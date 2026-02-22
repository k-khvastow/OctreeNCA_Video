"""
Agent_OctreeNCA_M2SingleStep.py
──────────────────────────────────────────────────────────────────────────────
Training and evaluation agent for the keyframe M1 + single-step M2 setup.

Training strategy
─────────────────
Each batch step draws ONE training pair (anchor, target):
  - anchor  = frame 0 (the keyframe) of the sequence
  - target  = frame k, where k ~ Uniform[1, max_step]  (clamped to T-1)

Forward:
  1. Run M1 on the anchor frame → upscale states to 512×512.
  2. Run M2 on the target frame, conditioned on the upscaled M1 states.
  3. Compute segmentation loss on M2's output vs GT at frame k.
  4. Optionally also supervise M1 output (use_t0_for_loss=True).

Inference strategy
──────────────────
Mirrors the intended deployment pattern:
  - Every ``keyframe_interval`` frames (default 20 = 1/s at 20 FPS):
    → Run M1 to produce fresh hidden states.
    → Score M1's prediction.
  - All other frames:
    → Run M2 with the most recent M1 states (fixed, not updated by M2).
    → Score M2's prediction.

Configuration keys consumed
───────────────────────────
  trainer.m2_single_step.max_step    (int, default 20)
  trainer.m2_single_step.keyframe_interval  (int, default 20)
  model.m1.use_first_frame           (bool, default True)
  model.m1.use_t0_for_loss           (bool, default False)
  trainer.use_amp                    (bool, default False)
  trainer.ema                        (bool)
  trainer.gradient_accumulation      (int, default 1)
  trainer.normalize_gradients        ("all" | "none")
  experiment.logging.track_gradient_norm  (bool)
"""
from __future__ import annotations

import numpy as np
import time
import warnings
import torch

from tqdm import tqdm

from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.utils.helper import merge_img_label_gt_simplified


class OctreeNCAM2SingleStepAgent(MedNCAAgent):
    """
    Agent for the keyframe-M1 + single-step-M2 temporal segmentation model.

    Expected data dict keys:
      image_a, label_a : (B, T, C, H, W)
      image_b, label_b : (B, T, C, H, W)
    """

    def __init__(self, model):
        super().__init__(model)
        self.accum_iter = 0
        self._warned_no_supervision = False
        self._pt_log_count = 0
        self._pt_log_interval = 20

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _log_phase_times(self, pt_times: dict[str, float]):
        """Log per-phase timing to console and tensorboard."""
        # Filter out internal _start_ keys
        times = {k: v for k, v in pt_times.items() if not k.startswith('_start_')}
        total = sum(times.values())

        # Log to tensorboard
        global_step = getattr(self, '_current_global_step', None)
        if global_step is not None:
            for k, v in times.items():
                self.exp.write_scalar(f'TimingPhases/m2_batch/{k}_ms', v * 1000.0, global_step)
            self.exp.write_scalar('TimingPhases/m2_batch/total_ms', total * 1000.0, global_step)

        # Print periodically
        self._pt_log_count += 1
        if self._pt_log_count % self._pt_log_interval == 1:
            parts = " ".join(f"{k}={v*1000:.1f}ms" for k, v in times.items())
            print(f"[m2_timing] step={global_step} {parts} total={total*1000:.1f}ms")

    def _as_device_tensor(self, x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(self.device)
        return torch.as_tensor(x, device=self.device)

    def _concat_views(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([view_a, view_b], dim=0)

    def _score_dual_frame(self, loss_f, pred_bchw, y_a_t, y_b_t, ids, bucket,
                          dist_a_t=None, dist_b_t=None):
        batch_size = y_a_t.shape[0]
        pred_a = pred_bchw[:batch_size]
        pred_b = pred_bchw[batch_size:]
        for b in range(batch_size):
            pid = ids[b] if b < len(ids) else f"unknown_{b}"
            for view_pred, view_y, view_dist, suffix in [
                (pred_a[b:b + 1], y_a_t[b:b + 1], dist_a_t[b:b + 1] if dist_a_t is not None else None, "_A"),
                (pred_b[b:b + 1], y_b_t[b:b + 1], dist_b_t[b:b + 1] if dist_b_t is not None else None, "_B"),
            ]:
                score_kwargs = dict(pred=view_pred, target=view_y, patient_id=f"{pid}{suffix}")
                if view_dist is not None:
                    score_kwargs["target_dist"] = view_dist
                scores = loss_f(**score_kwargs)
                for key, val in scores.items():
                    if key not in bucket:
                        bucket[key] = {}
                    p = f"{pid}{suffix}"
                    if p not in bucket[key]:
                        bucket[key][p] = []
                    bucket[key][p].append(val.item() if isinstance(val, torch.Tensor) else val)

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def batch_step(self, data: dict, loss_f: torch.nn.Module) -> dict:
        cfg = self.exp.config

        # ── Phase timing setup ──────────────────────────────────────────
        _pt = getattr(self, '_phase_timing_enabled', False)
        _pt_cuda = _pt and (self.device.type == 'cuda')
        _pt_times: dict[str, float] = {}
        def _pt_sync():
            if _pt_cuda:
                torch.cuda.synchronize(self.device)
        def _pt_start(name: str):
            if _pt:
                _pt_sync()
                _pt_times[f'_start_{name}'] = time.perf_counter()
        def _pt_end(name: str):
            if _pt:
                _pt_sync()
                _pt_times[name] = time.perf_counter() - _pt_times.pop(f'_start_{name}')

        _pt_start('data_to_device')

        accum_steps = int(cfg.get("trainer.gradient_accumulation", 1))
        use_amp = cfg.get("trainer.use_amp", False)
        max_step = int(cfg.get("trainer.m2_single_step.max_step", 20))
        use_t0_for_loss = bool(cfg.get("model.m1.use_t0_for_loss", False))

        x_a_seq = self._as_device_tensor(data["image_a"])
        y_a_seq = self._as_device_tensor(data["label_a"])
        x_b_seq = self._as_device_tensor(data["image_b"])
        y_b_seq = self._as_device_tensor(data["label_b"])

        # ── Precomputed boundary distance maps (optional) ───────────
        y_dist_a_seq = self._as_device_tensor(data.get("label_dist_a", None))
        y_dist_b_seq = self._as_device_tensor(data.get("label_dist_b", None))

        # Ensure temporal dimension exists
        if x_a_seq.ndim == 4:
            x_a_seq = x_a_seq.unsqueeze(1)
            y_a_seq = y_a_seq.unsqueeze(1)
            x_b_seq = x_b_seq.unsqueeze(1)
            y_b_seq = y_b_seq.unsqueeze(1)
            if y_dist_a_seq is not None:
                y_dist_a_seq = y_dist_a_seq.unsqueeze(1)
            if y_dist_b_seq is not None:
                y_dist_b_seq = y_dist_b_seq.unsqueeze(1)

        time_steps = x_a_seq.shape[1]
        if time_steps < 2:
            if not self._warned_no_supervision:
                warnings.warn(
                    "[M2SingleStep] Sequence length < 2; cannot form a target frame. Skipping batch."
                )
                self._warned_no_supervision = True
            return {}

        # ── Determine target frame index ────────────────────────────────
        # Sparse mode: the dataset already sampled k and only loaded frames
        # 0 (anchor) and k (target). The temporal dim has exactly 2 entries:
        #   index 0 = anchor,  index 1 = the randomly chosen target.
        # sparse_target_k holds the original k for logging purposes.
        sparse_k = data.get("sparse_target_k", None)
        if sparse_k is not None:
            # Dataset pre-sampled k; target is at temporal index 1
            target_idx = 1
            # Extract the original k value for logging (may be batched tensor)
            if torch.is_tensor(sparse_k):
                original_k = int(sparse_k[0].item()) if sparse_k.dim() > 0 else int(sparse_k.item())
            else:
                original_k = int(sparse_k[0]) if hasattr(sparse_k, '__len__') else int(sparse_k)
        else:
            # Full sequence: sample k ourselves
            k = torch.randint(1, max(2, min(max_step, time_steps - 1) + 1), (1,)).item()
            target_idx = int(k)
            original_k = target_idx

        _pt_end('data_to_device')

        # ── Gradient accumulation zero_grad ─────────────────────────────
        if self.accum_iter % accum_steps == 0:
            self.optimizer.zero_grad()

        normalize_grads = cfg["trainer.normalize_gradients"] == "all"
        track_grads = cfg.get("experiment.logging.track_gradient_norm", False)
        total_norm = 0.0
        did_backward = False
        loss_ret = {}
        total_supervised_steps = 1 + (1 if use_t0_for_loss else 0)

        def _backward(loss_tensor: torch.Tensor):
            nonlocal did_backward
            if not isinstance(loss_tensor, torch.Tensor):
                return
            scaled = loss_tensor
            if scaled.numel() > 1:
                scaled = scaled.mean()
            scaled = scaled / float(total_supervised_steps) / float(accum_steps)
            if not scaled.requires_grad:
                return
            if use_amp:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            did_backward = True

        # ── Step 1: M1 on anchor frame (t=0) ───────────────────────────
        _pt_start('m1_forward')
        x_a_0 = x_a_seq[:, 0]
        x_b_0 = x_b_seq[:, 0]
        y_a_0 = y_a_seq[:, 0]
        y_b_0 = y_b_seq[:, 0]

        if use_amp:
            with torch.amp.autocast("cuda"):
                m1_out, (state_a, state_b) = self.model.m1_forward_and_init_states(
                    x_a_0, x_b_0, y_a_0, y_b_0
                )
        else:
            m1_out, (state_a, state_b) = self.model.m1_forward_and_init_states(
                x_a_0, x_b_0, y_a_0, y_b_0
            )

        # Detach M1 states — M2 is trained independently
        state_a = state_a.detach()
        state_b = state_b.detach()
        _pt_end('m1_forward')

        # Optional: supervise M1 output at t=0
        if use_t0_for_loss and m1_out is not None:
            t0_out = dict(m1_out)
            t0_out["target_unpatched"] = self._concat_views(y_a_0, y_b_0)
            if y_dist_a_seq is not None and y_dist_b_seq is not None:
                t0_out["target_dist"] = self._concat_views(y_dist_a_seq[:, 0], y_dist_b_seq[:, 0])
            if use_amp:
                with torch.amp.autocast("cuda"):
                    l0, l0_dict = loss_f(**t0_out)
            else:
                l0, l0_dict = loss_f(**t0_out)
            if not (isinstance(l0, torch.Tensor) and torch.isnan(l0).any()):
                _backward(l0)
                for key, val in l0_dict.items():
                    loss_ret[f"m1/{key}"] = val.item() if isinstance(val, torch.Tensor) else val

        # ── Step 2: M2 on target frame ──────────────────────────────────
        _pt_start('m2_forward')
        x_a_t = x_a_seq[:, target_idx]
        x_b_t = x_b_seq[:, target_idx]
        y_a_t = y_a_seq[:, target_idx]
        y_b_t = y_b_seq[:, target_idx]

        if use_amp:
            with torch.amp.autocast("cuda"):
                out = self.model(
                    x_a_t, x_b_t,
                    y_a=y_a_t, y_b=y_b_t,
                    prev_state_a=state_a,
                    prev_state_b=state_b,
                    step_k=original_k,
                )
        else:
            out = self.model(
                x_a_t, x_b_t,
                y_a=y_a_t, y_b=y_b_t,
                prev_state_a=state_a,
                prev_state_b=state_b,
                step_k=original_k,
            )

        _pt_end('m2_forward')

        # NaN guard
        _logits = out.get("logits", None)
        if _logits is not None and torch.isnan(_logits).any():
            warnings.warn(f"[M2SingleStep] NaN in M2 logits (target_idx={target_idx}). Skipping.")
            return {}

        out["target_unpatched"] = self._concat_views(y_a_t, y_b_t)
        if y_dist_a_seq is not None and y_dist_b_seq is not None:
            out["target_dist"] = self._concat_views(
                y_dist_a_seq[:, target_idx], y_dist_b_seq[:, target_idx]
            )

        _pt_start('loss')
        if use_amp:
            with torch.amp.autocast("cuda"):
                l, l_dict = loss_f(**out)
        else:
            l, l_dict = loss_f(**out)
        _pt_end('loss')

        if isinstance(l, torch.Tensor) and torch.isnan(l).any():
            warnings.warn(f"[M2SingleStep] NaN in M2 loss (target_idx={target_idx}). Skipping.")
            return {}

        _pt_start('backward')
        _backward(l)
        _pt_end('backward')
        for key, val in l_dict.items():
            loss_ret[key] = (loss_ret.get(key, 0.0)
                             + (val.item() if isinstance(val, torch.Tensor) else val))

        # ── Log the sampled step for diagnostics ───────────────────────
        loss_ret["step_k"] = float(original_k)

        if not did_backward:
            if _pt:
                self._log_phase_times(_pt_times)
            return loss_ret

        self.accum_iter += 1
        _pt_start('optimizer_step')
        if self.accum_iter % accum_steps == 0:
            if use_amp and (normalize_grads or track_grads):
                self.scaler.unscale_(self.optimizer)
            if normalize_grads or track_grads:
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.detach().data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
            if normalize_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if track_grads:
                if not hasattr(self, "epoch_grad_norm"):
                    self.epoch_grad_norm = []
                self.epoch_grad_norm.append(total_norm)

            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.exp.config["trainer.ema"]:
                self.ema.update()
        _pt_end('optimizer_step')

        if _pt:
            self._log_phase_times(_pt_times)
        return loss_ret

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self, loss_f, split="test", tag="test/img/", **kwargs):
        self.model.eval()

        if split not in self.exp.data_loaders:
            return {}

        loader = self.exp.data_loaders[split]
        save_img = kwargs.get("save_img", None)
        if save_img is None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        keyframe_interval = int(self.exp.config.get("trainer.m2_single_step.keyframe_interval", 20))
        loss_log: dict = {}
        dataset = self.exp.datasets.get(split, None)
        is_rgb = bool(getattr(dataset, "is_rgb", False))

        pbar = tqdm(loader, desc=f"Eval {split}")

        for i, data in enumerate(pbar):
            x_a_seq = self._as_device_tensor(data["image_a"])
            y_a_seq = self._as_device_tensor(data["label_a"])
            x_b_seq = self._as_device_tensor(data["image_b"])
            y_b_seq = self._as_device_tensor(data["label_b"])

            # Precomputed boundary distance maps (optional)
            y_dist_a_seq = self._as_device_tensor(data.get("label_dist_a", None))
            y_dist_b_seq = self._as_device_tensor(data.get("label_dist_b", None))

            if x_a_seq.ndim == 4:
                x_a_seq = x_a_seq.unsqueeze(1)
                y_a_seq = y_a_seq.unsqueeze(1)
                x_b_seq = x_b_seq.unsqueeze(1)
                y_b_seq = y_b_seq.unsqueeze(1)
                if y_dist_a_seq is not None:
                    y_dist_a_seq = y_dist_a_seq.unsqueeze(1)
                if y_dist_b_seq is not None:
                    y_dist_b_seq = y_dist_b_seq.unsqueeze(1)

            batch_size, time_steps, _, _, _ = x_a_seq.shape
            raw_ids = data.get("id", None)
            if raw_ids is None:
                ids = [f"{i}_{b}" for b in range(batch_size)]
            elif isinstance(raw_ids, (list, tuple)):
                ids = [str(pid) for pid in raw_ids]
            else:
                ids = [str(raw_ids)] * batch_size

            batch_scores_time: dict = {}
            vis_x_a = vis_y_a = vis_pred_a = None
            vis_x_b = vis_y_b = vis_pred_b = None

            # Persistent M1 states (refreshed at keyframes)
            m1_state_a: torch.Tensor | None = None
            m1_state_b: torch.Tensor | None = None
            last_keyframe_t: int = 0

            for t in range(time_steps):
                x_a_t = x_a_seq[:, t]
                x_b_t = x_b_seq[:, t]
                y_a_t = y_a_seq[:, t]
                y_b_t = y_b_seq[:, t]

                dist_a_t = y_dist_a_seq[:, t] if y_dist_a_seq is not None else None
                dist_b_t = y_dist_b_seq[:, t] if y_dist_b_seq is not None else None

                if t % keyframe_interval == 0:
                    last_keyframe_t = t
                    # ── Keyframe: run M1 ────────────────────────────────
                    m1_out, (m1_state_a, m1_state_b) = self.model.m1_forward_and_init_states(
                        x_a_t, x_b_t, y_a_t, y_b_t,
                    )
                    # Score M1 prediction
                    pred_m1 = m1_out.get("probabilities", m1_out.get("logits", None))
                    if pred_m1 is not None:
                        pred_m1_bchw = pred_m1.permute(0, 3, 1, 2)
                        self._score_dual_frame(
                            loss_f=loss_f,
                            pred_bchw=pred_m1_bchw,
                            y_a_t=y_a_t, y_b_t=y_b_t,
                            ids=ids, bucket=batch_scores_time,
                            dist_a_t=dist_a_t, dist_b_t=dist_b_t,
                        )
                        vis_x_a, vis_y_a = x_a_t, y_a_t
                        vis_pred_a = pred_m1_bchw[:batch_size]
                        vis_x_b, vis_y_b = x_b_t, y_b_t
                        vis_pred_b = pred_m1_bchw[batch_size:]
                else:
                    # ── Intermediate frame: run M2 ──────────────────────
                    if m1_state_a is None or m1_state_b is None:
                        # No keyframe has been processed yet; run M1 now.
                        m1_out, (m1_state_a, m1_state_b) = self.model.m1_forward_and_init_states(
                            x_a_t, x_b_t, y_a_t, y_b_t,
                        )

                    out = self.model(
                        x_a_t, x_b_t,
                        y_a=y_a_t, y_b=y_b_t,
                        prev_state_a=m1_state_a,
                        prev_state_b=m1_state_b,
                        step_k=t - last_keyframe_t,
                    )
                    # M1 states are NOT updated from M2's output

                    pred = out.get("probabilities", out.get("logits", None))
                    if pred is None:
                        continue
                    pred_bchw = pred.permute(0, 3, 1, 2)
                    self._score_dual_frame(
                        loss_f=loss_f,
                        pred_bchw=pred_bchw,
                        y_a_t=y_a_t, y_b_t=y_b_t,
                        ids=ids, bucket=batch_scores_time,
                        dist_a_t=dist_a_t, dist_b_t=dist_b_t,
                    )
                    vis_x_a, vis_y_a = x_a_t, y_a_t
                    vis_pred_a = pred_bchw[:batch_size]
                    vis_x_b, vis_y_b = x_b_t, y_b_t
                    vis_pred_b = pred_bchw[batch_size:]

            for metric_name, metric_pid_values in batch_scores_time.items():
                if metric_name not in loss_log:
                    loss_log[metric_name] = {}
                for pid, values in metric_pid_values.items():
                    loss_log[metric_name][pid] = float(np.mean(values))

            if (
                i in save_img
                and vis_x_a is not None and vis_pred_a is not None
            ):
                for b in range(batch_size):
                    pid = ids[b] if b < len(ids) else f"unknown_{i}_{b}"

                    def _img(x):
                        return x[b:b + 1].detach().cpu().permute(0, 2, 3, 1)

                    self.exp.write_img(
                        f"{tag}{pid}_A_{i}",
                        merge_img_label_gt_simplified(*[_img(v) for v in (vis_x_a, vis_pred_a, vis_y_a)], is_rgb),
                        self.exp.currentStep,
                    )
                    self.exp.write_img(
                        f"{tag}{pid}_B_{i}",
                        merge_img_label_gt_simplified(*[_img(v) for v in (vis_x_b, vis_pred_b, vis_y_b)], is_rgb),
                        self.exp.currentStep,
                    )

            dice_keys = [k for k in batch_scores_time if "DiceScore" in k]
            if dice_keys:
                vals = [v for dk in dice_keys for vs in batch_scores_time[dk].values() for v in vs]
                if vals:
                    pbar.set_postfix({"dice": float(np.mean(vals))})

        print(f"\n[{split.upper()} SCORES]")
        for metric, scores_dict in loss_log.items():
            if scores_dict:
                print(f"  > {metric}: {np.mean(list(scores_dict.values())):.4f}")
        print("--------------------------\n")

        self.model.train()
        return loss_log
