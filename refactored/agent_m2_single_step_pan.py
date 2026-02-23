"""
agent_m2_single_step_pan.py
──────────────────────────────────────────────────────────────────────────────
Single-view (pan) variant of the M2 single-step agent.

The dataset provides single-view data:  ``image``, ``label``, ``label_dist``.
This agent adapts it for the dual-view M2 model by passing the same tensor
as both x_a and x_b.  Loss and scoring are computed on only one view's output
(the first B samples of the 2B-batch model output), avoiding gradient /
metric duplication.

Note: The model still processes a 2B batch internally (cat([x, x], dim=0))
because the DualView backbone is shared code.  With ``cross_fusion=none``
both halves produce identical results, so only the first B are used.
This is 2× compute for the backbone — a proper single-view model would
eliminate that overhead, but this avoids rewriting the model/warm-start
classes while giving mathematically correct gradients and scores.
"""
from __future__ import annotations

import numpy as np
import time
import warnings
import torch

from tqdm import tqdm

from src.agents.Agent_OctreeNCA_M2SingleStep import OctreeNCAM2SingleStepAgent
from src.utils.helper import merge_img_label_gt_simplified


class OctreeNCAM2SingleStepPanAgent(OctreeNCAM2SingleStepAgent):
    """
    Single-view agent for the keyframe-M1 + single-step-M2 model.

    Reads ``data["image"]`` / ``data["label"]`` (single-view) instead of
    ``data["image_a"]`` / ``data["image_b"]`` (dual-view).

    Passes the same tensor as both views to the dual-view model, then
    uses only the first B samples for loss/scoring.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Single-view helpers
    # ──────────────────────────────────────────────────────────────────────

    def _score_single_frame(self, loss_f, pred_bchw, y_t, ids, bucket, dist_t=None):
        """Score predictions for a single view (no _A/_B split)."""
        batch_size = y_t.shape[0]
        # Only use the first B outputs (view A half of the 2B model output)
        pred = pred_bchw[:batch_size]
        for b in range(batch_size):
            pid = ids[b] if b < len(ids) else f"unknown_{b}"
            score_kwargs = dict(
                pred=pred[b:b + 1],
                target=y_t[b:b + 1],
                patient_id=pid,
            )
            if dist_t is not None:
                score_kwargs["target_dist"] = dist_t[b:b + 1]
            scores = loss_f(**score_kwargs)
            for key, val in scores.items():
                if key not in bucket:
                    bucket[key] = {}
                if pid not in bucket[key]:
                    bucket[key][pid] = []
                bucket[key][pid].append(
                    val.item() if isinstance(val, torch.Tensor) else val
                )

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

        # ── Single-view data: read image/label (not image_a/image_b) ────
        x_seq = self._as_device_tensor(data["image"])
        y_seq = self._as_device_tensor(data["label"])
        y_dist_seq = self._as_device_tensor(data.get("label_dist", None))

        # Ensure temporal dimension exists: (B, C, H, W) → (B, 1, C, H, W)
        if x_seq.ndim == 4:
            x_seq = x_seq.unsqueeze(1)
            y_seq = y_seq.unsqueeze(1)
            if y_dist_seq is not None:
                y_dist_seq = y_dist_seq.unsqueeze(1)

        time_steps = x_seq.shape[1]
        if time_steps < 2:
            if not self._warned_no_supervision:
                warnings.warn(
                    "[M2SingleStepPan] Sequence length < 2; cannot form a target frame. Skipping."
                )
                self._warned_no_supervision = True
            return {}

        # ── Determine target frame index ────────────────────────────────
        sparse_k = data.get("sparse_target_k", None)
        if sparse_k is not None:
            target_idx = 1
            if torch.is_tensor(sparse_k):
                original_k = int(sparse_k[0].item()) if sparse_k.dim() > 0 else int(sparse_k.item())
            else:
                original_k = int(sparse_k[0]) if hasattr(sparse_k, '__len__') else int(sparse_k)
        else:
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
        x_0 = x_seq[:, 0]   # (B, C, H, W)
        y_0 = y_seq[:, 0]
        B = x_0.shape[0]

        # Pass same image as both views; clone the "b" copy so the model's
        # internal cat + inplace slice doesn't hit an autograd view error.
        if use_amp:
            with torch.amp.autocast("cuda"):
                m1_out, (state_a, state_b) = self.model.m1_forward_and_init_states(
                    x_0, x_0.clone(), y_0, y_0.clone()
                )
        else:
            m1_out, (state_a, state_b) = self.model.m1_forward_and_init_states(
                x_0, x_0.clone(), y_0, y_0.clone()
            )

        # Detach — M2 trained independently
        state_a = state_a.detach()
        state_b = state_b.detach()
        _pt_end('m1_forward')

        # Optional: supervise M1 output at t=0 (only B, not 2B)
        if use_t0_for_loss and m1_out is not None:
            t0_out = dict(m1_out)
            # Slice to first B — model output is 2B (cat of both views)
            if "logits" in t0_out and t0_out["logits"].shape[0] == 2 * B:
                t0_out["logits"] = t0_out["logits"][:B]
            if "probabilities" in t0_out and t0_out["probabilities"].shape[0] == 2 * B:
                t0_out["probabilities"] = t0_out["probabilities"][:B]
            if "target" in t0_out and t0_out["target"].shape[0] == 2 * B:
                t0_out["target"] = t0_out["target"][:B]
            t0_out["target_unpatched"] = y_0
            if y_dist_seq is not None:
                t0_out["target_dist"] = y_dist_seq[:, 0]
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
        x_t = x_seq[:, target_idx]
        y_t = y_seq[:, target_idx]

        if use_amp:
            with torch.amp.autocast("cuda"):
                out = self.model(
                    x_t, x_t.clone(),
                    y_a=y_t, y_b=y_t.clone(),
                    prev_state_a=state_a,
                    prev_state_b=state_b.clone(),
                    step_k=original_k,
                )
        else:
            out = self.model(
                x_t, x_t.clone(),
                y_a=y_t, y_b=y_t.clone(),
                prev_state_a=state_a,
                prev_state_b=state_b.clone(),
                step_k=original_k,
            )

        _pt_end('m2_forward')

        # NaN guard
        _logits = out.get("logits", None)
        if _logits is not None and torch.isnan(_logits).any():
            warnings.warn(f"[M2SingleStepPan] NaN in M2 logits (target_idx={target_idx}). Skipping.")
            return {}

        # Slice output to first B and compute loss on single view only
        out_single = dict(out)
        for key in ("logits", "probabilities", "hidden_channels", "target"):
            if key in out_single and out_single[key] is not None:
                t = out_single[key]
                if t.shape[0] == 2 * B:
                    out_single[key] = t[:B]

        out_single["target_unpatched"] = y_t
        if y_dist_seq is not None:
            out_single["target_dist"] = y_dist_seq[:, target_idx]

        _pt_start('loss')
        if use_amp:
            with torch.amp.autocast("cuda"):
                l, l_dict = loss_f(**out_single)
        else:
            l, l_dict = loss_f(**out_single)
        _pt_end('loss')

        if isinstance(l, torch.Tensor) and torch.isnan(l).any():
            warnings.warn(f"[M2SingleStepPan] NaN in M2 loss (target_idx={target_idx}). Skipping.")
            return {}

        _pt_start('backward')
        _backward(l)
        _pt_end('backward')
        for key, val in l_dict.items():
            loss_ret[key] = (
                loss_ret.get(key, 0.0)
                + (val.item() if isinstance(val, torch.Tensor) else val)
            )

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

        keyframe_interval = int(
            self.exp.config.get("trainer.m2_single_step.keyframe_interval", 20)
        )
        loss_log: dict = {}
        dataset = self.exp.datasets.get(split, None)
        is_rgb = bool(getattr(dataset, "is_rgb", False))

        pbar = tqdm(loader, desc=f"Eval {split}")

        for i, data in enumerate(pbar):
            # ── Single-view data ────────────────────────────────────────
            x_seq = self._as_device_tensor(data["image"])
            y_seq = self._as_device_tensor(data["label"])
            y_dist_seq = self._as_device_tensor(data.get("label_dist", None))

            if x_seq.ndim == 4:
                x_seq = x_seq.unsqueeze(1)
                y_seq = y_seq.unsqueeze(1)
                if y_dist_seq is not None:
                    y_dist_seq = y_dist_seq.unsqueeze(1)

            batch_size, time_steps, _, _, _ = x_seq.shape
            raw_ids = data.get("id", None)
            if raw_ids is None:
                ids = [f"{i}_{b}" for b in range(batch_size)]
            elif isinstance(raw_ids, (list, tuple)):
                ids = [str(pid) for pid in raw_ids]
            else:
                ids = [str(raw_ids)] * batch_size

            batch_scores_time: dict = {}
            vis_x = vis_y = vis_pred = None

            # Persistent M1 states
            m1_state_a: torch.Tensor | None = None
            m1_state_b: torch.Tensor | None = None
            last_keyframe_t: int = 0

            for t in range(time_steps):
                x_t = x_seq[:, t]
                y_t = y_seq[:, t]
                dist_t = y_dist_seq[:, t] if y_dist_seq is not None else None

                if t % keyframe_interval == 0:
                    last_keyframe_t = t
                    # ── Keyframe: run M1 ────────────────────────────────
                    m1_out, (m1_state_a, m1_state_b) = self.model.m1_forward_and_init_states(
                        x_t, x_t.clone(), y_t, y_t.clone(),
                    )
                    pred_m1 = m1_out.get("probabilities", m1_out.get("logits", None))
                    if pred_m1 is not None:
                        pred_m1_bchw = pred_m1.permute(0, 3, 1, 2)
                        self._score_single_frame(
                            loss_f=loss_f,
                            pred_bchw=pred_m1_bchw,
                            y_t=y_t,
                            ids=ids,
                            bucket=batch_scores_time,
                            dist_t=dist_t,
                        )
                        vis_x, vis_y = x_t, y_t
                        vis_pred = pred_m1_bchw[:batch_size]
                else:
                    # ── Intermediate frame: run M2 ──────────────────────
                    if m1_state_a is None or m1_state_b is None:
                        m1_out, (m1_state_a, m1_state_b) = self.model.m1_forward_and_init_states(
                            x_t, x_t.clone(), y_t, y_t.clone(),
                        )

                    out = self.model(
                        x_t, x_t.clone(),
                        y_a=y_t, y_b=y_t.clone(),
                        prev_state_a=m1_state_a,
                        prev_state_b=m1_state_b.clone(),
                        step_k=t - last_keyframe_t,
                    )

                    pred = out.get("probabilities", out.get("logits", None))
                    if pred is None:
                        continue
                    pred_bchw = pred.permute(0, 3, 1, 2)
                    self._score_single_frame(
                        loss_f=loss_f,
                        pred_bchw=pred_bchw,
                        y_t=y_t,
                        ids=ids,
                        bucket=batch_scores_time,
                        dist_t=dist_t,
                    )
                    vis_x, vis_y = x_t, y_t
                    vis_pred = pred_bchw[:batch_size]

            for metric_name, metric_pid_values in batch_scores_time.items():
                if metric_name not in loss_log:
                    loss_log[metric_name] = {}
                for pid, values in metric_pid_values.items():
                    loss_log[metric_name][pid] = float(np.mean(values))

            if (
                i in save_img
                and vis_x is not None
                and vis_pred is not None
            ):
                for b in range(batch_size):
                    pid = ids[b] if b < len(ids) else f"unknown_{i}_{b}"

                    def _img(x):
                        return x[b:b + 1].detach().cpu().permute(0, 2, 3, 1)

                    self.exp.write_img(
                        f"{tag}{pid}_{i}",
                        merge_img_label_gt_simplified(
                            *[_img(v) for v in (vis_x, vis_pred, vis_y)], is_rgb
                        ),
                        self.exp.currentStep,
                    )

            dice_keys = [k for k in batch_scores_time if "DiceScore" in k]
            if dice_keys:
                vals = [
                    v
                    for dk in dice_keys
                    for vs in batch_scores_time[dk].values()
                    for v in vs
                ]
                if vals:
                    pbar.set_postfix({"dice": float(np.mean(vals))})

        print(f"\n[{split.upper()} SCORES]")
        for metric, scores_dict in loss_log.items():
            if scores_dict:
                print(f"  > {metric}: {np.mean(list(scores_dict.values())):.4f}")
        print("--------------------------\n")

        self.model.train()
        return loss_log
