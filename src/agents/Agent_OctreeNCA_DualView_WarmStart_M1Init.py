import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings

from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.losses.TemporalConsistencyLoss import TemporalConsistencyLoss
from src.utils.helper import merge_img_label_gt_simplified


class OctreeNCADualViewWarmStartM1InitAgent(MedNCAAgent):
    """
    Temporal dual-view warm-start agent with M1 state initialization.

    Expected input tensors:
      - image_a, label_a: (B, T, C, H, W)
      - image_b, label_b: (B, T, C, H, W)
    """

    def __init__(self, model):
        super().__init__(model)
        self.accum_iter = 0
        self._warned_no_supervision = False
        self._temporal_consistency_loss_fn = TemporalConsistencyLoss()

    def _as_device_tensor(self, x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(self.device)
        return torch.as_tensor(x, device=self.device)

    def _concat_views(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([view_a, view_b], dim=0)

    def _init_prev_states(self, x_a_seq, x_b_seq, y_a_seq, y_b_seq):
        use_m1 = self.exp.config.get("model.m1.use_first_frame", True)
        if not use_m1 or x_a_seq.shape[1] == 0:
            return None, None, 0, None

        m1_out, (prev_state_a, prev_state_b) = self.model.m1_forward_and_init_states(
            x_a_seq[:, 0],
            x_b_seq[:, 0],
            y_a_seq[:, 0],
            y_b_seq[:, 0],
        )
        return prev_state_a, prev_state_b, 1, m1_out

    def _score_dual_frame(self, loss_f, pred_bchw, y_a_t, y_b_t, ids, bucket):
        batch_size = y_a_t.shape[0]
        pred_a = pred_bchw[:batch_size]
        pred_b = pred_bchw[batch_size:]

        for b in range(batch_size):
            pid = ids[b] if b < len(ids) else f"unknown_{b}"
            scores_a = loss_f(pred=pred_a[b:b + 1], target=y_a_t[b:b + 1], patient_id=f"{pid}_A")
            scores_b = loss_f(pred=pred_b[b:b + 1], target=y_b_t[b:b + 1], patient_id=f"{pid}_B")

            for key, val in scores_a.items():
                if key not in bucket:
                    bucket[key] = {}
                if f"{pid}_A" not in bucket[key]:
                    bucket[key][f"{pid}_A"] = []
                bucket[key][f"{pid}_A"].append(val.item() if isinstance(val, torch.Tensor) else val)

            for key, val in scores_b.items():
                if key not in bucket:
                    bucket[key] = {}
                if f"{pid}_B" not in bucket[key]:
                    bucket[key][f"{pid}_B"] = []
                bucket[key][f"{pid}_B"].append(val.item() if isinstance(val, torch.Tensor) else val)

    def batch_step(self, data: dict, loss_f: torch.nn.Module) -> dict:
        accum_steps = int(self.exp.config.get("trainer.gradient_accumulation", 1))
        seq_tbptt_steps = self.exp.config.get("model.sequence.tbptt_steps", None)
        seq_tbptt_steps = int(seq_tbptt_steps) if seq_tbptt_steps not in (None, 0, "0", "") else None
        seq_tbptt_mode_raw = str(self.exp.config.get("model.sequence.tbptt_mode", "detach")).strip().lower()
        if seq_tbptt_mode_raw in ("off", "none", "disabled", "0", ""):
            seq_tbptt_mode = "off"
        elif seq_tbptt_mode_raw in ("detach", "detach_only", "legacy", "1"):
            seq_tbptt_mode = "detach"
        elif seq_tbptt_mode_raw in ("chunked", "chunk", "backward_chunked", "true_tbptt"):
            seq_tbptt_mode = "chunked"
        else:
            raise ValueError(
                f"Unknown model.sequence.tbptt_mode '{seq_tbptt_mode_raw}'. "
                "Use one of: off, detach, chunked."
            )
        if seq_tbptt_steps is None or seq_tbptt_steps <= 0:
            seq_tbptt_mode = "off"
        use_amp = self.exp.config.get("trainer.use_amp", False)

        # --- Temporal consistency loss weight ---
        tc_weight = float(self.exp.config.get("trainer.temporal_consistency_weight", 0.0))

        # --- Curriculum schedule: compute effective sequence length ---
        cur_epoch = getattr(self, "_current_epoch", 0) or 0
        seq_len_min = int(self.exp.config.get("trainer.curriculum.seq_len_min", 0))
        seq_len_max = int(self.exp.config.get("trainer.curriculum.seq_len_max", 0))
        curriculum_epochs = int(self.exp.config.get("trainer.curriculum.warmup_epochs", 0))

        # Sync epoch counter to the warm-start model for noise annealing
        m2 = getattr(self.model, "m2", None)
        if m2 is not None and hasattr(m2, "_current_epoch"):
            m2._current_epoch = cur_epoch

        x_a_seq = self._as_device_tensor(data["image_a"])
        y_a_seq = self._as_device_tensor(data["label_a"])
        x_b_seq = self._as_device_tensor(data["image_b"])
        y_b_seq = self._as_device_tensor(data["label_b"])
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

        _, time_steps, _, _, _ = x_a_seq.shape

        # Curriculum truncation: if enabled, limit the effective time steps
        if seq_len_min > 0 and seq_len_max > seq_len_min and curriculum_epochs > 0:
            effective_len = int(
                seq_len_min + (seq_len_max - seq_len_min)
                * min(1.0, cur_epoch / float(curriculum_epochs))
            )
            effective_len = max(2, min(effective_len, time_steps))  # need ≥2 for warm-start
            if effective_len < time_steps:
                x_a_seq = x_a_seq[:, :effective_len]
                y_a_seq = y_a_seq[:, :effective_len]
                x_b_seq = x_b_seq[:, :effective_len]
                y_b_seq = y_b_seq[:, :effective_len]
                if y_dist_a_seq is not None:
                    y_dist_a_seq = y_dist_a_seq[:, :effective_len]
                if y_dist_b_seq is not None:
                    y_dist_b_seq = y_dist_b_seq[:, :effective_len]
                time_steps = effective_len

        loss_val = None
        chunk_loss = None
        loss_ret = {}
        batch_stat_sums = {}
        did_backward = False

        if self.accum_iter % accum_steps == 0:
            self.optimizer.zero_grad()

        use_t0_for_loss = self.exp.config.get("model.m1.use_t0_for_loss", False)
        prev_state_a, prev_state_b, start_t, m1_out = self._init_prev_states(x_a_seq, x_b_seq, y_a_seq, y_b_seq)
        supervised_t0 = bool(use_t0_for_loss and m1_out is not None)
        total_supervised_steps = max(0, time_steps - start_t) + (1 if supervised_t0 else 0)
        steps = 0
        prev_hidden = None  # for temporal consistency loss

        def _accumulate_loss(loss_tensor: torch.Tensor):
            nonlocal loss_val, chunk_loss
            if seq_tbptt_mode == "chunked":
                chunk_loss = loss_tensor if chunk_loss is None else (chunk_loss + loss_tensor)
            else:
                loss_val = loss_tensor if loss_val is None else (loss_val + loss_tensor)

        def _backward_scaled(loss_tensor):
            nonlocal did_backward
            if not isinstance(loss_tensor, torch.Tensor):
                return
            scaled_loss = loss_tensor
            if scaled_loss.numel() > 1:
                scaled_loss = scaled_loss.mean()
            scaled_loss = scaled_loss / float(total_supervised_steps)
            scaled_loss = scaled_loss / accum_steps
            if not scaled_loss.requires_grad:
                return
            if use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            did_backward = True

        if total_supervised_steps == 0:
            if not self._warned_no_supervision:
                print(
                    "[dual_warm_start_m1init] No supervised steps in this batch: "
                    "M1 handled t=0 and there are no later frames for M2."
                )
                self._warned_no_supervision = True
            return {}

        _nan_batch_count = 0  # track NaN frames within this batch_step

        if use_t0_for_loss and m1_out is not None:
            out = dict(m1_out)
            out["target_unpatched"] = self._concat_views(y_a_seq[:, 0], y_b_seq[:, 0])
            if y_dist_a_seq is not None and y_dist_b_seq is not None:
                out["target_dist"] = self._concat_views(y_dist_a_seq[:, 0], y_dist_b_seq[:, 0])

            if use_amp:
                with torch.amp.autocast("cuda"):
                    l, l_dict = loss_f(**out)
            else:
                l, l_dict = loss_f(**out)

            _accumulate_loss(l)
            steps += 1

            step_loss_ret = {}
            for key, value in l_dict.items():
                if key not in loss_ret:
                    loss_ret[key] = 0
                val = value.item() if isinstance(value, torch.Tensor) else value
                loss_ret[key] += val
                step_loss_ret[key] = val

            frame_data = {
                "image": self._concat_views(x_a_seq[:, 0], x_b_seq[:, 0]),
                "label": self._concat_views(y_a_seq[:, 0], y_b_seq[:, 0]),
            }
            self._log_batch_class_counts(frame_data, step_loss_ret)
            self._maybe_log_spike_batch(frame_data, out, step_loss_ret)

            for key, value in step_loss_ret.items():
                if key.startswith("batch_class_pixels/") or key.startswith("batch_class_frac/"):
                    if key not in batch_stat_sums:
                        batch_stat_sums[key] = 0.0
                    batch_stat_sums[key] += float(value)

        for t in range(start_t, time_steps):
            x_a_t = x_a_seq[:, t]
            y_a_t = y_a_seq[:, t]
            x_b_t = x_b_seq[:, t]
            y_b_t = y_b_seq[:, t]

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = self.model(
                        x_a_t,
                        x_b_t,
                        y_a=y_a_t,
                        y_b=y_b_t,
                        prev_state_a=prev_state_a,
                        prev_state_b=prev_state_b,
                    )
            else:
                out = self.model(
                    x_a_t,
                    x_b_t,
                    y_a=y_a_t,
                    y_b=y_b_t,
                    prev_state_a=prev_state_a,
                    prev_state_b=prev_state_b,
                )

            # --- NaN guard: detect NaN in model output and skip frame ---
            _logits = out.get("logits", None)
            if _logits is not None and torch.isnan(_logits).any():
                _nan_batch_count += 1
                if _nan_batch_count <= 3:
                    warnings.warn(
                        f"[dual_warm_start_m1init] NaN detected in model logits at t={t} "
                        f"(epoch={cur_epoch}). Skipping this frame for loss computation."
                    )
                # Still update prev_state with detached (non-NaN) values
                if "final_state_a" in out and "final_state_b" in out:
                    prev_state_a = torch.nan_to_num(out["final_state_a"], nan=0.0).detach()
                    prev_state_b = torch.nan_to_num(out["final_state_b"], nan=0.0).detach()
                continue

            out["target_unpatched"] = self._concat_views(y_a_t, y_b_t)
            if y_dist_a_seq is not None and y_dist_b_seq is not None:
                out["target_dist"] = self._concat_views(y_dist_a_seq[:, t], y_dist_b_seq[:, t])

            if use_amp:
                with torch.amp.autocast("cuda"):
                    l, l_dict = loss_f(**out)
            else:
                l, l_dict = loss_f(**out)

            # --- NaN guard: detect NaN in loss and skip frame ---
            if isinstance(l, torch.Tensor) and torch.isnan(l).any():
                _nan_batch_count += 1
                if _nan_batch_count <= 3:
                    warnings.warn(
                        f"[dual_warm_start_m1init] NaN detected in loss at t={t} "
                        f"(epoch={cur_epoch}). Skipping this frame."
                    )
                if "final_state_a" in out and "final_state_b" in out:
                    prev_state_a = out["final_state_a"].detach()
                    prev_state_b = out["final_state_b"].detach()
                continue

            _accumulate_loss(l)
            steps += 1

            # --- Temporal consistency loss on hidden states ---
            if tc_weight > 0.0 and "hidden_channels" in out:
                hidden_t = out["hidden_channels"]
                if prev_hidden is not None:
                    tc_loss, tc_dict = self._temporal_consistency_loss_fn(hidden_t, prev_hidden)
                    _accumulate_loss(tc_loss * tc_weight)
                    for k, v in tc_dict.items():
                        full_key = f"TemporalConsistencyLoss/{k}"
                        if full_key not in loss_ret:
                            loss_ret[full_key] = 0
                        loss_ret[full_key] += v * tc_weight
                prev_hidden = hidden_t.detach()

            step_loss_ret = {}
            for key, value in l_dict.items():
                if key not in loss_ret:
                    loss_ret[key] = 0
                val = value.item() if isinstance(value, torch.Tensor) else value
                loss_ret[key] += val
                step_loss_ret[key] = val

            frame_data = {
                "image": self._concat_views(x_a_t, x_b_t),
                "label": self._concat_views(y_a_t, y_b_t),
            }
            self._log_batch_class_counts(frame_data, step_loss_ret)
            self._maybe_log_spike_batch(frame_data, out, step_loss_ret)

            for key, value in step_loss_ret.items():
                if key.startswith("batch_class_pixels/") or key.startswith("batch_class_frac/"):
                    if key not in batch_stat_sums:
                        batch_stat_sums[key] = 0.0
                    batch_stat_sums[key] += float(value)

            prev_state_a = out["final_state_a"]
            prev_state_b = out["final_state_b"]
            reached_boundary = (
                seq_tbptt_mode != "off"
                and seq_tbptt_steps is not None
                and seq_tbptt_steps > 0
                and prev_state_a is not None
                and prev_state_b is not None
                and ((t - start_t + 1) % seq_tbptt_steps == 0)
                and (t < time_steps - 1)
            )
            if reached_boundary:
                if seq_tbptt_mode == "chunked":
                    _backward_scaled(chunk_loss)
                    chunk_loss = None
                prev_state_a = prev_state_a.detach()
                prev_state_b = prev_state_b.detach()

        if seq_tbptt_mode == "chunked":
            _backward_scaled(chunk_loss)
        else:
            _backward_scaled(loss_val)

        if steps == 0:
            # All frames were skipped due to NaN — nothing to log or backward
            warnings.warn(
                f"[dual_warm_start_m1init] All {total_supervised_steps} frames produced NaN "
                f"in this batch (epoch={cur_epoch}). Skipping entire batch."
            )
            return {}

        loss_ret = {key: value / steps for key, value in loss_ret.items()}
        for key, value in batch_stat_sums.items():
            loss_ret[key] = value / steps

        track_grads = self.exp.config.get("experiment.logging.track_gradient_norm", False)
        normalize_grads = self.exp.config["trainer.normalize_gradients"] == "all"
        total_norm = 0.0

        if not did_backward:
            return loss_ret

        self.accum_iter += 1
        if self.accum_iter % accum_steps == 0:
            if use_amp and (normalize_grads or track_grads):
                self.scaler.unscale_(self.optimizer)
            if normalize_grads or track_grads:
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
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

        return loss_ret

    @torch.no_grad()
    def test(self, loss_f, split="test", tag="test/img/", **kwargs):
        self.model.eval()

        if split not in self.exp.data_loaders:
            return {}

        loader = self.exp.data_loaders[split]
        save_img = kwargs.get("save_img", None)
        if save_img is None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]
        loss_log = {}
        dataset = self.exp.datasets.get(split, None)
        is_rgb = bool(getattr(dataset, "is_rgb", False))

        pbar = tqdm(loader, desc=f"Eval {split}")

        for i, data in enumerate(pbar):
            x_a_seq = self._as_device_tensor(data["image_a"])
            y_a_seq = self._as_device_tensor(data["label_a"])
            x_b_seq = self._as_device_tensor(data["image_b"])
            y_b_seq = self._as_device_tensor(data["label_b"])

            if x_a_seq.ndim == 4:
                x_a_seq = x_a_seq.unsqueeze(1)
                y_a_seq = y_a_seq.unsqueeze(1)
                x_b_seq = x_b_seq.unsqueeze(1)
                y_b_seq = y_b_seq.unsqueeze(1)

            batch_size, time_steps, _, _, _ = x_a_seq.shape
            raw_ids = data.get("id", None)
            if raw_ids is None:
                ids = [f"{i}_{b}" for b in range(batch_size)]
            elif isinstance(raw_ids, (list, tuple)):
                ids = [str(pid) for pid in raw_ids]
            else:
                ids = [str(raw_ids)] * batch_size

            prev_state_a, prev_state_b, start_t, m1_out = self._init_prev_states(x_a_seq, x_b_seq, y_a_seq, y_b_seq)

            batch_scores_time = {}
            vis_x_a = None
            vis_y_a = None
            vis_pred_a = None
            vis_x_b = None
            vis_y_b = None
            vis_pred_b = None

            if m1_out is not None:
                pred = m1_out.get("probabilities", m1_out.get("logits", None))
                if pred is not None:
                    pred_bchw = pred.permute(0, 3, 1, 2)
                    self._score_dual_frame(
                        loss_f=loss_f,
                        pred_bchw=pred_bchw,
                        y_a_t=y_a_seq[:, 0],
                        y_b_t=y_b_seq[:, 0],
                        ids=ids,
                        bucket=batch_scores_time,
                    )
                    vis_x_a = x_a_seq[:, 0]
                    vis_y_a = y_a_seq[:, 0]
                    vis_pred_a = pred_bchw[:batch_size]
                    vis_x_b = x_b_seq[:, 0]
                    vis_y_b = y_b_seq[:, 0]
                    vis_pred_b = pred_bchw[batch_size:]

            for t in range(start_t, time_steps):
                x_a_t = x_a_seq[:, t]
                y_a_t = y_a_seq[:, t]
                x_b_t = x_b_seq[:, t]
                y_b_t = y_b_seq[:, t]

                out = self.model(
                    x_a_t,
                    x_b_t,
                    y_a=y_a_t,
                    y_b=y_b_t,
                    prev_state_a=prev_state_a,
                    prev_state_b=prev_state_b,
                )
                prev_state_a = out["final_state_a"]
                prev_state_b = out["final_state_b"]

                pred = out.get("probabilities", out.get("logits", None))
                if pred is None:
                    continue
                pred_bchw = pred.permute(0, 3, 1, 2)
                self._score_dual_frame(
                    loss_f=loss_f,
                    pred_bchw=pred_bchw,
                    y_a_t=y_a_t,
                    y_b_t=y_b_t,
                    ids=ids,
                    bucket=batch_scores_time,
                )

                vis_x_a = x_a_t
                vis_y_a = y_a_t
                vis_pred_a = pred_bchw[:batch_size]
                vis_x_b = x_b_t
                vis_y_b = y_b_t
                vis_pred_b = pred_bchw[batch_size:]

            for metric_name, metric_pid_values in batch_scores_time.items():
                if metric_name not in loss_log:
                    loss_log[metric_name] = {}
                for pid, values in metric_pid_values.items():
                    loss_log[metric_name][pid] = float(np.mean(values))

            if (
                i in save_img
                and vis_x_a is not None
                and vis_y_a is not None
                and vis_pred_a is not None
                and vis_x_b is not None
                and vis_y_b is not None
                and vis_pred_b is not None
            ):
                image_a_bhwc = vis_x_a.detach().cpu().permute(0, 2, 3, 1)
                pred_a_bhwc = vis_pred_a.detach().cpu().permute(0, 2, 3, 1)
                gt_a_bhwc = vis_y_a.detach().cpu().permute(0, 2, 3, 1)

                image_b_bhwc = vis_x_b.detach().cpu().permute(0, 2, 3, 1)
                pred_b_bhwc = vis_pred_b.detach().cpu().permute(0, 2, 3, 1)
                gt_b_bhwc = vis_y_b.detach().cpu().permute(0, 2, 3, 1)

                for b in range(batch_size):
                    pid = ids[b] if b < len(ids) else f"unknown_{i}_{b}"
                    self.exp.write_img(
                        f"{tag}{pid}_A_{i}",
                        merge_img_label_gt_simplified(
                            image_a_bhwc[b:b + 1],
                            pred_a_bhwc[b:b + 1],
                            gt_a_bhwc[b:b + 1],
                            is_rgb,
                        ),
                        self.exp.currentStep,
                    )
                    self.exp.write_img(
                        f"{tag}{pid}_B_{i}",
                        merge_img_label_gt_simplified(
                            image_b_bhwc[b:b + 1],
                            pred_b_bhwc[b:b + 1],
                            gt_b_bhwc[b:b + 1],
                            is_rgb,
                        ),
                        self.exp.currentStep,
                    )

            dice_keys = [k for k in batch_scores_time.keys() if "DiceScore" in k]
            if len(dice_keys) > 0:
                dice_values = []
                for dk in dice_keys:
                    for vals in batch_scores_time[dk].values():
                        dice_values.extend(vals)
                if len(dice_values) > 0:
                    pbar.set_postfix({"dice": float(np.mean(dice_values))})

        print(f"\n[{split.upper()} SCORES]")
        for metric, scores_dict in loss_log.items():
            if len(scores_dict) > 0:
                avg = np.mean(list(scores_dict.values()))
                print(f"  > {metric}: {avg:.4f}")
        print("--------------------------\n")

        self.model.train()
        return loss_log
