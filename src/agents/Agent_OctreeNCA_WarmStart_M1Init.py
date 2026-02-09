import torch
import numpy as np
from tqdm import tqdm
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.utils.helper import merge_img_label_gt_simplified


class OctreeNCAWarmStartM1InitAgent(MedNCAAgent):
    """
    Warm-start agent with M1 initialization on the first frame.
    Expects data input to be (B, T, C, H, W).
    """
    def __init__(self, model):
        super().__init__(model)
        self.accum_iter = 0
        self._warned_no_supervision = False

    def _init_prev_state(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        use_m1 = self.exp.config.get('model.m1.use_first_frame', True)

        if not use_m1 or x_seq.shape[1] == 0:
            return None, 0, None

        # First frame is segmented by M1 only. M2 starts at the next frame.
        m1_out, prev_state = self.model.m1_forward_and_init_state(x_seq[:, 0], y_seq[:, 0])
        start_t = 1

        return prev_state, start_t, m1_out

    def batch_step(self, data: dict, loss_f: torch.nn.Module) -> dict:
        accum_steps = int(self.exp.config.get('trainer.gradient_accumulation', 1))

        x_seq = data['image'].to(self.device)
        y_seq = data['label'].to(self.device)
        y_dist_seq = data.get('label_dist', None)
        if y_dist_seq is not None:
            y_dist_seq = y_dist_seq.to(self.device)

        if x_seq.ndim == 4:
            x_seq = x_seq.unsqueeze(1)
            y_seq = y_seq.unsqueeze(1)
            if y_dist_seq is not None:
                y_dist_seq = y_dist_seq.unsqueeze(1)

        B, T, C, H, W = x_seq.shape

        loss_val = 0
        loss_ret = {}
        batch_stat_sums = {}

        if self.accum_iter % accum_steps == 0:
            self.optimizer.zero_grad()

        use_t0_for_loss = self.exp.config.get('model.m1.use_t0_for_loss', False)
        prev_state, start_t, m1_out = self._init_prev_state(x_seq, y_seq)
        steps = 0

        if use_t0_for_loss and m1_out is not None:
            out = dict(m1_out)
            out["target_unpatched"] = y_seq[:, 0]
            if y_dist_seq is not None:
                out["target_dist"] = y_dist_seq[:, 0]

            if self.exp.config.get('trainer.use_amp', False):
                with torch.amp.autocast('cuda'):
                    l, l_dict = loss_f(**out)
            else:
                l, l_dict = loss_f(**out)

            loss_val += l
            steps += 1

            step_loss_ret = {}
            for k, v in l_dict.items():
                if k not in loss_ret:
                    loss_ret[k] = 0
                val = v.item() if isinstance(v, torch.Tensor) else v
                loss_ret[k] += val
                step_loss_ret[k] = val

            frame_data = {"image": x_seq[:, 0], "label": y_seq[:, 0]}
            self._log_batch_class_counts(frame_data, step_loss_ret)
            self._maybe_log_spike_batch(frame_data, out, step_loss_ret)

            for k, v in step_loss_ret.items():
                if k.startswith("batch_class_pixels/") or k.startswith("batch_class_frac/"):
                    if k not in batch_stat_sums:
                        batch_stat_sums[k] = 0.0
                    batch_stat_sums[k] += float(v)

        for t in range(start_t, T):
            x_t = x_seq[:, t]
            y_t = y_seq[:, t]

            out = self.model(x_t, y_t, prev_state=prev_state)
            out["target_unpatched"] = y_t
            if y_dist_seq is not None:
                out["target_dist"] = y_dist_seq[:, t]

            if self.exp.config.get('trainer.use_amp', False):
                with torch.amp.autocast('cuda'):
                    l, l_dict = loss_f(**out)
            else:
                l, l_dict = loss_f(**out)

            loss_val += l
            steps += 1

            step_loss_ret = {}
            for k, v in l_dict.items():
                if k not in loss_ret:
                    loss_ret[k] = 0
                val = v.item() if isinstance(v, torch.Tensor) else v
                loss_ret[k] += val
                step_loss_ret[k] = val

            frame_data = {"image": x_t, "label": y_t}
            self._log_batch_class_counts(frame_data, step_loss_ret)
            self._maybe_log_spike_batch(frame_data, out, step_loss_ret)

            for k, v in step_loss_ret.items():
                if k.startswith("batch_class_pixels/") or k.startswith("batch_class_frac/"):
                    if k not in batch_stat_sums:
                        batch_stat_sums[k] = 0.0
                    batch_stat_sums[k] += float(v)

            prev_state = out['final_state']

        if steps == 0:
            if not self._warned_no_supervision:
                print(
                    "[warm_start_m1init] No supervised steps in this batch: "
                    "M1 handled t=0 and there are no later frames for M2."
                )
                self._warned_no_supervision = True
            return {}

        loss_val = loss_val / steps
        loss_ret = {k: v / steps for k, v in loss_ret.items()}
        for k, v in batch_stat_sums.items():
            loss_ret[k] = v / steps

        if isinstance(loss_val, torch.Tensor) and loss_val.numel() > 1:
            loss_val = loss_val.mean()

        loss_val = loss_val / accum_steps

        track_grads = self.exp.config.get('experiment.logging.track_gradient_norm', False)
        normalize_grads = self.exp.config['trainer.normalize_gradients'] == "all"
        total_norm = 0.0

        if not isinstance(loss_val, torch.Tensor) or not loss_val.requires_grad:
            return loss_ret

        if self.exp.config.get('trainer.use_amp', False):
            self.scaler.scale(loss_val).backward()

            self.accum_iter += 1
            if self.accum_iter % accum_steps == 0:
                if normalize_grads or track_grads:
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
                    if not hasattr(self, 'epoch_grad_norm'):
                        self.epoch_grad_norm = []
                    self.epoch_grad_norm.append(total_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.exp.config['trainer.ema']:
                    self.ema.update()
        else:
            loss_val.backward()

            self.accum_iter += 1
            if self.accum_iter % accum_steps == 0:
                if normalize_grads or track_grads:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                if normalize_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if track_grads:
                    if not hasattr(self, 'epoch_grad_norm'):
                        self.epoch_grad_norm = []
                    self.epoch_grad_norm.append(total_norm)

                self.optimizer.step()
                if self.exp.config['trainer.ema']:
                    self.ema.update()

        return loss_ret

    @torch.no_grad()
    def test(self, loss_f, split='test', tag='test/img/', **kwargs):
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
            x_seq = data['image'].to(self.device)
            y_seq = data['label'].to(self.device)
            ids = data.get('id', [f"{i}"] * x_seq.shape[0])

            if x_seq.ndim == 4:
                x_seq = x_seq.unsqueeze(1)
                y_seq = y_seq.unsqueeze(1)

            B, T, C, H, W = x_seq.shape
            prev_state, start_t, m1_out = self._init_prev_state(x_seq, y_seq)

            batch_scores_time = {}
            vis_x = None
            vis_y = None
            vis_pred = None

            if m1_out is not None:
                eval_out = {}
                if 'probabilities' in m1_out:
                    eval_out['pred'] = m1_out['probabilities'].permute(0, 3, 1, 2)
                elif 'logits' in m1_out:
                    eval_out['pred'] = m1_out['logits'].permute(0, 3, 1, 2)
                eval_out['target'] = y_seq[:, 0]

                scores = loss_f(**eval_out)
                vis_x = x_seq[:, 0]
                vis_y = y_seq[:, 0]
                vis_pred = eval_out['pred']

                for k, v in scores.items():
                    if k not in batch_scores_time:
                        batch_scores_time[k] = []
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    batch_scores_time[k].append(val)

            for t in range(start_t, T):
                x_t = x_seq[:, t]
                y_t = y_seq[:, t]

                out = self.model(x_t, y_t, prev_state=prev_state)
                prev_state = out['final_state']

                eval_out = {}
                if 'probabilities' in out:
                    eval_out['pred'] = out['probabilities'].permute(0, 3, 1, 2)
                elif 'logits' in out:
                    eval_out['pred'] = out['logits'].permute(0, 3, 1, 2)
                eval_out['target'] = y_t

                scores = loss_f(**eval_out)

                vis_x = x_t
                vis_y = y_t
                vis_pred = eval_out['pred']

                for k, v in scores.items():
                    if k not in batch_scores_time:
                        batch_scores_time[k] = []
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    batch_scores_time[k].append(val)

            for metric_name, values in batch_scores_time.items():
                avg_score = np.mean(values)

                if metric_name not in loss_log:
                    loss_log[metric_name] = {}

                for b in range(B):
                    pid = ids[b] if b < len(ids) else f"unknown_{i}_{b}"
                    loss_log[metric_name][pid] = avg_score

            if i in save_img and vis_x is not None and vis_y is not None and vis_pred is not None:
                image_bhwc = vis_x.detach().cpu().permute(0, 2, 3, 1)
                pred_bhwc = vis_pred.detach().cpu().permute(0, 2, 3, 1)
                gt_bhwc = vis_y.detach().cpu().permute(0, 2, 3, 1)
                for b in range(B):
                    pid = ids[b] if b < len(ids) else f"unknown_{i}_{b}"
                    self.exp.write_img(
                        f"{tag}{pid}_{i}",
                        merge_img_label_gt_simplified(
                            image_bhwc[b:b+1],
                            pred_bhwc[b:b+1],
                            gt_bhwc[b:b+1],
                            is_rgb,
                        ),
                        self.exp.currentStep,
                    )

            if 'DiceScore' in batch_scores_time:
                pbar.set_postfix({'dice': np.mean(batch_scores_time['DiceScore'])})

        print(f"\n[{split.upper()} SCORES]")
        for metric, scores_dict in loss_log.items():
            if len(scores_dict) > 0:
                avg = np.mean(list(scores_dict.values()))
                print(f"  > {metric}: {avg:.4f}")
        print("--------------------------\n")

        self.model.train()
        return loss_log
