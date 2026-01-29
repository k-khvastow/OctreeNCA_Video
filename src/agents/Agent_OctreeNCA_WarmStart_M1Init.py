from typing import Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from src.agents.Agent_MedNCA_Simple import MedNCAAgent


class OctreeNCAWarmStartM1InitAgent(MedNCAAgent):
    """
    Warm-start agent with M1 initialization on the first frame.
    Expects data input to be (B, T, C, H, W).
    """
    def __init__(self, model):
        super().__init__(model)
        self.accum_iter = 0

    def _init_prev_state(self, x_seq: torch.Tensor, y_seq: torch.Tensor) -> Tuple[Optional[torch.Tensor], int]:
        use_m1 = self.exp.config.get('model.m1.use_first_frame', True)
        use_t0_for_loss = self.exp.config.get('model.m1.use_t0_for_loss', False)

        if not use_m1 or x_seq.shape[1] == 0:
            return None, 0

        prev_state = self.model.init_state_from_m1(x_seq[:, 0], y_seq[:, 0])
        start_t = 0
        if not use_t0_for_loss and x_seq.shape[1] > 1:
            start_t = 1

        return prev_state, start_t

    def batch_step(self, data: dict, loss_f: torch.nn.Module) -> dict:
        accum_steps = int(self.exp.config.get('trainer.gradient_accumulation', 1))

        x_seq = data['image'].to(self.device)
        y_seq = data['label'].to(self.device)

        if x_seq.ndim == 4:
            x_seq = x_seq.unsqueeze(1)
            y_seq = y_seq.unsqueeze(1)

        B, T, C, H, W = x_seq.shape

        loss_val = 0
        loss_ret = {}

        if self.accum_iter % accum_steps == 0:
            self.optimizer.zero_grad()

        prev_state, start_t = self._init_prev_state(x_seq, y_seq)
        steps = max(1, T - start_t)

        for t in range(start_t, T):
            x_t = x_seq[:, t]
            y_t = y_seq[:, t]

            out = self.model(x_t, y_t, prev_state=prev_state)

            if self.exp.config.get('trainer.use_amp', False):
                with torch.amp.autocast('cuda'):
                    l, l_dict = loss_f(**out)
            else:
                l, l_dict = loss_f(**out)

            loss_val += l

            for k, v in l_dict.items():
                if k not in loss_ret:
                    loss_ret[k] = 0
                val = v.item() if isinstance(v, torch.Tensor) else v
                loss_ret[k] += val

            prev_state = out['final_state']

        loss_val = loss_val / steps
        loss_ret = {k: v / steps for k, v in loss_ret.items()}

        if isinstance(loss_val, torch.Tensor) and loss_val.numel() > 1:
            loss_val = loss_val.mean()

        loss_val = loss_val / accum_steps

        track_grads = self.exp.config.get('experiment.logging.track_gradient_norm', False)
        normalize_grads = self.exp.config['trainer.normalize_gradients'] == "all"
        total_norm = 0.0

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
        loss_log = {}

        pbar = tqdm(loader, desc=f"Eval {split}")

        for i, data in enumerate(pbar):
            x_seq = data['image'].to(self.device)
            y_seq = data['label'].to(self.device)
            ids = data.get('id', [f"{i}"] * x_seq.shape[0])

            if x_seq.ndim == 4:
                x_seq = x_seq.unsqueeze(1)
                y_seq = y_seq.unsqueeze(1)

            B, T, C, H, W = x_seq.shape
            prev_state, start_t = self._init_prev_state(x_seq, y_seq)

            batch_scores_time = {}

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
