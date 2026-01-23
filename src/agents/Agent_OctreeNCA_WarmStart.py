import torch
from src.agents.Agent_MedNCA_Simple import MedNCAAgent

class OctreeNCAWarmStartAgent(MedNCAAgent):
    """
    Agent for training OctreeNCA with temporal warm-start on video sequences.
    Expects data input to be (B, T, C, H, W).
    """
    def batch_step(self, data: dict, loss_f: torch.nn.Module) -> dict:
        # data['image_sequence']: (B, T, C, H, W)
        x_seq = data['image_sequence'].to(self.device)
        y_seq = data['label_sequence'].to(self.device)
        
        # Check dims
        if x_seq.ndim == 4: # If B, C, H, W (not sequence), unsqueeze
             x_seq = x_seq.unsqueeze(1)
             y_seq = y_seq.unsqueeze(1)

        B, T, C, H, W = x_seq.shape
        
        loss_val = 0
        loss_ret = {}
        prev_state = None
        
        self.optimizer.zero_grad()
        
        # Temporal Loop
        for t in range(T):
            x_t = x_seq[:, t] # (B, C, H, W)
            y_t = y_seq[:, t]
            
            # Forward Pass with State Injection
            out = self.model(x_t, y_t, prev_state=prev_state)
            
            # Accumulate Loss
            # We use the standard loss function, passing the dictionary 'out' which contains 'logits' and 'target'
            if self.exp.config.get('trainer.use_amp', False):
                 with torch.amp.autocast('cuda'):
                    l, l_dict = loss_f(**out)
            else:
                l, l_dict = loss_f(**out)
            
            loss_val += l
            
            # Aggregate logging metrics
            for k, v in l_dict.items():
                if k not in loss_ret: loss_ret[k] = 0
                val = v.item() if isinstance(v, torch.Tensor) else v
                loss_ret[k] += val
                
            # Pass state to next step
            # We use output 'final_state' from the model
            prev_state = out['final_state'] 

        # Average loss over time steps
        loss_val = loss_val / T
        loss_ret = {k: v / T for k, v in loss_ret.items()}
        
        # Backpropagation
        if self.exp.config.get('trainer.use_amp', False):
            self.scaler.scale(loss_val).backward()
            
            # Gradient Clipping / Norm
            if self.exp.config['trainer.normalize_gradients'] == "all":
                 self.scaler.unscale_(self.optimizer)
                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_val.backward()
            
            if self.exp.config['trainer.normalize_gradients'] == "all":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
            self.optimizer.step()
            
        if self.exp.config['trainer.ema']:
            self.ema.update()

        return loss_ret

    def test(self, loss_f, split='test', **kwargs):
        # Fallback to standard test (treats each frame independently or just runs cold start)
        # For true temporal eval, a specific test loop is needed, but for now we inherit base behavior
        # Note: Dataset for test must be compatible (e.g. standard Video2DDataset) 
        # or we assume 'val' split of Video2DSequentialDataset works here treating T as B in get_outputs if flattened.
        # However, MedNCAAgent.test is not implemented in the base class provided. 
        # Assuming the user runs validation via the standard loop or skips it.
        return {}