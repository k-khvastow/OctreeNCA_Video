import torch

class OverflowLoss(torch.nn.Module):
    r"""
    OverflowLoss for NCA.
    Constrains hidden states to [-1, 1] to ensure stability.
    Optionally ignores output constraints for segmentation tasks.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    def forward(self, output: torch.Tensor = None, target: torch.Tensor = None, **kwargs):
        loss_ret = {}
        
        # 1. Hidden State Overflow (CRITICAL FOR STABILITY)
        # We must penalize hidden states that go beyond [-1, 1]
        if 'hidden_channels' in kwargs:
            hidden: torch.Tensor = kwargs['hidden_channels']
            hidden_overflow_loss = (hidden - torch.clip(hidden, -1.0, 1.0)).abs().mean()
            loss_ret['hidden'] = hidden_overflow_loss.item()
        else:
            # Fallback if hidden_channels aren't provided (though they should be)
            hidden_overflow_loss = 0.0

        # 2. Output Overflow (SKIP FOR SEGMENTATION)
        # For segmentation (logits/probs), we rely on CrossEntropy/Dice, not value clamping.
        # We define rgb_overflow_loss as 0 to avoid interfering with segmentation learning.
        rgb_overflow_loss = 0.0
        loss_ret['rgb'] = 0.0

        # Total loss
        loss = hidden_overflow_loss + rgb_overflow_loss

        return loss, loss_ret

class MaskedOverflowLoss(torch.nn.Module):
    r"""
    Masked version if you are using masks, adapted to ignore output clamping.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    def forward(self, **kwargs):
        loss_ret = {}
        
        # 1. Hidden State Overflow
        if 'loss_mask' in kwargs and 'hidden_channels' in kwargs:
            mask: torch.Tensor = kwargs['loss_mask']
            hidden: torch.Tensor = kwargs['hidden_channels']
            
            hidden_dist = (hidden - torch.clip(hidden, -1.0, 1.0)).abs()
            hidden_overflow_loss = (hidden_dist * mask).sum() / (mask.sum() + 1e-5)
            loss_ret['hidden'] = hidden_overflow_loss.item()
        else:
            hidden_overflow_loss = 0.0

        # 2. Output Overflow (Skipped)
        rgb_overflow_loss = 0.0
        loss_ret['rgb'] = 0.0

        loss = hidden_overflow_loss + rgb_overflow_loss
        return loss, loss_ret