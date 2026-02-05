import torch
import torch.nn.functional as F

def _to_channels_first(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        # BHWC -> BCHW if last dim looks like channels.
        if tensor.shape[-1] < tensor.shape[1]:
            return tensor.permute(0, 3, 1, 2)
    elif tensor.dim() == 5:
        # BHWDC -> BCDHW if last dim looks like channels.
        if tensor.shape[-1] < tensor.shape[1]:
            return tensor.permute(0, 4, 1, 2, 3)
    return tensor

def soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    """Soft Tversky score (multiclass-friendly)."""
    assert output.size() == target.size()

    if dims is not None:
        output_sum = torch.sum(output, dim=dims)
        target_sum = torch.sum(target, dim=dims)
        difference = (output - target).abs().sum(dim=dims)
    else:
        output_sum = torch.sum(output)
        target_sum = torch.sum(target)
        difference = (output - target).abs().sum()

    intersection = (output_sum + target_sum - difference) / 2.0  # TP
    fp = output_sum - intersection
    fn = target_sum - intersection

    tversky_score = (intersection + smooth) / (
        intersection + alpha * fp + beta * fn + smooth
    ).clamp_min(eps)
    return tversky_score

class DiceLoss(torch.nn.Module):
    r"""Dice Loss
    """
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceLoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.tensor, target: torch.tensor, smooth: float = 1, binarize=False) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)
        if binarize:
            input = torch.round(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceLoss_mask(torch.nn.Module):
    r"""Dice Loss mask, that only calculates on masked values
    """
    def __init__(self, useSigmoid = True) -> None:
        r"""Initialisation method of DiceLoss mask
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss_mask, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, smooth: float=1) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
                mask: The mask which defines which values to consider
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        mask = torch.flatten(mask)

        input = input[~mask]  
        target = target[~mask]  
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceBCELoss(torch.nn.Module):
    r"""Dice BCE Loss
    """
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)
        input = torch.flatten(input) 
        target = torch.flatten(target)
        
        intersection = (input * target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class BCELoss(torch.nn.Module):
    r"""BCE Loss
    """
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)

        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        return BCE

class FocalLoss(torch.nn.Module):
    r"""Multi-class Focal Loss for segmentation (logits + class indices)."""
    def __init__(self, gamma: float = 2.0, alpha=None, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # None, scalar, or tensor/list of shape [C]
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor = None, target: torch.Tensor = None, logits=None, x=None, y=None, **kwargs) -> torch.Tensor:
        # Support common wrapper kwargs: logits/target or x/y.
        if input is None:
            input = logits if logits is not None else x
        if target is None:
            target = y

        # input: [B, C, H, W], target: [B, H, W] (class indices) or one-hot [B, C, H, W]
        if target.dtype in (torch.float16, torch.float32) and target.dim() == 4:
            target = torch.argmax(target, dim=1)

        log_probs = F.log_softmax(input, dim=1)  # [B, C, H, W]
        probs = log_probs.exp()

        # gather p_t and log_p_t
        target = target.unsqueeze(1)  # [B, 1, H, W]
        log_p_t = log_probs.gather(1, target).squeeze(1)  # [B, H, W]
        p_t = probs.gather(1, target).squeeze(1)

        # mask ignore_index
        if self.ignore_index is not None:
            valid = (target.squeeze(1) != self.ignore_index)
            log_p_t = log_p_t[valid]
            p_t = p_t[valid]
        else:
            valid = None

        focal = -(1 - p_t) ** self.gamma * log_p_t

        if self.alpha is not None:
            if not torch.is_tensor(self.alpha):
                alpha = torch.tensor(self.alpha, device=input.device, dtype=input.dtype)
            else:
                alpha = self.alpha.to(input.device, input.dtype)
            # alpha per pixel based on target class
            alpha_t = alpha.gather(0, target.squeeze(1).clamp_min(0))
            if valid is not None:
                alpha_t = alpha_t[valid]
            focal = focal * alpha_t

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class TverskyLoss(torch.nn.Module):
    r"""Multiclass Tversky Loss for segmentation.

    Expects logits/probabilities shaped [B, C, H, W] or [B, C, H, W, D] (channels first),
    or [B, H, W, C] / [B, H, W, D, C] (channels last). Targets can be class indices
    [B, H, W] / [B, H, W, D] or one-hot with the same shape as logits.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        from_logits: bool = True,
        log_loss: bool = False,
        smooth: float = 0.0,
        eps: float = 1e-7,
        ignore_index: int = None,
        classes=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.classes = classes
        self.reduction = reduction

    def _prepare_target(self, target: torch.Tensor, num_classes: int, input_dim: int):
        valid_mask = None
        is_one_hot = (
            target.dim() == input_dim
            and (target.shape[1] == num_classes or target.shape[-1] == num_classes)
        )

        if is_one_hot:
            # One-hot / soft target.
            target = _to_channels_first(target)
            if target.shape[1] != num_classes and target.shape[-1] == num_classes:
                if target.dim() == 4:
                    target = target.permute(0, 3, 1, 2)
                else:
                    target = target.permute(0, 4, 1, 2, 3)
        else:
            # Class indices (optionally with a singleton channel dim).
            if target.dim() == input_dim and target.shape[1] == 1:
                target = target.squeeze(1)
            elif target.dim() == input_dim and target.shape[-1] == 1:
                target = target.squeeze(-1)

            target = target.long()
            if self.ignore_index is not None:
                valid_mask = target != self.ignore_index
                target = target.clone()
                target[~valid_mask] = 0
            target = F.one_hot(target, num_classes=num_classes)
            if target.dim() == 4:
                target = target.permute(0, 3, 1, 2)
            else:
                target = target.permute(0, 4, 1, 2, 3)

        return target, valid_mask

    def forward(self, input: torch.Tensor = None, target: torch.Tensor = None, logits=None, x=None, y=None, **kwargs) -> torch.Tensor:
        # Support common wrapper kwargs: logits/target or x/y.
        if input is None:
            input = logits if logits is not None else x
        if target is None:
            target = y
        if input is None or target is None:
            raise ValueError("TverskyLoss requires input/logits and target.")

        input = _to_channels_first(input)
        input_dim = input.dim()
        num_classes = input.shape[1]

        if self.from_logits:
            probs = F.softmax(input, dim=1)
        else:
            probs = input

        target, valid_mask = self._prepare_target(target, num_classes, input_dim)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).to(dtype=probs.dtype)
            probs = probs * mask
            target = target * mask

        if self.classes is not None:
            probs = probs[:, self.classes]
            target = target[:, self.classes]

        dims = tuple([0] + list(range(2, probs.dim())))
        score = soft_tversky_score(
            probs, target, self.alpha, self.beta, self.smooth, self.eps, dims
        )

        if self.log_loss:
            loss = -torch.log(score.clamp_min(self.eps))
        else:
            loss = 1.0 - score

        if self.gamma != 1.0:
            loss = loss ** self.gamma

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceFocalLoss(FocalLoss):
    r"""Dice Focal Loss
    """
    def __init__(self, gamma: float = 2, eps: float = 1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + 1.)/(input.sum() + target.sum() + 1.)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        focal = loss_bce * (1 - logit) ** self.gamma  # focal loss
        dice_focal = focal.mean() + dice_loss
        return dice_focal
    
class WeightedDiceBCELoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        input = torch.sigmoid(input)
        input_flat = torch.flatten(input)
        target_flat = torch.flatten(target)
        variance_flat = 1 - 2*torch.flatten(variance)  # Assuming variance is already prepared for weighting

        # Weighted intersection for Dice
        weighted_intersection = (input_flat * target_flat * variance_flat).sum()
        weighted_input_sum = (input_flat * variance_flat).sum()
        weighted_target_sum = (target_flat * variance_flat).sum()
        
        dice_loss = 1 - (2. * weighted_intersection + 1.) / (weighted_input_sum + weighted_target_sum + 1.)

        # BCE Loss with variance weighting
        bce_loss = torch.nn.functional.binary_cross_entropy(input_flat, target_flat, weight=variance_flat, reduction='mean')

        # Combining Dice and BCE losses
        total_loss = bce_loss + dice_loss

        return total_loss
    


class DiceFocalLoss_2(FocalLoss):
    r"""Dice Focal Loss
    """
    def __init__(self, gamma: float = 2, eps: float = 1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(DiceFocalLoss_2, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        target = torch.sigmoid(target).detach()
        target = torch.round(target)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + 1.)/(input.sum() + target.sum() + 1.)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        focal = loss_bce * (1 - logit) ** self.gamma  # focal loss
        dice_focal = focal.mean() + dice_loss
        return dice_focal

import einops

class CrossEntropyLossWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits=None, target=None, x=None, y=None, **kwargs):
         if logits is None: logits = x
         if target is None: target = y
         
         # Heuristic to detect BHWC vs BCHW
         # Assume Channels C is much smaller than Spatial H, W.
         # BHWC: last dim is C. BCHW: dim 1 is C.
         
         # Permute logits if BHWC
         if logits.dim() == 4:
             # Check if last dim is likely channels (e.g. < dim 1)
             if logits.shape[-1] < logits.shape[1]: 
                  logits = einops.rearrange(logits, 'b h w c -> b c h w')

         # Handle one-hot target
         if target.dtype == torch.float32 or target.dtype == torch.float16:
              if target.dim() == 4:
                   # Check if last dim is likely channels (BHWC)
                   if target.shape[-1] < target.shape[1]:
                       target = torch.argmax(target, dim=-1)
                   else:
                       # Assume BCHW
                       target = torch.argmax(target, dim=1)
                   
         return self.loss(logits, target)
