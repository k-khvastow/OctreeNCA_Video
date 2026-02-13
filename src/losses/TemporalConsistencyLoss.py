import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    r"""
    Penalises large changes in hidden-state activations between consecutive
    time steps.

    Given hidden states ``h_t`` (current) and ``h_prev`` (previous, detached),
    the loss is:

        L = mean( ||h_t - sg(h_prev)||^2 )

    where ``sg`` denotes stop-gradient.  This encourages the latent dynamics to
    evolve smoothly rather than making chaotic jumps, which is the primary
    driver of the quality degeneration observed in long-horizon NCA rollouts.

    The loss is applied to the **hidden channels only** (i.e. the part of the
    NCA state that is neither input nor logit channels).  It expects tensors in
    BHWC layout (as returned by the model's ``_pack_outputs``).

    Args:
        reduction: ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        hidden_t: torch.Tensor,
        hidden_prev: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Args:
            hidden_t:    current hidden state, shape (B, H, W, C_hidden) or (B, C_hidden, H, W).
            hidden_prev: previous hidden state (will be detached), same shape.

        Returns:
            (loss, {"temporal_consistency": loss_value})
        """
        target = hidden_prev.detach()
        if self.reduction == "mean":
            loss = F.mse_loss(hidden_t, target, reduction="mean")
        elif self.reduction == "sum":
            loss = F.mse_loss(hidden_t, target, reduction="sum")
        else:
            raise ValueError(f"Unknown reduction '{self.reduction}'.")

        return loss, {"temporal_consistency": loss.item()}
