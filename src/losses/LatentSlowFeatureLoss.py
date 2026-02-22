import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSlowFeatureLoss(nn.Module):
    r"""
    Latent Slow Feature Analysis (SFA) regularizer.

    Implements:
      1) slow-change penalty      : ||h_t - sg(h_{t-1})||^2
      2) decorrelation constraint : off-diagonal covariance penalty on h_t

    The total loss is:
        L = L_slow + lambda_decorr * L_decorr
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = float(eps)

    @staticmethod
    def _to_observations(hidden: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden activations to 2D matrix [N_obs, C].

        Supports:
          - BHWC
          - BCHW
          - [N, C]
        """
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim != 4:
            raise ValueError(
                f"Expected hidden tensor with 2 or 4 dims, got shape {tuple(hidden.shape)}."
            )

        # Heuristic: channel-first tensors typically have the smallest second dim.
        is_bchw = (
            hidden.shape[1] <= hidden.shape[2]
            and hidden.shape[1] <= hidden.shape[3]
        )
        if is_bchw:
            hidden = hidden.permute(0, 2, 3, 1).contiguous()
        return hidden.reshape(-1, hidden.shape[-1])

    def _decorrelation_loss(self, hidden_t: torch.Tensor) -> torch.Tensor:
        z = self._to_observations(hidden_t)
        n_obs, n_channels = z.shape
        if n_obs < 2 or n_channels < 2:
            return hidden_t.new_tensor(0.0)

        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.t() @ z) / max(1, n_obs - 1)
        var = torch.diagonal(cov).clamp_min(self.eps)
        norm = torch.sqrt(var[:, None] * var[None, :])
        corr = cov / norm
        off_diag = corr - torch.diag_embed(torch.diagonal(corr))

        if self.reduction == "sum":
            return (off_diag * off_diag).sum()
        if self.reduction == "mean":
            return (off_diag * off_diag).mean()
        raise ValueError(f"Unknown reduction '{self.reduction}'.")

    def forward(
        self,
        hidden_t: torch.Tensor,
        hidden_prev: torch.Tensor,
        decorrelation_weight: float = 1.0,
        **kwargs,
    ) -> tuple:
        target = hidden_prev.detach()

        if self.reduction == "mean":
            slow_loss = F.mse_loss(hidden_t, target, reduction="mean")
        elif self.reduction == "sum":
            slow_loss = F.mse_loss(hidden_t, target, reduction="sum")
        else:
            raise ValueError(f"Unknown reduction '{self.reduction}'.")

        decorr_loss = self._decorrelation_loss(hidden_t)
        total = slow_loss + float(decorrelation_weight) * decorr_loss
        return total, {
            "slow": slow_loss.item(),
            "decorrelation": decorr_loss.item(),
            "total": total.item(),
        }
