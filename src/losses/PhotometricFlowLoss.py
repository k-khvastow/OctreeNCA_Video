"""
Self-supervised photometric flow loss + spatial smoothness regularization.

This loss does NOT require ground-truth optical flow.  It uses the
photometric consistency assumption: warping the previous frame with the
predicted flow should reconstruct the current frame.

Loss components:
  1. **Photometric L1**: |I_t − warp(I_{t-1}, flow)|
  2. **SSIM term** (optional): structural similarity between warped and current
  3. **Smoothness**: first-order spatial gradient penalty on the flow field

All losses are averaged over the spatial dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotometricFlowLoss(nn.Module):
    """Self-supervised optical flow loss for dual-view warm-start NCA.

    Computes photometric consistency between the warped previous image
    and the current image, plus a smoothness penalty on the flow.

    Args:
        photometric_weight: Weight for the L1 photometric term.
        ssim_weight: Weight for the SSIM photometric term (0 to disable).
        smoothness_weight: Weight for the spatial smoothness penalty.
        edge_aware_smoothness: If True, weight smoothness by image gradients
            (smaller penalty near edges where flow discontinuities are expected).
    """

    def __init__(
        self,
        photometric_weight: float = 1.0,
        ssim_weight: float = 0.0,
        smoothness_weight: float = 0.01,
        edge_aware_smoothness: bool = True,
    ):
        super().__init__()
        self.photometric_weight = photometric_weight
        self.ssim_weight = ssim_weight
        self.smoothness_weight = smoothness_weight
        self.edge_aware_smoothness = edge_aware_smoothness

    # ── Warping ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def _warp(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image (BCHW) by flow (B,2,H,W) in pixel units."""
        B, _, H, W = img.shape
        base = self._make_base_grid(H, W, img.device, img.dtype)
        flow_norm = torch.stack([
            flow[:, 0] / (W / 2.0),
            flow[:, 1] / (H / 2.0),
        ], dim=-1)
        grid = base + flow_norm
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

    # ── SSIM ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 3) -> torch.Tensor:
        """Compute mean SSIM between x and y (both BCHW). Returns per-pixel map."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = window_size // 2

        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

        sigma_x2 = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x ** 2
        sigma_y2 = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2))
        return ssim_map.clamp(0, 1)

    # ── Smoothness ───────────────────────────────────────────────────────

    @staticmethod
    def _flow_smoothness(flow: torch.Tensor, image: torch.Tensor = None, edge_aware: bool = True) -> torch.Tensor:
        """First-order spatial smoothness loss on flow (B,2,H,W).

        If edge_aware, down-weight smoothness near strong image gradients.
        """
        flow_dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # (B, 2, H, W-1)
        flow_dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # (B, 2, H-1, W)

        if edge_aware and image is not None:
            img_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)
            img_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)
            weight_x = torch.exp(-img_dx)
            weight_y = torch.exp(-img_dy)
            flow_dx = flow_dx * weight_x
            flow_dy = flow_dy * weight_y

        return flow_dx.mean() + flow_dy.mean()

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        flow_a: torch.Tensor,
        flow_b: torch.Tensor,
        prev_img_a: torch.Tensor,
        prev_img_b: torch.Tensor,
        current_img_a: torch.Tensor,
        current_img_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute flow losses for both views.

        Args:
            flow_a, flow_b: Predicted flow fields (B, 2, H, W) in pixel units.
            prev_img_a, prev_img_b: Previous frame images (B, C, H, W).
            current_img_a, current_img_b: Current frame images (B, C, H, W).

        Returns:
            (total_loss, loss_dict) where loss_dict contains named components.
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=flow_a.device, dtype=flow_a.dtype)

        # ── Photometric L1 ───────────────────────────────────────────────
        warped_a = self._warp(prev_img_a, flow_a)
        warped_b = self._warp(prev_img_b, flow_b)

        photo_a = F.l1_loss(warped_a, current_img_a)
        photo_b = F.l1_loss(warped_b, current_img_b)
        photometric_loss = (photo_a + photo_b) / 2.0

        if self.photometric_weight > 0:
            total_loss = total_loss + self.photometric_weight * photometric_loss
        loss_dict["FlowLoss/photometric"] = photometric_loss.item()

        # ── SSIM ─────────────────────────────────────────────────────────
        if self.ssim_weight > 0:
            ssim_a = self._ssim(warped_a, current_img_a).mean()
            ssim_b = self._ssim(warped_b, current_img_b).mean()
            ssim_loss = 1.0 - (ssim_a + ssim_b) / 2.0
            total_loss = total_loss + self.ssim_weight * ssim_loss
            loss_dict["FlowLoss/ssim"] = ssim_loss.item()

        # ── Smoothness ───────────────────────────────────────────────────
        if self.smoothness_weight > 0:
            smooth_a = self._flow_smoothness(flow_a, current_img_a, self.edge_aware_smoothness)
            smooth_b = self._flow_smoothness(flow_b, current_img_b, self.edge_aware_smoothness)
            smooth_loss = (smooth_a + smooth_b) / 2.0
            total_loss = total_loss + self.smoothness_weight * smooth_loss
            loss_dict["FlowLoss/smoothness"] = smooth_loss.item()

        # ── Flow magnitude statistics (for monitoring) ───────────────────
        with torch.no_grad():
            flow_mag = torch.cat([
                flow_a.norm(dim=1).mean().unsqueeze(0),
                flow_b.norm(dim=1).mean().unsqueeze(0),
            ]).mean()
            loss_dict["FlowLoss/mean_flow_magnitude"] = flow_mag.item()
            loss_dict["FlowLoss/max_flow_magnitude"] = max(
                flow_a.norm(dim=1).max().item(),
                flow_b.norm(dim=1).max().item(),
            )

        loss_dict["FlowLoss/total"] = total_loss.item()
        return total_loss, loss_dict
