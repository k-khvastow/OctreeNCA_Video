"""
Flow-augmented Dual-View Warm-Start OctreeNCA.

Extends OctreeNCA2DDualViewWarmStart with:
  - A lightweight flow head that predicts per-pixel 2D displacement (dx, dy)
    from hidden channels.
  - Spatial warping of the previous state before NCA refinement.
  - Photometric self-supervised loss output for training.
  - Optional flow-magnitude conditioning on the temporal gate.

The flow head is zero-initialized so the model starts identical to the
non-flow warm-start variant and gradually learns motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Model_OctreeNCA_2d_dual_view_warm import OctreeNCA2DDualViewWarmStart


class OctreeNCA2DDualViewWarmStartFlow(OctreeNCA2DDualViewWarmStart):
    """Warm-start dual-view OctreeNCA with learned optical flow warping."""

    def __init__(self, config: dict):
        super().__init__(config)

        hidden_dim = self.channel_n - self.input_channels - self.output_channels
        self.flow_enabled = bool(config.get("model.flow.enabled", True))
        self.flow_weight = float(config.get("model.flow.loss_weight", 0.1))
        self.flow_smoothness_weight = float(config.get("model.flow.smoothness_weight", 0.01))
        self.flow_warp_state = bool(config.get("model.flow.warp_state", True))
        self.flow_condition_gate = bool(config.get("model.flow.condition_gate", False))

        if self.flow_enabled and hidden_dim > 0:
            # Flow head: hidden channels → 2 (dx, dy)
            # Zero-initialized so it starts as identity (no warping).
            self.flow_head = nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True)
            with torch.no_grad():
                self.flow_head.weight.zero_()
                self.flow_head.bias.zero_()

            # Optional: augment the temporal gate input with flow magnitude
            if self.flow_condition_gate and self.temporal_gate_mode == "gru":
                # Rebuild the GRU gate with 1 extra input channel (flow mag)
                old_out = self.temporal_gate.out_channels
                self.temporal_gate = nn.Conv2d(
                    self.channel_n * 2 + 1, old_out, kernel_size=1, bias=True
                )
                with torch.no_grad():
                    self.temporal_gate.weight.zero_()
                    self.temporal_gate.bias.fill_(-2.0)
        else:
            self.flow_head = None

    # ── Flow utilities ────────────────────────────────────────────────────

    @staticmethod
    def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create a normalized identity sampling grid of shape (1, H, W, 2)."""
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    def _predict_flow(self, state: torch.Tensor) -> torch.Tensor:
        """Predict flow field from hidden channels of a BCHW state.

        Returns flow in pixel units (not normalized), shape (B, 2, H, W).
        """
        hidden_start = self.input_channels + self.output_channels
        hidden = state[:, hidden_start:]
        return self.flow_head(hidden)  # (B, 2, H, W)

    def _warp_with_flow(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp tensor *x* (BCHW) by *flow* (B, 2, H, W) in pixel units.

        Uses bilinear sampling with zero padding for out-of-bounds regions.
        """
        B, _, H, W = x.shape
        base_grid = self._make_base_grid(H, W, x.device, x.dtype)  # (1, H, W, 2)

        # Convert pixel-unit flow to normalized [-1, 1] displacements
        # flow[:, 0] = dx (horizontal), flow[:, 1] = dy (vertical)
        flow_norm = torch.stack([
            flow[:, 0] / (W / 2.0),   # dx normalized
            flow[:, 1] / (H / 2.0),   # dy normalized
        ], dim=-1)  # (B, H, W, 2)

        grid = base_grid + flow_norm  # (B, H, W, 2)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # ── Override warm-start forward ──────────────────────────────────────

    def _forward_warm_single_scale(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        prev_state_a: torch.Tensor,
        prev_state_b: torch.Tensor,
    ):
        """Warm-start forward with optional flow-based warping.

        Pipeline:
          1. Predict flow from previous hidden state
          2. Warp previous state using predicted flow
          3. Run standard NCA warm-start on the warped state
          4. Return flow fields + warped previous images for loss computation
        """
        input_ch = self.input_channels
        hidden_start = input_ch + self.output_channels

        # ── 1. Flow prediction ────────────────────────────────────────────
        flow_a = None
        flow_b = None
        if self.flow_enabled and self.flow_head is not None:
            flow_a = self._predict_flow(prev_state_a)  # (B, 2, H, W)
            flow_b = self._predict_flow(prev_state_b)

        # ── 2. Warp previous state ───────────────────────────────────────
        oc = self._orig_input_channels
        if flow_a is not None and self.flow_warp_state:
            # Keep a copy of the unwarped previous image for photometric loss
            prev_img_a = prev_state_a[:, :oc].clone()
            prev_img_b = prev_state_b[:, :oc].clone()

            warped_state_a = self._warp_with_flow(prev_state_a, flow_a)
            warped_state_b = self._warp_with_flow(prev_state_b, flow_b)
        else:
            prev_img_a = prev_state_a[:, :oc].clone()
            prev_img_b = prev_state_b[:, :oc].clone()
            warped_state_a = prev_state_a
            warped_state_b = prev_state_b

        # ── 3. Standard NCA warm-start on (possibly warped) state ────────
        state_a = warped_state_a.clone()
        state_b = warped_state_b.clone()

        # Reset spatial-only hidden channels before injecting new inputs
        state_a = self._reset_spatial_channels(state_a)
        state_b = self._reset_spatial_channels(state_b)

        # Inject current image (and per-pixel diff from previous frame)
        self._inject_image_and_diff(state_a, x_a, prev_state=prev_state_a)
        self._inject_image_and_diff(state_b, x_b, prev_state=prev_state_b)

        state_a = self._init_logits_for_warm_start_bchw(state_a)
        state_b = self._init_logits_for_warm_start_bchw(state_b)
        state_a = self._stabilize_hidden_state_bchw(state_a)
        state_b = self._stabilize_hidden_state_bchw(state_b)

        # Inject noise into hidden channels (training-time only, annealed)
        state_a = self._inject_hidden_noise(state_a)
        state_b = self._inject_hidden_noise(state_b)

        # Snapshot hidden channels BEFORE the backbone update for contractive reg
        pre_update_hidden_ab = torch.cat(
            [state_a[:, hidden_start:], state_b[:, hidden_start:]], dim=0
        )
        if self.training:
            pre_update_hidden_ab.requires_grad_(True)
            state_a = torch.cat([state_a[:, :hidden_start], pre_update_hidden_ab[:state_a.shape[0]]], dim=1)
            state_b = torch.cat([state_b[:, :hidden_start], pre_update_hidden_ab[state_a.shape[0]:]], dim=1)

        steps = int(self.warm_start_steps)
        state_ab = torch.cat([state_a, state_b], dim=0)
        state_ab = self._run_backbone_with_steps(state_ab, level=0, steps=steps)
        cand_state_a, cand_state_b = state_ab.chunk(2, dim=0)
        cand_state_a, cand_state_b = self._maybe_cross_fuse(cand_state_a, cand_state_b, level=0)

        # ── 4. Temporal gate (optionally conditioned on flow) ────────────
        if self.temporal_gate_mode == "gru":
            if self.flow_condition_gate and flow_a is not None:
                flow_mag_a = flow_a.norm(dim=1, keepdim=True)  # (B, 1, H, W)
                flow_mag_b = flow_b.norm(dim=1, keepdim=True)
                gate_in_a = torch.cat([warped_state_a, cand_state_a, flow_mag_a], dim=1)
                gate_in_b = torch.cat([warped_state_b, cand_state_b, flow_mag_b], dim=1)
            else:
                gate_in_a = torch.cat([warped_state_a, cand_state_a], dim=1)
                gate_in_b = torch.cat([warped_state_b, cand_state_b], dim=1)
            z_a = torch.sigmoid(self.temporal_gate(gate_in_a))
            z_b = torch.sigmoid(self.temporal_gate(gate_in_b))
            state_a = (1.0 - z_a) * warped_state_a + z_a * cand_state_a
            state_b = (1.0 - z_b) * warped_state_b + z_b * cand_state_b
        elif self.temporal_gate_mode == "simple":
            feat_a = torch.cat([cand_state_a[:, :input_ch], cand_state_a[:, hidden_start:]], dim=1)
            feat_b = torch.cat([cand_state_b[:, :input_ch], cand_state_b[:, hidden_start:]], dim=1)
            z_a = torch.sigmoid(self.temporal_gate(feat_a))
            z_b = torch.sigmoid(self.temporal_gate(feat_b))
            state_a = cand_state_a.clone()
            state_b = cand_state_b.clone()
            state_a[:, hidden_start:] = (1.0 - z_a) * warped_state_a[:, hidden_start:] + z_a * cand_state_a[:, hidden_start:]
            state_b[:, hidden_start:] = (1.0 - z_b) * warped_state_b[:, hidden_start:] + z_b * cand_state_b[:, hidden_start:]
        else:
            state_a = cand_state_a
            state_b = cand_state_b

        # Keep current-frame input channels exact after temporal blending
        self._inject_image_and_diff(state_a, x_a, prev_state=prev_state_a)
        self._inject_image_and_diff(state_b, x_b, prev_state=prev_state_b)

        state_a = self._stabilize_hidden_state_bchw(state_a)
        state_b = self._stabilize_hidden_state_bchw(state_b)

        # Post-update hidden channels for contractive regularization
        post_update_hidden_ab = torch.cat(
            [state_a[:, hidden_start:], state_b[:, hidden_start:]], dim=0
        )

        return (
            state_a,
            state_b,
            pre_update_hidden_ab,
            post_update_hidden_ab,
            flow_a,
            flow_b,
            prev_img_a,
            prev_img_b,
        )

    def _pack_outputs(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        pre_update_hidden: torch.Tensor = None,
        post_update_hidden: torch.Tensor = None,
        flow_a: torch.Tensor = None,
        flow_b: torch.Tensor = None,
        prev_img_a: torch.Tensor = None,
        prev_img_b: torch.Tensor = None,
        x_a: torch.Tensor = None,
        x_b: torch.Tensor = None,
    ):
        """Pack model outputs including flow fields and warped images."""
        ret_dict = super()._pack_outputs(
            state_a, state_b, y_a, y_b,
            pre_update_hidden=pre_update_hidden,
            post_update_hidden=post_update_hidden,
        )

        # Attach flow outputs for loss computation
        if flow_a is not None:
            ret_dict["flow_a"] = flow_a  # (B, 2, H, W)
            ret_dict["flow_b"] = flow_b
        if prev_img_a is not None:
            ret_dict["prev_img_a"] = prev_img_a  # (B, C, H, W) — previous frame's image
            ret_dict["prev_img_b"] = prev_img_b
        if x_a is not None:
            ret_dict["current_img_a"] = x_a[:, :self.input_channels]  # (B, C, H, W)
            ret_dict["current_img_b"] = x_b[:, :self.input_channels]

        return ret_dict

    def forward_train(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: torch.Tensor = None,
        y_b: torch.Tensor = None,
        prev_state_a: torch.Tensor = None,
        prev_state_b: torch.Tensor = None,
        batch_duplication=1,
    ):
        if y_a is None:
            y_a = torch.zeros(
                (x_a.shape[0], self.output_channels, x_a.shape[2], x_a.shape[3]),
                device=x_a.device, dtype=x_a.dtype,
            )
        if y_b is None:
            y_b = torch.zeros(
                (x_b.shape[0], self.output_channels, x_b.shape[2], x_b.shape[3]),
                device=x_b.device, dtype=x_b.dtype,
            )

        prev_state_a = self._to_bchw_state(prev_state_a, "prev_state_a")
        prev_state_b = self._to_bchw_state(prev_state_b, "prev_state_b")
        if (prev_state_a is None) ^ (prev_state_b is None):
            raise ValueError("prev_state_a and prev_state_b must both be provided or both be None.")

        if batch_duplication != 1:
            x_a = torch.cat([x_a] * batch_duplication, dim=0)
            x_b = torch.cat([x_b] * batch_duplication, dim=0)
            y_a = torch.cat([y_a] * batch_duplication, dim=0)
            y_b = torch.cat([y_b] * batch_duplication, dim=0)
            if prev_state_a is not None:
                prev_state_a = torch.cat([prev_state_a] * batch_duplication, dim=0)
                prev_state_b = torch.cat([prev_state_b] * batch_duplication, dim=0)

        flow_a = None
        flow_b = None
        prev_img_a = None
        prev_img_b = None
        pre_update_hidden = None
        post_update_hidden = None

        if prev_state_a is None:
            # Cold start — no flow
            state_a, state_b = self._forward_cold(x_a, x_b)
        else:
            if prev_state_a.shape[2:4] != tuple(self.octree_res[0]):
                prev_state_a = F.interpolate(prev_state_a, size=self.octree_res[0], mode="nearest")
            if prev_state_b.shape[2:4] != tuple(self.octree_res[0]):
                prev_state_b = F.interpolate(prev_state_b, size=self.octree_res[0], mode="nearest")

            if self.warm_start_multiscale:
                # Multiscale path does not support flow (yet)
                state_a, state_b = self._forward_warm_multiscale(x_a, x_b, prev_state_a, prev_state_b)
            else:
                (
                    state_a, state_b,
                    pre_update_hidden, post_update_hidden,
                    flow_a, flow_b,
                    prev_img_a, prev_img_b,
                ) = self._forward_warm_single_scale(x_a, x_b, prev_state_a, prev_state_b)

        return self._pack_outputs(
            state_a, state_b, y_a, y_b,
            pre_update_hidden=pre_update_hidden,
            post_update_hidden=post_update_hidden,
            flow_a=flow_a,
            flow_b=flow_b,
            prev_img_a=prev_img_a,
            prev_img_b=prev_img_b,
            x_a=x_a,
            x_b=x_b,
        )
