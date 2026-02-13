import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Model_OctreeNCA_2d_dual_view import OctreeNCA2DDualView


class OctreeNCA2DDualViewWarmStart(OctreeNCA2DDualView):
    """
    Temporal dual-view OctreeNCA with hidden-state warm start.

    - Keeps one recurrent state per view (A/B) across frames.
    - Injects current image channels into the carried state.
    - Supports optional logits reset/gating and hidden stabilization.
    - Returns final states for the next frame rollout.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.warm_start_steps = config.get("model.octree.warm_start_steps", self.inference_steps[0])
        self.warm_start_multiscale = bool(config.get("model.octree.warm_start_multiscale", False))
        self.warm_start_multiscale_steps = config.get("model.octree.warm_start_multiscale_steps", None)
        self.warm_start_multiscale_start_level = config.get(
            "model.octree.warm_start_multiscale_start_level", len(self.octree_res) - 1
        )
        self.warm_start_multiscale_downsample_mode = str(
            config.get("model.octree.warm_start_multiscale_downsample_mode", "nearest")
        )

        self.warm_start_hidden_norm = str(config.get("model.octree.warm_start_hidden_norm", "none")).lower()
        self.warm_start_hidden_clip = config.get("model.octree.warm_start_hidden_clip", None)
        self.warm_start_hidden_tanh_scale = config.get("model.octree.warm_start_hidden_tanh_scale", None)
        self.warm_start_hidden_gn_groups = int(config.get("model.octree.warm_start_hidden_gn_groups", 8))

        # Hidden noise injection (scheduled sampling for latent dynamics)
        self.warm_start_hidden_noise_std = float(
            config.get("model.octree.warm_start_hidden_noise_std", 0.0)
        )
        self.warm_start_hidden_noise_anneal_epochs = int(
            config.get("model.octree.warm_start_hidden_noise_anneal_epochs", 0)
        )
        # Will be set by the agent at each epoch to control annealing
        self._current_epoch = 0

        hidden_dim = self.channel_n - self.input_channels - self.output_channels
        self._warm_hidden_ln = None
        self._warm_hidden_gn = None
        if hidden_dim > 0:
            if self.warm_start_hidden_norm == "layer":
                self._warm_hidden_ln = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            elif self.warm_start_hidden_norm == "group":
                groups = max(1, int(self.warm_start_hidden_gn_groups))
                if hidden_dim % groups != 0:
                    groups = 1
                self._warm_hidden_gn = nn.GroupNorm(groups, hidden_dim, affine=True)
            elif self.warm_start_hidden_norm in ("none", "", None):
                pass
            else:
                raise ValueError(
                    "model.octree.warm_start_hidden_norm must be one of: 'none', 'layer', 'group'."
                )

        self.warm_start_logits_mode = str(config.get("model.octree.warm_start_logits_mode", "carry")).lower()
        self.warm_start_logits_gate_from = str(
            config.get("model.octree.warm_start_logits_gate_from", "hidden")
        ).lower()
        self._logits_gate_conv = None
        if self.warm_start_logits_mode == "gate":
            if self.warm_start_logits_gate_from == "hidden":
                gate_in = hidden_dim
            elif self.warm_start_logits_gate_from == "state":
                gate_in = self.channel_n
            elif self.warm_start_logits_gate_from in ("hidden+input", "input+hidden"):
                gate_in = hidden_dim + self.input_channels
            else:
                raise ValueError(
                    "model.octree.warm_start_logits_gate_from must be one of: "
                    "'hidden', 'state', 'hidden+input'."
                )
            self._logits_gate_conv = nn.Conv2d(gate_in, self.output_channels, kernel_size=1, bias=True)
            with torch.no_grad():
                self._logits_gate_conv.weight.zero_()
                self._logits_gate_conv.bias.fill_(2.0)

    def _run_backbone_with_steps(self, x_bchw: torch.Tensor, level: int, steps: int):
        old_steps = self.inference_steps[level]
        self.inference_steps[level] = int(steps)
        try:
            return self._run_backbone(x_bchw, level)
        finally:
            self.inference_steps[level] = old_steps

    def _interp_bchw(self, x: torch.Tensor, size, mode: str) -> torch.Tensor:
        align = False if mode in ("bilinear", "bicubic", "trilinear") else None
        return F.interpolate(x, size=size, mode=mode, align_corners=align)

    def _get_multiscale_steps(self):
        n_levels = len(self.octree_res)
        cfg = self.warm_start_multiscale_steps
        if cfg is None:
            fine = int(self.warm_start_steps)
            coarse = max(1, fine // 4)
            return [fine] + [coarse] * (n_levels - 1)
        if isinstance(cfg, (list, tuple, np.ndarray)):
            if len(cfg) != n_levels:
                raise ValueError(
                    f"model.octree.warm_start_multiscale_steps must have length {n_levels}, got {len(cfg)}."
                )
            return [int(s) for s in cfg]
        return [int(cfg)] * n_levels

    def _to_bchw_state(self, state: torch.Tensor, name: str) -> torch.Tensor:
        if state is None:
            return None
        if state.dim() != 4:
            raise ValueError(f"{name} must be 4D, got shape {tuple(state.shape)}.")
        if state.shape[1] == self.channel_n:
            return state
        if state.shape[-1] == self.channel_n:
            return state.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"{name} must be BCHW or BHWC with channel_n={self.channel_n}, got {tuple(state.shape)}."
        )

    def _init_logits_for_warm_start_bchw(self, state: torch.Tensor) -> torch.Tensor:
        mode = self.warm_start_logits_mode
        if mode in (None, "", "carry"):
            return state

        logits_start = self.input_channels
        logits_end = self.input_channels + self.output_channels

        if mode == "reset":
            out = state.clone()
            out[:, logits_start:logits_end] = 0
            return out

        if mode != "gate":
            raise ValueError(
                "model.octree.warm_start_logits_mode must be one of: 'carry', 'reset', 'gate'."
            )
        if self._logits_gate_conv is None:
            raise RuntimeError("Gate mode enabled but _logits_gate_conv is not initialized.")

        if self.warm_start_logits_gate_from == "hidden":
            feat = state[:, logits_end:]
        elif self.warm_start_logits_gate_from == "state":
            feat = state
        else:
            feat = torch.cat([state[:, :self.input_channels], state[:, logits_end:]], dim=1)

        gate = torch.sigmoid(self._logits_gate_conv(feat))
        out = state.clone()
        out[:, logits_start:logits_end] = out[:, logits_start:logits_end] * gate
        return out

    def _inject_hidden_noise(self, state: torch.Tensor) -> torch.Tensor:
        """Inject Gaussian noise into hidden channels during training.

        The noise standard deviation is annealed linearly from
        ``warm_start_hidden_noise_std`` to 0 over
        ``warm_start_hidden_noise_anneal_epochs`` epochs.  If
        ``anneal_epochs == 0`` the noise is constant.
        """
        if not self.training:
            return state
        base_std = self.warm_start_hidden_noise_std
        if base_std <= 0.0:
            return state

        anneal = self.warm_start_hidden_noise_anneal_epochs
        if anneal > 0 and self._current_epoch >= anneal:
            return state  # annealing finished
        if anneal > 0:
            scale = 1.0 - (self._current_epoch / float(anneal))
        else:
            scale = 1.0

        std = base_std * scale
        hidden_start = self.input_channels + self.output_channels
        if hidden_start >= self.channel_n:
            return state

        left = state[:, :hidden_start]
        hidden = state[:, hidden_start:]
        hidden = hidden + torch.randn_like(hidden) * std
        return torch.cat([left, hidden], dim=1)

    def _stabilize_hidden_state_bchw(self, state: torch.Tensor) -> torch.Tensor:
        hidden_start = self.input_channels + self.output_channels
        if hidden_start >= self.channel_n:
            return state

        left = state[:, :hidden_start]
        hidden = state[:, hidden_start:]

        clip = self.warm_start_hidden_clip
        if clip is not None:
            clip = float(clip)
            if clip > 0:
                hidden = hidden.clamp(-clip, clip)

        tanh_scale = self.warm_start_hidden_tanh_scale
        if tanh_scale is not None:
            tanh_scale = float(tanh_scale)
            if tanh_scale > 0:
                hidden = torch.tanh(hidden / tanh_scale) * tanh_scale

        if self._warm_hidden_ln is not None:
            hidden = hidden.permute(0, 2, 3, 1).contiguous()
            hidden = self._warm_hidden_ln(hidden)
            hidden = hidden.permute(0, 3, 1, 2).contiguous()

        if self._warm_hidden_gn is not None:
            hidden = self._warm_hidden_gn(hidden)

        return torch.cat([left, hidden], dim=1)

    def _forward_cold(self, x_a: torch.Tensor, x_b: torch.Tensor):
        input_ch = self.input_channels

        state_a = x_a.new_zeros((x_a.shape[0], self.channel_n, *self.octree_res[-1]))
        state_b = x_b.new_zeros((x_b.shape[0], self.channel_n, *self.octree_res[-1]))
        xa_coarse = self.downscale(x_a, -1, layout="BCHW")
        xb_coarse = self.downscale(x_b, -1, layout="BCHW")
        state_a[:, :input_ch] = xa_coarse[:, :input_ch]
        state_b[:, :input_ch] = xb_coarse[:, :input_ch]

        for level in range(len(self.octree_res) - 1, -1, -1):
            state_ab = torch.cat([state_a, state_b], dim=0)
            state_ab = self._run_backbone(state_ab, level)
            state_a, state_b = state_ab.chunk(2, dim=0)
            state_a, state_b = self._maybe_cross_fuse(state_a, state_b, level)

            if level > 0:
                scale_h, scale_w = self.computed_upsampling_scales[level - 1][0]
                scale_h, scale_w = int(scale_h), int(scale_w)

                state_a = state_a.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                state_b = state_b.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)

                inj_a = self.downscale(x_a, level - 1, layout="BCHW")
                inj_b = self.downscale(x_b, level - 1, layout="BCHW")
                state_a[:, :input_ch] = inj_a[:, :input_ch]
                state_b[:, :input_ch] = inj_b[:, :input_ch]

        return state_a, state_b

    def _forward_warm_single_scale(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        prev_state_a: torch.Tensor,
        prev_state_b: torch.Tensor,
    ):
        input_ch = self.input_channels
        state_a = prev_state_a.clone()
        state_b = prev_state_b.clone()

        state_a[:, :input_ch] = x_a[:, :input_ch]
        state_b[:, :input_ch] = x_b[:, :input_ch]

        state_a = self._init_logits_for_warm_start_bchw(state_a)
        state_b = self._init_logits_for_warm_start_bchw(state_b)
        state_a = self._stabilize_hidden_state_bchw(state_a)
        state_b = self._stabilize_hidden_state_bchw(state_b)

        # Inject noise into hidden channels (training-time only, annealed)
        state_a = self._inject_hidden_noise(state_a)
        state_b = self._inject_hidden_noise(state_b)

        steps = int(self.warm_start_steps)
        state_ab = torch.cat([state_a, state_b], dim=0)
        state_ab = self._run_backbone_with_steps(state_ab, level=0, steps=steps)
        state_a, state_b = state_ab.chunk(2, dim=0)
        state_a, state_b = self._maybe_cross_fuse(state_a, state_b, level=0)
        state_a = self._stabilize_hidden_state_bchw(state_a)
        state_b = self._stabilize_hidden_state_bchw(state_b)
        return state_a, state_b

    def _forward_warm_multiscale(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        prev_state_a: torch.Tensor,
        prev_state_b: torch.Tensor,
    ):
        start_level = int(self.warm_start_multiscale_start_level)
        if start_level < 0 or start_level >= len(self.octree_res):
            raise ValueError(
                f"model.octree.warm_start_multiscale_start_level out of range: {start_level} "
                f"(expected 0..{len(self.octree_res)-1})."
            )

        mode = self.warm_start_multiscale_downsample_mode
        if mode not in ("nearest", "bilinear", "bicubic", "area"):
            raise ValueError(
                "model.octree.warm_start_multiscale_downsample_mode must be one of "
                "'nearest', 'bilinear', 'bicubic', 'area'."
            )

        steps_per_level = self._get_multiscale_steps()

        inputs_a = {}
        inputs_b = {}
        for level in range(start_level, -1, -1):
            if level == 0:
                inputs_a[0] = x_a
                inputs_b[0] = x_b
            else:
                target_res = self.octree_res[level]
                inputs_a[level] = self._interp_bchw(x_a, target_res, mode="bilinear")
                inputs_b[level] = self._interp_bchw(x_b, target_res, mode="bilinear")

        if start_level == 0:
            state_a = prev_state_a.clone()
            state_b = prev_state_b.clone()
        else:
            state_a = self._interp_bchw(prev_state_a, self.octree_res[start_level], mode=mode)
            state_b = self._interp_bchw(prev_state_b, self.octree_res[start_level], mode=mode)

        input_ch = self.input_channels
        state_a[:, :input_ch] = inputs_a[start_level][:, :input_ch]
        state_b[:, :input_ch] = inputs_b[start_level][:, :input_ch]
        state_a = self._init_logits_for_warm_start_bchw(state_a)
        state_b = self._init_logits_for_warm_start_bchw(state_b)
        state_a = self._stabilize_hidden_state_bchw(state_a)
        state_b = self._stabilize_hidden_state_bchw(state_b)

        for level in range(start_level, -1, -1):
            if level != start_level:
                state_a[:, :input_ch] = inputs_a[level][:, :input_ch]
                state_b[:, :input_ch] = inputs_b[level][:, :input_ch]

            steps = int(steps_per_level[level])
            if steps > 0:
                state_ab = torch.cat([state_a, state_b], dim=0)
                state_ab = self._run_backbone_with_steps(state_ab, level=level, steps=steps)
                state_a, state_b = state_ab.chunk(2, dim=0)
                state_a, state_b = self._maybe_cross_fuse(state_a, state_b, level=level)
                state_a = self._stabilize_hidden_state_bchw(state_a)
                state_b = self._stabilize_hidden_state_bchw(state_b)

            if level > 0:
                scale_h, scale_w = self.computed_upsampling_scales[level - 1][0]
                scale_h, scale_w = int(scale_h), int(scale_w)
                state_a = state_a.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                state_b = state_b.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                state_a = self._stabilize_hidden_state_bchw(state_a)
                state_b = self._stabilize_hidden_state_bchw(state_b)

        return state_a, state_b

    def _pack_outputs(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
    ):
        input_ch = self.input_channels

        logits_a = state_a[:, input_ch:input_ch + self.output_channels]
        logits_b = state_b[:, input_ch:input_ch + self.output_channels]
        hidden_a = state_a[:, input_ch + self.output_channels:]
        hidden_b = state_b[:, input_ch + self.output_channels:]

        logits_a = self._bchw_to_bhwc(logits_a)
        logits_b = self._bchw_to_bhwc(logits_b)
        target_a = self._bchw_to_bhwc(y_a)
        target_b = self._bchw_to_bhwc(y_b)
        hidden_a = self._bchw_to_bhwc(hidden_a)
        hidden_b = self._bchw_to_bhwc(hidden_b)

        logits = torch.cat([logits_a, logits_b], dim=0)
        target = torch.cat([target_a, target_b], dim=0)
        hidden = torch.cat([hidden_a, hidden_b], dim=0)

        ret_dict = {
            "logits": logits,
            "target": target,
            "hidden_channels": hidden,
            "final_state_a": self._bchw_to_bhwc(state_a),
            "final_state_b": self._bchw_to_bhwc(state_b),
        }
        if self.apply_nonlin is not None:
            ret_dict["probabilities"] = self.apply_nonlin(logits)
        return ret_dict

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: torch.Tensor = None,
        y_b: torch.Tensor = None,
        prev_state_a: torch.Tensor = None,
        prev_state_b: torch.Tensor = None,
        batch_duplication=1,
    ):
        if self.training:
            if x_a.shape[2:4] != self.octree_res[0]:
                raise ValueError(f"Expected view A shape {self.octree_res[0]}, got {tuple(x_a.shape[2:4])}.")
            if x_b.shape[2:4] != self.octree_res[0]:
                raise ValueError(f"Expected view B shape {self.octree_res[0]}, got {tuple(x_b.shape[2:4])}.")
            return self.forward_train(
                x_a,
                x_b,
                y_a=y_a,
                y_b=y_b,
                prev_state_a=prev_state_a,
                prev_state_b=prev_state_b,
                batch_duplication=batch_duplication,
            )
        return self.forward_eval(
            x_a,
            x_b,
            prev_state_a=prev_state_a,
            prev_state_b=prev_state_b,
        )

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
                device=x_a.device,
                dtype=x_a.dtype,
            )
        if y_b is None:
            y_b = torch.zeros(
                (x_b.shape[0], self.output_channels, x_b.shape[2], x_b.shape[3]),
                device=x_b.device,
                dtype=x_b.dtype,
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

        if prev_state_a is None:
            state_a, state_b = self._forward_cold(x_a, x_b)
        else:
            if prev_state_a.shape[2:4] != tuple(self.octree_res[0]):
                prev_state_a = F.interpolate(prev_state_a, size=self.octree_res[0], mode="nearest")
            if prev_state_b.shape[2:4] != tuple(self.octree_res[0]):
                prev_state_b = F.interpolate(prev_state_b, size=self.octree_res[0], mode="nearest")

            if self.warm_start_multiscale:
                state_a, state_b = self._forward_warm_multiscale(x_a, x_b, prev_state_a, prev_state_b)
            else:
                state_a, state_b = self._forward_warm_single_scale(x_a, x_b, prev_state_a, prev_state_b)

        return self._pack_outputs(state_a, state_b, y_a, y_b)

    @torch.no_grad()
    def forward_eval(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        prev_state_a: torch.Tensor = None,
        prev_state_b: torch.Tensor = None,
    ):
        temp = self.patch_sizes
        self.patch_sizes = [None] * len(self.patch_sizes)
        out = self.forward_train(
            x_a,
            x_b,
            y_a=x_a,
            y_b=x_b,
            prev_state_a=prev_state_a,
            prev_state_b=prev_state_b,
            batch_duplication=1,
        )
        out.pop("target", None)
        self.patch_sizes = temp
        return out
