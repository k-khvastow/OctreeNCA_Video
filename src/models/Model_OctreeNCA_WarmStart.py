import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2

class OctreeNCA2DWarmStart(OctreeNCA2DPatch2):
    def __init__(self, config: dict):
        super().__init__(config)
        # Number of steps to run during the warm-start phase (finest level only)
        self.warm_start_steps = config.get('model.octree.warm_start_steps', self.inference_steps[0])
        self.warm_start_multiscale = bool(config.get("model.octree.warm_start_multiscale", False))
        self.warm_start_multiscale_steps = config.get("model.octree.warm_start_multiscale_steps", None)
        self.warm_start_multiscale_start_level = config.get(
            "model.octree.warm_start_multiscale_start_level", len(self.octree_res) - 1
        )
        self.warm_start_multiscale_downsample_mode = str(
            config.get("model.octree.warm_start_multiscale_downsample_mode", "nearest")
        )

        # Hidden-state stabilization (applied to the carried recurrent state).
        # These operate on the hidden slice only: state[..., input:input+out] are logits, state[..., hidden_start:] are hidden.
        # Options:
        # - model.octree.warm_start_hidden_norm: none|layer|group
        # - model.octree.warm_start_hidden_clip: float (clamps hidden to [-v, v])
        # - model.octree.warm_start_hidden_tanh_scale: float (bounds via tanh; keeps magnitude roughly <= scale)
        self.warm_start_hidden_norm = str(
            config.get("model.octree.warm_start_hidden_norm", "none")
        ).lower()
        self.warm_start_hidden_clip = config.get("model.octree.warm_start_hidden_clip", None)
        self.warm_start_hidden_tanh_scale = config.get("model.octree.warm_start_hidden_tanh_scale", None)
        self.warm_start_hidden_gn_groups = int(config.get("model.octree.warm_start_hidden_gn_groups", 8))

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

        # Logits carry policy during warm start.
        # State layout is: [input | logits | hidden]. Carrying logits across frames can amplify drift.
        # Options:
        # - carry (default): keep logits from prev_state
        # - reset: zero logits at start of each frame (carry hidden only)
        # - gate: learned per-pixel gate deciding how much prev logits to keep (vs reset-to-zero)
        self.warm_start_logits_mode = str(
            config.get("model.octree.warm_start_logits_mode", "carry")
        ).lower()
        self.warm_start_logits_gate_from = str(
            config.get("model.octree.warm_start_logits_gate_from", "hidden")
        ).lower()
        self._logits_gate_conv = None
        if self.warm_start_logits_mode == "gate":
            hidden_dim = self.channel_n - self.input_channels - self.output_channels
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
                # Bias > 0 starts closer to 'carry' (sigmoid(bias) ~= keep ratio).
                self._logits_gate_conv.weight.zero_()
                self._logits_gate_conv.bias.fill_(2.0)

    def _interp_bhwc(self, x: torch.Tensor, size, mode: str) -> torch.Tensor:
        # x: BHWC -> resize spatial to `size` -> BHWC
        x_bchw = x.permute(0, 3, 1, 2)
        align = False if mode in ("bilinear", "bicubic", "trilinear") else None
        out = F.interpolate(x_bchw, size=size, mode=mode, align_corners=align)
        return out.permute(0, 2, 3, 1)

    def _get_multiscale_steps(self):
        n_levels = len(self.octree_res)
        cfg = self.warm_start_multiscale_steps
        if cfg is None:
            fine = int(self.warm_start_steps)
            coarse = max(1, fine // 4)
            steps = [fine] + [coarse] * (n_levels - 1)
            return steps
        if isinstance(cfg, (list, tuple, np.ndarray)):
            if len(cfg) != n_levels:
                raise ValueError(
                    f"model.octree.warm_start_multiscale_steps must have length {n_levels}, got {len(cfg)}."
                )
            return [int(s) for s in cfg]
        return [int(cfg)] * n_levels

    def _init_logits_for_warm_start(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies warm-start logits initialization policy once at the start of a frame update.
        state: BHWC at the current level resolution.
        """
        mode = self.warm_start_logits_mode
        if mode in (None, "", "carry"):
            return state

        logits_start = self.input_channels
        logits_end = self.input_channels + self.output_channels

        if mode == "reset":
            left = state[..., :logits_start]
            logits = torch.zeros_like(state[..., logits_start:logits_end])
            right = state[..., logits_end:]
            return torch.cat([left, logits, right], dim=-1)

        if mode != "gate":
            raise ValueError(
                "model.octree.warm_start_logits_mode must be one of: 'carry', 'reset', 'gate'."
            )

        if self._logits_gate_conv is None:
            raise RuntimeError("Gate mode enabled but _logits_gate_conv is not initialized.")

        if self.warm_start_logits_gate_from == "hidden":
            feat = state[..., self.input_channels + self.output_channels :]
        elif self.warm_start_logits_gate_from == "state":
            feat = state
        else:  # hidden+input
            feat = torch.cat(
                [
                    state[..., : self.input_channels],
                    state[..., self.input_channels + self.output_channels :],
                ],
                dim=-1,
            )

        feat_bchw = feat.permute(0, 3, 1, 2).contiguous()
        gate_bchw = torch.sigmoid(self._logits_gate_conv(feat_bchw))
        gate = gate_bchw.permute(0, 2, 3, 1)

        left = state[..., :logits_start]
        logits = state[..., logits_start:logits_end] * gate
        right = state[..., logits_end:]
        return torch.cat([left, logits, right], dim=-1)

    def _stabilize_hidden_state(self, state: torch.Tensor) -> torch.Tensor:
        hidden_start = self.input_channels + self.output_channels
        if hidden_start >= self.channel_n:
            return state

        left = state[..., :hidden_start]
        hidden = state[..., hidden_start:]

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
            hidden = self._warm_hidden_ln(hidden)

        if self._warm_hidden_gn is not None:
            h_bchw = hidden.permute(0, 3, 1, 2).contiguous()
            h_bchw = self._warm_hidden_gn(h_bchw)
            hidden = h_bchw.permute(0, 2, 3, 1)

        return torch.cat([left, hidden], dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, prev_state: torch.Tensor = None, batch_duplication=1):
        # x: BCHW (Current Frame Image) - The dataset returns BCHW
        # y: BCHW (Current Frame Label)
        # prev_state: BHWC (State from previous frame at finest resolution)

        # Standardize inputs to BHWC for processing
        x = x.permute(0, 2, 3, 1) # -> BHWC
        if y is not None:
            y = y.permute(0, 2, 3, 1) # -> BHWC
        
        if prev_state is not None:
             # Ensure prev_state is BHWC
             if prev_state.shape[-1] != self.channel_n:
                 prev_state = prev_state.permute(0, 2, 3, 1)

        if self.training:
            return self.forward_train(x, y, prev_state, batch_duplication)
        else:
            return self.forward_eval_warm(x, prev_state)

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, prev_state: torch.Tensor = None, batch_duplication=1):
        # x, y: BHWC
        
        # Branch A: Cold Start (Standard Coarse-to-Fine)
        if prev_state is None:
            # We assume input x matches the finest resolution
            # Prepare inputs for all levels (for injection) using PyTorch interpolate (requires BCHW)
            x_bchw = x.permute(0, 3, 1, 2)
            
            inputs_at_levels = {} 
            for level in range(len(self.octree_res)):
                if level == 0:
                    inputs_at_levels[0] = x
                else:
                    target_res = self.octree_res[level]
                    down = torch.nn.functional.interpolate(x_bchw, size=target_res, mode='bilinear', align_corners=False)
                    inputs_at_levels[level] = down.permute(0, 2, 3, 1) # Store as BHWC

            # Start from Coarsest Level
            coarsest_lvl = len(self.octree_res) - 1
            # Initialize state: Zeros + Input Image
            # State shape: (B, H, W, C)
            state = torch.zeros(x.shape[0], *self.octree_res[coarsest_lvl], self.channel_n, device=self.device)
            state[..., :self.input_channels] = inputs_at_levels[coarsest_lvl][..., :self.input_channels]

            # Loop from Coarsest to Finest
            for level in range(len(self.octree_res)-1, -1, -1):
                steps = self.inference_steps[level]
                
                # FIX: Do NOT permute to BCHW here. 
                # The backbone (BasicNCA2DFast) expects BHWC and handles rearrangement internally.
                if self.separate_models:
                    state = self.backbone_ncas[level](state, steps=steps, fire_rate=self.fire_rate)
                else:
                    state = self.backbone_nca(state, steps=steps, fire_rate=self.fire_rate)
                
                # Upscale if not at finest level
                if level > 0:
                    # Upsampling requires BCHW
                    state = state.permute(0, 3, 1, 2) # BHWC -> BCHW
                    state = torch.nn.Upsample(scale_factor=tuple(self.computed_upsampling_scales[level-1][0]), mode='nearest')(state)
                    state = state.permute(0, 2, 3, 1) # BCHW -> BHWC
                    
                    # Inject details from input image at this new resolution
                    target_input = inputs_at_levels[level-1]
                    state[..., :self.input_channels] = target_input[..., :self.input_channels]

        # Branch B: Warm Start
        else:
            # prev_state is from t-1 at finest resolution (BHWC).
            if not self.warm_start_multiscale:
                # 1) Inject: overwrite input channels with the NEW image
                state = prev_state.clone()
                state[..., :self.input_channels] = x[..., :self.input_channels]
                state = self._init_logits_for_warm_start(state)
                state = self._stabilize_hidden_state(state)

                # 2) Run NCA at finest level (Level 0)
                steps = int(self.warm_start_steps)
                if self.separate_models:
                    state = self.backbone_ncas[0](state, steps=steps, fire_rate=self.fire_rate)
                else:
                    state = self.backbone_nca(state, steps=steps, fire_rate=self.fire_rate)
                state = self._stabilize_hidden_state(state)
            else:
                # Multi-scale warm start:
                # - downsample prev_state to a coarser level
                # - run a few NCA steps per level (coarse -> fine), injecting the current image at each level
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

                # Prepare current image at levels for stable input injection.
                inputs_at_levels = {}
                for level in range(start_level, -1, -1):
                    if level == 0:
                        inputs_at_levels[0] = x
                    else:
                        inputs_at_levels[level] = self._interp_bhwc(x, size=self.octree_res[level], mode="bilinear")

                # Start from downsampled previous state at start_level.
                if start_level == 0:
                    state = prev_state.clone()
                else:
                    state = self._interp_bhwc(prev_state, size=self.octree_res[start_level], mode=mode)
                # Inject image at start level then apply logits policy once for the frame.
                state[..., :self.input_channels] = inputs_at_levels[start_level][..., :self.input_channels]
                state = self._init_logits_for_warm_start(state)
                state = self._stabilize_hidden_state(state)

                # Coarse -> fine refinement
                for level in range(start_level, -1, -1):
                    if level != start_level:
                        state[..., :self.input_channels] = inputs_at_levels[level][..., :self.input_channels]
                    steps = int(steps_per_level[level])
                    if steps > 0:
                        if self.separate_models:
                            state = self.backbone_ncas[level](state, steps=steps, fire_rate=self.fire_rate)
                        else:
                            state = self.backbone_nca(state, steps=steps, fire_rate=self.fire_rate)
                        state = self._stabilize_hidden_state(state)

                    if level > 0:
                        # Upscale to next finer level (BHWC -> BCHW -> upsample -> BHWC)
                        scale_h, scale_w = self.computed_upsampling_scales[level - 1][0]
                        scale_h, scale_w = int(scale_h), int(scale_w)
                        state_bchw = state.permute(0, 3, 1, 2)
                        state_bchw = torch.nn.Upsample(
                            scale_factor=(scale_h, scale_w), mode="nearest"
                        )(state_bchw)
                        state = state_bchw.permute(0, 2, 3, 1)
                        state = self._stabilize_hidden_state(state)

        # Finalize Output
        logits = state[..., self.input_channels:self.input_channels+self.output_channels]
        hidden = state[..., self.input_channels+self.output_channels:]
        
        ret_dict = {'logits': logits, 'target': y, 'hidden_channels': hidden, 'final_state': state}

        if self.apply_nonlin is not None:
            ret_dict['probabilities'] = self.apply_nonlin(logits)

        return ret_dict

    @torch.no_grad()
    def forward_eval_warm(self, x: torch.Tensor, prev_state: torch.Tensor = None):
        # Evaluation wrapper
        out = self.forward_train(x, x, prev_state=prev_state)
        out.pop('target')
        return out
