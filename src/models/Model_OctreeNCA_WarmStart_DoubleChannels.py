import torch
import torch.nn as nn
import numpy as np
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2


class OctreeNCA2DWarmStartDoubleChannels(OctreeNCA2DPatch2):
    def __init__(self, config: dict):
        # Default to temporal append for the "double channels" variant.
        self.temporal_append_outputs = config.get('model.temporal_append_outputs', True)
        self.temporal_append_use_probs = config.get('model.temporal_append_use_probs', False)
        base_input_channels = config.get('model.input_channels', 1)

        adjusted_config = config
        if self.temporal_append_outputs:
            adjusted_config = dict(config)
            adjusted_config['model.input_channels'] = base_input_channels + config['model.output_channels']

        super().__init__(adjusted_config)

        self.base_input_channels = base_input_channels
        # Number of steps to run during the warm-start phase (finest level only)
        self.warm_start_steps = config.get('model.octree.warm_start_steps', self.inference_steps[0])
        if self.temporal_append_outputs:
            min_channels = self.input_channels + self.output_channels
            if self.channel_n < min_channels:
                raise ValueError(
                    f"channel_n ({self.channel_n}) must be >= input_channels ({self.input_channels}) "
                    f"+ output_channels ({self.output_channels}) when temporal append is enabled."
                )

    def _get_prev_output(self, prev_state: torch.Tensor, x: torch.Tensor):
        if prev_state is None:
            return torch.zeros(
                x.shape[0], x.shape[1], x.shape[2], self.output_channels,
                device=x.device, dtype=x.dtype
            )

        prev_logits = prev_state[..., self.input_channels:self.input_channels + self.output_channels]
        if self.temporal_append_use_probs and self.apply_nonlin is not None:
            return self.apply_nonlin(prev_logits)
        return prev_logits

    def _build_temporal_input(self, x: torch.Tensor, prev_state: torch.Tensor):
        if not self.temporal_append_outputs:
            return x

        # If caller already provided augmented channels, optionally refresh from prev_state.
        if x.shape[-1] == self.input_channels:
            if prev_state is None:
                return x
            x_aug = x.clone()
            x_aug[..., self.base_input_channels:self.input_channels] = self._get_prev_output(prev_state, x)
            return x_aug

        if x.shape[-1] != self.base_input_channels:
            raise ValueError(
                f"Expected input with {self.base_input_channels} channels, got {x.shape[-1]}."
            )

        prev_out = self._get_prev_output(prev_state, x)
        return torch.cat([x, prev_out], dim=-1)

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

        x = self._build_temporal_input(x, prev_state)

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
            for level in range(len(self.octree_res) - 1, -1, -1):
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

            # 1. Inject: Overwrite the input channels of the previous state with the NEW image
            state = prev_state.clone()
            state[..., :self.input_channels] = x[..., :self.input_channels]

            # 2. Run NCA at finest level (Level 0)
            steps = self.warm_start_steps

            # FIX: Pass BHWC directly to backbone
            if self.separate_models:
                state = self.backbone_ncas[0](state, steps=steps, fire_rate=self.fire_rate)
            else:
                state = self.backbone_nca(state, steps=steps, fire_rate=self.fire_rate)

            # State remains BHWC

        # Finalize Output
        logits = state[..., self.input_channels:self.input_channels + self.output_channels]
        hidden = state[..., self.input_channels + self.output_channels:]

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
