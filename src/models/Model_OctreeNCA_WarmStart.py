import torch
import torch.nn as nn
import numpy as np
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2

class OctreeNCA2DWarmStart(OctreeNCA2DPatch2):
    def __init__(self, config: dict):
        super().__init__(config)
        # Number of steps to run during the warm-start phase (finest level only)
        self.warm_start_steps = config.get('model.octree.warm_start_steps', self.inference_steps[0])

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, prev_state: torch.Tensor = None, batch_duplication=1):
        # x: BCHW (Current Frame Image)
        # y: BCHW (Current Frame Label)
        # prev_state: BHWC (State from previous frame at finest resolution)

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
            # Assume input x is at target resolution (finest)
            # We assume the dataset provides correct resolution, bypassing the complex patching logic of the parent
            x_bchw = x.permute(0, 3, 1, 2)
            
            # Prepare inputs for all levels (for injection)
            inputs_at_levels = {} 
            for level in range(len(self.octree_res)):
                if level == 0:
                    inputs_at_levels[0] = x
                else:
                    target_res = self.octree_res[level]
                    down = torch.nn.functional.interpolate(x_bchw, size=target_res, mode='bilinear', align_corners=False)
                    inputs_at_levels[level] = down.permute(0, 2, 3, 1)

            # Start from Coarsest Level
            coarsest_lvl = len(self.octree_res) - 1
            state = torch.zeros(x.shape[0], *self.octree_res[coarsest_lvl], self.channel_n, device=self.device)
            state[..., :self.input_channels] = inputs_at_levels[coarsest_lvl][..., :self.input_channels]

            # Upscaling Loop
            for level in range(len(self.octree_res)-1, -1, -1):
                steps = self.inference_steps[level]
                state = state.permute(0, 3, 1, 2)
                
                if self.separate_models:
                    state = self.backbone_ncas[level](state, steps=steps, fire_rate=self.fire_rate)
                else:
                    state = self.backbone_nca(state, steps=steps, fire_rate=self.fire_rate)
                
                if level > 0:
                    # Upscale
                    state = torch.nn.Upsample(scale_factor=tuple(self.computed_upsampling_scales[level-1][0]), mode='nearest')(state)
                    state = state.permute(0, 2, 3, 1)
                    # Inject Image
                    target_input = inputs_at_levels[level-1]
                    state[..., :self.input_channels] = target_input[..., :self.input_channels]
                else:
                    state = state.permute(0, 2, 3, 1)

        # Branch B: Warm Start
        else:
            # 1. Inject: Overwrite input channels of previous state with NEW image
            state = prev_state.clone()
            state[..., :self.input_channels] = x[..., :self.input_channels]
            
            # 2. Run NCA at finest level (Level 0) only
            steps = self.warm_start_steps
            state = state.permute(0, 3, 1, 2)
            
            if self.separate_models:
                state = self.backbone_ncas[0](state, steps=steps, fire_rate=self.fire_rate)
            else:
                state = self.backbone_nca(state, steps=steps, fire_rate=self.fire_rate)
                
            state = state.permute(0, 2, 3, 1)

        # Output
        logits = state[..., self.input_channels:self.input_channels+self.output_channels]
        hidden = state[..., self.input_channels+self.output_channels:]
        
        ret_dict = {'logits': logits, 'target': y, 'hidden_channels': hidden, 'final_state': state}

        if self.apply_nonlin is not None:
            ret_dict['probabilities'] = self.apply_nonlin(logits)

        return ret_dict

    @torch.no_grad()
    def forward_eval_warm(self, x: torch.Tensor, prev_state: torch.Tensor = None):
        out = self.forward_train(x, x, prev_state=prev_state)
        out.pop('target')
        return out