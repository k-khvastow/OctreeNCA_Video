###############################################
##########   Gemini version   #################
###############################################

from src.models.Model_BasicNCA2D import BasicNCA2D
from src.models.Model_BasicNCA2D_fast import BasicNCA2DFast
from src.models.Model_ViTCA import ViTCA

import torch
import torch.nn as nn
import torch.nn.functional as F

class OctreeNCA2DPatch2(torch.nn.Module):
    def __init__(self, config: dict):
        super(OctreeNCA2DPatch2, self).__init__()

        # --- Parameters ---
        self.channel_n = config['model.channel_n']
        self.fire_rate = config['model.fire_rate']
        self.input_channels = config['model.input_channels']
        self.output_channels = config['model.output_channels']
        self.device = config['experiment.device']
        self.patch_sizes = config['model.train.patch_sizes']
        self.loss_weighted_patching = config['model.train.loss_weighted_patching']
        
        # Octree Configuration
        octree_res_and_steps = config['model.octree.res_and_steps']
        self.octree_res = [tuple(r_s[0]) for r_s in octree_res_and_steps]
        self.inference_steps = [r_s[1] for r_s in octree_res_and_steps]
        self.separate_models = config['model.octree.separate_models']

        # --- Backbone Initialization ---
        # Map strings to classes to avoid eval()
        from src.models.Model_BasicNCA2D import BasicNCA2D
        from src.models.Model_ViTCA import ViTCA
        model_map = {"BasicNCA2D": BasicNCA2D, "ViTCA": ViTCA}
        backbone_class = model_map.get(config.get("model.backbone_class"), BasicNCA2D)
        
        kernel_size = config['model.kernel_size']
        normalization = config.get("model.normalization", "batch")

        if self.separate_models:
            ks_list = kernel_size if isinstance(kernel_size, list) else [kernel_size] * len(self.octree_res)
            self.backbone_ncas = nn.ModuleList([
                backbone_class(channel_n=self.channel_n, fire_rate=self.fire_rate, device=self.device, 
                               hidden_size=config['model.hidden_size'], input_channels=self.input_channels, 
                               kernel_size=ks_list[l], normalization=normalization)
                for l in range(len(self.octree_res))
            ])
        else:
            self.backbone_nca = backbone_class(channel_n=self.channel_n, fire_rate=self.fire_rate, device=self.device, 
                                              hidden_size=config['model.hidden_size'], input_channels=self.input_channels, 
                                              kernel_size=kernel_size, normalization=normalization)

        # Precompute Upsampling Scales
        self.scales = []
        for i in range(len(self.octree_res)-1):
            scale = (self.octree_res[i][0] // self.octree_res[i+1][0], 
                     self.octree_res[i][1] // self.octree_res[i+1][1])
            self.scales.append(scale)

    @torch.no_grad()
    def compute_probabilities_matrix(self, loss_map: torch.Tensor, level: int) -> torch.Tensor:
        """ Vectorized patch difficulty mapping using GPU AvgPool. """
        patch_size = self.patch_sizes[level]
        # (B, 1, H, W)
        loss_img = loss_map.unsqueeze(1) 
        
        # Pool to find patches with highest average loss
        # stride=1 checks every possible top-left corner
        patch_difficulty = F.avg_pool2d(loss_img, kernel_size=patch_size, stride=1)
        
        # Flatten and Softmax to get probabilities per patch corner
        B, C, H, W = patch_difficulty.shape
        probs = F.softmax(patch_difficulty.view(B, -1) * 10.0, dim=-1) # Temperature scaling
        return probs.view(B, H, W)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1, prev_loss=None):
        """ BHWC internal format to avoid repeated permutations. """
        x = x.permute(0, 2, 3, 1) # BCHW -> BHWC
        if y is not None:
            y = y.permute(0, 2, 3, 1)
        
        if self.training:
            return self.forward_train(x, y, batch_duplication, prev_loss)
        return self.forward_eval(x)

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, batch_duplication=1, prev_loss=None):
        B = x.shape[0] * batch_duplication
        device = x.device

        if batch_duplication != 1:
            x = x.repeat(batch_duplication, 1, 1, 1)
            y = y.repeat(batch_duplication, 1, 1, 1)

        # Initial State (Coarsest Resolution)
        curr_res = self.octree_res[-1]
        x_down = F.interpolate(x.permute(0, 3, 1, 2), size=curr_res).permute(0, 2, 3, 1)
        state = torch.zeros((B, *curr_res, self.channel_n), device=device)
        state[..., :self.input_channels] = x_down
        
        offsets = torch.zeros((B, 2), dtype=torch.long, device=device)
        b_idx = torch.arange(B, device=device).view(B, 1, 1)

        for level in range(len(self.octree_res)-1, -1, -1):
            model = self.backbone_ncas[level] if self.separate_models else self.backbone_nca
            state = model(state, steps=self.inference_steps[level], fire_rate=self.fire_rate)

            if level > 0:
                # Upscale State
                scale = self.scales[level-1]
                state = F.interpolate(state.permute(0, 3, 1, 2), scale_factor=scale, mode='nearest').permute(0, 2, 3, 1)
                offsets = offsets * torch.tensor(scale, device=device)
                
                target_patch_size = self.patch_sizes[level-1]
                if target_patch_size is not None:
                    # SAMPLE INDICES: Use Weighted (if loss exists) or Random
                    if self.loss_weighted_patching and prev_loss is not None:
                        # Vectorized weighted sampling
                        prob_matrix = self.compute_probabilities_matrix(prev_loss, level-1)
                        flat_idx = torch.multinomial(prob_matrix.view(B, -1), 1).squeeze(1)
                        local_h = flat_idx // prob_matrix.shape[2]
                        local_w = flat_idx % prob_matrix.shape[2]
                    else:
                        # Fast uniform random sampling
                        H_max, W_max = state.shape[1] - target_patch_size[0], state.shape[2] - target_patch_size[1]
                        local_h = torch.randint(0, H_max + 1, (B,), device=device)
                        local_w = torch.randint(0, W_max + 1, (B,), device=device)
                    
                    # Update state with patch and original high-res data
                    h_idx = local_h.view(B, 1, 1) + torch.arange(target_patch_size[0], device=device).view(1, -1, 1)
                    w_idx = local_w.view(B, 1, 1) + torch.arange(target_patch_size[1], device=device).view(1, 1, -1)
                    
                    state = state[b_idx, h_idx, w_idx, :]
                    offsets[:, 0] += local_h
                    offsets[:, 1] += local_w
                    
                    # Correct input channels from global resolution
                    orig_level = F.interpolate(x.permute(0, 3, 1, 2), size=self.octree_res[level-1]).permute(0, 2, 3, 1)
                    oh = offsets[:, 0].view(B, 1, 1) + torch.arange(target_patch_size[0], device=device).view(1, -1, 1)
                    ow = offsets[:, 1].view(B, 1, 1) + torch.arange(target_patch_size[1], device=device).view(1, 1, -1)
                    state[..., :self.input_channels] = orig_level[b_idx, oh, ow, :]

        # Final Crop for Loss Calculation
        fh = offsets[:, 0].view(B, 1, 1) + torch.arange(state.shape[1], device=device).view(1, -1, 1)
        fw = offsets[:, 1].view(B, 1, 1) + torch.arange(state.shape[2], device=device).view(1, 1, -1)
        y_cropped = y[b_idx, fh, fw, :]

        return {
            'logits': state[..., self.input_channels:self.input_channels+self.output_channels], 
            'target': y_cropped, 
            'hidden_channels': state[..., self.input_channels+self.output_channels:]
        }

    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        old_patches = self.patch_sizes
        self.patch_sizes = [None] * len(self.patch_sizes)
        res = self.forward_train(x, x, batch_duplication=1)
        self.patch_sizes = old_patches
        return res