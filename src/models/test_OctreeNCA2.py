####################################
# Looking for an optimized version of Model_OctreeNCA_2d_patching2.py
# Class OctreeNCA2DPatch2 
####################################

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# from src.models.Model_BasicNCA2D import BasicNCA2D
from src.models.test_OctreeNCA import BasicNCA2D
from src.models.Model_BasicNCA2D_fast import BasicNCA2DFast
from src.models.Model_ViTCA import ViTCA
import torchio as tio
import random
import math
import torch.nn.functional as F
import subprocess as sp


import matplotlib.pyplot as plt

class OctreeNCA2DPatch2(torch.nn.Module):
    def __init__(self, config: dict):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(OctreeNCA2DPatch2, self).__init__()

        channel_n = config['model.channel_n']
        fire_rate = config['model.fire_rate']
        hidden_size = config['model.hidden_size']
        input_channels = config['model.input_channels']
        output_channels = config['model.output_channels']
        kernel_size = config['model.kernel_size']
        track_running_stats = config['model.batchnorm_track_running_stats']

        octree_res_and_steps = config['model.octree.res_and_steps']
        separate_models = config['model.octree.separate_models']

        device = config['experiment.device']
        patch_sizes = config['model.train.patch_sizes']
        loss_weighted_patching = config['model.train.loss_weighted_patching']

        compile = config['performance.compile']

        normalization = config.get("model.normalization", "batch")

        self.apply_nonlin = config.get("model.apply_nonlin", None)
        self.apply_nonlin = eval(self.apply_nonlin) if self.apply_nonlin is not None else None

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate

        self.patch_sizes = patch_sizes
        self.loss_weighted_patching = loss_weighted_patching


        self.octree_res = [tuple(r_s[0]) for r_s in octree_res_and_steps]
        self.inference_steps = [r_s[1] for r_s in octree_res_and_steps]

        self.separate_models = separate_models

        if compile:
            torch.set_float32_matmul_precision('high')
        
        if separate_models:
            if isinstance(kernel_size, list):
                assert len(kernel_size) == len(octree_res_and_steps), "kernel_size must have same length as octree_res_and_steps"
            else:
                kernel_size = [kernel_size] * len(octree_res_and_steps)
        else:
            assert isinstance(kernel_size, int), "kernel_size must be an integer"

        backbone_class = eval(config.get("model.backbone_class", "BasicNCA2D"))
        assert backbone_class in [BasicNCA2D, BasicNCA2DFast], f"backbone_class must be either BasicNCA2D, got {backbone_class}"

        if separate_models:
            if config["model.vitca"]:
                self.backbone_ncas = []
                for l in range(len(octree_res_and_steps)):
                    conv_size = kernel_size[l]
                    self.backbone_ncas.append(ViTCA(patch_size=1, depth=config["model.vitca.depth"], heads=config["model.vitca.heads"],
                                           mlp_dim=config["model.vitca.mlp_dim"], dropout=config["model.vitca.dropout"], 
                                           cell_in_chns=input_channels, cell_out_chns=output_channels, 
                                           cell_hidden_chns=channel_n - input_channels - output_channels, 
                                           embed_cells=config["model.vitca.embed_cells"], embed_dim=config["model.vitca.embed_dim"],
                                           embed_dropout=config["model.vitca.embed_dropout"], 
                                           localize_attn=True, localized_attn_neighbourhood=[conv_size, conv_size], 
                                           device=config["experiment.device"]
                                           ))
                self.backbone_ncas = nn.ModuleList(self.backbone_ncas)
            else:
                self.backbone_ncas = nn.ModuleList([backbone_class(channel_n=channel_n, fire_rate=fire_rate, device=device, 
                                                            hidden_size=hidden_size, input_channels=input_channels, kernel_size=kernel_size[l],
                                                            normalization=normalization) 
                                                            for l in range(len(octree_res_and_steps))])
            if compile:
                for i, model in enumerate(self.backbone_ncas):
                    self.backbone_ncas[i] = torch.compile(model)
        else:
            if config["model.vitca"]:
                conv_size = config["kernel_size"]
                self.backbone_nca = ViTCA(patch_size=1, depth=config["model.vitca.depth"], heads=config["model.vitca.heads"],
                                           mlp_dim=config["model.vitca.mlp_dim"], dropout=config["model.vitca.dropout"], 
                                           cell_in_chns=input_channels, cell_out_chns=output_channels, cell_hidden_chns=channel_n - input_channels - output_channels,
                                           embed_cells=config["model.vitca.embed_cells"], embed_dim=config["model.vitca.embed_dim"],
                                           embed_dropout=config["model.vitca.embed_dropout"], 
                                           localize_attn=True, localized_attn_neighbourhood=[conv_size, conv_size], device=config["device"]
                                           )
            else:
                self.backbone_nca = backbone_class(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, 
                                            input_channels=input_channels, kernel_size=kernel_size, normalization=normalization)
                
            if compile:
                self.backbone_nca = torch.compile(self.backbone_nca)





        self.computed_upsampling_scales = []
        for i in range(len(self.octree_res)-1):
            t = []
            for c in range(2):
                t.append(self.octree_res[i][c]//self.octree_res[i+1][c])
            self.computed_upsampling_scales.append(np.array(t).reshape(1, 2))


        self.SAVE_VRAM_DURING_BATCHED_FORWARD = True

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1):
        #x: BCHW
        #y: BCHW

        x = x.permute(0, 2, 3, 1)
        if y is not None:
            y = y.permute(0, 2, 3, 1)

        #if y is not None:
        #    y = y.to(self.device)

        if self.training:
            assert x.shape[1:3] == self.octree_res[0], f"Expected shape {self.octree_res[0]}, got shape {x.shape[1:3]}"
            return self.forward_train(x, y, batch_duplication)
            
        else:
            return self.forward_eval(x)

    @torch.no_grad()
    def downscale(self, x: torch.Tensor, level: int):
        x = self.align_tensor_to(x, "BCHW")
        # self.remove_names(x)

        out = F.interpolate(x, size=self.octree_res[level])
        out.names = ('B', 'C', 'H', 'W')
        # x.names = ('B', 'C', 'H', 'W')
        return out
    
    """
    def remove_names(self, x: torch.Tensor):
        x.names = [None, None, None, None]
    """

    def align_tensor_to(self, x: torch.Tensor, to: str) -> torch.Tensor: # TEST
        # Use the number of channels position to determine current state
        # instead of string-based Named Tensors
        is_bchw = x.shape[1] == self.channel_n or x.shape[1] == self.input_channels
        
        if to == "BCHW":
            return x if is_bchw else x.permute(0, 3, 1, 2)
        if to == "BHWC":
            return x.permute(0, 2, 3, 1) if is_bchw else x
        return x

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, batch_duplication=1):
        #x: BHWC
        #y: BHWC

        #assert x.shape[:3] == y.shape[:3], f"Expected x and y to have the same number of channels, got {x.shape[:2]} and {y.shape[:2]}"
        #assert y.shape[3] == self.output_channels, f"Expected y to have {self.output_channels} channels, got {y.shape[3]}"
        #assert x.shape[3] == self.input_channels, f"Expected x to have {self.input_channels} channels, got {x.shape[3]}"

        input_ch = self.input_channels


        if self.loss_weighted_patching and not all([p is None for p in self.patch_sizes]):
            with torch.no_grad():
                # self.remove_names(x)
                self.remove_names(y)
                self.eval()
                if self.SAVE_VRAM_DURING_BATCHED_FORWARD: # activate this to minimize memory usage, resulting in lower runtime performance
                    initial_pred = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.output_channels), device=self.device)
                    for b in range(x.shape[0]):
                        initial_pred[b] = self.forward_eval(x[b:b+1])
                else:
                    initial_pred = self.forward_eval(x)
                self.train()

                loss = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]), device=self.device) # HWD 
                if len(initial_pred.shape) == 5 and y.shape[-1] == 1:
                    for m in range(y.shape[-1]):
                        temp = torch.nn.functional.binary_cross_entropy_with_logits(initial_pred[..., m].squeeze(), y[...].squeeze(), reduction='none')
                        loss += temp
                else:
                    for m in range(initial_pred.shape[-1]):
                        if 1 in y[..., m]:
                            temp = torch.nn.functional.binary_cross_entropy_with_logits(initial_pred[..., m].squeeze(), y[..., m].squeeze(), reduction='none')
                            loss += temp
            del initial_pred

        if batch_duplication != 1:
            x = torch.cat([x] * batch_duplication, dim=0)
            y = torch.cat([y] * batch_duplication, dim=0)
            if self.loss_weighted_patching and not all([p is None for p in self.patch_sizes]):
                loss = torch.cat([loss] * batch_duplication, dim=0)

        original = x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        # x.names = ('B', 'H', 'W', 'C')
        y.names = ('B', 'H', 'W', 'C')
        original.names = ('B', 'C', 'H', 'W')
        

        if self.patch_sizes[-1] is not None:
            x_new = torch.zeros(x.shape[0], *self.patch_sizes[-1], self.channel_n,
                                dtype=torch.float, device=self.device, 
                                names=('B', 'H', 'W', 'C'))
            current_patch = np.zeros((x.shape[0], 2, 2), dtype=int)
            current_patch = torch.from_numpy(current_patch).to(device)
            x = self.downscale(x, -1)
            x = self.align_tensor_to(x, "BHWC")
            self.remove_names(x_new)
            # self.remove_names(x)

            if self.loss_weighted_patching:
                loss_weighted_probabilities = self.compute_probabilities_matrix(loss, -1).cpu().numpy()

            for b in range(x.shape[0]):
                if self.loss_weighted_patching:
                    h_start, w_start = self.sample_index(loss_weighted_probabilities[b])
                else:
                    h_start = self.my_rand_int(0, self.octree_res[-1][0]-self.patch_sizes[-1][0])
                    w_start = self.my_rand_int(0, self.octree_res[-1][1]-self.patch_sizes[-1][1])
                current_patch[b] = np.array([[h_start, w_start], 
                                        [self.patch_sizes[-1][0] + h_start, 
                                        self.patch_sizes[-1][1] + w_start]
                                        ])
                
                x_new[b,:,:, :input_ch] = \
                x[b,    current_patch[b,0,0]:current_patch[b,1,0],
                        current_patch[b,0,1]:current_patch[b,1,1], :]
            x_new.names = ('B', 'H', 'W', 'C')
            x = x_new
        else:
            x_new = torch.zeros(x.shape[0], *self.octree_res[-1], self.channel_n,
                                dtype=torch.float, device=self.device)
            current_patch = np.array([[[0,0], [*self.octree_res[-1]]]] * x.shape[0])
            current_patch = torch.from_numpy(current_patch).to(device)
            x = self.downscale(x, -1)
            x = self.align_tensor_to(x, "BHWC")
            # self.remove_names(x)
            x_new[:,:,:, :input_ch] = x
            x = x_new
            # x.names = ('B', 'H', 'W', 'C')

        #x: BHWC

        # size_np   = np.array(test_size, dtype=current_patch.dtype) # Test
        offset_np = np.zeros(2, dtype=current_patch.dtype) # Test

        for level in range(len(self.octree_res)-1, -1, -1):

            x = self.align_tensor_to(x, "BHWC")
            # # self.remove_names(x) # Test

            if self.separate_models:
                x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            else:
                x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            # x.names = ('B', 'H', 'W', 'C') # Test


            if level > 0:
                #upscale states
                x = self.align_tensor_to(x, "BCHW")
                # # self.remove_names(x) # Test
                x = torch.nn.Upsample(scale_factor=tuple(self.computed_upsampling_scales[level-1][0]), 
                                      mode='nearest')(x)
                current_patch *= self.computed_upsampling_scales[level-1]
            
                original_right_resolution = self.downscale(original, level-1)
                assert original_right_resolution.names == ('B', 'C', 'H', 'W')
                self.remove_names(original_right_resolution)
                original_right_resolution = original_right_resolution.contiguous(memory_format=torch.channels_last)
                #cut out patch from input_channels
                test_size = self.patch_sizes[level-1]
                B, _, H, W = x.shape
                device = x.device
                if test_size is not None:
                    x_new = torch.zeros(x.shape[0], self.channel_n, *test_size, device=self.device, dtype=torch.float)
                    th, tw = self.patch_sizes[level-1][0], self.patch_sizes[level-1][1]
                    if self.loss_weighted_patching:
                        loss_weighted_probabilities = self.compute_probabilities_matrix(loss, level-1).cpu().numpy()

                    for b in range(x.shape[0]):
                        # ----- OFFSETS -----
                        h_offsets = torch.empty(B, dtype=torch.long, device=device)
                        w_offsets = torch.empty(B, dtype=torch.long, device=device)

                        for b in range(B):
                            if self.loss_weighted_patching:
                                temp = loss_weighted_probabilities[
                                    b,
                                    current_patch[b,0,0]:current_patch[b,1,0] + 1 - th,
                                    current_patch[b,0,1]:current_patch[b,1,1] + 1 - tw
                                ]

                                h, w, _ = self.sample_index(temp)

                                assert h <= H - th
                                assert w <= W - tw

                                h_offsets[b] = int(h)
                                w_offsets[b] = int(w)
                            else:
                                h_offsets[b] = self.my_rand_int(0, H - th)
                                w_offsets[b] = self.my_rand_int(0, W - tw)

                        # ----- UPDATE PATCH -----
                        
                        offsets = torch.stack([h_offsets, w_offsets], dim=1)   # shape [B, 2] # Test

                        current_patch[:, 0] += offsets # Test
                        current_patch[:, 1] = current_patch[:, 0] + torch.tensor(test_size, device=device) # Test

                        # ----- COPY (EXACT ORIGINAL ORDER) -----
                        # Build index grids from current_patch
                        hs = torch.tensor(current_patch[:,0,0], device=device)
                        ws = torch.tensor(current_patch[:,0,1], device=device)

                        # create ranges
                        h_range = hs[:,None] + torch.arange(th, device=device)
                        w_range = ws[:,None] + torch.arange(tw, device=device)

                        # ---- FIRST PART ----
                        x_new[:, :input_ch] = original_right_resolution[
                            torch.arange(B, device=device)[:,None,None],
                            :input_ch,
                            h_range[:,:,None],
                            w_range[:,None,:]
                        ]

                        # ---- SECOND PART (uses raw offsets) ----
                        h2 = h_offsets[:,None] + torch.arange(th, device=device)
                        w2 = w_offsets[:,None] + torch.arange(tw, device=device)

                        x_new[:, input_ch:] = x[
                            torch.arange(B, device=device)[:,None,None],
                            input_ch:,
                            h2[:,:,None],
                            w2[:,None,:]
                        ]

                    x = x_new # NOT the right x at the moment!!! # EDIT: seems good now
                else:
                    for b in range(x.shape[0]):
                        x[b, :input_ch] = original_right_resolution[b, :,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1]]
                # # x.names = ('B', 'C', 'H', 'W') # Test
        
        #x: BHWC
        y_new = torch.zeros(y.shape[0], x.shape[1], x.shape[2],
                             y.shape[3], device=self.device, dtype=torch.float)
        for b in range(x.shape[0]):
            y_new[b] = y[b, current_patch[b,0,0]:current_patch[b,1,0],
                        current_patch[b,0,1]:current_patch[b,1,1], :] 
        y = y_new
        
        # self.remove_names(x)
        logits = x[..., input_ch:input_ch+self.output_channels]
        hidden = x[..., input_ch+self.output_channels:]

        ret_dict = {'logits': logits, 'target': y, 'hidden_channels': hidden}


        if self.apply_nonlin is not None:
            probabilites = self.apply_nonlin(logits)
            ret_dict['probabilities'] = probabilites


        return ret_dict
    
    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        temp = self.patch_sizes
        self.patch_sizes = [None] * len(self.patch_sizes)



        if x.shape[1:3] != self.octree_res[0]:
            temp_octree_res = self.octree_res
            new_octree_res = [list(x.shape[1:3])]
            for i in range(1, len(self.octree_res)):
                downsample_factor = np.array(self.octree_res[i-1]) / np.array(self.octree_res[i])
                new_octree_res.append([math.ceil(new_octree_res[i-1][0] / downsample_factor[0]), 
                                        math.ceil(new_octree_res[i-1][1] / downsample_factor[1])])
            self.octree_res = new_octree_res
            print("running inference on different resolution, this is the new resolution:", self.octree_res)


        out = self.forward_train(x, x)
        out.pop("target")#target contains the input anyways so we remove it here!

        if x.shape[1:3] != tuple(self.octree_res[0]):
            self.octree_res = temp_octree_res

        self.patch_sizes = temp
        return out
    
    def create_inference_series(self, x: torch.Tensor, per_step: bool):
        inference_series = [] #list of BHWC tensors
        #x: BCHW
        original = x
        x = x.permute(0, 2, 3, 1)
        # x.names = ('B', 'H', 'W', 'C')
        original.names = ('B', 'C', 'H', 'W')
        
        x_new = torch.zeros(x.shape[0], *self.octree_res[-1], self.channel_n,
                                dtype=torch.float, device=self.device)
        current_patch = np.array([[[0,0], [*self.octree_res[-1]]]] * x.shape[0])
        x = self.downscale(x, -1)
        x = self.align_tensor_to(x, "BHWC")
        # self.remove_names(x)
        x_new[:,:,:, :self.input_channels] = x
        x = x_new
        # x.names = ('B', 'H', 'W', 'C')

        for level in range(len(self.octree_res)-1, -1, -1):

            x = self.align_tensor_to(x, "BHWC")
            # self.remove_names(x)

            if per_step:
                pass
            else:
                inference_series.append(x)

            if per_step:
                if self.separate_models:
                    x, gallery = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate, visualize=True)
                else:
                    x, gallery = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate, visualize=True)
                inference_series.append(gallery)
            else:
                if self.separate_models:
                    x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
                else:
                    x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            # x.names = ('B', 'H', 'W', 'C')

            if not per_step:
                inference_series.append(x)


            if level > 0:
                #upscale states
                x = self.align_tensor_to(x, "BCHW")
                # self.remove_names(x)
                x = torch.nn.Upsample(scale_factor=tuple(self.computed_upsampling_scales[level-1][0]), 
                                      mode='nearest')(x)
                current_patch *= self.computed_upsampling_scales[level-1]
            
                original_right_resolution = self.downscale(original, level-1)
                assert original_right_resolution.names == ('B', 'C', 'H', 'W')
                self.remove_names(original_right_resolution)
                original_right_resolution = original_right_resolution.contiguous(memory_format=torch.channels_last)

                for b in range(x.shape[0]):
                        x[b, :self.input_channels] = original_right_resolution[b, :,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1]]
                # x.names = ('B', 'C', 'H', 'W')
        
        return inference_series
    
    def my_rand_int(self, low, high):
        if high == low:
            return low
        assert high > low, f"high must be greater than low, got {low} and {high}"
        return random.randint(low, high)
        #return np.random.randint(low, high)
    
    def sample_index(self, p):
        #https://stackoverflow.com/questions/61047932/numpy-sampling-from-a-2d-numpy-array-of-probabilities
        p = p / np.sum(p)
        i = np.random.choice(np.arange(p.size), p=p.ravel())
        return np.unravel_index(i, p.shape)

    @torch.no_grad()
    def compute_probabilities_matrix(self, loss: torch.Tensor, level: int) -> torch.Tensor:
        assert False, "Not implemented yet"
        patch_size = self.patch_sizes[level]
        loss = loss.unsqueeze(1)
        loss = F.interpolate(loss, size=self.octree_res[level])
        loss_per_patch = F.conv3d(loss, torch.ones((1, 1, *patch_size), device=self.device), padding=(0,0,0))
        loss_per_patch = loss_per_patch[:,0]
        loss_per_patch = loss_per_patch / torch.sum(loss_per_patch, dim=(1,2,3)).view(loss_per_patch.shape[0], 1, 1, 1)
        return loss_per_patch