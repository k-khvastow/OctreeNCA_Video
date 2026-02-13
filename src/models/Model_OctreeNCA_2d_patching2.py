import numpy as np
import torch
import torch.nn as nn
from src.models.Model_BasicNCA2D import BasicNCA2D
from src.models.Model_BasicNCA2D_fast import BasicNCA2DFast
from src.models.Model_ViTCA import ViTCA
import torchio as tio
import random
import math
import torch.nn.functional as F
import subprocess as sp

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

        compile = bool(config['performance.compile'])
        compile_mode = config.get("performance.compile.mode", None)
        compile_backend = config.get("performance.compile.backend", None)
        compile_dynamic = config.get("performance.compile.dynamic", None)
        compile_fullgraph = bool(config.get("performance.compile.fullgraph", False))

        normalization = config.get("model.normalization", "batch")

        self.apply_nonlin = config.get("model.apply_nonlin", None)
        self.apply_nonlin = eval(self.apply_nonlin) if self.apply_nonlin is not None else None
        if isinstance(self.apply_nonlin, nn.Softmax) and self.apply_nonlin.dim == 1:
            print(
                "[OctreeNCA2DPatch2] model.apply_nonlin=Softmax(dim=1) detected, "
                "but logits are channel-last (BHWC). Switching to Softmax(dim=-1)."
            )
            self.apply_nonlin = nn.Softmax(dim=-1)

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
        self.use_vitca = bool(config["model.vitca"])

        if compile:
            torch.set_float32_matmul_precision('high')
        self._compile_enabled = compile
        self._compile_mode = compile_mode
        self._compile_backend = compile_backend
        self._compile_dynamic = compile_dynamic
        self._compile_fullgraph = compile_fullgraph
        
        if separate_models:
            if isinstance(kernel_size, list):
                assert len(kernel_size) == len(octree_res_and_steps), "kernel_size must have same length as octree_res_and_steps"
            else:
                kernel_size = [kernel_size] * len(octree_res_and_steps)
        else:
            assert isinstance(kernel_size, int), "kernel_size must be an integer"

        backbone_class = eval(config.get("model.backbone_class", "BasicNCA2D"))
        assert backbone_class in [BasicNCA2D, BasicNCA2DFast], f"backbone_class must be either BasicNCA2D, got {backbone_class}"
        backbone_inplace_relu = bool(config.get("performance.inplace_operations", False))
        backbone_tbptt_steps = config.get("model.backbone.tbptt_steps", None)
        backbone_spectral_norm = bool(config.get("model.spectral_norm", False))

        def _build_backbone(kernel_size_value):
            kwargs = dict(
                channel_n=channel_n,
                fire_rate=fire_rate,
                device=device,
                hidden_size=hidden_size,
                input_channels=input_channels,
                kernel_size=kernel_size_value,
                normalization=normalization,
            )
            if backbone_class == BasicNCA2DFast:
                kwargs["inplace_relu"] = backbone_inplace_relu
                kwargs["tbptt_steps"] = backbone_tbptt_steps
                kwargs["use_spectral_norm"] = backbone_spectral_norm
            return backbone_class(**kwargs)

        if separate_models:
            if self.use_vitca:
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
                self.backbone_ncas = nn.ModuleList([
                    _build_backbone(kernel_size[l]) for l in range(len(octree_res_and_steps))
                ])
            if compile:
                for i, model in enumerate(self.backbone_ncas):
                    self.backbone_ncas[i] = self._compile_module(model)
        else:
            if self.use_vitca:
                conv_size = config["kernel_size"]
                self.backbone_nca = ViTCA(patch_size=1, depth=config["model.vitca.depth"], heads=config["model.vitca.heads"],
                                           mlp_dim=config["model.vitca.mlp_dim"], dropout=config["model.vitca.dropout"], 
                                           cell_in_chns=input_channels, cell_out_chns=output_channels, cell_hidden_chns=channel_n - input_channels - output_channels,
                                           embed_cells=config["model.vitca.embed_cells"], embed_dim=config["model.vitca.embed_dim"],
                                           embed_dropout=config["model.vitca.embed_dropout"], 
                                           localize_attn=True, localized_attn_neighbourhood=[conv_size, conv_size], device=config["device"]
                                           )
            else:
                self.backbone_nca = _build_backbone(kernel_size)
                
            if compile:
                self.backbone_nca = self._compile_module(self.backbone_nca)





        self.computed_upsampling_scales = []
        for i in range(len(self.octree_res)-1):
            t = []
            for c in range(2):
                t.append(self.octree_res[i][c]//self.octree_res[i+1][c])
            self.computed_upsampling_scales.append(np.array(t).reshape(1, 2))


        self.SAVE_VRAM_DURING_BATCHED_FORWARD = True
        self._backbone_supports_bchw = (not self.use_vitca) and (backbone_class == BasicNCA2DFast)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1):
        #x: BCHW
        #y: BCHW

        if self.training:
            assert x.shape[2:4] == self.octree_res[0], f"Expected shape {self.octree_res[0]}, got shape {x.shape[2:4]}"
            return self.forward_train(x, y, batch_duplication)
            
        else:
            return self.forward_eval(x)

    @staticmethod
    def _bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)

    @staticmethod
    def _bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)

    def _compile_module(self, module: nn.Module) -> nn.Module:
        if not hasattr(torch, "compile"):
            return module
        compile_kwargs = {}
        if self._compile_mode not in (None, "", "none", "default"):
            compile_kwargs["mode"] = self._compile_mode
        if self._compile_backend not in (None, "", "none"):
            compile_kwargs["backend"] = self._compile_backend
        if self._compile_dynamic is not None:
            compile_kwargs["dynamic"] = bool(self._compile_dynamic)
        if self._compile_fullgraph:
            compile_kwargs["fullgraph"] = True
        try:
            return torch.compile(module, **compile_kwargs)
        except TypeError:
            # Fallback for older torch versions without all kwargs.
            return torch.compile(module)

    def _get_backbone_for_level(self, level: int) -> nn.Module:
        return self.backbone_ncas[level] if self.separate_models else self.backbone_nca

    def _run_backbone(self, x_bchw: torch.Tensor, level: int, visualize: bool = False):
        model = self._get_backbone_for_level(level)
        steps = self.inference_steps[level]
        kwargs = {"steps": steps, "fire_rate": self.fire_rate}
        if visualize:
            kwargs["visualize"] = True
        if self._backbone_supports_bchw:
            kwargs["input_layout"] = "BCHW"
            return model(x_bchw, **kwargs)

        x_bhwc = self._bchw_to_bhwc(x_bchw)
        out = model(x_bhwc, **kwargs)
        if visualize:
            out_state, gallery = out
            return self._bhwc_to_bchw(out_state), gallery
        return self._bhwc_to_bchw(out)

    @torch.no_grad()
    def downscale(self, x: torch.Tensor, level: int, layout: str = "BCHW"):
        if layout == "BCHW":
            return F.interpolate(x, size=self.octree_res[level])
        if layout == "BHWC":
            out = F.interpolate(self._bhwc_to_bchw(x), size=self.octree_res[level])
            return self._bchw_to_bhwc(out)
        raise ValueError(f"Unknown layout {layout}")
    
    def remove_names(self, x: torch.Tensor):
        # Named tensors are intentionally disabled for performance.
        return x

    def align_tensor_to(self, x: torch.Tensor, to: str) -> torch.Tensor:
        named = x.names is not None and any(n is not None for n in x.names)
        if named:
            if to == "BCHW":
                if x.names == ('B', 'H', 'W', 'C'):
                    return self._bhwc_to_bchw(x.rename(None))
                if x.names == ('B', 'C', 'H', 'W'):
                    return x.rename(None)
            elif to == "BHWC":
                if x.names == ('B', 'C', 'H', 'W'):
                    return self._bchw_to_bhwc(x.rename(None))
                if x.names == ('B', 'H', 'W', 'C'):
                    return x.rename(None)
            raise ValueError(f"Expected to be aligned to BCHW or BHWC, got {to}")

        is_bchw = x.shape[1] in (self.input_channels, self.output_channels, self.channel_n)
        is_bhwc = x.shape[-1] in (self.input_channels, self.output_channels, self.channel_n)
        if to == "BCHW":
            if is_bchw and not is_bhwc:
                return x
            if is_bhwc:
                return self._bhwc_to_bchw(x)
        elif to == "BHWC":
            if is_bhwc and not is_bchw:
                return x
            if is_bchw:
                return self._bchw_to_bhwc(x)
        raise ValueError(
            f"Ambiguous tensor layout for shape {tuple(x.shape)} when aligning to {to}."
        )

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, batch_duplication=1):
        #x: BCHW
        #y: BCHW
        input_ch = self.input_channels

        if self.loss_weighted_patching and not all([p is None for p in self.patch_sizes]):
            x_bhwc = self._bchw_to_bhwc(x)
            y_bhwc = self._bchw_to_bhwc(y)
            with torch.no_grad():
                self.eval()
                if self.SAVE_VRAM_DURING_BATCHED_FORWARD: # activate this to minimize memory usage, resulting in lower runtime performance
                    initial_pred = torch.zeros(
                        (x_bhwc.shape[0], x_bhwc.shape[1], x_bhwc.shape[2], self.output_channels),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    for b in range(x_bhwc.shape[0]):
                        pred = self.forward_eval(x[b:b+1])
                        pred_logits = pred["logits"] if isinstance(pred, dict) else pred
                        initial_pred[b] = pred_logits.squeeze(0)
                else:
                    pred = self.forward_eval(x)
                    initial_pred = pred["logits"] if isinstance(pred, dict) else pred
                self.train()

                loss = torch.zeros((x_bhwc.shape[0], x_bhwc.shape[1], x_bhwc.shape[2], x_bhwc.shape[3]), device=self.device)
                if len(initial_pred.shape) == 5 and y_bhwc.shape[-1] == 1:
                    for m in range(y_bhwc.shape[-1]):
                        temp = torch.nn.functional.binary_cross_entropy_with_logits(
                            initial_pred[..., m].squeeze(), y_bhwc[...].squeeze(), reduction='none'
                        )
                        loss += temp
                else:
                    for m in range(initial_pred.shape[-1]):
                        if 1 in y_bhwc[..., m]:
                            temp = torch.nn.functional.binary_cross_entropy_with_logits(
                                initial_pred[..., m].squeeze(), y_bhwc[..., m].squeeze(), reduction='none'
                            )
                            loss += temp
            del initial_pred

        if batch_duplication != 1:
            x = torch.cat([x] * batch_duplication, dim=0)
            y = torch.cat([y] * batch_duplication, dim=0)
            if self.loss_weighted_patching and not all([p is None for p in self.patch_sizes]):
                loss = torch.cat([loss] * batch_duplication, dim=0)

        original_bchw = x

        if self.patch_sizes[-1] is not None:
            patch_h, patch_w = self.patch_sizes[-1]
            x_new = x.new_zeros((x.shape[0], self.channel_n, patch_h, patch_w))
            current_patch = np.zeros((x.shape[0], 2, 2), dtype=np.int64)
            x = self.downscale(x, -1, layout="BCHW")

            if self.loss_weighted_patching:
                loss_weighted_probabilities = self.compute_probabilities_matrix(loss, -1).cpu().numpy()

            for b in range(x.shape[0]):
                if self.loss_weighted_patching:
                    h_start, w_start = self.sample_index(loss_weighted_probabilities[b])
                else:
                    h_start = self.my_rand_int(0, self.octree_res[-1][0] - patch_h)
                    w_start = self.my_rand_int(0, self.octree_res[-1][1] - patch_w)
                current_patch[b] = np.array([[h_start, w_start], 
                                        [patch_h + h_start, 
                                        patch_w + w_start]
                                        ])
                
                x_new[b, :input_ch] = x[b,
                                        :input_ch,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1]]
            x = x_new
        else:
            x_new = x.new_zeros((x.shape[0], self.channel_n, *self.octree_res[-1]))
            current_patch = np.zeros((x.shape[0], 2, 2), dtype=np.int64)
            current_patch[:, 1, 0] = self.octree_res[-1][0]
            current_patch[:, 1, 1] = self.octree_res[-1][1]
            x = self.downscale(x, -1, layout="BCHW")
            x_new[:, :input_ch] = x[:, :input_ch]
            x = x_new

        #x: BCHW
        for level in range(len(self.octree_res)-1, -1, -1):
            x = self._run_backbone(x, level)

            if level > 0:
                scale_h, scale_w = self.computed_upsampling_scales[level-1][0]
                scale_h, scale_w = int(scale_h), int(scale_w)
                x = x.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                current_patch[:, :, 0] *= scale_h
                current_patch[:, :, 1] *= scale_w

                original_right_resolution = self.downscale(original_bchw, level-1, layout="BCHW")

                if self.patch_sizes[level-1] is not None:
                    patch_h, patch_w = self.patch_sizes[level-1]
                    x_new = x.new_zeros((x.shape[0], self.channel_n, patch_h, patch_w))
                    
                    if self.loss_weighted_patching:
                        loss_weighted_probabilities = self.compute_probabilities_matrix(loss, level-1).cpu().numpy()

                    for b in range(x.shape[0]):
                        if self.loss_weighted_patching:
                            temp = loss_weighted_probabilities[b,
                                                                          current_patch[b,0,0]:current_patch[b,1,0]+1-patch_h,
                                                                          current_patch[b,0,1]:current_patch[b,1,1]+1-patch_w
                                                                          ]
                            h_start, w_start = self.sample_index(temp)
                            h_offset = int(h_start)
                            w_offset = int(w_start)
                            assert h_offset <= x.shape[2] - patch_h
                            assert w_offset <= x.shape[3] - patch_w
                        else:
                            h_offset = self.my_rand_int(0, x.shape[2] - patch_h)
                            w_offset = self.my_rand_int(0, x.shape[3] - patch_w)

                        current_patch[b, 0] += np.array([h_offset, w_offset], dtype=np.int64)
                        current_patch[b, 1] = current_patch[b, 0] + np.array([patch_h, patch_w], dtype=np.int64)
                        
                        x_new[b, :input_ch] = original_right_resolution[b,
                                        :input_ch,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1]]
                        
                        x_new[b, input_ch:] = x[b,
                                        input_ch:,
                                        h_offset:h_offset + patch_h,
                                        w_offset:w_offset + patch_w]
                    x = x_new
                else:
                    for b in range(x.shape[0]):
                        x[b, :input_ch] = original_right_resolution[b,
                                        :input_ch,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1]]
        
        #x: BCHW
        y_new = y.new_zeros((y.shape[0], y.shape[1], x.shape[2], x.shape[3]))
        for b in range(x.shape[0]):
            y_new[b] = y[b, :,
                        current_patch[b,0,0]:current_patch[b,1,0],
                        current_patch[b,0,1]:current_patch[b,1,1]]
        y = y_new
        
        logits = x[:, input_ch:input_ch+self.output_channels]
        hidden = x[:, input_ch+self.output_channels:]

        logits = self._bchw_to_bhwc(logits)
        y = self._bchw_to_bhwc(y)
        hidden = self._bchw_to_bhwc(hidden)

        ret_dict = {'logits': logits, 'target': y, 'hidden_channels': hidden}

        if self.apply_nonlin is not None:
            probabilites = self.apply_nonlin(logits)
            ret_dict['probabilities'] = probabilites

        return ret_dict
    
    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        temp = self.patch_sizes
        self.patch_sizes = [None] * len(self.patch_sizes)



        if x.shape[2:4] != self.octree_res[0]:
            temp_octree_res = self.octree_res
            new_octree_res = [list(x.shape[2:4])]
            for i in range(1, len(self.octree_res)):
                downsample_factor = np.array(self.octree_res[i-1]) / np.array(self.octree_res[i])
                new_octree_res.append([math.ceil(new_octree_res[i-1][0] / downsample_factor[0]), 
                                        math.ceil(new_octree_res[i-1][1] / downsample_factor[1])])
            self.octree_res = new_octree_res
            print("running inference on different resolution, this is the new resolution:", self.octree_res)


        out = self.forward_train(x, x)
        out.pop("target")#target contains the input anyways so we remove it here!

        if x.shape[2:4] != tuple(self.octree_res[0]):
            self.octree_res = temp_octree_res

        self.patch_sizes = temp
        return out
    
    def create_inference_series(self, x: torch.Tensor, per_step: bool):
        inference_series = [] #list of BHWC tensors
        #x: BCHW
        original = x
        
        x_new = x.new_zeros((x.shape[0], self.channel_n, *self.octree_res[-1]))
        current_patch = np.zeros((x.shape[0], 2, 2), dtype=np.int64)
        current_patch[:, 1, 0] = self.octree_res[-1][0]
        current_patch[:, 1, 1] = self.octree_res[-1][1]
        x = self.downscale(x, -1, layout="BCHW")
        x_new[:, :self.input_channels] = x[:, :self.input_channels]
        x = x_new

        for level in range(len(self.octree_res)-1, -1, -1):
            if per_step:
                pass
            else:
                inference_series.append(self._bchw_to_bhwc(x))

            if per_step:
                x, gallery = self._run_backbone(x, level, visualize=True)
                inference_series.append(gallery)
            else:
                x = self._run_backbone(x, level)

            if not per_step:
                inference_series.append(self._bchw_to_bhwc(x))


            if level > 0:
                #upscale states
                scale_h, scale_w = self.computed_upsampling_scales[level-1][0]
                scale_h, scale_w = int(scale_h), int(scale_w)
                x = x.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                current_patch[:, :, 0] *= scale_h
                current_patch[:, :, 1] *= scale_w
            
                original_right_resolution = self.downscale(original, level-1, layout="BCHW")

                for b in range(x.shape[0]):
                        x[b, :self.input_channels] = original_right_resolution[b,
                                        :self.input_channels,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1]]
        
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
