import torch
import torch.nn as nn

from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2


class OctreeNCA2DDualView(OctreeNCA2DPatch2):
    """
    Dual-view OctreeNCA.

    - Takes two separate images (view A + view B) at the same timestamp
    - Runs two OctreeNCA state evolutions (one per view) with shared backbone weights
    - Exchanges information between views via feature-level FiLM on hidden channels
    - Produces two segmentations (one per view) in a single forward pass
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.cross_fusion = str(config.get("model.dual_view.cross_fusion", "film")).lower()
        self.cross_strength = float(config.get("model.dual_view.cross_strength", 0.5))
        self.cross_use_tanh = bool(config.get("model.dual_view.cross_use_tanh", True))

        hidden_start = self.input_channels + self.output_channels
        hidden_dim = max(0, self.channel_n - hidden_start)
        self._dual_hidden_start = hidden_start
        self._dual_hidden_dim = hidden_dim

        self.cross_film = None
        if self.cross_fusion in ("film", "fi lm", "film_hidden") and hidden_dim > 0:
            layers = []
            for _ in range(len(self.octree_res)):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                )
                with torch.no_grad():
                    mlp[-1].weight.zero_()
                    mlp[-1].bias.zero_()
                layers.append(mlp)
            self.cross_film = nn.ModuleList(layers)

        if self.cross_fusion not in ("none", "film", "fi lm", "film_hidden"):
            raise ValueError(
                "model.dual_view.cross_fusion must be one of: 'none', 'film'. "
                f"Got '{self.cross_fusion}'."
            )

    def _maybe_cross_fuse(self, state_a: torch.Tensor, state_b: torch.Tensor, level: int):
        if self.cross_fusion == "none":
            return state_a, state_b
        if self.cross_film is None or self._dual_hidden_dim <= 0:
            return state_a, state_b

        hs = self._dual_hidden_start
        ha = state_a[:, hs:]
        hb = state_b[:, hs:]

        pooled_a = ha.mean(dim=(2, 3))
        pooled_b = hb.mean(dim=(2, 3))

        params_a = self.cross_film[level](pooled_b)
        params_b = self.cross_film[level](pooled_a)
        scale_a, shift_a = params_a.chunk(2, dim=1)
        scale_b, shift_b = params_b.chunk(2, dim=1)

        strength = self.cross_strength
        if self.cross_use_tanh:
            scale_a = torch.tanh(scale_a) * strength
            shift_a = torch.tanh(shift_a) * strength
            scale_b = torch.tanh(scale_b) * strength
            shift_b = torch.tanh(shift_b) * strength
        else:
            scale_a = scale_a * strength
            shift_a = shift_a * strength
            scale_b = scale_b * strength
            shift_b = shift_b * strength

        ha = ha * (1.0 + scale_a[:, :, None, None]) + shift_a[:, :, None, None]
        hb = hb * (1.0 + scale_b[:, :, None, None]) + shift_b[:, :, None, None]

        state_a = torch.cat([state_a[:, :hs], ha], dim=1)
        state_b = torch.cat([state_b[:, :hs], hb], dim=1)
        return state_a, state_b

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, y_a: torch.Tensor = None, y_b: torch.Tensor = None, batch_duplication=1):
        # x_*.shape: BCHW
        if self.training:
            if x_a.shape[2:4] != self.octree_res[0]:
                raise ValueError(f"Expected view A shape {self.octree_res[0]}, got {tuple(x_a.shape[2:4])}.")
            if x_b.shape[2:4] != self.octree_res[0]:
                raise ValueError(f"Expected view B shape {self.octree_res[0]}, got {tuple(x_b.shape[2:4])}.")
            return self.forward_train(x_a, x_b, y_a, y_b, batch_duplication=batch_duplication)
        return self.forward_eval(x_a, x_b)

    def forward_train(self, x_a: torch.Tensor, x_b: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, batch_duplication=1):
        # x_*, y_*: BCHW
        if batch_duplication != 1:
            x_a = torch.cat([x_a] * batch_duplication, dim=0)
            x_b = torch.cat([x_b] * batch_duplication, dim=0)
            y_a = torch.cat([y_a] * batch_duplication, dim=0)
            y_b = torch.cat([y_b] * batch_duplication, dim=0)

        input_ch = self.input_channels
        original_a = x_a
        original_b = x_b

        # Initialize state at coarsest level for both views.
        state_a = x_a.new_zeros((x_a.shape[0], self.channel_n, *self.octree_res[-1]))
        state_b = x_b.new_zeros((x_b.shape[0], self.channel_n, *self.octree_res[-1]))
        xa_coarse = self.downscale(x_a, -1, layout="BCHW")
        xb_coarse = self.downscale(x_b, -1, layout="BCHW")
        state_a[:, :input_ch] = xa_coarse[:, :input_ch]
        state_b[:, :input_ch] = xb_coarse[:, :input_ch]

        for level in range(len(self.octree_res) - 1, -1, -1):
            # NOTE: When the backbone is wrapped by `torch.compile()` with CUDA graphs,
            # sequential invocations can return tensors backed by the same internal
            # output buffer. Calling the backbone once on a concatenated batch avoids
            # "tensor output of CUDAGraphs ... overwritten by a subsequent run" errors.
            if state_a.shape != state_b.shape:
                raise ValueError(f"Expected both views to share state shape, got {tuple(state_a.shape)} vs {tuple(state_b.shape)}.")
            state_ab = torch.cat([state_a, state_b], dim=0)
            state_ab = self._run_backbone(state_ab, level)
            state_a, state_b = state_ab.chunk(2, dim=0)
            state_a, state_b = self._maybe_cross_fuse(state_a, state_b, level)

            if level > 0:
                scale_h, scale_w = self.computed_upsampling_scales[level - 1][0]
                scale_h, scale_w = int(scale_h), int(scale_w)

                state_a = state_a.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                state_b = state_b.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)

                inj_a = self.downscale(original_a, level - 1, layout="BCHW")
                inj_b = self.downscale(original_b, level - 1, layout="BCHW")
                state_a[:, :input_ch] = inj_a[:, :input_ch]
                state_b[:, :input_ch] = inj_b[:, :input_ch]

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

        ret_dict = {"logits": logits, "target": target, "hidden_channels": hidden}
        if self.apply_nonlin is not None:
            ret_dict["probabilities"] = self.apply_nonlin(logits)
        return ret_dict

    @torch.no_grad()
    def forward_eval(self, x_a: torch.Tensor, x_b: torch.Tensor):
        temp = self.patch_sizes
        self.patch_sizes = [None] * len(self.patch_sizes)
        out = self.forward_train(x_a, x_b, x_a, x_b, batch_duplication=1)
        out.pop("target", None)
        self.patch_sizes = temp
        return out
