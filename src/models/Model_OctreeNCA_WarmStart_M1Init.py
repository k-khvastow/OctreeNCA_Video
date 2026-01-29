import os
from typing import Optional

import torch
import torch.nn as nn

from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
from src.models.Model_OctreeNCA_WarmStart import OctreeNCA2DWarmStart


class OctreeNCA2DWarmStartM1Init(nn.Module):
    """
    Two-model warm start:
      - M1 (single-image NCA) initializes hidden state from the first frame.
      - M2 (warm-start NCA) rolls forward on subsequent frames.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.m1 = OctreeNCA2DPatch2(config)
        self.m2 = OctreeNCA2DWarmStart(config)

        self.channel_n = self.m2.channel_n
        self.input_channels = self.m2.input_channels
        self.output_channels = self.m2.output_channels

        self.freeze_m1 = bool(config.get('model.m1.freeze', False))
        self.m1_use_probs = bool(config.get('model.m1.use_probs', False))
        self.m1_eval_mode = bool(config.get('model.m1.eval_mode', self.freeze_m1))

        self._load_m1_weights()
        self._apply_m1_freeze()

    def _resolve_m1_path(self, path: Optional[str]) -> Optional[str]:
        if path is None or path == "":
            return None
        if os.path.isdir(path):
            return os.path.join(path, 'model.pth')
        return path

    def _load_m1_weights(self) -> None:
        path = self._resolve_m1_path(self.config.get('model.m1.pretrained_path', None))
        if path is None:
            return
        strict = bool(self.config.get('model.m1.load_strict', True))
        state = torch.load(path, map_location='cpu')
        self.m1.load_state_dict(state, strict=strict)

    def _apply_m1_freeze(self) -> None:
        if self.freeze_m1:
            for p in self.m1.parameters():
                p.requires_grad = False
        if self.m1_eval_mode:
            self.m1.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.m1_eval_mode:
            self.m1.eval()
        return self

    def init_state_from_m1(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: BCHW, y: BCHW
        if y is None:
            y = torch.zeros(
                (x.shape[0], self.output_channels, x.shape[2], x.shape[3]),
                device=x.device,
                dtype=x.dtype,
            )

        if self.freeze_m1:
            with torch.no_grad():
                out = self.m1(x, y)
        else:
            out = self.m1(x, y)

        logits = out['logits']
        hidden = out.get('hidden_channels', None)
        if hidden is None:
            raise RuntimeError("M1 output missing hidden_channels.")

        x_bhwc = x.permute(0, 2, 3, 1)
        state = torch.zeros(
            x_bhwc.shape[0],
            x_bhwc.shape[1],
            x_bhwc.shape[2],
            self.channel_n,
            device=x.device,
            dtype=x_bhwc.dtype,
        )
        state[..., :self.input_channels] = x_bhwc[..., :self.input_channels]

        prev_out = logits
        if self.m1_use_probs and 'probabilities' in out:
            prev_out = out['probabilities']

        if prev_out.shape[-1] != self.output_channels:
            raise RuntimeError(
                f"M1 output channels ({prev_out.shape[-1]}) do not match expected "
                f"output_channels ({self.output_channels})."
            )
        expected_hidden = self.channel_n - self.input_channels - self.output_channels
        if hidden.shape[-1] != expected_hidden:
            raise RuntimeError(
                f"M1 hidden_channels ({hidden.shape[-1]}) do not match expected ({expected_hidden})."
            )

        state[..., self.input_channels:self.input_channels + self.output_channels] = prev_out
        state[..., self.input_channels + self.output_channels:] = hidden
        return state

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        prev_state: Optional[torch.Tensor] = None,
        batch_duplication=1,
    ):
        return self.m2(x, y, prev_state=prev_state, batch_duplication=batch_duplication)
