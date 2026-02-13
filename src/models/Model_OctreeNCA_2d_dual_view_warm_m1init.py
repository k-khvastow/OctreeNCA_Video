import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.Model_OctreeNCA_2d_dual_view import OctreeNCA2DDualView
from src.models.Model_OctreeNCA_2d_dual_view_warm import OctreeNCA2DDualViewWarmStart


class OctreeNCA2DDualViewWarmStartM1Init(nn.Module):
    """
    Two-model temporal dual-view setup:
      - M1 (dual-view cold model) initializes recurrent states from t=0.
      - M2 (dual-view warm model) rolls forward with hidden-state injection.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.m1 = OctreeNCA2DDualView(config)
        self.m2 = OctreeNCA2DDualViewWarmStart(config)

        self.channel_n = self.m2.channel_n
        self.input_channels = self.m2.input_channels
        self.output_channels = self.m2.output_channels

        self.freeze_m1 = bool(config.get("model.m1.freeze", False))
        self.m1_use_probs = bool(config.get("model.m1.use_probs", False))
        self.m1_eval_mode = bool(config.get("model.m1.eval_mode", self.freeze_m1))

        self.m2_init_from_m1 = bool(config.get("model.m2.init_from_m1", False))
        self.m2_share_backbone_with_m1 = bool(config.get("model.m2.share_backbone_with_m1", False))

        self._load_m1_weights()
        self._sync_m2_from_m1()
        self._apply_m1_freeze()

    def _sync_m2_from_m1(self) -> None:
        if not self.m2_init_from_m1 and not self.m2_share_backbone_with_m1:
            return

        if self.m2_share_backbone_with_m1 and self.m2_init_from_m1:
            raise ValueError("Set only one of model.m2.init_from_m1 or model.m2.share_backbone_with_m1.")

        if getattr(self.m1, "separate_models", None) != getattr(self.m2, "separate_models", None):
            raise RuntimeError("M1 and M2 mismatch: separate_models differs.")

        if self.m2_share_backbone_with_m1:
            if self.freeze_m1 or self.m1_eval_mode:
                raise ValueError(
                    "model.m2.share_backbone_with_m1=True requires "
                    "model.m1.freeze=False and model.m1.eval_mode=False."
                )
            if hasattr(self.m1, "backbone_ncas"):
                self.m2.backbone_ncas = self.m1.backbone_ncas
            else:
                self.m2.backbone_nca = self.m1.backbone_nca
            if getattr(self.m1, "cross_film", None) is not None:
                self.m2.cross_film = self.m1.cross_film
            return

        with torch.no_grad():
            copied = self.m2.load_state_dict(self.m1.state_dict(), strict=False)
        if len(copied.unexpected_keys) > 0:
            raise RuntimeError(
                f"Unexpected keys while copying M1 -> M2: {copied.unexpected_keys}"
            )

    def _resolve_m1_path(self, path: Optional[str]) -> Optional[str]:
        if path is None or path == "":
            return None
        if os.path.isdir(path):
            return os.path.join(path, "model.pth")
        return path

    @staticmethod
    def _remap_spectral_norm_keys(
        state: dict, model_state: dict
    ) -> dict:
        """Remap plain weight keys to spectral-norm parametrized keys and vice-versa.

        Handles checkpoints saved *without* spectral norm being loaded into a
        model that *has* spectral norm (and the reverse).
        """
        remapped = dict(state)
        model_keys = set(model_state.keys())
        state_keys = set(state.keys())

        # Case 1: checkpoint has "...fc1.weight" but model expects
        #         "...fc1.parametrizations.weight.original" (+ _u, _v buffers)
        for key in list(state_keys):
            if key.endswith(".weight") and key not in model_keys:
                param_key = key.replace(".weight", ".parametrizations.weight.original")
                if param_key in model_keys and param_key not in state_keys:
                    remapped[param_key] = remapped.pop(key)
                    # Carry over the model's initialized _u / _v buffers
                    prefix = key.rsplit(".weight", 1)[0]
                    for suffix in (
                        ".parametrizations.weight.0._u",
                        ".parametrizations.weight.0._v",
                    ):
                        buf_key = prefix + suffix
                        if buf_key in model_keys and buf_key not in remapped:
                            remapped[buf_key] = model_state[buf_key]

        # Case 2: checkpoint has "...fc1.parametrizations.weight.original"
        #         but model expects plain "...fc1.weight"
        for key in list(remapped.keys()):
            if ".parametrizations.weight.original" in key and key not in model_keys:
                plain_key = key.replace(
                    ".parametrizations.weight.original", ".weight"
                )
                if plain_key in model_keys and plain_key not in set(remapped.keys()):
                    remapped[plain_key] = remapped.pop(key)
            # Drop _u / _v vectors that the model doesn't expect
            elif (
                ".parametrizations.weight.0._u" in key
                or ".parametrizations.weight.0._v" in key
            ) and key not in model_keys:
                remapped.pop(key)

        return remapped

    def _load_m1_weights(self) -> None:
        path = self._resolve_m1_path(self.config.get("model.m1.pretrained_path", None))
        if path is None:
            return
        strict = bool(self.config.get("model.m1.load_strict", True))
        state = torch.load(path, map_location="cpu")
        state = self._remap_spectral_norm_keys(state, self.m1.state_dict())
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

    def _forward_m1(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
    ) -> dict:
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

        if self.freeze_m1:
            with torch.no_grad():
                out = self.m1(x_a, x_b, y_a, y_b)
        else:
            out = self.m1(x_a, x_b, y_a, y_b)
        return out

    def _split_dual_batch(self, tensor: torch.Tensor, batch_size: int, name: str):
        if tensor.shape[0] != 2 * batch_size:
            raise RuntimeError(
                f"Expected {name} batch dimension to be 2*B={2 * batch_size}, got {tensor.shape[0]}."
            )
        return tensor[:batch_size], tensor[batch_size:]

    def _states_from_m1_output(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        out: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = out["logits"]
        hidden = out.get("hidden_channels", None)
        if hidden is None:
            raise RuntimeError("M1 output missing hidden_channels.")

        batch_size = x_a.shape[0]
        logits_a, logits_b = self._split_dual_batch(logits, batch_size, "logits")
        hidden_a, hidden_b = self._split_dual_batch(hidden, batch_size, "hidden_channels")

        if self.m1_use_probs and "probabilities" in out:
            probs_a, probs_b = self._split_dual_batch(out["probabilities"], batch_size, "probabilities")
            logits_a, logits_b = probs_a, probs_b

        x_a_bhwc = x_a.permute(0, 2, 3, 1)
        x_b_bhwc = x_b.permute(0, 2, 3, 1)

        state_a = torch.zeros(
            x_a_bhwc.shape[0],
            x_a_bhwc.shape[1],
            x_a_bhwc.shape[2],
            self.channel_n,
            device=x_a.device,
            dtype=x_a_bhwc.dtype,
        )
        state_b = torch.zeros(
            x_b_bhwc.shape[0],
            x_b_bhwc.shape[1],
            x_b_bhwc.shape[2],
            self.channel_n,
            device=x_b.device,
            dtype=x_b_bhwc.dtype,
        )

        expected_hidden = self.channel_n - self.input_channels - self.output_channels
        if hidden_a.shape[-1] != expected_hidden or hidden_b.shape[-1] != expected_hidden:
            raise RuntimeError(
                f"M1 hidden_channels mismatch. Expected {expected_hidden}, got "
                f"{hidden_a.shape[-1]} (A) and {hidden_b.shape[-1]} (B)."
            )

        state_a[..., :self.input_channels] = x_a_bhwc[..., :self.input_channels]
        state_b[..., :self.input_channels] = x_b_bhwc[..., :self.input_channels]
        state_a[..., self.input_channels:self.input_channels + self.output_channels] = logits_a
        state_b[..., self.input_channels:self.input_channels + self.output_channels] = logits_b
        state_a[..., self.input_channels + self.output_channels:] = hidden_a
        state_b[..., self.input_channels + self.output_channels:] = hidden_b
        return state_a, state_b

    def m1_forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
    ) -> dict:
        return self._forward_m1(x_a, x_b, y_a, y_b)

    def init_states_from_m1(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
        m1_out: Optional[dict] = None,
    ):
        if m1_out is None:
            m1_out = self._forward_m1(x_a, x_b, y_a, y_b)
        return self._states_from_m1_output(x_a, x_b, m1_out)

    def m1_forward_and_init_states(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
    ):
        m1_out = self._forward_m1(x_a, x_b, y_a, y_b)
        states = self._states_from_m1_output(x_a, x_b, m1_out)
        return m1_out, states

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
        prev_state_a: Optional[torch.Tensor] = None,
        prev_state_b: Optional[torch.Tensor] = None,
        batch_duplication=1,
    ):
        return self.m2(
            x_a,
            x_b,
            y_a=y_a,
            y_b=y_b,
            prev_state_a=prev_state_a,
            prev_state_b=prev_state_b,
            batch_duplication=batch_duplication,
        )
