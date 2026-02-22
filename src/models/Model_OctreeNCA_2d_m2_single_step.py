"""
Model_OctreeNCA_2d_m2_single_step.py
──────────────────────────────────────────────────────────────────────────────
Two-model setup for keyframe-based temporal segmentation:

  - M1 (dual-view cold model) runs at a coarser finest resolution
    (e.g. 256×256) and produces seed hidden states once per keyframe.
  - M2 (dual-view warm model) operates at the full resolution (e.g. 512×512)
    and receives a *single* upscaled state from M1 — it does NOT carry its
    own hidden state across frames.

Key differences versus OctreeNCA2DDualViewWarmStartM1Init:
  1. ``model.m1.level_offset`` (int, default 0):
       Skip the first N resolution levels when building M1.  With offset=1
       and 4 total levels [[512,…],[256,…],[128,…],[64,…]], M1 gets
       [[256,…],[128,…],[64,…]] — i.e. M1 finest = 256×256.
  2. M1 inputs are automatically downscaled to M1's finest resolution before
     the forward pass.
  3. ``_states_from_m1_output`` bilinearly upscales M1's logits + hidden
     channels from M1's finest resolution to M2's finest resolution (512×512)
     so M2 can start from a full-resolution warm state.
  4. State is NOT meant to be carried across frames by M2 — the agent handles
     that policy (it always feeds M1's upscaled state, never M2's output).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Model_OctreeNCA_2d_dual_view import OctreeNCA2DDualView
from src.models.Model_OctreeNCA_2d_dual_view_warm import OctreeNCA2DDualViewWarmStart


class OctreeNCA2DM2SingleStep(nn.Module):
    """
    Keyframe M1 + single-step M2 model.

    M1 runs at a coarser finest resolution (e.g. 256×256).
    M2 runs at the full finest resolution (e.g. 512×512), conditioned on
    the upscaled M1 state.  M2 does exactly one forward step per query frame.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        full_res = config["model.octree.res_and_steps"]  # e.g. [[512,s],[256,s],[128,s],[64,s]]

        # ── M1 config ─────────────────────────────────────────────────────
        m1_level_offset = int(config.get("model.m1.level_offset", 0))
        m1_channel_n = config.get("model.m1.channel_n", None)
        m1_num_levels = config.get("model.m1.num_levels", None)

        if m1_level_offset < 0 or m1_level_offset >= len(full_res):
            raise ValueError(
                f"model.m1.level_offset={m1_level_offset} out of range (0..{len(full_res)-1})."
            )

        m1_res = full_res[m1_level_offset:]   # skip the first N finest levels

        if m1_num_levels is not None:
            m1_levels = int(m1_num_levels)
            if m1_levels < 1 or m1_levels > len(m1_res):
                raise ValueError(
                    f"model.m1.num_levels={m1_levels} out of range (1..{len(m1_res)}) "
                    f"after applying level_offset={m1_level_offset}."
                )
            m1_res = m1_res[:m1_levels]

        m1_config = dict(config)
        m1_config["model.octree.res_and_steps"] = m1_res
        if m1_channel_n is not None:
            m1_config["model.channel_n"] = int(m1_channel_n)

        # Truncate kernel_size / patch_sizes to M1 resolution count
        n_m1 = len(m1_res)
        ks = m1_config.get("model.kernel_size", None)
        if isinstance(ks, (list, tuple)) and len(ks) > n_m1:
            m1_config["model.kernel_size"] = list(ks)[:n_m1]
        ps = m1_config.get("model.train.patch_sizes", None)
        if isinstance(ps, (list, tuple)) and len(ps) > n_m1:
            m1_config["model.train.patch_sizes"] = list(ps)[:n_m1]

        self.m1 = OctreeNCA2DDualView(m1_config)
        # M1's finest spatial resolution (e.g. (256, 256))
        self.m1_finest_res: tuple[int, int] = tuple(m1_res[0][0])  # type: ignore[assignment]

        # ── M2 config ─────────────────────────────────────────────────────
        m2_num_levels = config.get("model.m2.num_levels", None)
        if m2_num_levels is not None:
            m2_levels = int(m2_num_levels)
            if m2_levels < 1 or m2_levels > len(full_res):
                raise ValueError(
                    f"model.m2.num_levels={m2_levels} out of range (1..{len(full_res)})."
                )
            m2_res = full_res[:m2_levels]
            m2_config = dict(config)
            m2_config["model.octree.res_and_steps"] = m2_res
            n_m2 = len(m2_res)
            ks = m2_config.get("model.kernel_size", None)
            if isinstance(ks, (list, tuple)) and len(ks) > n_m2:
                m2_config["model.kernel_size"] = list(ks)[:n_m2]
            ps = m2_config.get("model.train.patch_sizes", None)
            if isinstance(ps, (list, tuple)) and len(ps) > n_m2:
                m2_config["model.train.patch_sizes"] = list(ps)[:n_m2]
        else:
            m2_config = dict(config)
            m2_res = full_res

        # Override M2 kernel_size if a separate value is provided
        m2_ks = config.get("model.m2.kernel_size", None)
        if m2_ks is not None:
            m2_config["model.kernel_size"] = int(m2_ks)

        # Override M2 hidden_size (FC inside the NCA backbone) if specified
        m2_hs = config.get("model.m2.hidden_size", None)
        if m2_hs is not None:
            m2_config["model.hidden_size"] = int(m2_hs)

        # ── M2 extra hidden channels (start empty, appended after M1 hidden) ─
        self.m2_extra_hidden = int(config.get("model.m2.extra_hidden", 0))
        if self.m2_extra_hidden > 0:
            base_ch = int(m2_config.get("model.channel_n", 32))
            m2_config["model.channel_n"] = base_ch + self.m2_extra_hidden
            print(
                f"[M2SingleStep] M2 extra hidden channels: {self.m2_extra_hidden} "
                f"(base {base_ch} → effective channel_n {m2_config['model.channel_n']})"
            )

        self.m2 = OctreeNCA2DDualViewWarmStart(m2_config)
        # M2's finest spatial resolution (e.g. (512, 512))
        self.m2_finest_res: tuple[int, int] = tuple(m2_res[0][0])  # type: ignore[assignment]

        # ── Shared attributes ──────────────────────────────────────────────
        self.channel_n = self.m2.channel_n
        self.input_channels = self.m2.input_channels
        self.output_channels = self.m2.output_channels
        self.m1_channel_n = self.m1.channel_n

        self.freeze_m1 = bool(config.get("model.m1.freeze", True))
        self.m1_use_probs = bool(config.get("model.m1.use_probs", False))
        self.m1_eval_mode = bool(config.get("model.m1.eval_mode", self.freeze_m1))
        self.m1_level_offset = m1_level_offset
        self.m1_disable_backbone_tbptt = bool(config.get("model.m1.disable_backbone_tbptt", False))

        # ── Hidden-state transfer level ────────────────────────────────────
        # When > 0, M1 captures its hidden channels at this octree level
        # (e.g. level 1 = 256×256) and those are used for M2 warm-start
        # instead of M1's finest-level hidden.  M1's finest logits (e.g.
        # 512×512) are still used for the M2 logits slot and M1 loss.
        # 0 = use M1's finest level (default / legacy behaviour).
        self.m1_transfer_level = int(config.get("model.m1.transfer_level", 0))
        if self.m1_transfer_level > 0:
            if self.m1_transfer_level >= len(m1_res):
                raise ValueError(
                    f"model.m1.transfer_level={self.m1_transfer_level} out of range "
                    f"(0..{len(m1_res)-1}) for M1 with {len(m1_res)} levels."
                )
            self.m1._capture_level = self.m1_transfer_level
            self.m1_transfer_res: tuple[int, int] = tuple(m1_res[self.m1_transfer_level][0])
            print(
                f"[M2SingleStep] M1 hidden transfer from level {self.m1_transfer_level} "
                f"({self.m1_transfer_res[0]}×{self.m1_transfer_res[1]}) → "
                f"upscaled to M2 {self.m2_finest_res[0]}×{self.m2_finest_res[1]}"
            )
        else:
            self.m1_transfer_res = self.m1_finest_res

        self.m2_init_from_m1 = bool(config.get("model.m2.init_from_m1", False))
        self.m2_init_identity = bool(config.get("model.m2.init_identity", False))

        # ── FC bottleneck: compress (frame_diff + M1 logits + hidden) → N channels
        # When enabled, the frame difference, M1 logits, and M1 hidden
        # channels are projected through a 1×1 conv (per-pixel FC) before
        # entering M2.  The output replaces the hidden portion of the
        # warm-start state (remaining hidden channels are zeroed).
        fc_bottleneck_dim = config.get("model.m2.fc_bottleneck", 0)
        self._fc_bottleneck_dim = int(fc_bottleneck_dim) if fc_bottleneck_dim else 0
        self._fc_bottleneck: nn.Module | None = None
        if self._fc_bottleneck_dim > 0:
            orig_ic = getattr(self.m2, "_orig_input_channels", self.input_channels)
            m2_hidden_dim = self.channel_n - self.input_channels - self.output_channels
            m1_hidden_dim = self.m1_channel_n - self.m1.input_channels - self.m1.output_channels
            hidden_copy = min(m1_hidden_dim, m2_hidden_dim)
            # FC input: frame_diff (orig_ic) + M1 logits (output_channels) + M1 hidden
            fc_in_dim = orig_ic + self.output_channels + hidden_copy
            self._fc_bottleneck = nn.Conv2d(
                fc_in_dim, self._fc_bottleneck_dim, kernel_size=1, bias=True,
            )
            # Initialize near-zero so the bottleneck starts conservatively
            nn.init.xavier_uniform_(self._fc_bottleneck.weight, gain=0.1)
            nn.init.zeros_(self._fc_bottleneck.bias)
            print(
                f"[M2SingleStep] FC bottleneck: ({orig_ic} frame_diff + "
                f"{self.output_channels} logits + {hidden_copy} hidden) "
                f"→ {self._fc_bottleneck_dim} channels"
            )

        # ── Temporal step-index (k) embedding ─────────────────────────────
        # When model.m2.temporal_embed_dim > 0, a learned nn.Embedding maps
        # the integer step k (frames since the M1 anchor) to a dense vector
        # that is projected and added to the hidden channels of the warm-start
        # state.  This breaks the symmetry between frames 1, 2, …, max_step
        # and gives M2 an explicit "how far from anchor?" signal without
        # touching the image or logit channels.
        self._k_embed: nn.Embedding | None = None
        self._k_proj: nn.Linear | None = None
        k_embed_dim = int(config.get("model.m2.temporal_embed_dim", 0) or 0)
        if k_embed_dim > 0:
            k_max_step = int(
                config.get("model.m2.max_temporal_step",
                           config.get("trainer.m2_single_step.max_step", 20))
            )
            n_hidden = self.channel_n - self.input_channels - self.output_channels
            self._k_embed = nn.Embedding(k_max_step + 1, k_embed_dim)
            self._k_proj = nn.Linear(k_embed_dim, n_hidden, bias=True)
            # Near-zero init so the embedding starts as a small perturbation
            nn.init.normal_(self._k_embed.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self._k_proj.weight, gain=0.01)
            nn.init.zeros_(self._k_proj.bias)
            print(
                f"[M2SingleStep] Temporal k-embedding: "
                f"vocab={k_max_step + 1}, embed_dim={k_embed_dim}, proj_out={n_hidden}"
            )

        # ── Optical-flow warp: replace M1 logits in state with flow-warped logits
        # When enabled, before M2 runs the M1 logits slot is replaced by
        # M1_logits warped via Farneback optical flow from the anchor frame
        # to the current target frame.  This gives M2 a motion-corrected
        # prior instead of the static anchor-frame prediction.
        self._warp_logits = bool(config.get("model.m2.warp_logits", False))
        if self._warp_logits:
            try:
                import cv2 as _cv2_check  # noqa: F401
                print("[M2SingleStep] Optical-flow warp enabled (OpenCV Farneback).")
            except ImportError:
                import warnings as _warn
                _warn.warn(
                    "[M2SingleStep] model.m2.warp_logits=True but 'cv2' not found. "
                    "Install opencv-python-headless and retry. Warp disabled."
                )
                self._warp_logits = False

        self._load_m1_weights()
        self._sync_m2_from_m1()
        self._apply_m1_freeze()

    # ──────────────────────────────────────────────────────────────────────
    # Init helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_m1_weights(self) -> None:
        path = self.config.get("model.m1.pretrained_path", "")
        if not path or not str(path).strip():
            return
        state = torch.load(path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        m1_keys = set(self.m1.state_dict().keys())
        filtered = {k: v for k, v in state.items() if k in m1_keys}
        missing, unexpected = self.m1.load_state_dict(filtered, strict=False)
        print(
            f"[M2SingleStep] Loaded M1 weights from {path}  "
            f"(loaded={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    @staticmethod
    def _partial_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy the overlapping slice of *src* into *dst* (in-place).

        For every dimension, copies ``min(dst.size(d), src.size(d))``
        elements.  Remaining elements in *dst* are left untouched (i.e.
        keep their original random / zero init).
        """
        slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
        dst[slices] = src[slices]

    def _sync_m2_from_m1(self) -> None:
        if not self.m2_init_from_m1 and not self.m2_init_identity:
            return

        m1_state = self.m1.state_dict()
        m2_state = self.m2.state_dict()

        exact, partial, skipped = 0, 0, 0
        with torch.no_grad():
            for key, m2_param in m2_state.items():
                if key not in m1_state:
                    skipped += 1
                    continue
                m1_param = m1_state[key]
                if m1_param.shape == m2_param.shape:
                    m2_param.copy_(m1_param)
                    exact += 1
                else:
                    # Partial (truncated) copy — take the overlapping sub-tensor
                    self._partial_copy(m2_param, m1_param)
                    partial += 1
            # Write back (some params may be views, safest to reload)
            self.m2.load_state_dict(m2_state, strict=True)

        print(
            f"[M2SingleStep] Synced M2 from M1 weights  "
            f"(exact={exact}, partial={partial}, skipped={skipped})"
        )
        if self.m2_init_identity:
            self._reset_m2_residual_to_identity()

    def _reset_m2_residual_to_identity(self) -> None:
        m2_raw = self._unwrap_compiled_module(self.m2)
        for mod in self._iter_backbone_modules(m2_raw):
            if hasattr(mod, "fc2"):
                with torch.no_grad():
                    mod.fc2.weight.zero_()
                    if mod.fc2.bias is not None:
                        mod.fc2.bias.zero_()

    def _apply_m1_freeze(self) -> None:
        if self.freeze_m1:
            for p in self.m1.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.m1_eval_mode:
            self.m1.eval()
        return self

    # ──────────────────────────────────────────────────────────────────────
    # M1 forward helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
        while hasattr(module, "_orig_mod"):
            module = module._orig_mod
        return module

    def _iter_backbone_modules(self, model: nn.Module):
        if hasattr(model, "backbone_ncas"):
            for backbone in model.backbone_ncas:
                yield self._unwrap_compiled_module(backbone)
        elif hasattr(model, "backbone_nca"):
            yield self._unwrap_compiled_module(model.backbone_nca)

    @contextmanager
    def _temporary_disable_m1_backbone_tbptt(self):
        if not self.m1_disable_backbone_tbptt:
            yield
            return
        restored = []
        for backbone in self._iter_backbone_modules(self.m1):
            if hasattr(backbone, "tbptt_steps"):
                restored.append((backbone, backbone.tbptt_steps))
                backbone.tbptt_steps = None
        try:
            yield
        finally:
            for backbone, tbptt_steps in restored:
                backbone.tbptt_steps = tbptt_steps

    def _downscale_to_m1_res(self, x: torch.Tensor) -> torch.Tensor:
        """Bilinearly downscale BCHW tensor to M1's finest spatial resolution."""
        target_h, target_w = self.m1_finest_res
        if x.shape[2] == target_h and x.shape[3] == target_w:
            return x
        return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def _forward_m1(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
    ) -> dict:
        # Downscale inputs to M1's finest resolution when M1 operates coarser.
        x_a_m1 = self._downscale_to_m1_res(x_a)
        x_b_m1 = self._downscale_to_m1_res(x_b)

        target_h, target_w = self.m1_finest_res
        if y_a is None:
            y_a_m1 = torch.zeros(
                (x_a.shape[0], self.m1.output_channels, target_h, target_w),
                device=x_a.device, dtype=x_a.dtype,
            )
        else:
            y_a_m1 = F.interpolate(
                y_a.float(), size=(target_h, target_w), mode="nearest"
            )
        if y_b is None:
            y_b_m1 = torch.zeros(
                (x_b.shape[0], self.m1.output_channels, target_h, target_w),
                device=x_b.device, dtype=x_b.dtype,
            )
        else:
            y_b_m1 = F.interpolate(
                y_b.float(), size=(target_h, target_w), mode="nearest"
            )

        with self._temporary_disable_m1_backbone_tbptt():
            if self.freeze_m1:
                with torch.no_grad():
                    out = self.m1(x_a_m1, x_b_m1, y_a_m1, y_b_m1)
            else:
                out = self.m1(x_a_m1, x_b_m1, y_a_m1, y_b_m1)
        return out

    def _split_dual_batch(self, tensor: torch.Tensor, batch_size: int, name: str):
        if tensor.shape[0] != 2 * batch_size:
            raise RuntimeError(
                f"Expected {name} batch dimension to be 2*B={2*batch_size}, got {tensor.shape[0]}."
            )
        return tensor[:batch_size], tensor[batch_size:]

    @staticmethod
    def _interp_bhwc(x: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
        """Interpolate BHWC tensor to target spatial size, returning BHWC.

        .. deprecated:: Use ``_to_bchw`` + ``F.interpolate`` directly.
        """
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        align = False if mode in ("bilinear", "bicubic") else None
        up = F.interpolate(x_bchw, size=size, mode=mode, align_corners=align)
        return up.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _to_bchw(x: torch.Tensor, channel_n: int) -> torch.Tensor:
        """Convert BHWC → BCHW if needed (contiguous). BCHW input is returned as-is."""
        if x.shape[1] == channel_n:
            return x  # already BCHW
        # BHWC → BCHW
        return x.permute(0, 3, 1, 2).contiguous()

    def _states_from_m1_output(
        self,
        x_a: torch.Tensor,         # original 512×512 dataset image (BCHW)
        x_b: torch.Tensor,
        out: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build M2-ready warm-start states (at M2's finest resolution) from M1 output.

        Returns BCHW states.  All intermediate work is done in BCHW to avoid
        unnecessary permute+contiguous copies between BHWC and BCHW layouts.

        Logits always come from M1's finest level (e.g. 512×512).
        Hidden channels come from the *transfer level* — by default M1's
        finest, but when ``model.m1.transfer_level > 0`` they come from a
        coarser captured level (e.g. 256×256) and are bilinearly upscaled.
        """
        logits = out["logits"]          # BHWC at M1's finest res
        hidden = out.get("hidden_channels", None)
        if hidden is None:
            raise RuntimeError("M1 output missing hidden_channels.")

        batch_size = x_a.shape[0]
        logits_a, logits_b = self._split_dual_batch(logits, batch_size, "logits")

        if self.m1_use_probs and "probabilities" in out:
            probs_a, probs_b = self._split_dual_batch(out["probabilities"], batch_size, "probabilities")
            logits_a, logits_b = probs_a, probs_b

        # Convert logits BHWC → BCHW once (single permute+contiguous)
        logits_a = self._to_bchw(logits_a, self.output_channels)
        logits_b = self._to_bchw(logits_b, self.output_channels)

        # ── Hidden channels: prefer captured intermediate level ──────────
        captured = out.get("captured_level_states", None)
        if self.m1_transfer_level > 0 and captured is not None:
            # Captured states are already BCHW (stored without BHWC conversion)
            hidden_a = captured["hidden_a"]
            hidden_b = captured["hidden_b"]
            # Ensure BCHW (handles legacy BHWC captures gracefully)
            if captured.get("_layout") != "BCHW":
                m1_hid_ch = self.m1_channel_n - self.m1.input_channels - self.m1.output_channels
                hidden_a = self._to_bchw(hidden_a, m1_hid_ch)
                hidden_b = self._to_bchw(hidden_b, m1_hid_ch)
        else:
            # Fallback: use M1's finest-level hidden (2B concatenated, BHWC from _pack_outputs)
            hidden_a, hidden_b = self._split_dual_batch(hidden, batch_size, "hidden_channels")
            m1_hid_ch = self.m1_channel_n - self.m1.input_channels - self.m1.output_channels
            hidden_a = self._to_bchw(hidden_a, m1_hid_ch)
            hidden_b = self._to_bchw(hidden_b, m1_hid_ch)

        # M2's target spatial size
        m2_h, m2_w = self.m2_finest_res

        # Upscale logits to M2 resolution if they differ (BCHW: spatial dims are [2],[3])
        if logits_a.shape[2] != m2_h or logits_a.shape[3] != m2_w:
            logits_a = F.interpolate(logits_a, size=(m2_h, m2_w), mode="bilinear", align_corners=False)
            logits_b = F.interpolate(logits_b, size=(m2_h, m2_w), mode="bilinear", align_corners=False)

        # Upscale hidden to M2 resolution if they differ (BCHW)
        if hidden_a.shape[2] != m2_h or hidden_a.shape[3] != m2_w:
            hidden_a = F.interpolate(hidden_a, size=(m2_h, m2_w), mode="bilinear", align_corners=False)
            hidden_b = F.interpolate(hidden_b, size=(m2_h, m2_w), mode="bilinear", align_corners=False)

        # Downscale original image to M2 finest res (if needed) — already BCHW
        if x_a.shape[2] != m2_h or x_a.shape[3] != m2_w:
            x_a_m2 = F.interpolate(x_a, size=(m2_h, m2_w), mode="bilinear", align_corners=False)
            x_b_m2 = F.interpolate(x_b, size=(m2_h, m2_w), mode="bilinear", align_corners=False)
        else:
            x_a_m2, x_b_m2 = x_a, x_b

        # Build state in BCHW — avoids all permute overhead downstream.
        # _to_bchw_state in the warm-start model detects BCHW and skips conversion.
        ic = self.input_channels
        oc = self.output_channels
        h_start = ic + oc
        orig_ic = getattr(self.m2, "_orig_input_channels", ic)

        state_a = x_a.new_zeros(batch_size, self.channel_n, m2_h, m2_w)
        state_b = x_b.new_zeros(batch_size, self.channel_n, m2_h, m2_w)

        m2_hidden_dim = self.channel_n - h_start
        m1_hidden_dim = hidden_a.shape[1]  # BCHW: channel dim
        hidden_copy = min(m1_hidden_dim, m2_hidden_dim)

        # Image channels (BCHW)
        state_a[:, :orig_ic] = x_a_m2[:, :orig_ic]
        state_b[:, :orig_ic] = x_b_m2[:, :orig_ic]
        # Extra input channels (e.g. frame-diff) stay zero → signals t=0 / keyframe

        # Logits from M1 (BCHW)
        state_a[:, ic:ic + oc] = logits_a
        state_b[:, ic:ic + oc] = logits_b

        # Hidden channels from M1 (BCHW)
        state_a[:, h_start:h_start + hidden_copy] = hidden_a[:, :hidden_copy]
        state_b[:, h_start:h_start + hidden_copy] = hidden_b[:, :hidden_copy]

        return state_a, state_b

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if m1_out is None:
            m1_out = self._forward_m1(x_a, x_b, y_a, y_b)
        return self._states_from_m1_output(x_a, x_b, m1_out)

    def m1_forward_and_init_states(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, Tuple[torch.Tensor, torch.Tensor]]:
        m1_out = self._forward_m1(x_a, x_b, y_a, y_b)
        states = self._states_from_m1_output(x_a, x_b, m1_out)
        return m1_out, states

    # ──────────────────────────────────────────────────────────────────────
    # Optical-flow warp helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_farneback_flow(
        anchor_bchw: torch.Tensor,
        target_bchw: torch.Tensor,
    ) -> torch.Tensor:
        """Dense Farneback optical flow from anchor to target (CPU via OpenCV).

        Args:
            anchor_bchw: (B, C, H, W) float in [0, 1]
            target_bchw: (B, C, H, W) float in [0, 1]

        Returns:
            flow: (B, 2, H, W) float32  — flow[b, 0]=dx, flow[b, 1]=dy in pixels
        """
        import cv2  # lazy import; guarded in __init__
        B, C, H, W = anchor_bchw.shape
        # Collapse to grayscale on CPU
        anc_gray = anchor_bchw.mean(dim=1) if C > 1 else anchor_bchw[:, 0]
        tgt_gray = target_bchw.mean(dim=1) if C > 1 else target_bchw[:, 0]
        anc_np = (anc_gray * 255).clamp(0, 255).byte().cpu().numpy()  # (B, H, W)
        tgt_np = (tgt_gray * 255).clamp(0, 255).byte().cpu().numpy()
        flows = []
        for b in range(B):
            f = cv2.calcOpticalFlowFarneback(
                anc_np[b], tgt_np[b], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )  # (H, W, 2): [dx, dy]
            flows.append(torch.from_numpy(f).permute(2, 0, 1))  # (2, H, W)
        return torch.stack(flows, dim=0).to(
            device=anchor_bchw.device, dtype=anchor_bchw.dtype
        )  # (B, 2, H, W)

    @staticmethod
    def _warp_with_flow(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Bilinear warp of *src* by dense *flow* using grid_sample.

        Args:
            src:  (B, C, H, W)
            flow: (B, 2, H, W)  flow[b, 0]=dx, flow[b, 1]=dy in pixels

        Returns:
            warped: (B, C, H, W)
        """
        B, C, H, W = src.shape
        # Base grid (normalized coords in [-1, 1])
        ys = torch.linspace(-1.0, 1.0, H, device=src.device, dtype=src.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=src.device, dtype=src.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        # Normalize pixel displacement to [-1, 1] range
        dx_norm = flow[:, 0:1] / (W / 2.0)  # (B, 1, H, W)
        dy_norm = flow[:, 1:2] / (H / 2.0)
        flow_norm = torch.cat([dx_norm, dy_norm], dim=1).permute(0, 2, 3, 1)  # (B, H, W, 2)
        sample_grid = (base + flow_norm).clamp(-1.0, 1.0)
        return F.grid_sample(
            src, sample_grid, mode="bilinear",
            align_corners=True, padding_mode="border",
        )

    def _apply_flow_warp_to_state(
        self,
        prev_state: torch.Tensor,
        x_current: torch.Tensor,
    ) -> torch.Tensor:
        """Replace the M1-logits slice of prev_state with flow-warped logits.

        The anchor image is extracted from the state itself (channels 0:orig_ic),
        and the optical flow from that anchor to x_current is used to warp
        the M1 logits before M2 processes them.  The hidden channels and all
        other slots are left untouched.

        Args:
            prev_state: (B, H, W, C) BHWC or (B, C, H, W) BCHW warm-start state.
            x_current:  (B, C_img, H, W) BCHW target frame.

        Returns:
            Modified prev_state (same layout as input).
        """
        orig_ic = getattr(self.m2, "_orig_input_channels", self.input_channels)
        ic = self.input_channels
        oc = self.output_channels

        is_bhwc = (prev_state.shape[-1] == self.channel_n)
        if is_bhwc:
            state = prev_state.permute(0, 3, 1, 2).contiguous()
        else:
            state = prev_state.clone()

        anchor_img = state[:, :orig_ic]         # (B, orig_ic, H, W)
        logits     = state[:, ic : ic + oc]    # (B, oc, H, W)

        with torch.no_grad():
            flow = self._compute_farneback_flow(anchor_img, x_current[:, :orig_ic])

        warped = self._warp_with_flow(logits, flow)  # differentiable

        new_state = state.clone()
        new_state[:, ic : ic + oc] = warped

        if is_bhwc:
            return new_state.permute(0, 2, 3, 1).contiguous()
        return new_state

    # ──────────────────────────────────────────────────────────────────────
    # Temporal embedding helper
    # ──────────────────────────────────────────────────────────────────────

    def _inject_k_embedding(
        self,
        prev_state: torch.Tensor,
        step_k,
    ) -> torch.Tensor:
        """Add a learned temporal embedding for step k to prev_state's hidden channels.

        Args:
            prev_state: (B, H, W, C) BHWC or (B, C, H, W) BCHW warm-start state.
            step_k:     int, 0-d tensor, or (B,) tensor with k values.

        Returns:
            Modified prev_state (same layout as input) with hidden channels
            perturbed by a small k-dependent delta.
        """
        assert self._k_embed is not None and self._k_proj is not None
        hidden_start = self.input_channels + self.output_channels
        B = prev_state.shape[0]
        device = prev_state.device

        # Normalise step_k to a (B,) int64 tensor, clamped to embedding vocab
        if isinstance(step_k, int):
            k_t = torch.full((B,), step_k, dtype=torch.long, device=device)
        else:
            k_t = torch.as_tensor(step_k, dtype=torch.long, device=device)
            if k_t.dim() == 0:
                k_t = k_t.unsqueeze(0).expand(B)
            elif k_t.shape[0] != B:
                k_t = k_t[:B] if k_t.shape[0] > B else k_t.expand(B)
        k_t = k_t.clamp(0, self._k_embed.num_embeddings - 1)

        k_emb = self._k_embed(k_t)              # (B, embed_dim)
        k_delta = self._k_proj(k_emb)           # (B, n_hidden)

        is_bhwc = (prev_state.shape[-1] == self.channel_n)
        prev_state = prev_state.clone()
        if is_bhwc:
            # (B, n_hidden) → broadcast over H, W
            prev_state[..., hidden_start:] = (
                prev_state[..., hidden_start:] + k_delta[:, None, None, :]
            )
        else:
            prev_state[:, hidden_start:] = (
                prev_state[:, hidden_start:] + k_delta[:, :, None, None]
            )
        return prev_state

    def _apply_fc_bottleneck_to_state(
        self,
        prev_state: torch.Tensor,
        x_current: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FC bottleneck to compress frame_diff + hidden into N channels.

        Args:
            prev_state: (B, H, W, C) warm-start state from M1 (BHWC format).
            x_current:  (B, C_img, H, W) current target frame (BCHW).

        Returns:
            Modified prev_state (BHWC) with hidden channels replaced by FC output.
        """
        orig_ic = getattr(self.m2, "_orig_input_channels", self.input_channels)
        hidden_start = self.input_channels + self.output_channels

        # Detect format: BHWC vs BCHW
        is_bhwc = (prev_state.shape[-1] == self.channel_n)
        if is_bhwc:
            # Work in BCHW for Conv2d
            state_bchw = prev_state.permute(0, 3, 1, 2).contiguous()
        else:
            state_bchw = prev_state

        # Anchor image stored in state (channels 0:orig_ic)
        anchor_img = state_bchw[:, :orig_ic]  # (B, orig_ic, H, W)
        # Frame difference: target - anchor
        frame_diff = x_current[:, :orig_ic] - anchor_img  # (B, orig_ic, H, W)
        # M1 logits stored in state
        logits = state_bchw[:, self.input_channels:hidden_start]  # (B, out_ch, H, W)
        # Hidden channels from M1
        hidden = state_bchw[:, hidden_start:]  # (B, hidden_dim, H, W)
        m2_hidden_dim = self.channel_n - hidden_start
        m1_hidden_dim = self.m1_channel_n - self.m1.input_channels - self.m1.output_channels
        hidden_copy = min(m1_hidden_dim, m2_hidden_dim)
        hidden = hidden[:, :hidden_copy]  # only the channels actually populated from M1

        # FC: [frame_diff, logits, hidden] → bottleneck_dim
        fc_in = torch.cat([frame_diff, logits, hidden], dim=1)
        fc_out = self._fc_bottleneck(fc_in)  # (B, bottleneck_dim, H, W)

        # Build new state: zero all hidden, place FC output in first slots
        new_state = state_bchw.clone()
        new_state[:, hidden_start:] = 0.0
        new_state[:, hidden_start:hidden_start + self._fc_bottleneck_dim] = fc_out

        if is_bhwc:
            return new_state.permute(0, 2, 3, 1).contiguous()
        return new_state

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
        prev_state_a: Optional[torch.Tensor] = None,
        prev_state_b: Optional[torch.Tensor] = None,
        step_k=None,
        batch_duplication: int = 1,
    ) -> dict:
        """Single M2 forward step.

        ``prev_state_a/b`` should be the upscaled M1 states from the most
        recent keyframe (obtained via ``m1_forward_and_init_states``).
        The agent controls when to call M1 vs M2 — this method is always M2.

        ``step_k``: int or (B,) tensor — number of frames since the M1 anchor.
        When provided (and temporal embedding is enabled), a learned k-embedding
        is added to the hidden channels of the warm-start state before M2 runs,
        giving M2 an explicit signal about how far it is from the keyframe.
        """
        # Warp M1 logits in state toward the target frame via optical flow
        if self._warp_logits and prev_state_a is not None:
            prev_state_a = self._apply_flow_warp_to_state(prev_state_a, x_a)
            prev_state_b = self._apply_flow_warp_to_state(prev_state_b, x_b)

        # Apply FC bottleneck to compress frame_diff + hidden before M2
        if self._fc_bottleneck is not None and prev_state_a is not None:
            prev_state_a = self._apply_fc_bottleneck_to_state(prev_state_a, x_a)
            prev_state_b = self._apply_fc_bottleneck_to_state(prev_state_b, x_b)

        # Inject learnable temporal step embedding into hidden channels
        if self._k_embed is not None and step_k is not None and prev_state_a is not None:
            prev_state_a = self._inject_k_embedding(prev_state_a, step_k)
            prev_state_b = self._inject_k_embedding(prev_state_b, step_k)

        return self.m2(
            x_a, x_b,
            y_a=y_a, y_b=y_b,
            prev_state_a=prev_state_a,
            prev_state_b=prev_state_b,
            batch_duplication=batch_duplication,
        )
