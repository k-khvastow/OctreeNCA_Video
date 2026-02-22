#!/usr/bin/env python3
"""
visualize_m2_single_step.py – M2 Single-Step model visualization
═══════════════════════════════════════════════════════════════════

Produces two video sections saved to one MP4:

  Part 1 – M1 Keyframe Inference (Octree Stages)
    Animates M1 running coarsest→finest on frame 0, showing each
    NCA step as a video frame. Dual-view A+B shown in two rows.

  Part 2 – M2 Temporal Evolution
    Shows M1 processing the keyframe (k=0), then M2 refining
    subsequent frames k = 1, 2, …, max_k. Each video frame
    reveals one additional temporal offset.

Usage:
    python refactored/visualize_m2_single_step.py \\
        --checkpoint  /path/to/epoch_6/model.pth \\
        --output      m2_vis.mp4 \\
        --sample      50 \\
        --fps         6

    # Override max_k (default: from config or 10)
    python refactored/visualize_m2_single_step.py \\
        --checkpoint ... --output ... --max-k 8

    # Use EMA weights instead
    python refactored/visualize_m2_single_step.py \\
        --checkpoint /path/to/epoch_6/ema.pth ...

Requires the full OctreeNCA project on sys.path (auto-detected
from the script's location).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ── Project root ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Colour palette ───────────────────────────────────────────────────────────
CLASS_COLORS = np.array([
    [30,  30,  30],   # 0  Background (dark grey so it's visible vs black border)
    [255,   0,   0],  # 1  Red
    [  0, 255, 209],  # 2  Cyan
    [ 61, 255,   0],  # 3  Green
    [  0,  78, 255],  # 4  Blue
    [255, 189,   0],  # 5  Yellow
    [218,   0, 255],  # 6  Magenta
    [255, 127,  80],  # 7  Coral
], dtype=np.uint8)

RGB_TO_CLASS = {
    (0, 0, 0):     0,
    (255, 0, 0):   1,
    (0, 255, 209): 2,
    (61, 255, 0):  3,
    (0, 78, 255):  4,
    (255, 189, 0): 5,
    (218, 0, 255): 6,
}

# ── Font helpers ────────────────────────────────────────────────────────────
def _try_font(size: int) -> ImageFont.FreeTypeFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Image helpers ────────────────────────────────────────────────────────────
def logits_bchw_to_rgb(logits_bchw: torch.Tensor, input_ch: int, out_ch: int,
                        view: int = 0) -> np.ndarray:
    """logits_bchw: (B, C, H, W) full NCA state. Returns HW3 uint8 RGB."""
    state = logits_bchw[view]                              # (C, H, W)
    seg = state[input_ch:input_ch + out_ch].argmax(0)      # (H, W)
    seg_np = seg.cpu().numpy().astype(np.int32)
    h, w = seg_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(out_ch, len(CLASS_COLORS))):
        mask = seg_np == c
        if mask.any():
            rgb[mask] = CLASS_COLORS[c]
    return rgb


def image_bchw_to_rgb(state_bchw: torch.Tensor, view: int = 0) -> np.ndarray:
    """Extract the input image channel from a state, returns HW3 uint8 RGB."""
    img = state_bchw[view, 0].cpu().float().numpy()
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def seg_mask_to_rgb(seg_hw: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Convert integer class mask (H, W) → RGB (H, W, 3)."""
    rgb = np.zeros((*seg_hw.shape, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(CLASS_COLORS))):
        rgb[seg_hw == c] = CLASS_COLORS[c]
    return rgb


def _pil_from_rgb(arr_hw3: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Resize numpy HW3 to PIL Image at target pixel (W, H)."""
    return Image.fromarray(arr_hw3).resize(size, Image.BILINEAR)


# ── Canvas helpers ───────────────────────────────────────────────────────────
HEADER_H = 50
ROW_LABEL_W = 60
FOOTER_H = 28
PAD = 4


def _make_canvas(n_cols: int, tile: int, n_rows: int = 2) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    w = ROW_LABEL_W + n_cols * tile + (n_cols + 1) * PAD
    h = HEADER_H + n_rows * (tile + PAD) + FOOTER_H
    canvas = Image.new("RGB", (w, h), (28, 28, 28))
    return canvas, ImageDraw.Draw(canvas)


def _header(draw: ImageDraw.ImageDraw, canvas_w: int,
            title: str, subtitle: str,
            f_title: ImageFont.FreeTypeFont, f_small: ImageFont.FreeTypeFont):
    draw.text((12, 10), title, fill=(240, 240, 240), font=f_title)
    bb = draw.textbbox((0, 0), subtitle, font=f_small)
    draw.text((canvas_w - (bb[2] - bb[0]) - 12, 14),
              subtitle, fill=(160, 160, 160), font=f_small)


def _tile_at(canvas: Image.Image, draw: ImageDraw.ImageDraw,
             col: int, row: int, tile: int,
             rgb_hw3: np.ndarray, label: str,
             f_label: ImageFont.FreeTypeFont,
             active: bool = False):
    x = ROW_LABEL_W + PAD + col * (tile + PAD)
    y = HEADER_H + PAD + row * (tile + PAD)
    pil = _pil_from_rgb(rgb_hw3, (tile, tile))
    canvas.paste(pil, (x, y))
    if active:
        draw.rectangle([x - 1, y - 1, x + tile, y + tile], outline=(255, 220, 0), width=2)
    lb = draw.textbbox((0, 0), label, font=f_label)
    lw = lb[2] - lb[0]
    draw.text((x + tile // 2 - lw // 2, y + tile + 2), label, fill=(200, 200, 200), font=f_label)


def _row_label(draw: ImageDraw.ImageDraw, row: int, tile: int, text: str,
               font: ImageFont.FreeTypeFont):
    y = HEADER_H + PAD + row * (tile + PAD) + tile // 2 - 8
    draw.text((4, y), text, fill=(180, 180, 180), font=font)


# ── Data loading ─────────────────────────────────────────────────────────────
def _rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.int64)
    for color, cls in RGB_TO_CLASS.items():
        mask = (rgb[:, :, 0] == color[0]) & (rgb[:, :, 1] == color[1]) & (rgb[:, :, 2] == color[2])
        out[mask] = cls
    return out


def load_sequences(data_root: str, n_frames: int = 11, input_size: tuple = (512, 512),
                   seq_idx: int = 50):
    """Load up to ``n_frames`` consecutive paired frames from iOCT dataset.

    Returns:
        imgs_a  : list of (1, H, W) float32 tensors in [0,1]
        imgs_b  : list of (1, H, W) float32 tensors in [0,1]
        gts_a   : list of (H, W) int64 gt class masks
        gts_b   : list of (H, W) int64 gt class masks
        seq_id  : str identifier for the loaded sequence
    """
    root = Path(data_root)
    H, W = input_size

    def _imgs_and_segs(ds, view):
        img_dir = root / ds / "Bscans-dt" / view / "Image"
        seg_dir = root / ds / "Bscans-dt" / view / "Segmentation"
        if not img_dir.exists():
            return [], []
        names = sorted(
            {p.name for p in img_dir.glob("*.png") if (seg_dir / p.name).exists()},
            key=lambda n: (0, int(Path(n).stem)) if Path(n).stem.isdigit() else (1, n)
        )
        return [img_dir / n for n in names], [seg_dir / n for n in names]

    img_paths_a, seg_paths_a = [], []
    img_paths_b, seg_paths_b = [], []
    ds_used = None
    for ds in ("peeling", "sri"):
        ia, sa = _imgs_and_segs(ds, "A")
        ib, sb = _imgs_and_segs(ds, "B")
        common = [(ia[i], sa[i], ib[i], sb[i])
                  for i in range(min(len(ia), len(ib)))
                  if ia[i].name == ib[i].name]
        if common:
            img_paths_a = [c[0] for c in common]
            seg_paths_a = [c[1] for c in common]
            img_paths_b = [c[2] for c in common]
            seg_paths_b = [c[3] for c in common]
            ds_used = ds
            break

    if not img_paths_a:
        raise RuntimeError(f"No paired A+B iOCT frames found under {data_root}")

    # Start from seq_idx (clamp)
    start = min(seq_idx, max(0, len(img_paths_a) - n_frames))
    end   = min(start + n_frames, len(img_paths_a))
    print(f"Loading {end - start} consecutive frames from '{ds_used}' "
          f"starting at index {start} (out of {len(img_paths_a)} total)")

    def _load(img_p, seg_p):
        img = np.array(Image.open(img_p))
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
        seg_rgb = np.array(Image.open(seg_p).resize((W, H), Image.NEAREST))
        if seg_rgb.ndim == 2:
            seg_rgb = np.stack([seg_rgb] * 3, axis=-1)
        return torch.tensor(img[None], dtype=torch.float32), _rgb_to_class(seg_rgb)

    imgs_a, imgs_b, gts_a, gts_b = [], [], [], []
    for i in range(start, end):
        ta, ga = _load(img_paths_a[i], seg_paths_a[i])
        tb, gb = _load(img_paths_b[i], seg_paths_b[i])
        imgs_a.append(ta)
        imgs_b.append(tb)
        gts_a.append(ga)
        gts_b.append(gb)

    return imgs_a, imgs_b, gts_a, gts_b, f"{ds_used}_frame{start}"


# ── Checkpoint loading ────────────────────────────────────────────────────────
def _find_config(ckpt_path: str) -> Optional[Path]:
    p = Path(ckpt_path).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        cfg = parent / "config.json"
        if cfg.exists():
            return cfg
    return None


def load_model(ckpt_path: str, device: torch.device):
    """Load OctreeNCA2DM2SingleStep from checkpoint + sibling config.json."""
    cfg_path = _find_config(ckpt_path)
    if cfg_path is None:
        raise FileNotFoundError(f"No config.json found near {ckpt_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    print(f"Config: {cfg_path}")

    from src.models.Model_OctreeNCA_2d_m2_single_step import OctreeNCA2DM2SingleStep

    # Silence the M1 pretrained-path loader if the path is stale —
    # we are loading the full composite checkpoint below anyway.
    config = dict(config)
    config["model.m1.pretrained_path"] = ""

    model = OctreeNCA2DM2SingleStep(config)
    model.to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and not any(k.startswith("m1.") or k.startswith("m2.") for k in list(raw.keys())[:3]):
        # Maybe it's wrapped
        for key in ("model_state_dict", "ema_state_dict"):
            if key in raw:
                sd = raw[key]; break
        else:
            sd = raw
    else:
        sd = raw

    # Strip torch.compile() prefix
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")
    print(f"Loaded checkpoint: {ckpt_path}")
    model.eval()
    return model, config


# ── Part 1: M1 Octree Stage Frames ──────────────────────────────────────────
def collect_m1_stage_frames(
    model,
    img_a: torch.Tensor,   # (1, 1, H, W)
    img_b: torch.Tensor,
    gt_a: np.ndarray,      # (H, W) int64
    gt_b: np.ndarray,
    tile: int,
    exp_name: str,
) -> list[Image.Image]:
    """Animate M1 running one NCA step at a time across all octree levels."""
    m1 = model.m1
    m1.eval()
    input_ch = m1.input_channels
    out_ch    = m1.output_channels
    n_levels  = len(m1.octree_res)

    f_title = _try_font(20)
    f_small = _try_font(13)
    f_label = _try_font(11)
    f_row   = _try_font(14)

    # Column layout: GT | Input | level_(n-1) coarsest | … | level_0 finest
    n_cols = 2 + n_levels

    # GT RGB images (constant across all frames)
    gt_rgb_a = seg_mask_to_rgb(gt_a)
    gt_rgb_b = seg_mask_to_rgb(gt_b)
    inp_rgb_a = image_bchw_to_rgb(img_a.cpu(), view=0)
    inp_rgb_b = image_bchw_to_rgb(img_b.cpu(), view=0)

    frames: list[Image.Image] = []

    orig_steps = list(m1.inference_steps)  # save to restore afterwards

    with torch.no_grad():
        # ── Initialise coarsest level ────────────────────────────────────
        state_a = img_a.new_zeros((1, m1.channel_n, *m1.octree_res[-1]))
        state_b = img_b.new_zeros((1, m1.channel_n, *m1.octree_res[-1]))
        xa_c = m1.downscale(img_a, -1, layout="BCHW")
        xb_c = m1.downscale(img_b, -1, layout="BCHW")
        state_a[:, :input_ch] = xa_c[:, :input_ch]
        state_b[:, :input_ch] = xb_c[:, :input_ch]

        # Current segmentation snapshots at each level slot (start as zeros)
        level_states = [None] * n_levels   # index 0 = finest, -1 = coarsest

        def _render(active_level: int, global_step: int) -> Image.Image:
            canvas, draw = _make_canvas(n_cols, tile)
            w_total = canvas.width
            _header(draw, w_total,
                    f"M1 Octree — step {global_step:4d}",
                    exp_name[:35], f_title, f_small)

            for row_idx, (row_label, gta, gtb, inp_rgb,
                          state_this) in enumerate([
                ("A", gt_rgb_a, gt_rgb_b, inp_rgb_a, (state_a, state_b)),
                ("B", gt_rgb_a, gt_rgb_b, inp_rgb_b, (state_a, state_b)),
            ]):
                gt_use   = gta if row_idx == 0 else gtb
                inp_use  = inp_rgb_a if row_idx == 0 else inp_rgb_b

                _row_label(draw, row_idx, tile, f"View {row_label}", f_row)
                _tile_at(canvas, draw, 0, row_idx, tile, gt_use, "GT", f_label)
                _tile_at(canvas, draw, 1, row_idx, tile, inp_use, "Input", f_label)

                # Octree levels coarsest → finest (left → right)
                for disp_idx in range(n_levels):
                    lv = n_levels - 1 - disp_idx   # coarsest first
                    state_at_lv = level_states[lv]
                    if state_at_lv is None:
                        blank = np.zeros((tile, tile, 3), dtype=np.uint8)
                        seg_rgb = blank
                    else:
                        view = 0 if row_idx == 0 else 1
                        # Gather the view's state (stacked A+B)
                        sa, sb = state_at_lv
                        s_use = sa if row_idx == 0 else sb
                        seg_rgb = logits_bchw_to_rgb(s_use, input_ch, out_ch, view=0)
                    res = m1.octree_res[lv]
                    col_lbl = f"L{lv} {res[0]}²"
                    is_active = (lv == active_level)
                    _tile_at(canvas, draw, 2 + disp_idx, row_idx, tile,
                             seg_rgb, col_lbl, f_label, active=is_active)

            return canvas

        global_step = 0

        for level in range(n_levels - 1, -1, -1):
            n_steps = orig_steps[level]
            if isinstance(n_steps, (list, tuple)):
                n_steps = int(n_steps[1])  # use max for eval
            else:
                n_steps = int(n_steps)

            m1.inference_steps[level] = 1  # patch to single-step mode

            for s in range(n_steps):
                state_ab = torch.cat([state_a, state_b], dim=0)
                state_ab = m1._run_backbone(state_ab, level)
                state_a, state_b = state_ab.chunk(2, dim=0)
                state_a_fused, state_b_fused = m1._maybe_cross_fuse(state_a, state_b, level)
                state_a, state_b = state_a_fused, state_b_fused

                level_states[level] = (state_a.clone(), state_b.clone())
                global_step += 1
                frames.append(_render(level, global_step))

            # Restore step count
            m1.inference_steps[level] = orig_steps[level]

            # Upscale to next finer level (if not already finest)
            if level > 0:
                scale_h, scale_w = m1.computed_upsampling_scales[level - 1][0]
                scale_h, scale_w = int(scale_h), int(scale_w)
                state_a = state_a.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                state_b = state_b.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
                inj_a = m1.downscale(img_a, level - 1, layout="BCHW")
                inj_b = m1.downscale(img_b, level - 1, layout="BCHW")
                state_a[:, :input_ch] = inj_a[:, :input_ch]
                state_b[:, :input_ch] = inj_b[:, :input_ch]

    # Restore in case of early exit
    m1.inference_steps = orig_steps

    # Pause on the final M1 frame (×3 = ~0.5 pause at 6 fps)
    if frames:
        frames.extend([frames[-1]] * 3)

    return frames


# ── Part 2: M2 Temporal Evolution Frames ────────────────────────────────────
def collect_m2_temporal_frames(
    model,
    imgs_a: list,   # list of (1, 1, H, W) tensors
    imgs_b: list,
    gts_a: list,
    gts_b: list,
    tile: int,
    exp_name: str,
    max_k: int,
) -> list[Image.Image]:
    """
    Animate M2 temporal evolution.

    Each video frame reveals one more temporal offset k. Layout:
      Row A: GT_A | Input_A | M1(k=0) | M2(k=1) | … | M2(k=K)
      Row B: GT_B | Input_B | M1(k=0) | M2(k=1) | … | M2(k=K)
    """
    model.eval()
    input_ch = model.m2.input_channels
    out_ch    = model.m2.output_channels
    device    = next(model.parameters()).device

    f_title = _try_font(20)
    f_small = _try_font(13)
    f_label = _try_font(11)
    f_row   = _try_font(14)

    max_k = min(max_k, len(imgs_a) - 1)
    n_display = max_k + 1   # k=0..max_k

    # ── Run M1 on frame 0 ───────────────────────────────────────────────
    # imgs_a[k] is (1, H, W) — add batch dim to get (1, 1, H, W)
    x_a0 = imgs_a[0].unsqueeze(0)  # (1, 1, H, W)
    x_b0 = imgs_b[0].unsqueeze(0)

    with torch.no_grad():
        m1_out = model._forward_m1(x_a0, x_b0)

    # Extract M1 logits (BHWC, 2×B stacked A+B)
    m1_logits_bhwc = m1_out["logits"]  # (2, H_m1, W_m1, C)
    B = x_a0.shape[0]

    def _bhwc_to_bchw_seg(bhwc: torch.Tensor) -> torch.Tensor:
        return bhwc.permute(0, 3, 1, 2)   # → (B, C, H, W)

    m1_logits_a = _bhwc_to_bchw_seg(m1_logits_bhwc[:B])   # (B, C, H_m1, W_m1)
    m1_logits_b = _bhwc_to_bchw_seg(m1_logits_bhwc[B:])

    # Upscale M1 logits to M2 resolution for display
    m2_h, m2_w = model.m2_finest_res
    m1_disp_a = F.interpolate(m1_logits_a, size=(m2_h, m2_w), mode="bilinear", align_corners=False)
    m1_disp_b = F.interpolate(m1_logits_b, size=(m2_h, m2_w), mode="bilinear", align_corners=False)

    def _seg_from_logits(logits_bchw: torch.Tensor) -> np.ndarray:
        """logits_bchw (1, C, H, W) → HW3 RGB"""
        seg = logits_bchw[0].argmax(0).cpu().numpy().astype(np.int32)
        rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for c in range(min(out_ch, len(CLASS_COLORS))):
            rgb[seg == c] = CLASS_COLORS[c]
        return rgb

    # Warm-start states for M2
    with torch.no_grad():
        state_a_bhwc, state_b_bhwc = model._states_from_m1_output(x_a0, x_b0, m1_out)

    # Convert BHWC → BCHW for the M2 forward pass
    state_a_bchw = state_a_bhwc.permute(0, 3, 1, 2).contiguous()
    state_b_bchw = state_b_bhwc.permute(0, 3, 1, 2).contiguous()

    # ── Run M2 for k=1..max_k ───────────────────────────────────────────
    # Collect list of seg RGBs: index 0 = M1@k=0, 1=M2@k=1, …
    seg_rgbs_a: list[np.ndarray] = [_seg_from_logits(m1_disp_a)]
    seg_rgbs_b: list[np.ndarray] = [_seg_from_logits(m1_disp_b)]
    labels_k: list[str] = ["M1 k=0"]

    for k in range(1, max_k + 1):
        x_ak = imgs_a[k].unsqueeze(0)  # (1, 1, H, W)
        x_bk = imgs_b[k].unsqueeze(0)

        with torch.no_grad():
            out_k = model.forward(
                x_ak, x_bk,
                prev_state_a=state_a_bchw,
                prev_state_b=state_b_bchw,
            )

        logits_k = out_k["logits"]  # BHWC, (2B, H, W, C)
        lk_a = _bhwc_to_bchw_seg(logits_k[:B])
        lk_b = _bhwc_to_bchw_seg(logits_k[B:])

        seg_rgbs_a.append(_seg_from_logits(lk_a))
        seg_rgbs_b.append(_seg_from_logits(lk_b))
        labels_k.append(f"M2 k={k}")

    # ── Build video frames: reveal one column at a time ─────────────────
    gt_rgb_a = seg_mask_to_rgb(gts_a[0])
    gt_rgb_b = seg_mask_to_rgb(gts_b[0])

    frames: list[Image.Image] = []

    for reveal in range(1, n_display + 1):
        # Number of data columns currently shown: GT | input | k=0 | k=1 | … | k=(reveal-1)
        n_cols = 2 + reveal
        canvas, draw = _make_canvas(n_cols, tile)
        _header(draw, canvas.width,
                f"M2 Temporal — revealing k=0..{reveal-1}",
                exp_name[:35], f_title, f_small)

        for row_idx, (row_lbl, gt_rgb, inp_rgb, segs) in enumerate([
            ("A", gt_rgb_a, image_bchw_to_rgb(imgs_a[reveal - 1].unsqueeze(0).cpu(), 0), seg_rgbs_a),
            ("B", gt_rgb_b, image_bchw_to_rgb(imgs_b[reveal - 1].unsqueeze(0).cpu(), 0), seg_rgbs_b),
        ]):
            _row_label(draw, row_idx, tile, f"View {row_lbl}", f_row)

            # GT column
            _tile_at(canvas, draw, 0, row_idx, tile,
                     gt_rgb, f"GT k={reveal-1}", f_label)
            # Input column
            _tile_at(canvas, draw, 1, row_idx, tile,
                     inp_rgb, f"Input k={reveal-1}", f_label)

            # Segmentation columns k=0..reveal-1
            for j in range(reveal):
                is_last = (j == reveal - 1)
                _tile_at(canvas, draw, 2 + j, row_idx, tile,
                         segs[j], labels_k[j], f_label, active=is_last)

        frames.append(canvas)

    # Pause on the final frame
    if frames:
        frames.extend([frames[-1]] * 4)

    return frames


# ── Write video ──────────────────────────────────────────────────────────────
def _save_mp4(frames: list[Image.Image], output: str, fps: int):
    import imageio
    writer = imageio.get_writer(output, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(np.array(f))
    writer.close()


def _save_gif(frames: list[Image.Image], output: str, fps: int):
    duration = int(1000 / fps)
    frames[0].save(
        output, save_all=True,
        append_images=frames[1:],
        loop=0, duration=duration,
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualize M2SingleStep model: M1 octree stages + M2 temporal evolution"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model.pth (or ema.pth)")
    parser.add_argument("--output", default="m2_single_step_vis.mp4",
                        help="Output path (.mp4 or .gif)")
    parser.add_argument("--sample", type=int, default=50,
                        help="Starting frame index in the dataset")
    parser.add_argument("--fps", type=int, default=6,
                        help="Frames per second for output video")
    parser.add_argument("--max-k", type=int, default=None,
                        help="Max temporal offset for M2 (default: from config or 10)")
    parser.add_argument("--tile", type=int, default=256,
                        help="Tile pixel size per panel (default 256)")
    parser.add_argument("--data-root", default="/vol/data/OctreeNCA_Video/ioct_data",
                        help="iOCT data root")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    model, config = load_model(args.checkpoint, device)
    exp_name = config.get("experiment.name", Path(args.checkpoint).parents[2].name)
    print(f"Experiment: {exp_name}")

    input_size_raw = config.get("experiment.dataset.input_size", [512, 512])
    input_size = tuple(int(x) for x in input_size_raw)

    # Determine max_k
    max_k = args.max_k
    if max_k is None:
        max_k = int(config.get("trainer.m2_single_step.max_step", 10))
    print(f"max_k = {max_k}")

    # ── Load data ────────────────────────────────────────────────────────
    n_frames = max_k + 1
    imgs_a, imgs_b, gts_a, gts_b, seq_id = load_sequences(
        args.data_root, n_frames=n_frames,
        input_size=input_size, seq_idx=args.sample,
    )
    imgs_a = [x.to(device) for x in imgs_a]
    imgs_b = [x.to(device) for x in imgs_b]

    print(f"\n── Part 1: M1 Octree Stage Frames ──────────────────")
    part1 = collect_m1_stage_frames(
        model,
        imgs_a[0][None],   # add batch dim: (1, 1, H, W)
        imgs_b[0][None],
        gts_a[0], gts_b[0],
        tile=args.tile,
        exp_name=exp_name,
    )
    print(f"   Generated {len(part1)} frames for Part 1")

    print(f"\n── Part 2: M2 Temporal Evolution Frames ────────────")
    part2 = collect_m2_temporal_frames(
        model,
        imgs_a, imgs_b, gts_a, gts_b,
        tile=args.tile,
        exp_name=exp_name,
        max_k=max_k,
    )
    print(f"   Generated {len(part2)} frames for Part 2")

    all_frames = part1 + part2
    print(f"\nTotal frames: {len(all_frames)}  ({len(all_frames)/args.fps:.1f}s at {args.fps} fps)")

    # ── Resize all to the same canvas size ────────────────────────────────
    # Part 2 grows wider — pad Part 1 frames to match Part 2's width
    if all_frames:
        max_w = max(f.width  for f in all_frames)
        max_h = max(f.height for f in all_frames)
        padded = []
        for f in all_frames:
            if f.width != max_w or f.height != max_h:
                canvas = Image.new("RGB", (max_w, max_h), (28, 28, 28))
                canvas.paste(f, (0, 0))
                padded.append(canvas)
            else:
                padded.append(f)
        all_frames = padded

    # ── Save ─────────────────────────────────────────────────────────────
    output = args.output
    try:
        if output.endswith(".gif"):
            _save_gif(all_frames, output, args.fps)
        else:
            _save_mp4(all_frames, output, args.fps)
        print(f"\nSaved → {output}")
    except Exception as e:
        print(f"MP4 save failed ({e}); falling back to GIF")
        output_gif = output.replace(".mp4", ".gif")
        _save_gif(all_frames, output_gif, args.fps)
        print(f"Saved → {output_gif}")


if __name__ == "__main__":
    main()
