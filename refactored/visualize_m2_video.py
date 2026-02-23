#!/usr/bin/env python3
"""
visualize_m2_video.py — M2 Single-Step temporal video
═════════════════════════════════════════════════════════

Produces an MP4 video showing a temporal sequence of iOCT frames
processed by the M1 (keyframe) + M2 (single-step) model.

Layout (4 columns side-by-side):
  ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │ View A: Input    │ View A:          │ View B: Input    │ View B:          │
  │ + GT overlay     │ Predicted seg    │ + GT overlay     │ Predicted seg    │
  └──────────────────┴──────────────────┴──────────────────┴──────────────────┘

A header banner shows the frame index and whether the prediction
was produced by M1 (keyframe) or M2 (intermediate frame).

Usage:
    python refactored/visualize_m2_video.py \\
        --checkpoint /path/to/epoch_6/model.pth \\
        --num-frames 40 \\
        --fps 4 \\
        --output m2_temporal.mp4

    # Override keyframe interval (default: from config)
    python refactored/visualize_m2_video.py \\
        --checkpoint ... --num-frames 40 --fps 4 --keyframe-interval 10

    # M1-only mode (standalone dual-view OctreeNCA checkpoint)
    python refactored/visualize_m2_video.py \\
        --m1-only \\
        --checkpoint /path/to/iOCT2D_dual_.../models/epoch_99/model.pth \\
        --num-frames 40 --fps 4 --output m1_only_video.mp4
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


# ── Colour palette (7 iOCT classes) ─────────────────────────────────────────
CLASS_COLORS = np.array([
    [ 30,  30,  30],  # 0  Background
    [255,   0,   0],  # 1  Red
    [  0, 255, 209],  # 2  Cyan
    [ 61, 255,   0],  # 3  Green
    [  0,  78, 255],  # 4  Blue
    [255, 189,   0],  # 5  Yellow
    [218,   0, 255],  # 6  Magenta
    [255, 127,  80],  # 7  Coral (fallback)
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

# ── Font helpers ─────────────────────────────────────────────────────────────
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


# ── Image / segmentation helpers ─────────────────────────────────────────────

def seg_mask_to_rgb(seg_hw: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Convert integer class mask (H, W) → RGB (H, W, 3)."""
    rgb = np.zeros((*seg_hw.shape, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(CLASS_COLORS))):
        rgb[seg_hw == c] = CLASS_COLORS[c]
    return rgb


def make_gt_overlay(input_gray_01: np.ndarray, gt_class_mask: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """Blend GT segmentation colours on top of grayscale input.

    Args:
        input_gray_01: (H, W) float in [0, 1]
        gt_class_mask: (H, W) int64 class indices
        alpha: blend factor for GT colour (1 = only GT colours)

    Returns:
        (H, W, 3) uint8 RGB image
    """
    # Normalise input to 0-255 uint8 with contrast stretch
    lo, hi = input_gray_01.min(), input_gray_01.max()
    if hi - lo > 1e-6:
        gray = ((input_gray_01 - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    else:
        gray = (input_gray_01 * 255).clip(0, 255).astype(np.uint8)
    base_rgb = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)

    gt_rgb = seg_mask_to_rgb(gt_class_mask)
    fg_mask = (gt_class_mask > 0)  # non-background pixels

    blended = base_rgb.copy().astype(np.float32)
    blended[fg_mask] = (alpha * gt_rgb[fg_mask].astype(np.float32)
                        + (1 - alpha) * base_rgb[fg_mask].astype(np.float32))
    return blended.clip(0, 255).astype(np.uint8)


def logits_bhwc_to_seg_rgb(logits_bhwc: torch.Tensor,
                           input_ch: int, out_ch: int,
                           batch_idx: int = 0) -> np.ndarray:
    """Extract segmentation from NCA state (BHWC) → (H, W, 3) uint8 RGB."""
    state = logits_bhwc[batch_idx]  # (H, W, C)
    seg_logits = state[..., input_ch:input_ch + out_ch]  # (H, W, out_ch)
    class_map = seg_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
    return seg_mask_to_rgb(class_map, out_ch)


def logits_bchw_to_seg_rgb(logits_bchw: torch.Tensor,
                           out_ch: int,
                           batch_idx: int = 0) -> np.ndarray:
    """Segmentation from logits in BCHW layout → (H, W, 3) uint8 RGB."""
    seg = logits_bchw[batch_idx].argmax(dim=0).cpu().numpy().astype(np.int32)
    return seg_mask_to_rgb(seg, out_ch)


def _pil_from_rgb(arr_hw3: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Resize (H, W, 3) uint8 numpy array to PIL at target (W, H)."""
    return Image.fromarray(arr_hw3).resize(size, Image.BILINEAR)


# ── Data loading ─────────────────────────────────────────────────────────────

def _rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.int64)
    for color, cls in RGB_TO_CLASS.items():
        mask = ((rgb[:, :, 0] == color[0])
                & (rgb[:, :, 1] == color[1])
                & (rgb[:, :, 2] == color[2]))
        out[mask] = cls
    return out


def load_sequences(data_root: str, n_frames: int, input_size: tuple = (512, 512),
                   seq_idx: int = 50):
    """Load consecutive paired A+B frames from iOCT dataset.

    Returns:
        imgs_a  : list of (1, H, W) float32 tensors
        imgs_b  : list of (1, H, W) float32 tensors
        gts_a   : list of (H, W) int64 class masks
        gts_b   : list of (H, W) int64 class masks
        seq_id  : str identifier
    """
    root = Path(data_root)
    H, W = input_size

    def _imgs_and_segs(ds, view):
        img_dir = root / ds / "Bscans-dt" / view / "Image"
        seg_dir = root / ds / "Bscans-dt" / view / "Segmentation"
        if not img_dir.exists():
            return [], []
        names = sorted(
            {p.name for p in img_dir.glob("*.png")
             if (seg_dir / p.name).exists()},
            key=lambda n: (0, int(Path(n).stem)) if Path(n).stem.isdigit() else (1, n),
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

    start = min(seq_idx, max(0, len(img_paths_a) - n_frames))
    end = min(start + n_frames, len(img_paths_a))
    print(f"Loading {end - start} consecutive frames from '{ds_used}' "
          f"starting at index {start} (total available: {len(img_paths_a)})")

    def _load(img_p, seg_p):
        img = np.array(Image.open(img_p))
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
        seg_rgb = np.array(Image.open(seg_p).resize((W, H), Image.NEAREST))
        if seg_rgb.ndim == 2:
            seg_rgb = np.stack([seg_rgb] * 3, axis=-1)
        return (torch.tensor(img[None], dtype=torch.float32),
                _rgb_to_class(seg_rgb))

    imgs_a, imgs_b, gts_a, gts_b = [], [], [], []
    for i in range(start, end):
        ta, ga = _load(img_paths_a[i], seg_paths_a[i])
        tb, gb = _load(img_paths_b[i], seg_paths_b[i])
        imgs_a.append(ta)
        imgs_b.append(tb)
        gts_a.append(ga)
        gts_b.append(gb)

    return imgs_a, imgs_b, gts_a, gts_b, f"{ds_used}_frame{start}"


# ── Checkpoint loading ───────────────────────────────────────────────────────

def _find_config(ckpt_path: str) -> Optional[Path]:
    p = Path(ckpt_path).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        cfg = parent / "config.json"
        if cfg.exists():
            return cfg
    return None


def _extract_state_dict(raw):
    """Unwrap checkpoint wrappers to get the plain state_dict."""
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "ema_state_dict"):
            if key in raw:
                return raw[key]
    return raw


def load_model(ckpt_path: str, device: torch.device):
    """Load OctreeNCA2DM2SingleStep from checkpoint + sibling config.json."""
    cfg_path = _find_config(ckpt_path)
    if cfg_path is None:
        raise FileNotFoundError(f"No config.json found near {ckpt_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    print(f"Config: {cfg_path}")

    from src.models.Model_OctreeNCA_2d_m2_single_step import OctreeNCA2DM2SingleStep

    # Silence M1 pretrained-path loader (we load the full composite ckpt below)
    config = dict(config)
    config["model.m1.pretrained_path"] = ""

    model = OctreeNCA2DM2SingleStep(config)
    model.to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = _extract_state_dict(raw)

    # If no m1./m2. prefix found, try harder
    if isinstance(raw, dict) and not any(
        k.startswith("m1.") or k.startswith("m2.") for k in list(sd.keys())[:10]
    ):
        sd = _extract_state_dict(raw)

    # Strip torch.compile prefix
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    print(f"Loaded checkpoint: {ckpt_path}")
    model.eval()
    return model, config


def load_m1_model(ckpt_path: str, device: torch.device):
    """Load a standalone OctreeNCA2DDualView (M1) checkpoint.

    This handles the case where the checkpoint is a plain dual-view model
    (not wrapped in the M2SingleStep composite), e.g. from the
    iOCT2D_dual_* experiments.
    """
    cfg_path = _find_config(ckpt_path)
    if cfg_path is None:
        raise FileNotFoundError(f"No config.json found near {ckpt_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    print(f"Config: {cfg_path}")

    from src.models.Model_OctreeNCA_2d_dual_view import OctreeNCA2DDualView

    model = OctreeNCA2DDualView(config)
    model.to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = _extract_state_dict(raw)

    # Strip torch.compile / DDP prefixes
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    print(f"Loaded M1 checkpoint: {ckpt_path}")
    model.eval()
    return model, config


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference_sequence(model, imgs_a, imgs_b, keyframe_interval, device):
    """Run M1/M2 inference on a sequence of frames.

    Returns:
        results: list of dicts, one per frame:
            {
                "pred_a": (H, W, 3) uint8 RGB segmentation,
                "pred_b": (H, W, 3) uint8 RGB segmentation,
                "model_label": str  ("M1 Keyframe" or "M2 k=N"),
            }
    """
    model.eval()
    input_ch = model.input_channels
    out_ch = model.output_channels

    results = []
    m1_state_a = None
    m1_state_b = None
    last_keyframe_t = 0

    with torch.no_grad():
        for t in range(len(imgs_a)):
            x_a = imgs_a[t].unsqueeze(0).to(device)  # (1, 1, H, W)
            x_b = imgs_b[t].unsqueeze(0).to(device)

            if t % keyframe_interval == 0:
                # ── Keyframe: run M1 ─────────────────────────────────
                m1_out, (m1_state_a, m1_state_b) = model.m1_forward_and_init_states(
                    x_a, x_b,
                )
                last_keyframe_t = t

                # M1 logits are BHWC with stacked A+B
                logits_bhwc = m1_out["logits"]  # (2, H_m1, W_m1, C)
                B = x_a.shape[0]
                logits_a = logits_bhwc[:B]  # (1, H_m1, W_m1, C)
                logits_b = logits_bhwc[B:]

                # Upscale M1 logits to M2 resolution for display
                m2_h, m2_w = model.m2_finest_res
                logits_a_bchw = logits_a.permute(0, 3, 1, 2)
                logits_b_bchw = logits_b.permute(0, 3, 1, 2)
                if logits_a_bchw.shape[2] != m2_h or logits_a_bchw.shape[3] != m2_w:
                    logits_a_bchw = F.interpolate(logits_a_bchw, size=(m2_h, m2_w),
                                                  mode="bilinear", align_corners=False)
                    logits_b_bchw = F.interpolate(logits_b_bchw, size=(m2_h, m2_w),
                                                  mode="bilinear", align_corners=False)

                pred_a = logits_bchw_to_seg_rgb(logits_a_bchw, out_ch)
                pred_b = logits_bchw_to_seg_rgb(logits_b_bchw, out_ch)
                label = "M1 Keyframe"
            else:
                # ── Intermediate: run M2 ─────────────────────────────
                assert m1_state_a is not None
                k = t - last_keyframe_t
                out = model(
                    x_a, x_b,
                    prev_state_a=m1_state_a,
                    prev_state_b=m1_state_b,
                    step_k=k,
                )
                logits_bhwc = out["logits"]
                B = x_a.shape[0]
                logits_a_bchw = logits_bhwc[:B].permute(0, 3, 1, 2)
                logits_b_bchw = logits_bhwc[B:].permute(0, 3, 1, 2)

                pred_a = logits_bchw_to_seg_rgb(logits_a_bchw, out_ch)
                pred_b = logits_bchw_to_seg_rgb(logits_b_bchw, out_ch)
                label = f"M2 (k={k})"

            results.append({
                "pred_a": pred_a,
                "pred_b": pred_b,
                "model_label": label,
            })

    return results


def run_m1_only_inference_sequence(model, imgs_a, imgs_b, device):
    """Run standalone M1 inference on every frame (no M2).

    Returns:
        results: list of dicts, one per frame:
            {
                "pred_a": (H, W, 3) uint8 RGB segmentation,
                "pred_b": (H, W, 3) uint8 RGB segmentation,
                "model_label": "M1",
            }
    """
    model.eval()
    out_ch = model.output_channels

    results = []

    with torch.no_grad():
        for t in range(len(imgs_a)):
            x_a = imgs_a[t].unsqueeze(0).to(device)  # (1, 1, H, W)
            x_b = imgs_b[t].unsqueeze(0).to(device)

            out = model.forward_eval(x_a, x_b)

            # logits: (2*B, H, W, out_ch) BHWC, A stacked above B
            logits_bhwc = out["logits"]
            B = x_a.shape[0]
            logits_a_bchw = logits_bhwc[:B].permute(0, 3, 1, 2)
            logits_b_bchw = logits_bhwc[B:].permute(0, 3, 1, 2)

            pred_a = logits_bchw_to_seg_rgb(logits_a_bchw, out_ch)
            pred_b = logits_bchw_to_seg_rgb(logits_b_bchw, out_ch)

            results.append({
                "pred_a": pred_a,
                "pred_b": pred_b,
                "model_label": "M1",
            })

    return results


# ── Frame rendering ──────────────────────────────────────────────────────────

HEADER_H = 56
FOOTER_H = 28
PAD = 6
COL_LABEL_H = 20


def render_frame(
    input_a_01: np.ndarray,   # (H, W) float [0, 1]
    gt_a: np.ndarray,         # (H, W) int64 class mask
    pred_a_rgb: np.ndarray,   # (H, W, 3) uint8
    input_b_01: np.ndarray,
    gt_b: np.ndarray,
    pred_b_rgb: np.ndarray,
    frame_idx: int,
    model_label: str,
    tile: int,
    exp_name: str,
    total_frames: int,
) -> Image.Image:
    """Compose one video frame with 4 columns.

    Layout:
      Col 0: View A input + GT overlay
      Col 1: View A predicted segmentation
      Col 2: View B input + GT overlay
      Col 3: View B predicted segmentation
    """
    n_cols = 4
    canvas_w = PAD + n_cols * (tile + PAD)
    canvas_h = HEADER_H + tile + COL_LABEL_H + FOOTER_H

    canvas = Image.new("RGB", (canvas_w, canvas_h), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)

    f_title = _try_font(20)
    f_small = _try_font(14)
    f_label = _try_font(12)
    f_model = _try_font(16)

    # ── Header ───────────────────────────────────────────────────────────
    # Left: frame counter
    frame_text = f"Frame {frame_idx}/{total_frames - 1}"
    draw.text((12, 8), frame_text, fill=(240, 240, 240), font=f_title)

    # Centre: model badge (M1 or M2)
    is_m1 = model_label.startswith("M1")
    badge_color = (0, 180, 80) if is_m1 else (60, 140, 255)
    badge_text = model_label

    badge_bb = draw.textbbox((0, 0), badge_text, font=f_model)
    badge_w = badge_bb[2] - badge_bb[0] + 16
    badge_h = badge_bb[3] - badge_bb[1] + 8
    badge_x = canvas_w // 2 - badge_w // 2
    badge_y = 6
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
        radius=6, fill=badge_color,
    )
    draw.text((badge_x + 8, badge_y + 3), badge_text,
              fill=(255, 255, 255), font=f_model)

    # Right: experiment name (truncated)
    name_short = exp_name[:40]
    name_bb = draw.textbbox((0, 0), name_short, font=f_small)
    draw.text((canvas_w - (name_bb[2] - name_bb[0]) - 12, 12),
              name_short, fill=(160, 160, 160), font=f_small)

    # ── Separator line under header ──────────────────────────────────────
    draw.line([(0, HEADER_H - 2), (canvas_w, HEADER_H - 2)],
              fill=(70, 70, 70), width=1)

    # ── Tiles ────────────────────────────────────────────────────────────
    gt_overlay_a = make_gt_overlay(input_a_01, gt_a)
    gt_overlay_b = make_gt_overlay(input_b_01, gt_b)

    tiles_data = [
        (gt_overlay_a, "View A: Input + GT"),
        (pred_a_rgb,   "View A: Prediction"),
        (gt_overlay_b, "View B: Input + GT"),
        (pred_b_rgb,   "View B: Prediction"),
    ]

    for col, (rgb_hw3, label) in enumerate(tiles_data):
        x = PAD + col * (tile + PAD)
        y = HEADER_H

        pil_tile = _pil_from_rgb(rgb_hw3, (tile, tile))
        canvas.paste(pil_tile, (x, y))

        # Thin border
        draw.rectangle([x - 1, y - 1, x + tile, y + tile],
                       outline=(80, 80, 80), width=1)

        # Column label centred below tile
        lb = draw.textbbox((0, 0), label, font=f_label)
        lw = lb[2] - lb[0]
        draw.text((x + tile // 2 - lw // 2, y + tile + 3),
                  label, fill=(200, 200, 200), font=f_label)

    return canvas


# ── Video saving ─────────────────────────────────────────────────────────────

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
        description="Visualize M2SingleStep model: 4-column temporal video "
                    "(GT+overlay | pred | GT+overlay | pred)"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model.pth or ema.pth")
    parser.add_argument("--m1-only", action="store_true",
                        help="Load a standalone M1 (OctreeNCA2DDualView) checkpoint "
                             "instead of the composite M1+M2 model")
    parser.add_argument("--output", default="m2_temporal_video.mp4",
                        help="Output path (.mp4 or .gif)")
    parser.add_argument("--num-frames", type=int, default=40,
                        help="Total number of temporal frames in the video")
    parser.add_argument("--fps", type=int, default=4,
                        help="Output video FPS")
    parser.add_argument("--sample", type=int, default=50,
                        help="Starting frame index in dataset")
    parser.add_argument("--tile", type=int, default=512,
                        help="Tile pixel size per column")
    parser.add_argument("--keyframe-interval", type=int, default=None,
                        help="M1 keyframe interval (default: from config)")
    parser.add_argument("--data-root",
                        default="/home/khvastow/ioct_data",
                        help="iOCT data root directory")
    args = parser.parse_args()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    m1_only = args.m1_only
    if m1_only:
        model, config = load_m1_model(args.checkpoint, device)
        print("Mode: M1-only (standalone dual-view OctreeNCA)")
    else:
        model, config = load_model(args.checkpoint, device)

    exp_name = config.get("experiment.name",
                          Path(args.checkpoint).parents[2].name)
    print(f"Experiment: {exp_name}")

    input_size_raw = config.get("experiment.dataset.input_size", [512, 512])
    input_size = tuple(int(x) for x in input_size_raw)

    keyframe_interval = args.keyframe_interval
    if not m1_only:
        if keyframe_interval is None:
            keyframe_interval = int(config.get(
                "trainer.m2_single_step.keyframe_interval", 10))
        print(f"Keyframe interval: {keyframe_interval}")

    # ── Load data ────────────────────────────────────────────────────────
    imgs_a, imgs_b, gts_a, gts_b, seq_id = load_sequences(
        args.data_root,
        n_frames=args.num_frames,
        input_size=input_size,
        seq_idx=args.sample,
    )
    n_frames = len(imgs_a)
    imgs_a_dev = [x.to(device) for x in imgs_a]
    imgs_b_dev = [x.to(device) for x in imgs_b]
    print(f"Loaded {n_frames} frames (requested {args.num_frames})")

    # ── Run inference ────────────────────────────────────────────────────
    print("\nRunning inference...")
    if m1_only:
        results = run_m1_only_inference_sequence(
            model, imgs_a_dev, imgs_b_dev, device,
        )
    else:
        results = run_inference_sequence(
            model, imgs_a_dev, imgs_b_dev, keyframe_interval, device,
        )
    for r in results:
        print(f"  {r['model_label']}")

    # ── Render video frames ──────────────────────────────────────────────
    print("\nRendering video frames...")
    frames: list[Image.Image] = []
    for t in range(n_frames):
        input_a_01 = imgs_a[t][0].numpy()  # (H, W) float
        input_b_01 = imgs_b[t][0].numpy()
        gt_a = gts_a[t]
        gt_b = gts_b[t]

        frame = render_frame(
            input_a_01=input_a_01,
            gt_a=gt_a,
            pred_a_rgb=results[t]["pred_a"],
            input_b_01=input_b_01,
            gt_b=gt_b,
            pred_b_rgb=results[t]["pred_b"],
            frame_idx=t,
            model_label=results[t]["model_label"],
            tile=args.tile,
            exp_name=exp_name,
            total_frames=n_frames,
        )
        frames.append(frame)

    # ── Pad to consistent canvas & round to 16-multiples ─────────────────
    if frames:
        max_w = max(f.width for f in frames)
        max_h = max(f.height for f in frames)
        max_w = ((max_w + 15) // 16) * 16
        max_h = ((max_h + 15) // 16) * 16
        padded = []
        for f in frames:
            if f.width != max_w or f.height != max_h:
                canvas = Image.new("RGB", (max_w, max_h), (28, 28, 28))
                canvas.paste(f, (0, 0))
                padded.append(canvas)
            else:
                padded.append(f)
        frames = padded

    # ── Save ─────────────────────────────────────────────────────────────
    output = args.output
    print(f"\nTotal frames: {len(frames)} ({len(frames) / args.fps:.1f}s at {args.fps} fps)")
    try:
        if output.endswith(".gif"):
            _save_gif(frames, output, args.fps)
        else:
            _save_mp4(frames, output, args.fps)
        print(f"Saved → {output}")
    except Exception as e:
        print(f"MP4 save failed ({e}); falling back to GIF")
        output_gif = output.replace(".mp4", ".gif")
        _save_gif(frames, output_gif, args.fps)
        print(f"Saved → {output_gif}")


if __name__ == "__main__":
    main()
