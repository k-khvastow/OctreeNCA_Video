#!/usr/bin/env python3
"""
visualize_peeling3_video.py — Dual-view OctreeNCA peeling3 video
═══════════════════════════════════════════════════════════════════

Produces an MP4/GIF video showing consecutive iOCT peeling3 frames
processed by a dual-view OctreeNCA model (M1).

Data layout expected (per-frame folder):
    <DATA_ROOT>/<frame_id>/00.png        – cross-section view A (grayscale 512×512)
    <DATA_ROOT>/<frame_id>/01.png        – cross-section view B
    <DATA_ROOT>/<frame_id>/00_seg.png    – segmentation mask for view A (uint8, 0–4)
    <DATA_ROOT>/<frame_id>/01_seg.png    – segmentation mask for view B

Layout (4 columns side-by-side):
  ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │ View A: Input    │ View A:          │ View B: Input    │ View B:          │
  │ + GT overlay     │ Predicted seg    │ + GT overlay     │ Predicted seg    │
  └──────────────────┴──────────────────┴──────────────────┴──────────────────┘

Usage:
    python refactored/visualize_peeling3_video.py \\
        --checkpoint "/path/to/models/epoch_99/model.pth" \\
        --num-frames 100 --fps 10 --output peeling3_video.mp4

    # Custom data root / starting frame
    python refactored/visualize_peeling3_video.py \\
        --checkpoint ... --data-root /path/to/peeling3/Bscan \\
        --start 50 --num-frames 200 --fps 20

    # Limit to specific classes
    python refactored/visualize_peeling3_video.py \\
        --checkpoint ... --max-class 4 --num-frames 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ── Project root ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Colour palette (classes 0-7) ────────────────────────────────────────────
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

def seg_mask_to_rgb(seg_hw: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Convert integer class mask (H, W) -> RGB (H, W, 3)."""
    rgb = np.zeros((*seg_hw.shape, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(CLASS_COLORS))):
        rgb[seg_hw == c] = CLASS_COLORS[c]
    return rgb


def make_gt_overlay(input_gray_01: np.ndarray, gt_class_mask: np.ndarray,
                    alpha: float = 0.5, num_classes: int = 5) -> np.ndarray:
    """Blend GT segmentation colours on top of grayscale input."""
    lo, hi = input_gray_01.min(), input_gray_01.max()
    if hi - lo > 1e-6:
        gray = ((input_gray_01 - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    else:
        gray = (input_gray_01 * 255).clip(0, 255).astype(np.uint8)
    base_rgb = np.stack([gray, gray, gray], axis=-1)

    gt_rgb = seg_mask_to_rgb(gt_class_mask, num_classes)
    fg_mask = gt_class_mask > 0

    blended = base_rgb.copy().astype(np.float32)
    blended[fg_mask] = (alpha * gt_rgb[fg_mask].astype(np.float32)
                        + (1 - alpha) * base_rgb[fg_mask].astype(np.float32))
    return blended.clip(0, 255).astype(np.uint8)


def logits_bchw_to_seg_rgb(logits_bchw: torch.Tensor,
                           out_ch: int,
                           batch_idx: int = 0) -> np.ndarray:
    """Segmentation from logits in BCHW layout -> (H, W, 3) uint8 RGB."""
    seg = logits_bchw[batch_idx].argmax(dim=0).cpu().numpy().astype(np.int32)
    return seg_mask_to_rgb(seg, out_ch)


def _pil_from_rgb(arr_hw3: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(arr_hw3).resize(size, Image.BILINEAR)


# ── Peeling3 data loading ───────────────────────────────────────────────────

def load_peeling3_sequences(
    data_root: str,
    n_frames: int,
    input_size: tuple[int, int] = (512, 512),
    start_idx: int = 0,
    max_class: int = 4,
):
    """Load consecutive paired A+B frames from peeling3 data.

    Directory layout:
        data_root/<frame_id>/00.png, 01.png, 00_seg.png, 01_seg.png

    Returns:
        imgs_a  : list of (1, H, W) float32 tensors
        imgs_b  : list of (1, H, W) float32 tensors
        gts_a   : list of (H, W) int64 class masks
        gts_b   : list of (H, W) int64 class masks
    """
    root = Path(data_root)
    H, W = input_size

    # Discover valid frame directories (sorted numerically)
    frame_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    valid = []
    for d in frame_dirs:
        if all((d / f).exists() for f in ("00.png", "01.png", "00_seg.png", "01_seg.png")):
            valid.append(d)

    if not valid:
        raise RuntimeError(f"No valid peeling3 frame directories found in {data_root}")

    start = min(start_idx, max(0, len(valid) - n_frames))
    end = min(start + n_frames, len(valid))
    selected = valid[start:end]
    print(f"Loading {len(selected)} frames from peeling3 "
          f"(start={start}, total available={len(valid)})")

    def _load(img_path: Path, seg_path: Path):
        img = np.array(Image.open(img_path))
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        # Resize if needed
        if img.shape != (H, W):
            img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
        img_f = img.astype(np.float32) / 255.0
        img_t = torch.tensor(img_f[None], dtype=torch.float32)  # (1, H, W)

        seg = np.array(Image.open(seg_path))
        if seg.ndim == 3:
            seg = seg[:, :, 0]  # take first channel
        if seg.shape != (H, W):
            seg = np.array(Image.fromarray(seg).resize((W, H), Image.NEAREST))
        # Clamp classes above max_class to background
        seg[seg > max_class] = 0
        return img_t, seg.astype(np.int64)

    imgs_a, imgs_b, gts_a, gts_b = [], [], [], []
    for d in selected:
        ia, ga = _load(d / "00.png", d / "00_seg.png")
        ib, gb = _load(d / "01.png", d / "01_seg.png")
        imgs_a.append(ia)
        imgs_b.append(ib)
        gts_a.append(ga)
        gts_b.append(gb)

    return imgs_a, imgs_b, gts_a, gts_b


# ── Checkpoint loading ───────────────────────────────────────────────────────

def _find_config(ckpt_path: str) -> Path | None:
    p = Path(ckpt_path).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        cfg = parent / "config.json"
        if cfg.exists():
            return cfg
    return None


def _extract_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "ema_state_dict"):
            if key in raw:
                return raw[key]
    return raw


def load_model(ckpt_path: str, device: torch.device):
    """Load OctreeNCA2DDualView from checkpoint + sibling config.json."""
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
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"Loaded checkpoint: {ckpt_path}")
    model.eval()
    return model, config


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, imgs_a, imgs_b, device):
    """Run M1 (dual-view) inference on every frame.

    Returns:
        results: list of dicts per frame:
            {"pred_a": (H, W, 3) uint8 RGB, "pred_b": (H, W, 3) uint8 RGB}
    """
    model.eval()
    out_ch = model.output_channels
    results = []

    with torch.no_grad():
        for t in range(len(imgs_a)):
            x_a = imgs_a[t].unsqueeze(0).to(device)  # (1, 1, H, W)
            x_b = imgs_b[t].unsqueeze(0).to(device)

            out = model.forward_eval(x_a, x_b)

            # logits: (2*B, H, W, out_ch) BHWC — A stacked above B
            logits_bhwc = out["logits"]
            B = x_a.shape[0]
            logits_a_bchw = logits_bhwc[:B].permute(0, 3, 1, 2)
            logits_b_bchw = logits_bhwc[B:].permute(0, 3, 1, 2)

            pred_a = logits_bchw_to_seg_rgb(logits_a_bchw, out_ch)
            pred_b = logits_bchw_to_seg_rgb(logits_b_bchw, out_ch)

            results.append({"pred_a": pred_a, "pred_b": pred_b})

    return results


# ── Frame rendering ──────────────────────────────────────────────────────────

HEADER_H = 56
COL_LABEL_H = 20
FOOTER_H = 28
PAD = 6


def render_frame(
    input_a_01: np.ndarray,   # (H, W) float [0, 1]
    gt_a: np.ndarray,         # (H, W) int64 class mask
    pred_a_rgb: np.ndarray,   # (H, W, 3) uint8
    input_b_01: np.ndarray,
    gt_b: np.ndarray,
    pred_b_rgb: np.ndarray,
    frame_idx: int,
    tile: int,
    exp_name: str,
    total_frames: int,
    num_classes: int,
) -> Image.Image:
    """Compose one video frame with 4 columns."""
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
    frame_text = f"Frame {frame_idx}/{total_frames - 1}"
    draw.text((12, 8), frame_text, fill=(240, 240, 240), font=f_title)

    # Centre badge
    badge_text = "Dual-View OctreeNCA"
    badge_color = (0, 180, 80)
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

    # Separator
    draw.line([(0, HEADER_H - 2), (canvas_w, HEADER_H - 2)],
              fill=(70, 70, 70), width=1)

    # ── Tiles ────────────────────────────────────────────────────────────
    gt_overlay_a = make_gt_overlay(input_a_01, gt_a, num_classes=num_classes)
    gt_overlay_b = make_gt_overlay(input_b_01, gt_b, num_classes=num_classes)

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

        draw.rectangle([x - 1, y - 1, x + tile, y + tile],
                       outline=(80, 80, 80), width=1)

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
        description="Visualize dual-view OctreeNCA on peeling3 data: "
                    "4-column temporal video (GT+overlay | pred) x2 views"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model.pth or ema.pth")
    parser.add_argument("--output", default="peeling3_video.mp4",
                        help="Output path (.mp4 or .gif)")
    parser.add_argument("--num-frames", type=int, default=100,
                        help="Number of consecutive frames to render")
    parser.add_argument("--fps", type=int, default=10,
                        help="Output video FPS")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting frame index in the dataset")
    parser.add_argument("--tile", type=int, default=512,
                        help="Tile pixel size per column")
    parser.add_argument("--max-class", type=int, default=4,
                        help="Max valid class id (higher ids mapped to bg 0)")
    parser.add_argument("--data-root",
                        default="/home/khvastow/ioct_data/canvas/peeling3/iOCT/Bscan",
                        help="Peeling3 data root (directory of frame folders)")
    parser.add_argument("--device", default="cuda:3",
                        help="Torch device (default: cuda:3)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    model, config = load_model(args.checkpoint, device)
    out_ch = model.output_channels
    num_classes = out_ch

    exp_name = config.get("experiment.name", Path(args.checkpoint).parents[2].name)
    print(f"Experiment: {exp_name}")
    print(f"Model: {out_ch} output classes, input_channels={model.input_channels}")

    input_size_raw = config.get("experiment.dataset.input_size", [512, 512])
    input_size = tuple(int(x) for x in input_size_raw)

    # ── Load data ────────────────────────────────────────────────────────
    imgs_a, imgs_b, gts_a, gts_b = load_peeling3_sequences(
        data_root=args.data_root,
        n_frames=args.num_frames,
        input_size=input_size,
        start_idx=args.start,
        max_class=args.max_class,
    )
    n_frames = len(imgs_a)
    imgs_a_dev = [x.to(device) for x in imgs_a]
    imgs_b_dev = [x.to(device) for x in imgs_b]
    print(f"Loaded {n_frames} frames (requested {args.num_frames})")

    # ── Run inference ────────────────────────────────────────────────────
    print("\nRunning inference...")
    results = run_inference(model, imgs_a_dev, imgs_b_dev, device)
    print(f"Inference complete: {len(results)} frames")

    # ── Render video frames ──────────────────────────────────────────────
    print("Rendering video frames...")
    frames: list[Image.Image] = []
    for t in range(n_frames):
        input_a_01 = imgs_a[t][0].numpy()   # (H, W) float
        input_b_01 = imgs_b[t][0].numpy()

        frame = render_frame(
            input_a_01=input_a_01,
            gt_a=gts_a[t],
            pred_a_rgb=results[t]["pred_a"],
            input_b_01=input_b_01,
            gt_b=gts_b[t],
            pred_b_rgb=results[t]["pred_b"],
            frame_idx=t,
            tile=args.tile,
            exp_name=exp_name,
            total_frames=n_frames,
            num_classes=num_classes,
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
        print(f"Saved -> {output}")
    except Exception as e:
        print(f"MP4 save failed ({e}); falling back to GIF")
        output_gif = output.replace(".mp4", ".gif")
        _save_gif(frames, output_gif, args.fps)
        print(f"Saved -> {output_gif}")


if __name__ == "__main__":
    main()
