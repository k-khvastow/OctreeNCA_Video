import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from train_baseline_unet import CONFIG, IOCTDataset, UNet

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import imageio.v2 as imageio  # type: ignore
except ImportError:
    imageio = None


# Matches the colors used in train_baseline_unet.IOCTDataset (Segmentation RGB).
PALETTE_RGB_255: List[Tuple[int, int, int]] = [
    (0, 0, 0),        # 0 background
    (255, 0, 0),      # 1 red
    (0, 255, 209),    # 2 cyan
    (61, 255, 0),     # 3 green
    (0, 78, 255),     # 4 blue
    (255, 189, 0),    # 5 yellow/orange
    (218, 0, 255),    # 6 magenta
]


def _frame_sort_key(image_path: Path):
    stem = image_path.stem
    try:
        return int(stem)
    except ValueError:
        return stem


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    min_val = torch.min(x)
    max_val = torch.max(x)
    if (max_val - min_val) > 0:
        x = (x - min_val) / (max_val - min_val)
    return x


def _class_indices_to_rgb(indices_bhw: torch.Tensor, palette_rgb01: torch.Tensor) -> torch.Tensor:
    if indices_bhw.ndim != 3:
        raise ValueError(f"Expected BHW class indices, got shape {tuple(indices_bhw.shape)}")
    return palette_rgb01[indices_bhw]  # (B, H, W, 3)


def _merge_three_panel_black_bg(
    image_b1hw: torch.Tensor,
    pred_logits_bchw: torch.Tensor,
    gt_idx_bhw: torch.Tensor,
    palette_rgb01: torch.Tensor,
) -> np.ndarray:
    device = pred_logits_bchw.device
    image_b1hw = image_b1hw.to(device)
    gt_idx_bhw = gt_idx_bhw.to(device)

    # Image -> BHWC (3)
    img = _normalize01(image_b1hw).permute(0, 2, 3, 1)
    img = img.repeat(1, 1, 1, 3)

    # Pred -> indices -> RGB
    pred_idx_bhw = torch.argmax(pred_logits_bchw, dim=1)
    pred_rgb = _class_indices_to_rgb(pred_idx_bhw, palette_rgb01)

    # GT -> RGB
    gt_rgb = _class_indices_to_rgb(gt_idx_bhw, palette_rgb01)

    merged = torch.cat([img, pred_rgb, gt_rgb], dim=2)  # concat along width
    return merged.squeeze(0).detach().cpu().numpy()


def _sequence_key(image_path: Path, data_root: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(data_root.resolve())
        # Expected: <dataset>/Bscans-dt/<view>/Image/<frame>.png
        parts = rel.parts
        if len(parts) >= 3:
            dataset = parts[0]
            view = parts[2]
            return f"{dataset}_{view}"
    except Exception:
        pass

    # Fallback: try to infer from tail parts
    parts = image_path.parts
    if len(parts) >= 5:
        dataset = parts[-5]
        view = parts[-3]
        return f"{dataset}_{view}"
    return "unknown"


def _resolve_checkpoint(model_dir: Path, checkpoint: Optional[str]) -> Path:
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    best = model_dir / "best_model.pth"
    if best.exists():
        return best

    checkpoints = sorted(model_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}. Expected best_model.pth or checkpoint_epoch_*.pth."
    )


def _init_writer(output_path: Path, fps: int, frame_size_wh: Tuple[int, int]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frame_size_wh

    if output_path.suffix.lower() == ".gif":
        return ("gif", [])

    if cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return ("cv2", cv2.VideoWriter(str(output_path), fourcc, fps, (w, h)))

    if imageio is not None:
        return ("imageio", imageio.get_writer(str(output_path), fps=fps))

    frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return ("frames", frames_dir)


def _write_frame(writer, writer_kind: str, frame_rgb_uint8: np.ndarray, frame_idx: int, fps: int):
    if writer_kind == "gif":
        writer.append(Image.fromarray(frame_rgb_uint8))
        return

    if writer_kind == "cv2":
        frame_bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        return

    if writer_kind == "imageio":
        writer.append_data(frame_rgb_uint8)
        return

    if writer_kind == "frames":
        out_png = writer / f"{frame_idx:06d}.png"
        Image.fromarray(frame_rgb_uint8).save(out_png)
        return

    raise ValueError(f"Unknown writer kind: {writer_kind}")


def _close_writer(writer, writer_kind: str, output_path: Path, fps: int):
    if writer_kind == "gif":
        # Note: can be large; intended for shorter sequences.
        if len(writer) == 0:
            return
        duration_ms = int(1000 / max(1, fps))
        writer[0].save(
            output_path,
            save_all=True,
            append_images=writer[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return

    if writer_kind == "cv2":
        writer.release()
        return

    if writer_kind == "imageio":
        writer.close()
        return

    if writer_kind == "frames":
        return

    raise ValueError(f"Unknown writer kind: {writer_kind}")


def create_visualisation_baseline_unet(
    model_dir: str,
    checkpoint: Optional[str],
    data_root: str,
    sequence: Optional[str],
    fps: int,
    max_frames: Optional[int],
    frame_stride: int,
    output_path: str,
    device: Optional[str],
):
    model_dir_path = Path(model_dir)
    data_root_path = Path(data_root)
    output_path = Path(output_path)

    dataset = IOCTDataset(str(data_root_path), input_size=CONFIG.get("input_size", (512, 512)))
    if len(dataset) == 0:
        raise ValueError(f"No frames found under {data_root_path}")

    # Group by (dataset, view)
    seq_to_indices: Dict[str, List[int]] = {}
    for i, info in enumerate(dataset.frames):
        image_path = Path(info["image_path"])
        key = _sequence_key(image_path, data_root_path)
        seq_to_indices.setdefault(key, []).append(i)

    if sequence is None:
        sequence = sorted(seq_to_indices.keys())[0]
    if sequence not in seq_to_indices:
        available = ", ".join(sorted(seq_to_indices.keys()))
        raise ValueError(f"Sequence '{sequence}' not found. Available: {available}")

    indices = seq_to_indices[sequence]
    indices.sort(key=lambda i: _frame_sort_key(Path(dataset.frames[i]["image_path"])))

    frame_stride = max(1, int(frame_stride))
    if frame_stride > 1:
        indices = indices[::frame_stride]
    if max_frames is not None:
        indices = indices[: int(max_frames)]

    ckpt_path = _resolve_checkpoint(model_dir_path, checkpoint)

    # Model
    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=CONFIG["input_channels"],
        channels=CONFIG["base_channels"],
        n_classes=CONFIG["n_classes"],
    ).to(resolved_device)
    state = torch.load(ckpt_path, map_location=resolved_device)
    model.load_state_dict(state)
    model.eval()

    palette_rgb01 = torch.tensor(PALETTE_RGB_255, dtype=torch.float32, device=resolved_device) / 255.0

    writer = None
    writer_kind = None
    frame_size_wh = None

    print(f"Sequence: {sequence} | Frames: {len(indices)} | Checkpoint: {ckpt_path}")
    if cv2 is None and imageio is None and output_path.suffix.lower() != ".gif":
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        print(
            "Note: cv2/imageio not available; writing PNG frames to "
            f"{frames_dir} instead of a video container."
        )

    with torch.inference_mode():
        for frame_i, ds_idx in enumerate(indices):
            img_1hw, gt_hw = dataset[ds_idx]  # torch tensors
            img_b1hw = img_1hw.unsqueeze(0).to(resolved_device)
            gt_bhw = gt_hw.unsqueeze(0).to(resolved_device)

            logits_bchw = model(img_b1hw)

            merged = _merge_three_panel_black_bg(img_b1hw, logits_bchw, gt_bhw, palette_rgb01)
            frame_rgb_uint8 = (merged * 255.0).clip(0, 255).astype(np.uint8)

            if writer is None:
                h, w, _ = frame_rgb_uint8.shape
                frame_size_wh = (w, h)
                writer_kind, writer = _init_writer(output_path, fps=fps, frame_size_wh=frame_size_wh)

            _write_frame(writer, writer_kind, frame_rgb_uint8, frame_i, fps=fps)

            if frame_i % 25 == 0:
                print(f"Processed {frame_i}/{len(indices)}")

    if writer is not None and writer_kind is not None:
        _close_writer(writer, writer_kind, output_path, fps=fps)

    if writer_kind == "frames":
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        print(f"Frames written to: {frames_dir.resolve()}")
    else:
        print(f"Output written to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise baseline U-Net outputs on iOCT (image | prediction | ground truth)."
    )
    parser.add_argument("--model_dir", type=str, default=CONFIG.get("model_save_path", "Models/iOCT_UNet"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth checkpoint. If omitted, uses best_model.pth in --model_dir (or latest checkpoint_epoch_*).",
    )
    parser.add_argument("--data_root", type=str, default=CONFIG.get("data_path", "ioct_data"))
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence key like 'peeling_A' or 'sri_B'. Omit to pick the first available sequence.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default="visualisationOCT/baseline_unet_vis.mp4",
        help="Output path. If .gif, writes an animated GIF. If cv2/imageio missing, writes PNG frames to <stem>_frames/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda', 'cuda:0', or 'cpu'. Default: auto.",
    )
    parser.add_argument(
        "--list_sequences",
        action="store_true",
        help="List available sequences and exit (no model required).",
    )
    args = parser.parse_args()

    if args.list_sequences:
        dataset = IOCTDataset(args.data_root, input_size=CONFIG.get("input_size", (512, 512)))
        seq_to_count: Dict[str, int] = {}
        for info in dataset.frames:
            key = _sequence_key(Path(info["image_path"]), Path(args.data_root))
            seq_to_count[key] = seq_to_count.get(key, 0) + 1
        for k in sorted(seq_to_count.keys()):
            print(f"{k}: {seq_to_count[k]} frames")
        return

    create_visualisation_baseline_unet(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        sequence=args.sequence,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()

