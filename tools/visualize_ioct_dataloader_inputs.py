#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw


DATA_ROOT = Path("/vol/data/OctreeNCA_Video/ioct_data")
DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]
SEQUENCE_LENGTH = 3
SEQUENCE_STEP = 1
INPUT_SIZE = (512, 512)

RGB_TO_CLASS = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 209): 2,
    (61, 255, 0): 3,
    (0, 78, 255): 4,
    (255, 189, 0): 5,
    (218, 0, 255): 6,
}

PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 209],
        [61, 255, 0],
        [0, 78, 255],
        [255, 189, 0],
        [218, 0, 255],
        [240, 240, 240],
    ],
    dtype=np.uint8,
)


def rgb_to_class(rgb_seg: np.ndarray) -> np.ndarray:
    h, w = rgb_seg.shape[:2]
    class_seg = np.zeros((h, w), dtype=np.int64)
    for rgb_val, class_idx in RGB_TO_CLASS.items():
        mask = (
            (rgb_seg[:, :, 0] == rgb_val[0])
            & (rgb_seg[:, :, 1] == rgb_val[1])
            & (rgb_seg[:, :, 2] == rgb_val[2])
        )
        class_seg[mask] = class_idx
    return class_seg


def collect_sequences():
    sequences = []
    required_span = (SEQUENCE_LENGTH - 1) * SEQUENCE_STEP + 1
    for dataset_name in DATASETS:
        for view in VIEWS:
            base_path = DATA_ROOT / dataset_name / "Bscans-dt" / view
            image_dir = base_path / "Image"
            seg_dir = base_path / "Segmentation"
            if not image_dir.exists() or not seg_dir.exists():
                continue
            image_files = sorted(image_dir.glob("*.png"))
            if len(image_files) < required_span:
                continue
            for i in range(0, len(image_files) - required_span + 1):
                indices = [i + j * SEQUENCE_STEP for j in range(SEQUENCE_LENGTH)]
                window_files = [image_files[idx] for idx in indices]
                valid = True
                for img_path in window_files:
                    if not (seg_dir / img_path.name).exists():
                        valid = False
                        break
                if not valid:
                    continue
                seq_id = f"{dataset_name}_{view}_{window_files[0].stem}"
                sequences.append(
                    {
                        "id": seq_id,
                        "image_paths": window_files,
                        "seg_dir": seg_dir,
                    }
                )
    return sequences


def load_sample(sample_meta):
    imgs = []
    masks = []
    img_paths = []
    seg_paths = []
    for img_path in sample_meta["image_paths"]:
        seg_path = sample_meta["seg_dir"] / img_path.name
        img = np.array(Image.open(img_path))
        seg_rgb = np.array(Image.open(seg_path))
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        seg = rgb_to_class(seg_rgb)
        if img.shape != INPUT_SIZE:
            raise ValueError(f"Unexpected image shape {img.shape} for {img_path}")
        if seg.shape != INPUT_SIZE:
            raise ValueError(f"Unexpected segmentation shape {seg.shape} for {seg_path}")
        img_f = (img.astype(np.float32) / 255.0)[None, :, :]
        seg_onehot = np.eye(len(RGB_TO_CLASS), dtype=np.float32)[seg].transpose(2, 0, 1)
        imgs.append(img_f)
        masks.append(seg_onehot)
        img_paths.append(str(img_path))
        seg_paths.append(str(seg_path))
    return np.stack(imgs), np.stack(masks), img_paths, seg_paths


def save_visualization(image_tchw, label_tchw, sample_id, out_path, img_paths, seg_paths):
    t, _, h, w = image_tchw.shape
    rows = []
    row_labels = []
    for i in range(t):
        img_u8 = (image_tchw[i, 0] * 255.0).clip(0, 255).astype(np.uint8)
        gt_cls = label_tchw[i].argmax(axis=0).astype(np.int64)
        gt_rgb = PALETTE[gt_cls % len(PALETTE)]
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
        overlay = (0.6 * img_rgb + 0.4 * gt_rgb).clip(0, 255).astype(np.uint8)
        row = np.concatenate([img_rgb, gt_rgb, overlay], axis=1)
        rows.append(row)
        row_labels.append(f"t={i} | image: {img_paths[i]} | seg: {seg_paths[i]}")
    canvas_np = np.concatenate(rows, axis=0)
    canvas = Image.fromarray(canvas_np, mode="RGB")
    header_h = 34 + 16 * t
    out = Image.new("RGB", (canvas.width, canvas.height + header_h), (20, 20, 20))
    out.paste(canvas, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 10), f"{sample_id} | rows=timestep, cols=[image|label|overlay]", fill=(230, 230, 230))
    y = 26
    for text in row_labels:
        draw.text((8, y), text, fill=(230, 230, 230))
        y += 16
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize iOCT dataloader-like inputs.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("debug/dataloader_viz"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences = collect_sequences()
    if len(sequences) == 0:
        raise RuntimeError("No valid sequences found.")

    selected = sequences[args.start_index : args.start_index + args.batch_size]
    if len(selected) == 0:
        raise RuntimeError("No samples selected. Adjust --start-index.")

    batch_images = []
    batch_labels = []
    batch_ids = []
    for sample in selected:
        image_tchw, label_tchw, img_paths, seg_paths = load_sample(sample)
        batch_images.append(image_tchw)
        batch_labels.append(label_tchw)
        batch_ids.append(sample["id"])
        sample["img_paths"] = img_paths
        sample["seg_paths"] = seg_paths

    batch_images = np.stack(batch_images)  # B,T,1,H,W
    batch_labels = np.stack(batch_labels)  # B,T,C,H,W

    print(f"Found sequences: {len(sequences)}")
    print(f"Batch image shape: {batch_images.shape}")
    print(f"Batch label shape: {batch_labels.shape}")
    print(f"Batch ids: {batch_ids}")

    for bi, sid in enumerate(batch_ids):
        out_img = args.out_dir / f"batch_sample{bi}_{sid}.png"
        save_visualization(
            batch_images[bi],
            batch_labels[bi],
            sid,
            out_img,
            selected[bi]["img_paths"],
            selected[bi]["seg_paths"],
        )
        print(f"Saved: {out_img}")

    class_counts = batch_labels.sum(axis=(0, 1, 3, 4)).astype(np.int64)
    counts_path = args.out_dir / "batch_class_counts.txt"
    with open(counts_path, "w", encoding="utf-8") as f:
        for c, count in enumerate(class_counts):
            f.write(f"class_{c}: {int(count)}\n")
    print(f"Saved: {counts_path}")


if __name__ == "__main__":
    main()
