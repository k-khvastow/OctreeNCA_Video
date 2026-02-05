#!/usr/bin/env python3
"""Overview for the Video2D dataset used in train_video2d.py.

Reads train_video2d.py to pull DATA_ROOT, LABEL_ROOT, SELECTED_CLASSES
without importing heavy dependencies.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional


@dataclass
class VideoTrainConfig:
    data_root: Optional[str]
    label_root: Optional[str]
    selected_classes: Optional[List[int]]


def _literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def parse_train_file(train_file: Path) -> VideoTrainConfig:
    src = train_file.read_text()
    tree = ast.parse(src)

    data_root = None
    label_root = None
    selected_classes = None

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "DATA_ROOT":
                data_root = _literal_eval(node.value)
            elif name == "LABEL_ROOT":
                label_root = _literal_eval(node.value)
            elif name == "SELECTED_CLASSES":
                selected_classes = _literal_eval(node.value)

    return VideoTrainConfig(
        data_root=data_root,
        label_root=label_root,
        selected_classes=selected_classes,
    )


def _scan_video2d(data_root: Path, label_root: Path, top_n: int = 5) -> Dict[str, Any]:
    seq_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    seq_names = {p.name for p in seq_dirs}

    label_files = list(label_root.glob("*.mat")) if label_root.exists() else []
    label_names = {p.stem for p in label_files}

    with_label = sorted([p for p in seq_dirs if p.name in label_names], key=lambda p: p.name)
    missing_label = sorted([p for p in seq_dirs if p.name not in label_names], key=lambda p: p.name)
    missing_seq = sorted([name for name in label_names if name not in seq_names])

    frame_counts = []
    per_sequence = []
    for seq in with_label:
        bmp_files = sorted(seq.glob("*.bmp"))
        count = len(bmp_files)
        frame_counts.append(count)
        per_sequence.append({"sequence": seq.name, "frames": count})

    stats = None
    if frame_counts:
        stats = {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": mean(frame_counts),
            "median": median(frame_counts),
            "total_frames": sum(frame_counts),
        }

    top_sequences = sorted(per_sequence, key=lambda x: x["frames"], reverse=True)[:top_n]

    return {
        "sequences_total": len(seq_dirs),
        "sequences_with_label": len(with_label),
        "sequences_missing_label": len(missing_label),
        "labels_total": len(label_files),
        "labels_missing_sequence": len(missing_seq),
        "frame_stats": stats,
        "top_sequences": top_sequences,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Overview Video2D dataset used in train_video2d.py")
    p.add_argument("--train-file", default="train_video2d.py", help="Path to train_video2d.py")
    p.add_argument("--data-root", default=None, help="Override DATA_ROOT")
    p.add_argument("--label-root", default=None, help="Override LABEL_ROOT")
    p.add_argument("--top", type=int, default=5, help="Show top N sequences by frame count")
    p.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train_file = Path(args.train_file)
    if not train_file.is_file():
        raise SystemExit(f"Train file not found: {train_file}")

    cfg = parse_train_file(train_file)
    data_root = Path(args.data_root or cfg.data_root or "")
    label_root = Path(args.label_root or cfg.label_root or "")

    overview = None
    if data_root.exists() and label_root.exists():
        overview = _scan_video2d(data_root, label_root, top_n=args.top)

    if args.json:
        import json
        payload = {
            "train_file": str(train_file),
            "data_root": str(data_root),
            "label_root": str(label_root),
            "data_root_exists": data_root.exists(),
            "label_root_exists": label_root.exists(),
            "selected_classes": cfg.selected_classes,
            "overview": overview,
        }
        print(json.dumps(payload, indent=2))
        return

    print("Video2D dataset overview (train_video2d.py)")
    print(f"Train file: {train_file}")
    print(f"Data root: {data_root} (exists: {data_root.exists()})")
    print(f"Label root: {label_root} (exists: {label_root.exists()})")
    print(f"Selected classes: {cfg.selected_classes}")

    if overview is None:
        print("Data root and/or label root do not exist. No scan performed.")
        return

    print("\nSequences")
    print(f"total={overview['sequences_total']} with_label={overview['sequences_with_label']} missing_label={overview['sequences_missing_label']}")
    print(f"labels_total={overview['labels_total']} labels_missing_sequence={overview['labels_missing_sequence']}")

    stats = overview["frame_stats"]
    if stats:
        print("\nFrames (per sequence with labels)")
        print(
            f"min={stats['min']} max={stats['max']} mean={stats['mean']:.2f} "
            f"median={stats['median']:.2f} total={stats['total_frames']}"
        )

    if overview["top_sequences"]:
        print("\nTop sequences by frame count")
        for item in overview["top_sequences"]:
            print(f"- {item['sequence']}: {item['frames']} frames")


if __name__ == "__main__":
    main()
