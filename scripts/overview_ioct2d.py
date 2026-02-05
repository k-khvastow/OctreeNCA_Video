#!/usr/bin/env python3
"""Overview for the iOCT dataset used in train_ioct2d.py.

Reads train_ioct2d.py to pull DATA_ROOT, DATASETS, VIEWS, SELECTED_CLASSES,
and the RGB_TO_CLASS mapping without importing heavy dependencies.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IoctTrainConfig:
    data_root: Optional[str]
    datasets: Optional[List[str]]
    views: Optional[List[str]]
    selected_classes: Optional[List[int]]
    rgb_to_class: Optional[Dict[Tuple[int, int, int], int]]


def _literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def parse_train_file(train_file: Path) -> IoctTrainConfig:
    src = train_file.read_text()
    tree = ast.parse(src)

    data_root = None
    datasets = None
    views = None
    selected_classes = None
    rgb_to_class = None

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "DATA_ROOT":
                data_root = _literal_eval(node.value)
            elif name == "DATASETS":
                datasets = _literal_eval(node.value)
            elif name == "VIEWS":
                views = _literal_eval(node.value)
            elif name == "SELECTED_CLASSES":
                selected_classes = _literal_eval(node.value)

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "iOCTDatasetForExperiment":
            for sub in node.body:
                if (
                    isinstance(sub, ast.Assign)
                    and len(sub.targets) == 1
                    and isinstance(sub.targets[0], ast.Name)
                    and sub.targets[0].id == "RGB_TO_CLASS"
                ):
                    rgb_to_class = _literal_eval(sub.value)

    return IoctTrainConfig(
        data_root=data_root,
        datasets=datasets,
        views=views,
        selected_classes=selected_classes,
        rgb_to_class=rgb_to_class,
    )


def _as_list(value: Optional[Any]) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _scan_ioct(root: Path, datasets: List[str], views: List[str]) -> Dict[str, Any]:
    per_group = []
    total_images = 0
    total_segs = 0
    total_pairs = 0
    total_missing_seg = 0
    total_missing_img = 0
    missing_dirs = []

    for dataset in datasets:
        for view in views:
            base = root / dataset / "Bscans-dt" / view
            image_dir = base / "Image"
            seg_dir = base / "Segmentation"
            image_exists = image_dir.exists()
            seg_exists = seg_dir.exists()

            if not image_exists or not seg_exists:
                missing_dirs.append({
                    "dataset": dataset,
                    "view": view,
                    "image_dir": str(image_dir),
                    "seg_dir": str(seg_dir),
                    "image_exists": image_exists,
                    "seg_exists": seg_exists,
                })
                per_group.append({
                    "dataset": dataset,
                    "view": view,
                    "image_dir": str(image_dir),
                    "seg_dir": str(seg_dir),
                    "images": 0,
                    "segs": 0,
                    "pairs": 0,
                    "missing_seg": 0,
                    "missing_img": 0,
                })
                continue

            img_files = sorted(image_dir.glob("*.png"))
            seg_files = sorted(seg_dir.glob("*.png"))

            img_stems = {p.stem for p in img_files}
            seg_stems = {p.stem for p in seg_files}

            pairs = img_stems & seg_stems
            missing_seg = img_stems - seg_stems
            missing_img = seg_stems - img_stems

            images = len(img_stems)
            segs = len(seg_stems)
            pair_count = len(pairs)

            per_group.append({
                "dataset": dataset,
                "view": view,
                "image_dir": str(image_dir),
                "seg_dir": str(seg_dir),
                "images": images,
                "segs": segs,
                "pairs": pair_count,
                "missing_seg": len(missing_seg),
                "missing_img": len(missing_img),
            })

            total_images += images
            total_segs += segs
            total_pairs += pair_count
            total_missing_seg += len(missing_seg)
            total_missing_img += len(missing_img)

    return {
        "per_group": per_group,
        "totals": {
            "images": total_images,
            "segs": total_segs,
            "pairs": total_pairs,
            "missing_seg": total_missing_seg,
            "missing_img": total_missing_img,
        },
        "missing_dirs": missing_dirs,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Overview iOCT dataset used in train_ioct2d.py")
    p.add_argument("--train-file", default="train_ioct2d.py", help="Path to train_ioct2d.py")
    p.add_argument("--data-root", default=None, help="Override DATA_ROOT")
    p.add_argument("--datasets", default=None, help="Override DATASETS, comma-separated")
    p.add_argument("--views", default=None, help="Override VIEWS, comma-separated")
    p.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train_file = Path(args.train_file)
    if not train_file.is_file():
        raise SystemExit(f"Train file not found: {train_file}")

    cfg = parse_train_file(train_file)

    data_root = Path(args.data_root or cfg.data_root or "")
    datasets = args.datasets.split(",") if args.datasets else _as_list(cfg.datasets) or []
    views = args.views.split(",") if args.views else _as_list(cfg.views) or []

    if not datasets and data_root.exists():
        datasets = [p.name for p in data_root.iterdir() if p.is_dir()]

    if not views and data_root.exists():
        # Try to infer views from folder structure
        view_set = set()
        for dataset in datasets:
            base = data_root / dataset / "Bscans-dt"
            if base.exists():
                for p in base.iterdir():
                    if p.is_dir():
                        view_set.add(p.name)
        views = sorted(view_set)

    overview = _scan_ioct(data_root, datasets, views) if data_root.exists() else None

    if args.json:
        import json
        rgb_to_class = None
        if cfg.rgb_to_class:
            rgb_to_class = [
                {"rgb": list(rgb), "class": cls} for rgb, cls in cfg.rgb_to_class.items()
            ]
        payload = {
            "train_file": str(train_file),
            "data_root": str(data_root),
            "data_root_exists": data_root.exists(),
            "datasets": datasets,
            "views": views,
            "selected_classes": cfg.selected_classes,
            "rgb_to_class": rgb_to_class,
            "overview": overview,
        }
        print(json.dumps(payload, indent=2))
        return

    print("iOCT dataset overview (train_ioct2d.py)")
    print(f"Train file: {train_file}")
    print(f"Data root: {data_root} (exists: {data_root.exists()})")
    print(f"Datasets: {', '.join(datasets) if datasets else '(none)'}")
    print(f"Views: {', '.join(views) if views else '(none)'}")
    print(f"Selected classes: {cfg.selected_classes}")
    if cfg.rgb_to_class:
        num_classes = len(set(cfg.rgb_to_class.values()))
        print(f"RGB->class entries: {len(cfg.rgb_to_class)} (classes: {num_classes})")
    else:
        print("RGB->class entries: (not found)")

    if overview is None:
        print("Data root does not exist or is not accessible. No scan performed.")
        return

    print("\nPer dataset/view")
    for row in overview["per_group"]:
        print(
            f"- {row['dataset']}/{row['view']}: images={row['images']} segs={row['segs']} "
            f"pairs={row['pairs']} missing_seg={row['missing_seg']} missing_img={row['missing_img']}"
        )

    totals = overview["totals"]
    print("\nTotals")
    print(
        f"images={totals['images']} segs={totals['segs']} pairs={totals['pairs']} "
        f"missing_seg={totals['missing_seg']} missing_img={totals['missing_img']}"
    )

    if overview["missing_dirs"]:
        print("\nMissing directories")
        for item in overview["missing_dirs"]:
            print(
                f"- {item['dataset']}/{item['view']} image_dir_exists={item['image_exists']} "
                f"seg_dir_exists={item['seg_exists']}"
            )


if __name__ == "__main__":
    main()
