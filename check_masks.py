"""
Mask sanity check script.
Run this BEFORE evaluate_metrics.py to verify that the colour palette
in load_mask() correctly decodes your ground-truth masks.

Usage:
    python3 check_masks.py --masks ioct_data/peeling/Bscans-dt/A/Segmentation
    python3 check_masks.py --masks ioct_data/peeling/Bscans-dt/A/Segmentation --samples 20
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tabulate import tabulate

# ── paste the same COLOUR_MAP as in evaluate_metrics.py ──────────────────────
COLOUR_MAP = {
    (  0,   0,   0): 0,   # Background (black)
    (255,   0,   0): 1,   # Class 1 (red)
    (  0, 255, 209): 2,   # Class 2 (cyan)
    ( 61, 255,   0): 3,   # Class 3 (green)
    (  0,  78, 255): 4,   # Class 4 (blue)
    (255, 189,   0): 5,   # Class 5 (yellow/orange)
    (218,   0, 255): 6,   # Class 6 (magenta)
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def decode_mask(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        decoded : (H, W) uint8 class map
        unmatched : (H, W) bool, True where no palette entry matched
    """
    decoded   = np.zeros(arr.shape[:2], dtype=np.uint8)
    matched   = np.zeros(arr.shape[:2], dtype=bool)
    for rgb, cls_idx in COLOUR_MAP.items():
        hit = (arr[:, :, 0] == rgb[0]) & \
              (arr[:, :, 1] == rgb[1]) & \
              (arr[:, :, 2] == rgb[2])
        decoded[hit] = cls_idx
        matched[hit] = True
    unmatched = ~matched
    return decoded, unmatched


def main(seg_dir: str, n_samples: int, verbose: bool):
    paths = sorted(
        p for p in Path(seg_dir).iterdir()
        if p.suffix.lower() in IMG_EXTS
    )
    if not paths:
        print(f"ERROR: No image files found in {seg_dir}")
        return

    print(f"Found {len(paths)} mask files in {seg_dir}\n")

    # ── PASS 1: collect ALL unique colours across entire dataset ─────────────
    print("=" * 60)
    print("PASS 1 — Scanning all masks for unique colours ...")
    print("=" * 60)
    global_colour_counts: dict[tuple, int] = defaultdict(int)

    for path in paths:
        arr = np.array(Image.open(path).convert("RGB"))
        pixels = arr.reshape(-1, 3)
        colours, counts = np.unique(pixels, axis=0, return_counts=True)
        for c, n in zip(colours, counts):
            global_colour_counts[tuple(int(x) for x in c)] += int(n)

    total_px = sum(global_colour_counts.values())
    sorted_colours = sorted(global_colour_counts.items(), key=lambda x: -x[1])

    rows = []
    unknown_colours = []
    for rgb, count in sorted_colours:
        in_map  = rgb in COLOUR_MAP
        cls_idx = COLOUR_MAP.get(rgb, "UNKNOWN")
        status  = f"class {cls_idx}" if in_map else "*** NOT IN PALETTE ***"
        pct     = 100 * count / total_px
        rows.append([str(rgb), f"{count:,}", f"{pct:.3f}%", status])
        if not in_map and rgb != (0, 0, 0):
            unknown_colours.append((rgb, count))

    print(tabulate(rows, headers=["RGB", "Total pixels", "% of all", "Mapping"],
                   tablefmt="rounded_outline"))

    if unknown_colours:
        print(f"\n⚠️  {len(unknown_colours)} colour(s) NOT covered by COLOUR_MAP:")
        for rgb, count in unknown_colours:
            print(f"   RGB{rgb}  →  {count:,} pixels  — add this to COLOUR_MAP in both scripts!")
    else:
        print("\n✓  All colours are covered by COLOUR_MAP.")

    # ── PASS 2: per-mask decode check on a sample ────────────────────────────
    sample_paths = paths[:n_samples]
    print(f"\n{'=' * 60}")
    print(f"PASS 2 — Per-mask decode check (first {len(sample_paths)} masks)")
    print("=" * 60)

    any_unmatched = False
    rows2 = []
    for path in sample_paths:
        arr = np.array(Image.open(path).convert("RGB"))
        decoded, unmatched = decode_mask(arr)
        total      = arr.shape[0] * arr.shape[1]
        n_unmatched = int(unmatched.sum())
        classes_found = sorted(np.unique(decoded).tolist())
        flag = "⚠️  UNMATCHED PIXELS" if n_unmatched > 0 else "✓"
        rows2.append([path.name, str(classes_found),
                      n_unmatched, f"{100*n_unmatched/total:.3f}%", flag])
        if n_unmatched > 0:
            any_unmatched = True

    print(tabulate(rows2,
                   headers=["Mask file", "Classes decoded", "Unmatched px", "Unmatched %", "Status"],
                   tablefmt="rounded_outline"))

    if any_unmatched:
        print("\n⚠️  Some masks have unmatched pixels — those are silently mapped")
        print("    to class 0 (background) in evaluate_metrics.py, corrupting metrics.")
        print("    Add the missing colours to COLOUR_MAP in both scripts.\n")
    else:
        print(f"\n✓  All pixels in the sampled masks are correctly decoded.\n")

    # ── PASS 3: class distribution summary ───────────────────────────────────
    if verbose:
        print("=" * 60)
        print("PASS 3 — Class frequency across sampled masks")
        print("=" * 60)
        class_totals: dict[int, int] = defaultdict(int)
        for path in sample_paths:
            arr = np.array(Image.open(path).convert("RGB"))
            decoded, _ = decode_mask(arr)
            for cls_idx in range(max(COLOUR_MAP.values()) + 1):
                class_totals[cls_idx] += int((decoded == cls_idx).sum())

        sample_px = sum(class_totals.values())
        rows3 = []
        for cls_idx in sorted(class_totals):
            rgb_str = str(next((k for k, v in COLOUR_MAP.items() if v == cls_idx), "?"))
            rows3.append([cls_idx, rgb_str,
                          f"{class_totals[cls_idx]:,}",
                          f"{100*class_totals[cls_idx]/sample_px:.3f}%"])
        print(tabulate(rows3, headers=["Class", "RGB", "Pixels", "% of sample"],
                       tablefmt="rounded_outline"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check RGB mask colour palette")
    parser.add_argument("--masks",   required=True,      help="Path to segmentation directory")
    parser.add_argument("--samples", type=int, default=10, help="Masks to check in Pass 2 (default: 10)")
    parser.add_argument("--verbose", action="store_true",  help="Also print class distribution (Pass 3)")
    args = parser.parse_args()
    main(args.masks, args.samples, args.verbose)