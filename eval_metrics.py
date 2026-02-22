"""
Evaluation script for UNetWrapper2D segmentation model.
Computes IoU, Dice, Boundary F1, and Pixel Accuracy over a dataset,
then prints a summary table.

The UNet architecture is reconstructed to exactly match the checkpoint:
  - No BatchNorm (Conv2d with bias=True)
  - Fixed 32 channels throughout (no doubling per level)
  - ConvTranspose2d upsampling (bilinear=False)
  - outc is a plain nn.Conv2d (key: outc.weight / outc.bias)

Directory layout expected:
    ioct_data/peeling/Bscans-dt/A/Image/          <- input images
    ioct_data/peeling/Bscans-dt/A/Segmentation/   <- ground-truth masks
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from tabulate import tabulate          # pip install tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.models.UNetWrapper2D import UNetWrapper2D


# ══════════════════════════════════════════════════════════════════════════════
#  UNet reconstructed from checkpoint weights
#  Key observations from the checkpoint:
#    inc.double_conv.0.weight  : [32,  1, 3, 3]  -> Conv(1,  32) no BN, bias
#    inc.double_conv.2.weight  : [32, 32, 3, 3]  -> Conv(32, 32) no BN, bias
#    down*.double_conv.0.weight: [32, 32, 3, 3]  -> channels stay at 32 always
#    up*.up.weight             : ConvTranspose2d (not bilinear upsample)
#    up1.conv.double_conv.0    : [32, 64, 3, 3]  -> input is cat(skip32, up32)=64
#    outc.weight               : direct Conv2d (no .conv wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Two Conv2d layers with ReLU, NO BatchNorm, bias=True."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bilinear upsample (no learnable weights) then DoubleConv.
    Confirmed by checkpoint: no up*.up.weight keys present.
    After cat with skip (bc): total input to DoubleConv = bc*2 -> bc
    """
    def __init__(self, channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(channels * 2, channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """
    UNet with fixed base_channels=32 throughout all levels (no doubling),
    no BatchNorm, ConvTranspose2d upsampling, plain Conv2d output head.
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 1,
                 base_channels: int = 32):
        super().__init__()
        bc = base_channels
        self.inc   = DoubleConv(n_channels, bc)      # -> bc
        self.down1 = Down(bc, bc)                    # -> bc
        self.down2 = Down(bc, bc)                    # -> bc
        self.down3 = Down(bc, bc)                    # -> bc
        self.down4 = Down(bc, bc)                    # -> bc  (bottleneck)

        self.up1   = Up(bc)
        self.up2   = Up(bc)
        self.up3   = Up(bc)
        self.up4   = Up(bc)
        # outc.weight shape will be [n_classes, bc, 1, 1]
        self.outc  = nn.Conv2d(bc, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# ══════════════════════════════════════════════════════════════════════════════
#  Paths & config
# ══════════════════════════════════════════════════════════════════════════════

# All image/segmentation root pairs to evaluate (can add more here or via --dirs)
DATASET_DIRS = [
    ("ioct_data/peeling/Bscans-dt/A/Image", "ioct_data/peeling/Bscans-dt/A/Segmentation"),
    ("ioct_data/peeling/Bscans-dt/B/Image", "ioct_data/peeling/Bscans-dt/B/Segmentation"),
    ("ioct_data/sri/Bscans-dt/A/Image",     "ioct_data/sri/Bscans-dt/A/Segmentation"),
    ("ioct_data/sri/Bscans-dt/B/Image",     "ioct_data/sri/Bscans-dt/B/Segmentation"),
]
MODEL_PATH = "Models/iOCT_UNet/best_model.pth"

N_CHANNELS    = 1      # 1 = grayscale, 3 = RGB
N_CLASSES     = 7      # 7 output classes (from checkpoint outc shape [7,32,1,1])
BASE_CHANNELS = 32     # fixed channel width (from checkpoint)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD   = 0.5
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OUTPUT_CSV  = "eval_results.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> UNetWrapper2D:
    unet    = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES,
                   base_channels=BASE_CHANNELS)
    wrapper = UNetWrapper2D(unet)

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state_dict, dict) and key in state_dict:
            state_dict = state_dict[key]
            break

    wrapper.model.load_state_dict(state_dict)
    wrapper.to(DEVICE)
    wrapper.eval()
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("L") if N_CHANNELS == 1 else img.convert("RGB")
    return TF.to_tensor(img).unsqueeze(0)   # (1, C, H, W) in [0,1]


def load_mask(path: str) -> np.ndarray:
    """
    Load a colour-coded RGB segmentation mask and convert to integer class map.
    Each colour maps to a class index 0-N; black (0,0,0) = background = class 0.
    """
    arr = np.array(Image.open(path).convert("RGB"))  # (H, W, 3)

    # ── colour palette from train_baseline_unet.py RGB_TO_CLASS ─────────────
    COLOUR_MAP = {
        (  0,   0,   0): 0,   # Background (black)
        (255,   0,   0): 1,   # Class 1 (red)
        (  0, 255, 209): 2,   # Class 2 (cyan)
        ( 61, 255,   0): 3,   # Class 3 (green)
        (  0,  78, 255): 4,   # Class 4 (blue)
        (255, 189,   0): 5,   # Class 5 (yellow/orange)
        (218,   0, 255): 6,   # Class 6 (magenta)
    }

    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for rgb, cls_idx in COLOUR_MAP.items():
        match = (arr[:, :, 0] == rgb[0]) & \
                (arr[:, :, 1] == rgb[1]) & \
                (arr[:, :, 2] == rgb[2])
        mask[match] = cls_idx

    return mask


def find_pairs(image_dir: str, seg_dir: str):
    img_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMG_EXTS
    )
    pairs = []
    for img_path in img_paths:
        for ext in [img_path.suffix] + list(IMG_EXTS - {img_path.suffix}):
            seg_path = Path(seg_dir) / (img_path.stem + ext)
            if seg_path.exists():
                pairs.append((str(img_path), str(seg_path)))
                break
    if not pairs:
        raise FileNotFoundError(
            f"No matching image/mask pairs found.\n"
            f"  images : {image_dir}\n"
            f"  masks  : {seg_dir}"
        )
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics — matching src/scores exactly
#
#  Two variants are computed, mirroring the training codebase:
#
#  1. Per-image (macro): IoU/Dice computed per image per class, then averaged.
#     Matches DiceScore / IoUScore behaviour.
#     Skips classes absent from BOTH pred and gt for that image (no epsilon).
#
#  2. Global (patchwise): TP/FP/FN accumulated across ALL images before
#     computing the final score. Matches PatchwiseDiceScore / PatchwiseIoUScore.
#     More robust — equivalent to weighting by class frequency.
#
#  Both ignore class 0 (background) to match training evaluation convention.
# ══════════════════════════════════════════════════════════════════════════════

def _boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray,
                 eps: float = 1e-7) -> float:
    struct = np.ones((2, 2), dtype=bool)
    def boundary(m):
        return m.astype(bool) & ~binary_erosion(m.astype(bool), structure=struct)
    pred_b, gt_b = boundary(pred_bin), boundary(gt_bin)
    tp        = float((pred_b & gt_b).sum())
    precision = (tp + eps) / (float(pred_b.sum()) + eps)
    recall    = (tp + eps) / (float(gt_b.sum()) + eps)
    return 2 * precision * recall / (precision + recall + eps)


def per_image_metrics(pred: np.ndarray, gt: np.ndarray,
                      ignore_background: bool = True) -> dict:
    """
    Matches DiceScore / IoUScore from the training codebase.
    Returns per-class dicts and macro-averaged scalars.
    Skips classes absent from both pred and gt (no epsilon added).
    """
    classes = list(range(N_CLASSES))
    if ignore_background:
        classes = [c for c in classes if c != 0]

    dice_per_class = {}
    iou_per_class  = {}
    bf1_per_class  = {}
    pa_vals        = []

    for c in classes:
        p = (pred == c)
        g = (gt   == c)
        gt_has  = g.any()
        pred_has = p.any()

        # skip if absent from both (matches training score behaviour)
        if not gt_has and not pred_has:
            continue

        tp = float((p & g).sum())
        fp = float((p & ~g).sum())
        fn = float((~p & g).sum())

        if gt_has:
            dice_per_class[c] = 2 * tp / (2 * tp + fp + fn + 1e-8)
            iou_per_class[c]  = tp / (tp + fp + fn + 1e-8)
        else:
            # gt absent but pred fired → score 0 (matches training)
            dice_per_class[c] = 0.0
            iou_per_class[c]  = 0.0

        bf1_per_class[c] = _boundary_f1(p, g)
        pa_vals.append(float((p == g).sum()) / g.size)

    if not dice_per_class:
        return {"dice": 0.0, "iou": 0.0, "bf1": 0.0, "pa": 1.0,
                "dice_per_class": {}, "iou_per_class": {}, "bf1_per_class": {}}

    return {
        "dice": float(np.mean(list(dice_per_class.values()))),
        "iou":  float(np.mean(list(iou_per_class.values()))),
        "bf1":  float(np.mean(list(bf1_per_class.values()))),
        "pa":   float(np.mean(pa_vals)) if pa_vals else 1.0,
        "dice_per_class": dice_per_class,
        "iou_per_class":  iou_per_class,
        "bf1_per_class":  bf1_per_class,
    }


class GlobalAccumulator:
    """
    Matches PatchwiseDiceScore / PatchwiseIoUScore from the training codebase.
    Accumulates TP/FP/FN globally across all images, then computes final scores.
    """
    def __init__(self, n_classes: int, ignore_background: bool = True):
        self.n_classes = n_classes
        self.ignore_background = ignore_background
        self.tp = np.zeros(n_classes, dtype=np.float64)
        self.fp = np.zeros(n_classes, dtype=np.float64)
        self.fn = np.zeros(n_classes, dtype=np.float64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        for c in range(self.n_classes):
            p = (pred == c)
            g = (gt   == c)
            self.tp[c] += (p & g).sum()
            self.fp[c] += (p & ~g).sum()
            self.fn[c] += (~p & g).sum()

    def scores(self) -> dict:
        classes = list(range(self.n_classes))
        if self.ignore_background:
            classes = [c for c in classes if c != 0]

        dice_per_class = {}
        iou_per_class  = {}
        for c in classes:
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            # skip if completely absent from entire dataset
            if tp + fp + fn == 0:
                continue
            dice_per_class[c] = 2 * tp / (2 * tp + fp + fn + 1e-5)
            iou_per_class[c]  = tp / (tp + fp + fn + 1e-4)

        if not dice_per_class:
            return {"global_dice": 0.0, "global_iou": 0.0}

        return {
            "global_dice": float(np.mean(list(dice_per_class.values()))),
            "global_iou":  float(np.mean(list(iou_per_class.values()))),
            "global_dice_per_class": dice_per_class,
            "global_iou_per_class":  iou_per_class,
        }




# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model: UNetWrapper2D, img_tensor: torch.Tensor) -> np.ndarray:
    logits = model(img_tensor.to(DEVICE))["logits"]  # (1, H, W, C)
    logits = logits.permute(0, 3, 1, 2)              # (1, C, H, W)
    if N_CLASSES == 1:
        pred = (torch.sigmoid(logits) > THRESHOLD).squeeze().cpu().numpy()
    else:
        # argmax across classes; treat class 0 as background, rest as foreground
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
    return pred.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(dataset_dirs: list, model_path: str):
    print(f"Device             : {DEVICE}")
    print(f"Loading model from : {model_path}")
    model = load_model(model_path)

    all_pairs = []
    for image_dir, seg_dir in dataset_dirs:
        pairs = find_pairs(image_dir, seg_dir)
        print(f"  {image_dir}: {len(pairs)} pairs")
        all_pairs.extend([(img, seg, image_dir) for img, seg in pairs])
    print(f"Total: {len(all_pairs)} image/mask pairs\n")

    import csv
    accumulator = GlobalAccumulator(n_classes=N_CLASSES, ignore_background=True)
    agg = {"pa": [], "iou": [], "dice": [], "bf1": []}

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "pixel_acc", "iou", "dice", "boundary_f1"])

        for img_path, seg_path, src_dir in all_pairs:
            img_tensor = load_image(img_path)
            gt_mask    = load_mask(seg_path)
            pred_mask  = predict(model, img_tensor)

            if pred_mask.shape != gt_mask.shape:
                pred_pil  = Image.fromarray(pred_mask * 255).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                pred_mask = (np.array(pred_pil) > 127).astype(np.uint8)

            m = per_image_metrics(pred_mask, gt_mask, ignore_background=True)
            accumulator.update(pred_mask, gt_mask)

            writer.writerow([Path(img_path).stem,
                             f"{m['pa']:.4f}", f"{m['iou']:.4f}",
                             f"{m['dice']:.4f}", f"{m['bf1']:.4f}"])
            agg["pa"].append(m["pa"])
            agg["iou"].append(m["iou"])
            agg["dice"].append(m["dice"])
            agg["bf1"].append(m["bf1"])

    print(f"Per-image results saved to: {OUTPUT_CSV}\n")
    global_sc = accumulator.scores()

    # ── terminal: two summary tables ─────────────────────────────────────────
    print("Per-image macro average (matches DiceScore / IoUScore):")
    rows1 = []
    for label, fn in [("MEAN", np.mean), ("STD", np.std),
                      ("MIN",  np.min),  ("MAX", np.max)]:
        rows1.append([label, f"{fn(agg['pa']):.4f}", f"{fn(agg['iou']):.4f}",
                      f"{fn(agg['dice']):.4f}", f"{fn(agg['bf1']):.4f}"])
    print(tabulate(rows1, headers=["", "Pixel Acc", "IoU", "Dice", "Boundary F1"],
                   tablefmt="rounded_outline"))

    print("\nGlobal aggregate (matches PatchwiseDiceScore / PatchwiseIoUScore):")
    rows2 = [["ALL IMAGES",
              f"{global_sc['global_iou']:.4f}",
              f"{global_sc['global_dice']:.4f}"]]
    print(tabulate(rows2, headers=["", "Global IoU", "Global Dice"],
                   tablefmt="rounded_outline"))

    print("\nPer-class global scores (foreground classes only):")
    class_rows = []
    for c in sorted(global_sc["global_iou_per_class"].keys()):
        class_rows.append([f"Class {c}",
                           f"{global_sc['global_iou_per_class'][c]:.4f}",
                           f"{global_sc['global_dice_per_class'][c]:.4f}"])
    print(tabulate(class_rows, headers=["Class", "Global IoU", "Global Dice"],
                   tablefmt="rounded_outline"))


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNetWrapper2D segmentation metrics")
    parser.add_argument("--dirs", nargs="+", metavar="IMG_DIR:SEG_DIR",
                        default=[f"{a}:{b}" for a, b in DATASET_DIRS],
                        help="One or more image:segmentation directory pairs (space-separated)")
    parser.add_argument("--model",    default=MODEL_PATH, help="Path to .pth checkpoint")
    parser.add_argument("--channels", type=int,   default=N_CHANNELS,    help="Input channels (default: 1)")
    parser.add_argument("--classes",  type=int,   default=N_CLASSES,     help="Output classes (default: 1)")
    parser.add_argument("--base-ch",  type=int,   default=BASE_CHANNELS, help="Base channel width (default: 32)")
    parser.add_argument("--thresh",   type=float, default=THRESHOLD,     help="Sigmoid threshold (default: 0.5)")
    parser.add_argument("--output",   default=OUTPUT_CSV,                help="Path for per-image CSV output (default: eval_results.csv)")
    args = parser.parse_args()

    N_CHANNELS    = args.channels
    N_CLASSES     = args.classes
    BASE_CHANNELS = args.base_ch
    OUTPUT_CSV    = args.output
    THRESHOLD     = args.thresh

    dataset_dirs = []
    for entry in args.dirs:
        parts = entry.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"--dirs entries must be in IMG_DIR:SEG_DIR format, got: {entry!r}")
        dataset_dirs.append((parts[0], parts[1]))
    evaluate(dataset_dirs, args.model)