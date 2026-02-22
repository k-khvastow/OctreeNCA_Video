"""
Evaluation script for OctreeNCA2DDualView segmentation model.
Computes IoU, Dice, Boundary F1, and Pixel Accuracy over paired A+B views,
then prints summary tables matching the training codebase metric conventions.

Model: OctreeNCA2DDualView
Checkpoint: <path>/octree_study_new/Experiments/
            iOCT2D_dual_dispute_32_Dual-view iOCT (A+B) OctreeNCA segmentation./
            models/epoch_99/model.pth

Datasets used: peeling + sri, views A + B (same as training).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tabulate import tabulate
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── project imports (same as training script) ─────────────────────────────────
from src.models.Model_OctreeNCA_2d_dual_view import OctreeNCA2DDualView
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  Paths & config
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT  = "ioct_data"
MODEL_PATH = ("<path>/<path>/octree_study_new/Experiments/"
              "iOCT2D_dual_dispute_32_Dual-view iOCT (A+B) OctreeNCA segmentation."
              "/models/epoch_99/model.pth")
              # <path>/<path>/octree_study_new/Experiments/iOCT2D_dual_dispute_32_Dual-view iOCT 
              # (A+B) OctreeNCA segmentation./models/epoch_99/model.pth

OUTPUT_CSV = "eval_results_octree.csv"

DATASETS   = ["peeling", "sri"]
VIEWS      = ["A", "B"]          # view_a=A, view_b=B

N_CLASSES  = 7                   # background + 6 foreground
INPUT_SIZE = (512, 512)

# OctreeNCA model config — must match training
MODEL_CONFIG = {
    # Experiment settings (required by model init)
    "experiment.device": "cuda",
    
    # Performance settings (from configs.default)
    "performance.compile": False,
    "performance.data_parallel": False,
    "performance.num_workers": 8,
    "performance.unlock_CPU": True,
    "performance.inplace_operations": True,
    "performance.cudnn_benchmark": True,
    "performance.allow_tf32": True,
    
    # Model architecture
    "model.output_channels": N_CLASSES,
    "model.input_channels":  1,
    "model.channel_n":       32,
    "model.hidden_size":     64,
    "model.kernel_size":     3,
    "model.normalization":   "none",
    "model.apply_nonlin":    "torch.nn.Softmax(dim=-1)",
    "model.backbone_class":  "BasicNCA2DFast",
    "model.fire_rate":       0.5,
    "model.batchnorm_track_running_stats": False,
    "model.train.patch_sizes": [None] * 5,
    "model.train.loss_weighted_patching": False,
    "model.eval.patch_wise": False,
    "model.octree.separate_models": False,
    "model.dual_view.cross_fusion":    "film",
    "model.dual_view.cross_strength":  0.5,
    "model.dual_view.cross_use_tanh":  True,
    # VitCA disabled (not used in iOCT2D_dual_view training)
    "model.vitca": False,
    "model.vitca.depth": 1,
    "model.vitca.heads": 4,
    "model.vitca.mlp_dim": 64,
    "model.vitca.dropout": 0.0,
    "model.vitca.positional_embedding": "vit_handcrafted",
    "model.vitca.embed_cells": True,
    "model.vitca.embed_dim": 128,
    "model.vitca.embed_dropout": 0.0,
    # res_and_steps built below to match training
    "model.octree.res_and_steps": None,
}

def _build_octree_resolutions(input_size, steps, final_steps):
    h, w = input_size
    resolutions = []
    for _ in range(4):
        resolutions.append([h, w])
        h = max(1, h // 2)
        w = max(1, w // 2)
    res_and_steps = []
    for i, res in enumerate(resolutions):
        s = final_steps if i == len(resolutions) - 1 else steps
        res_and_steps.append([res, s])
    return res_and_steps

# Use max of the training ranges for deterministic eval
MODEL_CONFIG["model.octree.res_and_steps"] = _build_octree_resolutions(
    INPUT_SIZE, steps=20, final_steps=20
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── RGB → class index (identical to training) ─────────────────────────────────
RGB_TO_CLASS = {
    (  0,   0,   0): 0,
    (255,   0,   0): 1,
    (  0, 255, 209): 2,
    ( 61, 255,   0): 3,
    (  0,  78, 255): 4,
    (255, 189,   0): 5,
    (218,   0, 255): 6,
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> OctreeNCA2DDualView:
    model = OctreeNCA2DDualView(MODEL_CONFIG)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state_dict, dict) and key in state_dict:
            state_dict = state_dict[key]
            break
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Disable stochastic cell firing for deterministic eval.
    # fire_rate=0.5 during training randomly drops 50% of updates;
    # at eval this should be 1.0 (all cells always update).
    for m in model.modules():
        if hasattr(m, "fire_rate"):
            print(f"  Setting fire_rate: {m.fire_rate} -> 1.0 on {m.__class__.__name__}")
            m.fire_rate = 1.0

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def rgb_to_class(arr: np.ndarray) -> np.ndarray:
    """Convert RGB mask (H,W,3) to integer class map (H,W)."""
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    for rgb, cls_idx in RGB_TO_CLASS.items():
        hit = (arr[:,:,0]==rgb[0]) & (arr[:,:,1]==rgb[1]) & (arr[:,:,2]==rgb[2])
        out[hit] = cls_idx
    return out


def load_image(path: Path) -> torch.Tensor:
    """Load grayscale image → (1, 1, H, W) float32 tensor in [0,1], channel-first: (B,C,H,W)."""
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    img = img.astype(np.float32) / 255.0
    # model expects (B, C, H, W) channel-first
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return t.to(DEVICE)


def load_mask(path: Path) -> np.ndarray:
    """Load RGB mask → integer class map (H,W)."""
    return rgb_to_class(np.array(Image.open(path).convert("RGB")))


def collect_pairs(data_root: str, datasets: list, views: list) -> list:
    """
    Collect paired (view_a, view_b) frame info dicts.
    Mirrors iOCTPairedViewsDatasetForExperiment._collect_pairs().
    """
    root = Path(data_root)
    view_a, view_b = views[0], views[1]

    def sort_key(name):
        try:    return (0, int(Path(name).stem))
        except: return (1, Path(name).stem)

    pairs = []
    for ds in datasets:
        base = root / ds / "Bscans-dt"
        img_dir_a = base / view_a / "Image"
        seg_dir_a = base / view_a / "Segmentation"
        img_dir_b = base / view_b / "Image"
        seg_dir_b = base / view_b / "Segmentation"

        if not all(d.exists() for d in [img_dir_a, seg_dir_a, img_dir_b, seg_dir_b]):
            print(f"  Warning: skipping {ds} — directories not found")
            continue

        names_a = {p.name for p in img_dir_a.glob("*.png") if (seg_dir_a / p.name).exists()}
        names_b = {p.name for p in img_dir_b.glob("*.png") if (seg_dir_b / p.name).exists()}
        common  = sorted(names_a & names_b, key=sort_key)

        for name in common:
            pairs.append({
                "id":        f"{ds}_{Path(name).stem}",
                "dataset":   ds,
                "frame":     Path(name).stem,
                "img_a":     img_dir_a / name,
                "seg_a":     seg_dir_a / name,
                "img_b":     img_dir_b / name,
                "seg_b":     seg_dir_b / name,
            })

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  Prefetching dataset
# ══════════════════════════════════════════════════════════════════════════════

class PairedFrameDataset(Dataset):
    """
    Loads image+mask pairs from disk in background worker threads.
    Images are returned as float32 tensors ready for the model.
    Masks are returned as uint8 numpy arrays (kept on CPU for metrics).
    """
    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        info = self.pairs[idx]

        def _load_img(path):
            img = np.array(Image.open(path))
            if img.ndim == 3:
                img = np.mean(img, axis=2).astype(np.uint8)
            return (img.astype(np.float32) / 255.0)[..., None]  # (H, W, 1)

        def _load_mask(path):
            return rgb_to_class(np.array(Image.open(path).convert("RGB")))

        # model expects (B, C, H, W) channel-first; _load_img returns (H, W, 1)
        def _to_bchw(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)

        return {
            "id":    info["id"],
            "img_a": _to_bchw(_load_img(info["img_a"])),   # (1, 1, H, W)
            "img_b": _to_bchw(_load_img(info["img_b"])),
            "gt_a":  torch.from_numpy(_load_mask(info["seg_a"])),  # (H, W)
            "gt_b":  torch.from_numpy(_load_mask(info["seg_b"])),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics (identical logic to evaluate_metrics.py)
# ══════════════════════════════════════════════════════════════════════════════

def _mask_to_boundary(mask: np.ndarray, boundary_width: int = 2) -> np.ndarray:
    """
    Fast boundary extraction via distance transform.
    A pixel is on the boundary if its EDT distance to the complement <= boundary_width.
    Far faster than binary_erosion: no structuring-element iteration, pure C under the hood.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & (distance_transform_edt(mask) <= boundary_width)


def _boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = _mask_to_boundary(pred_bin.astype(bool))
    gt_b   = _mask_to_boundary(gt_bin.astype(bool))
    tp        = float((pred_b & gt_b).sum())
    precision = (tp + eps) / (float(pred_b.sum()) + eps)
    recall    = (tp + eps) / (float(gt_b.sum())  + eps)
    return 2 * precision * recall / (precision + recall + eps)


def per_image_metrics(pred: np.ndarray, gt: np.ndarray,
                      ignore_background: bool = True) -> dict:
    classes = list(range(N_CLASSES))
    if ignore_background:
        classes = [c for c in classes if c != 0]

    dice_pc, iou_pc, bf1_pc, pa_vals = {}, {}, {}, []
    for c in classes:
        p, g = (pred == c), (gt == c)
        if not p.any() and not g.any():
            continue
        tp = float((p & g).sum())
        fp = float((p & ~g).sum())
        fn = float((~p & g).sum())
        if g.any():
            dice_pc[c] = 2 * tp / (2 * tp + fp + fn + 1e-8)
            iou_pc[c]  = tp / (tp + fp + fn + 1e-8)
        else:
            dice_pc[c] = iou_pc[c] = 0.0
        bf1_pc[c] = _boundary_f1(p, g)

    # Pixel accuracy: single global computation on the full class map (not per-class average)
    pa = float((pred == gt).sum()) / gt.size

    if not dice_pc:
        return {"dice": 0., "iou": 0., "bf1": 0., "pa": pa,
                "dice_per_class": {}, "iou_per_class": {}, "bf1_per_class": {}}
    return {
        "dice": float(np.mean(list(dice_pc.values()))),
        "iou":  float(np.mean(list(iou_pc.values()))),
        "bf1":  float(np.mean(list(bf1_pc.values()))),
        "pa":   pa,
        "dice_per_class": dice_pc,
        "iou_per_class":  iou_pc,
        "bf1_per_class":  bf1_pc,
    }


class GlobalAccumulator:
    def __init__(self, n_classes: int, ignore_background: bool = True):
        self.n_classes = n_classes
        self.ignore_background = ignore_background
        self.tp = np.zeros(n_classes, dtype=np.float64)
        self.fp = np.zeros(n_classes, dtype=np.float64)
        self.fn = np.zeros(n_classes, dtype=np.float64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        for c in range(self.n_classes):
            p, g = (pred == c), (gt == c)
            self.tp[c] += (p & g).sum()
            self.fp[c] += (p & ~g).sum()
            self.fn[c] += (~p & g).sum()

    def scores(self) -> dict:
        classes = list(range(self.n_classes))
        if self.ignore_background:
            classes = [c for c in classes if c != 0]
        dice_pc, iou_pc = {}, {}
        for c in classes:
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            if tp + fp + fn == 0:
                continue
            dice_pc[c] = 2 * tp / (2 * tp + fp + fn + 1e-5)
            iou_pc[c]  = tp / (tp + fp + fn + 1e-4)
        if not dice_pc:
            return {"global_dice": 0., "global_iou": 0.,
                    "global_dice_per_class": {}, "global_iou_per_class": {}}
        return {
            "global_dice": float(np.mean(list(dice_pc.values()))),
            "global_iou":  float(np.mean(list(iou_pc.values()))),
            "global_dice_per_class": dice_pc,
            "global_iou_per_class":  iou_pc,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_pair(model: OctreeNCA2DDualView,
                 img_a: torch.Tensor,
                 img_b: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the dual-view model and return (pred_a, pred_b) as integer class maps.
    The model is called with positional args (image_a, image_b) and returns
    a dict with 'logits_a' and 'logits_b', each (1, H, W, C) after softmax.
    Falls back to inspecting the return value if the API differs slightly.
    """
    out = model(img_a, img_b)

    # ── one-time diagnostic on first call ────────────────────────────────────
    if not hasattr(predict_pair, "_diagnosed"):
        predict_pair._diagnosed = True
        if isinstance(out, dict):
            print(f"\n[DIAG] model output keys: {list(out.keys())}")
            for k, v in out.items():
                t = v if isinstance(v, torch.Tensor) else None
                print(f"  {k}: {t.shape if t is not None else type(v)}")
        elif isinstance(out, (tuple, list)):
            print(f"\n[DIAG] model output: {type(out).__name__} of length {len(out)}")
            for i, v in enumerate(out):
                t = v if isinstance(v, torch.Tensor) else None
                print(f"  [{i}]: {t.shape if t is not None else type(v)}")
    # ─────────────────────────────────────────────────────────────────────────

    # probabilities: (2, H, W, 7) — batch dim 0=view_a, 1=view_b
    probs = out["probabilities"]          # softmax already applied
    pred_a = probs[0].argmax(dim=-1).cpu().numpy().astype(np.uint8)  # (H, W)
    pred_b = probs[1].argmax(dim=-1).cpu().numpy().astype(np.uint8)

    # ── sanity check: verify predictions contain non-background classes ───────
    if not hasattr(predict_pair, "_checked"):
        predict_pair._checked = True
        unique_a = np.unique(pred_a)
        unique_b = np.unique(pred_b)
        print(f"[DIAG] pred_a unique classes: {unique_a}")
        print(f"[DIAG] pred_b unique classes: {unique_b}")
        if len(unique_a) == 1 and unique_a[0] == 0:
            print("[DIAG] WARNING: pred_a is all background — output key or shape may be wrong")
        if len(unique_b) == 1 and unique_b[0] == 0:
            print("[DIAG] WARNING: pred_b is all background — output key or shape may be wrong")

    return pred_a, pred_b


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(data_root: str, datasets: list, views: list, model_path: str):
    print(f"Device             : {DEVICE}")
    print(f"Loading model from : {model_path}")
    model = load_model(model_path)

    print(f"Collecting paired frames from: {data_root}")
    pairs = collect_pairs(data_root, datasets, views)
    print(f"Found {len(pairs)} paired frames ({views[0]}+{views[1]})\n")

    accumulator_a = GlobalAccumulator(N_CLASSES)
    accumulator_b = GlobalAccumulator(N_CLASSES)
    agg = {"pa": [], "iou": [], "dice": [], "bf1": []}

    dataset    = PairedFrameDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=(DEVICE == "cuda"),
                            prefetch_factor=4)

    iterator = dataloader
    if tqdm is not None:
        iterator = tqdm(dataloader, desc="Evaluating", unit="pair")

    t0 = time.time()
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "view", "pixel_acc", "iou", "dice", "boundary_f1"])

        for batch in iterator:
            img_a = batch["img_a"].squeeze(0).to(DEVICE)   # (1,H,W,1)
            img_b = batch["img_b"].squeeze(0).to(DEVICE)
            gt_a  = batch["gt_a"].squeeze(0).numpy()       # (H,W)
            gt_b  = batch["gt_b"].squeeze(0).numpy()
            pair_id = batch["id"][0]

            pred_a, pred_b = predict_pair(model, img_a, img_b)

            for pred, gt, view, acc in [
                (pred_a, gt_a, views[0], accumulator_a),
                (pred_b, gt_b, views[1], accumulator_b),
            ]:
                m = per_image_metrics(pred, gt)
                acc.update(pred, gt)
                writer.writerow([pair_id, view,
                                 f"{m['pa']:.4f}", f"{m['iou']:.4f}",
                                 f"{m['dice']:.4f}", f"{m['bf1']:.4f}"])
                agg["pa"].append(m["pa"])
                agg["iou"].append(m["iou"])
                agg["dice"].append(m["dice"])
                agg["bf1"].append(m["bf1"])

    elapsed = time.time() - t0
    print(f"Inference + metrics: {elapsed/60:.1f} min "
          f"({elapsed/len(pairs):.2f} s/pair)")

    print(f"Per-frame results saved to: {OUTPUT_CSV}\n")

    # Combined global scores across both views
    sc_a = accumulator_a.scores()
    sc_b = accumulator_b.scores()
    global_dice = np.mean([sc_a["global_dice"], sc_b["global_dice"]])
    global_iou  = np.mean([sc_a["global_iou"],  sc_b["global_iou"]])
    all_classes = sorted(set(sc_a["global_dice_per_class"]) | set(sc_b["global_dice_per_class"]))
    global_dice_pc = {c: np.mean([sc_a["global_dice_per_class"].get(c, 0),
                                   sc_b["global_dice_per_class"].get(c, 0)])
                      for c in all_classes}
    global_iou_pc  = {c: np.mean([sc_a["global_iou_per_class"].get(c, 0),
                                   sc_b["global_iou_per_class"].get(c, 0)])
                      for c in all_classes}

    # ── Per-image macro average ───────────────────────────────────────────────
    print("Per-image macro average (matches DiceScore / IoUScore):")
    rows1 = []
    for label, fn in [("MEAN", np.mean), ("STD", np.std),
                      ("MIN",  np.min),  ("MAX", np.max)]:
        rows1.append([label, f"{fn(agg['pa']):.4f}", f"{fn(agg['iou']):.4f}",
                      f"{fn(agg['dice']):.4f}", f"{fn(agg['bf1']):.4f}"])
    print(tabulate(rows1, headers=["", "Pixel Acc", "IoU", "Dice", "Boundary F1"],
                   tablefmt="rounded_outline"))

    # ── Global aggregate ─────────────────────────────────────────────────────
    print("\nGlobal aggregate (matches PatchwiseDiceScore / PatchwiseIoUScore):")
    print(tabulate([["ALL FRAMES", f"{global_iou:.4f}", f"{global_dice:.4f}"]],
                   headers=["", "Global IoU", "Global Dice"],
                   tablefmt="rounded_outline"))

    # ── Per-class global scores ───────────────────────────────────────────────
    print("\nPer-class global scores (foreground classes only):")
    class_rows = [[f"Class {c}",
                   f"{global_iou_pc[c]:.4f}",
                   f"{global_dice_pc[c]:.4f}"] for c in all_classes]
    print(tabulate(class_rows, headers=["Class", "Global IoU", "Global Dice"],
                   tablefmt="rounded_outline"))

    # ── Per-view breakdown ────────────────────────────────────────────────────
    print("\nPer-view global breakdown:")
    view_rows = [
        [f"View {views[0]}", f"{sc_a['global_iou']:.4f}", f"{sc_a['global_dice']:.4f}"],
        [f"View {views[1]}", f"{sc_b['global_iou']:.4f}", f"{sc_b['global_dice']:.4f}"],
    ]
    print(tabulate(view_rows, headers=["", "Global IoU", "Global Dice"],
                   tablefmt="rounded_outline"))


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OctreeNCA2DDualView segmentation metrics")
    parser.add_argument("--data-root", default=DATA_ROOT,   help="iOCT data root (contains peeling/, sri/)")
    parser.add_argument("--model",     default=MODEL_PATH,  help="Path to model.pth checkpoint")
    parser.add_argument("--datasets",  nargs="+", default=DATASETS, help="Dataset names (default: peeling sri)")
    parser.add_argument("--views",     nargs=2,   default=VIEWS,    help="Two view names (default: A B)")
    parser.add_argument("--output",    default=OUTPUT_CSV,           help="CSV output path")
    args = parser.parse_args()

    OUTPUT_CSV = args.output
    evaluate(args.data_root, args.datasets, args.views, args.model)