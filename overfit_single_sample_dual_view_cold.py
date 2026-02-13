"""
Overfit a single iOCT dual-view (cold-start) sample to verify the model works.

Usage:
    python overfit_single_sample_dual_view_cold.py --iters 500 --lr 1e-3

This script:
  1. Builds the full model + loss + dataset from train_ioct2d_dual_view.py.
  2. Picks a single sample from the dataset.
  3. Runs a tight gradient-descent loop repeating only that sample.
  4. Prints the loss every iteration so you can confirm it goes down.
  5. Periodically prints per-class Dice on the same sample.
"""

import sys
import time
import argparse

import numpy as np
import torch

from train_ioct2d_dual_view import (
    get_study_config,
    get_dataset_args,
    iOCTPairedViewsDatasetForExperiment,
    OctreeNCA2DDualView,
    WeightedLosses,
)


def parse_args():
    p = argparse.ArgumentParser(description="Overfit one sample (dual-view cold-start)")
    p.add_argument("--iters", type=int, default=500, help="Number of gradient steps")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--sample-idx", type=int, default=0, help="Dataset index to overfit")
    p.add_argument("--print-every", type=int, default=5, help="Print loss every N iters")
    p.add_argument("--dice-every", type=int, default=25, help="Print Dice every N iters")
    p.add_argument("--amp", action="store_true", default=False, help="Use AMP (mixed precision)")
    p.add_argument("--no-compile", action="store_true", default=False, help="Disable torch.compile")
    return p.parse_args()


def compute_dice_per_class(logits_bhwc: torch.Tensor, target_bhwc: torch.Tensor, eps=1e-7):
    """
    logits_bhwc:  (B, H, W, C)  raw model output (BHWC)
    target_bhwc:  (B, H, W, C)  one-hot ground truth (BHWC)
    Returns dict {class_idx: dice_score} (skips background class 0).
    """
    probs = torch.softmax(logits_bhwc, dim=-1)
    num_classes = probs.shape[-1]
    dice = {}
    for c in range(1, num_classes):
        p = probs[..., c].flatten()
        t = target_bhwc[..., c].flatten()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice[c] = (2 * inter / (union + eps)).item()
    return dice


@torch.no_grad()
def evaluate_sample(model, img_a, img_b, lbl_a, lbl_b, num_classes):
    """Run forward on the single sample and print per-class Dice."""
    model.eval()

    out = model(img_a, img_b, y_a=lbl_a, y_b=lbl_b, batch_duplication=1)

    logits = out.get("logits", out.get("probabilities"))  # (2B, H, W, C)
    B = img_a.shape[0]
    logits_a, logits_b = logits[:B], logits[B:]

    # Build targets in BHWC to match model output format
    target_a = lbl_a.permute(0, 2, 3, 1)  # BCHW -> BHWC
    target_b = lbl_b.permute(0, 2, 3, 1)

    dice_a = compute_dice_per_class(logits_a, target_a)
    dice_b = compute_dice_per_class(logits_b, target_b)

    mean_a = np.mean(list(dice_a.values())) if dice_a else 0.0
    mean_b = np.mean(list(dice_b.values())) if dice_b else 0.0

    print(f"  View A Dice:  mean={mean_a:.4f}  per-class={{{', '.join(f'{c}:{v:.4f}' for c, v in dice_a.items())}}}")
    print(f"  View B Dice:  mean={mean_b:.4f}  per-class={{{', '.join(f'{c}:{v:.4f}' for c, v in dice_b.items())}}}")

    model.train()
    return mean_a, mean_b


def main():
    args = parse_args()

    # ---------- Config ----------
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    # Override settings for a clean overfit test
    study_config["experiment.use_wandb"] = False
    study_config["performance.compile"] = (not args.no_compile) and torch.cuda.is_available()
    study_config["trainer.use_amp"] = args.amp
    study_config["trainer.ema"] = False
    study_config["trainer.batch_size"] = 1
    study_config["trainer.gradient_accumulation"] = 1
    study_config["trainer.normalize_gradients"] = "none"

    # Only Dice + Focal loss
    study_config["trainer.losses"] = [
        "src.losses.DiceLoss.nnUNetSoftDiceLossSum",
        "src.losses.LossFunctions.FocalLoss",
    ]
    study_config["trainer.losses.parameters"] = [
        {"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05},
        {"gamma": 2.0, "alpha": None, "ignore_index": 0, "reduction": "mean"},
    ]
    study_config["trainer.loss_weights"] = [1.0, 1.0]
    study_config["experiment.dataset.precompute_boundary_dist"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study_config["experiment.device"] = str(device)
    num_classes = study_config["model.output_channels"]

    # ---------- Dataset ----------
    print("Building dataset ...")
    dataset_args["max_samples"] = args.sample_idx + 1
    dataset_args["precompute_boundary_dist"] = False
    ds = iOCTPairedViewsDatasetForExperiment(**dataset_args)

    if len(ds) == 0:
        print("ERROR: No pairs found. Check DATA_ROOT and that the dataset directories exist.")
        sys.exit(1)

    idx = min(args.sample_idx, len(ds) - 1)
    sample = ds[idx]
    print(f"Using sample {idx} / {len(ds)-1}  (id={sample.get('id', '?')})")
    print(f"  image_a shape: {sample['image_a'].shape}  (H, W, 1)")
    print(f"  label_a shape: {sample['label_a'].shape}  (H, W, C)")

    # Convert HWC -> BCHW for the model (agent's prepare_data does this)
    def to_bchw(arr):
        t = torch.from_numpy(arr).to(device)
        if t.ndim == 3:  # (H, W, C) -> (1, C, H, W)
            t = t.permute(2, 0, 1).unsqueeze(0)
        return t.float()

    img_a = to_bchw(sample["image_a"])
    img_b = to_bchw(sample["image_b"])
    lbl_a = to_bchw(sample["label_a"])
    lbl_b = to_bchw(sample["label_b"])

    print(f"  img_a tensor: {img_a.shape},  lbl_a tensor: {lbl_a.shape}")
    print(f"  num_classes={num_classes}")

    # ---------- Model ----------
    print("Building model ...")
    model = OctreeNCA2DDualView(study_config)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ---------- Loss ----------
    print("Building loss ...")
    loss_fn = WeightedLosses(study_config).to(device)

    # ---------- Optimizer ----------
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---------- Training loop ----------
    print(f"\n{'='*60}")
    print(f"Overfitting on sample {idx} for {args.iters} iterations  (lr={args.lr})")
    print(f"{'='*60}\n")

    best_loss = float("inf")

    for step in range(1, args.iters + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        t0 = time.time()

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            out = model(img_a, img_b, y_a=lbl_a, y_b=lbl_b, batch_duplication=1)

            # Build loss inputs the same way the agent does
            loss_kwargs = dict(out)
            loss_kwargs["target_unpatched"] = lbl_a  # agent uses view A label

            total_loss, loss_dict = loss_fn(**loss_kwargs)

        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0
        loss_val = total_loss.item()

        if loss_val < best_loss:
            best_loss = loss_val

        # ---------- Printing ----------
        if step % args.print_every == 0 or step == 1:
            parts = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if torch.is_tensor(v))
            print(f"[{step:4d}/{args.iters}]  loss={loss_val:.6f}  best={best_loss:.6f}  "
                  f"dt={dt:.2f}s  {parts}")

        if step % args.dice_every == 0 or step == args.iters:
            evaluate_sample(model, img_a, img_b, lbl_a, lbl_b, num_classes)

    print(f"\n{'='*60}")
    print(f"Final loss: {loss_val:.6f},  Best loss: {best_loss:.6f}")
    print(f"{'='*60}")

    if best_loss < 0.1:
        print("\n✓ Loss dropped well — model can fit a single sample. Pipeline looks correct.")
    elif best_loss < 0.5:
        print("\n~ Loss decreased but did not fully converge. Try more iterations or higher lr.")
    else:
        print("\n✗ Loss did not decrease significantly. Something may be wrong with the pipeline.")


if __name__ == "__main__":
    main()
