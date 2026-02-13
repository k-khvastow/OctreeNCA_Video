"""
Overfit a single iOCT dual-view sample to verify the model pipeline works.

Usage:
    IOCT_DUAL_WARM_M1_CHECKPOINT_PATH=/path/to/m1.pth python overfit_single_sample_dual_view.py

Optional env vars (same as train_ioct2d_dual_view_warm_preprocessed_m1init.py):
    IOCT_DUAL_WARM_SEQUENCE_LENGTH  (default: 3)
    IOCT_DUAL_WARM_SEQUENCE_STEP    (default: 1)
    ... and all IOCT_DUAL_WARM_* / IOCT_WARM_* variables.

This script:
  1. Builds the full model + loss + dataset as in training.
  2. Picks a single sample from the dataset.
  3. Runs a tight loop repeating only that sample for N iterations.
  4. Prints the loss every iteration so you can confirm it goes down.
  5. Periodically prints per-class Dice on the same sample.
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Re-use everything from the real training script
# ---------------------------------------------------------------------------
from train_ioct2d_dual_view_warm_preprocessed_m1init import (
    get_study_config,
    get_dataset_args,
    iOCTPairedSequentialDatasetForExperiment,
    OctreeNCA2DDualViewWarmStartM1Init,
    WeightedLosses,
    DATA_ROOT,
    DATASETS,
    VIEWS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Overfit one sample (dual-view warm-start M1-init)")
    p.add_argument("--iters", type=int, default=500, help="Number of gradient steps")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--sample-idx", type=int, default=0, help="Dataset index to overfit")
    p.add_argument("--print-every", type=int, default=5, help="Print loss every N iters")
    p.add_argument("--dice-every", type=int, default=25, help="Print Dice every N iters")
    p.add_argument("--amp", action="store_true", default=False, help="Use AMP (mixed precision)")
    p.add_argument("--no-compile", action="store_true", default=False, help="Disable torch.compile")
    return p.parse_args()


def compute_dice_per_class(logits: torch.Tensor, target_onehot: torch.Tensor, eps=1e-7):
    """
    logits:       (B, C, H, W) raw model output
    target_onehot: (B, C, H, W) one-hot ground truth
    Returns dict {class_idx: dice_score} (skips background class 0).
    """
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    dice = {}
    for c in range(1, num_classes):
        p = probs[:, c].flatten()
        t = target_onehot[:, c].flatten()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice[c] = (2 * inter / (union + eps)).item()
    return dice


@torch.no_grad()
def evaluate_sample(model, sample, device, num_classes):
    """Run forward on the single sample and print per-class Dice."""
    model.eval()

    img_a = torch.from_numpy(sample["image_a"]).unsqueeze(0).to(device)  # (1,T,1,H,W)
    img_b = torch.from_numpy(sample["image_b"]).unsqueeze(0).to(device)
    lbl_a = torch.from_numpy(sample["label_a"]).unsqueeze(0).to(device)
    lbl_b = torch.from_numpy(sample["label_b"]).unsqueeze(0).to(device)

    T = img_a.shape[1]

    # --- M1 init (t=0) ---
    m1_out, (prev_state_a, prev_state_b) = model.m1_forward_and_init_states(
        img_a[:, 0], img_b[:, 0],
        y_a=lbl_a[:, 0], y_b=lbl_b[:, 0],
    )

    all_dice_a, all_dice_b = [], []

    start_t = 0 if model.config.get("model.m1.use_t0_for_loss", False) else 1
    for t in range(start_t, T):
        out = model(
            img_a[:, t], img_b[:, t],
            y_a=lbl_a[:, t], y_b=lbl_b[:, t],
            prev_state_a=prev_state_a,
            prev_state_b=prev_state_b,
        )
        prev_state_a = out.get("final_state_a")
        prev_state_b = out.get("final_state_b")

        logits = out.get("logits", out.get("probabilities"))
        # logits is (2B, H, W, C) in BHWC format — permute to BCHW
        logits = logits.permute(0, 3, 1, 2)
        B = img_a.shape[0]
        logits_a, logits_b = logits[:B], logits[B:]

        dice_a = compute_dice_per_class(logits_a, lbl_a[:, t])
        dice_b = compute_dice_per_class(logits_b, lbl_b[:, t])
        all_dice_a.append(dice_a)
        all_dice_b.append(dice_b)

    # Average over time
    classes = sorted(all_dice_a[0].keys()) if all_dice_a else []
    avg_a = {c: np.mean([d[c] for d in all_dice_a]) for c in classes}
    avg_b = {c: np.mean([d[c] for d in all_dice_b]) for c in classes}
    mean_a = np.mean(list(avg_a.values())) if avg_a else 0.0
    mean_b = np.mean(list(avg_b.values())) if avg_b else 0.0

    print(f"  View A Dice:  mean={mean_a:.4f}  per-class={{{', '.join(f'{c}:{v:.4f}' for c,v in avg_a.items())}}}")
    print(f"  View B Dice:  mean={mean_b:.4f}  per-class={{{', '.join(f'{c}:{v:.4f}' for c,v in avg_b.items())}}}")

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
    # Only load 1 sample for fast overfitting
    dataset_args["max_samples"] = args.sample_idx + 1
    dataset_args["precompute_boundary_dist"] = False
    ds = iOCTPairedSequentialDatasetForExperiment(**dataset_args)

    if len(ds) == 0:
        print("ERROR: No sequences found. Check DATA_ROOT and that the dataset directories exist.")
        sys.exit(1)

    idx = min(args.sample_idx, len(ds) - 1)
    sample = ds[idx]
    print(f"Using sample {idx} / {len(ds)-1}  (id={sample.get('id', '?')})")
    print(f"  image_a shape: {sample['image_a'].shape}")
    print(f"  label_a shape: {sample['label_a'].shape}")
    print(f"  image_b shape: {sample['image_b'].shape}")
    print(f"  label_b shape: {sample['label_b'].shape}")

    # Batch dimension
    img_a = torch.from_numpy(sample["image_a"]).unsqueeze(0).to(device)  # (1,T,1,H,W)
    img_b = torch.from_numpy(sample["image_b"]).unsqueeze(0).to(device)
    lbl_a = torch.from_numpy(sample["label_a"]).unsqueeze(0).to(device)
    lbl_b = torch.from_numpy(sample["label_b"]).unsqueeze(0).to(device)
    dist_a = (
        torch.from_numpy(sample["label_dist_a"]).unsqueeze(0).to(device)
        if "label_dist_a" in sample
        else None
    )
    dist_b = (
        torch.from_numpy(sample["label_dist_b"]).unsqueeze(0).to(device)
        if "label_dist_b" in sample
        else None
    )

    T = img_a.shape[1]
    print(f"  Sequence length T={T}, num_classes={num_classes}")

    # ---------- Model ----------
    print("Building model ...")
    model = OctreeNCA2DDualViewWarmStartM1Init(study_config)
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

    start_t_model = 0 if study_config.get("model.m1.use_t0_for_loss", False) else 1
    best_loss = float("inf")

    for step in range(1, args.iters + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        t0 = time.time()

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            # --- M1 init (t=0) ---
            m1_out, (prev_state_a, prev_state_b) = model.m1_forward_and_init_states(
                img_a[:, 0], img_b[:, 0],
                y_a=lbl_a[:, 0], y_b=lbl_b[:, 0],
            )

            total_loss = torch.tensor(0.0, device=device)
            n_supervised = 0

            for t in range(start_t_model, T):
                out = model(
                    img_a[:, t], img_b[:, t],
                    y_a=lbl_a[:, t], y_b=lbl_b[:, t],
                    prev_state_a=prev_state_a,
                    prev_state_b=prev_state_b,
                )
                prev_state_a = out.get("final_state_a")
                prev_state_b = out.get("final_state_b")

                # Build loss inputs — mimic what the agent does:
                # pass the full model output dict and add target_unpatched + target_dist
                loss_kwargs = dict(out)
                loss_kwargs["target_unpatched"] = torch.cat([lbl_a[:, t], lbl_b[:, t]], dim=0)

                # Distance maps for boundary loss
                if dist_a is not None and dist_b is not None:
                    loss_kwargs["target_dist"] = torch.cat([dist_a[:, t], dist_b[:, t]], dim=0)

                frame_loss, loss_dict = loss_fn(**loss_kwargs)
                total_loss = total_loss + frame_loss
                n_supervised += 1

            if n_supervised > 0:
                total_loss = total_loss / n_supervised

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
            # Breakdown
            parts = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if torch.is_tensor(v))
            print(f"[{step:4d}/{args.iters}]  loss={loss_val:.6f}  best={best_loss:.6f}  "
                  f"dt={dt:.2f}s  {parts}")

        if step % args.dice_every == 0 or step == args.iters:
            evaluate_sample(model, sample, device, num_classes)

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
