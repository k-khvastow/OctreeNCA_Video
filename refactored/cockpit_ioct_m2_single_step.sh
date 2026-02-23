#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT — iOCT Keyframe M1 + Single-Step M2
# ═══════════════════════════════════════════════════════════════════════════
#
# Model concept:
#   - M1 (pretrained, frozen) runs at 256×256 finest resolution.
#     It is called once per second (every 20 frames at 20 FPS).
#   - M2 (trained from scratch or from M1) runs at 512×512.
#     For each training step it is called ONCE on a randomly chosen
#     frame k ∈ [1, 20] frames after the M1 anchor frame, conditioned
#     on the upscaled (256→512) M1 hidden state.
#   - At inference: M1 at frames 0, 20, 40, …; M2 for the 19 in-between.
#
# Usage:
#   chmod +x refactored/cockpit_ioct_m2_single_step.sh
#   ./refactored/cockpit_ioct_m2_single_step.sh
#
# Or dry-run (print config, no training):
#   source refactored/cockpit_ioct_m2_single_step.sh && \
#     python refactored/train_m2_single_step.py --dry-run
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── VRAM-saving tweaks ──────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="0"
export EXP_NAME_SUFFIX="GPU${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── M1 architecture ─────────────────────────────────────────────────────
# M1 runs all 4 levels (512×512 → 256×256 → 128×128 → 64×64) so its
# finest-level segmentation is at 512×512 (useful for visualisation / loss).
# Hidden states for M2 warm-start are taken from level 1 (256×256) and
# bilinearly upscaled to 512×512.
# These must match the pretrained M1 checkpoint.
export MODEL_M1_CHANNEL_N="32"
export M1_LEVEL_OFFSET="0"          # include all levels → M1 finest = 512×512
export M1_NUM_LEVELS="4"            # M1 levels: 512×512 → 256×256 → 128×128 → 64×64
export M1_TRANSFER_LEVEL="1"        # transfer hidden from level 1 = 256×256 to M2

# ── M1 checkpoint ───────────────────────────────────────────────────────
# Paste the path to your pretrained M1 .pth file here.
# export M1_CHECKPOINT="/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_dual_daybed_24_Dual-view iOCT (A+B) OctreeNCA segmentation./models/epoch_39/model.pth"
export M1_CHECKPOINT="/home/khvastow/iOCT2D_dual_dispute_32_Dual-view iOCT (A+B) OctreeNCA segmentation./models/epoch_99/model.pth"
# export M1_CHECKPOINT=""
export M1_FREEZE="1"
export M1_EVAL_MODE="1"
export M1_DISABLE_BACKBONE_TBPTT="1"

# ── M2 architecture ─────────────────────────────────────────────────────
# M2 operates at 512×512 only (1 level).
export MODEL_CHANNEL_N="32"
export M2_NUM_LEVELS="1"            # M2: 512×512 only
export M2_KERNEL_SIZE="3"           # M2 depthwise conv kernel size
export M2_FC_BOTTLENECK="0"         # FC(frame_diff + hidden) → 4 channels before M2
export M2_EXTRA_HIDDEN="8"          # extra hidden channels for M2 (start empty), total = 32+8 = 40
export M2_HIDDEN_SIZE="64"         # FC hidden size inside NCA backbone (M1 default = 64)
export M2_INIT_FROM_M1="1"         # partial weight transfer (handles different channel_n)
export M2_TEMPORAL_EMB_DIM="0"     # learned k-embedding: embed(k) → linear → added to hidden
export M2_WARP_LOGITS="1"          # warp M1 logits toward target frame via Farneback optical flow
# M2 warm-start details (single scale, no multiscale warm-up)
export WARM_START_STEPS="7"
export MODEL_TEMPORAL_GATE="none"
export MODEL_TEMPORAL_RATIO="1"
export MODEL_HIDDEN_NORM="none"
export MODEL_HIDDEN_CLIP="0"
export MODEL_HIDDEN_NOISE_STD="0"
export MODEL_HIDDEN_NOISE_ANNEAL="50"
export MODEL_SPECTRAL_NORM="0"

# ── Single-step temporal knobs ──────────────────────────────────────────
# During training: target frame k ~ U[1, M2_MAX_STEP]
# During inference: M1 every M2_KEYFRAME_INTERVAL frames (= 1/s at 20 FPS)
export M2_MAX_STEP="10"
export M2_KEYFRAME_INTERVAL="10"

# ── Octree steps for M1 (applied to all M1 levels) ──────────────────────
export OCTREE_STEPS="15"
export OCTREE_COARSEST_STEPS="20"
export OCTREE_FINEST_MULTIPLIER="1"

# ── Frame-difference input for M2 ───────────────────────────────────────
# M2 gets an extra channel: intensity difference vs its warm-start state
# (i.e. current frame minus the M1 anchor frame down-then-upscaled).
# Gives M2 an explicit delta-image signal without hidden-state memory.
export FRAME_DIFF_INPUT="1"

# ── Training ────────────────────────────────────────────────────────────
export LR="1e-3"
export EMA="1"
export TORCH_COMPILE="0"
export TORCH_COMPILE_MODE="default"
export GRADIENT_CLIP="1.0"

# Use gradient accumulation to compensate for small effective batch size
# (1 sample × 1 step per batch_step).
export GRADIENT_ACCUMULATION="5"
export BATCH_SIZE="4"
# ── Data path ───────────────────────────────────────────────────────────────
export IOCT_DATA_ROOT="/home/khvastow/ioct_data"
# ── Resume ──────────────────────────────────────────────────────────────
export RESUME_NAME=""
export RESUME_MODEL_PATH=""
# export RESUME_NAME="M2SingleStep_iOCT2D_dual_knight_32"
# export RESUME_MODEL_PATH="<path>/octree_study_new/Experiments/M2SingleStep_iOCT2D_dual_knight_32_Keyframe M1 (256×256) + single-step M2 (512×512) iOCT segmentation"
# ── Tracking ────────────────────────────────────────────────────────────
export USE_WANDB="1"
export WANDB_PROJECT="OctreeNCA_Video"

# ── Loss weights & parameters ───────────────────────────────────────────
# Weights for the two primary objectives:
#   DICE_LOSS_WEIGHT   λ for dice loss  (default 1.0)
#   FOCAL_LOSS_WEIGHT  λ for FocalLoss  (default 1.0)
export DICE_LOSS_WEIGHT="1.0"
export FOCAL_LOSS_WEIGHT="1.0"

# Dice-loss variant:
#   DICE_TYPE   nnunet      → nnUNetSoftDiceLossSum  (sum of per-class soft Dice)
#               generalized → GeneralizedDiceLoss    (inverse-squared volume weights)
#               tversky     → TverskyLoss            (asymmetric FP/FN via α/β)
#                              α < β punishes FN more (recall-oriented)
#                              α = β = 0.5 recovers standard Dice
export DICE_TYPE="tversky"

# Dice-loss params (shared by both variants):
#   DICE_SMOOTH        Laplace smoothing ε to avoid division by zero
#                      Use 1.0 (nnUNet default) for stability with imbalanced classes.
#                      Too small (e.g. 1e-5) causes GDL to collapse near 1.0.
#   DICE_BATCH_DICE    compute dice over entire batch rather than per-sample (1/0)
#   DICE_DO_BG         include background class in dice computation (1/0)
export DICE_SMOOTH="1.0"
export DICE_BATCH_DICE="1"
export DICE_DO_BG="1"

# GeneralizedDiceLoss-only param:
#   DICE_WEIGHT_EPS    ε added to inverse-squared volume weights to prevent div/0
#   GDL_WEIGHT_TYPE    "v2" = 1/g² (original, aggressive for rare classes)
#                      "v1" = 1/g  (gentler, better for very thin/imbalanced classes)
#                      "uniform" = no class weighting (standard Dice)
#   GDL_MAX_WEIGHT     clamp per-class weights to this value (0 = no clamp).
#                      Prevents a single tiny class from dominating the loss.
#                      Good range: 100–1000.  Set 0 to disable.
export DICE_WEIGHT_EPS="5e-2"
export GDL_WEIGHT_TYPE="v1"
export GDL_MAX_WEIGHT="0"

# Focal-loss params:
#   FOCAL_GAMMA          focusing exponent γ (default 2.0; higher → harder examples)
#   FOCAL_IGNORE_INDEX   class index to ignore in focal loss (default 0 = background)
#   FOCAL_REDUCTION      loss reduction: mean | sum | none
export FOCAL_GAMMA="2.0"
export FOCAL_IGNORE_INDEX=""
export FOCAL_REDUCTION="mean"

# ── Boundary loss (optional) ─────────────────────────────────────────────
# Penalises confident predictions far from the true boundary using signed
# distance maps.  Requires precomputed EDT per frame (automatic when =1).
#   BOUNDARY_LOSS                  enable boundary loss (0 = off, 1 = on)
#   BOUNDARY_LOSS_WEIGHT           λ for BoundaryLoss (try 0.05–0.2)
#   BOUNDARY_DIST_CLIP             clamp signed distance to [-clip, clip]
#   BOUNDARY_DO_BG                 include background class in boundary loss (1/0)
#   BOUNDARY_USE_PROBABILITIES     use softmax probabilities instead of hard argmax (1/0)
#   BOUNDARY_COMPUTE_MISSING_DIST  recompute EDT on-the-fly if not precomputed (1/0)
export BOUNDARY_LOSS="1"
export BOUNDARY_LOSS_WEIGHT="0.1"
export BOUNDARY_DIST_CLIP="20.0"
export BOUNDARY_DO_BG="1"
export BOUNDARY_USE_PROBABILITIES="0"
export BOUNDARY_COMPUTE_MISSING_DIST="1"

# TverskyLoss-only params (used when DICE_TYPE=tversky):
#   TVERSKY_ALPHA          FP penalty weight (default 0.3)
#   TVERSKY_BETA           FN penalty weight (default 0.7; > α → recall-oriented)
#   TVERSKY_GAMMA          focal exponent on (1 - Tversky); 1.0 = standard
#   TVERSKY_SMOOTH         Laplace smoothing ε
#   TVERSKY_IGNORE_INDEX   class index to ignore (empty = none)
export TVERSKY_ALPHA="0.3"
export TVERSKY_BETA="0.7"
export TVERSKY_GAMMA="1.0"
export TVERSKY_SMOOTH="0.001"
export TVERSKY_IGNORE_INDEX=""

# ── Loss: t=0 supervision of M1 output (optional) ───────────────────────
# Set to 1 to also compute segmentation loss on the M1 output at the anchor
# frame, which helps keep M1's representations meaningful.
# (Only useful when M1 is NOT frozen; keep 0 when M1_FREEZE=1.)
export M1_LOSS_ON_T0="0"

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/opt/conda/bin/conda run -p /opt/conda/envs/seg_v2 --no-capture-output \
    python "${SCRIPT_DIR}/train_m2_single_step.py"
