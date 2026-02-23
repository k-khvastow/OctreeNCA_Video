#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT — iOCT Keyframe M1 (Pan / Single-View) + Single-Step M2
# ═══════════════════════════════════════════════════════════════════════════
#
# Model concept:
#   - M1 (pretrained PAN model, frozen) runs at 256×256 finest resolution.
#     It is called once per second (every 20 frames at 20 FPS).
#   - M2 (trained from scratch or from M1) runs at 512×512.
#     For each training step it is called ONCE on a randomly chosen
#     frame k ∈ [1, 20] frames after the M1 anchor frame, conditioned
#     on the upscaled (256→512) M1 hidden state.
#   - At inference: M1 at frames 0, 20, 40, …; M2 for the 19 in-between.
#
# Key difference from the dual-view cockpit:
#   - M1 is a single-view (pan) checkpoint (no cross-fusion / FiLM).
#   - Uses separate_models=True (matching the pan M1 architecture).
#   - Dataset loads each view (A, B) independently — no paired A+B frames.
#
# Usage:
#   chmod +x refactored/cockpit_ioct_m2_single_step_pan.sh
#   ./refactored/cockpit_ioct_m2_single_step_pan.sh
#
# Or dry-run (print config, no training):
#   source refactored/cockpit_ioct_m2_single_step_pan.sh && \
#     python refactored/train_m2_single_step_pan.py --dry-run
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── VRAM-saving tweaks ──────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="1"
export EXP_NAME_SUFFIX="GPU${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── M1 architecture (pan / single-view) ─────────────────────────────────
# Pan M1 uses separate_models=True (one backbone per level).
# cross_fusion=none means no FiLM layers → state dict matches the non-dual
# pan checkpoint exactly.
export MODEL_M1_CHANNEL_N="24"
export M1_LEVEL_OFFSET="0"          # include all levels → M1 finest = 512×512
export M1_NUM_LEVELS="5"            # M1 levels: 512×512 → 256×256 → 128×128 → 64×64
export M1_TRANSFER_LEVEL="1"        # transfer hidden from level 1 = 256×256 to M2
export SEPARATE_MODELS="1"          # pan M1 uses separate backbones per level

# ── M1 checkpoint (pan) ─────────────────────────────────────────────────
export M1_CHECKPOINT="/home/khvastow/iOCT2D_pan_24_Training OctreeNCA on iOCT 2D frames./models/epoch_99/model.pth"
export M1_FREEZE="1"
export M1_EVAL_MODE="1"
export M1_DISABLE_BACKBONE_TBPTT="1"

# ── M2 architecture ─────────────────────────────────────────────────────
# M2 operates at 512×512 only (1 level).
export MODEL_CHANNEL_N="24"
export M2_NUM_LEVELS="1"            # M2: 512×512 only
export M2_KERNEL_SIZE="3"           # M2 depthwise conv kernel size
export M2_FC_BOTTLENECK="0"         # FC(frame_diff + hidden) → 4 channels before M2
export M2_EXTRA_HIDDEN="8"          # extra hidden channels for M2 (start empty), total = 24+8 = 32
export M2_HIDDEN_SIZE="64"          # FC hidden size inside NCA backbone (pan M1 default = 64)
export M2_INIT_FROM_M1="1"          # partial weight transfer (handles different channel_n)
export M2_TEMPORAL_EMB_DIM="0"      # learned k-embedding: embed(k) → linear → added to hidden
export M2_WARP_LOGITS="0"           # warp M1 logits toward target frame via Farneback optical flow
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
export OCTREE_STEPS="10"
export OCTREE_COARSEST_STEPS="20"
export OCTREE_FINEST_MULTIPLIER="1"

# ── Frame-difference input for M2 ───────────────────────────────────────
export FRAME_DIFF_INPUT="1"

# ── Training ────────────────────────────────────────────────────────────
export LR="1e-3"
export EMA="1"
export TORCH_COMPILE="0"
export TORCH_COMPILE_MODE="default"
export GRADIENT_CLIP="1.0"

# Use gradient accumulation to compensate for small effective batch size
export GRADIENT_ACCUMULATION="5"
export BATCH_SIZE="4"
# ── Data path ───────────────────────────────────────────────────────────
export IOCT_DATA_ROOT="/home/khvastow/ioct_data"
# ── Resume ──────────────────────────────────────────────────────────────
export RESUME_NAME=""
export RESUME_MODEL_PATH=""
# ── Tracking ────────────────────────────────────────────────────────────
export USE_WANDB="1"
export WANDB_PROJECT="OctreeNCA_Video"

# ── Loss weights & parameters ───────────────────────────────────────────
export DICE_LOSS_WEIGHT="1.0"
export FOCAL_LOSS_WEIGHT="1.0"

export DICE_TYPE="nnunet"

export DICE_SMOOTH="1.0"
export DICE_BATCH_DICE="1"
export DICE_DO_BG="1"

export DICE_WEIGHT_EPS="5e-2"
export GDL_WEIGHT_TYPE="v1"
export GDL_MAX_WEIGHT="0"

export FOCAL_GAMMA="2.0"
export FOCAL_IGNORE_INDEX=""
export FOCAL_REDUCTION="mean"

# ── Boundary loss (optional) ─────────────────────────────────────────────
export BOUNDARY_LOSS="0"
export BOUNDARY_LOSS_WEIGHT="0.1"
export BOUNDARY_DIST_CLIP="20.0"
export BOUNDARY_DO_BG="1"
export BOUNDARY_USE_PROBABILITIES="0"
export BOUNDARY_COMPUTE_MISSING_DIST="1"

# TverskyLoss-only params: 
export TVERSKY_ALPHA="0.3"
export TVERSKY_BETA="0.7"
export TVERSKY_GAMMA="1.0"
export TVERSKY_SMOOTH="0.001"
export TVERSKY_IGNORE_INDEX=""

# ── Loss: t=0 supervision of M1 output (optional) ───────────────────────
export M1_LOSS_ON_T0="0"

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/opt/conda/bin/conda run -p /opt/conda/envs/seg_v2 --no-capture-output \
    python "${SCRIPT_DIR}/train_m2_single_step_pan.py"
