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
export M1_CHECKPOINT="/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_dual_dispute_32_Dual-view iOCT (A+B) OctreeNCA segmentation./models/epoch_99/model.pth"
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
export MODEL_HIDDEN_NOISE_STD="0.05"
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
export LR="4e-4"
export EMA="1"
export TORCH_COMPILE="0"
export TORCH_COMPILE_MODE="default"
export GRADIENT_CLIP="1.0"

# Use gradient accumulation to compensate for small effective batch size
# (1 sample × 1 step per batch_step).
export GRADIENT_ACCUMULATION="5"
export BATCH_SIZE="2"

# ── Resume ──────────────────────────────────────────────────────────────
export RESUME_NAME=""
export RESUME_MODEL_PATH=""
# export RESUME_NAME="M2SingleStep_iOCT2D_dual_knight_32"
# export RESUME_MODEL_PATH="<path>/octree_study_new/Experiments/M2SingleStep_iOCT2D_dual_knight_32_Keyframe M1 (256×256) + single-step M2 (512×512) iOCT segmentation"
# ── Tracking ────────────────────────────────────────────────────────────
export USE_WANDB="1"
export WANDB_PROJECT="OctreeNCA_Video"

# ── Boundary loss (optional) ─────────────────────────────────────────────
# Penalises confident predictions far from the true boundary using signed
# distance maps.  Requires precomputed EDT per frame (automatic when =1).
export BOUNDARY_LOSS="0"
export BOUNDARY_LOSS_WEIGHT="0.1"     # λ for BoundaryLoss (try 0.05–0.2)
export BOUNDARY_DIST_CLIP="20.0"     # clamp signed distance to [-clip, clip]

# ── Loss: t=0 supervision of M1 output (optional) ───────────────────────
# Set to 1 to also compute segmentation loss on the M1 output at the anchor
# frame, which helps keep M1's representations meaningful.
# (Only useful when M1 is NOT frozen; keep 0 when M1_FREEZE=1.)
export M1_LOSS_ON_T0="0"

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/home/ubuntu/miniforge/bin/conda run -p /vol/data/conda_envs/seg_v2 --no-capture-output \
    python "${SCRIPT_DIR}/train_m2_single_step.py"
