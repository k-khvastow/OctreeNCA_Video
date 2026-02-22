#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT — iOCT Dual-View Flow (warp-then-refine, M1+M2 from scratch)
# ═══════════════════════════════════════════════════════════════════════════
#
# Flow-augmented OctreeNCA: M2 predicts per-pixel optical flow from its
# hidden state, warps the previous frame's state to align with the current
# frame, then refines with NCA steps.  A self-supervised photometric loss
# (L1 + optional SSIM + smoothness) trains the flow head without GT flow.
#
# Usage:
#   chmod +x refactored/cockpit_flow.sh
#   ./refactored/cockpit_flow.sh
#
# Or source it and run train.py manually:
#   source refactored/cockpit_flow.sh
#   python refactored/train.py --dry-run
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Preset selection ─────────────────────────────────────────────────────
export EXP_PRESET="ioct_dual_flow"

# ── VRAM-saving tweaks (16 GB GPU) ──────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Model architecture ──────────────────────────────────────────────────
export NUM_LEVELS="4"
export MODEL_CHANNEL_N="24"
export MODEL_TEMPORAL_GATE="simple"
export MODEL_TEMPORAL_RATIO="0.5"
export MODEL_SPECTRAL_NORM="0"
export MODEL_HIDDEN_NORM="none"
export MODEL_HIDDEN_CLIP="5.0"
export WARM_START_STEPS="5"

# ── Octree forward steps per resolution level ───────────────────────────
export OCTREE_STEPS="5,10"
export OCTREE_COARSEST_STEPS="10,16"
export OCTREE_FINEST_MULTIPLIER="2"

# ── M1 / M2 relationship ────────────────────────────────────────────────
# Both train from scratch, shared backbone
export SHARE_M1_M2_BACKBONE="1"

# ── Flow settings ───────────────────────────────────────────────────────
# Enable the flow head in M2 (warp previous state before NCA refinement)
export FLOW_ENABLED="1"
# Weight of the self-supervised photometric flow loss (relative to seg loss)
export FLOW_LOSS_WEIGHT="0.1"
# Spatial smoothness regularization on the flow field
export FLOW_SMOOTHNESS_WEIGHT="0.01"
# SSIM component of photometric loss (0 = disabled, try 0.85 to enable)
export FLOW_SSIM_WEIGHT="0.0"
# Warp the full previous state (not just logits) using predicted flow
export FLOW_WARP_STATE="1"
# Condition the temporal gate on flow magnitude (adds 1 channel to gate input)
export FLOW_CONDITION_GATE="0"

# ── Sequence / TBPTT ────────────────────────────────────────────────────
export SEQ_LENGTH="3"
export SEQ_STEP="1"
export TBPTT_MODE="off"
export TBPTT_STEPS="3"
export CURRICULUM_MIN="3"
export CURRICULUM_MAX="3"
export CURRICULUM_EPOCHS="20"
export TEMPORAL_CONSISTENCY_W="0.1"

# ── Contractive regularization ──────────────────────────────────────────
export CONTRACTIVE_W="0"

# ── Hidden state noise ──────────────────────────────────────────────────
export HIDDEN_NOISE_STD="0.001"

# ── Training ────────────────────────────────────────────────────────────
export LR="2e-4"
export EMA="1"
export TORCH_COMPILE="0"
export TORCH_COMPILE_MODE="reduce-overhead"
export GRADIENT_CLIP="1.0"

# ── Tracking ────────────────────────────────────────────────────────────
export USE_WANDB="1"
export WANDB_PROJECT="OctreeNCA_Video"

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/home/ubuntu/miniforge/bin/conda run -p /vol/data/conda_envs/seg_v2 --no-capture-output \
    python "${SCRIPT_DIR}/train.py"
