#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT — iOCT Dual-View (simple, shared weights, no warm-start)
# ═══════════════════════════════════════════════════════════════════════════
#
# Equivalent to the old: train_ioct2d_dual_view.py
#
# Single-frame paired-view (A+B) OctreeNCA segmentation with a single
# shared NCA backbone across all octree coarseness levels.
#
# Usage:
#   chmod +x refactored/cockpit_ioct_dual_view.sh
#   ./refactored/cockpit_ioct_dual_view.sh
#
# Or source it and run train.py manually:
#   source refactored/cockpit_ioct_dual_view.sh
#   python refactored/train.py --dry-run
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Preset selection ─────────────────────────────────────────────────────
export EXP_PRESET="ioct_dual"

# ── VRAM-saving tweaks ──────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Model architecture ──────────────────────────────────────────────────
export MODEL_CHANNEL_N="24"
export MODEL_HIDDEN_SIZE="64"
export BACKBONE_CLASS="BasicNCA2DFast"
# Shared weights for all coarseness levels (single backbone)
export SEPARATE_MODELS="0"

# ── Octree resolution levels and forward steps ──────────────────────────
# 3 levels: 512×512 → 256×256 → 128×128
# Steps given as "min,max" are sampled uniformly during training; eval uses max.
export NUM_LEVELS="3"
export OCTREE_STEPS="10,20"
export OCTREE_COARSEST_STEPS="5,15"
export OCTREE_FINEST_MULTIPLIER="1"

# ── Training ────────────────────────────────────────────────────────────
export LR="1e-4"
export BATCH_SIZE="2"
export USE_AMP="1"
export EMA="1"
export EMA_DECAY="0.99"
export GRADIENT_CLIP="1.0"
export TORCH_COMPILE="0"
export TORCH_COMPILE_MODE="reduce-overhead"

# ── Tracking ────────────────────────────────────────────────────────────
export USE_WANDB="1"
export WANDB_PROJECT="OctreeNCA_Video"

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/home/ubuntu/miniforge/bin/conda run -p /vol/data/conda_envs/seg_v2 --no-capture-output \
    python "${SCRIPT_DIR}/train.py"
