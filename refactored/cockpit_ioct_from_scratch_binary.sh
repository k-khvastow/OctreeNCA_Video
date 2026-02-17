#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT — iOCT Dual-View Back-to-Back BINARY (single merged class)
# ═══════════════════════════════════════════════════════════════════════════
#
# All segmentation labels are merged into a single foreground class
# (binary: background vs. foreground). M1+M2 trained from scratch.
#
# Usage:
#   chmod +x refactored/cockpit_ioct_from_scratch_binary.sh
#   ./refactored/cockpit_ioct_from_scratch_binary.sh
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Preset selection ─────────────────────────────────────────────────────
export EXP_PRESET="ioct_dual_b2b_binary"

# ── VRAM-saving tweaks (16 GB GPU) ──────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Model architecture ──────────────────────────────────────────────────
export MODEL_CHANNEL_N="24"
export MODEL_TEMPORAL_GATE="simple"
export MODEL_TEMPORAL_RATIO="0.5"
export MODEL_SPECTRAL_NORM="0"
export MODEL_HIDDEN_NORM="none"
export MODEL_HIDDEN_CLIP="5.0"

# ── M1 / M2 relationship ────────────────────────────────────────────────
# No pretrained M1 — both train from scratch
# M1_CHECKPOINT intentionally unset (empty = random init)
export SHARE_M1_M2_BACKBONE="1"

# ── Binary segmentation ─────────────────────────────────────────────────
export MERGE_ALL_CLASSES="1"
export NUM_CLASSES="2"

# ── Sequence / TBPTT ────────────────────────────────────────────────────
export SEQ_LENGTH="3"
export SEQ_STEP="1"
export TBPTT_MODE="off"
export TBPTT_STEPS="3"
export CURRICULUM_MIN="3"
export CURRICULUM_MAX="3"
export CURRICULUM_EPOCHS="20"
export TEMPORAL_CONSISTENCY_W="0.1"

# ── Hidden state ────────────────────────────────────────────────────────
export HIDDEN_NOISE_STD="0.001"

# ── Training ────────────────────────────────────────────────────────────
export LR="2e-4"
export EMA="1"
export TORCH_COMPILE="1"
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
