#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT — iOCT Dual-View Fine-Tune (warm-start with pretrained M1)
# ═══════════════════════════════════════════════════════════════════════════
#
# Equivalent to the old: script_train_dual_view_warm_tbptt.sh
#                       + train_ioct2d_dual_view_warm_preprocessed_m1init.py
#
# Usage:
#   chmod +x refactored/cockpit_ioct_finetune.sh
#   ./refactored/cockpit_ioct_finetune.sh
#
# Or source it and run train.py manually:
#   source refactored/cockpit_ioct_finetune.sh
#   python refactored/train.py --dry-run
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Preset selection ─────────────────────────────────────────────────────
export EXP_PRESET="ioct_dual_warm"

# ── VRAM-saving tweaks (16 GB GPU) ──────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Model architecture ──────────────────────────────────────────────────
export MODEL_CHANNEL_N="24"
export MODEL_M1_CHANNEL_N="24"
export MODEL_TEMPORAL_GATE="simple"
export MODEL_TEMPORAL_RATIO="0.5"
export MODEL_SPECTRAL_NORM="0.1"
export MODEL_HIDDEN_NORM="layer"
export MODEL_HIDDEN_CLIP="5.0"

# ── M1 checkpoint (pretrained) ──────────────────────────────────────────
export M1_CHECKPOINT="/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_dual_daybed_24_Dual-view iOCT (A+B) OctreeNCA segmentation./models/epoch_39/model.pth"
export M1_FREEZE="1"
export M1_DISABLE_BACKBONE_TBPTT="0"

# ── Sequence / TBPTT ────────────────────────────────────────────────────
export SEQ_LENGTH="6"
export SEQ_STEP="1"
export TBPTT_MODE="chunked"
export TBPTT_STEPS="3"
export CURRICULUM_MIN="6"
export CURRICULUM_MAX="6"
export CURRICULUM_EPOCHS="20"
export TEMPORAL_CONSISTENCY_W="0.1"

# ── Hidden state ────────────────────────────────────────────────────────
export HIDDEN_NOISE_STD="0.01"

# ── Training ────────────────────────────────────────────────────────────
export LR="1e-5"
export EMA="1"
export TORCH_COMPILE="1"
export TORCH_COMPILE_MODE="reduce-overhead"
export GRADIENT_CLIP="1.0"

# ── Tracking ────────────────────────────────────────────────────────────
export USE_WANDB="1"
export WANDB_PROJECT="OctreeNCA_Video"
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/home/ubuntu/miniforge/bin/conda run -p /vol/data/conda_envs/seg_v2 --no-capture-output \
    python "${SCRIPT_DIR}/train.py"
