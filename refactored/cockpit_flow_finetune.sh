#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COCKPIT — iOCT Dual-View Flow Fine-Tune (pretrained M1 + flow M2)
# ═══════════════════════════════════════════════════════════════════════════
#
# Flow-augmented warm-start with a FROZEN pretrained M1.
# M1 initialises the recurrent state at t=0, then M2 (with optical-flow
# warp-then-refine) rolls forward.
#
# Usage:
#   chmod +x refactored/cockpit_flow_finetune.sh
#   ./refactored/cockpit_flow_finetune.sh
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Preset selection ─────────────────────────────────────────────────────
export EXP_PRESET="ioct_dual_flow"

# ── VRAM-saving tweaks (16 GB GPU) ──────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Model architecture ──────────────────────────────────────────────────
export MODEL_CHANNEL_N="32"
export MODEL_M1_CHANNEL_N="32"
export MODEL_TEMPORAL_GATE="gru"
export MODEL_TEMPORAL_RATIO="0.5"
export MODEL_SPECTRAL_NORM="0"
export MODEL_HIDDEN_NORM="layer"
export MODEL_HIDDEN_CLIP="5.0"
export WARM_START_STEPS="10"

# ── Octree forward steps per resolution level ───────────────────────────
export OCTREE_STEPS="10"
export OCTREE_COARSEST_STEPS="10"
export OCTREE_FINEST_MULTIPLIER="1"
export NUM_LEVELS="4"

# ── M1 checkpoint (pretrained, frozen) ──────────────────────────────────
export M1_CHECKPOINT="/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_dual_dispute_32_Dual-view iOCT (A+B) OctreeNCA segmentation./models/epoch_99/model.pth"
export M1_FREEZE="1"
export M1_EVAL_MODE="0"
export M1_DISABLE_BACKBONE_TBPTT="1"
export M1_NUM_LEVELS="4"
export M1_LOSS_ON_T0="0"

# ── M2 configuration ────────────────────────────────────────────────────
export M2_IDENTITY_INIT="0"
export M2_INIT_FROM_M1="0"
export M2_NUM_LEVELS="1"
export SHARE_M1_M2_BACKBONE="0"

# ── Flow settings ───────────────────────────────────────────────────────
export FLOW_ENABLED="1"
export FLOW_LOSS_WEIGHT="0.1"
export FLOW_SMOOTHNESS_WEIGHT="0.01"
export FLOW_SSIM_WEIGHT="0.0"
export FLOW_WARP_STATE="1"
export FLOW_CONDITION_GATE="0"

# ── Sequence / TBPTT ────────────────────────────────────────────────────
export SEQ_LENGTH="3"
export SEQ_STEP="10"
export TBPTT_MODE="chunked"
export TBPTT_STEPS="2"
export CURRICULUM_MIN="3"
export CURRICULUM_MAX="3"
export CURRICULUM_EPOCHS="40"
export TEMPORAL_CONSISTENCY_W="0"

# ── Contractive regularization ──────────────────────────────────────────
export CONTRACTIVE_W="0"

# ── Latent Slow Feature Analysis (SFA) ──────────────────────────────────
export LATENT_SFA_W="0"
export LATENT_SFA_DECORR_W="0"

# ── Hidden state noise ──────────────────────────────────────────────────
export HIDDEN_NOISE_STD="0.01"

# ── Training ────────────────────────────────────────────────────────────
export LR="1e-4"
export EMA="1"
export TORCH_COMPILE="0"
export TORCH_COMPILE_MODE="default"
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
