#!/bin/bash

# Back-to-back dual-view training WITHOUT pretrained M1 weights.
# Both M1 and M2 are randomly initialised and trained jointly.

# ── VRAM-saving tweaks (16 GB GPU) ──────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IOCT_DUAL_B2B_TORCH_COMPILE_MODE="reduce-overhead"
export IOCT_DUAL_B2B_TORCH_COMPILE="1"

# EMA (keep on by default; set to "0" if VRAM is tight)
export IOCT_DUAL_B2B_EMA="1"

# Sequence-level TBPTT mode ("off" = full backprop, no truncation)
export IOCT_DUAL_B2B_SEQ_TBPTT_MODE="off"

# Hidden-state stabilisation: use clipping, no LayerNorm
export IOCT_DUAL_B2B_HIDDEN_NORM="none"
export IOCT_DUAL_B2B_HIDDEN_CLIP="5.0"

# ── Learning rate ────────────────────────────────────────────────
# Higher than the fine-tune script (2e-4 vs 1e-5) since we train from scratch.
export IOCT_DUAL_B2B_LR="2e-4"

# ── M1: no pretrained checkpoint ────────────────────────────────
# Leave empty → M1 is random-init and trains jointly with M2.
export IOCT_DUAL_B2B_M1_CHECKPOINT_PATH=""

# No TBPTT anywhere — full backprop through M1 backbone too.
export IOCT_DUAL_B2B_M1_DISABLE_BACKBONE_TBPTT="0"

# ── Sequence settings ───────────────────────────────────────────
export IOCT_DUAL_B2B_SEQUENCE_LENGTH="3"
export IOCT_DUAL_B2B_SEQUENCE_STEP="1"
export IOCT_DUAL_B2B_SEQ_TBPTT_STEPS="3"

# Curriculum schedule
export IOCT_DUAL_B2B_SEQ_LEN_MIN="3"
export IOCT_DUAL_B2B_SEQ_LEN_MAX="3"
export IOCT_DUAL_B2B_CURRICULUM_EPOCHS="20"

# Temporal consistency
export IOCT_DUAL_B2B_TEMPORAL_CONSISTENCY_WEIGHT="0.1"

# Disable spectral norm (interacts badly with zero-init fc1)
export IOCT_DUAL_B2B_SPECTRAL_NORM="0"

# Hidden-state noise injection
export IOCT_DUAL_B2B_HIDDEN_NOISE_STD="0.001"

# ── M2 initialisation ───────────────────────────────────────────
# Share backbone weights between M1 and M2 (same module objects, ~halves
# backbone VRAM).  Incompatible with INIT_M2_FROM_M1 / INIT_M2_IDENTITY.
export IOCT_DUAL_B2B_SHARE_M1_M2_BACKBONE="1"
export IOCT_DUAL_B2B_INIT_M2_FROM_M1="0"
export IOCT_DUAL_B2B_INIT_M2_IDENTITY="0"

# ── Channel count (same for M1 and M2) ──────────────────────────
export IOCT_DUAL_B2B_CHANNEL_N="24"

# Learned temporal gate
export IOCT_DUAL_B2B_TEMPORAL_GATE="simple"

# Temporal / spatial hidden channel split
export IOCT_DUAL_B2B_TEMPORAL_RATIO="0.5"

# Run
/home/ubuntu/miniforge/bin/conda run -p /vol/data/conda_envs/seg_v2 --no-capture-output python /vol/data/OctreeNCA_Video/train_ioct2d_dual_view_warm_no_m1_pretrain.py
