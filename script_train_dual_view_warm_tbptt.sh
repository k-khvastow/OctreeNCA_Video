#!/bin/bash

# Configuration for train_ioct2d_dual_view_warm_preprocessed_m1init.py

# ── VRAM-saving tweaks (16 GB GPU) ──────────────────────────────
# Reduce CUDA memory fragmentation; use expandable segments
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Use lighter torch.compile mode (max-autotune caches extra CUDA memory)
export IOCT_DUAL_WARM_TORCH_COMPILE_MODE="reduce-overhead"
export IOCT_DUAL_WARM_TORCH_COMPILE="1"
# Disable EMA (saves a full copy of model weights in VRAM)
export IOCT_DUAL_WARM_EMA="1"
# 

export IOCT_DUAL_WARM_SEQ_TBPTT_MODE="chunked"
# Use clipping instead of LayerNorm: LN shifts the distribution away from what
# M1's backbone was trained on, causing NaN at t=1.  Clipping bounds magnitudes
# without altering the distribution shape.
export IOCT_DUAL_WARM_HIDDEN_NORM="none"
export IOCT_DUAL_WARM_HIDDEN_CLIP="5.0"

# Learning rate override
export IOCT_DUAL_WARM_LR="1e-5"

# M1 checkpoint path
export IOCT_DUAL_WARM_M1_CHECKPOINT_PATH="/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_dual_daybed_24_Dual-view iOCT (A+B) OctreeNCA segmentation./models/epoch_39/model.pth"

export IOCT_M1_FREEZE="1"

# Toggle: 1 = disable backbone TBPTT for M1 first-frame pass (recommended), 0 = keep TBPTT active in M1
export IOCT_DUAL_WARM_M1_DISABLE_BACKBONE_TBPTT="0"

# Sequence settings  (reduced from 7 → 3 to fit 16 GB; increase if headroom allows)
export IOCT_DUAL_WARM_SEQUENCE_LENGTH="6"
export IOCT_DUAL_WARM_SEQUENCE_STEP="1"
export IOCT_DUAL_WARM_SEQ_TBPTT_STEPS="3"

# Curriculum schedule  (aligned with reduced sequence length)
export IOCT_DUAL_WARM_SEQ_LEN_MIN="3"
export IOCT_DUAL_WARM_SEQ_LEN_MAX="3"
export IOCT_DUAL_WARM_CURRICULUM_EPOCHS="20"

export IOCT_DUAL_WARM_TEMPORAL_CONSISTENCY_WEIGHT="0.1"
# Disable spectral norm: it interacts badly with fc1 zero-init (0/0 = NaN)
# and with checkpoint weights trained without it.
export IOCT_DUAL_WARM_SPECTRAL_NORM="0"

# Hidden state noise
export IOCT_DUAL_WARM_HIDDEN_NOISE_STD="0.001" 

# Initialization
export IOCT_DUAL_WARM_INIT_M2_FROM_M1="0"
# Identity init: copy M1 weights to M2 but zero the residual layer (fc1),
# so M2 starts as an identity mapping that preserves M1's hidden state.
# Requires INIT_M2_FROM_M1="1".  Set to "1" to enable.
export IOCT_DUAL_WARM_INIT_M2_IDENTITY="0"

# Number of NCA state channels (default: 24)
export IOCT_DUAL_WARM_CHANNEL_N="32"
# M1 checkpoint channel count (set when M1 was trained with fewer channels)
export IOCT_DUAL_WARM_M1_CHANNEL_N="24"

# Learned temporal gate on hidden channels ("none", "gru", or "simple")
export IOCT_DUAL_WARM_TEMPORAL_GATE="simple"

# Temporal vs. Spatial hidden channel split:
# Fraction of hidden channels carried across frames (temporal).
# The rest are reset to zero each frame (spatial).
# 1.0 = all temporal (default), 0.5 = half temporal / half spatial.
export IOCT_DUAL_WARM_TEMPORAL_RATIO="0.5"

# Run the training script
/home/ubuntu/miniforge/bin/conda run -p /vol/data/conda_envs/seg_v2 --no-capture-output python /vol/data/OctreeNCA_Video/train_ioct2d_dual_view_warm_preprocessed_m1init.py
