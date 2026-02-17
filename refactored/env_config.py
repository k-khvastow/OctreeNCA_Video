"""
env_config.py — Unified environment-variable-to-config mapping.

Reads flat env vars (set in cockpit.sh) and maps them into the
dot-delimited study_config dict used by the framework.

Design:
  - One canonical flat namespace: MODEL_*, SEQ_*, TRAIN_*, etc.
  - No 3-tier fallback chains.
  - Each env var has a well-defined type and config key target.
  - Empty string / unset means "use preset default" (not overridden).
"""

from __future__ import annotations

import os
from typing import Any


# ───────────────────────────────────────────────────────────────────────────
# Type casters
# ───────────────────────────────────────────────────────────────────────────
def _bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes")


def _optional_float(val: str) -> float | None:
    val = val.strip()
    return float(val) if val else None


def _optional_int(val: str) -> int | None:
    val = val.strip()
    return int(val) if val else None


def _int_or_csv(val: str) -> int | list[int]:
    """Parse '3' as int or '2,4,6' as list[int]."""
    val = val.strip()
    if "," in val:
        return [int(x) for x in val.split(",") if x.strip()]
    return int(val)


# ───────────────────────────────────────────────────────────────────────────
# Env var registry
#
# Format:  ENV_NAME  →  (config_key, caster)
#
# Only env vars that are *set and non-empty* will be written into the
# config dict.  Everything else stays at whatever the preset provides.
# ───────────────────────────────────────────────────────────────────────────
ENV_MAP: dict[str, tuple[str, Any]] = {
    # ── Experiment identity ──────────────────────────────────────────────
    "EXP_NAME":                     ("experiment.name", str),
    "EXP_DESCRIPTION":              ("experiment.description", str),
    "WANDB_PROJECT":                ("experiment.wandb_project", str),
    "WANDB_ENTITY":                 ("experiment.wandb_entity", str),
    "USE_WANDB":                    ("experiment.use_wandb", _bool),

    # ── Model architecture ───────────────────────────────────────────────
    "MODEL_CHANNEL_N":              ("model.channel_n", int),
    "MODEL_HIDDEN_SIZE":            ("model.hidden_size", int),
    "MODEL_M1_CHANNEL_N":           ("model.m1.channel_n", int),
    "MODEL_SPECTRAL_NORM":          ("model.spectral_norm", _bool),
    "MODEL_HIDDEN_NORM":            ("model.octree.warm_start_hidden_norm", str),
    "MODEL_HIDDEN_CLIP":            ("model.octree.warm_start_hidden_clip", float),
    "MODEL_HIDDEN_TANH_SCALE":      ("model.octree.warm_start_hidden_tanh_scale", float),
    "MODEL_HIDDEN_GN_GROUPS":       ("model.octree.warm_start_hidden_gn_groups", int),
    "MODEL_TEMPORAL_GATE":          ("model.octree.warm_start_temporal_gate", str),
    "MODEL_TEMPORAL_RATIO":         ("model.octree.warm_start_temporal_ratio", float),
    "MODEL_LOGITS_MODE":            ("model.octree.warm_start_logits_mode", str),
    "MODEL_LOGITS_GATE_FROM":       ("model.octree.warm_start_logits_gate_from", str),

    # ── M1 / M2 relationship (dual-view only) ───────────────────────────
    "M1_CHECKPOINT":                ("model.m1.pretrained_path", str),
    "M1_FREEZE":                    ("model.m1.freeze", _bool),
    "M1_LOSS_ON_T0":                ("model.m1.use_t0_for_loss", _bool),
    "M1_DISABLE_BACKBONE_TBPTT":    ("model.m1.disable_backbone_tbptt", _bool),
    "M2_INIT_FROM_M1":              ("model.m2.init_from_m1", _bool),
    "M2_IDENTITY_INIT":             ("model.m2.init_identity", _bool),
    "SHARE_M1_M2_BACKBONE":         ("model.m2.share_backbone_with_m1", _bool),

    # ── Sequence / temporal ──────────────────────────────────────────────
    "SEQ_LENGTH":                   ("_seq.length", int),
    "SEQ_STEP":                     ("_seq.step", int),
    "TBPTT_MODE":                   ("model.sequence.tbptt_mode", str),
    "TBPTT_STEPS":                  ("model.sequence.tbptt_steps", int),
    "BACKBONE_TBPTT_STEPS":         ("model.backbone.tbptt_steps", int),
    "CURRICULUM_MIN":               ("trainer.curriculum.seq_len_min", int),
    "CURRICULUM_MAX":               ("trainer.curriculum.seq_len_max", int),
    "CURRICULUM_EPOCHS":            ("trainer.curriculum.warmup_epochs", int),
    "TEMPORAL_CONSISTENCY_W":       ("trainer.temporal_consistency_weight", float),
    "HIDDEN_NOISE_STD":             ("model.octree.warm_start_hidden_noise_std", float),
    "HIDDEN_NOISE_ANNEAL":          ("model.octree.warm_start_hidden_noise_anneal_epochs", int),

    # ── Multiscale warm-start ────────────────────────────────────────────
    "MULTISCALE":                   ("model.octree.warm_start_multiscale", _bool),
    "MULTISCALE_START_LEVEL":       ("model.octree.warm_start_multiscale_start_level", int),
    "MULTISCALE_STEPS":             ("model.octree.warm_start_multiscale_steps", _int_or_csv),
    "MULTISCALE_DOWNSAMPLE_MODE":   ("model.octree.warm_start_multiscale_downsample_mode", str),

    # ── Training ─────────────────────────────────────────────────────────
    "LR":                           ("trainer.optimizer.lr", float),
    "LR_SCALE":                     ("_train.lr_scale", float),
    "BATCH_SIZE":                   ("trainer.batch_size", int),
    "GRADIENT_ACCUMULATION":        ("trainer.gradient_accumulation", int),
    "N_EPOCHS":                     ("trainer.n_epochs", int),
    "EMA":                          ("trainer.ema", _bool),
    "EMA_DECAY":                    ("trainer.ema.decay", float),
    "USE_AMP":                      ("trainer.use_amp", _bool),
    "GRADIENT_CLIP":                ("trainer.gradient_clip_val", float),

    # ── Torch compile ────────────────────────────────────────────────────
    "TORCH_COMPILE":                ("performance.compile", _bool),
    "TORCH_COMPILE_MODE":           ("performance.compile.mode", str),
    "TORCH_COMPILE_BACKEND":        ("performance.compile.backend", str),
    "TORCH_COMPILE_DYNAMIC":        ("performance.compile.dynamic", _bool),
    "TORCH_COMPILE_FULLGRAPH":      ("performance.compile.fullgraph", _bool),

    # ── Dataset ───────────────────────────────────────────────────────────
    "MERGE_ALL_CLASSES":            ("experiment.dataset.merge_all_classes", _bool),
    "NUM_CLASSES":                  ("model.output_channels", int),

    # ── Logging ──────────────────────────────────────────────────────────
    "TRACK_GRAD_NORM":              ("experiment.logging.track_gradient_norm", _bool),
    "SAVE_INTERVAL":                ("experiment.save_interval", int),
    "EVAL_INTERVAL":                ("experiment.logging.evaluate_interval", int),

    # ── Resume ───────────────────────────────────────────────────────────
    "RESUME_NAME":                  ("_resume.name", str),
    "RESUME_MODEL_PATH":            ("_resume.model_path", str),
}


def load_cockpit_env() -> dict[str, Any]:
    """Read all set env vars, cast to correct types, return config overrides.

    Keys prefixed with ``_`` are internal (not direct config keys) and are
    handled specially by the preset's ``apply_env_overrides()`` method.
    """
    overrides: dict[str, Any] = {}
    for env_key, (config_key, caster) in ENV_MAP.items():
        raw = os.environ.get(env_key, "").strip()
        if raw == "":
            continue
        try:
            overrides[config_key] = caster(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid value for env var {env_key}={raw!r}: {exc}"
            ) from exc
    return overrides


def validate_env() -> list[str]:
    """Warn about unrecognized env vars that look like cockpit variables."""
    known = set(ENV_MAP.keys())
    # Also recognise PYTORCH_CUDA_ALLOC_CONF and EXP_PRESET as valid
    known |= {"PYTORCH_CUDA_ALLOC_CONF", "EXP_PRESET"}
    warnings = []
    for key in os.environ:
        if key.startswith(("MODEL_", "SEQ_", "TRAIN_", "M1_", "M2_", "TBPTT_",
                           "EXP_", "WANDB_", "USE_", "LR", "EMA", "TORCH_COMPILE",
                           "CURRICULUM_", "TEMPORAL_", "HIDDEN_", "MULTISCALE_",
                           "BACKBONE_", "GRADIENT_", "RESUME_", "SAVE_", "EVAL_",
                           "TRACK_", "SHARE_", "BATCH_", "N_EPOCHS")):
            if key not in known:
                warnings.append(
                    f"  Warning: Env var '{key}' looks like a cockpit var but is not recognized. "
                    f"Did you mean one of: {', '.join(sorted(known))}?"
                )
    return warnings


def normalize_tbptt_mode(mode: str) -> str:
    """Normalize sequence-level TBPTT mode string."""
    mode = (mode or "").strip().lower()
    aliases = {
        "off": "off", "none": "off", "disabled": "off", "0": "off",
        "detach": "detach", "detach_only": "detach", "legacy": "detach", "1": "detach",
        "chunked": "chunked", "chunk": "chunked", "backward_chunked": "chunked", "true_tbptt": "chunked",
    }
    if mode in aliases:
        return aliases[mode]
    raise ValueError(f"Invalid TBPTT mode '{mode}'. Use: off|detach|chunked")
