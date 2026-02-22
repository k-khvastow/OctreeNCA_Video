#!/usr/bin/env python3
"""
train_m2_single_step.py
──────────────────────────────────────────────────────────────────────────────
Standalone training entry point for the keyframe-M1 + single-step-M2 model.

This script is entirely self-contained and does NOT touch any existing training
workflows (train.py, presets.py, env_config.py remain unmodified).

Usage:
    # Via cockpit (recommended):
    ./refactored/cockpit_ioct_m2_single_step.sh

    # Or directly:
    M1_CHECKPOINT="/path/to/model.pth" \\
    M1_LEVEL_OFFSET=1 \\
    M2_MAX_STEP=20 \\
    python refactored/train_m2_single_step.py

    # Dry run (print config, no training):
    python refactored/train_m2_single_step.py --dry-run

New env vars handled here (not in env_config.py):
    M1_LEVEL_OFFSET       → model.m1.level_offset         (int, default 1)
    M2_MAX_STEP           → trainer.m2_single_step.max_step (int, default 20)
    M2_KEYFRAME_INTERVAL  → trainer.m2_single_step.keyframe_interval (int, 20)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from refactored.env_config import load_cockpit_env


# ── Extra env vars specific to this entry point ─────────────────────────
_EXTRA_ENV_MAP: dict[str, tuple[str, type]] = {
    "M1_LEVEL_OFFSET":      ("model.m1.level_offset",                      int),
    "M1_TRANSFER_LEVEL":    ("model.m1.transfer_level",                    int),
    "M2_MAX_STEP":          ("trainer.m2_single_step.max_step",             int),
    "M2_KEYFRAME_INTERVAL": ("trainer.m2_single_step.keyframe_interval",    int),
}


def _load_extra_env() -> dict:
    """Parse env vars specific to the m2_single_step workflow."""
    overrides: dict = {}
    for env_key, (config_key, caster) in _EXTRA_ENV_MAP.items():
        raw = os.environ.get(env_key, "").strip()
        if raw:
            try:
                overrides[config_key] = caster(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid value for env var {env_key}={raw!r}: {exc}"
                ) from exc
    return overrides


def _validate_env_m2() -> list[str]:
    """Warn about recognized-but-handled-separately env vars — avoids noise."""
    from refactored.env_config import validate_env
    warnings = validate_env()
    # Suppress warnings for our own new keys (they're handled by _load_extra_env)
    known_extra = set(_EXTRA_ENV_MAP.keys())
    return [w for w in warnings if not any(k in w for k in known_extra)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2 Single-Step keyframe training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without training.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after training.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate env ─────────────────────────────────────────────────────
    for w in _validate_env_m2():
        print(w)

    # ── Load env overrides ───────────────────────────────────────────────
    env_overrides = load_cockpit_env()
    env_overrides.update(_load_extra_env())  # extra keys take lower priority than explicit sets

    # ── Build preset + config ────────────────────────────────────────────
    from refactored.preset_m2_single_step import build_ioct_m2_single_step

    print("Loading preset: ioct_m2_single_step")
    preset = build_ioct_m2_single_step()
    study_config = preset.build_config(env_overrides)

    # ── Build dataset args ───────────────────────────────────────────────
    dataset_args = preset.dataset_args_builder(study_config, env_overrides)

    # ── Print runtime summary ────────────────────────────────────────────
    summary = {
        "preset":                           "ioct_m2_single_step",
        "experiment_name":                  study_config.get("experiment.name"),
        "model.channel_n":                  study_config.get("model.channel_n"),
        "model.m1.channel_n":               study_config.get("model.m1.channel_n"),
        "model.m1.level_offset":            study_config.get("model.m1.level_offset"),
        "model.m1.transfer_level":          study_config.get("model.m1.transfer_level", 0),
        "model.m1.num_levels":              study_config.get("model.m1.num_levels"),
        "model.m2.num_levels":              study_config.get("model.m2.num_levels"),
        "model.m2.extra_hidden":            study_config.get("model.m2.extra_hidden", 0),
        "model.m2.hidden_size":             study_config.get("model.m2.hidden_size"),
        "model.m1.pretrained_path":         study_config.get("model.m1.pretrained_path", ""),
        "model.m1.freeze":                  study_config.get("model.m1.freeze"),
        "trainer.m2_single_step.max_step":  study_config.get("trainer.m2_single_step.max_step"),
        "trainer.m2_single_step.keyframe_interval":
                                            study_config.get("trainer.m2_single_step.keyframe_interval"),
        "trainer.optimizer.lr":             study_config.get("trainer.optimizer.lr"),
        "trainer.batch_size":               study_config.get("trainer.batch_size"),
        "trainer.gradient_accumulation":    study_config.get("trainer.gradient_accumulation"),
        "trainer.n_epochs":                 study_config.get("trainer.n_epochs"),
        "trainer.ema":                      study_config.get("trainer.ema"),
        "trainer.use_amp":                  study_config.get("trainer.use_amp"),
        "sequence._seq.length":             study_config.get("_seq.length"),
        "model.octree.res_and_steps":       study_config.get("model.octree.res_and_steps"),
        "use_wandb":                        study_config.get("experiment.use_wandb", False),
    }
    print("\n" + "=" * 60)
    print("RUNTIME CONFIG — M2 Single Step")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:45s} = {v}")
    print("=" * 60 + "\n")

    if args.dry_run:
        print("\n--- Full config (dry run) ---")
        display = {k: v for k, v in sorted(study_config.items()) if not k.startswith("_")}
        print(json.dumps(display, indent=2, default=str))
        print("\n--- Dataset args ---")
        print(json.dumps(dataset_args, indent=2, default=str))
        return

    # ── Create experiment ────────────────────────────────────────────────
    from src.utils.Study import Study

    study = Study(study_config)
    exp = preset.experiment_wrapper_class().createExperiment(
        study_config,
        detail_config={},
        dataset_class=preset.dataset_class,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    # ── Train ────────────────────────────────────────────────────────────
    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()

    # ── Eval ─────────────────────────────────────────────────────────────
    if not args.no_eval:
        print("Running evaluation...")
        study.eval_experiments()

    print("Done.")


if __name__ == "__main__":
    main()
