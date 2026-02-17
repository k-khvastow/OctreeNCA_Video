#!/usr/bin/env python3
"""
train.py — Single modular training entry point.

Reads EXP_PRESET and cockpit env vars, builds the config, creates the
experiment, and runs training.

Usage:
    # Via cockpit shell script (recommended):
    source refactored/cockpit_ioct_finetune.sh

    # Or directly:
    EXP_PRESET=ioct_dual_warm MODEL_CHANNEL_N=32 LR=1e-5 python refactored/train.py

    # List available presets:
    python refactored/train.py --list-presets

    # Dry-run (print config without training):
    python refactored/train.py --dry-run
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

from refactored.env_config import load_cockpit_env, validate_env
from refactored.presets import get_preset, list_presets, list_preset_descriptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified OctreeNCA training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        help="Preset name. Overrides EXP_PRESET env var.",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="List available presets and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the final config and exit without training.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── List presets ─────────────────────────────────────────────────────
    if args.list_presets:
        print("Available presets:")
        for name in list_presets():
            desc = list_preset_descriptions()[name]
            print(f"  {name:25s}  {desc}")
        return

    # ── Determine preset ─────────────────────────────────────────────────
    preset_name = args.preset or os.environ.get("EXP_PRESET", "").strip()
    if not preset_name:
        print("Error: No preset specified. Set EXP_PRESET env var or use --preset.")
        print(f"Available: {', '.join(list_presets())}")
        sys.exit(1)

    # ── Validate env vars ────────────────────────────────────────────────
    warnings = validate_env()
    for w in warnings:
        print(w)

    # ── Load env overrides ───────────────────────────────────────────────
    env_overrides = load_cockpit_env()

    # ── Build preset + config ────────────────────────────────────────────
    print(f"Loading preset: {preset_name}")
    preset = get_preset(preset_name)
    study_config = preset.build_config(env_overrides)

    # ── Build dataset args ───────────────────────────────────────────────
    dataset_args = preset.dataset_args_builder(study_config, env_overrides)

    # ── Print runtime summary ────────────────────────────────────────────
    runtime_summary = {
        "preset": preset_name,
        "experiment_name": study_config.get("experiment.name"),
        "model.channel_n": study_config.get("model.channel_n"),
        "trainer.optimizer.lr": study_config.get("trainer.optimizer.lr"),
        "trainer.batch_size": study_config.get("trainer.batch_size"),
        "trainer.gradient_accumulation": study_config.get("trainer.gradient_accumulation"),
        "trainer.n_epochs": study_config.get("trainer.n_epochs"),
        "trainer.ema": study_config.get("trainer.ema"),
        "trainer.use_amp": study_config.get("trainer.use_amp"),
        "performance.compile": study_config.get("performance.compile", False),
        "performance.compile.mode": study_config.get("performance.compile.mode"),
        "model.sequence.tbptt_mode": study_config.get("model.sequence.tbptt_mode"),
        "model.sequence.tbptt_steps": study_config.get("model.sequence.tbptt_steps"),
        "model.m1.pretrained_path": study_config.get("model.m1.pretrained_path", ""),
        "model.m1.freeze": study_config.get("model.m1.freeze", None),
        "model.m1.use_t0_for_loss": study_config.get("model.m1.use_t0_for_loss", None),
        "use_wandb": study_config.get("experiment.use_wandb", False),
    }
    print("\n" + "=" * 60)
    print("RUNTIME CONFIG")
    print("=" * 60)
    for k, v in runtime_summary.items():
        print(f"  {k:40s} = {v}")
    print("=" * 60 + "\n")

    # ── Dry run ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n--- Full config (dry run) ---")
        # Filter out internal keys for display
        display_config = {k: v for k, v in sorted(study_config.items()) if not k.startswith("_")}
        print(json.dumps(display_config, indent=2, default=str))
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
