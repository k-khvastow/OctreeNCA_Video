"""
preset_m2_single_step.py
──────────────────────────────────────────────────────────────────────────────
Self-contained preset definition for the keyframe-M1 + single-step-M2 model.

This module is intentionally SEPARATE from refactored/presets.py so it does
not disturb any existing training workflow.

It reuses infrastructure (Preset, config layer builders, dataset helpers) from
presets.py via normal imports, but adds nothing to the _PRESET_BUILDERS dict
in that module.  The standalone train_m2_single_step.py entry point calls
``build_ioct_m2_single_step()`` directly.

New config keys introduced
──────────────────────────
  model.m1.level_offset          (int, default 0)
    Skip the first N resolution levels when building M1.
    With offset=1 and full_res = [[512,…],[256,…],[128,…],[64,…]],
    M1 operates at finest = 256×256.

  trainer.m2_single_step.max_step          (int, default 20)
    Upper bound of the random step k ~ U[1, max_step] drawn each batch.

  trainer.m2_single_step.keyframe_interval (int, default 20)
    How many frames between M1 calls during inference.
    20 = call M1 once per second at 20 FPS.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── Project root ─────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import configs
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses

# Reuse shared helpers from the existing preset module (read-only, no modification)
from refactored.presets import (
    Preset,
    build_octree_resolutions,
    _ioct_common_overrides,   # "private" but importable: read-only use
    _ioct_losses,
    _ioct_dual_dataset_args,
    normalize_tbptt_mode,
)
from refactored.env_config import normalize_tbptt_mode  # noqa: F811 (same symbol, fine)
from dataclasses import dataclass

# Dataset class (defined inside presets.py, not a separate module)
from refactored.presets import iOCTPairedSequentialDataset


# ═══════════════════════════════════════════════════════════════════════════
# Experiment wrapper
# ═══════════════════════════════════════════════════════════════════════════

class EXP_M2SingleStep(ExperimentWrapper):
    """Factory for keyframe-M1 + single-step-M2 experiments."""

    def createExperiment(self, study_config: dict, detail_config: dict = {},
                         dataset_class=None, dataset_args=None):
        if dataset_class is None:
            raise ValueError("dataset_class must be provided")

        from src.models.Model_OctreeNCA_2d_m2_single_step import OctreeNCA2DM2SingleStep
        from src.agents.Agent_OctreeNCA_M2SingleStep import OctreeNCAM2SingleStepAgent

        model = OctreeNCA2DM2SingleStep(study_config)
        agent = OctreeNCAM2SingleStepAgent(model)
        loss_function = WeightedLosses(study_config)
        return super().createExperiment(
            study_config, model, agent,
            dataset_class, dataset_args or {}, loss_function,
        )


# ═══════════════════════════════════════════════════════════════════════════
# M2-aware Preset subclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class M2SingleStepPreset(Preset):
    """Preset subclass that rebuilds ``model.octree.res_and_steps`` with
    exactly ``model.m1.num_levels`` levels after the base config is built.

    The base ``Preset.build_config`` always builds the full default number of
    levels (typically 5), so ``OCTREE_COARSEST_STEPS`` ends up on the wrong
    level when M1 uses fewer levels.  By re-calling ``build_octree_resolutions``
    with the correct ``num_levels``, the coarsest-steps value is applied to the
    actual coarsest M1 level.
    """

    def build_config(self, env_overrides: dict) -> dict:
        config = super().build_config(env_overrides)
        m1_num_levels = config.get("model.m1.num_levels")
        if m1_num_levels is not None:
            config["model.octree.res_and_steps"] = build_octree_resolutions(
                config.get("experiment.dataset.input_size", (512, 512)),
                steps_per_level=config.get("model.octree.steps_per_level", 8),
                coarsest_steps=config.get("model.octree.coarsest_steps", 20),
                first_steps_multiplier=config.get("model.octree.finest_multiplier", 2),
                num_levels=int(m1_num_levels),
            )
        return config


# ═══════════════════════════════════════════════════════════════════════════
# Preset definition
# ═══════════════════════════════════════════════════════════════════════════

_IOCT_NUM_CLASSES = max(iOCTPairedSequentialDataset.RGB_TO_CLASS.values()) + 1

_ioct_m2_single_step_overrides: dict[str, Any] = {
    **_ioct_common_overrides(_IOCT_NUM_CLASSES),

    # ── M1: pretrained, frozen, runs all levels (512→256→128→64) ────────
    # M1 produces 512×512 segmentation (for vis/loss) but hidden states
    # for M2 warm-start are taken from level 1 (256×256) and upscaled.
    "model.m1.pretrained_path": "",
    "model.m1.freeze": True,
    "model.m1.eval_mode": True,
    "model.m1.use_first_frame": True,
    "model.m1.use_t0_for_loss": False,
    "model.m1.use_probs": False,
    "model.m1.disable_backbone_tbptt": True,
    "model.m1.level_offset": 0,      # include all levels → M1 finest = 512×512
    "model.m1.num_levels": 4,        # 512×512, 256×256, 128×128, 64×64
    "model.m1.transfer_level": 1,    # transfer hidden from level 1 = 256×256
    "model.m1.channel_n": 32,        # must match the pretrained checkpoint

    # ── M2: fresh init, operates at 512×512 only ────────────────────────
    "model.m2.init_from_m1": False,  # architectures may differ
    "model.m2.init_identity": False,
    "model.m2.num_levels": 1,        # 512×512 only
    "model.m2.extra_hidden": 0,      # extra hidden channels (start empty, appended)
    "model.m2.hidden_size": None,    # None = use model.hidden_size; set e.g. 128 for larger M2 FC

    # ── Single-step training knobs ───────────────────────────────────────
    "trainer.m2_single_step.max_step": 20,            # k ~ U[1, 20]
    "trainer.m2_single_step.keyframe_interval": 20,   # 1 M1 call/second

    # ── Dataset: sequences long enough to sample up to step 20 ──────────
    "_seq.length": 21,
    "_seq.step": 1,

    # ── Sparse loading: only load frames 0 and k instead of all 21 ──────
    # The dataset samples k during __getitem__ and returns 2 frames.
    # Cuts per-batch I/O from ~21 frame-pairs to 2.
    "experiment.dataset.sparse_temporal_loading": True,

    # ── No TBPTT — M2 is single-step ────────────────────────────────────
    "model.sequence.tbptt_mode": "off",
    "model.sequence.tbptt_steps": None,

    # ── Model channels ───────────────────────────────────────────────────
    "model.channel_n": 32,

    # ── Training ────────────────────────────────────────────────────────
    "trainer.optimizer.lr": 1e-4,
    "trainer.ema": True,
    "trainer.ema.decay": 0.99,
    "trainer.gradient_clip_val": 1.0,

    # ── Disable sequence-level regularizers (not applicable here) ────────
    "trainer.temporal_consistency_weight": 0.0,
    "trainer.contractive_weight": 0.0,
    "trainer.latent_sfa_weight": 0.0,
    "trainer.latent_sfa_decorrelation_weight": 1.0,
    "trainer.motion_loss_weight": 0.0,
}


def _build_ioct_m2_single_step() -> Preset:
    import os
    use_boundary = os.environ.get("BOUNDARY_LOSS", "0").strip() == "1"
    boundary_weight = float(os.environ.get("BOUNDARY_LOSS_WEIGHT", "0.1").strip())
    boundary_clip = float(os.environ.get("BOUNDARY_DIST_CLIP", "20.0").strip())

    overrides = dict(_ioct_m2_single_step_overrides)
    overrides.update(_ioct_losses(
        _IOCT_NUM_CLASSES,
        use_boundary_loss=use_boundary,
        boundary_loss_weight=boundary_weight,
        boundary_dist_clip=boundary_clip,
    ))
    # When boundary loss is active, dataset must precompute signed distance maps.
    if use_boundary:
        overrides["experiment.dataset.precompute_boundary_dist"] = True
    return M2SingleStepPreset(
        name="ioct_m2_single_step",
        description_template="Keyframe M1 (256×256) + single-step M2 (512×512) iOCT segmentation",
        name_prefix="M2SingleStep_iOCT2D_dual",
        experiment_wrapper_class=EXP_M2SingleStep,
        dataset_class=iOCTPairedSequentialDataset,
        dataset_args_builder=_ioct_dual_dataset_args,
        config_layers=[
            configs.models.peso.peso_model_config,
            configs.trainers.nca.nca_trainer_config,
            configs.tasks.segmentation.segmentation_task_config,
            configs.default.default_config,
        ],
        default_overrides=overrides,
    )


def build_ioct_m2_single_step() -> Preset:
    """Public entry point: build the ioct_m2_single_step preset."""
    return _build_ioct_m2_single_step()
