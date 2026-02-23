"""
preset_m2_single_step_pan.py
──────────────────────────────────────────────────────────────────────────────
Self-contained preset for keyframe-M1 + single-step-M2 with a SINGLE-VIEW
(pan) M1, as opposed to the dual-view variant in preset_m2_single_step.py.

Key differences from the dual-view preset:
  - Uses ``iOCTSingleViewSequentialDataset`` that loads ONE view per frame
    (no paired A+B), inspired by the original ``train_ioct2d.py`` dataset.
  - Uses ``OctreeNCAM2SingleStepPanAgent`` that reads single-view data
    (``image`` / ``label``) and adapts internally for the dual-view model.
  - M1 uses ``cross_fusion = "none"`` (no FiLM), ``separate_models = True``.
  - ``model.m1.channel_n`` defaults to 24 (pan checkpoint default).
  - Name prefix: ``M2SingleStep_iOCT2D_pan``.

The standalone ``train_m2_single_step_pan.py`` entry point calls
``build_ioct_m2_single_step_pan()`` directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

# ── Project root ─────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import configs
from src.datasets.Dataset_Base import Dataset_Base
from src.losses.WeightedLosses import WeightedLosses
from src.utils.DistanceMaps import signed_distance_map
from src.utils.ExperimentWrapper import ExperimentWrapper

# Reuse shared helpers from the existing preset modules (read-only)
from refactored.presets import (
    Preset,
    build_octree_resolutions,
    _ioct_common_overrides,
    _ioct_losses,
    normalize_tbptt_mode,
)
from refactored.env_config import normalize_tbptt_mode  # noqa: F811
from refactored.preset_m2_single_step import M2SingleStepPreset

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_IOCT_DATA_ROOT = (
    os.environ.get("IOCT_DATA_ROOT", "/vol/data/OctreeNCA_Video/ioct_data").strip()
    or "/vol/data/OctreeNCA_Video/ioct_data"
)
_IOCT_DATASETS = ["peeling", "sri"]
_IOCT_VIEWS = ["A", "B"]


# ═══════════════════════════════════════════════════════════════════════════
# Single-view sequential dataset (inspired by train_ioct2d.py)
# ═══════════════════════════════════════════════════════════════════════════

class iOCTSingleViewSequentialDataset(Dataset_Base):
    """Single-view sequential iOCT dataset.

    Unlike ``iOCTPairedSequentialDataset`` which loads paired A+B frames,
    this dataset treats each (dataset, view) combination as an independent
    pool of temporal sequences, loading only ONE view per frame.

    Each view (A and B) from each dataset (peeling, sri) contributes
    separate sequences to the training set — the model never sees paired
    views simultaneously.

    Returns:
        image:  (T, 1, H, W)   float32
        label:  (T, C, H, W)   float32 one-hot
        label_dist: (T, C, H, W) float32 signed distance maps (optional)
    """

    # Same colour map as iOCTPairedSequentialDataset / iOCTDatasetForExperiment
    RGB_TO_CLASS = {
        (0, 0, 0): 0,          # Background
        (255, 0, 0): 1,        # Class 1 (red)
        (0, 255, 209): 2,      # Class 2 (cyan)
        (61, 255, 0): 3,       # Class 3 (green)
        (0, 78, 255): 4,       # Class 4 (blue)
        (255, 189, 0): 5,      # Class 5 (yellow/orange)
        (218, 0, 255): 6,      # Class 6 (magenta)
    }

    def __init__(
        self,
        data_root: str,
        datasets=("peeling", "sri"),
        views=("A", "B"),
        sequence_length: int = 3,
        sequence_step: int = 1,
        num_classes: int = 7,
        input_size=(512, 512),
        class_subset=None,
        merge_all_classes: bool = False,
        precompute_boundary_dist: bool = False,
        boundary_dist_classes=None,
        max_samples: int = None,
        sparse_temporal_loading: bool = False,
        sparse_max_step: int = 20,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.datasets = list(datasets)
        self.views = list(views)
        self.sequence_length = int(sequence_length)
        self.sequence_step = int(sequence_step)
        self.num_classes = num_classes
        self.size = input_size
        self.merge_all_classes = merge_all_classes
        self.precompute_boundary_dist = precompute_boundary_dist
        self.boundary_dist_classes = boundary_dist_classes
        self.max_samples = max_samples
        self.sparse_temporal_loading = sparse_temporal_loading
        self.sparse_max_step = int(sparse_max_step)

        # Required by agents
        self.slice = -1
        self.delivers_channel_axis = True
        self.is_rgb = False

        # Optional class subset
        self.class_subset = None
        self.class_map = None
        if class_subset is not None:
            cleaned = sorted(set(int(c) for c in class_subset if int(c) != 0))
            if not cleaned:
                raise ValueError("class_subset must include at least one non-zero class id.")
            self.class_subset = cleaned
            self.class_map = {c: i + 1 for i, c in enumerate(self.class_subset)}
            self.num_classes = len(self.class_subset) + 1

        self.sequences = []
        self.sequences_dict = {}
        self._collect_sequences()

    def _collect_sequences(self):
        def _sort_key(name: str):
            stem = Path(name).stem
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        required_span = (self.sequence_length - 1) * self.sequence_step + 1

        for dataset_name in self.datasets:
            for view in self.views:
                base_path = self.data_root / dataset_name / "Bscans-dt" / view
                img_dir = base_path / "Image"
                seg_dir = base_path / "Segmentation"

                if not img_dir.exists() or not seg_dir.exists():
                    print(f"Warning: Skipping {dataset_name}/{view} — dirs not found")
                    continue

                # Collect matched image/segmentation filenames
                names = sorted(
                    [p.name for p in img_dir.glob("*.png") if (seg_dir / p.name).exists()],
                    key=_sort_key,
                )

                if len(names) < required_span:
                    continue

                # Build all possible sequences from this (dataset, view) pair
                for i in range(0, len(names) - required_span + 1):
                    seq_names = [names[i + j * self.sequence_step]
                                 for j in range(self.sequence_length)]
                    seq_id = f"{dataset_name}_{view}_{Path(seq_names[0]).stem}"
                    info = {
                        "id": seq_id,
                        "patient_id": f"{dataset_name}_{view}",
                        "dataset": dataset_name,
                        "view": view,
                        "seq_names": seq_names,
                        "img_dir": img_dir,
                        "seg_dir": seg_dir,
                    }
                    self.sequences.append(info)
                    self.sequences_dict[seq_id] = info

        if self.max_samples is not None and self.max_samples > 0:
            self.sequences = self.sequences[: self.max_samples]
            self.sequences_dict = {s["id"]: s for s in self.sequences}

        print(
            f"Found {len(self.sequences)} single-view iOCT sequences "
            f"(views={self.views}, length={self.sequence_length}, step={self.sequence_step})."
        )

    def getFilesInPath(self, path: str):
        return {k: {"id": k} for k in self.sequences_dict.keys()}

    def setPaths(self, images_path, images_list, labels_path, labels_list):
        super().setPaths(images_path, images_list, labels_path, labels_list)
        self.sequences = [
            self.sequences_dict[uid]
            for uid in self.images_list
            if uid in self.sequences_dict
        ]
        print(f"Dataset split set. Active single-view sequences: {len(self.sequences)}")

    def _rgb_to_class(self, rgb_seg: np.ndarray) -> np.ndarray:
        h, w = rgb_seg.shape[:2]
        class_seg = np.zeros((h, w), dtype=np.int64)
        for rgb_val, class_idx in self.RGB_TO_CLASS.items():
            mask = (
                (rgb_seg[:, :, 0] == rgb_val[0])
                & (rgb_seg[:, :, 1] == rgb_val[1])
                & (rgb_seg[:, :, 2] == rgb_val[2])
            )
            class_seg[mask] = class_idx
        return class_seg

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, img_path: Path, seg_path: Path):
        """Load a single frame: grayscale image (1,H,W) and one-hot label (C,H,W)."""
        img = np.array(Image.open(img_path))
        seg_rgb = np.array(Image.open(seg_path))

        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)

        seg = self._rgb_to_class(seg_rgb)

        expected_size = tuple(self.size)
        if img.shape != expected_size:
            raise ValueError(f"Image shape {img.shape} != expected {expected_size} for {img_path}.")
        if seg.shape != expected_size:
            raise ValueError(f"Seg shape {seg.shape} != expected {expected_size} for {seg_path}.")

        if self.class_map is not None:
            remapped = np.zeros_like(seg)
            for src, dst in self.class_map.items():
                remapped[seg == src] = dst
            seg = remapped

        if self.merge_all_classes:
            seg = (seg > 0).astype(np.int64)

        img = img.astype(np.float32) / 255.0
        img = img[None, :, :]  # (1, H, W)

        seg_tensor = torch.from_numpy(seg).long()
        max_class = int(seg_tensor.max().item())
        if max_class >= self.num_classes:
            raise ValueError(
                f"Seg class id {max_class} >= num_classes ({self.num_classes}). "
                "Update model.output_channels or class_subset."
            )
        label_onehot = (
            torch.nn.functional.one_hot(seg_tensor, num_classes=self.num_classes)
            .permute(2, 0, 1)
            .numpy()
            .astype(np.float32)
        )

        label_dist = None
        if self.precompute_boundary_dist:
            label_dist = signed_distance_map(
                label_onehot,
                class_ids=self.boundary_dist_classes,
                channel_first=True,
                compact=False,
                dtype=np.float32,
            )
        return img, label_onehot, label_dist

    def __getitem__(self, idx):
        info = self.sequences[idx]
        seq_names = info["seq_names"]

        # ── Sparse mode: only load frame 0 (anchor) + random frame k ──
        use_sparse = (
            self.sparse_temporal_loading
            and getattr(self, "state", "train") == "train"
        )
        if use_sparse:
            max_k = min(self.sparse_max_step, len(seq_names) - 1)
            k = int(np.random.randint(1, max(2, max_k + 1)))
            load_indices = [0, k]
            names_to_load = [seq_names[i] for i in load_indices]
        else:
            load_indices = None
            names_to_load = seq_names

        imgs, lbls = [], []
        dists = [] if self.precompute_boundary_dist else None

        for name in names_to_load:
            img, lbl, dist = self._load_frame(
                info["img_dir"] / name, info["seg_dir"] / name
            )
            imgs.append(img)
            lbls.append(lbl)
            if dists is not None:
                dists.append(dist)

        sample = {
            "image": np.stack(imgs),    # (T, 1, H, W)
            "label": np.stack(lbls),    # (T, C, H, W)
            "id": info["id"],
            "patient_id": info["patient_id"],
            "dataset": info["dataset"],
            "view": info["view"],
            "frame_start": Path(info["seq_names"][0]).stem,
            "path": str(info["img_dir"] / info["seq_names"][0]),
        }

        if use_sparse:
            sample["sparse_target_k"] = k

        if dists is not None:
            sample["label_dist"] = np.stack(dists)

        return sample


_IOCT_NUM_CLASSES = max(iOCTSingleViewSequentialDataset.RGB_TO_CLASS.values()) + 1


# ═══════════════════════════════════════════════════════════════════════════
# Experiment wrapper
# ═══════════════════════════════════════════════════════════════════════════

class EXP_M2SingleStepPan(ExperimentWrapper):
    """Factory for keyframe-M1 + single-step-M2 experiments (pan / single-view)."""

    def createExperiment(self, study_config: dict, detail_config: dict = {},
                         dataset_class=None, dataset_args=None):
        if dataset_class is None:
            raise ValueError("dataset_class must be provided")

        from src.models.Model_OctreeNCA_2d_m2_single_step import OctreeNCA2DM2SingleStep
        from refactored.agent_m2_single_step_pan import OctreeNCAM2SingleStepPanAgent

        model = OctreeNCA2DM2SingleStep(study_config)
        agent = OctreeNCAM2SingleStepPanAgent(model)
        loss_function = WeightedLosses(study_config)
        return super().createExperiment(
            study_config, model, agent,
            dataset_class, dataset_args or {}, loss_function,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Dataset args builder
# ═══════════════════════════════════════════════════════════════════════════

def _ioct_pan_dataset_args(study_config: dict, env_overrides: dict) -> dict:
    """Build dataset args for single-view sequential iOCT."""
    seq_length = study_config.get("_seq.length", 3)
    seq_step = study_config.get("_seq.step", 1)
    curriculum_max = study_config.get("trainer.curriculum.seq_len_max", seq_length)
    dataset_seq_len = max(seq_length, curriculum_max)

    return {
        "data_root": _IOCT_DATA_ROOT,
        "datasets": _IOCT_DATASETS,
        "views": _IOCT_VIEWS,
        "sequence_length": dataset_seq_len,
        "sequence_step": seq_step,
        "num_classes": study_config["model.output_channels"],
        "input_size": study_config["experiment.dataset.input_size"],
        "class_subset": study_config.get("experiment.dataset.class_subset", None),
        "merge_all_classes": study_config.get("experiment.dataset.merge_all_classes", False),
        "precompute_boundary_dist": study_config.get("experiment.dataset.precompute_boundary_dist", False),
        "boundary_dist_classes": study_config.get("experiment.dataset.boundary_dist_classes", None),
        "sparse_temporal_loading": study_config.get("experiment.dataset.sparse_temporal_loading", False),
        "sparse_max_step": study_config.get("trainer.m2_single_step.max_step",
                                            study_config.get("experiment.dataset.sparse_max_step", 20)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Preset overrides
# ═══════════════════════════════════════════════════════════════════════════

_ioct_m2_single_step_pan_overrides: dict[str, Any] = {
    **_ioct_common_overrides(_IOCT_NUM_CLASSES),

    # ── M1: pan (single-view) pretrained, frozen ────────────────────────
    # cross_fusion=none disables FiLM → state dict matches non-dual model.
    # separate_models=True matches the pan M1 checkpoint architecture.
    "model.dual_view.cross_fusion": "none",
    "model.dual_view.cross_strength": 0.0,
    "model.dual_view.cross_use_tanh": False,
    "model.octree.separate_models": True,

    "model.m1.pretrained_path": "",
    "model.m1.freeze": True,
    "model.m1.eval_mode": True,
    "model.m1.use_first_frame": True,
    "model.m1.use_t0_for_loss": False,
    "model.m1.use_probs": False,
    "model.m1.disable_backbone_tbptt": True,
    "model.m1.level_offset": 0,      # include all levels → M1 finest = 512×512
    "model.m1.num_levels": 4,        # 512→256→128→64 (skip 32×32 from pan)
    "model.m1.transfer_level": 1,    # transfer hidden from level 1 = 256×256
    "model.m1.channel_n": 24,        # must match the pan pretrained checkpoint

    # ── M2: fresh init, operates at 512×512 only ────────────────────────
    "model.m2.init_from_m1": False,
    "model.m2.init_identity": False,
    "model.m2.num_levels": 1,        # 512×512 only
    "model.m2.extra_hidden": 0,
    "model.m2.hidden_size": None,

    # ── Single-step training knobs ───────────────────────────────────────
    "trainer.m2_single_step.max_step": 20,
    "trainer.m2_single_step.keyframe_interval": 20,

    # ── Dataset: sequences long enough to sample up to step 20 ──────────
    "_seq.length": 21,
    "_seq.step": 1,

    # ── Sparse loading ───────────────────────────────────────────────────
    "experiment.dataset.sparse_temporal_loading": True,

    # ── No TBPTT — M2 is single-step ────────────────────────────────────
    "model.sequence.tbptt_mode": "off",
    "model.sequence.tbptt_steps": None,

    # ── Model channels ───────────────────────────────────────────────────
    "model.channel_n": 24,           # match pan M1 default

    # ── Per-level kernel sizes (separate_models=True) ────────────────────
    "model.kernel_size": [3, 3, 3, 3],

    # ── Training ────────────────────────────────────────────────────────
    "trainer.optimizer.lr": 1e-4,
    "trainer.ema": True,
    "trainer.ema.decay": 0.99,
    "trainer.gradient_clip_val": 1.0,

    # ── Disable sequence-level regularizers ──────────────────────────────
    "trainer.temporal_consistency_weight": 0.0,
    "trainer.contractive_weight": 0.0,
    "trainer.latent_sfa_weight": 0.0,
    "trainer.latent_sfa_decorrelation_weight": 1.0,
    "trainer.motion_loss_weight": 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Preset builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_ioct_m2_single_step_pan() -> Preset:
    def _env(key, default):
        return os.environ.get(key, str(default)).strip() or str(default)

    use_boundary    = _env("BOUNDARY_LOSS",       "0") == "1"
    boundary_weight = float(_env("BOUNDARY_LOSS_WEIGHT", "0.1"))
    boundary_clip   = float(_env("BOUNDARY_DIST_CLIP",   "20.0"))
    dice_weight     = float(_env("DICE_LOSS_WEIGHT",  "1.0"))
    focal_weight    = float(_env("FOCAL_LOSS_WEIGHT", "1.0"))
    focal_gamma     = float(_env("FOCAL_GAMMA",       "2.0"))
    dice_smooth     = float(_env("DICE_SMOOTH",       "1e-5"))
    dice_batch_dice = _env("DICE_BATCH_DICE", "1") in ("1", "true", "yes")
    dice_do_bg      = _env("DICE_DO_BG",      "0") in ("1", "true", "yes")
    focal_ignore_index = int(_env("FOCAL_IGNORE_INDEX", "0"))
    focal_reduction    = _env("FOCAL_REDUCTION",    "mean")
    tversky_alpha       = float(_env("TVERSKY_ALPHA",       "0.3"))
    tversky_beta        = float(_env("TVERSKY_BETA",        "0.7"))
    tversky_gamma       = float(_env("TVERSKY_GAMMA",       "1.0"))
    tversky_smooth      = float(_env("TVERSKY_SMOOTH",      "0.0"))
    _tversky_ign_raw    = _env("TVERSKY_IGNORE_INDEX",  "")
    tversky_ignore_idx  = int(_tversky_ign_raw) if _tversky_ign_raw else None

    overrides = dict(_ioct_m2_single_step_pan_overrides)
    overrides.update(_ioct_losses(
        _IOCT_NUM_CLASSES,
        use_boundary_loss=use_boundary,
        boundary_loss_weight=boundary_weight,
        boundary_dist_clip=boundary_clip,
        boundary_do_bg=_env("BOUNDARY_DO_BG", "0") in ("1", "true", "yes"),
        boundary_use_probabilities=_env("BOUNDARY_USE_PROBABILITIES", "0") in ("1", "true", "yes"),
        boundary_compute_missing_dist=_env("BOUNDARY_COMPUTE_MISSING_DIST", "0") in ("1", "true", "yes"),
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        focal_gamma=focal_gamma,
        dice_smooth=dice_smooth,
        dice_batch_dice=dice_batch_dice,
        dice_do_bg=dice_do_bg,
        focal_ignore_index=focal_ignore_index,
        focal_reduction=focal_reduction,
        dice_type=_env("DICE_TYPE", "nnunet"),
        dice_weight_eps=float(_env("DICE_WEIGHT_EPS", "1e-6")),
        gdl_weight_type=_env("GDL_WEIGHT_TYPE", "v2"),
        gdl_max_weight=float(_env("GDL_MAX_WEIGHT", "0")) or None,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        tversky_gamma=tversky_gamma,
        tversky_smooth=tversky_smooth,
        tversky_ignore_index=tversky_ignore_idx,
    ))
    if use_boundary:
        overrides["experiment.dataset.precompute_boundary_dist"] = True

    return M2SingleStepPreset(
        name="ioct_m2_single_step_pan",
        description_template="Keyframe M1-pan (256×256) + single-step M2 (512×512) iOCT segmentation",
        name_prefix="M2SingleStep_iOCT2D_pan",
        experiment_wrapper_class=EXP_M2SingleStepPan,
        dataset_class=iOCTSingleViewSequentialDataset,
        dataset_args_builder=_ioct_pan_dataset_args,
        config_layers=[
            configs.models.peso.peso_model_config,
            configs.trainers.nca.nca_trainer_config,
            configs.tasks.segmentation.segmentation_task_config,
            configs.default.default_config,
        ],
        default_overrides=overrides,
    )


def build_ioct_m2_single_step_pan() -> Preset:
    """Public entry point: build the ioct_m2_single_step_pan preset."""
    return _build_ioct_m2_single_step_pan()
