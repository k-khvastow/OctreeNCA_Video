"""
presets.py — Preset registry for all experiment families.

Each preset defines:
  - Which model/agent/experiment-wrapper classes to use
  - Which config layers to merge
  - Default overrides specific to that experiment family
  - How to build the dataset and dataset_args
  - How to generate focal-alpha weights (if applicable)

Usage:
    from refactored.presets import PRESETS
    preset = PRESETS["ioct_dual_warm"]
    config = preset.build_config(env_overrides)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Type

import numpy as np
from PIL import Image

# ── Framework imports ────────────────────────────────────────────────────
# Add project root to path so we can import src/ and configs/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import configs
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses
from src.datasets.Dataset_Base import Dataset_Base
from src.utils.DistanceMaps import signed_distance_map

from refactored.env_config import normalize_tbptt_mode


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_octree_resolutions(
    input_size: tuple[int, int],
    steps_per_level: int = 8,
    coarsest_steps: int = 20,
    first_steps_multiplier: int = 2,
    num_levels: int = 5,
) -> list:
    """Build the multi-resolution octree spec: [[res, steps], ...]."""
    h, w = input_size
    resolutions = []
    for _ in range(num_levels):
        resolutions.append([h, w])
        h = max(1, h // 2)
        w = max(1, w // 2)
    res_and_steps = []
    for i, res in enumerate(resolutions):
        if i == 0:
            res_and_steps.append([res, steps_per_level * first_steps_multiplier])
        elif i == len(resolutions) - 1:
            res_and_steps.append([res, coarsest_steps])
        else:
            res_and_steps.append([res, steps_per_level])
    return res_and_steps


def compute_class_alpha_weights(
    data_root: str,
    datasets: list[str],
    views: list[str],
    num_classes: int,
    rgb_to_class: dict,
    max_samples: int = 200,
    merge_to_binary: bool = False,
) -> list[float]:
    """Scan segmentation masks and return inverse-frequency alpha weights
    suitable for FocalLoss.

    If merge_to_binary is True, all non-zero classes are merged into class 1
    before computing frequencies (for binary bg/fg segmentation).
    """
    effective_classes = 2 if merge_to_binary else num_classes
    counts = np.zeros(effective_classes, dtype=np.float64)
    n_scanned = 0
    for ds_name in datasets:
        for view in views:
            seg_dir = Path(data_root) / ds_name / "Bscans-dt" / view / "Segmentation"
            if not seg_dir.exists():
                continue
            for seg_path in sorted(seg_dir.glob("*.png")):
                seg_rgb = np.array(Image.open(seg_path))
                class_map = np.zeros(seg_rgb.shape[:2], dtype=np.int64)
                for rgb_val, cls_idx in rgb_to_class.items():
                    mask = (
                        (seg_rgb[:, :, 0] == rgb_val[0])
                        & (seg_rgb[:, :, 1] == rgb_val[1])
                        & (seg_rgb[:, :, 2] == rgb_val[2])
                    )
                    class_map[mask] = cls_idx
                if merge_to_binary:
                    class_map = (class_map > 0).astype(np.int64)
                for c in range(effective_classes):
                    counts[c] += (class_map == c).sum()
                n_scanned += 1
                if n_scanned >= max_samples:
                    break
            if n_scanned >= max_samples:
                break
        if n_scanned >= max_samples:
            break
    print(f"Class pixel counts (from {n_scanned} samples): {counts}")
    total = counts.sum()
    alpha = np.where(counts > 0, total / (num_classes * counts), 1.0)
    alpha[0] = min(alpha[0], 0.25)
    fg_mean = alpha[1:].mean()
    if fg_mean > 0:
        alpha[1:] = alpha[1:] / fg_mean
    return alpha.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# iOCT sequential dataset (shared between dual-view presets)
# ═══════════════════════════════════════════════════════════════════════════

class iOCTPairedSequentialDataset(Dataset_Base):
    """Paired-view sequential iOCT dataset.

    Returns image_a, image_b: (T, 1, H, W) and label_a, label_b: (T, C, H, W).
    """

    RGB_TO_CLASS = {
        (0, 0, 0): 0,
        (255, 0, 0): 1,
        (0, 255, 209): 2,
        (61, 255, 0): 3,
        (0, 78, 255): 4,
        (255, 189, 0): 5,
        (218, 0, 255): 6,
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
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.datasets = list(datasets)
        self.views = list(views)
        if len(self.views) != 2:
            raise ValueError(f"Expected exactly two views, got {self.views}.")
        self.view_a, self.view_b = self.views[0], self.views[1]

        self.sequence_length = int(sequence_length)
        self.sequence_step = int(sequence_step)
        self.num_classes = num_classes
        self.size = input_size
        self.merge_all_classes = merge_all_classes
        self.precompute_boundary_dist = precompute_boundary_dist
        self.boundary_dist_classes = boundary_dist_classes
        self.max_samples = max_samples

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
            base_path = self.data_root / dataset_name / "Bscans-dt"
            img_dir_a = base_path / self.view_a / "Image"
            seg_dir_a = base_path / self.view_a / "Segmentation"
            img_dir_b = base_path / self.view_b / "Image"
            seg_dir_b = base_path / self.view_b / "Segmentation"

            if not all(d.exists() for d in [img_dir_a, seg_dir_a, img_dir_b, seg_dir_b]):
                print(f"Warning: Skipping {dataset_name} — directories not found "
                      f"({self.view_a}, {self.view_b})")
                continue

            names_a = {p.name for p in img_dir_a.glob("*.png") if (seg_dir_a / p.name).exists()}
            names_b = {p.name for p in img_dir_b.glob("*.png") if (seg_dir_b / p.name).exists()}
            common = sorted(names_a & names_b, key=_sort_key)

            if len(common) < required_span:
                continue

            for i in range(0, len(common) - required_span + 1):
                seq_names = [common[i + j * self.sequence_step] for j in range(self.sequence_length)]
                seq_id = f"{dataset_name}_{Path(seq_names[0]).stem}"
                info = {
                    "id": seq_id,
                    "patient_id": seq_id,
                    "dataset": dataset_name,
                    "view_a": self.view_a,
                    "view_b": self.view_b,
                    "seq_names": seq_names,
                    "img_dir_a": img_dir_a,
                    "seg_dir_a": seg_dir_a,
                    "img_dir_b": img_dir_b,
                    "seg_dir_b": seg_dir_b,
                }
                self.sequences.append(info)
                self.sequences_dict[seq_id] = info

        if self.max_samples is not None and self.max_samples > 0:
            self.sequences = self.sequences[:self.max_samples]
            self.sequences_dict = {s["id"]: s for s in self.sequences}

        print(f"Found {len(self.sequences)} paired iOCT sequences "
              f"(views={self.view_a}+{self.view_b}, length={self.sequence_length}, step={self.sequence_step}).")

    def getFilesInPath(self, path: str):
        return {k: {"id": k} for k in self.sequences_dict.keys()}

    def setPaths(self, images_path, images_list, labels_path, labels_list):
        super().setPaths(images_path, images_list, labels_path, labels_list)
        self.sequences = [self.sequences_dict[uid] for uid in self.images_list if uid in self.sequences_dict]
        print(f"Dataset split set. Active paired sequences: {len(self.sequences)}")

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

    def _load_view_frame(self, img_path: Path, seg_path: Path):
        import torch
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

        imgs_a, lbls_a = [], []
        imgs_b, lbls_b = [], []
        dists_a = [] if self.precompute_boundary_dist else None
        dists_b = [] if self.precompute_boundary_dist else None

        for name in info["seq_names"]:
            img_a, lbl_a, dist_a = self._load_view_frame(info["img_dir_a"] / name, info["seg_dir_a"] / name)
            img_b, lbl_b, dist_b = self._load_view_frame(info["img_dir_b"] / name, info["seg_dir_b"] / name)
            imgs_a.append(img_a)
            lbls_a.append(lbl_a)
            imgs_b.append(img_b)
            lbls_b.append(lbl_b)
            if dists_a is not None:
                dists_a.append(dist_a)
            if dists_b is not None:
                dists_b.append(dist_b)

        sample = {
            "image_a": np.stack(imgs_a),
            "label_a": np.stack(lbls_a),
            "image_b": np.stack(imgs_b),
            "label_b": np.stack(lbls_b),
            "image": np.stack(imgs_a),
            "label": np.stack(lbls_a),
            "id": info["id"],
            "patient_id": info["patient_id"],
            "dataset": info["dataset"],
            "view_a": info["view_a"],
            "view_b": info["view_b"],
            "frame_start": Path(info["seq_names"][0]).stem,
            "path_a": str(info["img_dir_a"] / info["seq_names"][0]),
            "path_b": str(info["img_dir_b"] / info["seq_names"][0]),
        }

        if dists_a is not None:
            sample["label_dist_a"] = np.stack(dists_a)
            sample["label_dist"] = sample["label_dist_a"]
        if dists_b is not None:
            sample["label_dist_b"] = np.stack(dists_b)
        return sample


# ═══════════════════════════════════════════════════════════════════════════
# Experiment wrapper (shared by dual-view presets)
# ═══════════════════════════════════════════════════════════════════════════

class EXP_DualViewWarmStart(ExperimentWrapper):
    """Factory for dual-view warm-start OctreeNCA experiments."""

    def createExperiment(self, study_config: dict, detail_config: dict = {},
                         dataset_class=None, dataset_args=None):
        if dataset_class is None:
            raise ValueError("dataset_class must be provided")

        from src.models.Model_OctreeNCA_2d_dual_view_warm_m1init import OctreeNCA2DDualViewWarmStartM1Init
        from src.agents.Agent_OctreeNCA_DualView_WarmStart_M1Init import OctreeNCADualViewWarmStartM1InitAgent

        model = OctreeNCA2DDualViewWarmStartM1Init(study_config)
        agent = OctreeNCADualViewWarmStartM1InitAgent(model)
        loss_function = WeightedLosses(study_config)
        return super().createExperiment(study_config, model, agent,
                                        dataset_class, dataset_args or {}, loss_function)


# ═══════════════════════════════════════════════════════════════════════════
# Preset dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Preset:
    """Complete specification of one experiment family."""

    name: str
    description_template: str

    # Name prefix for auto-generated experiment names
    name_prefix: str

    # The ExperimentWrapper subclass to use
    experiment_wrapper_class: type

    # Dataset class and how to build its args
    dataset_class: type
    dataset_args_builder: Callable  # (study_config, env_overrides) → dict

    # Config layers to merge (in order, later = lower priority via | merge)
    config_layers: list[dict] = field(default_factory=list)

    # Preset-specific default overrides (applied after config layers, before env)
    default_overrides: dict[str, Any] = field(default_factory=dict)

    # Whether to compute class-balanced focal alpha
    compute_focal_alpha: bool = True
    focal_alpha_params: dict[str, Any] = field(default_factory=dict)

    def build_config(self, env_overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge config layers → preset defaults → env overrides.

        Priority (highest wins): env_overrides > default_overrides > config_layers
        """
        import wonderwords

        # 1. Start with merged config layers
        config: dict[str, Any] = {}
        for layer in self.config_layers:
            config = config | layer

        # 2. Apply preset defaults (override config layers)
        config.update(self.default_overrides)

        # 3. Apply env overrides (highest priority).
        #    Filter out internal keys (prefixed with _) — they're handled below.
        for key, val in env_overrides.items():
            if not key.startswith("_"):
                config[key] = val

        # 4. Handle internal keys
        # LR scaling
        lr_scale = env_overrides.get("_train.lr_scale", 1.0)
        if lr_scale != 1.0:
            config["trainer.optimizer.lr"] = float(config["trainer.optimizer.lr"]) * lr_scale

        # Sequence length (used for dataset args, curriculum defaults)
        seq_length = env_overrides.get("_seq.length", config.get("_seq.length"))
        seq_step = env_overrides.get("_seq.step", config.get("_seq.step"))
        if seq_length is not None:
            config["_seq.length"] = seq_length
        if seq_step is not None:
            config["_seq.step"] = seq_step

        # Normalize TBPTT mode
        if "model.sequence.tbptt_mode" in config:
            config["model.sequence.tbptt_mode"] = normalize_tbptt_mode(
                config["model.sequence.tbptt_mode"]
            )

        # Resume handling
        resume_name = env_overrides.get("_resume.name", "")
        resume_path = env_overrides.get("_resume.model_path", "")
        if resume_name:
            config["experiment.name"] = resume_name
        else:
            r = wonderwords.RandomWord()
            random_word = r.word(include_parts_of_speech=["nouns"])
            config["experiment.name"] = (
                f"{self.name_prefix}_{random_word}_{config.get('model.channel_n', 24)}"
            )
        if resume_path:
            config["experiment.model_path"] = resume_path

        # Ensure experiment.description is set
        if "experiment.description" not in config:
            config["experiment.description"] = self.description_template

        # Store preset name for downstream auto-detection
        config["experiment.preset"] = self.name

        return config


# ═══════════════════════════════════════════════════════════════════════════
# iOCT dual-view dataset args builder
# ═══════════════════════════════════════════════════════════════════════════

_IOCT_DATA_ROOT = "/vol/data/OctreeNCA_Video/ioct_data"
_IOCT_DATASETS = ["peeling", "sri"]
_IOCT_VIEWS = ["A", "B"]


def _ioct_dual_dataset_args(study_config: dict, env_overrides: dict) -> dict:
    """Build dataset args for paired sequential iOCT."""
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
    }


# ═══════════════════════════════════════════════════════════════════════════
# Common iOCT config pieces
# ═══════════════════════════════════════════════════════════════════════════

def _ioct_common_overrides(num_classes: int = 7) -> dict[str, Any]:
    """Config keys shared between all iOCT dual-view presets."""
    input_size = (512, 512)
    return {
        "model.output_channels": num_classes,
        "model.input_channels": 1,
        "experiment.use_wandb": True,
        "experiment.wandb_project": "OctreeNCA_Video",
        "experiment.dataset.img_path": _IOCT_DATA_ROOT,
        "experiment.dataset.label_path": _IOCT_DATA_ROOT,
        "experiment.dataset.seed": 42,
        "experiment.data_split": [0.8, 0.1, 0.1],
        "experiment.dataset.input_size": input_size,
        "experiment.dataset.transform_mode": "none",

        "experiment.logging.also_eval_on_train": False,
        "experiment.save_interval": 3,
        "experiment.logging.evaluate_interval": 40,
        "experiment.task.score": [
            "src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
            "src.scores.PatchwiseIoUScore.PatchwiseIoUScore",
        ],

        "trainer.num_steps_per_epoch": 200,
        "trainer.batch_duplication": 1,
        "trainer.n_epochs": 100,
        "trainer.batch_size": 1,
        "trainer.gradient_accumulation": 16,
        "trainer.normalize_gradients": "all",
        "trainer.gradient_clip_val": 1.0,
        "trainer.use_amp": False,

        "model.backbone_class": "BasicNCA2DFast",
        "model.octree.separate_models": True,
        "model.octree.res_and_steps": build_octree_resolutions(input_size, 8, 20),
        "model.kernel_size": [3] * 5,
        "model.octree.warm_start_steps": 10,
        "model.hidden_size": 64,
        "model.normalization": "none",
        "model.apply_nonlin": "torch.nn.Softmax(dim=-1)",

        # Dual-view cross-fusion
        "model.dual_view.cross_fusion": "film",
        "model.dual_view.cross_strength": 0.5,
        "model.dual_view.cross_use_tanh": True,

        # Boundary loss
        "experiment.dataset.precompute_boundary_dist": True,
        "experiment.dataset.boundary_dist_classes": None,

        # Spike monitoring
        "experiment.logging.spike_watch.enabled": True,
        "experiment.logging.spike_watch.window": 50,
        "experiment.logging.spike_watch.zscore": 3.0,
        "experiment.logging.spike_watch.min_value": 0.2,
        "experiment.logging.spike_watch.max_images_per_epoch": 10,
        "experiment.logging.spike_watch.max_images_per_spike": 2,
        "experiment.logging.spike_watch.save_classes": list(range(1, num_classes)),

        # Phase timing
        "experiment.logging.batch_timing.enabled": False,
        "experiment.logging.batch_timing.print_interval": 20,
        "experiment.logging.batch_timing.warmup_steps": 5,
        "experiment.logging.phase_timing.enabled": True,
        "experiment.logging.phase_timing.print_interval": 20,
        "experiment.logging.phase_timing.warmup_steps": 5,

        # Internal keys for dataset builder
        "_seq.length": 3,
        "_seq.step": 1,
    }


def _ioct_losses(num_classes: int = 7, merge_to_binary: bool = False) -> dict[str, Any]:
    """Build the standard 4-loss setup with auto-computed focal alpha."""
    focal_alpha = compute_class_alpha_weights(
        data_root=_IOCT_DATA_ROOT,
        datasets=_IOCT_DATASETS,
        views=_IOCT_VIEWS,
        num_classes=num_classes,
        rgb_to_class=iOCTPairedSequentialDataset.RGB_TO_CLASS,
        max_samples=200,
        merge_to_binary=merge_to_binary,
    )
    print(f"Focal alpha (inverse-freq, normalized): {focal_alpha}")

    spike_keys = [
        "FocalLoss/loss",
        "BoundaryLoss/loss",
        "GeneralizedDiceLoss/overall",
    ] + [f"GeneralizedDiceLoss/mask_{i}" for i in range(max(0, num_classes - 1))]

    return {
        "trainer.losses": [
            "src.losses.DiceLoss.GeneralizedDiceLoss",
            "src.losses.LossFunctions.FocalLoss",
            "src.losses.DiceLoss.BoundaryLoss",
            "src.losses.OverflowLoss.OverflowLoss",
        ],
        "trainer.losses.parameters": [
            {"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05},
            {"gamma": 2.0, "alpha": focal_alpha, "ignore_index": 0, "reduction": "mean"},
            {
                "do_bg": False, "channel_last": True, "use_precomputed": True,
                "use_probabilities": False, "dist_clip": 20.0, "compute_missing_dist": False,
            },
            {},
        ],
        "trainer.loss_weights": [1.0, 1.0, 0.1, 1.0],
        "experiment.logging.spike_watch.keys": spike_keys,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Preset: ioct_dual_warm  (fine-tune with pretrained M1)
# ═══════════════════════════════════════════════════════════════════════════

_IOCT_NUM_CLASSES = max(iOCTPairedSequentialDataset.RGB_TO_CLASS.values()) + 1

_ioct_dual_warm_overrides = {
    **_ioct_common_overrides(_IOCT_NUM_CLASSES),

    # M1: pretrained, frozen by default
    "model.m1.pretrained_path": "",
    "model.m1.freeze": True,
    "model.m1.eval_mode": True,
    "model.m1.use_first_frame": True,
    "model.m1.use_t0_for_loss": False,
    "model.m1.use_probs": False,
    "model.m1.disable_backbone_tbptt": True,

    # M2: init from M1 by default
    "model.m2.init_from_m1": True,
    "model.m2.init_identity": False,
    "model.m2.share_backbone_with_m1": False,

    # Sequence TBPTT
    "model.sequence.tbptt_mode": "detach",

    # Channels
    "model.channel_n": 24,

    # EMA
    "trainer.ema": True,
    "trainer.ema.decay": 0.99,
}


def _build_ioct_dual_warm():
    overrides = dict(_ioct_dual_warm_overrides)
    overrides.update(_ioct_losses(_IOCT_NUM_CLASSES))
    return Preset(
        name="ioct_dual_warm",
        description_template="Dual-view iOCT warm-start with pretrained M1",
        name_prefix="WarmStart_M1Init_iOCT2D_dual",
        experiment_wrapper_class=EXP_DualViewWarmStart,
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


# ═══════════════════════════════════════════════════════════════════════════
# Preset: ioct_dual_b2b  (both M1+M2 from scratch)
# ═══════════════════════════════════════════════════════════════════════════

_ioct_dual_b2b_overrides = {
    **_ioct_common_overrides(_IOCT_NUM_CLASSES),

    # M1: from scratch, trains jointly
    "model.m1.pretrained_path": "",
    "model.m1.freeze": False,
    "model.m1.eval_mode": False,
    "model.m1.use_first_frame": True,
    "model.m1.use_t0_for_loss": True,   # Direct supervision at t=0
    "model.m1.use_probs": False,
    "model.m1.disable_backbone_tbptt": False,

    # M2: do NOT init from M1 by default
    "model.m2.init_from_m1": False,
    "model.m2.init_identity": False,
    "model.m2.share_backbone_with_m1": False,

    # Sequence TBPTT off by default (full backprop)
    "model.sequence.tbptt_mode": "off",

    # Channels
    "model.channel_n": 24,

    # Higher LR for from-scratch training
    "trainer.optimizer.lr": 2e-4,

    # EMA
    "trainer.ema": True,
    "trainer.ema.decay": 0.99,
}


def _build_ioct_dual_b2b():
    overrides = dict(_ioct_dual_b2b_overrides)
    overrides.update(_ioct_losses(_IOCT_NUM_CLASSES))
    return Preset(
        name="ioct_dual_b2b",
        description_template="Back-to-back dual-view: M1+M2 from random init, no pretrained M1",
        name_prefix="B2B_NoM1Pretrain_iOCT2D_dual",
        experiment_wrapper_class=EXP_DualViewWarmStart,
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


# ═══════════════════════════════════════════════════════════════════════════
# Preset: ioct_dual_b2b_binary  (M1+M2 from scratch, single merged class)
# ═══════════════════════════════════════════════════════════════════════════

_IOCT_BINARY_NUM_CLASSES = 2  # background + one merged foreground

_ioct_dual_b2b_binary_overrides = {
    **_ioct_common_overrides(_IOCT_BINARY_NUM_CLASSES),

    # Merge all segmentation classes into one foreground class
    "experiment.dataset.merge_all_classes": True,

    # M1: from scratch, trains jointly
    "model.m1.pretrained_path": "",
    "model.m1.freeze": False,
    "model.m1.eval_mode": False,
    "model.m1.use_first_frame": True,
    "model.m1.use_t0_for_loss": True,
    "model.m1.use_probs": False,
    "model.m1.disable_backbone_tbptt": False,

    # M2: do NOT init from M1
    "model.m2.init_from_m1": False,
    "model.m2.init_identity": False,
    "model.m2.share_backbone_with_m1": False,

    # Sequence TBPTT off by default (full backprop)
    "model.sequence.tbptt_mode": "off",

    # Channels
    "model.channel_n": 24,

    # Higher LR for from-scratch training
    "trainer.optimizer.lr": 2e-4,

    # EMA
    "trainer.ema": True,
    "trainer.ema.decay": 0.99,
}


def _build_ioct_dual_b2b_binary():
    overrides = dict(_ioct_dual_b2b_binary_overrides)
    overrides.update(_ioct_losses(_IOCT_BINARY_NUM_CLASSES, merge_to_binary=True))
    return Preset(
        name="ioct_dual_b2b_binary",
        description_template="Back-to-back dual-view: M1+M2 from scratch, all classes merged into one",
        name_prefix="B2B_Binary_iOCT2D_dual",
        experiment_wrapper_class=EXP_DualViewWarmStart,
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


# ═══════════════════════════════════════════════════════════════════════════
# Preset registry
# ═══════════════════════════════════════════════════════════════════════════

# Lazy builders to avoid computing focal alpha for unused presets
_PRESET_BUILDERS: dict[str, Callable[[], Preset]] = {
    "ioct_dual_warm": _build_ioct_dual_warm,
    "ioct_dual_b2b": _build_ioct_dual_b2b,
    "ioct_dual_b2b_binary": _build_ioct_dual_b2b_binary,
}

# Cache
_PRESET_CACHE: dict[str, Preset] = {}


def get_preset(name: str) -> Preset:
    """Get a preset by name, building it lazily on first access."""
    if name not in _PRESET_CACHE:
        if name not in _PRESET_BUILDERS:
            available = ", ".join(sorted(_PRESET_BUILDERS.keys()))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        _PRESET_CACHE[name] = _PRESET_BUILDERS[name]()
    return _PRESET_CACHE[name]


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    return sorted(_PRESET_BUILDERS.keys())


# Descriptions for --list-presets (avoids eagerly building presets)
_PRESET_DESCRIPTIONS: dict[str, str] = {
    "ioct_dual_warm": "Dual-view iOCT warm-start with pretrained M1",
    "ioct_dual_b2b": "Back-to-back dual-view: M1+M2 from random init, no pretrained M1",
    "ioct_dual_b2b_binary": "Back-to-back dual-view: M1+M2 from scratch, all classes merged into one",
}


def list_preset_descriptions() -> dict[str, str]:
    """Return preset name → description without building presets."""
    return dict(_PRESET_DESCRIPTIONS)
