"""Training script for dual-view iOCT OctreeNCA on BioProject13 peeling3 data.

Data layout (per-frame folder):
    <DATA_ROOT>/<frame_id>/00.png        – cross-section view A (grayscale 512×512)
    <DATA_ROOT>/<frame_id>/01.png        – cross-section view B
    <DATA_ROOT>/<frame_id>/00_seg.png    – segmentation mask for view A (uint8,  0–4)
    <DATA_ROOT>/<frame_id>/01_seg.png    – segmentation mask for view B
"""

import configs
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses
from src.models.Model_OctreeNCA_2d_dual_view import OctreeNCA2DDualView
from src.agents.Agent_MedNCA_DualView import MedNCADualViewAgent
from src.utils.Study import Study
import wonderwords
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch

from src.datasets.Dataset_Base import Dataset_Base
from src.utils.DistanceMaps import signed_distance_map


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/khvastow/ioct_data/canvas/peeling3/iOCT/Bscan"

# The segmentation masks in canvas/peeling3 use sparse class IDs
# {0, 1, 2, 3, 4, 5, 8, 13}.  We only keep classes 0-4; anything
# above 4 is mapped to background (0).
NUM_CLASSES = 5
MAX_VALID_CLASS = NUM_CLASSES - 1  # 4

# Optional: train only a subset of foreground classes (background 0 is always kept).
SELECTED_CLASSES = None  # e.g. [1, 2]

# ---------------------------------------------------------------------------
# Torch compile controls
# ---------------------------------------------------------------------------
ENABLE_TORCH_COMPILE = os.getenv("IOCT_TORCH_COMPILE", "0") == "1"
TORCH_COMPILE_MODE = os.getenv("IOCT_TORCH_COMPILE_MODE", "reduce-overhead")
TORCH_COMPILE_BACKEND = os.getenv("IOCT_TORCH_COMPILE_BACKEND", "inductor")
TORCH_COMPILE_DYNAMIC = os.getenv("IOCT_TORCH_COMPILE_DYNAMIC", "0") == "1"

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
TORCH_COMPILE_FULLGRAPH = os.getenv("IOCT_TORCH_COMPILE_FULLGRAPH", "0") == "1"
ENABLE_GRAD_NORM_LOGGING = os.getenv("IOCT_TRACK_GRAD_NORM", "0") == "1"
_tbptt_env = os.getenv("IOCT_TBPTT_STEPS", "").strip()
BACKBONE_TBPTT_STEPS = int(_tbptt_env) if _tbptt_env else None

r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


# ---------------------------------------------------------------------------
# Experiment wrapper
# ---------------------------------------------------------------------------
class EXP_OctreeNCA_DualView_Peeling3(ExperimentWrapper):
    def createExperiment(self, study_config: dict, detail_config: dict = {},
                         dataset_class=None, dataset_args: dict = {}):
        config = study_config
        if dataset_class is None:
            raise ValueError("dataset_class must be provided")
        model = OctreeNCA2DDualView(config)
        agent = MedNCADualViewAgent(model)
        loss_function = WeightedLosses(config)
        return super().createExperiment(config, model, agent, dataset_class,
                                        dataset_args, loss_function)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Peeling3DualViewDataset(Dataset_Base):
    """Paired dual-view iOCT dataset for BioProject13 peeling3 data.

    Each frame folder contains two cross-section images (``00.png``, ``01.png``)
    and their corresponding segmentation masks (``00_seg.png``, ``01_seg.png``)
    stored as single-channel uint8 class-index images (values 0 – ``num_classes-1``).

    Returns a dict compatible with ``MedNCADualViewAgent``:
        image_a / image_b  : (H, W, 1) float32 in [0, 1]
        label_a / label_b  : (H, W, C) float32 one-hot
    """

    def __init__(
        self,
        data_root: str,
        num_classes: int = NUM_CLASSES,
        input_size=(512, 512),
        class_subset=None,
        precompute_boundary_dist: bool = False,
        boundary_dist_classes=None,
        max_samples: int = None,
        sparse_class_ids=None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.num_classes = num_classes

        # Build a look-up table to remap sparse/non-contiguous mask ids
        # to contiguous 0..N-1
        self._remap_lut = None
        if sparse_class_ids is not None:
            max_id = max(sparse_class_ids)
            lut = np.zeros(max_id + 1, dtype=np.uint8)
            for new_id, old_id in enumerate(sparse_class_ids):
                lut[old_id] = new_id
            self._remap_lut = lut
        self.size = input_size
        self.precompute_boundary_dist = precompute_boundary_dist
        self.boundary_dist_classes = boundary_dist_classes
        self.max_samples = max_samples

        # Required by agents
        self.slice = -1
        self.delivers_channel_axis = True
        self.is_rgb = False

        # Optional class subset selection  (background 0 is always kept)
        self.class_subset = None
        self.class_map = None
        if class_subset is not None:
            cleaned = []
            seen = set()
            for c in class_subset:
                c_int = int(c)
                if c_int == 0:
                    continue
                if c_int not in seen:
                    cleaned.append(c_int)
                    seen.add(c_int)
            if len(cleaned) == 0:
                raise ValueError("class_subset must include at least one non-zero class id.")
            self.class_subset = cleaned
            self.class_map = {c: i + 1 for i, c in enumerate(self.class_subset)}
            self.num_classes = len(self.class_subset) + 1

        self.pairs = []
        self.pairs_dict = {}
        self._collect_pairs()

    # ---- discovery ----------------------------------------------------------
    def _collect_pairs(self):
        """Walk ``data_root`` and build the list of valid frame pairs."""
        frame_dirs = sorted(
            [d for d in self.data_root.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )

        for frame_dir in frame_dirs:
            img_a = frame_dir / "00.png"
            seg_a = frame_dir / "00_seg.png"
            img_b = frame_dir / "01.png"
            seg_b = frame_dir / "01_seg.png"

            if not (img_a.exists() and seg_a.exists() and img_b.exists() and seg_b.exists()):
                print(f"Warning: Skipping {frame_dir.name} – missing files")
                continue

            pair_id = frame_dir.name
            info = {
                "id": pair_id,
                "patient_id": pair_id,
                "dataset": "peeling3",
                "frame": pair_id,
                "view_a": "00",
                "view_b": "01",
                "image_path_a": img_a,
                "seg_path_a": seg_a,
                "image_path_b": img_b,
                "seg_path_b": seg_b,
            }
            self.pairs.append(info)
            self.pairs_dict[pair_id] = info

        if self.max_samples is not None and self.max_samples > 0:
            self.pairs = self.pairs[: self.max_samples]
            self.pairs_dict = {p["id"]: p for p in self.pairs}

        print(f"Found {len(self.pairs)} paired peeling3 frames (00 + 01).")

    # ---- interface required by Experiment / DataSplit -----------------------
    def getFilesInPath(self, path: str):
        return {k: {"id": k} for k in self.pairs_dict.keys()}

    def setPaths(self, images_path: str, images_list: list,
                 labels_path: str, labels_list: list) -> None:
        super().setPaths(images_path, images_list, labels_path, labels_list)
        self.pairs = [self.pairs_dict[uid] for uid in self.images_list
                      if uid in self.pairs_dict]
        print(f"Dataset split set. Active pairs: {len(self.pairs)}")

    # ---- I/O helpers --------------------------------------------------------
    def _load_view(self, img_path: Path, seg_path: Path):
        img = np.array(Image.open(img_path))
        seg = np.array(Image.open(seg_path))

        # Grayscale normalisation – handle potential RGB images gracefully
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)

        expected_size = tuple(self.size)
        if img.shape != expected_size:
            raise ValueError(
                f"Image shape {img.shape} != expected {expected_size} for {img_path}."
            )
        if seg.shape[:2] != expected_size:
            raise ValueError(
                f"Segmentation shape {seg.shape} != expected {expected_size} for {seg_path}."
            )

        # Clamp: any class id > MAX_VALID_CLASS becomes background (0)
        seg[seg > MAX_VALID_CLASS] = 0

        # Remap sparse class ids to contiguous 0..N-1
        if self._remap_lut is not None:
            seg = self._remap_lut[seg]

        # Remap classes if a subset was requested
        if self.class_map is not None:
            remapped = np.zeros_like(seg)
            for src, dst in self.class_map.items():
                remapped[seg == src] = dst
            seg = remapped

        img = img.astype(np.float32) / 255.0
        img = img[..., None]  # (H, W, 1)

        seg_tensor = torch.from_numpy(seg.astype(np.int64)).long()
        max_class = int(seg_tensor.max().item())
        if max_class >= self.num_classes:
            raise ValueError(
                f"Segmentation class id {max_class} >= num_classes ({self.num_classes}). "
                "Update model.output_channels or class_subset."
            )
        label_onehot = (
            torch.nn.functional.one_hot(seg_tensor, num_classes=self.num_classes)
            .numpy()
            .astype(np.float32)
        )  # (H, W, C)

        label_dist = None
        if self.precompute_boundary_dist:
            label_dist = signed_distance_map(
                label_onehot,
                class_ids=self.boundary_dist_classes,
                channel_first=False,
                compact=False,
                dtype=np.float32,
            )

        return img, label_onehot, label_dist

    # ---- __len__ / __getitem__ ----------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        info = self.pairs[idx]

        img_a, lbl_a, dist_a = self._load_view(info["image_path_a"], info["seg_path_a"])
        img_b, lbl_b, dist_b = self._load_view(info["image_path_b"], info["seg_path_b"])

        sample = {
            "image_a": img_a,
            "label_a": lbl_a,
            "image_b": img_b,
            "label_b": lbl_b,
            # Compatibility aliases for the generic transform pipeline
            "image": img_a,
            "label": lbl_a,
            "id": info["id"],
            "patient_id": info["patient_id"],
            "dataset": info["dataset"],
            "view_a": info["view_a"],
            "view_b": info["view_b"],
            "path_a": str(info["image_path_a"]),
            "path_b": str(info["image_path_b"]),
        }

        if dist_a is not None:
            sample["label_dist_a"] = dist_a
            sample["label_dist"] = dist_a
        if dist_b is not None:
            sample["label_dist_b"] = dist_b

        return sample


# ---------------------------------------------------------------------------
# Octree resolution builder (identical to train_ioct2d_dual_view.py)
# ---------------------------------------------------------------------------
def _build_octree_resolutions(input_size, steps, final_steps):
    h, w = input_size
    resolutions = []
    for _ in range(4):
        resolutions.append([h, w])
        h = max(1, h // 2)
        w = max(1, w // 2)
    res_and_steps = []
    for i, res in enumerate(resolutions):
        if i == len(resolutions) - 1:
            res_and_steps.append([res, final_steps])
        else:
            res_and_steps.append([res, steps])
    return res_and_steps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def get_study_config():
    num_classes = NUM_CLASSES

    # Apply class subset if requested
    selected_classes = SELECTED_CLASSES
    if selected_classes is not None:
        cleaned = [int(c) for c in selected_classes if int(c) != 0]
        cleaned = list(dict.fromkeys(cleaned))  # deduplicate, preserve order
        if not cleaned:
            raise ValueError("SELECTED_CLASSES must include at least one non-zero class id.")
        num_classes = len(cleaned) + 1
    else:
        cleaned = None

    study_config = {
        "experiment.name": r"OctreeNCA_iOCT_2D_DualView_Peeling3",
        "experiment.description": "Dual-view iOCT (00+01) OctreeNCA segmentation on peeling3.",
        "model.output_channels": num_classes,
        "model.input_channels": 1,
        "experiment.use_wandb": True,
        "experiment.wandb_project": "OctreeNCA_Video",
        "experiment.dataset.img_path": DATA_ROOT,
        "experiment.dataset.label_path": DATA_ROOT,
        "experiment.dataset.seed": 42,
        "experiment.data_split": [0.8, 0.1, 0.1],
        "experiment.dataset.input_size": (512, 512),
        "experiment.dataset.transform_mode": "none",
        "trainer.num_steps_per_epoch": 200,
        "trainer.batch_duplication": 1,
        "trainer.n_epochs": 10,
    }

    study_config = study_config | configs.models.peso.peso_model_config
    study_config = study_config | configs.trainers.nca.nca_trainer_config
    study_config = study_config | configs.tasks.segmentation.segmentation_task_config
    study_config = study_config | configs.default.default_config

    study_config["experiment.logging.also_eval_on_train"] = False
    study_config["experiment.save_interval"] = 3
    study_config["experiment.logging.evaluate_interval"] = 40
    study_config["experiment.task.score"] = [
        "src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
        "src.scores.PatchwiseIoUScore.PatchwiseIoUScore",
    ]
    study_config["trainer.n_epochs"] = 100

    # OctreeNCA Model specifics
    steps = (10, 20)
    alpha = 1.0
    final_steps = (int(alpha * 15), int(alpha * 20))
    input_size = study_config["experiment.dataset.input_size"]
    study_config["model.octree.res_and_steps"] = _build_octree_resolutions(
        input_size, steps, final_steps
    )
    study_config["model.kernel_size"] = 3
    study_config["model.channel_n"] = 32
    study_config["model.hidden_size"] = 64
    study_config["trainer.batch_size"] = 2
    study_config["model.octree.separate_models"] = False
    study_config["model.backbone_class"] = "BasicNCA2DFast"

    # Dual-view fusion
    study_config["model.dual_view.cross_fusion"] = "film"
    study_config["model.dual_view.cross_strength"] = 0.5
    study_config["model.dual_view.cross_use_tanh"] = True

    dice_loss_weight = 1.0
    boundary_loss_weight = 0.2
    ema_decay = 0.99
    study_config["trainer.ema"] = ema_decay > 0.0
    study_config["trainer.ema.decay"] = ema_decay
    study_config["trainer.use_amp"] = True

    study_config["trainer.losses"] = [
        "src.losses.DiceLoss.nnUNetSoftDiceLossSum",
        "src.losses.LossFunctions.FocalLoss",
        "src.losses.DiceLoss.BoundaryLoss",
    ]
    study_config["trainer.losses.parameters"] = [
        {"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True,
         "do_bg": False, "smooth": 1e-05},
        {"gamma": 2.0, "alpha": None, "ignore_index": 0, "reduction": "mean"},
        {
            "do_bg": False,
            "channel_last": True,
            "use_precomputed": True,
            "use_probabilities": False,
            "dist_clip": 20.0,
            "compute_missing_dist": False,
        },
    ]
    study_config["trainer.loss_weights"] = [
        dice_loss_weight, 2.0 - dice_loss_weight, boundary_loss_weight
    ]

    study_config["experiment.dataset.precompute_boundary_dist"] = True
    study_config["experiment.dataset.boundary_dist_classes"] = None
    study_config["experiment.dataset.class_subset"] = cleaned

    study_config["model.normalization"] = "none"
    study_config["model.apply_nonlin"] = "torch.nn.Softmax(dim=-1)"
    study_config["performance.compile"] = ENABLE_TORCH_COMPILE
    study_config["performance.compile.mode"] = TORCH_COMPILE_MODE
    study_config["performance.compile.backend"] = TORCH_COMPILE_BACKEND
    study_config["performance.compile.dynamic"] = TORCH_COMPILE_DYNAMIC
    study_config["performance.compile.fullgraph"] = TORCH_COMPILE_FULLGRAPH
    study_config["experiment.logging.track_gradient_norm"] = ENABLE_GRAD_NORM_LOGGING
    study_config["trainer.normalize_gradients"] = None
    study_config["model.backbone.tbptt_steps"] = BACKBONE_TBPTT_STEPS

    study_config["experiment.name"] = (
        f"iOCT2D_dual_peeling3_{random_word}_{study_config['model.channel_n']}"
    )
    return study_config


def get_dataset_args(study_config):
    return {
        "data_root": DATA_ROOT,
        "num_classes": study_config["model.output_channels"],
        "input_size": study_config["experiment.dataset.input_size"],
        "class_subset": study_config.get("experiment.dataset.class_subset", None),
        "precompute_boundary_dist": study_config.get(
            "experiment.dataset.precompute_boundary_dist", False
        ),
        "boundary_dist_classes": study_config.get(
            "experiment.dataset.boundary_dist_classes", None
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    print(
        "Runtime config:",
        {
            "data_root": DATA_ROOT,
            "num_classes": study_config["model.output_channels"],
            "enabled": study_config.get("performance.compile", False),
            "mode": study_config.get("performance.compile.mode"),
            "backend": study_config.get("performance.compile.backend"),
            "dynamic": study_config.get("performance.compile.dynamic"),
            "fullgraph": study_config.get("performance.compile.fullgraph"),
            "track_grad_norm": study_config.get(
                "experiment.logging.track_gradient_norm", False
            ),
            "tbptt_steps": study_config.get("model.backbone.tbptt_steps", None),
        },
    )

    study = Study(study_config)
    exp = EXP_OctreeNCA_DualView_Peeling3().createExperiment(
        study_config,
        detail_config={},
        dataset_class=Peeling3DualViewDataset,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
