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


# iOCT dataset root (contains peeling/ and sri/ subfolders)
DATA_ROOT = "/vol/data/OctreeNCA_Video/ioct_data"

# Optional: train only a subset of foreground classes (background 0 is always kept).
# Example: [1, 2] -> model outputs 3 classes (background + 2 selected).
SELECTED_CLASSES = None  # e.g. [1, 2]

DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]

# Torch compile controls for this training script.
ENABLE_TORCH_COMPILE = os.getenv("IOCT_TORCH_COMPILE", "1") == "1"
TORCH_COMPILE_MODE = os.getenv("IOCT_TORCH_COMPILE_MODE", "max-autotune")
TORCH_COMPILE_BACKEND = os.getenv("IOCT_TORCH_COMPILE_BACKEND", "inductor")
TORCH_COMPILE_DYNAMIC = os.getenv("IOCT_TORCH_COMPILE_DYNAMIC", "0") == "1"
TORCH_COMPILE_FULLGRAPH = os.getenv("IOCT_TORCH_COMPILE_FULLGRAPH", "0") == "1"
ENABLE_GRAD_NORM_LOGGING = os.getenv("IOCT_TRACK_GRAD_NORM", "0") == "1"
_tbptt_env = os.getenv("IOCT_TBPTT_STEPS", "").strip()
BACKBONE_TBPTT_STEPS = int(_tbptt_env) if _tbptt_env else None

r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


class EXP_OctreeNCA_DualView(ExperimentWrapper):
    def createExperiment(self, study_config: dict, detail_config: dict = {}, dataset_class=None, dataset_args: dict = {}):
        config = study_config
        if dataset_class is None:
            raise ValueError("dataset_class must be provided")
        model = OctreeNCA2DDualView(config)
        agent = MedNCADualViewAgent(model)
        loss_function = WeightedLosses(config)
        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)


class iOCTPairedViewsDatasetForExperiment(Dataset_Base):
    """
    iOCT paired-views dataset adapter compatible with Experiment/DataSplit.

    Returns:
      - image_a, image_b: channel-last (H, W, 1)
      - label_a, label_b: one-hot (H, W, C)
    """

    RGB_TO_CLASS = {
        (0, 0, 0): 0,          # Background (black)
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
        datasets=DATASETS,
        views=VIEWS,
        num_classes: int = 7,
        input_size=(512, 512),
        class_subset=None,
        precompute_boundary_dist: bool = False,
        boundary_dist_classes=None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.views = list(views)
        if len(self.views) != 2:
            raise ValueError(f"Expected exactly two views, got {self.views}.")
        self.view_a, self.view_b = self.views[0], self.views[1]

        self.num_classes = num_classes
        self.size = input_size
        self.precompute_boundary_dist = precompute_boundary_dist
        self.boundary_dist_classes = boundary_dist_classes

        # Required by agents
        self.slice = -1
        self.delivers_channel_axis = True
        self.is_rgb = False

        # Optional class subset selection
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

    def _collect_pairs(self):
        def _sort_key(name: str):
            stem = Path(name).stem
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        for dataset_name in self.datasets:
            base_path = self.data_root / dataset_name / "Bscans-dt"
            img_dir_a = base_path / self.view_a / "Image"
            seg_dir_a = base_path / self.view_a / "Segmentation"
            img_dir_b = base_path / self.view_b / "Image"
            seg_dir_b = base_path / self.view_b / "Segmentation"

            if not (img_dir_a.exists() and seg_dir_a.exists() and img_dir_b.exists() and seg_dir_b.exists()):
                print(f"Warning: Skipping {dataset_name} - paired view directories not found ({self.view_a}, {self.view_b})")
                continue

            names_a = {p.name for p in img_dir_a.glob("*.png") if (seg_dir_a / p.name).exists()}
            names_b = {p.name for p in img_dir_b.glob("*.png") if (seg_dir_b / p.name).exists()}
            common = sorted(names_a & names_b, key=_sort_key)

            for name in common:
                stem = Path(name).stem
                pair_id = f"{dataset_name}_{stem}"
                info = {
                    "id": pair_id,
                    "patient_id": pair_id,
                    "dataset": dataset_name,
                    "frame": stem,
                    "view_a": self.view_a,
                    "view_b": self.view_b,
                    "image_path_a": img_dir_a / name,
                    "seg_path_a": seg_dir_a / name,
                    "image_path_b": img_dir_b / name,
                    "seg_path_b": seg_dir_b / name,
                }
                self.pairs.append(info)
                self.pairs_dict[pair_id] = info

        print(f"Found {len(self.pairs)} paired iOCT frames ({self.view_a}+{self.view_b}).")

    def getFilesInPath(self, path: str):
        return {k: {"id": k} for k in self.pairs_dict.keys()}

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
        super().setPaths(images_path, images_list, labels_path, labels_list)
        self.pairs = [self.pairs_dict[uid] for uid in self.images_list if uid in self.pairs_dict]
        print(f"Dataset split set. Active pairs: {len(self.pairs)}")

    def _rgb_to_class(self, rgb_seg: np.ndarray) -> np.ndarray:
        h, w = rgb_seg.shape[:2]
        class_seg = np.zeros((h, w), dtype=np.int64)
        for rgb_val, class_idx in self.RGB_TO_CLASS.items():
            mask = (rgb_seg[:, :, 0] == rgb_val[0]) & (rgb_seg[:, :, 1] == rgb_val[1]) & (rgb_seg[:, :, 2] == rgb_val[2])
            class_seg[mask] = class_idx
        return class_seg

    def __len__(self):
        return len(self.pairs)

    def _load_view(self, img_path: Path, seg_path: Path):
        img = np.array(Image.open(img_path))
        seg_rgb = np.array(Image.open(seg_path))

        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)

        seg = self._rgb_to_class(seg_rgb)

        expected_size = tuple(self.size)
        if img.shape != expected_size:
            raise ValueError(f"Image shape {img.shape} does not match expected size {expected_size} for {img_path}.")
        if seg.shape != expected_size:
            raise ValueError(f"Segmentation shape {seg.shape} does not match expected size {expected_size} for {seg_path}.")

        if self.class_map is not None:
            remapped = np.zeros_like(seg)
            for src, dst in self.class_map.items():
                remapped[seg == src] = dst
            seg = remapped

        img = img.astype(np.float32) / 255.0
        img = img[..., None]  # (H, W, 1)

        seg_tensor = torch.from_numpy(seg).long()
        max_class = int(seg_tensor.max().item())
        if max_class >= self.num_classes:
            raise ValueError(
                f"Segmentation class id {max_class} is >= num_classes ({self.num_classes}). "
                "Update model.output_channels or class_subset to cover all label ids."
            )
        label_onehot = torch.nn.functional.one_hot(seg_tensor, num_classes=self.num_classes).numpy().astype(np.float32)

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

    def __getitem__(self, idx):
        info = self.pairs[idx]

        img_a, lbl_a, dist_a = self._load_view(info["image_path_a"], info["seg_path_a"])
        img_b, lbl_b, dist_b = self._load_view(info["image_path_b"], info["seg_path_b"])

        sample = {
            "image_a": img_a,
            "label_a": lbl_a,
            "image_b": img_b,
            "label_b": lbl_b,
            # Compatibility aliases for the generic Experiment transform pipeline,
            # which expects 'image'/'label' keys for batchgenerators + NumpyToTensor.
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


def _build_octree_resolutions(input_size, steps, final_steps):
    h, w = input_size
    resolutions = []
    for _ in range(5):
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


def get_study_config():
    full_num_classes = max(iOCTPairedViewsDatasetForExperiment.RGB_TO_CLASS.values()) + 1
    study_config = {
        "experiment.name": r"OctreeNCA_iOCT_2D_DualView",
        "experiment.description": "Dual-view iOCT (A+B) OctreeNCA segmentation.",
        "model.output_channels": full_num_classes,
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
    steps = 10
    alpha = 1.0
    input_size = study_config["experiment.dataset.input_size"]
    study_config["model.octree.res_and_steps"] = _build_octree_resolutions(input_size, steps, int(alpha * 20))
    study_config["model.kernel_size"] = [3] * len(study_config["model.octree.res_and_steps"])

    study_config["model.channel_n"] = 24
    study_config["model.hidden_size"] = 64
    study_config["trainer.batch_size"] = 4
    study_config["model.octree.separate_models"] = True
    study_config["model.backbone_class"] = "BasicNCA2DFast"

    # Dual-view fusion settings (keeps images separate; fuses via hidden-state FiLM).
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
        {"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05},
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
    study_config["trainer.loss_weights"] = [dice_loss_weight, 2.0 - dice_loss_weight, boundary_loss_weight]

    study_config["experiment.dataset.precompute_boundary_dist"] = True
    study_config["experiment.dataset.boundary_dist_classes"] = None

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

    selected_classes = SELECTED_CLASSES
    if selected_classes is not None:
        cleaned = []
        seen = set()
        for c in selected_classes:
            c_int = int(c)
            if c_int == 0:
                continue
            if c_int not in seen:
                cleaned.append(c_int)
                seen.add(c_int)
        if len(cleaned) == 0:
            raise ValueError("SELECTED_CLASSES must include at least one non-zero class id.")
        study_config["experiment.dataset.class_subset"] = cleaned
        study_config["model.output_channels"] = len(cleaned) + 1
    else:
        study_config["experiment.dataset.class_subset"] = None

    study_config["experiment.name"] = f"iOCT2D_dual_{random_word}_{study_config['model.channel_n']}"
    return study_config


def get_dataset_args(study_config):
    return {
        "data_root": DATA_ROOT,
        "datasets": DATASETS,
        "views": VIEWS,
        "num_classes": study_config["model.output_channels"],
        "input_size": study_config["experiment.dataset.input_size"],
        "class_subset": study_config.get("experiment.dataset.class_subset", None),
        "precompute_boundary_dist": study_config.get("experiment.dataset.precompute_boundary_dist", False),
        "boundary_dist_classes": study_config.get("experiment.dataset.boundary_dist_classes", None),
    }


if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    print(
        "Runtime config:",
        {
            "enabled": study_config.get("performance.compile", False),
            "mode": study_config.get("performance.compile.mode"),
            "backend": study_config.get("performance.compile.backend"),
            "dynamic": study_config.get("performance.compile.dynamic"),
            "fullgraph": study_config.get("performance.compile.fullgraph"),
            "track_grad_norm": study_config.get("experiment.logging.track_gradient_norm", False),
            "tbptt_steps": study_config.get("model.backbone.tbptt_steps", None),
        },
    )

    study = Study(study_config)
    exp = EXP_OctreeNCA_DualView().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTPairedViewsDatasetForExperiment,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
