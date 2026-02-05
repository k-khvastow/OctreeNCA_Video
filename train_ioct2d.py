from matplotlib import pyplot as plt
import configs
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.Study import Study
import wonderwords
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.datasets.Dataset_Base import Dataset_Base


# iOCT dataset root (contains peeling/ and sri/ subfolders)
DATA_ROOT = "/vol/data/OctreeNCA_Video/ioct_data"

# Optional: train only a subset of foreground classes (background 0 is always kept).
# Example: [1, 2] -> model outputs 3 classes (background + 2 selected).
SELECTED_CLASSES = None  # e.g. [1, 2]

DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]

r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


class iOCTDatasetForExperiment(Dataset_Base):
    """
    iOCT dataset adapter compatible with Experiment/DataSplit.
    Returns channel-last images (H, W, 1) and one-hot labels (H, W, C).
    """

    # Mapping from RGB values to class indices (from Dataset_iOCT_Sequential.py)
    RGB_TO_CLASS = {
        (0, 0, 0): 0,          # Background (black)
        (255, 0, 0): 1,        # Class 1 (red)
        (0, 255, 209): 2,      # Class 2 (cyan)
        (61, 255, 0): 3,       # Class 3 (green)
        (0, 78, 255): 4,       # Class 4 (blue)
        (255, 189, 0): 5,      # Class 5 (yellow/orange)
    }

    def __init__(
        self,
        data_root: str,
        datasets=DATASETS,
        views=VIEWS,
        num_classes: int = 6,
        input_size=(512, 512),
        class_subset=None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.views = views
        self.num_classes = num_classes
        self.size = input_size

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

        self.frames = []
        self.frames_dict = {}
        self._collect_frames()

    def _collect_frames(self):
        for dataset_name in self.datasets:
            for view in self.views:
                base_path = self.data_root / dataset_name / "Bscans-dt" / view
                image_dir = base_path / "Image"
                seg_dir = base_path / "Segmentation"

                if not image_dir.exists() or not seg_dir.exists():
                    print(f"Warning: Skipping {dataset_name}/{view} - directories not found")
                    continue

                for img_path in sorted(image_dir.glob("*.png")):
                    seg_path = seg_dir / img_path.name
                    if not seg_path.exists():
                        continue
                    frame_id = f"{dataset_name}_{view}_{img_path.stem}"
                    info = {
                        "id": frame_id,
                        "patient_id": f"{dataset_name}_{view}",
                        "dataset": dataset_name,
                        "view": view,
                        "image_path": img_path,
                        "seg_path": seg_path,
                    }
                    self.frames.append(info)
                    self.frames_dict[frame_id] = info

        print(f"Found {len(self.frames)} iOCT frames.")

    def getFilesInPath(self, path: str):
        return {k: {"id": k} for k in self.frames_dict.keys()}

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
        super().setPaths(images_path, images_list, labels_path, labels_list)
        self.frames = [self.frames_dict[uid] for uid in self.images_list if uid in self.frames_dict]
        print(f"Dataset split set. Active frames: {len(self.frames)}")

    def _rgb_to_class(self, rgb_seg: np.ndarray) -> np.ndarray:
        h, w = rgb_seg.shape[:2]
        class_seg = np.zeros((h, w), dtype=np.int64)
        for rgb_val, class_idx in self.RGB_TO_CLASS.items():
            mask = (rgb_seg[:, :, 0] == rgb_val[0]) & \
                   (rgb_seg[:, :, 1] == rgb_val[1]) & \
                   (rgb_seg[:, :, 2] == rgb_val[2])
            class_seg[mask] = class_idx
        return class_seg

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        info = self.frames[idx]

        img = np.array(Image.open(info["image_path"]))
        seg_rgb = np.array(Image.open(info["seg_path"]))

        # Convert to grayscale
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)

        # Convert segmentation to class indices
        seg = self._rgb_to_class(seg_rgb)

        # Resize to target size (H, W)
        if hasattr(self, "size") and self.size is not None and img.shape != tuple(self.size):
            img = np.array(Image.fromarray(img).resize(
                (self.size[1], self.size[0]), Image.BILINEAR
            ))
            seg = np.array(Image.fromarray(seg.astype(np.uint8)).resize(
                (self.size[1], self.size[0]), Image.NEAREST
            ))

        # Optional class subset remap
        if self.class_map is not None:
            remapped = np.zeros_like(seg)
            for src, dst in self.class_map.items():
                remapped[seg == src] = dst
            seg = remapped

        # Normalize image and add channel axis (H, W, 1)
        img = img.astype(np.float32) / 255.0
        img = img[..., None]

        # One-hot encode label to (H, W, C)
        seg_tensor = torch.from_numpy(seg).long()
        label_onehot = torch.nn.functional.one_hot(seg_tensor, num_classes=self.num_classes).numpy().astype(np.float32)

        return {
            "image": img,
            "label": label_onehot,
            "id": info["id"],
            "patient_id": info["patient_id"],
            "dataset": info["dataset"],
            "view": info["view"],
            "path": str(info["image_path"]),
        }


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
    study_config = {
        "experiment.name": r"OctreeNCA_iOCT_2D",
        "experiment.description": "Training OctreeNCA on iOCT 2D frames.",
        "model.output_channels": 6,
        "model.input_channels": 1,
        "experiment.use_wandb": True,
        "experiment.wandb_project": "OctreeNCA_Video",
        "experiment.dataset.img_path": DATA_ROOT,
        "experiment.dataset.label_path": DATA_ROOT,
        "experiment.dataset.seed": 42,
        "experiment.data_split": [0.8, 0.1, 0.1],
        "experiment.dataset.input_size": (512, 512),
        "experiment.dataset.transform_mode": "resize",
        "trainer.num_steps_per_epoch": 200,
        "trainer.batch_duplication": 1,
        "trainer.n_epochs": 10,
    }

    # Merge default configs
    study_config = study_config | configs.models.peso.peso_model_config
    study_config = study_config | configs.trainers.nca.nca_trainer_config
    study_config = study_config | configs.tasks.segmentation.segmentation_task_config
    study_config = study_config | configs.default.default_config

    # Experiment settings
    study_config["experiment.logging.also_eval_on_train"] = False
    study_config["experiment.save_interval"] = 3
    study_config["experiment.logging.evaluate_interval"] = 40
    study_config["experiment.task.score"] = [
        "src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
        "src.scores.PatchwiseIoUScore.PatchwiseIoUScore",
    ]
    study_config["trainer.n_epochs"] = 100

    # OctreeNCA Model Specifics
    steps = 10
    alpha = 1.0
    input_size = study_config["experiment.dataset.input_size"]
    study_config["model.octree.res_and_steps"] = _build_octree_resolutions(
        input_size, steps, int(alpha * 20)
    )
    study_config["model.kernel_size"] = [5] * len(study_config["model.octree.res_and_steps"])

    study_config["model.channel_n"] = 24
    study_config["model.hidden_size"] = 32
    study_config["trainer.batch_size"] = 6
    study_config["model.octree.separate_models"] = True
    study_config["model.backbone_class"] = "BasicNCA2DFast"

    dice_loss_weight = 1.0
    boundary_loss_weight = 0.2
    ema_decay = 0.99
    study_config["trainer.ema"] = ema_decay > 0.0
    study_config["trainer.ema.decay"] = ema_decay

    study_config["trainer.use_amp"] = True

    study_config["trainer.losses"] = [
        "src.losses.DiceLoss.GeneralizedDiceLoss",
        "src.losses.LossFunctions.FocalLoss",
        "src.losses.DiceLoss.BoundaryLoss",
    ]
    study_config["trainer.losses.parameters"] = [
        {"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05},
        {"gamma": 2.0, "alpha": None, "ignore_index": -100, "reduction": "mean"},
        {},
        {"do_bg": False, "channel_last": True, "use_precomputed": True, "use_probabilities": False, "dist_clip": 20.0},
    ]
    study_config["trainer.loss_weights"] = [
        dice_loss_weight,
        2.0 - dice_loss_weight,
        boundary_loss_weight,
    ]

    study_config["experiment.dataset.precompute_boundary_dist"] = False
    study_config["experiment.dataset.boundary_dist_classes"] = None

    study_config["model.normalization"] = "none"
    study_config["model.apply_nonlin"] = "torch.nn.Softmax(dim=1)"

    # Spike monitoring
    study_config["experiment.logging.spike_watch.enabled"] = True
    study_config["experiment.logging.spike_watch.keys"] = [
        "FocalLoss/loss",
        "nnUNetSoftDiceLoss/mask_3",
        "nnUNetSoftDiceLoss/mask_4",
    ]
    study_config["experiment.logging.spike_watch.window"] = 50
    study_config["experiment.logging.spike_watch.zscore"] = 3.0
    study_config["experiment.logging.spike_watch.min_value"] = 0.2
    study_config["experiment.logging.spike_watch.max_images_per_epoch"] = 10
    study_config["experiment.logging.spike_watch.max_images_per_spike"] = 2
    study_config["experiment.logging.spike_watch.save_classes"] = [1, 2, 3, 4, 5]

    # Optional class subset selection (foreground classes only; background is always class 0)
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
        study_config["experiment.logging.spike_watch.save_classes"] = list(range(1, len(cleaned) + 1))
    else:
        study_config["experiment.dataset.class_subset"] = None

    # Update experiment name with params
    study_config["experiment.name"] = f"iOCT2D_{random_word}_{study_config['model.channel_n']}"

    return study_config


def get_dataset_args(study_config):
    return {
        "data_root": DATA_ROOT,
        "datasets": DATASETS,
        "views": VIEWS,
        "num_classes": study_config["model.output_channels"],
        "input_size": study_config["experiment.dataset.input_size"],
        "class_subset": study_config.get("experiment.dataset.class_subset", None),
    }


if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    study = Study(study_config)
    exp = EXP_OctreeNCA().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTDatasetForExperiment,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
