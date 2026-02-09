import configs
from src.utils.WarmStartM1InitConfig import EXP_OctreeNCA_WarmStart_M1Init
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

# Set this to your M1 checkpoint (.pth) or directory containing model.pth.
M1_CHECKPOINT_PATH = "/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_hospital_24_Training OctreeNCA on iOCT 2D frames./models/epoch_99/model.pth"

SEQUENCE_LENGTH = 5
SEQUENCE_STEP = 5

DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]

r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


class iOCTSequentialDatasetForExperiment(Dataset_Base):
    """
    iOCT sequential dataset adapter compatible with Experiment/DataSplit.
    Returns images (T, 1, H, W) and one-hot labels (T, C, H, W).
    """

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
        sequence_length: int = 5,
        sequence_step: int = 1,
        num_classes: int = 6,
        input_size=(512, 512),
        class_subset=None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.views = views
        self.sequence_length = int(sequence_length)
        self.sequence_step = int(sequence_step)
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

        self.sequences = []
        self.sequences_dict = {}
        self._collect_sequences()

    def _collect_sequences(self):
        required_span = (self.sequence_length - 1) * self.sequence_step + 1
        for dataset_name in self.datasets:
            for view in self.views:
                base_path = self.data_root / dataset_name / "Bscans-dt" / view
                image_dir = base_path / "Image"
                seg_dir = base_path / "Segmentation"

                if not image_dir.exists() or not seg_dir.exists():
                    print(f"Warning: Skipping {dataset_name}/{view} - directories not found")
                    continue

                image_files = sorted(image_dir.glob("*.png"))
                if len(image_files) < required_span:
                    continue

                for i in range(0, len(image_files) - required_span + 1):
                    indices = [i + j * self.sequence_step for j in range(self.sequence_length)]
                    window_files = [image_files[idx] for idx in indices]

                    valid = True
                    for img_path in window_files:
                        seg_path = seg_dir / img_path.name
                        if not seg_path.exists():
                            valid = False
                            break

                    if not valid:
                        continue

                    start_stem = window_files[0].stem
                    seq_id = f"{dataset_name}_{view}_{start_stem}"
                    seq_data = {
                        "id": seq_id,
                        "patient_id": f"{dataset_name}_{view}",
                        "dataset": dataset_name,
                        "view": view,
                        "image_paths": window_files,
                        "seg_dir": seg_dir,
                    }
                    self.sequences.append(seq_data)
                    self.sequences_dict[seq_id] = seq_data

        print(
            f"Found {len(self.sequences)} iOCT sequences "
            f"(length={self.sequence_length}, step={self.sequence_step})."
        )

    def getFilesInPath(self, path: str):
        return {k: {"id": k} for k in self.sequences_dict.keys()}

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
        super().setPaths(images_path, images_list, labels_path, labels_list)
        self.sequences = [self.sequences_dict[uid] for uid in self.images_list if uid in self.sequences_dict]
        print(f"Dataset split set. Active sequences: {len(self.sequences)}")

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
        return len(self.sequences)

    def __getitem__(self, idx):
        meta = self.sequences[idx]
        imgs = []
        masks = []

        for img_path in meta["image_paths"]:
            seg_path = meta["seg_dir"] / img_path.name

            img = np.array(Image.open(img_path))
            seg_rgb = np.array(Image.open(seg_path))

            if img.ndim == 3:
                img = np.mean(img, axis=2).astype(np.uint8)

            seg = self._rgb_to_class(seg_rgb)

            if hasattr(self, "size") and self.size is not None and img.shape != tuple(self.size):
                img = np.array(Image.fromarray(img).resize(
                    (self.size[1], self.size[0]), Image.BILINEAR
                ))
                seg = np.array(Image.fromarray(seg.astype(np.uint8)).resize(
                    (self.size[1], self.size[0]), Image.NEAREST
                ))

            if self.class_map is not None:
                remapped = np.zeros_like(seg)
                for src, dst in self.class_map.items():
                    remapped[seg == src] = dst
                seg = remapped

            img = img.astype(np.float32) / 255.0
            img = img[None, :, :]

            imgs.append(img)
            masks.append(seg)

        imgs_np = np.stack(imgs)
        masks_np = np.stack(masks)

        masks_tensor = torch.from_numpy(masks_np).long()
        masks_onehot = torch.nn.functional.one_hot(masks_tensor, num_classes=self.num_classes)
        masks_onehot = masks_onehot.permute(0, 3, 1, 2).float().numpy()

        return {
            "image": imgs_np,
            "label": masks_onehot,
            "id": meta["id"],
        }


def _build_octree_resolutions(input_size, steps, final_steps, first_steps_multiplier=2):
    h, w = input_size
    resolutions = []
    for _ in range(5):
        resolutions.append([h, w])
        h = max(1, h // 2)
        w = max(1, w // 2)
    res_and_steps = []
    for i, res in enumerate(resolutions):
        if i == 0:
            res_and_steps.append([res, steps * first_steps_multiplier])
        elif i == len(resolutions) - 1:
            res_and_steps.append([res, final_steps])
        else:
            res_and_steps.append([res, steps])
    return res_and_steps


def get_study_config():
    study_config = {
        "experiment.name": r"OctreeNCA_iOCT_2D_WarmStart_M1Init",
        "experiment.description": "Warm start with M1 hidden state init, then M2 for sequential iOCT frames.",
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
        "trainer.n_epochs": 100,
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

    # Model specifics
    steps = 10
    alpha = 1.0
    input_size = study_config["experiment.dataset.input_size"]
    study_config["model.backbone_class"] = "BasicNCA2DFast"
    study_config["model.octree.separate_models"] = True
    study_config["model.octree.res_and_steps"] = _build_octree_resolutions(
        input_size, steps, int(alpha * 20 / 2)
    )
    study_config["model.kernel_size"] = [5] * len(study_config["model.octree.res_and_steps"])
    study_config["model.octree.warm_start_steps"] = 10
    study_config["model.channel_n"] = 24
    study_config["model.hidden_size"] = 32
    study_config["trainer.batch_size"] = 2
    study_config["trainer.gradient_accumulation"] = 8
    study_config["trainer.normalize_gradients"] = "all"

    # M1 init options
    study_config["model.m1.pretrained_path"] = M1_CHECKPOINT_PATH
    study_config["model.m1.freeze"] = True
    study_config["model.m1.use_first_frame"] = True
    study_config["model.m1.use_t0_for_loss"] = False
    study_config["model.m1.use_probs"] = False

    dice_loss_weight = 1.0
    boundary_loss_weight = 0.1
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

    # Spike monitoring (per-batch class counts + save batches on spikes)
    study_config["experiment.logging.spike_watch.enabled"] = True
    study_config["experiment.logging.spike_watch.keys"] = [
        "CrossEntropyLossWrapper/loss",
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

    study_config["experiment.name"] = f"WarmStart_M1Init_iOCT2D_{random_word}_{study_config['model.channel_n']}"

    return study_config


def get_dataset_args(study_config):
    return {
        "data_root": DATA_ROOT,
        "datasets": DATASETS,
        "views": VIEWS,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_step": SEQUENCE_STEP,
        "num_classes": study_config["model.output_channels"],
        "input_size": study_config["experiment.dataset.input_size"],
        "class_subset": study_config.get("experiment.dataset.class_subset", None),
    }


if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    study = Study(study_config)
    exp = EXP_OctreeNCA_WarmStart_M1Init().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTSequentialDatasetForExperiment,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
