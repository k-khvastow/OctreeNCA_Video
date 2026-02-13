import configs
from src.utils.WarmStartM1InitConfig import EXP_OctreeNCA_WarmStart_M1Init
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

# Set this to your M1 checkpoint (.pth) or directory containing model.pth.
M1_CHECKPOINT_PATH = "/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_pan_24_Training OctreeNCA on iOCT 2D frames./models/epoch_99/model.pth"

SEQUENCE_LENGTH = 2
SEQUENCE_STEP = 10
INIT_M2_FROM_M1 = os.getenv("IOCT_WARM_INIT_M2_FROM_M1", "0") == "1"
SHARE_M1_M2_BACKBONE = os.getenv("IOCT_WARM_SHARE_M1_M2_BACKBONE", "0") == "1"
_seq_tbptt_env = os.getenv("IOCT_WARM_SEQ_TBPTT_STEPS", os.getenv("IOCT_SEQ_TBPTT_STEPS", "")).strip()
SEQUENCE_TBPTT_STEPS = int(_seq_tbptt_env) if _seq_tbptt_env else None

WARM_MULTISCALE = os.getenv("IOCT_WARM_MULTISCALE", "0") == "1"
WARM_MULTISCALE_START_LEVEL = os.getenv("IOCT_WARM_MULTISCALE_START_LEVEL", "").strip()
WARM_MULTISCALE_STEPS = os.getenv("IOCT_WARM_MULTISCALE_STEPS", "").strip()
WARM_MULTISCALE_DOWNSAMPLE_MODE = os.getenv("IOCT_WARM_MULTISCALE_DOWNSAMPLE_MODE", "nearest").strip()
WARM_LOGITS_MODE = os.getenv("IOCT_WARM_LOGITS_MODE", "carry").strip().lower()
WARM_LOGITS_GATE_FROM = os.getenv("IOCT_WARM_LOGITS_GATE_FROM", "hidden").strip().lower()
WARM_HIDDEN_NORM = os.getenv("IOCT_WARM_HIDDEN_NORM", "none").strip().lower()
WARM_HIDDEN_CLIP = os.getenv("IOCT_WARM_HIDDEN_CLIP", "").strip()
WARM_HIDDEN_TANH_SCALE = os.getenv("IOCT_WARM_HIDDEN_TANH_SCALE", "").strip()
WARM_HIDDEN_GN_GROUPS = os.getenv("IOCT_WARM_HIDDEN_GN_GROUPS", "").strip()

DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]

# Torch compile controls for this training script.
ENABLE_TORCH_COMPILE = os.getenv("IOCT_WARM_TORCH_COMPILE", os.getenv("IOCT_TORCH_COMPILE", "1")) == "1"
TORCH_COMPILE_MODE = os.getenv("IOCT_WARM_TORCH_COMPILE_MODE", os.getenv("IOCT_TORCH_COMPILE_MODE", "max-autotune"))  # "default", "reduce-overhead", "max-autotune", "max-autotune-internal", "loose", "force_fallback"
TORCH_COMPILE_BACKEND = os.getenv("IOCT_WARM_TORCH_COMPILE_BACKEND", os.getenv("IOCT_TORCH_COMPILE_BACKEND", "inductor"))
TORCH_COMPILE_DYNAMIC = os.getenv("IOCT_WARM_TORCH_COMPILE_DYNAMIC", os.getenv("IOCT_TORCH_COMPILE_DYNAMIC", "0")) == "1"
TORCH_COMPILE_FULLGRAPH = os.getenv("IOCT_WARM_TORCH_COMPILE_FULLGRAPH", os.getenv("IOCT_TORCH_COMPILE_FULLGRAPH", "0")) == "1"
ENABLE_GRAD_NORM_LOGGING = os.getenv("IOCT_WARM_TRACK_GRAD_NORM", os.getenv("IOCT_TRACK_GRAD_NORM", "0")) == "1"
_tbptt_env = os.getenv("IOCT_WARM_TBPTT_STEPS", os.getenv("IOCT_TBPTT_STEPS", "")).strip()
BACKBONE_TBPTT_STEPS = int(_tbptt_env) if _tbptt_env else None
LR_OVERRIDE = os.getenv("IOCT_WARM_LR", "").strip()
LR_SCALE = float(os.getenv("IOCT_WARM_LR_SCALE", "1.0"))
RESUME_EXPERIMENT_NAME = os.getenv("IOCT_WARM_RESUME_EXPERIMENT_NAME", "").strip()
RESUME_MODEL_PATH = os.getenv("IOCT_WARM_RESUME_MODEL_PATH", "").strip()



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
        (218, 0, 255): 6,      # Class 6 (magenta)
    }

    def __init__(
        self,
        data_root: str,
        datasets=DATASETS,
        views=VIEWS,
        sequence_length: int = 5,
        sequence_step: int = 1,
        num_classes: int = 7,
        input_size=(512, 512),
        class_subset=None,
        precompute_boundary_dist: bool = False,
        boundary_dist_classes=None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.views = views
        self.sequence_length = int(sequence_length)
        self.sequence_step = int(sequence_step)
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
        label_dists = [] if self.precompute_boundary_dist else None

        for img_path in meta["image_paths"]:
            seg_path = meta["seg_dir"] / img_path.name

            img = np.array(Image.open(img_path))
            seg_rgb = np.array(Image.open(seg_path))

            if img.ndim == 3:
                img = np.mean(img, axis=2).astype(np.uint8)

            seg = self._rgb_to_class(seg_rgb)

            # No resizing in this training script: enforce expected native size.
            if hasattr(self, "size") and self.size is not None:
                expected_size = tuple(self.size)
                if img.shape != expected_size:
                    raise ValueError(
                        f"Image shape {img.shape} does not match expected size {expected_size} for {meta['id']} ({img_path.name})."
                    )
                if seg.shape != expected_size:
                    raise ValueError(
                        f"Segmentation shape {seg.shape} does not match expected size {expected_size} for {meta['id']} ({seg_path.name})."
                    )

            if self.class_map is not None:
                remapped = np.zeros_like(seg)
                for src, dst in self.class_map.items():
                    remapped[seg == src] = dst
                seg = remapped

            img = img.astype(np.float32) / 255.0
            img = img[None, :, :]

            seg_tensor = torch.from_numpy(seg).long()
            max_class = int(seg_tensor.max().item())
            if max_class >= self.num_classes:
                raise ValueError(
                    f"Segmentation class id {max_class} is >= num_classes ({self.num_classes}) in {meta['id']} ({seg_path.name}). "
                    "Update model.output_channels or class_subset to cover all label ids."
                )
            seg_onehot = torch.nn.functional.one_hot(seg_tensor, num_classes=self.num_classes).permute(2, 0, 1).numpy().astype(np.float32)

            if label_dists is not None:
                seg_dist = signed_distance_map(
                    seg_onehot,
                    class_ids=self.boundary_dist_classes,
                    channel_first=True,
                    compact=False,
                    dtype=np.float32,
                )
                label_dists.append(seg_dist)

            imgs.append(img)
            masks.append(seg_onehot)

        imgs_np = np.stack(imgs)
        masks_onehot = np.stack(masks)

        sample = {
            "image": imgs_np,
            "label": masks_onehot,
            "id": meta["id"],
        }
        if label_dists is not None:
            sample["label_dist"] = np.stack(label_dists)
        return sample


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
    full_num_classes = max(iOCTSequentialDatasetForExperiment.RGB_TO_CLASS.values()) + 1
    study_config = {
        "experiment.name": r"OctreeNCA_iOCT_2D_WarmStart_M1Init",
        "experiment.description": "Warm start with M1 hidden state init, then M2 for sequential iOCT frames.",
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
        input_size, steps, int(alpha * 20)
    )
    study_config["model.kernel_size"] = [3] * len(study_config["model.octree.res_and_steps"])
    study_config["model.octree.warm_start_steps"] = 10
    study_config["model.channel_n"] = 24
    study_config["model.hidden_size"] = 64
    study_config["trainer.batch_size"] = 4
    study_config["trainer.gradient_accumulation"] = 8
    study_config["trainer.normalize_gradients"] = "all"

    # M1 init options
    study_config["model.m1.pretrained_path"] = M1_CHECKPOINT_PATH
    study_config["model.m1.freeze"] = True
    study_config["model.m1.use_first_frame"] = True
    study_config["model.m1.use_t0_for_loss"] = False
    study_config["model.m1.use_probs"] = False
    # M2 init / weight sharing options
    # - Set IOCT_WARM_INIT_M2_FROM_M1=1 to copy M1 backbone weights into M2 at init time.
    # - Set IOCT_WARM_SHARE_M1_M2_BACKBONE=1 to share the same backbone modules (requires model.m1.freeze=0 and model.m1.eval_mode=0).
    study_config["model.m2.init_from_m1"] = INIT_M2_FROM_M1
    study_config["model.m2.share_backbone_with_m1"] = SHARE_M1_M2_BACKBONE
    study_config["model.sequence.tbptt_steps"] = SEQUENCE_TBPTT_STEPS

    # Multi-scale warm-start refinement across octree levels (coarse -> fine).
    # Enable with IOCT_WARM_MULTISCALE=1.
    # Optional:
    # - IOCT_WARM_MULTISCALE_START_LEVEL: integer level index to start from (default: coarsest).
    # - IOCT_WARM_MULTISCALE_STEPS: int (all levels) or comma list of length n_levels (level0..levelN).
    # - IOCT_WARM_MULTISCALE_DOWNSAMPLE_MODE: nearest|bilinear|bicubic|area for downsampling prev_state.
    study_config["model.octree.warm_start_multiscale"] = WARM_MULTISCALE
    study_config["model.octree.warm_start_multiscale_downsample_mode"] = WARM_MULTISCALE_DOWNSAMPLE_MODE
    if WARM_MULTISCALE_START_LEVEL != "":
        study_config["model.octree.warm_start_multiscale_start_level"] = int(WARM_MULTISCALE_START_LEVEL)
    if WARM_MULTISCALE_STEPS != "":
        if "," in WARM_MULTISCALE_STEPS:
            study_config["model.octree.warm_start_multiscale_steps"] = [
                int(x) for x in WARM_MULTISCALE_STEPS.split(",") if x.strip() != ""
            ]
        else:
            study_config["model.octree.warm_start_multiscale_steps"] = int(WARM_MULTISCALE_STEPS)

    # Warm-start logits policy (carry logits across frames can cause drift/instability).
    # IOCT_WARM_LOGITS_MODE: carry|reset|gate
    # IOCT_WARM_LOGITS_GATE_FROM (gate mode): hidden|state|hidden+input
    study_config["model.octree.warm_start_logits_mode"] = WARM_LOGITS_MODE
    study_config["model.octree.warm_start_logits_gate_from"] = WARM_LOGITS_GATE_FROM

    # Hidden-state stabilization (applies to the carried recurrent hidden channels).
    # IOCT_WARM_HIDDEN_NORM: none|layer|group
    # IOCT_WARM_HIDDEN_CLIP: float (clamp to [-v, v])
    # IOCT_WARM_HIDDEN_TANH_SCALE: float (bounds via tanh; keeps magnitude roughly <= scale)
    # IOCT_WARM_HIDDEN_GN_GROUPS: int (groupnorm groups; falls back to 1 if incompatible)
    study_config["model.octree.warm_start_hidden_norm"] = WARM_HIDDEN_NORM
    if WARM_HIDDEN_CLIP != "":
        study_config["model.octree.warm_start_hidden_clip"] = float(WARM_HIDDEN_CLIP)
    if WARM_HIDDEN_TANH_SCALE != "":
        study_config["model.octree.warm_start_hidden_tanh_scale"] = float(WARM_HIDDEN_TANH_SCALE)
    if WARM_HIDDEN_GN_GROUPS != "":
        study_config["model.octree.warm_start_hidden_gn_groups"] = int(WARM_HIDDEN_GN_GROUPS)

    dice_loss_weight = 1.0
    boundary_loss_weight = 0.1
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
        {"do_bg": False, "channel_last": True, "use_precomputed": True, "use_probabilities": False, "dist_clip": 20.0,
         "compute_missing_dist": False},
    ]
    study_config["trainer.loss_weights"] = [
        dice_loss_weight,
        2.0 - dice_loss_weight,
        boundary_loss_weight,
    ]

    study_config["experiment.dataset.precompute_boundary_dist"] = True
    study_config["experiment.dataset.boundary_dist_classes"] = None
    study_config["trainer.gradient_clip_val"] = 1.0


    study_config["model.normalization"] = "none"
    study_config["model.apply_nonlin"] = "torch.nn.Softmax(dim=-1)"
    study_config["performance.compile"] = ENABLE_TORCH_COMPILE
    study_config["performance.compile.mode"] = TORCH_COMPILE_MODE
    study_config["performance.compile.backend"] = TORCH_COMPILE_BACKEND
    study_config["performance.compile.dynamic"] = TORCH_COMPILE_DYNAMIC
    study_config["performance.compile.fullgraph"] = TORCH_COMPILE_FULLGRAPH
    study_config["experiment.logging.track_gradient_norm"] = ENABLE_GRAD_NORM_LOGGING
    study_config["model.backbone.tbptt_steps"] = BACKBONE_TBPTT_STEPS

    # Optional learning-rate controls for quick tuning without editing defaults.
    # - IOCT_WARM_LR: set an explicit optimizer LR.
    # - IOCT_WARM_LR_SCALE: multiply the current LR (default: 1.0).
    if LR_OVERRIDE != "":
        study_config["trainer.optimizer.lr"] = float(LR_OVERRIDE)
    if LR_SCALE != 1.0:
        study_config["trainer.optimizer.lr"] = float(study_config["trainer.optimizer.lr"]) * LR_SCALE

    # Spike monitoring (per-batch class counts + save batches on spikes)
    study_config["experiment.logging.spike_watch.enabled"] = True
    dice_spike_keys = [f"nnUNetSoftDiceLossSum/mask_{i}" for i in range(max(0, full_num_classes - 1))]
    study_config["experiment.logging.spike_watch.keys"] = [
        "FocalLoss/loss",
        "BoundaryLoss/loss",
        "nnUNetSoftDiceLossSum/overall",
        *dice_spike_keys,
    ]
    study_config["experiment.logging.spike_watch.window"] = 50
    study_config["experiment.logging.spike_watch.zscore"] = 3.0
    study_config["experiment.logging.spike_watch.min_value"] = 0.2
    study_config["experiment.logging.spike_watch.max_images_per_epoch"] = 10
    study_config["experiment.logging.spike_watch.max_images_per_spike"] = 2
    study_config["experiment.logging.spike_watch.save_classes"] = list(range(1, full_num_classes))
    study_config["experiment.logging.batch_timing.enabled"] = False
    study_config["experiment.logging.batch_timing.print_interval"] = 20
    study_config["experiment.logging.batch_timing.warmup_steps"] = 5
    study_config["experiment.logging.phase_timing.enabled"] = True
    study_config["experiment.logging.phase_timing.print_interval"] = 20
    study_config["experiment.logging.phase_timing.warmup_steps"] = 5

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

    if RESUME_EXPERIMENT_NAME != "":
        study_config["experiment.name"] = RESUME_EXPERIMENT_NAME
    else:
        study_config["experiment.name"] = f"WarmStart_M1Init_iOCT2D_{random_word}_{study_config['model.channel_n']}"

    if RESUME_MODEL_PATH != "":
        study_config["experiment.model_path"] = RESUME_MODEL_PATH

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
            "seq_tbptt_steps": study_config.get("model.sequence.tbptt_steps", None),
            "optimizer_lr": study_config.get("trainer.optimizer.lr"),
            "lr_scale": LR_SCALE,
        },
    )

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
