import configs
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses
from src.models.Model_OctreeNCA_2d_dual_view_warm_m1init import OctreeNCA2DDualViewWarmStartM1Init
from src.agents.Agent_OctreeNCA_DualView_WarmStart_M1Init import OctreeNCADualViewWarmStartM1InitAgent
from src.utils.Study import Study
import wonderwords
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch

from src.datasets.Dataset_Base import Dataset_Base
from src.utils.DistanceMaps import signed_distance_map


def _normalize_seq_tbptt_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    aliases = {
        "off": "off",
        "none": "off",
        "disabled": "off",
        "0": "off",
        "detach": "detach",
        "detach_only": "detach",
        "legacy": "detach",
        "1": "detach",
        "chunked": "chunked",
        "chunk": "chunked",
        "backward_chunked": "chunked",
        "true_tbptt": "chunked",
    }
    if mode in aliases:
        return aliases[mode]
    raise ValueError(
        "Invalid sequence TBPTT mode. Use one of: off|detach|chunked "
        f"(received '{mode}')."
    )


# iOCT dataset root (contains peeling/ and sri/ subfolders)
DATA_ROOT = "/vol/data/OctreeNCA_Video/ioct_data"

# Optional: train only a subset of foreground classes (background 0 is always kept).
# Example: [1, 2] -> model outputs 3 classes (background + 2 selected).
SELECTED_CLASSES = None  # e.g. [1, 2]

# Set this to a dual-view M1 checkpoint (.pth) or directory containing model.pth.
M1_CHECKPOINT_PATH = os.getenv("IOCT_DUAL_WARM_M1_CHECKPOINT_PATH", "").strip()

SEQUENCE_LENGTH = int(os.getenv("IOCT_DUAL_WARM_SEQUENCE_LENGTH", "3"))
SEQUENCE_STEP = int(os.getenv("IOCT_DUAL_WARM_SEQUENCE_STEP", "1"))

# --- Curriculum schedule on sequence length ---
# The dataset loads sequences of SEQUENCE_LENGTH_MAX frames, but training
# starts with SEQUENCE_LENGTH_MIN and linearly ramps up to SEQUENCE_LENGTH_MAX
# over the first CURRICULUM_WARMUP_EPOCHS epochs.  This forces the model to
# learn long-horizon stability gradually.
SEQUENCE_LENGTH_MIN = int(os.getenv("IOCT_DUAL_WARM_SEQ_LEN_MIN", str(SEQUENCE_LENGTH)))
SEQUENCE_LENGTH_MAX = int(os.getenv("IOCT_DUAL_WARM_SEQ_LEN_MAX", str(SEQUENCE_LENGTH)))
CURRICULUM_WARMUP_EPOCHS = int(os.getenv("IOCT_DUAL_WARM_CURRICULUM_EPOCHS", "0"))

# --- Hidden-state noise injection (training only) ---
WARM_HIDDEN_NOISE_STD = os.getenv(
    "IOCT_DUAL_WARM_HIDDEN_NOISE_STD", os.getenv("IOCT_WARM_HIDDEN_NOISE_STD", "")
).strip()
WARM_HIDDEN_NOISE_ANNEAL_EPOCHS = int(os.getenv(
    "IOCT_DUAL_WARM_HIDDEN_NOISE_ANNEAL_EPOCHS",
    os.getenv("IOCT_WARM_HIDDEN_NOISE_ANNEAL_EPOCHS", "0"),
))

# --- Spectral norm on NCA backbone residual layer ---
ENABLE_SPECTRAL_NORM = os.getenv(
    "IOCT_DUAL_WARM_SPECTRAL_NORM", os.getenv("IOCT_WARM_SPECTRAL_NORM", "0")
) == "1"

# --- Temporal consistency loss on hidden states ---
TEMPORAL_CONSISTENCY_WEIGHT = os.getenv(
    "IOCT_DUAL_WARM_TEMPORAL_CONSISTENCY_WEIGHT",
    os.getenv("IOCT_WARM_TEMPORAL_CONSISTENCY_WEIGHT", ""),
).strip()

INIT_M2_FROM_M1 = os.getenv(
    "IOCT_DUAL_WARM_INIT_M2_FROM_M1", os.getenv("IOCT_WARM_INIT_M2_FROM_M1", "1")
) == "1"
SHARE_M1_M2_BACKBONE = os.getenv(
    "IOCT_DUAL_WARM_SHARE_M1_M2_BACKBONE", os.getenv("IOCT_WARM_SHARE_M1_M2_BACKBONE", "0")
) == "1"
_seq_tbptt_env = os.getenv(
    "IOCT_DUAL_WARM_SEQ_TBPTT_STEPS",
    os.getenv("IOCT_WARM_SEQ_TBPTT_STEPS", os.getenv("IOCT_SEQ_TBPTT_STEPS", "")),
).strip()
SEQUENCE_TBPTT_STEPS = int(_seq_tbptt_env) if _seq_tbptt_env else None
_seq_tbptt_mode_env = os.getenv(
    "IOCT_DUAL_WARM_SEQ_TBPTT_MODE",
    os.getenv("IOCT_WARM_SEQ_TBPTT_MODE", os.getenv("IOCT_SEQ_TBPTT_MODE", "detach")),
)
SEQUENCE_TBPTT_MODE = _normalize_seq_tbptt_mode(_seq_tbptt_mode_env)

WARM_MULTISCALE = os.getenv(
    "IOCT_DUAL_WARM_MULTISCALE", os.getenv("IOCT_WARM_MULTISCALE", "0")
) == "1"
WARM_MULTISCALE_START_LEVEL = os.getenv(
    "IOCT_DUAL_WARM_MULTISCALE_START_LEVEL", os.getenv("IOCT_WARM_MULTISCALE_START_LEVEL", "")
).strip()
WARM_MULTISCALE_STEPS = os.getenv(
    "IOCT_DUAL_WARM_MULTISCALE_STEPS", os.getenv("IOCT_WARM_MULTISCALE_STEPS", "")
).strip()
WARM_MULTISCALE_DOWNSAMPLE_MODE = os.getenv(
    "IOCT_DUAL_WARM_MULTISCALE_DOWNSAMPLE_MODE",
    os.getenv("IOCT_WARM_MULTISCALE_DOWNSAMPLE_MODE", "nearest"),
).strip()
WARM_LOGITS_MODE = os.getenv(
    "IOCT_DUAL_WARM_LOGITS_MODE", os.getenv("IOCT_WARM_LOGITS_MODE", "carry")
).strip().lower()
WARM_LOGITS_GATE_FROM = os.getenv(
    "IOCT_DUAL_WARM_LOGITS_GATE_FROM", os.getenv("IOCT_WARM_LOGITS_GATE_FROM", "hidden")
).strip().lower()
WARM_HIDDEN_NORM = os.getenv(
    "IOCT_DUAL_WARM_HIDDEN_NORM", os.getenv("IOCT_WARM_HIDDEN_NORM", "none")
).strip().lower()
WARM_HIDDEN_CLIP = os.getenv(
    "IOCT_DUAL_WARM_HIDDEN_CLIP", os.getenv("IOCT_WARM_HIDDEN_CLIP", "")
).strip()
WARM_HIDDEN_TANH_SCALE = os.getenv(
    "IOCT_DUAL_WARM_HIDDEN_TANH_SCALE", os.getenv("IOCT_WARM_HIDDEN_TANH_SCALE", "")
).strip()
WARM_HIDDEN_GN_GROUPS = os.getenv(
    "IOCT_DUAL_WARM_HIDDEN_GN_GROUPS", os.getenv("IOCT_WARM_HIDDEN_GN_GROUPS", "")
).strip()

DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]

# Torch compile controls for this training script.
ENABLE_TORCH_COMPILE = os.getenv(
    "IOCT_DUAL_WARM_TORCH_COMPILE",
    os.getenv("IOCT_WARM_TORCH_COMPILE", os.getenv("IOCT_TORCH_COMPILE", "1")),
) == "1"
TORCH_COMPILE_MODE = os.getenv(
    "IOCT_DUAL_WARM_TORCH_COMPILE_MODE",
    os.getenv("IOCT_WARM_TORCH_COMPILE_MODE", os.getenv("IOCT_TORCH_COMPILE_MODE", "max-autotune")),
)
TORCH_COMPILE_BACKEND = os.getenv(
    "IOCT_DUAL_WARM_TORCH_COMPILE_BACKEND",
    os.getenv("IOCT_WARM_TORCH_COMPILE_BACKEND", os.getenv("IOCT_TORCH_COMPILE_BACKEND", "inductor")),
)
TORCH_COMPILE_DYNAMIC = os.getenv(
    "IOCT_DUAL_WARM_TORCH_COMPILE_DYNAMIC",
    os.getenv("IOCT_WARM_TORCH_COMPILE_DYNAMIC", os.getenv("IOCT_TORCH_COMPILE_DYNAMIC", "0")),
) == "1"
TORCH_COMPILE_FULLGRAPH = os.getenv(
    "IOCT_DUAL_WARM_TORCH_COMPILE_FULLGRAPH",
    os.getenv("IOCT_WARM_TORCH_COMPILE_FULLGRAPH", os.getenv("IOCT_TORCH_COMPILE_FULLGRAPH", "0")),
) == "1"
ENABLE_GRAD_NORM_LOGGING = os.getenv(
    "IOCT_DUAL_WARM_TRACK_GRAD_NORM",
    os.getenv("IOCT_WARM_TRACK_GRAD_NORM", os.getenv("IOCT_TRACK_GRAD_NORM", "0")),
) == "1"
_tbptt_env = os.getenv(
    "IOCT_DUAL_WARM_TBPTT_STEPS",
    os.getenv("IOCT_WARM_TBPTT_STEPS", os.getenv("IOCT_TBPTT_STEPS", "")),
).strip()
BACKBONE_TBPTT_STEPS = int(_tbptt_env) if _tbptt_env else None
LR_OVERRIDE = os.getenv("IOCT_DUAL_WARM_LR", os.getenv("IOCT_WARM_LR", "")).strip()
LR_SCALE = float(os.getenv("IOCT_DUAL_WARM_LR_SCALE", os.getenv("IOCT_WARM_LR_SCALE", "1.0")))
RESUME_EXPERIMENT_NAME = os.getenv(
    "IOCT_DUAL_WARM_RESUME_EXPERIMENT_NAME", os.getenv("IOCT_WARM_RESUME_EXPERIMENT_NAME", "")
).strip()
RESUME_MODEL_PATH = os.getenv(
    "IOCT_DUAL_WARM_RESUME_MODEL_PATH", os.getenv("IOCT_WARM_RESUME_MODEL_PATH", "")
).strip()


r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


class EXP_OctreeNCA_DualView_WarmStart_M1Init(ExperimentWrapper):
    def createExperiment(self, study_config: dict, detail_config: dict = {}, dataset_class=None, dataset_args=None):
        if dataset_args is None:
            dataset_args = {}
        if dataset_class is None:
            raise ValueError("dataset_class must be provided")

        model = OctreeNCA2DDualViewWarmStartM1Init(study_config)
        agent = OctreeNCADualViewWarmStartM1InitAgent(model)
        loss_function = WeightedLosses(study_config)
        return super().createExperiment(study_config, model, agent, dataset_class, dataset_args, loss_function)


class iOCTPairedSequentialDatasetForExperiment(Dataset_Base):
    """
    Paired-view sequential iOCT dataset adapter.

    Returns:
      - image_a, image_b: (T, 1, H, W)
      - label_a, label_b: (T, C, H, W)
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
        sequence_length: int = 3,
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
        self.views = list(views)
        if len(self.views) != 2:
            raise ValueError(f"Expected exactly two views, got {self.views}.")
        self.view_a, self.view_b = self.views[0], self.views[1]

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

            if not (img_dir_a.exists() and seg_dir_a.exists() and img_dir_b.exists() and seg_dir_b.exists()):
                print(
                    f"Warning: Skipping {dataset_name} - paired sequence directories not found "
                    f"({self.view_a}, {self.view_b})"
                )
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

        print(
            f"Found {len(self.sequences)} paired iOCT sequences "
            f"(views={self.view_a}+{self.view_b}, length={self.sequence_length}, step={self.sequence_step})."
        )

    def getFilesInPath(self, path: str):
        return {k: {"id": k} for k in self.sequences_dict.keys()}

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
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
        img = np.array(Image.open(img_path))
        seg_rgb = np.array(Image.open(seg_path))

        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)

        seg = self._rgb_to_class(seg_rgb)

        expected_size = tuple(self.size)
        if img.shape != expected_size:
            raise ValueError(f"Image shape {img.shape} does not match expected size {expected_size} for {img_path}.")
        if seg.shape != expected_size:
            raise ValueError(
                f"Segmentation shape {seg.shape} does not match expected size {expected_size} for {seg_path}."
            )

        if self.class_map is not None:
            remapped = np.zeros_like(seg)
            for src, dst in self.class_map.items():
                remapped[seg == src] = dst
            seg = remapped

        img = img.astype(np.float32) / 255.0
        img = img[None, :, :]  # (1, H, W)

        seg_tensor = torch.from_numpy(seg).long()
        max_class = int(seg_tensor.max().item())
        if max_class >= self.num_classes:
            raise ValueError(
                f"Segmentation class id {max_class} is >= num_classes ({self.num_classes}). "
                "Update model.output_channels or class_subset to cover all label ids."
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
            # Compatibility aliases for generic transform pipeline.
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
    full_num_classes = max(iOCTPairedSequentialDatasetForExperiment.RGB_TO_CLASS.values()) + 1
    study_config = {
        "experiment.name": r"OctreeNCA_iOCT_2D_DualView_WarmStart_M1Init",
        "experiment.description": "Dual-view warm start with M1 hidden state init on paired iOCT sequences.",
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
    study_config["trainer.batch_size"] = 1
    study_config["trainer.gradient_accumulation"] = 16
    study_config["trainer.normalize_gradients"] = "all"

    # Dual-view cross-fusion settings.
    study_config["model.dual_view.cross_fusion"] = "film"
    study_config["model.dual_view.cross_strength"] = 0.5
    study_config["model.dual_view.cross_use_tanh"] = True

    # M1 init options
    study_config["model.m1.pretrained_path"] = M1_CHECKPOINT_PATH
    study_config["model.m1.freeze"] = True
    study_config["model.m1.use_first_frame"] = True
    study_config["model.m1.use_t0_for_loss"] = False
    study_config["model.m1.use_probs"] = False

    # M2 init / weight sharing options
    study_config["model.m2.init_from_m1"] = INIT_M2_FROM_M1
    study_config["model.m2.share_backbone_with_m1"] = SHARE_M1_M2_BACKBONE
    study_config["model.sequence.tbptt_steps"] = SEQUENCE_TBPTT_STEPS
    study_config["model.sequence.tbptt_mode"] = SEQUENCE_TBPTT_MODE

    # Multi-scale warm-start refinement
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

    # Warm-start logits policy
    study_config["model.octree.warm_start_logits_mode"] = WARM_LOGITS_MODE
    study_config["model.octree.warm_start_logits_gate_from"] = WARM_LOGITS_GATE_FROM

    # Hidden-state stabilization
    study_config["model.octree.warm_start_hidden_norm"] = WARM_HIDDEN_NORM

    # Hidden noise injection (scheduled sampling for hidden states)
    if WARM_HIDDEN_NOISE_STD != "":
        study_config["model.octree.warm_start_hidden_noise_std"] = float(WARM_HIDDEN_NOISE_STD)
    study_config["model.octree.warm_start_hidden_noise_anneal_epochs"] = WARM_HIDDEN_NOISE_ANNEAL_EPOCHS

    # Spectral norm on NCA backbone
    study_config["model.spectral_norm"] = ENABLE_SPECTRAL_NORM

    # Temporal consistency loss
    if TEMPORAL_CONSISTENCY_WEIGHT != "":
        study_config["trainer.temporal_consistency_weight"] = float(TEMPORAL_CONSISTENCY_WEIGHT)

    # Curriculum schedule
    study_config["trainer.curriculum.seq_len_min"] = SEQUENCE_LENGTH_MIN
    study_config["trainer.curriculum.seq_len_max"] = SEQUENCE_LENGTH_MAX
    study_config["trainer.curriculum.warmup_epochs"] = CURRICULUM_WARMUP_EPOCHS
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
        "src.losses.OverflowLoss.OverflowLoss",
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
        {},
    ]
    study_config["trainer.loss_weights"] = [
        dice_loss_weight,
        2.0 - dice_loss_weight,
        boundary_loss_weight,
        1.0,
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
    if LR_OVERRIDE != "":
        study_config["trainer.optimizer.lr"] = float(LR_OVERRIDE)
    if LR_SCALE != 1.0:
        study_config["trainer.optimizer.lr"] = float(study_config["trainer.optimizer.lr"]) * LR_SCALE

    # Spike monitoring
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
        study_config["experiment.name"] = (
            f"WarmStart_M1Init_iOCT2D_dual_{random_word}_{study_config['model.channel_n']}"
        )

    if RESUME_MODEL_PATH != "":
        study_config["experiment.model_path"] = RESUME_MODEL_PATH

    return study_config


def get_dataset_args(study_config):
    # Use the maximum sequence length for the dataset so the agent can
    # truncate dynamically via the curriculum schedule.
    dataset_seq_len = max(SEQUENCE_LENGTH, SEQUENCE_LENGTH_MAX)
    return {
        "data_root": DATA_ROOT,
        "datasets": DATASETS,
        "views": VIEWS,
        "sequence_length": dataset_seq_len,
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
            "seq_tbptt_mode": study_config.get("model.sequence.tbptt_mode", "detach"),
            "optimizer_lr": study_config.get("trainer.optimizer.lr"),
            "lr_scale": LR_SCALE,
            "m1_checkpoint_path": study_config.get("model.m1.pretrained_path", ""),
        },
    )

    study = Study(study_config)
    exp = EXP_OctreeNCA_DualView_WarmStart_M1Init().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTPairedSequentialDatasetForExperiment,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
