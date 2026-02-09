import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import types

import configs
import numpy as np
import torch
from PIL import Image

from src.datasets.Dataset_Base import Dataset_Base

try:
    import colormaps as cmaps
except ImportError:
    cmaps = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# iOCT dataset root (contains peeling/ and sri/ subfolders)
DATA_ROOT = "/vol/data/OctreeNCA_Video/ioct_data"

# Optional: train only a subset of foreground classes (background 0 is always kept).
SELECTED_CLASSES = None  # e.g. [1, 2]

# Set this to your M1 checkpoint (.pth) or directory containing model.pth.
M1_CHECKPOINT_PATH = "/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_hospital_24_Training OctreeNCA on iOCT 2D frames./models/epoch_99/model.pth"

SEQUENCE_LENGTH = 5
SEQUENCE_STEP = 5
DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]


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


def _build_octree_resolutions(
    input_size: Tuple[int, int],
    steps: int,
    final_steps: int,
    first_steps_multiplier: int = 2,
):
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


def get_study_config() -> dict:
    study_config = {
        "experiment.name": "OctreeNCA_iOCT_2D_WarmStart_M1Init_VIS",
        "experiment.description": "Warm start with M1 hidden state init, then M2 for sequential iOCT frames.",
        "model.output_channels": 7,
        "model.input_channels": 1,
        "experiment.use_wandb": False,
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
        {"gamma": 2.0, "alpha": None, "ignore_index": 0, "reduction": "mean"},
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

    study_config["experiment.name"] = f"WarmStart_M1Init_iOCT2D_VIS_{study_config['model.channel_n']}"
    return study_config


def get_dataset_args(study_config: dict) -> dict:
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


def _frame_sort_key(image_path: Path):
    stem = image_path.stem
    try:
        return int(stem)
    except ValueError:
        return stem


def _extract_prediction_scores(output, output_channels: int) -> torch.Tensor:
    if isinstance(output, dict):
        pred = output.get("probabilities", None)
        if pred is None:
            pred = output.get("logits", None)
        if pred is None:
            pred = output.get("output", None)
        if pred is None:
            pred = next(iter(output.values()))
    else:
        pred = output

    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Unexpected prediction type: {type(pred)}")
    if pred.ndim != 4:
        raise ValueError(f"Expected prediction tensor with 4 dims, got shape {tuple(pred.shape)}")

    # Convert BCHW -> BHWC when necessary.
    if pred.shape[1] == output_channels:
        pred = pred.permute(0, 2, 3, 1)

    return pred


def _normalize_image(img: torch.Tensor) -> torch.Tensor:
    img = img.to(torch.float32)
    min_val = torch.min(img)
    max_val = torch.max(img)
    if (max_val - min_val) > 0:
        img = (img - min_val) / (max_val - min_val)
    return img


def _scores_to_classes(mask_or_scores_bhwc: torch.Tensor) -> torch.Tensor:
    if mask_or_scores_bhwc.ndim != 4:
        raise ValueError(f"Expected BHWC tensor, got shape {tuple(mask_or_scores_bhwc.shape)}")

    if mask_or_scores_bhwc.shape[-1] == 1:
        return (mask_or_scores_bhwc[..., 0] > 0.5).long()

    return torch.argmax(mask_or_scores_bhwc, dim=-1).long()


def _classes_to_rgb(class_map_bhw: torch.Tensor, num_classes: int) -> torch.Tensor:
    b, h, w = class_map_bhw.shape
    rgb = torch.zeros((b, h, w, 3), dtype=torch.float32, device=class_map_bhw.device)
    fallback_palette = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.82),
        (0.24, 1.0, 0.0),
        (0.0, 0.31, 1.0),
        (1.0, 0.74, 0.0),
        (0.85, 0.0, 1.0),
    ]

    def _get_color_from_cmaps(class_idx: int):
        if cmaps is None:
            return None
        bold = getattr(cmaps, "bold", None)
        if bold is None:
            return None

        # API variant 1: colormap exposes .colors as Nx3 array/list.
        colors = getattr(bold, "colors", None)
        if colors is not None:
            try:
                n = len(colors)
            except TypeError:
                n = 0
            if n > 0:
                return colors[class_idx % n]

        # API variant 2: colormap is indexable and each item has .colors or is RGB itself.
        try:
            item = bold[class_idx]
        except Exception:
            return None
        return getattr(item, "colors", item)

    for cls_idx in range(1, num_classes):
        color_vals = _get_color_from_cmaps(cls_idx)
        if color_vals is None:
            palette_idx = cls_idx if cls_idx < len(fallback_palette) else cls_idx % len(fallback_palette)
            color_vals = fallback_palette[palette_idx]
        color = torch.tensor(color_vals, dtype=torch.float32, device=class_map_bhw.device)
        rgb[class_map_bhw == cls_idx] = color
    return rgb


def _merge_three_panel_black_bg(
    image_bchw: torch.Tensor,
    pred_scores_bhwc: torch.Tensor,
    gt_bchw: torch.Tensor,
) -> np.ndarray:
    image_bhwc = image_bchw.permute(0, 2, 3, 1)
    if image_bhwc.shape[-1] != 3:
        image_bhwc = image_bhwc.repeat(1, 1, 1, 3)
    image_bhwc = _normalize_image(image_bhwc)

    gt_bhwc = gt_bchw.permute(0, 2, 3, 1)
    pred_classes = _scores_to_classes(pred_scores_bhwc)
    gt_classes = _scores_to_classes(gt_bhwc)

    num_classes = gt_bhwc.shape[-1]
    pred_rgb = _classes_to_rgb(pred_classes, num_classes=num_classes)
    gt_rgb = _classes_to_rgb(gt_classes, num_classes=num_classes)

    merged = torch.cat([image_bhwc, pred_rgb, gt_rgb], dim=2)
    return merged.squeeze(0).cpu().numpy()


def _resolve_checkpoint_path(weights_path: str) -> Path:
    path = Path(weights_path)
    if path.is_dir():
        path = path / "model.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _ensure_optional_model_imports() -> None:
    # Some model files import matplotlib/torchio but do not require them during plain inference.
    # Create minimal stubs so inference can still run in lightweight environments.
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        if "matplotlib" not in sys.modules:
            matplotlib_stub = types.ModuleType("matplotlib")
            pyplot_stub = types.ModuleType("pyplot")
            matplotlib_stub.pyplot = pyplot_stub
            sys.modules["matplotlib"] = matplotlib_stub
            sys.modules["matplotlib.pyplot"] = pyplot_stub

    try:
        import torchio  # noqa: F401
    except Exception:
        if "torchio" not in sys.modules:
            sys.modules["torchio"] = types.ModuleType("torchio")

    try:
        import einops  # noqa: F401
    except Exception:
        if "einops" not in sys.modules:
            einops_stub = types.ModuleType("einops")

            def _rearrange(x: torch.Tensor, pattern: str):
                normalized = " ".join(pattern.strip().split())
                if normalized == "b h w c -> b c h w":
                    return x.permute(0, 3, 1, 2)
                if normalized == "b c h w -> b h w c":
                    return x.permute(0, 2, 3, 1)
                raise NotImplementedError(f"einops fallback does not support pattern: {pattern}")

            einops_stub.rearrange = _rearrange
            sys.modules["einops"] = einops_stub

    # Octree model imports ViTCA module even when model.vitca=False.
    # Provide a lightweight fallback module to avoid importing heavy optional deps.
    try:
        import src.models.Model_ViTCA  # noqa: F401
    except Exception:
        if "src.models.Model_ViTCA" not in sys.modules:
            vitca_stub = types.ModuleType("src.models.Model_ViTCA")

            class ViTCA(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    raise RuntimeError("ViTCA fallback was used unexpectedly.")

            vitca_stub.ViTCA = ViTCA
            sys.modules["src.models.Model_ViTCA"] = vitca_stub


def _make_video_writer(out_path: str, fps: int, width: int, height: int):
    if cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return ("cv2", cv2.VideoWriter(out_path, fourcc, fps, (width, height)), out_path)
    if imageio is not None:
        return ("imageio", imageio.get_writer(out_path, fps=fps, codec="libx264"), out_path)
    gif_path = str(Path(out_path).with_suffix(".gif"))
    print("Warning: No MP4 backend found. Falling back to GIF output.")
    return ("pil_gif", {"frames": [], "output_path": gif_path}, gif_path)


def _write_video_frame(writer_backend, writer, frame_rgb: np.ndarray) -> None:
    if writer_backend == "cv2":
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        return
    if writer_backend == "imageio":
        writer.append_data(frame_rgb)
        return
    writer["frames"].append(frame_rgb)


def _close_video_writer(writer_backend, writer, fps: int) -> None:
    if writer_backend == "cv2":
        writer.release()
    elif writer_backend == "imageio":
        writer.close()
    else:
        frames = writer["frames"]
        if len(frames) == 0:
            return
        first = Image.fromarray(frames[0])
        append_images = [Image.fromarray(frame) for frame in frames[1:]]
        duration_ms = max(1, int(round(1000.0 / max(1, fps))))
        first.save(
            writer["output_path"],
            save_all=True,
            append_images=append_images,
            duration=duration_ms,
            loop=0,
        )


def _extract_state_dict(raw_checkpoint):
    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state_dict = raw_checkpoint.get(key, None)
            if isinstance(state_dict, dict) and len(state_dict) > 0:
                return state_dict
    if isinstance(raw_checkpoint, dict):
        return raw_checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(raw_checkpoint)}")


def _strip_prefix_if_needed(state_dict: dict, prefix: str) -> dict:
    if len(state_dict) == 0:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _load_model_weights(model: torch.nn.Module, weights_path: Path, device: str) -> None:
    checkpoint = torch.load(str(weights_path), map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    candidates = [
        state_dict,
        _strip_prefix_if_needed(state_dict, "module."),
        _strip_prefix_if_needed(state_dict, "model."),
    ]

    last_error = None
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            print(f"Loaded checkpoint: {weights_path}")
            return
        except RuntimeError as err:
            last_error = err

    missing, unexpected = model.load_state_dict(candidates[-1], strict=False)
    print(f"Warning: loaded checkpoint with strict=False: {weights_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if last_error is not None:
        print(f"Last strict load error: {last_error}")


def _build_patient_indices(dataset: iOCTSequentialDatasetForExperiment) -> Dict[str, List[int]]:
    patient_to_indices: Dict[str, List[int]] = {}
    for i, seq in enumerate(dataset.sequences):
        patient_id = seq["patient_id"]
        patient_to_indices.setdefault(patient_id, []).append(i)

    for patient_id, indices in patient_to_indices.items():
        indices.sort(key=lambda idx: _frame_sort_key(dataset.sequences[idx]["image_paths"][0]))
        patient_to_indices[patient_id] = indices

    return patient_to_indices


def create_video_ioct2d_warm_m1init(
    weights_path: str,
    target_patient_id: Optional[str] = None,
    fps: int = 20,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    output_dir: str = "visualisationOCT",
    device: Optional[str] = None,
    use_m1_init: bool = True,
    render_t0_with_m1: bool = True,
    dry_run: bool = False,
) -> List[str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    study_config = get_study_config()
    study_config["experiment.use_wandb"] = False
    study_config["experiment.device"] = device

    dataset_args = get_dataset_args(study_config)
    dataset_args["sequence_length"] = 1
    dataset_args["sequence_step"] = 1
    dataset = iOCTSequentialDatasetForExperiment(**dataset_args)
    print(f"Loaded full-frame dataset view: {len(dataset)} samples.")

    patient_to_indices = _build_patient_indices(dataset)
    available_patients = sorted(patient_to_indices.keys())
    if len(available_patients) == 0:
        raise RuntimeError("No patient/view samples were found in the dataset.")

    if target_patient_id is not None:
        if target_patient_id not in patient_to_indices:
            raise ValueError(
                f"Patient '{target_patient_id}' not found. Available: {available_patients}"
            )
        selected_patients = [target_patient_id]
    else:
        selected_patients = available_patients

    if dry_run:
        print("Dry run enabled. No model/video operations executed.")
        for patient_id in selected_patients:
            print(f"  {patient_id}: {len(patient_to_indices[patient_id])} frames")
        return []

    _ensure_optional_model_imports()
    from src.models.Model_OctreeNCA_WarmStart_M1Init import OctreeNCA2DWarmStartM1Init

    model = OctreeNCA2DWarmStartM1Init(study_config).to(device)
    model.eval()

    checkpoint_path = _resolve_checkpoint_path(weights_path)
    _load_model_weights(model, checkpoint_path, device)

    os.makedirs(output_dir, exist_ok=True)
    frame_stride = max(1, int(frame_stride))
    output_paths: List[str] = []
    output_channels = int(study_config["model.output_channels"])
    checkpoint_tag = checkpoint_path.parent.name if checkpoint_path.parent.name.startswith("epoch_") else checkpoint_path.stem

    with torch.no_grad():
        for patient_id in selected_patients:
            sequence_indices = list(patient_to_indices[patient_id])
            if frame_stride > 1:
                sequence_indices = sequence_indices[::frame_stride]
            if max_frames is not None:
                sequence_indices = sequence_indices[:max_frames]
            if len(sequence_indices) == 0:
                continue

            out_path = os.path.join(output_dir, f"ioct_warm_m1init_{checkpoint_tag}_{patient_id}.mp4")
            actual_out_path = out_path
            video_writer = None
            writer_backend = None
            prev_state = None

            print(f"Rendering patient/view '{patient_id}' with {len(sequence_indices)} frames...")

            for i, idx in enumerate(sequence_indices):
                sample = dataset[idx]
                image_t = torch.from_numpy(sample["image"]).to(device=device, dtype=torch.float32)[:1]
                label_t = torch.from_numpy(sample["label"]).to(device=device, dtype=torch.float32)[:1]

                if i == 0 and use_m1_init:
                    prev_state = model.init_state_from_m1(image_t)
                    if render_t0_with_m1:
                        out = model.m1(image_t)
                    else:
                        out = model(image_t, prev_state=prev_state, batch_duplication=1)
                        prev_state = out.get("final_state", prev_state)
                else:
                    out = model(image_t, prev_state=prev_state, batch_duplication=1)
                    prev_state = out.get("final_state", prev_state)

                pred_scores = _extract_prediction_scores(out, output_channels)
                merged = _merge_three_panel_black_bg(image_t, pred_scores, label_t)

                frame_rgb = (merged * 255.0).clip(0, 255).astype(np.uint8)
                if cv2 is not None:
                    frame_name = dataset.sequences[idx]["image_paths"][0].stem
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        frame_bgr,
                        f"{patient_id} | frame {frame_name}",
                        (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if video_writer is None:
                    height, width, _ = frame_rgb.shape
                    writer_backend, video_writer, actual_out_path = _make_video_writer(out_path, fps, width, height)

                _write_video_frame(writer_backend, video_writer, frame_rgb)

                if i % 25 == 0:
                    print(f"  frame {i}/{len(sequence_indices)}")

            if video_writer is not None:
                _close_video_writer(writer_backend, video_writer, fps=fps)
                output_paths.append(os.path.abspath(actual_out_path))
                print(f"Saved video: {os.path.abspath(actual_out_path)}")

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create warm-start M1-init iOCT videos (image | prediction | ground truth)."
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint (.pth or epoch directory).")
    parser.add_argument("--patient", type=str, default=None, help="Patient/view id (e.g. peeling_A). If omitted, renders all.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="visualisationOCT")
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Defaults to auto.")
    parser.add_argument("--no_m1_init", action="store_true", help="Disable M1 hidden-state initialization.")
    parser.add_argument(
        "--m2_on_t0",
        action="store_true",
        help="When M1 init is enabled, use M2 prediction on the first frame (instead of M1 prediction).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Load dataset and report frame counts without running inference.")
    args = parser.parse_args()

    created = create_video_ioct2d_warm_m1init(
        weights_path=args.weights,
        target_patient_id=args.patient,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        output_dir=args.output_dir,
        device=args.device,
        use_m1_init=not args.no_m1_init,
        render_t0_with_m1=not args.m2_on_t0,
        dry_run=args.dry_run,
    )

    if len(created) == 0:
        print("No videos were generated.")
