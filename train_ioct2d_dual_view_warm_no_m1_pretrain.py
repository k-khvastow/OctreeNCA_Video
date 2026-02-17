"""Back-to-back dual-view warm-start training **without** any pretrained M1
weights.  Both M1 and M2 are randomly initialised and trained jointly
end-to-end.

Key differences from ``train_ioct2d_dual_view_warm_preprocessed_m1init.py``:

* ``model.m1.pretrained_path`` is empty  →  M1 starts from random init.
* ``model.m1.freeze``  = False           →  M1 must learn, not be frozen.
* ``model.m1.eval_mode`` = False         →  BN/dropout in train mode.
* ``model.m1.use_t0_for_loss`` = True    →  direct supervision at t=0 so M1
                                            gets gradient signal on the very
                                            first frame.
* ``model.m1.disable_backbone_tbptt`` = False  →  keep TBPTT active in M1
                                                   since its backbone also
                                                   trains.
* ``model.m2.init_from_m1`` defaults to ``False``; setting it to ``True``
  copies the *random* M1 weights into M2 so both branches share the same
  starting point (only works when channel_n is the same for M1 and M2).
* ``model.m1.channel_n`` is removed – M1 and M2 share the same channel
  count since there is no legacy checkpoint to accommodate.
* Learning rate defaults higher (``2e-4``) since we train from scratch.
"""

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


# ---------------------------------------------------------------------------
# Helper: normalise the sequence-level TBPTT mode string
# ---------------------------------------------------------------------------
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


# ── Environment variables ──────────────────────────────────────────────────
DATA_ROOT = "/vol/data/OctreeNCA_Video/ioct_data"

SELECTED_CLASSES = None  # e.g. [1, 2]

# No pretrained M1 path – this is the whole point of this script.
M1_CHECKPOINT_PATH = os.getenv("IOCT_DUAL_B2B_M1_CHECKPOINT_PATH", "").strip()

SEQUENCE_LENGTH = int(os.getenv("IOCT_DUAL_B2B_SEQUENCE_LENGTH", "3"))
SEQUENCE_STEP = int(os.getenv("IOCT_DUAL_B2B_SEQUENCE_STEP", "1"))

# Curriculum schedule
SEQUENCE_LENGTH_MIN = int(os.getenv("IOCT_DUAL_B2B_SEQ_LEN_MIN", str(SEQUENCE_LENGTH)))
SEQUENCE_LENGTH_MAX = int(os.getenv("IOCT_DUAL_B2B_SEQ_LEN_MAX", str(SEQUENCE_LENGTH)))
CURRICULUM_WARMUP_EPOCHS = int(os.getenv("IOCT_DUAL_B2B_CURRICULUM_EPOCHS", "0"))

# Hidden-state noise injection
WARM_HIDDEN_NOISE_STD = os.getenv(
    "IOCT_DUAL_B2B_HIDDEN_NOISE_STD", os.getenv("IOCT_WARM_HIDDEN_NOISE_STD", "")
).strip()
WARM_HIDDEN_NOISE_ANNEAL_EPOCHS = int(os.getenv(
    "IOCT_DUAL_B2B_HIDDEN_NOISE_ANNEAL_EPOCHS",
    os.getenv("IOCT_WARM_HIDDEN_NOISE_ANNEAL_EPOCHS", "0"),
))

# Spectral norm
ENABLE_SPECTRAL_NORM = os.getenv(
    "IOCT_DUAL_B2B_SPECTRAL_NORM", os.getenv("IOCT_WARM_SPECTRAL_NORM", "0")
) == "1"

# Temporal consistency loss
TEMPORAL_CONSISTENCY_WEIGHT = os.getenv(
    "IOCT_DUAL_B2B_TEMPORAL_CONSISTENCY_WEIGHT",
    os.getenv("IOCT_WARM_TEMPORAL_CONSISTENCY_WEIGHT", ""),
).strip()

# M1 settings – all default to "training" mode since there are no pretrained
# weights.
IOCT_M1_FREEZE = False  # NEVER freeze a random M1
M1_DISABLE_BACKBONE_TBPTT = os.getenv(
    "IOCT_DUAL_B2B_M1_DISABLE_BACKBONE_TBPTT", "0"
) == "1"  # default: keep TBPTT active in M1

INIT_M2_FROM_M1 = os.getenv(
    "IOCT_DUAL_B2B_INIT_M2_FROM_M1", os.getenv("IOCT_WARM_INIT_M2_FROM_M1", "0")
) == "1"
INIT_M2_IDENTITY = os.getenv(
    "IOCT_DUAL_B2B_INIT_M2_IDENTITY", os.getenv("IOCT_WARM_INIT_M2_IDENTITY", "0")
) == "1"
SHARE_M1_M2_BACKBONE = os.getenv(
    "IOCT_DUAL_B2B_SHARE_M1_M2_BACKBONE", os.getenv("IOCT_WARM_SHARE_M1_M2_BACKBONE", "0")
) == "1"
_seq_tbptt_env = os.getenv(
    "IOCT_DUAL_B2B_SEQ_TBPTT_STEPS",
    os.getenv("IOCT_WARM_SEQ_TBPTT_STEPS", os.getenv("IOCT_SEQ_TBPTT_STEPS", "")),
).strip()
SEQUENCE_TBPTT_STEPS = int(_seq_tbptt_env) if _seq_tbptt_env else None
_seq_tbptt_mode_env = os.getenv(
    "IOCT_DUAL_B2B_SEQ_TBPTT_MODE",
    os.getenv("IOCT_WARM_SEQ_TBPTT_MODE", os.getenv("IOCT_SEQ_TBPTT_MODE", "detach")),
)
SEQUENCE_TBPTT_MODE = _normalize_seq_tbptt_mode(_seq_tbptt_mode_env)

WARM_MULTISCALE = os.getenv(
    "IOCT_DUAL_B2B_MULTISCALE", os.getenv("IOCT_WARM_MULTISCALE", "0")
) == "1"
WARM_MULTISCALE_START_LEVEL = os.getenv(
    "IOCT_DUAL_B2B_MULTISCALE_START_LEVEL", os.getenv("IOCT_WARM_MULTISCALE_START_LEVEL", "")
).strip()
WARM_MULTISCALE_STEPS = os.getenv(
    "IOCT_DUAL_B2B_MULTISCALE_STEPS", os.getenv("IOCT_WARM_MULTISCALE_STEPS", "")
).strip()
WARM_MULTISCALE_DOWNSAMPLE_MODE = os.getenv(
    "IOCT_DUAL_B2B_MULTISCALE_DOWNSAMPLE_MODE",
    os.getenv("IOCT_WARM_MULTISCALE_DOWNSAMPLE_MODE", "nearest"),
).strip()
WARM_LOGITS_MODE = os.getenv(
    "IOCT_DUAL_B2B_LOGITS_MODE", os.getenv("IOCT_WARM_LOGITS_MODE", "carry")
).strip().lower()
WARM_LOGITS_GATE_FROM = os.getenv(
    "IOCT_DUAL_B2B_LOGITS_GATE_FROM", os.getenv("IOCT_WARM_LOGITS_GATE_FROM", "hidden")
).strip().lower()
WARM_HIDDEN_NORM = os.getenv(
    "IOCT_DUAL_B2B_HIDDEN_NORM", os.getenv("IOCT_WARM_HIDDEN_NORM", "none")
).strip().lower()
WARM_HIDDEN_CLIP = os.getenv(
    "IOCT_DUAL_B2B_HIDDEN_CLIP", os.getenv("IOCT_WARM_HIDDEN_CLIP", "")
).strip()
WARM_HIDDEN_TANH_SCALE = os.getenv(
    "IOCT_DUAL_B2B_HIDDEN_TANH_SCALE", os.getenv("IOCT_WARM_HIDDEN_TANH_SCALE", "")
).strip()
WARM_HIDDEN_GN_GROUPS = os.getenv(
    "IOCT_DUAL_B2B_HIDDEN_GN_GROUPS", os.getenv("IOCT_WARM_HIDDEN_GN_GROUPS", "")
).strip()

# Learned temporal gate
WARM_TEMPORAL_GATE = os.getenv(
    "IOCT_DUAL_B2B_TEMPORAL_GATE", os.getenv("IOCT_WARM_TEMPORAL_GATE", "none")
).strip().lower()

# Temporal vs. Spatial hidden channel split
WARM_TEMPORAL_RATIO = os.getenv(
    "IOCT_DUAL_B2B_TEMPORAL_RATIO", os.getenv("IOCT_WARM_TEMPORAL_RATIO", "")
).strip()

# --- Number of NCA state channels ---
CHANNEL_N = int(os.getenv("IOCT_DUAL_B2B_CHANNEL_N", os.getenv("IOCT_WARM_CHANNEL_N", "24")))

# No separate M1_CHANNEL_N – M1 and M2 share the same channel count.

DATASETS = ["peeling", "sri"]
VIEWS = ["A", "B"]


# ---------------------------------------------------------------------------
# Inverse-frequency class-weight computation (same as parent script)
# ---------------------------------------------------------------------------
def _compute_class_alpha_weights(
    data_root: str,
    datasets: list,
    views: list,
    num_classes: int,
    rgb_to_class: dict,
    max_samples: int = 200,
) -> list:
    from pathlib import Path
    counts = np.zeros(num_classes, dtype=np.float64)
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
                for c in range(num_classes):
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


# ---------------------------------------------------------------------------
# Torch compile
# ---------------------------------------------------------------------------
ENABLE_TORCH_COMPILE = os.getenv(
    "IOCT_DUAL_B2B_TORCH_COMPILE",
    os.getenv("IOCT_WARM_TORCH_COMPILE", os.getenv("IOCT_TORCH_COMPILE", "1")),
) == "1"
TORCH_COMPILE_MODE = os.getenv(
    "IOCT_DUAL_B2B_TORCH_COMPILE_MODE",
    os.getenv("IOCT_WARM_TORCH_COMPILE_MODE", os.getenv("IOCT_TORCH_COMPILE_MODE", "max-autotune")),
)
TORCH_COMPILE_BACKEND = os.getenv(
    "IOCT_DUAL_B2B_TORCH_COMPILE_BACKEND",
    os.getenv("IOCT_WARM_TORCH_COMPILE_BACKEND", os.getenv("IOCT_TORCH_COMPILE_BACKEND", "inductor")),
)
TORCH_COMPILE_DYNAMIC = os.getenv(
    "IOCT_DUAL_B2B_TORCH_COMPILE_DYNAMIC",
    os.getenv("IOCT_WARM_TORCH_COMPILE_DYNAMIC", os.getenv("IOCT_TORCH_COMPILE_DYNAMIC", "0")),
) == "1"
TORCH_COMPILE_FULLGRAPH = os.getenv(
    "IOCT_DUAL_B2B_TORCH_COMPILE_FULLGRAPH",
    os.getenv("IOCT_WARM_TORCH_COMPILE_FULLGRAPH", os.getenv("IOCT_TORCH_COMPILE_FULLGRAPH", "0")),
) == "1"
ENABLE_GRAD_NORM_LOGGING = os.getenv(
    "IOCT_DUAL_B2B_TRACK_GRAD_NORM",
    os.getenv("IOCT_WARM_TRACK_GRAD_NORM", os.getenv("IOCT_TRACK_GRAD_NORM", "0")),
) == "1"
_tbptt_env = os.getenv(
    "IOCT_DUAL_B2B_TBPTT_STEPS",
    os.getenv("IOCT_WARM_TBPTT_STEPS", os.getenv("IOCT_TBPTT_STEPS", "")),
).strip()
BACKBONE_TBPTT_STEPS = int(_tbptt_env) if _tbptt_env else None
LR_OVERRIDE = os.getenv("IOCT_DUAL_B2B_LR", os.getenv("IOCT_WARM_LR", "")).strip()
LR_SCALE = float(os.getenv("IOCT_DUAL_B2B_LR_SCALE", os.getenv("IOCT_WARM_LR_SCALE", "1.0")))
RESUME_EXPERIMENT_NAME = os.getenv(
    "IOCT_DUAL_B2B_RESUME_EXPERIMENT_NAME", os.getenv("IOCT_WARM_RESUME_EXPERIMENT_NAME", "")
).strip()
RESUME_MODEL_PATH = os.getenv(
    "IOCT_DUAL_B2B_RESUME_MODEL_PATH", os.getenv("IOCT_WARM_RESUME_MODEL_PATH", "")
).strip()


r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


# ---------------------------------------------------------------------------
# Experiment wrapper – identical to the m1-init variant
# ---------------------------------------------------------------------------
class EXP_OctreeNCA_DualView_B2B(ExperimentWrapper):
    def createExperiment(self, study_config: dict, detail_config: dict = {}, dataset_class=None, dataset_args=None):
        if dataset_args is None:
            dataset_args = {}
        if dataset_class is None:
            raise ValueError("dataset_class must be provided")

        model = OctreeNCA2DDualViewWarmStartM1Init(study_config)
        agent = OctreeNCADualViewWarmStartM1InitAgent(model)
        loss_function = WeightedLosses(study_config)
        return super().createExperiment(study_config, model, agent, dataset_class, dataset_args, loss_function)


# ---------------------------------------------------------------------------
# Dataset – re-use the paired-sequence dataset from the m1-init script
# ---------------------------------------------------------------------------
from train_ioct2d_dual_view_warm_preprocessed_m1init import iOCTPairedSequentialDatasetForExperiment


# ---------------------------------------------------------------------------
# Octree resolution builder
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Study config
# ---------------------------------------------------------------------------
def get_study_config():
    full_num_classes = max(iOCTPairedSequentialDatasetForExperiment.RGB_TO_CLASS.values()) + 1
    study_config = {
        "experiment.name": r"OctreeNCA_iOCT_2D_DualView_B2B_NoM1Pretrain",
        "experiment.description": (
            "Back-to-back dual-view training: M1 + M2 from random init (no pretrained M1)."
        ),
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
    steps = 8
    alpha = 1.0
    input_size = study_config["experiment.dataset.input_size"]
    study_config["model.backbone_class"] = "BasicNCA2DFast"
    study_config["model.octree.separate_models"] = True
    study_config["model.octree.res_and_steps"] = _build_octree_resolutions(
        input_size, steps, int(alpha * 20)
    )
    study_config["model.kernel_size"] = [3] * len(study_config["model.octree.res_and_steps"])
    study_config["model.octree.warm_start_steps"] = 10
    study_config["model.channel_n"] = CHANNEL_N
    study_config["model.hidden_size"] = 64
    study_config["trainer.batch_size"] = 1
    study_config["trainer.gradient_accumulation"] = 16
    study_config["trainer.normalize_gradients"] = "all"

    # Dual-view cross-fusion settings
    study_config["model.dual_view.cross_fusion"] = "film"
    study_config["model.dual_view.cross_strength"] = 0.5
    study_config["model.dual_view.cross_use_tanh"] = True

    # ── M1 init: NO pretrained weights, train from scratch ──────────────
    study_config["model.m1.pretrained_path"] = M1_CHECKPOINT_PATH  # empty → random init
    study_config["model.m1.freeze"] = False   # must train M1
    study_config["model.m1.eval_mode"] = False  # keep BN/dropout in train mode
    study_config["model.m1.use_first_frame"] = True
    study_config["model.m1.use_t0_for_loss"] = True  # supervise M1 at t=0
    study_config["model.m1.use_probs"] = False
    study_config["model.m1.disable_backbone_tbptt"] = M1_DISABLE_BACKBONE_TBPTT

    # M2 init / weight sharing  (no separate M1 channel count needed)
    study_config["model.m2.init_from_m1"] = INIT_M2_FROM_M1
    study_config["model.m2.init_identity"] = INIT_M2_IDENTITY
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

    # Hidden noise injection
    if WARM_HIDDEN_NOISE_STD != "":
        study_config["model.octree.warm_start_hidden_noise_std"] = float(WARM_HIDDEN_NOISE_STD)
    study_config["model.octree.warm_start_hidden_noise_anneal_epochs"] = WARM_HIDDEN_NOISE_ANNEAL_EPOCHS

    # Spectral norm
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

    # Learned temporal gate
    study_config["model.octree.warm_start_temporal_gate"] = WARM_TEMPORAL_GATE

    # Temporal vs. Spatial hidden channel split
    if WARM_TEMPORAL_RATIO != "":
        study_config["model.octree.warm_start_temporal_ratio"] = float(WARM_TEMPORAL_RATIO)

    dice_loss_weight = 1
    focal_loss_weight = 1
    boundary_loss_weight = 0.1
    _ema_env = os.getenv("IOCT_DUAL_B2B_EMA", os.getenv("IOCT_WARM_EMA", "1"))
    ema_decay = 0.99 if _ema_env == "1" else 0.0
    study_config["trainer.ema"] = ema_decay > 0.0
    study_config["trainer.ema.decay"] = ema_decay
    study_config["trainer.use_amp"] = False

    # --- Compute per-class inverse-frequency alpha for FocalLoss ---
    focal_alpha = _compute_class_alpha_weights(
        data_root=DATA_ROOT,
        datasets=DATASETS,
        views=VIEWS,
        num_classes=full_num_classes,
        rgb_to_class=iOCTPairedSequentialDatasetForExperiment.RGB_TO_CLASS,
        max_samples=200,
    )
    print(f"Focal alpha (inverse-freq, normalized): {focal_alpha}")

    study_config["trainer.losses"] = [
        "src.losses.DiceLoss.GeneralizedDiceLoss",
        "src.losses.LossFunctions.FocalLoss",
        "src.losses.DiceLoss.BoundaryLoss",
        "src.losses.OverflowLoss.OverflowLoss",
    ]
    study_config["trainer.losses.parameters"] = [
        {"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05},
        {"gamma": 2.0, "alpha": focal_alpha, "reduction": "mean"},
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
        focal_loss_weight,
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

    # Higher default LR for training from scratch (2e-4 vs 1e-5 fine-tune).
    if LR_OVERRIDE != "":
        study_config["trainer.optimizer.lr"] = float(LR_OVERRIDE)
    else:
        study_config["trainer.optimizer.lr"] = 2e-4
    if LR_SCALE != 1.0:
        study_config["trainer.optimizer.lr"] = float(study_config["trainer.optimizer.lr"]) * LR_SCALE

    # Spike monitoring
    study_config["experiment.logging.spike_watch.enabled"] = True
    dice_spike_keys = [f"GeneralizedDiceLoss/mask_{i}" for i in range(max(0, full_num_classes - 1))]
    study_config["experiment.logging.spike_watch.keys"] = [
        "FocalLoss/loss",
        "BoundaryLoss/loss",
        "GeneralizedDiceLoss/overall",
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

    # Class subset
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
            f"B2B_NoM1Pretrain_iOCT2D_dual_{random_word}_{study_config['model.channel_n']}"
        )

    if RESUME_MODEL_PATH != "":
        study_config["experiment.model_path"] = RESUME_MODEL_PATH

    return study_config


def get_dataset_args(study_config):
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
            "m1_freeze": study_config.get("model.m1.freeze", False),
            "m1_use_t0_for_loss": study_config.get("model.m1.use_t0_for_loss", False),
        },
    )

    study = Study(study_config)
    exp = EXP_OctreeNCA_DualView_B2B().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTPairedSequentialDatasetForExperiment,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
