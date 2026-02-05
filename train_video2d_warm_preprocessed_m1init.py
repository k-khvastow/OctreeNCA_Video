import configs
from src.datasets.Dataset_Video2D_Sequential_cached import Video2DSequentialDatasetCached
from src.utils.WarmStartM1InitConfig import EXP_OctreeNCA_WarmStart_M1Init
from src.utils.Study import Study
import wonderwords

DATA_ROOT = "/vol/data/BioProject13/data_OCT/npy_Cropped_400/"
LABEL_ROOT = "/vol/data/BioProject13/data_OCT/npy_Cropped_400/"

# Optional: train only a subset of foreground classes (background 0 is always kept).
# Example: [1, 2] -> model outputs 3 classes (background + 2 selected).
SELECTED_CLASSES = None  # e.g. [1, 2]

# Set this to your M1 checkpoint (.pth) or directory containing model.pth.
M1_CHECKPOINT_PATH = ""

r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


def get_study_config():
    study_config = {
        'experiment.name': r'OctreeNCA_Video_WarmStart_M1Init',
        'experiment.description': 'Warm start with M1 hidden state init, then M2 for sequential frames.',
        'model.output_channels': 6,
        'model.input_channels': 1,
        'experiment.use_wandb': True,
        'experiment.wandb_project': 'OctreeNCA_Video',
        'experiment.dataset.img_path': DATA_ROOT,
        'experiment.dataset.label_path': LABEL_ROOT,
        'experiment.dataset.seed': 42,
        'experiment.data_split': [0.7, 0.29, 0.01],
        'experiment.dataset.input_size': (400, 400),
        'experiment.dataset.transform_mode': 'crop', # Options: 'resize', 'crop'
        'trainer.num_steps_per_epoch': 200,
        'trainer.batch_duplication': 1,
        'trainer.n_epochs': 100,
    }

    # Merge default configs
    study_config = study_config | configs.models.peso.peso_model_config
    study_config = study_config | configs.trainers.nca.nca_trainer_config
    study_config = study_config | configs.tasks.segmentation.segmentation_task_config
    study_config = study_config | configs.default.default_config

    # Experiment settings
    study_config['experiment.logging.also_eval_on_train'] = False
    study_config['experiment.save_interval'] = 3
    study_config['experiment.logging.evaluate_interval'] = 40
    study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                             "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]
    study_config['trainer.n_epochs'] = 100

    # Model specifics
    steps = 10
    alpha = 1.0
    study_config['model.backbone_class'] = "BasicNCA2DFast"
    study_config['model.octree.separate_models'] = True
    study_config['model.octree.res_and_steps'] = [[[400,400], steps * 2], [[200,200], steps], [[100,100], steps], [[50,50], steps], [[25,25], int(alpha * 20 / 2)]]
    study_config['model.kernel_size'] = [5, 5, 5, 5, 5]
    study_config['model.octree.warm_start_steps'] = 10
    study_config['model.channel_n'] = 24
    study_config['model.hidden_size'] = 32
    study_config['trainer.batch_size'] = 6
    study_config['trainer.gradient_accumulation'] = 8
    study_config['trainer.normalize_gradients'] = 'all'

    # M1 init options
    study_config['model.m1.pretrained_path'] = "/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/Video2D_w3losses_deliberation_24_Training OctreeNCA on 2D video slices from OCTA dataset with 7 classes./models/epoch_99/model.pth"
    study_config['model.m1.freeze'] = True
    study_config['model.m1.use_first_frame'] = True
    study_config['model.m1.use_t0_for_loss'] = False
    study_config['model.m1.use_probs'] = False

    dice_loss_weight = 1.0
    boundary_loss_weight = 1
    ema_decay = 0.99
    study_config['trainer.ema'] = ema_decay > 0.0
    study_config['trainer.ema.decay'] = ema_decay
    study_config['trainer.use_amp'] = True

    study_config['trainer.losses'] = [
        "src.losses.DiceLoss.GeneralizedDiceLoss",
        "src.losses.LossFunctions.CrossEntropyLossWrapper",
        # "src.losses.OverflowLoss.OverflowLoss",
        "src.losses.DiceLoss.BoundaryLoss",
    ]
    study_config['trainer.losses.parameters'] = [
        {"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05},
        {},
        {},
        {"do_bg": False, "channel_last": True, "use_precomputed": True, "use_probabilities": False, "dist_clip": 20.0},
    ]
    study_config['trainer.loss_weights'] = [
        dice_loss_weight,
        2.0 - dice_loss_weight,
        # 1,
        boundary_loss_weight,
    ]

    study_config['experiment.dataset.precompute_boundary_dist'] = False
    study_config['experiment.dataset.boundary_dist_classes'] = None

    study_config['model.normalization'] = "none"
    study_config['model.apply_nonlin'] = "torch.nn.Softmax(dim=1)"

    # Spike monitoring (per-batch class counts + save batches on spikes)
    study_config['experiment.logging.spike_watch.enabled'] = True
    study_config['experiment.logging.spike_watch.keys'] = [
        "CrossEntropyLossWrapper/loss",
        "nnUNetSoftDiceLoss/mask_3",
        "nnUNetSoftDiceLoss/mask_4",
    ]
    study_config['experiment.logging.spike_watch.window'] = 50
    study_config['experiment.logging.spike_watch.zscore'] = 3.0
    study_config['experiment.logging.spike_watch.min_value'] = 0.2
    study_config['experiment.logging.spike_watch.max_images_per_epoch'] = 10
    study_config['experiment.logging.spike_watch.max_images_per_spike'] = 2
    study_config['experiment.logging.spike_watch.save_classes'] = [1, 2, 3, 4, 5]

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
        study_config['experiment.dataset.class_subset'] = cleaned
        study_config['model.output_channels'] = len(cleaned) + 1
        study_config['experiment.logging.spike_watch.save_classes'] = list(range(1, len(cleaned) + 1))
    else:
        study_config['experiment.dataset.class_subset'] = None

    study_config['experiment.name'] = f"WarmStart_M1Init_w3losses_{random_word}_{study_config['model.channel_n']}"

    return study_config


def get_dataset_args(study_config):
    return {
        'data_root': DATA_ROOT,
        'label_root': LABEL_ROOT,
        'sequence_length': 5,
        'num_classes': study_config['model.output_channels'],
        'input_size': study_config['experiment.dataset.input_size'],
    }


if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    study = Study(study_config)

    exp = EXP_OctreeNCA_WarmStart_M1Init().createExperiment(
        study_config,
        detail_config={},
        dataset_class=Video2DSequentialDatasetCached,
        dataset_args=dataset_args,
    )

    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
