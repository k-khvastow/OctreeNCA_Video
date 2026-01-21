from matplotlib import pyplot as plt
import configs
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.datasets.Dataset_Video2D import Video2DDataset
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
import os, torch
from datetime import datetime


DATA_ROOT = "/vol/data/BioProject13/data_OCT/OCT"
LABEL_ROOT = "/vol/data/BioProject13/data_OCT/Label/GT_Layers"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

study_config = {
    'experiment.name': f"Video2D_{timestamp}",
    'experiment.description': 'Training OctreeNCA on 2D video slices from OCTA dataset with 7 classes.',
    'model.output_channels': 8,
    'model.input_channels': 1,
    'experiment.use_wandb': True,
    'experiment.wandb_project': 'OctreeNCA_Video',
    'experiment.dataset.img_path': DATA_ROOT,
    'experiment.dataset.label_path': LABEL_ROOT,
    'experiment.dataset.seed': 42,
    'experiment.data_split': [0.7, 0.2, 0.1],
    'experiment.dataset.input_size': (400, 400),
    'experiment.dataset.transform_mode': 'resize', # Options: 'resize', 'crop'
    'trainer.num_steps_per_epoch': 1000,
    'trainer.batch_duplication': 1,
    'trainer.n_epochs': 10
}

# Merge default configs
study_config = study_config | configs.models.peso.peso_model_config # Using peso config as base for OctreeNCA params
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

# Experiment settings
study_config['experiment.logging.also_eval_on_train'] = False
study_config['experiment.logging.evaluate_interval'] = study_config['trainer.n_epochs'] + 1
study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                         "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]

# OctreeNCA Model Specifics
steps = 10
alpha = 1.0
# Adjust resolutions as per image size (640x400 from history). 
# Octree needs usually square or compatible dims.
# Let's assume we resize to something standard or keep as is if the model supports it.
# The user's PESO config had [[320,320], ...].
# 640x400 is not square. OctreeNCA might require padding or specific handling. 
# For now, let's use a config that fits 400x400 or just try to run it.
# Actually, the user's OctreeNCAV2 implementation seems to handle levels.
# Let's set a resolution hierarchy.
# Adjust resolutions for 400x400 input
study_config['model.octree.res_and_steps'] = [[[400,400], steps], [[200,200], steps], [[100,100], steps], [[50,50], steps], [[25,25], int(alpha * 20)]]

study_config['model.channel_n'] = 32
study_config['model.hidden_size'] = 128
study_config['trainer.batch_size'] = 4

dice_loss_weight = 0.95
ema_decay = 0.99
study_config['trainer.ema'] = ema_decay > 0.0
study_config['trainer.ema.decay'] = ema_decay

study_config['trainer.use_amp'] = True

study_config['trainer.losses'] = ["src.losses.DiceLoss.nnUNetSoftDiceLoss", "src.losses.LossFunctions.CrossEntropyLossWrapper"]
study_config['trainer.losses.parameters'] = [{"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05}, {}]
study_config['trainer.loss_weights'] = [dice_loss_weight, 2.0-dice_loss_weight]

study_config['model.normalization'] = "none"
study_config['model.apply_nonlin'] = "torch.nn.Softmax(dim=1)"

# Update experiment name with params

# Prepare Study and Experiment
study = Study(study_config)

# Initialize Experiment with Video2DDataset
# Note: Dataset_Video2D takes data_root, label_root, preload
dataset_args = {
    'data_root': DATA_ROOT,
    'label_root': LABEL_ROOT,
    'preload': False, # Set to True if RAM allows
    'num_classes': 8
}

# The EXP_OctreeNCA wrapper uses the dataset_class and dataset_args to instantiate the dataset
# It internally handles the train/test split if the dataset supports it or if the wrapper does.
# Looking at EXP_OctreeNCA in BaselineConfigs, it calls proper super().createExperiment.
# However, usually datasets in this project might need `set_experiment` or `setPaths` calls 
# if they inherit from Dataset_Base and the ExperimentWrapper expects that.
# Our Video2DDataset inherits from torch.utils.data.Dataset directly.
# Let's check ExperimentWrapper.py to see how it handles datasets.
# But for now, we pass the class and args.

exp = EXP_OctreeNCA().createExperiment(study_config, detail_config={}, dataset_class=Video2DDataset, dataset_args=dataset_args)

study.add_experiment(exp)

if __name__ == "__main__":
    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
    # study.eval_experiments(export_prediction=False) 
