from matplotlib import pyplot as plt
import configs
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.datasets.Dataset_Video2D import Video2DDataset
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
import os, torch

# Define dataset paths
# Assuming standard paths based on user request history, but should double check or provide variables
# History mentions: 
# Data: /Users/khvastow/Work/TUM/m2nd semester/Comp Bio/data/OCTA_6mm/OCT
# Labels: /Users/khvastow/Work/TUM/m2nd semester/Comp Bio/data/Label/GT_Layers

DATA_ROOT = '/vol/data/BioProject13/data_OCT/OCT'
LABEL_ROOT = '/vol/data/BioProject13/data_OCT/Label/GT_Layers'

study_config = {
    'experiment.name': r'OctreeNCA_Video2D',
    'experiment.description': "OctreeNCA 2D Training on Video Frames",
    'model.output_channels': 3,
    'experiment.use_wandb': True,
    'experiment.wandb_project': 'OctreeNCA_Video'
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
study_config['model.octree.res_and_steps'] = [[[320,320], steps], [[160,160], steps], [[80,80], steps], [[40,40], steps], [[20,20], int(alpha * 20)]]

study_config['model.channel_n'] = 16
study_config['model.hidden_size'] = 64
study_config['trainer.batch_size'] = 4 

dice_loss_weight = 1.0
ema_decay = 0.99
study_config['trainer.ema'] = ema_decay > 0.0
study_config['trainer.ema.decay'] = ema_decay

study_config['trainer.losses'] = ["src.losses.DiceLoss.DiceLoss", "src.losses.BCELoss.BCELoss"]
study_config['trainer.losses.parameters'] = [{}, {}]
study_config['trainer.loss_weights'] = [dice_loss_weight, 2.0-dice_loss_weight]

study_config['model.normalization'] = "none"

# Update experiment name with params
study_config['experiment.name'] = f"Video2D_{study_config['model.normalization']}_{steps}_{alpha}_{study_config['model.channel_n']}"

# Prepare Study and Experiment
study = Study(study_config)

# Initialize Experiment with Video2DDataset
# Note: Dataset_Video2D takes data_root, label_root, preload
dataset_args = {
    'data_root': DATA_ROOT,
    'label_root': LABEL_ROOT,
    'preload': False # Set to True if RAM allows
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
