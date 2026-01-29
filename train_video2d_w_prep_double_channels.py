import configs
from src.datasets.Dataset_Video2D_Sequential_cached import Video2DSequentialDatasetCached
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses
from src.models.Model_OctreeNCA_WarmStart_DoubleChannels import OctreeNCA2DWarmStartDoubleChannels
from src.agents.Agent_OctreeNCA_WarmStart import OctreeNCAWarmStartAgent
from src.utils.Study import Study
import torch
import wonderwords

DATA_ROOT = "/vol/data/BioProject13/data_OCT/npy_Cropped_400/"
LABEL_ROOT = "/vol/data/BioProject13/data_OCT/npy_Cropped_400/"
r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])


class EXP_OctreeNCA_WarmStart_DoubleChannels(ExperimentWrapper):
    def createExperiment(self, study_config: dict, detail_config: dict = {}, dataset_class=None, dataset_args={}):
        config = study_config
        if dataset_class is None:
            assert False, "Dataset is None"

        model = OctreeNCA2DWarmStartDoubleChannels(config)
        agent = OctreeNCAWarmStartAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)


def get_study_config():
    study_config = {
        'experiment.name': r'OctreeNCA_Video_WarmStart_DoubleChannels',
        'experiment.description': 'Training OctreeNCA on Video with Warm Start and Double-Channel Temporal Input.',
        'model.output_channels': 6,
        'model.input_channels': 1,
        'experiment.use_wandb': True,
        'experiment.wandb_project': 'OctreeNCA_Video',
        'experiment.dataset.img_path': DATA_ROOT,
        'experiment.dataset.label_path': LABEL_ROOT,
        'experiment.dataset.seed': 42,
        'experiment.data_split': [0.998, 0.001, 0.001],
        'experiment.dataset.input_size': (400, 400),
        'trainer.num_steps_per_epoch': 1000,
        'trainer.batch_duplication': 1,
        'trainer.n_epochs': 20
    }

    # Merge default configs
    study_config = study_config | configs.models.peso.peso_model_config
    study_config = study_config | configs.trainers.nca.nca_trainer_config
    study_config = study_config | configs.tasks.segmentation.segmentation_task_config
    study_config = study_config | configs.default.default_config

    # Experiment settings
    study_config['experiment.logging.also_eval_on_train'] = False
    study_config['experiment.save_interval'] = 5
    study_config['experiment.logging.evaluate_interval'] = 100

    # Model Specifics
    # Resolution must match input_size for the top level
    steps = 10
    alpha = 1.0
    study_config['model.octree.res_and_steps'] = [[[400, 400], steps], [[200, 200], steps], [[100, 100], steps], [[50, 50], steps], [[25, 25], int(alpha * 20)]]
    study_config['model.octree.warm_start_steps'] = 10  # Reduced steps for warm start
    study_config['model.channel_n'] = 16
    study_config['model.hidden_size'] = 32
    study_config['trainer.batch_size'] = 3  # Sequence batch size
    study_config['trainer.gradient_accumulation'] = 8
    study_config['trainer.normalize_gradients'] = 'all'

    dice_loss_weight = 1.0
    ema_decay = 0.99
    study_config['trainer.ema'] = ema_decay > 0.0
    study_config['trainer.ema.decay'] = ema_decay
    study_config['trainer.use_amp'] = True

    study_config['trainer.losses'] = ["src.losses.DiceLoss.nnUNetSoftDiceLoss", "src.losses.LossFunctions.CrossEntropyLossWrapper", "src.losses.OverflowLoss.OverflowLoss"]
    study_config['trainer.losses.parameters'] = [{"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05}, {}, {}]
    study_config['trainer.loss_weights'] = [dice_loss_weight, 2.0 - dice_loss_weight, 1]

    study_config['model.normalization'] = "none"
    study_config['model.apply_nonlin'] = "torch.nn.Softmax(dim=1)"

    # Append previous frame outputs to the next-frame input (temporal conditioning)
    study_config['model.temporal_append_outputs'] = True
    # If True, use probabilities instead of logits for the appended channels
    study_config['model.temporal_append_use_probs'] = False

    study_config['experiment.name'] = f"WarmStart_DoubleCh_{random_word}_{study_config['model.channel_n']}"

    return study_config


def get_dataset_args(study_config):
    return {
        'data_root': DATA_ROOT,
        'label_root': LABEL_ROOT,
        'sequence_length': 5,  # Temporal sequence length
        'num_classes': study_config['model.output_channels'],
        'input_size': study_config['experiment.dataset.input_size']
    }


if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    study = Study(study_config)

    exp = EXP_OctreeNCA_WarmStart_DoubleChannels().createExperiment(
        study_config,
        detail_config={},
        dataset_class=Video2DSequentialDatasetCached,
        dataset_args=dataset_args
    )

    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()
