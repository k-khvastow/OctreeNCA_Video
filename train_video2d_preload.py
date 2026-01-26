from matplotlib import pyplot as plt
import configs
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.datasets.Dataset_Video2D_cached import Video2DDatasetCached
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
import os, torch
import wonderwords
import numpy as np
from multiprocessing import Manager
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2

# ==========================================
# 1. Custom Dataset with Shared Cache
# ==========================================
class PreloadableVideo2DDataset(Video2DDatasetCached):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep a reference to all samples since self.samples will be subsetted
        self.all_samples_list = list(self.all_samples.values())
        # Active subset for the current epoch
        self.samples = self.all_samples_list
        
        # Shared cache for multiprocessing
        self.manager = Manager()
        self.cache = self.manager.dict()

    def set_active_subset(self, indices):
        """Restrict the dataset to a specific list of indices for the current epoch."""
        self.samples = [self.all_samples_list[i] for i in indices]

    def preload(self, indices):
        """Load specific indices into the shared cache."""
        print(f"Preloading {len(indices)} samples...")
        for i in indices:
            meta = self.all_samples_list[i]
            sample_id = meta['id']
            
            if sample_id not in self.cache:
                try:
                    # Replicating load logic from Video2DDatasetCached
                    file_name = meta['file_name']
                    img_container = np.load(os.path.join(self.image_root, file_name))
                    mask_container = np.load(os.path.join(self.label_root, file_name))
                    
                    img = img_container['data'].astype(np.float32) / 255.0
                    mask = mask_container['data'] # uint8
                    
                    # Store tuple in cache
                    self.cache[sample_id] = (img, mask)
                except Exception as e:
                    print(f"Error preloading {sample_id}: {e}")

    def unload(self, indices):
        """Remove specific indices from the shared cache."""
        for i in indices:
            meta = self.all_samples_list[i]
            sample_id = meta['id']
            if sample_id in self.cache:
                del self.cache[sample_id]

    def __getitem__(self, index):
        # Map the subset index to the actual sample
        meta = self.samples[index]
        sample_id = meta['id']

        # 1. Try Cache
        if sample_id in self.cache:
            img, mask = self.cache[sample_id]
        else:
            # 2. Fallback Load
            file_name = meta['file_name']
            img_container = np.load(os.path.join(self.image_root, file_name))
            mask_container = np.load(os.path.join(self.label_root, file_name))
            img = img_container['data'].astype(np.float32) / 255.0
            mask = mask_container['data']

        # Process (Normalization done above)
        # One-Hot Encoding
        mask_tensor = torch.from_numpy(mask).long()
        label_onehot = torch.nn.functional.one_hot(mask_tensor, num_classes=self.num_classes).numpy().astype(np.float32)

        return {
            'image': img, 
            'label': label_onehot,
            'id': meta['id'],
            'patient_id': meta['patient_id'],
            'frame_index': meta['frame_index']
        }

# ==========================================
# 2. Custom Agent to Manage Preloading
# ==========================================
class PreloadAgent(MedNCAAgent):
    # REMOVED initialize() to prevent early access to self.exp.datasets
    
    def initialize_epoch(self):
        # 1. Lazy Initialization of Indices
        # We do this here because self.exp.datasets is not ready during agent.__init__
        if not hasattr(self, 'all_indices'):
            dataset = self.exp.datasets['train']
            n_samples = len(dataset.all_samples_list)
            # Create a fixed permutation to ensure deterministic chunks across epochs
            self.all_indices = np.random.permutation(n_samples)

        super().initialize_epoch()
        
        current_step = self.exp.currentStep # Logic step/epoch
        dataset = self.exp.datasets['train']
        
        # Calculate samples needed per epoch
        batch_size = self.exp.config['trainer.batch_size']
        # If using StepsPerEpochGenerator
        steps = self.exp.config.get('trainer.num_steps_per_epoch', len(dataset)//batch_size)
        samples_per_epoch = batch_size * steps
        
        def get_epoch_indices(epoch_idx):
            n = len(self.all_indices)
            start = (epoch_idx * samples_per_epoch) % n
            end = start + samples_per_epoch
            
            if end > n:
                return np.concatenate([self.all_indices[start:], self.all_indices[:end - n]])
            else:
                return self.all_indices[start:end]

        # 2. Calculate Windows
        current_indices = get_epoch_indices(current_step)
        next_indices = get_epoch_indices(current_step + 1)
        next_next_indices = get_epoch_indices(current_step + 2)
        
        # 3. Preload (Current + Next 2 Epochs)
        # Unique indices to load
        indices_to_load = np.unique(np.concatenate([current_indices, next_indices, next_next_indices]))
        dataset.preload(indices_to_load)
        
        # 4. Unload (Anything not in the active window)
        # Note: This scans the whole cache keys, which is safe
        active_ids = set([dataset.all_samples_list[i]['id'] for i in indices_to_load])
        cached_ids = list(dataset.cache.keys())
        ids_to_remove = [uid for uid in cached_ids if uid not in active_ids]
        
        if ids_to_remove:
            print(f"Unloading {len(ids_to_remove)} samples...")
            for uid in ids_to_remove:
                del dataset.cache[uid]

        # 5. Set Active Subset
        dataset.set_active_subset(current_indices)
        
        # 6. Restart DataLoader to apply subset change
        if 'train' in self.exp.data_loaders:
            loader = self.exp.data_loaders['train']
            # Check if it's the MultiThreadedAugmenter which supports restart
            if hasattr(loader, 'restart'):
                print("Restarting DataLoader workers...")
                loader.restart()

# ==========================================
# 3. Custom Experiment Wrapper
# ==========================================
class EXP_OctreeNCA_Preload(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        if dataset_class is None:
            assert False, "Dataset is None"
        
        model = OctreeNCA2DPatch2(config)
        
        assert config['model.batchnorm_track_running_stats'] == False
        assert config['trainer.gradient_accumulation'] == False
        assert config['trainer.train_quality_control'] == False
        assert config['experiment.task'] == 'segmentation'
        
        # USE CUSTOM AGENT
        agent = PreloadAgent(model)
        loss_function = WeightedLosses(config) 

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

# ==========================================
# Main Script
# ==========================================

DATA_ROOT = "/vol/data/BioProject13/data_OCT/npy_Cropped_400/"
LABEL_ROOT = "/vol/data/BioProject13/data_OCT/npy_Cropped_400/"
r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])

def get_study_config():
    study_config = {
        'experiment.name': r'OctreeNCA_Video2D_7Class',
        'experiment.description': 'Training OctreeNCA on 2D video slices from OCTA dataset with 7 classes.',
        'model.output_channels': 6,
        'model.input_channels': 1,
        'experiment.use_wandb': True,
        'experiment.wandb_project': 'OctreeNCA_Video',
        'experiment.dataset.img_path': DATA_ROOT,
        'experiment.dataset.label_path': LABEL_ROOT,
        'experiment.dataset.seed': 42,
        'experiment.data_split': [0.7, 0.29, 0.01],
        'experiment.dataset.input_size': (400, 400),
        'experiment.dataset.transform_mode': 'crop',
        'trainer.num_steps_per_epoch': 1000,
        'trainer.batch_duplication': 1,
        'trainer.n_epochs': 10
    }

    study_config = study_config | configs.models.peso.peso_model_config
    study_config = study_config | configs.trainers.nca.nca_trainer_config
    study_config = study_config | configs.tasks.segmentation.segmentation_task_config
    study_config = study_config | configs.default.default_config

    study_config['experiment.logging.also_eval_on_train'] = False
    study_config['experiment.save_interval'] = 3
    study_config['experiment.logging.evaluate_interval'] = 40
    study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                             "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]

    steps = 10
    alpha = 1.0
    study_config['model.octree.res_and_steps'] = [[[400,400], steps], [[200,200], steps], [[100,100], steps], [[50,50], steps], [[25,25], int(alpha * 20)]]

    study_config['model.channel_n'] = 24
    study_config['model.hidden_size'] = 32
    study_config['trainer.batch_size'] = 8

    dice_loss_weight = 1.0
    ema_decay = 0.99
    study_config['trainer.ema'] = ema_decay > 0.0
    study_config['trainer.ema.decay'] = ema_decay
    study_config['trainer.use_amp'] = True
    study_config['trainer.losses'] = ["src.losses.DiceLoss.nnUNetSoftDiceLoss", "src.losses.LossFunctions.CrossEntropyLossWrapper"]
    study_config['trainer.losses.parameters'] = [{"apply_nonlin": "torch.nn.Softmax(dim=1)", "batch_dice": True, "do_bg": False, "smooth": 1e-05}, {}]
    study_config['trainer.loss_weights'] = [dice_loss_weight, 2.0-dice_loss_weight]
    study_config['model.normalization'] = "none"
    study_config['model.apply_nonlin'] = "torch.nn.Softmax(dim=1)"
    study_config['experiment.name'] = f"Video2D_{random_word}_{study_config['model.channel_n']}"
    
    return study_config

def get_dataset_args(study_config):
    return {
        'data_root': DATA_ROOT,
        'label_root': LABEL_ROOT,
        'preload': False, # Disabled standard preload to use our custom dynamic preload
        'num_classes': study_config['model.output_channels'],
        'transform_mode': study_config['experiment.dataset.transform_mode'],
        'input_size': study_config['experiment.dataset.input_size']
    }

if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)
    
    study = Study(study_config)
    # Use the custom ExperimentWrapper and Dataset
    exp = EXP_OctreeNCA_Preload().createExperiment(
        study_config, 
        detail_config={}, 
        dataset_class=PreloadableVideo2DDataset, 
        dataset_args=dataset_args
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()