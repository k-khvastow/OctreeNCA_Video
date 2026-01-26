from src.agents.Agent_MedNCA_Simple import MedNCAAgent
import numpy as np

class PreloadAgent(MedNCAAgent):
    def initialize(self):
        super().initialize()
        # Create a fixed permutation of the entire dataset to chunk deterministically
        dataset = self.exp.datasets['train']
        n_samples = len(dataset.all_samples_list)
        self.all_indices = np.random.permutation(n_samples)
        
    def initialize_epoch(self):
        super().initialize_epoch()
        
        current_step = self.exp.currentStep
        dataset = self.exp.datasets['train']
        
        # Calculate how many samples we need per epoch
        batch_size = self.exp.config['trainer.batch_size']
        steps = self.exp.config['trainer.num_steps_per_epoch']
        samples_per_epoch = batch_size * steps
        
        # Helper to get indices for a specific epoch number (handling wrap-around)
        def get_epoch_indices(epoch_idx):
            start = (epoch_idx * samples_per_epoch) % len(self.all_indices)
            end = start + samples_per_epoch
            
            if end > len(self.all_indices):
                # Wrap around
                return np.concatenate([self.all_indices[start:], self.all_indices[:end - len(self.all_indices)]])
            else:
                return self.all_indices[start:end]

        # 1. Identify chunks
        current_indices = get_epoch_indices(current_step)
        next_indices = get_epoch_indices(current_step + 1)
        next_next_indices = get_epoch_indices(current_step + 2)
        
        # Identify old indices to unload (from 2 epochs ago)
        # We keep current, next, and next_next. We drop anything before that.
        # Simplest way: unload everything NOT in (current + next + next_next)
        # Ideally, just unload the specific epoch that just fell out of the window.
        prev_indices = get_epoch_indices(current_step - 1)
        
        # 2. Trigger Preload / Unload
        # Note: Preload unique indices to avoid redundant work
        indices_to_load = np.unique(np.concatenate([current_indices, next_indices, next_next_indices]))
        dataset.preload(indices_to_load)
        
        # Unload previous epoch indices ONLY if they are not reused in the upcoming window (unlikely with sequential, but good practice)
        indices_to_keep = set(indices_to_load)
        indices_to_unload = [i for i in prev_indices if i not in indices_to_keep]
        dataset.unload(indices_to_unload)
        
        # 3. Set Active Subset for THIS epoch
        # This forces the generator to only sample from 'current_indices'
        dataset.set_active_subset(current_indices)
        
        # 4. Restart DataLoader workers
        # This is CRITICAL. Workers are forked processes; they need to be killed and respawned 
        # to see the updated 'dataset.samples' list we just set.
        # The shared 'cache' (Manager dict) persists across restarts.
        if 'train' in self.exp.data_loaders and hasattr(self.exp.data_loaders['train'], 'restart'):
            print("Restarting dataloader workers to apply new subset...")
            self.exp.data_loaders['train'].restart()