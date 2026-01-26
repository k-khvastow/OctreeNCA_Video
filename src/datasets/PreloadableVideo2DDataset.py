from multiprocessing import Manager
import numpy as np
import torch
import os
from src.datasets.Dataset_Video2D_cached import Video2DDatasetCached

class PreloadableVideo2DDataset(Video2DDatasetCached):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store all samples permanently
        self.all_samples_list = list(self.all_samples.values())
        # Active samples for the current epoch (subset)
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
                # Load data using the logic from the original __getitem__
                # We replicate the loading logic here to store it in cache
                try:
                    file_name = meta['file_name']
                    img_container = np.load(os.path.join(self.image_root, file_name))
                    mask_container = np.load(os.path.join(self.label_root, file_name))
                    
                    img = img_container['data'].astype(np.float32) / 255.0
                    mask = mask_container['data']
                    
                    # Store in cache
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
        # Map the index (0 to subset_len) to the actual sample metadata
        meta = self.samples[index]
        sample_id = meta['id']

        # 1. Try to get from Cache
        if sample_id in self.cache:
            img, mask = self.cache[sample_id]
        else:
            # 2. Fallback: Load from disk (same logic as original)
            file_name = meta['file_name']
            img_container = np.load(os.path.join(self.image_root, file_name))
            mask_container = np.load(os.path.join(self.label_root, file_name))
            img = img_container['data'].astype(np.float32) / 255.0
            mask = mask_container['data']

        # Process as usual (One-hot, etc.)
        mask_tensor = torch.from_numpy(mask).long()
        label_onehot = torch.nn.functional.one_hot(mask_tensor, num_classes=self.num_classes).numpy().astype(np.float32)

        return {
            'image': img, 
            'label': label_onehot,
            'id': meta['id'],
            'patient_id': meta['patient_id'],
            'frame_index': meta['frame_index']
        }