import os
import torch
import numpy as np
import natsort
from torch.utils.data import Dataset
from src.datasets.Dataset_Base import Dataset_Base

class Video2DDatasetCached(Dataset_Base):
    def __init__(self, data_root, label_root=None, preload=False, num_classes=8, transform_mode='none', input_size=None):
        super().__init__()
        
        self.image_root = os.path.join(data_root, "images")
        self.label_root = os.path.join(data_root, "labels")
        self.num_classes = num_classes
        
        self.slice = -1 
        self.delivers_channel_axis = True
        self.is_rgb = False

        self.samples = [] 
        self.all_samples = {}

        if not os.path.exists(self.image_root):
             raise ValueError(f"Processed image root not found: {self.image_root}")

        print(f"Scanning cached dataset at {self.image_root}...")
        
        # CHANGE 1: Search for .npz instead of .npy
        files = natsort.natsorted([f for f in os.listdir(self.image_root) if f.endswith('.npz')])
        
        for f in files:
            # Filename format: {patient_id}_{frame_idx}.npz
            base_name = f.replace('.npz', '')
            parts = base_name.rsplit('_', 1)
            patient_id = parts[0]
            frame_idx = int(parts[1])
            
            self.all_samples[base_name] = {
                'id': base_name,
                'patient_id': patient_id,
                'frame_index': frame_idx,
                'file_name': f
            }
            
        self.samples = list(self.all_samples.values())
        print(f"Found {len(self.samples)} cached samples.")

    def getFilesInPath(self, path: str):
        return self.all_samples

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
        super().setPaths(images_path, images_list, labels_path, labels_list)
        self.samples = [self.all_samples[uid] for uid in self.images_list if uid in self.all_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        meta = self.samples[index]
        file_name = meta['file_name']
        
        # CHANGE 2: Load .npz and extract the 'data' key
        # We use 'data' because in the preprocessing script we used: np.savez_compressed(..., data=img)
        img_container = np.load(os.path.join(self.image_root, file_name))
        mask_container = np.load(os.path.join(self.label_root, file_name))
        
        img = img_container['data']   # stored as uint8
        mask = mask_container['data'] # stored as uint8

        # CHANGE 3: Normalize on the fly
        # Convert uint8 (0-255) -> float32 (0.0-1.0)
        img = img.astype(np.float32) / 255.0

        # Mask is uint8, convert to int64 (long) for One-Hot
        mask_tensor = torch.from_numpy(mask).long()
        
        label_onehot = torch.nn.functional.one_hot(mask_tensor, num_classes=self.num_classes).numpy().astype(np.float32)

        return {
            'image': img, 
            'label': label_onehot,
            'id': meta['id'],
            'patient_id': meta['patient_id'],
            'frame_index': meta['frame_index']
        }