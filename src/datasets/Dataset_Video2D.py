import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import natsort
import cv2

class Video2DDataset(Dataset):
    def __init__(self, data_root, label_root, preload=False, num_classes=8):
        """
        Args:
            data_root (str): Root directory containing subdirectories of image sequences (e.g., .../OCT).
            label_root (str): Root directory containing .mat files (e.g., .../GT_Layers).
            preload (bool): If True, load all data into RAM.
            num_classes (int): Number of segmentation classes (output channels).
        """
        self.data_root = data_root
        self.label_root = label_root
        self.preload = preload
        self.num_classes = num_classes

        
        self.samples = [] # List of dicts: {'folder': str, 'image_name': str, 'frame_index': int}
        
        # 1. Scan for valid data pairs
        if not os.path.exists(data_root):
             raise ValueError(f"Data root does not exist: {data_root}")
             
        candidates = natsort.natsorted(os.listdir(data_root))
        print(f"Scanning dataset at {data_root}...")
        
        for item in candidates:
            data_path = os.path.join(data_root, item)
            if os.path.isdir(data_path):
                # Check if corresponding label file exists
                label_path = os.path.join(label_root, f"{item}.mat")
                if os.path.exists(label_path):
                    # List all images in the subdirectory
                    img_list = natsort.natsorted([x for x in os.listdir(data_path) if x.endswith('.bmp')])
                    
                    if not img_list:
                        continue
                        
                    # Add each image as a separate sample
                    for idx, img_name in enumerate(img_list):
                        self.samples.append({
                            'folder': item,
                            'image_name': img_name,
                            'frame_index': idx
                        })
        
        print(f"Found {len(self.samples)} valid 2D samples.")
        
        if len(self.samples) == 0:
            print(f"Warning: No valid samples found in {data_root} with labels in {label_root}")

        # 2. Preload if requested
        self.cache = {}
        if self.preload:
            print("Preloading data... (This may take a while and consume a lot of RAM)")
            for i in range(len(self.samples)):
                self.cache[i] = self.load_sample(i)
                if i % 100 == 0:
                    print(f"Loaded {i}/{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.preload:
            return self.cache[index]
        else:
            return self.load_sample(index)

    def load_sample(self, index):
        meta = self.samples[index]
        folder = meta['folder']
        img_name = meta['image_name']
        frame_idx = meta['frame_index']
        
        data_dir = os.path.join(self.data_root, folder)
        img_path = os.path.join(data_dir, img_name)
        label_path = os.path.join(self.label_root, f"{folder}.mat")
        
        # 1. Load Image
        # Read as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
             raise RuntimeError(f"Failed to read image: {img_path}")
             
        # Convert to float32
        img = img.astype(np.float32)
        # Normalize? User provided code did not normalize, but typically we want [0,1] or similar
        # For now, keeping it raw as per user snippet, but converting to tensor requires float/double.
        
        depth, width = img.shape
        
        # 2. Generate Segmentation Mask
        # We need the layers data. 
        # Caching optimization: if we just loaded this file for the previous frame, reuse it.
        layers_data = self._get_mat_data(label_path)
        
        # layers_data shape: (num_layers, num_frames, width)
        # Check standard consistency
        num_layers = layers_data.shape[0]
        total_frames = layers_data.shape[1]
        
        if frame_idx >= total_frames:
            # Fallback or error?
            # If the image list has more frames than the label file, we cannot generate a mask.
            raise RuntimeError(f"Frame index {frame_idx} out of bounds for label {label_path} with {total_frames} frames.")
            
        mask = np.zeros((depth, width), dtype=np.float32)
        y_grid = np.arange(depth).reshape(depth, 1) # (H, 1)
        
        for i in range(num_layers):
            layer_surface = layers_data[i, frame_idx, :] # (width,)
            layer_surface_expanded = layer_surface.reshape(1, width) # (1, W)
            mask += (y_grid > layer_surface_expanded).astype(np.float32)
            
        # Convert to Torch Tensors
        # Image: (1, H, W)
        data_tensor = torch.from_numpy(img).unsqueeze(0) 
        
        # Label: One-Hot Encoding for num_classes
        # mask is (H, W) with values 0..N
        # We need (C, H, W)
        mask = torch.from_numpy(mask).long()
        label_tensor = torch.nn.functional.one_hot(mask, num_classes=self.num_classes).permute(2, 0, 1).float()
        
        # Return dict as expected by MedNCAAgent
        return {
            'image': data_tensor,
            'label': label_tensor,
            'id': f"{folder}_{frame_idx}",
            'folder': folder,
            'frame_index': frame_idx
        }

    # Simple LRU-1 cache for the .mat file content
    _last_mat_path = None
    _last_mat_data = None

    def _get_mat_data(self, path):
        if self._last_mat_path == path:
            return self._last_mat_data
        
        mat = scipy.io.loadmat(path)
        if 'Layer' not in mat:
             raise ValueError(f".mat file {path} does not contain 'Layer' key")
        
        data = mat['Layer']
        self._last_mat_path = path
        self._last_mat_data = data
        return data
