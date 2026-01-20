import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import natsort
import cv2
from src.datasets.Dataset_Base import Dataset_Base

class Video2DDataset(Dataset_Base):
    def __init__(self, data_root, label_root, preload=False, num_classes=8, transform_mode='resize'):
        super().__init__()
        """
        Args:
            data_root (str): Root directory containing subdirectories of image sequences (e.g., .../OCT).
            label_root (str): Root directory containing .mat files (e.g., .../GT_Layers).
            preload (bool): If True, load all data into RAM.
            num_classes (int): Number of segmentation classes (output channels).
            transform_mode (str): 'resize' or 'crop'.
        """
        self.data_root = data_root
        self.label_root = label_root
        self.preload = preload
        self.num_classes = num_classes
        self.transform_mode = transform_mode

        # Attributes required by Agent_UNet / Agent_MedNCA
        self.slice = -1 
        self.delivers_channel_axis = True # We return HWC now
        self.is_rgb = False # Grayscale images

        self.all_samples = {}
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
                        sample_id = f"{item}_{idx}"
                        self.all_samples[sample_id] = {
                            'folder': item,
                            'image_name': img_name,
                            'frame_index': idx,
                            'id': sample_id
                        }
        
        self.samples = list(self.all_samples.values())
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

    def getFilesInPath(self, path: str):
        """
        Return a dictionary of all files/samples. 
        Experiment expects {id: ...}. The values don't matter keenly for splitting as long as keys are consistent.
        """
        return self.all_samples

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
        """
        Called by Experiment to set the active split (train/val/test).
        """
        super().setPaths(images_path, images_list, labels_path, labels_list)
        # Filter samples based on images_list (which contains IDs)
        self.samples = [self.all_samples[uid] for uid in self.images_list if uid in self.all_samples]

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
             
        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0

        if hasattr(self, 'size') and self.size is not None:
             target_size = (self.size[1], self.size[0]) # size is (H, W) -> (W, H)
             
             if self.transform_mode == 'crop':
                 # Center crop
                 h, w = img.shape
                 th, tw = self.size[0], self.size[1]
                 x1 = int(round((w - tw) / 2.))
                 y1 = int(round((h - th) / 2.))
                 # Handle cases where image is smaller than target
                 if x1 < 0 or y1 < 0:
                      # Fallback to resize or pad? User asked for crop from middle.
                      # Ideally pad. For now, let's just resize if too small, or let array slicing handle it (risky)
                      # Safest simple approach: Resize if smaller, Crop if larger
                      if w < tw or h < th:
                          img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                      else:
                          img = img[y1:y1+th, x1:x1+tw]
                 else:
                     img = img[y1:y1+th, x1:x1+tw]
             else:
                 # Resize
                 img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

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
            layer_surface_expanded = layer_surface.reshape(1, width) # (1, W)
            mask += (y_grid > layer_surface_expanded).astype(np.float32)

        # Resize Mask if self.size is set
        if hasattr(self, 'size') and self.size is not None:
             target_size = (self.size[1], self.size[0])
             
             if self.transform_mode == 'crop':
                 # Re-calculate crop coordinates for mask (reuse logic or recompute)
                 h, w = mask.shape
                 th, tw = self.size[0], self.size[1]
                 x1 = int(round((w - tw) / 2.))
                 y1 = int(round((h - th) / 2.))
                 
                 if w < tw or h < th:
                      mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                 else:
                      mask = mask[y1:y1+th, x1:x1+tw]
             else:
                 mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
        
        # 3. Format Output
        # Return HWC numpy arrays for compatibility with BatchgeneratorsDataLoader -> NumpyToTensor
        
        # Image: (H, W) -> (H, W, 1)
        img = img[..., np.newaxis]
        
        # Label: One-Hot Encoding
        # mask is (H, W). We want (H, W, C)
        # Using torch for convenient one_hot, then back to numpy
        mask_tensor = torch.from_numpy(mask).long()
        label_onehot = torch.nn.functional.one_hot(mask_tensor, num_classes=self.num_classes).numpy().astype(np.float32)
        
        # Return dict
        return {
            'image': img, 
            'label': label_onehot,
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
