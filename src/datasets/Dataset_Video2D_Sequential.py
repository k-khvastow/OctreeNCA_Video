import os
import torch
import numpy as np
import scipy.io
import natsort
import cv2
from src.datasets.Dataset_Base import Dataset_Base

class Video2DSequentialDataset(Dataset_Base):
    def __init__(self, data_root, label_root, sequence_length=5, num_classes=7, input_size=(400, 400)):
        super().__init__()
        self.data_root = data_root
        self.label_root = label_root
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.size = input_size 

        self.sequences = [] 
        self.sequences_dict = {}

        if not os.path.exists(data_root):
            raise ValueError(f"Data root not found: {data_root}")

        candidates = natsort.natsorted(os.listdir(data_root))
        for item in candidates:
            data_path = os.path.join(data_root, item)
            label_path = os.path.join(label_root, f"{item}.mat")
            
            if os.path.isdir(data_path) and os.path.exists(label_path):
                img_list = natsort.natsorted([x for x in os.listdir(data_path) if x.endswith('.bmp')])
                num_frames = len(img_list)
                
                if num_frames >= sequence_length:
                    for i in range(0, num_frames - sequence_length + 1):
                        seq_id = f"{item}_{i}"
                        seq_data = {
                            'folder': item,
                            'start_frame': i,
                            'image_names': img_list[i : i+sequence_length],
                            'id': seq_id
                        }
                        self.sequences.append(seq_data)
                        self.sequences_dict[seq_id] = seq_data
        
        print(f"Found {len(self.sequences)} valid sequences of length {sequence_length}.")

    def getFilesInPath(self, path: str):
        return {k: {'id': k} for k in self.sequences_dict.keys()}

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
        super().setPaths(images_path, images_list, labels_path, labels_list)
        # Filter sequences to only those in the split
        self.sequences = [self.sequences_dict[uid] for uid in self.images_list if uid in self.sequences_dict]
        print(f"Dataset split set. Active samples: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # 1. Recursive Skip Logic for Metadata/Folder
        try:
            meta = self.sequences[index]
            folder = meta['folder']
            start_frame = meta['start_frame']
            img_names = meta['image_names']
            
            # Check folder existence
            if not os.path.exists(os.path.join(self.data_root, folder)):
                raise FileNotFoundError
        except Exception:
            # Skip to next
            return self.__getitem__((index + 1) % len(self.sequences))

        label_path = os.path.join(self.label_root, f"{folder}.mat")
        try:
            layers_data = self._load_mat_data(label_path)
        except Exception as e:
            print(f"Error loading labels for {folder}: {e}. Skipping.")
            return self.__getitem__((index + 1) % len(self.sequences))

        imgs = []
        masks = []
        
        # 2. Determine Crop/Resize Parameters based on the first image
        # (Assuming first image is representative; if corrupt, loop handles it)
        mode = 'resize'
        crop_x, crop_y = 0, 0
        
        # We need the dimensions. Try reading the first image.
        first_img_path = os.path.join(self.data_root, folder, img_names[0])
        probe_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
        
        if probe_img is not None:
            orig_h, orig_w = probe_img.shape
            th, tw = self.size
            if orig_w >= tw and orig_h >= th:
                crop_x = (orig_w - tw) // 2
                crop_y = (orig_h - th) // 2
                mode = 'crop'
            else:
                mode = 'resize'
        else:
            # If first image is dead, skip sequence
            print(f"First image corrupt: {first_img_path}. Skipping sequence.")
            return self.__getitem__((index + 1) % len(self.sequences))

        # 3. Load Sequence
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.data_root, folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # --- Safety Check: Skip Entire Sequence if any frame is bad ---
            if img is None:
                print(f"Warning: Corrupt image {img_name} in {folder}. Skipping sequence.")
                return self.__getitem__((index + 1) % len(self.sequences))
            
            # Convert to Float32 and Normalize immediately
            img_processed = img.astype(np.float32) / 255.0
            
            # Load Mask
            abs_frame_idx = start_frame + i
            mask = self._generate_mask(layers_data, abs_frame_idx, orig_h, orig_w)
            
            # Apply Transforms
            if mode == 'crop':
                img_processed = img_processed[crop_y:crop_y+th, crop_x:crop_x+tw]
                mask = mask[crop_y:crop_y+th, crop_x:crop_x+tw]
            else:
                img_processed = cv2.resize(img_processed, (tw, th), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            
            imgs.append(img_processed[np.newaxis, ...]) # (1, H, W)
            masks.append(mask)

        # 4. Stack and Finalize
        imgs_np = np.stack(imgs) # (T, 1, H, W)
        masks_np = np.stack(masks) # (T, H, W)
        
        imgs_np = imgs_np.astype(np.float32)
        
        masks_tensor = torch.from_numpy(masks_np).long()
        masks_onehot = torch.nn.functional.one_hot(masks_tensor, num_classes=self.num_classes)
        masks_onehot = masks_onehot.permute(0, 3, 1, 2).float().numpy() # (T, C, H, W)

        return {
            'image': imgs_np, 
            'label': masks_onehot,
            'id': f"{folder}_{start_frame}"
        }

    def _generate_mask(self, layers_data, frame_idx, h, w):
        num_layers = layers_data.shape[0]
        mask = np.zeros((h, w), dtype=np.float32)
        y_grid = np.arange(h).reshape(h, 1)
        for i in range(num_layers):
            layer_surface = layers_data[i, frame_idx, :].reshape(1, w)
            mask += (y_grid > layer_surface).astype(np.float32)
        return mask

    def _load_mat_data(self, path):
        mat = scipy.io.loadmat(path)
        return mat['Layer']