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
        self.size = input_size # (H, W)

        self.sequences = [] # List of {'folder': str, 'start_frame': int}

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
                        self.sequences.append({
                            'folder': item,
                            'start_frame': i,
                            'image_names': img_list[i : i+sequence_length]
                        })
        
        print(f"Found {len(self.sequences)} valid sequences of length {sequence_length}.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        meta = self.sequences[index]
        folder = meta['folder']
        start_frame = meta['start_frame']
        img_names = meta['image_names']
        
        label_path = os.path.join(self.label_root, f"{folder}.mat")
        layers_data = self._load_mat_data(label_path) 

        imgs = []
        masks = []
        
        # Determine strict Center Crop
        first_img_path = os.path.join(self.data_root, folder, img_names[0])
        first_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = first_img.shape
        
        th, tw = self.size
        # Fallback to resize if smaller, otherwise crop
        if orig_w >= tw and orig_h >= th:
            crop_x = (orig_w - tw) // 2
            crop_y = (orig_h - th) // 2
            mode = 'crop'
        else:
            mode = 'resize'
            crop_x, crop_y = 0, 0

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.data_root, folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            
            abs_frame_idx = start_frame + i
            mask = self._generate_mask(layers_data, abs_frame_idx, orig_h, orig_w)
            
            if mode == 'crop':
                img = img[crop_y:crop_y+th, crop_x:crop_x+tw]
                mask = mask[crop_y:crop_y+th, crop_x:crop_x+tw]
            else:
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            
            imgs.append(img[np.newaxis, ...]) # (1, H, W)
            masks.append(mask)

        imgs_tensor = torch.from_numpy(np.stack(imgs)).float() # (T, 1, H, W)
        masks_np = np.stack(masks) # (T, H, W)
        masks_tensor = torch.from_numpy(masks_np).long()
        masks_onehot = torch.nn.functional.one_hot(masks_tensor, num_classes=self.num_classes)
        masks_onehot = masks_onehot.permute(0, 3, 1, 2).float() # (T, C, H, W)

        return {
            'image_sequence': imgs_tensor, 
            'label_sequence': masks_onehot,
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