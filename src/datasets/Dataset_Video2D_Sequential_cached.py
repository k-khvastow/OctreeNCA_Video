import os
import torch
import numpy as np
import natsort
from src.datasets.Dataset_Base import Dataset_Base

class Video2DSequentialDatasetCached(Dataset_Base):
    def __init__(self, data_root, label_root=None, sequence_length=5, step=5, num_classes=7, input_size=(400, 400)):
        super().__init__()
        # Point to the folder containing the flat .npz files
        # data_root should be the parent folder, e.g. ".../npy_Cropped_400"
        self.image_root = os.path.join(data_root, "images")
        self.label_root = os.path.join(data_root, "labels")
        self.sequence_length = sequence_length
        self.step = step
        self.num_classes = num_classes
        self.size = input_size 

        self.sequences = [] 
        self.sequences_dict = {}

        if not os.path.exists(self.image_root):
            raise ValueError(f"Image root not found: {self.image_root}")

        print(f"Scanning cached sequences at {self.image_root}...")
        
        # 1. Gather all files and group by Patient ID
        all_files = natsort.natsorted([f for f in os.listdir(self.image_root) if f.endswith('.npz')])
        patient_groups = {}

        for f in all_files:
            # Filename format: {patient_id}_{frame_idx}.npz
            # Example: "Patient001_0.npz"
            base_name = f.replace('.npz', '')
            parts = base_name.rsplit('_', 1)
            patient_id = parts[0]
            
            if patient_id not in patient_groups:
                patient_groups[patient_id] = []
            patient_groups[patient_id].append(f)

        # 2. Generate Sequences (Sliding Window with Step)
        for patient_id, file_list in patient_groups.items():
            num_frames = len(file_list)

            # The required number of frames to form a sequence with stepping
            required_span = (self.sequence_length - 1) * self.step + 1
            
            if num_frames >= required_span:
                # We slide the window one frame at a time
                for i in range(num_frames - required_span + 1):
                    # Get the indices for the current sequence based on step
                    indices = [i + j * self.step for j in range(self.sequence_length)]
                    
                    # Select the window of files using the calculated indices
                    window_files = [file_list[idx] for idx in indices]
                    
                    # Use the first file's frame index for the sequence ID
                    start_frame_str = window_files[0].replace('.npz', '').rsplit('_', 1)[1]
                    seq_id = f"{patient_id}_{start_frame_str}"
                    
                    seq_data = {
                        'id': seq_id,
                        'patient_id': patient_id,
                        'file_names': window_files
                    }
                    
                    self.sequences.append(seq_data)
                    self.sequences_dict[seq_id] = seq_data

        print(f"Found {len(self.sequences)} valid sequences of length {sequence_length} with step {self.step}.")

    def getFilesInPath(self, path: str):
        # Helper for the Splitter to know what IDs exist
        return {k: {'id': k} for k in self.sequences_dict.keys()}

    def setPaths(self, images_path: str, images_list: list, labels_path: str, labels_list: list) -> None:
        super().setPaths(images_path, images_list, labels_path, labels_list)
        # Filter based on the experiment split
        # Expects images_list to contain the sequence IDs (e.g., "Patient01_0")
        self.sequences = [self.sequences_dict[uid] for uid in self.images_list if uid in self.sequences_dict]
        print(f"Dataset split set. Active samples: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        meta = self.sequences[index]
        file_names = meta['file_names']
        
        imgs = []
        masks = []
        
        try:
            # 3. Load Sequence
            for f_name in file_names:
                img_path = os.path.join(self.image_root, f_name)
                lbl_path = os.path.join(self.label_root, f_name)
                
                # Load compressed arrays
                # Extract 'data' because we used np.savez_compressed(..., data=arr)
                img_arr = np.load(img_path)['data'] # Shape: (H, W, 1), uint8
                msk_arr = np.load(lbl_path)['data'] # Shape: (H, W), uint8
                
                # Normalize Image (uint8 -> float32 0..1)
                img_processed = img_arr.astype(np.float32) / 255.0
                
                # Handle Image Dimensions (Permute if needed)
                # Current shape: (H, W, 1). We usually want (1, H, W) for PyTorch stacking
                img_processed = np.transpose(img_processed, (2, 0, 1)) 
                
                imgs.append(img_processed)
                masks.append(msk_arr)
                
            # 4. Stack and Format
            # imgs stack: (T, 1, H, W)
            imgs_np = np.stack(imgs) 
            
            # masks stack: (T, H, W)
            masks_np = np.stack(masks)
            
            # One-Hot Encoding for Masks
            masks_tensor = torch.from_numpy(masks_np).long()
            masks_onehot = torch.nn.functional.one_hot(masks_tensor, num_classes=self.num_classes)
            
            # Current OneHot shape: (T, H, W, C).
            # We need: (T, C, H, W)
            masks_onehot = masks_onehot.permute(0, 3, 1, 2).float().numpy()

            return {
                'image': imgs_np, 
                'label': masks_onehot,
                'id': meta['id']
            }

        except Exception as e:
            print(f"Error loading sequence {meta['id']}: {e}. Skipping...")
            # Fallback to next item
            return self.__getitem__((index + 1) % len(self.sequences))