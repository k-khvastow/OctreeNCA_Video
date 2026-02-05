import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


class iOCTSequentialDataset(Dataset):
    """
    Dataset for iOCT data with temporal sequences.
    
    Structure:
        data_root/
        ├── peeling/Bscans-dt/
        │   ├── A/
        │   │   ├── Image/
        │   │   └── Segmentation/
        │   └── B/
        │       ├── Image/
        │       └── Segmentation/
        └── sri/Bscans-dt/
            ├── A/
            └── B/
    
    Args:
        data_root: Root directory containing peeling/sri folders
        datasets: List of datasets to include, e.g., ['peeling', 'sri']
        views: List of views to include, e.g., ['A', 'B']
        sequence_length: Number of consecutive frames per sequence
        sequence_step: Step size between sequences (default: 1 for overlap)
        num_classes: Number of segmentation classes
        input_size: Target size (H, W) for resizing
        split: 'train', 'val', or 'test'
        train_val_test_split: Tuple of (train, val, test) ratios
        seed: Random seed for splits
    """
    
    # Mapping from RGB values to class indices
    # Based on actual iOCT data inspection
    RGB_TO_CLASS = {
        (0, 0, 0): 0,          # Background (black)
        (255, 0, 0): 1,        # Class 1 (red)
        (0, 255, 209): 2,      # Class 2 (cyan)
        (61, 255, 0): 3,       # Class 3 (green)
        (0, 78, 255): 4,       # Class 4 (blue)
        (255, 189, 0): 5,      # Class 5 (yellow/orange)
    }
    
    def __init__(
        self,
        data_root: str,
        datasets: List[str] = ['peeling', 'sri'],
        views: List[str] = ['A', 'B'],
        sequence_length: int = 5,
        sequence_step: int = 1,
        num_classes: int = 6,
        input_size: Tuple[int, int] = (400, 400),
        split: str = 'train',
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.views = views
        self.sequence_length = sequence_length
        self.sequence_step = sequence_step
        self.num_classes = num_classes
        self.input_size = input_size
        self.split = split
        
        # Collect all valid sequences
        self.sequences = self._build_sequences()
        
        # Split into train/val/test
        self.sequences = self._split_data(self.sequences, train_val_test_split, seed)
        
        print(f"Loaded {len(self.sequences)} sequences for {split} split")
        print(f"  Datasets: {datasets}, Views: {views}")
        print(f"  Sequence length: {sequence_length}, Step: {sequence_step}")
    
    def _build_sequences(self) -> List[dict]:
        """Build list of all valid sequences."""
        sequences = []
        
        for dataset_name in self.datasets:
            for view in self.views:
                base_path = self.data_root / dataset_name / "Bscans-dt" / view
                image_dir = base_path / "Image"
                seg_dir = base_path / "Segmentation"
                
                if not image_dir.exists() or not seg_dir.exists():
                    print(f"Warning: Skipping {dataset_name}/{view} - directories not found")
                    continue
                
                # Get sorted list of frame numbers
                image_files = sorted(list(image_dir.glob("*.png")))
                frame_nums = [int(f.stem) for f in image_files]
                
                # Create sequences
                for i in range(0, len(frame_nums) - self.sequence_length + 1, self.sequence_step):
                    seq_frames = frame_nums[i:i + self.sequence_length]
                    
                    # Verify all frames exist
                    valid = True
                    for frame_num in seq_frames:
                        img_path = image_dir / f"{frame_num:05d}.png"
                        seg_path = seg_dir / f"{frame_num:05d}.png"
                        if not img_path.exists() or not seg_path.exists():
                            valid = False
                            break
                    
                    if valid:
                        sequences.append({
                            'dataset': dataset_name,
                            'view': view,
                            'frames': seq_frames,
                            'image_dir': image_dir,
                            'seg_dir': seg_dir
                        })
        
        return sequences
    
    def _split_data(
        self, 
        sequences: List[dict], 
        split_ratios: Tuple[float, float, float],
        seed: int
    ) -> List[dict]:
        """Split sequences into train/val/test."""
        np.random.seed(seed)
        n_total = len(sequences)
        
        # Shuffle sequences
        indices = np.random.permutation(n_total)
        
        # Calculate split sizes
        train_ratio, val_ratio, test_ratio = split_ratios
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1"
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split indices
        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            selected_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        return [sequences[i] for i in selected_indices]
    
    def _rgb_to_class(self, rgb_seg: np.ndarray) -> np.ndarray:
        """Convert RGB segmentation to class indices."""
        h, w = rgb_seg.shape[:2]
        class_seg = np.zeros((h, w), dtype=np.int64)
        
        for rgb_val, class_idx in self.RGB_TO_CLASS.items():
            # Match all three channels
            mask = (rgb_seg[:, :, 0] == rgb_val[0]) & \
                   (rgb_seg[:, :, 1] == rgb_val[1]) & \
                   (rgb_seg[:, :, 2] == rgb_val[2])
            class_seg[mask] = class_idx
        
        return class_seg
    
    def _load_frame(self, image_path: Path, seg_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a single frame."""
        # Load image (RGB)
        img = np.array(Image.open(image_path))
        
        # Load segmentation (RGB) and convert to class indices
        seg_rgb = np.array(Image.open(seg_path))
        seg = self._rgb_to_class(seg_rgb)
        
        # Convert RGB to grayscale for input
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        
        # Resize if needed
        if img.shape != self.input_size:
            img = np.array(Image.fromarray(img).resize(
                (self.input_size[1], self.input_size[0]), 
                Image.BILINEAR
            ))
            seg = np.array(Image.fromarray(seg.astype(np.uint8)).resize(
                (self.input_size[1], self.input_size[0]), 
                Image.NEAREST
            ))
        
        return img, seg
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'image': Tensor of shape (T, 1, H, W)
                'label': Tensor of shape (T, H, W)
                'metadata': Dict with dataset info
        """
        seq_info = self.sequences[idx]
        
        images = []
        labels = []
        
        for frame_num in seq_info['frames']:
            img_path = seq_info['image_dir'] / f"{frame_num:05d}.png"
            seg_path = seq_info['seg_dir'] / f"{frame_num:05d}.png"
            
            img, seg = self._load_frame(img_path, seg_path)
            images.append(img)
            labels.append(seg)
        
        # Stack into sequences
        images = np.stack(images, axis=0)  # (T, H, W)
        labels = np.stack(labels, axis=0)  # (T, H, W)
        
        # Add channel dimension for images
        images = images[:, None, :, :]  # (T, 1, H, W)
        
        # Normalize images to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Convert to tensors
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels).long()
        
        return {
            'image': images,
            'label': labels,
            'metadata': {
                'dataset': seq_info['dataset'],
                'view': seq_info['view'],
                'frames': seq_info['frames']
            }
        }


# Alternative: Single-frame dataset (no temporal sequences)
class iOCTDataset(Dataset):
    """Simple single-frame iOCT dataset without temporal sequences."""
    
    RGB_TO_CLASS = iOCTSequentialDataset.RGB_TO_CLASS
    
    def __init__(
        self,
        data_root: str,
        datasets: List[str] = ['peeling', 'sri'],
        views: List[str] = ['A', 'B'],
        num_classes: int = 6,
        input_size: Tuple[int, int] = (400, 400),
        split: str = 'train',
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.views = views
        self.num_classes = num_classes
        self.input_size = input_size
        self.split = split
        
        # Collect all frames
        self.frames = self._collect_frames()
        
        # Split
        self.frames = self._split_data(self.frames, train_val_test_split, seed)
        
        print(f"Loaded {len(self.frames)} frames for {split} split")
    
    def _collect_frames(self) -> List[dict]:
        """Collect all individual frames."""
        frames = []
        
        for dataset_name in self.datasets:
            for view in self.views:
                base_path = self.data_root / dataset_name / "Bscans-dt" / view
                image_dir = base_path / "Image"
                seg_dir = base_path / "Segmentation"
                
                if not image_dir.exists():
                    continue
                
                for img_path in sorted(image_dir.glob("*.png")):
                    seg_path = seg_dir / img_path.name
                    if seg_path.exists():
                        frames.append({
                            'dataset': dataset_name,
                            'view': view,
                            'image_path': img_path,
                            'seg_path': seg_path
                        })
        
        return frames
    
    def _split_data(self, frames: List[dict], split_ratios: Tuple[float, float, float], seed: int) -> List[dict]:
        """Split frames into train/val/test."""
        np.random.seed(seed)
        n_total = len(frames)
        indices = np.random.permutation(n_total)
        
        train_ratio, val_ratio, _ = split_ratios
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        else:
            selected_indices = indices[n_train + n_val:]
        
        return [frames[i] for i in selected_indices]
    
    def _rgb_to_class(self, rgb_seg: np.ndarray) -> np.ndarray:
        """Convert RGB segmentation to class indices."""
        h, w = rgb_seg.shape[:2]
        class_seg = np.zeros((h, w), dtype=np.int64)
        
        for rgb_val, class_idx in self.RGB_TO_CLASS.items():
            mask = (rgb_seg[:, :, 0] == rgb_val[0]) & \
                   (rgb_seg[:, :, 1] == rgb_val[1]) & \
                   (rgb_seg[:, :, 2] == rgb_val[2])
            class_seg[mask] = class_idx
        
        return class_seg
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'image': Tensor of shape (1, H, W)
                'label': Tensor of shape (H, W)
                'metadata': Dict with dataset info
        """
        frame_info = self.frames[idx]
        
        # Load image
        img = np.array(Image.open(frame_info['image_path']))
        seg_rgb = np.array(Image.open(frame_info['seg_path']))
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        
        # Convert seg to class indices
        seg = self._rgb_to_class(seg_rgb)
        
        # Resize
        if img.shape != self.input_size:
            img = np.array(Image.fromarray(img).resize(
                (self.input_size[1], self.input_size[0]), 
                Image.BILINEAR
            ))
            seg = np.array(Image.fromarray(seg.astype(np.uint8)).resize(
                (self.input_size[1], self.input_size[0]), 
                Image.NEAREST
            ))
        
        # Normalize and add channel dim
        img = img.astype(np.float32) / 255.0
        img = img[None, :, :]  # (1, H, W)
        
        # Convert to tensors
        img = torch.from_numpy(img)
        seg = torch.from_numpy(seg).long()
        
        return {
            'image': img,
            'label': seg,
            'metadata': {
                'dataset': frame_info['dataset'],
                'view': frame_info['view'],
                'path': str(frame_info['image_path'])
            }
        }
