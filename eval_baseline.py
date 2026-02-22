import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import random
#import copy
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from PIL import Image
from pathlib import Path


class IOCTDataset(Dataset):
        
    # Mapping from RGB values to class indices
    RGB_TO_CLASS = {
        (0, 0, 0): 0,          # Background (black)
        (255, 0, 0): 1,        # Class 1 (red)
        (0, 255, 209): 2,      # Class 2 (cyan)
        (61, 255, 0): 3,       # Class 3 (green)
        (0, 78, 255): 4,       # Class 4 (blue)
        (255, 189, 0): 5,      # Class 5 (yellow/orange)
        (218, 0, 255): 6,      # Class 6 (magenta)
    }

    def __init__(self, data_root, input_size=(512, 512), subset_videos=None):
        self.data_root = Path(data_root)
        self.input_size = input_size
        self.num_classes = 7
        self.subset_videos = subset_videos
        self.frames = []
        self._collect_frames_new()
    # method for the first iOCT data (peeling/Bscans-dt and sri/Bscans-dt)
    def _collect_frames(self):
        datasets = ["peeling/Bscans-dt", "sri/Bscans-dt"]
        views = ["A", "B"]
        total_count = 0
        print(f"Scanning {self.data_root}...")
        
        for d_name in datasets:
            for view in views:
                base_path = self.data_root / d_name / view
                img_dir = base_path / "Image"
                seg_dir = base_path / "Segmentation"

                if not img_dir.exists() or not seg_dir.exists():
                    continue
                
                # Find files
                files = sorted(list(img_dir.glob("*.png")))
                #count = 0
                for img_path in files:
                    mask_path = seg_dir / img_path.name
                
                    if mask_path.exists():
                        self.frames.append({
                            'image_path': str(img_path),
                            'mask_path': str(mask_path),
                            'id': f"{d_name}_{view}_{img_path.name}"
                        })
                        total_count += 1
                
        print(f"Total frames collected: {total_count}")  
    # method for the new iOCT data (peeling1, peeling2, peeling3)
    def _collect_frames_new(self):
        print(f"Scanning {self.data_root} for videos: {self.subset_videos}...")
        
        # Look for peeling1, peeling2, peeling3...
        for video_folder in self.data_root.glob("peeling*"):
            if not video_folder.is_dir(): continue
            
            video_id = video_folder.name # e.g. "peeling1"
            
            # Skip if this video is not in our requested subset
            if self.subset_videos is not None and video_id not in self.subset_videos:
                continue
                
            bscan_dir = video_folder / "iOCT" / "Bscan"
            if not bscan_dir.exists():
                print(f"  Warning: No iOCT/Bscan folder found in {video_id}")
                continue
            
            # Iterate through frame indices (005, 006, etc.)
            count = 0
            for frame_folder in bscan_dir.iterdir():
                if not frame_folder.is_dir(): continue
                
                # Check View 00
                img_00 = frame_folder / "00.png"
                seg_00 = frame_folder / "00_seg.png"
                if img_00.exists() and seg_00.exists():
                    self.frames.append({
                        'image_path': str(img_00),
                        'mask_path': str(seg_00),
                        'video_id': video_id
                    })
                    count += 1
                
                # Check View 01
                img_01 = frame_folder / "01.png"
                seg_01 = frame_folder / "01_seg.png"
                if img_01.exists() and seg_01.exists():
                    self.frames.append({
                        'image_path': str(img_01),
                        'mask_path': str(seg_01),
                        'video_id': video_id
                    })
                    count += 1
                    
            if count > 0:
                print(f"  -> Loaded {count} frames from {video_id}")

    def _rgb_to_class(self, rgb_seg: np.ndarray) -> np.ndarray:
        h, w = rgb_seg.shape[:2]
        class_seg = np.zeros((h, w), dtype=np.int64)
        for rgb_val, class_idx in self.RGB_TO_CLASS.items():
            # Check for close match (sometimes compression adds noise) or exact match
            mask = (rgb_seg[:, :, 0] == rgb_val[0]) & \
                   (rgb_seg[:, :, 1] == rgb_val[1]) & \
                   (rgb_seg[:, :, 2] == rgb_val[2])
            class_seg[mask] = class_idx
        return class_seg

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        info = self.frames[idx]

        # Load
        img_pil = Image.open(info['image_path']).convert("RGB") # Ensure 3 channels to start
        mask_pil = Image.open(info['mask_path']).convert('L')#.convert("RGB")
        
        # Resize
        if self.input_size:
            # PIL uses (W, H)
            target_wh = (self.input_size[1], self.input_size[0])
            img_pil = img_pil.resize(target_wh, Image.BILINEAR)
            mask_pil = mask_pil.resize(target_wh, Image.NEAREST)

        img = np.array(img_pil)
        #mask_rgb = np.array(mask_pil)
        mask_arr = np.array(mask_pil) 
        mask_arr[mask_arr >= 5] = 0

        img_gray = np.mean(img, axis=2).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_gray).unsqueeze(0).float() # (1, H, W)

        # Convert Mask RGB to Indices (H, W)
        #mask_indices = self._rgb_to_class(mask_rgb)
        #mask_tensor = torch.from_numpy(mask_indices).long()

        mask_tensor = torch.from_numpy(mask_arr.astype(np.int64)).long()

        return img_tensor, mask_tensor

# ==========================================
# 2. U-NET MODEL
# ==========================================
class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(channels, channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels // 2, channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(channels*2, channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv2D(in_channels, channels)
        self.down1 = Down(channels)
        self.down2 = Down(channels)
        self.down3 = Down(channels)
        self.down4 = Down(channels)
        self.up1 = Up(channels)
        self.up2 = Up(channels)
        self.up3 = Up(channels)
        self.up4 = Up(channels)
        self.outc = nn.Conv2d(channels, n_classes, kernel_size=1) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        logits = self.outc(feature)
        return logits

CONFIG = {
    'project_name': 'iOCT_UNet_new',
    'run_name': 'train_peeling1_test_2_3',
    'data_path': '/vol/data/BioProject13/data_OCT',  
    'model_save_path': '/vol/data/OctreeNCA_Video/Models/iOCT_UNet_newdata',
    
    'train_video': ['peeling1'],
    'test_videos': ['peeling2', 'peeling3'],
    'val_split_ratio': 0.15,

    'seed': 42,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'epochs': 10,
    'save_interval': 2,
    'input_size': (512, 512),
    
    'input_channels': 1,  # Grayscale
    'base_channels': 32,
    'n_classes': 5       
}

# mixed split of data for the first iOCT dataset (peeling/Bscans-dt and sri/Bscans-dt)
def get_mixed_splits(config):
    """
    Mixes ALL data and splits randomly (80% Train, 10% Val, 10% Test).
    Matches colleague's strategy.
    """
    # 1. Load Everything
    full_ds = IOCTDataset(config['data_path'], config['input_size'])
    total_size = len(full_ds)
    
    if total_size == 0:
        raise ValueError("No data found!")
        
    # 2. Calculate Split Sizes
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    print("-" * 30)
    print("MIXED RANDOM SPLIT (Colleague's Strategy)")
    print(f"Total: {total_size}")
    print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print("-" * 30)
    
    # 3. Random Split
    generator = torch.Generator().manual_seed(config['seed'])
    train_ds, val_ds, test_ds = random_split(
        full_ds, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    return train_ds, val_ds, test_ds
# cross-validation split by video for the new iOCT dataset (peeling1, peeling2, peeling3)
def get_crossval_splits(config):
    print("\n" + "="*40)
    print(f"CROSS VALIDATION SETUP")
    print(f"Training on: {config['train_video']}")
    print(f"Testing on:  {config['test_videos']}")
    print("="*40)
    
    # 1. TEST SET (Strictly unseen videos)
    test_ds = IOCTDataset(config['data_path'], config['input_size'], subset_videos=config['test_videos'])
    if len(test_ds) == 0:
        raise ValueError("Test dataset is empty! Check folder names and paths.")

    # 2. TRAIN & VAL SETS (From the single training video)
    full_train_video_ds = IOCTDataset(config['data_path'], config['input_size'], subset_videos=config['train_video'])
    if len(full_train_video_ds) == 0:
        raise ValueError("Train dataset is empty! Check folder names and paths.")
        
    # Split the single video into Train and Val
    total_train_frames = len(full_train_video_ds)
    val_size = int(total_train_frames * config['val_split_ratio'])
    train_size = total_train_frames - val_size
    
    generator = torch.Generator().manual_seed(config['seed'])
    train_ds, val_ds = random_split(
        full_train_video_ds, 
        [train_size, val_size],
        generator=generator
    )
    
    print("\nDataset Sizes:")
    print(f"  Train: {len(train_ds)} frames (from {config['train_video']})")
    print(f"  Val:   {len(val_ds)} frames (from {config['train_video']} - used for Model Selection)")
    print(f"  Test:  {len(test_ds)} frames (from {config['test_videos']} - STRICTLY UNSEEN)")
    
    return train_ds, val_ds, test_ds

def calculate_dice(inputs, targets, num_classes):
    # inputs: (B, C, H, W) logits
    # targets: (B, H, W) indices
    inputs = F.softmax(inputs, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
    union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    
    dice = (2. * intersection) / (union + 1e-6)
    return dice.mean().item()

def soft_dice_loss(logits, targets, num_classes, smooth=1e-6):
    """
    Multi-class soft Dice loss computed from probabilities (softmax).
    Returns:
      dice_loss_sum: sum over classes of (1 - dice_c)
      dice_loss_mean: mean over classes of (1 - dice_c)
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)  # sum over batch and spatial dims
    intersection = (probs * targets_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    dice_loss_per_class = 1.0 - dice

    dice_loss_sum = dice_loss_per_class.sum()
    dice_loss_mean = dice_loss_per_class.mean()
    return dice_loss_sum, dice_loss_mean

def run_validation(model, val_loader, criterion, device, num_classes, max_batches=None):
    model.eval()
    val_dice_sum = 0.0
    val_ce_loss_sum = 0.0
    val_dice_loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            val_ce_loss_sum += criterion(outputs, masks).item()
            dice_loss_sum, _dice_loss_mean = soft_dice_loss(outputs, masks, num_classes=num_classes)
            val_dice_loss_sum += dice_loss_sum.item()
            val_dice_sum += calculate_dice(outputs, masks, num_classes)
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    if n_batches == 0:
        return {
            "val_ce_loss": float("nan"),
            "val_dice_loss_sum": float("nan"),
            "val_dice_loss_mean": float("nan"),
            "val_dice": float("nan"),
        }
    return {
        "val_ce_loss": val_ce_loss_sum / n_batches,
        "val_dice_loss_sum": val_dice_loss_sum / n_batches,
        "val_dice_loss_mean": (val_dice_loss_sum / n_batches) / num_classes,
        "val_dice": val_dice_sum / n_batches,
    }


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG['model_save_path'], exist_ok=True)


    wandb.init(project=CONFIG['project_name'], config=CONFIG, name=CONFIG['run_name'])
    
    # 1. Data
    
    test_ds = IOCTDataset(CONFIG['data_path'], CONFIG['input_size'], subset_videos=CONFIG['test_videos'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # 2. Model
    model = UNet(CONFIG['input_channels'], CONFIG['base_channels'], CONFIG['n_classes']).to(device)
    
    global_step = 0
    
    
    # Final Test
    print("\n" + "="*30)
    print("FINAL EVALUATION ON TEST SET")
    print("="*30)
    
    model.load_state_dict(torch.load(os.path.join(CONFIG['model_save_path'], "best_model.pth")))
    model.eval()
    
    total_dice = 0
    class_dice_sums = np.zeros(CONFIG['n_classes'])
    test_dice_loss_sum = 0.0
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            dice_loss_sum, _dice_loss_mean = soft_dice_loss(outputs, masks, num_classes=CONFIG["n_classes"])
            test_dice_loss_sum += dice_loss_sum.item()
            
            # Dice calculation per class
            inputs_soft = F.softmax(outputs, dim=1)
            targets_one_hot = F.one_hot(masks, num_classes=CONFIG['n_classes']).permute(0, 3, 1, 2).float()
            
            inter = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
            union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
            dice = (2. * inter) / (union + 1e-6)
            
            total_dice += dice.mean().item()
            class_dice_sums += dice.mean(dim=0).cpu().numpy()

    print(f"Overall Test Dice: {total_dice / len(test_loader):.4f}")
    print("Per-Class Scores:")
    #class_names = ["Background", "Red", "Cyan", "Green", "Blue", "Yellow"]
    class_names = ["Background", "Class1", "Class2", "Class3", "Class4"]
    avg_class_dice = class_dice_sums / len(test_loader)
    
    for i, name in enumerate(class_names):
        print(f"  {name}: {avg_class_dice[i]:.4f}")

    wandb.log(
        {
            "test_dice": total_dice / len(test_loader),
            "test_dice_loss_sum": test_dice_loss_sum / len(test_loader),
            "test_dice_loss_mean": (test_dice_loss_sum / len(test_loader)) / CONFIG["n_classes"],
        },
        step=global_step,
    )
    wandb.finish()

if __name__ == "__main__":
    run()
