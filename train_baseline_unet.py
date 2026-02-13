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
from PIL import Image
from pathlib import Path


class IOCTDataset(Dataset):
    """
    Standalone version of iOCTDatasetForExperiment.
    Adapted to handle both 'Bscans-dt' structure and direct 'image' structure.
    """
    
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

    def __init__(self, data_root, input_size=(512, 512)):
        self.data_root = Path(data_root)
        self.input_size = input_size
        self.frames = []
        self._collect_all_frames()
        

    def _collect_all_frames(self):
        # Backwards-compatible name (some versions used `_collect_frames`).
        self._collect_frames()

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
        mask_pil = Image.open(info['mask_path']).convert("RGB")
        
        # Resize
        if self.input_size:
            # PIL uses (W, H)
            target_wh = (self.input_size[1], self.input_size[0])
            img_pil = img_pil.resize(target_wh, Image.BILINEAR)
            mask_pil = mask_pil.resize(target_wh, Image.NEAREST)

        img = np.array(img_pil)
        mask_rgb = np.array(mask_pil)

        img_gray = np.mean(img, axis=2).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_gray).unsqueeze(0).float() # (1, H, W)

        # Convert Mask RGB to Indices (H, W)
        mask_indices = self._rgb_to_class(mask_rgb)
        mask_tensor = torch.from_numpy(mask_indices).long()

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
    'project_name': 'iOCT_UNet',
    'run_name': 'Run_Mixed_Split_10Ep',
    'data_path': 'ioct_data',  
    'model_save_path': 'Models/iOCT_UNet',
    
    'seed': 42,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'epochs': 10,
    'save_interval': 2,
    'input_size': (512, 512),
    
    'input_channels': 1,  # Grayscale
    'base_channels': 32,
    'n_classes': 7        
}


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
    try:
        import wandb  # type: ignore
    except ImportError as e:
        raise ImportError(
            "wandb is required to run training. Install it (e.g. `pip install wandb`) "
            "or modify train_baseline_unet.py to disable wandb logging."
        ) from e

    wandb.init(project=CONFIG['project_name'], config=CONFIG, name=CONFIG['run_name'])
    
    # 1. Data
    train_ds, val_ds, test_ds = get_mixed_splits(CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # 2. Model
    model = UNet(CONFIG['input_channels'], CONFIG['base_channels'], CONFIG['n_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    dice_loss_weight = CONFIG.get("dice_loss_weight", 1.0)
    
    best_val_dice = 0.0
    global_step = 0
    val_log_interval_steps = CONFIG.get("val_log_interval_steps", 100)
    val_max_batches = CONFIG.get("val_max_batches", None)
    
    print("\nStarting Training...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for imgs, masks in pbar:
            global_step += 1
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            ce_loss = criterion(outputs, masks)
            dice_loss_sum, dice_loss_mean = soft_dice_loss(outputs, masks, num_classes=CONFIG["n_classes"])
            loss = ce_loss + dice_loss_weight * dice_loss_mean
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_batch_loss": loss.item(),
                    "train_batch_ce_loss": ce_loss.item(),
                    "train_batch_dice_loss_sum": dice_loss_sum.item(),
                    "train_batch_dice_loss_mean": dice_loss_mean.item(),
                },
                step=global_step,
            )
            pbar.set_postfix({"loss": loss.item(), "ce": ce_loss.item(), "dice": dice_loss_mean.item()})

            if val_log_interval_steps and (global_step % val_log_interval_steps == 0):
                val_metrics_step = run_validation(
                    model,
                    val_loader,
                    criterion,
                    device,
                    CONFIG["n_classes"],
                    max_batches=val_max_batches,
                )
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "val_ce_loss_step": val_metrics_step["val_ce_loss"],
                        "val_dice_loss_sum_step": val_metrics_step["val_dice_loss_sum"],
                        "val_dice_loss_mean_step": val_metrics_step["val_dice_loss_mean"],
                    },
                    step=global_step,
                )
                model.train()
            
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_metrics = run_validation(model, val_loader, criterion, device, CONFIG["n_classes"], max_batches=None)
        
        print(
            f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val CE: {val_metrics['val_ce_loss']:.4f} | "
            f"Val Dice Loss(sum): {val_metrics['val_dice_loss_sum']:.4f} | Val Dice: {val_metrics['val_dice']:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_ce_loss": val_metrics["val_ce_loss"],
                "val_dice_loss_sum": val_metrics["val_dice_loss_sum"],
                "val_dice_loss_mean": val_metrics["val_dice_loss_mean"],
                "val_dice": val_metrics["val_dice"],
            },
            step=global_step,
        )
        
        if val_metrics["val_dice"] > best_val_dice:
            best_val_dice = val_metrics["val_dice"]
            torch.save(model.state_dict(), os.path.join(CONFIG['model_save_path'], "best_model.pth"))
            print(f"--> New Best Model ({best_val_dice:.4f})")

        if (epoch + 1) % CONFIG['save_interval'] == 0:
            save_name = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), os.path.join(CONFIG['model_save_path'], save_name))
            print(f"--> Saved Checkpoint: {save_name}")

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
    class_names = ["Background", "Red", "Cyan", "Green", "Blue", "Yellow"]
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
