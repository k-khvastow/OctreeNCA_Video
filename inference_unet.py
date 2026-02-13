import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader

#from src.datasets.Dataset_Video2D import Video2DDataset
from train_baseline_unet import UNet, IOCTDataset

# --- CONFIGURATION ---
CONFIG = {
    # Data Paths (Same as training)
    'data_path': 'ioct_data',      
   # 'label_path': 'ioct_data/peeling/Bscans-dt/A/Segmentation', 
    
    # Model/Inference Settings
    'input_channels': 1,
    'base_channels': 64,
    'n_classes': 9,
    'input_size': (400, 400), # Crop size
    'transform_mode': 'crop',
    'batch_size': 1,          # Inference is usually single-batch
    'model_path': 'Models/iOCT_Unet/best_model.pth' # Path to trained model weights
}

def load_real_sample(device):
    """
    Loads a single batch of REAL data from the dataset.
    """
    print(f"Loading real data from {CONFIG['data_path']}...")
    
    # Initialize dataset
    dataset = IOCTDataset(
        data_root=CONFIG['data_path'],
        #label_root=CONFIG['label_path'],
        preload=False, 
        num_classes=CONFIG['n_classes'],
        transform_mode=CONFIG['transform_mode'], 
        input_size=CONFIG['input_size']
    )
    
    # Create loader
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Fetch ONE batch
    # The dataset returns a dict: {'image': (B, H, W, 1), 'label': ...}
    batch_dict = next(iter(dataloader))
    
    # Extract image
    images = batch_dict['image'] # (B, H, W, 1)
    
    # Permute for PyTorch: (B, 1, H, W)
    inputs = images.permute(0, 3, 1, 2).float().to(device)
    
    print(f"Loaded Real Sample. Shape: {inputs.shape}")
    print(f"Value Range: {inputs.min():.2f} to {inputs.max():.2f}")
    
    return inputs

def benchmark(model, input_tensor, warmup_iters=20, benchmark_runs=100):
    """
    Measures inference speed using CUDA Events for high precision.
    """
    print(f"\n--- Benchmarking Latency ({benchmark_runs} runs) ---")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 1. Warmup
    print(f"Warming up ({warmup_iters} iters)...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # 2. Timing Loop
    timings = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            torch.cuda.synchronize()
            start_event.record()
            
            _ = model(input_tensor)
            
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
            
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    fps = 1000.0 / avg_ms
    
    print("-" * 40)
    print(f"Average Inference Time: {avg_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"Throughput:             {fps:.2f} FPS")
    print("-" * 40)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = UNet(
        in_channels=CONFIG['input_channels'],
        channels=CONFIG['base_channels'],
        n_classes=CONFIG['n_classes']
    ).to(device)
    model.eval()

    # 2. Load Weights (Optional)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
            

    # 3. Load REAL Data
    real_input = load_real_sample(device)

    # 4. Run Benchmark
    benchmark(model, real_input)

if __name__ == "__main__":
    main()