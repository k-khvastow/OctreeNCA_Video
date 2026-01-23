import torch
import os
import numpy as np
import time
import datetime
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity

# --- Imports from your project structure ---
# Assumes this script is in the same directory as train_video2d.py
from train_video2d import get_study_config, get_dataset_args
from src.datasets.Dataset_Video2D import Video2DDataset
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.ProjectConfiguration import ProjectConfiguration as pc

# --- Profiling Settings ---
PROFILE_SETTINGS = {
    "compile_mode": "max-autotune",  # Options: 'none', 'default', 'reduce-overhead', 'max-autotune'
    "batch_size": 1,                    # Override config batch size for profiling
    "warmup_iters": 20,
    "benchmark_runs": 50,
    "profile_kernels": True,            # Whether to generate the chrome trace
    "device": "cuda:0"
}

def setup_experiment():
    """
    Reuses the logic from visualize_results.py to setup the exact experiment config.
    """
    print("--- Setting up Experiment from Config ---")
    
    # 1. Get Configuration from train_video2d.py
    study_config = get_study_config()

    random_word = "subsidence"  # Fixed word for visualization naming
    study_config['experiment.name'] = f"Video2D_{random_word}_{study_config['model.channel_n']}"
    
    # 2. Apply Profiling Overrides
    study_config['experiment.dataset.preload'] = False 
    study_config['trainer.batch_size'] = PROFILE_SETTINGS['batch_size']
    
    # 3. Get Dataset Args
    dataset_args = get_dataset_args(study_config)
    dataset_args['preload'] = False
    
    # 4. Create Experiment Wrapper
    # This initializes the model with the complex octree structure defined in get_study_config
    exp = EXP_OctreeNCA().createExperiment(
        study_config, 
        detail_config={}, 
        dataset_class=Video2DDataset, 
        dataset_args=dataset_args
    )
    
    # 5. Load Model Weights (Optional but recommended for consistency)
    # This logic mirrors visualize_results.py to find the latest checkpoint
    model_dir = os.path.join(pc.FILER_BASE_PATH, exp.config['experiment.model_path'], 'models')
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        epochs = [f for f in files if f.startswith("epoch_")]
        if epochs:
            epochs.sort(key=lambda x: int(x.split('_')[1]))
            latest_epoch = epochs[-1]
            checkpoint_path = os.path.join(model_dir, latest_epoch)
            print(f"Loading weights from: {checkpoint_path}")
            exp.agent.load_state(checkpoint_path, pretrained=True)
        else:
            print("No checkpoints found. Using random weights.")
    else:
        print(f"Model directory {model_dir} not found. Using random weights.")

    return exp

def prepare_dummy_input(exp):
    """
    Generates a dummy tensor matching the shape expected by the agent/model.
    """
    config = exp.config
    
    # Extract dimensions from config
    input_size = config.get('experiment.dataset.input_size', (400, 400))
    if isinstance(input_size, int):
        h, w = input_size, input_size
    else:
        h, w = input_size
        
    c = config['model.input_channels']
    b = PROFILE_SETTINGS['batch_size']
    
    print(f"Preparing Input: Batch={b}, Channels={c}, Height={h}, Width={w}")
    
    # OctreeNCAV2 expects BCHW input
    dummy_input = torch.randn(b, c, h, w, device=exp.agent.device, dtype=torch.float32)
    
    return dummy_input

def benchmark(model, input_tensor):
    """
    Runs the timing benchmark.
    """
    print(f"\n--- Benchmarking Latency ({PROFILE_SETTINGS['benchmark_runs']} runs) ---")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    print(f"Warming up ({PROFILE_SETTINGS['warmup_iters']} iters)...")
    with torch.no_grad():
        for _ in range(PROFILE_SETTINGS['warmup_iters']):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # Timing Loop
    timings = []
    with torch.no_grad():
        for _ in range(PROFILE_SETTINGS['benchmark_runs']):
            torch.cuda.synchronize()
            start_event.record()
            
            _ = model(input_tensor)
            
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
            
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    fps = 1000.0 / avg_ms
    
    print(f"Average Inference Time: {avg_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")

def profile_kernels(model, input_tensor, filename_prefix):
    """
    Generates PyTorch Profiler trace.
    """
    print("\n--- Profiling Kernels ---")
    
    output_dir = Path("inference_reports")
    output_dir.mkdir(exist_ok=True)
    trace_path = output_dir / f"{filename_prefix}_trace.json"
    
    def trace_handler(p):
        print(f"Exporting chrome trace to: {trace_path.resolve()}")
        p.export_chrome_trace(str(trace_path))

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
                prof.step()
    
    # Save text summary
    summary_path = output_dir / f"{filename_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print(f"Kernel summary saved to: {summary_path}")

def main():
    # 1. Setup Experiment
    exp = setup_experiment()
    model = exp.agent.model
    device = exp.agent.device
    
    model.eval()
    model.to(device)
    
    # 2. Prepare Data
    input_tensor = prepare_dummy_input(exp)
    
    # 3. Compile (Optional)
    mode = PROFILE_SETTINGS["compile_mode"]
    if mode != "none":
        print(f"Compiling model with mode='{mode}'...")
        # Note: We compile the model instance directly
        try:
            model = torch.compile(model, mode=mode)
        except Exception as e:
            print(f"Compilation failed: {e}. Proceeding without compilation.")

    # 4. Run Benchmark
    benchmark(model, input_tensor)
    
    # 5. Run Profiler
    if PROFILE_SETTINGS["profile_kernels"]:
        # Generate filename based on config
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        res = exp.config.get('experiment.dataset.input_size', 'unknown')
        prefix = f"Profile_OctreeNCA_{res}_{mode}_{timestamp}"
        profile_kernels(model, input_tensor, prefix)

if __name__ == "__main__":
    # Ensure reproducibility/stability for profiling
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    main()