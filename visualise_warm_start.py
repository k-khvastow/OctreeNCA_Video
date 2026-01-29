import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
import glob
import cv2

# Import configurations and classes from your provided files
from train_video2d_warm import get_study_config, get_dataset_args
from src.datasets.Dataset_Video2D_Sequential import Video2DSequentialDataset
from src.utils.WarmStartConfig import EXP_OctreeNCA_WarmStart
from src.utils.ProjectConfiguration import ProjectConfiguration as pc

def visualize_warm_start(num_samples=2, experiment_name_override=None, sequence_length_override=None, step_override=None):
    """
    Visualizes the output of the Warm Start OctreeNCA model on video sequences.
    
    Args:
        num_samples: Number of sequences to visualize.
        experiment_name_override: Exact name of the experiment folder (e.g., 'WarmStart_apple_24').
                                  If None, attempts to find the most recently modified WarmStart experiment.
        sequence_length_override: Optional int to override the dataset sequence length (e.g., 10).
        step_override: Optional int to override the dataset step size (e.g., 2).
    """
    
    # 1. Setup Configuration
    # We load the base config from the training script
    study_config = get_study_config()
    # study_config['experiment.dataset.preload'] = False # Not used by SequentialDataset
    
    # Handle Experiment Name matching
    if experiment_name_override:
         study_config['experiment.name'] = experiment_name_override
    else:
        # Auto-detect the latest experiment folder starting with "WarmStart_"
        exp_path = os.path.join(pc.FILER_BASE_PATH, 'Experiments')
        if os.path.exists(exp_path):
            candidates = sorted(glob.glob(os.path.join(exp_path, "WarmStart_*")), key=os.path.getmtime)
            if candidates:
                latest_exp = os.path.basename(candidates[-1])
                print(f"Automatically selected latest experiment: {latest_exp}")
                study_config['experiment.name'] = latest_exp
            else:
                print("No 'WarmStart_*' experiment folders found in Experiments directory.")
                return
        else:
            print(f"Experiments directory not found at {exp_path}")
            return

    # Get Dataset Arguments (for experiment reload)
    dataset_args = get_dataset_args(study_config)
    # FIX: Video2DSequentialDataset does not accept 'preload', so we do not add it here.
    
    # 2. Initialize Experiment & Model
    print(f"Initializing Experiment: {study_config['experiment.name']}")
    # Use the specific WarmStart experiment wrapper
    try:
        exp = EXP_OctreeNCA_WarmStart().createExperiment(
            study_config, 
            detail_config={}, 
            dataset_class=Video2DSequentialDataset, 
            dataset_args=dataset_args
        )
    except TypeError as e:
        print(f"Error initializing experiment: {e}")
        print("Please ensure Video2DSequentialDataset matches the arguments provided.")
        return

    # 3. Load Latest Checkpoint
    # Construct path: <Base>/Experiments/<Name>/models/
    model_dir = os.path.join(pc.FILER_BASE_PATH, exp.config['experiment.model_path'], 'models')
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return

    files = os.listdir(model_dir)
    epochs = [f for f in files if f.startswith("epoch_")]
    if not epochs:
        print("No checkpoints found.")
        return
    
    # Sort to find the latest epoch
    epochs.sort(key=lambda x: int(x.split('_')[1]))
    latest_epoch = epochs[-1]
    checkpoint_path = os.path.join(model_dir, latest_epoch)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    exp.agent.load_state(checkpoint_path, pretrained=True)
    exp.agent.model.eval()
    device = exp.agent.device

    # 4. Load Dataset
    print("Loading Dataset (this may take a moment)...")
    # For visualization, allow a different sequence length without breaking reload.
    vis_dataset_args = dict(dataset_args)
    if sequence_length_override is not None:
        try:
            sequence_length_override = int(sequence_length_override)
        except (TypeError, ValueError):
            raise ValueError("sequence_length_override must be an integer.")
        if sequence_length_override <= 0:
            raise ValueError("sequence_length_override must be > 0.")
        vis_dataset_args['sequence_length'] = sequence_length_override
        print(f"Overriding sequence length to {sequence_length_override}")

    if step_override is not None:
        try:
            step_override = int(step_override)
        except (TypeError, ValueError):
            raise ValueError("step_override must be an integer.")
        if step_override <= 0:
            raise ValueError("step_override must be > 0.")
        vis_dataset_args['step'] = step_override
        print(f"Overriding step size to {step_override}")

    dataset = Video2DSequentialDataset(**vis_dataset_args)
    print(f"Dataset loaded. Total sequences: {len(dataset)}")

    # 5. Inference & Visualization
    # Pick random sequences
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    indices = range(len(dataset))
    sample_indices = random.sample(indices, min(num_samples, len(dataset)))
    
    sequence_length = dataset.sequence_length # Default is 5 based on training config
    
    # Setup Plot: 
    # Rows: 2 per sample (GT Sequence, Prediction Sequence)
    # Cols: Time steps (sequence_length)
    fig, axes = plt.subplots(len(sample_indices) * 2, sequence_length, 
                             figsize=(3 * sequence_length, 3 * len(sample_indices) * 2))
    
    # Ensure axes is always 2D array [row, col]
    if len(axes.shape) == 1:
        axes = axes.reshape(-1, sequence_length)
    elif len(axes.shape) == 2 and len(sample_indices) == 1:
         # If only 1 sample (2 rows), axes is (2, T). Correct.
         pass

    print("Generating visualizations...")
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            print(f"Processing sample {idx}...")
            
            # Load sequence data
            # Dataset returns: {'image': (T, 1, H, W), 'label': (T, C, H, W), 'id': str}
            data = dataset.__getitem__(idx)
            
            images_np = data['image'] # Sequence of inputs
            labels_np = data['label'] # Sequence of GT labels
            seq_id = data['id']
            
            images_tensor = torch.from_numpy(images_np).to(device)
            
            # Initialize state as None for the first frame (Cold Start)
            prev_state = None
            
            row_gt = i * 2
            row_pred = i * 2 + 1
            
            # Loop through the sequence
            for t in range(sequence_length):
                # -- 1. Prepare Input --
                # Extract single frame: (1, 1, H, W)
                img_t = images_tensor[t].unsqueeze(0) 
                
                # -- 2. Forward Pass (Warm Start) --
                # Use the model's forward to ensure BCHW -> BHWC permutation is applied.
                out = exp.agent.model(img_t, prev_state=prev_state)
                
                # -- 3. Update State for next frame --
                # The model returns 'final_state' which we feed back into the next step
                prev_state = out['final_state']
                
                # -- 4. Process Output --
                logits = out['logits'] # (1, H, W, C) - Model outputs BHWC directly here
                
                # Squeeze batch dim -> (H, W, C)
                pred_np = logits.squeeze(0).cpu().numpy()
                
                # Create Prediction Mask (Argmax)
                pred_mask = np.argmax(pred_np, axis=-1)
                
                # Process Ground Truth Mask
                # label is (C, H, W) -> argmax over dim 0 -> (H, W)
                lbl_t = labels_np[t] 
                gt_mask = np.argmax(lbl_t, axis=0) 
                
                # Get Image for Visualization (H, W)
                img_vis = images_np[t, 0]
                
                # -- 5. Plotting --
                
                # Row 1: Ground Truth
                ax_gt = axes[row_gt, t]
                ax_gt.imshow(img_vis, cmap='gray')
                ax_gt.imshow(gt_mask, cmap='jet', alpha=0.5, interpolation='nearest')
                
                if t == 0:
                    ax_gt.set_ylabel(f"Seq {seq_id}\nGround Truth", fontsize=10, fontweight='bold')
                ax_gt.set_title(f"t={t}", fontsize=10)
                ax_gt.axis('off')
                
                # Row 2: Prediction
                ax_pred = axes[row_pred, t]
                ax_pred.imshow(img_vis, cmap='gray')
                ax_pred.imshow(pred_mask, cmap='jet', alpha=0.5, interpolation='nearest')
                
                if t == 0:
                    ax_pred.set_ylabel(f"Prediction", fontsize=10, fontweight='bold')
                ax_pred.axis('off')

    plt.tight_layout()
    
    # Save output
    # Use the experiment name in the filename
    safe_exp_name = study_config['experiment.name'].replace('/', '_')
    out_path = f"visualization_{safe_exp_name}.png"
    plt.savefig(out_path)
    print(f"Visualization saved to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    # Call the function. 
    # Pass 'experiment_name_override' if you want to visualize a specific past run.
    # Otherwise, it picks the latest one.
    visualize_warm_start(
        num_samples=2,
        experiment_name_override="WarmStart_conspiracy_24",
        sequence_length_override=10,
        step_override=10,
    )
