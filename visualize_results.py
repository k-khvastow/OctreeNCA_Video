
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import random

from train_video2d import get_study_config, get_dataset_args
from src.datasets.Dataset_Video2D import Video2DDataset
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.ProjectConfiguration import ProjectConfiguration as pc

def visualize(num_samples=3):
    # 1. Setup
    study_config = get_study_config()
    study_config['experiment.dataset.preload'] = False 
    random_word = "airline"  # Fixed word for visualization naming
    
    # Get dataset args
    dataset_args = get_dataset_args(study_config)
    dataset_args['preload'] = False # Disable preload for visualization

    study_config['experiment.name'] = f"Video2D_{random_word}_{study_config['model.channel_n']}"
    # Create Experiment
    print("Initialize Experiment...")
    exp = EXP_OctreeNCA().createExperiment(study_config, detail_config={}, dataset_class=Video2DDataset, dataset_args=dataset_args)
    
    # 2. Load Model
    # Determine the path. 
    # Usually models are saved in <STUDY_PATH>/Experiments/<ExperimentName>/models/
    model_dir = os.path.join(pc.FILER_BASE_PATH, exp.config['experiment.model_path'], 'models')
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return

    # Find latest epoch
    files = os.listdir(model_dir)
    epochs = [f for f in files if f.startswith("epoch_")]
    if not epochs:
        print("No checkpoints found.")
        return
    
    # Sort by epoch number
    epochs.sort(key=lambda x: int(x.split('_')[1]))
    latest_epoch = epochs[-1]
    checkpoint_path = os.path.join(model_dir, latest_epoch)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    exp.agent.load_state(checkpoint_path, pretrained=True)
    exp.agent.model.eval()

    # 3. Get Data
    # Dataset_Video2D handles train/test splitting if configured, or we can just access the dataset directly.
    # The agent/exp usually sets up dataloaders.
    # Let's verify how exp sets up data. 
    # EXP_OctreeNCA -> ExperimentWrapper -> setup -> creates dataloaders
    # Let's manually create a split or just pick from the full dataset if split unavailable.
    
    # Try to reuse experiment's data setup if possible, or just create dataset instance
    dataset = Video2DDataset(**dataset_args)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 4. Inference & Visualization
    indices = range(len(dataset))
    # Pick random indices
    sample_indices = random.sample(indices, num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = [axes] # ensure list of lists if needed? No, (2,) array usually. 
        # Actually (samples, 2) makes 2D array.
        if num_samples == 1: axes = np.expand_dims(axes, axis=0)

    print("Generating visualizations...")
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            data = dataset.load_sample(idx)
            # data is dict: {'id', 'image' (HWC), 'label' (HWC), ...}
            
            img_np = data['image'] # (H, W, 1) usually or (H,W,C)
            label_np = data['label'] # (H, W, 7 or 8) one-hot
            
            # Prepare for model
            # Agent expects dict with BCHW tensors?
            # Agent.prepare_data usually calls my_default_collate if using loader.
            # Here we manual.
            
            input_tensor = torch.from_numpy(img_np).unsqueeze(0) # (1, H, W, C)
            input_tensor = input_tensor.permute(0, 3, 1, 2).float() # (1, C, H, W)
            input_tensor = input_tensor.to(exp.agent.device)
            
            # Predict
            # Agent.get_outputs expects input dict?
            # Agent.get_outputs(data_dict) -> self.model(inputs, ...)
            # Let's define dummy dict
            input_dict = {
                'image': input_tensor,
                'label': torch.from_numpy(label_np).unsqueeze(0).to(exp.agent.device) # Not used for prediction but model might need shape?
            }
            
            # Actually OctreeNCA forward(x, y) uses y?
            # forward(x, y, batch_duplication)
            # Yes it typically passes targets. But for inference (eval), y is optional?
            # Model_OctreeNCAV2.forward(self, x, y=None, ...)
            # So y is optional.
            
            # However, MedNCAAgent.get_outputs gets inputs, targets = data['image'], data['label']
            # So we better provide label or modify it.
            # Let's provide dummy label matches output shape
            
            output = exp.agent.model(input_tensor, batch_duplication=1) 
            
            # Handle dictionary output
            if isinstance(output, dict):
                if 'logits' in output:
                   pred = output['logits']
                elif 'output' in output:
                   pred = output['output']
                else:
                   # Fallback: take the first value
                   pred = list(output.values())[0]
            else:
                pred = output 
            if pred.shape[1] == study_config['model.output_channels']: 
                 # BCHW -> BHWC
                 pred = pred.permute(0, 2, 3, 1)
            
            pred_np = pred.squeeze(0).cpu().numpy() # (H, W, C)
            
            # Convert Prediction to Mask (argmax)
            pred_mask = np.argmax(pred_np, axis=-1)
            
            # Convert GT to Mask
            gt_mask = np.argmax(label_np, axis=-1)
            
            # Convert Image to visualize
            img_vis = img_np
            # If 1 channel, squeeze
            if img_vis.shape[-1] == 1:
                img_vis = img_vis.squeeze(-1)
            
            # Plot Left: Image + GT
            ax_left = axes[i][0]
            ax_left.imshow(img_vis, cmap='gray')
            ax_left.imshow(gt_mask, cmap='jet', alpha=0.5, interpolation='nearest') # Overlay
            ax_left.set_title(f"Sample {idx}: GT")
            ax_left.axis('off')
            
            # Plot Right: Image + Pred
            ax_right = axes[i][1]
            ax_right.imshow(img_vis, cmap='gray')
            ax_right.imshow(pred_mask, cmap='jet', alpha=0.5, interpolation='nearest') # Overlay
            ax_right.set_title(f"Sample {idx}: Prediction")
            ax_right.axis('off')

    out_path = "visualization_results.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Visualization saved to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    visualize()
