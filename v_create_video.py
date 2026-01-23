import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Import your existing modules
from train_video2d import get_study_config, get_dataset_args
from src.datasets.Dataset_Video2D import Video2DDataset
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.ProjectConfiguration import ProjectConfiguration as pc

def create_video_segmentation():
    # --- CONFIGURATION ---
    # Update this to match the experiment name you actually trained
    random_word = "subsidence" 
    
    # Specify the folder/patient ID you want to render. 
    # Set to None to automatically pick the first available folder.
    target_folder_name = "10001" # Set to "10001" based on your log
    
    output_video_name = f"segmentation_video_{random_word}.mp4"
    fps = 20  # Frames per second
    # ---------------------

    # 1. Setup
    study_config = get_study_config()
    study_config['experiment.dataset.preload'] = False 
    
    dataset_args = get_dataset_args(study_config)
    dataset_args['preload'] = False 

    # Construct experiment name
    study_config['experiment.name'] = f"Video2D_{random_word}_{study_config['model.channel_n']}"
    
    print("Initialize Experiment...")
    exp = EXP_OctreeNCA().createExperiment(study_config, detail_config={}, dataset_class=Video2DDataset, dataset_args=dataset_args)
    
    # 2. Load Model
    model_dir = os.path.join(pc.FILER_BASE_PATH, exp.config['experiment.model_path'], 'models')
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Please check if 'random_word' matches your trained experiment.")
        return

    epochs = [f for f in os.listdir(model_dir) if f.startswith("epoch_")]
    if not epochs:
        print("No checkpoints found.")
        return
    
    epochs.sort(key=lambda x: int(x.split('_')[1]))
    latest_epoch = epochs[-1]
    checkpoint_path = os.path.join(model_dir, latest_epoch)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    exp.agent.load_state(checkpoint_path, pretrained=True)
    exp.agent.model.eval()

    # 3. Get Data and Select Sequence
    dataset = Video2DDataset(**dataset_args)
    print(f"Total dataset size: {len(dataset)}")
    
    # Group indices by folder
    folder_to_indices = {}
    for i, sample_meta in enumerate(dataset.samples):
        folder = sample_meta['folder']
        if folder not in folder_to_indices:
            folder_to_indices[folder] = []
        folder_to_indices[folder].append(i)
    
    # Select the target folder
    if target_folder_name is None:
        target_folder_name = list(folder_to_indices.keys())[0]
    
    if target_folder_name not in folder_to_indices:
        print(f"Error: Folder '{target_folder_name}' not found in dataset.")
        print("Available folders:", list(folder_to_indices.keys())[:5], "...")
        return

    # Get indices for this folder and sort them by frame_index
    sequence_indices = folder_to_indices[target_folder_name]
    sequence_indices.sort(key=lambda i: dataset.samples[i]['frame_index'])
    
    print(f"Selected folder: {target_folder_name} with {len(sequence_indices)} frames.")

    # 4. Initialize Video Writer
    video_writer = None
    out_path = f"visualisationOCT/{random_word}_segmentation_video.mp4"
    print("Generating video frames...")
    
    # Create a persistent figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    canvas = FigureCanvas(fig)

    with torch.no_grad():
        for i, idx in enumerate(sequence_indices):
            # Load Data
            data = dataset.load_sample(idx)
            img_np = data['image'] # (H, W, 1)
            label_np = data['label'] # (H, W, Classes)
            
            # Prepare Input
            input_tensor = torch.from_numpy(img_np).unsqueeze(0).permute(0, 3, 1, 2).float()
            input_tensor = input_tensor.to(exp.agent.device)
            
            # Inference
            output = exp.agent.model(input_tensor, batch_duplication=1) 
            
            # Handle Output
            if isinstance(output, dict):
                pred = output.get('logits', output.get('output', list(output.values())[0]))
            else:
                pred = output 

            if pred.shape[1] == study_config['model.output_channels']: 
                 pred = pred.permute(0, 2, 3, 1) # BCHW -> BHWC
            
            pred_np = pred.squeeze(0).cpu().numpy()
            pred_mask = np.argmax(pred_np, axis=-1)
            gt_mask = np.argmax(label_np, axis=-1)
            
            # Prepare Image for Vis
            img_vis = img_np
            if img_vis.shape[-1] == 1:
                img_vis = img_vis.squeeze(-1)

            # --- PLOTTING ---
            axes[0].clear()
            axes[1].clear()

            # Plot Left: Image + GT
            axes[0].imshow(img_vis, cmap='gray')
            axes[0].imshow(gt_mask, cmap='jet', alpha=0.5, interpolation='nearest')
            axes[0].set_title(f"GT (Frame {data['frame_index']})")
            axes[0].axis('off')
            
            # Plot Right: Image + Pred
            axes[1].imshow(img_vis, cmap='gray')
            axes[1].imshow(pred_mask, cmap='jet', alpha=0.5, interpolation='nearest')
            axes[1].set_title(f"Pred (Frame {data['frame_index']})")
            axes[1].axis('off')
            
            # Render to buffer
            canvas.draw()
            
            # --- FIX APPLIED HERE ---
            # Retrieve the RGBA buffer directly from the canvas
            buf = canvas.buffer_rgba()
            # Convert to a numpy array (Height, Width, 4)
            frame = np.asarray(buf)
            # Convert RGBA (Matplotlib) to BGR (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            height, width, _ = frame_bgr.shape

            # Initialize Writer once we know the frame size
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            video_writer.write(frame_bgr)
            
            if i % 10 == 0:
                print(f"Processed frame {i}/{len(sequence_indices)}")

    # Cleanup
    plt.close(fig)
    
    if video_writer:
        video_writer.release()
        print(f"Video saved successfully to: {os.path.abspath(out_path)}")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    create_video_segmentation()