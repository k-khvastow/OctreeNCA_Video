"""
Visualize OctreeNCA inference stages as a video.
Shows all 5 stages side-by-side, with each frame showing one update step.
Usage: python visualize_octree_stages.py --model_path <path> --output video.mp4
"""

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from train_ioct2d import get_study_config, get_dataset_args, iOCTDatasetForExperiment


def load_model(model_path, study_config):
    """Load trained OctreeNCA model."""
    from src.models.Model_OctreeNCA import OctreeNCA
    
    model = OctreeNCA(
        input_channels=study_config['model.input_channels'],
        output_channels=study_config['model.output_channels'],
        channel_n=study_config['model.channel_n'],
        hidden_size=study_config['model.hidden_size'],
        kernel_size=study_config['model.kernel_size'],
        res_and_steps=study_config['model.octree.res_and_steps'],
        backbone_class_name=study_config['model.backbone_class'],
        separate_models=study_config['model.octree.separate_models'],
    )
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_to_rgb(tensor, num_classes):
    """Convert prediction tensor (C, H, W) to RGB (H, W, 3)."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    class_pred = torch.argmax(tensor, dim=0).cpu().numpy()
    h, w = class_pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    colors = [
        [0, 0, 0],       # Background
        [255, 0, 0],     # Red
        [0, 255, 209],   # Cyan
        [61, 255, 0],    # Green
        [0, 78, 255],    # Blue
        [255, 189, 0],   # Yellow
        [218, 0, 255],   # (magenta)
    ]
    
    for c in range(min(num_classes, len(colors))):
        rgb[class_pred == c] = colors[c]
    
    return rgb


def create_visualization_video(
    model_path,
    output_path='octree_stages.mp4',
    sample_idx=100,
    fps=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print("Loading config and dataset...")
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)
    
    # Load full dataset
    dataset = iOCTDatasetForExperiment(**dataset_args)
    all_ids = list(dataset.frames_dict.keys())
    
    # Split train/val/test
    n = len(all_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    test_ids = all_ids[n_train + n_val:]
    
    dataset.setPaths('', test_ids, '', [])
    print(f"Test set: {len(dataset)} samples")
    
    # Get test sample
    sample = dataset[min(sample_idx, len(dataset) - 1)]
    image = torch.from_numpy(sample['image']).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 1, H, W)
    
    print(f"Sample: {sample['id']}")
    print(f"Input shape: {image.shape}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, study_config).to(device)
    
    # Setup video
    res_and_steps = study_config['model.octree.res_and_steps']
    num_classes = study_config['model.output_channels']
    num_stages = len(res_and_steps)
    
    model_name = "OctreeNCA"
    
    # Video dimensions
    tile_h = 256
    frame_w = tile_h * num_stages
    frame_h = tile_h + 60
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    
    print(f"\nCreating video: {frame_w}x{frame_h} @ {fps}fps")
    print(f"Stages: {[f'{r[0]}x{r[1]}' for r, _ in res_and_steps]}")
    
    with torch.no_grad():
        # Store outputs at each stage for visualization
        stage_outputs = [None] * num_stages
        
        # Process from smallest (stage 4) to largest (stage 0)
        for stage_idx in range(num_stages - 1, -1, -1):
            res, steps = res_and_steps[stage_idx]
            h, w = res
            
            print(f"\nStage {stage_idx}: {h}x{w}, {steps} steps")
            
            # Downsample input
            input_stage = torch.nn.functional.interpolate(
                image, size=(h, w), mode='bilinear', align_corners=False
            )
            
            # Initialize hidden state
            if stage_idx == num_stages - 1:
                hidden = torch.zeros(1, study_config['model.channel_n'], h, w, device=device)
            else:
                hidden = torch.nn.functional.interpolate(
                    hidden, size=(h, w), mode='bilinear', align_corners=False
                )
            
            # Run updates
            backbone = model.backbone_ncas[stage_idx]
            
            for step in range(steps):
                x = torch.cat([input_stage, hidden], dim=1)
                delta = backbone(x)
                hidden = hidden + delta
                logits = hidden[:, :num_classes, :, :]
                
                # Store current output
                stage_outputs[stage_idx] = logits.clone()
                
                # Create frame
                canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                
                # Add text
                step_text = f"Step {step + 1}/{steps}"
                cv2.putText(canvas, step_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(canvas, model_name, (frame_w - 180, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw all 5 stages
                for s_idx in range(num_stages):
                    x_offset = s_idx * tile_h
                    
                    # Stage label
                    stage_res = res_and_steps[s_idx][0]
                    label = f"{stage_res[0]}x{stage_res[1]}"
                    cv2.putText(canvas, label, (x_offset + 10, frame_h - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    if s_idx > stage_idx:
                        # Future stage - show gray
                        gray_tile = np.full((tile_h, tile_h, 3), 40, dtype=np.uint8)
                        canvas[60:60+tile_h, x_offset:x_offset+tile_h] = gray_tile
                    elif s_idx == stage_idx or stage_outputs[s_idx] is not None:
                        # Current or completed stage
                        out = stage_outputs[s_idx] if stage_outputs[s_idx] is not None else logits
                        
                        # Upsample to tile size
                        out_resized = torch.nn.functional.interpolate(
                            out, size=(tile_h, tile_h), mode='nearest'
                        )
                        
                        rgb = tensor_to_rgb(out_resized[0], num_classes)
                        canvas[60:60+tile_h, x_offset:x_offset+tile_h] = rgb
                
                video.write(canvas)
    
    video.release()
    print(f"\n✓ Video saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model.pth')
    parser.add_argument('--output', type=str, default='octree_stages.mp4', help='Output video path')
    parser.add_argument('--sample_idx', type=int, default=100, help='Test sample index')
    parser.add_argument('--fps', type=int, default=10, help='Video FPS')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    create_visualization_video(
        model_path=args.model_path,
        output_path=args.output,
        sample_idx=args.sample_idx,
        fps=args.fps,
        device=args.device
    )
