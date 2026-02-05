import argparse
import os

import cv2
import numpy as np
import torch

from train_ioct2d import get_study_config, get_dataset_args, iOCTDatasetForExperiment
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
import colormaps as cmaps


def _extract_logits(output, output_channels):
    if isinstance(output, dict):
        pred = output.get('logits', None)
        if pred is None:
            pred = output.get('output', None)
        if pred is None:
            pred = next(iter(output.values()))
    else:
        pred = output

    if pred.dim() == 4 and pred.shape[1] == output_channels:
        pred = pred.permute(0, 2, 3, 1)
    return pred


def _frame_sort_key(info):
    stem = info["image_path"].stem
    try:
        return int(stem)
    except ValueError:
        return stem


def _normalize_image(img: torch.Tensor) -> torch.Tensor:
    img = img.to(torch.float32)
    min_val = torch.min(img)
    max_val = torch.max(img)
    if (max_val - min_val) > 0:
        img = (img - min_val) / (max_val - min_val)
    return img


def _mask_to_rgb_bg(mask_bhwc: torch.Tensor, background_color=(0.0, 0.0, 0.0)) -> torch.Tensor:
    # mask_bhwc: BHWC, values are logits/probabilities or one-hot
    if isinstance(mask_bhwc, torch.Tensor):
        mask = mask_bhwc
    else:
        mask = torch.from_numpy(mask_bhwc)

    if mask.dtype != torch.bool:
        mask = mask > 0

    b, h, w, c = mask.shape
    bg = torch.tensor(background_color, dtype=torch.float32, device=mask.device)
    rgb = bg.view(1, 1, 1, 3).expand(b, h, w, 3).clone()
    for ci in range(1, c):
        color = torch.tensor(cmaps.bold[ci].colors, dtype=torch.float32, device=mask.device)
        rgb[mask[..., ci]] = color
    return rgb


def _merge_three_panel_black_bg(image_bhwc: torch.Tensor, pred_bhwc: torch.Tensor, gt_bhwc: torch.Tensor) -> np.ndarray:
    device = pred_bhwc.device
    img = image_bhwc.to(device)
    gt_bhwc = gt_bhwc.to(device)
    if img.shape[-1] != 3:
        img = img.repeat(1, 1, 1, 3)
    img = _normalize_image(img)

    pred_rgb = _mask_to_rgb_bg(pred_bhwc, background_color=(0.0, 0.0, 0.0))
    gt_rgb = _mask_to_rgb_bg(gt_bhwc, background_color=(0.0, 0.0, 0.0))

    merged = torch.cat([img, pred_rgb, gt_rgb], dim=2)
    return merged.squeeze(0).cpu().numpy()


def create_video_ioct2d(
    random_word,
    target_patient_id=None,
    fps=20,
    max_frames=None,
    frame_stride=1,
    output_dir="visualisationOCT",
):
    # 1. Setup
    study_config = get_study_config()
    study_config['experiment.dataset.preload'] = False

    dataset_args = get_dataset_args(study_config)

    # Match experiment name used during training
    study_config['experiment.name'] = f"iOCT2D_{random_word}_{study_config['model.channel_n']}"

    print("Initialize Experiment...")
    exp = EXP_OctreeNCA().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTDatasetForExperiment,
        dataset_args=dataset_args,
    )

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
    dataset = iOCTDatasetForExperiment(**dataset_args)
    print(f"Total dataset size: {len(dataset)}")

    patient_to_indices = {}
    for i, info in enumerate(dataset.frames):
        patient_id = info["patient_id"]
        patient_to_indices.setdefault(patient_id, []).append(i)

    if target_patient_id is None:
        target_patient_id = list(patient_to_indices.keys())[0]

    if target_patient_id not in patient_to_indices:
        print(f"Error: Patient '{target_patient_id}' not found in dataset.")
        print("Available patients:", list(patient_to_indices.keys()))
        return

    sequence_indices = patient_to_indices[target_patient_id]
    sequence_indices.sort(key=lambda i: _frame_sort_key(dataset.frames[i]))

    frame_stride = max(1, int(frame_stride))
    if frame_stride > 1:
        sequence_indices = sequence_indices[::frame_stride]

    if max_frames is not None:
        sequence_indices = sequence_indices[:max_frames]

    print(f"Selected patient: {target_patient_id} with {len(sequence_indices)} frames.")

    # 4. Initialize Video Writer
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"ioct2d_{random_word}_{target_patient_id}.mp4")
    video_writer = None

    print("Generating video frames...")
    with torch.no_grad():
        for i, idx in enumerate(sequence_indices):
            data = dataset[idx]
            img_np = data['image']  # (H, W, 1)
            label_np = data['label']  # (H, W, C)

            input_tensor = torch.from_numpy(img_np).unsqueeze(0).permute(0, 3, 1, 2).float()
            input_tensor = input_tensor.to(exp.agent.device)

            output = exp.agent.model(input_tensor, batch_duplication=1)
            pred = _extract_logits(output, study_config['model.output_channels'])

            image_bhwc = torch.from_numpy(img_np).unsqueeze(0)
            gt_bhwc = torch.from_numpy(label_np).unsqueeze(0)

            merged = _merge_three_panel_black_bg(image_bhwc, pred, gt_bhwc)

            frame_rgb = (merged * 255.0).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if video_writer is None:
                height, width, _ = frame_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            video_writer.write(frame_bgr)

            if i % 10 == 0:
                print(f"Processed frame {i}/{len(sequence_indices)}")

    if video_writer:
        video_writer.release()
        print(f"Video saved successfully to: {os.path.abspath(out_path)}")
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create iOCT2D video (image | prediction | ground truth).")
    parser.add_argument("--random_word", type=str, required=True, help="Experiment random word used during training.")
    parser.add_argument("--patient", type=str, default=None, help="Patient id (e.g., peeling_A, sri_B).")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="visualisationOCT")
    args = parser.parse_args()

    create_video_ioct2d(
        random_word=args.random_word,
        target_patient_id=args.patient,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        output_dir=args.output_dir,
    )
