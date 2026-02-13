import argparse
import os

import cv2
import numpy as np
import torch

from train_ioct2d_dual_view import (
    EXP_OctreeNCA_DualView,
    get_dataset_args as get_dataset_args_dual,
    get_study_config as get_study_config_dual,
    iOCTPairedViewsDatasetForExperiment,
)
from train_ioct2d_dual_view_warm_preprocessed_m1init import (
    EXP_OctreeNCA_DualView_WarmStart_M1Init,
    get_dataset_args as get_dataset_args_warm,
    get_study_config as get_study_config_warm,
    iOCTPairedSequentialDatasetForExperiment,
)
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
import colormaps as cmaps


def _extract_logits(output, output_channels):
    if isinstance(output, dict):
        pred = output.get("probabilities", None)
        if pred is None:
            pred = output.get("logits", None)
        if pred is None:
            pred = output.get("output", None)
        if pred is None:
            pred = next(iter(output.values()))
    else:
        pred = output

    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Unexpected prediction type: {type(pred)}")
    if pred.dim() == 4 and pred.shape[1] == output_channels:
        pred = pred.permute(0, 2, 3, 1)
    return pred


def _frame_sort_key(info):
    stem = info.get("frame", None)
    if stem is None:
        stem = info.get("frame_start", None)
    if stem is None:
        seq_names = info.get("seq_names", None)
        if seq_names is not None and len(seq_names) > 0:
            stem = os.path.splitext(seq_names[0])[0]
        else:
            stem = info.get("id", "")
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
    if isinstance(mask_bhwc, torch.Tensor):
        mask = mask_bhwc
    else:
        mask = torch.from_numpy(mask_bhwc)

    if mask.ndim != 4:
        raise ValueError(f"Expected BHWC tensor, got shape {tuple(mask.shape)}")

    b, h, w, c = mask.shape
    if c > 1:
        class_idx = torch.argmax(mask, dim=-1)
        mask = torch.nn.functional.one_hot(class_idx, num_classes=c).to(torch.bool)
    elif mask.dtype != torch.bool:
        mask = mask > 0.5

    bg = torch.tensor(background_color, dtype=torch.float32, device=mask.device)
    rgb = bg.view(1, 1, 1, 3).expand(b, h, w, 3).clone()
    for ci in range(1, c):
        color = torch.tensor(cmaps.bold[ci].colors, dtype=torch.float32, device=mask.device)
        rgb[mask[..., ci]] = color
    return rgb


def _merge_three_panel_black_bg(image_bhwc: torch.Tensor, pred_bhwc: torch.Tensor, gt_bhwc: torch.Tensor) -> torch.Tensor:
    device = pred_bhwc.device
    img = image_bhwc.to(device)
    gt_bhwc = gt_bhwc.to(device)
    if img.shape[-1] != 3:
        img = img.repeat(1, 1, 1, 3)
    img = _normalize_image(img)

    pred_rgb = _mask_to_rgb_bg(pred_bhwc, background_color=(0.0, 0.0, 0.0))
    gt_rgb = _mask_to_rgb_bg(gt_bhwc, background_color=(0.0, 0.0, 0.0))

    return torch.cat([img, pred_rgb, gt_rgb], dim=2)


def _merge_dual_view_rows(
    image_a_bhwc: torch.Tensor,
    pred_a_bhwc: torch.Tensor,
    gt_a_bhwc: torch.Tensor,
    image_b_bhwc: torch.Tensor,
    pred_b_bhwc: torch.Tensor,
    gt_b_bhwc: torch.Tensor,
) -> np.ndarray:
    row_a = _merge_three_panel_black_bg(image_a_bhwc, pred_a_bhwc, gt_a_bhwc)
    row_b = _merge_three_panel_black_bg(image_b_bhwc, pred_b_bhwc, gt_b_bhwc)
    merged = torch.cat([row_a, row_b], dim=1)
    return merged.squeeze(0).cpu().numpy()


def create_video_ioct2d_dual_view(
    random_word,
    target_dataset=None,
    fps=20,
    max_frames=None,
    frame_stride=1,
    output_dir="visualisationOCT",
    warm_m1init=False,
    use_m1_init=True,
    render_t0_with_m1=True,
):
    if warm_m1init:
        study_config = get_study_config_warm()
        study_config["experiment.name"] = f"WarmStart_M1Init_iOCT2D_dual_{random_word}_{study_config['model.channel_n']}"
        exp_class = EXP_OctreeNCA_DualView_WarmStart_M1Init
        dataset_class = iOCTPairedSequentialDatasetForExperiment
        dataset_args = get_dataset_args_warm(study_config)
        dataset_args["sequence_length"] = 1
        dataset_args["sequence_step"] = 1
        out_prefix = "long_ioct2d_dual_warm_m1init"
    else:
        study_config = get_study_config_dual()
        study_config["experiment.name"] = f"iOCT2D_dual_{random_word}_{study_config['model.channel_n']}"
        exp_class = EXP_OctreeNCA_DualView
        dataset_class = iOCTPairedViewsDatasetForExperiment
        dataset_args = get_dataset_args_dual(study_config)
        out_prefix = "long_ioct2d_dual"

    study_config["experiment.use_wandb"] = False
    study_config["experiment.dataset.preload"] = False

    print("Initialize Experiment...")
    exp = exp_class().createExperiment(
        study_config,
        detail_config={},
        dataset_class=dataset_class,
        dataset_args=dataset_args,
    )

    model_dir = os.path.join(pc.FILER_BASE_PATH, exp.config["experiment.model_path"], "models")

    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Please check if 'random_word' matches your trained dual-view experiment.")
        return

    epochs = [f for f in os.listdir(model_dir) if f.startswith("epoch_")]
    if not epochs:
        print("No checkpoints found.")
        return

    epochs.sort(key=lambda x: int(x.split("_")[1]))
    latest_epoch = epochs[-1]
    checkpoint_path = os.path.join(model_dir, latest_epoch)

    print(f"Loading checkpoint: {checkpoint_path}")
    exp.agent.load_state(checkpoint_path, pretrained=True)
    exp.agent.model.eval()

    dataset = dataset_class(**dataset_args)
    print(f"Total dataset size: {len(dataset)}")

    dataset_to_indices = {}
    if warm_m1init:
        for i, info in enumerate(dataset.sequences):
            dataset_name = info["dataset"]
            dataset_to_indices.setdefault(dataset_name, []).append(i)
    else:
        for i, info in enumerate(dataset.pairs):
            dataset_name = info["dataset"]
            dataset_to_indices.setdefault(dataset_name, []).append(i)

    if target_dataset is None:
        target_dataset = list(dataset_to_indices.keys())[0]

    if target_dataset not in dataset_to_indices:
        print(f"Error: Dataset '{target_dataset}' not found in paired dataset.")
        print("Available datasets:", list(dataset_to_indices.keys()))
        return

    sequence_indices = dataset_to_indices[target_dataset]
    if warm_m1init:
        sequence_indices.sort(key=lambda i: _frame_sort_key(dataset.sequences[i]))
    else:
        sequence_indices.sort(key=lambda i: _frame_sort_key(dataset.pairs[i]))

    frame_stride = max(1, int(frame_stride))
    if frame_stride > 1:
        sequence_indices = sequence_indices[::frame_stride]

    if max_frames is not None:
        sequence_indices = sequence_indices[:max_frames]

    print(f"Selected dataset: {target_dataset} with {len(sequence_indices)} frames.")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{out_prefix}_{random_word}_{target_dataset}.mp4")
    video_writer = None

    print("Generating video frames...")
    prev_state_a = None
    prev_state_b = None
    with torch.no_grad():
        for i, idx in enumerate(sequence_indices):
            data = dataset[idx]
            if warm_m1init:
                img_a_t = torch.from_numpy(data["image_a"]).to(exp.agent.device, dtype=torch.float32)[:1]
                lbl_a_t = torch.from_numpy(data["label_a"]).to(exp.agent.device, dtype=torch.float32)[:1]
                img_b_t = torch.from_numpy(data["image_b"]).to(exp.agent.device, dtype=torch.float32)[:1]
                lbl_b_t = torch.from_numpy(data["label_b"]).to(exp.agent.device, dtype=torch.float32)[:1]

                if i == 0 and use_m1_init:
                    m1_out, (prev_state_a, prev_state_b) = exp.agent.model.m1_forward_and_init_states(
                        img_a_t,
                        img_b_t,
                        lbl_a_t,
                        lbl_b_t,
                    )
                    if render_t0_with_m1:
                        pred = _extract_logits(m1_out, study_config["model.output_channels"])
                    else:
                        output = exp.agent.model(
                            img_a_t,
                            img_b_t,
                            y_a=lbl_a_t,
                            y_b=lbl_b_t,
                            prev_state_a=prev_state_a,
                            prev_state_b=prev_state_b,
                            batch_duplication=1,
                        )
                        pred = _extract_logits(output, study_config["model.output_channels"])
                        prev_state_a = output.get("final_state_a", prev_state_a)
                        prev_state_b = output.get("final_state_b", prev_state_b)
                else:
                    output = exp.agent.model(
                        img_a_t,
                        img_b_t,
                        y_a=lbl_a_t,
                        y_b=lbl_b_t,
                        prev_state_a=prev_state_a,
                        prev_state_b=prev_state_b,
                        batch_duplication=1,
                    )
                    pred = _extract_logits(output, study_config["model.output_channels"])
                    prev_state_a = output.get("final_state_a", prev_state_a)
                    prev_state_b = output.get("final_state_b", prev_state_b)

                image_a_bhwc = img_a_t.permute(0, 2, 3, 1)
                gt_a_bhwc = lbl_a_t.permute(0, 2, 3, 1)
                image_b_bhwc = img_b_t.permute(0, 2, 3, 1)
                gt_b_bhwc = lbl_b_t.permute(0, 2, 3, 1)
            else:
                img_a_np = data["image_a"]
                lbl_a_np = data["label_a"]
                img_b_np = data["image_b"]
                lbl_b_np = data["label_b"]

                input_a = torch.from_numpy(img_a_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(exp.agent.device)
                input_b = torch.from_numpy(img_b_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(exp.agent.device)

                output = exp.agent.model(input_a, input_b, batch_duplication=1)
                pred = _extract_logits(output, study_config["model.output_channels"])

                image_a_bhwc = torch.from_numpy(img_a_np).unsqueeze(0)
                gt_a_bhwc = torch.from_numpy(lbl_a_np).unsqueeze(0)
                image_b_bhwc = torch.from_numpy(img_b_np).unsqueeze(0)
                gt_b_bhwc = torch.from_numpy(lbl_b_np).unsqueeze(0)

            if pred.shape[0] < 2:
                raise RuntimeError(
                    f"Expected dual-view logits with batch >= 2 after concatenation, got shape {tuple(pred.shape)}"
                )

            pred_a = pred[0:1]
            pred_b = pred[1:2]

            merged = _merge_dual_view_rows(
                image_a_bhwc=image_a_bhwc,
                pred_a_bhwc=pred_a,
                gt_a_bhwc=gt_a_bhwc,
                image_b_bhwc=image_b_bhwc,
                pred_b_bhwc=pred_b,
                gt_b_bhwc=gt_b_bhwc,
            )

            frame_rgb = (merged * 255.0).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if video_writer is None:
                height, width, _ = frame_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
    parser = argparse.ArgumentParser(
        description="Create dual-view iOCT2D video with A(top)/B(bottom), each as image | prediction | ground truth."
    )
    parser.add_argument("--random_word", type=str, required=True, help="Experiment random word used during training.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset split to visualize (e.g., peeling, sri).")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="visualisationOCT")
    parser.add_argument(
        "--warm_m1init",
        action="store_true",
        help="Load checkpoints from train_ioct2d_dual_view_warm_preprocessed_m1init.py experiments.",
    )
    parser.add_argument(
        "--no_m1_init",
        action="store_true",
        help="Only for --warm_m1init: disable M1 hidden-state initialization for frame 0.",
    )
    parser.add_argument(
        "--m2_on_t0",
        action="store_true",
        help="Only for --warm_m1init: render M2 output on first frame instead of M1 output.",
    )
    args = parser.parse_args()

    create_video_ioct2d_dual_view(
        random_word=args.random_word,
        target_dataset=args.dataset,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        output_dir=args.output_dir,
        warm_m1init=args.warm_m1init,
        use_m1_init=not args.no_m1_init,
        render_t0_with_m1=not args.m2_on_t0,
    )
