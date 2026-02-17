#
# python v_create_video_tbptt_multiscale.py --random_word <word>
#
# Visualisation for models trained with script_train_dual_view_warm_tbptt.sh.
#
# Layout:
#   Left:  5 upscaling stages of the M1 cold-start on the *first* frame
#          (coarsest → finest), each upscaled to full resolution.
#   Sep:   A thin black vertical separator bar.
#   Right: Warm-start M2 predictions on subsequent frames (sliding through time).
#
# Each panel shows dual-view rows (A on top, B on bottom) with
# image | prediction | ground-truth sub-columns.
#
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_ioct2d_dual_view_warm_preprocessed_m1init import (
    EXP_OctreeNCA_DualView_WarmStart_M1Init,
    get_dataset_args as get_dataset_args_warm,
    get_study_config as get_study_config_warm,
    iOCTPairedSequentialDatasetForExperiment,
)
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
import colormaps as cmaps

# ── Architecture config keys to pre-load from checkpoint ────────────────
_ARCHITECTURE_CONFIG_KEYS = [
    "model.octree.warm_start_temporal_gate",
    "model.octree.warm_start_hidden_clip",
    "model.octree.warm_start_hidden_tanh_scale",
    "model.octree.warm_start_hidden_gn_groups",
]

SEP_WIDTH = 8  # pixel width of the black separator bar


# ── Helpers ─────────────────────────────────────────────────────────────

def _preload_saved_config(study_config: dict) -> None:
    model_path_base = os.path.join(
        pc.FILER_BASE_PATH,
        study_config.get(
            "experiment.model_path",
            os.path.join(
                pc.STUDY_PATH,
                "Experiments",
                study_config["experiment.name"] + "_" + study_config["experiment.description"],
            ),
        ),
    )
    config_path = os.path.join(model_path_base, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, "r") as f:
        saved = json.load(f)
    for key in _ARCHITECTURE_CONFIG_KEYS:
        if key in saved:
            study_config[key] = saved[key]


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
    lo = torch.min(img)
    hi = torch.max(img)
    if (hi - lo) > 0:
        img = (img - lo) / (hi - lo)
    return img


def _mask_to_rgb(mask_bhwc: torch.Tensor, bg=(0.0, 0.0, 0.0)) -> torch.Tensor:
    mask = mask_bhwc if isinstance(mask_bhwc, torch.Tensor) else torch.from_numpy(mask_bhwc)
    if mask.ndim != 4:
        raise ValueError(f"Expected BHWC, got {tuple(mask.shape)}")
    b, h, w, c = mask.shape
    if c > 1:
        class_idx = torch.argmax(mask, dim=-1)
        mask = torch.nn.functional.one_hot(class_idx, num_classes=c).to(torch.bool)
    elif mask.dtype != torch.bool:
        mask = mask > 0.5

    bg_t = torch.tensor(bg, dtype=torch.float32, device=mask.device)
    rgb = bg_t.view(1, 1, 1, 3).expand(b, h, w, 3).clone()
    for ci in range(1, c):
        color = torch.tensor(cmaps.bold[ci].colors, dtype=torch.float32, device=mask.device)
        rgb[mask[..., ci]] = color
    return rgb


def _three_panel(image_bhwc, pred_bhwc, gt_bhwc):
    """image | prediction | ground-truth  →  (B, H, 3W, 3)"""
    device = pred_bhwc.device
    img = image_bhwc.to(device)
    gt = gt_bhwc.to(device)
    if img.shape[-1] != 3:
        img = img.repeat(1, 1, 1, 3)
    img = _normalize_image(img)
    pred_rgb = _mask_to_rgb(pred_bhwc)
    gt_rgb = _mask_to_rgb(gt)
    return torch.cat([img, pred_rgb, gt_rgb], dim=2)


def _dual_view_panel(img_a, pred_a, gt_a, img_b, pred_b, gt_b):
    """Two rows (A/B) stacked vertically → (H*2, W*3, 3) numpy."""
    row_a = _three_panel(img_a, pred_a, gt_a)
    row_b = _three_panel(img_b, pred_b, gt_b)
    merged = torch.cat([row_a, row_b], dim=1)  # stack rows
    return merged.squeeze(0).cpu().numpy()


def _pred_only_dual(pred_a_bhwc, pred_b_bhwc):
    """Prediction-only dual-view panel (no image/gt columns)."""
    pred_a_rgb = _mask_to_rgb(pred_a_bhwc)
    pred_b_rgb = _mask_to_rgb(pred_b_bhwc)
    merged = torch.cat([pred_a_rgb, pred_b_rgb], dim=1)
    return merged.squeeze(0).cpu().numpy()


# ── Per-level cold-start extraction ────────────────────────────────────

@torch.no_grad()
def _m1_cold_start_per_level(model, x_a_bchw, x_b_bchw):
    """Run the M1 dual-view cold-start and collect per-level logits (BHWC)
    for both views.  Returns list of (logits_a, logits_b) from coarsest
    to finest, each upscaled to full resolution."""
    m1 = model.m1
    # unwrap torch.compile if present
    while hasattr(m1, "_orig_mod"):
        m1 = m1._orig_mod

    input_ch = m1.input_channels
    out_ch = m1.output_channels
    octree_res = m1.octree_res
    full_h, full_w = octree_res[0]

    state_a = x_a_bchw.new_zeros((x_a_bchw.shape[0], m1.channel_n, *octree_res[-1]))
    state_b = x_b_bchw.new_zeros((x_b_bchw.shape[0], m1.channel_n, *octree_res[-1]))
    xa_c = m1.downscale(x_a_bchw, -1, layout="BCHW")
    xb_c = m1.downscale(x_b_bchw, -1, layout="BCHW")
    state_a[:, :input_ch] = xa_c[:, :input_ch]
    state_b[:, :input_ch] = xb_c[:, :input_ch]

    per_level = []  # (logits_a_BHWC, logits_b_BHWC) at full res

    for level in range(len(octree_res) - 1, -1, -1):
        state_ab = torch.cat([state_a, state_b], dim=0)
        state_ab = m1._run_backbone(state_ab, level)
        state_a, state_b = state_ab.chunk(2, dim=0)
        state_a, state_b = m1._maybe_cross_fuse(state_a, state_b, level)

        # Extract logits at this level and upscale to full res for display
        logits_a = state_a[:, input_ch:input_ch + out_ch]
        logits_b = state_b[:, input_ch:input_ch + out_ch]
        if logits_a.shape[2:] != (full_h, full_w):
            logits_a = F.interpolate(logits_a, size=(full_h, full_w), mode="nearest")
            logits_b = F.interpolate(logits_b, size=(full_h, full_w), mode="nearest")
        # Apply softmax for display if the model has one
        logits_a = logits_a.permute(0, 2, 3, 1)  # BHWC
        logits_b = logits_b.permute(0, 2, 3, 1)
        if m1.apply_nonlin is not None:
            logits_a = m1.apply_nonlin(logits_a)
            logits_b = m1.apply_nonlin(logits_b)
        per_level.append((logits_a, logits_b))

        if level > 0:
            sh, sw = m1.computed_upsampling_scales[level - 1][0]
            sh, sw = int(sh), int(sw)
            state_a = state_a.repeat_interleave(sh, dim=2).repeat_interleave(sw, dim=3)
            state_b = state_b.repeat_interleave(sh, dim=2).repeat_interleave(sw, dim=3)
            inj_a = m1.downscale(x_a_bchw, level - 1, layout="BCHW")
            inj_b = m1.downscale(x_b_bchw, level - 1, layout="BCHW")
            state_a[:, :input_ch] = inj_a[:, :input_ch]
            state_b[:, :input_ch] = inj_b[:, :input_ch]

    # Also return final hidden for state init
    hidden_a = state_a[:, input_ch + out_ch:]
    hidden_b = state_b[:, input_ch + out_ch:]
    final_logits_a = state_a[:, input_ch:input_ch + out_ch]
    final_logits_b = state_b[:, input_ch:input_ch + out_ch]

    return per_level, (state_a, state_b, final_logits_a, final_logits_b, hidden_a, hidden_b)


# ── Main ───────────────────────────────────────────────────────────────

def create_video(
    random_word,
    target_dataset=None,
    fps=10,
    max_frames=None,
    frame_stride=1,
    output_dir="visualisationOCT",
    use_m1_init=True,
    warm_frames_per_video_frame=2,
):
    study_config = get_study_config_warm()
    study_config["experiment.name"] = (
        f"WarmStart_M1Init_iOCT2D_dual_{random_word}_{study_config['model.channel_n']}"
    )
    exp_class = EXP_OctreeNCA_DualView_WarmStart_M1Init
    dataset_class = iOCTPairedSequentialDatasetForExperiment
    dataset_args = get_dataset_args_warm(study_config)
    dataset_args["sequence_length"] = 1
    dataset_args["sequence_step"] = 1

    study_config["experiment.use_wandb"] = False
    study_config["experiment.dataset.preload"] = False
    _preload_saved_config(study_config)

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
    for i, info in enumerate(dataset.sequences):
        dataset_name = info["dataset"]
        dataset_to_indices.setdefault(dataset_name, []).append(i)

    if target_dataset is None:
        target_dataset = list(dataset_to_indices.keys())[0]
    if target_dataset not in dataset_to_indices:
        print(f"Error: Dataset '{target_dataset}' not found.")
        print("Available:", list(dataset_to_indices.keys()))
        return

    sequence_indices = dataset_to_indices[target_dataset]
    sequence_indices.sort(key=lambda i: _frame_sort_key(dataset.sequences[i]))

    frame_stride = max(1, int(frame_stride))
    if frame_stride > 1:
        sequence_indices = sequence_indices[::frame_stride]
    if max_frames is not None:
        sequence_indices = sequence_indices[:max_frames]

    print(f"Selected dataset: {target_dataset} with {len(sequence_indices)} frames.")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"tbptt_multiscale_{random_word}_{target_dataset}.mp4",
    )
    video_writer = None
    output_channels = study_config["model.output_channels"]
    model = exp.agent.model

    # ── Collect all frame data first ──────────────────────────────────
    print("Loading all frame tensors...")
    all_img_a, all_lbl_a, all_img_b, all_lbl_b = [], [], [], []
    for idx in sequence_indices:
        data = dataset[idx]
        all_img_a.append(torch.from_numpy(data["image_a"]).to(exp.agent.device, dtype=torch.float32)[:1])
        all_lbl_a.append(torch.from_numpy(data["label_a"]).to(exp.agent.device, dtype=torch.float32)[:1])
        all_img_b.append(torch.from_numpy(data["image_b"]).to(exp.agent.device, dtype=torch.float32)[:1])
        all_lbl_b.append(torch.from_numpy(data["label_b"]).to(exp.agent.device, dtype=torch.float32)[:1])

    n_frames = len(all_img_a)
    if n_frames < 1:
        print("No frames to process.")
        return

    print(f"Processing {n_frames} frames...")

    # ── First frame: M1 cold-start with per-level stages ─────────────
    print("Running M1 cold-start on first frame (per-level extraction)...")
    with torch.no_grad():
        img_a_0 = all_img_a[0]
        img_b_0 = all_img_b[0]
        lbl_a_0 = all_lbl_a[0]
        lbl_b_0 = all_lbl_b[0]

        per_level_preds, (final_sa, final_sb, fin_logits_a, fin_logits_b, hid_a, hid_b) = \
            _m1_cold_start_per_level(model, img_a_0, img_b_0)

        # Build the 5 upscaling-stage panels (pred-only, no image/gt, to save space)
        # per_level_preds is coarsest→finest
        n_levels = len(per_level_preds)
        print(f"  Captured {n_levels} upscaling levels.")

        # Build stage panels: each shows prediction at that level
        stage_panels = []
        for li, (pred_a_bhwc, pred_b_bhwc) in enumerate(per_level_preds):
            panel = _pred_only_dual(pred_a_bhwc, pred_b_bhwc)
            stage_panels.append(panel)

        # Also build the combined M1 final panel with image|pred|gt
        img_a_bhwc = img_a_0.permute(0, 2, 3, 1)
        img_b_bhwc = img_b_0.permute(0, 2, 3, 1)
        gt_a_bhwc = lbl_a_0.permute(0, 2, 3, 1)
        gt_b_bhwc = lbl_b_0.permute(0, 2, 3, 1)
        m1_final_pred_a = per_level_preds[-1][0]  # finest level
        m1_final_pred_b = per_level_preds[-1][1]
        m1_panel = _dual_view_panel(img_a_bhwc, m1_final_pred_a, gt_a_bhwc,
                                    img_b_bhwc, m1_final_pred_b, gt_b_bhwc)

        # Initialize warm-start states from M1 output
        if use_m1_init:
            m1_out_dict = {
                "logits": torch.cat([
                    fin_logits_a.permute(0, 2, 3, 1),
                    fin_logits_b.permute(0, 2, 3, 1),
                ], dim=0),
                "hidden_channels": torch.cat([
                    hid_a.permute(0, 2, 3, 1),
                    hid_b.permute(0, 2, 3, 1),
                ], dim=0),
            }
            if model.m1.apply_nonlin is not None:
                m1_out_dict["probabilities"] = model.m1.apply_nonlin(m1_out_dict["logits"])
            # Unwrap compiled module for _states_from_m1_output
            real_model = model
            while hasattr(real_model, "_orig_mod"):
                real_model = real_model._orig_mod
            prev_state_a, prev_state_b = real_model._states_from_m1_output(
                img_a_0, img_b_0, m1_out_dict
            )
        else:
            prev_state_a = None
            prev_state_b = None

    # ── Build left-side (fixed) image: stage panels ──────────────────
    # Assemble: [stage0 | stage1 | ... | stageN-1 | M1_final_panel]
    # All panels should have the same height. Stage panels may differ in width
    # from the full m1_panel. We'll normalize heights.
    target_h = m1_panel.shape[0]

    def _resize_panel(panel, target_h):
        ph, pw, _ = panel.shape
        if ph != target_h:
            scale = target_h / ph
            new_w = int(pw * scale)
            panel = cv2.resize(panel, (new_w, target_h), interpolation=cv2.INTER_NEAREST)
        return panel

    stage_panels_resized = [_resize_panel((p * 255).clip(0, 255).astype(np.uint8), target_h)
                            for p in stage_panels]
    m1_panel_u8 = (m1_panel * 255).clip(0, 255).astype(np.uint8)

    # Add level labels on each stage panel
    for li, sp in enumerate(stage_panels_resized):
        label = f"L{n_levels - 1 - li}"
        level_idx = n_levels - 1 - li
        res = model.m1.octree_res[level_idx] if hasattr(model.m1, "octree_res") else "?"
        while hasattr(res, "_orig_mod"):
            pass  # res is not a module
        label = f"L{level_idx} ({res[0]}x{res[1]})" if isinstance(res, (tuple, list)) else f"L{level_idx}"
        cv2.putText(sp, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Label on M1 final panel
    cv2.putText(m1_panel_u8, "M1 final (img|pred|gt)", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Separator
    sep = np.zeros((target_h, SEP_WIDTH, 3), dtype=np.uint8)
    sep[:, :, :] = 40  # dark grey

    left_side = np.concatenate(stage_panels_resized + [m1_panel_u8, sep], axis=1)
    left_w = left_side.shape[1]

    # ── Process warm-start frames and write video ────────────────────
    print("Generating video frames with warm-start progression...")

    with torch.no_grad():
        for t in range(1, n_frames):
            img_a_t = all_img_a[t]
            img_b_t = all_img_b[t]
            lbl_a_t = all_lbl_a[t]
            lbl_b_t = all_lbl_b[t]

            output = model(
                img_a_t,
                img_b_t,
                y_a=lbl_a_t,
                y_b=lbl_b_t,
                prev_state_a=prev_state_a,
                prev_state_b=prev_state_b,
                batch_duplication=1,
            )
            pred = _extract_logits(output, output_channels)
            prev_state_a = output.get("final_state_a", prev_state_a)
            prev_state_b = output.get("final_state_b", prev_state_b)

            pred_a = pred[0:1]
            pred_b = pred[1:2]

            img_a_bhwc = img_a_t.permute(0, 2, 3, 1)
            gt_a_bhwc = lbl_a_t.permute(0, 2, 3, 1)
            img_b_bhwc = img_b_t.permute(0, 2, 3, 1)
            gt_b_bhwc = lbl_b_t.permute(0, 2, 3, 1)

            warm_panel = _dual_view_panel(img_a_bhwc, pred_a, gt_a_bhwc,
                                          img_b_bhwc, pred_b, gt_b_bhwc)
            warm_panel_u8 = (warm_panel * 255).clip(0, 255).astype(np.uint8)
            cv2.putText(warm_panel_u8, f"M2 t={t} (img|pred|gt)", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Also try to grab the next 1-2 frames for preview (without advancing state)
            right_panels = [warm_panel_u8]

            # Peek at the next `warm_frames_per_video_frame - 1` frames
            for dt in range(1, warm_frames_per_video_frame):
                peek_t = t + dt
                if peek_t >= n_frames:
                    break
                peek_out = model(
                    all_img_a[peek_t],
                    all_img_b[peek_t],
                    y_a=all_lbl_a[peek_t],
                    y_b=all_lbl_b[peek_t],
                    prev_state_a=prev_state_a,
                    prev_state_b=prev_state_b,
                    batch_duplication=1,
                )
                peek_pred = _extract_logits(peek_out, output_channels)
                pp_a = peek_pred[0:1]
                pp_b = peek_pred[1:2]
                pp_img_a = all_img_a[peek_t].permute(0, 2, 3, 1)
                pp_gt_a = all_lbl_a[peek_t].permute(0, 2, 3, 1)
                pp_img_b = all_img_b[peek_t].permute(0, 2, 3, 1)
                pp_gt_b = all_lbl_b[peek_t].permute(0, 2, 3, 1)
                peek_panel = _dual_view_panel(pp_img_a, pp_a, pp_gt_a,
                                              pp_img_b, pp_b, pp_gt_b)
                peek_u8 = (peek_panel * 255).clip(0, 255).astype(np.uint8)
                cv2.putText(peek_u8, f"M2 t={peek_t} (preview)", (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                right_panels.append(peek_u8)

            # Resize right panels to match left height
            right_panels_resized = [_resize_panel(p, target_h) for p in right_panels]
            right_side = np.concatenate(right_panels_resized, axis=1)

            # Compose full frame
            frame_bgr = np.concatenate([left_side, right_side], axis=1)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

            if video_writer is None:
                h_out, w_out, _ = frame_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w_out, h_out))
                print(f"  Video resolution: {w_out} x {h_out}")

            video_writer.write(frame_bgr)

            if t % 10 == 0:
                print(f"  Frame {t}/{n_frames - 1}")

    if video_writer:
        video_writer.release()
        print(f"\nVideo saved: {os.path.abspath(out_path)}")
    else:
        print("No frames written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create TBPTT dual-view video: M1 upscaling stages on the left, "
            "warm-start M2 predictions sliding through time on the right."
        ),
    )
    parser.add_argument("--random_word", type=str, required=True,
                        help="Experiment random word from training.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset split to visualize (e.g. peeling, sri).")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="visualisationOCT")
    parser.add_argument("--warm_preview_frames", type=int, default=2,
                        help="How many warm-start frames to show side-by-side (current + N-1 previews).")
    parser.add_argument("--no_m1_init", action="store_true",
                        help="Disable M1 hidden-state initialization.")
    args = parser.parse_args()

    create_video(
        random_word=args.random_word,
        target_dataset=args.dataset,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        output_dir=args.output_dir,
        use_m1_init=not args.no_m1_init,
        warm_frames_per_video_frame=args.warm_preview_frames,
    )
