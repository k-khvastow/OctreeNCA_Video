"""
Visualize OctreeNCA dual-view inference (M1 cold-start + M2 warm-start on
subsequent frames), saved as MP4 video.

Phase 1 — M1 cold-start: multi-resolution octree (4 levels), coarse→fine.
Phase 2 — M2 warm-start: single-scale NCA with temporal gating on next frames.

Usage:
    python visualize_octree_video.py \\
        --checkpoint '/vol/data/OctreeNCA_Video/refactored/<path>/<path>/octree_study_new/Experiments/WarmStart_M1Init_iOCT2D_dual_abbey_32_Dual-view iOCT warm-start with pretrained M1/models/epoch_6/model.pth' \\
        --output octree_inference.mp4 \\
        --sample 50 --num-frames 5 --fps 8
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import imageio


# ─── Minimal reimplementation of BasicNCA2DFast (matches checkpoint) ─────────

class BasicNCA2DFast(nn.Module):
    """Single NCA backbone matching the trained checkpoint structure.
    All ops in BCHW.  Perception via depthwise conv, update via 1x1 convs.
    No batch-norm (normalization='none' → Identity)."""

    def __init__(self, channel_n=32, hidden_size=64, input_channels=1,
                 kernel_size=3):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        padding = (kernel_size - 1) // 2
        # Perception: depthwise conv
        self.conv = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size,
                              padding=padding, groups=channel_n,
                              padding_mode='reflect')
        # Update: 1x1 convs  (input = identity + conv = 2*channel_n)
        self.fc0 = nn.Conv2d(channel_n * 2, hidden_size, 1)
        self.bn = nn.Identity()  # normalization='none'
        self.fc1 = nn.Conv2d(hidden_size, channel_n - input_channels, 1, bias=False)
        nn.init.zeros_(self.fc1.weight)

    def single_step_bchw(self, x, fire_rate=0.5):
        """One NCA step on a BCHW tensor.  Returns BCHW."""
        # Perceive
        y = self.conv(x)
        y = torch.cat([x, y], dim=1)  # (B, 2*C, H, W)
        # Update
        dx = self.fc0(y)
        dx = self.bn(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)  # (B, C - input_channels, H, W)
        # Stochastic update
        if fire_rate < 1.0:
            mask = (torch.rand(dx.shape[0], 1, dx.shape[2], dx.shape[3],
                               device=x.device) < fire_rate).float()
            dx = dx * mask
        # Residual — only update non-input channels
        new_non_input = x[:, self.input_channels:] + dx
        return torch.cat([x[:, :self.input_channels], new_non_input], dim=1)

    def forward_n_steps_bchw(self, x, steps=10, fire_rate=0.5):
        """Run *steps* NCA updates.  x: BCHW, returns BCHW."""
        for _ in range(steps):
            x = self.single_step_bchw(x, fire_rate)
        return x


# ─── Cross-view FiLM ────────────────────────────────────────────────────────

class CrossViewFiLM(nn.Module):
    """Per-level FiLM: pool hidden features of the other view and modulate."""

    def __init__(self, hidden_dim, n_levels, cross_strength=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_strength = cross_strength
        layers = []
        for _ in range(n_levels):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2 * hidden_dim),
            )
            with torch.no_grad():
                mlp[-1].weight.zero_()
                mlp[-1].bias.zero_()
            layers.append(mlp)
        self.cross_film = nn.ModuleList(layers)

    def fuse(self, state_a, state_b, level, hidden_start):
        """state_a, state_b: BCHW.  Mutually modulate hidden channels."""
        ha = state_a[:, hidden_start:]
        hb = state_b[:, hidden_start:]
        pooled_a = ha.mean(dim=(2, 3))
        pooled_b = hb.mean(dim=(2, 3))

        params_a = self.cross_film[level](pooled_b)
        params_b = self.cross_film[level](pooled_a)
        s = self.cross_strength

        for params, h, state, label in [
            (params_a, ha, state_a, 'a'),
            (params_b, hb, state_b, 'b'),
        ]:
            scale, shift = params.chunk(2, dim=1)
            scale = torch.tanh(scale) * s
            shift = torch.tanh(shift) * s
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            if label == 'a':
                state_a = torch.cat([state_a[:, :hidden_start], h], dim=1)
            else:
                state_b = torch.cat([state_b[:, :hidden_start], h], dim=1)
        return state_a, state_b


# ─── Dual-view OctreeNCA M1 (cold-start) ────────────────────────────────────

class OctreeM1DualView(nn.Module):
    """M1 submodel: shared-backbone dual-view multi-resolution OctreeNCA."""

    def __init__(self, channel_n=32, hidden_size=64, input_channels=1,
                 output_channels=7, kernel_size=3, fire_rate=0.5,
                 octree_res_and_steps=None, cross_strength=0.5):
        super().__init__()
        if octree_res_and_steps is None:
            octree_res_and_steps = [
                [[512, 512], 10], [[256, 256], 10],
                [[128, 128], 10], [[64, 64], 10],
            ]
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fire_rate = fire_rate

        self.octree_res = [tuple(rs[0]) for rs in octree_res_and_steps]
        self.steps_per_level = [rs[1] for rs in octree_res_and_steps]

        hidden_start = input_channels + output_channels
        hidden_dim = channel_n - hidden_start
        self._hidden_start = hidden_start

        # Shared backbone
        self.backbone_nca = BasicNCA2DFast(channel_n, hidden_size,
                                           input_channels, kernel_size)
        # Cross-view FiLM (one per level)
        self.cross_film_mod = CrossViewFiLM(hidden_dim, len(self.octree_res),
                                            cross_strength)
        # Precompute upsampling scales
        self._up_scales = []
        for i in range(len(self.octree_res) - 1):
            sh = self.octree_res[i][0] // self.octree_res[i + 1][0]
            sw = self.octree_res[i][1] // self.octree_res[i + 1][1]
            self._up_scales.append((sh, sw))

    @torch.no_grad()
    def downscale(self, x_bchw, level):
        return F.interpolate(x_bchw, size=self.octree_res[level])


# ─── Checkpoint loading ─────────────────────────────────────────────────────

def load_m1_from_checkpoint(checkpoint_path, device='cpu', config_path=None):
    """Load M1 dual-view OctreeNCA from a WarmStartM1Init checkpoint.

    Handles the _orig_mod prefix from torch.compile and m1.cross_film mapping.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # --- Read config if available ---
    if config_path is None:
        p = Path(checkpoint_path)
        for parent in p.parents:
            candidate = parent / 'config.json'
            if candidate.exists():
                config_path = str(candidate)
                break
    cfg = {}
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)

    channel_n = int(cfg.get('model.m1.channel_n', cfg.get('model.channel_n', 32)))
    hidden_size = int(cfg.get('model.hidden_size', 64))
    input_channels = int(cfg.get('model.input_channels', 1))
    output_channels = int(cfg.get('model.output_channels', 7))
    kernel_size = int(cfg.get('model.kernel_size', 3))
    fire_rate = float(cfg.get('model.fire_rate', 0.5))
    cross_strength = float(cfg.get('model.dual_view.cross_strength', 0.5))

    m1_num_levels = cfg.get('model.m1.num_levels', None)
    full_res = cfg.get('model.octree.res_and_steps',
                       [[[512, 512], 10], [[256, 256], 10],
                        [[128, 128], 10], [[64, 64], 10]])
    if m1_num_levels is not None:
        full_res = full_res[:int(m1_num_levels)]

    model = OctreeM1DualView(
        channel_n=channel_n, hidden_size=hidden_size,
        input_channels=input_channels, output_channels=output_channels,
        kernel_size=kernel_size, fire_rate=fire_rate,
        octree_res_and_steps=full_res, cross_strength=cross_strength,
    ).to(device)

    # --- Extract and remap M1 keys ---
    m1_state = {}
    for k, v in ckpt.items():
        if not k.startswith('m1.'):
            continue
        new_k = k[len('m1.'):]
        # Strip _orig_mod. prefix from torch.compile
        new_k = new_k.replace('._orig_mod.', '.')
        # Remap cross_film.X.Y -> cross_film_mod.cross_film.X.Y
        if new_k.startswith('cross_film.'):
            new_k = 'cross_film_mod.' + new_k
        m1_state[new_k] = v

    missing, unexpected = model.load_state_dict(m1_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    model.eval()
    print(f"Loaded M1 from checkpoint ({len(m1_state)} keys)")
    return model, cfg


# ─── M2 Warm-Start Model ────────────────────────────────────────────────────

class OctreeM2WarmStart(nn.Module):
    """M2 submodel: single-scale warm-start with temporal gating."""

    def __init__(self, channel_n=32, hidden_size=64, input_channels=1,
                 output_channels=7, kernel_size=3, fire_rate=0.5,
                 warm_start_steps=10, temporal_ratio=0.5, hidden_clip=5.0,
                 cross_strength=0.5):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fire_rate = fire_rate
        self.warm_start_steps = warm_start_steps

        hidden_start = input_channels + output_channels
        hidden_dim = channel_n - hidden_start
        self._hidden_start = hidden_start
        self._hidden_dim = hidden_dim
        self.hidden_clip = hidden_clip

        # Temporal / spatial split
        self.n_temporal = int(hidden_dim * temporal_ratio)
        self.n_spatial = hidden_dim - self.n_temporal

        # Shared backbone NCA
        self.backbone_nca = BasicNCA2DFast(channel_n, hidden_size,
                                           input_channels, kernel_size)
        # Cross-view FiLM (1 level for M2)
        self.cross_film_mod = CrossViewFiLM(hidden_dim, 1, cross_strength)

        # GRU-style temporal gate
        self.temporal_gate = nn.Conv2d(channel_n * 2, channel_n, 1, bias=True)
        with torch.no_grad():
            self.temporal_gate.weight.zero_()
            self.temporal_gate.bias.fill_(-2.0)

        # Hidden LayerNorm
        self._warm_hidden_ln = nn.LayerNorm(hidden_dim, elementwise_affine=True) \
            if hidden_dim > 0 else None

    def stabilize_hidden(self, state):
        """Clip + LayerNorm on hidden channels. state: BCHW."""
        hs = self._hidden_start
        if hs >= self.channel_n:
            return state
        left = state[:, :hs]
        hidden = state[:, hs:]
        if self.hidden_clip and self.hidden_clip > 0:
            hidden = hidden.clamp(-self.hidden_clip, self.hidden_clip)
        if self._warm_hidden_ln is not None:
            hidden = hidden.permute(0, 2, 3, 1).contiguous()
            hidden = self._warm_hidden_ln(hidden)
            hidden = hidden.permute(0, 3, 1, 2).contiguous()
        return torch.cat([left, hidden], dim=1)

    def reset_spatial_channels(self, state):
        if self.n_spatial <= 0:
            return state
        temporal_end = self._hidden_start + self.n_temporal
        state = state.clone()
        state[:, temporal_end:] = 0.0
        return state


def load_m2_from_checkpoint(checkpoint_path, cfg, device='cpu'):
    """Load M2 warm-start model from the same WarmStartM1Init checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    channel_n = int(cfg.get('model.channel_n', 32))
    hidden_size = int(cfg.get('model.hidden_size', 64))
    input_channels = int(cfg.get('model.input_channels', 1))
    output_channels = int(cfg.get('model.output_channels', 7))
    kernel_size = int(cfg.get('model.kernel_size', 3))
    fire_rate = float(cfg.get('model.fire_rate', 0.5))
    cross_strength = float(cfg.get('model.dual_view.cross_strength', 0.5))
    warm_start_steps = int(cfg.get('model.octree.warm_start_steps', 10))
    temporal_ratio = float(cfg.get('model.octree.warm_start_temporal_ratio', 0.5))
    hidden_clip = float(cfg.get('model.octree.warm_start_hidden_clip', 5.0))

    model = OctreeM2WarmStart(
        channel_n=channel_n, hidden_size=hidden_size,
        input_channels=input_channels, output_channels=output_channels,
        kernel_size=kernel_size, fire_rate=fire_rate,
        warm_start_steps=warm_start_steps,
        temporal_ratio=temporal_ratio, hidden_clip=hidden_clip,
        cross_strength=cross_strength,
    ).to(device)

    # Extract and remap M2 keys
    m2_state = {}
    for k, v in ckpt.items():
        if not k.startswith('m2.'):
            continue
        new_k = k[len('m2.'):]
        new_k = new_k.replace('._orig_mod.', '.')
        if new_k.startswith('cross_film.'):
            new_k = 'cross_film_mod.' + new_k
        m2_state[new_k] = v

    missing, unexpected = model.load_state_dict(m2_state, strict=False)
    if missing:
        print(f"[M2 WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[M2 WARN] Unexpected keys: {unexpected}")
    model.eval()
    print(f"Loaded M2 from checkpoint ({len(m2_state)} keys)")
    return model


# ─── Visualization helpers ──────────────────────────────────────────────────

# 7-class colours (iOCT dataset)
CLASS_COLORS = np.array([
    [0, 0, 0],       # 0 Background
    [255, 0, 0],     # 1 Red
    [0, 255, 209],   # 2 Cyan
    [61, 255, 0],    # 3 Green
    [0, 78, 255],    # 4 Blue
    [255, 189, 0],   # 5 Yellow
    [180, 80, 255],  # 6 Purple
], dtype=np.uint8)


def state_to_segmentation_rgb(state_bchw, input_channels, output_channels):
    """Extract segmentation output from state, return RGB (H, W, 3)."""
    logits = state_bchw[0, input_channels:input_channels + output_channels]
    class_map = logits.argmax(dim=0).cpu().numpy()
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(output_channels, len(CLASS_COLORS))):
        rgb[class_map == c] = CLASS_COLORS[c]
    return rgb


def state_to_input_rgb(state_bchw):
    """Extract input channel as grayscale RGB."""
    gray = (state_bchw[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def try_load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_gt_overlay(input_gray, gt_rgb, alpha=0.6):
    """Blend GT segmentation over greyscale input.
    input_gray: (H,W) uint8; gt_rgb: (H,W,3) uint8. Returns (H,W,3) uint8."""
    input_rgb = np.stack([input_gray] * 3, axis=-1)
    mask = np.any(gt_rgb > 0, axis=-1, keepdims=True)
    blended = (alpha * gt_rgb.astype(np.float32)
               + (1 - alpha) * input_rgb.astype(np.float32))
    result = np.where(mask, blended.clip(0, 255).astype(np.uint8), input_rgb)
    return result


def create_frame(level_states_a, level_states_b, active_level, global_step,
                 model_name, input_channels, output_channels,
                 tile_size=200, show_both_views=True, phase_label=None,
                 frame_idx=None, fixed_canvas_size=None,
                 gt_seg_a=None, gt_seg_b=None):
    """Render one frame with up to 4 rows: pred A, GT A, pred B, GT B."""
    n_levels = len(level_states_a)
    header_h = 50
    footer_h = 24
    row_label_w = 80 if show_both_views else 0
    padding = 4

    has_gt = gt_seg_a is not None
    n_sub_rows = 2 if has_gt else 1  # predictions + GT overlay per view
    n_views = 2 if show_both_views else 1
    n_total_rows = n_sub_rows * n_views

    canvas_w = row_label_w + tile_size * n_levels + padding * (n_levels + 1)
    canvas_h = (header_h
                + tile_size * n_total_rows
                + footer_h * n_views
                + padding * (n_total_rows + n_views))

    if fixed_canvas_size:
        final_w, final_h = fixed_canvas_size
    else:
        final_w, final_h = canvas_w, canvas_h

    canvas = Image.new('RGB', (final_w, final_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    x_offset = max(0, (final_w - canvas_w) // 2)

    font_title = try_load_font(22)
    font_small = try_load_font(14)
    font_label = try_load_font(12)
    font_view = try_load_font(14)

    # Header
    draw.text((12, 12), f"Step {global_step}", fill='white', font=font_title)
    if phase_label:
        pl_bbox = draw.textbbox((0, 0), phase_label, font=font_small)
        pl_w = pl_bbox[2] - pl_bbox[0]
        pl_color = (100, 200, 255) if "M2" in phase_label else (255, 200, 100)
        draw.text((final_w // 2 - pl_w // 2, 14), phase_label,
                  fill=pl_color, font=font_small)
    name_bbox = draw.textbbox((0, 0), model_name, font=font_small)
    name_w = name_bbox[2] - name_bbox[0]
    draw.text((final_w - name_w - 12, 14), model_name,
              fill=(180, 180, 180), font=font_small)

    views = [(level_states_a, gt_seg_a, "A")]
    if show_both_views:
        views.append((level_states_b, gt_seg_b, "B"))

    for view_idx, (states, gt_seg, vlabel) in enumerate(views):
        # y_base for this view group
        group_h = n_sub_rows * (tile_size + padding) + footer_h + padding
        y_base = header_h + view_idx * group_h

        # ── Sub-row 0: Predictions ──────────────────────────────────
        y_pred = y_base
        if show_both_views:
            draw.text((x_offset + 4, y_pred + tile_size // 2 - 8),
                      f"Pred {vlabel}", fill='white', font=font_view)

        for disp in range(n_levels):
            lvl = n_levels - 1 - disp
            state = states[lvl]
            h_res, w_res = state.shape[2], state.shape[3]
            rgb = state_to_segmentation_rgb(state, input_channels,
                                            output_channels)
            tile = Image.fromarray(rgb).resize((tile_size, tile_size),
                                               Image.NEAREST)
            x_pos = (x_offset + row_label_w + padding
                     + disp * (tile_size + padding))
            canvas.paste(tile, (x_pos, y_pred))

            if lvl == active_level:
                for t in range(3):
                    draw.rectangle([x_pos - t, y_pred - t,
                                    x_pos + tile_size + t - 1,
                                    y_pred + tile_size + t - 1],
                                   outline='red')

        # ── Sub-row 1: Input + GT overlay ───────────────────────────
        if has_gt and gt_seg is not None:
            y_gt = y_pred + tile_size + padding
            if show_both_views:
                draw.text((x_offset + 4, y_gt + tile_size // 2 - 8),
                          f"GT {vlabel}", fill=(150, 255, 150),
                          font=font_view)

            for disp in range(n_levels):
                lvl = n_levels - 1 - disp
                state = states[lvl]
                h_res, w_res = state.shape[2], state.shape[3]
                input_gray = (state[0, 0].cpu().numpy() * 255) \
                    .clip(0, 255).astype(np.uint8)
                gt_resized = np.array(Image.fromarray(gt_seg).resize(
                    (w_res, h_res), Image.NEAREST))
                overlay = make_gt_overlay(input_gray, gt_resized, alpha=0.6)
                tile = Image.fromarray(overlay).resize(
                    (tile_size, tile_size), Image.NEAREST)
                x_pos = (x_offset + row_label_w + padding
                         + disp * (tile_size + padding))
                canvas.paste(tile, (x_pos, y_gt))

        # ── Resolution labels at bottom of view group ───────────────
        y_labels = y_base + n_sub_rows * (tile_size + padding)
        for disp in range(n_levels):
            lvl = n_levels - 1 - disp
            state = states[lvl]
            h_res, w_res = state.shape[2], state.shape[3]
            label = f"{h_res}\u00d7{w_res}"
            lb = draw.textbbox((0, 0), label, font=font_label)
            lw = lb[2] - lb[0]
            x_pos = (x_offset + row_label_w + padding
                     + disp * (tile_size + padding))
            draw.text((x_pos + tile_size // 2 - lw // 2, y_labels),
                      label, fill='white', font=font_label)

    return canvas


# ─── Data loading ───────────────────────────────────────────────────────────

def load_dual_view_images(data_root, idx=50, input_size=(512, 512)):
    """Load a paired (view A, view B) iOCT image."""
    data_root = Path(data_root)
    pairs = _get_all_pairs(data_root)
    if not pairs:
        raise FileNotFoundError(f"No paired images found under {data_root}")
    idx = min(idx, len(pairs) - 1)
    print(f"Loading pair #{idx}: {pairs[idx][0].name}")
    return _load_pair(pairs[idx], input_size)


def load_dual_view_sequence(data_root, start_idx=50, num_frames=5,
                            seq_step=1, input_size=(512, 512)):
    """Load a sequence of (img_a, img_b, gt_a, gt_b) tuples.

    seq_step: stride between consecutive frames (like SEQ_STEP in cockpit).
    """
    data_root = Path(data_root)
    pairs = _get_all_pairs(data_root)
    if not pairs:
        raise FileNotFoundError(f"No paired images found under {data_root}")
    max_start = len(pairs) - 1 - (num_frames - 1) * seq_step
    start_idx = max(0, min(start_idx, max_start))
    sequence = []
    for i in range(num_frames):
        actual_idx = start_idx + i * seq_step
        if actual_idx >= len(pairs):
            break
        a, b, gt_a, gt_b = _load_pair(pairs[actual_idx], input_size)
        sequence.append((a, b, gt_a, gt_b))
    print(f"Loaded {len(sequence)} frame pairs starting at #{start_idx}, "
          f"step={seq_step}")
    return sequence


def _get_all_pairs(data_root):
    pairs = []
    for ds in ["peeling", "sri"]:
        dir_a_img = data_root / ds / "Bscans-dt" / "A" / "Image"
        dir_b_img = data_root / ds / "Bscans-dt" / "B" / "Image"
        dir_a_seg = data_root / ds / "Bscans-dt" / "A" / "Segmentation"
        dir_b_seg = data_root / ds / "Bscans-dt" / "B" / "Segmentation"
        if dir_a_img.exists() and dir_b_img.exists():
            names_a = sorted(dir_a_img.glob("*.png"))
            names_b = sorted(dir_b_img.glob("*.png"))
            for pa, pb in zip(names_a, names_b):
                sa = dir_a_seg / pa.name if dir_a_seg.exists() else None
                sb = dir_b_seg / pb.name if dir_b_seg.exists() else None
                pairs.append((pa, pb, sa, sb))
    return pairs


def _load_pair(pair, input_size):
    """Load image pair + optional GT segmentation masks.
    pair: (img_a_path, img_b_path[, seg_a_path, seg_b_path])."""
    tensors = []
    for p in pair[:2]:
        img = np.array(Image.open(p))
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize(
            (input_size[1], input_size[0]), Image.BILINEAR))
        img = img.astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0))
    gt_arrays = []
    for seg_path in pair[2:]:
        if seg_path is not None and Path(seg_path).exists():
            seg = np.array(Image.open(seg_path))
            seg = np.array(Image.fromarray(seg).resize(
                (input_size[1], input_size[0]), Image.NEAREST))
            if seg.ndim == 2:
                seg = np.stack([seg] * 3, axis=-1)
            gt_arrays.append(seg)
        else:
            gt_arrays.append(None)
    return tensors[0], tensors[1], gt_arrays[0], gt_arrays[1]


# ─── Main ───────────────────────────────────────────────────────────────────

def run_visualization(checkpoint_path, output_path, sample_idx=50, fps=8,
                      data_root="/vol/data/OctreeNCA_Video/ioct_data",
                      tile_size=200, show_both_views=True, num_frames=5,
                      seq_step=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load models ---
    m1_model, cfg = load_m1_from_checkpoint(checkpoint_path, device=device)
    m2_model = load_m2_from_checkpoint(checkpoint_path, cfg, device=device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # --- Load sequence of frames ---
    sequence = load_dual_view_sequence(data_root, sample_idx, num_frames,
                                       seq_step=seq_step)
    img_a_0, img_b_0 = sequence[0][0].to(device), sequence[0][1].to(device)
    gt_a_0, gt_b_0 = sequence[0][2], sequence[0][3]
    print(f"Input shapes: A={tuple(img_a_0.shape)}, B={tuple(img_b_0.shape)}")

    n_levels = len(m1_model.octree_res)
    input_ch = m1_model.input_channels
    output_ch = m1_model.output_channels
    model_name = "OctreeNCA"

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: M1 Cold-Start on frame 0
    # ═══════════════════════════════════════════════════════════════════════
    coarsest = n_levels - 1
    state_a = img_a_0.new_zeros((1, m1_model.channel_n, *m1_model.octree_res[coarsest]))
    state_b = img_b_0.new_zeros((1, m1_model.channel_n, *m1_model.octree_res[coarsest]))
    xa_coarse = m1_model.downscale(img_a_0, coarsest)
    xb_coarse = m1_model.downscale(img_b_0, coarsest)
    state_a[:, :input_ch] = xa_coarse[:, :input_ch]
    state_b[:, :input_ch] = xb_coarse[:, :input_ch]

    # Snapshot all levels for visualization
    level_snapshots_a = []
    level_snapshots_b = []
    for lv in range(n_levels):
        z_a = img_a_0.new_zeros((1, m1_model.channel_n, *m1_model.octree_res[lv]))
        z_b = img_b_0.new_zeros((1, m1_model.channel_n, *m1_model.octree_res[lv]))
        z_a[:, :input_ch] = m1_model.downscale(img_a_0, lv)[:, :input_ch]
        z_b[:, :input_ch] = m1_model.downscale(img_b_0, lv)[:, :input_ch]
        level_snapshots_a.append(z_a)
        level_snapshots_b.append(z_b)
    level_snapshots_a[coarsest] = state_a
    level_snapshots_b[coarsest] = state_b

    print(f"\n=== Phase 1: M1 Cold-Start (frame 0) ===")
    print(f"Octree levels ({n_levels}):")
    for i in range(n_levels):
        tag = 'coarsest' if i == coarsest else ('finest' if i == 0 else '')
        print(f"  Level {i}: {m1_model.octree_res[i][0]}\u00d7"
              f"{m1_model.octree_res[i][1]}  "
              f"{m1_model.steps_per_level[i]} steps  {tag}")

    frames = []
    global_step = 0

    # Compute fixed canvas size based on M1 (the widest: 4 levels)
    # to keep all frames the same size for the video encoder
    row_label_w = 80 if show_both_views else 0
    padding = 4
    n_views = 2 if show_both_views else 1
    n_sub_rows = 2  # predictions + GT overlay
    n_total_rows = n_sub_rows * n_views
    header_h = 50
    footer_h = 24
    fixed_w = row_label_w + tile_size * n_levels + padding * (n_levels + 1)
    fixed_h = (header_h
               + tile_size * n_total_rows
               + footer_h * n_views
               + padding * (n_total_rows + n_views))
    # Round up to multiple of 16 for ffmpeg compatibility
    fixed_w = ((fixed_w + 15) // 16) * 16
    fixed_h = ((fixed_h + 15) // 16) * 16
    fixed_canvas_size = (fixed_w, fixed_h)

    with torch.no_grad():
        # --- M1: coarsest to finest ---
        for level_idx in range(coarsest, -1, -1):
            n_steps = m1_model.steps_per_level[level_idx]
            nca = m1_model.backbone_nca

            print(f"  Level {level_idx} "
                  f"({m1_model.octree_res[level_idx][0]}"
                  f"\u00d7{m1_model.octree_res[level_idx][1]}): "
                  f"{n_steps} steps")

            for step in range(n_steps):
                frame = create_frame(
                    level_snapshots_a, level_snapshots_b,
                    active_level=level_idx,
                    global_step=global_step,
                    model_name=model_name,
                    input_channels=input_ch,
                    output_channels=output_ch,
                    tile_size=tile_size,
                    show_both_views=show_both_views,
                    phase_label="M1 Cold-Start (frame 1)",
                    fixed_canvas_size=fixed_canvas_size,
                    gt_seg_a=gt_a_0, gt_seg_b=gt_b_0,
                )
                frames.append(np.array(frame))

                ab = torch.cat([state_a, state_b], dim=0)
                ab = nca.single_step_bchw(ab, fire_rate=m1_model.fire_rate)
                state_a, state_b = ab.chunk(2, dim=0)
                level_snapshots_a[level_idx] = state_a
                level_snapshots_b[level_idx] = state_b
                global_step += 1

            # Cross-view FiLM
            state_a, state_b = m1_model.cross_film_mod.fuse(
                state_a, state_b, level_idx, m1_model._hidden_start)
            level_snapshots_a[level_idx] = state_a
            level_snapshots_b[level_idx] = state_b

            # Upscale to finer level
            if level_idx > 0:
                sh, sw = m1_model._up_scales[level_idx - 1]
                state_a = state_a.repeat_interleave(sh, dim=2) \
                                 .repeat_interleave(sw, dim=3)
                state_b = state_b.repeat_interleave(sh, dim=2) \
                                 .repeat_interleave(sw, dim=3)
                inj_a = m1_model.downscale(img_a_0, level_idx - 1)
                inj_b = m1_model.downscale(img_b_0, level_idx - 1)
                state_a[:, :input_ch] = inj_a[:, :input_ch]
                state_b[:, :input_ch] = inj_b[:, :input_ch]
                level_snapshots_a[level_idx - 1] = state_a
                level_snapshots_b[level_idx - 1] = state_b

        # Final M1 frame
        frames.append(np.array(create_frame(
            level_snapshots_a, level_snapshots_b,
            active_level=0, global_step=global_step,
            model_name=model_name,
            input_channels=input_ch, output_channels=output_ch,
            tile_size=tile_size, show_both_views=show_both_views,
            phase_label="M1 Cold-Start (frame 1) — done",
            fixed_canvas_size=fixed_canvas_size,
            gt_seg_a=gt_a_0, gt_seg_b=gt_b_0,
        )))
        # Hold final M1 frame for a moment
        for _ in range(fps):  # 1 second pause
            frames.append(frames[-1])

        # ═══════════════════════════════════════════════════════════════════
        # Extract M1 final state → M2 initial state (at finest resolution)
        # ═══════════════════════════════════════════════════════════════════
        # state_a, state_b are already at finest (512x512) in BCHW
        prev_state_a = state_a.clone()
        prev_state_b = state_b.clone()

        # ═══════════════════════════════════════════════════════════════════
        # Phase 2: M2 Warm-Start on frames 1..N
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n=== Phase 2: M2 Warm-Start ({len(sequence)-1} frames) ===")
        print(f"  Steps per frame: {m2_model.warm_start_steps}")
        print(f"  Temporal ratio: {m2_model.n_temporal}/{m2_model._hidden_dim}")

        for frame_t in range(1, len(sequence)):
            img_a_t = sequence[frame_t][0].to(device)
            img_b_t = sequence[frame_t][1].to(device)
            gt_a_t, gt_b_t = sequence[frame_t][2], sequence[frame_t][3]
            print(f"  Frame {frame_t+1}/{len(sequence)}")

            # --- Prepare state for warm-start ---
            ws_a = prev_state_a.clone()
            ws_b = prev_state_b.clone()

            # Reset spatial channels
            ws_a = m2_model.reset_spatial_channels(ws_a)
            ws_b = m2_model.reset_spatial_channels(ws_b)

            # Inject new input
            ws_a[:, :input_ch] = img_a_t[:, :input_ch]
            ws_b[:, :input_ch] = img_b_t[:, :input_ch]

            # Stabilize hidden
            ws_a = m2_model.stabilize_hidden(ws_a)
            ws_b = m2_model.stabilize_hidden(ws_b)

            # For M2 visualization: show single-scale state (finest only)
            # We display all octree levels but only level 0 is active
            m2_snapshots_a = [ws_a.clone()]  # level 0 = finest
            m2_snapshots_b = [ws_b.clone()]

            n_steps = m2_model.warm_start_steps
            nca = m2_model.backbone_nca

            for step in range(n_steps):
                # Frame before step
                frame = create_frame(
                    m2_snapshots_a, m2_snapshots_b,
                    active_level=0,
                    global_step=global_step,
                    model_name=model_name,
                    input_channels=input_ch,
                    output_channels=output_ch,
                    tile_size=tile_size,
                    show_both_views=show_both_views,
                    phase_label=f"M2 Warm-Start (frame {frame_t+1}/{len(sequence)})",
                    fixed_canvas_size=fixed_canvas_size,
                    gt_seg_a=gt_a_t, gt_seg_b=gt_b_t,
                )
                frames.append(np.array(frame))

                # Single NCA step
                ab = torch.cat([ws_a, ws_b], dim=0)
                ab = nca.single_step_bchw(ab, fire_rate=m2_model.fire_rate)
                ws_a, ws_b = ab.chunk(2, dim=0)
                m2_snapshots_a[0] = ws_a
                m2_snapshots_b[0] = ws_b
                global_step += 1

            # Cross-view FiLM (level 0 for M2)
            cand_a, cand_b = m2_model.cross_film_mod.fuse(
                ws_a, ws_b, 0, m2_model._hidden_start)

            # Temporal gate (GRU-style)
            z_a = torch.sigmoid(m2_model.temporal_gate(
                torch.cat([prev_state_a, cand_a], dim=1)))
            z_b = torch.sigmoid(m2_model.temporal_gate(
                torch.cat([prev_state_b, cand_b], dim=1)))
            state_a = (1.0 - z_a) * prev_state_a + z_a * cand_a
            state_b = (1.0 - z_b) * prev_state_b + z_b * cand_b

            # Re-inject input
            state_a[:, :input_ch] = img_a_t[:, :input_ch]
            state_b[:, :input_ch] = img_b_t[:, :input_ch]

            # Stabilize
            state_a = m2_model.stabilize_hidden(state_a)
            state_b = m2_model.stabilize_hidden(state_b)

            # Final frame for this timestep
            m2_snapshots_a[0] = state_a
            m2_snapshots_b[0] = state_b
            frames.append(np.array(create_frame(
                m2_snapshots_a, m2_snapshots_b,
                active_level=0, global_step=global_step,
                model_name=model_name,
                input_channels=input_ch, output_channels=output_ch,
                tile_size=tile_size, show_both_views=show_both_views,
                phase_label=f"M2 (frame {frame_t+1}/{len(sequence)}) — gated",
                fixed_canvas_size=fixed_canvas_size,
                gt_seg_a=gt_a_t, gt_seg_b=gt_b_t,
            )))

            # Hold last frame briefly
            for _ in range(max(1, fps // 2)):
                frames.append(frames[-1])

            prev_state_a = state_a
            prev_state_b = state_b

    # --- Save video ---
    output_path = str(output_path)
    if output_path.endswith('.gif'):
        print(f"\nSaving GIF ({len(frames)} frames) ...")
        imageio.mimsave(output_path, frames, duration=1000 // fps, loop=0)
    else:
        print(f"\nSaving MP4 ({len(frames)} frames) ...")
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                    pixelformat='yuv420p', quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()

    print(f"Saved to {output_path} ({len(frames)} frames, "
          f"{len(frames)/fps:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize OctreeNCA M1 cold-start + M2 warm-start inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model.pth")
    parser.add_argument("--output", type=str, default="octree_inference.mp4",
                        help="Output path (.mp4 or .gif)")
    parser.add_argument("--sample", type=int, default=50,
                        help="Starting image pair index")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of sequential frames (1=M1 only)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second")
    parser.add_argument("--data-root", type=str,
                        default="/vol/data/OctreeNCA_Video/ioct_data",
                        help="iOCT data root")
    parser.add_argument("--tile-size", type=int, default=200,
                        help="Tile size for each stage in pixels")
    parser.add_argument("--single-view", action='store_true',
                        help="Show only view A (default: show both A and B)")
    parser.add_argument("--seq-step", type=int, default=1,
                        help="Step between consecutive frames (like SEQ_STEP "
                             "in cockpit, e.g. 10)")
    args = parser.parse_args()

    run_visualization(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        sample_idx=args.sample,
        fps=args.fps,
        data_root=args.data_root,
        tile_size=args.tile_size,
        show_both_views=not args.single_view,
        num_frames=args.num_frames,
        seq_step=args.seq_step,
    )
