"""
Visualize OctreeNCA inference: all resolution stages side-by-side,
stepping from coarsest to finest, saved as MP4 video.

Automatically detects from the experiment config.json:
  - Number of octree levels and steps per level
  - Single-view vs dual-view (FiLM) mode
  - Model hyper-parameters (channel_n, hidden_size, kernel_size, etc.)

Usage:
    python visualize_octree_stages.py \
        --checkpoint /vol/data/OctreeNCA_Video/.../model.pth \
        --output octree_inference.mp4 \
        --sample 50 --fps 8
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


# ─── NCA backbone ───────────────────────────────────────────────────────────

class BasicNCA2DFast(nn.Module):
    """Single-level NCA backbone matching the trained checkpoint structure."""

    def __init__(self, channel_n=24, hidden_size=32, input_channels=1, kernel_size=3):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.conv = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=channel_n,
                              padding_mode='reflect')
        self.fc0 = nn.Conv2d(channel_n * 2, hidden_size, 1)
        self.fc1 = nn.Conv2d(hidden_size, channel_n - input_channels, 1, bias=False)
        nn.init.zeros_(self.fc1.weight)

    def forward_bchw(self, x_bchw, fire_rate=1.0):
        return self._step(x_bchw, fire_rate)

    def _step(self, x, fire_rate):
        y = self.conv(x)
        y = torch.cat([x, y], dim=1)
        dx = F.relu(self.fc0(y))
        dx = self.fc1(dx)
        if fire_rate < 1.0:
            mask = (torch.rand(dx.shape[0], 1, dx.shape[2], dx.shape[3],
                               device=x.device) < fire_rate).float()
            dx = dx * mask
        x_inp = x[:, :self.input_channels]
        x_state = x[:, self.input_channels:] + dx
        return torch.cat([x_inp, x_state], dim=1)


# ─── OctreeNCA model (single-view) ──────────────────────────────────────────

class OctreeNCA_SeparateModels(nn.Module):
    def __init__(self, channel_n=24, hidden_size=32, input_channels=1,
                 output_channels=7, n_levels=4, kernel_size=3,
                 separate_models=True):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_levels = n_levels
        self.separate_models = separate_models
        if separate_models:
            self.backbone_ncas = nn.ModuleList([
                BasicNCA2DFast(channel_n, hidden_size, input_channels, kernel_size)
                for _ in range(n_levels)
            ])
        else:
            self.backbone_nca = BasicNCA2DFast(channel_n, hidden_size, input_channels, kernel_size)

    def get_backbone(self, level):
        return self.backbone_ncas[level] if self.separate_models else self.backbone_nca

    def make_seed(self, x_bchw):
        B, C, H, W = x_bchw.shape
        seed = torch.zeros(B, self.channel_n, H, W,
                           dtype=x_bchw.dtype, device=x_bchw.device)
        seed[:, :C] = x_bchw
        return seed


# ─── OctreeNCA dual-view model with FiLM cross-fusion ───────────────────────

class OctreeNCA_DualView(nn.Module):
    """Dual-view OctreeNCA with FiLM cross-fusion between views A and B."""

    def __init__(self, channel_n=24, hidden_size=32, input_channels=1,
                 output_channels=7, n_levels=4, kernel_size=3,
                 cross_strength=0.5, cross_use_tanh=True,
                 separate_models=True):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_levels = n_levels
        self.cross_strength = cross_strength
        self.cross_use_tanh = cross_use_tanh
        self.separate_models = separate_models

        if separate_models:
            self.backbone_ncas = nn.ModuleList([
                BasicNCA2DFast(channel_n, hidden_size, input_channels, kernel_size)
                for _ in range(n_levels)
            ])
        else:
            self.backbone_nca = BasicNCA2DFast(channel_n, hidden_size, input_channels, kernel_size)

        # FiLM cross-fusion MLPs
        hidden_start = input_channels + output_channels
        hidden_dim = max(0, channel_n - hidden_start)
        self._dual_hidden_start = hidden_start
        self._dual_hidden_dim = hidden_dim

        self.cross_film = None
        if hidden_dim > 0:
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

    def make_seed(self, x_bchw):
        B, C, H, W = x_bchw.shape
        seed = torch.zeros(B, self.channel_n, H, W,
                           dtype=x_bchw.dtype, device=x_bchw.device)
        seed[:, :C] = x_bchw
        return seed

    def get_backbone(self, level):
        return self.backbone_ncas[level] if self.separate_models else self.backbone_nca

    def cross_fuse(self, state_a, state_b, level):
        """Apply FiLM cross-fusion between views."""
        if self.cross_film is None or self._dual_hidden_dim <= 0:
            return state_a, state_b

        hs = self._dual_hidden_start
        ha = state_a[:, hs:]
        hb = state_b[:, hs:]

        pooled_a = ha.mean(dim=(2, 3))
        pooled_b = hb.mean(dim=(2, 3))

        params_a = self.cross_film[level](pooled_b)
        params_b = self.cross_film[level](pooled_a)
        scale_a, shift_a = params_a.chunk(2, dim=1)
        scale_b, shift_b = params_b.chunk(2, dim=1)

        strength = self.cross_strength
        if self.cross_use_tanh:
            scale_a = torch.tanh(scale_a) * strength
            shift_a = torch.tanh(shift_a) * strength
            scale_b = torch.tanh(scale_b) * strength
            shift_b = torch.tanh(shift_b) * strength
        else:
            scale_a = scale_a * strength
            shift_a = shift_a * strength
            scale_b = scale_b * strength
            shift_b = shift_b * strength

        ha = ha * (1.0 + scale_a[:, :, None, None]) + shift_a[:, :, None, None]
        hb = hb * (1.0 + scale_b[:, :, None, None]) + shift_b[:, :, None, None]

        state_a = torch.cat([state_a[:, :hs], ha], dim=1)
        state_b = torch.cat([state_b[:, :hs], hb], dim=1)
        return state_a, state_b


# ─── Config auto-detection ──────────────────────────────────────────────────

def find_config_json(checkpoint_path):
    """Walk up from checkpoint to find the experiment config.json."""
    p = Path(checkpoint_path).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        cfg = parent / "config.json"
        if cfg.exists():
            return cfg
    return None


def extract_state_dict(raw_checkpoint):
    """Support plain state_dict or checkpoint wrappers."""
    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state_dict = raw_checkpoint.get(key, None)
            if isinstance(state_dict, dict) and len(state_dict) > 0:
                return state_dict
    if isinstance(raw_checkpoint, dict):
        return raw_checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(raw_checkpoint)}")


def strip_prefix_if_all(state_dict, prefix):
    if len(state_dict) == 0:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def unwrap_submodel_state_dict(state_dict, submodel="auto"):
    """Handle warm-start checkpoints storing weights under m1./m2. prefixes."""
    available = []
    for name in ("m1", "m2"):
        if any(k.startswith(f"{name}.") for k in state_dict.keys()):
            available.append(name)

    if not available:
        return state_dict, None

    if submodel == "auto":
        # For warm-start checkpoints, m1 is the stable per-frame model.
        chosen = "m1" if "m1" in available else available[0]
    else:
        chosen = submodel
        if chosen not in available:
            raise ValueError(
                f"Requested --submodel {submodel}, but checkpoint has: {available}"
            )

    prefix = f"{chosen}."
    unwrapped = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    return unwrapped, chosen


def detect_dual_view(state_dict):
    """Check if checkpoint contains cross_film keys -> dual-view model."""
    return any(k.startswith("cross_film.") or ".cross_film." in k for k in state_dict.keys())


def detect_separate_models(state_dict):
    """Check if checkpoint uses separate per-level backbones (backbone_ncas) or a single shared one (backbone_nca)."""
    return any(k.startswith("backbone_ncas.") or ".backbone_ncas." in k for k in state_dict.keys())


def load_config_and_model(checkpoint_path, device, submodel="auto"):
    """Auto-detect model type and hyper-parameters, load weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)

    # Strip _orig_mod. prefix from torch.compile() checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('._orig_mod.', '.')] = v
    cleaned = strip_prefix_if_all(cleaned, "module.")
    cleaned = strip_prefix_if_all(cleaned, "model.")
    cleaned, selected_submodel = unwrap_submodel_state_dict(cleaned, submodel=submodel)
    if selected_submodel is not None:
        print(f"Using submodel: {selected_submodel}")

    is_dual = detect_dual_view(cleaned)
    separate_models = detect_separate_models(cleaned)

    # Try to load config.json
    cfg_path = find_config_json(checkpoint_path)
    if cfg_path:
        print(f"Config: {cfg_path}")
        with open(cfg_path) as f:
            config = json.load(f)
    else:
        config = {}
        print("Warning: No config.json found, using defaults.")

    # Extract model parameters from config (with sensible defaults)
    channel_n = int(config.get("model.channel_n", 24))
    hidden_size = int(config.get("model.hidden_size", 64))
    input_channels = int(config.get("model.input_channels", 1))
    output_channels = int(config.get("model.output_channels", 7))

    # Kernel sizes: config may have a list (one per level) or single int
    ks_raw = config.get("model.kernel_size", 3)
    if isinstance(ks_raw, list):
        kernel_size = int(ks_raw[0])  # assume uniform for backbone init
    else:
        kernel_size = int(ks_raw)

    # Octree levels and steps from res_and_steps
    res_and_steps = config.get("model.octree.res_and_steps", None)
    if res_and_steps:
        n_levels = len(res_and_steps)
        # res_and_steps is ordered finest->coarsest: [[512,512],10], [[256,256],10], ...
        # steps_per_level[i] corresponds to level i (index 0 = finest)
        # Steps can be int (fixed) or [min, max] (random range from training).
        # For visualization we default to the max of the range.
        def _resolve_steps(s):
            if isinstance(s, (list, tuple)):
                return int(max(s))
            return int(s)
        steps_per_level = [_resolve_steps(entry[1]) for entry in res_and_steps]
    else:
        # Fallback: count backbone_ncas in checkpoint
        nca_indices = set()
        for k in cleaned:
            if k.startswith("backbone_ncas."):
                idx = int(k.split('.')[1])
                nca_indices.add(idx)
        if nca_indices:
            n_levels = max(nca_indices) + 1
        else:
            film_indices = set()
            for k in cleaned:
                if k.startswith("cross_film."):
                    film_indices.add(int(k.split('.')[1]))
            n_levels = max(film_indices) + 1 if film_indices else 4
        steps_per_level = [10] * (n_levels - 1) + [20]

    # Cross-fusion params (dual-view only)
    cross_strength = float(config.get("model.dual_view.cross_strength", 0.5))
    cross_use_tanh = bool(config.get("model.dual_view.cross_use_tanh", True))

    experiment_name = config.get("experiment.name", "OctreeNCA")
    fire_rate = float(config.get("model.fire_rate", 0.5))

    # Input size from config (used for loading images at the right resolution)
    input_size_raw = config.get("experiment.dataset.input_size", [512, 512])
    input_size = tuple(int(x) for x in input_size_raw)

    print(f"Detected: {'Dual-view (FiLM)' if is_dual else 'Single-view'} | "
          f"{'separate' if separate_models else 'shared'} backbone | "
          f"{n_levels} levels | channel_n={channel_n} | hidden={hidden_size} | "
          f"kernel={kernel_size} | out_ch={output_channels} | fire_rate={fire_rate}")
    print(f"Steps per level (finest->coarsest): {steps_per_level}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")

    if is_dual:
        model = OctreeNCA_DualView(
            channel_n=channel_n, hidden_size=hidden_size,
            input_channels=input_channels, output_channels=output_channels,
            n_levels=n_levels, kernel_size=kernel_size,
            cross_strength=cross_strength, cross_use_tanh=cross_use_tanh,
            separate_models=separate_models,
        ).to(device)
    else:
        model = OctreeNCA_SeparateModels(
            channel_n=channel_n, hidden_size=hidden_size,
            input_channels=input_channels, output_channels=output_channels,
            n_levels=n_levels, kernel_size=kernel_size,
            separate_models=separate_models,
        ).to(device)

    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as err:
        print(f"Strict checkpoint load failed: {err}")
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded with strict=False. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  First missing keys: {missing[:8]}")
        if unexpected:
            print(f"  First unexpected keys: {unexpected[:8]}")
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    return model, steps_per_level, is_dual, experiment_name, fire_rate, input_size


# ─── Octree helpers ─────────────────────────────────────────────────────────

def build_octree_levels(seed_bchw, n_levels):
    """Downsample seed into n_levels. Returns [level0=finest, ..., levelN-1=coarsest].
    Uses nearest-neighbor interpolation to match training code (F.interpolate default)."""
    levels = [seed_bchw]
    x = seed_bchw
    for i in range(1, n_levels):
        h, w = seed_bchw.shape[2] // (2 ** i), seed_bchw.shape[3] // (2 ** i)
        x = F.interpolate(seed_bchw, size=(h, w), mode='nearest')
        levels.append(x)
    return levels


def upscale_state(state_bchw, input_channels):
    """Upscale hidden (non-input) channels by 2x nearest, return only hidden part."""
    hidden = state_bchw[:, input_channels:]
    return F.interpolate(hidden, scale_factor=2, mode='nearest')


# ─── Visualization helpers ──────────────────────────────────────────────────

CLASS_COLORS = np.array([
    [0, 0, 0],       # 0 Background
    [255, 0, 0],     # 1 Red
    [0, 255, 209],   # 2 Cyan
    [61, 255, 0],    # 3 Green
    [0, 78, 255],    # 4 Blue
    [255, 189, 0],   # 5 Yellow
    [218, 0, 255],   # 6 Magenta
    [255, 127, 80],  # 7 Coral
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


def state_to_input_rgb(state_bchw, input_channels):
    """Extract input image from state, apply contrast stretching, return RGB (H, W, 3).
    iOCT images are often very dark; min-max normalization makes them visible."""
    inp = state_bchw[0, :input_channels].mean(dim=0).cpu().numpy()
    lo, hi = inp.min(), inp.max()
    if hi - lo > 1e-6:
        inp = (inp - lo) / (hi - lo)  # stretch to [0, 1]
    inp = np.clip(inp * 255, 0, 255).astype(np.uint8)
    return np.stack([inp, inp, inp], axis=-1)


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


def create_frame_single(level_states, active_level, global_step, model_name,
                        input_channels, output_channels, tile_size=512,
                        gt_rgb=None):
    """Render one frame for single-view: GT + input image + levels coarsest->finest left-to-right."""
    n_levels = len(level_states)
    header_h = 50
    footer_h = 30
    padding = 4
    # Extra columns for GT and input image
    n_cols = n_levels + 2  # +1 GT, +1 input
    canvas_w = tile_size * n_cols + padding * (n_cols + 1)
    canvas_h = header_h + tile_size + footer_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font_title = try_load_font(22)
    font_small = try_load_font(14)
    font_label = try_load_font(12)

    draw.text((12, 12), f"Step {global_step}", fill='white', font=font_title)
    name_bbox = draw.textbbox((0, 0), model_name, font=font_small)
    draw.text((canvas_w - (name_bbox[2] - name_bbox[0]) - 12, 14),
              model_name, fill=(180, 180, 180), font=font_small)

    col = 0

    # Column 0: ground truth
    x_pos = padding + col * (tile_size + padding)
    y_pos = header_h
    if gt_rgb is not None:
        gt_pil = Image.fromarray(gt_rgb).resize((tile_size, tile_size), Image.NEAREST)
        canvas.paste(gt_pil, (x_pos, y_pos))
    else:
        # blank placeholder
        draw.rectangle([x_pos, y_pos, x_pos + tile_size - 1, y_pos + tile_size - 1],
                       fill=(50, 50, 50))
    label = "GT"
    lb = draw.textbbox((0, 0), label, font=font_label)
    draw.text((x_pos + tile_size // 2 - (lb[2] - lb[0]) // 2, y_pos + tile_size + 4),
              label, fill='white', font=font_label)
    col += 1

    # Column 1: input image (from finest level)
    inp_rgb = state_to_input_rgb(level_states[0], input_channels)
    inp_pil = Image.fromarray(inp_rgb).resize((tile_size, tile_size), Image.BILINEAR)
    x_pos = padding + col * (tile_size + padding)
    y_pos = header_h
    canvas.paste(inp_pil, (x_pos, y_pos))
    label = "Input"
    lb = draw.textbbox((0, 0), label, font=font_label)
    draw.text((x_pos + tile_size // 2 - (lb[2] - lb[0]) // 2, y_pos + tile_size + 4),
              label, fill='white', font=font_label)
    col += 1

    # Columns 2..n_levels+1: segmentation stages coarsest->finest
    for display_idx in range(n_levels):
        level_idx = n_levels - 1 - display_idx
        state = level_states[level_idx]
        h_res, w_res = state.shape[2], state.shape[3]
        rgb = state_to_segmentation_rgb(state, input_channels, output_channels)
        img_pil = Image.fromarray(rgb).resize((tile_size, tile_size), Image.NEAREST)

        x_pos = padding + (col + display_idx) * (tile_size + padding)
        y_pos = header_h
        canvas.paste(img_pil, (x_pos, y_pos))

        if level_idx == active_level:
            for t in range(3):
                draw.rectangle([x_pos - t, y_pos - t,
                                x_pos + tile_size + t - 1, y_pos + tile_size + t - 1],
                               outline='red')

        label = f"{h_res}x{w_res}"
        lb = draw.textbbox((0, 0), label, font=font_label)
        draw.text((x_pos + tile_size // 2 - (lb[2] - lb[0]) // 2, y_pos + tile_size + 4),
                  label, fill='white', font=font_label)

    return canvas


def create_frame_dual(levels_a, levels_b, active_level, global_step, model_name,
                      input_channels, output_channels, tile_size=512,
                      gt_a=None, gt_b=None):
    """Render one frame for dual-view: two rows (View A top, View B bottom),
    GT + input image + levels coarsest->finest left-to-right."""
    n_levels = len(levels_a)
    header_h = 50
    row_label_w = 60
    footer_h = 30
    padding = 4
    # Extra columns for GT and input image
    n_cols = n_levels + 2  # +1 GT, +1 input
    canvas_w = row_label_w + tile_size * n_cols + padding * (n_cols + 1)
    canvas_h = header_h + tile_size * 2 + padding * 3 + footer_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font_title = try_load_font(22)
    font_small = try_load_font(14)
    font_label = try_load_font(12)
    font_row = try_load_font(16)

    draw.text((12, 12), f"Step {global_step}", fill='white', font=font_title)
    name_bbox = draw.textbbox((0, 0), model_name, font=font_small)
    draw.text((canvas_w - (name_bbox[2] - name_bbox[0]) - 12, 14),
              model_name, fill=(180, 180, 180), font=font_small)

    gt_views = [gt_a, gt_b]
    for row_idx, (levels, view_label) in enumerate([(levels_a, "View A"), (levels_b, "View B")]):
        y_row = header_h + row_idx * (tile_size + padding)

        # Row label
        rl_bbox = draw.textbbox((0, 0), view_label, font=font_row)
        rl_h = rl_bbox[3] - rl_bbox[1]
        draw.text((8, y_row + tile_size // 2 - rl_h // 2),
                  view_label, fill=(200, 200, 255), font=font_row)

        col = 0

        # Column 0: ground truth
        x_pos = row_label_w + padding + col * (tile_size + padding)
        y_pos = y_row
        gt_cur = gt_views[row_idx]
        if gt_cur is not None:
            gt_pil = Image.fromarray(gt_cur).resize((tile_size, tile_size), Image.NEAREST)
            canvas.paste(gt_pil, (x_pos, y_pos))
        else:
            draw.rectangle([x_pos, y_pos, x_pos + tile_size - 1, y_pos + tile_size - 1],
                           fill=(50, 50, 50))
        if row_idx == 1:
            label = "GT"
            lb = draw.textbbox((0, 0), label, font=font_label)
            draw.text((x_pos + tile_size // 2 - (lb[2] - lb[0]) // 2,
                       y_pos + tile_size + 4),
                      label, fill='white', font=font_label)
        col += 1

        # Column 1: input image (from finest level)
        inp_rgb = state_to_input_rgb(levels[0], input_channels)
        inp_pil = Image.fromarray(inp_rgb).resize((tile_size, tile_size), Image.BILINEAR)
        x_pos = row_label_w + padding + col * (tile_size + padding)
        y_pos = y_row
        canvas.paste(inp_pil, (x_pos, y_pos))
        if row_idx == 1:
            label = "Input"
            lb = draw.textbbox((0, 0), label, font=font_label)
            draw.text((x_pos + tile_size // 2 - (lb[2] - lb[0]) // 2,
                       y_pos + tile_size + 4),
                      label, fill='white', font=font_label)
        col += 1

        # Columns 2..n_levels+1: segmentation stages coarsest->finest
        for display_idx in range(n_levels):
            level_idx = n_levels - 1 - display_idx
            state = levels[level_idx]
            h_res, w_res = state.shape[2], state.shape[3]
            rgb = state_to_segmentation_rgb(state, input_channels, output_channels)
            img_pil = Image.fromarray(rgb).resize((tile_size, tile_size), Image.NEAREST)

            x_pos = row_label_w + padding + (col + display_idx) * (tile_size + padding)
            y_pos = y_row
            canvas.paste(img_pil, (x_pos, y_pos))

            if level_idx == active_level:
                for t in range(3):
                    draw.rectangle([x_pos - t, y_pos - t,
                                    x_pos + tile_size + t - 1, y_pos + tile_size + t - 1],
                                   outline='red')

            # Resolution labels only on the bottom row
            if row_idx == 1:
                label = f"{h_res}x{w_res}"
                lb = draw.textbbox((0, 0), label, font=font_label)
                draw.text((x_pos + tile_size // 2 - (lb[2] - lb[0]) // 2,
                           y_pos + tile_size + 4),
                          label, fill='white', font=font_label)

    return canvas


# ─── Data loading ───────────────────────────────────────────────────────────

def load_sample_image(data_root, idx=50, input_size=(512, 512), view="A"):
    """Load a single iOCT image and its ground truth segmentation for the given view.
    Returns (image_tensor, gt_rgb) where gt_rgb is an (H, W, 3) numpy array."""
    data_root = Path(data_root)
    img_paths = []
    seg_paths = []
    for ds in ["peeling", "sri"]:
        d_img = data_root / ds / "Bscans-dt" / view / "Image"
        d_seg = data_root / ds / "Bscans-dt" / view / "Segmentation"
        if d_img.exists():
            sorted_imgs = sorted(d_img.glob("*.png"))
            img_paths.extend(sorted_imgs)
            if d_seg.exists():
                sorted_segs = sorted(d_seg.glob("*.png"))
                seg_paths.extend(sorted_segs)
            else:
                seg_paths.extend([None] * len(sorted_imgs))
    if not img_paths:
        raise FileNotFoundError(f"No images found for view {view} under {data_root}")
    idx = min(idx, len(img_paths) - 1)
    print(f"Loading {view}: {img_paths[idx]}")
    img = np.array(Image.open(img_paths[idx]))
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    img = np.array(Image.fromarray(img).resize(
        (input_size[1], input_size[0]), Image.BILINEAR))
    img = img.astype(np.float32) / 255.0

    # Load ground truth segmentation mask
    gt_rgb = None
    if idx < len(seg_paths) and seg_paths[idx] is not None:
        seg = Image.open(seg_paths[idx]).convert('RGB')
        seg = seg.resize((input_size[1], input_size[0]), Image.NEAREST)
        gt_rgb = np.array(seg)

    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0), gt_rgb


def load_paired_images(data_root, idx=50, input_size=(512, 512)):
    """Load paired A + B view images with ground truth."""
    img_a, gt_a = load_sample_image(data_root, idx, input_size, view="A")
    img_b, gt_b = load_sample_image(data_root, idx, input_size, view="B")
    return img_a, img_b, gt_a, gt_b


# ─── Inference runners ──────────────────────────────────────────────────────

def run_single_view_inference(model, img, steps_per_level, tile_size, model_name,
                              fire_rate=0.5, gt_rgb=None):
    """Run single-view inference, collecting frames at each step."""
    seed = model.make_seed(img)
    n_levels = model.n_levels
    levels = build_octree_levels(seed, n_levels)

    frames = []
    global_step = 0

    with torch.no_grad():
        for level_idx in range(n_levels - 1, -1, -1):
            n_steps = steps_per_level[level_idx]
            nca = model.get_backbone(level_idx)
            x = levels[level_idx]
            print(f"  Level {level_idx} ({x.shape[2]}x{x.shape[3]}): {n_steps} steps")

            for step in range(n_steps):
                frame = create_frame_single(
                    levels, active_level=level_idx, global_step=global_step,
                    model_name=model_name,
                    input_channels=model.input_channels,
                    output_channels=model.output_channels,
                    tile_size=tile_size,
                    gt_rgb=gt_rgb,
                )
                frames.append(np.array(frame))
                x = nca.forward_bchw(x, fire_rate=fire_rate)
                levels[level_idx] = x
                global_step += 1

            if level_idx > 0:
                hidden_up = upscale_state(x, model.input_channels)
                finer = levels[level_idx - 1]
                levels[level_idx - 1] = torch.cat([
                    finer[:, :model.input_channels], hidden_up
                ], dim=1)

        # Final frame
        frames.append(np.array(create_frame_single(
            levels, active_level=0, global_step=global_step,
            model_name=model_name,
            input_channels=model.input_channels,
            output_channels=model.output_channels,
            tile_size=tile_size,
            gt_rgb=gt_rgb,
        )))

    return frames


def run_dual_view_inference(model, img_a, img_b, steps_per_level, tile_size, model_name,
                            fire_rate=0.5, gt_a=None, gt_b=None):
    """Run dual-view inference with FiLM cross-fusion, collecting frames."""
    seed_a = model.make_seed(img_a)
    seed_b = model.make_seed(img_b)
    n_levels = model.n_levels
    levels_a = build_octree_levels(seed_a, n_levels)
    levels_b = build_octree_levels(seed_b, n_levels)

    frames = []
    global_step = 0

    with torch.no_grad():
        for level_idx in range(n_levels - 1, -1, -1):
            n_steps = steps_per_level[level_idx]
            nca = model.get_backbone(level_idx)
            xa = levels_a[level_idx]
            xb = levels_b[level_idx]
            print(f"  Level {level_idx} ({xa.shape[2]}x{xa.shape[3]}): {n_steps} steps")

            for step in range(n_steps):
                frame = create_frame_dual(
                    levels_a, levels_b, active_level=level_idx,
                    global_step=global_step, model_name=model_name,
                    input_channels=model.input_channels,
                    output_channels=model.output_channels,
                    tile_size=tile_size,
                    gt_a=gt_a, gt_b=gt_b,
                )
                frames.append(np.array(frame))

                # Shared backbone: concatenate both views for a single forward
                xab = torch.cat([xa, xb], dim=0)
                xab = nca.forward_bchw(xab, fire_rate=fire_rate)
                xa, xb = xab.chunk(2, dim=0)

                levels_a[level_idx] = xa
                levels_b[level_idx] = xb
                global_step += 1

            # FiLM cross-fusion ONCE per level (after all NCA steps),
            # matching the training code's forward_train flow.
            xa, xb = model.cross_fuse(xa, xb, level_idx)
            levels_a[level_idx] = xa
            levels_b[level_idx] = xb

            if level_idx > 0:
                inp_ch = model.input_channels
                hidden_up_a = upscale_state(xa, inp_ch)
                hidden_up_b = upscale_state(xb, inp_ch)
                finer_a = levels_a[level_idx - 1]
                finer_b = levels_b[level_idx - 1]
                levels_a[level_idx - 1] = torch.cat([
                    finer_a[:, :inp_ch], hidden_up_a
                ], dim=1)
                levels_b[level_idx - 1] = torch.cat([
                    finer_b[:, :inp_ch], hidden_up_b
                ], dim=1)

        # Final frame
        frames.append(np.array(create_frame_dual(
            levels_a, levels_b, active_level=0, global_step=global_step,
            model_name=model_name,
            input_channels=model.input_channels,
            output_channels=model.output_channels,
            tile_size=tile_size,
            gt_a=gt_a, gt_b=gt_b,
        )))

    return frames


# ─── Main ───────────────────────────────────────────────────────────────────

def run_visualization(checkpoint_path, output_path, sample_idx=50, fps=8,
                      data_root="/vol/data/OctreeNCA_Video/ioct_data",
                      tile_size=512, steps_override=None, submodel="auto"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Auto-detect and load model
    model, steps_per_level, is_dual, experiment_name, fire_rate, input_size = load_config_and_model(
        checkpoint_path, device, submodel=submodel)

    # Override steps per level if provided
    if steps_override is not None:
        if len(steps_override) != len(steps_per_level):
            print(f"Note: --steps has {len(steps_override)} values, config has {len(steps_per_level)} levels. "
                  f"Using {len(steps_override)} levels from --steps.")
        print(f"Overriding steps per level: {steps_per_level} -> {steps_override}")
        steps_per_level = steps_override

    model_name = experiment_name if len(experiment_name) <= 30 else experiment_name[:27] + "..."

    # Compute resolutions automatically from input_size and number of levels
    n_levels = len(steps_per_level)
    print(f"\nOctree levels ({n_levels}):")
    h, w = input_size
    for i in range(n_levels):
        lvl_h = h // (2 ** i)
        lvl_w = w // (2 ** i)
        tag = 'finest' if i == 0 else ('coarsest' if i == n_levels - 1 else '')
        print(f"  Level {i}: {lvl_h}x{lvl_w}  steps={steps_per_level[i]}  ({tag})")

    # Load data and run inference
    if is_dual:
        print(f"\nDual-view mode: loading paired A+B images")
        img_a, img_b, gt_a, gt_b = load_paired_images(data_root, sample_idx, input_size=input_size)
        img_a, img_b = img_a.to(device), img_b.to(device)
        print(f"Input shapes: A={tuple(img_a.shape)}, B={tuple(img_b.shape)}")
        frames = run_dual_view_inference(model, img_a, img_b, steps_per_level,
                                         tile_size, model_name,
                                         fire_rate=fire_rate,
                                         gt_a=gt_a, gt_b=gt_b)
    else:
        print(f"\nSingle-view mode")
        img, gt_rgb = load_sample_image(data_root, sample_idx, input_size=input_size)
        img = img.to(device)
        print(f"Input shape: {tuple(img.shape)}")
        frames = run_single_view_inference(model, img, steps_per_level,
                                           tile_size, model_name,
                                           fire_rate=fire_rate,
                                           gt_rgb=gt_rgb)

    # Save video
    output_path = str(output_path)
    if output_path.endswith('.gif'):
        print(f"Saving GIF ({len(frames)} frames) ...")
        imageio.mimsave(output_path, frames, duration=1000 // fps, loop=0)
    else:
        print(f"Saving MP4 ({len(frames)} frames) ...")
        try:
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                        quality=8, format='FFMPEG',
                                        output_params=['-pix_fmt', 'yuv420p'])
            for f in frames:
                writer.append_data(f)
            writer.close()
        except Exception as e:
            print(f"FFMPEG writer failed ({e}), falling back to GIF...")
            gif_path = output_path.rsplit('.', 1)[0] + '.gif'
            imageio.mimsave(gif_path, frames, duration=1000 // fps, loop=0)
            output_path = gif_path

    print(f"Saved to {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OctreeNCA inference stages")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model.pth or ema.pth")
    parser.add_argument("--output", type=str, default="octree_inference.mp4",
                        help="Output path (.mp4 or .gif)")
    parser.add_argument("--sample", type=int, default=50,
                        help="Image index to use")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second")
    parser.add_argument("--data-root", type=str,
                        default="/vol/data/OctreeNCA_Video/ioct_data",
                        help="iOCT data root")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile size for each stage in pixels")
    parser.add_argument("--steps", type=str, default=None,
                        help="Override steps per level (finest->coarsest), comma-separated. "
                             "E.g. '10,12,16' for 3 levels. The number of values "
                             "determines how many downscaling levels are used. "
                             "If not set, uses max of the range from training config.")
    parser.add_argument("--submodel", type=str, default="auto", choices=["auto", "m1", "m2"],
                        help="For warm-start checkpoints with nested keys (m1./m2.), "
                             "select which submodel to visualize.")
    args = parser.parse_args()

    # Parse --steps into list of ints
    steps_override = None
    if args.steps is not None:
        steps_override = [int(s.strip()) for s in args.steps.split(',')]

    run_visualization(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        sample_idx=args.sample,
        fps=args.fps,
        data_root=args.data_root,
        tile_size=args.tile_size,
        steps_override=steps_override,
        submodel=args.submodel,
    )
