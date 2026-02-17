"""
Visualize OctreeNCA inference: all 5 resolution stages side-by-side,
stepping from coarsest to finest, saved as MP4 video.

Usage:
    python visualize_octree_video.py \
        --checkpoint /vol/data/OctreeNCA_Video/.../model.pth \
        --output octree_inference.mp4 \
        --sample 50 --fps 8
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import imageio

# ─── Minimal reimplementation of BasicNCA2DFast (matches checkpoint) ─────────

class BasicNCA2DFast(nn.Module):
    """Single-level NCA backbone matching the trained checkpoint structure.
    All ops in BCHW. Perception via depthwise conv, update via 1x1 convs."""

    def __init__(self, channel_n=24, hidden_size=32, input_channels=1, kernel_size=5):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        # Perception: depthwise conv
        self.conv = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=channel_n,
                              padding_mode='reflect')
        # Update: 1x1 convs  (input = identity + conv = 2*channel_n)
        self.fc0 = nn.Conv2d(channel_n * 2, hidden_size, 1)
        self.fc1 = nn.Conv2d(hidden_size, channel_n - input_channels, 1, bias=False)
        nn.init.zeros_(self.fc1.weight)

    def forward(self, x, steps=1, fire_rate=1.0):
        """x: BHWC tensor. Returns BHWC tensor."""
        x = x.permute(0, 3, 1, 2)  # -> BCHW
        for _ in range(steps):
            x = self._step(x, fire_rate)
        return x.permute(0, 2, 3, 1)  # -> BHWC

    def forward_bchw(self, x_bchw, fire_rate=1.0):
        """Single step on BCHW tensor, returns BCHW. Used for per-step viz."""
        return self._step(x_bchw, fire_rate)

    def _step(self, x, fire_rate):
        # Perceive
        y = self.conv(x)
        y = torch.cat([x, y], dim=1)  # (B, 2*C, H, W)
        # Update
        dx = F.relu(self.fc0(y))
        dx = self.fc1(dx)  # (B, C - input_channels, H, W)
        # Stochastic update
        if fire_rate < 1.0:
            mask = (torch.rand(dx.shape[0], 1, dx.shape[2], dx.shape[3],
                               device=x.device) < fire_rate).float()
            dx = dx * mask
        # Residual — only update non-input channels
        x_inp = x[:, :self.input_channels]
        x_state = x[:, self.input_channels:] + dx
        return torch.cat([x_inp, x_state], dim=1)


# ─── OctreeNCA with separate models (matches checkpoint) ────────────────────

class OctreeNCA_SeparateModels(nn.Module):
    def __init__(self, channel_n=24, hidden_size=32, input_channels=1,
                 output_channels=6, n_levels=5, kernel_size=5):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_levels = n_levels
        self.backbone_ncas = nn.ModuleList([
            BasicNCA2DFast(channel_n, hidden_size, input_channels, kernel_size)
            for _ in range(n_levels)
        ])

    def make_seed(self, x_bchw):
        """x_bchw: (B, input_channels, H, W) -> (B, channel_n, H, W)"""
        B, C, H, W = x_bchw.shape
        seed = torch.zeros(B, self.channel_n, H, W,
                           dtype=x_bchw.dtype, device=x_bchw.device)
        seed[:, :C] = x_bchw
        return seed


# ─── Octree helper ──────────────────────────────────────────────────────────

def build_octree_levels(seed_bchw, n_levels=5):
    """Downsample seed into n_levels. Returns list [level0=full, ..., levelN=coarsest] in BCHW."""
    levels = [seed_bchw]
    x = seed_bchw
    for _ in range(n_levels - 1):
        x = F.avg_pool2d(x, 2)
        levels.append(x)
    return levels  # index 0 = finest, index -1 = coarsest


def upscale_hidden_states(state_bchw, input_channels):
    """Upscale hidden (non-input) channels by 2x nearest."""
    inp = state_bchw[:, :input_channels]
    hidden = state_bchw[:, input_channels:]
    hidden_up = F.interpolate(hidden, scale_factor=2, mode='nearest')
    return hidden_up  # only hidden part, to be added to finer level


# ─── Visualization helpers ──────────────────────────────────────────────────

CLASS_COLORS = np.array([
    [0, 0, 0],       # 0 Background
    [255, 0, 0],     # 1 Red
    [0, 255, 209],   # 2 Cyan
    [61, 255, 0],    # 3 Green
    [0, 78, 255],    # 4 Blue
    [255, 189, 0],   # 5 Yellow
], dtype=np.uint8)


def state_to_segmentation_rgb(state_bchw, input_channels, output_channels):
    """Extract segmentation output from state, return RGB (H, W, 3)."""
    logits = state_bchw[0, input_channels:input_channels + output_channels]  # (C, H, W)
    class_map = logits.argmax(dim=0).cpu().numpy()  # (H, W)
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


def create_frame(level_states_bchw, active_level, global_step, model_name,
                 input_channels, output_channels, tile_size=200, show_input=True):
    """
    Render one frame: all 5 stages left-to-right (coarsest on left, finest on right).
    Active level gets a red border. Shows segmentation output.
    Optionally shows a small input thumbnail.
    """
    n_levels = len(level_states_bchw)
    header_h = 50
    footer_h = 30
    padding = 4
    canvas_w = tile_size * n_levels + padding * (n_levels + 1)
    canvas_h = header_h + tile_size + footer_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    font_title = try_load_font(22)
    font_small = try_load_font(14)
    font_label = try_load_font(12)

    # Header
    draw.text((12, 12), f"Step {global_step}", fill='white', font=font_title)
    # Model name upper-right
    name_bbox = draw.textbbox((0, 0), model_name, font=font_small)
    name_w = name_bbox[2] - name_bbox[0]
    draw.text((canvas_w - name_w - 12, 14), model_name, fill=(180, 180, 180), font=font_small)

    # Draw levels: left = coarsest, right = finest
    # level_states_bchw[0] = finest (512), [-1] = coarsest (32)
    # Display order: coarsest first (left) -> finest (right)
    for display_idx in range(n_levels):
        level_idx = n_levels - 1 - display_idx  # map display position to level index
        state = level_states_bchw[level_idx]
        h_res, w_res = state.shape[2], state.shape[3]

        rgb = state_to_segmentation_rgb(state, input_channels, output_channels)
        img_pil = Image.fromarray(rgb).resize((tile_size, tile_size), Image.NEAREST)

        x_pos = padding + display_idx * (tile_size + padding)
        y_pos = header_h

        canvas.paste(img_pil, (x_pos, y_pos))

        # Red border for active level
        if level_idx == active_level:
            for t in range(3):
                draw.rectangle(
                    [x_pos - t, y_pos - t,
                     x_pos + tile_size + t - 1, y_pos + tile_size + t - 1],
                    outline='red')

        # Resolution label below tile
        label = f"{h_res}×{w_res}"
        lb = draw.textbbox((0, 0), label, font=font_label)
        lw = lb[2] - lb[0]
        draw.text((x_pos + tile_size // 2 - lw // 2, y_pos + tile_size + 4),
                  label, fill='white', font=font_label)

    return canvas


# ─── Main ───────────────────────────────────────────────────────────────────

def load_sample_image(data_root, idx=50, input_size=(512, 512)):
    """Load a single iOCT image without importing the dataset class."""
    data_root = Path(data_root)
    img_paths = []
    for ds in ["peeling", "sri"]:
        for view in ["A", "B"]:
            d = data_root / ds / "Bscans-dt" / view / "Image"
            if d.exists():
                img_paths.extend(sorted(d.glob("*.png")))
    if not img_paths:
        raise FileNotFoundError(f"No images found under {data_root}")
    idx = min(idx, len(img_paths) - 1)
    print(f"Loading image: {img_paths[idx]}")
    img = np.array(Image.open(img_paths[idx]))
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    img = np.array(Image.fromarray(img).resize(
        (input_size[1], input_size[0]), Image.BILINEAR))
    img = img.astype(np.float32) / 255.0
    # -> (1, 1, H, W) tensor
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def run_visualization(checkpoint_path, output_path, sample_idx=50, fps=8,
                      data_root="/vol/data/OctreeNCA_Video/ioct_data",
                      tile_size=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load model ---
    model = OctreeNCA_SeparateModels(
        channel_n=24, hidden_size=32, input_channels=1,
        output_channels=6, n_levels=5, kernel_size=5
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # --- Load image ---
    img = load_sample_image(data_root, sample_idx).to(device)
    print(f"Input shape: {img.shape}")

    # --- Build octree ---
    seed = model.make_seed(img)  # (1, 24, 512, 512)
    levels = build_octree_levels(seed, n_levels=5)
    # levels[0]=512, [1]=256, [2]=128, [3]=64, [4]=32

    steps_per_level = [10, 10, 10, 10, 20]  # from train config: steps=10, final=alpha*20=20
    # Inference order: coarsest (level 4) first -> finest (level 0)
    # This matches: for level in range(n_levels-1, -1, -1)

    model_name = Path(checkpoint_path).parts[-4] if len(Path(checkpoint_path).parts) > 4 else "OctreeNCA"
    # Truncate long names
    if len(model_name) > 30:
        model_name = "OctreeNCA_iOCT2D"

    print(f"Octree levels:")
    for i, lv in enumerate(levels):
        print(f"  Level {i}: {lv.shape[2]}×{lv.shape[3]}  "
              f"({'coarsest' if i == len(levels)-1 else 'finest' if i == 0 else ''})")

    frames = []
    global_step = 0

    with torch.no_grad():
        # Process coarsest to finest
        for level_idx in range(len(levels) - 1, -1, -1):
            n_steps = steps_per_level[level_idx]
            nca = model.backbone_ncas[level_idx]
            x = levels[level_idx]  # BCHW

            print(f"Level {level_idx} ({x.shape[2]}×{x.shape[3]}): {n_steps} steps")

            for step in range(n_steps):
                # Capture frame BEFORE update
                frame = create_frame(
                    levels, active_level=level_idx,
                    global_step=global_step,
                    model_name=model_name,
                    input_channels=model.input_channels,
                    output_channels=model.output_channels,
                    tile_size=tile_size,
                )
                frames.append(np.array(frame))

                # Single NCA step (BCHW)
                x = nca.forward_bchw(x, fire_rate=1.0)
                levels[level_idx] = x
                global_step += 1

            # After all steps at this level, upscale hidden states to finer level
            if level_idx > 0:
                hidden_up = upscale_hidden_states(x, model.input_channels)
                finer = levels[level_idx - 1]
                # Replace hidden channels of finer level with upscaled hidden
                levels[level_idx - 1] = torch.cat([
                    finer[:, :model.input_channels],
                    hidden_up
                ], dim=1)

        # Capture final frame
        frames.append(np.array(create_frame(
            levels, active_level=0, global_step=global_step,
            model_name=model_name,
            input_channels=model.input_channels,
            output_channels=model.output_channels,
            tile_size=tile_size,
        )))

    # --- Save video ---
    output_path = str(output_path)
    if output_path.endswith('.gif'):
        print(f"Saving GIF ({len(frames)} frames) ...")
        imageio.mimsave(output_path, frames, duration=1000 // fps, loop=0)
    else:
        print(f"Saving MP4 ({len(frames)} frames) ...")
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                    pixelmt='yuv420p', quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()

    print(f"Saved to {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OctreeNCA inference stages")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model.pth")
    parser.add_argument("--output", type=str, default="octree_inference.mp4",
                        help="Output path (.mp4 or .gif)")
    parser.add_argument("--sample", type=int, default=50,
                        help="Image index to use")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second")
    parser.add_argument("--data-root", type=str,
                        default="/vol/data/OctreeNCA_Video/ioct_data",
                        help="iOCT data root")
    parser.add_argument("--tile-size", type=int, default=200,
                        help="Tile size for each stage in pixels")
    args = parser.parse_args()

    run_visualization(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        sample_idx=args.sample,
        fps=args.fps,
        data_root=args.data_root,
        tile_size=args.tile_size,
    )
