#!/usr/bin/env python3
"""
profile_m2_single_step.py — Inference profiling for M1+M2 single-step model
═══════════════════════════════════════════════════════════════════════════════

Profiles the inference latency and throughput of the keyframe-M1 + single-step-M2
model, breaking down timing for:

  1. **M1 keyframe** forward pass (coarse-to-fine octree NCA)
  2. **M2 single-step** forward pass (warm-start NCA at full resolution)
  3. **End-to-end per-frame** average (amortized M1 cost over keyframe interval)

Reports GPU VRAM usage, per-component timings, and optionally exports a
PyTorch profiler chrome trace for kernel-level analysis.

Usage:
    # Basic profiling (GPU, 50 runs)
    python refactored/profile_m2_single_step.py \\
        --checkpoint "/path/to/epoch_21/model.pth"

    # CPU profiling
    python refactored/profile_m2_single_step.py \\
        --checkpoint "/path/to/model.pth" --device cpu

    # With kernel profiling and torch.compile
    python refactored/profile_m2_single_step.py \\
        --checkpoint "/path/to/model.pth" \\
        --compile-mode max-autotune \\
        --profile-kernels

    # Override keyframe interval
    python refactored/profile_m2_single_step.py \\
        --checkpoint "/path/to/model.pth" \\
        --keyframe-interval 20

    # Sweep over multiple input resolutions
    python refactored/profile_m2_single_step.py \\
        --checkpoint "/path/to/model.pth" \\
        --resolutions 256 512

    # Save results to JSON
    python refactored/profile_m2_single_step.py \\
        --checkpoint "/path/to/model.pth" \\
        --output profiling_results.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ── Project root ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint + model loading (reused from visualize_m2_video.py)
# ═══════════════════════════════════════════════════════════════════════════

def _find_config(ckpt_path: str) -> Optional[Path]:
    p = Path(ckpt_path).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        cfg = parent / "config.json"
        if cfg.exists():
            return cfg
    return None


def _extract_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "ema_state_dict"):
            if key in raw:
                return raw[key]
    return raw


def load_model(ckpt_path: str, device: torch.device):
    """Load OctreeNCA2DM2SingleStep from checkpoint + sibling config.json."""
    cfg_path = _find_config(ckpt_path)
    if cfg_path is None:
        raise FileNotFoundError(f"No config.json found near {ckpt_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    print(f"Config: {cfg_path}")

    from src.models.Model_OctreeNCA_2d_m2_single_step import OctreeNCA2DM2SingleStep

    # Don't re-load M1 pretrained weights — full composite checkpoint below
    config = dict(config)
    config["model.m1.pretrained_path"] = ""

    model = OctreeNCA2DM2SingleStep(config)
    model.to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = _extract_state_dict(raw)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"Loaded checkpoint: {ckpt_path}")
    model.eval()
    return model, config


# ═══════════════════════════════════════════════════════════════════════════
# Dummy input generation
# ═══════════════════════════════════════════════════════════════════════════

def make_dummy_inputs(batch_size: int, resolution: int, device: torch.device):
    """Create random dummy inputs mimicking dual-view iOCT frames (1-channel)."""
    x_a = torch.randn(batch_size, 1, resolution, resolution, device=device, dtype=torch.float32)
    x_b = torch.randn(batch_size, 1, resolution, resolution, device=device, dtype=torch.float32)
    return x_a, x_b


# ═══════════════════════════════════════════════════════════════════════════
# Timing utilities
# ═══════════════════════════════════════════════════════════════════════════

class Timer:
    """Context-manager timer that handles both CUDA and CPU timing."""

    def __init__(self, device: torch.device):
        self.is_cuda = device.type == "cuda" and torch.cuda.is_available()
        self.device = device
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.is_cuda:
            self._end.record()
            torch.cuda.synchronize(self.device)
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark functions
# ═══════════════════════════════════════════════════════════════════════════

def _warmup(fn, n_iters: int, device: torch.device):
    """Run fn() n_iters times to warm up CUDA kernels / JIT."""
    with torch.no_grad():
        for _ in range(n_iters):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_m1(model, x_a, x_b, device, warmup_iters=20, n_runs=50):
    """Benchmark M1 keyframe forward + state init."""
    def _fn():
        return model.m1_forward_and_init_states(x_a, x_b)

    _warmup(_fn, warmup_iters, device)

    timings_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t = Timer(device)
            with t:
                m1_out, (state_a, state_b) = _fn()
            timings_ms.append(t.elapsed_ms)

    return timings_ms, state_a.detach(), state_b.detach()


def benchmark_m2(model, x_a, x_b, state_a, state_b, step_k, device,
                 warmup_iters=20, n_runs=50):
    """Benchmark a single M2 forward step."""
    def _fn():
        return model(
            x_a, x_b,
            prev_state_a=state_a,
            prev_state_b=state_b,
            step_k=step_k,
        )

    _warmup(_fn, warmup_iters, device)

    timings_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t = Timer(device)
            with t:
                _fn()
            timings_ms.append(t.elapsed_ms)

    return timings_ms


def benchmark_end_to_end(model, x_a, x_b, keyframe_interval, device,
                         warmup_iters=10, n_runs=20):
    """Benchmark an end-to-end sequence of keyframe_interval frames.

    Simulates real inference: M1 at frame 0, M2 for frames 1..interval-1.
    Reports total time for the interval and amortized per-frame time.
    """
    def _fn():
        m1_out, (state_a, state_b) = model.m1_forward_and_init_states(x_a, x_b)
        for k in range(1, keyframe_interval):
            _ = model(
                x_a, x_b,
                prev_state_a=state_a,
                prev_state_b=state_b,
                step_k=k,
            )

    _warmup(_fn, warmup_iters, device)

    timings_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t = Timer(device)
            with t:
                _fn()
            timings_ms.append(t.elapsed_ms)

    return timings_ms


def profile_kernels(model, x_a, x_b, state_a, state_b, step_k,
                    device, filename_prefix: str, profile_m1: bool = True):
    """Export a PyTorch profiler chrome trace for kernel-level analysis."""
    from torch.profiler import ProfilerActivity

    output_dir = Path("inference_reports")
    output_dir.mkdir(exist_ok=True)
    trace_path = output_dir / f"{filename_prefix}_trace.json"

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    def trace_handler(prof):
        print(f"  Exporting chrome trace: {trace_path.resolve()}")
        prof.export_chrome_trace(str(trace_path))

    def _step():
        if profile_m1:
            m1_out, (sa, sb) = model.m1_forward_and_init_states(x_a, x_b)
        model(x_a, x_b, prev_state_a=state_a, prev_state_b=state_b, step_k=step_k)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                _step()
                prof.step()

    summary_path = output_dir / f"{filename_prefix}_summary.txt"
    sort_key = "cuda_time_total" if ProfilerActivity.CUDA in activities else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=30)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Kernel summary: {summary_path}")
    print(table)


# ═══════════════════════════════════════════════════════════════════════════
# Results formatting
# ═══════════════════════════════════════════════════════════════════════════

def _stats(timings_ms: list[float]) -> dict:
    arr = np.array(timings_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "n_runs": len(arr),
    }


def print_results(label: str, stats: dict, extra: str = ""):
    mean = stats["mean_ms"]
    std = stats["std_ms"]
    fps = 1000.0 / max(mean, 1e-9)
    print(f"  {label:40s}  {mean:8.2f} +/- {std:5.2f} ms  "
          f"({fps:7.1f} FPS)  [min={stats['min_ms']:.2f}, max={stats['max_ms']:.2f}]"
          f"  {extra}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Profile M1+M2 single-step inference latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to M2SingleStep model.pth checkpoint.")
    p.add_argument("--device", default="cuda:0",
                   help="Device (default: cuda:0)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (default: 1)")
    p.add_argument("--resolutions", type=int, nargs="+", default=[512],
                   help="Input resolutions to profile (default: 512)")
    p.add_argument("--keyframe-interval", type=int, default=None,
                   help="Override keyframe interval (default: read from config)")
    p.add_argument("--warmup-iters", type=int, default=20,
                   help="Warmup iterations (default: 20)")
    p.add_argument("--benchmark-runs", type=int, default=50,
                   help="Timed benchmark iterations (default: 50)")
    p.add_argument("--compile-mode", default="none",
                   choices=["none", "default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode (default: none)")
    p.add_argument("--profile-kernels", action="store_true",
                   help="Export PyTorch profiler chrome trace")
    p.add_argument("--output", type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument("--no-e2e", action="store_true",
                   help="Skip end-to-end sequence benchmark")
    p.add_argument("--m2-step-k", type=int, default=5,
                   help="Step k value for M2 benchmark (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    device = torch.device(args.device)
    is_cuda = device.type == "cuda" and torch.cuda.is_available()

    if is_cuda:
        torch.cuda.set_device(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA: {torch.version.cuda}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    keyframe_interval = args.keyframe_interval
    if keyframe_interval is None:
        keyframe_interval = int(config.get(
            "trainer.m2_single_step.keyframe_interval",
            config.get("M2_KEYFRAME_INTERVAL", 10),
        ))
    print(f"Keyframe interval: {keyframe_interval}")

    # ── Model summary ───────────────────────────────────────────────────
    m1_params = sum(p.numel() for p in model.m1.parameters())
    m2_params = sum(p.numel() for p in model.m2.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters:")
    print(f"  M1: {m1_params:>12,d}")
    print(f"  M2: {m2_params:>12,d}")
    print(f"  Total: {total_params:>9,d}")
    print(f"  M1 finest res: {model.m1_finest_res}")
    print(f"  M2 finest res: {model.m2_finest_res}")
    print(f"  M1 transfer level: {model.m1_transfer_level} ({model.m1_transfer_res})")
    print(f"  Warp logits: {model._warp_logits}")
    print(f"  FC bottleneck: {model._fc_bottleneck_dim}")
    print(f"  K-embedding: {model._k_embed is not None}")

    # ── Optional compilation ────────────────────────────────────────────
    if args.compile_mode != "none":
        print(f"\nCompiling model with mode='{args.compile_mode}'...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print("  Compilation successful.")
        except Exception as exc:
            print(f"  Compilation failed: {exc}. Continuing without compile.")

    # ── Profile each resolution ─────────────────────────────────────────
    all_results = []

    for resolution in args.resolutions:
        print(f"\n{'='*70}")
        print(f"PROFILING @ {resolution}x{resolution}  batch_size={args.batch_size}")
        print(f"{'='*70}")

        x_a, x_b = make_dummy_inputs(args.batch_size, resolution, device)

        # VRAM baseline
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            vram_before = torch.cuda.memory_allocated(device) / 1024**2

        # ── M1 benchmark ────────────────────────────────────────────────
        print(f"\n--- M1 Keyframe Benchmark ({args.benchmark_runs} runs) ---")
        m1_timings, state_a, state_b = benchmark_m1(
            model, x_a, x_b, device,
            warmup_iters=args.warmup_iters,
            n_runs=args.benchmark_runs,
        )
        m1_stats = _stats(m1_timings)
        print_results("M1 keyframe (forward + init states)", m1_stats)

        # ── M2 benchmark ────────────────────────────────────────────────
        step_k = args.m2_step_k
        print(f"\n--- M2 Single-Step Benchmark (k={step_k}, {args.benchmark_runs} runs) ---")
        m2_timings = benchmark_m2(
            model, x_a, x_b, state_a, state_b, step_k, device,
            warmup_iters=args.warmup_iters,
            n_runs=args.benchmark_runs,
        )
        m2_stats = _stats(m2_timings)
        print_results(f"M2 single step (k={step_k})", m2_stats)

        # ── Amortized per-frame cost ────────────────────────────────────
        m1_amortized = m1_stats["mean_ms"] / keyframe_interval
        per_frame_avg = m1_amortized + m2_stats["mean_ms"] * (keyframe_interval - 1) / keyframe_interval
        per_frame_fps = 1000.0 / max(per_frame_avg, 1e-9)
        print(f"\n--- Amortized Per-Frame (interval={keyframe_interval}) ---")
        print(f"  M1 amortized:        {m1_amortized:8.2f} ms/frame")
        print(f"  Per-frame average:   {per_frame_avg:8.2f} ms  ({per_frame_fps:.1f} FPS)")

        # ── End-to-end sequence benchmark ───────────────────────────────
        e2e_stats = None
        if not args.no_e2e:
            print(f"\n--- End-to-End Sequence Benchmark ({keyframe_interval} frames, "
                  f"{min(args.benchmark_runs, 20)} runs) ---")
            e2e_timings = benchmark_end_to_end(
                model, x_a, x_b, keyframe_interval, device,
                warmup_iters=min(args.warmup_iters, 10),
                n_runs=min(args.benchmark_runs, 20),
            )
            e2e_stats = _stats(e2e_timings)
            per_frame_e2e = e2e_stats["mean_ms"] / keyframe_interval
            fps_e2e = 1000.0 / max(per_frame_e2e, 1e-9)
            print_results(
                f"E2E {keyframe_interval}-frame sequence", e2e_stats,
                extra=f"= {per_frame_e2e:.2f} ms/frame ({fps_e2e:.1f} FPS)"
            )

        # ── VRAM ────────────────────────────────────────────────────────
        vram_info = {}
        if is_cuda:
            vram_peak = torch.cuda.max_memory_allocated(device) / 1024**2
            vram_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            vram_info = {
                "peak_allocated_mb": round(vram_peak, 1),
                "peak_reserved_mb": round(vram_reserved, 1),
            }
            print(f"\n--- VRAM ---")
            print(f"  Peak allocated: {vram_peak:.1f} MB")
            print(f"  Peak reserved:  {vram_reserved:.1f} MB")

        # ── Kernel profiling ────────────────────────────────────────────
        if args.profile_kernels:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"M2SingleStep_{resolution}x{resolution}_{args.compile_mode}_{timestamp}"
            print(f"\n--- Kernel Profiling ---")
            profile_kernels(
                model, x_a, x_b, state_a, state_b, step_k,
                device, prefix, profile_m1=True,
            )

        # ── Collect results ─────────────────────────────────────────────
        result = {
            "resolution": resolution,
            "batch_size": args.batch_size,
            "keyframe_interval": keyframe_interval,
            "compile_mode": args.compile_mode,
            "m2_step_k": step_k,
            "device": str(device),
            "m1_keyframe": m1_stats,
            "m2_single_step": m2_stats,
            "amortized_per_frame_ms": round(per_frame_avg, 3),
            "amortized_fps": round(per_frame_fps, 1),
            "vram": vram_info,
        }
        if e2e_stats is not None:
            result["e2e_sequence"] = e2e_stats
            result["e2e_per_frame_ms"] = round(e2e_stats["mean_ms"] / keyframe_interval, 3)
        result["model_params"] = {
            "m1": m1_params,
            "m2": m2_params,
            "total": total_params,
        }
        all_results.append(result)

    # ── Summary across resolutions ──────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS RESOLUTIONS")
        print(f"{'='*70}")
        print(f"  {'Resolution':>12s}  {'M1 (ms)':>10s}  {'M2 (ms)':>10s}  "
              f"{'Amortized':>10s}  {'FPS':>8s}  {'VRAM (MB)':>10s}")
        for r in all_results:
            vram_str = f"{r['vram']['peak_allocated_mb']:.0f}" if r['vram'] else "N/A"
            print(f"  {r['resolution']:>8d}x{r['resolution']:<4d}"
                  f"  {r['m1_keyframe']['mean_ms']:>10.2f}"
                  f"  {r['m2_single_step']['mean_ms']:>10.2f}"
                  f"  {r['amortized_per_frame_ms']:>10.2f}"
                  f"  {r['amortized_fps']:>8.1f}"
                  f"  {vram_str:>10s}")

    # ── Save to JSON ────────────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"inference_reports/m2_single_step_profile_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(device) if is_cuda else "CPU",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if is_cuda else None,
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    main()
