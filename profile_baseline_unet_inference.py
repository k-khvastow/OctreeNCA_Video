import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_baseline_unet import CONFIG as TRAIN_CONFIG
from train_baseline_unet import IOCTDataset, UNet


def resolve_checkpoint(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    best = path / "best_model.pth"
    if best.exists():
        return best

    checkpoints = sorted(path.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {path}. Expected best_model.pth or checkpoint_epoch_*.pth."
    )


def load_input_batch(
    data_root: str,
    input_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = IOCTDataset(data_root=data_root, input_size=input_size)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples found in dataset root '{data_root}'. "
            "Expected peeling/sri Bscans-dt structure."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, _masks = next(iter(loader))
    return images.to(device=device, dtype=torch.float32, non_blocking=True)


def benchmark(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup_iters: int,
    benchmark_runs: int,
    use_amp: bool,
    device: torch.device,
) -> dict:
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    timings_ms: list[float] = []

    amp_dtype = torch.float16 if is_cuda else torch.bfloat16
    amp_enabled = bool(use_amp and is_cuda)

    with torch.inference_mode():
        for _ in range(warmup_iters):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _ = model(input_tensor)

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            for _ in range(benchmark_runs):
                torch.cuda.synchronize(device)
                start_event.record()
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize(device)
                timings_ms.append(float(start_event.elapsed_time(end_event)))

        peak_alloc_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    else:
        with torch.inference_mode():
            for _ in range(benchmark_runs):
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=False):
                    _ = model(input_tensor)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
        peak_alloc_bytes = None
        peak_reserved_bytes = None

    avg_ms = float(np.mean(timings_ms))
    std_ms = float(np.std(timings_ms))
    fps = float(1000.0 / max(avg_ms, 1e-9))
    throughput = float(fps * input_tensor.shape[0])

    param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))
    input_bytes = int(input_tensor.numel() * input_tensor.element_size())

    return {
        "device": str(device),
        "batch_size": int(input_tensor.shape[0]),
        "input_shape": [int(v) for v in input_tensor.shape],
        "amp": amp_enabled,
        "warmup_iters": warmup_iters,
        "benchmark_runs": benchmark_runs,
        "latency_ms_mean": avg_ms,
        "latency_ms_std": std_ms,
        "fps": fps,
        "images_per_second": throughput,
        "model_parameter_bytes": param_bytes,
        "input_tensor_bytes": input_bytes,
        "peak_memory_allocated_bytes": peak_alloc_bytes,
        "peak_memory_reserved_bytes": peak_reserved_bytes,
    }


def print_report(report: dict) -> None:
    print("\n=== Baseline UNet Inference Profile ===")
    print(f"Device:                    {report['device']}")
    print(f"Input shape:               {tuple(report['input_shape'])}")
    print(f"AMP enabled:               {report['amp']}")
    print(f"Latency:                   {report['latency_ms_mean']:.2f} +- {report['latency_ms_std']:.2f} ms")
    print(f"FPS (batch=1):             {report['fps']:.2f}")
    print(f"Throughput (img/s):        {report['images_per_second']:.2f}")

    if report["peak_memory_allocated_bytes"] is not None:
        alloc_mb = report["peak_memory_allocated_bytes"] / (1024**2)
        reserved_mb = report["peak_memory_reserved_bytes"] / (1024**2)
        print(f"Peak CUDA allocated VRAM:  {alloc_mb:.2f} MB")
        print(f"Peak CUDA reserved VRAM:   {reserved_mb:.2f} MB")
    else:
        print("Peak CUDA VRAM:            N/A (running on CPU)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile inference latency/FPS/VRAM for train_baseline_unet.py checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=TRAIN_CONFIG.get("model_save_path", "Models/iOCT_UNet"),
        help="Path to checkpoint file or model directory.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=TRAIN_CONFIG.get("data_path", "ioct_data"),
        help="Dataset root for a real input batch.",
    )
    parser.add_argument(
        "--input_h",
        type=int,
        default=int(TRAIN_CONFIG.get("input_size", (512, 512))[0]),
    )
    parser.add_argument(
        "--input_w",
        type=int,
        default=int(TRAIN_CONFIG.get("input_size", (512, 512))[1]),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--json_out", type=str, default="", help="Optional path to save JSON report.")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    model = UNet(
        in_channels=int(TRAIN_CONFIG["input_channels"]),
        channels=int(TRAIN_CONFIG["base_channels"]),
        n_classes=int(TRAIN_CONFIG["n_classes"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    input_batch = load_input_batch(
        data_root=args.data_root,
        input_size=(args.input_h, args.input_w),
        batch_size=args.batch_size,
        device=device,
    )

    report = benchmark(
        model=model,
        input_tensor=input_batch,
        warmup_iters=args.warmup,
        benchmark_runs=args.runs,
        use_amp=args.amp,
        device=device,
    )
    report["checkpoint"] = str(checkpoint_path)
    report["data_root"] = args.data_root

    print_report(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report: {out_path}")


if __name__ == "__main__":
    main()
