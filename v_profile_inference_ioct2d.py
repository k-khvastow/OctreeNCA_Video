import argparse
import datetime
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.profiler import ProfilerActivity

from src.utils.ProjectConfiguration import ProjectConfiguration as pc


PROFILE_SETTINGS = {
    "compile_mode": "max-autotune",  # "none", "default", "reduce-overhead", "max-autotune"
    "batch_size": 1,
    "warmup_iters": 20,
    "benchmark_runs": 50,
    "profile_kernels": True,
    "device": "cuda:0",
    "strict_checkpoint": True,
}


def _infer_experiment_model_path(config: dict) -> str:
    if "experiment.model_path" in config:
        return config["experiment.model_path"]
    return os.path.join(
        pc.STUDY_PATH,
        "Experiments",
        f"{config['experiment.name']}_{config['experiment.description']}",
    )


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        new_key = new_key.replace("._orig_mod.", ".")
        normalized[new_key] = value
    return normalized


def _load_model_weights(model: torch.nn.Module, checkpoint_dir: str, device: torch.device) -> None:
    checkpoint_file = os.path.join(checkpoint_dir, "model.pth")
    try:
        state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_file, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_file}: {type(state_dict)}")

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint with strict=True (raw state dict).")
        return
    except RuntimeError as raw_exc:
        normalized = _normalize_state_dict_keys(state_dict)
        try:
            model.load_state_dict(normalized, strict=True)
            print("Loaded checkpoint with strict=True after key normalization.")
            return
        except RuntimeError as norm_exc:
            if PROFILE_SETTINGS["strict_checkpoint"]:
                raise RuntimeError(
                    "Strict checkpoint load failed (raw and normalized keys)."
                ) from norm_exc
            missing, unexpected = model.load_state_dict(normalized, strict=False)
            print(
                "Warning: loaded checkpoint with strict=False. "
                f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
            )
            print(f"Raw strict error: {raw_exc}")


def _resolve_checkpoint_dir(model_dir: str, checkpoint_epoch: int | None, checkpoint_dir: str | None) -> str:
    if checkpoint_dir:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if os.path.basename(checkpoint_dir) == "model.pth":
            checkpoint_dir = os.path.dirname(checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
        if not os.path.exists(os.path.join(checkpoint_dir, "model.pth")):
            raise FileNotFoundError(f"model.pth not found in checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    if checkpoint_epoch is not None:
        checkpoint_dir = os.path.join(model_dir, f"epoch_{int(checkpoint_epoch)}")
        if not os.path.exists(os.path.join(checkpoint_dir, "model.pth")):
            raise FileNotFoundError(f"checkpoint epoch not found: {checkpoint_dir}")
        return checkpoint_dir

    epochs = [f for f in os.listdir(model_dir) if f.startswith("epoch_") and f.split("_")[-1].isdigit()]
    if not epochs:
        raise FileNotFoundError(f"no epoch_* checkpoints found in: {model_dir}")
    epochs.sort(key=lambda x: int(x.split("_")[1]))
    return os.path.join(model_dir, epochs[-1])


def setup_experiment(random_word: str | None, checkpoint_epoch: int | None, checkpoint_dir: str | None):
    print("--- Setting up iOCT2D Experiment ---")
    try:
        from train_ioct2d import get_dataset_args, get_study_config, iOCTDatasetForExperiment
        from src.utils.BaselineConfigs import EXP_OctreeNCA
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import train_ioct2d dependencies. Install training runtime dependencies first."
        ) from exc

    train_config = get_study_config()
    if random_word:
        train_config["experiment.name"] = f"iOCT2D_{random_word}_{train_config['model.channel_n']}"

    train_model_path = _infer_experiment_model_path(train_config)
    train_model_dir = os.path.join(pc.FILER_BASE_PATH, train_model_path, "models")
    checkpoint_path = _resolve_checkpoint_dir(train_model_dir, checkpoint_epoch, checkpoint_dir)
    print(f"Resolved checkpoint: {checkpoint_path}")

    study_config = get_study_config()
    profile_tag = random_word if random_word is not None else "manual"
    profile_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_config["experiment.name"] = f"iOCT2D_profile_{profile_tag}_{profile_stamp}"
    study_config["experiment.description"] = "Inference profiling run (no resume)"
    study_config["experiment.model_path"] = _infer_experiment_model_path(study_config)

    study_config["experiment.dataset.preload"] = False
    study_config["trainer.batch_size"] = PROFILE_SETTINGS["batch_size"]
    study_config["performance.compile"] = False
    study_config["experiment.device"] = PROFILE_SETTINGS["device"]
    study_config["experiment.use_wandb"] = False

    dataset_args = get_dataset_args(study_config)
    exp = EXP_OctreeNCA().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTDatasetForExperiment,
        dataset_args=dataset_args,
    )

    try:
        print(f"Loading weights from: {checkpoint_path}")
        _load_model_weights(exp.agent.model, checkpoint_path, exp.agent.device)
    except Exception as exc:
        if PROFILE_SETTINGS["strict_checkpoint"]:
            raise
        print(f"Warning: could not load checkpoint ({exc}). Profiling random weights.")

    return exp


def prepare_dummy_input(exp):
    config = exp.config
    h, w = config.get("experiment.dataset.input_size", (512, 512))
    c = int(config["model.input_channels"])
    b = int(PROFILE_SETTINGS["batch_size"])
    print(f"Preparing input: batch={b}, channels={c}, height={h}, width={w}")
    return torch.randn(b, c, h, w, device=exp.agent.device, dtype=torch.float32)


def _run_model(model, input_tensor):
    return model(input_tensor, batch_duplication=1)


def benchmark(model, input_tensor, device):
    print(f"\n--- Benchmarking Latency ({PROFILE_SETTINGS['benchmark_runs']} runs) ---")
    is_cuda = device.type == "cuda" and torch.cuda.is_available()

    print(f"Warming up ({PROFILE_SETTINGS['warmup_iters']} iters)...")
    with torch.no_grad():
        for _ in range(PROFILE_SETTINGS["warmup_iters"]):
            _ = _run_model(model, input_tensor)
    if is_cuda:
        torch.cuda.synchronize(device)

    timings_ms = []
    with torch.no_grad():
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(PROFILE_SETTINGS["benchmark_runs"]):
                torch.cuda.synchronize(device)
                start_event.record()
                _ = _run_model(model, input_tensor)
                end_event.record()
                torch.cuda.synchronize(device)
                timings_ms.append(start_event.elapsed_time(end_event))
        else:
            for _ in range(PROFILE_SETTINGS["benchmark_runs"]):
                t0 = time.perf_counter()
                _ = _run_model(model, input_tensor)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(timings_ms))
    std_ms = float(np.std(timings_ms))
    fps = 1000.0 / max(avg_ms, 1e-9)
    print(f"Average Inference Time: {avg_ms:.2f} ms +- {std_ms:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


def profile_kernels(model, input_tensor, filename_prefix, device):
    print("\n--- Profiling Kernels ---")
    output_dir = Path("inference_reports")
    output_dir.mkdir(exist_ok=True)
    trace_path = output_dir / f"{filename_prefix}_trace.json"

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    def trace_handler(prof):
        print(f"Exporting chrome trace to: {trace_path.resolve()}")
        prof.export_chrome_trace(str(trace_path))

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                _ = _run_model(model, input_tensor)
                prof.step()

    summary_path = output_dir / f"{filename_prefix}_summary.txt"
    sort_key = "cuda_time_total" if ProfilerActivity.CUDA in activities else "cpu_time_total"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(prof.key_averages().table(sort_by=sort_key, row_limit=30))
    print(f"Kernel summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile inference for train_ioct2d.py models.")
    parser.add_argument("--random_word", type=str, default=None, help="Training random word from experiment name iOCT2D_<word>_<channel_n>.")
    parser.add_argument("--checkpoint_epoch", type=int, default=None, help="Specific epoch number to load from model dir.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Direct path to checkpoint epoch directory (or model.pth).")
    parser.add_argument("--device", type=str, default=PROFILE_SETTINGS["device"])
    parser.add_argument("--compile_mode", type=str, default=PROFILE_SETTINGS["compile_mode"])
    parser.add_argument("--batch_size", type=int, default=PROFILE_SETTINGS["batch_size"])
    parser.add_argument("--warmup_iters", type=int, default=PROFILE_SETTINGS["warmup_iters"])
    parser.add_argument("--benchmark_runs", type=int, default=PROFILE_SETTINGS["benchmark_runs"])
    parser.add_argument("--profile_kernels", action="store_true", default=PROFILE_SETTINGS["profile_kernels"])
    parser.add_argument("--no_profile_kernels", action="store_false", dest="profile_kernels")
    parser.add_argument("--strict_checkpoint", action="store_true", default=PROFILE_SETTINGS["strict_checkpoint"])
    parser.add_argument("--allow_random_weights", action="store_false", dest="strict_checkpoint")
    args = parser.parse_args()

    PROFILE_SETTINGS["device"] = args.device
    PROFILE_SETTINGS["compile_mode"] = args.compile_mode
    PROFILE_SETTINGS["batch_size"] = args.batch_size
    PROFILE_SETTINGS["warmup_iters"] = args.warmup_iters
    PROFILE_SETTINGS["benchmark_runs"] = args.benchmark_runs
    PROFILE_SETTINGS["profile_kernels"] = args.profile_kernels
    PROFILE_SETTINGS["strict_checkpoint"] = args.strict_checkpoint

    if PROFILE_SETTINGS["strict_checkpoint"] and not any(
        [args.random_word is not None, args.checkpoint_dir is not None]
    ):
        raise ValueError(
            "Provide --random_word or --checkpoint_dir when strict checkpoint loading is enabled."
        )

    exp = setup_experiment(
        random_word=args.random_word,
        checkpoint_epoch=args.checkpoint_epoch,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = exp.agent.model
    device = exp.agent.device
    model.eval()
    model.to(device)

    input_tensor = prepare_dummy_input(exp)

    mode = PROFILE_SETTINGS["compile_mode"]
    if mode != "none":
        print(f"Compiling model with mode='{mode}'...")
        try:
            model = torch.compile(model, mode=mode)
        except Exception as exc:
            print(f"Compilation failed: {exc}. Proceeding without compilation.")

    benchmark(model, input_tensor, device)

    if PROFILE_SETTINGS["profile_kernels"]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        res = exp.config.get("experiment.dataset.input_size", "unknown")
        prefix = f"Profile_iOCT2D_{res}_{mode}_{timestamp}"
        profile_kernels(model, input_tensor, prefix, device)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    main()
