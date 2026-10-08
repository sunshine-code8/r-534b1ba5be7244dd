"""Benchmark full-frame NeuralDSP candidates with dense four-class outputs.

Run from Ghost-FWL: python benchmark_original_neuraldsp.py --candidate all --skip-flops
Edit CANDIDATES below or override --patch-size, --dim/--channels, and --depths.
The legacy point-return benchmark is retained as --candidate original (needs einops).
No dataset, training configuration, or checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import math
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ORIGINAL_COMMIT = "9049e6b2a5fc7c20c5485d4ec0e1a6ecd8dd4da2"
ORIGINAL_FILES = (
    "neuraldsp/__init__.py",
    "neuraldsp/networks/__init__.py",
    "neuraldsp/networks/neural_dsp_swinunet.py",
    "neuraldsp/networks/swin_helpers.py",
    "neuraldsp/networks/transformer_helpers.py",
)
HEIGHT, WIDTH, TIME_BINS = 336, 400, 256

# Editable candidate configurations. channels = (full, half, quarter, eighth).
# depths = (half encoder, quarter encoder, eighth bottleneck,
#           quarter decoder, half decoder). Zero disables that Transformer only.
CANDIDATES = {
    "preferred": {"patch_size": 16, "channels": (16, 16, 32, 64), "depths": (0, 1, 2, 1, 0)},
    "fast": {"patch_size": 16, "channels": (16, 16, 32, 64), "depths": (0, 1, 1, 1, 0)},
    "patch32": {"patch_size": 32, "channels": (16, 16, 32, 64), "depths": (0, 1, 1, 1, 0)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", choices=[*CANDIDATES, "all", "original"], default="preferred"
    )
    parser.add_argument("--patch-size", type=int, help="Override temporal patch length")
    widths = parser.add_mutually_exclusive_group()
    widths.add_argument("--dim", type=int, help="Base width D; uses channels (D,D,2D,4D)")
    widths.add_argument(
        "--channels", type=int, nargs=4, metavar=("FULL", "HALF", "QUARTER", "EIGHTH")
    )
    parser.add_argument("--depths", type=int, nargs=5, metavar=("E1", "E2", "BOT", "D2", "D1"))
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--time-bins", type=int, default=TIME_BINS)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--window-size", type=int, nargs=2, default=(2, 4))
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument(
        "--tf32",
        choices=["off", "on"],
        default="off",
        help="Candidate CUDA matmul AND cuDNN TF32; default: off",
    )
    parser.add_argument(
        "--overall-only", action="store_true", help="Skip isolated stage benchmarks"
    )
    parser.add_argument("--csv", type=Path, help="Write benchmark rows and configurations to CSV")
    parser.add_argument(
        "--neuraldsp-repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "NeuralLidarDSP",
        help="Path to the NeuralLidarDSP Git checkout",
    )
    parser.add_argument(
        "--revision", default=ORIGINAL_COMMIT, help="Original Git commit to benchmark"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-flops", action="store_true", help="Skip the operator FLOP counter")
    parser.add_argument(
        "--disable-cudnn-tf32", action="store_true", help="Use full FP32 for cuDNN convolutions"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1 or args.repeats < 2:
        parser.error("batch-size and iterations must be positive, warmup >= 0, repeats >= 2")
    if args.candidate == "original" and not (args.neuraldsp_repo / ".git").exists():
        parser.error(f"NeuralDSP Git checkout not found: {args.neuraldsp_repo}")
    if args.candidate == "original" and (
        args.patch_size is not None
        or args.dim is not None
        or args.channels is not None
        or args.depths is not None
        or args.precision != "fp32"
        or args.csv is not None
        or args.overall_only
        or args.tf32 != "off"
        or args.heads != 2
        or tuple(args.window_size) != (2, 4)
        or args.mlp_ratio != 2
        or (args.height, args.width, args.time_bins) != (HEIGHT, WIDTH, TIME_BINS)
    ):
        parser.error(
            "Candidate architecture/precision/CSV options do not apply to --candidate original"
        )
    return args


def load_pristine_module(repo: Path, revision: str, archive: Path) -> Any:
    """Import just the original tracked files from a temporary ZIP, never the worktree."""
    command = [
        "git",
        "-C",
        str(repo),
        "archive",
        "--format=zip",
        f"--output={archive}",
        revision,
        *ORIGINAL_FILES,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Cannot read original NeuralDSP revision {revision}: {exc.stderr}"
        ) from exc
    sys.path.insert(0, str(archive))
    try:
        return importlib.import_module("neuraldsp.networks.neural_dsp_swinunet")
    except ModuleNotFoundError as exc:
        if exc.name == "einops":
            raise RuntimeError(
                "einops is required; run with `uv run --with einops python ...`"
            ) from exc
        raise


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def measure_flops(fn: Callable[[], Any], device: torch.device) -> int:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        raise RuntimeError("FLOP counting requires newer PyTorch; pass --skip-flops") from exc
    synchronize(device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    synchronize(device)
    return int(counter.get_total_flops())


def measure_speed(
    fn: Callable[[], Any], device: torch.device, warmup: int, iterations: int, repeats: int
) -> tuple[float, float]:
    latencies_ms = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000 / iterations)
    return statistics.mean(latencies_ms), statistics.stdev(latencies_ms)


def matched_filter(model: nn.Module, waveform: Tensor) -> Tensor:
    batch, rows, cols, n_samples = waveform.shape[:4]
    flattened = waveform.permute(0, 1, 2, 4, 3).view(batch * rows * cols, -1, n_samples)
    filtered = F.conv1d(flattened, model.matched_filter_values, padding="same")
    return filtered.view(batch, rows, cols, -1, n_samples).permute(0, 1, 2, 4, 3)


def add_original_positions(model: nn.Module, tokens: Tensor) -> Tensor:
    """Original sinusoidal-position calculation and in-place addition."""
    dtype, device = tokens.dtype, tokens.device
    pe_t = torch.zeros((model.num_patches, model.dim), dtype=dtype, device=device)
    pos_t = torch.arange(0, model.num_patches, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, model.dim, 2, dtype=dtype, device=device) * -(math.log(10000.0) / model.dim)
    )
    pe_t[:, 0::2] = torch.sin(pos_t * div_term)
    pe_t[:, 1::2] = torch.cos(pos_t * div_term)
    tokens += pe_t[None, None, None, ...]
    return tokens


def capture_stage_inputs(model: nn.Module, waveform: Tensor) -> dict[str, tuple[Tensor, ...]]:
    captured: dict[str, tuple[Tensor, ...]] = {}
    modules = {
        "patch": model.to_patch_embedding,
        "block1": model.block1,
        "block2": model.block2,
        "block3": model.block3,
        "block4": model.block4,
        "block5": model.block5,
        "head": model.linear_score,
    }
    handles = [
        module.register_forward_pre_hook(
            lambda _module, inputs, name=name: captured.__setitem__(name, inputs)
        )
        for name, module in modules.items()
    ]
    try:
        with torch.inference_mode():
            model(waveform)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def make_cases(
    model: nn.Module, module: Any, waveform: Tensor, captured: dict[str, tuple[Tensor, ...]]
) -> list[tuple[str, Callable[[], Any], int]]:
    patch_input = captured["patch"][0]
    # The original forward mutates patch tokens when adding positions. Keep a
    # separate scratch tensor so repeated position-only calls do not alter
    # the saved Block 1 input.
    with torch.no_grad():
        position_scratch = model.to_patch_embedding(patch_input)
    head_input = captured["head"][0]

    def prediction_head() -> dict[str, Tensor]:
        batch, rows, cols = head_input.shape[:3]
        mask = F.softmax(model.linear_score(head_input), dim=-1)
        offset = torch.sigmoid(model.linear_dist(head_input))
        tof = module.map_patch_offset_to_full_tof_batched(offset, model.patch_dim)
        return {
            "patch_class": mask.view(batch, rows, cols, model.num_patches, -1),
            "patch_offset": offset.view(batch, rows, cols, model.num_patches, 1),
            "tof": tof.view(batch, rows, cols, model.num_patches, 1),
        }

    cases = []
    if model.do_matched_filtering:
        cases.append(
            (
                "Matched filter",
                lambda: matched_filter(model, waveform),
                model.matched_filter_values.numel(),
            )
        )
    cases.extend(
        [
            (
                "Patch embedding",
                lambda: model.to_patch_embedding(patch_input),
                parameter_count(model.to_patch_embedding),
            ),
            ("Position encoding", lambda: add_original_positions(model, position_scratch), 0),
            (
                "Down block 1",
                lambda: model.block1(*captured["block1"]),
                parameter_count(model.block1),
            ),
            (
                "Down block 2",
                lambda: model.block2(*captured["block2"]),
                parameter_count(model.block2),
            ),
            (
                "Bottleneck block 3",
                lambda: model.block3(*captured["block3"]),
                parameter_count(model.block3),
            ),
            (
                "Up block 4",
                lambda: model.block4(*captured["block4"]),
                parameter_count(model.block4),
            ),
            (
                "Up block 5",
                lambda: model.block5(*captured["block5"]),
                parameter_count(model.block5),
            ),
            (
                "Prediction head",
                prediction_head,
                parameter_count(model.linear_score) + parameter_count(model.linear_dist),
            ),
            ("Overall", lambda: model(waveform), parameter_count(model)),
        ]
    )
    return cases


def run_original(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu if needed")
    if args.disable_cudnn_tf32:
        torch.backends.cudnn.allow_tf32 = False

    with tempfile.TemporaryDirectory(prefix="original_neuraldsp_") as temp_dir:
        original = load_pristine_module(
            args.neuraldsp_repo.resolve(), args.revision, Path(temp_dir) / "neuraldsp_original.zip"
        )
        torch.manual_seed(42)
        model = (
            original.NeuralDSP(
                n_samples=TIME_BINS,
                n_channels=1,
                n_rows=HEIGHT,
                n_cols=WIDTH,
                dim=128,
                waveform_patch_size=64,
                heads=[2, 2, 2],
                depths=[2, 2, 2],
                window_size=(2, 4),
                mlp_ratio=2,
                add_temporal_pos_embed=True,
                dropout=[0.3] * 5,
                do_matched_filtering=True,
            )
            .to(device)
            .eval()
        )
        waveform = torch.randn(args.batch_size, HEIGHT, WIDTH, TIME_BINS, 1, device=device)
        captured = capture_stage_inputs(model, waveform)
        cases = make_cases(model, original, waveform, captured)

        device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        print(f"Original NeuralDSP commit: {args.revision}")
        print(f"Source Git checkout: {args.neuraldsp_repo.resolve()} (working-tree edits ignored)")
        print(
            f"Device: {device} ({device_name}); PyTorch: {torch.__version__}; FP32; batch: {args.batch_size}"
        )
        print(
            f"Input: {tuple(waveform.shape)} = (B,H,W,T,C); waveform patches: {model.num_patches}"
        )
        print(
            f"cuDNN allow_tf32: {torch.backends.cudnn.allow_tf32}; matmul allow_tf32: {torch.backends.cuda.matmul.allow_tf32}"
        )
        print(
            f"Warmup: {args.warmup}; iterations/repeat: {args.iterations}; repeats: {args.repeats}"
        )
        print(
            "FLOPs: PyTorch operator counter, one batch; multiply-add = 2; some pointwise ops omitted"
        )
        print("FPS: batch-size / latency; isolated stage FPS is NOT full-model FPS")
        print("Input creation, checkpoint loading, and host-to-device transfers are excluded")
        print(
            f"{'Stage':<21} {'Params (M)':>12} {'FLOPs (G)':>12} {'ms/call ± std':>19} {'FPS':>12}"
        )
        for name, fn, params in cases:
            flops = None if args.skip_flops else measure_flops(fn, device)
            mean_ms, std_ms = measure_speed(fn, device, args.warmup, args.iterations, args.repeats)
            flops_text = "skipped" if flops is None else f"{flops / 1e9:.3f}"
            print(
                f"{name:<21} {params / 1e6:12.6f} {flops_text:>12} "
                f"{mean_ms:8.3f} ± {std_ms:<7.3f} {args.batch_size * 1000 / mean_ms:12.2f}",
                flush=True,
            )
        print(
            "Note: position encoding reuses a scratch tensor; the original model also has an unused learned position parameter."
        )


def run_candidate(args: argparse.Namespace, name: str) -> list[dict[str, Any]]:
    from experiments.neuraldsp_fwl.benchmark_candidates import FullFrameNeuralDSP

    config = dict(CANDIDATES[name])
    if args.patch_size is not None:
        config["patch_size"] = args.patch_size
    if args.dim is not None:
        config["channels"] = (args.dim, args.dim, 2 * args.dim, 4 * args.dim)
    if args.channels is not None:
        config["channels"] = tuple(args.channels)
    if args.depths is not None:
        config["depths"] = tuple(args.depths)
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use --precision fp32 for CPU validation")
    torch.manual_seed(42)
    model = (
        FullFrameNeuralDSP(
            args.height,
            args.width,
            args.time_bins,
            **config,
            heads=args.heads,
            window_size=tuple(args.window_size),
            mlp_ratio=args.mlp_ratio,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    waveform = torch.randn(
        args.batch_size, args.height, args.width, args.time_bins, 1, device=device, dtype=dtype
    )
    expected = (args.batch_size, args.height, args.width, args.time_bins, 4)
    with torch.inference_mode():
        output = model(waveform)
        if tuple(output.shape) != expected or not output.is_contiguous():
            raise RuntimeError(
                f"Incorrect output layout: {tuple(output.shape)}, expected {expected}"
            )
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Model output contains non-finite logits")
        del output
    synchronize(device)
    print(
        f"\nCandidate: {name}; patch_size={config['patch_size']}; channels={config['channels']}; depths={config['depths']}",
        flush=True,
    )
    print(f"Input: {tuple(waveform.shape)}; output: {expected} (B,H,W,T,4), dense logits")
    print(
        f"Heads={args.heads}; window={tuple(args.window_size)}; MLP ratio={args.mlp_ratio}; parameters={parameter_count(model):,}"
    )
    print(f"{'Stage':<23} {'Params (M)':>12} {'FLOPs (G)':>12} {'ms/call ± std':>19} {'FPS':>12}")
    rows = []

    def report(stage: str, module: nn.Module, fn: Callable[[], Tensor]) -> None:
        # Overall is timed before hooks/captured stage tensors exist. Each call
        # includes the dense head and output materialization, with no host copy.
        if stage == "Overall" and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        mean_ms, std_ms = measure_speed(fn, device, args.warmup, args.iterations, args.repeats)
        peak = (
            torch.cuda.max_memory_allocated(device) / 2**20
            if stage == "Overall" and device.type == "cuda"
            else None
        )
        flops = None if args.skip_flops else measure_flops(fn, device)
        params = parameter_count(module)
        fps = args.batch_size * 1000 / mean_ms
        flops_text = "skipped" if flops is None else f"{flops / 1e9:.3f}"
        print(
            f"{stage:<23} {params / 1e6:12.6f} {flops_text:>12} "
            f"{mean_ms:8.3f} ± {std_ms:<7.3f} {fps:12.2f}",
            flush=True,
        )
        rows.append(
            {
                "candidate": name,
                "patch_size": config["patch_size"],
                "channels": str(config["channels"]),
                "depths": str(config["depths"]),
                "height": args.height,
                "width": args.width,
                "time_bins": args.time_bins,
                "batch_size": args.batch_size,
                "precision": args.precision,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "torch_version": torch.__version__,
                "heads": args.heads,
                "window": str(tuple(args.window_size)),
                "mlp_ratio": args.mlp_ratio,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "stage": stage,
                "parameters": params,
                "flops": flops,
                "latency_ms": mean_ms,
                "std_ms": std_ms,
                "fps": fps,
                "peak_allocated_mib": peak,
            }
        )
        if peak is not None:
            print(f"  Overall peak allocated GPU memory: {peak:.1f} MiB")

    report("Overall", model, lambda: model(waveform))
    if not args.overall_only:
        stages = [
            ("Matched filter", model.matched_filter),
            ("Patch embedding", model.patch_embedding),
            ("Position encoding", model.position_encoding),
            ("Down block 1", model.block1),
            ("Down block 2", model.block2),
            ("Bottleneck block 3", model.block3),
            ("Up block 4", model.block4),
            ("Up block 5", model.block5),
            ("Full-res recovery", model.full_resolution),
            ("4-class head", model.classification_head),
        ]
        captured = {}
        handles = [
            module.register_forward_pre_hook(
                lambda _module, inputs, label=label: captured.__setitem__(label, inputs)
            )
            for label, module in stages
        ]
        try:
            with torch.inference_mode():
                model(waveform)
        finally:
            for handle in handles:
                handle.remove()
        # FLOP counting uses no_grad, not inference_mode. Clone captured inputs
        # outside inference_mode so older torch versions can also run that pass.
        captured = {label: tuple(x.clone() for x in inputs) for label, inputs in captured.items()}
        for label, module in stages:
            report(label, module, lambda module=module, label=label: module(*captured[label]))
    return rows


def main() -> None:
    args = parse_args()
    if args.candidate == "original":
        run_original(args)
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for small-shape validation")
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This synchronized benchmark supports CPU and CUDA only")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on" and not args.disable_cudnn_tf32
    print(
        f"Device: {device}; PyTorch: {torch.__version__}; precision: {args.precision}; batch: {args.batch_size}"
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"TF32 matmul={torch.backends.cuda.matmul.allow_tf32}; cuDNN={torch.backends.cudnn.allow_tf32}"
    )
    print(
        "Eager inference, random weights, no checkpoint. FP16/BF16 use model AND input in that dtype."
    )
    print("Overall includes filtering, backbone, full-resolution recovery and four-class logits.")
    print("Excludes data loading, host/device transfer, argmax and other postprocessing.")
    print("FPS = batch size / mean batch latency; isolated stage FPS is not model FPS.")
    print("FLOPs: operator counter, multiply-add=2; some elementwise operations omitted.")
    names = list(CANDIDATES) if args.candidate == "all" else [args.candidate]
    rows = []
    for name in names:
        rows.extend(run_candidate(args, name))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\nOverall comparison:")
    for row in rows:
        if row["stage"] == "Overall":
            print(f"{row['candidate']:<12} {row['latency_ms']:.3f} ms/batch; {row['fps']:.2f} FPS")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
