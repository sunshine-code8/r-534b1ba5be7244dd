"""Benchmark FastWaveDSP and FineWaveDSP grouped 2D U-Net inference; no dataset or checkpoint required.

python benchmark_wavedsp.py --model all --precision fp16 --device cuda:0
Uses the same synchronized wall-clock timing as benchmark_original_neuraldsp.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch

from benchmark_original_neuraldsp import (
    CANDIDATES,
    measure_flops,
    measure_speed,
    parameter_count,
    synchronize,
)
from experiments.wavedsp import FastWaveDSP, FineWaveDSP
from experiments.wavedsp.model import ARCHITECTURE_VERSION

NEURALDSP_CONFIGS = {
    **CANDIDATES,
    "user_patch32": {"patch_size": 32, "channels": (16, 16, 32, 64), "depths": (0, 1, 2, 1, 0)},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fast", "fine", "all"), default="all")
    parser.add_argument(
        "--include-neuraldsp",
        choices=tuple(NEURALDSP_CONFIGS),
        help="Also benchmark an existing dense NeuralDSP candidate under identical settings",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--tf32", choices=("off", "on"), default="off")
    parser.add_argument(
        "--cudnn-benchmark", action="store_true", help="Enable convolution autotuning"
    )
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--time-bins", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--patch-size", type=int, help="Temporal patch length: default fast=16, fine=4"
    )
    parser.add_argument("--patch-dim", type=int, help="Embedding dim: default fast=8, fine=4")
    parser.add_argument(
        "--full-head", action="store_true", help="Fast: use groups=1 in the classification head"
    )
    parser.add_argument(
        "--stem-channels",
        type=int,
        help="Embedding compression width: default fast=96, fine=128",
    )
    parser.add_argument(
        "--no-bottleneck-mixer",
        action="store_true",
        help="Fast: omit the extra 1x1 bottleneck mixer",
    )
    parser.add_argument(
        "--no-temporal-attention",
        action="store_true",
        help="Fine: omit bottleneck temporal attention",
    )
    parser.add_argument(
        "--no-relative-bias", action="store_true", help="Fine: omit relative token-distance bias"
    )
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--attention-token-dim", type=int, default=8)
    parser.add_argument(
        "--attention-chunk-size",
        type=int,
        default=0,
        help="Fine: bottleneck spatial sequences per attention chunk; 0 = all",
    )
    parser.add_argument(
        "--output",
        choices=("logits", "labels", "both"),
        default="logits",
        help="labels includes argmax to uint8; both times each path separately",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Report parameters and FLOPs without repeated FPS timing",
    )
    parser.add_argument(
        "--skip-flops", action="store_true",
        help="Skip the extra operator-counting forward pass",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if (
        min(
            args.height,
            args.width,
            args.time_bins,
            args.batch_size,
            args.iterations,
            args.cpu_threads,
        )
        < 1
        or args.repeats < 2
        or args.warmup < 0
    ):
        parser.error(
            "Dimensions/iterations/CPU threads must be positive; repeats >= 2, warmup >= 0"
        )
    if args.patch_size is not None and args.patch_size < 1:
        parser.error("patch-size must be positive")
    if args.patch_dim is not None and args.patch_dim < 1:
        parser.error("patch-dim must be positive")
    names = ("fast", "fine") if args.model == "all" else (args.model,)
    if args.full_head and "fast" not in names:
        parser.error("--full-head applies to Fast only; select --model fast or all")
    for name in names:
        patch = args.patch_size or (16 if name == "fast" else 4)
        if args.time_bins % patch:
            parser.error(f"time-bins must be divisible by {name}'s patch-size={patch}")
    if args.attention_chunk_size < 0:
        parser.error("attention-chunk-size must be >= 0")
    if (
        min(args.attention_heads, args.attention_token_dim) < 1
        or args.attention_token_dim % args.attention_heads
    ):
        parser.error(
            "attention-token-dim must be positive and divisible by positive attention-heads"
        )
    if args.stem_channels is not None and args.stem_channels < 1:
        parser.error("stem-channels must be positive")
    if "fast" in names:
        if args.patch_dim is not None and (args.patch_dim < 2 or args.patch_dim % 2):
            parser.error("Fast+ dual projection requires an even patch-dim >= 2")
        if args.patch_size is not None and args.patch_size < 2:
            parser.error("Fast+ requires patch-size >= 2 for within-patch differences")
    return args


def run_model(args: argparse.Namespace, name: str) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    torch.manual_seed(42)
    config: dict[str, Any] = {"time_bins": args.time_bins}
    if args.stem_channels is not None:
        config["stem_channels"] = args.stem_channels
    if args.patch_size is not None:
        config["patch_size"] = args.patch_size
    if args.patch_dim is not None:
        config["patch_dim"] = args.patch_dim
    if name == "fast":
        model = FastWaveDSP(
            **config, bottleneck_mixer=not args.no_bottleneck_mixer, full_head=args.full_head
        )
    elif name == "fine":
        model = FineWaveDSP(
            **config,
            temporal_attention=not args.no_temporal_attention,
            relative_bias=not args.no_relative_bias,
            heads=args.attention_heads,
            token_dim=args.attention_token_dim,
            attention_chunk_size=args.attention_chunk_size,
        )
    else:
        from experiments.neuraldsp_fwl.benchmark_candidates import FullFrameNeuralDSP

        config = dict(NEURALDSP_CONFIGS[name.removeprefix("neuraldsp_")])
        model = FullFrameNeuralDSP(args.height, args.width, args.time_bins, **config)
    version = (
        model.architecture_version if name in ("fast", "fine") else "dense_neuraldsp_candidate"
    )
    if name in ("fast", "fine"):
        config = dict(model.config)
    model = model.to(device=device, dtype=dtype).eval()
    x = torch.randn(
        args.batch_size, args.height, args.width, args.time_bins, 1, device=device, dtype=dtype
    )
    output_shape = (args.batch_size, args.height, args.width, args.time_bins, 4)
    with torch.inference_mode():
        output = model(x)
        if tuple(output.shape) != output_shape or not output.is_contiguous():
            raise RuntimeError(f"Unexpected output layout: {tuple(output.shape)}")
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Non-finite output logits")
        del output
    synchronize(device)
    params = parameter_count(model)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_mib = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 2**20
    print(
        f"\n{name} [{version}]: Params [M]={params / 1e6:.6f} "
        f"({params:,} parameters; trainable={trainable_params / 1e6:.6f} M); "
        f"parameter storage={parameter_mib:.2f} MiB; {config}",
        flush=True,
    )
    print(f"Input={tuple(x.shape)}, dense logits={output_shape}")
    # predict() is decorated with inference_mode, which can hide its internal
    # dispatch from FlopCounterMode. Both paths use the same counted core ops;
    # argmax and output-layout copies are outside the operator FLOP estimate.
    flops = None if args.skip_flops else measure_flops(lambda: model(x), device)
    modes = ("logits", "labels") if args.output == "both" else (args.output,)
    rows = []
    for mode in modes:

        def invoke() -> torch.Tensor:
            if mode == "logits":
                return model(x)
            if isinstance(model, (FastWaveDSP, FineWaveDSP)):
                return model.predict(x)
            return model(x).argmax(dim=-1).to(torch.uint8)

        if args.metrics_only:
            mean = std = fps = peak = None
        else:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            mean, std = measure_speed(invoke, device, args.warmup, args.iterations, args.repeats)
            peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            fps = args.batch_size * 1000 / mean
        row = dict(
            model=name,
            architecture_version=version,
            output=mode,
            status="ok",
            error="",
            config=json.dumps(config),
            height=args.height,
            width=args.width,
            time_bins=args.time_bins,
            batch_size=args.batch_size,
            precision=args.precision,
            device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cpu_threads=torch.get_num_threads(),
            seed=42,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            parameters=params,
            parameters_m=params / 1e6,
            trainable_parameters=trainable_params,
            trainable_parameters_m=trainable_params / 1e6,
            parameter_storage_mib=parameter_mib,
            flops_per_batch=flops,
            gflops_per_batch=flops / 1e9 if flops is not None else None,
            gflops_per_sample=flops / (args.batch_size * 1e9) if flops is not None else None,
            flops_scope="counted PyTorch forward operators; MAC=2 FLOPs; pointwise ops may be omitted",
            latency_ms=mean,
            std_ms=std,
            fps=fps,
            peak_allocated_mib=peak,
        )
        rows.append(row)
        flops_text = (
            "FLOPs [G]=skipped" if flops is None
            else f"FLOPs [G]={flops / (args.batch_size * 1e9):.3f}/sample"
        )
        if flops is not None and args.batch_size > 1:
            flops_text += f" ({flops / 1e9:.3f} G/batch)"
        if args.metrics_only:
            print(f"  {mode:6s}: {flops_text}; FPS timing skipped", flush=True)
        else:
            peak_text = f"{peak:.2f} MiB" if peak is not None else "n/a"
            print(
                f"  {mode:6s}: {flops_text}; {mean:.3f} ± {std:.3f} ms/batch; "
                f"{fps:.2f} FPS; peak allocated={peak_text}",
                flush=True,
            )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA supported by this synchronized benchmark")
    if device.type == "cpu" and args.precision != "fp32":
        raise ValueError("Use --precision fp32 for CPU validation")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for small-shape validation")
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected GPU does not support BF16")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    print(f"Device={device}; PyTorch={torch.__version__}; precision={args.precision}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}")
    print(
        f"WaveDSP architecture={ARCHITECTURE_VERSION}; TF32={args.tf32}; cuDNN autotune={args.cudnn_benchmark}"
    )
    print("Eager inference with random weights; model and input use the selected dtype.")
    print(
        "Includes input layout conversion, grouped embedding, full 2D U-Net and classification head."
    )
    print("Excludes data loading, host/device transfers, checkpoint loading and input allocation.")
    print("FPS = batch size / mean batch latency; std is across repeat means.")
    print(
        "Logits includes contiguous output conversion; labels includes argmax on native GPU logits."
    )
    print("Both paths allocate native four-class logits; labels avoids their final layout copy.")
    print("Params [M] = parameter count / 1e6; FLOPs [G] = counted forward FLOPs per sample / 1e9.")
    print(f"Each sample has shape ({args.height}, {args.width}, {args.time_bins}); FPS counts these samples unchanged.")
    print("FLOPs counts PyTorch logits-forward operators; multiply-add = 2 FLOPs.")
    print("Pointwise ops, layout copies and labels argmax may be omitted; labels shares the core FLOP estimate.")
    if args.metrics_only:
        print("Metrics-only mode: repeated FPS timing is skipped.")
    names = ["fast", "fine"] if args.model == "all" else [args.model]
    if args.include_neuraldsp:
        names.append(f"neuraldsp_{args.include_neuraldsp}")
    rows: list[dict[str, Any]] = []
    failed = False
    for name in names:
        try:
            rows.extend(run_model(args, name))
        except (RuntimeError, ValueError) as exc:
            # Continue comparisons after OOM, but never report a failed model as zero latency.
            failed = True
            print(f"\n{name} FAILED: {exc}", flush=True)
            rows.append(
                dict(
                    model=name,
                    architecture_version=(
                        ARCHITECTURE_VERSION
                        if name in ("fast", "fine")
                        else "dense_neuraldsp_candidate"
                    ),
                    output=args.output,
                    status="failed",
                    error=str(exc),
                )
            )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {args.csv}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
