"""Benchmark patch embedding, encoder, head layers, and the full FWLMAE model.

Run from the repository root:
    uv run python benchmark_fwlmae.py --config configs/config_test.yaml

FLOPs are counted by PyTorch's FlopCounterMode for one batch of size 1.
Timing uses preallocated FP32 inputs, eager execution, and synchronous wall time.
Data loading, preprocessing, transfers, and prediction postprocessing are excluded.
"""

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from src.config import TestConfig, load_config_from_yaml
from src.utils import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_test.yaml")
    parser.add_argument("--device", default=None, help="Override the device in the config")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--linear1-iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--disable-cudnn-tf32",
        action="store_true",
        help="Use full FP32 precision for cuDNN convolutions",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.linear1_iterations < 1:
        parser.error("warmup must be nonnegative; iteration counts must be positive")
    if args.repeats < 2:
        parser.error("repeats must be at least 2 to calculate a standard deviation")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def encoder_after_patch(model: nn.Module, patch_tokens: Tensor) -> Tensor:
    """Run forward_features after patch_embed, including positions and normalization."""
    tokens = patch_tokens
    if model.pos_embed is not None:
        positions = model.pos_embed.expand(tokens.shape[0], -1, -1)
        tokens = tokens + positions.type_as(tokens).to(tokens.device).clone().detach()
    tokens = model.pos_drop(tokens)
    for block in model.blocks:
        tokens = block(tokens)
    tokens = model.norm(tokens)
    if model.fc_norm is not None:
        return model.fc_norm(tokens.mean(1))
    return tokens


def measure_flops(fn: Callable[[], Tensor], device: torch.device) -> int:
    synchronize(device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    synchronize(device)
    return int(counter.get_total_flops())


def measure_speed(
    fn: Callable[[], Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, float, float]:
    """Return mean latency (ms), its sample std, and throughput (calls/s)."""
    elapsed_seconds = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)

        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            synchronize(device)
            elapsed_seconds.append(time.perf_counter() - start)

    latencies_ms = [seconds * 1000 / iterations for seconds in elapsed_seconds]
    mean_latency_ms = statistics.mean(latencies_ms)
    return (
        mean_latency_ms,
        statistics.stdev(latencies_ms),
        1000 / mean_latency_ms,
    )


def main() -> None:
    args = parse_args()
    if args.disable_cudnn_tf32:
        torch.backends.cudnn.allow_tf32 = False
    config = load_config_from_yaml(args.config)
    if not isinstance(config, TestConfig) or config.model_name.lower() != "fwl_mae":
        raise ValueError("The config must describe a test-mode fwl_mae model")

    device = torch.device(args.device or config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu to run on CPU")

    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Second-stage checkpoint not found: {checkpoint_path}")

    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    model = model.to(device).eval()

    input_shape = (1, config.n_channels, *config.voxel_size)
    if list(config.voxel_size) != [config.downsample_z, config.target_size[1], config.target_size[0]]:
        raise ValueError("voxel_size does not match downsample_z and target_size in the config")
    x = torch.randn(input_shape, device=device, dtype=torch.float32)

    linears = [layer for layer in model.head.modules() if isinstance(layer, nn.Linear)]
    if len(linears) != 2:
        raise ValueError(f"Expected two head Linear layers; found {len(linears)}")
    linear1, linear2 = linears

    # Intermediate inputs are prepared once, outside the timed sections.
    with torch.inference_mode():
        patch_tokens = model.patch_embed(x)
        features = encoder_after_patch(model, patch_tokens)
        torch.testing.assert_close(features, model.forward_features(x))
        linear2_input = model.head[2](model.head[1](linear1(features)))
    synchronize(device)

    encoder_modules = nn.ModuleList([model.blocks, model.norm])
    if model.fc_norm is not None:
        encoder_modules.append(model.fc_norm)
    encoder_params = parameter_count(encoder_modules)
    if isinstance(model.pos_embed, nn.Parameter):
        encoder_params += model.pos_embed.numel()

    cases: list[tuple[str, Callable[[], Tensor], int, int]] = [
        (
            "Patch Embedding",
            lambda: model.patch_embed(x),
            parameter_count(model.patch_embed),
            args.iterations,
        ),
        (
            "Encoder",
            lambda: encoder_after_patch(model, patch_tokens),
            encoder_params,
            args.iterations,
        ),
        ("Head Linear 1", lambda: linear1(features), parameter_count(linear1), args.linear1_iterations),
        ("Head Linear 2", lambda: linear2(linear2_input), parameter_count(linear2), args.iterations),
        ("Overall", lambda: model(x), parameter_count(model), args.iterations),
    ]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"PyTorch: {torch.__version__}; dtype: FP32; batch size: 1; input: {input_shape}")
    print(f"Tokens: {features.shape[1]}; repeats: {args.repeats}; warmup: {args.warmup}")
    print(f"cuDNN allow_tf32: {torch.backends.cudnn.allow_tf32}")
    print("FLOPs: PyTorch operator counter, one input; multiply-add = 2 FLOPs")
    print("FPS: synchronous eager calls/s; excludes data I/O and preprocessing")
    print(f"{'Part':<16} {'Params (M)':>12} {'FLOPs (G)':>12} {'ms/call ± std':>19} {'FPS':>12}")

    for name, fn, params, iterations in cases:
        flops = measure_flops(fn, device)
        mean_ms, std_ms, fps = measure_speed(fn, device, args.warmup, iterations, args.repeats)
        print(f"{name:<16} {params / 1e6:12.3f} {flops / 1e9:12.3f} "
              f"{mean_ms:8.3f} ± {std_ms:<7.3f} {fps:12.2f}", flush=True)


if __name__ == "__main__":
    main()
