"""Benchmark independently recreated FWLMAE patch embedding and head Linear 2.

Run from the repository root:
    uv run python benchmark_standalone_patch_head2.py

No model code, configuration, dataset, or checkpoint is loaded. Modules and inputs
are newly created with the same shapes and operations as the original model.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

VOXEL_SIZE = (256, 128, 128)  # (T, H, W)
PATCH_SIZE = (256, 16, 16)
EMBED_DIM = 768
NUM_CLASSES = 4
NUM_PATCHES = (
    (VOXEL_SIZE[0] // PATCH_SIZE[0])
    * (VOXEL_SIZE[1] // PATCH_SIZE[1])
    * (VOXEL_SIZE[2] // PATCH_SIZE[2])
)
HEAD_LINEAR2_IN = EMBED_DIM // 2
HEAD_LINEAR2_OUT = NUM_CLASSES * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]


class StandalonePatchEmbedding(nn.Module):
    """Replicate VoxelPatchEmbed's Conv3d, flatten, and transpose."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1:] == (1, *VOXEL_SIZE), f"Unexpected input shape: {tuple(x.shape)}"
        return self.proj(x).flatten(2).transpose(1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50, help="Untimed warmup calls per module")
    parser.add_argument("--iterations", type=int, default=200, help="Timed calls per repeat")
    parser.add_argument("--repeats", type=int, default=5, help="Independent timed repeats")
    parser.add_argument(
        "--disable-cudnn-tf32",
        action="store_true",
        help="Disable cuDNN TF32 for a full-FP32 convolution comparison",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1 or args.repeats < 2:
        parser.error("batch-size and iterations must be positive, warmup >= 0, repeats >= 2")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_flops(fn: Callable[[], Tensor], device: torch.device) -> int:
    synchronize(device)
    # PyTorch 2.5's counter needs no_grad here; inference_mode can count zero FLOPs.
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    synchronize(device)
    return int(counter.get_total_flops())


def measure_speed(
    fn: Callable[[], Tensor], device: torch.device, warmup: int, iterations: int, repeats: int
) -> tuple[float, float, float]:
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

    mean_ms = statistics.mean(latencies_ms)
    return mean_ms, statistics.stdev(latencies_ms), 1000 / mean_ms


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu if needed")
    if args.disable_cudnn_tf32:
        torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(42)
    patch_embedding = StandalonePatchEmbedding().to(device).eval()
    head_linear2 = nn.Linear(HEAD_LINEAR2_IN, HEAD_LINEAR2_OUT).to(device).eval()
    patch_input = torch.randn(args.batch_size, 1, *VOXEL_SIZE, device=device)
    head_input = torch.randn(args.batch_size, NUM_PATCHES, HEAD_LINEAR2_IN, device=device)

    with torch.inference_mode():
        patch_output_shape = tuple(patch_embedding(patch_input).shape)
        head_output_shape = tuple(head_linear2(head_input).shape)
    assert patch_output_shape == (args.batch_size, NUM_PATCHES, EMBED_DIM)
    assert head_output_shape == (args.batch_size, NUM_PATCHES, HEAD_LINEAR2_OUT)

    cases: list[tuple[str, nn.Module, Callable[[], Tensor], tuple[int, ...], tuple[int, ...]]] = [
        (
            "Patch Embedding",
            patch_embedding,
            lambda: patch_embedding(patch_input),
            tuple(patch_input.shape),
            patch_output_shape,
        ),
        (
            "Head Linear 2",
            head_linear2,
            lambda: head_linear2(head_input),
            tuple(head_input.shape),
            head_output_shape,
        ),
    ]

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("Standalone modules: freshly initialized; no checkpoint or full model")
    print(f"Device: {device} ({device_name}); PyTorch: {torch.__version__}; dtype: FP32")
    print(
        f"Batch size: {args.batch_size}; tokens: {NUM_PATCHES}; cuDNN allow_tf32: {torch.backends.cudnn.allow_tf32}"
    )
    print(
        f"Warmup: {args.warmup} calls/module; repeats: {args.repeats}; iterations/repeat: {args.iterations}"
    )
    print("FLOPs: PyTorch operator counter, one batch; multiply-add = 2 FLOPs")
    print("FPS: synchronous eager calls/s; excludes input creation, data I/O, and transfers")
    print(
        f"{'Part':<16} {'Input -> Output':<51} {'Params (M)':>12} {'FLOPs (G)':>12} {'ms/call ± std':>19} {'FPS':>12}"
    )
    for name, module, fn, input_shape, output_shape in cases:
        flops = measure_flops(fn, device)
        mean_ms, std_ms, fps = measure_speed(fn, device, args.warmup, args.iterations, args.repeats)
        params = sum(parameter.numel() for parameter in module.parameters())
        shape_text = f"{input_shape} -> {output_shape}"
        print(
            f"{name:<16} {shape_text:<51} {params / 1e6:12.3f} {flops / 1e9:12.3f} "
            f"{mean_ms:8.3f} ± {std_ms:<7.3f} {fps:12.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
