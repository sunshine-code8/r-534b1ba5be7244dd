"""Synthetic two-GPU smoke test for the isolated experiment stack."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments.neuraldsp_fwl.model import NeuralDSPClassifier, build_backbone


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"expected two processes, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    config = {
        "model": {
            "n_samples": 256,
            "n_channels": 1,
            "n_rows": 128,
            "n_cols": 128,
            "dim": 32,
            "waveform_patch_size": 32,
            "heads": [2, 2, 2],
            "depths": [2, 2, 2],
            "window_size": [2, 4],
            "mlp_ratio": 2,
            "dropout": [0.3] * 5,
            "do_matched_filtering": True,
        }
    }
    model = DDP(
        NeuralDSPClassifier(build_backbone(config)).to(device),
        device_ids=[local_rank],
        output_device=local_rank,
    )
    voxels = torch.randn(1, 1, 256, 128, 128, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(voxels)
        loss = logits.float().mean()
    loss.backward()
    result = torch.tensor([float(torch.isfinite(logits).all())], device=device)
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        print(f"two_gpu_smoke_ok={int(result.item()) == world_size}; shape={tuple(logits.shape)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
