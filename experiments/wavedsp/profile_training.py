"""Short, checkpoint-free full-frame training profile on one selected GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from experiments.wavedsp.cache import NativeCachedView
from experiments.wavedsp.train import (
    MaskedFocalLoss,
    SupervisedWaveDSP,
    load_yaml,
    make_datasets,
    make_loader,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--native-layout", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--loss", choices=("legacy", "current"), default="current")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        raise ValueError("steps must be positive and warmup nonnegative")
    config = load_yaml(args.config)
    config["training"] = dict(config["training"])
    config["training"]["batch_size_per_gpu"] = args.batch_size
    config["training"]["num_workers"] = args.workers
    config["training"]["prefetch_factor"] = args.prefetch_factor
    dataset, _ = make_datasets(config, None)
    if args.native_layout:
        dataset = NativeCachedView(dataset)
    loader, _ = make_loader(dataset, config, True, rank=0, world_size=1)
    iterator = iter(loader)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    network = SupervisedWaveDSP(
        config["model"], config["target_size"][2], **config.get("model_kwargs", {})
    ).to(device)
    network.train()
    optimizer = AdamW(network.parameters(), lr=float(config["training"]["lr"]))
    alpha = list(config["supervised"]["focal_alpha"])
    gamma = float(config["supervised"]["focal_gamma"])
    ignore_index = int(config["supervised"].get("ignore_index", -1))
    current_loss = MaskedFocalLoss(alpha, gamma, ignore_index).to(device)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)
    accumulation = 2
    results: list[dict[str, float]] = []
    total = args.warmup + args.steps
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(total):
            started = time.perf_counter()
            batch = next(iterator)
            data_wait = time.perf_counter() - started
            boundaries = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            boundaries[0].record()
            voxels = batch["voxel_grids"].to(device, non_blocking=True).float()
            targets = batch["annotations"].to(device, non_blocking=True).long()
            if args.native_layout:
                voxels = voxels.permute(0, 4, 3, 2, 1)
                targets = targets.permute(0, 3, 2, 1).contiguous()
            boundaries[1].record()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(config["training"].get("bf16", True))):
                logits = network(voxels)
                boundaries[2].record()
                if args.loss == "legacy":
                    valid = targets != ignore_index
                    if not valid.any():
                        raise ValueError("Batch contains no valid annotation voxels")
                    ce = F.cross_entropy(logits.float(), targets, reduction="none", ignore_index=ignore_index)
                    ce = ce[valid]
                    weights = alpha_tensor[targets[valid]]
                    loss = (weights * (1 - torch.exp(-ce)).pow(gamma) * ce).mean()
                else:
                    loss = current_loss(logits, targets)
                boundaries[3].record()
            (loss / accumulation).backward()
            boundaries[4].record()
            if (step + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), float(config["training"].get("gradient_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            finish = torch.cuda.Event(enable_timing=True)
            finish.record()
            finish.synchronize()
            wall = time.perf_counter() - started
            row = {
                "step": step + 1,
                "data_wait_ms": 1000 * data_wait,
                "transfer_ms": boundaries[0].elapsed_time(boundaries[1]),
                "forward_ms": boundaries[1].elapsed_time(boundaries[2]),
                "loss_ms": boundaries[2].elapsed_time(boundaries[3]),
                "backward_ms": boundaries[3].elapsed_time(boundaries[4]),
                "optimizer_ms": boundaries[4].elapsed_time(finish),
                "wall_ms": 1000 * wall,
            }
            print(json.dumps({"warmup": step < args.warmup, **row}), flush=True)
            if step >= args.warmup:
                results.append(row)
            del batch, voxels, targets, logits, loss
    finally:
        del iterator, loader
    summary = {
        "model": config["model"],
        "loss": args.loss,
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "native_layout": args.native_layout,
        "steps": args.steps,
        "mean_ms": {
            key: statistics.mean(row[key] for row in results)
            for key in ("data_wait_ms", "transfer_ms", "forward_ms", "loss_ms", "backward_ms", "optimizer_ms", "wall_ms")
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    print(json.dumps({"summary": summary}), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "steps": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
