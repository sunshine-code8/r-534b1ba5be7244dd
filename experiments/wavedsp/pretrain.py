"""Independent reconstruction-only FAST/FINE MAE training, on one GPU or torchrun."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, Sampler
from tqdm.auto import tqdm

from experiments.wavedsp.mae_data import MAECachedDataset, MAERawDataset, check_disjoint
from experiments.wavedsp.mae_model import (
    ReconstructionWaveDSP,
    fixed_spatial_masks,
    masked_mse_parts,
    spatial_mask,
)
from experiments.wavedsp.model import ARCHITECTURE_VERSION
from experiments.wavedsp.train import create_run_dir, load_yaml, save_checkpoint_atomic


class EvaluationSampler(Sampler[int]):
    """No padding/duplicate frames in distributed validation."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def make_datasets(
    config: dict[str, Any],
    data_root: Path | None = None,
    cache_root: Path | None = None,
    raw_data: bool = False,
) -> dict[str, Dataset]:
    source = data_root or Path(config["data_root"])
    data_config = load_yaml(Path(config["data_config"]))
    size = tuple(config["target_size"])
    if raw_data:
        datasets = {s: MAERawDataset(data_config, s, size, source) for s in ("train", "valid")}
        check_disjoint(datasets)
        return datasets
    cache = cache_root or Path(config["cache_root"])
    datasets = {
        s: MAECachedDataset(cache, s, data_config, size, source) for s in ("train", "valid")
    }
    check_disjoint(datasets)
    return datasets


def dataset_identity(datasets: dict[str, Dataset]) -> dict[str, Any]:
    import hashlib

    identity = {}
    for split, dataset in datasets.items():
        identity[split] = hashlib.sha256(
            json.dumps(dataset.entries, sort_keys=True).encode()
        ).hexdigest()
    dataset = datasets["train"]
    identity["preprocessing"] = (
        dataset.meta["preprocessing"] if isinstance(dataset, MAECachedDataset) else dataset.settings
    )
    return identity


def make_loader(
    dataset: Dataset, settings: dict[str, Any], training: bool, rank: int, world_size: int
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(settings["seed"]),
            drop_last=False,
        )
        if training and world_size > 1
        else None
    )
    chosen_sampler = (
        (sampler if sampler is not None else RandomSampler(dataset, generator=torch.Generator()))
        if training
        else EvaluationSampler(dataset, rank, world_size)
    )
    workers = int(settings.get("num_workers", 8))
    if workers < 0 or int(settings.get("prefetch_factor", 1)) < 1:
        raise ValueError("num_workers must be >= 0 and prefetch_factor >= 1")
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size_per_gpu"]),
        sampler=chosen_sampler,
        shuffle=training and chosen_sampler is None,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(settings.get("prefetch_factor", 1)) if workers else None,
        generator=torch.Generator().manual_seed(int(settings["seed"]) + rank),
    )
    return loader, sampler


def validate_config(config: dict[str, Any], world_size: int) -> int:
    settings, masking = config["training"], config["masking"]
    micro = int(settings["batch_size_per_gpu"]) * world_size
    effective = int(settings["effective_batch_size"])
    if micro < 1 or effective < micro or effective % micro:
        raise ValueError(
            "effective_batch_size must be a positive multiple of batch_size_per_gpu * GPUs"
        )
    if int(settings["epochs"]) < 1 or float(settings["lr"]) <= 0:
        raise ValueError("Positive epochs and learning rate required")
    if float(settings["weight_decay"]) < 0 or float(settings["gradient_clip"]) <= 0:
        raise ValueError("Nonnegative weight_decay and positive gradient_clip required")
    if settings.get("scheduler", "none") != "none":
        raise ValueError("This baseline pretrainer uses scheduler: none")
    if config.get("loss", "masked_mse") != "masked_mse":
        raise ValueError("Only reconstruction loss masked_mse is supported")
    if len(masking["block_size_yx"]) != 2 or min(masking["block_size_yx"]) < 1:
        raise ValueError("masking.block_size_yx must contain two positive dimensions")
    if not 0 < float(masking["ratio"]) < 1 or not math.isfinite(float(masking["value"])):
        raise ValueError("Mask ratio must be in (0,1) and mask value must be finite")
    x, y, _ = config["target_size"]
    by, bx = masking["block_size_yx"]
    if math.ceil(y / by) * math.ceil(x / bx) < 2:
        raise ValueError("Mask grid needs at least two blocks")
    return effective // micro


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    rank: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    settings, masking = config["training"], config["masking"]
    blocks = tuple(int(v) for v in masking["block_size_yx"])
    mask_generator = torch.Generator(device=device).manual_seed(
        int(settings["seed"]) + epoch * 100003 + rank
    )
    if loader.generator is not None:
        loader.generator.manual_seed(int(settings["seed"]) + epoch * 100003 + rank)
    if isinstance(loader.sampler, RandomSampler):
        # Shuffle RNG is independent of worker creation, including after resume.
        loader.sampler.generator.manual_seed(int(settings["seed"]) + epoch * 100003 + rank)
    # SSE, hidden voxel count, all voxel count, and frame count. One reduction per epoch.
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    if training:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader, desc=f"{'Train' if training else 'Valid'} {epoch + 1}", disable=rank != 0
    )
    with torch.enable_grad() if training else torch.no_grad():
        for step, batch in enumerate(progress):
            # CPU stacks contiguous XYT source dtype; restore the supervised YXT convention on GPU.
            target = (
                batch["waveform"]
                .to(device, non_blocking=True)
                .float()
                .permute(0, 2, 1, 3)
                .unsqueeze(-1)
            )
            b, y, x = target.shape[:3]
            if training:
                mask = spatial_mask(
                    b, y, x, blocks, float(masking["ratio"]), device, mask_generator
                )
            else:
                mask = fixed_spatial_masks(
                    batch["sample_id"],
                    y,
                    x,
                    blocks,
                    float(masking["ratio"]),
                    int(masking["validation_seed"]),
                    device,
                )
            update = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            group_size = min(accumulation, len(loader) - step // accumulation * accumulation)
            sync = (
                model.no_sync()
                if training and isinstance(model, DDP) and not update
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and bool(settings.get("bf16", True)),
                ):
                    reconstruction = model(target, mask, float(masking["value"]))
                numerator, denominator = masked_mse_parts(reconstruction, target, mask)
                loss = numerator / denominator.clamp_min(1)
                if training:
                    (loss / group_size).backward()
            if training and update:
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["gradient_clip"]), error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals[0] += numerator.detach().double()
            totals[1] += denominator.detach().double()
            totals[2] += target.numel()
            totals[3] += b
            if rank == 0 and ((step + 1) % 20 == 0 or step + 1 == len(loader)):
                progress.set_postfix(masked_mse=float(totals[0] / totals[1].clamp_min(1)))
            del reconstruction, target, mask, loss, numerator, denominator
    if dist.is_initialized():
        dist.all_reduce(totals)
    if not torch.isfinite(totals).all() or totals[1] <= 0:
        raise RuntimeError("Nonfinite reconstruction loss or no masked validation/training voxels")
    return {
        "masked_mse": float(totals[0] / totals[1]),
        "masked_fraction": float(totals[1] / totals[2]),
        "frames": int(totals[3]),
    }


def check_resume(
    checkpoint: dict[str, Any], config: dict[str, Any], identity: dict[str, Any], world_size: int
) -> None:
    if (
        checkpoint.get("stage") != "wavedsp_mae_reconstruction"
        or checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
    ):
        raise ValueError("Not a compatible WaveDSP reconstruction checkpoint")
    previous = checkpoint["config"]
    for key in ("model", "model_kwargs", "target_size", "masking", "loss"):
        if previous.get(key) != config.get(key):
            raise ValueError(f"Cannot resume: {key} differs")
    for key in (
        "seed",
        "lr",
        "weight_decay",
        "scheduler",
        "gradient_clip",
        "effective_batch_size",
        "batch_size_per_gpu",
    ):
        if previous["training"].get(key) != config["training"].get(key):
            raise ValueError(f"Cannot resume: training.{key} differs")
    if checkpoint.get("data_identity") != identity or checkpoint.get("world_size") != world_size:
        raise ValueError("Cannot resume: dataset/preprocessing or GPU count differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--raw-data", action="store_true", help="Read waveform files directly; no peaks"
    )
    parser.add_argument(
        "--check-data", action="store_true", help="CPU-only data/mask check, no training"
    )
    parser.add_argument("--resume", type=Path, help="Continue this stage from last.pt")
    args = parser.parse_args()
    config = load_yaml(args.config)
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    accumulation = validate_config(config, world_size)
    datasets = make_datasets(config, args.data_root, args.cache_root, args.raw_data)
    identity = dataset_identity(datasets)
    if args.check_data:
        for split, dataset in datasets.items():
            sample = dataset[0]
            waveform = sample["waveform"]
            mask = fixed_spatial_masks(
                [sample["sample_id"]],
                waveform.shape[1],
                waveform.shape[0],
                tuple(config["masking"]["block_size_yx"]),
                float(config["masking"]["ratio"]),
                int(config["masking"]["validation_seed"]),
                torch.device("cpu"),
            )
            print(
                json.dumps(
                    {
                        "split": split,
                        "frames": len(dataset),
                        "sample_id": sample["sample_id"],
                        "shape_XYT": list(waveform.shape),
                        "dtype": str(waveform.dtype),
                        "masked_fraction": float(mask.float().mean()),
                    }
                )
            )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training; use --check-data for CPU-only checks")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    try:
        settings = config["training"]
        torch.manual_seed(int(settings["seed"]))
        torch.cuda.manual_seed_all(int(settings["seed"]))
        # Fail on a missing/corrupt sample before allocating the full model.
        for dataset in datasets.values():
            dataset[0]
        train_loader, sampler = make_loader(datasets["train"], settings, True, rank, world_size)
        valid_loader, _ = make_loader(datasets["valid"], settings, False, rank, world_size)
        network = ReconstructionWaveDSP(
            config["model"], int(config["target_size"][2]), **config.get("model_kwargs", {})
        ).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(settings["lr"]),
            weight_decay=float(settings["weight_decay"]),
        )
        start_epoch, best = 0, float("inf")
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            check_resume(state, config, identity, world_size)
            network.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            start_epoch, best = int(state["epoch"]), float(state["best_masked_mse"])
            if start_epoch >= int(settings["epochs"]):
                raise ValueError(
                    "Checkpoint already reached configured epochs; increase epochs to continue"
                )
            del state
        model = DDP(network, device_ids=[local_rank]) if world_size > 1 else network
        output_dir = create_run_dir(Path(config["output_dir"]), args.resume, rank)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "model": config["model"],
                        "stage": "reconstruction_only",
                        "output_dir": str(output_dir),
                        "data_source": "raw" if args.raw_data else "cache",
                        "data_identity": identity,
                        "frames": {s: len(d) for s, d in datasets.items()},
                        "GPUs": world_size,
                        "gradient_accumulation": accumulation,
                        "effective_batch_size": settings["effective_batch_size"],
                        "parameters": sum(p.numel() for p in network.parameters()),
                    }
                ),
                flush=True,
            )
        for epoch in range(start_epoch, int(settings["epochs"])):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train = run_epoch(
                model, train_loader, config, device, epoch, rank, optimizer, accumulation
            )
            # Uneven validation shards must not call DDP.forward (buffer broadcasts can deadlock).
            valid = run_epoch(network, valid_loader, config, device, epoch, rank)
            improved = valid["masked_mse"] < best
            best = min(best, valid["masked_mse"])
            if rank == 0:
                print(json.dumps({"epoch": epoch + 1, "train": train, "valid": valid}), flush=True)
                state = {
                    "stage": "wavedsp_mae_reconstruction",
                    "architecture_version": ARCHITECTURE_VERSION,
                    "epoch": epoch + 1,
                    "best_masked_mse": best,
                    "model": network.state_dict(),
                    "backbone": network.backbone_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "data_identity": identity,
                    "world_size": world_size,
                    "train_metrics": train,
                    "valid_metrics": valid,
                }
                save_checkpoint_atomic(state, output_dir / "last.pt")
                if improved:
                    save_checkpoint_atomic(state, output_dir / "best.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
