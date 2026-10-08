"""Two-GPU-capable training entry point for isolated NeuralDSP/FWL experiments."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pathlib
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from experiments.neuraldsp_fwl.data import (
    VoxelOnlyDataset,
    build_supervised_dataset,
    seed_worker,
    supervised_collate,
)
from experiments.neuraldsp_fwl.model import (
    NeuralDSPClassifier,
    NeuralDSPMAE,
    build_backbone,
    patchify_voxels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    return parser.parse_args()


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("These experiments require CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, message: str | dict[str, Any]) -> None:
    if is_main(rank):
        print(
            json.dumps(message, ensure_ascii=False) if isinstance(message, dict) else message,
            flush=True,
        )


def reduce_pair(value: float, count: int, device: torch.device) -> tuple[float, int]:
    pair = torch.tensor([value, float(count)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float(pair[0].item()), int(pair[1].item())


def make_macro_mask(
    batch_size: int,
    num_patches: int,
    ratio: float,
    device: torch.device,
    sample_indices: Tensor | None = None,
    seed: int = 42,
) -> Tensor:
    num_masked = int(num_patches * ratio)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    if sample_indices is None:
        selected = (
            torch.rand(batch_size, num_patches, device=device).topk(num_masked, dim=1).indices
        )
    else:
        rows = []
        for sample_index in sample_indices.tolist():
            generator = torch.Generator().manual_seed(seed + int(sample_index))
            rows.append(torch.randperm(num_patches, generator=generator)[:num_masked])
        selected = torch.stack(rows).to(device)
    return mask.scatter_(1, selected, True)


class FocalLoss(nn.Module):
    def __init__(self, alpha: list[float], gamma: float, ignore_index: int) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        valid = targets != self.ignore_index
        weights = torch.zeros_like(ce)
        weights[valid] = self.alpha[targets[valid]]
        focal = weights * (1.0 - torch.exp(-ce)).pow(self.gamma) * ce
        return focal[valid].mean()


def build_loader(
    dataset: torch.utils.data.Dataset,
    config: dict[str, Any],
    training: bool,
    world_size: int,
    rank: int,
    collate_fn: Any = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            drop_last=training,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size_per_gpu"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(config["training"].get("num_workers", 8)),
        pin_memory=True,
        persistent_workers=int(config["training"].get("num_workers", 8)) > 0,
        drop_last=training,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    return loader, sampler


def cosine_with_warmup(
    optimizer: AdamW, epochs: int, warmup: int, min_lr: float, lr: float
) -> LambdaLR:
    def multiplier(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr / lr + (1.0 - min_lr / lr) * cosine

    return LambdaLR(optimizer, multiplier)


def save_checkpoint(
    path: pathlib.Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    metric: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "epoch": epoch,
            "metric": metric,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_pretrained_backbone(model: NeuralDSPClassifier, path: pathlib.Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"pretraining checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefix = "backbone."
    backbone_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not backbone_state:
        raise RuntimeError(f"{path} contains no {prefix} weights")
    model.backbone.load_state_dict(backbone_state, strict=True)


def optimizer_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    return parameters


def autocast_context(enabled: bool) -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def gradient_accumulation_steps(config: dict[str, Any]) -> int:
    training = config["training"]
    if "effective_batch_size" not in training:
        return int(training.get("gradient_accumulation", 1))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    micro_batch = int(training["batch_size_per_gpu"]) * world_size
    effective_batch = int(training["effective_batch_size"])
    if effective_batch < micro_batch or effective_batch % micro_batch != 0:
        raise ValueError(
            f"effective_batch_size={effective_batch} must be a positive multiple of "
            f"batch_size_per_gpu * world_size = {micro_batch}"
        )
    return effective_batch // micro_batch


def progress_bar(loader: DataLoader, description: str) -> tqdm:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return tqdm(
        loader,
        desc=description,
        disable=rank != 0,
        dynamic_ncols=True,
        leave=False,
    )


def train_pretrain_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    config: dict[str, Any],
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = gradient_accumulation_steps(config)
    ratio = float(config["pretrain"]["mask_ratio"])
    patch_size = tuple(config["pretrain"]["patch_size"])
    num_patches = (256 // patch_size[0]) * (128 // patch_size[1]) * (128 // patch_size[2])
    total_loss = 0.0
    total_samples = 0

    progress = progress_bar(loader, "Pretrain")
    for step, batch in enumerate(progress):
        voxels = batch["voxels"].to(device, non_blocking=True)
        mask = make_macro_mask(voxels.shape[0], num_patches, ratio, device)
        synchronize = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        sync_context = (
            contextlib.nullcontext()
            if synchronize or not isinstance(model, DDP)
            else model.no_sync()
        )
        with sync_context:
            with autocast_context(bool(config["training"].get("bf16", True))):
                output = model(voxels, mask)["reconstruction"]
                targets = patchify_voxels(voxels, patch_size)
                targets = targets[mask].reshape(voxels.shape[0], -1, targets.shape[-1])
                loss = F.mse_loss(output.float(), targets.float())
            (loss / accumulation).backward()
        if synchronize:
            torch.nn.utils.clip_grad_norm_(
                optimizer_parameters(model), float(config["training"].get("gradient_clip", 1.0))
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * voxels.shape[0]
        total_samples += voxels.shape[0]
        progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.6f}")
    reduced_loss, reduced_samples = reduce_pair(total_loss, total_samples, device)
    return reduced_loss / max(1, reduced_samples)


@torch.no_grad()
def validate_pretrain(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> float:
    model.eval()
    ratio = float(config["pretrain"]["mask_ratio"])
    patch_size = tuple(config["pretrain"]["patch_size"])
    num_patches = (256 // patch_size[0]) * (128 // patch_size[1]) * (128 // patch_size[2])
    total_loss = 0.0
    total_samples = 0
    progress = progress_bar(loader, "Validate")
    for batch in progress:
        voxels = batch["voxels"].to(device, non_blocking=True)
        indices = batch["indices"]
        mask = make_macro_mask(
            voxels.shape[0],
            num_patches,
            ratio,
            device,
            sample_indices=indices,
            seed=int(config["training"]["seed"]),
        )
        with autocast_context(bool(config["training"].get("bf16", True))):
            output = model(voxels, mask)["reconstruction"]
            targets = patchify_voxels(voxels, patch_size)
            targets = targets[mask].reshape(voxels.shape[0], -1, targets.shape[-1])
            loss = F.mse_loss(output.float(), targets.float())
        total_loss += float(loss) * voxels.shape[0]
        total_samples += voxels.shape[0]
        progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.6f}")
    reduced_loss, reduced_samples = reduce_pair(total_loss, total_samples, device)
    return reduced_loss / max(1, reduced_samples)


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = gradient_accumulation_steps(config)
    total_loss = 0.0
    total_samples = 0
    progress = progress_bar(loader, "Supervised")
    for step, batch in enumerate(progress):
        voxels = batch["voxel_grids"].to(device, non_blocking=True)
        targets = batch["annotations"].to(device, non_blocking=True)
        synchronize = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        sync_context = (
            contextlib.nullcontext()
            if synchronize or not isinstance(model, DDP)
            else model.no_sync()
        )
        with sync_context:
            with autocast_context(bool(config["training"].get("bf16", True))):
                logits = model(voxels)
                loss = loss_fn(logits.float(), targets)
            (loss / accumulation).backward()
        if synchronize:
            torch.nn.utils.clip_grad_norm_(
                optimizer_parameters(model), float(config["training"].get("gradient_clip", 1.0))
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * voxels.shape[0]
        total_samples += voxels.shape[0]
        progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.6f}")
    reduced_loss, reduced_samples = reduce_pair(total_loss, total_samples, device)
    return reduced_loss / max(1, reduced_samples)


@torch.no_grad()
def validate_supervised(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    num_classes = int(config["supervised"]["num_classes"])
    ignore_index = int(config["supervised"].get("ignore_index", -1))
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    total_loss = 0.0
    total_samples = 0
    progress = progress_bar(loader, "Validate")
    for batch in progress:
        voxels = batch["voxel_grids"].to(device, non_blocking=True)
        targets = batch["annotations"].to(device, non_blocking=True)
        with autocast_context(bool(config["training"].get("bf16", True))):
            logits = model(voxels)
            loss = loss_fn(logits.float(), targets)
        predictions = logits.argmax(dim=1)
        valid = targets != ignore_index
        encoded = targets[valid] * num_classes + predictions[valid]
        confusion += torch.bincount(encoded, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
        total_loss += float(loss) * voxels.shape[0]
        total_samples += voxels.shape[0]
        progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.6f}")

    if dist.is_initialized():
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    reduced_loss, reduced_samples = reduce_pair(total_loss, total_samples, device)
    true_positive = confusion.diag()
    false_positive = confusion.sum(0) - true_positive
    false_negative = confusion.sum(1) - true_positive
    iou = true_positive / (true_positive + false_positive + false_negative).clamp_min(1)
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "loss": reduced_loss / max(1, reduced_samples),
        "miou": float(iou.mean()),
        "macro_f1": float(f1.mean()),
    }


def run_pretrain(
    config: dict[str, Any],
    data_config: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    seed = int(config["training"]["seed"])
    train_dataset = VoxelOnlyDataset(
        data_config["train_voxel_dirs"], data_config, training=True, seed=seed
    )
    valid_dataset = VoxelOnlyDataset(
        data_config["valid_voxel_dirs"], data_config, training=False, seed=seed
    )
    train_loader, train_sampler = build_loader(train_dataset, config, True, world_size, rank)
    valid_loader, _ = build_loader(valid_dataset, config, False, world_size, rank)

    model: nn.Module = NeuralDSPMAE(
        build_backbone(config), spatial_patch=tuple(config["pretrain"]["patch_size"][1:])
    ).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index)
    optimizer = AdamW(
        optimizer_parameters(model),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    output_dir = pathlib.Path(config["output_dir"])
    best = float("inf")
    epochs = int(config["training"]["epochs"])
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_pretrain_epoch(model, train_loader, optimizer, device, config)
        valid_loss = validate_pretrain(model, valid_loader, device, config)
        log(rank, {"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
        if is_main(rank):
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, valid_loss, config)
            if valid_loss < best:
                best = valid_loss
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1, best, config)


def run_supervised(
    config: dict[str, Any],
    data_config: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    train_dataset = build_supervised_dataset(data_config, training=True)
    valid_dataset = build_supervised_dataset(data_config, training=False)
    train_loader, train_sampler = build_loader(
        train_dataset, config, True, world_size, rank, supervised_collate
    )
    valid_loader, _ = build_loader(
        valid_dataset, config, False, world_size, rank, supervised_collate
    )

    classifier = NeuralDSPClassifier(
        build_backbone(config), num_classes=int(config["supervised"]["num_classes"])
    )
    pretrained = config["supervised"].get("pretrained_backbone")
    if pretrained:
        load_pretrained_backbone(classifier, pathlib.Path(pretrained))
    if bool(config["supervised"].get("freeze_backbone", False)):
        if not pretrained:
            raise ValueError("freeze_backbone=true requires pretrained_backbone")
        classifier.freeze_backbone()
    classifier = classifier.to(device)

    trainable = sum(
        parameter.numel() for parameter in classifier.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in classifier.parameters())
    log(rank, {"total_parameters": total, "trainable_parameters": trainable})
    model: nn.Module = classifier
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    optimizer = AdamW(
        optimizer_parameters(model),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = None
    if config["training"].get("scheduler", "none") == "cosine":
        scheduler = cosine_with_warmup(
            optimizer,
            int(config["training"]["epochs"]),
            int(config["training"].get("warmup_epochs", 0)),
            float(config["training"].get("min_lr", 1e-6)),
            float(config["training"]["lr"]),
        )
    loss_fn = FocalLoss(
        alpha=list(config["supervised"]["focal_alpha"]),
        gamma=float(config["supervised"]["focal_gamma"]),
        ignore_index=int(config["supervised"].get("ignore_index", -1)),
    ).to(device)

    output_dir = pathlib.Path(config["output_dir"])
    best = -float("inf")
    epochs = int(config["training"]["epochs"])
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_supervised_epoch(model, train_loader, optimizer, loss_fn, device, config)
        metrics = validate_supervised(model, valid_loader, loss_fn, device, config)
        current_lr = optimizer.param_groups[0]["lr"]
        log(rank, {"epoch": epoch + 1, "lr": current_lr, "train_loss": train_loss, **metrics})
        if is_main(rank):
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, epoch + 1, metrics["macro_f1"], config
            )
            if metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1, best, config)
        if scheduler is not None:
            scheduler.step()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = load_yaml(pathlib.Path(config["data_config"]))
    rank, local_rank, world_size, device = setup_distributed()
    try:
        set_seed(int(config["training"]["seed"]), rank)
        log(
            rank,
            {
                "config": str(args.config),
                "task": config["task"],
                "world_size": world_size,
                "local_rank": local_rank,
                "gradient_accumulation": gradient_accumulation_steps(config),
                "effective_batch_size": int(config["training"]["batch_size_per_gpu"])
                * world_size
                * gradient_accumulation_steps(config),
            },
        )
        if config["task"] == "pretrain":
            run_pretrain(config, data_config, rank, world_size, device)
        elif config["task"] == "supervised":
            run_supervised(config, data_config, rank, world_size, device)
        else:
            raise ValueError(f"unknown task: {config['task']}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
