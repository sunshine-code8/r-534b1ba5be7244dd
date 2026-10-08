"""Full-frame Ghost supervised training for FastWaveDSP and FineWaveDSP.

Run from the repository root with ``uv run python -m experiments.wavedsp.train --config ...``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from experiments.wavedsp import FastWaveDSP, FineWaveDSP
from experiments.wavedsp.cache import GhostCachedDataset, NativeCachedView
from experiments.wavedsp.data import GhostSupervisedDataset


class SupervisedWaveDSP(nn.Module):
    """Adapt [B,1,T,H,W] Ghost batches to the existing WaveDSP API."""

    def __init__(self, kind: str, time_bins: int, **kwargs: Any) -> None:
        super().__init__()
        network = {"fast": FastWaveDSP, "fine": FineWaveDSP}.get(kind)
        if network is None:
            raise ValueError(f"Unknown model: {kind}")
        self.network = network(time_bins=time_bins, layout="channels_last", **kwargs)
        self.backbone_frozen = False

    def set_backbone_frozen(self, frozen: bool) -> None:
        self.backbone_frozen = frozen
        for name, parameter in self.network.named_parameters():
            parameter.requires_grad_(not frozen or name.startswith("head."))
        self.train(self.training)

    def train(self, mode: bool = True) -> "SupervisedWaveDSP":
        super().train(mode)
        if self.backbone_frozen:
            self.network.eval()
            self.network.head.train(mode)
        return self

    def forward(self, voxels: Tensor) -> Tensor:
        # [B,1,T,H,W] -> [B,H,W,T,1] -> [B,H,W,T,4] -> [B,4,T,H,W]
        inputs = voxels.permute(0, 3, 4, 2, 1)
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.network.forward_features(inputs)
            logits = self.network.head.format_logits(self.network.head(features), self.network.layout)
            return logits.permute(0, 4, 3, 1, 2)
        return self.network(inputs).permute(0, 4, 3, 1, 2)


class MaskedFocalLoss(nn.Module):
    """Same focal formula as NeuralDSP training, averaged over valid voxels only."""

    def __init__(self, alpha: list[float], gamma: float, ignore_index: int = -1) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Keep shapes fixed: boolean indexing would invoke CUDA nonzero and synchronize
        # the host on every full-frame batch. Ignored voxels contribute zero.
        ce = F.cross_entropy(
            logits.float(), targets, reduction="none", ignore_index=self.ignore_index
        )
        valid = targets != self.ignore_index
        safe_targets = targets.clamp(0, self.alpha.numel() - 1)
        weights = self.alpha[safe_targets]
        focal = weights * (1 - torch.exp(-ce)).pow(self.gamma) * ce
        return (focal * valid).sum() / valid.sum().clamp_min(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, help="Ghost dataset root if not at data/ghost_dataset"
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Inspect one sample per split without CUDA"
    )
    parser.add_argument("--cache-root", type=Path, help="Override the completed preprocessing cache path")
    parser.add_argument("--raw-data", action="store_true", help="Use original Ghost files instead of the configured cache")
    parser.add_argument("--resume", type=Path, help="Resume from a last.pt checkpoint")
    parser.add_argument("--check-transfer", action="store_true", help="CPU-only check of MAE loading and freezing; no training")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def make_datasets(
    config: dict[str, Any], data_root: Path | None,
    cache_root: Path | None = None, raw_data: bool = False,
) -> tuple[GhostSupervisedDataset | GhostCachedDataset, GhostSupervisedDataset | GhostCachedDataset]:
    data_config = load_yaml(Path(config["data_config"]))
    size = tuple(int(value) for value in config["target_size"])
    if len(size) != 3 or min(size) <= 0:
        raise ValueError("target_size must be positive [X,Y,T]")
    classes = int(config["supervised"]["num_classes"])
    ignore_index = int(config["supervised"].get("ignore_index", -1))
    selected_cache = None if raw_data else (cache_root or config.get("cache_root"))
    if selected_cache is not None:
        train = GhostCachedDataset(Path(selected_cache), "train", size, classes, ignore_index, data_config)
        valid = GhostCachedDataset(Path(selected_cache), "valid", size, classes, ignore_index, data_config)
        overlap = {item["voxel"] for item in train.entries} & {item["voxel"] for item in valid.entries}
    else:
        train = GhostSupervisedDataset(data_config, "train", size, data_root, classes, ignore_index)
        valid = GhostSupervisedDataset(data_config, "valid", size, data_root, classes, ignore_index)
        overlap = {path for path, _ in train.pairs} & {path for path, _ in valid.pairs}
    if overlap:
        raise ValueError(f"Train and validation share voxel files: {sorted(overlap)[:3]}")
    return train, valid


def inspect_data(datasets: tuple[GhostSupervisedDataset | GhostCachedDataset, GhostSupervisedDataset | GhostCachedDataset]) -> None:
    for name, dataset in zip(("train", "valid"), datasets):
        sample = dataset[0]
        labels = sample["annotations"]
        print(
            json.dumps(
                {
                    "split": name,
                    "samples": len(dataset),
                    "frame_id": sample["frame_id"],
                    "voxel_shape": list(sample["voxel_grids"].shape),
                    "annotation_shape": list(labels.shape),
                    "classes_in_sample": torch.unique(labels).tolist(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def make_loader(
    dataset: GhostSupervisedDataset | GhostCachedDataset | NativeCachedView,
    config: dict[str, Any],
    training: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    settings = config["training"]
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=training, drop_last=training
        )
        if world_size > 1
        else None
    )
    workers = int(settings.get("num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size_per_gpu"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(settings.get("prefetch_factor", 1)) if workers > 0 else None,
        drop_last=False,
    )
    if not len(loader):
        raise ValueError("No batches; reduce batch size or number of GPUs")
    return loader, sampler


def make_confusion(logits: Tensor, targets: Tensor, classes: int, ignore_index: int) -> Tensor:
    predicted = logits.argmax(dim=1)
    valid = targets != ignore_index
    encoded = targets[valid] * classes + predicted[valid]
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def score(confusion: Tensor) -> dict[str, float]:
    tp = confusion.diag()
    fp = confusion.sum(0) - tp
    fn = confusion.sum(1) - tp
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    iou = tp / (tp + fp + fn).clamp_min(1)
    return {"macro_f1": float(f1.mean()), "miou": float(iou.mean())}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MaskedFocalLoss,
    device: torch.device,
    config: dict[str, Any],
    optimizer: AdamW | None = None,
    epoch: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    settings = config["training"]
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    micro_batch = int(settings["batch_size_per_gpu"]) * world_size
    effective_batch = int(settings["effective_batch_size"])
    if effective_batch < micro_batch or effective_batch % micro_batch:
        raise ValueError(
            f"effective_batch_size={effective_batch} must be a positive multiple of "
            f"batch_size_per_gpu={settings['batch_size_per_gpu']} * GPUs={world_size} "
            f"(micro-batch={micro_batch}); reduce batch_size_per_gpu or raise "
            "effective_batch_size"
        )
    accumulation = effective_batch // micro_batch
    if training:
        optimizer.zero_grad(set_to_none=True)
    classes = int(config["supervised"]["num_classes"])
    ignore_index = int(config["supervised"].get("ignore_index", -1))
    confusion = torch.zeros(classes, classes, dtype=torch.float64, device=device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    grad_context = contextlib.nullcontext() if training else torch.no_grad()
    show_progress = not dist.is_initialized() or dist.get_rank() == 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{settings['epochs']} {'Train' if training else 'Valid'}",
        disable=not show_progress,
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
        unit="batch",
    )
    native_cache_layout = isinstance(loader.dataset, NativeCachedView)
    with grad_context:
        for step, batch in enumerate(progress):
            voxels = batch["voxel_grids"].to(device, non_blocking=True).float()
            targets = batch["annotations"].to(device, non_blocking=True).long()
            if native_cache_layout:
                # Collate contiguous [X,Y,T] on CPU; restore [1,T,Y,X] on GPU.
                voxels = voxels.permute(0, 4, 3, 2, 1)
                targets = targets.permute(0, 3, 2, 1).contiguous()
            update = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            sync = (
                model.no_sync()
                if training and isinstance(model, DDP) and not update
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=bool(settings.get("bf16", True))
                ):
                    logits = model(voxels)
                    loss = loss_fn(logits, targets)
                if training:
                    (loss / group_size).backward()
            if training:
                if update:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(settings.get("gradient_clip", 1.0))
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                confusion += make_confusion(logits.detach(), targets, classes, ignore_index)
            loss_sum += loss.detach().double() * voxels.shape[0]
            count += voxels.shape[0]
            if show_progress and ((step + 1) % 20 == 0 or step + 1 == len(loader)):
                progress.set_postfix(avg_loss=f"{(loss_sum / count).item():.4f}", refresh=False)
            del voxels, targets, logits, loss
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(count)
        if not training:
            dist.all_reduce(confusion)
    result = {"loss": float(loss_sum / count)}
    if not training:
        result.update(score(confusion))
    return result


def create_run_dir(base_dir: Path, resume: Path | None, rank: int) -> Path:
    """Create one unique per-launch directory and share it among DDP ranks."""
    path: str | None = None
    if rank == 0:
        base_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now().astimezone()
        run_dir = base_dir / started.strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(exist_ok=False)
        if resume is not None:
            shutil.copy2(resume, run_dir / "last.pt")
            previous_best = resume.parent / "best.pt"
            if previous_best.is_file():
                shutil.copy2(previous_best, run_dir / "best.pt")
        with (run_dir / "run_info.json").open("w") as handle:
            json.dump(
                {"started_at": started.isoformat(),
                 "resumed_from": str(resume.resolve()) if resume else None},
                handle, indent=2,
            )
            handle.write("\n")
        path = str(run_dir)
    if dist.is_initialized():
        paths = [path]
        dist.broadcast_object_list(paths, src=0)
        path = paths[0]
    assert path is not None
    return Path(path)


def save_checkpoint_atomic(state: dict[str, Any], path: Path) -> None:
    """Publish a complete checkpoint even if training is interrupted during save."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.check_transfer:
        if args.resume or not config.get("transfer"):
            raise ValueError("--check-transfer requires transfer config and cannot be combined with --resume")
        from experiments.wavedsp.transfer import load_mae_backbone
        network = SupervisedWaveDSP(config["model"], config["target_size"][2], **config.get("model_kwargs", {}))
        info = load_mae_backbone(network.network, config, load_yaml(Path(config["data_config"])))
        network.set_backbone_frozen(config["transfer"]["freeze_backbone"])
        print(json.dumps({**info, "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                          "frozen_parameters": sum(p.numel() for p in network.parameters() if not p.requires_grad),
                          "trainable_names": [name for name, p in network.named_parameters() if p.requires_grad]}, ensure_ascii=False, indent=2))
        return
    datasets = make_datasets(config, args.data_root, args.cache_root, args.raw_data)
    if args.check_data:
        inspect_data(datasets)
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for full-frame training; use --check-data for data inspection"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    try:
        seed = int(config["training"]["seed"]) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Check actual preprocessing and label values before allocating the model.
        datasets[0][0]
        datasets[1][0]
        using_cache = isinstance(datasets[0], GhostCachedDataset)
        native_cache_layout = using_cache and bool(config["training"].get("native_cache_layout", True))
        if native_cache_layout:
            datasets = (NativeCachedView(datasets[0]), NativeCachedView(datasets[1]))
        train_loader, train_sampler = make_loader(datasets[0], config, True, rank, world_size)
        valid_loader, _ = make_loader(datasets[1], config, False, rank, world_size)
        network = SupervisedWaveDSP(
            str(config["model"]), int(config["target_size"][2]), **config.get("model_kwargs", {})
        ).to(device)
        transfer_info = None
        if config.get("transfer"):
            from experiments.wavedsp.transfer import load_mae_backbone
            if not args.resume:
                transfer_info = load_mae_backbone(network.network, config, load_yaml(Path(config["data_config"])))
            network.set_backbone_frozen(config["transfer"]["freeze_backbone"])
        model: nn.Module = (
            DDP(network, device_ids=[local_rank], output_device=local_rank)
            if world_size > 1
            else network
        )
        settings = config["training"]
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(settings["lr"]),
            weight_decay=float(settings["weight_decay"]),
        )
        epochs = int(settings["epochs"])
        scheduler = None
        if settings.get("scheduler", "none") == "cosine":
            warmup = int(settings.get("warmup_epochs", 0))
            min_lr = float(settings.get("min_lr", 1e-6))
            lr = float(settings["lr"])

            def multiplier(epoch: int) -> float:
                if warmup and epoch < warmup:
                    return (epoch + 1) / warmup
                fraction = min(1.0, (epoch - warmup) / max(1, epochs - warmup - 1))
                cosine = 0.5 * (1 + math.cos(math.pi * fraction))
                return min_lr / lr + (1 - min_lr / lr) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        elif settings.get("scheduler", "none") != "none":
            raise ValueError("scheduler must be none or cosine")
        loss_fn = MaskedFocalLoss(
            list(config["supervised"]["focal_alpha"]),
            float(config["supervised"]["focal_gamma"]),
            int(config["supervised"].get("ignore_index", -1)),
        ).to(device)
        start_epoch, best = 0, -float("inf")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            from experiments.wavedsp.transfer import check_transfer_resume
            check_transfer_resume(checkpoint, config)
            transfer_info = checkpoint.get("transfer_info")
            previous_config = checkpoint.get("config", {})
            for key in ("model", "model_kwargs", "target_size"):
                previous_value, current_value = previous_config.get(key), config.get(key)
                if key == "model_kwargs" and config["model"] == "fast":
                    # Older checkpoints omit the equivalent full_head=False default.
                    previous_value = {"full_head": False, **(previous_value or {})}
                    current_value = {"full_head": False, **(current_value or {})}
                if previous_value != current_value:
                    raise ValueError(f"Cannot resume: checkpoint {key} differs from current config")
            network.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"])
            best = float(checkpoint["best_macro_f1"])
        output_dir = create_run_dir(Path(config["output_dir"]), args.resume, rank)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "model": config["model"],
                        "output_dir": str(output_dir),
                        "resume_from": str(args.resume) if args.resume else None,
                        "start_epoch": start_epoch + 1,
                        "data_source": "cache" if using_cache else "raw",
                        "cache_layout": "native_XYT" if native_cache_layout else "legacy_TYX",
                        "train_frames": len(datasets[0]),
                        "valid_frames": len(datasets[1]),
                        "target_size_XYT": config["target_size"],
                        "GPUs": world_size,
                        "parameters": sum(p.numel() for p in network.parameters()),
                        "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                        "transfer_info": transfer_info,
                    }
                ),
                flush=True,
            )
        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model, train_loader, loss_fn, device, config, optimizer, epoch
            )
            valid_metrics = run_epoch(model, valid_loader, loss_fn, device, config, epoch=epoch)
            if scheduler is not None:
                scheduler.step()
            if rank == 0:
                print(
                    json.dumps(
                        {"epoch": epoch + 1, "train_loss": train_metrics["loss"], **valid_metrics}
                    ),
                    flush=True,
                )
                improved = valid_metrics["macro_f1"] > best
                best = max(best, valid_metrics["macro_f1"])
                state = {
                    "epoch": epoch + 1,
                    "best_macro_f1": best,
                    "model": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "config": config,
                }
                if config.get("transfer"):
                    state["stage"] = "wavedsp_supervised_transfer"
                    state["transfer_info"] = transfer_info
                save_checkpoint_atomic(state, output_dir / "last.pt")
                if improved:
                    save_checkpoint_atomic(state, output_dir / "best.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
