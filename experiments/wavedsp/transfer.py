"""Strict MAE backbone transfer, opt-in for WaveDSP supervised training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from experiments.wavedsp.mae_data import preprocessing_settings
from experiments.wavedsp.model import ARCHITECTURE_VERSION, FastWaveDSP, FineWaveDSP, WaveDSPBase


def load_mae_backbone(
    network: WaveDSPBase,
    config: dict[str, Any],
    data_config: dict[str, Any],
) -> dict[str, Any]:
    transfer = config["transfer"]
    if not isinstance(transfer, dict) or not isinstance(transfer.get("freeze_backbone"), bool):
        raise ValueError("transfer requires pretrained_checkpoint and boolean freeze_backbone")
    if not transfer.get("pretrained_checkpoint"):
        raise ValueError("transfer.pretrained_checkpoint is required")
    path = Path(transfer["pretrained_checkpoint"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("stage") != "wavedsp_mae_reconstruction"
        or checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
    ):
        raise ValueError("Expected a compatible WaveDSP MAE reconstruction checkpoint")
    previous = checkpoint["config"]
    if previous["model"] != config["model"]:
        raise ValueError("Pretrained model kind differs from the supervised model")
    if previous["target_size"] != config["target_size"]:
        raise ValueError("Pretrained target_size differs from the supervised target_size")
    expected_preprocessing = preprocessing_settings(data_config, tuple(config["target_size"]))
    if checkpoint.get("data_identity", {}).get("preprocessing") != expected_preprocessing:
        raise ValueError("Pretrained crop/time sampling differs from supervised preprocessing")
    if network.auxiliary_head is not None:
        raise ValueError(
            "MAE transfer uses the classification head only; disable auxiliary_reconstruction"
        )
    cls = {"fast": FastWaveDSP, "fine": FineWaveDSP}[config["model"]]
    # Canonical defaults allow explicit stage-two settings to match empty stage-one kwargs.
    # Preserve RNG so validation of the source configuration cannot change training randomness.
    with torch.random.fork_rng(devices=[]):
        source = cls(
            time_bins=previous["target_size"][2],
            layout="channels_last",
            **previous.get("model_kwargs", {}),
        )
    # Classification heads are newly initialized and absent from MAE backbones.
    source_backbone_config = {k: v for k, v in source.config.items() if k != "full_head"}
    target_backbone_config = {k: v for k, v in network.config.items() if k != "full_head"}
    if source_backbone_config != target_backbone_config:
        raise ValueError(
            f"Pretrained architecture differs: source={source.config}, target={network.config}"
        )
    current = network.state_dict()
    expected_keys = {key for key in current if not key.startswith("head.")}
    backbone = checkpoint["backbone"]
    if set(backbone) != expected_keys:
        raise ValueError(
            f"Backbone keys differ: missing={sorted(expected_keys - set(backbone))}, "
            f"unexpected={sorted(set(backbone) - expected_keys)}"
        )
    for key, value in backbone.items():
        if not isinstance(value, torch.Tensor) or value.shape != current[key].shape:
            raise ValueError(f"Backbone shape mismatch: {key}")
    # Only the newly initialized classification head is retained.
    network.load_state_dict({**current, **backbone}, strict=True)
    return {
        "pretrained_checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pretrain_epoch": int(checkpoint["epoch"]),
        "pretrain_best_masked_mse": float(checkpoint["best_masked_mse"]),
        "freeze_backbone": transfer["freeze_backbone"],
        "loaded_backbone_tensors": len(backbone),
        "model_config": network.config,
    }


def check_transfer_resume(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    """Do not resume a different optimization strategy, even when weights have identical shapes."""
    previous = checkpoint.get("config", {}).get("transfer")
    current = config.get("transfer")
    if bool(previous) != bool(current):
        raise ValueError("Cannot resume: transfer strategy differs from the checkpoint")
    if current:
        if not isinstance(current.get("freeze_backbone"), bool):
            raise ValueError("transfer.freeze_backbone must be a boolean")
        if previous.get("freeze_backbone") != current["freeze_backbone"]:
            raise ValueError("Cannot resume: freeze_backbone differs from the checkpoint")
        # Resume uses the complete stage-two state, not a new stage-one initialization.
        if previous.get("pretrained_checkpoint") != current.get("pretrained_checkpoint"):
            raise ValueError("Cannot resume: pretrained_checkpoint differs from the checkpoint")
