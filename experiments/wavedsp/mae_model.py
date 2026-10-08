"""Reconstruction-only WaveDSP adapter; supervised model definitions stay unchanged."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn

from experiments.wavedsp.model import FastWaveDSP, FineWaveDSP


class ReconstructionWaveDSP(nn.Module):
    """Mask raw [B,Y,X,T,1] inputs and reconstruct through the entire U-Net.

    The supervised head is removed from this instance only. No unused trainable
    parameters remain, including under DDP. Backbone keys retain their original
    names for subsequent transfer to an ordinary FastWaveDSP/FineWaveDSP.
    """

    def __init__(self, kind: str, time_bins: int = 256, **kwargs: Any) -> None:
        super().__init__()
        cls = {"fast": FastWaveDSP, "fine": FineWaveDSP}.get(kind)
        if cls is None:
            raise ValueError(f"Unknown model: {kind}")
        if "layout" in kwargs or kwargs.get("auxiliary_reconstruction", False):
            raise ValueError("Pretraining uses channels_last and no embedding auxiliary head")
        self.network = cls(time_bins=time_bins, layout="channels_last", **kwargs)
        channels = self.network.config["decoder_channels"][-1]
        self.network.head = nn.Identity()
        self.reconstruction_head = nn.Conv2d(
            channels, time_bins, 1, groups=self.network.patch_count
        )

    def forward(self, waveform: Tensor, mask: Tensor, mask_value: float = 0.0) -> Tensor:
        if waveform.ndim != 5 or waveform.shape[-1] != 1:
            raise ValueError("Expected waveform [B,Y,X,T,1]")
        if mask.dtype != torch.bool or mask.shape != (*waveform.shape[:3], 1, 1):
            raise ValueError("Expected boolean spatial mask [B,Y,X,1,1]")
        masked = waveform.masked_fill(mask, mask_value)
        features = self.network.forward_features(masked)
        return self.reconstruction_head(features).permute(0, 2, 3, 1).unsqueeze(-1)

    def backbone_state_dict(self) -> dict[str, Tensor]:
        """Full shared backbone, without reconstruction or classification parameters."""
        return self.network.state_dict()


def spatial_mask(
    batch_size: int,
    height: int,
    width: int,
    block_yx: tuple[int, int],
    ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Random spatial blocks spanning all time bins; clip partial edge blocks."""
    if min(batch_size, height, width, *block_yx) < 1 or not 0 < ratio < 1:
        raise ValueError("Positive dimensions and 0 < mask_ratio < 1 required")
    gy, gx = math.ceil(height / block_yx[0]), math.ceil(width / block_yx[1])
    count = gy * gx
    if count < 2:
        raise ValueError("Mask grid needs at least two blocks (visible and masked)")
    hidden = min(count - 1, max(1, int(count * ratio)))
    order = torch.rand(batch_size, count, device=device, generator=generator).argsort(dim=1)
    grid = torch.zeros(batch_size, count, dtype=torch.bool, device=device)
    grid.scatter_(1, order[:, :hidden], True)
    mask = grid.reshape(batch_size, gy, gx)
    mask = mask.repeat_interleave(block_yx[0], 1).repeat_interleave(block_yx[1], 2)
    return mask[:, :height, :width, None, None]


def fixed_spatial_masks(
    sample_ids: list[str],
    height: int,
    width: int,
    block_yx: tuple[int, int],
    ratio: float,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Validation masks independent of loader ordering, workers, batch size and rank."""
    masks = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
        generator = torch.Generator(device=device)
        generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
        masks.append(spatial_mask(1, height, width, block_yx, ratio, device, generator))
    return torch.cat(masks)


def masked_mse_parts(reconstruction: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """FP32 masked squared-error sum and voxel count, with no boolean gather."""
    if reconstruction.shape != target.shape or target.ndim != 5 or target.shape[-1] != 1:
        raise ValueError("Reconstruction and target must have matching [B,Y,X,T,1] shapes")
    if mask.dtype != torch.bool or mask.shape != (*target.shape[:3], 1, 1):
        raise ValueError("Expected boolean spatial mask [B,Y,X,1,1]")
    error = (reconstruction.float() - target.float()).square()
    numerator = error.masked_fill(~mask, 0).sum()
    denominator = mask.sum().to(torch.float32) * target.shape[3]
    return numerator, denominator
