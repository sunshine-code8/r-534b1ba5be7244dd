"""NeuralDSP backbone adapted to the FWL-MAE tensor contracts.

The backbone follows NeuralLidarDSP commit 9049e6b. The original point-return
heads are replaced by either a masked-waveform reconstruction head or a dense
four-class head. This module is deliberately kept outside ``src`` so the
original FWL-MAE model and entry points remain untouched.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TemporalAttention(nn.Module):
    """Original NeuralDSP temporal attention, including its shortcut norm."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.norm_shortcut = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim) if heads != 1 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        batch, length, dim = x.shape
        qkv = self.to_qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.to_out(out) + self.norm_shortcut(shortcut)


class WindowAttention(nn.Module):
    """Native-PyTorch equivalent of NeuralDSP's 2-D shifted window attention."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        shift: bool,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by heads={num_heads}")
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.shift_size = (window_size[0] // 2, window_size[1] // 2) if shift else (0, 0)
        window_area = window_size[0] * window_size[1]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        rows = torch.arange(window_size[0])
        cols = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(rows, cols, indexing="ij")).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size[0] - 1
        relative[:, :, 1] += window_size[1] - 1
        relative[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _partition(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        wh, ww = self.window_size
        return (
            x.reshape(batch, height // wh, wh, width // ww, ww, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, wh * ww, channels)
        )

    def _reverse(self, windows: Tensor, batch: int, height: int, width: int) -> Tensor:
        wh, ww = self.window_size
        channels = windows.shape[-1]
        return (
            windows.reshape(batch, height // wh, width // ww, wh, ww, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, channels)
        )

    def _shift_mask(self, height: int, width: int, device: torch.device) -> Tensor:
        wh, ww = self.window_size
        sh, sw = self.shift_size
        image_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
        w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
        counter = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                image_mask[:, h_slice, w_slice, :] = counter
                counter += 1
        mask_windows = self._partition(image_mask).squeeze(-1)
        mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        wh, ww = self.window_size
        if height % wh or width % ww:
            raise ValueError(
                f"feature size {(height, width)} is not divisible by window {self.window_size}"
            )
        if self.shift_size != (0, 0):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

        windows = self._partition(x)
        count, length, _ = windows.shape
        qkv = self.qkv(windows).reshape(count, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = q @ k.transpose(-2, -1) * self.scale
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(length, length, self.num_heads)
        attention = attention + relative_bias.permute(2, 0, 1).unsqueeze(0)

        if self.shift_size != (0, 0):
            mask = self._shift_mask(height, width, x.device)
            windows_per_image = mask.shape[0]
            attention = attention.reshape(batch, windows_per_image, self.num_heads, length, length)
            attention = attention + mask[None, :, None, :, :]
            attention = attention.reshape(-1, self.num_heads, length, length)

        attention = attention.softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(count, length, channels)
        out = self.proj(out)
        out = self._reverse(out, batch, height, width)
        if self.shift_size != (0, 0):
            out = torch.roll(out, shifts=self.shift_size, dims=(1, 2))
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        window_size: tuple[int, int],
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_index in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TemporalAttention(dim, heads),
                        nn.LayerNorm(dim),
                        WindowAttention(dim, window_size, heads, shift=layer_index % 2 == 1),
                        nn.LayerNorm(dim),
                        Mlp(dim, mlp_ratio * dim, dropout),
                        nn.LayerNorm(dim),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        batch, rows, cols, patches, dim = x.shape
        for temporal, norm1, spatial, norm2, mlp, norm3 in self.layers:
            shortcut = x
            x = temporal(x.reshape(batch * rows * cols, patches, dim))
            x = norm1(x.reshape(batch, rows, cols, patches, dim) + shortcut)

            shortcut = x
            x = x.permute(0, 3, 1, 2, 4).reshape(batch * patches, rows, cols, dim)
            x = spatial(x).reshape(batch, patches, rows, cols, dim)
            x = norm2(x.permute(0, 2, 3, 1, 4) + shortcut)
            x = norm3(mlp(x) + x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, resolution: tuple[int, int], dim: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: Tensor) -> Tensor:
        height, width = self.resolution
        parts = [
            x[:, 0::2, 0::2],
            x[:, 1::2, 0::2],
            x[:, 0::2, 1::2],
            x[:, 1::2, 1::2],
        ]
        x = torch.cat(parts, dim=-1).reshape(
            x.shape[0], height // 2, width // 2, -1, x.shape[-1] * 4
        )
        return self.norm(self.reduction(x))


class PatchExpanding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, patches, _ = x.shape
        x = self.expand(x)
        channels = x.shape[-1] // 4
        x = x.reshape(batch, height, width, patches, 2, 2, channels)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch, height * 2, width * 2, patches, channels)
        return self.norm(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        resolution: tuple[int, int],
        dim: int,
        depth: int,
        heads: int,
        window_size: tuple[int, int],
        mlp_ratio: int,
        dropout: float,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.transformer = TransformerBlock(dim, depth, heads, window_size, mlp_ratio, dropout)
        self.downsampler = PatchMerging(resolution, dim) if downsample else nn.Identity()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        residual = self.transformer(x)
        return self.downsampler(residual), residual


class UpBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        residual_dim: int,
        depth: int,
        heads: int,
        window_size: tuple[int, int],
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.upsampler = PatchExpanding(2 * dim)
        self.linear = nn.Linear(dim + residual_dim, dim)
        self.transformer = TransformerBlock(dim, depth, heads, window_size, mlp_ratio, dropout)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        x = self.upsampler(x)
        return self.transformer(self.linear(torch.cat([x, residual], dim=-1)))


def sinusoidal_encoding(length: int, dim: int) -> Tensor:
    encoding = torch.zeros(length, dim)
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class NeuralDSPBackbone(nn.Module):
    """NeuralDSP Swin U-Net without its original score/offset heads."""

    def __init__(
        self,
        n_samples: int = 256,
        n_channels: int = 1,
        n_rows: int = 128,
        n_cols: int = 128,
        dim: int = 32,
        waveform_patch_size: int = 32,
        heads: Sequence[int] = (2, 2, 2),
        depths: Sequence[int] = (2, 2, 2),
        window_size: tuple[int, int] = (2, 4),
        mlp_ratio: int = 2,
        dropout: Sequence[float] = (0.3, 0.3, 0.3, 0.3, 0.3),
        do_matched_filtering: bool = True,
    ) -> None:
        super().__init__()
        if n_samples % waveform_patch_size:
            raise ValueError("n_samples must be divisible by waveform_patch_size")
        if len(heads) != 3 or len(depths) != 3 or len(dropout) != 5:
            raise ValueError("heads/depths/dropout must have lengths 3/3/5")
        self.n_samples = n_samples
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.dim = dim
        self.patch_dim = waveform_patch_size
        self.num_patches = n_samples // waveform_patch_size

        self.patch_norm1 = nn.LayerNorm(n_channels * waveform_patch_size)
        self.patch_linear = nn.Linear(n_channels * waveform_patch_size, dim)
        self.patch_norm2 = nn.LayerNorm(dim)
        self.register_buffer(
            "temporal_position_encoding",
            sinusoidal_encoding(self.num_patches, dim)[None, None, None],
            persistent=True,
        )

        self.block1 = DownBlock(
            (n_rows, n_cols), dim, depths[0], heads[0], window_size, mlp_ratio, dropout[0]
        )
        self.block2 = DownBlock(
            (n_rows // 2, n_cols // 2),
            2 * dim,
            depths[1],
            heads[1],
            window_size,
            mlp_ratio,
            dropout[1],
        )
        self.block3 = DownBlock(
            (n_rows // 4, n_cols // 4),
            4 * dim,
            depths[2],
            heads[2],
            window_size,
            mlp_ratio,
            dropout[2],
            downsample=False,
        )
        self.block4 = UpBlock(
            2 * dim,
            2 * dim,
            depths[1],
            heads[1],
            window_size,
            mlp_ratio,
            dropout[3],
        )
        self.block5 = UpBlock(
            dim,
            dim,
            depths[1],
            heads[1],
            window_size,
            mlp_ratio,
            dropout[4],
        )

        self.do_matched_filtering = do_matched_filtering
        if do_matched_filtering:
            self.matched_filter_values = nn.Parameter(torch.randn(1, 1, 39))

    def _to_waveform_layout(self, voxels: Tensor) -> Tensor:
        expected = (1, self.n_samples, self.n_rows, self.n_cols)
        if tuple(voxels.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(voxels.shape)}")
        return voxels.permute(0, 3, 4, 2, 1).contiguous()

    def _patch_embed(self, waveform: Tensor) -> Tensor:
        batch, rows, cols, samples, channels = waveform.shape
        x = waveform.reshape(batch, rows, cols, self.num_patches, self.patch_dim * channels)
        return self.patch_norm2(self.patch_linear(self.patch_norm1(x)))

    def forward_features(self, voxels: Tensor, token_mask: Tensor | None = None) -> Tensor:
        waveform = self._to_waveform_layout(voxels)
        batch, rows, cols, samples, _ = waveform.shape
        if self.do_matched_filtering:
            flattened = waveform.permute(0, 1, 2, 4, 3).reshape(
                batch * rows * cols, self.n_channels, samples
            )
            flattened = F.conv1d(flattened, self.matched_filter_values, padding="same")
            waveform = flattened.reshape(batch, rows, cols, self.n_channels, samples).permute(
                0, 1, 2, 4, 3
            )

        x = self._patch_embed(waveform)
        if token_mask is not None:
            if token_mask.shape != x.shape[:-1]:
                raise ValueError(
                    f"token mask {tuple(token_mask.shape)} != token grid {tuple(x.shape[:-1])}"
                )
        x = x + self.temporal_position_encoding.to(dtype=x.dtype)
        x, residual1 = self.block1(x)
        x, residual2 = self.block2(x)
        x, _ = self.block3(x)
        x = self.block4(x, residual2)
        return self.block5(x, residual1)


def expand_macro_mask(
    macro_mask: Tensor,
    rows: int = 128,
    cols: int = 128,
    spatial_patch: tuple[int, int] = (16, 16),
    temporal_patches: int = 8,
) -> Tensor:
    patch_h, patch_w = spatial_patch
    grid_h, grid_w = rows // patch_h, cols // patch_w
    if macro_mask.shape[1] != grid_h * grid_w:
        raise ValueError(f"expected {grid_h * grid_w} macro patches, got {macro_mask.shape[1]}")
    mask = macro_mask.reshape(macro_mask.shape[0], grid_h, grid_w)
    mask = mask.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)
    return mask.unsqueeze(-1).expand(-1, -1, -1, temporal_patches)


def patchify_voxels(voxels: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    batch, channels, depth, height, width = voxels.shape
    pd, ph, pw = patch_size
    if depth % pd or height % ph or width % pw:
        raise ValueError("voxel dimensions must be divisible by patch dimensions")
    patches = voxels.reshape(
        batch,
        channels,
        depth // pd,
        pd,
        height // ph,
        ph,
        width // pw,
        pw,
    )
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return patches.reshape(batch, -1, channels * pd * ph * pw)


class NeuralDSPMAE(nn.Module):
    def __init__(
        self, backbone: NeuralDSPBackbone, spatial_patch: tuple[int, int] = (16, 16)
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.spatial_patch = spatial_patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, backbone.dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(backbone.dim), nn.Linear(backbone.dim, backbone.patch_dim)
        )
        self.patch_size = (backbone.n_samples, spatial_patch[0], spatial_patch[1])

    def forward(self, voxels: Tensor, macro_mask: Tensor) -> dict[str, Tensor]:
        waveform = self.backbone._to_waveform_layout(voxels)
        batch, rows, cols, samples, _ = waveform.shape
        if self.backbone.do_matched_filtering:
            flattened = waveform.permute(0, 1, 2, 4, 3).reshape(
                batch * rows * cols, self.backbone.n_channels, samples
            )
            flattened = F.conv1d(flattened, self.backbone.matched_filter_values, padding="same")
            waveform = flattened.reshape(
                batch, rows, cols, self.backbone.n_channels, samples
            ).permute(0, 1, 2, 4, 3)

        tokens = self.backbone._patch_embed(waveform)
        token_mask = expand_macro_mask(
            macro_mask,
            rows=self.backbone.n_rows,
            cols=self.backbone.n_cols,
            spatial_patch=self.spatial_patch,
            temporal_patches=self.backbone.num_patches,
        )
        tokens = torch.where(token_mask[..., None], self.mask_token.to(tokens.dtype), tokens)
        tokens = tokens + self.backbone.temporal_position_encoding.to(dtype=tokens.dtype)
        x, residual1 = self.backbone.block1(tokens)
        x, residual2 = self.backbone.block2(x)
        x, _ = self.backbone.block3(x)
        x = self.backbone.block4(x, residual2)
        features = self.backbone.block5(x, residual1)

        reconstruction = self.reconstruction_head(features)
        reconstruction = reconstruction.reshape(batch, rows, cols, self.backbone.n_samples)
        reconstruction = reconstruction.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        patches = patchify_voxels(reconstruction, self.patch_size)
        masked = patches[macro_mask].reshape(batch, -1, patches.shape[-1])
        return {"reconstruction": masked}


class NeuralDSPClassifier(nn.Module):
    def __init__(self, backbone: NeuralDSPBackbone, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.classification_head = nn.Sequential(
            nn.LayerNorm(backbone.dim),
            nn.Linear(backbone.dim, backbone.patch_dim * num_classes),
        )
        self.backbone_frozen = False

    def freeze_backbone(self) -> None:
        self.backbone_frozen = True
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "NeuralDSPClassifier":
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, voxels: Tensor) -> Tensor:
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.backbone.forward_features(voxels)
        else:
            features = self.backbone.forward_features(voxels)
        batch, rows, cols, temporal_patches, _ = features.shape
        logits = self.classification_head(features)
        logits = logits.reshape(
            batch,
            rows,
            cols,
            temporal_patches,
            self.backbone.patch_dim,
            self.num_classes,
        )
        logits = logits.reshape(batch, rows, cols, self.backbone.n_samples, self.num_classes)
        return logits.permute(0, 4, 3, 1, 2).contiguous()


def build_backbone(config: dict) -> NeuralDSPBackbone:
    model = config["model"]
    return NeuralDSPBackbone(
        n_samples=int(model.get("n_samples", 256)),
        n_channels=int(model.get("n_channels", 1)),
        n_rows=int(model.get("n_rows", 128)),
        n_cols=int(model.get("n_cols", 128)),
        dim=int(model.get("dim", 32)),
        waveform_patch_size=int(model.get("waveform_patch_size", 32)),
        heads=tuple(model.get("heads", [2, 2, 2])),
        depths=tuple(model.get("depths", [2, 2, 2])),
        window_size=tuple(model.get("window_size", [2, 4])),
        mlp_ratio=int(model.get("mlp_ratio", 2)),
        dropout=tuple(model.get("dropout", [0.3] * 5)),
        do_matched_filtering=bool(model.get("do_matched_filtering", True)),
    )
