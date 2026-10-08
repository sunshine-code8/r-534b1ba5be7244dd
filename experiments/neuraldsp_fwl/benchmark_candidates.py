"""Full-frame NeuralDSP candidates used only by the standalone benchmark.

Input: [B,H,W,T,1]. Output: contiguous [B,H,W,T,4] logits.
Training models and entry points are deliberately independent of this module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import Mlp, TemporalAttention, WindowAttention, sinusoidal_encoding


class PaddedWindowAttention(WindowAttention):
    """Window attention with fixed-resolution, cached shift and padding masks."""

    def __init__(
        self,
        dim: int,
        resolution: tuple[int, int],
        window_size: tuple[int, int],
        heads: int,
        shift: bool,
    ) -> None:
        super().__init__(dim, window_size, heads, shift)
        self.height, self.width = resolution
        wh, ww = window_size
        self.padded_height = (self.height + wh - 1) // wh * wh
        self.padded_width = (self.width + ww - 1) // ww * ww
        hp, wp = self.padded_height, self.padded_width
        length = wh * ww
        mask = torch.zeros(hp // wh * (wp // ww), length, length)
        if shift:
            mask = self._shift_mask(hp, wp, torch.device("cpu"))
        if (hp, wp) != resolution:
            valid = torch.zeros(1, hp, wp, 1, dtype=torch.bool)
            valid[:, : self.height, : self.width] = True
            if shift:
                valid = torch.roll(valid, shifts=tuple(-s for s in self.shift_size), dims=(1, 2))
            valid = self._partition(valid).squeeze(-1)
            # Mask keys for real queries. Give discarded padded queries a safe
            # row so fully padded windows cannot produce softmax NaNs.
            mask = mask.masked_fill(~valid[:, None, :], float("-inf"))
            mask = torch.where(valid[:, :, None], mask, torch.zeros_like(mask))
        self.register_buffer("attention_mask", mask, persistent=False)
        self.has_mask = shift or (hp, wp) != resolution

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        if (height, width) != (self.height, self.width):
            raise ValueError("Window attention received an unexpected spatial resolution")
        hp, wp = self.padded_height, self.padded_width
        if (hp, wp) != (height, width):
            x = F.pad(x, (0, 0, 0, wp - width, 0, hp - height))
        if self.shift_size != (0, 0):
            x = torch.roll(x, shifts=tuple(-s for s in self.shift_size), dims=(1, 2))
        windows = self._partition(x)
        count, length, _ = windows.shape
        qkv = self.qkv(windows).reshape(count, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(length, length, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if self.has_mask:
            attention = attention.reshape(batch, -1, self.num_heads, length, length)
            attention = attention + self.attention_mask.to(attention.dtype)[None, :, None]
            attention = attention.reshape(count, self.num_heads, length, length)
        out = (attention.softmax(dim=-1) @ v).transpose(1, 2).reshape(count, length, channels)
        out = self._reverse(self.proj(out), batch, hp, wp)
        if self.shift_size != (0, 0):
            out = torch.roll(out, shifts=self.shift_size, dims=(1, 2))
        return out[:, :height, :width].contiguous()


class CandidateTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        resolution: tuple[int, int],
        heads: int,
        window_size: tuple[int, int],
        mlp_ratio: int,
        start_shifted: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TemporalAttention(dim, heads),
                        nn.LayerNorm(dim),
                        PaddedWindowAttention(
                            dim, resolution, window_size, heads, bool((i + int(start_shifted)) % 2)
                        ),
                        nn.LayerNorm(dim),
                        Mlp(dim, mlp_ratio * dim, 0.3),
                        nn.LayerNorm(dim),
                    ]
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, patches, dim = x.shape
        for temporal, norm1, spatial, norm2, mlp, norm3 in self.layers:
            x = norm1(temporal(x.reshape(-1, patches, dim)).reshape_as(x) + x)
            spatial_input = x.permute(0, 3, 1, 2, 4).reshape(-1, height, width, dim)
            y = spatial(spatial_input).reshape(batch, patches, height, width, dim)
            x = norm2(y.permute(0, 2, 3, 1, 4) + x)
            x = norm3(mlp(x) + x)
        return x


class SpatialMerge(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(
            [x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1
        )
        return self.norm(self.linear(x))


class SpatialExpand(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 4 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, patches, _ = x.shape
        x = self.linear(x).reshape(batch, height, width, patches, 2, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, height * 2, width * 2, patches, self.out_dim
        )
        return self.norm(x)


class CandidateDown(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, transformer: nn.Module) -> None:
        super().__init__()
        self.merge = SpatialMerge(in_dim, out_dim)
        self.transformer = transformer

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer(self.merge(x))


class CandidateUp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, transformer: nn.Module) -> None:
        super().__init__()
        self.expand = SpatialExpand(in_dim, out_dim)
        self.transformer = transformer

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.transformer(self.expand(x) + skip)


class MatchedFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 1, 39))

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x.reshape(-1, 1, x.shape[-2]), self.weight, padding=19).reshape_as(x)


class TokenEmbedding(nn.Module):
    def __init__(self, patch_size: int, dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, samples, _ = x.shape
        return self.linear(
            x.reshape(batch, height, width, samples // self.patch_size, self.patch_size)
        )


class PositionEncoding(nn.Module):
    def __init__(self, patches: int, dim: int) -> None:
        super().__init__()
        self.register_buffer("encoding", sinusoidal_encoding(patches, dim)[None, None, None])

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding.to(x.dtype)


class DenseHead(nn.Module):
    def __init__(self, dim: int, patch_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, patch_size * 4)
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, patches, _ = x.shape
        return self.linear(self.norm(x)).reshape(batch, height, width, patches * self.patch_size, 4)


class FullFrameNeuralDSP(nn.Module):
    def __init__(
        self,
        height: int = 336,
        width: int = 400,
        samples: int = 256,
        patch_size: int = 16,
        channels: tuple[int, ...] = (16, 16, 32, 64),
        depths: tuple[int, ...] = (0, 1, 2, 1, 0),
        heads: int = 2,
        window_size: tuple[int, int] = (2, 4),
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        if min(height, width, samples, patch_size, heads, mlp_ratio, *window_size) <= 0:
            raise ValueError("Dimensions, patch size, heads, window and MLP ratio must be positive")
        if height % 8 or width % 8 or samples % patch_size:
            raise ValueError("H/W must be divisible by 8 and T must be divisible by patch_size")
        if len(channels) != 4 or any(c <= 0 or c % 2 for c in channels):
            raise ValueError(
                "channels must contain four positive even widths: full, half, quarter, eighth"
            )
        if len(depths) != 5 or any(d < 0 for d in depths):
            raise ValueError("depths must contain five nonnegative integers")
        stage_channels = (channels[1], channels[2], channels[3], channels[2], channels[1])
        if any(d and c % heads for c, d in zip(stage_channels, depths)):
            raise ValueError("Active Transformer channel widths must be divisible by heads")
        self.input_shape = (height, width, samples, 1)
        self.matched_filter = MatchedFilter()
        self.patch_embedding = TokenEmbedding(patch_size, channels[0])
        self.position_encoding = PositionEncoding(samples // patch_size, channels[0])

        def transformer(index: int, divisor: int, shifted: bool = False) -> nn.Module:
            return CandidateTransformer(
                stage_channels[index],
                depths[index],
                (height // divisor, width // divisor),
                heads,
                window_size,
                mlp_ratio,
                shifted,
            )

        c0, c1, c2, c3 = channels
        self.block1 = CandidateDown(c0, c1, transformer(0, 2))
        self.block2 = CandidateDown(c1, c2, transformer(1, 4))
        self.block3 = CandidateDown(c2, c3, transformer(2, 8))
        self.block4 = CandidateUp(c3, c2, transformer(3, 4, True))
        self.block5 = CandidateUp(c2, c1, transformer(4, 2, True))
        self.full_resolution = CandidateUp(c1, c0, nn.Identity())
        self.classification_head = DenseHead(c0, patch_size)

    def forward(self, waveform: Tensor) -> Tensor:
        if tuple(waveform.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected [B,{self.input_shape}], got {tuple(waveform.shape)}")
        s0 = self.position_encoding(self.patch_embedding(self.matched_filter(waveform)))
        s1 = self.block1(s0)
        s2 = self.block2(s1)
        x = self.block3(s2)
        x = self.block4(x, s2)
        x = self.block5(x, s1)
        return self.classification_head(self.full_resolution(x, s0))
