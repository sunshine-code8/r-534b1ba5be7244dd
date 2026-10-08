"""FastWaveDSP and FineWaveDSP grouped waveform embeddings with a 2D U-Net.

Existing class names and layouts are retained. Default input/output:
[B,H,W,T,1] -> [B,H,W,T,4], or channels_first [B,1,H,W,T] -> [B,4,H,W,T].
All backbone tensors are 4D. Only Fine's H/8,W/8 bottleneck forms sequences.
The old full-bin branches and their checkpoints are intentionally incompatible.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ARCHITECTURE_VERSION = "grouped_unet_v3_plus"


class DualProjectionEmbedding(nn.Module):
    """Per-patch amplitude + within-patch first differences, interleaved by patch."""

    def __init__(self, time_bins: int, patch_size: int, patch_dim: int) -> None:
        super().__init__()
        if patch_size < 2 or patch_dim < 2 or patch_dim % 2:
            raise ValueError("Dual projection requires patch_size >= 2 and even patch_dim >= 2")
        self.patch_size, self.patch_count = patch_size, time_bins // patch_size
        self.branch_dim = patch_dim // 2
        self.amplitude = nn.Conv2d(
            time_bins, self.patch_count * self.branch_dim, 1, groups=self.patch_count
        )
        self.derivative = nn.Conv2d(
            self.patch_count * (patch_size - 1),
            self.patch_count * self.branch_dim,
            1,
            groups=self.patch_count,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        patches = x.reshape(b, self.patch_count, self.patch_size, h, w)
        # No difference is taken across a patch boundary. This is a view of the raw input,
        # not a full-bin multichannel hidden feature.
        delta = (patches[:, :, 1:] - patches[:, :, :-1]).reshape(b, -1, h, w)
        amplitude = self.amplitude(x).reshape(b, self.patch_count, self.branch_dim, h, w)
        derivative = self.derivative(delta).reshape(b, self.patch_count, self.branch_dim, h, w)
        return F.silu(torch.cat((amplitude, derivative), dim=2).reshape(b, -1, h, w))


class ResidualPatchEmbedding(nn.Module):
    """Nonlinear grouped projection plus a raw local waveform shortcut.

    For default P=D=4 the shortcut is exactly identity. SiLU before the sum keeps
    the enhanced embedding from being merely two collapsible linear projections.
    """

    def __init__(self, time_bins: int, patch_size: int, patch_dim: int) -> None:
        super().__init__()
        groups = time_bins // patch_size
        self.project = nn.Conv2d(time_bins, groups * patch_dim, 1, groups=groups)
        self.skip = (
            nn.Identity()
            if patch_size == patch_dim
            else nn.Conv2d(time_bins, groups * patch_dim, 1, groups=groups, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.project(x)) + self.skip(x)


def waveform_auxiliary_loss(
    reconstruction: Tensor,
    target: Tensor,
    layout: str = "channels_last",
    wave_weight: float = 0.1,
    derivative_weight: float = 0.05,
) -> dict[str, Tensor]:
    """Optional training-only L1 waveform and temporal-derivative supervision."""
    if reconstruction.shape != target.shape or target.ndim != 5:
        raise ValueError("Reconstruction and target must have the same five-dimensional shape")
    if layout not in ("channels_last", "channels_first") or min(wave_weight, derivative_weight) < 0:
        raise ValueError("Expected a supported layout and nonnegative auxiliary weights")
    time_dim = 3 if layout == "channels_last" else 4
    reconstruction, target = reconstruction.float(), target.float()
    wave = F.l1_loss(reconstruction, target)
    derivative = (
        F.l1_loss(reconstruction.diff(dim=time_dim), target.diff(dim=time_dim))
        if target.shape[time_dim] > 1
        else reconstruction.sum() * 0
    )
    return {
        "waveform": wave,
        "derivative": derivative,
        "total": wave_weight * wave + derivative_weight * derivative,
    }


class SpatialBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, kernel, stride=stride, padding=kernel // 2, groups=cin),
            nn.SiLU(),
            nn.Conv2d(cin, cout, 1),
            nn.SiLU(),
        )
        self.residual = cin == cout and stride == 1

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return x + y if self.residual else y


class SpatialMerge(nn.Module):
    """NeuralDSP-style 2x2 merge, with right/bottom padding for odd sizes."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.project = nn.Conv2d(4 * cin, cout, 1)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2))
        return F.silu(self.project(self.unshuffle(x)))


class SpatialExpand(nn.Module):
    """Upsample, concatenate the encoder skip, then project with a 1x1 conv."""

    def __init__(self, cin: int, skip_channels: int, cout: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(cin + skip_channels, cout, 1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return F.silu(self.project(torch.cat((x, skip), dim=1)))


class GroupedBinHead(nn.Module):
    """Channel order is [patch, sub-bin, class], with class fastest within a bin."""

    def __init__(self, time_bins: int, patch_size: int, patch_dim: int, full_head: bool = False) -> None:
        super().__init__()
        self.time_bins = time_bins
        self.patch_count = time_bins // patch_size
        self.project = nn.Conv2d(
            self.patch_count * patch_dim, time_bins * 4, 1,
            groups=1 if full_head else self.patch_count
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.project(x)  # [B,T*4,H,W], final logits only

    def format_logits(self, logits: Tensor, layout: str) -> Tensor:
        b, _, h, w = logits.shape
        out = logits.reshape(b, self.time_bins, 4, h, w)
        if layout == "channels_first":
            return out.permute(0, 2, 3, 4, 1).contiguous()
        return out.permute(0, 3, 4, 1, 2).contiguous()

    def labels(self, logits: Tensor) -> Tensor:
        b, _, h, w = logits.shape
        return (
            logits.reshape(b, self.time_bins, 4, h, w)
            .argmax(dim=2)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
        )


class BottleneckTemporalAttention(nn.Module):
    """One global temporal block, only on the 1/8-resolution spatial grid.

    A dense projection creates learned temporal slots. These are contextual
    latent slots, not unmodified raw waveform patches after the spatial merges.
    Relative range bias is a learned table indexed by signed token distance.
    """

    def __init__(
        self,
        channels: int,
        tokens: int,
        token_dim: int = 8,
        heads: int = 2,
        relative_bias: bool = True,
        chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if min(channels, tokens, token_dim, heads) < 1 or token_dim % heads or chunk_size < 0:
            raise ValueError(
                "Positive dimensions, token_dim divisible by heads, chunk_size >= 0 required"
            )
        self.tokens, self.token_dim, self.heads = tokens, token_dim, heads
        self.chunk_size = chunk_size
        self.to_tokens = nn.Conv2d(channels, tokens * token_dim, 1)
        self.norm = nn.LayerNorm(token_dim)
        self.qkv = nn.Linear(token_dim, 3 * token_dim)
        self.proj = nn.Linear(token_dim, token_dim)
        self.norm_ffn = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, 2 * token_dim), nn.GELU(), nn.Linear(2 * token_dim, token_dim)
        )
        self.to_spatial = nn.Conv2d(tokens * token_dim, channels, 1)
        if relative_bias:
            self.relative_bias = nn.Parameter(torch.zeros(2 * tokens - 1, heads))
            nn.init.normal_(self.relative_bias, std=0.02)
        else:
            self.register_parameter("relative_bias", None)
        positions = torch.arange(tokens)
        self.register_buffer(
            "relative_index", positions[:, None] - positions[None, :] + tokens - 1, persistent=False
        )

    def attend(self, x: Tensor) -> Tensor:
        n, length, dim = x.shape
        qkv = self.qkv(self.norm(x)).reshape(n, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q * (dim // self.heads) ** -0.5) @ k.transpose(-2, -1)
        if self.relative_bias is not None:
            scores = scores + self.relative_bias[self.relative_index].permute(2, 0, 1)[None]
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        out = (weights @ v).transpose(1, 2).reshape(n, length, dim)
        x = x + self.proj(out)
        return x + self.ffn(self.norm_ffn(x))

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        tokens = (
            self.to_tokens(x).permute(0, 2, 3, 1).reshape(b * h * w, self.tokens, self.token_dim)
        )
        chunk = self.chunk_size or tokens.shape[0]
        if chunk >= tokens.shape[0]:
            tokens = self.attend(tokens)
        else:
            tokens = torch.cat([self.attend(part) for part in tokens.split(chunk)], dim=0)
        out = tokens.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x + self.to_spatial(out)


class WaveDSPBase(nn.Module):
    architecture_version = ARCHITECTURE_VERSION

    def __init__(
        self,
        time_bins: int,
        patch_size: int,
        patch_dim: int,
        encoder_channels: tuple[int, int, int],
        decoder_channels: tuple[int, int],
        layout: str,
        bottleneck: nn.Module,
        embedding_kind: str,
        stem_channels: int,
        decoder_patch_dim: int,
        auxiliary_reconstruction: bool = False,
        full_head: bool = False,
    ) -> None:
        super().__init__()
        if min(time_bins, patch_size, patch_dim, *encoder_channels, *decoder_channels) < 1:
            raise ValueError("Time bins, patch dimensions and channel widths must be positive")
        if time_bins % patch_size:
            raise ValueError("time_bins must be divisible by patch_size")
        if layout not in ("channels_last", "channels_first"):
            raise ValueError("layout must be channels_last or channels_first")
        self.time_bins, self.patch_size, self.patch_dim = time_bins, patch_size, patch_dim
        self.layout = layout
        self.patch_count = time_bins // patch_size
        embedded_channels = self.patch_count * patch_dim
        c0 = stem_channels
        head_channels = self.patch_count * decoder_patch_dim
        if min(c0, decoder_patch_dim) < 1:
            raise ValueError("Stem and decoder dimensions must be positive")
        c1, c2, c3 = encoder_channels
        d2, d1 = decoder_channels
        if embedding_kind == "dual":
            self.patch_embed = DualProjectionEmbedding(time_bins, patch_size, patch_dim)
        elif embedding_kind == "residual":
            self.patch_embed = ResidualPatchEmbedding(time_bins, patch_size, patch_dim)
        else:
            raise ValueError("Unknown embedding kind")
        self.embedding_compression = nn.Conv2d(embedded_channels, c0, 1)
        self.auxiliary_head = (
            nn.Conv2d(embedded_channels, time_bins, 1, groups=self.patch_count)
            if auxiliary_reconstruction
            else None
        )
        # Only the compressed representation is stored as the level-0 U-Net skip.
        self.down1 = SpatialMerge(c0, c1)
        self.encoder1 = SpatialBlock(c1, c1)
        self.down2 = SpatialMerge(c1, c2)
        self.encoder2 = nn.Sequential(SpatialBlock(c2, c2), SpatialBlock(c2, c2))
        self.down3 = SpatialMerge(c2, c3)
        self.bottleneck = bottleneck
        self.up2 = SpatialExpand(c3, c2, d2)
        self.up1 = SpatialExpand(d2, c1, d1)
        self.up0 = SpatialExpand(d1, c0, head_channels)
        self.head = GroupedBinHead(time_bins, patch_size, decoder_patch_dim, full_head=full_head)
        self.config = dict(
            time_bins=time_bins,
            patch_size=patch_size,
            patch_dim=patch_dim,
            encoder_channels=(c0, c1, c2, c3),
            decoder_channels=(d2, d1, head_channels),
            embedding_kind=embedding_kind,
            embedding_channels=embedded_channels,
            stem_channels=c0,
            decoder_patch_dim=decoder_patch_dim,
            auxiliary_reconstruction=auxiliary_reconstruction,
            layout=layout,
        )

    def _unpack(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError("Expected a five-dimensional waveform input tensor")
        if self.layout == "channels_first":
            if x.shape[1] != 1:
                raise ValueError("Expected [B,1,H,W,T]")
            raw = x[:, 0]
        else:
            if x.shape[-1] != 1:
                raise ValueError("Expected [B,H,W,T,1]")
            raw = x[..., 0]
        if min(raw.shape) < 1 or raw.shape[-1] != self.time_bins:
            raise ValueError(
                f"Expected positive B/H/W and T={self.time_bins}; got {tuple(raw.shape)}"
            )
        # Explicit bin-to-channel conversion is included in forward timing.
        return raw.permute(0, 3, 1, 2).contiguous()

    def _decode(self, e0: Tensor) -> Tensor:
        e1 = self.encoder1(self.down1(e0))
        e2 = self.encoder2(self.down2(e1))
        out = self.bottleneck(self.down3(e2))
        return self.up0(self.up1(self.up2(out, e2), e1), e0)

    def forward_features(self, x: Tensor) -> Tensor:
        # Release the expanded embedding before running the full encoder/decoder in inference.
        e0 = self.embedding_compression(self.patch_embed(self._unpack(x)))
        return self._decode(e0)

    def forward_with_aux(self, x: Tensor) -> dict[str, Tensor]:
        """Explicit training path; ordinary forward/predict never execute the auxiliary head."""
        if self.auxiliary_head is None:
            raise RuntimeError(
                "Construct with auxiliary_reconstruction=True to use forward_with_aux"
            )
        embedding = self.patch_embed(self._unpack(x))
        features = self._decode(self.embedding_compression(embedding))
        logits = self.head.format_logits(self.head(features), self.layout)
        reconstruction = self.auxiliary_head(embedding).permute(0, 2, 3, 1)
        reconstruction = (
            reconstruction.unsqueeze(-1)
            if self.layout == "channels_last"
            else reconstruction.unsqueeze(1)
        )
        return {"logits": logits, "reconstruction": reconstruction.contiguous()}

    def forward(self, x: Tensor) -> Tensor:
        """Return contiguous dense logits; no full-bin hidden feature is reconstructed."""
        return self.head.format_logits(self.head(self.forward_features(x)), self.layout)

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        """Return GPU uint8 [B,H,W,T], taking argmax before dense-logit layout conversion.

        The native [B,T*4,H,W] logits are still allocated. This is not a fused head
        and argmax kernel; it only avoids a full four-class permutation/copy.
        """
        return self.head.labels(self.head(self.forward_features(x)))


class FastWaveDSP(WaveDSPBase):
    """P=16, D=4+4 amplitude/derivative embedding followed by 1x1 compression."""

    def __init__(
        self,
        time_bins: int = 256,
        patch_size: int = 16,
        patch_dim: int = 8,
        layout: str = "channels_last",
        bottleneck_mixer: bool = True,
        stem_channels: int = 96,
        auxiliary_reconstruction: bool = False,
        full_head: bool = False,
    ) -> None:
        bottleneck = nn.Sequential(
            SpatialBlock(128, 128),
            nn.Sequential(nn.Conv2d(128, 128, 1), nn.SiLU()) if bottleneck_mixer else nn.Identity(),
            SpatialBlock(128, 128),
        )
        super().__init__(
            time_bins,
            patch_size,
            patch_dim,
            (96, 128, 128),
            (96, 64),
            layout,
            bottleneck,
            embedding_kind="dual",
            stem_channels=stem_channels,
            decoder_patch_dim=4,
            auxiliary_reconstruction=auxiliary_reconstruction,
            full_head=full_head,
        )
        self.config.update(bottleneck_mixer=bottleneck_mixer, full_head=full_head)


class FineWaveDSP(WaveDSPBase):
    """P=D=4 residual embedding followed by 1x1 compression."""

    def __init__(
        self,
        time_bins: int = 256,
        patch_size: int = 4,
        patch_dim: int = 4,
        layout: str = "channels_last",
        temporal_attention: bool = True,
        token_dim: int = 8,
        heads: int = 2,
        relative_bias: bool = True,
        attention_chunk_size: int = 0,
        stem_channels: int = 128,
        auxiliary_reconstruction: bool = False,
    ) -> None:
        if min(time_bins, patch_size, patch_dim, token_dim, heads) < 1 or time_bins % patch_size:
            raise ValueError("Positive dimensions and time_bins divisible by patch_size required")
        if token_dim % heads or attention_chunk_size < 0:
            raise ValueError("token_dim must be divisible by heads; attention_chunk_size >= 0")
        attention = (
            BottleneckTemporalAttention(
                256, time_bins // patch_size, token_dim, heads, relative_bias, attention_chunk_size
            )
            if temporal_attention
            else nn.Identity()
        )
        bottleneck = nn.Sequential(SpatialBlock(256, 256), attention, SpatialBlock(256, 256))
        super().__init__(
            time_bins,
            patch_size,
            patch_dim,
            (128, 160, 256),
            (160, 128),
            layout,
            bottleneck,
            embedding_kind="residual",
            stem_channels=stem_channels,
            decoder_patch_dim=2,
            auxiliary_reconstruction=auxiliary_reconstruction,
        )
        self.config.update(
            temporal_attention=temporal_attention,
            token_dim=token_dim,
            heads=heads,
            relative_bias=relative_bias if temporal_attention else False,
            attention_chunk_size=attention_chunk_size,
        )
