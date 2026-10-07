"""Configurable multi-resolution residual U-Net for serious image experiments.

Time/noise embeddings, scale-shift normalization, zero-initialized residual
branches, multiple blocks per level and attention follow standard diffusion
design principles. This is an original implementation, not a paper checkpoint.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import Conditioning, normalization


def zero(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ImageResidual(nn.Module):
    def __init__(self, incoming, outgoing, embedding, dropout=0.0):
        super().__init__()
        self.first = nn.Sequential(
            normalization(incoming), nn.SiLU(), nn.Conv2d(incoming, outgoing, 3, padding=1)
        )
        self.condition = nn.Linear(embedding, 2 * outgoing)
        self.norm = normalization(outgoing)
        self.last = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), zero(nn.Conv2d(outgoing, outgoing, 3, padding=1))
        )
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, embedding):
        h = self.first(x)
        scale, shift = self.condition(F.silu(embedding)).chunk(2, dim=1)
        h = self.norm(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return (self.skip(x) + self.last(h)) / math.sqrt(2)


class ImageAttention(nn.Module):
    def __init__(self, channels, head_channels=32):
        super().__init__()
        self.heads = max(1, channels // head_channels)
        if channels % self.heads:
            raise ValueError("Attention channels must be divisible by number of heads")
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.output = zero(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        attention = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        return x + self.output(attention.transpose(-1, -2).reshape(b, c, h, w))


class ImageUNet(nn.Module):
    def __init__(
        self,
        width=64,
        classes=0,
        channels=3,
        out_channels=None,
        channel_mult=(1, 2, 2, 2),
        blocks_per_level=2,
        attention_levels=(1, 2),
        dropout=0.1,
        time_embedding="fourier",
        **_,
    ):
        super().__init__()
        if width % 2 or min(channel_mult) < 1 or blocks_per_level < 1:
            raise ValueError("Expected even width and positive depth/channel multipliers")
        self.divisor = 2 ** (len(channel_mult) - 1)
        self.condition = Conditioning(width * 4, classes, time_embedding)
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.encoders, self.decoders, self.down, self.up = (nn.ModuleList() for _ in range(4))
        features, current = [width], width
        for level, multiplier in enumerate(channel_mult):
            stage = nn.ModuleList()
            for _ in range(blocks_per_level):
                stage.append(
                    nn.ModuleList(
                        [
                            ImageResidual(current, width * multiplier, width * 4, dropout),
                            ImageAttention(width * multiplier)
                            if level in attention_levels
                            else nn.Identity(),
                        ]
                    )
                )
                current = width * multiplier
                features.append(current)
            self.encoders.append(stage)
            if level != len(channel_mult) - 1:
                self.down.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
                features.append(current)
        self.middle0 = ImageResidual(current, current, width * 4, dropout)
        self.middle_attention = ImageAttention(current)
        self.middle1 = ImageResidual(current, current, width * 4, dropout)
        for level in reversed(range(len(channel_mult))):
            stage = nn.ModuleList()
            for _ in range(blocks_per_level + 1):
                incoming = current + features.pop()
                current = width * channel_mult[level]
                stage.append(
                    nn.ModuleList(
                        [
                            ImageResidual(incoming, current, width * 4, dropout),
                            ImageAttention(current) if level in attention_levels else nn.Identity(),
                        ]
                    )
                )
            self.decoders.append(stage)
            if level:
                self.up.append(nn.Conv2d(current, current, 3, padding=1))
        assert not features
        self.output = nn.Sequential(
            normalization(current),
            nn.SiLU(),
            zero(nn.Conv2d(current, out_channels or channels, 3, padding=1)),
        )

    def forward(self, x, t, labels=None):
        if x.shape[-1] % self.divisor or x.shape[-2] % self.divisor:
            raise ValueError(f"Spatial dimensions must be divisible by {self.divisor}")
        embedding = self.condition(t, labels)
        h = self.input(x)
        skips = [h]
        for level, stage in enumerate(self.encoders):
            for residual, attention in stage:
                h = attention(residual(h, embedding))
                skips.append(h)
            if level < len(self.down):
                h = self.down[level](h)
                skips.append(h)
        h = self.middle1(self.middle_attention(self.middle0(h, embedding)), embedding)
        for level, stage in enumerate(self.decoders):
            for residual, attention in stage:
                h = attention(residual(torch.cat([h, skips.pop()], 1), embedding))
            if level < len(self.up):
                h = self.up[level](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(h)
