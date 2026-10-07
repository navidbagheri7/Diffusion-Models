"""Original compact networks: a time-conditioned MLP and a residual U-Net."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width, kind="fourier"):
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("Time embedding width must be positive and even")
        if kind not in {"fourier", "positional"}:
            raise ValueError("Time embedding must be fourier or positional")
        self.phase_scale = 2 * math.pi if kind == "fourier" else 1.0
        frequencies = (
            torch.exp(torch.linspace(0, math.log(1000), width // 2))
            if kind == "fourier"
            else torch.exp(torch.linspace(0, -math.log(10000), width // 2))
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t):
        phase = t[:, None] * self.frequencies[None] * self.phase_scale
        return torch.cat([phase.sin(), phase.cos()], -1)


class Conditioning(nn.Module):
    def __init__(self, width, classes, time_embedding="fourier"):
        super().__init__()
        self.classes = classes
        self.time = nn.Sequential(
            TimeEmbedding(width, time_embedding),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        # Last embedding is the null token for unconditional training and CFG.
        self.label = nn.Embedding(classes + 1, width) if classes else None

    def forward(self, t, labels=None):
        h = self.time(t)
        if self.label is not None:
            if labels is None:
                labels = torch.full_like(t, self.classes, dtype=torch.long)
            h = h + self.label(labels)
        return h


class TimeMLP(nn.Module):
    def __init__(self, width=128, depth=3, classes=0, output_dim=2):
        super().__init__()
        self.condition = Conditioning(width, classes)
        self.input = nn.Linear(2, width)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(width, width)) for _ in range(depth)]
        )
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(width, output_dim))

    def forward(self, x, t, labels=None):
        condition = self.condition(t, labels)
        h = self.input(x) + condition
        for block in self.blocks:
            h = h + block(h + condition) / math.sqrt(2)
        return self.output(h)


def normalization(channels):
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, embedding):
        super().__init__()
        self.first = nn.Sequential(
            normalization(incoming), nn.SiLU(), nn.Conv2d(incoming, outgoing, 3, padding=1)
        )
        self.condition = nn.Linear(embedding, outgoing * 2)
        self.norm = normalization(outgoing)
        self.second = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, condition):
        h = self.first(x)
        scale, shift = self.condition(F.silu(condition)).chunk(2, -1)
        h = self.norm(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        return (self.skip(x) + self.second(F.silu(h))) / math.sqrt(2)


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).flatten(2).transpose(1, 2).chunk(3, -1)
        attention = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])
        return x + self.output(attention[:, 0].transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Three resolution levels, FiLM time/label conditioning, bottleneck attention."""

    def __init__(self, width=32, classes=0):
        super().__init__()
        embedding = width * 4
        self.condition = Conditioning(embedding, classes)
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.enc0 = ResidualBlock(width, width, embedding)
        self.down0 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc1 = ResidualBlock(width * 2, width * 2, embedding)
        self.down1 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.middle = ResidualBlock(width * 4, width * 4, embedding)
        self.attention = SpatialAttention(width * 4)
        self.dec1 = ResidualBlock(width * 6, width * 2, embedding)
        self.dec0 = ResidualBlock(width * 3, width, embedding)
        self.output = nn.Sequential(
            normalization(width), nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1)
        )

    def forward(self, x, t, labels=None):
        if x.shape[-1] % 4 or x.shape[-2] % 4:
            raise ValueError("U-Net spatial dimensions must be divisible by four")
        condition = self.condition(t, labels)
        h0 = self.enc0(self.input(x), condition)
        h1 = self.enc1(self.down0(h0), condition)
        h = self.attention(self.middle(self.down1(h1), condition))
        h = self.dec1(
            torch.cat([F.interpolate(h, size=h1.shape[-2:], mode="nearest"), h1], 1), condition
        )
        h = self.dec0(
            torch.cat([F.interpolate(h, size=h0.shape[-2:], mode="nearest"), h0], 1), condition
        )
        return self.output(h)


def make_model(config, classes=0):
    if "_target_" in config:
        from hydra.utils import instantiate

        return instantiate(config, classes=classes)
    if config.kind == "dit":
        from .dit import DiT

        return DiT(classes=classes, **{k: v for k, v in config.items() if k != "kind"})
    if config.kind == "image_unet":
        from .image_models import ImageUNet

        return ImageUNet(classes=classes, **{k: v for k, v in config.items() if k != "kind"})
    if config.kind == "mlp":
        return TimeMLP(config.width, config.depth, classes)
    if config.kind == "unet":
        return UNet(config.width, classes)
    raise ValueError(f"Unknown model: {config.kind}")


class ImageClassifier(nn.Module):
    """Small noisy-image classifier for demonstrating classifier guidance."""

    def __init__(self, width, classes, channels=3):
        super().__init__()
        self.condition = Conditioning(width * 4, 0)
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.block0 = ResidualBlock(width, width * 2, width * 4)
        self.block1 = ResidualBlock(width * 2, width * 4, width * 4)
        self.output = nn.Linear(width * 4, classes)

    def forward(self, x, t):
        c = self.condition(t)
        h = self.block0(F.avg_pool2d(self.input(x), 2), c)
        h = self.block1(F.avg_pool2d(h, 2), c)
        return self.output(F.silu(h).mean((2, 3)))
