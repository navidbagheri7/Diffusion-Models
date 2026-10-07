"""Diffusion transformer with patch tokens, fixed 2D positions and adaLN-Zero.

Original implementation of the architecture described by Peebles & Xie
(arXiv:2212.09748). The time argument follows the enclosing process: it is
log(sigma)/4 for EDM and normalized corruption time for discrete diffusion.
Class dropout is handled by the common trainer, using the last embedding as null.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .models import Conditioning


def position_embedding(size, width):
    """Fixed separable sin/cos positions; each axis receives half the features."""
    if size < 1 or width % 4:
        raise ValueError("Positions require positive grid size and width divisible by four")
    frequency = 10000 ** (-torch.arange(width // 4, dtype=torch.float64) / (width // 4))
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    embeddings = []
    for coordinates in [x.flatten(), y.flatten()]:
        phase = coordinates[:, None] * frequency[None]
        embeddings.extend([phase.sin(), phase.cos()])
    return torch.cat(embeddings, dim=-1).float()[None]


def modulate(tokens, shift, scale):
    return tokens * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4):
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, width)
        )
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, tokens, condition):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.condition(condition).chunk(6, -1)
        h = modulate(self.norm_attention(tokens), shift_a, scale_a)
        b, n, width = h.shape
        q, k, v = (
            self.qkv(h)
            .reshape(b, n, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attention = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, width)
        tokens = tokens + gate_a[:, None] * self.projection(attention)
        return tokens + gate_m[:, None] * self.mlp(
            modulate(self.norm_mlp(tokens), shift_m, scale_m)
        )


class DiT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        width=384,
        depth=8,
        heads=6,
        channels=3,
        out_channels=None,
        classes=0,
        mlp_ratio=4,
        gradient_checkpointing=False,
        time_embedding="fourier",
        **_,
    ):
        super().__init__()
        if min(image_size, patch_size, width, depth, heads, channels) < 1:
            raise ValueError("DiT dimensions must be positive")
        if width % 4 or width % heads or image_size % patch_size or mlp_ratio <= 0:
            raise ValueError("DiT width must divide heads/four and image size must divide patches")
        self.image_size, self.patch_size = image_size, patch_size
        self.channels, self.out_channels = channels, out_channels or channels
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = nn.Conv2d(channels, width, patch_size, stride=patch_size)
        self.register_buffer("positions", position_embedding(image_size // patch_size, width))
        self.condition = Conditioning(width, classes, time_embedding)
        self.blocks = nn.ModuleList([DiTBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.output_condition = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, patch_size**2 * self.out_channels)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        if self.condition.label is not None:
            nn.init.normal_(self.condition.label.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.condition[-1].weight)
            nn.init.zeros_(block.condition[-1].bias)
        for module in [self.output_condition[-1], self.output]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def unpatchify(self, tokens):
        b = len(tokens)
        grid, p = self.image_size // self.patch_size, self.patch_size
        return (
            tokens.reshape(b, grid, grid, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, self.out_channels, self.image_size, self.image_size)
        )

    def forward(self, x, t, labels=None):
        if x.ndim != 4 or x.shape[1:] != (self.channels, self.image_size, self.image_size):
            raise ValueError("DiT input must match configured channels and square image size")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.positions.to(tokens.dtype)
        condition = self.condition(t, labels)
        for block in self.blocks:
            tokens = (
                checkpoint(block, tokens, condition, use_reentrant=False)
                if self.gradient_checkpointing and self.training
                else block(tokens, condition)
            )
        shift, scale = self.output_condition(condition).chunk(2, -1)
        return self.unpatchify(self.output(modulate(self.norm(tokens), shift, scale)))
