"""Inspectable token conditioning, masked cross-attention, and native LoRA.

Image queries attend to text keys/values. The compact U-Net is our educational
architecture. The separate pretrained path adapts a local Diffusers U-Net;
neither loading a codec nor a text encoder implies a pretrained denoiser.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .image_models import ImageUNet
from .models import normalization


class TextCrossAttention(nn.Module):
    """NCHW image features + NLD token features -> NCHW residual update.

    A boolean NL mask marks valid tokens. Empty prompts still contain CLIP's
    special tokens; an all-masked sequence is rejected instead of producing NaN.
    """

    def __init__(self, channels, text_dim, heads=4):
        super().__init__()
        if channels % heads or min(channels, text_dim, heads) < 1:
            raise ValueError("Positive dimensions and channels divisible by heads are required")
        self.heads = heads
        self.norm = normalization(channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Linear(text_dim, channels)
        self.value = nn.Linear(text_dim, channels)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context, mask=None, return_weights=False):
        b, c, h, w = x.shape
        if context.ndim != 3 or len(context) != b:
            raise ValueError("Text context must have shape [batch, tokens, text_dim]")
        if mask is not None:
            if mask.shape != context.shape[:2] or not mask.bool().any(1).all():
                raise ValueError("Every text sequence needs at least one valid token")
            mask = mask.bool()[:, None, None, :]
        q = self.query(self.norm(x)).flatten(2).transpose(1, 2)
        q = q.reshape(b, h * w, self.heads, c // self.heads).transpose(1, 2)
        k = self.key(context).reshape(b, -1, self.heads, c // self.heads).transpose(1, 2)
        v = self.value(context).reshape(b, -1, self.heads, c // self.heads).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        result = x + self.output(
            attention.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        )
        if return_weights:
            logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(c // self.heads)
            if mask is not None:
                logits = logits.masked_fill(~mask, -torch.inf)
            return result, logits.softmax(-1).mean(1).reshape(b, h, w, -1)
        return result


class TextUNet(ImageUNet):
    """Our residual/time-conditioned U-Net with token attention at every level."""

    def __init__(self, text_dim=1024, width=64, channel_mult=(1, 2, 3), time_scale=1.0, **kwargs):
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Text time scale must be finite and positive")
        super().__init__(width=width, channel_mult=channel_mult, classes=0, **kwargs)
        self.time_scale = time_scale
        self.encoder_text = nn.ModuleList(
            [TextCrossAttention(width * m, text_dim) for m in channel_mult]
        )
        self.decoder_text = nn.ModuleList(
            [TextCrossAttention(width * m, text_dim) for m in reversed(channel_mult)]
        )
        self.middle_text = TextCrossAttention(width * channel_mult[-1], text_dim)

    def forward(self, x, t, context, mask=None):
        if any(size % self.divisor for size in x.shape[-2:]):
            raise ValueError("Latent spatial dimensions must match the U-Net divisor")
        embedding = self.condition(t * self.time_scale)
        h, skips = self.input(x), []
        skips.append(h)
        for level, stage in enumerate(self.encoders):
            for residual, attention in stage:
                h = attention(residual(h, embedding))
                h = self.encoder_text[level](h, context, mask)
                skips.append(h)
            if level < len(self.down):
                h = self.down[level](h)
                skips.append(h)
        h = self.middle1(
            self.middle_text(self.middle_attention(self.middle0(h, embedding)), context, mask),
            embedding,
        )
        for level, stage in enumerate(self.decoders):
            for residual, attention in stage:
                h = attention(residual(torch.cat([h, skips.pop()], 1), embedding))
                h = self.decoder_text[level](h, context, mask)
            if level < len(self.up):
                h = self.up[level](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(h)


class LowRankLinear(nn.Module):
    """W x + (alpha/r) B A x; frozen W, random A, initially zero B."""

    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or alpha <= 0:
            raise ValueError("LoRA needs a linear layer and positive rank/alpha")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.a = nn.Linear(
            base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32
        )
        self.b = nn.Linear(
            rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        # Adapter arithmetic is FP32 outside autocast; gradients stay FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = self.b(self.a(x.float())) * self.scale
        return self.base(x) + update.to(x.dtype)


def inject_lora(model, rank=8, alpha=8):
    model.requires_grad_(False)
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(("to_q", "to_k", "to_v", "to_out.0"))
    ]
    if not names:
        raise ValueError("No supported attention projections found")
    for name in names:
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, LowRankLinear(model.get_submodule(name), rank, alpha))
    return names


def trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(model, state):
    parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if parameters.keys() != state.keys():
        raise ValueError("Trainable parameter names differ from checkpoint")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter))


def text_guided_prediction(predict, x, t, context, empty_context, scale=1.0):
    """CFG in native epsilon/v coordinates, with measured branch-call counts."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("CFG scale must be finite and nonnegative")
    if scale == 1:
        return predict(x, t, context), 1
    unconditional = predict(x, t, empty_context)
    if scale == 0:
        return unconditional, 1
    conditional = predict(x, t, context)
    return unconditional + scale * (conditional - unconditional), 2
