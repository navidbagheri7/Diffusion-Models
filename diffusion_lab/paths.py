"""Gaussian corruption paths and their SDE coefficients.

Diffusion time increases from data (0) to noise (1). Flow time uses the opposite
direction: noise (0) to data (1). Names and docstrings always specify the direction.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def expand(value: Tensor, x: Tensor) -> Tensor:
    """Broadcast a scalar or one coefficient per example over spatial dimensions."""
    return value.reshape(-1, *([1] * (x.ndim - 1)))


@dataclass
class GaussianPath:
    """X_t = alpha(t) X_0 + sigma(t) epsilon, epsilon ~ N(0,I)."""

    kind: str = "vp_linear"
    beta_min: float = 0.1
    beta_max: float = 20.0
    sigma_min: float = 0.01
    sigma_max: float = 20.0
    logsnr_max: float = 15.0
    logsnr_min: float = -15.0

    def __post_init__(self):
        if self.kind not in {
            "vp_linear",
            "vp_cosine",
            "ve",
            "sub_vp",
            "vp_logsnr",
            "vp_exponential",
        }:
            raise ValueError(f"Unknown continuous path: {self.kind}")
        if not 0 < self.beta_min <= self.beta_max or not 0 < self.sigma_min < self.sigma_max:
            raise ValueError("Expected positive ordered beta and sigma bounds")
        if self.logsnr_min >= self.logsnr_max:
            raise ValueError("log-SNR must decrease during forward diffusion")

    def coefficients(self, t: Tensor) -> tuple[Tensor, Tensor]:
        if self.kind in {"vp_linear", "sub_vp", "vp_exponential"}:
            if self.kind == "vp_exponential":
                integrated_beta = self.beta_max * t
            else:
                integrated_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            log_alpha = -0.5 * integrated_beta
            alpha = log_alpha.exp()
            variance = (-torch.expm1(2 * log_alpha)).clamp_min(0)
            return alpha, variance if self.kind == "sub_vp" else variance.sqrt()
        if self.kind == "vp_logsnr":
            logsnr = self.logsnr_max + t * (self.logsnr_min - self.logsnr_max)
            return logsnr.sigmoid().sqrt(), (-logsnr).sigmoid().sqrt()
        if self.kind == "vp_cosine":
            offset = 0.008
            angle = (t + offset) / (1 + offset) * math.pi / 2
            alpha = angle.cos() / math.cos(offset / (1 + offset) * math.pi / 2)
            return alpha, (1 - alpha.square()).clamp_min(0).sqrt()
        # Conventional VE marginals start with a small Gaussian smoothing at t=0.
        return torch.ones_like(t), self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return forward drift f(x,t) and diffusion coefficient g(t)."""
        if self.kind == "ve":
            _, sigma = self.coefficients(t)
            return torch.zeros_like(x), sigma * math.sqrt(
                2 * math.log(self.sigma_max / self.sigma_min)
            )
        if self.kind == "vp_logsnr":
            alpha, _ = self.coefficients(t)
            beta = (self.logsnr_max - self.logsnr_min) * (1 - alpha.square())
        elif self.kind == "vp_exponential":
            beta = torch.full_like(t, self.beta_max)
        elif self.kind in {"vp_linear", "sub_vp"}:
            beta = self.beta_min + (self.beta_max - self.beta_min) * t
        else:
            angle = (t + 0.008) / 1.008 * math.pi / 2
            beta = math.pi / 1.008 * angle.tan()
        diffusion_squared = beta
        if self.kind == "sub_vp":
            integrated_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            diffusion_squared = beta * (-torch.expm1(-2 * integrated_beta))
        return -0.5 * expand(beta, x) * x, diffusion_squared.sqrt()

    def prior_std(self) -> float:
        return self.sigma_max if self.kind == "ve" else 1.0


class DiscreteVP:
    """DDPM table. Table index i corresponds to mathematical step i+1.

    A clean endpoint is represented separately, by index -1 during sampling.
    Training uses t=(i+1)/T, preserving exact agreement with this table.
    """

    def __init__(self, steps: int = 1000, schedule: str = "cosine", device="cpu"):
        if steps < 2:
            raise ValueError("At least two diffusion steps are required")
        self.steps = steps
        if schedule == "linear":
            betas = torch.linspace(
                1e-4 * 1000 / steps, 0.02 * 1000 / steps, steps, dtype=torch.float64
            )
            if (betas >= 1).any():
                raise ValueError(
                    "Scaled linear schedule requires >20 steps; use cosine for tiny tests"
                )
        elif schedule in {"logsnr", "exponential"}:
            t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
            # Upper bound 10 keeps the first table variance representable in float32.
            acp = torch.sigmoid(10 - 25 * t) if schedule == "logsnr" else torch.exp(-12 * t)
            acp = acp / acp[0]
            betas = (1 - acp[1:] / acp[:-1]).clamp(max=0.999)
        elif schedule == "cosine":
            t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
            acp = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
            acp = acp / acp[0]
            betas = (1 - acp[1:] / acp[:-1]).clamp(max=0.999)
        else:
            raise ValueError(f"Unknown discrete VP schedule: {schedule}")
        self.betas = betas.float().to(device)
        self.alpha_bar = (1 - betas).cumprod(0).float().to(device)

    def coefficients(self, t: Tensor) -> tuple[Tensor, Tensor]:
        index = (t * self.steps).round().long().sub(1).clamp(0, self.steps - 1)
        a = self.alpha_bar[index]
        # t=0 means exactly clean data, rather than the first noised table entry.
        a = torch.where(t == 0, torch.ones_like(a), a)
        return a.sqrt(), (1 - a).sqrt()

    def prior_std(self) -> float:
        return 1.0


def make_path(config, device="cpu"):
    if config.kind == "ddpm":
        return DiscreteVP(config.steps, config.schedule, device)
    return GaussianPath(
        config.kind,
        config.beta_min,
        config.beta_max,
        config.sigma_min,
        config.sigma_max,
        config.get("logsnr_max", 15.0),
        config.get("logsnr_min", -15.0),
    )


def flow_coefficients(t: Tensor, path: str = "linear"):
    """Generation direction: X_t=a(t)data+s(t)noise; return a,s,a',s'."""
    if path == "linear":
        return t, 1 - t, torch.ones_like(t), -torch.ones_like(t)
    if path == "trigonometric":
        theta = t * math.pi / 2
        return theta.sin(), theta.cos(), math.pi / 2 * theta.cos(), -math.pi / 2 * theta.sin()
    raise ValueError(f"Unknown flow path: {path}")
