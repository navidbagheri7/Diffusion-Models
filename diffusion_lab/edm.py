"""EDM noise preconditioning, log-normal training and Karras sampling.

Reference: Karras et al., arXiv:2206.00364, equations 5/8 and algorithms 1/2.
Sigma is a noise standard deviation, not diffusion time or variance.
"""

import math
from itertools import pairwise

import torch
from torch import nn

from .guidance import classifier_free
from .paths import expand


class EDMPreconditioner(nn.Module):
    def __init__(self, network, sigma_data=0.5):
        super().__init__()
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        self.network = network
        self.sigma_data = sigma_data

    def forward(self, x, sigma, labels=None):
        if (sigma <= 0).any():
            raise ValueError("EDM network evaluations require strictly positive sigma")
        s = expand(sigma, x)
        denominator = s.square() + self.sigma_data**2
        c_skip = self.sigma_data**2 / denominator
        c_out = s * self.sigma_data / denominator.sqrt()
        c_in = denominator.rsqrt()
        c_noise = sigma.log() / 4
        return c_skip * x + c_out * self.network(c_in * x, c_noise, labels)


def edm_loss(model, clean, labels=None, p_mean=-1.2, p_std=1.2, generator=None):
    sigma = (
        torch.randn(len(clean), device=clean.device, generator=generator) * p_std + p_mean
    ).exp()
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    noisy = clean + expand(sigma, clean) * noise
    prediction = model(noisy, sigma, labels)
    sigma_data = model.module.sigma_data if hasattr(model, "module") else model.sigma_data
    weight = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()
    return (prediction.float() - clean).square().flatten(1).mean(1) * weight


def karras_sigmas(steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    if steps < 1 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError("Expected positive steps/rho and ordered sigma bounds")
    ramp = torch.linspace(0, 1, steps, device=device, dtype=torch.float64)
    sigmas = (
        sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)]).float()


@torch.no_grad()
def edm_sample(
    model,
    initial,
    steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    solver="heun",
    labels=None,
    scale=1.0,
    churn=0.0,
    churn_min=0.0,
    churn_max=float("inf"),
    noise_scale=1.0,
    generator=None,
    classifier=None,
    classifier_scale=0.0,
):
    """Initial is standard normal; result includes true measured network calls.

    An optional sigma-conditioned classifier contributes sigma^2 grad log p(y|x).
    No evaluation at sigma=0: the final Euler step returns the denoised estimate.
    """
    from .guidance import classifier_gradient

    if solver not in {"euler", "heun"} or churn < 0 or noise_scale <= 0:
        raise ValueError("Invalid EDM sampling configuration")
    sigmas = karras_sigmas(steps, sigma_min, sigma_max, rho, initial.device)
    x, calls = initial * sigmas[0], 0
    history = [x[:32].cpu()]

    def denoise(state, sigma):
        nonlocal calls
        time = torch.full((len(state),), float(sigma), device=state.device)
        calls += 2 if labels is not None and scale not in {0.0, 1.0} else 1
        value = classifier_free(model, state, time, labels, scale)
        if classifier is not None and getattr(
            classifier, "guidance_sigma_min", 0.0
        ) <= sigma <= getattr(classifier, "guidance_sigma_max", float("inf")):
            if labels is None:
                raise ValueError("Classifier guidance requires labels")
            value = value + sigma**2 * classifier_scale * classifier_gradient(
                classifier, state, time.log() / 4, labels
            )
        return value

    for index, (current, following) in enumerate(pairwise(sigmas)):
        gamma = min(churn / steps, math.sqrt(2) - 1) if churn_min <= current <= churn_max else 0
        inflated = current * (1 + gamma)
        if gamma:
            x = x + (inflated.square() - current.square()).sqrt() * noise_scale * torch.randn(
                x.shape, device=x.device, generator=generator
            )
        drift = (x - denoise(x, inflated)) / inflated
        proposal = x + (following - inflated) * drift
        if solver == "heun" and index < len(sigmas) - 2:
            second = (proposal - denoise(proposal, following)) / following
            x = x + (following - inflated) * (drift + second) / 2
        else:
            x = proposal
        if not torch.isfinite(x).all():
            raise FloatingPointError("EDM sampler diverged")
        history.append(x[:32].cpu())
    return x, torch.stack(history), calls
