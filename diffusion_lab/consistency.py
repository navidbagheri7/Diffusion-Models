"""Boundary-conditioned consistency training and diffusion distillation.

Original CT/CD use shared-noise MSE and an EMA target. Improved consistency
training uses a detached online target, lognormal interval sampling and
Pseudo-Huber loss; architectures and training budgets remain configurable.
"""

import torch
from torch import nn

from .edm import karras_sigmas
from .guidance import classifier_free
from .paths import expand


def improved_scales(step, total_steps, minimum=10, maximum=1280):
    """iCT discretization doubles on a fixed training-budget schedule."""
    import math

    if not 0 < minimum <= maximum or total_steps < 1:
        raise ValueError("Invalid improved consistency scale schedule")
    period = max(1, math.ceil(total_steps / (math.log2(maximum / minimum) + 1)))
    return min(minimum * 2 ** (step // period), maximum) + 1


def improved_consistency_loss(
    student,
    clean,
    labels=None,
    intervals=11,
    sigma_max=80.0,
    p_mean=-1.1,
    p_std=2.0,
    generator=None,
):
    """iCT: detached ONLINE target, log-normal interval weights, Pseudo-Huber.

    Generation can still use EMA weights. The target is the current student,
    not its EMA; EMA-target consistency training is a different objective.
    """
    if intervals < 2 or p_std <= 0:
        raise ValueError("iCT requires >=2 levels and positive log-noise deviation")
    core = student.module if hasattr(student, "module") else student
    grid = karras_sigmas(intervals, core.sigma_min, sigma_max, device=clean.device)[:-1].flip(0)
    cdf = torch.special.ndtr((grid.log() - p_mean) / p_std)
    probabilities = (cdf[1:] - cdf[:-1]).clamp_min(0)
    probabilities = probabilities / probabilities.sum()
    index = torch.multinomial(probabilities, len(clean), replacement=True, generator=generator)
    low, high = grid[index], grid[index + 1]
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    # Populate autocast's weight cache with gradient-tracked casts first.
    # A no_grad online-target forward first can cache detached Conv/Linear
    # weights for the student forward in the SAME autocast context.
    prediction = student(clean + expand(high, clean) * noise, high, labels)
    with torch.no_grad():
        destination = core(clean + expand(low, clean) * noise, low, labels)
    delta = prediction.float() - destination.float()
    c = 0.00054 * clean[0].numel() ** 0.5
    # Stable equivalent of sqrt(delta^2+c^2)-c; avoids cancellation at small delta.
    squared_distance = delta.square().flatten(1).sum(1)
    robust = squared_distance / (torch.sqrt(squared_distance + c * c) + c)
    return robust / (high - low)


class ConsistencyModel(nn.Module):
    def __init__(self, network, sigma_min=0.002, sigma_data=0.5):
        super().__init__()
        if sigma_min <= 0 or sigma_data <= 0:
            raise ValueError("Consistency boundary and data standard deviation must be positive")
        self.network = network
        self.sigma_min, self.sigma_data = sigma_min, sigma_data

    def forward(self, x, sigma, labels=None):
        if (sigma <= 0).any():
            raise ValueError("Consistency network evaluations require strictly positive sigma")
        delta = expand(sigma - self.sigma_min, x)
        s = expand(sigma, x)
        c_skip = self.sigma_data**2 / (delta.square() + self.sigma_data**2)
        c_out = self.sigma_data * delta / (s.square() + self.sigma_data**2).sqrt()
        value = self.network(x / (s.square() + self.sigma_data**2).sqrt(), sigma.log() / 4, labels)
        return c_skip * x + c_out * value


def consistency_loss(
    student,
    target,
    clean,
    labels,
    intervals=40,
    sigma_max=80.0,
    teacher=None,
    generator=None,
    weighting="uniform",
    loss_norm="mse",
):
    """CT uses shared-noise neighbors; CD uses a teacher PF-ODE Heun step.

    Target is detached. With a teacher only it defines the neighboring trajectory.
    """
    if intervals < 2:
        raise ValueError("Consistency training needs at least two noise levels")
    if weighting not in {"uniform", "karras", "inverse_gap"}:
        raise ValueError("Unknown consistency loss weighting")
    if loss_norm not in {"mse", "pseudo_huber"}:
        raise ValueError("Unknown consistency discrepancy")
    sigma_min = student.module.sigma_min if hasattr(student, "module") else student.sigma_min
    grid = karras_sigmas(intervals, sigma_min, sigma_max, device=clean.device)[:-1].flip(0)
    indices = torch.randint(intervals - 1, (len(clean),), device=clean.device, generator=generator)
    low, high = grid[indices], grid[indices + 1]
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    noisy_high = clean + expand(high, clean) * noise
    with torch.no_grad():
        if teacher is None:
            noisy_low = clean + expand(low, clean) * noise
        else:
            first = (noisy_high - teacher(noisy_high, high, labels)) / expand(high, clean)
            step = expand(low - high, clean)
            proposal = noisy_high + step * first
            second = (proposal - teacher(proposal, low, labels)) / expand(low, clean)
            noisy_low = noisy_high + step * (first + second) / 2
        destination = target(noisy_low, low, labels)
    prediction = student(noisy_high, high, labels)
    squared = (prediction.float() - destination.float()).square().flatten(1)
    if loss_norm == "pseudo_huber":
        c = 0.00054 * clean[0].numel() ** 0.5
        distance = squared.sum(1)
        loss = distance / (torch.sqrt(distance + c * c) + c)
    else:
        loss = squared.mean(1)
    if weighting == "karras":
        core = student.module if hasattr(student, "module") else student
        loss = loss * (high.reciprocal().square() + core.sigma_data**-2)
    elif weighting == "inverse_gap":
        loss = loss / (high - low)
    return loss


@torch.no_grad()
def consistency_sample(
    model,
    initial,
    steps=1,
    sigma_max=80.0,
    labels=None,
    scale=1.0,
    generator=None,
    sigmas=None,
    clip_denoised=False,
):
    # Every refinement level must exceed the identity boundary sigma_min.
    if sigmas is None:
        sigmas = karras_sigmas(steps + 1, model.sigma_min, sigma_max, device=initial.device)[:-2]
    else:
        sigmas = torch.as_tensor(sigmas, device=initial.device, dtype=initial.dtype)
        if (
            sigmas.ndim != 1
            or len(sigmas) != steps
            or not torch.isfinite(sigmas).all()
            or (sigmas <= model.sigma_min).any()
            or not torch.isclose(sigmas[0], sigmas.new_tensor(sigma_max)).item()
            or not torch.all(sigmas[1:] < sigmas[:-1])
        ):
            raise ValueError(
                "Consistency sigmas must be descending, start at sigma_max, and exceed sigma_min"
            )
    x = initial * sigmas[0]
    history = [x[:32].cpu()]
    calls = 0
    for index, sigma in enumerate(sigmas):
        if index:
            x = x + (sigma.square() - model.sigma_min**2).clamp_min(0).sqrt() * torch.randn(
                x.shape, device=x.device, generator=generator
            )
        t = torch.full((len(x),), float(sigma), device=x.device)
        x = classifier_free(model, x, t, labels, scale)
        if clip_denoised:
            x = x.clamp(-1, 1)
        calls += 2 if labels is not None and scale not in {0.0, 1.0} else 1
        history.append(x[:32].cpu())
    return x, torch.stack(history), calls
