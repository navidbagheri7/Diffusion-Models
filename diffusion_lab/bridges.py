"""Brownian bridges and empirical Schrödinger bridge Markovian projection.

A converged endpoint Sinkhorn solution followed by conditional Brownian bridges
gives a finite-support discretization of Brownian-reference SB. Atomic targets
are singular to unrestricted Wiener measure: interpret this construction using
the finite endpoint kernel, not a literal finite path KL against Wiener measure.
A learned drift approximates its Markovian projection. This educational model
does not reproduce all features of the DSBM paper.
"""

import torch

from .paths import expand


def bridge_training_pair(source, target, t, epsilon, noise=None):
    """Return a bridge state, conditional forward drift, and continuity velocity."""
    if epsilon <= 0:
        raise ValueError("Brownian diffusion variance must be positive")
    if noise is None:
        noise = torch.randn_like(source)
    time = expand(t, source)
    sigma = (epsilon * time * (1 - time)).sqrt()
    x = (1 - time) * source + time * target + sigma * noise
    drift = (target - x) / (1 - time).clamp_min(1e-8)
    velocity = target - source + epsilon * (1 - 2 * time) / (2 * sigma.clamp_min(1e-8)) * noise
    return x, drift, velocity


def brownian_bridge_trajectory(source, target, times, epsilon=0.5, generator=None):
    """Exact joint bridge law on a uniform or nonuniform increasing [0,1] grid."""
    if times[0] != 0 or times[-1] != 1 or not torch.all(times[1:] > times[:-1]):
        raise ValueError("Bridge grid must increase from 0 to 1")
    increments = torch.randn(
        len(times) - 1, *source.shape, device=source.device, generator=generator
    )
    dt = (times[1:] - times[:-1]).reshape(-1, *([1] * source.ndim))
    motion = torch.cat([torch.zeros_like(source)[None], (increments * dt.sqrt()).cumsum(0)])
    t = times.reshape(-1, *([1] * source.ndim))
    return (1 - t) * source + t * target + epsilon**0.5 * (motion - t * motion[-1])


def empirical_bridge_field(x, t, source, target, plan, epsilon, velocity=False):
    """Analytic mixture Markov drift (or probability-flow velocity) at 0<t<1.

    Enumerates endpoint pairs; intentionally restricted to small 2D lessons.
    """
    time = t[:, None, None, None]
    mean = (1 - time) * source[None, :, None] + time * target[None, None, :]
    variance = epsilon * time * (1 - time)
    delta = x[:, None, None] - mean
    log_weight = plan.clamp_min(1e-30).log()[None] - delta.square().sum(-1) / (
        2 * variance.squeeze(-1)
    )
    weights = log_weight.flatten(1).softmax(-1).reshape_as(log_weight)
    conditional_drift = (target[None, None] - x[:, None, None]) / (1 - time)
    field = (weights[..., None] * conditional_drift).sum((1, 2))
    if velocity:
        score = (weights[..., None] * (-delta / variance)).sum((1, 2))
        field = field - epsilon / 2 * score
    return field
