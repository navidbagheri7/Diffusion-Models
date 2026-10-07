"""Prediction targets, loss weighting, and exact algebraic conversions."""

import torch
from torch import Tensor

from .paths import expand


def training_target(x0: Tensor, noise: Tensor, alpha: Tensor, sigma: Tensor, prediction: str):
    a, s = expand(alpha, x0), expand(sigma, x0)
    if prediction == "epsilon":
        return noise, torch.ones_like(a)
    if prediction == "x0":
        return x0, torch.ones_like(a)
    if prediction == "score":
        # Conditional Gaussian score; its least-squares optimum is the marginal score.
        return -noise / s.clamp_min(1e-8), s.square()
    if prediction == "v":
        return a * noise - s * x0, torch.ones_like(a)
    raise ValueError(f"Unknown diffusion prediction target: {prediction}")


def convert_prediction(x: Tensor, output: Tensor, alpha: Tensor, sigma: Tensor, prediction: str):
    """Return (x0 estimate, epsilon estimate, marginal score estimate).

    Tweedie: E[X0|Xt=x]=(x+sigma^2 score(x,t))/alpha for Gaussian corruption.
    Endpoint divisions are guarded; callers avoid singular times during integration.
    """
    a, s = expand(alpha, x), expand(sigma, x)
    if prediction == "epsilon":
        eps = output
        x0 = (x - s * eps) / a.clamp_min(1e-8)
    elif prediction == "x0":
        x0 = output
        eps = (x - a * x0) / s.clamp_min(1e-8)
    elif prediction == "score":
        eps = -s * output
        x0 = (x + s.square() * output) / a.clamp_min(1e-8)
    elif prediction == "v":
        denominator = (a.square() + s.square()).clamp_min(1e-8)
        x0 = (a * x - s * output) / denominator
        eps = (s * x + a * output) / denominator
    else:
        raise ValueError(f"Unknown diffusion prediction target: {prediction}")
    return x0, eps, -eps / s.clamp_min(1e-8)
