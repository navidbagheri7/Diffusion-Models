"""Learned diagonal variance and hybrid/VLB losses from Improved DDPM.

The predicted covariance is diagonal per pixel, not a dense covariance matrix.
Equations are implemented independently from Nichol & Dhariwal (2021).
"""

import math

import torch

from .paths import expand
from .targets import convert_prediction


def posterior(path, x0, x, indices):
    a = path.alpha_bar[indices]
    previous = path.alpha_bar[(indices - 1).clamp_min(0)]
    previous = torch.where(indices == 0, torch.ones_like(previous), previous)
    beta = path.betas[indices]
    variance = beta * (1 - previous) / (1 - a)
    mean = (
        expand(previous.sqrt() * beta / (1 - a), x) * x0
        + expand((1 - beta).sqrt() * (1 - previous) / (1 - a), x) * x
    )
    # q(x_0 | x_1, x_0) is exactly a point mass at x_0. Computing
    # beta_1 / (1-alpha_bar_1) in float32 loses precision for small beta_1.
    mean = torch.where(expand(indices == 0, x), x0, mean)
    # First-step posterior is a point mass. Use the second-step variance for logs.
    second = path.betas[1] * (1 - path.alpha_bar[0]) / (1 - path.alpha_bar[1])
    log_variance = torch.where(indices == 0, second, variance).clamp_min(1e-20).log()
    return mean, log_variance


def learned_log_variance(values, minimum_log, maximum_log):
    # The published learned-range parameterization does not clamp the network.
    fraction = (values + 1) / 2
    return fraction * expand(maximum_log, values) + (1 - fraction) * expand(minimum_log, values)


def discretized_log_likelihood(x, mean, log_variance):
    """8-bit [-1,1] discretized Gaussian likelihood with exact Gaussian CDF."""
    inverse_std = (-0.5 * log_variance).exp()
    upper = (x - mean + 1 / 255) * inverse_std
    lower = (x - mean - 1 / 255) * inverse_std
    cdf_upper, cdf_lower = torch.special.ndtr(upper), torch.special.ndtr(lower)
    center = (cdf_upper - cdf_lower).clamp_min(1e-12).log()
    return torch.where(
        x < -0.999,
        cdf_upper.clamp_min(1e-12).log(),
        torch.where(x > 0.999, (1 - cdf_lower).clamp_min(1e-12).log(), center),
    )


def stable_discretized_log_likelihood(x, mean, log_variance):
    """Float64 log-CDF intervals; no probability floor caps reported ELBOs."""
    x, mean, log_variance = x.double(), mean.double(), log_variance.double()
    inverse_std = (-0.5 * log_variance).exp()
    center = (x - mean) * inverse_std
    upper, lower = center + inverse_std / 255, center - inverse_std / 255
    positive = center >= 0
    large = torch.special.log_ndtr(torch.where(positive, -lower, upper))
    small = torch.special.log_ndtr(torch.where(positive, -upper, lower))
    difference = (small - large).clamp_max(-torch.finfo(torch.float64).eps)
    middle = large + (-torch.expm1(difference)).log()
    # Thin-interval midpoint expansion has relative error below ~1e-8 here.
    thin = (2 * inverse_std / 255) * (1 + center.abs()) < 1e-4
    midpoint = (
        -0.5 * center.square()
        - 0.5 * math.log(2 * math.pi)
        + math.log(2 / 255)
        - 0.5 * log_variance
    )
    middle = torch.where(thin, midpoint, middle)
    return torch.where(
        x < -0.999,
        torch.special.log_ndtr(upper),
        torch.where(x > 0.999, torch.special.log_ndtr(-lower), middle),
    )


def variational_bound(
    path,
    clean,
    noisy,
    indices,
    estimate,
    variance_values=None,
    prediction="epsilon",
    detach_mean=False,
    clip_x0=False,
    stable_decoder=False,
):
    a = path.alpha_bar[indices]
    x0 = convert_prediction(
        noisy, estimate.detach() if detach_mean else estimate, a.sqrt(), (1 - a).sqrt(), prediction
    )[0]
    if clip_x0:
        x0 = x0.clamp(-1, 1)
    true_mean, true_logvar = posterior(path, clean, noisy, indices)
    predicted_mean, minimum_log = posterior(path, x0, noisy, indices)
    predicted_logvar = (
        expand(minimum_log, noisy)
        if variance_values is None
        else learned_log_variance(variance_values, minimum_log, path.betas[indices].log())
    )
    kl = 0.5 * (
        -1
        + predicted_logvar
        - expand(true_logvar, noisy)
        + (expand(true_logvar, noisy) - predicted_logvar).exp()
        + (true_mean - predicted_mean).square() * (-predicted_logvar).exp()
    )
    decoder_function = (
        stable_discretized_log_likelihood if stable_decoder else discretized_log_likelihood
    )
    decoder = -decoder_function(clean, predicted_mean, predicted_logvar)
    per_example = torch.where(expand(indices == 0, clean), decoder, kl).flatten(1).mean(
        1
    ) / math.log(2)
    return per_example


def improved_loss(
    path,
    clean,
    noisy,
    t,
    target,
    output,
    prediction="epsilon",
    objective="hybrid",
    hybrid_weight=0.001,
    loss_weight=1.0,
    learned_variance=True,
):
    indices = (t * path.steps).round().long() - 1
    estimate, variance = output.chunk(2, dim=1) if learned_variance else (output, None)
    mse = ((estimate.float() - target).square() * loss_weight).flatten(1).mean(1)
    if objective == "simple":
        if learned_variance:
            raise ValueError("A variance head requires hybrid or VLB supervision")
        return mse
    vb = variational_bound(
        path,
        clean,
        noisy,
        indices,
        estimate.float(),
        variance.float() if variance is not None else None,
        prediction,
        detach_mean=objective == "hybrid",
    )
    if objective == "vlb":
        return path.steps * vb
    if objective != "hybrid":
        raise ValueError("objective must be simple, hybrid or vlb")
    return mse + hybrid_weight * path.steps * vb


class LossSecondMomentSampler:
    """Online p(t) proportional to RMS loss, with an unbiased 1/(T p(t)) factor."""

    def __init__(self, steps, history=10, uniform_probability=0.001, device="cpu"):
        self.values = torch.zeros(steps, history, dtype=torch.float64, device=device)
        self.counts = torch.zeros(steps, dtype=torch.long, device=device)
        self.uniform_probability = uniform_probability

    def probabilities(self):
        if (self.counts < self.values.shape[1]).any():
            return torch.full_like(self.counts, 1 / len(self.counts), dtype=torch.float64)
        rms = self.values.square().mean(1).sqrt().clamp_min(1e-12)
        return (1 - self.uniform_probability) * rms / rms.sum() + self.uniform_probability / len(
            rms
        )

    def sample(self, n, generator=None):
        probabilities = self.probabilities()
        indices = torch.multinomial(probabilities, n, replacement=True, generator=generator)
        return indices, (1 / (len(probabilities) * probabilities[indices])).float()

    @torch.no_grad()
    def update(self, indices, losses):
        for index, loss in zip(indices, losses, strict=True):
            self.values[index] = self.values[index].roll(-1)
            self.values[index, -1] = loss.detach()
            self.counts[index] += 1

    def state_dict(self):
        return {"values": self.values, "counts": self.counts}

    def load_state_dict(self, state):
        self.values.copy_(state["values"])
        self.counts.copy_(state["counts"])
