"""First/second-order multistep DPM-Solver and DPM-Solver++.

Integrates the linear Gaussian part exactly in half-log-SNR lambda=log(alpha/sigma).
This independent implementation supports discrete VP and continuous paths.
Discrete coefficients are interpolated in log alpha between their table nodes.
"""

from itertools import pairwise

import torch

from .guidance import classifier_free
from .paths import DiscreteVP
from .targets import convert_prediction


def interpolated_coefficients(path, t):
    if not isinstance(path, DiscreteVP):
        return path.coefficients(t)
    location = (t * path.steps - 1).clamp(0, path.steps - 1)
    lower = location.floor().long()
    upper = (lower + 1).clamp(max=path.steps - 1)
    log_alpha = path.alpha_bar.log() / 2
    value = log_alpha[lower] + (location - lower) * (log_alpha[upper] - log_alpha[lower])
    return value.exp(), (-torch.expm1(2 * value)).sqrt()


def logsnr_grid(path, steps, device, time_min=None, time_max=0.999):
    if steps < 1:
        raise ValueError("steps must be positive")
    time_min = time_min or (1 / path.steps if isinstance(path, DiscreteVP) else 0.001)
    time_max = 1.0 if isinstance(path, DiscreteVP) else time_max
    endpoints = torch.tensor([time_max, time_min], device=device)
    a, s = interpolated_coefficients(path, endpoints)
    lambdas = torch.linspace((a[0] / s[0]).log(), (a[1] / s[1]).log(), steps + 1, device=device)
    low, high = torch.full_like(lambdas, time_min), torch.full_like(lambdas, time_max)
    for _ in range(40):
        mid = (low + high) / 2
        alpha, sigma = interpolated_coefficients(path, mid)
        value = (alpha / sigma).log()
        low = torch.where(value > lambdas, mid, low)
        high = torch.where(value > lambdas, high, mid)
    return (low + high) / 2


@torch.no_grad()
def dpm_sample(
    model,
    path,
    initial,
    prediction="epsilon",
    steps=20,
    algorithm="dpmpp_2m",
    labels=None,
    scale=1.0,
    clip_x0=False,
    denoise_final=True,
):
    if algorithm not in {"dpm_2m", "dpmpp_2m", "dpm_1", "dpmpp_1"}:
        raise ValueError("Unknown DPM solver variant")
    times = logsnr_grid(path, steps, initial.device)
    x, previous_value, previous_h = initial, None, None
    history = [x[:32].cpu()]
    for current, following in pairwise(times):
        time = current.expand(len(x))
        a, s = interpolated_coefficients(path, time)
        an, sn = interpolated_coefficients(path, following[None])
        output = classifier_free(model, x, time, labels, scale)
        if output.shape[1] == 2 * x.shape[1]:
            output = output.chunk(2, dim=1)[0]
        clean, noise, _ = convert_prediction(x, output, a, s, prediction)
        if clip_x0:
            clean = clean.clamp(-1, 1)
            noise = (x - a[0] * clean) / s[0]
        value = clean if algorithm.startswith("dpmpp") else noise
        h = (an / sn).log()[0] - (a[0] / s[0]).log()
        extrapolation = value
        if previous_value is not None and algorithm.endswith("2m"):
            extrapolation = value + 0.5 * h / previous_h * (value - previous_value)
        if algorithm.startswith("dpmpp"):
            x = sn[0] / s[0] * x - an[0] * torch.expm1(-h) * extrapolation
        else:
            x = an[0] / a[0] * x - sn[0] * torch.expm1(h) * extrapolation
        previous_value, previous_h = value, h
        history.append(x[:32].cpu())
    if denoise_final:
        a, s = interpolated_coefficients(path, times[-1].expand(len(x)))
        output = classifier_free(model, x, times[-1].expand(len(x)), labels, scale)
        if output.shape[1] == 2 * x.shape[1]:
            output = output.chunk(2, dim=1)[0]
        x = convert_prediction(x, output, a, s, prediction)[0]
        if clip_x0:
            x = x.clamp(-1, 1)
        history.append(x[:32].cpu())
    if not torch.isfinite(x).all():
        raise FloatingPointError("DPM solver diverged")
    calls = (steps + int(denoise_final)) * (2 if labels is not None and scale not in {0, 1} else 1)
    return x, torch.stack(history), calls
