"""Progressive distillation: two deterministic DDIM teacher steps -> one student step.

The student's x0 regression target is solved algebraically from the teacher's
endpoint, using a nested, rounded discrete time grid shared by training/sampling.
This is separate from consistency distillation.
"""

from itertools import pairwise

import torch

from .guidance import classifier_free
from .paths import expand
from .targets import convert_prediction


def progressive_grid(diffusion_steps, steps, device="cpu"):
    if not 1 <= steps <= diffusion_steps:
        raise ValueError("Progressive steps must be within the diffusion table")
    return (
        torch.linspace(diffusion_steps - 1, -1, steps + 1, dtype=torch.float64, device=device)
        .round()
        .long()
    )


def coefficients(path, indices):
    a = path.alpha_bar[indices.clamp_min(0)]
    a = torch.where(indices < 0, torch.ones_like(a), a)
    return a.sqrt(), (1 - a).sqrt()


def ddim_destination(x, x0, high_a, high_s, low_a, low_s):
    ratio = low_s / high_s
    return expand(ratio, x) * x + expand(low_a - ratio * high_a, x) * x0


def distillation_target(x, endpoint, high_a, high_s, low_a, low_s):
    ratio = low_s / high_s
    denominator = low_a - ratio * high_a
    if (denominator.abs() < 1e-8).any():
        raise ValueError("Distillation interval is numerically degenerate")
    return (endpoint - expand(ratio, x) * x) / expand(denominator, x)


@torch.no_grad()
def teacher_prediction(teacher, teacher_config, path, x, indices, labels):
    a, s = coefficients(path, indices)
    output = teacher(x, (indices.float() + 1) / path.steps, labels)
    if teacher_config.learned_variance:
        output = output.chunk(2, 1)[0]
    x0 = convert_prediction(x, output, a, s, teacher_config.prediction)[0]
    return x0.clamp(-1, 1) if teacher_config.algorithm == "improved_ddpm" else x0


def progressive_loss(
    student, teacher, teacher_config, path, clean, labels, steps=16, generator=None
):
    if 2 * steps > path.steps:
        raise ValueError("Teacher's two-step grid exceeds the diffusion table")
    grid = progressive_grid(path.steps, 2 * steps, clean.device)
    intervals = torch.randint(steps, (len(clean),), device=clean.device, generator=generator)
    high, mid, low = grid[2 * intervals], grid[2 * intervals + 1], grid[2 * intervals + 2]
    ah, sh = coefficients(path, high)
    am, sm = coefficients(path, mid)
    al, sl = coefficients(path, low)
    noisy = expand(ah, clean) * clean + expand(sh, clean) * torch.randn(
        clean.shape, device=clean.device, generator=generator
    )
    with torch.no_grad():
        middle = ddim_destination(
            noisy,
            teacher_prediction(teacher, teacher_config, path, noisy, high, labels),
            ah,
            sh,
            am,
            sm,
        )
        endpoint = ddim_destination(
            middle,
            teacher_prediction(teacher, teacher_config, path, middle, mid, labels),
            am,
            sm,
            al,
            sl,
        )
        target = distillation_target(noisy, endpoint, ah, sh, al, sl)
    estimate = student(noisy, (high.float() + 1) / path.steps, labels)
    weight = (ah.square() / sh.square()).clamp_min(1)
    return (estimate.float() - target.float()).square().flatten(1).mean(1) * weight


@torch.no_grad()
def progressive_sample(model, path, initial, steps=16, labels=None, scale=1.0):
    grid = progressive_grid(path.steps, steps, initial.device)
    x, history, calls = initial, [initial[:32].cpu()], 0
    for high, low in pairwise(grid):
        high = high.expand(len(x))
        low = low.expand(len(x))
        ah, sh = coefficients(path, high)
        al, sl = coefficients(path, low)
        output = classifier_free(model, x, (high.float() + 1) / path.steps, labels, scale)
        x = ddim_destination(x, output, ah, sh, al, sl)
        calls += 2 if labels is not None and scale not in {0, 1} else 1
        history.append(x[:32].cpu())
    return x, torch.stack(history), calls


@torch.no_grad()
def nested_teacher_sample(teacher, config, path, initial, steps=32, labels=None, scale=1.0):
    """Initial teacher on the same nested grid used by progressive distillation."""

    def clean_prediction(x, t, condition):
        indices = (t * path.steps).round().long() - 1
        return teacher_prediction(teacher, config, path, x, indices, condition)

    return progressive_sample(clean_prediction, path, initial, steps, labels, scale)
