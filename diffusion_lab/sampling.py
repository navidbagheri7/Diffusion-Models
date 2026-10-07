"""DDPM/DDIM, probability-flow ODE, reverse SDE, and generation-time flows."""

from itertools import pairwise

import torch

from .guidance import classifier_free, classifier_gradient
from .paths import DiscreteVP, expand
from .solvers import integrate, sde_step
from .targets import convert_prediction


@torch.no_grad()
def sample_batched(
    model,
    process,
    sampler,
    initial,
    prediction="epsilon",
    labels=None,
    guidance_scale=1.0,
    classifier=None,
    classifier_scale=1.0,
    generator=None,
    batch_size=64,
):
    """Bound image network memory; retain trajectories for the first batch only.

    Samples are returned on CPU. Stochastic seed reproduction also depends on
    batch size because the random-number traversal order changes.
    """
    outputs, history = [], None
    for offset in range(0, len(initial), batch_size):
        condition = labels[offset : offset + batch_size] if labels is not None else None
        result, trajectory = sample(
            model,
            process,
            sampler,
            initial[offset : offset + batch_size],
            prediction,
            condition,
            guidance_scale,
            classifier,
            classifier_scale,
            generator,
        )
        outputs.append(result.cpu())
        if history is None:
            history = trajectory[:, :32]
    return torch.cat(outputs), history


@torch.no_grad()
def discrete_sample(
    model,
    path,
    initial,
    prediction="epsilon",
    method="ddim",
    steps=100,
    labels=None,
    guidance_scale=1.0,
    eta=0.0,
    clip_x0=False,
    classifier=None,
    classifier_scale=1.0,
    generator=None,
    learned_variance=False,
):
    if method not in {"ddpm", "ddim"} or not 0 <= eta <= 1:
        raise ValueError("Expected DDPM or DDIM and eta in [0,1]")
    if not 1 <= steps <= path.steps:
        raise ValueError("Sampling steps must be in [1, diffusion steps]")
    indices = (
        torch.linspace(path.steps - 1, 0, steps, device=initial.device)
        .round()
        .long()
        .unique(sorted=True)
        .flip(0)
    )
    following = torch.cat([indices[1:], indices.new_tensor([-1])])
    x, history = initial, [initial.cpu()]
    for index, previous in zip(indices, following, strict=True):
        t = torch.full((len(x),), (index.item() + 1) / path.steps, device=x.device)
        a = path.alpha_bar[index]
        ap = path.alpha_bar[previous] if previous >= 0 else torch.ones_like(a)
        variance_values = None
        if learned_variance:
            # CFG applies to the prediction; use the conditional variance head.
            conditional, variance_values = model(x, t, labels).chunk(2, dim=1)
            if labels is not None and guidance_scale != 1:
                unconditional = model(x, t, None).chunk(2, dim=1)[0]
                output = unconditional + guidance_scale * (conditional - unconditional)
            else:
                output = conditional
        else:
            output = classifier_free(model, x, t, labels, guidance_scale)
        x0, eps, score = convert_prediction(
            x, output, a.sqrt().expand(len(x)), (1 - a).sqrt().expand(len(x)), prediction
        )
        if classifier is not None:
            if labels is None:
                raise ValueError("Classifier guidance requires target labels")
            score = score + classifier_scale * classifier_gradient(classifier, x, t, labels)
            eps = -(1 - a).sqrt() * score
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
        if clip_x0:
            x0 = x0.clamp(-1, 1)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
        variance = ((1 - ap) / (1 - a) * (1 - a / ap)).clamp_min(0)
        if method == "ddpm":
            ratio = a / ap
            mean = ap.sqrt() * (1 - ratio) / (1 - a) * x0 + ratio.sqrt() * (1 - ap) / (1 - a) * x
            x = mean
            if previous >= 0:
                if learned_variance:
                    from .improved_ddpm import learned_log_variance

                    log_variance = learned_log_variance(
                        variance_values,
                        variance.clamp_min(1e-20).log().expand(len(x)),
                        (1 - ratio).log().expand(len(x)),
                    )
                    std = (0.5 * log_variance).exp()
                else:
                    std = variance.sqrt()
                x = x + std * torch.randn(x.shape, device=x.device, generator=generator)
        else:
            std = eta * variance.sqrt()
            x = ap.sqrt() * x0 + (1 - ap - std.square()).clamp_min(0).sqrt() * eps
            if eta > 0 and previous >= 0:
                x = x + std * torch.randn(x.shape, device=x.device, generator=generator)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Discrete sampler produced nonfinite samples")
        history.append(x.cpu())
    return x, torch.stack(history)


@torch.no_grad()
def sample(
    model,
    process,
    sampler,
    initial,
    prediction="epsilon",
    labels=None,
    guidance_scale=1.0,
    classifier=None,
    classifier_scale=1.0,
    generator=None,
):
    if process.kind in {"flow", "bridge"}:

        def field(x, t):
            return classifier_free(model, x, t, labels, guidance_scale)

        if classifier is not None:
            raise ValueError("Score-based classifier guidance is implemented for diffusion only")
        if process.kind == "bridge":
            if sampler.method == "bridge_sde":
                if prediction != "bridge_drift":
                    raise ValueError("Bridge SDE requires a forward-drift model")
                times = torch.linspace(
                    process.time_eps, 1 - process.time_eps, sampler.steps + 1, device=initial.device
                )
                return integrate(
                    field,
                    initial,
                    times,
                    sampler.sde_solver,
                    lambda t: torch.ones_like(t) * process.bridge_epsilon**0.5,
                    generator,
                )
            if prediction != "velocity":
                raise ValueError("Bridge ODE requires a continuity-velocity model")
            times = torch.linspace(
                process.time_eps, 1 - process.time_eps, sampler.steps + 1, device=initial.device
            )
        else:
            times = torch.linspace(0, 1, sampler.steps + 1, device=initial.device)
        if sampler.method != "ode":
            raise ValueError("Flow models are sampled with an ODE")
        return integrate(field, initial, times, sampler.ode_solver)
    path = process.path
    if sampler.method in {"dpm_1", "dpm_2m", "dpmpp_1", "dpmpp_2m"}:
        from .dpm_solver import dpm_sample

        if classifier is not None:
            raise ValueError("This DPM lesson uses CFG; classifier-guided DPM is not configured")
        result, history, _ = dpm_sample(
            model,
            path,
            initial,
            prediction,
            sampler.steps,
            sampler.method,
            labels,
            guidance_scale,
            sampler.get("clip_x0", False)
            if hasattr(sampler, "get")
            else getattr(sampler, "clip_x0", False),
        )
        return result, history
    if isinstance(path, DiscreteVP):
        return discrete_sample(
            model,
            path,
            initial,
            prediction,
            sampler.method,
            sampler.steps,
            labels,
            guidance_scale,
            sampler.eta,
            sampler.clip_x0,
            classifier,
            classifier_scale,
            generator,
        )
    if sampler.method not in {"ode", "sde", "predictor_corrector"}:
        raise ValueError("Continuous diffusion requires an ODE or SDE sampler")

    def score_field(x, t):
        alpha, sigma = path.coefficients(t)
        output = classifier_free(model, x, t, labels, guidance_scale)
        score = convert_prediction(x, output, alpha, sigma, prediction)[2]
        if classifier is not None:
            if labels is None:
                raise ValueError("Classifier guidance requires target labels")
            score = score + classifier_scale * classifier_gradient(classifier, x, t, labels)
        return score

    factor = 0.5 if sampler.method == "ode" else 1.0

    def drift(x, t):
        f, g = path.sde(x, t)
        return f - factor * expand(g.square(), x) * score_field(x, t)

    times = torch.linspace(
        1 - process.time_eps, process.time_eps, sampler.steps + 1, device=initial.device
    )
    if sampler.method == "predictor_corrector":
        x, history = initial, [initial.cpu()]
        for start, end in pairwise(times):
            t = torch.full((len(x),), start.item(), device=x.device)
            for _ in range(sampler.corrector_steps):
                gradient = score_field(x, t)
                noise = torch.randn(x.shape, device=x.device, generator=generator)
                ratio = (
                    sampler.snr
                    * noise.flatten(1).norm(dim=1).mean()
                    / gradient.flatten(1).norm(dim=1).mean().clamp_min(1e-8)
                )
                step = 2 * ratio.square()
                x = x + step * gradient + (2 * step).sqrt() * noise
            x = sde_step(
                drift,
                lambda u: path.sde(initial, u)[1],
                x,
                t,
                (end - start).item(),
                sampler.sde_solver,
                generator,
            )
            if not torch.isfinite(x).all():
                raise FloatingPointError("Predictor-corrector sampler diverged")
            history.append(x.cpu())
        return x, torch.stack(history)
    diffusion = None if sampler.method == "ode" else lambda t: path.sde(initial, t)[1]
    return integrate(
        drift,
        initial,
        times,
        sampler.ode_solver if diffusion is None else sampler.sde_solver,
        diffusion,
        generator,
    )
