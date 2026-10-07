"""Continuous change of variables and complete DDPM negative ELBO estimators.

Numerical ODE densities, stochastic trace estimates, and variational bounds
are distinct quantities. Neither Monte Carlo ELBO nor an approximate ODE solve
is labeled an exact image likelihood.
"""

import math
from itertools import pairwise

import torch

from .improved_ddpm import posterior, variational_bound
from .paths import expand
from .targets import convert_prediction


def gaussian_log_density(x, std=1.0):
    if std <= 0:
        raise ValueError("Gaussian standard deviation must be positive")
    return (
        (-0.5 * (x / std).square() - math.log(std) - 0.5 * math.log(2 * math.pi)).flatten(1).sum(1)
    )


def field_and_divergence(field, x, t, probes=None):
    """Exact trace (small dimensions) or fixed-probe Hutchinson estimate.

    Each batch element must have an independent field: batch-statistic layers
    that couple examples are incompatible with this per-example calculation.
    """
    with torch.enable_grad():
        state = x.detach().requires_grad_(True)
        value = field(state, t)
        if value.shape != state.shape:
            raise ValueError("Density field must have the same shape as its state")
        if not value.requires_grad:
            return value.detach(), x.new_zeros(len(x))
        traces = []
        if probes is None:
            dimension = x[0].numel()
            if dimension > 32:
                raise ValueError("Use stochastic trace estimation above 32 dimensions")
            for index in range(dimension):
                gradient = torch.autograd.grad(
                    value.flatten(1)[:, index].sum(),
                    state,
                    retain_graph=index < dimension - 1,
                    allow_unused=True,
                )[0]
                traces.append(
                    gradient.flatten(1)[:, index] if gradient is not None else x.new_zeros(len(x))
                )
            trace = torch.stack(traces).sum(0)
        else:
            for index, probe in enumerate(probes):
                gradient = torch.autograd.grad(
                    (value * probe).sum(),
                    state,
                    retain_graph=index < len(probes) - 1,
                    allow_unused=True,
                )[0]
                traces.append(
                    (gradient * probe).flatten(1).sum(1)
                    if gradient is not None
                    else x.new_zeros(len(x))
                )
            trace = torch.stack(traces).mean(0)
    return value.detach(), trace.detach()


def ode_log_density(
    field,
    data,
    times,
    terminal_log_density,
    solver="rk4",
    trace="exact",
    probe_count=1,
    generator=None,
):
    """Integrate data -> a known base density with d(log p)/dt=-div(v).

    The returned log p(data) is log p(terminal) + integral div(v) dt.
    Supports increasing corruption time AND decreasing generation time.
    Probes remain fixed throughout a solve; resampling at RHS evaluations
    would turn the numerical ODE right-hand side into stochastic noise.
    """
    if len(times) < 2 or not (
        torch.all(times[1:] > times[:-1]) or torch.all(times[1:] < times[:-1])
    ):
        raise ValueError("Likelihood times must be strictly monotone")
    if (
        solver not in {"euler", "heun", "rk4"}
        or trace not in {"exact", "hutchinson"}
        or probe_count < 1
    ):
        raise ValueError("Invalid density solver/trace settings")
    probes = None
    if trace == "hutchinson":
        probes = (
            torch.randint(2, (probe_count, *data.shape), device=data.device, generator=generator)
            * 2
            - 1
        ).to(data.dtype)
    state, integral = data.detach(), data.new_zeros(len(data))
    calls = 0

    def rhs(x, time):
        nonlocal calls
        calls += 1
        t = torch.full((len(x),), float(time), device=x.device, dtype=x.dtype)
        return field_and_divergence(field, x, t, probes)

    for start, end in pairwise(times):
        dt = float(end - start)
        v1, d1 = rhs(state, start)
        if solver == "euler":
            state, integral = state + dt * v1, integral + dt * d1
        elif solver == "heun":
            v2, d2 = rhs(state + dt * v1, end)
            state, integral = state + dt * (v1 + v2) / 2, integral + dt * (d1 + d2) / 2
        else:
            v2, d2 = rhs(state + dt * v1 / 2, start + dt / 2)
            v3, d3 = rhs(state + dt * v2 / 2, start + dt / 2)
            v4, d4 = rhs(state + dt * v3, end)
            state, integral = (
                state + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6,
                integral + dt * (d1 + 2 * d2 + 2 * d3 + d4) / 6,
            )
        if not torch.isfinite(state).all() or not torch.isfinite(integral).all():
            raise FloatingPointError("Likelihood integration diverged")
    return terminal_log_density(state) + integral, {
        "field_calls": calls,
        "trace": trace,
        "probe_count": probe_count if probes is not None else 0,
        "solver": solver,
        "integration_steps": len(times) - 1,
        "start_time": float(times[0]),
        "end_time": float(times[-1]),
        "dimensions_per_example": data[0].numel(),
        "terminal": state,
        "integrated_divergence": integral,
    }


def prior_kl_bpd(path, clean):
    a = path.alpha_bar[-1].double()
    variance = 1 - a
    return (0.5 * (a * clean.double().square() + variance - 1 - variance.log())).flatten(1).mean(
        1
    ) / math.log(2)


@torch.no_grad()
def ddpm_elbo(
    model,
    path,
    clean,
    prediction="epsilon",
    labels=None,
    learned_variance=False,
    quantized=True,
    timestep_samples=None,
    generator=None,
    clip_x0=None,
):
    """Complete negative ELBO in bits/dimension, including terminal prior KL.

    Full mode sums ALL T decoder/transition terms, with one forward-noise
    Monte Carlo draw per term. Optional uniformly sampled terms give an
    unbiased estimate of that sum, not a deterministic exact NLL.
    For continuous toy data, replace the 8-bit decoder by a Gaussian density.
    """
    if timestep_samples is not None and timestep_samples < 1:
        raise ValueError("Timestep sample count must be positive")
    clip_x0 = quantized if clip_x0 is None else clip_x0
    prior = prior_kl_bpd(path, clean)
    decoder, transitions = torch.zeros_like(prior), torch.zeros_like(prior)
    terms = path.steps if timestep_samples is None else timestep_samples
    # Always evaluate the decoder explicitly; sample transition terms separately.
    indices_list = [0] + (
        list(range(1, path.steps))
        if timestep_samples is None
        else torch.randint(
            1, path.steps, (terms,), device=clean.device, generator=generator
        ).tolist()
    )
    transition_factor = 1.0 if timestep_samples is None else (path.steps - 1) / terms
    for index in indices_list:
        indices = torch.full((len(clean),), index, device=clean.device, dtype=torch.long)
        t = (indices.float() + 1) / path.steps
        a = path.alpha_bar[index]
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        output = model(noisy, t, labels).float()
        estimate, variance = output.chunk(2, dim=1) if learned_variance else (output, None)
        values = variational_bound(
            path,
            clean,
            noisy,
            indices,
            estimate,
            variance,
            prediction,
            clip_x0=clip_x0,
            stable_decoder=True,
        )
        if index == 0 and not quantized:
            from .improved_ddpm import learned_log_variance

            x0 = convert_prediction(
                noisy,
                estimate,
                a.sqrt().expand(len(clean)),
                (1 - a).sqrt().expand(len(clean)),
                prediction,
            )[0]
            if clip_x0:
                x0 = x0.clamp(-1, 1)
            mean, minimum = posterior(path, x0, noisy, indices)
            logvar = (
                expand(minimum, noisy)
                if variance is None
                else learned_log_variance(variance, minimum, path.betas[indices].log())
            )
            values = (
                0.5 * (math.log(2 * math.pi) + logvar + (clean - mean).square() * (-logvar).exp())
            ).flatten(1).mean(1) / math.log(2)
        if index == 0:
            decoder += values
        else:
            transitions += transition_factor * values
    return {
        "negative_elbo_bpd": prior + decoder + transitions,
        "terminal_prior_kl_bpd": prior,
        "decoder_bpd": decoder,
        "transition_kl_bpd": transitions,
        "network_calls": len(indices_list),
        "clean_prediction_clipped": bool(clip_x0),
        "decoder_numerics": "Float64 tail-stable log CDF difference; no probability floor"
        if quantized
        else "Continuous Gaussian log density",
        "decoder": "8-bit discretized Gaussian"
        if quantized
        else "Continuous Gaussian density (bits/dimension)",
        "estimator": "All T terms; one Monte Carlo noise draw per term"
        if timestep_samples is None
        else "Exact terminal prior, explicit MC decoder, uniformly sampled transition sum",
        "is_exact_nll": False,
    }
