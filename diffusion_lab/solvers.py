"""Fixed-step ODE and additive-noise Itô SDE integrators.

Negative dt is supported for reverse diffusion. The Brownian increment always
uses sqrt(abs(dt)); its variance cannot be negative.
"""

from itertools import pairwise

import torch

from .paths import expand


def ode_step(field, x, t, dt, method="heun"):
    k1 = field(x, t)
    if method == "euler":
        return x + dt * k1
    if method == "heun":
        return x + dt / 2 * (k1 + field(x + dt * k1, t + dt))
    if method == "rk4":
        k2 = field(x + dt / 2 * k1, t + dt / 2)
        k3 = field(x + dt / 2 * k2, t + dt / 2)
        k4 = field(x + dt * k3, t + dt)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"Unknown ODE solver: {method}")


def sde_step(drift, diffusion, x, t, dt, method="euler_maruyama", generator=None):
    """Stochastic Heun is for additive noise only (Itô=Stratonovich here)."""
    dw = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * abs(dt) ** 0.5
    g0 = expand(diffusion(t), x)
    b0 = drift(x, t)
    proposal = x + dt * b0 + g0 * dw
    if method == "euler_maruyama":
        return proposal
    if method == "stochastic_heun":
        g1 = expand(diffusion(t + dt), x)
        return x + dt / 2 * (b0 + drift(proposal, t + dt)) + (g0 + g1) / 2 * dw
    raise ValueError(f"Unknown SDE solver: {method}")


@torch.no_grad()
def integrate(
    field, initial, times, method="heun", diffusion=None, generator=None, rtol=1e-5, atol=1e-6
):
    if len(times) < 2:
        raise ValueError("An integration grid needs at least two times")
    if not (torch.all(times[1:] > times[:-1]) or torch.all(times[1:] < times[:-1])):
        raise ValueError("Times must be strictly monotone")
    if method == "rk45":
        if diffusion is not None:
            raise ValueError("Adaptive RK45 is an ODE integrator")
        from scipy.integrate import solve_ivp

        shape = initial.shape

        def rhs(time, values):
            x = torch.as_tensor(values.reshape(shape), device=initial.device, dtype=initial.dtype)
            t = torch.full((len(x),), time, device=x.device, dtype=x.dtype)
            return field(x, t).detach().cpu().double().numpy().reshape(-1)

        result = solve_ivp(
            rhs,
            (times[0].item(), times[-1].item()),
            initial.detach().cpu().double().numpy().reshape(-1),
            t_eval=times.cpu().double().numpy(),
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(f"Adaptive integration failed: {result.message}")
        history = torch.as_tensor(result.y.T.copy()).reshape(len(times), *shape).to(initial.dtype)
        return history[-1].to(initial.device), history
    x = initial
    trajectory = [x.detach().cpu()]
    for start, end in pairwise(times):
        t = torch.full((len(x),), start.item(), device=x.device, dtype=x.dtype)
        dt = (end - start).item()
        if diffusion is None:
            x = ode_step(field, x, t, dt, method)
        else:
            x = sde_step(field, diffusion, x, t, dt, method, generator)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Integrator diverged; inspect endpoints, field, or step size")
        trajectory.append(x.detach().cpu())
    return x, torch.stack(trajectory)
