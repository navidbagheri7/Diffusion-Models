"""Generate mathematical lessons without a trained network or dataset download.

python -m diffusion_lab.lessons --output assets/teaching/gaussian_diffusion
All fields labeled analytic use a known Gaussian-mixture density, not neural fits.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .bridges import brownian_bridge_trajectory, empirical_bridge_field
from .data import GaussianMixture
from .metrics import gmm_metrics
from .paths import DiscreteVP, GaussianPath, expand, flow_coefficients
from .solvers import integrate
from .transport import exact_pairing, sample_coupling, sinkhorn_plan
from .utils import save_json, seed_everything
from .visualize import particle_animation, scatter_comparison


def save_figure(figure, directory, name):
    figure.tight_layout()
    figure.savefig(directory / name, dpi=160)
    plt.close(figure)


def schedules(directory):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    t = torch.linspace(0.001, 0.999, 400)
    for kind in ["vp_linear", "vp_cosine", "ve"]:
        path = GaussianPath(kind)
        a, s = path.coefficients(t)
        axes[0].plot(t, a, label=kind)
        axes[1].plot(t, s, label=kind)
        axes[2].plot(t, path.sde(torch.ones(len(t), 2), t)[1], label=kind)
    for ax, title in zip(
        axes,
        [
            "Signal coefficient alpha(t)",
            "Noise standard deviation sigma(t)",
            "SDE diffusion coefficient g(t)",
        ],
        strict=True,
    ):
        ax.set(title=title, xlabel="Forward diffusion time (data → noise)")
        ax.legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    save_figure(figure, directory, "schedules.png")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for schedule in ["linear", "cosine"]:
        path = DiscreteVP(1000, schedule)
        axes[0].plot(path.betas, label=schedule)
        axes[1].plot(path.alpha_bar, label=schedule)
    axes[0].set(title="Discrete per-step beta", xlabel="Table index")
    axes[1].set(title="Cumulative signal variance alpha_bar", xlabel="Table index", yscale="log")
    for ax in axes:
        ax.legend()
    save_figure(figure, directory, "discrete_schedules.png")


def corruption_and_tweedie(directory, mixture):
    x0, _ = mixture.sample(1500)
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharex=True, sharey=True)
    path = GaussianPath()
    for ax, time in zip(axes, [0.0, 0.15, 0.35, 0.65, 1.0], strict=True):
        a, s = path.coefficients(torch.tensor([time]))
        x = a * x0 + s * torch.randn_like(x0)
        ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.4)
        ax.set(title=f"t={time:g}", xlim=(-5, 5), ylim=(-5, 5), aspect="equal")
    figure.suptitle("Forward corruption: marginal snapshots (not particle trajectories)")
    save_figure(figure, directory, "forward_corruption.png")
    t = torch.full((200,), 0.25)
    a, s = path.coefficients(t)
    x = a[:, None] * x0[:200] + s[:, None] * torch.randn(200, 2)
    score = mixture.score(x, a, s)
    denoised = (x + s[:, None].square() * score) / a[:, None]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x[:, 0], x[:, 1], s=10, label="Noisy observation")
    axes[0].quiver(x[:, 0], x[:, 1], score[:, 0], score[:, 1], alpha=0.5)
    axes[0].set_title("Score points toward higher noised density")
    axes[1].scatter(x[:, 0], x[:, 1], s=10, alpha=0.4, label="Noisy")
    axes[1].scatter(denoised[:, 0], denoised[:, 1], s=10, label="Posterior mean E[X0|Xt]")
    axes[1].quiver(
        x[:, 0],
        x[:, 1],
        (denoised - x)[:, 0],
        (denoised - x)[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        alpha=0.2,
    )
    axes[1].set_title("Tweedie denoiser: a conditional mean")
    for ax in axes:
        ax.set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal")
        ax.legend(fontsize=8)
    save_figure(figure, directory, "score_tweedie.png")


def pde_lesson(directory, mixture):
    """Finite-difference comparison of density evolution and both PDE right sides."""
    axis = np.linspace(-6, 6, 181)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = torch.tensor(np.stack([xx.ravel(), yy.ravel()], -1), dtype=torch.float64)
    path = GaussianPath()
    time, dt, dx = 0.3, 1e-4, axis[1] - axis[0]

    def density(t):
        a, s = path.coefficients(torch.full((len(points),), t, dtype=torch.float64))
        return mixture.log_prob(points, a, s).exp().numpy().reshape(xx.shape)

    p = density(time)
    derivative = (density(time + dt) - density(time - dt)) / (2 * dt)
    t = torch.full((len(points),), time, dtype=torch.float64)
    a, s = path.coefficients(t)
    f, g = path.sde(points, t)
    v = f - expand(g.square() / 2, points) * mixture.score(points, a, s)
    f, v = f.numpy().reshape(*xx.shape, 2), v.numpy().reshape(*xx.shape, 2)

    def divergence(flux):
        return np.gradient(flux[..., 0], dx, axis=1, edge_order=2) + np.gradient(
            flux[..., 1], dx, axis=0, edge_order=2
        )

    laplacian = np.gradient(
        np.gradient(p, dx, axis=0, edge_order=2), dx, axis=0, edge_order=2
    ) + np.gradient(np.gradient(p, dx, axis=1, edge_order=2), dx, axis=1, edge_order=2)
    fp = -divergence(f * p[..., None]) + g[0].item() ** 2 / 2 * laplacian
    continuity = -divergence(v * p[..., None])
    interior = np.s_[3:-3, 3:-3]
    scale = np.sqrt(np.mean(derivative[interior] ** 2))
    results = {
        "fokker_planck_relative_rms_residual": float(
            np.sqrt(np.mean((derivative - fp)[interior] ** 2)) / scale
        ),
        "continuity_relative_rms_residual": float(
            np.sqrt(np.mean((derivative - continuity)[interior] ** 2)) / scale
        ),
        "spatial_dx": dx,
        "temporal_dt": dt,
        "time": time,
        "note": "Finite differences on a truncated domain; boundary cells excluded",
    }
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmax = np.max(np.abs(derivative))
    for ax, values, title in zip(
        axes,
        [p, derivative, fp, continuity],
        ["Density p_t", "Finite difference ∂t p", "Fokker–Planck RHS", "Continuity RHS"],
        strict=True,
    ):
        artist = ax.imshow(
            values,
            origin="lower",
            extent=(-6, 6, -6, 6),
            cmap="viridis" if title == "Density p_t" else "coolwarm",
            vmin=0 if title == "Density p_t" else -vmax,
            vmax=None if title == "Density p_t" else vmax,
        )
        ax.set_title(title)
        figure.colorbar(artist, ax=ax, shrink=0.65)
    save_figure(figure, directory, "density_equations.png")
    save_json(directory / "pde_residuals.json", results)
    return results


def reverse_dynamics(directory, mixture):
    path = GaussianPath()
    n = 1024
    initial_data, _ = mixture.sample(n)
    times = torch.linspace(0.999, 0.001, 301)
    a, s = path.coefficients(times[:1])
    initial = a * initial_data + s * torch.randn_like(initial_data)
    histories = {}
    metrics = {}
    reference = mixture.sample(n)[0]
    for method, factor, solver in [("ode", 0.5, "heun"), ("sde", 1.0, "euler_maruyama")]:

        def field(x, t, factor=factor):
            alpha, sigma = path.coefficients(t)
            f, g = path.sde(x, t)
            return f - expand(g.square() * factor, x) * mixture.score(x, alpha, sigma)

        diffusion = (lambda t: path.sde(initial, t)[1]) if method == "sde" else None
        result, history = integrate(field, initial, times, solver, diffusion)
        histories[method] = history
        scatter_comparison(
            reference,
            result,
            directory / f"analytic_{method}_samples.png",
            f"Analytic {method.upper()}",
        )
        particle_animation(
            history, directory / f"analytic_{method}.gif", f"Analytic VP {method.upper()}"
        )
        metrics[method] = gmm_metrics(result, reference, mixture)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, method in zip(axes, ["ode", "sde"], strict=True):
        history = histories[method]
        for particle in range(12):
            ax.plot(history[:, particle, 0], history[:, particle, 1], linewidth=0.8, alpha=0.8)
        ax.scatter(
            history[0, :12, 0], history[0, :12, 1], marker="x", c="black", label="Noise start"
        )
        ax.scatter(
            history[-1, :12, 0], history[-1, :12, 1], s=20, c="red", label="Generated endpoint"
        )
        ax.set(
            title=f"{method.upper()} trajectories: same initial particles",
            xlim=(-5, 5),
            ylim=(-5, 5),
            aspect="equal",
        )
        ax.legend(fontsize=8)
    save_figure(figure, directory, "ode_sde_trajectories.png")
    save_json(directory / "analytic_sampling_metrics.json", metrics)


def solver_lesson(directory):
    steps = [4, 8, 16, 32, 64]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    results = {}
    for method in ["euler", "heun", "rk4"]:
        errors = []
        for n in steps:
            x, _ = integrate(
                lambda x, t: -x,
                torch.ones(1, 1, dtype=torch.float64),
                torch.linspace(0, 1, n + 1, dtype=torch.float64),
                method,
            )
            errors.append(abs(x.item() - np.exp(-1)))
        axes[0].loglog(steps, errors, "o-", label=method)
        results[method] = errors
    axes[0].set(
        title="ODE error: dx/dt=-x, x(0)=1", xlabel="Steps", ylabel="Absolute endpoint error"
    )
    axes[0].legend()
    times = torch.linspace(0, 1, 101)
    for method in ["euler_maruyama", "stochastic_heun"]:
        result, history = integrate(
            lambda x, t: -x, torch.zeros(4096, 1), times, method, lambda t: torch.ones_like(t)
        )
        axes[1].plot(times, history[:, :, 0].var(1), label=method)
        results[method + "_endpoint_variance"] = result.var().item()
    axes[1].plot(times, (1 - torch.exp(-2 * times)) / 2, "k--", label="Analytic OU variance")
    axes[1].set(title="SDE ensemble variance (Monte Carlo)", xlabel="Time", ylabel="Variance")
    axes[1].legend(fontsize=8)
    save_figure(figure, directory, "solver_accuracy.png")
    save_json(directory / "solver_accuracy.json", results)


def flow_and_ot(directory, mixture):
    source = torch.randn(128, 2)
    target, _ = mixture.sample(128)
    assigned = exact_pairing(source, target)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ends, title in zip(
        axes,
        [target, assigned],
        ["Independent endpoint coupling", "Exact empirical OT coupling"],
        strict=True,
    ):
        for start, end in zip(source[:48], ends[:48], strict=True):
            ax.plot([start[0], end[0]], [start[1], end[1]], alpha=0.25, linewidth=0.7)
        ax.scatter(source[:, 0], source[:, 1], s=8, label="Noise")
        ax.scatter(ends[:, 0], ends[:, 1], s=8, label="Data")
        ax.set(title=title, xlim=(-5, 5), ylim=(-5, 5), aspect="equal")
        ax.legend(fontsize=8)
    save_figure(figure, directory, "ot_couplings.png")
    times = torch.linspace(0, 1, 101)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind in zip(axes, ["linear", "trigonometric"], strict=True):
        a, s, _, _ = flow_coefficients(times, kind)
        trajectory = a[:, None, None] * target[None] + s[:, None, None] * source[None]
        for i in range(12):
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], alpha=0.6)
        ax.set(title=f"Conditional {kind} flow paths", xlim=(-5, 5), ylim=(-5, 5), aspect="equal")
        particle_animation(
            trajectory,
            directory / f"flow_{kind}_conditional.gif",
            f"Conditional {kind} interpolation",
        )
    save_figure(figure, directory, "flow_paths.png")

    # Analytic marginal velocity: average conditional velocities given the state.
    def velocity(x, t):
        a, s, _, _ = flow_coefficients(t)
        weights = mixture.log_components(x, a, s).softmax(-1)
        variance = a.square() * mixture.std**2 + s.square()
        conditional_data = mixture.means[None] + (mixture.std**2 * a / variance)[:, None, None] * (
            x[:, None] - a[:, None, None] * mixture.means[None]
        )
        data_mean = (weights[..., None] * conditional_data).sum(1)
        noise_mean = (x - a[:, None] * data_mean) / s[:, None].clamp_min(1e-8)
        return data_mean - noise_mean

    initial = torch.randn(1024, 2)
    # Stop a tiny distance short of the algebraically singular noise conversion.
    result, history = integrate(velocity, initial, torch.linspace(0, 0.999, 301), "heun")
    particle_animation(
        history,
        directory / "flow_marginal.gif",
        "Analytic marginal flow (not independent line segments)",
    )
    scatter_comparison(
        mixture.sample(1024)[0],
        result,
        directory / "flow_marginal_samples.png",
        "Analytic marginal flow",
    )
    save_json(
        directory / "ot_costs.json",
        {
            "independent_mean_squared_cost": (source - target).square().sum(1).mean().item(),
            "empirical_ot_mean_squared_cost": (source - assigned).square().sum(1).mean().item(),
            "note": "Batch-level OT is not a guarantee of population OT",
        },
    )


def bridge_lesson(directory, mixture):
    source = torch.randn(48, 2)
    target = mixture.sample(48)[0]
    times = torch.linspace(0, 1, 81)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    costs = {}
    for ax, epsilon in zip(axes, [0.2, 0.7, 2.0], strict=True):
        plan = sinkhorn_plan(source, target, epsilon, iterations=4000)
        z, y = sample_coupling(source, target, plan, 256)
        history = brownian_bridge_trajectory(z, y, times, epsilon)
        for i in range(10):
            ax.plot(history[:, i, 0], history[:, i, 1], alpha=0.6, linewidth=0.7)
        ax.set(
            title=f"Empirical Brownian-reference SB, ε={epsilon}",
            xlim=(-5, 5),
            ylim=(-5, 5),
            aspect="equal",
        )
        particle_animation(
            history,
            directory / f"brownian_bridge_{epsilon:g}.gif",
            f"Endpoint-coupled Brownian bridges, ε={epsilon}",
        )
        costs[str(epsilon)] = {
            "coupled_squared_cost": (plan * torch.cdist(source, target).square()).sum().item(),
            "row_residual": (plan.sum(1) - 1 / len(source)).abs().max().item(),
        }
    save_figure(figure, directory, "entropic_bridges.png")
    epsilon = 0.7
    plan = sinkhorn_plan(source, target, epsilon)
    axis = torch.linspace(-4, 4, 15)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], -1)
    t = torch.full((len(grid),), 0.5)
    drift = empirical_bridge_field(grid, t, source, target, plan, epsilon)
    velocity = empirical_bridge_field(grid, t, source, target, plan, epsilon, velocity=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, title in zip(
        axes,
        [drift, velocity],
        [
            "Markovian projected SDE drift E[(Y-X)/(1-t)|X]",
            "Continuity velocity: drift − ε score/2",
        ],
        strict=True,
    ):
        ax.quiver(grid[:, 0], grid[:, 1], field[:, 0], field[:, 1])
        ax.set(title=title, xlim=(-5, 5), ylim=(-5, 5), aspect="equal")
        ax.title.set_fontsize(9)
    save_figure(figure, directory, "bridge_matching_fields.png")
    save_json(directory / "bridge_couplings.json", costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/teaching/gaussian_diffusion")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    seed_everything(args.seed, threads=4)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    mixture = GaussianMixture()
    for name, function in [
        ("schedules", lambda: schedules(directory)),
        ("corruption and Tweedie", lambda: corruption_and_tweedie(directory, mixture)),
        ("density PDEs", lambda: pde_lesson(directory, mixture)),
        ("reverse dynamics", lambda: reverse_dynamics(directory, mixture)),
        ("numerical solvers", lambda: solver_lesson(directory)),
        ("flow and OT", lambda: flow_and_ot(directory, mixture)),
        ("bridges", lambda: bridge_lesson(directory, mixture)),
    ]:
        print(f"Generating lesson: {name}", flush=True)
        function()
    save_json(
        directory / "provenance.json",
        {
            "command": f"python -m diffusion_lab.lessons --seed {args.seed} --output {args.output}",
            "seed": args.seed,
            "fields": "analytic GMM; no trained networks",
            "torch": torch.__version__,
        },
    )


if __name__ == "__main__":
    main()
