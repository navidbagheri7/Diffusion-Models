"""Original diagrams inspired by flow papers, plus animated density transport."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from .models import TimeMLP
from .paths import GaussianPath
from .solvers import integrate
from .toy_data import ToyDistribution
from .utils import save_json, seed_everything


def crossing_field(x, t, std=0.12):
    source = x.new_tensor([[-2, -1], [-2, 1]])
    target = x.new_tensor([[2, 1], [2, -1]])
    mean = (1 - t[:, None, None]) * source[None] + t[:, None, None] * target[None]
    variance = std**2 * ((1 - t).square() + t.square())
    delta = x[:, None] - mean
    weights = (-delta.square().sum(-1) / (2 * variance[:, None])).softmax(-1)
    derivative = std**2 * (4 * t - 2)
    velocity = (
        target[None]
        - source[None]
        + derivative[:, None, None] / (2 * variance[:, None, None]) * delta
    )
    return (weights[..., None] * velocity).sum(1)


def flow_reflow_lesson(output):
    source_centers = torch.tensor([[-2, -1], [-2, 1]])
    indices = torch.randint(2, (256,))
    source = source_centers[indices] + 0.12 * torch.randn(256, 2)
    linear_target = torch.tensor([[2, 1], [2, -1]])[indices] + 0.12 * torch.randn_like(source)
    times = torch.linspace(0, 1, 101)
    final, original = integrate(crossing_field, source, times, "rk4")
    linear = (1 - times[:, None, None]) * source[None] + times[:, None, None] * linear_target[None]
    stages = [original]
    models = []
    records = []
    current_target = final
    for iteration in range(3):
        network = TimeMLP(128, 4)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003)
        # Fresh endpoints cover the full smoothed source, not just plotted particles.
        train_source = source_centers[torch.randint(2, (4096,))] + 0.12 * torch.randn(4096, 2)
        if iteration == 0:
            with torch.no_grad():
                train_target = integrate(
                    crossing_field, train_source, torch.linspace(0, 1, 101), "heun"
                )[0]
        else:
            with torch.no_grad():
                train_target = integrate(
                    lambda x, t: models[-1](x, t), train_source, torch.linspace(0, 1, 51), "heun"
                )[0]
        for _ in range(6000):
            index = torch.randint(len(train_source), (256,))
            t = torch.rand(256)
            z, y = train_source[index], train_target[index]
            x = (1 - t[:, None]) * z + t[:, None] * y
            loss = (network(x, t) - (y - z)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval().requires_grad_(False)
        models.append(network)
        with torch.no_grad():
            current_target, trajectory = integrate(
                lambda x, t, net=network: net(x, t), source, times, "heun"
            )
        stages.append(trajectory)
        length = trajectory.diff(dim=0).norm(dim=-1).sum(0)
        distance = (current_target - source).norm(dim=-1)
        records.append(
            {
                "reflow_iteration": iteration + 1,
                "transport_cost_squared": (current_target - source).square().sum(1).mean().item(),
                "mean_excess_path_length": (length - distance).mean().item(),
            }
        )
        torch.save(network.state_dict(), output / f"reflow_{iteration + 1}.pt")
    figure, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True, sharey=True)
    for ax, traj, title in zip(
        axes,
        [linear, *stages],
        [
            "Crossing endpoint interpolation",
            "Marginal rectified flow",
            "Reflow 1",
            "Reflow 2",
            "Reflow 3",
        ],
        strict=True,
    ):
        data = traj.numpy()
        for j in range(48):
            ax.plot(
                data[:, j, 0],
                data[:, j, 1],
                color=plt.cm.tab10(int(indices[j])),
                alpha=0.25,
                lw=0.7,
            )
        ax.scatter(*data[0].T, color="#9467bd", s=5)
        ax.scatter(*data[-1].T, color="#d62728", s=5)
        ax.set(title=title, aspect="equal", xlim=(-2.5, 2.5), ylim=(-1.6, 1.6))
    figure.tight_layout()
    figure.savefig(output / "rectification_reflow.png", dpi=180)
    plt.close(figure)
    save_json(output / "reflow_diagnostics.json", records)
    # Animate particle concentration, using trajectories rather than endpoint lines.
    density_animation(
        stages[-1], output / "reflow_mass.gif", limit=3, title="Learned reflow: probability mass"
    )


def density_animation(trajectory, path, limit=4, title="Probability mass transport", labels=None):
    values = trajectory.detach().cpu().numpy()
    frames = np.unique(np.linspace(0, len(values) - 1, min(50, len(values))).astype(int))
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    scatter = axes[0].scatter(
        *values[0].T,
        s=4,
        c=labels.cpu().numpy() if labels is not None else "#3f77b5",
        cmap="tab10" if labels is not None else None,
        alpha=0.55,
    )
    histogram = np.histogram2d(*values[0].T, bins=80, range=[[-limit, limit], [-limit, limit]])[0].T
    image = axes[1].imshow(
        histogram,
        origin="lower",
        extent=(-limit, limit, -limit, limit),
        cmap="magma",
        vmin=0,
        vmax=max(5, histogram.max()),
    )
    for ax in axes:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")

    def update(index):
        scatter.set_offsets(values[index])
        histogram = np.histogram2d(
            *values[index].T, bins=80, range=[[-limit, limit], [-limit, limit]]
        )[0].T
        image.set_data(histogram)
        figure.suptitle(f"{title}; fraction {index / (len(values) - 1):.2f}")
        return scatter, image

    animation = FuncAnimation(figure, update, frames=frames, interval=80)
    animation.save(path, writer=PillowWriter(fps=12))
    plt.close(figure)


def schedules_and_path_lesson(output):
    t = torch.linspace(0.001, 0.999, 400)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for kind in ["vp_linear", "vp_cosine", "sub_vp", "vp_logsnr", "vp_exponential", "ve"]:
        a, s = GaussianPath(kind).coefficients(t)
        axes[0].plot(t, a, label=kind)
        axes[1].plot(t, s, label=kind)
        axes[2].plot(t, 2 * (a / s).log(), label=kind)
    for ax, title in zip(axes, ["Signal alpha", "Noise std sigma", "log-SNR"], strict=True):
        ax.set(title=title, xlabel="Forward diffusion time")
    axes[1].set_yscale("log")
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "extended_schedules.png", dpi=180)
    plt.close(figure)
    # Generation-direction endpoint paths of translated Gaussians: exact conditional
    # marginal velocity, to isolate path geometry and numerical error.
    z = torch.randn(128, 2) * 0.12 + torch.tensor([-2.0, 0.0])
    mean_data = torch.tensor([2.0, 0.0])
    choices = ["linear", "vp_linear", "sub_vp", "constant_speed_vp"]
    figure, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True, sharey=True)
    for row, steps in enumerate([1, 2, 5, 100]):
        for col, kind in enumerate(choices):
            # a increases from 0 to1, s decreases from1 to0; finite endpoints for VP.
            def coefficients(r, kind=kind):
                if kind == "linear":
                    return r, 1 - r, torch.ones_like(r), -torch.ones_like(r)
                if kind == "constant_speed_vp":
                    a = r.clamp(max=0.9999)
                    s = (1 - a.square()).sqrt()
                    return a, s, torch.ones_like(r), -a / s
                time = (1 - r) * 0.998 + 0.001
                path = GaussianPath(kind)
                a, s = path.coefficients(time)
                beta = path.beta_min + (path.beta_max - path.beta_min) * time
                da = 0.998 * 0.5 * beta * a
                ds = (
                    -0.998 * beta * a.square()
                    if kind == "sub_vp"
                    else -0.998 * 0.5 * beta * a.square() / s
                )
                return a, s, da, ds

            def field(x, r, coefficients=coefficients):
                a, s, da, ds = coefficients(r)
                m = a[:, None] * mean_data + s[:, None] * torch.tensor([-2.0, 0.0])
                v = 0.12**2 * (a.square() + s.square())
                dv = 2 * 0.12**2 * (a * da + s * ds)
                return (
                    da[:, None] * mean_data
                    + ds[:, None] * torch.tensor([-2.0, 0.0])
                    + dv[:, None] / (2 * v[:, None]) * (x - m)
                )

            _, trajectory = integrate(field, z, torch.linspace(0, 1, steps + 1), "euler")
            for j in range(32):
                axes[row, col].plot(*trajectory[:, j].T, color="#397eaf", alpha=0.15, lw=0.8)
            axes[row, col].scatter(*trajectory[0].T, color="#9467bd", s=3)
            axes[row, col].scatter(*trajectory[-1].T, color="#ff7f0e", s=3)
            axes[row, col].set(title=f"{kind}; Euler N={steps}", xlim=(-3, 3), ylim=(-1, 1))
    figure.tight_layout()
    figure.savefig(output / "path_discretization.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/teaching/transport"))
    parser.add_argument("--skip-reflow", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    names = list(ToyDistribution.COUNTS)
    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, name in zip(axes.flat, names, strict=False):
        data = ToyDistribution(name)
        points, labels = data.sample(4000)
        ax.scatter(*points.T, c=labels, cmap="tab10", vmin=0, vmax=9, s=1, alpha=0.6)
        ax.set(title=name.replace("_", " "), aspect="equal", xlim=(-4, 4), ylim=(-4, 4))
    axes.flat[-1].axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "toy_datasets.png", dpi=180)
    plt.close(figure)
    schedules_and_path_lesson(args.output)
    if not args.skip_reflow:
        flow_reflow_lesson(args.output)
    # Same points followed under forward sub-VP corruption with Brownian increments.
    points, labels = ToyDistribution("wavy_rings").sample(2000)
    path = GaussianPath("sub_vp")
    _, trajectory = integrate(
        lambda x, t: path.sde(x, t)[0],
        points,
        torch.linspace(0, 1, 101),
        "euler_maruyama",
        lambda t: path.sde(points, t)[1],
    )
    density_animation(
        trajectory,
        args.output / "subvp_probability_mass.gif",
        title="Forward sub-VP diffusion",
        labels=labels,
    )
    save_json(
        args.output / "provenance.json",
        {
            "seed": 42,
            "figures": "original code-generated figures inspired by rectified-flow literature; not copied paper images",
            "path_comparison": "analytic translated Gaussian example, finite VP endpoints, Euler errors",
            "density_animation": "empirical histogram of simulated particles, not exact PDE solution",
        },
    )


if __name__ == "__main__":
    main()
