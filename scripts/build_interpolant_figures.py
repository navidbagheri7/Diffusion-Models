"""Reproduce illustrative conditional paths; these are not marginal ODE trajectories."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion_lab.interpolants import StochasticInterpolant


def curved_base(source, target, time):
    time = time[:, None]
    return (1 - time) * source + time * target + 3 * time * (1 - time) * torch.sin(source + target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/teaching/interpolants"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    source = torch.randn(40, 2, generator=generator) * 0.18 + torch.tensor([-2.0, -1.0])
    target = torch.randn(40, 2, generator=generator) * 0.18 + torch.tensor([2.0, 1.6])
    times = torch.linspace(0, 1, 201)
    # A conditional interpolant trajectory keeps the SAME Z across time.
    noise = torch.randn(source.shape, generator=generator)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    for axis, title, base, amplitude in zip(
        axes,
        ["Linear conditional paths", "Nonlinear conditional paths", "Nonlinear + Gaussian bridge"],
        [None, curved_base, curved_base],
        [0.0, 0.0, 0.5],
        strict=True,
    ):
        interpolant = StochasticInterpolant(amplitude=0.5, base_fn=base)
        paths = []
        for time in times:
            batch_time = time.expand(len(source))
            deterministic = (
                (1 - time) * source + time * target
                if base is None
                else base(source, target, batch_time)
            )
            gamma = interpolant.coefficients(batch_time)[2][:, None] * (amplitude / 0.5)
            paths.append(deterministic + gamma * noise)
        paths = torch.stack(paths).numpy()
        axis.plot(paths[:, :, 0], paths[:, :, 1], color="#4475b7", alpha=0.3, linewidth=0.8)
        axis.scatter(*source.T.numpy(), color="#9554a4", s=12, zorder=3)
        axis.scatter(*target.T.numpy(), color="#e34a44", s=12, zorder=3)
        axis.set_title(title, fontsize=11)
        axis.set_aspect("equal")
        axis.set_xlabel("x₁")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("x₂")
    figure.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "nonlinear_paths.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
