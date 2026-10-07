"""Learn stochastic interpolants on labeled 2D data and visualize ODE/SDE transport."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import GaussianMixture
from .guidance import drop_labels
from .interpolants import GaussianInterpolantOracle, StochasticInterpolant
from .metrics import gmm_metrics
from .models import TimeMLP
from .solvers import integrate
from .toy_data import ToyDistribution, toy_metrics
from .utils import EMA, atomic_checkpoint, save_json, seed_everything
from .visualize import particle_animation


def plot_comparison(reference, samples, labels, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, values, name in zip(axes, [reference, samples], ["Reference", title], strict=True):
        values = values.detach().cpu()
        ax.scatter(
            values[:, 0],
            values[:, 1],
            c=labels.cpu() if labels is not None else None,
            cmap="tab10" if labels is not None else None,
            vmin=0 if labels is not None else None,
            vmax=9 if labels is not None else None,
            s=3,
            alpha=0.5,
        )
        ax.set(
            title=name, xlabel="$x_1$", ylabel="$x_2$", aspect="equal", xlim=(-5, 5), ylim=(-5, 5)
        )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_oracle_fields(oracle, directory, device):
    """Analytic marginal densities and coefficient curves, without a learned fit."""
    grid = torch.linspace(-5, 5, 140, device=device)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    points = torch.stack([xx.flatten(), yy.flatten()], 1)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharex=True, sharey=True)
    for ax, time in zip(axes, [0, 0.25, 0.5, 0.75, 1], strict=True):
        density = oracle.log_density(points, torch.full((len(points),), time, device=device)).exp()
        ax.imshow(
            density.reshape(140, 140).cpu(),
            origin="lower",
            extent=(-5, 5, -5, 5),
            cmap="magma",
            vmin=0,
            vmax=0.22,
        )
        ax.set(title=f"Exact marginal, t={time}", xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    fig.tight_layout()
    fig.savefig(directory / "density_evolution.png", dpi=160)
    plt.close(fig)
    time = torch.linspace(0, 1, 201, device=device)
    a, b, gamma, *_ = oracle.interpolant.coefficients(time)
    fig, ax = plt.subplots(figsize=(7, 4))
    for values, label in [
        (a, "Source coefficient a"),
        (b, "Target coefficient b"),
        (gamma, "Bridge noise gamma"),
    ]:
        ax.plot(time.cpu(), values.cpu(), label=label)
    ax.set(
        xlabel="Source-to-target time", ylabel="Coefficient", title="Cubic stochastic interpolant"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "coefficients.png", dpi=160)
    plt.close(fig)


def oracle_lesson(directory, device):
    source, target = GaussianMixture(1, 0, 1, device), GaussianMixture(8, 3, 0.25, device)
    interpolant = StochasticInterpolant("cubic", 0.5)
    oracle = GaussianInterpolantOracle(source, target, interpolant)
    plot_oracle_fields(oracle, directory, device)
    generator = torch.Generator(device=device).manual_seed(123)
    initial, _ = source.sample(2048, generator)
    reference, _ = target.sample(2048, generator)
    rows = []
    for strength in [0.0, 0.5, 2.0]:
        folder = directory / f"oracle_strength_{strength}"
        folder.mkdir(parents=True, exist_ok=True)
        if strength == 0:

            def field(x, t):
                return oracle.fields(x, t)[0]

            diffusion = None
            solver = "heun"
        else:

            def field(x, t, strength=strength):
                return oracle.drift(x, t, strength)

            def diffusion(t, strength=strength):
                return oracle.diffusion(t, strength)

            solver = "euler_maruyama"
        result, trajectory = integrate(
            field,
            initial.clone(),
            torch.linspace(0, 1, 401, device=device),
            solver,
            diffusion,
            generator,
        )
        metric = gmm_metrics(result, reference, target)
        metric.update(
            {
                "diffusion_strength": strength,
                "field": "Exact Gaussian-mixture stochastic-interpolant marginal fields",
                "integrator": solver,
                "steps": 400,
            }
        )
        save_json(folder / "metrics.json", metric)
        plot_comparison(
            reference,
            result,
            None,
            folder / "samples.png",
            f"Oracle transport, strength {strength}",
        )
        particle_animation(
            trajectory, folder / "generation.gif", f"Stochastic interpolant, strength {strength}"
        )
        atomic_checkpoint(
            folder / "samples.pt",
            {"samples": result.cpu(), "reference": reference.cpu(), "trajectory": trajectory},
        )
        rows.append(metric)
    save_json(directory / "oracle_marginals.json", rows)
    save_json(
        directory / "metrics.json",
        {"dataset": "Exact GMM interpolant", "seed": 123, "generated_count": 2048, "results": rows},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gmm", *ToyDistribution.COUNTS], default="gmm")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--path", choices=["linear", "trigonometric", "cubic"], default="linear")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--conditional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/gmm_stochastic_interpolant"))
    args = parser.parse_args()
    seed_everything(args.seed, 4)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.oracle_only:
        return oracle_lesson(args.output, device)
    if min(args.steps, args.batch_size, args.samples) < 1:
        raise ValueError("Training/evaluation counts must be positive")
    if (args.output / "checkpoint.pt").exists() and not args.resume:
        raise ValueError("Choose a new run directory or resume its checkpoint")
    data = (
        GaussianMixture(device=device)
        if args.dataset == "gmm"
        else ToyDistribution(args.dataset, device=device)
    )
    classes = data.components if args.conditional else 0
    process = StochasticInterpolant(args.path, args.amplitude)
    model = TimeMLP(args.width, 4, classes, output_dim=4).to(device)
    ema = EMA(model, 0.999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    first_step, history = 0, []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ["dataset", "width", "conditional", "amplitude", "path", "batch_size", "seed"]:
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume changes interpolant {key}")
        model.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        first_step, history = state["step"], state["history"]
        torch.set_rng_state(state["rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    OmegaConf.save(OmegaConf.create(config), args.output / "config.yaml")
    for step in range(first_step + 1, args.steps + 1):
        target, labels = data.sample(args.batch_size)
        source = torch.randn_like(target)
        t = torch.rand(len(target), device=device) * 0.998 + 0.001
        x, velocity, noise = process.targets(source, target, t)
        condition = drop_labels(labels, classes, 0.1) if classes else None
        predicted_velocity, predicted_noise = model(x, t, condition).chunk(2, 1)
        velocity_loss = (predicted_velocity - velocity).square().mean()
        noise_loss = (predicted_noise - noise).square().mean()
        loss = velocity_loss + noise_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Interpolant objective is nonfinite")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        ema.update(model)
        if step % 1000 == 0 or step == args.steps:
            row = {
                "step": step,
                "velocity_loss": float(velocity_loss.detach()),
                "noise_regression_loss": float(noise_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            atomic_checkpoint(
                args.output / "checkpoint.pt",
                {
                    "format": "interpolant_v1",
                    "config": config,
                    "model": model.state_dict(),
                    "ema": ema.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "classes": classes,
                    "history": history,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
            )
    generator = torch.Generator(device=device).manual_seed(args.seed + 10000)
    labels = torch.arange(args.samples, device=device) % classes if classes else None
    reference, _reference_labels = data.sample(args.samples, generator, labels)
    initial = torch.randn(reference.shape, device=device, generator=generator)
    results = []
    checkpoint_hash = hashlib.sha256((args.output / "checkpoint.pt").read_bytes()).hexdigest()
    for strength in [0.0, 0.5, 2.0]:
        velocity, drift, diffusion = process.learned_fields(ema.model, labels, args.scale, strength)
        field = velocity if strength == 0 else drift
        solver = "heun" if strength == 0 else "euler_maruyama"
        samples, trajectory = integrate(
            field,
            initial.clone(),
            torch.linspace(0, 1, 401, device=device),
            solver,
            None if strength == 0 else diffusion,
            generator,
        )
        folder = args.output / f"strength_{strength}"
        folder.mkdir(exist_ok=True)
        metric = (
            gmm_metrics(samples, reference, data)
            if args.dataset == "gmm"
            else toy_metrics(samples, reference)
        )
        metric.update(
            {
                "training_steps": args.steps,
                "checkpoint": str((args.output / "checkpoint.pt").resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "generated_count": len(samples),
                "reference_count": len(reference),
                "nfe_per_image": (800 if strength == 0 else 400)
                * (2 if classes and args.scale not in {0, 1} else 1),
                "dataset": args.dataset,
                "strength": strength,
                "conditional": bool(classes),
                "cfg_scale": args.scale,
                "integrator": solver,
                "steps": 400,
                "seed": args.seed,
                "noise_amplitude": args.amplitude,
                "path": args.path,
                "conditional_noise_pairs": "Independent Gaussian source/noise, labels attached to target",
            }
        )
        save_json(folder / "metrics.json", metric)
        plot_comparison(
            reference,
            samples,
            labels,
            folder / "samples.png",
            f"Learned interpolant, strength {strength}",
        )
        particle_animation(
            trajectory, folder / "generation.gif", f"Interpolant strength {strength}", labels=labels
        )
        atomic_checkpoint(
            folder / "samples.pt",
            {
                "samples": samples.cpu(),
                "reference": reference.cpu(),
                "labels": labels.cpu() if labels is not None else None,
                "trajectory": trajectory,
            },
        )
        results.append(metric)
    save_json(args.output / "metrics.json", results)
    save_json(args.output / "history.json", history)
    save_json(
        args.output / "training_complete.json",
        {
            "step": args.steps,
            "status": "Configured interpolation training and ODE/SDE comparisons completed",
        },
    )


if __name__ == "__main__":
    main()
