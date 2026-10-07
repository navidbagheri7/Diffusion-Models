"""Visualize trained learned variance, EDM coefficients and measured solver costs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import ImageDataset, balanced_image_indices
from .image_train import load_image_run
from .improved_ddpm import learned_log_variance, posterior
from .paths import DiscreteVP, expand
from .utils import save_json


@torch.no_grad()
def variance_lesson(checkpoint, output, device):
    cfg, model, classes, training_step = load_image_run(checkpoint, device)
    if cfg.algorithm != "improved_ddpm" or not cfg.learned_variance:
        raise ValueError("This lesson requires trained learned-range DDPM weights")
    dataset = ImageDataset(cfg.data, "validation")
    selected = balanced_image_indices(dataset, 64)
    clean = torch.stack([dataset[i][0] for i in selected]).to(device)
    labels = torch.tensor([dataset[i][1] for i in selected], device=device) if classes else None
    path = DiscreteVP(cfg.diffusion_steps, cfg.schedule, device)
    generator = torch.Generator(device=device).manual_seed(123)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    rows = []
    for index in torch.linspace(1, path.steps - 1, 32).round().long().unique():
        indices = torch.full((len(clean),), index.item(), dtype=torch.long, device=device)
        time = (indices.float() + 1) / path.steps
        a, s = path.coefficients(time)
        noisy = expand(a, clean) * clean + expand(s, clean) * noise
        variance = model(noisy, time, labels).float().chunk(2, dim=1)[1]
        _, low = posterior(path, clean, noisy, indices)
        high = path.betas[indices].log()
        predicted = learned_log_variance(variance, low, high).flatten()
        quantiles = predicted.quantile(torch.tensor([0.1, 0.5, 0.9], device=device)).exp().cpu()
        rows.append(
            {
                "t": time[0].item(),
                "posterior_variance": low[0].exp().item(),
                "beta": high[0].exp().item(),
                "predicted_variance_p10": quantiles[0].item(),
                "predicted_variance_median": quantiles[1].item(),
                "predicted_variance_p90": quantiles[2].item(),
            }
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    t = [r["t"] for r in rows]
    ax.semilogy(t, [r["posterior_variance"] for r in rows], label=r"Posterior $\tilde\beta_t$")
    ax.semilogy(t, [r["beta"] for r in rows], "--", label=r"Forward $\beta_t$")
    ax.semilogy(t, [r["predicted_variance_median"] for r in rows], label="Learned median")
    ax.fill_between(
        t,
        [r["predicted_variance_p10"] for r in rows],
        [r["predicted_variance_p90"] for r in rows],
        alpha=0.18,
        label="Learned pixel 10–90% range",
    )
    ax.set(
        xlabel="Forward diffusion time",
        ylabel="Reverse variance",
        title="Learned diagonal reverse variance",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learned_variance.png", dpi=180)
    plt.close(fig)
    save_json(
        output / "learned_variance.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": model.checkpoint_sha256,
            "training_steps": training_step,
            "images": len(clean),
            "seed": 123,
            "rows": rows,
            "interpretation": "Learned-range outputs are not clamped; individual variances may lie outside interpolation bounds. Quantiles do not establish calibration.",
        },
    )


def edm_lesson(output):
    sigma = torch.logspace(-3, 2, 200)
    data = 0.5
    denominator = sigma.square() + data**2
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for values, label in [
        (data**2 / denominator, r"$c_{skip}$"),
        (sigma * data / denominator.sqrt(), r"$c_{out}$"),
        (denominator.rsqrt(), r"$c_{in}$"),
    ]:
        axes[0].semilogx(sigma, values, label=label)
    axes[0].set(
        xlabel=r"Noise standard deviation $\sigma$", title="EDM preconditioning, data std = 0.5"
    )
    axes[0].legend()
    axes[1].loglog(sigma, denominator / (sigma * data).square())
    axes[1].set(xlabel=r"$\sigma$", ylabel=r"$\lambda(\sigma)$", title="Denoiser loss weight")
    fig.tight_layout()
    fig.savefig(output / "edm_preconditioning.png", dpi=180)
    plt.close(fig)


def solver_lesson(root, output):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    families = {
        "Learned-variance DDPM": [
            "mnist_ddpm1000",
            "mnist_ddim20",
            "mnist_ddim50",
            "mnist_dpm20",
            "mnist_dpmpp20",
        ],
        "EDM": ["mnist_edm18", "mnist_edm_euler18"],
        "Consistency distillation": ["mnist_cd1", "mnist_cd2", "mnist_cd4"],
        "Standalone consistency": ["mnist_ct1", "mnist_ct4"],
    }
    records = []
    for family, names in families.items():
        points = []
        for name in names:
            path = root / "outputs" / name / "metrics.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            points.append((record["nfe_per_image"], record["frechet_digit_features_128"]))
            records.append({"experiment": name, **record})
            if family == "Learned-variance DDPM":
                offset = {
                    "mnist_dpm20": (5, 12),
                    "mnist_dpmpp20": (5, -15),
                    "mnist_ddpm1000": (-60, 12),
                }.get(name, (4, 5))
                ax.annotate(
                    name.removeprefix("mnist_"),
                    points[-1],
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                )
        if points:
            ax.scatter(*zip(*points, strict=True), label=family, s=45)
    ax.set(
        xscale="log",
        xlabel="Measured denoiser calls per image",
        ylabel="Digit feature Fréchet distance (lower is better)",
        title="MNIST: sampling cost and measured quality",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_nfe.png", dpi=180)
    plt.close(fig)
    save_json(output / "quality_vs_nfe.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/mnist_improved_ddpm/checkpoint.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("assets/teaching/solvers"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variance_lesson(args.checkpoint, args.output, torch.device(args.device))
    edm_lesson(args.output)
    solver_lesson(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
