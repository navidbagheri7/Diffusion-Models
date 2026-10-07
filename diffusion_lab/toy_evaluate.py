"""Conditional Olympic-ring generation with exact noisy ring classifier guidance."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .models import TimeMLP
from .paths import DiscreteVP
from .sampling import discrete_sample
from .toy_data import ToyDistribution, toy_metrics
from .utils import save_json, seed_everything
from .visualize import particle_animation


class AnalyticRingClassifier:
    """Uniform circle convolved with isotropic Gaussian: density uses Bessel I0."""

    def __init__(self, path, std=0.06):
        self.path, self.std = path, std

    def __call__(self, x, t):
        centers = x.new_tensor([[-2, 0.6], [0, 0.6], [2, 0.6], [-1, -0.6], [1, -0.6]])
        a, s = self.path.coefficients(t)
        variance = a.square() * self.std**2 + s.square()
        radius = (x[:, None] - a[:, None, None] * centers[None]).norm(dim=-1)
        z = radius * a[:, None] / variance[:, None]
        log_bessel = torch.special.i0e(z).clamp_min(1e-30).log() + z.abs()
        return (
            -(radius.square() + a[:, None].square()) / (2 * variance[:, None])
            + log_bessel
            - variance[:, None].log()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/olympic_rings_conditional/checkpoint.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/olympic_rings_analytic_guidance")
    )
    args = parser.parse_args()
    seed_everything(123)
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TimeMLP(state["config"]["width"], 4, classes=5).eval().requires_grad_(False)
    model.load_state_dict(state["ema"])
    path = DiscreteVP()
    classifier = AnalyticRingClassifier(path)
    data = ToyDistribution("olympic_rings")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    reference, labels = data.sample(4096)
    axes[0].scatter(*reference.T, c=labels, cmap="tab10", vmin=0, vmax=9, s=2, alpha=0.5)
    axes[0].set_title("Target: actual class labels")
    noise = torch.randn(1024, 2)
    records = []
    for column, guidance in enumerate(["cfg", "analytic_classifier"], 1):
        for label in range(5):
            labels = torch.full((len(noise),), label, dtype=torch.long)
            values, trajectory = discrete_sample(
                model,
                path,
                noise.clone(),
                "x0",
                "ddim",
                100,
                labels,
                1.0 if guidance == "cfg" else 0.0,
                classifier=classifier if guidance != "cfg" else None,
            )
            reference = data.sample(len(values), labels=labels)[0]
            records.append({"guidance": guidance, "class": label, **toy_metrics(values, reference)})
            axes[column].scatter(*values.T, color=plt.cm.tab10(label), s=2, alpha=0.35)
            particle_animation(
                trajectory,
                args.output / f"{guidance}_{label}.gif",
                title=f"{guidance}: ring {label}",
                labels=labels,
            )
        axes[column].set_title(guidance.replace("_", " "))
    for ax in axes:
        ax.set(aspect="equal", xlim=(-4, 4), ylim=(-3, 3))
    figure.tight_layout()
    figure.savefig(args.output / "conditional_gallery.png", dpi=180)
    plt.close(figure)
    save_json(args.output / "metrics.json", records)


if __name__ == "__main__":
    main()
