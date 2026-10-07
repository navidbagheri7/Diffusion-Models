"""Conditional diffusion/flow lessons on nonlinear 2D datasets."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .guidance import drop_labels
from .models import TimeMLP
from .paths import DiscreteVP
from .sampling import sample
from .toy_data import ToyDistribution, toy_metrics
from .train import perturb
from .utils import EMA, atomic_checkpoint, save_json, seed_everything
from .visualize import particle_animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(ToyDistribution.COUNTS), default="olympic_rings")
    parser.add_argument("--algorithm", choices=["diffusion", "flow"], default="diffusion")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--classifier-steps", type=int, default=5000)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/toy_conditional"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = ToyDistribution(args.dataset, device=device)
    process = (
        SimpleNamespace(
            kind="flow", prediction="velocity", coupling="independent", flow_path="linear"
        )
        if args.algorithm == "flow"
        else SimpleNamespace(
            kind="ddpm", prediction="x0", steps=1000, path=DiscreteVP(device=device)
        )
    )
    sampler = (
        SimpleNamespace(method="ode", steps=100, ode_solver="heun")
        if args.algorithm == "flow"
        else SimpleNamespace(method="ddim", steps=100, eta=0.0, clip_x0=False)
    )
    model = TimeMLP(args.width, 4, classes=data.components).to(device)
    ema = EMA(model, 0.995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    history = []
    for step in range(args.steps):
        clean, labels = data.sample(256)
        x, t, target, weight = perturb(clean, labels, process)
        estimate = model(x, t, drop_labels(labels, data.components, 0.1))
        loss = ((estimate - target).square() * weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        ema.update(model)
        if (step + 1) % 500 == 0:
            history.append({"step": step + 1, "loss": loss.item()})
            print(history[-1], flush=True)
            atomic_checkpoint(
                args.output / "checkpoint.pt",
                {
                    "model": model.state_dict(),
                    "ema": ema.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "config": {
                        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    },
                    "history": history,
                },
            )
    classifier = None
    if args.algorithm == "diffusion" and args.classifier_steps:
        classifier = TimeMLP(args.width, 4, output_dim=data.components).to(device)
        opt = torch.optim.AdamW(classifier.parameters(), lr=0.0003)
        for _ in range(args.classifier_steps):
            clean, labels = data.sample(256)
            x, t, _, _ = perturb(clean, labels, process)
            loss = F.cross_entropy(classifier(x, t), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        classifier.eval().requires_grad_(False)
        torch.save(classifier.state_dict(), args.output / "classifier.pt")
    generator = torch.Generator(device=device).manual_seed(args.seed + 900)
    noise = torch.randn(1024, 2, device=device, generator=generator)
    metrics = []
    figure, axes = plt.subplots(
        1, 3 if classifier else 2, figsize=(14 if classifier else 10, 4), sharex=True, sharey=True
    )
    reference, ref_labels = data.sample(2048, generator)
    axes[0].scatter(
        *reference.cpu().T, c=ref_labels.cpu(), cmap="tab10", vmin=0, vmax=9, s=3, alpha=0.6
    )
    axes[0].set_title("Labeled target distribution")
    for column, guidance in enumerate(["cfg", "classifier"] if classifier else ["cfg"], 1):
        samples, all_labels = [], []
        for label in range(data.components):
            labels = torch.full((1024,), label, device=device, dtype=torch.long)
            values, trajectory = sample(
                ema.model,
                process,
                sampler,
                noise.clone(),
                process.prediction,
                labels,
                1.5 if guidance == "cfg" else 0.0,
                classifier if guidance == "classifier" else None,
                1.0,
                generator,
            )
            ref = data.sample(len(values), generator, labels)[0]
            row = {"guidance": guidance, "class": label, **toy_metrics(values, ref)}
            metrics.append(row)
            samples.append(values.cpu())
            all_labels.append(labels.cpu())
            particle_animation(
                trajectory.cpu(),
                args.output / f"{guidance}_class_{label}.gif",
                title=f"{args.dataset}: {guidance} class {label}",
                labels=labels.cpu(),
            )
        values, labels = torch.cat(samples), torch.cat(all_labels)
        axes[column].scatter(*values.T, c=labels, cmap="tab10", vmin=0, vmax=9, s=3, alpha=0.4)
        axes[column].set_title(f"Generated classes: {guidance}")
        torch.save({"samples": values, "labels": labels}, args.output / f"{guidance}_samples.pt")
    for ax in axes:
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    figure.tight_layout()
    figure.savefig(args.output / "conditional_gallery.png", dpi=180)
    plt.close(figure)
    save_json(args.output / "metrics.json", metrics)
    save_json(args.output / "history.json", history)


if __name__ == "__main__":
    main()
