"""Compare learned fields with analytic GMM quantities and inspect flow straightness."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import GaussianMixture
from .evaluate import load_run
from .guidance import AnalyticGMMClassifier, classifier_gradient
from .metrics import gmm_metrics
from .models import TimeMLP
from .paths import expand
from .sampling import sample
from .targets import convert_prediction
from .train import prepare_process
from .utils import save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/reports/diagnostics")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seed_everything(321)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    times = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 0.98]
    figure, ax = plt.subplots(figsize=(9, 5))
    errors = {}
    root = Path("outputs")
    for run in sorted(root.iterdir()):
        checkpoint = run / "checkpoint.pt"
        if not checkpoint.exists() or run.name.startswith("smoke"):
            continue
        cfg, model, _ = load_run(checkpoint, device)
        if cfg.data.kind != "gmm" or cfg.process.kind in {"flow", "bridge"} or cfg.conditional:
            continue
        process = prepare_process(cfg.process, device)
        mixture = GaussianMixture(cfg.data.components, cfg.data.radius, cfg.data.std, device)
        values = []
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(321)
            for time in times:
                data, _ = mixture.sample(1024, generator)
                noise = torch.randn(1024, 2, device=device, generator=generator)
                t = torch.full((1024,), time, device=device)
                alpha, sigma = process.path.coefficients(t)
                x = expand(alpha, data) * data + expand(sigma, data) * noise
                learned = convert_prediction(x, model(x, t), alpha, sigma, process.prediction)[2]
                exact = mixture.score(x, alpha, sigma)
                values.append(((learned - exact) * sigma[:, None]).square().mean().sqrt().item())
        errors[run.name] = values
        ax.plot(times, values, "o-", label=run.name)
    ax.set(
        title="Learned score error against analytic noised GMM score",
        xlabel="Diffusion time",
        ylabel="RMSE of sigma(t) × (learned score − true score)",
    )
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "score_errors.png", dpi=160)
    plt.close(figure)
    save_json(
        directory / "score_errors.json",
        {
            "times": times,
            "errors": errors,
            "note": "Unconditional fields; sigma weighting makes scales interpretable; this is not the noisy-target validation loss",
        },
    )

    cfg, model, classes = load_run(root / "gmm_guidance/checkpoint.pt", device)
    process = prepare_process(cfg.process, device)
    mixture = GaussianMixture(cfg.data.components, cfg.data.radius, cfg.data.std, device)
    classifier = TimeMLP(cfg.model.width, cfg.model.depth, output_dim=classes).to(device).eval()
    classifier.load_state_dict(
        torch.load(root / "gmm_guidance/classifier.pt", map_location=device, weights_only=True)
    )
    analytic = AnalyticGMMClassifier(mixture, process.path)
    classifier_rows = []
    for time in times:
        data, labels = mixture.sample(2048)
        t = torch.full((2048,), time, device=device)
        a, s = process.path.coefficients(t)
        x = a[:, None] * data + s[:, None] * torch.randn_like(data)
        with torch.no_grad():
            accuracy = (classifier(x, t).argmax(1) == labels).float().mean().item()
            oracle_accuracy = (analytic(x, t).argmax(1) == labels).float().mean().item()
        gradient = classifier_gradient(classifier, x, t, labels)
        oracle_gradient = classifier_gradient(analytic, x, t, labels)
        error = ((gradient - oracle_gradient) * s[:, None]).square().mean().sqrt().item()
        classifier_rows.append(
            {
                "time": time,
                "accuracy": accuracy,
                "bayes_accuracy": oracle_accuracy,
                "sigma_weighted_gradient_rmse": error,
            }
        )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(times, [r["accuracy"] for r in classifier_rows], "o-", label="Learned classifier")
    axes[0].plot(
        times,
        [r["bayes_accuracy"] for r in classifier_rows],
        "--",
        label="Analytic Bayes classifier",
    )
    axes[0].axhline(1 / classes, color="gray", linestyle=":", label="Chance")
    axes[0].set(
        title="Class accuracy is not gradient accuracy", xlabel="Diffusion time", ylabel="Accuracy"
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(times, [r["sigma_weighted_gradient_rmse"] for r in classifier_rows], "o-")
    axes[1].set(
        title="Guidance gradient error",
        xlabel="Diffusion time",
        ylabel="Sigma-weighted gradient RMSE",
    )
    figure.tight_layout()
    figure.savefig(directory / "classifier_diagnostics.png", dpi=160)
    plt.close(figure)
    save_json(directory / "classifier_diagnostics.json", classifier_rows)

    rows = {}
    for name in ["gmm_flow", "gmm_flow_ot", "gmm_reflow"]:
        cfg, model, _ = load_run(root / name / "checkpoint.pt", device)
        process = prepare_process(cfg.process, device)
        generator = torch.Generator(device=device).manual_seed(123)
        initial = torch.randn(256, 2, device=device, generator=generator)
        _, history = sample(model, process, cfg.sampler, initial, "velocity")
        increments = history[1:] - history[:-1]
        endpoint = history[-1] - history[0]
        length = increments.norm(dim=-1).sum(0)
        straight = endpoint.norm(dim=-1)
        rows[name] = {
            "mean_excess_length": (length - straight).mean().item(),
            "mean_length_ratio": (length / straight.clamp_min(1e-5)).mean().item(),
            "normalized_velocity_variation": (
                (increments * len(increments) - endpoint[None]).square().sum(-1).mean()
                / endpoint.square().sum(-1).mean()
            ).item(),
        }
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(rows), [row["mean_excess_length"] for row in rows.values()])
    ax.set(title="Measured excess trajectory length", ylabel="Path length − endpoint distance")
    figure.tight_layout()
    figure.savefig(directory / "reflow_straightness.png", dpi=160)
    plt.close(figure)
    save_json(directory / "reflow_straightness.json", rows)
    mixture = GaussianMixture()
    save_json(
        directory / "real_vs_real_metrics.json",
        gmm_metrics(mixture.sample(1024)[0], mixture.sample(1024)[0], mixture),
    )


if __name__ == "__main__":
    main()
