"""Reproducible consistency, likelihood and independent-training-seed studies.

Subprocesses retain exact CLI commands, logs, checkpoints and raw reports. No
failed or unfinished experiment is included in a completed summary.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from diffusion_lab.data import ImageDataset, balanced_image_indices
from diffusion_lab.distribution_metrics import mean_interval
from diffusion_lab.image_train import load_image_run
from diffusion_lab.paths import expand
from diffusion_lab.utils import save_json, seed_everything

ROOT = Path(__file__).resolve().parents[1]


def run(command, output):
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "command.json", command)
    with (output / "execution.log").open("a") as log:
        subprocess.run(
            [sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
        )


def consistency_study(args):
    checkpoint = ROOT / "outputs/cifar10_consistency_distilled/checkpoint.pt"
    cfg, model, _classes, _step = load_image_run(checkpoint, args.device)
    teacher = load_image_run(ROOT / cfg.teacher, args.device)[1]
    data = ImageDataset(cfg.data, "validation")
    selected = balanced_image_indices(data, 256)
    clean = torch.stack([data[i][0] for i in selected]).to(args.device)
    labels = torch.tensor([data[i][1] for i in selected], device=args.device)
    generator = torch.Generator(device=args.device).manual_seed(123)
    noise = torch.randn(clean.shape, device=args.device, generator=generator)
    rows = []
    with torch.no_grad():
        for sigma in [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 80]:
            t = torch.full((len(clean),), sigma, device=args.device)
            x = clean + expand(t, clean) * noise
            student = model(x, t, labels)
            baseline = teacher(x, t, labels)
            rows.append(
                {
                    "sigma": sigma,
                    "student_mse": (student - clean).square().mean().item(),
                    "teacher_mse": (baseline - clean).square().mean().item(),
                    "student_out_of_range_fraction": (student.abs() > 1).float().mean().item(),
                    "student_absolute_max": student.abs().max().item(),
                }
            )
    save_json(
        args.output / "denoising_profile.json",
        {"checkpoint_sha256": model.checkpoint_sha256, "samples": len(clean), "results": rows},
    )
    del teacher, model, clean, noise
    torch.cuda.empty_cache()
    levels = [[80], *[[80, x] for x in [0.1, 0.3, 1, 3, 10, 30]], [80, 10, 3, 1], [80, 3, 1, 0.3]]
    reports = []
    for index, sigmas in enumerate(levels):
        folder = args.output / f"selection_{index:02d}"
        command = [
            "-m",
            "diffusion_lab.image_evaluate",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(folder),
            "--device",
            args.device,
            "--samples",
            "1000",
            "--batch-size",
            "64",
            "--scale",
            "1",
            "--seed",
            "123",
            "--consistency-sigmas",
            *map(str, sigmas),
        ]
        if not (folder / "metrics.json").exists():
            run(command, folder)
        report = json.loads((folder / "metrics.json").read_text())
        reports.append(report)
        print(f"Consistency levels {sigmas}: FID {report['fid_2048']:.3f}", flush=True)
    winners = [
        min((r for r in reports if r["steps"] == steps), key=lambda r: r["fid_2048"])
        for steps in [1, 2, 4]
    ]
    finals = []
    for winner in winners:
        folder = args.output / f"final_{winner['steps']}"
        if not (folder / "metrics.json").exists():
            run(
                [
                    "-m",
                    "diffusion_lab.image_evaluate",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(folder),
                    "--device",
                    args.device,
                    "--samples",
                    "5000",
                    "--reference-offset",
                    "1000",
                    "--batch-size",
                    "64",
                    "--scale",
                    "1",
                    "--seed",
                    "124",
                    "--consistency-sigmas",
                    *map(str, winner["consistency_sigmas"]),
                ],
                folder,
            )
        finals.append(json.loads((folder / "metrics.json").read_text()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ["student_mse", "teacher_mse"]:
        axes[0].loglog([r["sigma"] for r in rows], [r[key] for r in rows], marker="o", label=key)
    axes[0].set(xlabel="Input noise standard deviation", ylabel="Denoising MSE")
    axes[0].legend()
    axes[1].plot([r["steps"] for r in finals], [r["fid_2048"] for r in finals], marker="o")
    axes[1].set(
        xlabel="Model calls", ylabel="FID (5k/5k)", title="Selected on separate 1k reference subset"
    )
    fig.tight_layout()
    fig.savefig(args.output / "consistency_study.png", dpi=160)
    plt.close(fig)
    save_json(
        args.output / "metrics.json",
        {
            "study": "Consistency refinement schedules",
            "selection": reports,
            "final": finals,
            "scope": "Same frozen weights; 1k selection then disjoint 5k reference subset and new generation seed. CIFAR test data were used historically during model development; this is not a pristine benchmark.",
            "causality": "Schedule sensitivity is measured; this study alone cannot prove a training-related root cause.",
        },
    )


def likelihood_study(args):
    cases = [
        ("mnist_elbo", "outputs/mnist_improved_ddpm/checkpoint_0020000.pt", 1024, 0, 1),
        ("mnist_edm", "outputs/mnist_edm_quality/checkpoint.pt", 1024, 100, 1),
        ("cifar_edm", "outputs/cifar10_edm_quality/checkpoint.pt", 512, 100, 1),
        ("cifar_edm_grid200", "outputs/cifar10_edm_quality/checkpoint.pt", 128, 200, 1),
        ("cifar_edm_probes4", "outputs/cifar10_edm_quality/checkpoint.pt", 128, 100, 4),
        ("cifar_edm_noise124", "outputs/cifar10_edm_quality/checkpoint.pt", 128, 100, 1),
    ]
    summaries = []
    for name, checkpoint, count, steps, probes in cases:
        if not (ROOT / checkpoint).exists():
            raise FileNotFoundError(checkpoint)
        values = []
        for offset in range(0, count, 32):
            folder = args.output / name / f"batch_{offset:04d}"
            if not (folder / "metrics.json").exists():
                command = [
                    "-m",
                    "diffusion_lab.likelihood_experiments",
                    "--checkpoint",
                    str(ROOT / checkpoint),
                    "--output",
                    str(folder),
                    "--device",
                    args.device,
                    "--samples",
                    str(min(32, count - offset)),
                    "--reference-offset",
                    str(offset),
                    "--steps",
                    str(steps or 100),
                    "--probes",
                    str(probes),
                    "--seed",
                    "124" if name.endswith("124") else "123",
                ]
                run(command, folder)
            per = json.loads((folder / "per_example.json").read_text())
            if steps == 0:
                values.extend(per["negative_elbo_bpd"])
            else:
                dimensions = 784 if "mnist" in name else 3072
                # Actual MNIST config is resized to 32x32; determine dimensions from saved config.
                config = OmegaConf.load(folder / "config.yaml")
                dimensions = config.data.channels * config.data.image_size**2
                values.extend(7 - x / (dimensions * math.log(2)) for x in per["log_density_nats"])
        summary = {
            "name": name,
            "samples": len(values),
            "integration_steps": steps or None,
            "probes": probes,
            "bpd": mean_interval(values),
            "is_exact_nll": False,
            "quantity": "Monte Carlo negative DDPM ELBO"
            if steps == 0
            else "Truncated dequantized numerical ODE density estimate",
        }
        summaries.append(summary)
        print(summary, flush=True)
        save_json(args.output / name / "summary.json", summary)
    save_json(
        args.output / "metrics.json",
        {
            "study": "Expanded likelihood and numerical sensitivity",
            "results": summaries,
            "scope": "Intervals describe variation across held-out examples, not numerical bias. Same checkpoint; known class labels; no CFG. Paired numerical sensitivity uses the first 128 examples.",
        },
    )


def seed_study(args):
    reports = []
    for family in ["ddpm", "flow"]:
        for seed in [42, 43, 44]:
            folder = args.output / f"gmm_{family}_seed{seed}"
            if not (folder / "metrics.json").exists():
                run(
                    [
                        "-m",
                        "diffusion_lab.train",
                        f"process={family}",
                        f"sampler={'ddim' if family == 'ddpm' else 'ode'}",
                        f"seed={seed}",
                        f"device={args.device}",
                        f"run_dir={folder}",
                        "steps=6000",
                        "evaluation_samples=4096",
                        "validation_samples=1024",
                        "sampler.steps=100",
                    ],
                    folder,
                )
            # Matched evaluation noise and target reference seed across all trained models.
            evaluation = folder / "matched_nfe_evaluation"
            if not (evaluation / "metrics.json").exists():
                run(
                    [
                        "-m",
                        "diffusion_lab.evaluate",
                        "--checkpoint",
                        str(folder / "checkpoint.pt"),
                        "--output",
                        str(evaluation),
                        "--device",
                        args.device,
                        "--samples",
                        "4096",
                        "--seed",
                        "123",
                        "--reference-seed",
                        "999",
                        "--steps",
                        "100" if family == "ddpm" else "50",
                    ],
                    evaluation,
                )
            row = json.loads((evaluation / "metrics.json").read_text())
            row["training_seed"] = seed
            row["family"] = family
            row["checkpoint_sha256"] = hashlib.sha256(
                (folder / "checkpoint.pt").read_bytes()
            ).hexdigest()
            row["training_steps"] = 6000
            assert row["nfe_per_image"] == 100
            reports.append(row)
    keys = ["sliced_w2", "mmd_squared_unbiased", "empirical_w2_subsample", "fraction_within_3sigma"]
    summary = {
        family: {
            key: mean_interval([r[key] for r in reports if r["family"] == family]) for key in keys
        }
        for family in ["ddpm", "flow"]
    }
    save_json(
        args.output / "metrics.json",
        {
            "study": "Independent training seeds: GMM DDPM versus flow",
            "training_seeds": [42, 43, 44],
            "matched_evaluation_seed": 123,
            "matched_reference_seed": 999,
            "matched_network_calls_per_image": 100,
            "steps_per_model": 6000,
            "results": reports,
            "summary": summary,
            "scope": "Three independently initialized/trained models per family; identical architecture/data/training updates/evaluation counts and 100 measured model calls. Independent fixed reference seed 999. These intervals are not evidence of image-model training uncertainty.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=["consistency", "likelihood", "training_seeds"], required=True
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(123, 4)
    {
        "consistency": consistency_study,
        "likelihood": likelihood_study,
        "training_seeds": seed_study,
    }[args.study](args)
    save_json(args.output / "completed.json", {"study": args.study, "status": "completed"})


if __name__ == "__main__":
    main()
