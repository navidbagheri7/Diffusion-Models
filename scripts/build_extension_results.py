"""Build the extension evidence page from measured reports and saved previews.

Pending training never becomes a final benchmark row. Run --follow alongside
long jobs to refresh progress; final evaluators call this after actual metrics.
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

if __package__:
    from .asset_layout import experiment_group
else:
    from asset_layout import experiment_group


def atomic_copy(source, destination):
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def build(root):
    outputs, assets = root / "outputs", root / "assets"
    lines = [
        "# Extension results: likelihood, interpolants, fast generation and latent DiT",
        "",
        "This page is generated from actual reports. Configured budgets are separate",
        "from completed training and measured final quality. Interim previews are",
        "labeled as such; no paper benchmark score is claimed.",
        "",
        "## Likelihood diagnostics",
        "",
        "All image values below are conditional on the held-out class label, without CFG.",
        "Small batches are demonstrations, not full test-set benchmark estimates.",
        "",
        "| Experiment | Examples | Quantity | Measured value |",
        "|---|---:|---|---:|",
    ]
    for name in [
        "likelihood_mnist_full_elbo",
        "likelihood_mnist_sampled_elbo",
        "likelihood_mnist_full_elbo_unclipped",
        "likelihood_mnist_edm",
        "likelihood_cifar10_edm",
        "likelihood_cifar10_edm_200steps",
        "likelihood_cifar10_flow",
        "likelihood_gmm_flow",
        "likelihood_gmm_ddpm_epsilon",
    ]:
        row = read(outputs / name / "metrics.json")
        if not row:
            continue
        key = (
            "negative_elbo_bpd_mean"
            if "negative_elbo_bpd_mean" in row
            else "dequantized_density_bpd_estimate"
            if "dequantized_density_bpd_estimate" in row
            else "mean_log_density_nats"
        )
        quantity = (
            "negative ELBO, bpd"
            if key.startswith("negative")
            else "ODE bpd estimate"
            if key.startswith("dequant")
            else "ODE log density, nats"
        )
        lines.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['samples']} | {quantity} | {row[key]:.4f} |"
        )
    analytic = read(outputs / "likelihood_analytic_gmm/metrics.json")
    if analytic:
        exact, hutch = analytic["results"]
        lines += [
            "",
            f"Analytic GMM: exact-trace log-density RMSE **{exact['rmse_log_density']:.3g}**;",
            f"8-probe Hutchinson RMSE **{hutch['rmse_log_density']:.3g}** on {analytic['samples']} samples.",
            "The same exact finite-time GMM terminal density isolates integration/trace error.",
        ]
    lines += [
        "",
        "## Learned stochastic interpolants",
        "",
        "All three datasets completed 15,000 optimizer updates. Each row uses 2,048",
        "generated/reference points, conditional labels, and 400 integration steps.",
        "",
        "| Dataset | Diffusion strength | Sliced W₂ ↓ | Unbiased MMD² |",
        "|---|---:|---:|---:|",
    ]
    for dataset in ["gmm", "wavy_rings", "two_spirals"]:
        name = dataset + "_stochastic_interpolant"
        rows = read(outputs / name / "metrics.json", [])
        for row in rows:
            lines.append(
                f"| [{dataset}](../assets/experiments/{experiment_group(name)}/{name}/strength_{row['strength']}/metrics.json) | {row['strength']} | {row['sliced_w2']:.4f} | {row['mmd_squared_unbiased']:.5f} |"
            )
    lines += [
        "",
        "An unbiased finite-sample MMD estimate can be negative. ODE and SDE agreement",
        "is exact only with exact fields and integration; learned fits retain error.",
        "",
        "![Conditional wavy-ring probability transport](../assets/experiments/toy/wavy_rings_stochastic_interpolant/strength_0.0/generation.gif)",
        "",
        "## Distribution quality beyond FID",
        "",
        "| RGB experiment | Precision | Recall | Density | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in read(assets / "reports/distribution_quality/rgb/comparison.json", []):
        name = row["experiment"]
        lines.append(
            f"| [{name}](../assets/reports/distribution_quality/rgb/{name}/metrics.json) | {row['precision']:.3f} | {row['recall']:.3f} | {row['density']:.3f} | {row['coverage']:.3f} |"
        )
    lines += [
        "",
        "CIFAR uses 2,048/2,048 Inception features; the face example has 2,000 references.",
        "Compare only matching data, representation, counts and k=3. Per-class reports",
        "and equal-size training/held-out nearest-neighbor galleries are retained.",
        "",
    ]
    for group in ["interim_digits", "interim_rgb", "extension_digits", "extension_rgb"]:
        rows = read(assets / "reports/distribution_quality" / group / "comparison.json", [])
        if rows:
            lines += [
                f"### Additional coverage diagnostics: {group}",
                "",
                "Compare within matching data, features, counts and k; digit and RGB spaces differ.",
                "",
                "| Experiment | Precision | Recall | Density | Coverage |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in rows:
                name = row["experiment"]
                lines.append(
                    f"| [{name}](../assets/reports/distribution_quality/{group}/{name}/metrics.json) | {row['precision']:.3f} | {row['recall']:.3f} | {row['density']:.3f} | {row['coverage']:.3f} |"
                )
            lines.append("")
    repeats = read(assets / "reports/distribution_quality/seed_intervals.json")
    if repeats:
        fid = repeats["metrics"]["fid_2048"]
        interval = fid["confidence_interval"]
        lines += [
            "Three independent CIFAR generation seeds at 2,048/2,048 samples: mean FID",
            f"**{fid['mean']:.2f}**, Student-t 95% interval **[{interval[0]:.2f}, {interval[1]:.2f}]**.",
            "This smaller-reference protocol is not comparable to the 10k/10k final FID.",
            "The interval excludes training and reference-sampling uncertainty.",
            "[Complete repeated-seed report](../assets/reports/distribution_quality/seed_intervals.json).",
            "",
        ]
    lines += [
        "## Stronger few-step experiments",
        "",
        "The following rows share the independent clean digit assessor and 2,048",
        "held-out/generated images. Its feature distance is not Inception FID.",
        "",
        "| MNIST method | Network calls | Digit feature distance ↓ | Requested-class accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name in [
        "mnist_progressive_teacher32",
        *[f"mnist_progressive_{n}_eval" for n in [16, 8, 4, 2, 1]],
        "mnist_ict1",
        "mnist_ict2",
        "mnist_ict4",
        "mnist_ict_smooth1",
        "mnist_ict_smooth2",
        "mnist_ict_smooth4",
        "mnist_ict_corrected1",
        "mnist_ict_corrected2",
        "mnist_ict_corrected4",
        "mnist_ict_corrected_5k1",
        "mnist_ict_corrected_5k4",
    ]:
        row = read(outputs / name / "metrics.json")
        if row:
            lines.append(
                f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['nfe_per_image']} | {row['frechet_digit_features_128']:.2f} | {row['requested_class_accuracy']:.3f} |"
            )
    lines += [
        "",
        "The first two iCT runs were invalidated by an autocast weight-cache bug:",
        "the detached online-target forward prevented convolution/linear gradients.",
        "They are retained as failed implementation diagnostics, not method benchmarks.",
        "The corrected run keeps the smooth recipe and evaluates the student first.",
        "Rows containing `5k` are immutable 5,000-update interim checkpoints;",
        "they do not establish completion of the 40,000-update corrected budget.",
        "",
        "## DiT, latent diffusion and CIFAR distillation",
        "",
        "The latent cache contains 28,000 training and 2,000 held-out 128px faces.",
        "The external frozen VAE alone is pretrained. Posterior-mean reconstruction",
        "PSNR is 32.11 dB on 128 held-out images; denoisers start from scratch.",
        "",
        "| Training run | Latest saved step / budget | Status |",
        "|---|---:|---|",
    ]
    status = []
    for name in [
        "cifar10_dit_quality",
        "celeba128_latent_unet_quality",
        "celeba128_latent_dit_quality",
        "cifar10_consistency_distilled",
        "mnist_ict_quality",
        "mnist_ict_smooth_quality",
        "mnist_ict_corrected_quality",
    ]:
        directory = outputs / name
        history = read(directory / "history.json", [])
        completed = read(directory / "training_complete.json")
        latest = max((row["step"] for row in history), default=0)
        budget = 20000 if name.startswith("mnist") else 30000 if "consistency" in name else 100000
        if name in {"mnist_ict_smooth_quality", "mnist_ict_corrected_quality"}:
            budget = 40000
        stage = (
            "Training completed; see measured evaluation rows below"
            if completed
            else "Training in progress"
        )
        lines.append(f"| {name} | {latest:,} / {budget:,} | {stage} |")
        preview = read(directory / "preview.json")
        if preview:
            target = assets / "archive/training_previews" / name
            target.mkdir(parents=True, exist_ok=True)
            snapshot = directory / f"preview_{preview['step']:07d}.png"
            if snapshot.exists():
                atomic_copy(snapshot, target / "samples.png")
                for filename in ["config.yaml", "environment.json"]:
                    if (directory / filename).exists():
                        shutil.copy2(directory / filename, target / filename)
                (target / "preview.json").write_text(
                    json.dumps(
                        {
                            **preview,
                            "training_budget": budget,
                            "training_complete": bool(completed),
                            "interim": not bool(completed),
                        },
                        indent=2,
                    )
                    + "\n"
                )
        status.append(
            {
                "run": name,
                "latest_saved_step": latest,
                "budget": budget,
                "training_complete": bool(completed),
            }
        )
    lines += [
        "",
        "Previews below are fixed-seed progress snapshots. Their metadata records the",
        "actual step. They are not substitutes for a completed final metric evaluation.",
        "",
        "![128px latent DiT training preview](../assets/archive/training_previews/celeba128_latent_dit_quality/samples.png)",
        "",
        "| Completed RGB evaluation | Generated / reference | FID ↓ | IS ↑ | Calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in [
        "cifar10_dit_final",
        "cifar10_dit_cfg_final",
        "celeba128_latent_unet_final",
        "celeba128_latent_dit_final",
        "cifar10_cd1",
        "cifar10_cd2",
        "cifar10_cd4",
    ]:
        row = read(outputs / name / "metrics.json")
        if row:
            lines.append(
                f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['generated_count']} / {row['real_count']} | {row['fid_2048']:.2f} | {row['inception_score_mean']:.2f} | {row['nfe_per_image']:g} |"
            )
    lines += [
        "",
        "### Immutable interim evaluations",
        "",
        "These use 2,048 generated images and 2,048 CIFAR / 2,000 face references.",
        "They are progress diagnostics, not completed final benchmark results.",
        "",
        "| Interim model | Training updates | FID ↓ | IS ↑ |",
        "|---|---:|---:|---:|",
    ]
    for name in ["cifar10_dit_30k", "celeba128_latent_dit_30k", "celeba128_latent_unet_20k"]:
        row = read(outputs / name / "metrics.json")
        if row:
            lines.append(
                f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['training_steps']} | {row['fid_2048']:.2f} | {row['inception_score_mean']:.2f} |"
            )
    lines += [
        "",
        "Final evaluations wait for training_complete.json and verify the loaded weight",
        "hash. Checkpoints, optimizer/RNG state and immutable milestones stay in outputs/.",
        "The public assets contain figures, configurations and measured reports.",
        "",
    ]
    (root / "docs/21_extension_results.md").write_text("\n".join(lines))
    (assets / "metadata/expansion_status.json").write_text(json.dumps(status, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("Interval must be positive")
    root = Path(__file__).resolve().parents[1]
    while True:
        build(root)
        if not args.follow or (root / "outputs/expansion_complete.json").exists():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
