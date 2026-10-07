"""Build a readable gallery from measured new experiments, without invented scores."""

import json
from pathlib import Path

if __package__:
    from .asset_layout import experiment_group
else:
    from asset_layout import experiment_group


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets/experiments"
    lines = [
        "# Advanced experiments and measured results",
        "",
        "Results below come from saved runs. Interim runs identify the actual checkpoint step;",
        "they are not completed-budget benchmark claims. Metric protocols are in [image practice](12_image_experiments.md).",
        "",
        "## MNIST quality versus network evaluations",
        "",
        "All final comparisons use 2,048 generated images, balanced references, seed 123,",
        "scale 1, and the same saved digit feature classifier. The feature distance is not Inception FID.",
        "",
        "![Quality versus denoiser calls](../assets/teaching/solvers/quality_vs_nfe.png)",
        "",
        "Within the learned-variance DDPM family, changing only the solver provides the fairest comparison.",
        "EDM and consistency have different training objectives/checkpoints, so their points compare complete pipelines.",
        "",
        "| Experiment | Training steps | Denoiser calls/image | Digit feature distance | Requested-class agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    comparisons = [
        "mnist_ddpm1000",
        "mnist_ddim20",
        "mnist_ddim50",
        "mnist_dpm20",
        "mnist_dpmpp20",
        "mnist_edm18",
        "mnist_edm_euler18",
        "mnist_cd1",
        "mnist_cd2",
        "mnist_cd4",
        "mnist_ct1",
        "mnist_ct4",
    ]
    for name in comparisons:
        path = assets / experiment_group(name) / name / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        lines.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {m['training_steps']} | {m['nfe_per_image']} | {m.get('frechet_digit_features_128', float('nan')):.3f} | {m.get('requested_class_accuracy', float('nan')):.3f} |"
        )
    for name in ["mnist_edm18", "mnist_cd1", "mnist_ct4"]:
        if (assets / experiment_group(name) / name / "classes.png").exists():
            lines.extend(
                [
                    "",
                    f"### {name}",
                    "",
                    f"![Class-conditioned digits](../assets/experiments/{experiment_group(name)}/{name}/classes.png)",
                ]
            )
    lines.extend(
        [
            "",
            "## CIFAR-10 and face images",
            "",
            "These use the reported reference count and pretrained Inception metrics.",
            "Small evaluations are noisy; comparisons across datasets/resolutions are not rankings.",
            "",
            "| Experiment | Steps | Generated / real | FID | IS | Denoiser calls/image |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    image_names = [
        "cifar10_edm_balanced_progress",
        "cifar10_edm15k_10k_eval",
        "cifar10_cfg15k_10k_eval",
        "cifar10_cfg27k_10k_eval",
        "cifar10_edm_final",
        "cifar10_edm_cfg_final",
        "cifar10_flow20_final",
        "cifar10_flow50_final",
        "celeba_flow_quality_progress",
        "celeba_flow_balanced_progress",
        "celeba_flow_final",
    ]
    for name in image_names:
        path = assets / experiment_group(name) / name / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        lines.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {m['training_steps']} | {m['generated_count']} / {m['real_count']} | {m['fid_2048']:.2f} | {m['inception_score_mean']:.2f} | {m['nfe_per_image']} |"
        )
    for name in image_names:
        if (assets / experiment_group(name) / name / "classes.png").exists():
            figure = "classes.png"
        elif (assets / experiment_group(name) / name / "samples.png").exists():
            figure = "samples.png"
        else:
            continue
        lines.extend(
            [
                "",
                f"### {name}",
                "",
                f"![{name} samples](../assets/experiments/{experiment_group(name)}/{name}/{figure})",
                "",
            ]
        )
    for name in [
        "cifar10_edm_final",
        "cifar10_cfg27k_10k_eval",
        "cifar10_edm_balanced_progress",
        "celeba_flow_final",
        "celeba_flow_quality_progress",
        "celeba_flow_balanced_progress",
        "mnist_edm_balanced_progress",
    ]:
        if (assets / experiment_group(name) / name / "interpolation.png").exists():
            lines.extend(
                [
                    f"### {name} interpolation",
                    "",
                    f"![Latent interpolation](../assets/experiments/{experiment_group(name)}/{name}/interpolation.png)",
                    "",
                ]
            )
    lines.extend(
        [
            "## Guidance",
            "",
            "CFG is the strongest current image demonstration. Noisy classifiers have measured",
            "held-out accuracy, but their gradients can still distort samples. Strong scales",
            "increase requested-class agreement at a quality cost. CLIP examples use a denoised",
            "heuristic energy and are not claimed as reliable open-vocabulary generation.",
            "",
        ]
    )
    for name in [
        "mnist_classifier_final_digit3",
        "cifar10_clip_airplane",
        "cifar10_clip_scale0.03",
        "cifar10_clip_scale0.3",
        "cifar10_clip_scale1.0",
    ]:
        if (assets / experiment_group(name) / name / "samples.png").exists():
            lines.extend(
                [
                    f"### {name}",
                    "",
                    f"![Guided samples](../assets/experiments/{experiment_group(name)}/{name}/samples.png)",
                    "",
                ]
            )
    for name in ["mnist", "cifar10"]:
        if (root / f"assets/teaching/guidance/{name}_comparison.png").exists():
            lines.extend(
                [
                    f"### {name} guidance comparison",
                    "",
                    f"![Matched-seed guidance comparison](../assets/teaching/guidance/{name}_comparison.png)",
                    "",
                    f"[Recorded protocol and assessment classifier](../assets/teaching/guidance/{name}_comparison.json).",
                    "The unconditional row is measured against the requested class as a baseline,",
                    "not as its unconditional distribution-quality score. CG also incurs the",
                    "separately recorded classifier calls. Guidance can exploit classifier",
                    "gradients; prefer an independent assessment classifier.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Alternating DSBM",
            "",
            "Both four-cycle runs use 6,000 fitting steps per direction per cycle.",
            "",
            "| Initialization | Final forward sliced W2 | Final backward sliced W2 |",
            "|---|---:|---:|",
        ]
    )
    for name in ["dsbm_imf", "dsbm_ipf"]:
        path = assets / experiment_group(name) / name / "metrics.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        forward = [r for r in rows if r["direction"] == "forward"][-1]
        backward = [r for r in rows if r["direction"] == "backward"][-1]
        lines.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {forward['sliced_w2']:.3f} | {backward['sliced_w2']:.3f} |"
        )
    lines.extend(["", "## Rich conditional 2D data", ""])
    for name in [
        "olympic_rings_analytic_guidance",
        "wavy_rings_conditional",
        "checkerboard_conditional",
    ]:
        if (assets / experiment_group(name) / name / "conditional_gallery.png").exists():
            lines.extend(
                [
                    f"![{name}](../assets/experiments/{experiment_group(name)}/{name}/conditional_gallery.png)",
                    "",
                ]
            )
    (root / "docs/14_advanced_results.md").write_text("\n".join(lines) + "\n")
    print("Updated docs/14_advanced_results.md from real saved metrics")


if __name__ == "__main__":
    main()
