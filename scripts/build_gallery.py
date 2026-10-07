"""Build the results chapter directly from exported metrics and local sample tensors."""

import json
from pathlib import Path

if __package__:
    from .asset_layout import experiment_group
else:
    from asset_layout import experiment_group

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]


def particle_grid(names, filename, columns=3, conditional=False):
    figure, axes = plt.subplots(
        (len(names) + columns - 1) // columns,
        columns,
        figsize=(columns * 4, ((len(names) + columns - 1) // columns) * 3.7),
        squeeze=False,
    )
    for ax, name in zip(axes.flat, names, strict=False):
        samples = torch.load(ROOT / "outputs" / name / "samples.pt", weights_only=True)["samples"]
        ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.45)
        if conditional and name != "cfg_scale_0":
            circle = plt.Circle((3, 0), 0.75, color="red", fill=False, linestyle="--", linewidth=1)
            ax.add_patch(circle)
        ax.set(
            title=name.replace("gmm_", "").replace("_", " "),
            xlim=(-5, 5),
            ylim=(-5, 5),
            aspect="equal",
        )
        ax.title.set_fontsize(10)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    figure.tight_layout()
    figure.savefig(ROOT / "assets/teaching/galleries" / filename, dpi=160)
    plt.close(figure)


def main():
    rows = json.loads((ROOT / "assets/metadata/results.json").read_text())
    by_name = {row["experiment"]: row for row in rows}
    gmm_names = [
        "gmm_ddpm_x0",
        "gmm_linear_epsilon",
        "gmm_cosine_v",
        "gmm_vp_epsilon",
        "gmm_ve_score",
        "gmm_flow",
        "gmm_flow_ot",
        "gmm_reflow",
        "gmm_bridge_sde",
    ]
    particle_grid(gmm_names, "gmm_gallery.png")
    guidance_names = [
        "cfg_scale_0",
        "cfg_scale_1",
        "cfg_scale_2",
        "cfg_scale_4",
        "classifier_scale_1",
        "classifier_analytic_scale_1",
    ]
    particle_grid(guidance_names, "guidance_gallery.png", conditional=True)
    content = [
        "# 8. Results from actual runs",
        "",
        "These figures and values come from the saved local experiments, not paper",
        "benchmarks. Training used the simple_udm environment on the three L40S GPUs",
        "of chapelle. Configurations and metrics are exported alongside each figure.",
        "The tables below are regenerated from assets/metadata/results.json.",
        "",
        "## Learned Gaussian-mixture models",
        "",
        "![Learned GMM samples](../assets/teaching/galleries/gmm_gallery.png)",
        "",
        "Each panel contains 1,024 generated particles. The target is eight Gaussian",
        "components on a radius-three ring, with component standard deviation 0.25.",
        "Look at cloud width, off-mode samples, and occupancy before reading a score.",
        "Most runs use 6,000 optimizer steps; bridge runs use 8,000.",
        "",
        "| Experiment | Sliced W2 | Empirical W2 (512) | Within 3σ | Mode coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in gmm_names:
        row = by_name[name]
        content.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['sliced_w2']:.4f} | {row['empirical_w2_subsample']:.4f} | {row['fraction_within_3sigma']:.3f} | {row['mode_coverage_3sigma']:.3f} |"
        )
    content += [
        "",
        "These are single-seed runs with different objectives and some different",
        "sampling grids. Their values do not establish a general ranking of methods.",
        "The real-versus-real diagnostic shows that finite sample measures differ",
        "even when both are drawn from the correct target distribution.",
        "",
        "## Conditional generation",
        "",
        "![Guidance comparison](../assets/teaching/galleries/guidance_gallery.png)",
        "",
        "All conditional panels target component zero at (3,0). The red circle is",
        "the 3σ diagnostic region. CFG scale zero generates unconditionally. Strong",
        "CFG changes the cloud's spread and location; bigger scale need not improve",
        "distribution fidelity.",
        "",
        "The learned classifier panel is retained even though it distorts the cloud.",
        "The analytic classifier reference isolates the guidance formula from",
        "classifier approximation error. Both still use the learned diffusion model.",
        "",
        "| Guidance experiment | Sliced W2 to appropriate reference | Within 3σ |",
        "|---|---:|---:|",
    ]
    for name in guidance_names:
        row = by_name[name]
        content.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['sliced_w2']:.4f} | {row['fraction_within_3sigma']:.3f} |"
        )
    content += [
        "",
        "![Noisy classifier accuracy and gradient error](../assets/reports/diagnostics/classifier_diagnostics.png)",
        "",
        "At small noise, the learned classifier has near-perfect classification",
        "accuracy. That alone does not constrain the input gradient sufficiently",
        "for reliable generative guidance. The separate gradient diagnostic compares",
        "it with the known posterior classifier.",
        "",
        "## Rectification and reflow",
        "",
        "![Measured straightness](../assets/reports/diagnostics/reflow_straightness.png)",
        "",
        "This same-initial-seed measurement compares path length with direct endpoint",
        "distance. Here OT flow and reflow have much smaller excess path length.",
        "This is measured behavior of these checkpoints, not a general OT guarantee.",
        "",
        "## Conditional flow generation",
        "",
        "![Conditional flow samples](../assets/experiments/toy/gmm_flow_cfg_class0/samples.png)",
        "",
        "![Conditional flow generation](../assets/experiments/toy/gmm_flow_cfg_class0/generation.gif)",
        "",
        "This separately trained flow uses component labels and null-token dropout.",
        "The conditional velocity generates component zero; no diffusion score or",
        "classifier is needed for this ordinary conditional flow example.",
        "",
        "## Short image-training baselines",
        "",
        "The image experiments resize to 32×32. CelebA/FFHQ use 4,096 training images;",
        "ImageNet uses 10,000, selected across all 1,000 classes. These are deliberately",
        "small teaching runs. ImageNet is trained conditionally with null-token",
        "dropout; the table evaluates its unconditional branch.",
        "",
        "| Experiment | Training steps | FID (2048) | KID mean | IS mean | Real/generated count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["celeba_ddpm", "celeba_flow", "imagenet_conditional", "ffhq_ddpm"]:
        row = by_name[name]
        content.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['training_steps']} | {row['fid_2048']:.2f} | {row['kid_mean']:.4f} | {row['inception_score_mean']:.3f} | {row['real_count']}/{row['generated_count']} |"
        )
    content += [
        "",
        "**128-example FID is a noisy diagnostic, not a publishable benchmark.**",
        "Different dataset rows are not directly comparable. IS on face datasets",
        "has limited semantic relevance. Samples remain rough after these short runs.",
        "",
        "### CelebA diffusion",
        "",
        "![CelebA diffusion samples](../assets/experiments/faces/celeba_ddpm/samples.png)",
        "",
        "![CelebA diffusion generation](../assets/experiments/faces/celeba_ddpm/generation.gif)",
        "",
        "### CelebA flow matching",
        "",
        "![CelebA flow samples](../assets/experiments/faces/celeba_flow/samples.png)",
        "",
        "![CelebA flow generation](../assets/experiments/faces/celeba_flow/generation.gif)",
        "",
        "### ImageNet",
        "",
        "![ImageNet early-run samples](../assets/experiments/imagenet/imagenet_conditional/samples.png)",
        "",
        "#### Conditional image guidance examples",
        "",
        "![ImageNet CFG class zero](../assets/experiments/imagenet/imagenet_cfg_class0/samples.png)",
        "",
        "![ImageNet learned classifier guidance](../assets/experiments/imagenet/imagenet_classifier_class0/samples.png)",
        "",
        "These panels target folder class_0000 using CFG and a separately trained",
        "noisy classifier. The short run does not establish accurate ImageNet class",
        "adherence. Metrics are disabled for these 32-example visual comparisons;",
        "the bounded validation selection has only one reference image per class.",
        "",
        "### FFHQ",
        "",
        "![FFHQ early-run samples](../assets/experiments/faces/ffhq_ddpm/samples.png)",
        "",
        "## Sampling budgets",
        "",
        "The comparisons below use evaluation seed 123 and the same checkpoint",
        "within each method family. Field evaluations differ between Euler, Heun,",
        "and RK4. CPU/GPU runtime is not mixed into a claim of algorithmic superiority.",
        "",
        "| Experiment | Sliced W2 |",
        "|---|---:|",
    ]
    for name in [
        "gmm_ddpm_ancestral",
        "gmm_ddim_20",
        "gmm_ddim_5",
        "gmm_flow_euler_10",
        "gmm_flow_heun_10",
        "gmm_flow_rk4_10",
        "gmm_reflow_euler_10",
    ]:
        content.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {by_name[name]['sliced_w2']:.4f} |"
        )
    content += [
        "",
        "Coarser steps can coincidentally compensate model error. A lower learned",
        "sample metric at fewer steps does not contradict an integrator's analytic",
        "convergence order. Use the analytic solver lesson to isolate numerical error.",
        "",
        "## Reproduce this page",
        "",
        "~~~bash",
        "python scripts/export_assets.py",
        "python -m diffusion_lab.diagnostics",
        "python scripts/build_gallery.py",
        "~~~",
        "",
        "The exporter keeps checkpoints and raw sample tensors in Git-ignored outputs.",
        "Every exported experiment retains its metrics; training runs also retain",
        "resolved configuration, environment information, and loss history.",
    ]
    (ROOT / "docs/08_results.md").write_text("\n".join(content) + "\n")


if __name__ == "__main__":
    main()
