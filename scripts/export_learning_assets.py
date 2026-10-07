"""Export curated course/evidence/text artifacts and generate the results chapter.

Large raw tensors and weights stay in outputs. Public text grids contain the
first eight outputs at original resolution, not cherry-picked images.
"""

import json
import shutil
from pathlib import Path

if __package__:
    from .asset_layout import experiment_group
else:
    from asset_layout import experiment_group

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion_lab.utils import save_json
from diffusion_lab.visualize import save_rendered_image

ROOT = Path(__file__).resolve().parents[1]


def compact_copy(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "metrics.json",
        "config.yaml",
        "denoising_profile.json",
        "summary.json",
        "consistency_study.png",
        "classes.png",
        "prompts.json",
        "command.json",
        "paired_sensitivity.json",
        "per_example_bpd.json",
        "likelihood_sensitivity.png",
        "contrast_alignment.json",
        "matched_alignment.png",
    ]:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    if (source / "samples.pt").exists() and source.name.startswith("coco_text"):
        samples = torch.load(
            source / "samples.pt", map_location="cpu", weights_only=True, mmap=True
        )
        save_rendered_image(samples[:8], target / "samples.png", nrow=4)
    elif (source / "samples.png").exists():
        shutil.copy2(source / "samples.png", target / "samples.png")


def merge_weights(rows):
    path = ROOT / "assets/metadata/checkpoints.json"
    manifest = json.loads(path.read_text())
    entries = {entry["sha256"]: entry for entry in manifest["checkpoints"]}
    for name, row in rows:
        if not row.get("checkpoint_sha256") or row.get("weights") == "base":
            continue
        digest = row["checkpoint_sha256"]
        entry = entries.setdefault(
            digest,
            {
                "sha256": digest,
                "checkpoint_at_evaluation": row["checkpoint"],
                "training_steps": row.get("training_steps"),
                "evaluations": [],
            },
        )
        if name not in entry["evaluations"]:
            entry["evaluations"].append(name)
    manifest["checkpoints"] = list(entries.values())
    save_json(path, manifest)


def main():
    local = ROOT / "outputs"
    has_studies = any(
        (local / name / "completed.json").exists()
        for name in [
            "consistency_refinement_study",
            "expanded_likelihood_study",
            "training_seed_study",
        ]
    )
    has_images = any((local / "mnist_training_seed_study").glob("*_summary.json"))
    has_text = any((path / "metrics.json").exists() for path in local.glob("coco_text*"))
    if not any([has_studies, has_images, has_text]):
        print(
            "No local learning-extension evaluations; published chapter and assets were preserved."
        )
        return
    evidence = ROOT / "assets/studies"
    evidence.mkdir(exist_ok=True)
    rows = []
    for name in [
        "consistency_refinement_study",
        "expanded_likelihood_study",
        "training_seed_study",
    ]:
        source = ROOT / "outputs" / name
        if not (source / "completed.json").exists():
            continue
        target = evidence / name
        compact_copy(source, target)
        if name == "consistency_refinement_study":
            for child in sorted(source.glob("final_*")):
                compact_copy(child, target / child.name)
                rows.append(
                    (
                        f"evidence/{name}/{child.name}",
                        json.loads((child / "metrics.json").read_text()),
                    )
                )
        if name == "expanded_likelihood_study":
            for child in sorted(source.iterdir()):
                if child.is_dir() and (child / "summary.json").exists():
                    compact_copy(child, target / child.name)
        if name == "training_seed_study":
            report = json.loads((source / "metrics.json").read_text())
            for row in report["results"]:
                child = (
                    source
                    / f"gmm_{row['family']}_seed{row['training_seed']}"
                    / "matched_nfe_evaluation"
                )
                compact_copy(child, target / child.parent.name)
                rows.append((f"evidence/{name}/{child.parent.name}", row))
    image_source = ROOT / "outputs/mnist_training_seed_study"
    image_reports = {}
    for algorithm in ["edm", "flow"]:
        published = evidence / "mnist_training_seed_study" / f"{algorithm}_summary.json"
        if published.exists():
            image_reports[algorithm] = json.loads(published.read_text())
        summary = image_source / f"{algorithm}_summary.json"
        if summary.exists():
            target = evidence / "mnist_training_seed_study"
            target.mkdir(exist_ok=True)
            shutil.copy2(summary, target / summary.name)
            image_reports[algorithm] = json.loads(summary.read_text())
            for row in image_reports[algorithm]["results"]:
                child = image_source / f"{algorithm}_seed{row['training_seed']}" / "evaluation"
                compact_copy(child, target / child.parent.name)
                rows.append((f"evidence/mnist_training_seed_study/{child.parent.name}", row))
    data = Path("/data/tii/Image_data/coco2017/verification.json")
    if data.exists():
        target = ROOT / "assets/reports/datasets"
        target.mkdir(exist_ok=True)
        shutil.copy2(data, target / "coco2017_verification.json")
    for source in sorted((ROOT / "outputs").glob("coco_text*")):
        if not (source / "metrics.json").exists():
            continue
        row = json.loads((source / "metrics.json").read_text())
        target = ROOT / "assets/experiments" / experiment_group(source.name) / source.name
        compact_copy(source, target)
        rows.append((source.name, row))
    text_rows = [
        (path.parent.name, json.loads(path.read_text()))
        for path in sorted((ROOT / "assets/experiments/text").glob("coco_text*/metrics.json"))
    ]
    merge_weights(rows)
    result_index = ROOT / "assets/metadata/results.json"
    indexed = {row["experiment"]: row for row in json.loads(result_index.read_text())}
    for name, row in text_rows:
        indexed[name] = {"experiment": name, **row}
    save_json(result_index, [indexed[name] for name in sorted(indexed)])
    alignment = ROOT / "outputs/text_alignment_study"
    if (alignment / "metrics.json").exists():
        compact_copy(alignment, evidence / "text_alignment_study")
    notebook_count = len(list((ROOT / "notebooks").glob("*.ipynb")))
    lines = [
        "# Curriculum, stronger evidence and COCO text results — version 0.4",
        "",
        f"The curriculum uses Markdown reference chapters plus {notebook_count} CPU lesson notebooks. "
        "See the [course index](../notebooks/README.md). "
        "Training/evaluation results below are generated from saved reports; smoke runs are integration checks.",
        "",
        "## Completed data and lessons",
        "",
        "- [Notebook validation](../assets/reports/diagnostics/notebook_validation.json) records default execution status and scope; "
        "[the course revision](29_notebook_course.md) describes expanded experiments and separate checks.",
        "- COCO 2017 train: 118,287 images / 591,753 captions; val: 5,000 images / 25,014 captions. "
        "All images decoded and matched annotation dimensions; extracted ZIP members passed CRC checks, with local hashes recorded in "
        "[verification](../assets/reports/datasets/coco2017_verification.json).",
        "",
        "## Frozen consistency student: refinement remains weak",
        "",
        "The old scores and this study use different reference/generation counts; compare within a protocol. "
        "Schedules were selected on 1k/1k, then evaluated on a disjoint 5k reference subset with a new generation seed.",
        "",
        "| Calls | Selected sigma levels | FID, 5k/5k | IS |",
        "|---:|---|---:|---:|",
    ]
    study = evidence / "consistency_refinement_study/metrics.json"
    if study.exists():
        report = json.loads(study.read_text())
        for row in report["final"]:
            lines.append(
                f"| {row['steps']} | {row['consistency_sigmas']} | {row['fid_2048']:.2f} | {row['inception_score_mean']:.2f} |"
            )
    lines.extend(
        [
            "",
            "Refinement did not beat one call in this sweep. The denoising profile flags large intermediate-noise errors "
            "but does not prove a unique cause. [Protocol and limits](25_evidence_studies.md).",
            "",
            "## Expanded image likelihood diagnostics",
            "",
            "| Case | Examples | Quantity | Mean bpd | 95% interval across examples |",
            "|---|---:|---|---:|---|",
        ]
    )
    path = evidence / "expanded_likelihood_study/metrics.json"
    if path.exists():
        for row in json.loads(path.read_text())["results"]:
            ci = row["bpd"]["confidence_interval"]
            lines.append(
                f"| {row['name']} | {row['samples']} | {'DDPM negative ELBO' if row['integration_steps'] is None else 'Numerical ODE density'} | {row['bpd']['mean']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
            )
    lines.extend(
        [
            "",
            "These are known-class conditional estimates, without CFG. Example intervals do not establish numerical convergence "
            "or remove truncation/trace/Monte Carlo bias. Grid/probe comparisons use the first 128 CIFAR examples; "
            "compare to the same subset rather than the full 512-example mean.",
            "[Matched numerical sensitivity](../assets/studies/expanded_likelihood_study/paired_sensitivity.json) "
            "and per-example bpd files retain the actual values and batch provenance.",
        ]
    )
    sensitivity = evidence / "expanded_likelihood_study/paired_sensitivity.json"
    if sensitivity.exists():
        report = json.loads(sensitivity.read_text())
        lines.extend(
            [
                "",
                "Matched CIFAR baseline (first 128 examples): "
                f"**{report['results'][0]['baseline_bpd']['mean']:.4f} bpd**. "
                "Changing 100 to 200 Heun steps changes mean bpd by "
                f"**{report['results'][0]['paired_difference_bpd']['mean']:+.4f}**; "
                "the finite grid is therefore a material part of this diagnostic.",
            ]
        )
    lines.extend(
        [
            "",
            "## Independently trained models",
            "",
            "GMM: three DDPM and three flow checkpoints, 6,000 updates each. Sampling uses 100 measured "
            "model calls, seed 123 and a separate reference seed 999.",
            "",
            "| Model | Sliced W2 mean | Training-seed SD | 95% training-seed interval |",
            "|---|---:|---:|---|",
        ]
    )
    path = evidence / "training_seed_study/metrics.json"
    if path.exists():
        for family, metrics in json.loads(path.read_text())["summary"].items():
            metric = metrics["sliced_w2"]
            ci = metric["confidence_interval"]
            lines.append(
                f"| {family} | {metric['mean']:.4f} | {metric['std']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
            )
    lines.extend(
        [
            "",
            "MNIST: three EDM and three flow models use the same U-Net, 20,000 updates, ordinary class conditioning, "
            "2,048 reference/generated digits and an independent digit assessor. EDM uses 31 versus flow 32 calls. "
            "Digit feature distance is not Inception FID.",
            "",
            "| Model | Completed training seeds | Digit feature distance mean ± seed SD | Requested-class accuracy mean |",
            "|---|---|---|---|",
        ]
    )
    for algorithm in ["edm", "flow"]:
        if algorithm in image_reports:
            report = image_reports[algorithm]
            metric = report["summary"]["frechet_digit_features_128"]
            accuracy = report["summary"]["requested_class_accuracy"]["mean"]
            lines.append(
                f"| {algorithm} | 42,43,44 | {metric['mean']:.2f} ± {metric['std']:.2f} | {accuracy:.1%} |"
            )
        else:
            completed = [
                seed
                for seed in [42, 43, 44]
                if (image_source / f"{algorithm}_seed{seed}/training_complete.json").exists()
            ]
            lines.append(f"| {algorithm} | {completed}; remaining runs pending | — | — |")
    lines.extend(
        [
            "",
            "These controlled comparisons do not estimate training uncertainty for the earlier CIFAR/face headline models.",
            "",
            "## COCO text fine-tuning and matched pretrained baseline",
            "",
        ]
    )
    status = ROOT / "outputs/learning_extension_status.json"
    if status.exists():
        lines.append("Status: **" + json.loads(status.read_text())["status"] + "**.\n")
    lines.extend(
        [
            "LoRA adapts local pretrained SD2 attention projections at 512px; the VAE and text encoder remain frozen. "
            "The compact text U-Net is implemented/tested and configured for scratch training, "
            "but this chapter does not claim a completed full-budget scratch COCO run.",
            "The 3,000-update adapter budget presents 24,000 image-caption pairs; it does not cover a full COCO epoch.",
            "",
            "| Evaluation | Weights | CFG | Captions | CLIP cosine | Paired minus mismatched cosine | Calls/image |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in text_rows:
        lines.append(
            f"| [{name}](../assets/experiments/{experiment_group(name)}/{name}/metrics.json) | {row['weights']} | {row['cfg_scale']} | {row['generated_count']} | {row.get('clip_cosine_mean', 0):.4f} | {row.get('paired_minus_mismatched_mean', 0):.4f} | {row['nfe_per_image']} |"
        )
    lines.extend(
        [
            "",
            "CLIP is an independent broad-alignment diagnostic, not a counting/spatial-binding benchmark. "
            "These 256-caption runs do not report headline FID. External pretraining overlap is unknown. "
            "Base/adapted comparisons use identical captions, batch sizes and initial-noise seeds. "
            "Prompt-pair grids use identical noise for every prompt and show the first eight outputs at original resolution.",
        ]
    )
    for name in ["coco_text_base_pairs", "coco_text_lora_pairs"]:
        if (ROOT / "assets/experiments" / experiment_group(name) / name / "samples.png").exists():
            lines.extend(
                [
                    "",
                    f"![{name}](../assets/experiments/{experiment_group(name)}/{name}/samples.png)",
                ]
            )
        contrast = (
            ROOT / "assets/experiments" / experiment_group(name) / name / "contrast_alignment.json"
        )
        if contrast.exists():
            result = json.loads(contrast.read_text())
            lines.extend(
                [
                    "",
                    f"{name}: CLIP prefers the intended caption in "
                    f"{result['contrast_caption_preference_fraction']:.0%} of six controlled decisions. "
                    "Both base and adapted models pass these small contrasts; this does not establish superior "
                    "counting or anatomy. The two final prompts are paraphrases, excluded from contrast scoring.",
                ]
            )
    for name, title in [
        ("prompt_contrasts.png", "Labeled base/adapted text contrasts"),
        ("text_cfg.png", "Same caption and initial noise across guidance strengths"),
    ]:
        if (evidence / "text_gallery" / name).exists():
            lines.extend(["", f"![{title}](../assets/studies/text_gallery/{name})"])
    if (alignment / "metrics.json").exists():
        result = json.loads((alignment / "metrics.json").read_text())
        metric = result["paired_difference_clip_cosine"]
        ci = metric["confidence_interval"]
        lines.extend(
            [
                "",
                "Matched base/adapted CLIP cosine difference: "
                f"**{metric['mean']:+.4f}**, caption-wise 95% interval "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]. "
                "This is one fine-tuning run and does not establish a general perceptual-quality improvement. "
                "[Per-caption evidence](../assets/studies/text_alignment_study/metrics.json).",
                "",
                "![Matched alignment and CFG sensitivity](../assets/studies/text_alignment_study/matched_alignment.png)",
            ]
        )
    (ROOT / "docs/26_learning_and_text_results.md").write_text("\n".join(lines) + "\n")
    if text_rows:
        regular = [(n, r) for n, r in text_rows if not n.endswith("_pairs")]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        names = [n.removeprefix("coco_text_") for n, _ in regular]
        axes[0].bar(names, [r["clip_cosine_mean"] for _, r in regular])
        axes[0].set(ylabel="Independent CLIP cosine", title="Held-out captions; matched noise")
        axes[1].bar(names, [r["paired_minus_mismatched_mean"] for _, r in regular])
        axes[1].set(ylabel="Paired minus cyclic mismatch cosine")
        for ax in axes:
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(evidence / "text_alignment.png", dpi=160)
        plt.close(fig)
    print("Exported curated learning assets and refreshed measured results.")


if __name__ == "__main__":
    main()
