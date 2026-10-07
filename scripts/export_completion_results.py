"""Curate a completed scratch/text/refinement study without exporting weights.

The source contains final.json with text evaluation directories and consistency
comparison directories. Every comparison must have its own completed.json.
Other incomplete training output remains private. Existing historical results
are preserved and the new report is explicitly scoped to one training seed.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from diffusion_lab.utils import save_json
from diffusion_lab.visualize import save_rendered_image

ROOT = Path(__file__).resolve().parents[1]


def copy_reports(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "metrics.json",
        "config.yaml",
        "prompts.json",
        "command.json",
        "classes.png",
        "comparison.png",
        "completed.json",
        "contrast_alignment.json",
    ]:
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)
    if (source / "samples.pt").is_file():
        values = torch.load(source / "samples.pt", map_location="cpu", weights_only=True, mmap=True)
        is_text = not isinstance(values, dict)
        pixels = values if is_text else values["samples"]
        save_rendered_image(
            pixels[: 8 if is_text else 64], target / "samples.png", nrow=4 if is_text else 8
        )
    elif (source / "samples.png").is_file():
        shutil.copyfile(source / "samples.png", target / "samples.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/completion_studies")
    parser.add_argument("--target", type=Path, default=ROOT / "assets/studies/scratch_refinement")
    args = parser.parse_args()
    manifest_file = args.source / "final.json"
    if not manifest_file.exists():
        print("Final study manifest is absent; published results were preserved.")
        return
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("status") != "completed_evaluations":
        raise ValueError("Publish only a completed evaluation manifest")
    entries = []
    # Check completeness before copying anything.
    for item in manifest["text_evaluations"]:
        folder = ROOT / item
        if not (folder / "metrics.json").is_file():
            raise ValueError(f"Missing text evaluation: {folder}")
        entries.append((folder, args.target / folder.name))
    protocol_links = []
    for item in manifest["consistency_comparisons"]:
        folder = ROOT / item
        if not (folder / "completed.json").is_file():
            raise ValueError(f"Incomplete consistency comparison: {folder}")
        entries.append((folder, args.target / folder.name))
        for family in ["baseline", "candidate"]:
            for calls in [1, 2, 4]:
                child = folder / f"{family}_final_{calls}"
                if not (child / "metrics.json").is_file():
                    raise ValueError(f"Missing matched final case: {child}")
                entries.append((child, args.target / folder.name / child.name))
    for source, target in entries:
        copy_reports(source, target)
    save_json(args.target / "manifest.json", manifest)
    hashes_file = ROOT / "assets/metadata/checkpoints.json"
    hashes = json.loads(hashes_file.read_text())
    by_hash = {entry["sha256"]: entry for entry in hashes["checkpoints"]}
    indexed_file = ROOT / "assets/metadata/results.json"
    indexed = {row["experiment"]: row for row in json.loads(indexed_file.read_text())}
    for source, target in entries:
        row = json.loads((source / "metrics.json").read_text())
        name = str(target.relative_to(ROOT / "assets"))
        indexed[name] = {"experiment": name, **row}
        if row.get("checkpoint_sha256"):
            entry = by_hash.setdefault(
                row["checkpoint_sha256"],
                {
                    "sha256": row["checkpoint_sha256"],
                    "checkpoint_at_evaluation": row["checkpoint"],
                    "training_steps": row["training_steps"],
                    "evaluations": [],
                },
            )
            if name not in entry["evaluations"]:
                entry["evaluations"].append(name)
    hashes["checkpoints"] = list(by_hash.values())
    save_json(hashes_file, hashes)
    save_json(indexed_file, [indexed[name] for name in sorted(indexed)])
    document = ROOT / "docs/28_scratch_and_refinement.md"
    content = document.read_text()
    marker = "\n## Completed evaluation reports\n"
    content = content.split(marker)[0]
    content = content.replace(
        "Training is in progress.",
        "The configured training budgets and evaluation jobs have completed.",
    )
    content = content.replace(
        "No new final FID, prompt-adherence result or completed training budget is claimed\nuntil the corresponding evaluation reports exist.",
        "The completed reports below record the observed quality and limitations; a completed budget is not a universal quality guarantee.",
    )
    lines = [
        "",
        "## Completed evaluation reports",
        "",
        "Each model has one training seed. Selection and final protocols are separated.",
        "",
        "| Comparison | Candidate calls | Candidate FID | Baseline FID | Candidate IS |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in manifest["consistency_comparisons"]:
        folder = ROOT / item
        report = json.loads((folder / "metrics.json").read_text())
        for baseline, candidate in zip(
            report["final"]["baseline"], report["final"]["candidate"], strict=True
        ):
            lines.append(
                f"| {folder.name} | {candidate['steps']} | {candidate['fid_2048']:.2f} | {baseline['fid_2048']:.2f} | {candidate['inception_score_mean']:.2f} |"
            )
        link = (args.target / folder.name / "metrics.json").relative_to(ROOT)
        protocol_links.append(f"[{folder.name} protocol and scores](../{link})")
    lines.extend(
        [
            "",
            *protocol_links,
            "",
            "| Text case | Updates | CFG | CLIP paired mean | CLIP paired − mismatched | FID |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    galleries = []
    for item in manifest["text_evaluations"]:
        folder = ROOT / item
        row = json.loads((folder / "metrics.json").read_text())
        if "clip_cosine_mean" not in row:
            continue
        fid = f"{row['fid_2048']:.2f}" if "fid_2048" in row else "—"
        lines.append(
            f"| {folder.name} | {row['training_steps']} | {row['cfg_scale']} | {row['clip_cosine_mean']:.4f} | {row['paired_minus_mismatched_mean']:.4f} | {fid} |"
        )
        link = (args.target / folder.name / "samples.png").relative_to(ROOT)
        galleries.extend(["", f"![First generated examples: {folder.name}](../{link})", ""])
    lines.extend(galleries)
    lines.extend(
        [
            "",
            "CLIP is a coarse semantic diagnostic. Visual quality, counting and spatial binding require separate interpretation.",
            "Call-budget comparisons use their recorded sample counts; historical scores with different counts should not be compared directly.",
            "Release packaging and external checkpoint hosting remain separate from these training results.",
        ]
    )
    document.write_text(content + "\n".join(lines) + "\n")
    readme = ROOT / "README.md"
    text = readme.read_text().replace(
        "is now running. Its final results are pending; the reported LoRA experiment",
        "has completed its configured training/evaluation budgets. See the report for observed quality; the reported LoRA experiment",
    )
    readme.write_text(text)
    audit = ROOT / "docs/22_repository_audit.md"
    if audit.exists():
        text = (
            audit.read_text()
            .replace(
                "Historical CIFAR refinement worsens FID; a new 4k pilot improves its tested four-call result, while final retraining comparisons are pending",
                "Historical CIFAR refinement worsens FID; new 40k CD/iCT budgets and matched one/two/four-call comparisons are complete, with observed quality recorded in the new study",
            )
            .replace(
                "The 100k-update budget/evaluation are in progress; its VAE/text encoder are external and frozen, and early outputs remain weak",
                "The 100k-update budget and held-out evaluation completed; its VAE/text encoder remain external and frozen, and generation quality is reported separately",
            )
            .replace(
                "Its full training,\nselection and final-evaluation jobs are running or queued; it does not claim\nthat the remaining quality gaps have already been closed.",
                "Its configured training, selection and final-evaluation jobs completed.\nThe measured scores and visual artifacts determine which quality gaps remain.",
            )
            .replace(
                "COCO training is ongoing;\n60k samples show clearer buses/fruit but distorted anatomy and geometry.",
                "COCO's configured training and evaluation budgets completed.\nThe study retains the intermediate visual review and final measured results;\nfinal showcase quality requires visual assessment of the completed model.",
            )
        )
        audit.write_text(text)
    validation = ROOT / "assets/metadata/validation.json"
    if validation.exists():
        report = json.loads(validation.read_text())
        report["completed_training_extension"] = manifest
        report.pop("running_training", None)
        report.setdefault("training_extension_checks", {})["training_status"] = (
            "Configured training and evaluation budgets completed; post-export final checks are a separate stage."
        )
        save_json(validation, report)
    print(f"Exported {len(entries)} compact reports; weights and raw tensors stayed local.")


if __name__ == "__main__":
    main()
