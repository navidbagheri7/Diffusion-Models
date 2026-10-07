"""Build dataset browsing indexes without copying or selecting individual samples."""

import json
from pathlib import Path

if __package__:
    from .asset_layout import experiment_group
else:
    from asset_layout import experiment_group

TITLES = {
    "toy": "Two-dimensional datasets and guidance",
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "faces": "Face datasets and latent models",
    "text": "Text-conditioned generation",
    "likelihood": "Likelihood and ELBO diagnostics",
    "bridges": "Bridge experiments",
    "imagenet": "ImageNet integration examples",
    "custom": "Custom datasets",
}


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    experiment_root = assets / "experiments"
    groups, inventory = [], []
    for folder in sorted(experiment_root.iterdir()):
        if not folder.is_dir():
            continue
        lines = [
            f"# {TITLES.get(folder.name, folder.name)}",
            "",
            "[Asset showcase](../../README.md) · [All experiment groups](../README.md)",
            "",
            "Browse saved outputs and original reports. Alphabetical order is not a quality ranking.",
            "Read validity labels, training steps, evaluation counts and protocols before comparing results.",
            "Intermediate and invalidated examples remain when the course discusses them.",
            "",
            "| Experiment | Visuals | Evidence |",
            "|---|---|---|",
        ]
        count = 0
        for run in sorted(folder.iterdir()):
            if not run.is_dir():
                continue
            expected = experiment_group(run.name)
            if expected != folder.name:
                raise ValueError(f"Misplaced experiment {run.name}: expected {expected}")
            visuals, reports = [], []
            for filename, label in (
                ("classes.png", "classes"),
                ("conditional_gallery.png", "conditional"),
                ("samples.png", "samples"),
                ("generation.gif", "animation"),
                ("interpolation.png", "interpolation"),
            ):
                if (run / filename).is_file():
                    visuals.append(f"[{label}]({run.name}/{filename})")
            for filename, label in (("metrics.json", "metrics"), ("config.yaml", "config")):
                if (run / filename).is_file():
                    reports.append(f"[{label}]({run.name}/{filename})")
            lines.append(
                f"| [{run.name}]({run.name}/) | {' · '.join(visuals) or 'See run folder'} | {' · '.join(reports) or 'See run folder'} |"
            )
            inventory.append(
                {
                    "experiment": run.name,
                    "group": folder.name,
                    "directory": str(run.relative_to(root)),
                    "files": [
                        str(p.relative_to(run)) for p in sorted(run.rglob("*")) if p.is_file()
                    ],
                }
            )
            count += 1
        (folder / "README.md").write_text("\n".join(lines) + "\n")
        groups.append((folder.name, count))
    lines = [
        "# Experiment evidence by dataset and topic",
        "",
        "[Asset showcase](../README.md) · [Metric protocols](../../docs/12_image_experiments.md)",
        "",
        "Each run keeps figures, configurations and reports together. Weights and training state remain outside this directory.",
        "",
        "| Group | Saved run folders |",
        "|---|---:|",
    ]
    lines.extend(
        f"| [{TITLES.get(group, group)}]({group}/README.md) | {count} |" for group, count in groups
    )
    (experiment_root / "README.md").write_text("\n".join(lines) + "\n")
    (assets / "metadata/asset_index.json").write_text(
        json.dumps(
            {
                "scope": "Browsing inventory; inclusion does not establish image quality or scientific validity",
                "groups": dict(groups),
                "experiments": inventory,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Indexed {len(inventory)} experiment folders in {len(groups)} groups.")


if __name__ == "__main__":
    main()
