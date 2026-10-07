"""Publish compact figures/configs/metrics locally, keeping checkpoints in outputs/."""

import hashlib
import json
import shutil
import tempfile
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


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "assets/experiments"
    if not (root / "outputs").exists():
        print("No local outputs directory; existing published evidence was preserved.")
        return
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest_path = root / "assets/metadata/checkpoints.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    checkpoints = {entry["sha256"]: entry for entry in manifest.get("checkpoints", [])}
    classifier_checkpoints = {entry["run"]: entry for entry in manifest.get("classifiers", [])}
    for directory in sorted((root / "outputs").iterdir()):
        publication = directory / "publication.json"
        if publication.is_file() and not json.loads(publication.read_text()).get("include", True):
            continue
        if (
            not directory.is_dir()
            or directory.name.startswith(("smoke", "coco_text_"))
            or directory.name
            in {
                "cifar10_edm_progress",
                "mnist_edm_progress",
                "consistency_refinement_study",
                "expanded_likelihood_study",
                "training_seed_study",
                "mnist_training_seed_study",
                "text_alignment_study",
            }
            or not (directory / "metrics.json").exists()
        ):
            continue
        target = destination / experiment_group(directory.name) / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix in {".png", ".gif", ".yaml", ".json"}
                and "hydra" not in path.parts
            ):
                destination_path = target / path.relative_to(directory)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_copy(path, destination_path)
        metrics = json.loads((directory / "metrics.json").read_text())
        metric_records = [metrics] if isinstance(metrics, dict) else metrics
        for record in metric_records:
            if (
                not isinstance(record, dict)
                or not record.get("checkpoint_sha256")
                or not record.get("checkpoint")
            ):
                continue
            digest = record["checkpoint_sha256"]
            entry = checkpoints.setdefault(
                digest,
                {
                    "sha256": digest,
                    "checkpoint_at_evaluation": record["checkpoint"],
                    "training_steps": record.get("training_steps"),
                    "evaluations": [],
                },
            )
            if entry["training_steps"] is None and record.get("training_steps") is not None:
                entry["training_steps"] = record["training_steps"]
            if directory.name not in entry["evaluations"]:
                entry["evaluations"].append(directory.name)
        rows.append(
            {"experiment": directory.name, **metrics}
            if isinstance(metrics, dict)
            else {"experiment": directory.name, "metrics": metrics}
        )
    for directory in sorted((root / "outputs").glob("*classifier")):
        if not (directory / "diagnostics.json").exists():
            continue
        target = root / "assets/reports/classifiers" / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ["config.json", "history.json", "diagnostics.json"]:
            if (directory / filename).exists():
                shutil.copy2(directory / filename, target / filename)
        checkpoint = directory / "classifier.pt"
        if checkpoint.exists():
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            entry = {
                "run": directory.name,
                "checkpoint": str(checkpoint.relative_to(root)),
                "sha256": digest.hexdigest(),
                "heldout_report": str((target / "diagnostics.json").relative_to(root)),
            }
            classifier_checkpoints[directory.name] = entry
            (target / "checkpoint.json").write_text(json.dumps(entry, indent=2) + "\n")
    index_path = root / "assets/metadata/results.json"
    existing = json.loads(index_path.read_text()) if index_path.exists() else []
    indexed = {row["experiment"]: row for row in existing}
    indexed.update({row["experiment"]: row for row in rows})
    index_path.write_text(json.dumps([indexed[name] for name in sorted(indexed)], indent=2) + "\n")
    (root / "assets/metadata/checkpoints.json").write_text(
        json.dumps(
            {
                "description": "Hashes of the exact weight files loaded for evaluations. Weights remain in outputs; a latest checkpoint path may subsequently be replaced. Prefer a matching immutable milestone for reproduction.",
                "checkpoints": list(checkpoints.values()),
                "classifiers": list(classifier_checkpoints.values()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported {len(rows)} experiments to {destination}")
    # The learning extension has nested studies and full-resolution text grids;
    # its own exporter curates compact figures and merges their weight hashes.
    if __package__:
        from .export_learning_assets import main as export_learning
    else:
        from export_learning_assets import main as export_learning

    export_learning()

    if __package__:
        from .build_asset_index import main as build_index
    else:
        from build_asset_index import main as build_index

    build_index()


if __name__ == "__main__":
    main()
