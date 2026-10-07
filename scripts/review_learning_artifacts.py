"""Verify completed learning-extension evidence against actual weights and pixels."""

import hashlib
import json
from pathlib import Path

if __package__:
    from .asset_layout import experiment_group
else:
    from asset_layout import experiment_group

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from diffusion_lab.metrics import image_pixels
from diffusion_lab.text_train import model_fingerprint
from diffusion_lab.utils import save_json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def image_review(directory, public, hashes, text=False):
    report = json.loads((directory / "metrics.json").read_text())
    checkpoint = report["checkpoint"]
    if checkpoint not in hashes:
        hashes[checkpoint] = digest(checkpoint)
    assert hashes[checkpoint] == report["checkpoint_sha256"], directory
    saved = torch.load(directory / "samples.pt", map_location="cpu", weights_only=True, mmap=True)
    samples = saved if text else saved["samples"]
    assert len(samples) == report["generated_count"], directory
    for start in range(0, len(samples), 32):
        assert torch.isfinite(samples[start : start + 32]).all(), directory
    count, nrow = (8, 4) if text else (64, 8)
    expected = make_grid(image_pixels(samples[:count]), nrow=nrow).permute(1, 2, 0).numpy()
    with Image.open(public / "samples.png") as image:
        assert np.array_equal(expected, np.asarray(image.convert("RGB"))), public
    assert json.loads((public / "metrics.json").read_text()) == report, public
    if text:
        required_calls = report["steps"] * (1 if report["cfg_scale"] in {0, 1} else 2)
        assert report["nfe_per_image"] == required_calls
        prompts = json.loads((directory / "prompts.json").read_text())
        assert len(prompts["captions"]) == len(samples)
        assert len(report["base_identity"]) >= 12
    return {
        "run": str(directory.relative_to(ROOT)),
        "checkpoint_sha256": hashes[checkpoint],
        "samples_finite": True,
        "generated_count": len(samples),
        "public_png_matches_metric_pixels": True,
        "public_report_matches": True,
    }


def main():
    torch.set_num_threads(2)
    complete = json.loads((ROOT / "outputs/learning_extension_complete.json").read_text())
    assert complete["steps"] == 3000
    weights = torch.load(
        ROOT / "outputs/coco2017_sd2_lora/checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert weights["step"] == complete["steps"]
    assert weights["base_identity"] == model_fingerprint(weights["config"]["base_model"])
    assert weights["manifest_sha256"] == digest(
        Path(weights["config"]["data_root"]) / "train.jsonl"
    )
    assert all(torch.isfinite(value).all() for value in weights["model"].values())
    train_count = (
        weights["step"]
        * weights["config"]["batch_size"]
        * weights["config"]["accumulation"]
        * len(weights["rank_states"])
    )
    hashes, rows = {}, []
    for name in complete["evaluations"]:
        rows.append(
            image_review(
                ROOT / "outputs" / name,
                ROOT / "assets/experiments" / experiment_group(name) / name,
                hashes,
                text=True,
            )
        )
    for left, right in [
        ("coco_text_base", "coco_text_lora"),
        ("coco_text_base_pairs", "coco_text_lora_pairs"),
    ]:
        a, b = ROOT / "outputs" / left, ROOT / "outputs" / right
        assert (a / "prompts.json").read_bytes() == (b / "prompts.json").read_bytes()
        ra, rb = [json.loads((folder / "metrics.json").read_text()) for folder in [a, b]]
        for key in [
            "seed",
            "cfg_scale",
            "steps",
            "generated_count",
            "batch_seeds",
            "base_identity",
        ]:
            assert ra[key] == rb[key], key
    for algorithm in ["edm", "flow"]:
        folder = ROOT / "outputs/mnist_training_seed_study"
        family = json.loads((folder / f"{algorithm}_summary.json").read_text())
        assert family["training_seeds"] == [42, 43, 44]
        assert family["steps_per_model"] == 20000
        digests = []
        for seed in [42, 43, 44]:
            name = f"{algorithm}_seed{seed}"
            done = json.loads((folder / name / "training_complete.json").read_text())
            assert done["step"] == 20000
            row = image_review(
                folder / name / "evaluation",
                ROOT / "assets/studies/mnist_training_seed_study" / name,
                hashes,
            )
            rows.append(row)
            digests.append(row["checkpoint_sha256"])
        assert len(set(digests)) == 3
    for calls in [1, 2, 4]:
        folder = ROOT / "outputs/consistency_refinement_study" / f"final_{calls}"
        row = image_review(
            folder, ROOT / "assets/studies/consistency_refinement_study" / folder.name, hashes
        )
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["generated_count"] == metrics["real_count"] == 5000
        assert metrics["reference_offset"] == 1000 and metrics["nfe_per_image"] == calls
        rows.append(row)
    gmm = json.loads((ROOT / "assets/studies/training_seed_study/metrics.json").read_text())
    assert len(gmm["results"]) == 6
    assert len({row["checkpoint_sha256"] for row in gmm["results"]}) == 6
    for row in gmm["results"]:
        assert row["nfe_per_image"] == 100 and row["reference_seed"] == 999
        assert digest(row["checkpoint"]) == row["checkpoint_sha256"]
    likelihood = ROOT / "assets/studies/expanded_likelihood_study"
    for case in [
        "mnist_elbo",
        "mnist_edm",
        "cifar_edm",
        "cifar_edm_grid200",
        "cifar_edm_probes4",
        "cifar_edm_noise124",
    ]:
        raw = json.loads((likelihood / case / "per_example_bpd.json").read_text())
        summary = json.loads((likelihood / case / "summary.json").read_text())
        assert len(raw["bpd"]) == summary["samples"] and np.isfinite(raw["bpd"]).all()
        assert np.isclose(np.mean(raw["bpd"]), summary["bpd"]["mean"])
        for batch in raw["batches"]:
            checkpoint = batch["checkpoint"]
            if checkpoint not in hashes:
                hashes[checkpoint] = digest(checkpoint)
            assert hashes[checkpoint] == batch["checkpoint_sha256"]
    data = Path(weights["config"]["data_root"])
    ids = [
        {json.loads(line)["image_id"] for line in (data / filename).read_text().splitlines()}
        for filename in ["train.jsonl", "val.jsonl"]
    ]
    assert ids[0].isdisjoint(ids[1])
    report = {
        "image_evaluations": rows,
        "gmm_training_models": 6,
        "likelihood_cases": 6,
        "full_base_identity_verified": True,
        "coco_train_val_image_ids_disjoint": True,
        "text_training_pairs_presented": train_count,
        "text_optimizer_steps": weights["step"],
        "base_identity_provenance": weights["base_identity_provenance"],
        "scope": "Actual hashes, finite tensors, counts, matched-caption/noise protocol and exact exported pixels. These checks do not establish subjective quality or eliminate external pretraining overlap.",
    }
    save_json(ROOT / "assets/reports/diagnostics/learning_integrity.json", report)
    print(f"Verified {len(rows)} image evaluations, six GMM models and six likelihood cases.")


if __name__ == "__main__":
    main()
