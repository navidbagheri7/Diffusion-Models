"""Audit completed extension weights, finite samples and rendered metric pixels."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


def review(directory):
    report = json.loads((directory / "metrics.json").read_text())
    digest = hashlib.sha256()
    with Path(report["checkpoint"]).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    state = torch.load(directory / "samples.pt", map_location="cpu", weights_only=True)
    samples = state["samples"]
    pixels = ((samples[:64].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    expected = make_grid(pixels, nrow=8).permute(1, 2, 0).numpy()
    actual = np.asarray(Image.open(directory / "samples.png").convert("RGB"))
    row = {
        "experiment": directory.name,
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_matches_report": digest.hexdigest() == report["checkpoint_sha256"],
        "sample_png_matches_metric_pixels": np.array_equal(expected, actual),
        "samples_finite": bool(torch.isfinite(samples).all()),
        "generated_count_matches": len(samples) == report["generated_count"],
        "training_steps": report["training_steps"],
        "experiment_validity": report.get("experiment_validity", "Measured evaluation"),
    }
    if not all(
        row[key]
        for key in [
            "checkpoint_matches_report",
            "sample_png_matches_metric_pixels",
            "samples_finite",
            "generated_count_matches",
        ]
    ):
        raise ValueError(f"Artifact integrity failed: {row}")
    return row


def main():
    root = Path(__file__).resolve().parents[1]
    names = [
        "cifar10_dit_30k",
        "celeba128_latent_unet_20k",
        "celeba128_latent_dit_30k",
        "cifar10_dit_final",
        "cifar10_dit_cfg_final",
        "celeba128_latent_unet_final",
        "celeba128_latent_dit_final",
        *[f"cifar10_cd{n}" for n in [1, 2, 4]],
        *[f"mnist_ict_smooth{n}" for n in [1, 2, 4]],
        *[f"mnist_ict_corrected{n}" for n in [1, 2, 4]],
        *[f"mnist_ict_corrected_5k{n}" for n in [1, 4]],
        *[f"mnist_ict{n}" for n in [1, 2, 4]],
        "mnist_progressive_teacher32",
        *[f"mnist_progressive_{n}_eval" for n in [16, 8, 4, 2, 1]],
    ]
    rows = [
        review(root / "outputs" / name)
        for name in names
        if (root / "outputs" / name / "samples.pt").exists()
    ]
    (root / "assets/reports/diagnostics/extension_integrity.json").write_text(
        json.dumps(
            {
                "scope": "File hashes, finite tensors and exact rendered metric pixels; not a subjective image-quality guarantee",
                "experiments": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
