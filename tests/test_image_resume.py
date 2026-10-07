"""End-to-end image resume across epochs and prefetched deterministic workers."""

import subprocess
import sys

import pytest
import torch
from PIL import Image


@pytest.mark.parametrize("workers", [0, 2])
def test_general_image_resume_preserves_augmentation_and_data_order(tmp_path, workers):
    root = tmp_path / "data"
    for split in ["train", "validation"]:
        folder = root / split
        folder.mkdir(parents=True)
        for i in range(10):
            image = Image.new("RGB", (8, 8), (i * 20, 50, 70))
            for j in range(4):
                image.putpixel((j, i % 8), (240, 10, 30))
            image.save(folder / f"{i:02}.png")

    def run(name, steps, resume=False):
        directory = tmp_path / name
        cmd = [
            sys.executable,
            "-m",
            "diffusion_lab.train",
            "data=custom",
            f"data.root={root}",
            "data.image_size=8",
            "data.holdout_fraction=0",
            "data.horizontal_flip=true",
            "model=unet",
            "model.width=8",
            "conditional=false",
            "process=flow",
            "sampler=ode",
            "sampler.steps=2",
            "device=cpu",
            "cpu_threads=1",
            f"workers={workers}",
            "batch_size=2",
            "validation_samples=4",
            "evaluation_samples=4",
            "image_metrics=false",
            "validate_every=4",
            "log_every=4",
            f"steps={steps}",
            f"run_dir={directory}",
        ]
        if resume:
            cmd.append(f"resume={directory}/checkpoint.pt")
        subprocess.run(cmd, check=True, capture_output=True, timeout=90)
        return torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)

    full = run("full", 12)
    run("split", 4)
    split = run("split", 12, True)
    for name in ["model", "ema"]:
        for key in full[name]:
            torch.testing.assert_close(full[name][key], split[name][key], rtol=0, atol=0)
    assert full["rank_states"][0]["epoch"] == split["rank_states"][0]["epoch"] == 2
