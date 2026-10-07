"""End-to-end resume oracle and independent codec/data boundary regressions."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from diffusion_lab.data import ImageDataset, balanced_image_indices
from diffusion_lab.text_train import codec_geometry


@pytest.mark.parametrize("process", ["ddpm", "flow"])
def test_exact_gmm_resume_matches_continuous_updates(tmp_path, process):
    def run(folder, steps, checkpoint=None):
        cmd = [
            sys.executable,
            "-m",
            "diffusion_lab.train",
            "device=cpu",
            "cpu_threads=1",
            "workers=0",
            f"process={process}",
            "sampler=ode" if process == "flow" else "sampler=ddim",
            f"run_dir={folder}",
            f"steps={steps}",
            "batch_size=16",
            "model.width=16",
            "model.depth=2",
            "validation_samples=16",
            "evaluation_samples=16",
            "sampler.steps=2",
            "validate_every=4",
            "log_every=4",
        ]
        if checkpoint:
            cmd.append(f"resume={checkpoint}")
        subprocess.run(cmd, check=True, capture_output=True, timeout=90)
        return torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)

    continuous = run(tmp_path / "continuous", 12)
    run(tmp_path / "split", 4)
    resumed = run(tmp_path / "split", 12, tmp_path / "split/checkpoint.pt")
    for group in ("model", "ema"):
        for key in continuous[group]:
            torch.testing.assert_close(continuous[group][key], resumed[group][key], rtol=0, atol=0)
    assert torch.equal(
        continuous["rank_states"][0]["rng"]["torch"], resumed["rank_states"][0]["rng"]["torch"]
    )


def test_balanced_reference_offset_preserves_declared_disjoint_subsets(tmp_path):
    for name in ["alpha", "beta"]:
        folder = tmp_path / "validation" / name
        folder.mkdir(parents=True)
        for i in range(6):
            Image.new("RGB", (8, 8), (i * 30, 50, 0)).save(folder / f"{i}.png")
    data = ImageDataset(
        OmegaConf.create(
            {
                "root": str(tmp_path),
                "channels": 3,
                "image_size": 8,
                "holdout_fraction": 0,
                "validation_limit": 0,
            }
        ),
        "validation",
    )
    first = balanced_image_indices(data, 4)
    later = balanced_image_indices(data, 8)[4:]
    assert set(first).isdisjoint(later)
    assert [data[i][1] for i in later] == [0, 1, 0, 1]


def test_codec_rejects_unsupported_channel_standard_deviation():
    codec = SimpleNamespace(
        config=SimpleNamespace(
            block_out_channels=[8, 16],
            scaling_factor=0.5,
            shift_factor=None,
            latents_mean=None,
            latents_std=[1.0, 2.0],
        )
    )
    with pytest.raises(ValueError, match="normalization"):
        codec_geometry(codec, 16)
