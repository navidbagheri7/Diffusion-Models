"""Held-out splitting and evaluation checks independent of external datasets."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from PIL import Image

from diffusion_lab.data import ImageDataset, balanced_image_indices
from diffusion_lab.metrics import mmd_squared
from diffusion_lab.paths import DiscreteVP
from diffusion_lab.sampling import sample, sample_batched


def test_flat_holdout_split_is_reproducible_disjoint_and_normalized(tmp_path):
    folder = tmp_path / "images/train"
    folder.mkdir(parents=True)
    for i in range(20):
        Image.new("RGB", (20, 20), (i * 10, 0, 255)).save(folder / f"{i:02d}.png")
    cfg = OmegaConf.create(
        {
            "root": str(tmp_path),
            "image_size": 16,
            "holdout_fraction": 0.25,
            "split_seed": 123,
            "train_limit": 0,
            "validation_limit": 0,
        }
    )
    train, validation = ImageDataset(cfg), ImageDataset(cfg, "validation")
    assert len(train) == 15 and len(validation) == 5
    assert {p for p, _ in train.files}.isdisjoint(p for p, _ in validation.files)
    assert validation.files == ImageDataset(cfg, "validation").files
    x, label = train[0]
    assert x.shape == (3, 16, 16) and label == 0
    assert x.min() >= -1 and x.max() <= 1


def test_bounded_class_dataset_covers_multiple_classes(tmp_path):
    for label in range(3):
        folder = tmp_path / f"images/train/class_{label:04d}"
        folder.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (16, 16)).save(folder / f"{i}.png")
    cfg = OmegaConf.create(
        {
            "root": str(tmp_path),
            "image_size": 16,
            "holdout_fraction": 0.0,
            "split_seed": 123,
            "train_limit": 6,
            "validation_limit": 0,
        }
    )
    dataset = ImageDataset(cfg)
    assert [label for _, label in dataset.files] == [0, 1, 2, 0, 1, 2]


def test_unsplit_custom_grayscale_root_and_balanced_selection(tmp_path):
    for label in range(3):
        folder = tmp_path / f"class_{label}"
        folder.mkdir()
        for index in range(10):
            Image.new("L", (16, 16), color=index * 20).save(folder / f"{index}.png")
    config = OmegaConf.create(
        {
            "root": str(tmp_path),
            "channels": 1,
            "image_size": 16,
            "holdout_fraction": 0.2,
            "split_seed": 1,
            "train_limit": 0,
            "validation_limit": 0,
        }
    )
    train, validation = ImageDataset(config), ImageDataset(config, "validation")
    assert len(train) == 24 and len(validation) == 6
    assert {p for p, _ in train.files}.isdisjoint(p for p, _ in validation.files)
    assert train[0][0].shape == (1, 16, 16)
    indices = balanced_image_indices(train, 6)
    assert sorted(train.files[index][1] for index in indices) == [0, 0, 1, 1, 2, 2]


def test_chunked_mmd_matches_dense_formula():
    torch.manual_seed(1)
    x, y = torch.randn(19, 2), torch.randn(23, 2)
    bandwidths = (0.5, 1.0)

    def kernel(a, b):
        return torch.stack(
            [torch.exp(-torch.cdist(a, b).square() / (2 * h * h)) for h in bandwidths]
        ).mean(0)

    xx, yy, xy = kernel(x, x), kernel(y, y), kernel(x, y)
    expected = (
        (xx.sum() - xx.diag().sum()) / (19 * 18)
        + (yy.sum() - yy.diag().sum()) / (23 * 22)
        - 2 * xy.mean()
    ).item()
    assert abs(mmd_squared(x, y, bandwidths, block_size=7) - expected) < 1e-6


def test_deterministic_batched_sampling_matches_full_sampling():
    path = DiscreteVP(40)
    process = SimpleNamespace(kind="ddpm", path=path)
    sampler = SimpleNamespace(method="ddim", steps=10, eta=0.0, clip_x0=False)

    def model(x, t, labels=None):
        return x * 0.2

    x = torch.randn(11, 3, 8, 8)
    full, _ = sample(model, process, sampler, x, "x0")
    batched, _ = sample_batched(model, process, sampler, x, "x0", batch_size=4)
    torch.testing.assert_close(full, batched)
