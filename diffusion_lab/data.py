"""Analytic Gaussian mixtures and deterministic image splits."""

import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def balanced_image_indices(dataset, limit, class_label=None):
    """Round-robin examples instead of a class-sorted prefix."""
    groups = {}
    for index, (_, label) in enumerate(dataset.files):
        if class_label is None or label == class_label:
            groups.setdefault(label, []).append(index)
    result = []
    for row in range(max(map(len, groups.values()), default=0)):
        for entries in groups.values():
            if row < len(entries):
                result.append(entries[row])
                if len(result) == limit:
                    return result
    return result


class GaussianMixture:
    """Uniform isotropic mixture on a ring; each component is a class."""

    def __init__(self, components=8, radius=3.0, std=0.25, device="cpu"):
        if components < 1 or std <= 0 or radius < 0:
            raise ValueError("Invalid Gaussian mixture parameters")
        theta = torch.arange(components, device=device) * 2 * math.pi / components
        self.means = radius * torch.stack((theta.cos(), theta.sin()), dim=1)
        self.std = std
        self.components = components

    def sample(self, n, generator=None, labels=None):
        if labels is None:
            labels = torch.randint(
                self.components, (n,), device=self.means.device, generator=generator
            )
        noise = torch.randn(n, 2, device=self.means.device, generator=generator)
        return self.means[labels] + self.std * noise, labels

    def log_components(self, x, alpha=1.0, sigma=0.0):
        a = torch.as_tensor(alpha, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        s = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1, 1)
        variance = a.squeeze(-1).square() * self.std**2 + s.square()
        delta = x[:, None] - a * self.means[None]
        return -delta.square().sum(-1) / (2 * variance) - torch.log(2 * math.pi * variance)

    def log_prob(self, x, alpha=1.0, sigma=0.0):
        return self.log_components(x, alpha, sigma).logsumexp(-1) - math.log(self.components)

    def score(self, x, alpha, sigma, labels=None):
        a = alpha.reshape(-1, 1, 1)
        variance = alpha.square() * self.std**2 + sigma.square()
        scores = -(x[:, None] - a * self.means[None]) / variance[:, None, None]
        if labels is not None:
            return scores[torch.arange(len(x), device=x.device), labels]
        weights = self.log_components(x, alpha, sigma).softmax(-1)
        return (weights[..., None] * scores).sum(1)


class ImageDataset(Dataset):
    """Flat folders or class folders. Files are sorted before selecting a split.

    For FFHQ only, an explicit deterministic fraction forms a held-out split.
    Dataset files remain read-only. Class folders must agree across official splits.
    """

    def __init__(self, config, split="train"):
        root = Path(config.root).expanduser()
        image_root = root / "images"
        if not image_root.exists():
            image_root = root
        folder = image_root / split
        fallback = not folder.exists() and split == "validation"
        if fallback:
            folder = image_root / "train"
        if not folder.is_dir() and not (image_root / "train").is_dir():
            # User-supplied unsplit root: flat images or direct class folders.
            if config.holdout_fraction <= 0:
                raise ValueError("Unsplit image roots require an explicit held-out fraction")
            folder = image_root
        if not folder.is_dir():
            raise FileNotFoundError(f"Image split not found: {folder}")
        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        self.class_names = classes
        if classes:
            self.files = [
                (p, i)
                for i, name in enumerate(classes)
                for p in sorted((folder / name).iterdir())
                if p.suffix.lower() in extensions
            ]
        else:
            self.files = [
                (p, 0) for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions
            ]
        if config.holdout_fraction > 0:
            # A fixed shuffled index split avoids filename/source ordering bias.
            indices = torch.randperm(
                len(self.files), generator=torch.Generator().manual_seed(config.split_seed)
            )
            boundary = int(len(indices) * (1 - config.holdout_fraction))
            chosen = indices[:boundary] if split == "train" else indices[boundary:]
            self.files = [self.files[i] for i in chosen.tolist()]
        elif fallback:
            raise ValueError("No validation split: set data.holdout_fraction explicitly")
        limit = config.train_limit if split == "train" else config.validation_limit
        if limit:
            # Stratified round-robin prevents a bounded ImageNet run using only class 0.
            by_class = {}
            for entry in self.files:
                by_class.setdefault(entry[1], []).append(entry)
            ordered = []
            for index in range(max(map(len, by_class.values()), default=0)):
                for entries in by_class.values():
                    if index < len(entries):
                        ordered.append(entries[index])
                    if len(ordered) >= limit:
                        break
                if len(ordered) >= limit:
                    break
            self.files = ordered
        if not self.files:
            raise ValueError(f"No images selected from {folder}")
        self.channels = config.get("channels", 3)
        if self.channels not in {1, 3}:
            raise ValueError("Image data supports one or three channels")
        augmentation = (
            [transforms.RandomHorizontalFlip()]
            if split == "train" and config.get("horizontal_flip", False)
            else []
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.CenterCrop(config.image_size),
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * self.channels, (0.5,) * self.channels),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, label = self.files[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("L" if self.channels == 1 else "RGB"))
        return tensor, label
