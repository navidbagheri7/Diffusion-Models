"""Portable image/caption JSONL datasets with image-level split boundaries."""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import pil_to_tensor


class CaptionDataset(Dataset):
    """Each JSONL row has image, captions (list[str]), and image_id.

    COCO val is divided by sorted image ID: first 1,000 for model selection,
    remaining 4,000 for final evaluation. Split by image, never by caption.
    Horizontal flips are deliberately disabled to preserve spatial language.
    """

    def __init__(self, root, split="train", resolution=512, limit=0, manifest=None):
        self.root = Path(root).resolve()
        path = (
            Path(manifest)
            if manifest
            else self.root / ("train.jsonl" if split == "train" else "val.jsonl")
        )
        content = path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(content).hexdigest()
        self.rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
        if len({row["image_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("Caption manifest has duplicate image IDs")
        self.rows.sort(key=lambda row: row["image_id"])
        if manifest is None:
            if split == "validation":
                self.rows = self.rows[:1000]
            elif split == "evaluation":
                self.rows = self.rows[1000:]
            elif split != "train":
                raise ValueError("Use train, validation or evaluation")
        if limit:
            self.rows = self.rows[:limit]
        if resolution < 8 or resolution % 8 or not self.rows:
            raise ValueError("Need nonempty paired data and resolution divisible by eight")
        self.resolution = resolution
        for row in self.rows:
            image = (self.root / row["image"]).resolve()
            if not image.is_relative_to(self.root):
                raise ValueError("Image path escapes dataset root")
            if not row["captions"] or any(
                not isinstance(c, str) or not c.strip() for c in row["captions"]
            ):
                raise ValueError("Each image requires nonempty text captions")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            image = ImageOps.fit(
                image.convert("RGB"),
                (self.resolution, self.resolution),
                method=Image.Resampling.BICUBIC,
            )
            pixels = pil_to_tensor(image).float().div(127.5).sub(1)
        return pixels, row["captions"], row["image_id"]


def caption_collate(rows):
    return torch.stack([row[0] for row in rows]), [row[1] for row in rows], [row[2] for row in rows]


class EpochSampler(Sampler):
    """A deterministic rank-sharded permutation; worker prefetch cannot move resume state."""

    def __init__(self, data, seed, rank=0, world=1):
        self.size, self.seed, self.rank, self.world, self.epoch = len(data), seed, rank, world, 0

    def __len__(self):
        return self.size // self.world

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator)[: len(self) * self.world]
        return iter(order[self.rank :: self.world].tolist())
