"""Exact training state and deterministic, rank-sharded image permutations."""

import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data import Sampler


def capture_rng(device):
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def signature(config):
    config = dict(config)
    for key in ("steps", "run_dir", "resume", "hydra"):
        config.pop(key, None)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


class PermutationSampler(Sampler):
    """Drop the rank-incomplete tail; offset counts consumed examples, not prefetch."""

    def __init__(self, data, seed, rank=0, world=1, epoch=0, offset=0):
        self.size, self.seed, self.rank, self.world = len(data), seed, rank, world
        self.epoch, self.offset = epoch, offset

    def __iter__(self):
        order = torch.randperm(
            self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)
        )
        order = order[: self.size // self.world * self.world][self.rank :: self.world]
        return iter(order[self.offset :].tolist())

    def __len__(self):
        return self.size // self.world - self.offset
