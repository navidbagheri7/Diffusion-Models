"""Labeled online 2D distributions with controlled additive smoothing.

These are sampleable distributions; we do not pretend their exact scores are
known. Colors denote actual generative classes, not inferred cluster identities.
"""

import math
from typing import ClassVar

import torch


class ToyDistribution:
    COUNTS: ClassVar[dict[str, int]] = {
        "two_spirals": 2,
        "pinwheel": 5,
        "warped_checkerboard": 2,
        "wavy_rings": 4,
        "galaxy": 5,
        "crescents": 4,
        "deformed_lattice": 4,
        "sierpinski": 3,
        "map": 4,
        "rose": 5,
        "olympic_rings": 5,
    }

    def __init__(self, name="wavy_rings", std=0.06, device="cpu"):
        if name not in self.COUNTS or std <= 0:
            raise ValueError("Unknown toy dataset or nonpositive smoothing")
        self.name, self.std, self.device = name, std, torch.device(device)
        self.components = self.COUNTS[name]

    def sample(self, n, generator=None, labels=None):
        def uniform(*shape):
            return torch.rand(*shape, device=self.device, generator=generator)

        def normal(*shape):
            return torch.randn(*shape, device=self.device, generator=generator)

        if labels is None:
            labels = torch.randint(self.components, (n,), device=self.device, generator=generator)
        if len(labels) != n or (labels < 0).any() or (labels >= self.components).any():
            raise ValueError("Invalid requested toy class labels")
        u, v, label = uniform(n), uniform(n), labels.float()
        if self.name == "two_spirals":
            angle = 3 * math.pi * u + label * math.pi
            radius = 0.2 + 2.8 * u
            x = torch.stack([radius * angle.cos(), radius * angle.sin()], 1)
        elif self.name in {"pinwheel", "galaxy"}:
            radius = 0.3 + 2.7 * u.sqrt()
            angle = (
                label * 2 * math.pi / self.components
                + (radius if self.name == "pinwheel" else 1.7 * radius)
                + 0.06 * normal(n)
            )
            x = torch.stack([radius * angle.cos(), radius * angle.sin()], 1)
        elif self.name == "warped_checkerboard":
            row = torch.randint(4, (n,), device=self.device, generator=generator)
            side = torch.randint(2, (n,), device=self.device, generator=generator)
            col = 2 * side + (row + labels) % 2
            first, second = col.float() - 2 + u, row.float() - 2 + v
            x = torch.stack([first + 0.2 * (2 * second).sin(), second + 0.2 * (2 * first).sin()], 1)
        elif self.name == "wavy_rings":
            angle = u * 2 * math.pi
            radius = 0.6 + 0.75 * label + 0.12 * (7 * angle + label).sin()
            x = torch.stack([radius * angle.cos(), radius * angle.sin()], 1)
        elif self.name == "crescents":
            angle = math.pi * u
            x = torch.stack(
                [
                    2 * angle.cos() + 0.4 * (label % 2),
                    (-1) ** label * angle.sin() + (label - 1.5) * 0.8,
                ],
                1,
            )
        elif self.name == "deformed_lattice":
            grid = torch.randint(7, (n, 2), device=self.device, generator=generator).float() - 3
            grid[:, 0] = (grid[:, 0] + label / 4) * 0.8
            x = grid + 0.2 * grid.flip(1).sin()
        elif self.name == "sierpinski":
            vertices = torch.tensor(
                [[0, 2], [-1.8, -1.1], [1.8, -1.1]], device=self.device, dtype=u.dtype
            )
            x = torch.zeros(n, 2, device=self.device)
            for _ in range(18):
                selection = torch.randint(3, (n,), device=self.device, generator=generator)
                x = (x + vertices[selection]) / 2
            # Final branch is the class, producing three recognizable subtriangles.
            x = (x + vertices[labels]) / 2
        elif self.name == "map":
            angle = 2 * math.pi * u
            radius = 0.6 + 0.2 * normal(n)
            center = torch.tensor(
                [[-2, 1.5], [2, 1.5], [-1.7, -1.5], [1.8, -1.5]], device=self.device
            )[labels]
            x = center + torch.stack([radius * angle.cos(), 0.8 * radius * angle.sin()], 1)
            corridor = v < 0.15
            x[corridor] = torch.stack(
                [4 * u[corridor] - 2, 0.35 * (4 * math.pi * u[corridor]).sin()], 1
            )
        elif self.name == "rose":
            angle = (label + u - 0.5) * 2 * math.pi / 5
            radius = 3 * (2.5 * angle).cos().abs()
            x = torch.stack([radius * angle.cos(), radius * angle.sin()], 1)
        else:
            centers = torch.tensor(
                [[-2, 0.6], [0, 0.6], [2, 0.6], [-1, -0.6], [1, -0.6]], device=self.device
            )
            angle = 2 * math.pi * u
            x = centers[labels] + torch.stack([angle.cos(), angle.sin()], 1)
        return x + self.std * normal(n, 2), labels


def toy_metrics(samples, reference):
    from .metrics import empirical_wasserstein, mmd_squared, sliced_wasserstein

    return {
        "mmd_squared_unbiased": mmd_squared(samples, reference),
        "sliced_w2": sliced_wasserstein(samples, reference),
        "empirical_w2_subsample": empirical_wasserstein(samples[:512], reference[:512]),
        "sample_count": len(samples),
        "exact_density_available": False,
    }
