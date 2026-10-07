"""The clean DDPM posterior endpoint must stay exact at small float32 beta."""

import pytest
import torch

from diffusion_lab.improved_ddpm import posterior
from diffusion_lab.paths import DiscreteVP


@pytest.mark.parametrize("schedule", ["linear", "cosine", "logsnr", "exponential"])
def test_first_posterior_mean_is_clean_and_independent_of_noisy_state(schedule):
    path = DiscreteVP(1000, schedule)
    clean = torch.tensor([[1.3, -0.7], [0.1, 2.0]], requires_grad=True)
    noisy = torch.tensor([[8.0, -9.0], [-2.0, 5.0]], requires_grad=True)
    mean, _ = posterior(path, clean, noisy, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(mean, clean, atol=0, rtol=0)
    clean_gradient, noisy_gradient = torch.autograd.grad(mean.sum(), [clean, noisy])
    torch.testing.assert_close(clean_gradient, torch.ones_like(clean), atol=0, rtol=0)
    torch.testing.assert_close(noisy_gradient, torch.zeros_like(noisy), atol=0, rtol=0)
