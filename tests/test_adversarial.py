"""Independent GAN objective and portable model invariants."""

import numpy as np
import pytest
import torch
from torch import nn

from diffusion_lab.adversarial import export, gradient_penalty, load_generator, network


def test_gradient_penalty_linear_critic_has_analytic_norm():
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 4.0]]))
    real, fake = torch.randn(16, 2), torch.randn(16, 2)
    penalty = gradient_penalty(layer, real, fake, torch.Generator().manual_seed(9))
    assert float(penalty) == pytest.approx(16.0)
    penalty.backward()
    np.testing.assert_allclose(layer.weight.grad.numpy(), [[4.8, 6.4]], rtol=1e-6)


def test_safe_generator_export_roundtrip_and_digest(tmp_path):
    model = network().eval()
    export(model, tmp_path, {"method": "least_squares", "seed": 7, "selected_step": 500})
    z = torch.randn(20, 2)
    torch.testing.assert_close(model(z), load_generator(tmp_path)(z), atol=0, rtol=0)
    path = tmp_path / "generator.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="modified"):
        load_generator(tmp_path)
