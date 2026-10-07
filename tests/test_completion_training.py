"""Scientific invariants for independent target updates and clipped refinement."""

import copy

import pytest
import torch
from torch import nn

from diffusion_lab.consistency import ConsistencyModel, consistency_loss, consistency_sample
from diffusion_lab.text import TextUNet
from diffusion_lab.utils import EMA


class Constant(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, t, labels=None):
        return x * 0 + self.value


def test_consistency_weighted_loss_has_target_detached_and_finite_student_gradient():
    student = ConsistencyModel(Constant())
    target = copy.deepcopy(student)
    clean = torch.randn(8, 2)
    for weighting in ("uniform", "karras", "inverse_gap"):
        student.zero_grad()
        result = consistency_loss(student, target, clean, None, intervals=10, weighting=weighting)
        result.mean().backward()
        assert torch.isfinite(result).all()
        assert student.network.value.grad.abs() > 0
        assert target.network.value.grad is None


def test_clipping_occurs_before_refinement_noise_not_only_at_rendering():
    class Recorder(nn.Module):
        sigma_min = 0.002

        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x, t, labels=None):
            self.inputs.append(x.clone())
            return x * 0 + 3

    model = Recorder()
    consistency_sample(
        model,
        torch.zeros(2, 2),
        steps=2,
        sigmas=[80, 1],
        generator=torch.Generator().manual_seed(7),
        clip_denoised=True,
    )
    noise = torch.randn(2, 2, generator=torch.Generator().manual_seed(7))
    expected = 1 + (1 - model.sigma_min**2) ** 0.5 * noise
    torch.testing.assert_close(model.inputs[1], expected)


def test_distinct_target_and_sampling_emas_follow_their_own_decays():
    model = Constant()
    target, inference = EMA(model, 0.95), EMA(model, 0.9999)
    with torch.no_grad():
        model.value.fill_(4)
    target.update(model)
    inference.update(model)
    assert target.model.value.item() == pytest.approx(2.1)
    assert inference.model.value.item() == pytest.approx(2.0002)
    assert target.model.value.data_ptr() != inference.model.value.data_ptr()


def test_text_time_scale_preserves_shape_and_has_a_visible_embedding_effect():
    torch.manual_seed(7)
    model = TextUNet(
        text_dim=8,
        width=8,
        channel_mult=(1, 2),
        channels=4,
        blocks_per_level=1,
        time_embedding="positional",
        time_scale=1000,
    )
    embeddings = []
    handle = model.condition.register_forward_hook(lambda _m, _i, o: embeddings.append(o.detach()))
    output = model(torch.randn(2, 4, 8, 8), torch.tensor([0.001, 0.999]), torch.randn(2, 3, 8))
    handle.remove()
    assert output.shape == (2, 4, 8, 8)
    assert not torch.allclose(embeddings[0][0], embeddings[0][1])
