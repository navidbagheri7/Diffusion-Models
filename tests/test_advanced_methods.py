"""Scientific invariants for advanced methods, independent of trained weights."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from diffusion_lab.consistency import ConsistencyModel, consistency_loss, consistency_sample
from diffusion_lab.dpm_solver import dpm_sample, interpolated_coefficients, logsnr_grid
from diffusion_lab.dsbm import projection_target
from diffusion_lab.edm import EDMPreconditioner, edm_loss, edm_sample, karras_sigmas
from diffusion_lab.image_evaluate import slerp
from diffusion_lab.image_models import ImageUNet
from diffusion_lab.improved_ddpm import (
    LossSecondMomentSampler,
    improved_loss,
    learned_log_variance,
    posterior,
)
from diffusion_lab.models import make_model
from diffusion_lab.paths import DiscreteVP, GaussianPath
from diffusion_lab.toy_data import ToyDistribution
from diffusion_lab.transport import conditional_pairing, exact_pairing


class ZeroNetwork(nn.Module):
    def forward(self, x, t, labels=None):
        return torch.zeros_like(x)


@pytest.mark.parametrize("kind", ["sub_vp", "vp_logsnr", "vp_exponential"])
def test_new_sde_moment_equations(kind):
    path = GaussianPath(kind)
    t = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64, requires_grad=True)
    a, s = path.coefficients(t)
    da = torch.autograd.grad(a.sum(), t, retain_graph=True)[0]
    dvar = torch.autograd.grad(s.square().sum(), t)[0]
    f, g = path.sde(torch.ones(3, 1, dtype=torch.float64), t)
    rate = f[:, 0]
    torch.testing.assert_close(da, rate * a)
    torch.testing.assert_close(dvar, 2 * rate * s.square() + g.square())


@pytest.mark.parametrize("schedule", ["logsnr", "exponential"])
def test_discrete_new_schedule(schedule):
    path = DiscreteVP(1000, schedule)
    assert (path.betas > 0).all() and (path.betas < 1).all()
    assert (path.alpha_bar[1:] < path.alpha_bar[:-1]).all()
    assert path.alpha_bar[-1] < 1e-4


def test_edm_preconditioning_matches_gaussian_conditional_mean():
    model = EDMPreconditioner(ZeroNetwork(), 0.5)
    x, sigma = torch.randn(4, 2), torch.tensor([0.01, 0.1, 1, 10])
    torch.testing.assert_close(model(x, sigma), x * (0.25 / (sigma.square() + 0.25))[:, None])
    assert torch.isfinite(edm_loss(model, x)).all()


def test_karras_sampler_boundary_and_measured_calls():
    grid = karras_sigmas(18)
    assert grid[0] == 80 and grid[-1] == 0
    assert (grid[1:] < grid[:-1]).all()
    calls = []

    class Denoiser:
        def __call__(self, x, t, labels=None):
            calls.append(1)
            return torch.ones_like(x)

    x, history, nfe = edm_sample(Denoiser(), torch.randn(8, 2), steps=18)
    torch.testing.assert_close(x, torch.ones_like(x), atol=1e-5, rtol=1e-5)
    assert nfe == len(calls) == 35 and history.shape[0] == 19


def test_consistency_exact_boundary_and_sampling_calls():
    model = ConsistencyModel(ZeroNetwork())
    clean = torch.randn(4, 2)
    torch.testing.assert_close(model(clean, torch.full((4,), model.sigma_min)), clean)
    values = consistency_loss(model, model, clean, None, intervals=5)
    assert torch.isfinite(values).all()
    _, _, nfe = consistency_sample(model, clean, steps=2)
    assert nfe == 2


def test_consistency_refinements_avoid_identity_boundary():
    calls = []

    class Recorder:
        sigma_min = 0.002

        def __call__(self, x, sigma, labels=None):
            calls.append(sigma[0].item())
            return x * 0

    consistency_sample(Recorder(), torch.randn(4, 2), steps=4)
    assert len(calls) == 4 and min(calls) > Recorder.sigma_min


@pytest.mark.parametrize("algorithm", ["dpm_1", "dpmpp_1", "dpm_2m", "dpmpp_2m"])
def test_dpm_exact_constant_clean_endpoint(algorithm):
    path = DiscreteVP(1000, "linear")
    clean = torch.randn(4, 2)

    class Oracle:
        def __call__(self, x, t, labels=None):
            return clean

    output, _, nfe = dpm_sample(Oracle(), path, torch.randn_like(clean), "x0", 20, algorithm)
    torch.testing.assert_close(output, clean, atol=1e-4, rtol=1e-4)
    assert nfe == 21
    times = logsnr_grid(path, 20, "cpu")
    a, s = interpolated_coefficients(path, times)
    increments = (a / s).log().diff()
    torch.testing.assert_close(
        increments, increments.mean().expand_as(increments), atol=1e-4, rtol=1e-4
    )


def test_improved_variance_and_hybrid_mean_stop_gradient():
    path = DiscreteVP(50)
    clean, noise = torch.randn(4, 1, 4, 4).clamp(-1, 1), torch.randn(4, 1, 4, 4)
    indices = torch.tensor([0, 1, 10, 49])
    a = path.alpha_bar[indices]
    noisy = a[:, None, None, None].sqrt() * clean + (1 - a[:, None, None, None]).sqrt() * noise
    output = torch.cat([noise, torch.zeros_like(noise)], 1).requires_grad_()
    loss = improved_loss(path, clean, noisy, (indices + 1) / path.steps, noise, output).mean()
    loss.backward()
    assert torch.isfinite(loss)
    torch.testing.assert_close(output.grad[:, :1], torch.zeros_like(noise), atol=1e-6, rtol=0)
    assert output.grad[:, 1:].abs().sum() > 0
    _, lower = posterior(path, clean, noisy, indices)
    values = torch.ones_like(noise)
    upper = path.betas[indices].log()
    torch.testing.assert_close(
        learned_log_variance(values, lower, upper), upper[:, None, None, None].expand_as(noise)
    )


@pytest.mark.parametrize("algorithm", ["dpm_2m", "dpmpp_2m"])
def test_dpm_converges_to_gaussian_probability_flow_without_final_oracle(algorithm):
    path = GaussianPath("vp_linear")
    variance_data = 0.25

    class GaussianPosterior:
        def __call__(self, x, t, labels=None):
            a, s = path.coefficients(t)
            return (a * variance_data / (a.square() * variance_data + s.square()))[:, None] * x

    endpoint_a, endpoint_s = path.coefficients(torch.tensor([0.999, 0.001]))
    variance = endpoint_a.square() * variance_data + endpoint_s.square()
    exact = (variance[1] / variance[0]).sqrt()
    errors = []
    for steps in [16, 32, 64]:
        result, _, _ = dpm_sample(
            GaussianPosterior(), path, torch.ones(1, 2), "x0", steps, algorithm, denoise_final=False
        )
        errors.append((result[0, 0] - exact).abs().item())
    assert errors[1] < errors[0] and errors[2] < errors[1] * 0.6


def test_loss_second_moment_unbiased_importance():
    sampler = LossSecondMomentSampler(3, history=2)
    sampler.update(torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([1, 1, 2, 2, 4, 4]))
    p = sampler.probabilities()
    assert p[2] > p[1] > p[0] > 0
    loss = torch.tensor([1, 2, 4], dtype=torch.float64)
    torch.testing.assert_close((p * loss / (3 * p)).sum(), loss.mean())


def test_dsbm_reverse_clock_target():
    source, target = torch.randn(4, 2), torch.randn(4, 2)
    t, noise = torch.full((4,), 0.3), torch.randn(4, 2)
    x, reverse = projection_target(source, target, t, 0.5, "backward", noise)
    torch.testing.assert_close(reverse, (source - x) / 0.7)
    x_forward, forward = projection_target(target, source, t, 0.5, "forward", noise)
    torch.testing.assert_close(x_forward, x)
    torch.testing.assert_close(forward, reverse)


@pytest.mark.parametrize("name", list(ToyDistribution.COUNTS))
def test_labeled_toys_finite_reproducible(name):
    data = ToyDistribution(name)
    labels = torch.arange(128) % data.components
    first, returned = data.sample(128, torch.Generator().manual_seed(7), labels)
    second, _ = data.sample(128, torch.Generator().manual_seed(7), labels)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(returned, labels)
    assert first.shape == (128, 2) and torch.isfinite(first).all()


def test_image_architecture_channels_and_hydra_custom_model():
    model = ImageUNet(
        width=16, channels=1, out_channels=2, classes=10, channel_mult=(1, 2, 2), blocks_per_level=1
    )
    x = torch.randn(2, 1, 32, 32)
    output = model(x, torch.rand(2), torch.tensor([0, 10]))
    assert output.shape == (2, 2, 32, 32)
    output.sum().backward()
    custom = make_model(
        OmegaConf.create(
            {
                "_target_": "diffusion_lab.image_models.ImageUNet",
                "width": 16,
                "channels": 1,
                "channel_mult": [1, 2],
                "blocks_per_level": 1,
            }
        ),
        classes=10,
    )
    assert custom(x, torch.rand(2)).shape == x.shape


def test_spherical_interpolation_endpoints():
    first, second = torch.randn(1, 4, 4), torch.randn(1, 4, 4)
    values = slerp(first, second, torch.linspace(0, 1, 9))
    torch.testing.assert_close(values[0], first)
    torch.testing.assert_close(values[-1], second)
    torch.testing.assert_close(values[4].norm(), (first.norm() + second.norm()) / 2)


def test_conditional_ot_preserves_each_classes_noise_marginal():
    data = torch.tensor([[-10.0], [-9.0], [9.0], [10.0]])
    noise = torch.tensor([[-3.0], [3.0], [-2.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    paired = conditional_pairing(data, noise, labels)
    for label in labels.unique():
        selected = labels == label
        torch.testing.assert_close(paired[selected].sort(0).values, noise[selected].sort(0).values)
    global_pairing = exact_pairing(data, noise)
    assert global_pairing[labels == 0].mean() < -2
    assert global_pairing[labels == 1].mean() > 2


def test_conditional_flow_perturb_uses_original_class_groups():
    from diffusion_lab.train import perturb

    process = OmegaConf.create(
        {"kind": "flow", "coupling": "ot", "flow_path": "linear", "prediction": "velocity"}
    )
    data = torch.tensor([[-10.0], [-9.0], [9.0], [10.0]])
    labels = torch.tensor([0, 0, 1, 1])
    generator = torch.Generator().manual_seed(7)
    original_noise = torch.randn(data.shape, generator=torch.Generator().manual_seed(7))
    _, _, velocity, _ = perturb(data, labels, process, generator, conditional=True)
    paired_noise = data - velocity
    for label in labels.unique():
        selected = labels == label
        torch.testing.assert_close(
            paired_noise[selected].sort(0).values, original_noise[selected].sort(0).values
        )
