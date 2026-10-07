"""Independent scientific invariants for the second research expansion."""

import hashlib
import json
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from diffusion_lab.consistency import ConsistencyModel, improved_consistency_loss, improved_scales
from diffusion_lab.data import GaussianMixture
from diffusion_lab.distribution_metrics import manifold_metrics
from diffusion_lab.dit import DiT, DiTBlock, position_embedding
from diffusion_lab.improved_ddpm import stable_discretized_log_likelihood
from diffusion_lab.interpolants import GaussianInterpolantOracle, StochasticInterpolant
from diffusion_lab.latents import (
    LatentDataset,
    denormalize_latents,
    normalize_latents,
    posterior_samples,
)
from diffusion_lab.likelihood import (
    ddpm_elbo,
    field_and_divergence,
    gaussian_log_density,
    ode_log_density,
)
from diffusion_lab.models import TimeEmbedding
from diffusion_lab.paths import DiscreteVP
from diffusion_lab.progressive import ddim_destination, distillation_target, progressive_grid


@pytest.mark.parametrize("direction", [1, -1])
def test_ode_density_sign_and_affine_gaussian(direction):
    x = torch.tensor([[0.4, -1.2], [1.0, 0.3]], dtype=torch.float64)
    density, metadata = ode_log_density(
        lambda x, t: 0.7 * x,
        x,
        torch.linspace(0, direction, 21, dtype=x.dtype),
        gaussian_log_density,
    )
    torch.testing.assert_close(
        density, gaussian_log_density(x, math.exp(-direction * 0.7)), atol=2e-7, rtol=2e-7
    )
    assert metadata["field_calls"] == 80


def test_divergence_orthogonal_probes_and_constant_field():
    x = torch.randn(3, 2, dtype=torch.float64)
    matrix = x.new_tensor([[2.0, 5.0], [-3.0, 4.0]])
    probes = x.new_tensor([[1.0, 1.0], [1.0, -1.0]])[:, None].expand(2, 3, 2)
    _, trace = field_and_divergence(lambda x, t: x @ matrix, x, torch.ones(3), probes)
    torch.testing.assert_close(trace, x.new_full((3,), 6.0))
    _, trace = field_and_divergence(lambda x, t: torch.ones_like(x), x, torch.ones(3))
    assert torch.equal(trace, torch.zeros_like(trace))


def test_full_elbo_stationary_gaussian_expected_nll():
    path = DiscreteVP(8, "cosine")
    generator = torch.Generator().manual_seed(123)
    data = torch.randn(20000, 2, generator=generator)

    class GaussianReverse(nn.Module):
        def forward(self, x, t, labels=None):
            _, s = path.coefficients(t)
            return torch.cat([s[:, None] * x, torch.ones_like(x)], 1)

    result = ddpm_elbo(
        GaussianReverse(), path, data, learned_variance=True, quantized=False, generator=generator
    )
    assert (
        abs(
            result["negative_elbo_bpd"].mean() + gaussian_log_density(data).mean() / 2 / math.log(2)
        )
        < 0.025
    )
    torch.testing.assert_close(
        result["negative_elbo_bpd"],
        result["terminal_prior_kl_bpd"] + result["decoder_bpd"] + result["transition_kl_bpd"],
    )
    assert result["network_calls"] == 8 and not result["is_exact_nll"]


def test_discrete_decoder_tail_probability_without_floor():
    from scipy.integrate import quad
    from scipy.stats import norm

    actual = stable_discretized_log_likelihood(
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([math.log(0.05**2)], dtype=torch.float64),
    )
    probability, _ = quad(norm.pdf, (0.5 - 1 / 255) / 0.05, (0.5 + 1 / 255) / 0.05, epsabs=1e-30)
    assert abs(float(actual) - math.log(probability)) < 1e-10
    assert float(actual) < math.log(1e-12)


def test_saved_image_matches_metric_quantization_at_half_integer(tmp_path):
    from PIL import Image

    from diffusion_lab.metrics import image_pixels
    from diffusion_lab.visualize import save_rendered_image

    images = torch.tensor([127.5, 128.5, 129.5, 130.5]).reshape(1, 1, 2, 2) / 127.5 - 1
    path = tmp_path / "ties.png"
    save_rendered_image(images, path)
    actual = torch.from_numpy(np.asarray(Image.open(path)).copy()).permute(2, 0, 1)
    torch.testing.assert_close(actual, image_pixels(images)[0].expand(3, -1, -1))


@pytest.mark.parametrize("path", ["linear", "trigonometric", "cubic"])
def test_interpolant_endpoints_and_density_equations(path):
    interpolant = StochasticInterpolant(path, 0.5)
    a, b, g, *_ = interpolant.coefficients(torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(a, torch.tensor([1.0, 0.0], dtype=a.dtype), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(b, torch.tensor([0.0, 1.0], dtype=a.dtype))
    assert torch.equal(g, torch.zeros_like(g))
    oracle = GaussianInterpolantOracle(
        GaussianMixture(1, 0, 1), GaussianMixture(3, 2, 0.4), interpolant
    )
    x = torch.tensor([[0.2, -0.4], [1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.3, 0.6], dtype=x.dtype, requires_grad=True)
    density = oracle.log_density(x, t).exp()
    dt = torch.autograd.grad(density.sum(), t, retain_graph=True, create_graph=True)[0]
    v, score = oracle.fields(x, t)
    grad = torch.autograd.grad(density.sum(), x, retain_graph=True, create_graph=True)[0]
    torch.testing.assert_close(grad, density[:, None] * score, atol=1e-10, rtol=1e-8)
    div = 0
    lap = 0
    div_drift = 0
    drift = oracle.drift(x, t, strength=0.7)
    for axis in range(2):
        div = (
            div
            + torch.autograd.grad((density * v[:, axis]).sum(), x, retain_graph=True)[0][:, axis]
        )
        div_drift = (
            div_drift
            + torch.autograd.grad((density * drift[:, axis]).sum(), x, retain_graph=True)[0][
                :, axis
            ]
        )
        lap = lap + torch.autograd.grad(grad[:, axis].sum(), x, retain_graph=True)[0][:, axis]
    torch.testing.assert_close(dt, -div, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(
        dt,
        -div_drift + 0.7 * interpolant.coefficients(t)[2].square() * lap,
        atol=1e-10,
        rtol=1e-8,
    )


def test_custom_nonlinear_interpolant_time_derivative():
    from diffusion_lab.paths import expand

    def base(source, target, t):
        return (
            expand(1 - t, source) * source
            + expand(t, target) * target
            + expand(t * (1 - t), source) * (source + target).sin()
        )

    source, target = torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    process = StochasticInterpolant(base_fn=base)
    t = torch.linspace(0.1, 0.9, 5, dtype=source.dtype)
    _, velocity, noise = process.targets(source, target, t)
    dgamma = process.coefficients(t)[-1]
    expected = target - source + expand(1 - 2 * t, source) * (source + target).sin()
    torch.testing.assert_close(velocity - expand(dgamma, source) * noise, expected)
    for time, endpoint in [(0, source), (1, target)]:
        values, _, _ = process.targets(source, target, torch.full_like(t, time))
        torch.testing.assert_close(values, endpoint)


def test_latent_whitening_and_posterior_variance():
    norm = {"mean": [2.0, -1.0], "std": [3.0, 0.5]}
    x = torch.randn(4, 2, 3, 3)
    torch.testing.assert_close(denormalize_latents(normalize_latents(x, norm), norm), x)
    moments = torch.tensor([2.0, -1.0, 3.0, 0.5]).reshape(1, 1, 4, 1, 1).expand(30000, 1, 4, 1, 1)
    z = posterior_samples(moments, norm, torch.Generator().manual_seed(7))
    assert z.mean().abs() < 0.015 and abs(float(z.var()) - 1) < 0.025


def test_custom_interpolant_noise_schedule_derivative_and_endpoints():
    process = StochasticInterpolant(noise_std_fn=lambda t: 2 * t * (1 - t))
    t = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0], dtype=torch.float64)
    gamma, derivative = process.coefficients(t)[2], process.coefficients(t)[-1]
    torch.testing.assert_close(gamma, 2 * t * (1 - t))
    torch.testing.assert_close(derivative, 2 - 4 * t)
    assert gamma[0] == gamma[-1] == 0
    sine = StochasticInterpolant(noise_std_fn=lambda t: torch.sin(math.pi * t))
    assert sine.coefficients(t)[2][-1] == 0
    assert sine.coefficients(t.float())[2][-1] == 0
    with pytest.raises(ValueError, match="vanish at endpoints"):
        StochasticInterpolant(noise_std_fn=lambda t: torch.ones_like(t)).coefficients(t)


def test_latent_cache_integrity_and_rng_continuation(tmp_path):
    moments = np.ones((4, 2, 2, 2, 2), dtype=np.float16)
    np.save(tmp_path / "train_moments.npy", moments)
    np.save(tmp_path / "train_labels.npy", np.array([0, 1, 0, 1], dtype=np.int64))
    manifest = {
        "format": "latent_cache_v1",
        "latent_channels": 1,
        "latent_size": 2,
        "class_names": ["a", "b"],
        "splits": {
            "train": {
                "count": 4,
                "variants": 2,
                "moments_sha256": hashlib.sha256(
                    (tmp_path / "train_moments.npy").read_bytes()
                ).hexdigest(),
                "labels_sha256": hashlib.sha256(
                    (tmp_path / "train_labels.npy").read_bytes()
                ).hexdigest(),
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cfg = OmegaConf.create(
        {"root": str(tmp_path), "channels": 1, "image_size": 2, "train_limit": 0}
    )
    data = LatentDataset(cfg)
    batch = torch.stack([data[i][0] for i in range(4)])
    norm = {"mean": [1.0], "std": [1.0]}
    generator = torch.Generator().manual_seed(3)
    posterior_samples(batch, norm, generator)
    resume_state = generator.get_state()
    expected = posterior_samples(batch, norm, generator)
    restored = torch.Generator().set_state(resume_state)
    assert torch.equal(expected, posterior_samples(batch, norm, restored))
    moments[0] = 2
    np.save(tmp_path / "train_moments.npy", moments)
    with pytest.raises(ValueError, match="content hash"):
        LatentDataset(cfg)


def test_dit_initialization_conditioning_and_patch_layout():
    block = DiTBlock(32, 4)
    nn.init.zeros_(block.condition[-1].weight)
    nn.init.zeros_(block.condition[-1].bias)
    x = torch.randn(2, 4, 32)
    torch.testing.assert_close(block(x, torch.randn(2, 32)), x)
    model = DiT(
        image_size=4,
        patch_size=2,
        width=16,
        depth=2,
        heads=4,
        channels=1,
        classes=2,
        gradient_checkpointing=True,
    )
    tokens = torch.arange(16).reshape(1, 4, 4)
    torch.testing.assert_close(
        model.unpatchify(tokens),
        torch.tensor([[[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]]]),
    )
    images = torch.randn(2, 1, 4, 4)
    assert torch.equal(model(images, torch.rand(2)), torch.zeros_like(images))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(3):
        loss = (model(images, torch.rand(2), torch.tensor([0, 2])) - images).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert model.condition.label.weight.grad.abs().sum() > 0
    assert position_embedding(4, 32).shape == (1, 16, 32)


def test_time_embedding_styles_and_old_checkpoint_compatibility():
    time = torch.tensor([0.25, -0.5])
    previous_frequencies = torch.exp(torch.linspace(0, math.log(1000), 16))
    phase = time[:, None] * previous_frequencies * 2 * math.pi
    expected = torch.cat([phase.sin(), phase.cos()], -1)
    original = TimeEmbedding(32)
    assert torch.equal(original(time), expected)
    original.load_state_dict({"frequencies": previous_frequencies})
    assert torch.equal(original(time), expected)
    positional = TimeEmbedding(32, "positional")
    small = positional(time + 1e-3) - positional(time)
    assert small.norm(dim=1).max() < 0.002
    with pytest.raises(ValueError, match="even"):
        TimeEmbedding(31)


def test_progressive_grid_and_two_to_one_target():
    for steps in [1, 2, 4, 8, 16, 32]:
        torch.testing.assert_close(
            progressive_grid(1000, 2 * steps)[::2], progressive_grid(1000, steps)
        )
    x, endpoint = torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8)
    ah, sh = torch.full((4,), 0.3), torch.full((4,), math.sqrt(1 - 0.3**2))
    al, sl = torch.full((4,), 0.8), torch.full((4,), 0.6)
    target = distillation_target(x, endpoint, ah, sh, al, sl)
    torch.testing.assert_close(ddim_destination(x, target, ah, sh, al, sl), endpoint)


def test_ict_online_target_and_doubling():
    class Online(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, x, t, labels=None):
            self.calls.append(torch.is_grad_enabled())
            return self.weight * x

    network = Online()
    student = ConsistencyModel(network)
    loss = improved_consistency_loss(student, torch.randn(4, 1, 8, 8), intervals=11).mean()
    loss.backward()
    assert network.calls == [True, False] and torch.isfinite(network.weight.grad)
    assert improved_scales(0, 10000) == 11 and improved_scales(10000, 10000) == 1281


def test_ict_bfloat16_online_target_does_not_detach_cached_weights():
    class Convolution(nn.Module):
        def __init__(self):
            super().__init__()
            self.output = nn.Conv2d(1, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, x, t, labels=None):
            return self.output(x)

    network = Convolution()
    student = ConsistencyModel(network)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    clean = torch.ones(8, 1, 4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = improved_consistency_loss(
            student, clean, intervals=11, generator=torch.Generator().manual_seed(4)
        ).mean()
    assert loss.requires_grad
    loss.backward()
    assert network.output.weight.grad is not None
    assert network.output.weight.grad.abs().max() > 0
    optimizer.step()
    assert network.output.weight.abs().max() > 0


def test_manifold_metrics_detect_collapse():
    real = torch.arange(20, dtype=torch.float64)[:, None] * 10
    equal = manifold_metrics(real, real, k=2, block_size=7)
    assert equal["precision"] == equal["recall"] == equal["coverage"] == 1
    collapsed = manifold_metrics(real, torch.zeros_like(real), k=2, block_size=7)
    assert collapsed["precision"] == 1 and collapsed["recall"] < 0.1 and collapsed["coverage"] < 0.3
    distant = manifold_metrics(real, real + 10000, k=2)
    assert distant["precision"] == distant["recall"] == distant["coverage"] == 0
