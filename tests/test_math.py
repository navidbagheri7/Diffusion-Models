"""Scientific invariants, analytic references, and convergence checks."""

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from diffusion_lab.bridges import bridge_training_pair, brownian_bridge_trajectory
from diffusion_lab.data import GaussianMixture
from diffusion_lab.guidance import AnalyticGMMClassifier, classifier_free, classifier_gradient
from diffusion_lab.metrics import empirical_wasserstein, sliced_wasserstein
from diffusion_lab.models import UNet
from diffusion_lab.paths import DiscreteVP, GaussianPath, expand, flow_coefficients
from diffusion_lab.sampling import discrete_sample
from diffusion_lab.solvers import integrate
from diffusion_lab.targets import convert_prediction, training_target
from diffusion_lab.transport import exact_pairing, sinkhorn_plan
from diffusion_lab.utils import EMA


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(3)
    torch.set_num_threads(2)


@pytest.mark.parametrize("kind", ["vp_linear", "vp_cosine", "ve"])
@pytest.mark.parametrize("prediction", ["epsilon", "x0", "score", "v"])
def test_prediction_round_trip(kind, prediction):
    x0, noise = torch.randn(16, 2), torch.randn(16, 2)
    alpha, sigma = GaussianPath(kind).coefficients(torch.linspace(0.05, 0.8, 16))
    x = expand(alpha, x0) * x0 + expand(sigma, x0) * noise
    target, _ = training_target(x0, noise, alpha, sigma, prediction)
    clean, eps, score = convert_prediction(x, target, alpha, sigma, prediction)
    torch.testing.assert_close(clean, x0, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(eps, noise, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(score, -noise / sigma[:, None], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["vp_linear", "vp_cosine", "ve"])
def test_sde_marginal_variance_ode(kind):
    path = GaussianPath(kind)
    t = torch.tensor([0.3, 0.6], dtype=torch.float64, requires_grad=True)
    alpha, sigma = path.coefficients(t)
    derivative_a = (
        torch.autograd.grad(alpha.sum(), t, retain_graph=True)[0]
        if alpha.requires_grad
        else torch.zeros_like(t)
    )
    derivative_v = torch.autograd.grad(sigma.square().sum(), t)[0]
    drift, g = path.sde(torch.ones(2, 1, dtype=torch.float64), t)
    torch.testing.assert_close(derivative_a, drift[:, 0] * alpha)
    torch.testing.assert_close(derivative_v, 2 * drift[:, 0] * sigma.square() + g.square())


def test_gmm_score_matches_log_density_gradient():
    mixture = GaussianMixture()
    x = torch.randn(19, 2, requires_grad=True)
    alpha, sigma = GaussianPath().coefficients(torch.full((19,), 0.25))
    gradient = torch.autograd.grad(mixture.log_prob(x, alpha, sigma).sum(), x)[0]
    torch.testing.assert_close(gradient, mixture.score(x, alpha, sigma), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_discrete_table_and_clean_endpoint(schedule):
    path = DiscreteVP(1000, schedule)
    assert torch.all((path.betas > 0) & (path.betas < 1))
    assert torch.all(path.alpha_bar[1:] < path.alpha_bar[:-1])
    a, s = path.coefficients(torch.tensor([0.0, 1.0]))
    assert a[0] == 1 and s[0] == 0
    torch.testing.assert_close(a[1].square(), path.alpha_bar[-1])


@pytest.mark.parametrize("method", ["ddpm", "ddim"])
def test_discrete_final_step_is_clean(method):
    path = DiscreteVP(40)

    class Oracle:
        def __call__(self, x, t, labels=None):
            return torch.full_like(x, 0.75)

    output, history = discrete_sample(Oracle(), path, torch.randn(12, 2), "x0", method, 20)
    torch.testing.assert_close(output, torch.full_like(output, 0.75), atol=1e-5, rtol=1e-5)
    assert len(history) == 21


def test_ddim_eta_zero_is_deterministic():
    path = DiscreteVP(40)

    def oracle(x, t, labels=None):
        return x * 0.1

    initial = torch.randn(4, 2)
    first = discrete_sample(oracle, path, initial, "x0", steps=10)[0]
    second = discrete_sample(oracle, path, initial, "x0", steps=10)[0]
    torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.parametrize("method,order", [("euler", 1), ("heun", 2), ("rk4", 4)])
def test_ode_convergence_order(method, order):
    def error(steps):
        result, _ = integrate(
            lambda x, t: x,
            torch.ones(1, 1, dtype=torch.float64),
            torch.linspace(0, 1, steps + 1, dtype=torch.float64),
            method,
        )
        return abs(result.item() - math.e)

    ratio = error(8) / error(16)
    assert ratio > 2 ** (order - 0.3)


def test_reverse_time_and_brownian_variance():
    times = torch.linspace(1, 0, 11)
    result, _ = integrate(
        lambda x, t: torch.zeros_like(x),
        torch.zeros(12000, 1),
        times,
        "euler_maruyama",
        lambda t: torch.ones_like(t),
    )
    assert abs(result.var().item() - 1) < 0.04
    assert abs(result.mean().item()) < 0.03


def test_flow_targets_are_path_derivatives():
    t = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    for kind in ["linear", "trigonometric"]:
        a, s, da, ds = flow_coefficients(t, kind)
        torch.testing.assert_close(torch.autograd.grad(a.sum(), t, retain_graph=True)[0], da)
        torch.testing.assert_close(torch.autograd.grad(s.sum(), t, retain_graph=True)[0], ds)


def test_bridge_joint_covariance_and_endpoints():
    z, y = torch.zeros(10000, 1), torch.ones(10000, 1)
    times = torch.tensor([0.0, 0.25, 0.75, 1.0])
    paths = brownian_bridge_trajectory(z, y, times, epsilon=0.8)
    torch.testing.assert_close(paths[0], z)
    torch.testing.assert_close(paths[-1], y)
    residual = paths.squeeze(-1) - times[:, None]
    assert abs(residual[1].var().item() - 0.8 * 0.25 * 0.75) < 0.01
    assert abs((residual[1] * residual[2]).mean().item() - 0.8 * (0.25 - 0.25 * 0.75)) < 0.01


def test_bridge_velocity_equals_drift_minus_half_diffusion_score():
    z, y, noise = torch.randn(11, 2), torch.randn(11, 2), torch.randn(11, 2)
    t = torch.linspace(0.1, 0.9, 11)
    x, b, v = bridge_training_pair(z, y, t, 0.7, noise)
    mean = (1 - t[:, None]) * z + t[:, None] * y
    score = -(x - mean) / (0.7 * t[:, None] * (1 - t[:, None]))
    torch.testing.assert_close(v, b - 0.7 / 2 * score, atol=2e-5, rtol=2e-5)


def test_transport_marginals_and_known_assignment():
    z = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    y = z.flip(0)
    torch.testing.assert_close(exact_pairing(z, y), z)
    plan = sinkhorn_plan(z, y, epsilon=1.0)
    torch.testing.assert_close(plan.sum(0), torch.full((3,), 1 / 3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(plan.sum(1), torch.full((3,), 1 / 3), atol=1e-6, rtol=1e-6)
    assert empirical_wasserstein(z, y) == 0
    assert sliced_wasserstein(z, y) == 0


def test_cfg_and_classifier_score_identity():
    def model(x, t, labels=None):
        return torch.zeros_like(x) if labels is None else torch.ones_like(x) * 2

    x, t, labels = torch.zeros(3, 2), torch.ones(3), torch.zeros(3, dtype=torch.long)
    torch.testing.assert_close(classifier_free(model, x, t, labels, 3), torch.ones_like(x) * 6)

    class Classifier:
        def __call__(self, x, t):
            return torch.stack([x[:, 0], -x[:, 0]], -1)

    gradient = classifier_gradient(Classifier(), x, t, labels)
    torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.0]]).expand_as(x))


def test_unet_and_ema_buffers():
    model = UNet(8, classes=3)
    x = torch.randn(2, 3, 16, 16)
    assert model(x, torch.tensor([0.2, 0.5]), torch.tensor([0, 3])).shape == x.shape
    ema = EMA(model)
    ema.update(model)
    torch.testing.assert_close(
        ema.model(x, torch.tensor([0.2, 0.5])), model(x, torch.tensor([0.2, 0.5]))
    )


def test_classifier_gradient_matches_conditional_minus_unconditional_score():
    mixture, path = GaussianMixture(), GaussianPath()
    x, t = torch.randn(31, 2), torch.full((31,), 0.3)
    labels = torch.arange(31) % mixture.components
    alpha, sigma = path.coefficients(t)
    gradient = classifier_gradient(AnalyticGMMClassifier(mixture, path), x, t, labels)
    expected = mixture.score(x, alpha, sigma, labels) - mixture.score(x, alpha, sigma)
    torch.testing.assert_close(gradient, expected, atol=2e-5, rtol=2e-5)


def test_adaptive_reverse_ode():
    initial = torch.full((2, 1), math.exp(-1), dtype=torch.float64)
    output, _ = integrate(
        lambda x, t: -x, initial, torch.linspace(1, 0, 21, dtype=torch.float64), "rk45"
    )
    torch.testing.assert_close(output, torch.ones_like(output), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "experiment",
    [
        "gmm_flow",
        "gmm_vp",
        "gmm_ve",
        "gmm_guidance",
        "gmm_bridge",
        "celeba_ddpm",
        "celeba_flow",
        "imagenet_conditional",
        "ffhq_ddpm",
        "gmm_reflow",
        "gmm_cosine_x0",
    ],
)
def test_hydra_experiments_compose(experiment):
    from diffusion_lab.train import validate_configuration

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])
    validate_configuration(cfg)


@pytest.mark.parametrize(
    "overrides",
    [
        ["process=ve"],
        ["experiment=gmm_flow", "sampler=sde"],
        ["experiment=gmm_bridge", "sampler=ode"],
        ["process.prediction=velocity"],
        ["evaluation_samples=1"],
    ],
)
def test_invalid_combinations_rejected_before_training(overrides):
    from diffusion_lab.train import validate_configuration

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train", overrides=overrides)
    with pytest.raises(ValueError):
        validate_configuration(cfg)
