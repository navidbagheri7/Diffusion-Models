"""General Gaussian-noise stochastic interpolants and their marginal fields.

X_t=a_t X_0+b_t X_1+gamma_t Z; Z is independent standard Gaussian.
Velocity target is its derivative; the noise regression gives score=-E[Z|x]/gamma.
Any nonnegative epsilon(t) gives forward drift v+epsilon*score, g=sqrt(2epsilon).
These SDEs share the ODE's time marginals in the exact-field limit.
"""

import math

import torch

from .guidance import classifier_free
from .paths import expand, flow_coefficients


class StochasticInterpolant:
    def __init__(self, path="linear", amplitude=0.5, base_fn=None, noise_std_fn=None):
        if path not in {"linear", "trigonometric", "cubic"} or amplitude <= 0:
            raise ValueError("Choose a supported path and positive interpolant noise amplitude")
        self.path, self.amplitude = path, amplitude
        self.base_fn = base_fn
        self.noise_std_fn = noise_std_fn

    def coefficients(self, t):
        if self.path == "cubic":
            b, db = 3 * t.square() - 2 * t.pow(3), 6 * t * (1 - t)
            a, da = 1 - b, -db
        else:
            b, a, db, da = flow_coefficients(t, self.path)
        if self.noise_std_fn is None:
            gamma = self.amplitude * torch.sin(math.pi * t)
            # Remove floating-point sine's tiny nonzero endpoint residual.
            gamma = torch.where((t == 0) | (t == 1), torch.zeros_like(gamma), gamma)
            derivative = self.amplitude * math.pi * torch.cos(math.pi * t)
        else:
            gamma, derivative = torch.func.jvp(self.noise_std_fn, (t,), (torch.ones_like(t),))
            if gamma.shape != t.shape:
                raise ValueError("Custom noise standard deviation must preserve time batch shape")
            boundary = (t == 0) | (t == 1)
            if not torch.isfinite(gamma).all() or (gamma[~boundary] < 0).any():
                raise ValueError(
                    "Interpolant noise standard deviation must be finite and nonnegative"
                )
            if not torch.isclose(
                gamma[boundary],
                torch.zeros_like(gamma[boundary]),
                atol=8 * torch.finfo(gamma.dtype).eps,
                rtol=0,
            ).all():
                raise ValueError("Interpolant noise standard deviation must vanish at endpoints")
            gamma = torch.where(boundary, torch.zeros_like(gamma), gamma)
        return a, b, gamma, da, db, derivative

    def targets(self, source, target, t, generator=None):
        noise = torch.randn(source.shape, device=source.device, generator=generator)
        a, b, gamma, da, db, dgamma = self.coefficients(t)
        if self.base_fn is None:
            base = expand(a, source) * source + expand(b, source) * target
            derivative = expand(da, source) * source + expand(db, source) * target
        else:
            # A user-supplied differentiable I(t,x0,x1) must act independently
            # on examples and satisfy I(0)=x0, I(1)=x1. Gaussian bridge noise
            # remains independent, so the same noise-to-score identity applies.
            base, derivative = torch.func.jvp(
                lambda time: self.base_fn(source, target, time), (t,), (torch.ones_like(t),)
            )
            if base.shape != source.shape:
                raise ValueError("Custom interpolant base must preserve endpoint shape")
        x = base + expand(gamma, source) * noise
        velocity = derivative + expand(dgamma, source) * noise
        return x, velocity, noise

    def learned_fields(self, model, labels=None, scale=1.0, strength=1.0, backward=False):
        if strength < 0:
            raise ValueError("Diffusion strength must be nonnegative")

        def velocity_and_noise(x, t):
            return classifier_free(model, x, t, labels, scale).chunk(2, dim=1)

        def velocity(x, t):
            return velocity_and_noise(x, t)[0]

        def drift(x, t):
            v, noise = velocity_and_noise(x, t)
            gamma = self.coefficients(t)[2]
            # epsilon=strength*gamma^2 cancels the score's gamma denominator.
            return v + (1 if backward else -1) * strength * expand(gamma, x) * noise

        def diffusion(t):
            return math.sqrt(2 * strength) * self.coefficients(t)[2]

        return velocity, drift, diffusion


class GaussianInterpolantOracle:
    """Exact density/score/velocity for independent uniform isotropic GMM endpoints."""

    def __init__(self, source, target, interpolant):
        if interpolant.base_fn is not None:
            raise ValueError("The analytic Gaussian oracle requires an affine endpoint interpolant")
        self.source, self.target, self.interpolant = source, target, interpolant

    def components(self, x, t):
        a, b, g, da, db, dg = self.interpolant.coefficients(t)
        source = self.source.means.to(x)
        target = self.target.means.to(x)
        means = (
            a[:, None, None, None] * source[None, :, None, :]
            + b[:, None, None, None] * target[None, None, :, :]
        ).flatten(1, 2)
        means_derivative = (
            da[:, None, None, None] * source[None, :, None, :]
            + db[:, None, None, None] * target[None, None, :, :]
        ).flatten(1, 2)
        variance = a.square() * self.source.std**2 + b.square() * self.target.std**2 + g.square()
        covariance = a * da * self.source.std**2 + b * db * self.target.std**2 + g * dg
        residual = x[:, None] - means
        log_density = (
            -0.5 * (residual.square() / variance[:, None, None]).sum(-1)
            - x.shape[1] / 2 * torch.log(2 * math.pi * variance[:, None])
            - math.log(means.shape[1])
        )
        return log_density, residual, variance, covariance, means_derivative

    def log_density(self, x, t):
        return torch.logsumexp(self.components(x, t)[0], dim=1)

    def fields(self, x, t):
        logits, residual, variance, covariance, mean_derivative = self.components(x, t)
        weights = logits.softmax(-1)
        score = -(weights[:, :, None] * residual / variance[:, None, None]).sum(1)
        velocity = (
            weights[:, :, None]
            * (mean_derivative + (covariance / variance)[:, None, None] * residual)
        ).sum(1)
        return velocity, score

    def drift(self, x, t, strength=1.0, backward=False):
        v, s = self.fields(x, t)
        gamma = self.interpolant.coefficients(t)[2]
        return v + (-1 if backward else 1) * strength * expand(gamma.square(), x) * s

    def diffusion(self, t, strength=1.0):
        return math.sqrt(2 * strength) * self.interpolant.coefficients(t)[2]
