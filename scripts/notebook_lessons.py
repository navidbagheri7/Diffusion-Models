"""Course cell sources: equations, experiments and worked exercises.

Notebooks import library implementations. Locally defined analytic fields and
short training loops are explicitly educational references.
"""

SETUP = r"""
from pathlib import Path
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image, display

project = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "diffusion_lab").is_dir())
if str(project) not in sys.path:
    sys.path.insert(0, str(project))
torch.set_num_threads(2)
torch.manual_seed(7)
np.random.seed(7)
plt.rcParams.update({"figure.figsize": (10, 3.6), "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})
ENABLE_WIDGETS = False  # Set True in Jupyter; static figures work on GitHub.
print(f"PyTorch {torch.__version__}; CPU lesson; seed=7; no automatic downloads.")

def scatter(ax, samples, title, labels=None, limits=(-4.5, 4.5)):
    points = samples.detach().cpu().numpy()
    colors = labels.detach().cpu().numpy() if labels is not None else "steelblue"
    ax.scatter(points[:, 0], points[:, 1], c=colors, cmap="tab10" if labels is not None else None,
               s=5, alpha=.55, linewidths=0)
    ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$", aspect="equal", xlim=limits, ylim=limits)

def saved_gallery(relative, caption):
    path = project / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing curated course evidence: {relative}")
    print(caption)
    display(Image(filename=str(path)))

def animate_particles(history, name, title, labels=None, limits=(-4.5, 4.5)):
    from matplotlib.animation import FuncAnimation, PillowWriter
    points = history.detach().cpu().numpy()
    indices = np.linspace(0, len(points)-1, min(32, len(points))).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = labels.detach().cpu().numpy() if labels is not None else "steelblue"
    dots = ax.scatter(*points[0].T, c=colors, cmap="tab10" if labels is not None else None,
                      s=9, alpha=.65, linewidths=0)
    ax.set(xlim=limits, ylim=limits, aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    def update(frame):
        dots.set_offsets(points[indices[frame]])
        ax.set_title(f"{title}: frame {frame+1}/{len(indices)}")
        return (dots,)
    animation = FuncAnimation(fig, update, frames=len(indices), interval=100)
    folder = project / "assets/teaching/notebooks"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}.gif"
    animation.save(target, writer=PillowWriter(fps=10), dpi=75)
    plt.close(fig)
    display(Image(filename=str(target)))
"""

LESSONS = [
    {
        "name": "00_probability_and_noising.ipynb",
        "title": "00 · Probability, tensors and Gaussian corruption",
        "intro": "Follow signal, noise, density and score on a distribution we can evaluate exactly, before introducing a network.",
        "prerequisites": "Python functions, vectors, Gaussian moments and derivatives.",
        "outcomes": "Broadcast coefficients; distinguish a marginal coupling from an SDE; compare schedules and derive corrupted moments.",
        "cells": [
            (
                "md",
                r"""
## 1. Samples, densities and scores

Our data law is $p_0(x)=K^{-1}\sum_k\mathcal N(x;\mu_k,s_0^2 I)$.
A sample has shape `[B,2]`, means `[K,2]`, and `log_prob(x)` returns `[B]`.
The score $\nabla_x\log p(x)$ has the **same shape as x**, not the shape of a probability.
Labels identify sampled components; real datasets need not expose such labels.
**Equation → code:** `GaussianMixture.sample`, `.log_prob`, `.score` in [data.py](../diffusion_lab/data.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import GaussianPath, expand
mixture = GaussianMixture(components=8, radius=3, std=.25)
generator = torch.Generator().manual_seed(7)
clean, labels = mixture.sample(1600, generator)
grid = torch.linspace(-4.2, 4.2, 100)
yy, xx = torch.meshgrid(grid, grid, indexing="ij")
points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
density = mixture.log_prob(points).exp().reshape(100, 100)
fig, axes = plt.subplots(1, 2)
scatter(axes[0], clean, "Samples by component", labels)
axes[1].contourf(xx, yy, density, levels=24, cmap="magma")
axes[1].set(title="Analytic density", aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
plt.tight_layout(); plt.show()
print("sample / label / density shapes:", clean.shape, labels.shape, mixture.log_prob(clean).shape)
""",
            ),
            (
                "md",
                r"""
## 2. Corruption and broadcasting

Diffusion time increases **data → noise**:
$$X_t=\alpha(t)X_0+\sigma(t)\epsilon,\qquad\epsilon\sim\mathcal N(0,I),\quad\epsilon\perp X_0.$$
$\sigma$ is a standard deviation, not variance. `t`, `alpha`, `sigma` have shape `[B]`.
`expand(alpha,x)` returns `[B,1]` for points or `[B,1,1,1]` for images `[B,C,H,W]`.
The conditional law is $q_t(x|x_0)=\mathcal N(\alpha x_0,\sigma^2I)$;
the marginal $p_t$ averages it over data.
**Code:** `GaussianPath.coefficients`, `expand` in [paths.py](../diffusion_lab/paths.py).
""",
            ),
            (
                "code",
                """
noise = torch.randn(clean.shape, generator=generator)
path = GaussianPath("vp_linear")
times = [0., .15, .4, .75, .99]
fig, axes = plt.subplots(1, len(times), figsize=(15, 3))
for ax, time in zip(axes, times, strict=True):
    a, s = path.coefficients(torch.full((len(clean),), time))
    scatter(ax, expand(a, clean)*clean+expand(s, clean)*noise, f"t={time:.2f}", labels)
plt.tight_layout(); plt.show()
assert expand(torch.ones(4), torch.zeros(4, 3, 16, 16)).shape == (4, 1, 1, 1)
""",
            ),
            (
                "md",
                r"""
## 3. Different schedules, different paths

For VP, $\alpha^2+\sigma^2=1$. With $B(t)=\int_0^t\beta(s)ds$,
$\alpha=e^{-B/2}$, $\sigma=\sqrt{1-e^{-B}}$. Linear **beta** is not linear noise.
Sub-VP uses $\sigma=1-e^{-B}$. VE uses $\alpha=1$ and
$\sigma=\sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$.
Here `vp_exponential` means constant beta/exponentially decaying signal;
`vp_logsnr` linearly interpolates $\log(\alpha^2/\sigma^2)$.
VE starts with small smoothing; finite log-SNR bounds give approximate endpoints.
The cosine SDE is singular at $t=1$; evaluate its field inside the interval.
""",
            ),
            (
                "code",
                """
grid = torch.linspace(.001, .999, 400)
kinds = ["vp_linear", "vp_cosine", "sub_vp", "ve", "vp_logsnr", "vp_exponential"]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for kind in kinds:
    a, s = GaussianPath(kind).coefficients(grid)
    axes[0].plot(grid, a, label=kind); axes[1].plot(grid, s, label=kind)
    axes[2].plot(grid, 2*(a.log()-s.log()), label=kind)
for ax, title in zip(axes, ["Signal", "Noise standard deviation", "Log-SNR"], strict=True):
    ax.set(title=title, xlabel="Forward diffusion t")
axes[1].set_yscale("log"); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
""",
            ),
            (
                "md",
                r"""
## 4. Check the moments

Independence implies $\mathbb E[X_t]=\alpha\mathbb E[X_0]$ and
$\operatorname{Cov}(X_t)=\alpha^2\operatorname{Cov}(X_0)+\sigma^2I$.
VP preserves **unit Gaussian variance**, not every dataset's variance.
""",
            ),
            (
                "code",
                """
g = torch.Generator().manual_seed(11)
x0 = 2*torch.randn(100000, 1, generator=g)+3
eps = torch.randn(x0.shape, generator=g)
a, s = GaussianPath().coefficients(torch.tensor([.25]))
xt = a*x0+s*eps
expected_mean, expected_variance = 3*a.item(), 4*a.square().item()+s.square().item()
print({"empirical_mean": xt.mean().item(), "analytic_mean": expected_mean,
       "empirical_variance": xt.var().item(), "analytic_variance": expected_variance})
assert abs(xt.mean().item()-expected_mean) < .03
assert abs(xt.var().item()-expected_variance) < .05
""",
            ),
            (
                "md",
                r"""
## 5. Watch the marginal change

We reuse one noise vector per point: a **coupling of marginal samples**, not
a forward SDE trajectory. The time marginals are correct but the joint law across times differs.
The animation is also saved as a standalone [GIF](../assets/teaching/notebooks/00_gaussian_corruption.gif).
""",
            ),
            (
                "code",
                """
history = []
for time in torch.linspace(0, .99, 32):
    a, s = path.coefficients(time.repeat(240))
    history.append(expand(a, clean[:240])*clean[:240]+expand(s, clean[:240])*noise[:240])
animate_particles(torch.stack(history), "00_gaussian_corruption", "Gaussian marginal coupling", labels[:240])
""",
            ),
            (
                "code",
                """
def explore_noising(time=.4, schedule="vp_linear"):
    a, s = GaussianPath(schedule).coefficients(torch.full((len(clean),), float(time)))
    _fig, ax = plt.subplots()
    scatter(ax, expand(a, clean)*clean+expand(s, clean)*noise, f"{schedule}, t={time:.2f}", labels)
    plt.show()
if ENABLE_WIDGETS:
    from ipywidgets import interact
    interact(explore_noising, time=(.01, .99, .02), schedule=kinds)
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. For $X_0\sim\mathcal N(3,4)$, find the corrupted mean/variance. What variance does VP preserve?
2. Why would fresh independent noise at every animation frame not produce an SDE trajectory?
3. Change component radius and standard deviation. Which ratio controls visible mode separation?

<details><summary>Worked answers</summary>

1. Mean $3\alpha$, variance $4\alpha^2+\sigma^2$. VP preserves variance 1.
2. Brownian increments have variance proportional to $dt$; independent marginal redraws impose a different temporal covariance.
3. Separation scales with $\alpha\|\mu_i-\mu_j\|/\sqrt{\alpha^2s_0^2+\sigma^2}$, rather than time alone.

</details>

**Continue:** [DDPM/DDIM](01_prediction_targets_and_ddim.ipynb).
**Read:** [Foundations](../docs/01_foundations.md), [SDE paper](https://arxiv.org/abs/2011.13456).
""",
            ),
        ],
    },
    {
        "name": "01_prediction_targets_and_ddim.ipynb",
        "title": "01 · Denoising regression, DDPM and DDIM",
        "intro": "Separate what the network learns, how its output is converted, and which update generates a sample.",
        "prerequisites": "Notebook 00; conditional expectation; PyTorch autograd and MSE.",
        "outcomes": "Derive Tweedie/VP-v conversions; inspect the posterior; train a small denoiser; compare sampling methods with measured calls.",
        "cells": [
            (
                "md",
                r"""
## 1. What MSE learns

Minimizing $\mathbb E\|h(X_t,t)-Y\|^2$ gives $h^*(x,t)=\mathbb E[Y|X_t=x,t]$.
An epsilon model predicts an average compatible noise, not the hidden draw.
The conditional score is $-\epsilon/\sigma$; its average is the marginal score:
$$s_t(x)=-\mathbb E[\epsilon|x]/\sigma,\qquad
\widehat x_0=(x+\sigma^2s_t(x))/\alpha.$$
The second identity is Gaussian Tweedie. Avoid singular coefficients.
VP-v is $v=\alpha\epsilon-\sigma x_0$, giving the general conversions
$$\widehat x_0=\frac{\alpha x-\sigma\widehat v}{\alpha^2+\sigma^2},\qquad
\widehat\epsilon=\frac{\sigma x+\alpha\widehat v}{\alpha^2+\sigma^2}.$$
VP makes the denominator 1. Flow velocity is different (03).
**Code:** `training_target`, `convert_prediction` in [targets.py](../diffusion_lab/targets.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.paths import GaussianPath, DiscreteVP, expand
from diffusion_lab.targets import training_target, convert_prediction
x0, eps = torch.randn(64, 2), torch.randn(64, 2)
for kind in ["vp_linear", "ve", "sub_vp"]:
    a, s = GaussianPath(kind).coefficients(torch.full((64,), .3))
    xt = expand(a, x0)*x0+expand(s, x0)*eps
    for name in ["epsilon", "x0", "score", "v"]:
        target, weight = training_target(x0, eps, a, s, name)
        recovered, recovered_eps, _ = convert_prediction(xt, target, a, s, name)
        torch.testing.assert_close(recovered, x0, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(recovered_eps, eps, atol=3e-5, rtol=3e-5)
    print(kind, "four conversions verified; alpha^2+sigma^2=", (a*a+s*s)[0].item())
""",
            ),
            (
                "md",
                r"""
## 2. DDPM table and forward posterior

Mathematical steps are $k=1,\ldots,T$, $a_k=1-\beta_k$ and
$\bar a_k=\prod_{j=1}^k a_j$. Signal $\alpha_k=\sqrt{\bar a_k}$,
noise $\sigma_k=\sqrt{1-\bar a_k}$. **$a_k$ is not $\alpha_k$.**
Table `i` means $k=i+1$; network time is `(i+1)/T`; clean sampling index is `-1`.
$$q(x_{k-1}|x_k,x_0)=\mathcal N(\widetilde\mu_k,\widetilde\beta_kI),$$
$$\widetilde\beta_k=\beta_k\frac{1-\bar a_{k-1}}{1-\bar a_k},\quad
\widetilde\mu_k=\frac{\sqrt{\bar a_{k-1}}\beta_k}{1-\bar a_k}x_0+
\frac{\sqrt{a_k}(1-\bar a_{k-1})}{1-\bar a_k}x_k.$$
At $k=1$ the posterior is a point mass. Code uses second-step posterior variance
for the finite first log-variance convention; the decoder is handled separately.
**Code:** [DiscreteVP](../diffusion_lab/paths.py), [posterior](../diffusion_lab/improved_ddpm.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.improved_ddpm import posterior
path = DiscreteVP(1000, "cosine")
indices = torch.tensor([0, 10, 200, 800]); time = (indices.float()+1)/path.steps
a, s = path.coefficients(time)
clean = torch.randn(4, 2)
noisy = expand(a, clean)*clean+expand(s, clean)*torch.randn_like(clean)
mean, logvar = posterior(path, clean, noisy, indices)
torch.testing.assert_close(mean[0], clean[0], atol=1e-5, rtol=1e-5)
fig, axes = plt.subplots(1, 2)
for schedule in ["linear", "cosine", "logsnr", "exponential"]:
    table = DiscreteVP(1000, schedule)
    axes[0].plot(torch.arange(1, 1001)/1000, table.alpha_bar, label=schedule)
    axes[1].plot(torch.arange(1, 1001)/1000, table.betas, label=schedule)
axes[0].set(title="Cumulative signal power", xlabel="k/T"); axes[0].legend()
axes[1].set(title="Per-step beta", xlabel="k/T", yscale="log")
plt.tight_layout(); plt.show()
""",
            ),
            (
                "md",
                r"""
## 3. Train and check on independent draws

Use the residual, time-conditioned `TimeMLP`, uniform table timesteps and clean-data MSE.
We choose x0 here because approximate epsilon predictions at the tiny terminal
alpha of a cosine table can amplify error through division by alpha. A smooth
positional time embedding is used for this short normalized-time training run.
This short CPU run is a teaching baseline. Tweedie's exact conditional clean-data average lets us measure **field error**, independent of pair-target noise.
**Code:** [models.py](../diffusion_lab/models.py); production Hydra/EMA/checkpoint loop [train.py](../diffusion_lab/train.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.data import GaussianMixture
from diffusion_lab.models import TimeMLP,Conditioning
mixture = GaussianMixture()
torch.manual_seed(21)
model = TimeMLP(width=64, depth=2)
model.condition = Conditioning(64,0,time_embedding="positional")
optimizer = torch.optim.Adam(model.parameters(), lr=.002)
train_rng = torch.Generator().manual_seed(21)
losses = []
for _step in range(1000):
    clean, _ = mixture.sample(256, train_rng)
    indices = torch.randint(path.steps, (len(clean),), generator=train_rng)
    time = (indices.float()+1)/path.steps
    a, s = path.coefficients(time)
    eps = torch.randn(clean.shape, generator=train_rng)
    noisy = expand(a, clean)*clean+expand(s, clean)*eps
    loss = (model(noisy, time)-clean).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
model.eval()
plt.plot(losses, alpha=.3, label="Batch MSE")
plt.plot(np.convolve(losses, np.ones(25)/25, mode="valid"), label="25-update mean")
plt.xlabel("Optimizer update"); plt.ylabel("Clean-data MSE"); plt.legend(); plt.show()
""",
            ),
            (
                "code",
                """
heldout_rng = torch.Generator().manual_seed(22)
heldout, _ = mixture.sample(512, heldout_rng)
eps = torch.randn(heldout.shape, generator=heldout_rng)
errors = []
for value in [.02, .1, .3, .6, .9]:
    time = torch.full((len(heldout),), value)
    a, s = path.coefficients(time)
    noisy = expand(a, heldout)*heldout+expand(s, heldout)*eps
    oracle = (noisy+expand(s.square(),noisy)*mixture.score(noisy,a,s))/expand(a,noisy)
    with torch.no_grad():
        prediction = model(noisy, time)
    errors.append((prediction-oracle).square().mean().sqrt().item())
plt.plot([.02, .1, .3, .6, .9], errors, marker="o")
plt.xlabel("Corruption time"); plt.ylabel("Held-out RMSE vs conditional mean"); plt.show()
""",
            ),
            (
                "md",
                r"""
## 4. DDPM versus DDIM

For cumulative powers $\bar a\to\bar a'$ (toward cleaner data), DDIM uses
$$x'=\sqrt{\bar a'}\widehat x_0+\sqrt{1-\bar a'-\nu^2}\widehat\epsilon+\nu z,$$
$$\nu=\eta\sqrt{\frac{1-\bar a'}{1-\bar a}(1-\bar a/\bar a')}.$$
$\eta=0$ is deterministic given initial noise/model/grid. DDPM uses posterior
mean plus posterior noise; this code also supports ancestral respaced transitions.
The oracle predicts exact marginal conditional means, but finite-grid Gaussian
reverse transitions still incur discretization error.
**Code:** `discrete_sample` in [sampling.py](../diffusion_lab/sampling.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.sampling import discrete_sample
class OracleNoise(torch.nn.Module):
    def forward(self, x, t, labels=None):
        a, s = path.coefficients(t)
        return -expand(s, x)*mixture.score(x, a, s, labels)
class CountCalls(torch.nn.Module):
    def __init__(self, network):
        super().__init__(); self.network = network; self.calls = 0
    def forward(self, x, t, labels=None):
        self.calls += 1
        return self.network(x, t, labels)
initial = torch.randn(512, 2, generator=torch.Generator().manual_seed(23))
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
cases = [(OracleNoise(), "ddpm", 1000,"epsilon"), (OracleNoise(), "ddim", 20,"epsilon"),
         (OracleNoise(), "ddim", 5,"epsilon"), (model, "ddim", 20,"x0")]
for ax, (network, method, steps,prediction) in zip(axes, cases, strict=True):
    counted = CountCalls(network)
    generated, _ = discrete_sample(counted, path, initial, prediction=prediction,method=method, steps=steps,
                                   generator=torch.Generator().manual_seed(24))
    scatter(ax, generated, f"{'Learned' if network is model else 'Oracle'} {method}: {counted.calls} calls")
plt.tight_layout(); plt.show()
""",
            ),
            (
                "code",
                """
first, _ = discrete_sample(OracleNoise(), path, initial, steps=20, eta=0)
second, _ = discrete_sample(OracleNoise(), path, initial, steps=20, eta=0)
torch.testing.assert_close(first, second, atol=0, rtol=0)
stochastic, _ = discrete_sample(OracleNoise(), path, initial, steps=20, eta=1,
                              generator=torch.Generator().manual_seed(25))
print("Same-noise eta=0 versus eta=1 RMS difference:", (first-stochastic).square().mean().sqrt().item())
""",
            ),
            (
                "code",
                """
class ParameterizedOracle(OracleNoise):
    def __init__(self,prediction):
        super().__init__(); self.prediction = prediction
    def forward(self,x,t,labels=None):
        eps = super().forward(x,t,labels)
        a,s = path.coefficients(t)
        x0,_,_ = convert_prediction(x,eps,a,s,"epsilon")
        return training_target(x0,eps,a,s,self.prediction)[0]
def explore_targets(prediction="epsilon",steps=20):
    samples,_ = discrete_sample(ParameterizedOracle(prediction),path,initial,
                                 prediction=prediction,steps=steps)
    _fig,ax = plt.subplots()
    scatter(ax,samples,f"Oracle {prediction}, DDIM {steps} calls"); plt.show()
if ENABLE_WIDGETS:
    from ipywidgets import interact
    interact(explore_targets,prediction=["epsilon","x0","score","v"],steps=(2,100,2))
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Derive Tweedie by differentiating $p_t(x)=\int q_t(x|x_0)p_0(x_0)dx_0$.
2. Why can an exact conditional-average epsilon network have nonzero training MSE?
3. Change training budget; compare field error and mode coverage with fixed evaluation noise.

<details><summary>Worked answers</summary>

1. $\nabla\log p_t=(\alpha\mathbb E[X_0|x]-x)/\sigma^2$; rearrange.
2. Irreducible conditional variance remains, separate from approximation error.
3. Averaged loss can hide poor low-noise fields; extra sampling calls do not necessarily repair model error.

</details>

Full recipe: `python -m diffusion_lab.train experiment=gmm_cosine_x0 device=cpu`
([config](../configs/experiment/gmm_cosine_x0.yaml)).
**Continue:** [SDE/ODE/PDE](02_scores_sdes_and_odes.ipynb).
**Sources:** [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502).
""",
            ),
        ],
    },
    {
        "name": "02_scores_sdes_and_odes.ipynb",
        "title": "02 · Scores, SDEs, ODEs and density evolution",
        "intro": "Connect trajectories to the equations that move probability mass. Compare exact fields before attributing differences to learned models.",
        "prerequisites": "00–01; gradients, divergence and basic stochastic increments.",
        "outcomes": "Check analytic scores; derive VP/VE/sub-VP coefficients; explain reverse-time signs; verify a Fokker–Planck/continuity identity numerically.",
        "cells": [
            (
                "md",
                r"""
## 1. A score points toward increasing log density

For an isotropic Gaussian, $s(x)=-(x-\mu)/v$. A mixture score is a
**posterior-responsibility-weighted** average of component scores, not their
unweighted sum. It need not point toward the nearest mean everywhere.
We compare `GaussianMixture.score` with differentiation of `log_prob`.
**Code:** [data.py](../diffusion_lab/data.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import GaussianPath, expand
from diffusion_lab.solvers import integrate
mixture, path = GaussianMixture(), GaussianPath("vp_linear")
grid = torch.linspace(-4, 4, 19)
yy, xx = torch.meshgrid(grid, grid, indexing="ij")
x = torch.stack([xx.flatten(), yy.flatten()], dim=1).requires_grad_(True)
t = torch.full((len(x),), .25)
a, s = path.coefficients(t)
automatic = torch.autograd.grad(mixture.log_prob(x, a, s).sum(), x)[0]
analytic = mixture.score(x, a, s)
torch.testing.assert_close(analytic, automatic, atol=2e-5, rtol=2e-5)
fig, ax = plt.subplots()
ax.quiver(x[:,0].detach(), x[:,1].detach(), analytic[:,0].detach(), analytic[:,1].detach())
ax.set(title="Marginal score at t=.25", aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
plt.show()
""",
            ),
            (
                "md",
                r"""
## 2. From corruption coefficients to an SDE

For $dX_t=F(t)X_tdt+g(t)dW_t$, Gaussian corruption requires
$$F=\alpha'/\alpha,\qquad g^2=(\sigma^2)'-2F\sigma^2.$$
This follows from mean and variance evolution. With $B=\int\beta$:

| Family | $f(x,t)$ | $g(t)^2$ | Noise standard deviation |
|---|---|---|---|
| VP | $-\beta x/2$ | $\beta$ | $\sqrt{1-e^{-B}}$ |
| sub-VP | $-\beta x/2$ | $\beta(1-e^{-2B})$ | $1-e^{-B}$ |
| VE | $0$ | $(\sigma^2)'$ | Exponential sigma in this implementation |

All examples use state-independent diffusion (additive noise); general
state-dependent diffusion needs additional terms. **Code:** `GaussianPath.sde` in [paths.py](../diffusion_lab/paths.py).
""",
            ),
            (
                "code",
                """
for kind in ["vp_linear", "vp_cosine", "sub_vp", "ve", "vp_logsnr", "vp_exponential"]:
    candidate = GaussianPath(kind)
    time = torch.linspace(.1, .8, 40, dtype=torch.float64).requires_grad_(True)
    a, s = candidate.coefficients(time)
    da = torch.autograd.grad(a.sum(), time, retain_graph=True)[0] if a.requires_grad else torch.zeros_like(a)
    ds2 = torch.autograd.grad(s.square().sum(), time)[0]
    F = da/a
    f, g = candidate.sde(torch.ones(len(time),2,dtype=time.dtype), time)
    torch.testing.assert_close(f[:,0], F, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(g.square(), ds2-2*F*s.square(), atol=1e-8, rtol=1e-8)
    print(kind, "mean/variance SDE identity passed")
""",
            ),
            (
                "md",
                r"""
## 3. Reverse clock: signs matter

Forward $t$ runs data → noise. Using the **same decreasing clock** for generation:
$$dX_t=[f-g^2s_t]dt+g\,d\overline W_t,\qquad dt<0,$$
$$\frac{dX_t}{dt}=f-\tfrac12g^2s_t \quad\text{(probability-flow ODE)}.$$
Brownian increment variance is $|dt|$. Alternatively use $r=1-t$ increasing:
reverse drift becomes $-f+g^2s_t$, ODE drift $-f+g^2s_t/2$.
Do not apply a reversed sign **and** a decreasing grid.

Exact score, matching initial marginal, regularity and exact integration give
the same time marginals for SDE and ODE. Their individual trajectories differ.
We initialize from the actual finite-time marginal, avoiding an approximate-prior confound.
**Code:** [solvers.py](../diffusion_lab/solvers.py), [sampling.py](../diffusion_lab/sampling.py).
""",
            ),
            (
                "code",
                """
g = torch.Generator().manual_seed(31)
clean, labels = mixture.sample(512, g)
a, s = path.coefficients(torch.full((len(clean),), .99))
initial = expand(a,clean)*clean+expand(s,clean)*torch.randn(clean.shape,generator=g)
def score(x,t):
    return mixture.score(x,*path.coefficients(t))
def ode_field(x,t):
    f,g = path.sde(x,t)
    return f-.5*expand(g.square(),x)*score(x,t)
def reverse_drift(x,t):
    f,g = path.sde(x,t)
    return f-expand(g.square(),x)*score(x,t)
def diffusion(t):
    return path.sde(torch.zeros(len(t),2),t)[1]
times = torch.linspace(.99,.005,301)
ode, ode_history = integrate(ode_field, initial, times, "heun")
sde, sde_history = integrate(reverse_drift, initial, times, "euler_maruyama", diffusion,
                             torch.Generator().manual_seed(32))
fig, axes = plt.subplots(1, 3, figsize=(14,4))
scatter(axes[0], initial, "Matching initial marginal")
for ax, samples, history, title in zip(axes[1:], [ode,sde], [ode_history,sde_history], ["PF-ODE","Reverse SDE"], strict=True):
    ax.plot(history[:,:12,0], history[:,:12,1], alpha=.35, linewidth=.7)
    scatter(ax, samples, title)
plt.tight_layout(); plt.show()
print("Finite endpoint t=.005 and solver error remain; trajectories need not coincide.")
""",
            ),
            (
                "code",
                """
animate_particles(ode_history[:,:180], "02_probability_flow", "PF-ODE mass transport")
animate_particles(sde_history[:,:180], "02_reverse_sde", "Reverse SDE mass transport")
""",
            ),
            (
                "md",
                r"""
## 4. Fokker–Planck becomes continuity

For state-independent $g(t)$,
$$\partial_t p=-\nabla\cdot(fp)+\tfrac12g^2\Delta p.$$
Since $p s_t=\nabla p$, define $u=f-g^2s_t/2$ and obtain
$$\partial_t p+\nabla\cdot(up)=0.$$
One equation evolves density under an SDE; the other under a deterministic
velocity. Equal PDEs explain equal marginals, not equal paths.
The following uses **autograd in double precision** to check a local residual
of the analytic GMM density. This is a derivative consistency check, not a proof
for all fields. **Code connection:** `field_and_divergence` in [likelihood.py](../diffusion_lab/likelihood.py).
""",
            ),
            (
                "code",
                """
x = torch.randn(48,2,dtype=torch.float64,requires_grad=True)
t = torch.full((len(x),),.35,dtype=torch.float64,requires_grad=True)
a,s = path.coefficients(t)
p = mixture.log_prob(x,a,s).exp()
dpdt = torch.autograd.grad(p.sum(),t,create_graph=True)[0]
gradp = torch.autograd.grad(p.sum(),x,create_graph=True)[0]
f,g = path.sde(x,t)
u = f-.5*expand(g.square(),x)*mixture.score(x,a,s)
def divergence(vector):
    return sum(torch.autograd.grad(vector[:,j].sum(),x,create_graph=True,retain_graph=True)[0][:,j]
               for j in range(2))
laplacian = divergence(gradp)
fp_residual = dpdt+divergence(f*p[:,None])-.5*g.square()*laplacian
continuity_residual = dpdt+divergence(u*p[:,None])
print("Max Fokker–Planck / continuity residual:", fp_residual.abs().max().item(), continuity_residual.abs().max().item())
assert fp_residual.abs().max() < 1e-8
assert continuity_residual.abs().max() < 1e-8
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Write the reverse-SDE drift with an increasing generation clock.
2. Why does PF-ODE use half the score coefficient?
3. Compare 50 and 600 steps; use [sliced Wasserstein](../diffusion_lab/metrics.py) against independently drawn endpoint reference samples. Repeat SDE noise seeds.

<details><summary>Worked answers</summary>

1. $-f(x,1-r)+g(1-r)^2s_{1-r}(x)$; Brownian coefficient remains $g(1-r)$.
2. The other half is represented by the SDE's diffusion/Laplacian term.
3. Empirical distribution distance mixes numerical and finite-sample error. A shared initial sample makes comparisons more controlled but does not eliminate Monte Carlo error.

</details>

**Animations:** [ODE](../assets/teaching/notebooks/02_probability_flow.gif), [SDE](../assets/teaching/notebooks/02_reverse_sde.gif).
**Continue:** [Flow matching](03_flow_matching_and_solvers.ipynb).
**Source:** [Song et al.](https://arxiv.org/abs/2011.13456).
""",
            ),
        ],
    },
    {
        "name": "03_flow_matching_and_solvers.ipynb",
        "title": "03 · Flow matching, learned transport and numerical solvers",
        "intro": "Begin with endpoint pairs, derive a marginal velocity, then learn a field and compare integration cost.",
        "prerequisites": "00–02; ODEs, MSE and conditional expectation.",
        "outcomes": "Distinguish straight training pairs from curved marginal paths; train a flow; compare Euler/Heun/RK4/RK45 by actual field calls; explore interpolation and reflow.",
        "cells": [
            (
                "md",
                r"""
## 1. Generation time goes noise → data

Use $\tau\in[0,1]$, source noise $Z$ and data $X$:
$$I_\tau=(1-\tau)Z+\tau X,\quad \dot I_\tau=X-Z,$$
$$\mathcal L=\mathbb E\|u_\theta(I_\tau,\tau)-(X-Z)\|^2.$$
The optimum is $u_\tau(x)=\mathbb E[X-Z|I_\tau=x]$, which satisfies continuity.
Straight endpoint-pair paths can cross; the averaged field's ODE trajectories
need not be straight. Under usual uniqueness assumptions, ODE trajectories
cannot cross at the same time. VP-v from 01 is not this derivative.
**Code:** `flow_coefficients` in [paths.py](../diffusion_lab/paths.py), training in [train.py](../diffusion_lab/train.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.paths import flow_coefficients, expand
from diffusion_lab.solvers import integrate
from diffusion_lab.data import GaussianMixture
from diffusion_lab.models import TimeMLP,Conditioning
g = torch.Generator().manual_seed(41)
source = torch.randn(200,2,generator=g)
mixture = GaussianMixture()
target, labels = mixture.sample(200,g)
grid = torch.linspace(0,1,51)
fig, axes = plt.subplots(1,2)
for ax, kind in zip(axes,["linear","trigonometric"],strict=True):
    history = []
    for time in grid:
        a,s,da,ds = flow_coefficients(time.repeat(len(source)),kind)
        history.append(expand(a,target)*target+expand(s,source)*source)
    history = torch.stack(history)
    ax.plot(history[:,:12,0],history[:,:12,1],alpha=.6)
    scatter(ax,target,f"Paired {kind} paths",labels)
plt.tight_layout(); plt.show()
""",
            ),
            (
                "md",
                r"""
## 2. An exact Gaussian flow separates solver error from model error

For independent $Z\sim N(0,I)$, $X\sim N(\mu,\operatorname{diag}(s_1^2))$,
$I_\tau$ has mean $\tau\mu$ and variance $q=(1-\tau)^2+\tau^2s_1^2$.
$$u_\tau(x)=\mu+\frac{-(1-\tau)+\tau s_1^2}{q}(x-\tau\mu).$$
The exact marginal ODE map at time $\tau$ is $\tau\mu+\sqrt q Z$;
at 1 it is $\mu+s_1Z$. This map is **not** the original independent pair.
Euler uses 1 call/step, Heun 2, RK4 4. RK45 chooses internal steps adaptively;
requested output times do not specify its call count.
**Code:** [solvers.py](../diffusion_lab/solvers.py).
""",
            ),
            (
                "code",
                """
mean,std = torch.tensor([2.,1.]),torch.tensor([.5,1.5])
initial = torch.randn(128,2,generator=torch.Generator().manual_seed(42))
def gaussian_field(x,t):
    time = t[:,None]
    variance = (1-time)**2+time**2*std.square()
    return mean+(-(1-time)+time*std.square())/variance*(x-time*mean)
exact = mean+std*initial
fig,axes = plt.subplots(1,2)
records = []
for method in ["euler","heun","rk4","rk45"]:
    costs,errors = [],[]
    for steps in [2,5,10,20,40]:
        counter = {"calls":0}
        def counted_field(x,t,counter=counter):
            counter["calls"] += 1
            return gaussian_field(x,t)
        out,history = integrate(counted_field,initial,torch.linspace(0,1,steps+1),method)
        error = (out-exact).square().mean().sqrt().item()
        costs.append(counter["calls"]); errors.append(error)
        records.append((method,steps,counter["calls"],error))
    axes[0].loglog(costs,errors,marker="o",label=method)
axes[0].set(xlabel="Measured field calls",ylabel="Paired endpoint RMSE"); axes[0].legend()
_,history = integrate(gaussian_field,initial,torch.linspace(0,1,101),"heun")
axes[1].plot(history[:,:12,0],history[:,:12,1],alpha=.6)
scatter(axes[1],exact,"Curved marginal trajectories")
plt.tight_layout(); plt.show()
print("RK45 call counts:", [row[2] for row in records if row[0]=="rk45"])
""",
            ),
            (
                "md",
                r"""
## 3. Learn the mixture velocity

No score network is needed for deterministic flow matching. Use a smooth positional embedding for normalized time. Draw new independent
endpoints at every update and regress their derivative. Below we use a small
unconditional CPU model; the full [flow recipe](../configs/experiment/gmm_flow.yaml)
adds saved training state and evaluation. Plot held-out samples instead of using loss as a quality claim.
""",
            ),
            (
                "code",
                """
torch.manual_seed(43)
flow = TimeMLP(width=64,depth=2)
flow.condition = Conditioning(64,0,time_embedding="positional")
optimizer = torch.optim.Adam(flow.parameters(),lr=.002)
g = torch.Generator().manual_seed(43)
losses = []
for _step in range(1500):
    data,_ = mixture.sample(256,g)
    noise = torch.randn(data.shape,generator=g)
    time = torch.rand(len(data),generator=g)
    a,s,da,ds = flow_coefficients(time)
    state = expand(a,data)*data+expand(s,data)*noise
    velocity_target = expand(da,data)*data+expand(ds,data)*noise
    loss = (flow(state,time)-velocity_target).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
flow.eval()
initial = torch.randn(512,2,generator=torch.Generator().manual_seed(44))
generated, learned_history = integrate(flow,initial,torch.linspace(0,1,51),"heun")
reference,ref_labels = mixture.sample(512,torch.Generator().manual_seed(45))
fig,axes = plt.subplots(1,3,figsize=(14,4))
axes[0].plot(losses,alpha=.25)
axes[0].plot(np.convolve(losses,np.ones(25)/25,mode="valid")); axes[0].set(title="Velocity MSE",xlabel="Update")
scatter(axes[1],generated,"Short-trained flow")
scatter(axes[2],reference,"Independent data",ref_labels)
plt.tight_layout(); plt.show()
animate_particles(learned_history[:,:180],"03_learned_flow","Short-trained flow")
""",
            ),
            (
                "md",
                r"""
## 4. Noise interpolation and reflow are different operations

Interpolating two initial noise vectors explores a fixed generator's map.
Linear interpolation reduces variance at its midpoint: independent standard
Gaussian endpoints give variance $1/2$ there. Spherical interpolation approximately
maintains typical radius in high dimensions; neither promises semantic linearity.
This low-dimensional example intentionally uses simple linear interpolation.

Reflow instead generates teacher endpoints from known initial noise, then
trains a new field on those pairs. It can reduce trajectory curvature, but a
trained finite model need not improve every metric or become globally optimal transport.
**Implementation:** `reflow_checkpoint` in [train.py](../diffusion_lab/train.py),
[recipe](../configs/experiment/gmm_reflow.yaml).
""",
            ),
            (
                "code",
                """
z0,z1 = initial[0],initial[1]
fractions = torch.linspace(0,1,11)
latent_line = (1-fractions[:,None])*z0+fractions[:,None]*z1
decoded,_ = integrate(flow,latent_line,torch.linspace(0,1,51),"heun")
fig,ax = plt.subplots()
scatter(ax,reference,"Noise interpolation through the learned flow")
ax.plot(decoded[:,0],decoded[:,1],"o-",color="crimson")
for i,p in enumerate(decoded):
    ax.annotate(str(i),p.numpy(),fontsize=8)
plt.show()
# Chord deviation: zero only for a trajectory linear in this time parameter.
tau = torch.linspace(0,1,len(learned_history))[:,None,None]
chord = (1-tau)*learned_history[0]+tau*learned_history[-1]
print("Mean squared trajectory-to-chord deviation:",(learned_history-chord).square().sum(-1).mean().item())
""",
            ),
            (
                "md",
                r"""
## 5. Perform one small reflow round

Generate endpoint pairs with the fixed learned teacher, then fit a new linear
pair-path velocity. The source endpoints are known teacher initial noise;
the target endpoints come from integrating the teacher, not independent data draws.
Both fields below are evaluated on **new shared noise**. Compare curvature and
few-call samples instead of assuming rectification succeeds at a finite budget.
""",
            ),
            (
                "code",
                """
teacher_source = torch.randn(1024,2,generator=torch.Generator().manual_seed(46))
teacher_target,_ = integrate(flow,teacher_source,torch.linspace(0,1,51),"heun")
torch.manual_seed(47)
reflow = TimeMLP(width=64,depth=2)
reflow.condition = Conditioning(64,0,time_embedding="positional")
flow.condition = Conditioning(64,0,time_embedding="positional")
optimizer = torch.optim.Adam(reflow.parameters(),lr=.002)
g = torch.Generator().manual_seed(47)
for _step in range(800):
    index = torch.randint(len(teacher_source),(256,),generator=g)
    source,target = teacher_source[index],teacher_target[index]
    time = torch.rand(len(source),generator=g)
    state = (1-time[:,None])*source+time[:,None]*target
    loss = (reflow(state,time)-(target-source)).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
reflow.eval()
heldout_noise = torch.randn(256,2,generator=torch.Generator().manual_seed(48))
fig,axes = plt.subplots(1,2)
for ax,(network,title) in zip(axes,[(flow,"Original learned flow"),(reflow,"One short reflow round")],strict=True):
    _,trajectory = integrate(network,heldout_noise,torch.linspace(0,1,51),"heun")
    tau = torch.linspace(0,1,len(trajectory))[:,None,None]
    chord = (1-tau)*trajectory[0]+tau*trajectory[-1]
    print(title,"mean squared chord deviation:",(trajectory-chord).square().sum(-1).mean().item())
    few_calls,_ = integrate(network,heldout_noise,torch.linspace(0,1,6),"euler")
    scatter(ax,few_calls,title+": 5 Euler calls")
plt.tight_layout(); plt.show()
""",
            ),
            (
                "code",
                """
def explore_solver(steps=10,method="heun"):
    generated,_ = integrate(flow,initial,torch.linspace(0,1,steps+1),method)
    _fig,ax = plt.subplots(); scatter(ax,generated,f"Learned flow: {method}, {steps} steps"); plt.show()
if ENABLE_WIDGETS:
    from ipywidgets import interact
    interact(explore_solver,steps=(2,60,2),method=["euler","heun","rk4"])
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Explain why straight pair paths do not imply a constant learned velocity.
2. Compare Euler 20 steps and Heun 10 steps at equal call budget.
3. Change the pair path to trigonometric and use its derivatives in the target.

<details><summary>Worked answers</summary>

1. Compatible endpoint pairs depend on the observed state and time; their conditional average changes.
2. Both use 20 calls here. Compare field error and endpoint distributions; higher order only helps when field smoothness and model accuracy permit it.
3. For $a=\sin(\pi\tau/2)$ and $s=\cos(\pi\tau/2)$, target is $a'X+s'Z$, not $X-Z$.

</details>

**Animation:** [Learned flow](../assets/teaching/notebooks/03_learned_flow.gif).
**Continue:** [Guidance](04_guidance.ipynb), then [OT/interpolants/bridges](07_interpolants_transport_and_bridges.ipynb).
**Sources:** [Flow matching](https://arxiv.org/abs/2210.02747), [rectified flow](https://arxiv.org/abs/2209.03003).
""",
            ),
        ],
    },
    {
        "name": "04_guidance.ipynb",
        "title": "04 · Conditioning, CFG and classifier guidance",
        "intro": "Use an exactly known class posterior to understand guidance before assessing learned classifiers or text encoders.",
        "prerequisites": "01–03; Bayes' rule and gradients of log probabilities.",
        "outcomes": "Verify the conditional-score identity; compare CG/CFG; inspect null-label training and branch costs; interpret guidance strength and classifier calibration.",
        "cells": [
            (
                "md",
                r"""
## 1. Bayes connects a classifier to a score

$$\nabla_x\log p_t(x|y)=s_t(x)+\nabla_x\log p_t(y|x).$$
Classifier guidance (CG) uses $s_{\rm CG}=s_u+w\nabla\log p_\phi(y|x,t)$.
A classifier trained only on clean images need not estimate a noisy posterior.
Classifier-free guidance (CFG) mixes $s_u+w(s_c-s_u)$, or mixes equivalent
linear prediction coordinates at a fixed time.
Here $w=0$ is null, $w=1$ ordinary conditioning, $w>1$ extrapolation.
Some papers/libraries instead write $(1+w)s_c-ws_u$; always check conventions.
**Code:** `classifier_gradient`, `classifier_free`, `AnalyticGMMClassifier` in [guidance.py](../diffusion_lab/guidance.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import GaussianPath,expand
from diffusion_lab.guidance import AnalyticGMMClassifier,classifier_gradient,classifier_free,drop_labels
from diffusion_lab.solvers import integrate
mixture,path = GaussianMixture(),GaussianPath()
classifier = AnalyticGMMClassifier(mixture,path)
x = torch.randn(96,2)
t = torch.full((len(x),),.3)
labels = torch.arange(len(x)) % mixture.components
a,s = path.coefficients(t)
unconditional = mixture.score(x,a,s)
conditional = mixture.score(x,a,s,labels)
gradient = classifier_gradient(classifier,x,t,labels)
torch.testing.assert_close(unconditional+gradient,conditional,atol=2e-5,rtol=2e-5)
print("Bayes conditional-score identity verified against a noisy-posterior oracle.")
""",
            ),
            (
                "md",
                r"""
## 2. Learn a null branch through condition dropout

A class-conditioned model has `K+1` label embeddings; index `K` is null.
Randomly replacing labels teaches unconditional and conditional fields in one
network. `labels=None` selects the null embedding at inference.
Dropping every label prevents learning class-specific behavior; dropping none
leaves the null branch unsupervised.
**Code:** `Conditioning` in [models.py](../diffusion_lab/models.py), `drop_labels` in [guidance.py](../diffusion_lab/guidance.py).
""",
            ),
            (
                "code",
                """
labels = torch.arange(10000) % 8
dropped = drop_labels(labels,8,.15,torch.Generator().manual_seed(51))
print("Observed null-label fraction:",(dropped==8).float().mean().item())
assert set(dropped.unique().tolist()) == set(range(9))
""",
            ),
            (
                "md",
                r"""
## 3. Hold initial noise fixed while changing guidance

The exact GMM score below isolates the mechanism. Use the actual finite-time
unconditional marginal at $t=.999$. Its small difference from a conditional
initial marginal is disclosed; strong guidance also changes the intended law.
The plot does not assess a trained noisy classifier.
""",
            ),
            (
                "code",
                """
class ExactScore(torch.nn.Module):
    def forward(self,x,t,labels=None):
        return mixture.score(x,*path.coefficients(t),labels)
g = torch.Generator().manual_seed(52)
clean,_ = mixture.sample(384,g)
a,s = path.coefficients(torch.full((len(clean),),.999))
initial = expand(a,clean)*clean+expand(s,clean)*torch.randn(clean.shape,generator=g)
requested = torch.zeros(len(initial),dtype=torch.long)
scales = [0.,1.,3.]
fig,axes = plt.subplots(1,3,figsize=(13,4))
generated_by_scale = {}
for ax,scale in zip(axes,scales,strict=True):
    def field(x,t,scale=scale):
        f,g = path.sde(x,t)
        guided = classifier_free(ExactScore(),x,t,requested,scale)
        return f-.5*expand(g.square(),x)*guided
    generated,_ = integrate(field,initial,torch.linspace(.999,.005,251),"heun")
    generated_by_scale[scale] = generated
    scatter(ax,generated,f"Requested class 0, CFG={scale:g}")
    ax.scatter(*mixture.means[0],marker="*",s=150,c="crimson")
plt.tight_layout(); plt.show()
for scale,generated in generated_by_scale.items():
    print(scale,"distance to requested mean:",torch.linalg.vector_norm(generated-mixture.means[0],dim=1).mean().item())
""",
            ),
            (
                "code",
                """
# At an identical state/time, exact CG and exact CFG fields coincide.
state = torch.randn(64,2); time = torch.full((64,),.4)
requested = torch.full((64,),2,dtype=torch.long)
for scale in [0.,1.,3.]:
    cfg = classifier_free(ExactScore(),state,time,requested,scale)
    cg = ExactScore()(state,time)+scale*classifier_gradient(classifier,state,time,requested)
    torch.testing.assert_close(cfg,cg,atol=2e-5,rtol=2e-5)
print("CG=CFG with exact compatible fields; learned estimation errors need not match.")
""",
            ),
            (
                "md",
                r"""
## 4. Count calls, not just solver steps

At $w=0$ or $1$, the implementation uses one branch. For other scales with a
condition it uses two. A Heun step with extrapolated CFG therefore uses four
denoiser branch calls. Batched branches may share one Python invocation but
still require two examples' worth of denoiser work. CG adds classifier forward/backward work.
CLIP energy on a denoised estimate is a separate heuristic, not a calibrated noisy posterior.
**Code:** [image_guidance.py](../diffusion_lab/image_guidance.py).
""",
            ),
            (
                "code",
                """
class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0
    def forward(self,x,t,labels=None):
        self.calls += 1
        return torch.zeros_like(x)
for scale in [0.,1.,3.]:
    counted = Counter()
    classifier_free(counted,state,time,requested,scale)
    print(f"CFG={scale:g}: {counted.calls} denoiser branch calls")
""",
            ),
            (
                "code",
                """
def explore_guidance(scale=1.,class_id=0):
    labels = torch.full((len(initial),),class_id,dtype=torch.long)
    def guided_field(x,t):
        f,g = path.sde(x,t)
        score = classifier_free(ExactScore(),x,t,labels,scale)
        return f-.5*expand(g.square(),x)*score
    samples,_ = integrate(guided_field,initial,torch.linspace(.999,.005,251),"heun")
    _fig,ax = plt.subplots(); scatter(ax,samples,f"Class {class_id}, CFG={scale:g}"); plt.show()
if ENABLE_WIDGETS:
    from ipywidgets import interact
    interact(explore_guidance,scale=(0.,4.,.5),class_id=(0,7))
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Can strong CFG improve prompt adherence and simultaneously hurt diversity?
2. Why must a classifier be evaluated at the same noise convention as the sampler?
3. Replace class 0 with class 4 and reuse the initial noise. Inspect the sample cloud, not only its mean.

<details><summary>Worked answers</summary>

1. Yes. It extrapolates a field and can concentrate mass, distort geometry or reduce mode coverage.
2. Otherwise its gradient estimates the wrong conditional law. EDM classifiers here receive `log(sigma)/4`; discrete diffusion uses its normalized table time.
3. Mean agreement alone can hide variance collapse and missing modes. Use held-out classification and distribution diagnostics when studying learned image models.

</details>

**Image evidence:** [Guidance chapter](../docs/04_guidance.md), [results](../docs/14_advanced_results.md).
**Continue:** [Likelihood and evaluation](05_likelihood_and_evaluation.ipynb).
**Sources:** [CG](https://arxiv.org/abs/2105.05233), [CFG](https://arxiv.org/abs/2207.12598).
""",
            ),
        ],
    },
    {
        "name": "05_likelihood_and_evaluation.ipynb",
        "title": "05 · Full DDPM ELBO, ODE likelihood and distribution evaluation",
        "intro": "Work through each likelihood term, inspect the learned variance head, and test metrics on deliberately flawed generators.",
        "prerequisites": "01–03; Gaussian KL, log densities and change of variables.",
        "outcomes": "Decompose the complete ELBO; distinguish a bound from NLL; check divergence signs and probe variance; diagnose mode collapse beyond FID/IS.",
        "cells": [
            (
                "md",
                r"""
## 1. The full DDPM negative ELBO

$$\mathcal L_{\rm VLB}=\underbrace{D_{KL}(q(x_T|x_0)\|p(x_T))}_{L_T}
+\underbrace{\sum_{k=2}^T\mathbb E_qD_{KL}(q(x_{k-1}|x_k,x_0)\|p_\theta(x_{k-1}|x_k))}_{\text{transition sum}}
-\underbrace{\mathbb E_q\log p_\theta(x_0|x_1)}_{\text{decoder log likelihood}}.$$
Divide nats by $D\log2$ for bits/dimension. It bounds NLL from above in expectation;
a noisy single Monte Carlo estimate is not an exact likelihood or a pointwise guaranteed bound.
The continuous toy decoder uses a Gaussian density; 8-bit images use discretized
Gaussian bin probabilities, with special treatment at pixel boundaries.
The first decoder term and terminal prior must not be omitted.

**Equation → code:** `ddpm_elbo`, `prior_kl_bpd` in [likelihood.py](../diffusion_lab/likelihood.py);
`variational_bound` and tail-stable decoder in [improved_ddpm.py](../diffusion_lab/improved_ddpm.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import DiscreteVP,expand
from diffusion_lab.likelihood import ddpm_elbo
mixture = GaussianMixture()
path = DiscreteVP(32,"cosine")
class OracleNoise(torch.nn.Module):
    def forward(self,x,t,labels=None):
        a,s = path.coefficients(t)
        return -expand(s,x)*mixture.score(x,a,s,labels)
clean,_ = mixture.sample(64,torch.Generator().manual_seed(61))
full = ddpm_elbo(OracleNoise(),path,clean,quantized=False,clip_x0=False,
                 generator=torch.Generator().manual_seed(62))
parts = ["terminal_prior_kl_bpd","decoder_bpd","transition_kl_bpd"]
torch.testing.assert_close(full["negative_elbo_bpd"],sum(full[key] for key in parts))
fig,ax = plt.subplots()
ax.bar(["Terminal prior","Continuous decoder","Transition sum"],[full[key].mean().item() for key in parts])
ax.set(ylabel="Mean bits/dimension",title="All DDPM ELBO terms, 32-step toy oracle")
plt.show()
print("Mean total:",full["negative_elbo_bpd"].mean().item(),"calls:",full["network_calls"])
print("Exact NLL?",full["is_exact_nll"],"—",full["estimator"])
""",
            ),
            (
                "md",
                r"""
## 2. Learn diagonal variance and reweight objectives

Improved DDPM predicts a per-coordinate variance head, not a dense covariance.
Its learned-range convention is
$$\log\Sigma_\theta=v\log\beta_k+(1-v)\log\widetilde\beta_k,\quad
v=(h_\theta+1)/2.$$
The published parameterization does **not** clamp $h_\theta$.
Hybrid training combines simple MSE with a small weighted VLB term, detaching
the mean prediction inside that VLB term so variance supervision does not
change its mean branch gradient. A pure VLB objective uses a different gradient path.
Loss-second-moment timestep sampling changes sampling probabilities and uses
importance weights; it is not just changing the beta schedule.
**Code:** `learned_log_variance`, `improved_loss`, `LossSecondMomentSampler` in [improved_ddpm.py](../diffusion_lab/improved_ddpm.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.improved_ddpm import posterior,learned_log_variance,variational_bound
index = torch.full((len(clean),),12,dtype=torch.long)
t = (index.float()+1)/path.steps
a,s = path.coefficients(t)
eps = torch.randn_like(clean)
noisy = expand(a,clean)*clean+expand(s,clean)*eps
_, minimum = posterior(path,clean,noisy,index)
maximum = path.betas[index].log()
fig,ax = plt.subplots()
values = torch.linspace(-2,2,100)[:,None].repeat(1,2)
interpolated = learned_log_variance(values,minimum[:1].repeat(100),maximum[:1].repeat(100))
ax.plot(values[:,0],interpolated[:,0]); ax.axvline(-1,ls=":"); ax.axvline(1,ls=":")
ax.set(xlabel="Raw variance-head output h",ylabel="Predicted log variance",title="Learned range permits extrapolation")
plt.show()
estimate = torch.randn_like(clean,requires_grad=True)
variance_head = torch.zeros_like(clean,requires_grad=True)
vb = variational_bound(path,clean,noisy,index,estimate,variance_head,detach_mean=True).mean()
vb.backward()
assert estimate.grad is None
assert variance_head.grad is not None and torch.isfinite(variance_head.grad).all()
print("Detached mean: no VLB mean gradient; variance gradient norm:",variance_head.grad.norm().item())
""",
            ),
            (
                "code",
                """
sampled_means = []
for seed in range(10):
    estimate = ddpm_elbo(OracleNoise(),path,clean,quantized=False,clip_x0=False,timestep_samples=8,
                         generator=torch.Generator().manual_seed(70+seed))
    sampled_means.append(estimate["negative_elbo_bpd"].mean().item())
plt.plot(sampled_means,"o",label="8 sampled transition terms, independent MC draws")
plt.axhline(full["negative_elbo_bpd"].mean().item(),color="black",label="One all-term MC estimate")
plt.xlabel("Monte Carlo repetition"); plt.ylabel("Estimated negative ELBO bpd"); plt.legend(fontsize=8); plt.show()
print("Each sampled run uses",estimate["network_calls"],"calls: one decoder plus eight transitions.")
""",
            ),
            (
                "md",
                r"""
## 3. ODE likelihood: get the sign right

Along $\dot x=u$, $d\log p_t(x_t)/dt=-\nabla\cdot u$.
Integrating a datum toward a known base gives
$$\log p_{\rm data}(x)=\log p_{\rm base}(x_T)+\int\nabla\cdot u(x_t,t)dt.$$
For $u=ax$ over unit time and a unit Gaussian terminal law, the initial
standard deviation is $e^{-a}$. This analytic case isolates numerical error.
**Code:** `ode_log_density`, `field_and_divergence` in [likelihood.py](../diffusion_lab/likelihood.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.likelihood import ode_log_density,gaussian_log_density,field_and_divergence
x = torch.randn(64,2,generator=torch.Generator().manual_seed(81),dtype=torch.float64)
exact = gaussian_log_density(x,std=math.exp(-.4))
fig,ax = plt.subplots()
for method in ["euler","heun","rk4"]:
    costs,errors = [],[]
    for steps in [2,5,10,20,40]:
        estimated,metadata = ode_log_density(lambda x,t:.4*x,x,torch.linspace(0,1,steps+1,dtype=x.dtype),
                                             gaussian_log_density,solver=method,trace="exact")
        costs.append(metadata["field_calls"]); errors.append((estimated-exact).abs().mean().item())
    ax.loglog(costs,errors,"o-",label=method)
ax.set(xlabel="Field/divergence evaluations",ylabel="Mean absolute log-density error"); ax.legend(); plt.show()
assert errors[-1] < 1e-8
""",
            ),
            (
                "md",
                r"""
## 4. Stochastic traces have their own uncertainty

Hutchinson uses $\mathbb E_z[z^TJz]=\operatorname{tr}J$, with zero-mean,
unit-covariance probes. The implementation uses Rademacher probes, held fixed
through each ODE solve. Resampling at every RHS evaluation would create a noisy
ODE right-hand side. Probe repetitions differ from sampling seeds or training seeds.
""",
            ),
            (
                "code",
                """
matrix = torch.tensor([[.4,1.2],[.8,-.1]],dtype=torch.float64)
def field(x,t):
    return x@matrix.T
t = torch.zeros(len(x),dtype=x.dtype)
_,exact_trace = field_and_divergence(field,x,t)
counts = [1,4,16,64]
deviations = []
for count in counts:
    g = torch.Generator().manual_seed(82)
    probes = (2*torch.randint(2,(count,*x.shape),generator=g)-1).to(x)
    _,estimated_trace = field_and_divergence(field,x,t,probes)
    deviations.append((estimated_trace-exact_trace).square().mean().sqrt().item())
plt.loglog(counts,deviations,"o-"); plt.xlabel("Probe count"); plt.ylabel("Trace RMSE across examples"); plt.show()
print("Exact divergence:",matrix.trace().item())
""",
            ),
            (
                "md",
                r"""
## 5. Metrics respond differently to collapse and outliers

MMD compares kernel means; sliced $W_2$ compares projected quantiles.
Precision asks whether generated features lie in the reference manifold;
recall asks how much reference support is represented. Density/coverage add
neighborhood information. Results depend on feature space, counts and $k$.
Below the **features are 2D coordinates**, not Inception embeddings; these are not image FID scores.
Unbiased MMD² can be negative. IS uses generated class predictions and does not
compare against real images. FID matches Gaussian feature moments and can miss
differences beyond them. No single score proves good images or non-memorization.
**Code:** [metrics.py](../diffusion_lab/metrics.py), [distribution_metrics.py](../diffusion_lab/distribution_metrics.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.metrics import mmd_squared,sliced_wasserstein
from diffusion_lab.distribution_metrics import manifold_metrics
g = torch.Generator().manual_seed(83)
real,_ = mixture.sample(256,g)
good,_ = mixture.sample(256,g)
collapsed,_ = mixture.sample(256,g,labels=torch.zeros(256,dtype=torch.long))
outliers = good+torch.tensor([4.,0.])
fig,axes = plt.subplots(1,3,figsize=(14,4))
rows = []
for ax,(name,samples) in zip(axes,[("Independent data",good),("One-mode collapse",collapsed),("Shifted generator",outliers)],strict=True):
    scatter(ax,samples,name,limits=(-8,8))
    row = manifold_metrics(real,samples,k=3)
    row.update(name=name,mmd2=mmd_squared(real,samples),sliced_w2=sliced_wasserstein(real,samples,64))
    rows.append(row)
plt.tight_layout(); plt.show()
for row in rows:
    print({key:round(value,4) if isinstance(value,float) else value for key,value in row.items()
           if key in ["name","precision","recall","density","coverage","mmd2","sliced_w2"]})
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Which DDPM term is missing if you average only noise MSE over time?
2. Does increasing ODE steps remove Hutchinson error? Does increasing probes remove discretization error?
3. Can a collapsed model have high precision but low recall?

<details><summary>Worked answers</summary>

1. Noise MSE is a surrogate with different time weights; it is not the decoder + transition KL sum + terminal prior decomposition.
2. Neither eliminates the other source of error. Evaluate grid, probe and endpoint sensitivity separately.
3. Yes: plausible samples from one mode may lie in the reference manifold while failing to cover other modes.

</details>

**Actual reports:** [Likelihood](../docs/16_likelihood_and_elbo.md), [evaluation](../docs/19_evaluation_beyond_fid.md),
[seed studies](../docs/25_evidence_studies.md), [latest measured failures](../docs/28_scratch_and_refinement.md).
**Continue:** [Text conditioning](06_text_cross_attention.ipynb).
**Sources:** [Improved DDPM](https://arxiv.org/abs/2102.09672), [density/coverage](https://proceedings.mlr.press/v119/naeem20a.html).
""",
            ),
        ],
    },
    {
        "name": "06_text_cross_attention.ipynb",
        "title": "06 · Text tokens, cross-attention, LoRA and text CFG",
        "intro": "Trace tensor shapes through token conditioning, learn an explicitly synthetic attention exercise, and inspect the actual COCO results without confusing attention mechanics with semantic success.",
        "prerequisites": "04; matrix products and attention basics. No pretrained models needed for default cells.",
        "outcomes": "Check token masking; distinguish spatial queries and text keys; train a toy attention task; inspect frozen/trainable boundaries and text CFG; assess real prompt failures.",
        "cells": [
            (
                "md",
                r"""
## 1. Cross-attention is a tensor operation

Image features `[B,C,H,W]` become $HW$ queries; text features `[B,L,D]`
become keys and values. With $h$ heads and $d=C/h$,
$$A=\operatorname{softmax}(QK^T/\sqrt d+M),\qquad O=AV.$$
Per-head $A$ has shape `[B,h,HW,L]`. The inspectable returned weights average
heads and reshape to `[B,H,W,L]`. The output is a residual image update.
Padding gets $-\infty$ logits. An all-masked sequence is rejected.
**Code:** `TextCrossAttention` in [text.py](../diffusion_lab/text.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.text import TextCrossAttention,LowRankLinear,text_guided_prediction,TextUNet
torch.manual_seed(91)
layer = TextCrossAttention(8,6,heads=2)
features = torch.randn(1,8,4,4)
tokens = torch.randn(1,5,6)
mask = torch.tensor([[True,True,True,True,False]])
output,weights = layer(features,tokens,mask,return_weights=True)
print("features / tokens / output / weights:",features.shape,tokens.shape,output.shape,weights.shape)
assert (weights[...,4]==0).all()
torch.testing.assert_close(weights.sum(-1),torch.ones(1,4,4))
changed = tokens.clone(); changed[:,4] = 1e6
torch.testing.assert_close(output,layer(features,changed,mask))
untrained_weights = weights.detach().clone()
""",
            ),
            (
                "md",
                r"""
## 2. Learn a tiny attention task and label its scope

We assign four arbitrary token vectors to four quadrants and train attention
weights with explicit quadrant supervision. This lets us see attention become
structured. **It does not train diffusion or prove word grounding**: the token
names below are artificial category names; no language encoder or image-caption data is used.
The real diffusion model receives supervision through the denoising objective,
not hand-labeled attention heatmaps.
""",
            ),
            (
                "code",
                """
target_tokens = torch.tensor([[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,3]])
optimizer = torch.optim.Adam(layer.parameters(),lr=.025)
losses = []
for _step in range(200):
    _,weights = layer(features,tokens,mask,return_weights=True)
    selected = weights[0].gather(-1,target_tokens[...,None]).squeeze(-1)
    loss = -selected.clamp_min(1e-12).log().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
_,trained_weights = layer(features,tokens,mask,return_weights=True)
fig,axes = plt.subplots(2,4,figsize=(11,5))
for row,(weight,title) in enumerate([(untrained_weights,"Untrained"),(trained_weights.detach(),"Synthetic supervised")]):
    for i in range(4):
        axes[row,i].imshow(weight[0,:,:,i],vmin=0,vmax=1,cmap="viridis")
        axes[row,i].set_title(f"{title}: token {i}"); axes[row,i].axis("off")
plt.tight_layout(); plt.show()
print("Toy attention CE before / after:",losses[0],losses[-1])
""",
            ),
            (
                "md",
                r"""
## 3. Time-conditioned text U-Net integration

Our compact `TextUNet` inherits residual blocks, normalization modulation and
time embeddings from the image U-Net, inserting cross-attention at encoder,
middle and decoder levels. A text sequence is not reduced to one class ID.
The output is in the trainer's native epsilon or VP-v coordinate; it is not an RGB image when working in latents.
This random small network demonstrates shape/gradient integration, not generation quality.
""",
            ),
            (
                "code",
                """
network = TextUNet(text_dim=6,width=16,channel_mult=(1,2),blocks_per_level=1,
                   attention_levels=(),channels=4,time_embedding="positional",time_scale=1000.)
latent = torch.randn(2,4,8,8)
context = torch.randn(2,5,6)
valid = mask.repeat(2,1)
prediction = network(latent,torch.tensor([.2,.7]),context,valid)
assert prediction.shape == latent.shape
loss = (prediction-torch.randn_like(prediction)).square().mean()
loss.backward()
print("Native latent prediction:",prediction.shape,"trainable parameters:",sum(p.numel() for p in network.parameters()))
""",
            ),
            (
                "md",
                r"""
## 4. LoRA adapts a frozen projection

$$y=Wx+(\alpha/r)BAx.$$
$W$ stays frozen, $A$ starts random, $B$ starts zero. Initially the adapter
exactly preserves the base function. At the first update, $B$ can receive
gradient even though $A$ initially receives zero through $B$.
LoRA adapts a pretrained denoiser; a random compact denoiser is a separate experiment.
**Code:** `LowRankLinear`, `inject_lora` in [text.py](../diffusion_lab/text.py).
""",
            ),
            (
                "code",
                """
base = torch.nn.Linear(6,8)
adapter = LowRankLinear(base,rank=2,alpha=2)
x = torch.randn(10,6)
torch.testing.assert_close(adapter(x),base(x),atol=0,rtol=0)
original = base.weight.detach().clone()
optimizer = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad],lr=.1)
adapter(x).square().mean().backward()
print("Initial A / B gradient norms:",adapter.a.weight.grad.norm().item(),adapter.b.weight.grad.norm().item())
optimizer.step()
torch.testing.assert_close(base.weight,original,atol=0,rtol=0)
print("Adapter changed function:",not torch.equal(adapter(x),base(x)))
""",
            ),
            (
                "md",
                r"""
## 5. Text CFG uses an encoded empty prompt

The null branch receives the text encoder's empty-prompt sequence (including
special tokens), **not an arbitrary all-zero tensor**. Mix native predictions
$y_u+w(y_c-y_u)$ before scheduler conversion. Caption dropout teaches the null
branch. Epsilon versus VP-v, latent scaling and scheduler beta convention
must agree between training and sampling.
**Code:** `text_guided_prediction` in [text.py](../diffusion_lab/text.py),
[text_train.py](../diffusion_lab/text_train.py), [text_evaluate.py](../diffusion_lab/text_evaluate.py).
""",
            ),
            (
                "code",
                """
# Synthetic contexts isolate the mixing identity; they are not real encoded prompts.
state = torch.randn(2,4,8,8); time = torch.ones(2)*.4
conditional_context = torch.ones(2,3,6); synthetic_null = torch.zeros_like(conditional_context)
def predict(x,t,context):
    return x+context.mean((1,2))[:,None,None,None]
for scale in [0.,1.,3.]:
    guided,calls = text_guided_prediction(predict,state,time,conditional_context,synthetic_null,scale)
    torch.testing.assert_close(guided,state+scale)
    print("scale / branch calls:",scale,calls)
""",
            ),
            (
                "md",
                r"""
## 6. Inspect real evidence, including failures

The following is the saved 100k-update **randomly initialized denoiser**, with
frozen pretrained VAE/text encoder, not an all-components-from-scratch model.
Prompt order: red bus, blue bus, two dogs, one dog; apples, oranges, cyclist
beside a red car, red car beside a cyclist. Vehicle color and fruit category
respond, while anatomy, counting and cyclist relationships fail visibly.
The held-out report uses 2,048 real/generated examples, 256px, CFG 7.5, 30 DDIM
steps. CLIP similarity is a coarse alignment diagnostic, not an anatomy/counting guarantee.
""",
            ),
            (
                "code",
                """
saved_gallery("assets/studies/scratch_refinement/text_step0100000_prompts/samples.png",
              "Actual COCO scratch-denoiser prompt contrasts, 100k updates")
report = json.loads((project/"assets/studies/scratch_refinement/text_final_heldout/metrics.json").read_text())
keys = ["training_steps","generated_count","real_count","cfg_scale","steps","fid_2048","inception_score_mean","clip_cosine_mean"]
print({key:report[key] for key in keys})
""",
            ),
            (
                "md",
                r"""
## Exercises, answers and GPU extension

1. Why does masking padding affect keys but not remove image queries?
2. Does changing a pretrained VAE into a frozen component make the whole system trained from scratch?
3. Evaluate red/blue contrasts with shared noise, then counting and spatial relations separately.

<details><summary>Worked answers</summary>

1. Padding is a property of the text sequence. Every valid spatial location still needs to attend to valid text tokens.
2. No. State explicitly which components were randomly initialized and which were externally pretrained.
3. A mean CLIP increase can coexist with wrong counts or geometry. Use contrast diagnostics and direct visual inspection; split captions by image ID and disclose unknown external pretraining overlap.

</details>

Read [COCO commands and splits](../docs/24_text_conditioning.md) and
[scratch recipe](../configs/text_coco_scratch.yaml). On your cluster use
[text Slurm template](../slurm/train_coco_scratch.sbatch). These commands train
image-scale models separately; the notebook never silently starts a long GPU job.
**Continue:** [Interpolants/bridges](07_interpolants_transport_and_bridges.ipynb).
**Sources:** [Latent diffusion](https://arxiv.org/abs/2112.10752), [LoRA](https://arxiv.org/abs/2106.09685).
""",
            ),
        ],
    },
    {
        "name": "07_interpolants_transport_and_bridges.ipynb",
        "title": "07 · Stochastic interpolants, optimal transport and bridges",
        "intro": "Connect endpoint coupling choices to path geometry, then add noise and distinguish an interpolant, an SDE and a Brownian bridge.",
        "prerequisites": "02–03; conditional expectation, continuity and Gaussian covariance.",
        "outcomes": "Construct interpolant targets with derivatives; check equal-marginal fields; solve finite OT/Sinkhorn problems; simulate the joint Brownian bridge law; understand bridge-matching scope.",
        "cells": [
            (
                "md",
                r"""
## 1. A general stochastic interpolant

For an endpoint coupling $(X_s,X_d)\sim\pi$ and independent $Z\sim N(0,I)$,
$$I_\tau=a(\tau)X_s+b(\tau)X_d+\gamma(\tau)Z,$$
with $(a,b)=(1,0)$ at 0 and $(0,1)$ at 1, $\gamma(0)=\gamma(1)=0$.
Velocity regression targets $a'X_s+b'X_d+\gamma'Z$; its conditional average
is $v_\tau(x)$. Noise regression gives $s_\tau(x)=-\mathbb E[Z|I_\tau=x]/\gamma$ inside the interval.
For any nonnegative diffusion strength $e(\tau)$,
$$dX_\tau=[v_\tau+e s_\tau]d\tau+\sqrt{2e}\,dW_\tau$$
has the same continuity equation as $\dot X=v$, assuming exact fields and matching initial law.
This code uses $e=\kappa\gamma^2$ to avoid singular endpoint divisions in the learned drift.
**Code:** `StochasticInterpolant.targets`, `.learned_fields`, `GaussianInterpolantOracle` in [interpolants.py](../diffusion_lab/interpolants.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.interpolants import StochasticInterpolant,GaussianInterpolantOracle
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import expand
from diffusion_lab.solvers import integrate
source_law = GaussianMixture(components=1,radius=0,std=1)
target_law = GaussianMixture(components=8,radius=3,std=.3)
interpolant = StochasticInterpolant(path="linear",amplitude=.5)
g = torch.Generator().manual_seed(101)
source,_ = source_law.sample(320,g)
target,_ = target_law.sample(320,g)
t = torch.rand(len(source),generator=g)*.98+.01
state,velocity_target,noise = interpolant.targets(source,target,t,g)
a,b,gamma,da,db,dgamma = interpolant.coefficients(t)
torch.testing.assert_close(state,expand(a,source)*source+expand(b,target)*target+expand(gamma,noise)*noise)
torch.testing.assert_close(velocity_target,expand(da,source)*source+expand(db,target)*target+expand(dgamma,noise)*noise)
grid = torch.linspace(0,1,100)
fig,ax = plt.subplots()
for values,label in zip(interpolant.coefficients(grid)[:3],["source a","data b","noise gamma"],strict=True):
    ax.plot(grid,values,label=label)
ax.set(xlabel="Generation time",title="Linear stochastic interpolant"); ax.legend(); plt.show()
""",
            ),
            (
                "md",
                r"""
## 2. Matching ODE/SDE marginals, different trajectories

The analytic oracle enumerates Gaussian endpoint components for **affine**
interpolants and computes marginal fields exactly. It cannot evaluate general
nonlinear endpoint paths. Use its ODE and SDE to isolate numerical/Monte Carlo error.
Each method below starts from the same actual source samples; SDE paths add Brownian noise.
""",
            ),
            (
                "code",
                """
oracle = GaussianInterpolantOracle(source_law,target_law,interpolant)
times = torch.linspace(0,1,251)
ode,ode_history = integrate(lambda x,t:oracle.fields(x,t)[0],source,times,"heun")
sde,sde_history = integrate(lambda x,t:oracle.drift(x,t,strength=1),source,times,"euler_maruyama",
                             lambda t:oracle.diffusion(t,strength=1),torch.Generator().manual_seed(102))
fig,axes = plt.subplots(1,3,figsize=(14,4))
scatter(axes[0],source,"Source Gaussian")
scatter(axes[1],ode,"Oracle interpolant ODE")
scatter(axes[2],sde,"Matching interpolant SDE")
plt.tight_layout(); plt.show()
animate_particles(sde_history[:,:150],"07_stochastic_interpolant","Stochastic interpolant SDE")
""",
            ),
            (
                "md",
                r"""
## 3. Nonlinear paths need the correct derivative

A custom $I(\tau,x_s,x_d)$ can bend endpoint-pair curves while preserving
endpoints. `targets` uses a Jacobian-vector product in time; manually keeping
the linear target after changing the path would supervise the wrong velocity.
The example is a hand-designed nonlinear path, **not a learned optimized path**.
""",
            ),
            (
                "code",
                """
def nonlinear_base(source,target,t):
    time = t[:,None]
    bend = torch.stack([torch.zeros_like(t),torch.sin((target-source)[:,0])],dim=1)
    return (1-time)*source+time*target+2*time*(1-time)*bend
nonlinear = StochasticInterpolant(amplitude=.2,base_fn=nonlinear_base)
torch.testing.assert_close(nonlinear_base(source,target,torch.zeros(len(source))),source)
torch.testing.assert_close(nonlinear_base(source,target,torch.ones(len(source))),target)
state,velocity,noise = nonlinear.targets(source,target,t,torch.Generator().manual_seed(103))
h = 1e-3
finite_difference = (nonlinear_base(source,target,t+h)-nonlinear_base(source,target,t-h))/(2*h)
dgamma = nonlinear.coefficients(t)[-1]
torch.testing.assert_close(velocity-expand(dgamma,noise)*noise,finite_difference,atol=6e-4,rtol=6e-4)
fig,ax = plt.subplots()
curves = torch.stack([nonlinear_base(source[:12],target[:12],time.repeat(12)) for time in torch.linspace(0,1,50)])
ax.plot(curves[:,:,0],curves[:,:,1],alpha=.6); ax.set(title="Custom nonlinear pair paths",aspect="equal")
plt.show()
""",
            ),
            (
                "md",
                r"""
## 4. Exact empirical OT versus entropic OT

For uniform finite endpoints and $C_{ij}=\|x_i-y_j\|^2/2$,
exact OT minimizes $\langle\pi,C\rangle$ subject to uniform marginals.
Equal counts admit a permutation/Hungarian assignment.
Sinkhorn minimizes $\langle\pi,C\rangle+\varepsilon KL(\pi\|a\otimes b)$.
Larger $\varepsilon$ generally spreads the coupling. It trades transport cost
against entropy; entropy-regularized cost is not unregularized Wasserstein distance.
Minibatch OT is a finite empirical choice, not proof of population-optimal learned transport.
**Code:** `exact_pairing`, `sinkhorn_plan`, `sample_coupling` in [transport.py](../diffusion_lab/transport.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.transport import exact_pairing,sinkhorn_plan,sample_coupling
g = torch.Generator().manual_seed(104)
left = torch.randn(40,2,generator=g)*torch.tensor([.15,1.])+torch.tensor([-2.,0.])
right = torch.randn(40,2,generator=g)*torch.tensor([.15,1.])+torch.tensor([2.,0.])
paired = exact_pairing(left,right)
cost = torch.cdist(left,right).square()/2
fig,axes = plt.subplots(1,3,figsize=(13,4))
for ax,endpoints,title in [(axes[0],right,"Independent pairing"),(axes[1],paired,"Exact empirical OT")]:
    for a,b in zip(left,endpoints,strict=True):
        ax.plot([a[0],b[0]],[a[1],b[1]],alpha=.3,color="steelblue")
    ax.scatter(*left.T,c="purple",s=12); ax.scatter(*right.T,c="crimson",s=12)
    ax.set(title=title,aspect="equal")
plan = sinkhorn_plan(left,right,epsilon=.5,iterations=3000)
axes[2].imshow(plan,cmap="magma"); axes[2].set(title="Entropic coupling, epsilon=.5",xlabel="Target index",ylabel="Source index")
plt.tight_layout(); plt.show()
torch.testing.assert_close(plan.sum(1),torch.full((40,),1/40),atol=1e-6,rtol=1e-6)
torch.testing.assert_close(plan.sum(0),torch.full((40,),1/40),atol=1e-6,rtol=1e-6)
print("Independent / exact / entropic transport costs:",
      (left-right).square().sum(1).mean().item()/2,(left-paired).square().sum(1).mean().item()/2,(plan*cost).sum().item())
""",
            ),
            (
                "md",
                r"""
## 5. Brownian bridges and the Schrödinger connection

For Brownian-reference variance rate $\varepsilon$ over unit time,
the conditional bridge between endpoints is
$$X_\tau=(1-\tau)x_s+\tau x_d+\sqrt\varepsilon(W_\tau-\tau W_1).$$
Its covariance is $\varepsilon(\min(s,t)-st)I$. At one time this gives noise
standard deviation $\sqrt{\varepsilon\tau(1-\tau)}$, but independent redraws of
this noise would not give the joint bridge law.
The conditional forward drift is $(x_d-x)/(1-\tau)$; bridge matching regresses
it and averages over endpoint posteriors to obtain a Markov drift.
For Brownian reference the endpoint kernel is $e^{-C/\varepsilon}$, explaining
the entropic-OT connection. Continuous Schrödinger bridges minimize path-space
KL to a reference process subject to endpoint marginals.
Our finite atomic endpoint construction uses a discrete kernel: it is not
a finite path-KL claim against unrestricted Wiener measure with atomic endpoints.
**Code:** [bridges.py](../diffusion_lab/bridges.py), [iterative neural projections](../diffusion_lab/dsbm.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.bridges import brownian_bridge_trajectory,bridge_training_pair
bridge_source,bridge_target = sample_coupling(left,right,plan,180,torch.Generator().manual_seed(105))
times = torch.linspace(0,1,51)
trajectory = brownian_bridge_trajectory(bridge_source,bridge_target,times,epsilon=.5,
                                        generator=torch.Generator().manual_seed(106))
torch.testing.assert_close(trajectory[0],bridge_source)
torch.testing.assert_close(trajectory[-1],bridge_target,atol=1e-6,rtol=1e-6)
fig,ax = plt.subplots()
ax.plot(trajectory[:,:12,0],trajectory[:,:12,1],alpha=.6,linewidth=.8)
ax.scatter(*bridge_source[:12].T,c="purple"); ax.scatter(*bridge_target[:12].T,c="crimson")
ax.set(title="Joint Brownian bridge trajectories",aspect="equal"); plt.show()
animate_particles(trajectory,"07_brownian_bridges","Entropic-coupled Brownian bridges",limits=(-4,4))
zeros = torch.zeros(8000,2)
bridge_noise = brownian_bridge_trajectory(zeros,zeros,torch.linspace(0,1,11),epsilon=.5,
                                          generator=torch.Generator().manual_seed(107))
empirical = (bridge_noise[3,:,0]*bridge_noise[7,:,0]).mean().item()
expected = .5*(.3-.3*.7)
print("Covariance at times .3/.7, empirical / analytic:",empirical,expected)
assert abs(empirical-expected) < .006
""",
            ),
            (
                "code",
                """
time = torch.full((len(bridge_source),),.4)
state,drift,velocity = bridge_training_pair(bridge_source,bridge_target,time,.5)
assert state.shape == drift.shape == velocity.shape == bridge_source.shape
print("Drift vs continuity velocity RMSE:",(drift-velocity).square().mean().sqrt().item())
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Show that drift $v+es$ and diffusion $\sqrt{2e}$ cancel their score/Laplacian contributions in Fokker–Planck.
2. Why should exact OT pairing be restricted within classes when preserving a class-independent noise prior?
3. Explain why a conditional bridge drift and a continuity velocity are different targets.

<details><summary>Worked answers</summary>

1. $-\nabla\cdot(esp)+e\Delta p=0$ because $sp=\nabla p$ and $e$ depends only on time.
2. Global assignment can make the paired noise distribution conditional on data geometry/class; class-wise matching preserves each empirical class's source-noise marginal.
3. Markov diffusion drift includes probability current plus a score contribution. For variance rate $\varepsilon$, continuity velocity is drift $-\varepsilon s/2$.

</details>

**Animations:** [Interpolant SDE](../assets/teaching/notebooks/07_stochastic_interpolant.gif), [bridges](../assets/teaching/notebooks/07_brownian_bridges.gif).
**Research extensions:** [Interpolant scope](../docs/17_stochastic_interpolants.md), [bridges](../docs/06_bridges.md), [iterative matching](../docs/11_iterative_bridges.md).
**Continue:** [EDM and consistency](08_edm_and_few_step_models.ipynb).
**Sources:** [Stochastic interpolants](https://arxiv.org/abs/2303.08797), [Sinkhorn](https://arxiv.org/abs/1306.0895), [DSBM](https://arxiv.org/abs/2303.16852).
""",
            ),
        ],
    },
    {
        "name": "08_edm_and_few_step_models.ipynb",
        "title": "08 · EDM preconditioning, consistency and few-step generation",
        "intro": "Separate a better numerical sampler from a model trained to jump across a trajectory. Inspect boundary conditions and the real refinement failures.",
        "prerequisites": "01–05; denoising, ODE integration and detached training targets.",
        "outcomes": "Explain EDM scales and its sigma clock; compare Karras grids with measured calls; check consistency boundaries; train a small consistency student and distinguish CT/CD/iCT/progressive distillation.",
        "cells": [
            (
                "md",
                r"""
## 1. EDM uses noise standard deviation as a coordinate

With $x=x_0+\sigma\epsilon$, EDM predicts a denoiser
$$D_\theta(x,\sigma)=c_{skip}x+c_{out}F_\theta(c_{in}x,c_{noise}),$$
$$c_{skip}=\frac{\sigma_d^2}{\sigma^2+\sigma_d^2},\quad
c_{out}=\frac{\sigma\sigma_d}{\sqrt{\sigma^2+\sigma_d^2}},\quad
c_{in}=\frac1{\sqrt{\sigma^2+\sigma_d^2}},\quad c_{noise}=\tfrac14\log\sigma.$$
Train with log-normal sigma and weight $\lambda(\sigma)=\sigma^{-2}+\sigma_d^{-2}$.
$\sigma_d$ is a data-scale design choice, not the noisy variance at every sigma.
**Code:** `EDMPreconditioner`, `edm_loss` in [edm.py](../diffusion_lab/edm.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.edm import EDMPreconditioner,karras_sigmas,edm_sample
from diffusion_lab.models import TimeMLP,Conditioning
sigma = torch.logspace(-3,2,200)
sd = .5
fig,axes = plt.subplots(1,2)
axes[0].semilogx(sigma,sd**2/(sigma**2+sd**2),label="c_skip")
axes[0].semilogx(sigma,sigma*sd/(sigma**2+sd**2).sqrt(),label="c_out")
axes[0].semilogx(sigma,(sigma**2+sd**2).rsqrt(),label="c_in")
axes[0].set(xlabel="Noise sigma",title="EDM scales"); axes[0].legend()
axes[1].loglog(sigma,sigma.reciprocal().square()+sd**-2)
axes[1].set(xlabel="Noise sigma",ylabel="Loss weight")
plt.tight_layout(); plt.show()
wrapper = EDMPreconditioner(TimeMLP(width=32,depth=1),sigma_data=sd)
assert wrapper(torch.randn(8,2),torch.ones(8)).shape == (8,2)
""",
            ),
            (
                "md",
                r"""
## 2. Karras grid and probability-flow integration

For additive Gaussian corruption with sigma as clock,
$$dx/d\sigma=(x-D(x,\sigma))/\sigma.$$
The descending Karras grid interpolates $\sigma^{1/\rho}$ linearly and appends zero.
Heun uses two denoiser calls except the final Euler step to zero (no division by zero),
so $N$ steps cost $2N-1$ branch calls at CFG 1. Stochastic churn is optional and is off below.
We use an analytic denoiser to isolate discretization from training.
""",
            ),
            (
                "code",
                """
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import expand
mixture = GaussianMixture()
class OracleDenoiser(torch.nn.Module):
    def forward(self,x,sigma,labels=None):
        score = mixture.score(x,torch.ones_like(sigma),sigma,labels)
        return x+expand(sigma.square(),x)*score
oracle = OracleDenoiser()
initial = torch.randn(384,2,generator=torch.Generator().manual_seed(111))
fig,axes = plt.subplots(1,3,figsize=(13,4))
for ax,(method,steps) in zip(axes,[("euler",18),("heun",18),("heun",5)],strict=True):
    samples,history,calls = edm_sample(oracle,initial,steps=steps,solver=method)
    assert calls == steps if method=="euler" else calls==2*steps-1
    scatter(ax,samples,f"Oracle EDM {method}: {calls} calls")
plt.tight_layout(); plt.show()
fig,ax = plt.subplots()
for rho in [1.,3.,7.]:
    grid = karras_sigmas(18,rho=rho)
    ax.semilogy(range(18),grid[:-1],"o-",label=f"rho={rho:g}")
ax.set(xlabel="Step index",ylabel="Sigma"); ax.legend(); plt.show()
print("Initial sigma=80 Gaussian prior is approximate for this finite-time GMM.")
""",
            ),
            (
                "md",
                r"""
## 3. A consistency map has a boundary condition

A consistency function maps points on one PF-ODE trajectory to its low-noise endpoint:
$F(x_\sigma,\sigma)=F(x_{\sigma'},\sigma')$.
At $\sigma_{min}$ require identity. The repository uses
$$c_{skip}=\frac{\sigma_d^2}{(\sigma-\sigma_{min})^2+\sigma_d^2},\quad
c_{out}=\frac{\sigma_d(\sigma-\sigma_{min})}{\sqrt{\sigma^2+\sigma_d^2}}.$$
This differs from ordinary EDM preconditioning: it makes the boundary exact.

| Method | Neighbor target |
|---|---|
| CT | Shared-noise adjacent states, detached EMA target |
| CD | Teacher PF-ODE step, detached target student |
| iCT | Detached **online** student, lognormal interval sampling, Pseudo-Huber and scale schedule |
| Progressive DDIM distillation | Two deterministic teacher steps matched by one student step |

Inference EMA and a training target EMA have different roles.
**Code:** [consistency.py](../diffusion_lab/consistency.py), [progressive.py](../diffusion_lab/progressive.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.consistency import ConsistencyModel,consistency_loss,consistency_sample,improved_scales
torch.manual_seed(112)
backbone = TimeMLP(width=64,depth=2)
backbone.condition = Conditioning(64,0,time_embedding="positional")
student = ConsistencyModel(backbone,sigma_min=.002,sigma_data=.5)
x = torch.randn(32,2)
torch.testing.assert_close(student(x,torch.full((32,),student.sigma_min)),x,atol=0,rtol=0)
plt.step(range(0,10001,250),[improved_scales(s,10000,10,160) for s in range(0,10001,250)],where="post")
plt.xlabel("Training update"); plt.ylabel("iCT noise levels"); plt.title("Example scale curriculum"); plt.show()
""",
            ),
            (
                "md",
                r"""
## 4. Short CPU consistency distillation

Use the exact GMM denoiser as a teacher and a separate EMA target student.
This is a small CD demonstration, not the published training budget/architecture.
The oracle teacher guides neighboring states; the student still must learn the
trajectory endpoint map. A lower neighbor loss alone does not prove good one-step samples.
""",
            ),
            (
                "code",
                """
import copy
target = copy.deepcopy(student).eval().requires_grad_(False)
optimizer = torch.optim.Adam(student.parameters(),lr=.001)
g = torch.Generator().manual_seed(113)
losses = []
for _step in range(1200):
    clean,_ = mixture.sample(256,g)
    loss = consistency_loss(student,target,clean,None,intervals=20,sigma_max=10,
                             teacher=oracle,generator=g,weighting="inverse_gap").mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        for destination,source in zip(target.parameters(),student.parameters(),strict=True):
            destination.lerp_(source,.05)
    losses.append(loss.item())
student.eval()
fig,axes = plt.subplots(1,3,figsize=(13,4))
axes[0].plot(losses,alpha=.3); axes[0].set(title="Neighbor CD loss",xlabel="Update")
for ax,steps in zip(axes[1:],[1,4],strict=True):
    generated,_,calls = consistency_sample(target,initial,steps=steps,sigma_max=10,
                                           generator=torch.Generator().manual_seed(114))
    scatter(ax,generated,f"Short-trained CD: {calls} calls")
plt.tight_layout(); plt.show()
""",
            ),
            (
                "md",
                r"""
## 5. Refinement is a measured hypothesis, not a guarantee

Consistency sampling re-noises a denoised estimate to each requested noise
level and applies the map again. Model error, chosen levels, clipping and
training target affect whether extra calls improve samples.
The actual revised CIFAR CD model achieved one-call FID 24.27 versus old 27.66,
but two-call FID worsened to 42.58 and four-call remained 100.57 (5k/5k).
Four-call iCT reached 43.10, improving its own one-call result but trailing one-call CD.
These results have one training seed; different recipes/budgets are not a causal ablation of one change.
""",
            ),
            (
                "code",
                """
saved_gallery("assets/studies/scratch_refinement/cd_vs_old/candidate_final_1/classes.png",
              "Actual 40k CIFAR CD, one call; rows are requested classes")
for calls in [1,2,4]:
    report = json.loads((project/f"assets/studies/scratch_refinement/cd_vs_old/candidate_final_{calls}/metrics.json").read_text())
    print({k:report[k] for k in ["nfe_per_image","fid_2048","inception_score_mean","real_count","generated_count"]})
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Why append zero to the EDM grid but avoid evaluating the denoiser at zero?
2. Why is a detached online iCT target different from an EMA target?
3. Design a fair comparison of one/four calls: fix weights, guidance, counts and evaluation seed; select levels on separate examples.

<details><summary>Worked answers</summary>

1. The final endpoint should be denoised, but log sigma and drift divisions are singular. Final Euler returns the current denoiser estimate without a zero-sigma network call.
2. Stop-gradient changes differentiation, not the target weights. Online target uses current weights; EMA averages previous updates and defines a different objective.
3. Report actual branch calls, timing and uncertainty; compare shared datasets/protocols. Do not select settings on the final evaluation results.

</details>

**Full recipes:** [Consistency chapter](../docs/10_consistency_models.md), [few-step chapter](../docs/18_stronger_few_step_generation.md), [final refinement](../docs/28_scratch_and_refinement.md).
**Continue:** [Latent diffusion and DiT](09_latent_diffusion_and_dit.ipynb).
**Sources:** [EDM](https://arxiv.org/abs/2206.00364), [consistency](https://arxiv.org/abs/2303.01469), [iCT](https://arxiv.org/abs/2310.14189).
""",
            ),
        ],
    },
    {
        "name": "09_latent_diffusion_and_dit.ipynb",
        "title": "09 · Image U-Nets, latent diffusion, DiT and reproducible experiments",
        "intro": "Trace an image model end to end: residual/time conditioning, patch tokens, stochastic latent caches, Hydra composition and evaluation evidence.",
        "prerequisites": "01, 05–06, 08; image tensors, convolutions and transformer attention.",
        "outcomes": "Inspect U-Net/DiT shapes and gradients; invert train-only latent normalization; sample posterior caches reproducibly; compose custom-data configs; assess reconstruction and generation separately.",
        "cells": [
            (
                "md",
                r"""
## 1. Architecture is separate from the generative process

The same U-Net or transformer can predict epsilon, clean data or a velocity
when its loss, output dimension and sampler agree. Time embedding is required
to distinguish noise levels; residual blocks and normalization modulation
inject conditioning throughout the network.
`ImageUNet` has down/up resolutions, skip connections, FiLM-style time/label
modulation and spatial attention. Its output is zero-initialized; an initial
zero prediction is intentional, not evidence the condition path is unused forever.
**Code:** [image_models.py](../diffusion_lab/image_models.py), [models.py](../diffusion_lab/models.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.image_models import ImageUNet
from diffusion_lab.dit import DiT
torch.manual_seed(121)
unet = ImageUNet(width=16,channels=3,classes=10,channel_mult=(1,2),
                 blocks_per_level=1,attention_levels=(1,),dropout=0)
pixels = torch.randn(2,3,16,16); time = torch.tensor([.2,.8]); labels = torch.tensor([1,7])
output = unet(pixels,time,labels)
assert output.shape == pixels.shape
loss = (output-torch.randn_like(output)).square().mean()
optimizer = torch.optim.Adam(unet.parameters(),lr=.001)
optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad():
    first = unet(pixels,time,labels)
    changed = unet(pixels,torch.tensor([.8,.2]),labels)
print("Image shape / parameter count:",output.shape,sum(p.numel() for p in unet.parameters()))
print("Time-dependent RMS change after one update:",(first-changed).square().mean().sqrt().item())
""",
            ),
            (
                "md",
                r"""
## 2. DiT patches and adaLN-Zero

For `[B,C,H,W]` and patch width $p$, patch embedding gives
`[B,N,d]`, $N=(H/p)(W/p)$. Self-attention mixes image tokens; class/time
conditioning modulates LayerNorm and residual gates. Cross-attention to text
is a separate architectural choice, not implied by calling a model DiT.
Unpatchify maps `[B,N,p²*C_out]` back to an image/latent prediction.
adaLN-Zero initially closes residual gates and the final prediction head;
gradients open those paths during training.
**Code:** `DiT.patch_embed`, `DiTBlock`, `unpatchify` in [dit.py](../diffusion_lab/dit.py).
""",
            ),
            (
                "code",
                """
dit = DiT(image_size=16,patch_size=2,width=64,depth=2,heads=4,channels=3,classes=10)
tokens = dit.patch_embed(pixels).flatten(2).transpose(1,2)
prediction = dit(pixels,time,labels)
print("Patch tokens:",tokens.shape,"output:",prediction.shape)
assert tokens.shape == (2,64,64)
torch.testing.assert_close(prediction,torch.zeros_like(prediction),atol=0,rtol=0)
loss = (prediction-torch.randn_like(prediction)).square().mean()
loss.backward()
assert dit.output.weight.grad is not None and dit.output.weight.grad.norm() > 0
print("Zero initial output still permits learning; output-head gradient:",dit.output.weight.grad.norm().item())
for patch in [2,4,8]:
    token_count = (32//patch)**2
    print(f"32px, patch {patch}: {token_count} tokens; {token_count**2} pairwise attention entries/head")
""",
            ),
            (
                "md",
                r"""
## 3. Latent diffusion starts with a codec

A frozen encoder gives $q(z|x)$; a decoder turns generated latents back into
pixels. A 256px image encoded with spatial factor 8 becomes `[B,4,32,32]` in
the SD2-family codec used here. Reconstruction quality limits what decoding
can preserve even before the generative model is trained.
The cache stores posterior **mean and standard deviation**, not one frozen
random sample. Sample $z=\mu+\sigma_q\eta$ during training and normalize using
training-only statistics. Validation/test statistics must not determine normalization.
The codec's own scale/shift and cache whitening must be inverted in the right order.
**Code:** `posterior_samples`, `normalize_latents`, `FrozenAutoencoder` in [latents.py](../diffusion_lab/latents.py).
""",
            ),
            (
                "code",
                """
from diffusion_lab.latents import normalize_latents,denormalize_latents,posterior_samples
normalization = {"mean":[.1,-.2,.3,0.],"std":[.5,.8,1.2,.6]}
means = torch.randn(16,4,8,8)
stds = torch.full_like(means,.2)
# [B,variants,2*C,H,W]; variant 0 only for this illustrative cache.
moments = torch.cat([means,stds],dim=1)[:,None]
first = posterior_samples(moments,normalization,torch.Generator().manual_seed(122),augment=False)
repeated = posterior_samples(moments,normalization,torch.Generator().manual_seed(122),augment=False)
changed = posterior_samples(moments,normalization,torch.Generator().manual_seed(123),augment=False)
torch.testing.assert_close(first,repeated,atol=0,rtol=0)
assert not torch.equal(first,changed)
torch.testing.assert_close(denormalize_latents(normalize_latents(means,normalization),normalization),means)
fig,axes = plt.subplots(1,3)
for ax,image,title in zip(axes,[means[0,0],first[0,0],changed[0,0]],
                         ["Posterior mean channel","Normalized draw, seed 122","Normalized draw, seed 123"],strict=True):
    ax.imshow(image,cmap="coolwarm"); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
""",
            ),
            (
                "md",
                r"""
## 4. Compose a real Hydra recipe

Hydra connects data, architecture, process, sampler and training state.
Compose a toy recipe to see the resolved choices; composition does not train.
For image folders use the dataset's supported layouts and explicit root/classes,
then set channel count and resolution to match the model. The repository
supports custom `_target_` models as documented in the image chapter.
**Code:** [train config](../configs/train.yaml), `make_model` in [models.py](../diffusion_lab/models.py), [ImageDataset](../diffusion_lab/data.py).
""",
            ),
            (
                "code",
                """
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(config_dir=str(project/"configs"),version_base=None):
    config = compose(config_name="train",overrides=["experiment=gmm_flow","device=cpu","steps=10"])
print(OmegaConf.to_yaml({key:config[key] for key in ["data","model","process","sampler","device","steps"]}))
""",
            ),
            (
                "md",
                r"""
## 5. A GPU codec exercise you explicitly enable

The next cell is off by default and never downloads weights or data.
To inspect **your own local** image and VAE, set `DIFFUSION_NOTEBOOK_VAE` and
`DIFFUSION_NOTEBOOK_IMAGE`, optionally `DIFFUSION_NOTEBOOK_DEVICE=cuda:0`.
The codec remains frozen; this exercise only compares reconstruction to its input.
Use the full training scripts for multi-GPU runs, rather than hiding cluster jobs in a notebook.
""",
            ),
            (
                "code",
                """
vae_path = os.environ.get("DIFFUSION_NOTEBOOK_VAE")
image_path = os.environ.get("DIFFUSION_NOTEBOOK_IMAGE")
if vae_path and image_path:
    from PIL import Image as PILImage, ImageOps
    from diffusion_lab.latents import FrozenAutoencoder
    device = os.environ.get("DIFFUSION_NOTEBOOK_DEVICE","cpu")
    codec = FrozenAutoencoder(vae_path,device)
    image = ImageOps.fit(PILImage.open(image_path).convert("RGB"),(256,256),method=PILImage.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float()/127.5-1
    with torch.no_grad():
        moments = codec.encode_moments(pixels)
        identity_normalization = {"mean":[0.]*codec.channels,"std":[1.]*codec.channels}
        reconstruction = codec.decode(moments[:,:codec.channels],identity_normalization)
    fig,axes = plt.subplots(1,2)
    for ax,value,title in zip(axes,[pixels,reconstruction],["Input","Frozen codec reconstruction"],strict=True):
        ax.imshow(((value[0].clamp(-1,1)+1)/2).permute(1,2,0)); ax.set_title(title); ax.axis("off")
    plt.show()
else:
    print("Optional codec exercise skipped: set both local VAE and image paths to enable it.")
""",
            ),
            (
                "md",
                r"""
## 6. From lesson to a reproducible image experiment

Record: resolved config, source version, data split, training RNG state, model
and inference EMA, codec identity, evaluation seed/counts, guidance, solver
grid and actual NFE. A final weight hash identifies the exact evaluated file;
a mutable `checkpoint.pt` path alone does not.
Report image metrics on decoded/rendered pixels with a fixed feature pipeline.
Latent tensor distance must not be called pixel-image FID.
The public checkout includes curated results, not training caches or weights.
**Read:** [Image training commands](../docs/12_image_experiments.md),
[latent/DiT recipes](../docs/20_latent_diffusion_and_dit.md),
[Slurm templates](../slurm/README.md), [checkpoint manifest](../assets/metadata/checkpoints.json).
""",
            ),
            (
                "code",
                """
from scripts.asset_layout import experiment_group

for name in ["cifar10_edm_cfg_final","cifar10_dit_cfg_final","cifar10_flow50_final"]:
    report = json.loads((project/f"assets/experiments/{experiment_group(name)}/{name}/metrics.json").read_text())
    print(name,{k:report[k] for k in ["fid_2048","inception_score_mean","real_count","generated_count"]})
saved_gallery("assets/experiments/cifar10/cifar10_dit_cfg_final/classes.png",
              "Actual trained CIFAR DiT class-conditioned samples; rows are requested classes")
saved_gallery("assets/experiments/faces/celeba128_latent_dit_final/interpolation.png",
              "Actual latent face DiT noise interpolation; codec/denoiser identities are recorded in the linked report")
""",
            ),
            (
                "md",
                r"""
## Exercises and worked answers

1. Why can a latent denoiser trained from random initialization still depend on external pretraining?
2. How does changing patch width affect token count and attention cost?
3. Why save posterior variance rather than cache one draw per image?

<details><summary>Worked answers</summary>

1. Its frozen encoder/decoder and possibly text encoder were pretrained. Describe those boundaries and unknown overlap with evaluation data.
2. Doubling patch width divides token count by four and full attention entries by sixteen, while losing spatial token granularity. Overall cost also includes projections/MLPs.
3. Repeated posterior draws represent codec uncertainty and require controlled RNG. One cached draw removes that variability from training.

</details>

**Finish the course:** reproduce one small experiment, change one modeling choice,
and report its benefit and limitation using the [evidence protocol](../docs/25_evidence_studies.md).
**Sources:** [Latent diffusion](https://arxiv.org/abs/2112.10752), [DiT](https://arxiv.org/abs/2212.09748).
""",
            ),
        ],
    },
]
