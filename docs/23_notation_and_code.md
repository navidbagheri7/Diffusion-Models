# Notation, time directions, shapes and equations in code

This is the shared reference for the course. Individual papers use different
symbols; the mappings below describe this repository's conventions.

## Random variables and network outputs

| Symbol | Meaning | Code/shape |
|---|---|---|
| $x_0$ | A clean observed example in forward diffusion | `clean`, `[B,2]` or `[B,C,H,W]` |
| $x_t$ | A corrupted state at diffusion time $t$ | Same shape as `clean` |
| $\epsilon,Z$ | Independent standard Gaussian vectors | Same shape as the endpoint; never a variance |
| $\alpha(t),\sigma(t)$ | Signal multiplier and noise **standard deviation** | `[B]`, broadcast by `paths.expand` |
| $s_t(x)$ | Marginal score $\nabla_x\log p_t(x)$ | Same shape as $x$ |
| $\widehat x_0,\widehat\epsilon$ | Network-based conditional-mean estimates | Same shape as $x$ |
| $v_{\rm VP}$ | Diffusion target $\alpha\epsilon-\sigma x_0$ | Prediction label `v`; distinct from flow velocity |
| $u_\tau(x)$ | Marginal flow/interpolant velocity | Prediction label `velocity` |
| $f,g$ | Forward SDE drift and Brownian coefficient | `f`: shape of $x$; `g`: `[B]` |
| $c,\varnothing$ | A condition and its null counterpart | Class IDs `[B]`; text context `[B,L,D]` |
| $\theta,\theta_{\rm EMA}$ | Online trainable parameters and sampling EMA | EMA is not necessarily the training target |

`B` is batch size; `C` is pixel/latent channel count; `H,W` are spatial sizes.
Text has `L` tokens with `D` features. Masks `[B,L]` use `True` for valid tokens
in our explicit attention module. Latents have codec-dependent channels and
spatial reduction; they are not RGB images.

## Two time directions

**Forward diffusion:** $t=0$ is data and $t=1$ is noise. Reverse generation
integrates from large $t$ to small $t$. Brownian increments use $\sqrt{|dt|}$.

**Flow/interpolant generation:** $\tau=0$ is the source (usually noise) and
$\tau=1$ is target data. If changing coordinates with $\tau=1-t$, the ODE field
changes sign: $dx/d\tau=-dx/dt$. A time reparameterization also changes SDE
coefficients; swapping endpoint labels is not sufficient.

**Discrete DDPM:** mathematical step $k\in\{1,\ldots,T\}$ has Python index
$i=k-1$. Our `DiscreteVP` networks receive $t=(i+1)/T$. Index `-1` represents
clean data during sampling. SD2 uses its own native integer indices $i=0,
\ldots,999$ and scheduler tables; `text_train` preserves those tables. The
compact text network receives $(i+1)/T$, with $T$ from the native scheduler.

**EDM/consistency:** the model is indexed by a noise standard deviation $\sigma$,
not normalized diffusion time. EDM corruption is $x=x_0+\sigma\epsilon$.
The network sees $\log\sigma/4$ through its preconditioner.

## Equation-to-code map

| Equation or operation | Implementation | Runnable lesson |
|---|---|---|
| $x_t=\alpha x_0+\sigma\epsilon$ and $f,g$ | [GaussianPath/DiscreteVP](../diffusion_lab/paths.py) | [Noising](../notebooks/00_probability_and_noising.ipynb) |
| Per-example coefficient broadcasting | [expand](../diffusion_lab/paths.py) | [Targets](../notebooks/01_prediction_targets_and_ddim.ipynb) |
| Noise/clean/score/VP-v targets and weights | [training_target](../diffusion_lab/targets.py) | [Targets](../notebooks/01_prediction_targets_and_ddim.ipynb) |
| $\widehat x_0=(x+\sigma^2\widehat s)/\alpha$ (Tweedie) | [convert_prediction](../diffusion_lab/targets.py) | [Targets](../notebooks/01_prediction_targets_and_ddim.ipynb) |
| Reverse SDE/PF-ODE coefficients | [sampling](../diffusion_lab/sampling.py) | [SDE/ODE](../notebooks/02_scores_sdes_and_odes.ipynb) |
| Euler/Heun/RK and $\sqrt{|dt|}$ Brownian increments | [solvers](../diffusion_lab/solvers.py) | [Flow/solvers](../notebooks/03_flow_matching_and_solvers.ipynb) |
| DDPM posterior and learned diagonal variance | [improved_ddpm](../diffusion_lab/improved_ddpm.py) | [Advanced diffusion](09_advanced_diffusion.md) |
| DDIM respaced update | [discrete_sample](../diffusion_lab/sampling.py) | [Targets/DDIM](../notebooks/01_prediction_targets_and_ddim.ipynb) |
| $u_\theta\leftarrow\mathbb E[\dot I_\tau\mid I_\tau=x]$ | [perturb/train](../diffusion_lab/train.py), [interpolants](../diffusion_lab/interpolants.py) | [Flows](03_flows_and_transport.md) |
| CFG $y_u+w(y_c-y_u)$ | [classifier_free](../diffusion_lab/guidance.py), [text_guided_prediction](../diffusion_lab/text.py) | [Guidance](../notebooks/04_guidance.ipynb) |
| $d\log p(x_t)/dt=-\nabla\cdot u_t$ | [ode_log_density](../diffusion_lab/likelihood.py) | [Likelihood](../notebooks/05_likelihood_and_evaluation.ipynb) |
| Complete DDPM negative ELBO | [ddpm_elbo](../diffusion_lab/likelihood.py) | [ELBO experiment](../notebooks/05_likelihood_and_evaluation.ipynb) |
| $\mathrm{softmax}(QK^\top/\sqrt d)V$ | [TextCrossAttention](../diffusion_lab/text.py) | [Text attention](../notebooks/06_text_cross_attention.ipynb) |
| $Wx+(a/r)BAx$ | [LowRankLinear](../diffusion_lab/text.py) | [Text training](24_text_conditioning.md) |
| Min-SNR epsilon/v-specific weights | [diffusion_training_loss](../diffusion_lab/text_train.py) | [Text training](24_text_conditioning.md) |
| Rendered pixels and RGB Inception metrics | [image_pixels/image_metrics](../diffusion_lab/metrics.py) | [Evaluation](19_evaluation_beyond_fid.md) |
| Fokker–Planck / continuity residual | [GaussianPath](../diffusion_lab/paths.py), [analytic density](../diffusion_lab/data.py) | [PDE checks](../notebooks/02_scores_sdes_and_odes.ipynb) |
| Interpolant JVP, score and matching stochastic drift | [StochasticInterpolant](../diffusion_lab/interpolants.py) | [Interpolants](../notebooks/07_interpolants_transport_and_bridges.ipynb) |
| OT assignment, Sinkhorn and joint Brownian bridge covariance | [transport](../diffusion_lab/transport.py), [bridges](../diffusion_lab/bridges.py) | [OT/bridges](../notebooks/07_interpolants_transport_and_bridges.ipynb) |
| EDM preconditioning and consistency boundary | [EDM](../diffusion_lab/edm.py), [consistency](../diffusion_lab/consistency.py) | [EDM/few-step](../notebooks/08_edm_and_few_step_models.ipynb) |
| DiT patch tokens and latent posterior normalization | [DiT](../diffusion_lab/dit.py), [latents](../diffusion_lab/latents.py) | [Latent/DiT](../notebooks/09_latent_diffusion_and_dit.ipynb) |

## Identities and assumptions to keep visible

The conditional Gaussian corruption score is $-\epsilon/\sigma$. It is a noisy
training target. Squared regression learns the **marginal** score through
conditional expectation, not that particular hidden noise realization.

For $dX_t=f(X_t,t)dt+g(t)dW_t$, with scalar spatially independent $g$, the
Fokker–Planck equation is

$$\partial_t p=-\nabla\cdot(fp)+\tfrac12g^2\Delta p.$$

The reverse-clock SDE drift is $f-g^2s$; the PF-ODE drift is $f-\tfrac12g^2s$.
Both statements require a matching initial marginal, the exact score and suitable
regularity. Our finite-noise priors, learned fields, endpoint cutoffs and solvers
are approximations. The continuity equation is
$\partial_\tau p=-\nabla\cdot(pu)$ and has no diffusion term.

VP obeys $\alpha^2+\sigma^2=1$; VE and sub-VP generally do not. Prediction
conversion uses $\alpha^2+\sigma^2$ where necessary. VE starts with small
Gaussian smoothing when $\sigma_{\min}>0$. Dividing by zero at clean/noise
endpoints is not part of a valid score or Tweedie calculation.

CFG scale is $w=0$ for null conditioning and $w=1$ for ordinary conditioning.
Other papers sometimes shift this convention. Text dropout uses the encoded
empty prompt, not a zero vector. Pretrained SD2 padding semantics are preserved;
our compact U-Net explicitly masks padding. Attention maps alone do not establish
semantic grounding.

The DDPM negative ELBO includes terminal prior KL, all transition KL terms and
the decoder negative log likelihood. A Monte Carlo bound is not exact NLL.
Numerical PF-ODE likelihood includes trace/integration/dequantization assumptions.
Known-class conditional likelihood is not a class-marginal likelihood. Frozen
latent density alone is not pixel image likelihood.

Use [references](references.md) for paper attribution and
[exercises](exercises.md) for worked checks of these conventions.
