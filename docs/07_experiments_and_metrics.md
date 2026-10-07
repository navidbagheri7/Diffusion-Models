# 7. Reproducible experiments and honest evaluation

## Configuration is part of the experiment

Hydra composes five groups: data, model, process, sampler, and experiment. The
experiment group provides a convenient preset; command-line overrides are saved
in the resolved run configuration. Training does not silently substitute an
unsupported sampler or target.

Examples:

~~~bash
python -m diffusion_lab.train --cfg job --resolve experiment=celeba_flow
python -m diffusion_lab.train experiment=gmm_vp seed=7 \
  sampler.ode_solver=rk4 sampler.steps=100 run_dir=outputs/vp_seed7_rk4
python -m diffusion_lab.train --multirun seed=0,1,2 \
  experiment=gmm_flow
~~~

Defaults include the seed in run_dir. Other overrides are not all encoded in that
path: explicitly name comparison runs to avoid accidental overwrites.

## Training pipeline

1. Read data and sample time/noise.
2. Construct a corruption/interpolation state and its target.
3. Optionally replace labels with the null token.
4. Evaluate the MLP or U-Net and compute the configured regression objective.
5. Backpropagate, clip gradient norm, update AdamW, then update EMA.
6. Periodically validate with fixed noise draws and save a checkpoint.
7. Sample with the EMA model, calculate metrics, and produce figures/GIFs.

EMA computes θ_EMA←d θ_EMA+(1−d)θ. It smooths optimizer fluctuations. Its
parameters and buffers are retained separately from training weights. Do not
confuse EMA decay with an optimizer learning rate.

The U-Net uses three spatial resolutions, residual blocks, GroupNorm,
time/class FiLM conditioning, skip connections, and bottleneck attention. Its
purpose is readability and small-scale experimentation, rather than reproducing
a large published architecture.

Optional BF16 autocast is used on CUDA. The loss is accumulated in float32.
NaN/Inf losses or diverging samplers raise explicit errors.

## Validation

GMM training samples are drawn online. Validation uses a separate fixed seed.
For images, validation reads a disjoint split and uses fixed noising randomness.
Changing prediction targets changes the meaning of the loss; changing label
dropout changes the mix of conditional and null-token training.

Conditional validation uses true labels, whereas training mixes true and null
labels. Therefore their losses are not expected to coincide even with no
overfitting. Validation uses the EMA model; the training loss uses live weights.

For reflow, validation measures regression on teacher endpoint pairs, while
generation metrics still compare with the real GMM. A low replay regression
loss does not establish that the teacher's target distribution is correct.

For bridges, matching loss measures the conditional bridge target. Evaluation
against the data distribution includes finite-support, learned-field,
integration, and endpoint-truncation effects.

## Data splits and preprocessing

Images are decoded as RGB, resized and center-cropped to the requested size,
then mapped to [-1,1]. No training-time random augmentation is enabled by default,
making the baseline easier to reproduce and inspect.

CelebA and ImageNet use their provided local train/validation folders.
FFHQ's single train folder is partitioned by a fixed shuffled index list.
That partition is formed before optional limits, so train and validation remain
disjoint. The local split_seed is separate from training seed.

Class-folder limits are selected round-robin across classes instead of taking
only the first class's images. train_limit=0 or validation_limit=0 means all
images in that split. Bounded subsets are teaching settings; record them when
reporting results.

## Gaussian-mixture metrics

### Maximum mean discrepancy

For kernel k and samples x₁,…,x_n and y₁,…,y_m, the unbiased squared estimate is

$$
\widehat{\mathrm{MMD}}^2=
\frac{\sum_{i\ne j}k(x_i,x_j)}{n(n-1)}
+\frac{\sum_{i\ne j}k(y_i,y_j)}{m(m-1)}
-\frac{2\sum_{i,j}k(x_i,y_j)}{nm}.
$$

We average RBF kernels exp(−‖x−y‖²/(2h²)) with h∈{0.25,0.5,1,2}. The bandwidth
choice is recorded in code; it is not fitted to favor a particular model.
Unbiased estimates can be slightly negative. They are not negative distances.
Implementation uses bounded blocks rather than allocating a full large Gram
matrix.

### Sliced Wasserstein-2

Draw 256 seeded unit directions θ. Project both equal-size sample sets, sort
each projection, and compare matching empirical quantiles:

$$
\widehat{\mathrm{SW}}_2=
\sqrt{\frac1{Ln}\sum_{\ell=1}^L\sum_{i=1}^n
(\langle\theta_\ell,x\rangle_{(i)}
-\langle\theta_\ell,y\rangle_{(i)})^2}.
$$

This averages one-dimensional W2² and takes a square root. It is a sliced
metric, not exact multidimensional W2.

### Exact empirical Wasserstein-2

For equal-weight sample sets of equal size, minimize assignment cost:

$$
\widehat W_2=
\sqrt{\frac1n\min_{\rho\ \mathrm{permutation}}
\sum_i\|x_i-y_{\rho(i)}\|^2}.
$$

The code uses at most 512 examples because Hungarian assignment has cubic
scaling. It is exact for those empirical measures, not exact population W2.
wasserstein_sample_count records the actual count.

### Density and mode diagnostics

The known GMM density permits E_generated[−log p_target(X)]. This is a cross-
entropy-like diagnostic, not a learned model likelihood. A generator collapsed
at component centers can achieve a deceptively favorable value, so use it
alongside coverage and distribution distances.

A sample is accepted if its nearest center lies within 3τ. Mode coverage is
the fraction of modes with at least one accepted sample. Occupancy diagnostics
include an explicit outlier bin rather than renormalizing away rejected samples.
The target diagnostic distribution allocates equal mass to each mode and zero
to the outlier bin. Because a true Gaussian has tails, this is a deliberately
idealized mode diagnostic, not exact total variation of continuous densities.
For a 2D isotropic Gaussian, the true mass inside radius 3τ is about 98.9%.

Real-versus-real samples give a finite-sample reference:
[baseline diagnostics](../assets/reports/diagnostics/real_vs_real_metrics.json).
Even two correct generators compared through independent sample sets have
nonzero empirical Wasserstein distances.

## Image metrics

### FID

Extract pretrained Inception features, estimate real/generated means μ and
covariances C, and compute the Gaussian W2 approximation:

$$
\mathrm{FID}=\|\mu_r-\mu_g\|^2+
\operatorname{Tr}\left(C_r+C_g-
2(C_r^{1/2}C_gC_r^{1/2})^{1/2}\right).
$$

This lab uses torchmetrics with torch-fidelity's 2048-dimensional Inception
features. It does not call a pixel covariance distance FID.

### KID

Kernel Inception Distance is an unbiased MMD² estimate on Inception features,
using the polynomial kernel (xᵀy/d+1)³. The saved mean and standard deviation
describe ten random subset evaluations, each of size min(50,n_real,n_generated).
Negative estimates are possible.

### Inception Score

$$\mathrm{IS}=\exp\left(\mathbb E_x
\operatorname{KL}(p_{\rm Inc}(y\mid x)\|p_{\rm Inc}(y))\right).$$

IS combines confident pretrained classifications with variety across generated
images. It has no real-data reference and does not establish dataset fidelity.
Its class semantics are especially poorly aligned with an unconditional face
dataset. The reported split standard deviation is not a confidence interval.

### Protocol details

Both real and generated images undergo the same rounded uint8 conversion from
[-1,1]. torch-fidelity handles Inception resizing. Generated values are clipped
for image display/metrics; unclipped sampled tensors are also saved locally.

Record resolution, resize/crop policy, feature extractor, real/generated counts,
checkpoint, sampler, seed, and guidance. Changing any of these can change FID.
128-example FID covariances have very low rank in 2048 dimensions and produce
highly noisy estimates. These teaching scores cannot be compared with published
50k-sample benchmark values.

Image sampling is batched to bound network activation memory. The current
implementation retains generated tensors and metric feature buffers, so very
large evaluation datasets still require adequate host memory.

Conditional evaluation must compare with the requested class. If a bounded
ImageNet validation selection leaves fewer than two images of that class,
figures are still saved but covariance metrics are reported as unavailable.
Select a larger validation subset for conditional quantitative evaluation.

## Reproducibility and distributed training

Seeds cover Python, NumPy, and PyTorch. Generator seeds separate training,
validation, endpoint support construction, and evaluation. CUDA kernels and
different hardware/library versions can prevent bitwise equivalence.
Stochastic batching changes RNG traversal; compare matching batch settings.

DDP synchronizes model gradients. Per-rank image sampling uses a
DistributedSampler with epoch seeds. batch_size is per GPU. The first validation
draws are fixed, and rank zero records results. Changing GPU count changes global
batch size unless you adjust per-rank batch size.

resume restores weights, EMA, optimizer state, and step/history. It does not
restore every RNG or exact loader cursor, so it continues optimization without
claiming bitwise identical resumed training.

## A scientifically useful comparison

Hold data, architecture, training budget, evaluation count, initial seeds, and
preprocessing constant. Vary one factor. Compare several seeds and match network
function evaluations when studying solvers. Use analytic fields to isolate
numerical error before comparing learned fields.

The [score diagnostics](../assets/reports/diagnostics/score_errors.png) compare learned
scores with the analytic GMM score after multiplying the error by σ(t).
The [classifier diagnostics](../assets/reports/diagnostics/classifier_diagnostics.png)
show why classification accuracy alone does not guarantee useful guidance
gradients. The [straightness diagnostic](../assets/reports/diagnostics/reflow_straightness.png)
measures path length beyond endpoint distance.

**Exercise:** repeat a comparison for at least three seeds. Report mean and spread,
including the matched real-versus-real baseline. State which error sources your
experiment actually controls.

Sources: [MMD](https://jmlr.org/papers/v13/gretton12a.html),
[FID](https://arxiv.org/abs/1706.08500),
[KID](https://arxiv.org/abs/1801.01401),
[Inception Score](https://arxiv.org/abs/1606.03498),
[torchmetrics image metrics](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html).
