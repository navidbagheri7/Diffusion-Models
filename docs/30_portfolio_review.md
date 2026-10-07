# October 7 review and generative comparisons

The laboratory now includes an [eleven-concept explorer](explorer.html),
200 recorded experiment/lesson presets and 567 image/animation selectors.
It also calculates six continuous Gaussian corruption paths, exact mixture
scores, Tweedie posterior means and prediction identities directly in the browser.
The [browser record](explorer_browser_validation.json) binds the exact HTML and
checks every view, eleven mobile concept areas and 36 independent calculations
against the Python implementation.

## GAN objectives and stabilization

Five owned unconditional 2D studies use the same eight-component ring mixture,
128-wide generator/critic networks, batch 256 and 6,000 generator updates.
Each uses fitting seeds 42–44. EMA generators are selected by the first minimum
validation MMD from 1,024 fixed independent target/noise draws, including the
initial state and every 500 updates. Test evaluation uses 4,096 separate draws;
the test set never selects an epoch or hyperparameter.

| Combination | Critic updates per generator update | Optimizer | Defining choice |
|---|---:|---|---|
| Non-saturating / R1 | 1 | Adam, 1e-4, betas 0/0.9 | Gamma 1; lazy R1 every 16 critic updates, scaled by 16 |
| Least squares | 1 | Adam, 1e-4, betas 0/0.9 | Real 1, fake 0, generator 1; half squared losses |
| Hinge / spectral normalization | 1 | Adam, 1e-4, betas 0/0.9 | Per-linear-layer spectral normalization; hinge critic |
| Wasserstein / gradient penalty | 5 | Adam, 1e-4, betas 0/0.9 | Interpolated input norm penalty, coefficient 10 |
| Historical clipped Wasserstein | 5 | RMSprop, 5e-5 | Parameters clipped to ±0.01 after each critic update |

These are controlled objective/stabilization recipes, not a claim that their
hyperparameters were optimized or that every regularizer represents a new method.
The clipped recipe is a historical teaching contrast. Their optimizer/critic
costs differ despite matched generator updates. All generators make one network
call per sample. Exact input-gradient penalties avoid discriminator batch
normalization. Published [generators and test evidence](../assets/adversarial/)
are small pickle-free arrays; complete optimizer/RNG state stays private.
The [independent audit](../assets/reports/adversarial_validation.json) reloaded all
fifteen generators, reproduced all 61,440 test outputs exactly, checked private
selected weights and recomputed every numerical metric.

The existing six GMM DDPM/flow fits have the same 6,000 training-update budget
and 4,096 evaluation count, but different training objectives, model structures,
critic work, evaluation seeds and 100-call sampling. They are historical context,
not an equal-compute causal comparison. The explorer labels this distinction.
Unbiased MMD can be negative. Target cross-entropy measures how generated points
align with the known target; it is not the GAN's negative log likelihood.
Mode coverage alone can hide outliers and unequal mode mass, so report the
three-sigma inlier fraction and total variation with an explicit outlier bin.

Primary references: [WGAN](https://arxiv.org/abs/1701.07875),
[gradient penalty](https://arxiv.org/abs/1704.00028),
[least-squares GAN](https://arxiv.org/abs/1611.04076),
[spectral normalization](https://arxiv.org/abs/1802.05957), and
[R1 and convergence](https://arxiv.org/abs/1801.04406).

## Concrete corrections

The general Hydra trainer previously restored weights/optimizer only. Its new
checkpoint schema stores per-rank Python, NumPy, Torch and CUDA random state,
consumed image position, deterministic sharded permutations and partial loss
accumulators. Image flips happen after loading on the training process, avoiding
worker-prefetch augmentation ambiguity. Resume rejects a different training
configuration or rank count; legacy checkpoints remain usable for evaluation,
but cannot establish exact continuation. Independent CPU split-run tests compare
all network and EMA values against uninterrupted DDPM and flow updates.
The image-specific and text-specific historical resume studies retain their
original documented scope; no new GPU-bitwise claim is made.

Latent evaluation now applies the declared balanced-reference offset and records
it in metadata. Codec geometry rejects per-channel standard deviations as well
as means when no normalization adapter is supplied. Tests cover disjoint balanced
subsets and unsupported codec parameters, alongside existing endpoint, density,
interpolant, BF16 gradient, codec and data-isolation oracles.

## Scope and release evidence

Historical fitting source is preserved at local commit `8b31895`. New changes
are documented rather than silently attributed to those historical experiments.
The original 45 runtime modules, notebook curriculum, configurations, study
runners, export/audit utilities and existing result reports are inventoried in the
[current component review](component_review.json). Numerical tests, fresh kernels,
public asset decoding and installed-wheel/source-archive checks have distinct
records; none implies that every generated image is high quality.

The weak scratch text denoiser and problematic CIFAR consistency refinement
remain documented negative results. Existing official-test reuse and small face
reference sets also remain explicit. The release adds stronger explanation,
correctness checks and accessible examples without relabeling those limitations
as benchmark success. Code is MIT licensed; datasets, external codecs and
borrowed excluded source retain their own terms. The owner publishes locally.

The current release checks all 132 numerical/integration tests, eleven freshly
executed English lessons (73 code cells), fifteen separate portable generation
commands, all 567 explorer views and all eleven mobile concept areas. The
[release validation](validation.json),
[notebook record](../assets/reports/diagnostics/notebook_validation.json),
[component inventory](component_review.json) and
locally generated `docs/release_artifacts.json` distribution record report their separate scopes.
The default notebook kernel uses the interpreter launching the runner, so an
unrelated registered environment cannot accidentally satisfy the lesson checks.

Distribution verification installs the wheel outside the checkout, runs the
132 tests and bundled DDPM/flow CPU training commands there, then executes every
lesson and validates public GAN evidence again from the extracted source archive.
It also decodes public images/animation frames, checks local documentation links
and applies publication exclusions. The supplied CI repeats those CPU checks;
no remote GitHub CI execution is claimed.
