# Coverage, evidence and implementation scope

This map connects the requested concepts to code, explanations and actual
experiments. A working implementation, a completed training budget and a strong
benchmark result are three separate claims. The [live training report](../assets/metadata/training_status.json)
tracks budgets; the [results](14_advanced_results.md) track measured outcomes.

| Concept | Implementation | Explanation and visual evidence |
|---|---|---|
| Gaussian corruption, denoising and Tweedie's identity | [targets](../diffusion_lab/targets.py), [paths](../diffusion_lab/paths.py) | [Foundations](01_foundations.md), [diffusion](02_diffusion.md); analytic score/denoising plots |
| Epsilon, clean-data, score and diffusion-v targets | [targets](../diffusion_lab/targets.py) | Algebraic conversion identities and trained GMM target comparisons |
| Flow velocity versus diffusion v | [paths](../diffusion_lab/paths.py), [training](../diffusion_lab/train.py) | [Flow matching](03_flows_and_transport.md); independent and OT couplings |
| Forward/reverse SDE and probability-flow ODE | [paths](../diffusion_lab/paths.py), [sampling](../diffusion_lab/sampling.py) | [Foundations](01_foundations.md); paired stochastic/deterministic paths and density evolution |
| Fokker–Planck and continuity equations | [lessons](../diffusion_lab/lessons.py) | Analytic density evolution and finite-difference residual diagnostics |
| VP, VE and sub-VP | [paths](../diffusion_lab/paths.py) | [Diffusion](02_diffusion.md), [advanced diffusion](09_advanced_diffusion.md); coefficient curves, SDE variance identities and trained GMMs |
| Linear/cosine/log-SNR/exponential schedules | [paths](../diffusion_lab/paths.py) | Schedule plots, finite-endpoint conventions and positivity tests |
| DDPM and DDIM | [sampling](../diffusion_lab/sampling.py) | [Diffusion](02_diffusion.md); full 1,000-call and respaced MNIST generation |
| Learned diagonal reverse variance | [Improved DDPM](../diffusion_lab/improved_ddpm.py) | Trained two-head MNIST model and predicted-variance diagnostic plot |
| Simple/hybrid/VLB losses and time importance sampling | [Improved DDPM](../diffusion_lab/improved_ddpm.py), [image trainer](../diffusion_lab/image_train.py) | Hybrid trained run; finite VLB/adaptive smoke run; unbiased-weight and gradient tests |
| EDM preconditioning, loss and Karras grid | [EDM](../diffusion_lab/edm.py) | [Advanced diffusion](09_advanced_diffusion.md); coefficient plots, trained MNIST and measured CIFAR galleries |
| Euler/Heun/RK4/RK45 | [solvers](../diffusion_lab/solvers.py) | [Numerical methods](05_numerical_methods.md); analytic accuracy comparisons |
| Euler–Maruyama/stochastic Heun/predictor–corrector | [solvers](../diffusion_lab/solvers.py), [sampling](../diffusion_lab/sampling.py) | Stochastic moments, trajectories and moving-mass GIFs |
| DPM-Solver and DPM-Solver++ | [DPM solver](../diffusion_lab/dpm_solver.py) | Analytic Gaussian ODE convergence and same-checkpoint 21-call versus 1,000-call image comparison |
| Consistency training and distillation | [consistency](../diffusion_lab/consistency.py) | [Consistency chapter](10_consistency_models.md); completed 20,000-step CT/CD runs, 1/2/4-call samples |
| Rectified flow and repeated reflow | [training](../diffusion_lab/train.py), [advanced lessons](../diffusion_lab/advanced_lessons.py) | [Flows](03_flows_and_transport.md), [visual transport](13_visual_transport.md); crossing pairs, learned reflow, cost/straightness diagnostics |
| Optimal transport and entropic OT | [transport](../diffusion_lab/transport.py) | Assignment/Sinkhorn comparisons, marginal checks and transport-cost figures |
| Brownian Schrödinger bridges and bridge matching | [bridges](../diffusion_lab/bridges.py) | [Bridge chapter](06_bridges.md); covariance checks and entropic bridge trajectories |
| Alternating DSBM-IMF/IPF | [DSBM](../diffusion_lab/dsbm.py) | [Iterative bridges](11_iterative_bridges.md); four learned forward/backward cycles for both initializations |
| Conditional/unconditional generation and CFG | [guidance](../diffusion_lab/guidance.py), [image evaluator](../diffusion_lab/image_evaluate.py) | Class galleries, null-token convention, rich labeled toy data and independent image assessment |
| Noisy classifier guidance | [image guidance](../diffusion_lab/image_guidance.py), [toy oracle](../diffusion_lab/toy_evaluate.py) | Measured classifier accuracy versus noise; actual image guidance tradeoffs and exact ring-posterior reference |
| CLIP guidance | [image guidance](../diffusion_lab/image_guidance.py) | Differentiable denoised-image energy, actual saved small galleries and explicit failure/heuristic scope |
| Image interpolation | [image evaluator](../diffusion_lab/image_evaluate.py) | Latent interpolation strips/GIFs and approximate real-image ODE inversion |
| CIFAR-10/MNIST verification | [downloader](../scripts/download_benchmarks.py) | Every exported PNG checked against source pixels; saved counts and ordered hashes |
| Custom image folders and architectures | [data](../diffusion_lab/data.py), [model factory](../diffusion_lab/models.py) | [Image practice](12_image_experiments.md); disjoint holdouts and Hydra constructor/forward contract |
| Checkpoints, reproducibility and DDP | [image trainer](../diffusion_lab/image_train.py), [resume check](../scripts/check_resume.py) | CPU bitwise continuation, three-rank GPU smoke run, immutable milestones and evaluated weight hashes |
| Cluster execution | [Slurm templates](../slurm/README.md) | Single-GPU, three-GPU DDP and bridge jobs; user-controlled site settings |
| MMD/Wasserstein/FID/KID/IS | [metrics](../diffusion_lab/metrics.py) | Explicit sample counts/reference selection; genuine Inception metrics for RGB and separate digit features for MNIST |
| Full DDPM ELBO and ODE likelihood | [likelihood](../diffusion_lab/likelihood.py) | [Likelihood chapter](16_likelihood_and_elbo.md); complete MNIST bound and analytic GMM density checks |
| General Gaussian-noise stochastic interpolants | [interpolants](../diffusion_lab/interpolants.py) | [Interpolant chapter](17_stochastic_interpolants.md); exact PDE tests and trained GMM/rings/spiral ODE/SDE GIFs |
| Improved consistency and progressive distillation | [consistency](../diffusion_lab/consistency.py), [progressive](../diffusion_lab/progressive.py) | [Few-step chapter](18_stronger_few_step_generation.md); online-target invariants and staged saved experiments |
| Precision/recall, density/coverage and seed uncertainty | [distribution metrics](../diffusion_lab/distribution_metrics.py) | [Evaluation chapter](19_evaluation_beyond_fid.md); RGB/digit class diagnostics, nearest neighbors and independent seeds |
| Latent diffusion and DiT | [latent codec](../diffusion_lab/latents.py), [DiT](../diffusion_lab/dit.py) | [Latent/DiT chapter](20_latent_diffusion_and_dit.md); verified posterior cache and completed 100k-update image generators with decoded final metrics |
| Caption conditioning, token cross-attention and text CFG | [text architectures](../diffusion_lab/text.py), [caption data](../diffusion_lab/caption_data.py), [text trainer](../diffusion_lab/text_train.py) | [Text chapter](24_text_conditioning.md); verified COCO images, masked-attention notebook, 3,000-update pretrained SD2 LoRA run and matched prompt/CFG galleries |
| Target-specific Min-SNR weights | [text training loss](../diffusion_lab/text_train.py) | Epsilon and VP-v formulas, unit checks and saved COCO configuration; a matched weighting ablation is not claimed |
| Guided course and shared conventions | [curriculum](00_start_here.md), [notation](23_notation_and_code.md) | Seven executed CPU notebooks and [worked exercises](exercises.md) |
| Expanded likelihood and independent training seeds | [evidence runner](../scripts/run_evidence_studies.py), [image-seed runner](../scripts/run_image_seed_study.py) | [Current results](26_learning_and_text_results.md); larger example sets, numerical sensitivity, three GMM/MNIST training seeds per model family |

## Boundaries of the claims

- Learned covariance is diagonal, not a dense pixel covariance. The training
  VLB omits the constant terminal prior term. The separate likelihood evaluator
  includes it in the complete negative ELBO; neither is called exact NLL.
- DPM implementations cover first-order and second-order multistep methods.
  They do not include every adaptive/third-order variant from those papers.
- Original CT/CD and the main iCT objective improvements are available. The
  configurable educational architectures/budgets do not reproduce paper scores.
- Stochastic interpolants provide affine endpoint families with Gaussian bridge
  noise and a custom differentiable nonlinear-base API. The saved learned
  experiments use the affine families; nonlinear targets have derivative tests.
- Earlier latent generators use an external frozen VAE and denoisers trained
  from scratch. Text examples fine-tune pretrained SD2 through LoRA; the compact
  text U-Net is implemented and tested, but has not completed scratch COCO
  training. Pretrained DiT weight import and a full latent image-likelihood
  variational calculation are not implemented.
- DSBM is implemented and trained in two dimensions. The image generators are
  diffusion/consistency/flow models, not image-scale Schrödinger bridges.
- Finite neural fits and endpoint-truncated simulations approximate the theory.
  Reflow/DSBM metrics need not improve monotonically in the saved experiments.
- CLIP is a heuristic denoised-image energy. The saved examples do not establish
  calibrated noisy posteriors or reliable open-vocabulary generation.
- Noisy classification accuracy does not establish correct guidance gradients.
  Independent assessment detects classifier exploitation that self-assessment
  can miss. Strong guidance can reduce diversity or distort images.
- Toy samplers are smoothed continuous distributions, not exact densities for
  every geometry. Analytic scores are used only where explicitly derived.
- Original short image runs are teaching baselines. Advanced runs retain their
  actual step counts; low FID/high IS is a measured outcome, not a guarantee
  provided by an architecture or a configured training budget.
- RGB scores use the declared local reference split/count and preprocessing.
  They are not directly comparable to every published 50k-reference protocol.
  The official test split is used for validation here, including development;
  disclose that use before making external benchmark claims.
- Local checks are recorded in [validation](../assets/metadata/validation.json). The CI
  workflow is provided but no GitHub CI execution is claimed before publication.

Large checkpoints and datasets remain outside the public source tree. The
[hash manifest](../assets/metadata/checkpoints.json) identifies evaluated local weights;
publishing downloadable weight artifacts is a separate release step.
The [current repository audit](22_repository_audit.md) separates remaining
release/evaluation work from optional additional research concepts.

Version 0.4 adds the [guided curriculum](00_start_here.md), executable notebooks,
[notation reference](23_notation_and_code.md), [COCO text module](24_text_conditioning.md)
and [evidence studies](25_evidence_studies.md). See [current measured results](26_learning_and_text_results.md)
for completed budgets and limitations. Text-conditioned checkpoints are separate
from the earlier class/unconditional image generators.
