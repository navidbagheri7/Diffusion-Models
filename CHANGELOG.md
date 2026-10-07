# Changelog

## 0.5.0 — 2026-10-07

Add controlled non-saturating/R1, least-squares, spectral hinge, Wasserstein/GP
and historical clipped-Wasserstein GAN comparisons: three fitting seeds,
6,000 generator updates, validation-selected EMA generators and 4,096 independent
test draws. Include small safe generators and independent prediction/metric audits.
Add eleven-concept offline explorer, all-view browser checks and previews.
Correct latent reference offsets, unsupported codec normalization checks and
general-trainer exact resume state/configuration/data-order handling. Preserve
historical source in Git and document its distinct evidence scope. Add MIT
licensing and complete notebook/evidence/source distributions.

# Changes

## 0.4.0 — 2026-10-05

- Guided prerequisite-to-research curriculum, shared notation/equation-to-code
  reference and worked exercises; seven executed notebooks with schedule,
  prediction-target, guidance and solver controls.
- Verified/resumable COCO 2017 downloader and portable image-caption manifests.
- Token cross-attention U-Net, frozen text/VAE components, native attention
  LoRA, text CFG, target-specific Min-SNR, DDP and checkpoint continuation.
- Matched pretrained/fine-tuned text galleries and independent CLIP alignment
  diagnostics, with explicit pretraining and compositional-evaluation limits.
- Frozen CIFAR consistency refinement investigation, larger full-ELBO/ODE
  studies and matched numerical sensitivity checks.
- Independent GMM/MNIST training-seed comparisons with actual model-call budgets.
- Curated results exporter, additional Slurm templates, citation metadata and
  installed-wheel checks. [Current evidence](docs/26_learning_and_text_results.md)
  records completion and remaining quality limitations.

## 0.3.0 — 2026-10-04

- Complete DDPM negative ELBO with explicit terminal/decoder/transition terms,
  sampled-sum estimator and a declared reverse-mean clipping convention.
- Continuous ODE density estimation with exact/Hutchinson Jacobian traces,
  coupled numerical integration, dequantization and analytic Gaussian/GMM checks.
- Three Gaussian-noise stochastic-interpolant path families, velocity/noise
  regression, exact GMM fields and learned class-colored ODE/SDE comparisons.
- Improved consistency objective and progressive two-to-one DDIM distillation.
- Precision/recall, density/coverage, per-class and nearest-neighbor diagnostics,
  plus independent generation-seed intervals.
- Configurable adaLN-Zero DiT, frozen-autoencoder posterior caches, train-only
  whitening, latent U-Net/DiT training and decoded image-space evaluation.
- Five new educational chapters, an evidence-based extension results page,
  Hydra presets and cache/DiT Slurm templates.

Implementation checks, completed experiments and ongoing long training budgets
are reported separately in [extension results](docs/21_extension_results.md).

## 0.2.0 — 2026-10-02

- Verified CIFAR-10/MNIST downloader, grayscale data and custom unsplit folders.
- Configurable multi-block U-Net, image trainer, DDP, RNG/loader resume,
  immutable milestones and measured inference costs.
- Sub-VP, linear log-SNR and exponential schedules.
- Learned diagonal DDPM variance, hybrid/VLB objectives and loss-second-moment sampling.
- EDM preconditioning, log-normal loss, Karras grid, Heun and optional churn.
- DPM-Solver/DPM-Solver++ first-order and second-order multistep integration.
- Standalone consistency training, EDM-teacher distillation and one/few-step sampling.
- Alternating 2D DSBM-IMF/IPF, richer labeled toy data and density GIFs.
- Image CFG, noisy classifiers, optional local CLIP energy, and interpolation.
- Additional mathematical lessons, evaluation protocols and Slurm templates.

Long image runs and final evaluation status are tracked separately in
[training status](assets/metadata/training_status.json) and
[advanced results](docs/14_advanced_results.md). Implementing a method does not
establish its paper's reported benchmark score.

## 0.1.0 — 2026-10-01

Initial original toolkit with DDPM/DDIM, VP/VE, target conversions, SDE/ODE
solvers, flow/reflow, direct finite Sinkhorn bridges, guidance, GMM/image lessons
and documented short-run baselines. Original borrowed files retained in legacy.
