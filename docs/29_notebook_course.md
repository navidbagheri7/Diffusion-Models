# Notebook course revision — October 6, 2026

The [course index](../notebooks/README.md) now links ten independent lessons,
expanded from the original seven introductory notebooks. Each includes
prerequisites, learning outcomes, equations linked to library functions, worked
experiments, interpretation, exercises and answers. The core implementation
remains in `diffusion_lab/`; notebooks import it rather than replacing it.

## What changed

- Foundations explain samples/densities/scores, per-example broadcasting, six
  continuous schedule families, covariance identities and marginal versus
  stochastic trajectory coupling.
- DDPM/DDIM includes all four target conversions, Tweedie, the posterior and
  table/clean-endpoint conventions, a small trained denoiser, independent field
  checks and measured sampler calls.
- SDE/ODE derives coefficients from Gaussian moments, checks score gradients,
  animates both processes and verifies local Fokker–Planck/continuity residuals.
- Flow matching trains a velocity field, compares Euler/Heun/RK4/RK45 by calls,
  explores noise interpolation and runs a teacher-endpoint reflow round.
- Guidance verifies Bayes' noisy-posterior gradient identity, compares exact
  CG/CFG, teaches null-label dropout and checks branch costs.
- Likelihood computes every DDPM ELBO term, tests the learned diagonal variance
  head and detached mean, compares ODE density solvers and stochastic traces,
  and studies collapse/outliers using MMD, sliced Wasserstein and PRDC.
- Text conditioning checks masks, trains an explicitly artificial attention
  task, traces a text U-Net forward/backward pass, verifies LoRA and text CFG,
  and displays actual COCO successes and failures.
- Interpolants/bridges constructs general targets and nonlinear-path JVPs,
  compares matching oracle ODE/SDE fields, solves empirical OT/Sinkhorn,
  simulates the joint Brownian bridge law and checks its temporal covariance.
- EDM/few-step explains preconditioning, Karras grids and consistency boundaries,
  trains a small CD student and interprets the unsuccessful CIFAR refinement results.
- Image/latent/DiT traces U-Net and transformer shapes/gradients, checks stochastic
  latent caches and whitening, composes Hydra configs, shows image interpolation
  evidence and provides an explicitly optional frozen-codec reconstruction exercise.

## Interpretation matters

Short training budgets are teaching demonstrations. The flow run forms the
eight modes, whereas the short denoiser has less resolved mode structure.
The reflow run reduces chord deviation but its five-call endpoints are less
sharply clustered. Lower curvature alone does not establish better distribution
matching. The consistency lesson similarly separates neighbor loss from
generation quality and retains the actual weak CIFAR results.

The synthetic attention experiment uses artificial token/quadrant supervision.
It is not a diffusion-training experiment or proof of language grounding.
General nonlinear interpolant target construction is demonstrated; this does
not establish a trained, optimized nonlinear image-scale transport model.
Bridge examples remain finite-support/two-dimensional and explicitly distinguish
that approximation from continuous-distribution path-space KL statements.

Image-scale generation evidence is read from curated assets. No default cell
downloads models, opens training caches or starts a long GPU job. Pretrained
codec/text dependencies and experimental limitations are stated alongside the images.

## Validation and maintenance

All ten default lessons passed in fresh CPU kernels: **69 code cells**, **69
Markdown cells**, **43 saved visual outputs** and six standalone GIFs. The
post-organization default execution took approximately 94 seconds on the development machine;
runtime varies by hardware. [Execution evidence](../assets/reports/diagnostics/notebook_validation.json)
records per-notebook counts, timings and hashes.

The same lessons were executed successfully from a public-file copy without
`outputs/`, checkpoints or `.local/` helpers. Four widget-enabled notebook
copies were checked headlessly with the notebook extra installed. The optional
codec exercise passed using the existing local SD2-family VAE and one COCO
image on an L40S GPU. These checks do not test a browser frontend or establish
image-generation quality. [Extended validation](../assets/reports/diagnostics/notebook_course_review.json)
records these scopes separately.

The posterior lesson caught float32 rounding in the first DDPM posterior mean.
`posterior` now enforces its exact clean-data endpoint; four schedule variants
also verify the clean/noisy-state gradients. The scientific suite passes
**124 tests**. Existing saved training metrics retain their original provenance;
they were not recomputed or relabeled as new results after this numerical fix.

Cell sources are reviewable in [notebook_lessons.py](../scripts/notebook_lessons.py).
Regenerate with [build_notebooks](../scripts/build_notebooks.py), then execute with
[execute_notebooks](../scripts/execute_notebooks.py) before publishing changes.
The [publication guide](27_publication.md) still lists the separate license,
checkpoint-hosting and release-packaging tasks.

Teaching GIFs and saved experiment galleries now use the [organized asset layout](../assets/README.md). All ten defaults were rerun after the move; the text, consistency and latent/DiT lessons also passed from a new public-file-only copy. [Reorganization evidence](../assets/reports/diagnostics/asset_reorganization.json) records these checks separately from the earlier widget/codec runs.
