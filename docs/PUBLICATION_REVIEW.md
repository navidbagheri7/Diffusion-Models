# Publication review — updated 2026-10-06

The requested `/home/labcmap/navid.bagheri-shouraki/Diffusion` path resolves to
the same checkout as `/home/navid.bagheri-shouraki/Diffusion`.

## Concept coverage

The [coverage map](15_coverage_and_scope.md) connects the requested methods to
their implementations, explanations, tests, and generated examples. It covers
DDPM/DDIM, VP/VE/sub-VP, prediction targets and Tweedie's identity, forward and
reverse SDEs, probability-flow ODEs, density equations, numerical solvers,
flow matching/reflow, OT/entropic OT, Brownian bridges and alternating DSBM,
CFG/classifier guidance, Improved DDPM, EDM, consistency models, full DDPM ELBO,
ODE likelihood estimates, stochastic interpolants, progressive distillation,
distribution coverage metrics, latent diffusion and adaLN-Zero DiT.

Version 0.4 adds a guided course, worked exercises and shared notation. The
October 6 revision expands the original seven introductory notebooks into
[ten guided executable lessons](../notebooks/README.md). It also adds COCO captions, token cross-attention,
text CFG, native LoRA and target-specific Min-SNR, plus expanded likelihood,
consistency diagnostics and independent GMM/MNIST training comparisons. The
[current results](26_learning_and_text_results.md) record completed budgets
separately from quality limitations and pretrained-model contributions.

The scope is explicit: DSBM examples are two-dimensional; reverse covariance
is diagonal; CT/CD use the original MSE construction, while iCT implements
online targets, interval sampling and Pseudo-Huber loss; and DPM
solvers cover first-order and second-order multistep variants. ImageNet and
FFHQ configurations are examples, not evidence of fully trained large-scale
models. Custom data/models and Slurm/DDP interfaces are included.

## Completed training and sample quality

All three long RGB runs completed 100,000 optimizer steps. Four MNIST models
completed their 20,000-step budgets. Final RGB evaluations and exports completed;
the [training report](../assets/metadata/training_status.json) records saved milestones.
The extension also completed 100k-update CIFAR DiT and both 128px latent models,
30k-update CIFAR consistency distillation and 40k-update corrected MNIST iCT.

The version 0.4 evidence extension completed six additional MNIST models
(three EDM and three flow seeds, 20,000 updates each), six GMM models
(three DDPM and three flow seeds, 6,000 updates each), and a 3,000-update
COCO LoRA run on pretrained SD2. The adapter saw 24,000 image-caption pairs,
less than one COCO epoch. Eight matched text evaluations retain 1,552 generated
images. A later compact COCO denoiser completed 100k updates with frozen
pretrained VAE/text components; [its final review](28_scratch_and_refinement.md)
records visible failures and an unmet showcase-quality goal. The [new results](26_learning_and_text_results.md)
include numerical likelihood sensitivity and the unsuccessful CIFAR consistency
refinement sweep.

| Final run | Generated / reference images | FID | IS |
|---|---:|---:|---:|
| [CIFAR EDM with CFG](../assets/experiments/cifar10/cifar10_edm_cfg_final/metrics.json) | 10,000 / 10,000 | 8.97 | 8.96 |
| [CIFAR flow, 50 steps with CFG](../assets/experiments/cifar10/cifar10_flow50_final/metrics.json) | 10,000 / 10,000 | 10.00 | 9.13 |
| [Face flow](../assets/experiments/faces/celeba_flow_final/metrics.json) | 10,000 / 2,000 | 16.31 | 2.27 |
| [CIFAR DiT with CFG](../assets/experiments/cifar10/cifar10_dit_cfg_final/metrics.json) | 10,000 / 10,000 | 8.93 | 9.27 |
| [128px latent face DiT](../assets/experiments/faces/celeba128_latent_dit_final/metrics.json) | 10,000 / 2,000 | 23.72 | 2.71 |

Visual inspection of final image grids, one-step consistency digits, and the
labeled 2D gallery supports featuring those examples. Not every output is
equally strong. The 20-step CIFAR flow has FID 22.24 despite IS 9.53, showing
that high IS alone does not establish fidelity. Short baselines, standalone
one-step consistency training, and classifier/CLIP guidance have visible
limitations; retain them as documented experiments rather than headline results.

The official CIFAR test split was used during development. Face metrics use
only 2,000 reference images. Read the [protocol](12_image_experiments.md) before
comparing numbers with paper benchmarks; face IS is not an object-diversity goal.

## Checks and release state

The current scientific suite and notebook checks are recorded in
[validation](../assets/metadata/validation.json). Lint and formatting include notebook
source cells. The ten lessons execute independently in fresh kernels, with
static plots, animations, small training runs and mathematical checks; the
[notebook report](../assets/reports/diagnostics/notebook_validation.json) records counts,
timings and hashes. The posterior clean-endpoint lesson identified a float32
rounding issue; the endpoint is now enforced exactly with a regression test.
Version 0.3 adds the five research extensions documented in
[extension results](21_extension_results.md). Their code and analytic/small-data
checks and final DiT/latent/CIFAR-distillation evaluations are available.
The baseline and extension artifact audits have separate scopes.
Progressive MNIST distillation completed all five stages, reaching 96.3%
requested-class accuracy with one call. Initial iCT runs were invalidated by
an autocast gradient-cache issue; the corrected run completed training from scratch.
Corrected one-call iCT reaches digit feature distance 43.60 and 96.1% class accuracy.
The regression suite now verifies an actual BF16 convolution update, and the
corrected objective also passed a three-GPU DDP training smoke test.
The [validation report](../assets/metadata/validation.json) also retains the earlier
installed-wheel, resume, and three-GPU DDP checks. The
[publication integrity report](../assets/reports/diagnostics/publication_review.json)
records final checkpoint hashes and sample/export checks.
The [extension integrity audit](../assets/reports/diagnostics/extension_integrity.json)
checks saved checkpoint hashes, finite samples, counts and exact PNG metric
pixels for new completed evaluations. These checks do not establish subjective
sample quality.

The [learning-extension integrity audit](../assets/reports/diagnostics/learning_integrity.json)
checks 17 completed image evaluations, six GMM checkpoint identities and six
likelihood cases against actual weights, counts and exported pixels. The
[version 0.4 installed-wheel audit](../assets/reports/diagnostics/wheel_v040.json)
verifies 45 Python modules and 50 bundled Hydra configurations outside the
checkout. These audits establish reproducible evidence and integration; they
do not establish uniformly strong generated images or bitwise text-training
continuation.

The [public asset audit](../assets/reports/diagnostics/public_asset_review.json) records
the current file count, footprint and successful PNG/GIF frame decoding.
The report measures the candidate source/assets footprint and largest files.
The [Git file-set audit](../assets/reports/diagnostics/publication_files.json) evaluates
the actual ignore rules and checks essential public files, representative
exclusions, file sizes and limited credential patterns. All 45 generator and
four classifier checkpoint files were rehashed against the manifest in the
[checkpoint review](../assets/reports/diagnostics/checkpoint_review.json).
Outputs, weights, build artifacts, and borrowed
legacy source are excluded by `.gitignore`. A limited credential-pattern scan
found no matches in candidate text files; that is not an exhaustive security audit.

The [publication layout](27_publication.md) keeps four historical completion/watch
helpers in the ignored local session directory and indexes the reusable tools.
Package builds exclude local configurations and non-config payloads; both wheel
and source builds passed a synthetic exclusion fixture. Asset export preserves
published evidence when local outputs are absent and merges new experiment
reports without clearing unrelated public baselines. The implemented core and
its documented experiments are ready for portfolio review; the release tasks
below still apply.

The educational implementation is suitable for a GitHub portfolio. These release
tasks remain:

1. Choose and add a license for the new implementation. No license is currently
   present; none was selected on the author's behalf.
2. Publish downloadable checkpoints, with the existing
   [hash manifest](../assets/metadata/checkpoints.json), so readers can sample immediately
   as well as retrain from configurations. The weights currently remain local.
3. Initialize a Git repository and publish the reviewed source/assets. Author
   citation metadata is supplied in [CITATION.cff](../CITATION.cff). The
   checkout currently has no Git repository or remote, and no push was performed.
   Continue excluding the borrowed legacy source unless its provenance is resolved.

The README now features final trained examples. Keep weak and interim runs
clearly identified in the full results gallery; documenting failed experiments
is useful evidence, while presenting all samples as uniformly good would be inaccurate.
The [repository audit](22_repository_audit.md) prioritizes remaining release,
quality and evaluation work and lists optional new research modules.
