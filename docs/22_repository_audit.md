# Repository audit and completion status — version 0.4

The toolkit implements the requested diffusion/flow core and the version 0.4
educational and text-conditioning extension. A supported implementation, a
completed training budget and a strong general benchmark result are separate
claims. The [coverage map](15_coverage_and_scope.md) records the supported scope;
[current results](26_learning_and_text_results.md) records actual measurements.
This review does not claim that every research concept or experiment is complete.

## Completed implementation and evidence

- DDPM/DDIM, continuous VP/VE/sub-VP, schedule families, prediction conversions,
  Tweedie, score SDEs/PF-ODEs, density equations and numerical solvers.
- Conditional/unconditional generation, CFG, noisy classifier guidance,
  documented heuristic CLIP energy, and image interpolation.
- Flow matching, OT/reflow, Gaussian-noise stochastic interpolants, 2D Brownian
  bridges and alternating DSBM.
- Improved DDPM, diagonal learned variance, full DDPM ELBO, numerical ODE
  likelihood, EDM, consistency/iCT and progressive distillation.
- Image U-Nets, pixel DiT, frozen-codec latent diffusion and latent DiT,
  custom data/model interfaces and cluster configurations.
- Guided lessons, shared notation, worked exercises and
  [ten independently executable notebooks](../notebooks/README.md) covering
  69 code cells and 43 saved visual outputs.
- Verified COCO images and caption manifests, masked token cross-attention,
  text CFG, native LoRA and epsilon/VP-v-specific Min-SNR weights.

The current scientific suite contains 124 passing tests. The
[validation report](../assets/metadata/validation.json) records the tested environment
and distinguishes current checks from earlier DDP/resume/wheel checks. Tests
cover analytic identities, finite-difference derivatives, numerical behavior,
conditioning, dataset isolation and model updates; they are not a proof that
every numerical or modeling choice is universally correct.

The [learning integrity audit](../assets/reports/diagnostics/learning_integrity.json)
verifies hashes, finite samples, counts and exported pixels for 17 image
evaluations, six independently trained GMM models and six likelihood cases.
The [extension audit](../assets/reports/diagnostics/extension_integrity.json) retains
the earlier DiT/latent/few-step evidence. The
[public asset audit](../assets/reports/diagnostics/public_asset_review.json) records
actual decoding and publication footprint. Scientific scores were not changed
by this publication-layout review.

## Completed experiments and remaining limitations

| Area | Completed evidence | Remaining work or limit |
|---|---|---|
| Main image generators | CIFAR/face diffusion and flow, CIFAR DiT and both latent face models completed their configured 100k-update budgets | Artifact quality varies; face references are only 2,000 images, and development used official test splits |
| Few-step generation | Progressive MNIST distillation and corrected iCT; old 30k CD and new 40k CIFAR CD/iCT students | Matched 5k/5k comparisons completed: revised CD improves one call modestly, two/four-call CD remain weak; iCT refinement helps its own model but trails the best one-call CD |
| Likelihood | Full MNIST negative ELBO on 1,024 examples; MNIST ODE on 1,024 and CIFAR ODE on 512; matched 128-example grid/probe/noise checks | ELBO is a bound, ODE density is numerical, and the measured grid sensitivity is material |
| Training uncertainty | Three independent GMM DDPM/flow seeds and three independent MNIST EDM/flow seeds per family | Earlier CIFAR/face headline models still lack training-seed repetitions |
| Text conditioning | SD2 LoRA at 512px, 3,000 updates/24,000 image-caption pairs, eight matched evaluations and prompt/CFG galleries | Pretrained-model fine-tuning, less than one COCO epoch; anatomy/counting errors and unknown external pretraining overlap |
| Compact text U-Net | Implemented, gradient/masking-tested; 256px scratch-denoiser training launched on COCO | The 100k-update budget and held-out evaluation completed; its VAE/text encoder remain external and frozen, and generation quality is reported separately |
| Nonlinear interpolants | Differentiable custom-base API and derivative tests | Saved learned transport examples use affine endpoint families |
| Bridges and latent density | Trained 2D bridge examples and frozen-codec latent generation | Image-scale bridges and a full encoder/decoder/prior pixel-likelihood calculation are absent |

Both base and adapted text models pass six small caption-preference decisions.
The larger matched CLIP cosine difference is +0.0056, with caption-wise interval
[0.0026, 0.0086]; this single fine-tuning seed does not establish improved visual
quality, counting or spatial binding. No headline text FID is reported.

## Publication cleanup completed

The public file set keeps the source, configs, tests, documentation, notebooks,
reusable experiment tools, Slurm templates and educational assets. The ignored
local set includes `outputs/`, datasets, archives, weight formats, raw tensors,
logs, caches, environments, editor settings, credentials and historical session
helpers. Borrowed legacy source remains local, with its provenance README public.
Four completed-run watch/orchestration scripts were preserved in `.local/`.
Their older report schemas and fixed run names are not public APIs.

The publication audit checks Git's actual ignore rules. Documentation checks
also verify that linked files are included in the publication set. The
[script index](../scripts/README.md) explains which tools can run on a fresh
clone and which require the local experiment tensors/checkpoints. Read the
[publication guide](27_publication.md) before creating a release.

## Remaining release tasks and optional extensions

The version 0.5 update adds an MIT license, local Git history, portable small
GAN generators and an offline explorer. Large diffusion/image checkpoints remain
local and are not publicly hosted. Author metadata is present in
[CITATION.cff](../CITATION.cff). No GitHub CI run or remote publication is claimed.

Optional research additions include inverse problems/posterior sampling, UniPC,
a matched latent-DiT flow experiment, learned nonlinear-interpolant transport
and image-scale bridges. These are future modules, rather than requirements
for releasing the documented educational core. Scratch COCO training and the
weak CIFAR consistency refinement should remain visibly unfinished until their
experiments support stronger claims.

The [new study](28_scratch_and_refinement.md) retains intermediate examples
and records the random-denoiser/pretrained-codec boundary. CIFAR training and
matched comparisons completed with mixed results. COCO's configured training and evaluation budgets completed.
The study retains the intermediate visual review and final measured results;
final showcase quality requires visual assessment of the completed model.
The overall showcase-quality goals remain unmet.
