# Start here: a guided course through the laboratory

The reference chapters explain the mathematics; the notebooks let you change a
modeling choice and inspect its consequences. Start with CPU examples before
training image models. Each unit below has a question, an experiment and a
checkpoint for understanding. A GPU is optional until the image-training units.

## Preparation

You need Python functions, basic PyTorch tensors, vectors/matrices, derivatives,
and the mean/variance of a Gaussian. Conditional expectation means averaging a
hidden quantity over the possibilities compatible with what you observed.
Least-squares regression learns this conditional average, which explains why
noisy per-pair training targets can learn deterministic marginal fields.

```bash
python -m pip install -e ".[dev,notebooks]"
python -m pytest -q
jupyter lab notebooks
```

JupyterLab also opens the notebooks. Each notebook includes static default plots
for GitHub readers and optional widgets for a running kernel. Cells use fixed
seeds, CPU-sized examples and no automatic downloads or image-scale training.
The denoising, flow, attention and consistency lessons include short educational
training runs. Each notebook is independently executable from a fresh kernel;
follow the [ten-notebook course index](../notebooks/README.md) for its contents,
worked exercises and optional GPU extensions.
Run `python -m scripts.execute_notebooks` to execute every lesson headlessly.

Read the [notation reference](23_notation_and_code.md) whenever a symbol or time
direction is unfamiliar. Try the [exercises](exercises.md) before checking their
worked answers. Actual measured results are linked separately from demonstrations.

## Course sequence

| Unit | Question and experiment | Read | Run | Check your understanding |
|---|---|---|---|---|
| 0. Probability and corruption | How does Gaussian noise change a distribution? Change time and the schedule. | [Foundations](01_foundations.md), [notation](23_notation_and_code.md) | [Noising notebook](../notebooks/00_probability_and_noising.ipynb) | Distinguish a random sample, density, score and noise standard deviation. |
| 1. DDPM and prediction targets | Why can the same corruption be modeled through noise, clean data, score or VP-v? Verify conversions and compare DDIM step counts. | [Diffusion](02_diffusion.md), [Improved DDPM](09_advanced_diffusion.md) | [Targets/DDIM notebook](../notebooks/01_prediction_targets_and_ddim.ipynb) | Explain what is learned by MSE, and why a conditional Gaussian score differs from the marginal score. |
| 2. Scores, SDEs and ODEs | How can stochastic and deterministic trajectories transport the same density? Generate from an analytic GMM score. | [Foundations](01_foundations.md), [solvers](05_numerical_methods.md) | [SDE/ODE notebook](../notebooks/02_scores_sdes_and_odes.ipynb) | Write reverse-SDE and probability-flow drifts and identify the time direction. |
| 3. Flow matching and transport | What velocity does conditional regression learn? Compare solvers on a known Gaussian flow. | [Flows/OT](03_flows_and_transport.md), [interpolants](17_stochastic_interpolants.md) | [Flow/solver notebook](../notebooks/03_flow_matching_and_solvers.ipynb) | Distinguish endpoint-pair paths from marginal ODE trajectories and integration steps from model calls. |
| 4. Guidance | What changes when a generator receives a condition? Sweep exact-score CFG strength. | [Guidance](04_guidance.md) | [Guidance notebook](../notebooks/04_guidance.ipynb) | Identify null/conditional branches and explain why strong guidance can change diversity. |
| 5. Likelihood and evaluation | What does a likelihood estimate or FID actually establish? Verify numerical change of variables and examine measured failures. | [ELBO](16_likelihood_and_elbo.md), [distribution metrics](19_evaluation_beyond_fid.md), [evidence studies](25_evidence_studies.md) | [Likelihood/evaluation notebook](../notebooks/05_likelihood_and_evaluation.ipynb) | Separate model error, numerical error, finite-sample error and training uncertainty. |
| 6. Image/text models | How does each image location access caption tokens? Inspect cross-attention and run the training pipeline. | [Latent/DiT](20_latent_diffusion_and_dit.md), [text conditioning](24_text_conditioning.md) | [Cross-attention notebook](../notebooks/06_text_cross_attention.ipynb) | Explain which components are pretrained and which parameters are trained. |
| 7. Interpolants, OT and bridges | How do coupling, noise and Markov projection change a transport? | [Interpolants](17_stochastic_interpolants.md), [bridges](06_bridges.md), [iterative matching](11_iterative_bridges.md) | [Interpolants/bridges notebook](../notebooks/07_interpolants_transport_and_bridges.ipynb) | Distinguish an interpolant sample, Brownian bridge joint law and learned Markov drift. |
| 8. EDM and few-step models | What is preconditioning, and why can extra refinement fail? | [Consistency](10_consistency_models.md), [distillation](18_stronger_few_step_generation.md) | [EDM/consistency notebook](../notebooks/08_edm_and_few_step_models.ipynb) | Explain CT/CD/iCT targets, boundary conditions and actual branch-call counts. |
| 9. Latent diffusion and DiT | How do image architectures and stochastic latent representations connect to training? | [Latent/DiT](20_latent_diffusion_and_dit.md), [image practice](12_image_experiments.md) | [Latent/DiT notebook](../notebooks/09_latent_diffusion_and_dit.ipynb) | Trace patch tokens, frozen codec scaling, posterior draws and Hydra composition. |

## From a lesson to your own research

The [coverage map](15_coverage_and_scope.md) distinguishes implementations,
experiments and scope limits. After understanding a unit, reproduce its command,
then change one factor while holding the rest fixed. Save the resolved config,
training seed, checkpoint hash, evaluation seed, reference subset and actual
network calls. Follow the [experiment chapter](07_experiments_and_metrics.md)
and [new evidence protocol](25_evidence_studies.md).

For image training, begin with MNIST/CIFAR. For text conditioning, start with a
short integration run before the full COCO budget. A passed integration run is
not a sample-quality claim. The repository includes unsuccessful experiments
because diagnosing them is part of learning the methods.
