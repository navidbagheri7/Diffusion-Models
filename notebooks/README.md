# A guided, executable course

These eleven notebooks connect the library's Python implementation to equations,
small experiments, plots, animations and worked exercises. They are teaching
companions to the `.py` package; the training and sampling implementations
remain in `diffusion_lab/`.

Start at 00 and follow the numbered sequence. Each notebook is independently
executable from a fresh kernel; no variables or checkpoints from another
notebook are required. Prerequisites and learning outcomes appear at its start.

| Notebook | What you will do |
|---|---|
| [00 · Probability and noising](00_probability_and_noising.ipynb) | Read densities and tensor shapes, compare six schedules, check moments and animate corruption |
| [01 · Targets and DDPM/DDIM](01_prediction_targets_and_ddim.ipynb) | Derive Tweedie/VP-v conversions, inspect the posterior, train a small denoiser and count sampler calls |
| [02 · Scores, SDEs and ODEs](02_scores_sdes_and_odes.ipynb) | Check score gradients and SDE coefficients, animate transport and verify Fokker–Planck/continuity residuals |
| [03 · Flow matching and solvers](03_flow_matching_and_solvers.ipynb) | Train a velocity field, compare Euler/Heun/RK4/RK45 by actual calls, explore interpolation and reflow |
| [04 · Guidance](04_guidance.ipynb) | Verify Bayes' score identity, compare CG/CFG, inspect null-label dropout and branch costs |
| [05 · Likelihood and evaluation](05_likelihood_and_evaluation.ipynb) | Decompose the full ELBO, inspect learned variance, test ODE density/traces and diagnose collapse with complementary metrics |
| [06 · Text conditioning](06_text_cross_attention.ipynb) | Inspect token masks and attention shapes, train an explicitly synthetic attention exercise, check LoRA/text CFG and examine COCO failures |
| [07 · Interpolants, OT and bridges](07_interpolants_transport_and_bridges.ipynb) | Differentiate custom paths, compare matching ODE/SDE fields, solve empirical OT/Sinkhorn and simulate joint Brownian bridges |
| [08 · EDM and few-step models](08_edm_and_few_step_models.ipynb) | Inspect preconditioning and Karras grids, verify consistency boundaries, train a small CD student and assess actual refinement results |
| [09 · Latent diffusion and DiT](09_latent_diffusion_and_dit.ipynb) | Trace architecture shapes/gradients, sample latent caches, compose Hydra configs and optionally inspect a frozen codec |
| [10 · Adversarial objectives](10_adversarial_objectives.ipynb) | Compare fifteen selected GAN generators, verify a gradient-penalty oracle and inspect objective/stabilization tradeoffs |

## Install, run and explore

From the repository root:

```bash
python -m pip install -e ".[dev,notebooks]"
jupyter lab notebooks
```

Choose a kernel from that same Python environment. If Jupyter lists only a
different environment, register this one with
`python -m ipykernel install --user --name diffusion-flow-lab --display-name "Diffusion Flow Lab"`.
Select it in Jupyter, or pass `--kernel diffusion-flow-lab` to the execution runner.

Default cells run on CPU and do not download datasets, weights or start long
image-training jobs. Small training loops use fixed seeds and independent
evaluation draws. Their samples demonstrate learning, not a competitive benchmark.
The text and latent lessons show saved public evidence alongside those demonstrations.

Static outputs render without an active kernel. Set `ENABLE_WIDGETS = True`
after the setup cell to enable interactive controls where provided, or simply
edit the experiment parameters and rerun. GitHub will show the saved default
outputs; it does not execute sliders. GIFs also live in
[the notebook assets folder](../assets/teaching/notebooks/).

Use each lesson's exercises before opening its worked answers. Differentiate
oracle results, short-trained models and actual saved image experiments; each
is labeled explicitly. Pay attention to time direction and endpoint cutoffs.

## Image-scale and GPU extensions

Follow the chapter links inside each notebook for full training commands,
custom datasets/models and [Slurm templates](../slurm/README.md). A multi-GPU
training job belongs in the scripts with saved checkpoints, rather than an
implicit notebook side effect.

Notebook 09 has an explicitly optional frozen-codec reconstruction cell. Install
the latent extra and set your own local paths before launching Jupyter:

```bash
python -m pip install -e ".[latent,notebooks]"
export DIFFUSION_NOTEBOOK_VAE=/path/to/local/vae
export DIFFUSION_NOTEBOOK_IMAGE=/path/to/example.jpg
export DIFFUSION_NOTEBOOK_DEVICE=cuda:0
jupyter lab notebooks
```

Reconstruction is distinct from generation. Neither codec access nor a passed
shape test establishes image quality.

## Maintain and validate

The reviewable source cells live in
[notebook_lessons.py](../scripts/notebook_lessons.py); the
[builder](../scripts/build_notebooks.py) creates the notebooks and clears old
outputs. Execute them after changing their sources:

```bash
python -m scripts.build_notebooks
python -m scripts.execute_notebooks --timeout 180
python -m scripts.check_docs
python -m scripts.check_publication
```

For a focused rerun add `--notebook 05_likelihood_and_evaluation.ipynb`.
Each lesson gets a fresh kernel. The runner verifies execution, records static
visual counts and hashes in [the validation report](../assets/reports/diagnostics/notebook_validation.json).
The default `python3` execution kernel uses the interpreter running the command;
an explicit custom `--kernel` uses that registered kernel's interpreter instead.
Do not interpret CPU default validation as a test of optional GPU/model downloads.

The [curriculum](../docs/00_start_here.md), [notation](../docs/23_notation_and_code.md)
and [primary references](../docs/references.md) supply the larger context.
The [course revision report](../docs/29_notebook_course.md) records the expanded
content, observed short-training limitations and separate validation scopes.
