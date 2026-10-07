# Repository tools

Run these tools from the repository root with `python -m scripts.<name>`.
Model training, sampling and numerical components live in `diffusion_lab/`;
these scripts reproduce studies, build educational figures or review evidence.
Install the optional extras needed by the chosen tool as described in the
[main README](../README.md).

| Purpose | Tools |
|---|---|
| Obtain and verify image data | `download_benchmarks`, `download_coco` |
| Build and execute the eleven-lesson course | `build_notebooks`, `execute_notebooks` (reviewable cells in `notebook_lessons.py`) |
| Compare/export portable GANs | `audit_adversarial` (`--public-only` on a fresh clone), `generate_adversarial` |
| Interactive concept explorer | `create_explorer`, `check_explorer_browser` (optional Playwright), `validate_release` |
| Review and test distribution contents | `review_components`, `check_distribution` |
| Run teaching comparisons | `run_research_suite`, `run_comparisons`, `run_quality_comparisons`, `run_guidance_examples` |
| Run advanced evidence studies | `run_evidence_studies`, `run_image_seed_study`, `run_progressive`, `run_extension_checks` |
| Summarize measured evaluations | `summarize_repeats`, `summarize_likelihood_sensitivity`, `summarize_text_evaluation`, `evaluate_prompt_pairs` |
| Make result galleries | `build_gallery`, `build_advanced_gallery`, `build_guidance_gallery`, `build_interpolant_figures`, `build_extension_results`, `build_text_gallery` |
| Export compact results and hashes | `export_assets`, `export_learning_assets` |
| Refresh dataset asset browsing indexes | `build_asset_index` (grouping shared in `asset_layout.py`) |
| Review publication files and figures | `check_publication`, `check_docs`, `review_public_assets` |
| Review locally retained experiment tensors and checkpoints | `review_extension_artifacts`, `review_learning_artifacts` |
| Verify or continue a training run | `check_resume`, `continue_image_training` |
| Compare consistency refinement with separate selection/final subsets | `compare_consistency_refinement` |
| Test caption use on shared held-out corruptions | `evaluate_text_denoising` |
| Curate a completed scratch/refinement study | `export_completion_results` |

`_publication.py` is the shared Git file-set helper used by the publication
checks. It evaluates project ignore rules with an empty temporary Git index;
it does not initialize or modify this checkout's Git repository.

## Common commands

```bash
# Data downloads are explicit and stay outside the source tree.
python -m scripts.download_benchmarks --root /your/image_data
python -m scripts.download_coco --root /your/image_data/coco2017

# CPU checks; do not train models or download datasets.
python -m scripts.check_docs
python -m scripts.check_publication
python -m scripts.review_public_assets
python -m scripts.execute_notebooks

# Publish selected local experiment evidence, excluding checkpoints/tensors.
python -m scripts.export_assets
```

A local run can include `publication.json` with `{"include": false}` to opt
out of the broad exporter. This is useful for intermediate diagnostics.
`export_completion_results` validates its final study manifest and copies only
explicit reports and the first representative outputs; it preserves historical
results and never exports checkpoints or raw sample tensors.

Artifact-integrity tools and report builders expect the local checkpoints,
saved sample tensors and directory conventions of the published experiments.
They are optional on a fresh clone; the exported reports remain readable
without those weights. Study runners also require their declared teacher or
assessment checkpoints. See [evidence protocols](../docs/25_evidence_studies.md)
and [text reproduction](../docs/24_text_conditioning.md) before launching them.

Four historical completion/watch scripts were moved into the ignored local
session directory. They orchestrated earlier runs and are not public APIs.
Use the documented model entrypoints and Slurm templates for new training.
