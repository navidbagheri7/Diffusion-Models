# Public repository layout and release checks

The public repository contains the implementation, Hydra configurations,
scientific tests, course notebooks, documentation, cluster templates and
selected experiment evidence. Training state and machine-specific working
files remain local. [`.gitignore`](../.gitignore) implements these exclusions;
[`.gitattributes`](../.gitattributes) keeps text line endings consistent.

The [asset showcase](../assets/README.md) is the visual entry point. Six groups
separate teaching figures, dataset-specific experiments, controlled studies,
reports, metadata and historical previews. Exporters use the same layout;
`python -m scripts.build_asset_index` refreshes the browsing indexes.

| Include in Git | Keep local or host separately |
|---|---|
| `diffusion_lab/`, `configs/`, `tests/` | `outputs/`, Hydra `multirun/`, logs and raw sample tensors |
| `docs/`, `notebooks/`, README and citation metadata | Dataset copies, downloaded archives and model caches |
| Reusable `scripts/`, `slurm/`, CI workflow | Historical completion helpers in `.local/` |
| `assets/` figures, GIFs, configs, metric reports and hashes | Model, optimizer and classifier weights, including `.pt`, `.ckpt`, `.bin` and `.safetensors` |
| `legacy/README.md` provenance note | Borrowed legacy source whose attribution/license is unresolved |

Weak, interim and invalidated experimental reports remain when a lesson uses
them to explain a limitation. They must retain their validity labels. A public
figure is not a quality guarantee. The public evidence currently includes the
complete educational gallery; its measured footprint is in the
[asset audit](../assets/reports/diagnostics/public_asset_review.json).

## Check the actual publication file set

```bash
python -m pytest -q
ruff check diffusion_lab tests scripts notebooks train_diffusion.py setup.py
ruff format --check diffusion_lab tests scripts notebooks train_diffusion.py setup.py
python -m scripts.check_docs
python -m scripts.check_publication
python -m scripts.review_public_assets
python -m scripts.execute_notebooks
```

The publication check evaluates Git's own ignore rules using a temporary empty
index. It verifies representative exclusions, essential public files, symlinks,
the project's 20 MiB per-file review limit and limited credential patterns.
Reports never print matching credential values. This is not an exhaustive
security scan. When this checkout has a Git index, the tool also reports ignored
files which are already tracked. It does not alter the index, initialize Git,
push commits or delete local files.
The [saved file-set review](../assets/reports/diagnostics/publication_files.json) records
the checks for this checkout. Wheel/source builds also exclude `configs/local/`
and raw weights; those boundaries were checked with synthetic local-only files.

Documentation checks verify that linked local files will be included, rather
than merely existing on this machine. Raw `outputs/` paths in command examples
are produced by running those commands; they are not downloadable assets.

If files were previously committed, `.gitignore` does not untrack them. Review
the index and remove unwanted paths from the index before publication, keeping
their local copies. No Git repository currently exists in this checkout, so
there is no existing index to clean here.

## Reproducibility and checkpoint distribution

Resolved experiment configs, metric preprocessing/counts, seeds, evaluation
commands, validity labels and exact weight hashes accompany exported evidence.
Export merges new runs into the public index and preserves existing evidence
when local training state is absent.
The [checkpoint manifest](../assets/metadata/checkpoints.json) records historical local
paths; those paths are provenance, not portable download links. Use the matching
immutable checkpoint, rather than assuming a mutable latest path still has the
same weights. Host selected model checkpoints separately and add verified
download locations when they are available. External pretrained model weights
and raw datasets are not bundled; obtain them from their sources.

Some scientific audits require the locally retained `outputs/` files. They are
not prerequisites for opening the lessons or running the CPU tests on a fresh
clone. The [script index](../scripts/README.md) distinguishes those tools from
the public-file checks.

## Current completion and remaining work

The [coverage map](15_coverage_and_scope.md) and [current audit](22_repository_audit.md)
state what is implemented, trained, evaluated and still limited. Text examples
include SD2 LoRA fine-tuning and a compact COCO denoiser trained from random
initialization with frozen pretrained VAE/text components. The 100k COCO and
40k CIFAR consistency budgets and final evaluations have completed; the
[final visual review](28_scratch_and_refinement.md) records an unmet
showcase-quality goal. CIFAR multi-call consistency refinement remains weak.
Likelihood numbers have numerical
and estimator limitations; independent image training repetitions cover MNIST,
not every earlier headline image generator.

Author metadata is supplied in [CITATION.cff](../CITATION.cff). A license for the
new implementation has not been selected. Public checkpoint hosting and GitHub
publication also remain release tasks. These choices do not prevent local
review and training, but must be resolved before describing a public release
as fully packaged and independently reproducible from downloaded weights.

## Version 0.5 local release and optional static hosting

The owned code now has an [MIT license](../LICENSE), and local Git history
preserves the pre-extension fitting source. Small trained GAN generators are
public; large historical diffusion/flow weights remain local and optional.
No remote push or site deployment has been performed.

To publish the static explorer yourself, push the reviewed repository, then
choose GitHub Pages **Deploy from a branch → main → / (root)**. The supplied
`.nojekyll` preserves the exact offline assets and relative evidence links.
Your demo URL is `https://YOUR_ACCOUNT.github.io/YOUR_REPOSITORY/docs/explorer.html`.
Keep both `docs/` and `assets/` in the published tree. The README screenshot is a
preview; the linked HTML supplies the actual controls. Opening that same file
locally or serving the root with `python -m http.server 8000` requires no account.
