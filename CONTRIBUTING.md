# Contributing an experiment or mathematical component

Keep a concept traceable from its equation to its implementation and visual
example. State the time direction, assumptions, supported process/target
combinations, and endpoint behavior. Avoid silently treating approximations
as exact identities.

For a new process, expose Gaussian coefficients when available and supply the
forward SDE coefficients. For a new target, document its loss weighting and
conversions. For a new solver, check an analytic convergence example before
interpreting learned-model results.

Run from the repository root:

~~~bash
python -m pytest -q
ruff format --check diffusion_lab tests scripts notebooks train_diffusion.py setup.py
ruff check diffusion_lab tests scripts notebooks train_diffusion.py setup.py
python scripts/check_docs.py
python -m scripts.check_publication
python -m scripts.execute_notebooks
~~~

Tests should establish scientific identities, conservation properties, endpoint
behavior, numerical convergence, or data isolation. Do not merely restate the
implementation in a second function.

Use named Hydra configurations and explicit run directories. Save real results,
including unsuccessful baselines, with their configuration and metric protocol.
Compare several seeds before drawing conclusions about a method. Keep raw
datasets and large checkpoints out of Git; export selected figures and metrics
with scripts/export_assets.py.
The [publication guide](docs/27_publication.md) describes the public layout.
Do not commit raw tensors, credentials, local environments, dataset copies or
cluster logs. The publication audit uses the actual `.gitignore` rules and
detects ignored files which were previously tracked; adding a pattern does not
untrack an existing file. Maintain explanatory figures and failure reports when
they are referenced by a lesson.
The [course](docs/00_start_here.md) and [notation reference](docs/23_notation_and_code.md)
define the common vocabulary. Notebook source cells live in
`scripts/build_notebooks.py`; build with the dev/notebooks extras installed,
then execute the notebooks before exporting their static outputs.

For external contributions, retain the source's attribution and applicable
license. The preserved legacy code is excluded from the new package and from
the intended public code tree.
