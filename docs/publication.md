# GitHub publication and interactive demo

Repository: [navidbagheri7/Diffusion-Models](https://github.com/navidbagheri7/Diffusion-Models). Interactive explorer: [https://navidbagheri7.github.io/Diffusion-Models/](https://navidbagheri7.github.io/Diffusion-Models/).

The repository contains authored English code, executed notebooks and measured evidence. The numerical comparisons and documented limitations remain unchanged by publication. Private course originals, local environments, cached datasets and large training checkpoints remain outside Git history.

## Deploy the website

GitHub Pages uses **GitHub Actions** as its source. Run **Actions → Publish interactive preview → Run workflow** to refresh the site after pushing changes. The workflow uses `git archive HEAD` to stage exactly the committed public tree. It preserves the repository-relative links to numerical evidence, images, rollouts and notebooks. It does not copy ignored local files.

The root page opens `docs/explorer.html`.

## Reproduce the research

Use the installation and reproduction instructions in the README. GitHub Pages serves the recorded evidence and browser-side controls; fitting models and running Python experiments requires the documented Python environment.

Existing version tags preserve the local release history. The `publication-2026-10-12` tag identifies the first published portfolio snapshot, including its website deployment configuration and repository metadata. The package versions describe the preserved scientific implementation.
