# Validation and assessment reports

[Asset showcase](../README.md)

| Folder | Purpose |
|---|---|
| [diagnostics/](diagnostics/) | Scientific, artifact, notebook, package and publication checks |
| [datasets/](datasets/) | CIFAR-10, MNIST and COCO decoding verification; no dataset images |
| [classifiers/](classifiers/) | Classifier configurations, histories and assessment reports; no weights |
| [distribution_quality/](distribution_quality/) | Coverage, class diagnostics, nearest-image comparisons and seed intervals |

The [notebook execution](diagnostics/notebook_validation.json),
[asset decoding audit](diagnostics/public_asset_review.json) and
[validation summary](../metadata/validation.json) describe their specific scopes.
Earlier reports retain their provenance and are not automatically rerun when
files are reorganized. No one check establishes image quality, mathematical
correctness and release readiness together.
