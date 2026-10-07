# Teaching plots and animations

[Asset showcase](../README.md) · [Guided course](../../notebooks/README.md)

| Lesson family | Examples |
|---|---|
| `gaussian_diffusion/` | [Noising](gaussian_diffusion/forward_corruption.png), [schedules](gaussian_diffusion/schedules.png), [score/Tweedie](gaussian_diffusion/score_tweedie.png), [density equations](gaussian_diffusion/density_equations.png) |
| `transport/` | [Labeled datasets](transport/toy_datasets.png), [rectification](transport/rectification_reflow.png), [moving mass](transport/reflow_mass.gif) |
| `solvers/` | [Generation quality versus calls](solvers/quality_vs_nfe.png) |
| `notebooks/` | [Corruption](notebooks/00_gaussian_corruption.gif), [learned flow](notebooks/03_learned_flow.gif), [Brownian bridges](notebooks/07_brownian_bridges.gif) |
| [interpolants/](interpolants/) | Interpolant figures and numerical diagnostics |
| [guidance/](guidance/) | Matched-seed image guidance comparisons and protocols |
| `galleries/` | [Learned GMM methods](galleries/gmm_gallery.png), [GMM guidance](galleries/guidance_gallery.png) |

Chapters distinguish analytic reference fields from trained models. The
Gaussian corruption GIF uses shared noise to show marginals; it does not
simulate the joint law of an SDE.
