# Primary references and implementation scope

The explanations are written for this project, and the new code is an original
implementation of the equations. The sources below identify the mathematical
methods and provide deeper derivations. No paper's reported scores are presented
as results of this repository.

| Topic | Primary source | Scope in this lab |
|---|---|---|
| DDPM | Ho, Jain, Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | Discrete VP corruption, denoising regression, fixed-posterior ancestral sampling |
| DDIM | Song, Meng, Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | Deterministic and eta-controlled updates, respaced grids |
| Improved DDPM | Nichol, Dhariwal, [Improved DDPM](https://arxiv.org/abs/2102.09672) | Cosine schedule, learned diagonal variance, hybrid/VLB objectives and loss-second-moment timestep sampling |
| Continuous score models | Song et al., [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | VP/VE/sub-VP, reverse SDE, probability-flow ODE, Langevin predictor–corrector |
| Diffusion v | Salimans, Ho, [Progressive Distillation](https://arxiv.org/abs/2202.00512) | Prediction target and conversions; trained MNIST progressive distillation chain |
| Classifier guidance | Dhariwal, Nichol, [Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233) | Noisy-time classifiers and score gradients |
| CFG | Ho, Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | Null-token training and conditional/unconditional mixing |
| Text cross-attention | Rombach et al., [Latent Diffusion](https://arxiv.org/abs/2112.10752) | Token-sequence conditioning in our compact U-Net and the local pretrained SD2 U-Net |
| Captioned data | Lin et al., [Microsoft COCO](https://arxiv.org/abs/1405.0312) | Verified COCO 2017 image/caption manifests with image-disjoint fine-tuning validation/evaluation splits |
| Low-rank adaptation | Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) | Native attention-projection adapters; frozen pretrained denoiser |
| Min-SNR | Hang et al., [Efficient Diffusion Training via Min-SNR Weighting](https://arxiv.org/abs/2303.09556) | Optional target-specific noise/VP-v loss weights for caption training |
| Flow matching | Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Conditional velocity regression and marginal transport |
| Rectification | Liu, Gong, Liu, [Flow Straight and Fast](https://arxiv.org/abs/2209.03003) | Linear interpolation, teacher-endpoint reflow, measured straightness |
| Minibatch OT | Tong et al., [Improving and Generalizing Flow-Based Generative Models with Minibatch OT](https://arxiv.org/abs/2302.00482) | Equal-weight exact batch assignment; no population OT guarantee |
| Entropic OT | Cuturi, [Sinkhorn Distances](https://arxiv.org/abs/1306.0895) | Stable finite-support Sinkhorn coupling and marginal checks |
| Schrödinger problem | Léonard, [A Survey of the Schrödinger Problem](https://arxiv.org/abs/1308.0215) | Brownian-reference path KL and endpoint OT connection |
| Bridge matching | Shi et al., [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852) | Direct empirical endpoint solve and alternating neural DSBM-IMF/IPF on 2D data |
| Diffusion solvers | Lu et al., [DPM-Solver](https://arxiv.org/abs/2206.00927), [DPM-Solver++](https://arxiv.org/abs/2211.01095) | First-order and second-order multistep integrators; no third-order/adaptive DPM variant |
| EDM | Karras et al., [Elucidating the Design Space](https://arxiv.org/abs/2206.00364) | Noise preconditioning, log-normal weighted denoising, Karras grid, Heun and optional churn |
| Consistency | Song et al., [Consistency Models](https://arxiv.org/abs/2303.01469) | Boundary parameterization, standalone MSE CT, teacher Heun CD and one/few-step generation |
| CLIP guidance | Liu et al., [Semantic Diffusion Guidance](https://arxiv.org/abs/2112.05744) | Differentiable CLIP energy on denoised estimates; heuristic rather than a calibrated noisy posterior |
| Adaptive ODE | [SciPy RK45](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html) | Small-problem Dormand–Prince wrapper |
| MMD | Gretton et al., [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html) | Unbiased squared RBF statistic |
| FID | Heusel et al., [GANs Trained by a Two Time-Scale Update Rule](https://arxiv.org/abs/1706.08500) | Pretrained Inception Gaussian feature distance |
| KID | Bińkowski et al., [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401) | Polynomial feature-kernel MMD via torchmetrics |
| IS | Salimans et al., [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) | Pretrained classification KL score via torchmetrics |

## Exact statements versus approximations

Additional implementation references for version 0.3:

- Albergo, Boffi and Vanden-Eijnden, [Stochastic Interpolants](https://arxiv.org/abs/2303.08797): conditional velocity/score regression and matching ODE/SDE marginals.
- Salimans and Ho, [Progressive Distillation](https://arxiv.org/abs/2202.00512): two deterministic teacher steps distilled into one.
- Song and Dhariwal, [Improved Consistency Training](https://arxiv.org/abs/2310.14189): online target, lognormal intervals, doubling schedule and Pseudo-Huber.
- Kynkäänniemi et al., [Improved Precision and Recall](https://arxiv.org/abs/1904.06991), and Naeem et al., [Density and Coverage](https://proceedings.mlr.press/v119/naeem20a.html): complementary feature-neighborhood metrics.
- Rombach et al., [Latent Diffusion](https://arxiv.org/abs/2112.10752): generation in a frozen autoencoder representation.
- Peebles and Xie, [DiT](https://arxiv.org/abs/2212.09748): patch tokens and adaLN-Zero transformer denoisers.

Algebraic target conversions and analytic GMM scores can be tested to floating
point tolerance. Equality of SDE/ODE marginals assumes an exact score, suitable
regularity, matching initial marginals, and exact integration. Learned models,
finite terminal priors, solver grids, and truncated endpoints violate those
ideal conditions to varying degrees.

Straight conditional paths do not imply straight learned ODE trajectories.
Minibatch OT does not imply population OT. Entropic empirical bridge solutions
do not reproduce a full image-scale continuous-distribution Schrödinger bridge.
Short-run metrics are teaching diagnostics and are reported with their protocols.
