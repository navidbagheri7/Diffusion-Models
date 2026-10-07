# Evaluation beyond FID and Inception Score

A public result should expose quality, diversity, conditioning, cost and
uncertainty. FID compresses differences in feature means and covariances into
one number. IS evaluates generated classifier distributions without a real
reference. Neither alone reveals every missing class or memorized example.
[distribution_metrics.py](../diffusion_lab/distribution_metrics.py) adds
complementary measurements to the existing FID/KID/IS and 2D metrics.

## Precision, recall, density and coverage

Let $r_i$ be real feature vectors and $g_j$ generated vectors. For each real
example, let $R_i$ be its distance to the $k$th nearest **other** real example.
For each generated example define $G_j$ analogously among generated features.
Use closed Euclidean balls $B(c,R)$; all distances are computed in float64,
in blocks, without constructing an $N\times N\times D$ tensor.

$$\mathrm{Precision}=\frac1M\sum_j\mathbf1[g_j\in\bigcup_i B(r_i,R_i)],$$
$$\mathrm{Recall}=\frac1N\sum_i\mathbf1[r_i\in\bigcup_j B(g_j,G_j)],$$
$$\mathrm{Density}=\frac1{kM}\sum_j\sum_i\mathbf1[g_j\in B(r_i,R_i)],$$
$$\mathrm{Coverage}=\frac1N\sum_i\mathbf1[\min_j\|r_i-g_j\|\leq R_i].$$

Precision asks whether generated examples lie in the real feature manifold.
Recall asks how much of the real manifold the generated manifold contains.
Density counts overlapping real neighborhoods and can exceed one. Coverage
requires generated support near each real example using real-data radii only.
These are neighborhood diagnostics, not calibrated human-quality probabilities.
Feature choice, sample count, $k$, outliers and ties affect the measurements.
Identical settings are required for comparisons. A test uses a collapsed mode
and a distant distribution to verify the expected precision/recall distinction.

For RGB images we use the same torch-fidelity Inception pool3 representation
as FID. Every tensor is clipped, rounded to uint8 and processed as a rendered
image. For digits we use the held-out, clean-only assessment classifier at
$\sigma=0.001$, with its checkpoint hash retained. For toy data we use native
coordinates. Scores in these three feature spaces are not interchangeable.

~~~bash
python -m diffusion_lab.distribution_metrics \
  --experiment outputs/cifar10_edm_final outputs/cifar10_edm_cfg_final \
  --limit 2048 --k 3 --per-class --nearest-training --device cuda:0 \
  --output assets/reports/distribution_quality/rgb
python -m diffusion_lab.distribution_metrics \
  --experiment outputs/mnist_ddpm1000 outputs/mnist_cd1 \
  --feature-classifier outputs/mnist_assessment_classifier/classifier.pt \
  --limit 2048 --k 3 --per-class --device cuda:0 \
  --output assets/reports/distribution_quality/digits
~~~

The evaluator records its subsampling seed and indices. `--per-class` measures
each requested class separately when it has enough examples. Requested labels
describe the generator input; independent classifier accuracy checks whether
the output actually belongs to that class. Small per-class neighborhoods have
different estimation variance from the full distribution.

## Nearest neighbors and memorization

`--nearest-training` uses equally sized training and held-out image subsets.
Each saved row contains the generated image, its nearest training feature
neighbor and nearest held-out feature neighbor. The report saves their original
indices and distances. A much smaller training distance can motivate a closer
memorization investigation. Comparable distances do not prove non-memorization:
finite feature embeddings discard information, and only a subset is searched.
Do not call this a privacy guarantee or a complete membership-inference test.

## Uncertainty and repeated seeds

KID subset standard deviation and IS split standard deviation are not
confidence intervals across independently generated datasets. The repetition
script verifies matching checkpoint hashes, sample counts, scales and solvers,
then computes Student-t intervals across distinct generation seeds.

~~~bash
for seed in 123 124 125; do
  python -m diffusion_lab.image_evaluate \
    --checkpoint outputs/cifar10_edm_quality/checkpoint.pt --device cuda:0 \
    --samples 2048 --steps 18 --scale 1.5 --seed "$seed" \
    --output "outputs/cifar10_edm_repeat_$seed"
done
python -m scripts.summarize_repeats \
  --experiment outputs/cifar10_edm_repeat_123 outputs/cifar10_edm_repeat_124 \
    outputs/cifar10_edm_repeat_125 --output assets/reports/distribution_quality/seed_intervals.json
~~~

The reference is fixed across these runs; their interval measures generation
seed variability, not training or reference-dataset uncertainty. With three
repetitions, the interval itself is uncertain. Finite-sample FID is biased:
the 2,048-image repetition protocol must not be ranked against the 10,000-image
final evaluation as if only model quality changed.

## Measured example: correct labels do not establish diversity

The corrected iCT **5,000-update interim** one-call model achieves 98.5%
requested-class accuracy, yet its distribution recall is only 45.1%.
The completed one-call progressive student achieves 96.3% class accuracy
and 83.4% recall. Both use the same independent 128-dimensional digit assessor,
2,048 generated/held-out images, rounded pixels and $k=3$.
Four-call iCT at the same interim checkpoint raises recall to 62.3%.

These [measured diagnostics](../assets/reports/distribution_quality/interim_digits/comparison.json)
show that producing recognizable requested classes can coexist with restricted
within-class variation. That interim comparison does not establish final method
rankings. The corrected iCT run has now completed 40,000 updates: its one-call
distribution recall is 86.0%, with 96.1% class accuracy. Read the
[final diagnostics](../assets/reports/distribution_quality/extension_digits/mnist_ict_corrected1/metrics.json)
and [complete results](21_extension_results.md) alongside the interim report.
Nearest-image galleries show examples alongside the quantitative report.

## A useful report

Keep held-out split/counts and preprocessing beside FID/KID/IS. Add precision/
recall and density/coverage, per-class diagnostics, nearest-neighbor examples,
actual NFE, wall time, weight hashes and repeated seeds. For 2D data retain
MMD, sliced Wasserstein, exact assignment Wasserstein on its declared subset,
mode occupancy and target density when an analytic density exists. An unbiased
finite-sample MMD estimate can be slightly negative.

Image reconstruction PSNR and reconstruction FID establish the latent codec's
behavior separately from generation. Likelihood bounds and numerical ODE
densities are documented in [the likelihood chapter](16_likelihood_and_elbo.md).
Our official test splits were also used during development/validation; disclose
that before claiming a pristine external benchmark protocol.

References: [improved precision and recall](https://arxiv.org/abs/1904.06991),
[density and coverage](https://proceedings.mlr.press/v119/naeem20a.html).
Measured diagnostics appear in [extension results](21_extension_results.md).
