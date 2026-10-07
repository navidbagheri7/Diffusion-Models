# 4. Conditional generation and guidance

## Conditioning versus guidance

Unconditional generation samples from p(x). Conditional generation aims at
p(x∣y), where y can identify a mixture component or an image class.
A model can receive y during both training and generation. Guidance changes
its sampling field to adjust the strength of conditioning.

Our GMM component index supplies an unambiguous label. ImageNet folder indices
supply image classes. The flat face folders are used unconditionally.

## Classifier-free guidance (CFG)

Use one model with a class embedding and a learned null embedding ∅. During
training, replace a fraction of labels with ∅. The resulting model learns both
conditional and unconditional predictions.

Combine them at sampling:

$$h_{\rm CFG}(x,t,y)=h_\theta(x,t,\varnothing)
+w[h_\theta(x,t,y)-h_\theta(x,t,\varnothing)].$$

Our convention: w=0 is unconditional, w=1 is the ordinary conditional model,
and w>1 extrapolates the conditional direction. Other libraries sometimes
define their scale using a shifted convention; inspect the formula.

For diffusion, h can be epsilon, x0, score, or v. Their fixed-state conversions
are affine, so this mixing is algebraically consistent before nonlinear
operations such as clean-image clipping. For flows we mix velocities; this is a
practical conditional sampling heuristic rather than the same score derivation.

Code: [classifier_free and drop_labels](../diffusion_lab/guidance.py).
The final embedding index is the null token. Omit labels at evaluation for
unconditional generation.

## The score viewpoint

Bayes' rule gives

$$\nabla_x\log p_t(x\mid y)=\nabla_x\log p_t(x)+\nabla_x\log p_t(y\mid x).$$

Thus CFG score mixing is

$$s_{\rm CFG}=s_{\rm uncond}+w(s_{\rm cond}-s_{\rm uncond}).$$

At each fixed time, this is the gradient of
log[p_t(x∣y)^w p_t(x)^(1−w)], up to a normalization constant.
That local identity does not guarantee that a complete guided reverse process
samples an exactly corresponding powered endpoint density.

Large guidance may improve label adherence while reducing diversity, distorting
component variance, or creating artifacts. Measure these outcomes.

## Classifier guidance

Train a separate classifier p_φ(y∣x,t) on noised states, not only clean data.
It must recognize classes at the noise levels visited during sampling.
Add its input gradient to the unconditional diffusion score:

$$s_{\rm guided}(x,t)=s_\theta(x,t)+w\nabla_x\log p_\phi(y\mid x,t).$$

For epsilon prediction, the equivalent update is

$$\epsilon_{\rm guided}=\epsilon_\theta
-w\sigma_t\nabla_x\log p_\phi(y\mid x,t).$$

The minus sign follows from ε=−σs. This lab computes the score update directly
and converts to the clean-data/noise estimates required by DDPM or DDIM.

The classifier is trained with cross entropy on the same corruption path.
Its weights are frozen at sampling, but gradients with respect to x remain
necessary. classifier_gradient temporarily enables autograd even inside a
no_grad sampler, detaches its input, and differentiates the selected log-softmax.

There is no clean-image classifier shortcut: a classifier trained only at zero
noise may provide unreliable gradients at high noise.

## Train and run

~~~bash
python -m diffusion_lab.train experiment=gmm_guidance \
  process.prediction=x0 run_dir=outputs/gmm_guidance

python -m diffusion_lab.evaluate \
  --checkpoint outputs/gmm_guidance/checkpoint.pt \
  --output outputs/cfg_example --class-label 0 --scale 2

python -m diffusion_lab.evaluate \
  --checkpoint outputs/gmm_guidance/checkpoint.pt \
  --output outputs/classifier_example --class-label 0 \
  --guidance classifier --scale 1

python scripts/run_comparisons.py --device cpu
~~~

For classifier guidance, evaluation uses the diffusion model's null prediction,
then adds the classifier gradient. It does not accidentally stack CFG and
classifier guidance. Both mechanisms are separately implemented.

For a known GMM, an analytic posterior classifier is also available as a
reference. It uses the exact noised class-conditional densities and gives an
exact classifier gradient; the diffusion model remains learned:

~~~bash
python -m diffusion_lab.evaluate \
  --checkpoint outputs/gmm_guidance/checkpoint.pt \
  --output outputs/analytic_classifier_example --class-label 0 \
  --guidance classifier --classifier-kind analytic --scale 1
python -m diffusion_lab.diagnostics
~~~

Compare the learned and analytic classifier results before attributing sampling
distortion to the guidance identity. Classification accuracy and input-gradient
accuracy are different diagnostics.

## Read the visual examples

The [results gallery](08_results.md) compares CFG scales 0, 1, 2, and 4 and
classifier scales 1 and 2 for component zero. With w=0, particles should populate
the whole ring. With conditional guidance, particles should concentrate near
the requested component, preserving its small Gaussian spread.

Compare the width and offset of the generated cloud with the reference. If the
cloud narrows too much, stronger label adherence has reduced diversity.
Conditional Wasserstein/MMD metrics use that component as the reference.

**Exercise:** select a different class. Verify that both methods rotate the
generated cloud appropriately. Evaluate a noisy classifier across several
noise levels; near the terminal prior, accuracy should approach chance because
little class information remains.

Sources: [classifier guidance](https://arxiv.org/abs/2105.05233),
[classifier-free guidance](https://arxiv.org/abs/2207.12598).

**Next:** [numerical integration](05_numerical_methods.md).
