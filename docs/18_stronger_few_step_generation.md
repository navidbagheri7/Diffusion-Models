# Stronger few-step generation

Reducing solver steps and learning a faster generator are different operations.
DDIM and DPM-Solver integrate a trained diffusion field more efficiently.
Distillation changes the network so a coarse step approximates several teacher
steps. Consistency models learn a map that sends several noise levels on a
trajectory to the same clean endpoint.

## Progressive distillation

[progressive.py](../diffusion_lab/progressive.py) distills two deterministic
DDIM steps into one. Let $(a_h,s_h)$ and $(a_l,s_l)$ be corruption coefficients
at a high and low noise level. With clean prediction $D$, DDIM's destination is

$$x_l=\frac{s_l}{s_h}x_h+
\left(a_l-\frac{s_l}{s_h}a_h\right)D(x_h,h).$$

Run the frozen teacher from $h$ to an intermediate level and then to $l$,
obtaining $x_l^{teacher}$. Solve the one-step equation for the student's target:

$$D^*=\frac{x_l^{teacher}-(s_l/s_h)x_h}
{a_l-(s_l/s_h)a_h}.$$

The student predicts clean data with weight $\max(\mathrm{SNR}(h),1)$.
Its target is not necessarily inside [-1,1], so progressive students are not
clamped during their native updates. Rendering still clips pixels. The initial
Improved-DDPM teacher uses its existing clipped clean prediction; subsequent
teachers use the learned progressive map without this clamp.

The integer time grid includes a clean endpoint with $(a,s)=(1,0)$.
Every student's grid is the teacher grid with alternating points removed.
Training and sampling share that exact rounded grid. Substituting a different
few-step grid after training changes the learned map; the sampler rejects it.
An algebraic test confirms a student's solved target reproduces the teacher's
two-step endpoint to floating-point precision.

~~~bash
python scripts/run_progressive.py \
  --teacher outputs/mnist_improved_ddpm/checkpoint.pt \
  --device cuda:0 --stage-steps 5000 --samples 2048
# Or train a single 16-call stage from the 32-call teacher grid:
python -m diffusion_lab.image_train --config-name image_progressive \
  teacher=outputs/mnist_improved_ddpm/checkpoint.pt student_steps=16 \
  model.width=32 'model.channel_mult=[1,2,2]' 'model.attention_levels=[2]' \
  accumulation=1 device=cuda:0
~~~

The MNIST chain is 32 → 16 → 8 → 4 → 2 → 1, with five separately saved student
stages. Compatible teacher body weights initialize each student; a changed
epsilon-to-clean output head is initialized independently. The teacher always
uses frozen EMA weights. Each stage has its own optimizer and resume state.
The paper's original large benchmark budgets and scores are not claimed here.

## Improved consistency training (iCT)

The original CT/CD objectives remain available. The new `improved_consistency`
objective implements the main improvements from
[Song and Dhariwal](https://arxiv.org/abs/2310.14189):

- The lower-noise target is the detached **current online student**. It is not
  an EMA teacher. EMA weights can still be used for generation.
- Lognormal probability mass selects neighboring noise intervals.
- The number of noise intervals doubles during training, capped at 1,280.
- A Pseudo-Huber loss replaces MSE. With $\delta$ the flattened output
  difference, use $\sqrt{\|\delta\|_2^2+c^2}-c$, $c=0.00054\sqrt D$, weighted
  by $1/(\sigma_{high}-\sigma_{low})$.

The same Gaussian noise perturbs the clean example at both adjacent levels.
The boundary parameterization satisfies $f_\theta(x,\sigma_{min})=x$.
Our presets use zero dropout so the online target semantics are explicit;
this is an educational configurable U-Net, not the paper's exact architecture
and full training recipe. The stable implementation rationalizes the
Pseudo-Huber expression to avoid subtracting nearly equal numbers.

The first two runs were invalidated by an implementation bug: evaluating the
detached online target first in the same autocast context cached detached
convolution/linear weights on PyTorch 2.7.1. Those weights never updated.
Their identical noise-like results are failure diagnostics, not evidence that
iCT fails as a method or that the time embedding caused it. This behavior is
documented in [PyTorch's issue tracker](https://github.com/pytorch/pytorch/issues/160438).

The corrected implementation evaluates the gradient-tracked student first,
then its detached online target. A BF16 regression test checks that a zero
convolution receives a nonzero gradient and updates after an optimizer step.
`image_ict_corrected` completed the same 40k-step positional-embedding recipe
with EMA decay 0.999. Original checkpoints remain loadable and their failure
reports are retained explicitly.

~~~bash
python -m diffusion_lab.image_train --config-name image_ict_corrected device=cuda:0 \
  model.width=32 'model.channel_mult=[1,2,2]' 'model.attention_levels=[2]' \
  accumulation=1
python -m diffusion_lab.image_evaluate --checkpoint outputs/mnist_ict_corrected_quality/checkpoint.pt \
  --steps 1 --scale 1 --samples 2048 --device cuda:0 \
  --feature-classifier outputs/mnist_assessment_classifier/classifier.pt \
  --output outputs/mnist_ict_corrected1
~~~

## Measured cost and quality

Evaluate 1, 2 and 4 calls for consistency models; evaluate each progressive
stage only at its trained step count. The evaluator records actual network
forward calls, including both branches of CFG. A nominal 4-step solver with
two-stage Heun and CFG is not a 4-NFE generator. Classifier and VAE decoder
costs have separate meanings and are recorded separately.

The existing MNIST benchmark compares 1,000-call DDPM with 21-call DPM-Solver++
and original CT/CD. The expansion adds iCT, every progressive stage and
CIFAR consistency distillation from the completed 100k-step EDM teacher.
Use the independent digit assessor, not a guidance model grading itself.
All configured budgets completed. Corrected one-call iCT reaches digit feature
distance 43.60 and requested-class accuracy 96.1%; four-call iCT reaches 40.78
and 96.5% using 2,048 images and the same independent digit assessor.
CIFAR one-call distillation reaches FID 22.22, while default two/four-call
sampling worsens it to 69.57/83.41. More refinements need not improve a finite
learned consistency map; schedule and training coverage require investigation.
See [extension results](21_extension_results.md) for the measured reports;
a low-NFE implementation does not guarantee strong image quality.

Original references: [progressive distillation](https://arxiv.org/abs/2202.00512)
and [consistency models](https://arxiv.org/abs/2303.01469).
