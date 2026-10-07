# Consistency models: one-step and few-step generation

Read [EDM and noise-level ODEs](09_advanced_diffusion.md) first. A diffusion
denoiser predicts clean data from a noisy state. A consistency function instead
learns to map points on the **same probability-flow ODE trajectory** to the same
small-noise endpoint. These are different training constraints.

For a trajectory x(sigma), the desired property is

\[
f_\theta(x(\sigma_1),\sigma_1)=f_\theta(x(\sigma_2),\sigma_2),\qquad
f_\theta(x,\sigma_{\min})=x.
\]

The toolkit enforces the second property structurally:

\[
f_\theta(x,\sigma)=c_{\rm skip}(\sigma)x+c_{\rm out}(\sigma)
F_\theta(c_{\rm in}(\sigma)x,\tfrac14\log\sigma),
\]

with
\(c_{\rm skip}=\sigma_{\rm data}^2/((\sigma-\sigma_{\min})^2+\sigma_{\rm data}^2)\),
\(c_{\rm out}=\sigma_{\rm data}(\sigma-\sigma_{\min})/\sqrt{\sigma^2+\sigma_{\rm data}^2}\),
and the EDM input normalization. At sigma_min, c_skip=1 and c_out=0, irrespective
of network weights. A test checks this boundary exactly.

Code: [consistency.py](../diffusion_lab/consistency.py).

## Consistency distillation (CD)

1. Freeze a trained EDM teacher and sample a high-noise state from data.
2. Use its probability-flow ODE to take one Heun step to the adjacent lower
   noise level. The teacher supplies the trajectory direction.
3. Evaluate the trainable student at the higher level and a detached EMA target
   at the lower level. Minimize their discrepancy.

Our lesson uses MSE discrepancy. It is an educational implementation of the
original consistency construction, not a reproduction of every perceptual-loss
or architecture choice used for paper benchmarks. The EMA target and inference
weights are the same network in this compact trainer; their decay is configurable.

~~~bash
# Copy the chosen teacher checkpoint first so later training cannot replace it.
mkdir -p outputs/mnist_cd
cp outputs/mnist_edm_quality/checkpoint.pt outputs/mnist_cd/teacher.pt
python -m diffusion_lab.image_train data=mnist algorithm=distillation \
  teacher=outputs/mnist_cd/teacher.pt model.width=32 \
  model.channel_mult='[1,2,2]' model.attention_levels='[2]' \
  model.dropout=0 sampling_steps=1 preview_scale=1 run_dir=outputs/mnist_cd
~~~

Teacher and student must agree on their architecture, channel count and classes.
The fixed teacher is retained beside the student. Distillation cannot make a
bad teacher into a proven high-quality one-step generator by itself.

## Consistency training (CT)

No pretrained teacher is needed. For adjacent sigma levels and one shared
Gaussian noise z, form

\[
x_{\rm high}=x_0+\sigma_{\rm high}z,\qquad
x_{\rm low}=x_0+\sigma_{\rm low}z.
\]

Regress the online prediction at x_high toward the detached EMA prediction at
x_low. These paired samples are conditional corruption neighbors, rather than
exact known marginal ODE trajectories. The connection to consistency is a
small-grid-spacing argument; coarse shared-noise straight lines should not be
called exact marginal probability-flow paths.

The trainer increases the number of noise levels with a square-root schedule.
Its consistency presets require dropout=0, removing mismatched stochastic
dropout masks between the online and target evaluations. Conditioning labels
(including the null token) are the same on both sides.

~~~bash
python -m diffusion_lab.image_train data=mnist algorithm=consistency \
  model.width=32 model.channel_mult='[1,2,2]' model.attention_levels='[2]' \
  model.dropout=0 sampling_steps=1 preview_scale=1 run_dir=outputs/mnist_ct
~~~

CT and CD can give a very small training discrepancy while still producing weak
images. Inspect generation and held-out metrics; loss alone is insufficient.
The later improved consistency training recipes are related further work and
are not silently claimed as part of this original-paper MSE lesson.

## Sampling

For one-step generation, draw z and evaluate
\(f_\theta(\sigma_{\max}z,\sigma_{\max})\). For another refinement, add noise of
standard deviation \(\sqrt{\tau^2-\sigma_{\min}^2}\) and evaluate at tau.
Refinement levels must be **strictly above** the identity boundary. Evaluating
at sigma_min would simply return the input and provide no refinement.

~~~bash
python -m diffusion_lab.image_evaluate \
  --checkpoint outputs/mnist_cd/checkpoint.pt --output outputs/mnist_cd_1 \
  --samples 2048 --steps 1 --scale 1 \
  --feature-classifier outputs/mnist_classifier/classifier.pt
python -m diffusion_lab.image_evaluate \
  --checkpoint outputs/mnist_cd/checkpoint.pt --output outputs/mnist_cd_4 \
  --samples 2048 --steps 4 --scale 1 \
  --feature-classifier outputs/mnist_classifier/classifier.pt
~~~

At scale 1, these use one and four student evaluations per image. Two-pass CFG
doubles those counts. The evaluator counts actual network calls and uses a
separate initial-noise RNG stream for meaningful solver comparisons.

Exercise: compare EDM with 35 evaluations against consistency with 1/2/4
evaluations, recording quality, class agreement, wall time and training compute.
Use the same classifier checkpoint for every MNIST feature metric. These metrics
are domain-specific and are explicitly **not ImageNet Inception FID or IS**.

Source: [Consistency Models](https://arxiv.org/abs/2303.01469).

## Improving refinement: separate the training and sampling choices

The optional `image_cd_refined` preset uses the same frozen EDM teacher and
Heun neighbor step, but weights its squared discrepancy by
\(w(\sigma)=\sigma^{-2}+\sigma_{\rm data}^{-2}\). This weighting is supported by
the [original implementation](https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py).
The default educational preset still uses uniform MSE. A separate target EMA
(`target_ema_decay=0.95`) supplies the lower-noise training target, while the
slower `ema_decay=0.9999` supplies generation weights. Both states are saved.
The generation EMA ramps with `min(configured_decay, (1+step)/(10+step))`;
its effective decay at 40k updates is approximately 0.999775. The separate
training-target decay remains 0.95 throughout.
Changing multiple choices together tests a recipe; it does not isolate which
choice caused an improvement. Pseudo-Huber discrepancy and inverse-gap
weighting are also exposed for controlled experiments; enabling them for CD
does not turn teacher distillation into the teacher-free iCT objective.

`--clip-consistency` clips **each** generated prediction into the pixel range
before adding refinement noise. This differs from clipping only the final
rendered image. It is appropriate to test for pixels normalized into [-1,1],
and is rejected for latent consistency sampling. Clipping changes the sampler;
report it alongside the sigma levels rather than hiding it in visualization.

```bash
python -m diffusion_lab.image_train --config-name=image_cd_refined
python -m scripts.compare_consistency_refinement \
  --baseline outputs/cifar10_consistency_distilled/checkpoint.pt \
  --candidate outputs/cifar10_cd_refined/checkpoint.pt \
  --output outputs/cifar_consistency_comparison --device cuda:0
```

The comparison freezes both input files, selects schedules/clipping on 1,000
reference examples, then uses 5,000 different reference examples and a new
generation seed for the final one/two/four-call scores. Class CFG is fixed at
one, and the evaluator measures actual denoiser calls. The historical CIFAR
test split has already been used for development: this is a repository
comparison rather than an untouched benchmark. The
[new training study](28_scratch_and_refinement.md) records its current status.
The [short CPU resume check](../assets/reports/diagnostics/cd_target_resume.json)
restores both EMA states exactly in its three-update comparison; it does not
promise deterministic CUDA training.
