# Stronger evidence: failures, larger likelihood batches and training seeds

The new [study runner](../scripts/run_evidence_studies.py) stores commands, logs,
raw measurements and checkpoint-backed summaries. Its three modes are:

```bash
python -m scripts.run_evidence_studies --study consistency --device cuda:1 \
  --output outputs/consistency_refinement_study
python -m scripts.run_evidence_studies --study likelihood --device cuda:0 \
  --output outputs/expanded_likelihood_study
python -m scripts.run_evidence_studies --study training_seeds --device cuda:2 \
  --output outputs/training_seed_study
```

## CIFAR consistency refinement

The original student gives poorer two/four-call images than one-call images.
Keep that result visible. We evaluate its denoising MSE, range excursions and
teacher comparison across noise levels. We then sweep explicit refinement
levels with **unchanged weights**. Selection uses 1k generated/1k reference
images; final measurements use a new sampling seed and a disjoint 5k reference
subset. These small-selection FIDs are noisy and are not headline benchmarks.
Historical CIFAR development used test data, so the new subset split does not
retroactively establish a pristine test protocol.

The [sampler](../diffusion_lab/consistency.py) validates explicit descending
levels, initial sigma and the identity boundary. The default schedule is
preserved, so old commands still reproduce old behavior. The first sweep finds
substantial schedule sensitivity but does not resolve the weak refinement.
Around sigma 1, the student denoising MSE is about 0.165 versus the teacher's
0.030 on 256 examples, with about 29.6% of student output values outside [-1,1].
This is diagnostic evidence; it does not identify a unique training-related
cause. Denoising MSE also differs conceptually from consistency along teacher
trajectories. Further retraining/target ablations need their own experiment.

## Expanded likelihood diagnostics

The full MNIST DDPM negative ELBO now uses 1,024 distinct held-out examples,
processing all 1,000 terms for each example in batches. MNIST ODE diagnostics
target 1,024 examples and CIFAR ODE diagnostics target 512. Paired sensitivity
checks use the first 128 CIFAR examples with 100/200 Heun steps, one/four trace
probes and two noise seeds. Every batch records its offset in a deterministic
balanced reference order. The runner aggregates per-example results rather
than treating a mean-of-means as independent raw observations.

Student-t intervals describe held-out-example variation. They do not remove
integration bias, trace error, finite endpoint cutoffs or terminal prior error.
Known-class conditional density is not a class-marginal density. Negative ELBO
and dequantized ODE estimates are different quantities. The earlier 8–32-example
reports remain labeled as small diagnostics; the expanded study does not replace
them with an unexplained number.

Run `python -m scripts.summarize_likelihood_sensitivity` to export per-example
bpd values with batch provenance and matched numerical differences. On the
same first 128 CIFAR images, moving from 100 to 200 Heun steps changes mean
bpd from 3.3454 to 3.3621. The approximately +0.0167 difference demonstrates
remaining grid sensitivity; a larger image count does not make the ODE solve
exact. Changing the trace/noise seed also changes the dequantization draw.

## Independent training seeds

DDPM and flow each train three separate GMM models with seeds 42, 43 and 44,
6,000 updates and the same architecture/data/batch sizes. All six use matched
sampling seed 123, independent reference seed 999 and 4,096 evaluation examples.
DDIM uses 100 steps and flow Heun uses 50 steps: both make **100 network calls**,
verified by forward hooks. Sliced Wasserstein, MMD, finite empirical Wasserstein
and mode coverage are summarized by family. Earlier unmatched comparisons are
retained as diagnostics and are excluded from this summary.

The [image seed runner](../scripts/run_image_seed_study.py) also trains three
independent MNIST EDM and three flow models for 20,000 updates each, with the
same residual U-Net, data and effective batch size. Evaluation uses 2,048
examples, ordinary class conditioning and an independent digit classifier.
Its 128-dimensional feature distance is **not Inception FID**. At 16 Heun steps,
EDM makes 31 calls and flow makes 32; this small budget difference is disclosed.
Only completed families receive a summary, and individual checkpoint hashes
are retained. These are training-seed comparisons for **GMM and MNIST**, not
uncertainty estimates for headline CIFAR/face runs. Earlier sampling
seed intervals use one fixed checkpoint and retain their distinct scope.
Three training seeds are a useful first comparison, not a guarantee of a stable
ranking for every data set and budget.

The [notebook](../notebooks/05_likelihood_and_evaluation.ipynb) connects these
protocols to numerical checks. The [results chapter](26_learning_and_text_results.md)
contains actual completed measurements and training status.
