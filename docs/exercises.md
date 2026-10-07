# Exercises and worked checks

Try each task before opening its answer. Use the matching notebook for numerical
checks; training or a GPU is not required for these questions.

1. For $X_t=\alpha X_0+\sigma\epsilon$, with independent $X_0$ of covariance
   $\Sigma$ and standard Gaussian $\epsilon$, derive the mean and covariance.

   <details><summary>Answer</summary>
   The mean is $\alpha\mathbb E[X_0]$ and covariance is
   $\alpha^2\Sigma+\sigma^2I$. Independence removes cross-covariance. In the
   first notebook, sample a non-unit-variance Gaussian to show that VP preserves
   unit variance only when the data covariance is the identity.
   </details>

2. Recover $x_0$ and $\epsilon$ from $x=\alpha x_0+\sigma\epsilon$ and
   $v=\alpha\epsilon-\sigma x_0$. Does the answer require VP normalization?

   <details><summary>Answer</summary>
   $x_0=(\alpha x-\sigma v)/(\alpha^2+\sigma^2)$ and
   $\epsilon=(\sigma x+\alpha v)/(\alpha^2+\sigma^2)$. The denominator becomes
   one for VP. `convert_prediction` retains the general denominator.
   </details>

3. Why does Gaussian score regression target $-\epsilon/\sigma$, even though
   this is not the marginal score for an individual training pair?

   <details><summary>Answer</summary>
   The target is the conditional corruption score. Least-squares regression
   learns its conditional expectation given the corrupted state, which is the
   marginal score under the usual interchange/integration assumptions.
   </details>

4. Starting from Fokker–Planck, verify that the PF-ODE drift
   $f-\tfrac12g^2s$ generates the same density equation when $ps=\nabla p$.

   <details><summary>Answer</summary>
   $-\nabla\cdot[p(f-\tfrac12g^2s)]
   =-\nabla\cdot(pf)+\tfrac12g^2\Delta p$ for spatially constant $g$.
   The same density evolution does not mean the same particle trajectories.
   </details>

5. How many calls do $N$ fixed Heun steps use? How does two-branch CFG change
   that? Why does an EDM sampler sometimes report $2N-1$ instead?

   <details><summary>Answer</summary>
   Ordinary Heun uses $2N$ field calls; separate null/conditional branches give
   $4N$ network calls. EDM's final zero-noise update uses Euler to avoid a
   singular final evaluation, giving $2N-1$ before CFG. Measure actual calls.
   </details>

6. Is three-sampling-seed variation for one checkpoint evidence of uncertainty
   across model training? Can low FID establish that a red bus prompt is followed?

   <details><summary>Answer</summary>
   No. Training uncertainty requires independently initialized and trained
   checkpoints. FID measures an aggregate feature-distribution discrepancy;
   prompt adherence requires paired text/image diagnostics and targeted tests.
   </details>

7. Why must all captions of one COCO image belong to the same split? What does
   using a pretrained denoiser change about claims of evaluation isolation?

   <details><summary>Answer</summary>
   Splitting captions would place the same image in train and evaluation. An
   image-level split isolates our fine-tuning data, but cannot prove that the
   base model's external pretraining never used an evaluation image.
   </details>

8. For a Gaussian flow $dx/dt=ax$, calculate its exact log-density change and
   use it to check the ODE likelihood sign.

   <details><summary>Answer</summary>
   In $D$ dimensions, divergence is $aD$ and density along a particle changes
   by $-aDt$. Backward density reconstruction adds the integrated divergence to
   the terminal log density. The likelihood notebook checks this numerically.
   </details>
