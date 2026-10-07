# Iterative diffusion Schrödinger bridge matching

[The bridge chapter](06_bridges.md) explains endpoint Sinkhorn coupling followed
by Brownian bridges. That direct finite-support lesson is now complemented by
an **alternating neural DSBM implementation** in
[dsbm.py](../diffusion_lab/dsbm.py).

The goal is a stochastic transport between prescribed marginals pi_0 and pi_1,
close in path-space relative entropy to a Brownian reference. Sinkhorn handles
a discretized endpoint problem. DSBM learns time-dependent drifts from sampled
endpoint pairs and repeatedly changes the coupling.

## One outer iteration

Start from a current endpoint coupling (Z,Y). Construct a conditional Brownian
bridge at time t:

\[
X_t=(1-t)Z+tY+\sqrt{\epsilon t(1-t)}\,\xi.
\]

The forward conditional drift is \((Y-X_t)/(1-t)\). Its conditional expectation
given X_t defines the Markovian projection. In our backward network, use the
**increasing reverse clock** r=1-t, swap endpoints, and regress
\((Z-X_{1-r})/(1-r)\). Its simulation starts from the target marginal.

Each outer iteration then:

1. Fits the backward projection on the current bridge mixture.
2. Starts fresh samples from the prescribed target distribution and simulates
   the backward SDE. Cache the resulting paired source/target endpoints.
3. Fits the forward projection using Brownian bridges from this new coupling.
4. Starts fresh samples from the prescribed source marginal and simulates the
   forward SDE. Cache the endpoints for the next iteration.

Fresh starting marginals correct accumulated endpoint bias on alternating sides.
Reconstructing bridges from the endpoint cache returns to the Brownian reciprocal
class. Just training one drift on a fixed Sinkhorn plan is not this iteration.

The regression uses positive time weights (1-t)^2 to reduce the exploding
conditional-drift variance near the target. This retains the same population
conditional-expectation minimizer at each interior time.

## IMF and IPF initialization

DSBM-IMF begins with an independent coupling of the two prescribed distributions.
DSBM-IPF begins with endpoints from the Brownian reference, Y=Z+sqrt(epsilon)xi.
The subsequent alternating bridge-projection machinery is the same. The latter
connects to iterative proportional fitting of path measures.

~~~bash
python -m diffusion_lab.dsbm --device cuda:0 --initialization imf \
  --iterations 4 --inner-steps 6000 --output outputs/dsbm_imf
python -m diffusion_lab.dsbm --device cuda:0 --initialization ipf \
  --iterations 4 --inner-steps 6000 --output outputs/dsbm_ipf
python -m diffusion_lab.dsbm --dataset wavy_rings --device cuda:0
~~~

Each direction saves metrics, samples, an animation, regression history and its
model/optimizer checkpoint. The outer checkpoint saves both drifts, the replay
coupling, iteration number and random state. Resume at an outer boundary:

~~~bash
python -m diffusion_lab.dsbm --output outputs/dsbm_imf --iterations 8 \
  --resume outputs/dsbm_imf/checkpoint.pt
~~~

## What the numerical experiment establishes

These are sample-based continuous-GMM/toy experiments with finite replay caches.
They approximate the ideal projection and fixed point; neither the neural fit
nor the Euler–Maruyama simulation is exact. Training avoids singular endpoints;
simulation extrapolates the learned drift at the start and stops at 1-time_eps.
This endpoint truncation introduces bias, which must be studied as the step size
and time_eps decrease. Metrics can fluctuate rather than improve monotonically.

Unlike the atomic Sinkhorn lesson, the prescribed smoothed GMM/toy marginals are
continuous. A finite replay cache is a Monte Carlo approximation, not a claim
that atomic terminal measures have finite KL against unrestricted Wiener measure.

The implementation currently demonstrates iterative DSBM on 2D distributions.
Image-scale DSBM introduces substantially more replay, compute and validation;
it is not represented by the image diffusion or face flow checkpoints.

Exercise: compare independent and Brownian initialization at equal compute.
Plot both endpoint errors after each half-iteration, rather than only the final
forward result. Then vary epsilon and inspect transport cost and path spread.

Source: [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852).
