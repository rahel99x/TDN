# Protected formulation experiments: causal closure, rational propagation, tangent flow

These three CPU development units complement the advance campaign's attribution
and architecture comparisons. They do not consume fresh confirmation fields.
They investigate a different representation or equation formulation, rather than
adding parameters to the current rank-one conditioner. A useful negative result
is retained, including when a mathematical implementation is correct but its
scientific or computational hypothesis fails.

The source is `tdn/analysis/advance/exploration.py`. The advance protocol fixes the
seed namespace, independent-parent counts, prototype identity and bounded unit
budget before execution. Full exploratory units have a 900-second numerical
budget each; the entire campaign's Slurm job limits still apply. Numerical
development probes run on the CPU, with no GPU performance claims. No unit
silently increases its budget or accesses the confirmation cohort.

## Common mathematical task and evidence

All three probes use the finite, non-aliased Fourier Galerkin ODE for the periodic
one-dimensional logistic reaction–diffusion equation on `[0, 2π]`:

\[
 \dot a_k=(r-\kappa k^2)a_k-r\!\sum_{p+q=k,\ |p|,|q|\le K}a_pa_q,
 \qquad |k|\le K.
\]

The convolution is a linear convolution followed by projection, **not** a circular
FFT product. Reality follows from `a[-k] = conjugate(a[k])`. The generated fields
start inside `[0,1]`, with independent unresolved phases. This task is not the
2D benchmark, a continuum-convergence study, or a turbulence experiment.

DOP853 produces forward trajectories in complex coefficient coordinates.
Tighter tolerances and an independently implemented, fixed-step RK4 endpoint
cross-check audit the XH and XR test teachers. These checks estimate numerical
uncertainty; they are not rigorous error bounds or independent PDE solvers.
The default modes are `K=6` in smoke/development and `K=8` in full. Every raw row
records its parent, physical parameters and schedule. Repeated times, noise
conditions, methods and schedules are paired observations of that parent.

RMS error uses Parseval's identity, `sqrt(sum_k |error_k|²)`, with spatial averages
over `[0,2π]`. Maximum error and interval violations are evaluated on 257 points;
they are observed maxima, not certified supremum norms. Mean error is the
zero-mode error. Raw files retain finite-state checks, conjugacy residuals,
state minimum/maximum, reference uncertainty and the relevant acquisition,
training and inference costs.

Each experiment logs separate mathematical, scientific-gap and utility checks,
with GOOD/BAD/NA and reproducible 1–100 evidence-attainment scores. The scores
are neither probabilities nor a universal model ranking. Missing deployment
evidence stays NA even when a representation or mathematical check succeeds.

## XH: genuinely causal forward-history closure

Write `c = P_m a` for resolved modes, with `m=2`. The exact coarse equation is

\[
 \dot c = F_m(c)+\eta(a),\qquad
 \eta(a)=-r\{P_m(a*a)-(c*c)\}.
\]

An instantaneous map of `c` cannot generally recover `η`, because unresolved
phase information can affect resolved nonlinear products. XH asks whether an
observed previous coarse state supplies enough information to make a compact
closure useful during subsequent autonomous prediction.

For step `h`, the training target is deliberately an **endpoint increment**:

\[
 y_n=\frac{c(t_n+h)-\Psi_h(c(t_n))}{h},
\]

where `Ψ` is coarse RK4. It is not mislabeled as the exact instantaneous closure.
The causal memory feature is

\[
 \widehat\eta_n=\frac{c_n-\Psi_h(c_{n-1})}{h}.
\]

The following controls receive the same noisy current observation when noise is
enabled:

| Method | Prediction | Purpose |
|---|---|---|
| Coarse RK4 | `Ψ_h(c_n)` | Strong contained numerical baseline |
| State persistence | `c_n` | Zero-dynamics control |
| Instantaneous fit | `Ψ_h(c_n) + h f(c_n)` | Whether state alone suffices |
| Deterministic secant memory | `Ψ_h(c_n) + h η_hat_n` | Whether learning is necessary |
| Fitted compact memory | `Ψ_h(c_n) + h g(c_n, η_hat_n)` | Additional learned utility |

Both fitted models use standardized real Fourier coordinates and their squares;
the memory model adds `η_hat`. They are linear fits in those features, solved
with SVD ridge regularization. Three ridge settings are fitted on training
parents and selected only on validation parents. We retain all validation
scores, singular values, scales and fitted coefficients. The memory model has
80 fitted coefficients for the declared cutoff. No neural network is needed.

All history is produced by integrating **forward** from `t=0`. Startup
observations occur at `t=0.10-h` and `t=0.10`; there is no backward diffusion or
future-state leakage. Subsequent steps use previous predictions exclusively;
teacher trajectories supply evaluation targets but never refresh model state.
The forward startup solve is timed and charged to every complete trajectory.
Test cases include doubled step, tripled diffusion, independently noisy startup
observations at standard deviations `10^-4` and `10^-3`, and—in full—a combined
shift. Startup noise is identical across competing methods for a paired parent.

The advance full declaration uses 24 training parents, six validation parents,
24 test parents and 12 autonomous steps. Three training observation times per
parent are repeated measurements, not independent examples. The advance smoke
has four/two/four parents and three autonomous steps.

Advance criteria are a median paired error ratio at most `0.8` relative to coarse
RK4, and additionally at most `0.9` relative to deterministic memory for the
fitted-memory claim. These are bounded development criteria without confidence
claims. Deployment economics remain NA because a repeated-workload use case
and acquisition strategy are not established.

The expected failure mode is explicit in the mathematics: noise in two states is
divided by `h`, so a difference-based history estimate can amplify noise even
when clean-history closure works very well. The current pilot does not hide
that failure using oracle denoising, clipping or a fresh teacher at each step.

## XR: resolvent and rational temporal response

This branch splits diffusion alone into `L`, leaving logistic reaction in `N`:

\[
 a_{n+1}=E(hL)a_n+h\Phi(hL)N(a_n),\qquad E(z)=1+z\Phi(z).
\]

The controlled responses are:

| Response | `E(z)` | `Φ(z)` | Stiff negative-axis behavior |
|---|---|---|---|
| Exponential ETD1 | `exp(z)` | `expm1(z)/z` | Decays to zero |
| Backward-Euler resolvent | `1/(1-z)` | `1/(1-z)` | Decays to zero |
| Padé (1,1) | `(1+z/2)/(1-z/2)` | `1/(1-z/2)` | Tends to **−1** |
| Padé (0,2) | `1/(1-z+z²/2)` | `(1-z/2)/(1-z+z²/2)` | Decays to zero |

All have `E(0)=Φ(0)=1` and exactly preserve zero-time identity. The response
identity and small-step limit are checked numerically. Padé (1,1) is A-stable
but not L-stable; its weak stiff damping is an intentional kill test. A higher
order rational approximation of the exponential does **not** make this nonlinear
ETD1 formula a higher order integrator. The nonlinear convergence test verifies
first order, and the full experiment measures actual nonlinear Galerkin endpoint
error at one, two, four and eight steps with mild and strong diffusion.

Repeated-query tests grant **every** response the same opportunity to cache the
state's nonlinear product and all query response factors. They measure uncached
calls, cached query-only calls and complete rebuild-plus-query calls. A classical
DOP853 dense-output control includes its solve/setup cost and receives equivalent
query caching. Queries are `[1,4,16,64]` in the advance declaration. Quality is
audited against a declared relative RMS tolerance; an accurate scalar response
alone cannot substantiate a PDE benefit.

Randomized interleaving retains every CPU timing round. A first invocation is
recorded separately and is **not process-cold latency**. Short per-call endpoint
timings are descriptive; no matched-accuracy deployment speedup is declared from
them. Building an encoding for one state does not authorize reuse after the
state changes. These standard rational formulas are controls, not claims of
literature novelty.

## XT: variational flow and limits of moment augmentation

The exact tangent equation for perturbation `v` is

\[
 \dot v_k=(r-\kappa k^2)v_k-2r\!\sum_{p+q=k}a_pv_q.
\]

XT integrates the state and tangent together, then checks the result against a
three-epsilon sweep of centered finite differences of independently integrated
forward solutions. It separately checks the semigroup chain rule:

\[
 D\varphi_{h}(a)v=
 D\varphi_{h/2}(\varphi_{h/2}(a))D\varphi_{h/2}(a)v.
\]

The unresolved initial perturbation satisfies `P_m v(0)=0`. A nonzero resolved
tangent later therefore measures missing information, not instability created by
a poor learned fit. We retain full and resolved tangent norms across horizons,
finite-difference truncation/cancellation behavior and composition errors.

The mean identity is checked independently:

\[
 \dot\mu=r(\mu-\mu^2-\operatorname{Var}(u)),\qquad
 D\dot\mu[a]v=r\left(v_0-2\sum_k a_kv_{-k}\right).
\]

Logistic reaction does not conserve mean. A useful mean correction must respect
this law, rather than simply forcing zero mean change.

Finally, paired fields have identical resolved coefficients and identical total
unresolved energy but different unresolved phases. Their instantaneous mean
derivatives agree while their later resolved spatial states differ. This is a
bounded counterexample to the sufficiency of **scalar variance augmentation**.
It does not rule out richer moments, temporal memory, or an approximate closure
on a narrower distribution. The negative scientific check deliberately remains
BAD even when every identity is mathematically correct.

## Artifacts and interpretation

Each prototype writes `prototype_rows.json` and `exploration_provenance.json`.
The latter records the exact source SHA, raw artifact SHA, effective settings,
parent IDs and CPU device. Its raw data are respectively:

- `XH_history.json`: trajectories of coarse prediction errors, acquisition and
  inference costs, split identities, fit/validation data, coefficients and all
  noise/shift failures.
- `XR_resolvent.json`: response curves, nonlinear endpoint errors and finite
  checks, teacher cross-checks, exact timing rounds and equal caching controls.
- `XT_tangent.json`: derivative sweeps, semigroup and mean identities, hidden-to-
  resolved sensitivities and paired variance-counterexample measurements.

The atlas reads these files without converting observed ranges into confidence
intervals. Broad PDE superiority, competitive GPU speed and deployment economics
remain unestablished by these cheap experiments. Successful mathematical checks
permit interpretation of the measured outcomes; they do not make the research
hypothesis successful automatically.
