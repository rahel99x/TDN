# Response: learned finite-time dynamics

This program implements the [next research handout](TDN_NEXT_RESEARCH_HANDOUT.md)
in its own namespace. Advance, Resolve and earlier measurements remain intact.
The user authorized the complete staged implementation and work-proportional
budgets on 11 October 2026. The handout's proposed three-hour pilot and eight-hour
development ceilings are superseded by that instruction. A time allowance is
neither a runtime prediction nor evidence that training needs the whole allowance.

The scientific objective is a useful **learned operator**, with separate claims
about representation, learning, generalization and complete solver cost. No
neural accuracy result by itself establishes a numerical-solver speedup. No local
FNO adaptation establishes reproduction or defeat of a published FNO benchmark.

## Target and precision contracts

- `fd_same_grid`: the declared periodic finite-difference diffusion symbol and
  nodal quadratic reaction, integrated on that same grid.
- `galerkin_same_grid`: the declared Fourier symbol and dealiased projected
  quadratic product, integrated on that same finite representation.
- `galerkin_projected_fine`: a separately specified fine initial field, evolved
  on independently refined fine grids and then projected. A coarse observation
  does not uniquely specify hidden initial modes; its lift or hidden-state
  distribution must be declared.

The coarse Galerkin RHS is not the exact projected-fine dynamics. Numerical
operators may share code while their reference and learning contracts remain
different. FD sampled norms and Galerkin physical interpolant norms remain
separate, with half mass per even-grid Nyquist axis in the latter.

The model input is an actual FP32 state. Its deployment teacher is generated for
that input under the declared lift. Deployment residuals subtract the actual
FP32 anchor, cast to FP64 **before** subtraction. Physical attribution residuals
use an FP64 anchor. Targets and differences remain FP64 until deliberately
normalized for learning. Training-only mean and centered scales include a
precision floor; easy examples are not divided by their own tiny residual.
Final FP32 addition can erase a sub-ULP correction and is measured separately.

## Mathematical structure

For `u'=Lu+ru-rB(u,u)`, a differentiable reference path `a(0)=u0` has residual
`rho=a'-f(a)`. Its exact error satisfies

\[
e'=Le+re-2rB(a,e)-rB(e,e)-\rho.
\]

The logistic background transform `u=c+q w`, with
`q=∂Phi_t(c0)/∂c0`, gives `w'=Lw-rqB(w,w)`. It does not say that `c` is the
actual mean or that `w` remains centered. Background derivatives can amplify;
forcing every latent response to decay would remove valid dynamics.

For actual mean `m` and centered state `v`,

\[
m'=r[m(1-m)-\langle v,v\rangle],\quad
v'=Lv+r(1-2m)v-rP_0B(v,v).
\]

Writing the reference as `a=ma+va` and centered error as `d=v-va` yields

\[
d'=Ld+r(1-2m)d+F,\qquad
F=-P_0\rho-2r(m-ma)va-rP_0[2B(va,d)+B(d,d)].
\]

The coupled model learns an approximation to this **full** finite-time source.
Keeping only `-P0 rho` is a separately labeled linear-response control. A
homogeneous-background response is approximate even when the error equation is
exact. Numerical checks do not become a proof or a certified residual bound.

For the Galerkin path, projected reaction uses a fixed substep count over the
declared differentiable path. A time-dependent `ceil(h*r)` loop is not a smooth
reference at its transitions. Differentiable physics parameters remain tensors;
teacher-only conversion to scalar solver parameters is explicit.

Reaction-only projected-fine closure can begin at `h²` for resolved initial
conditions and at `h` with hidden initial modes. It must not inherit the
diffusion-zero or cubic-time null of an ideal same-equation Strang defect.
Bounded mean clocks require a validated bounded-state assumption; finite
Galerkin projection does not generally preserve a nodal maximum principle.

## Architecture and attribution

Stage A is a shared endpoint trainability control: a declared physical anchor
plus separately normalized mean and centered learned corrections. If its
implementation matches residual FNO, aliases identify the same model; it is not
counted twice as an Ours/Theirs comparison.

The distinct Response architecture encodes phase-sensitive spatial features
once per current state, predicts a mean trajectory and zero-mean source fields,
and integrates those sources through the mean-dependent physical propagator.
The source ranks 2/4/8, temporal bases, mean clock/unrestricted mean, frozen
encoder, power-only features and linear-response limits are controlled variants.
Source products are formed before output compression. The time-query axis is
explicit and never treated as an extra spatial dimension.

The transparent quadrature decoder is retained beside a cached spectral decoder
that transforms source channels during encoding and uses one inverse transform
for their integrated response per query. It still pays for the anchor and mean
path. Fixed physical channels, direct cubic-spline source interpolation, learned
time rates, a separate numerical-remainder channel and FP64 final assembly are
declared controls. FP64 assembly retains the same deployed anchor/correction;
macrostep re-encoding returns to the model's declared parameter precision.

Selected coupled decoders receive a separate FP64 quadrature refinement audit
with frozen weights, independent source/mean refinements and joint 8/16/32-node
rules. Their differences from the finest completed rule are numerical
sensitivities, not error certificates. Deployment FP32-versus-FP64 assembly
differences are recorded separately. Diagnostic time is charged separately from
inference timing. A censored refinement remains unresolved.

The FD bounded mean-clock arm is tested both with and without its variance
consistency penalty. Inspection compares the clock-implied variance
`m(1-m)-m'/r` with the reconstructed field's physical variance and records both
path and actual decoded mean-ODE residuals. Zero reaction or invalid boundedness
gives NA. A penalized residual is not an enforced identity.

A cached encoding is tied to the state, geometry, physics, device, dtype and
model state. Every autonomous macrostep re-encodes its new state. Horizon-aware
encoders cannot claim horizon-independent reuse. Learnable source channels have
scale/rotation/permutation ambiguity; visual channel labels are not causal
physical identifications.

Only the final correction readout starts at zero. Multiplying several zero
factors would kill every gradient. The adapted original temporal model remains
a learned control; compact direct FNO and matched-anchor residual FNO receive
properly normalized targets and the same known physical information.
The direct FNO's default loss uses its training-only full-target scale. A
separate `shared_physical_objective` arm uses the residual-channel loss scales
shared by the corrected models. Its common-scale variant is compared with that
shared-objective arm, so changing mean/centered scaling is distinguishable from
changing the learning target. These are diagonal rescalings, not covariance
whitening.

## Data, trainability and selection

Parent specifications—not horizons, grids, translations or seeds—define the
independent unit. Deterministic namespaced digests generate seeds. Relative
phases, support, means, amplitudes, localization and independent reaction and
diffusion variation create actual shape diversity. Null and floor-limited cases
remain visible without dominating aggregate accuracy.

Matched phase twins are actual paired inputs. Separate descriptive panels test
held-out reaction/diffusion values, amplitude and mean changes, and grid-relative
high-frequency support. They preserve parent identities and do not alter the
primary estimand. Features include fixed physical frequency bands and normalized
extrema; power-only controls do not receive hidden phase-sensitive extrema.

The diverse tiny gate uses six resolvable regimes plus separate numerical-floor
and null controls. Fixed-feature linear fitting distinguishes feature limitations
from end-to-end optimization. The optimizer factorial separates learning rate,
clipping and objective multiplier. Each setting receives the same declared
opportunity; selected configurations then receive paired-seed replication.
Source and short-trajectory supervision is counted as additional data and
offline computation, with matched baseline opportunities. Tangent responses
are independently evaluated diagnostics; the primary loss does not receive
unreported tangent labels. Direct FNO uses a full-state training objective,
including variation between parent means, while residual families use their
mean/centered defect scales. A separately declared shared-physical-objective
arm isolates this optimization choice. Normalized losses from different
contracts must not be used as cross-family quality rankings.

Initialization remains selection eligible. Censored training is unresolved,
not proof of incapacity. A negative trainability result blocks the current development expansion while
leaving its evidence and protected exploration reportable. The current global
gate requires all declared family/target groups; revising that scope requires
an explicitly new protocol, not silently dropping a failed competitor.
The `.18/.24` time panel is extrapolation only while excluded from selection.

Response-weighted compression is compared to source-energy compression at
equal rank and through the same decoder. Small source reconstruction error does
not imply small endpoint error. Protected closure uses explicit shadow modes;
protected observation experiments distinguish two noisy startup measurements
from subsequent autonomous dynamics and include deterministic filtering and
zero-memory controls.

## Work-based budgets and recovery

Native profiling/calibration measures the new model's update and validation
costs. Planning records those measurements and their hashes, the estimator,
margin, requested updates and total allowance. A forecast is not an optimality
claim. CPU timings must not be used as measured RTX 4090 throughput.

Timing representatives are fixed from architecture/configuration, preferring
the primary arm, largest training cohort and first seed; measured accuracy or
speed never chooses them. Native warm timing jobs contain at most two learned
architectures and retain the same one/two/four/eight-step opportunities and
analytic controls. Fresh-process startup is measured once at one step per
representative. Dense-query/batching experiments have separate allocations.
Architecture-identical training variants retain all accuracy measurements, but
receive no fabricated timings or accuracy-qualified cost claims from an alias.

Every allocation remains at most 35 minutes, with science ending early enough
to save state and reports. A trial needing hours is partitioned into continuation
jobs bound to the same data, model, optimizer, RNG, sampler, source and software.
Both original and recovery attempts count. Completed trials do not train again
merely because a later continuation was scheduled. Budget changes are explicit
and frozen before the affected evaluation, never hidden in recovery.

The node's 110000 MiB Slurm host ceiling is respected. CPU preparation uses at
most eight cores and 48 GiB; GPU work requests one actual RTX 4090, at most four
CPUs and 32/48 GiB. Existing 18 GiB/75% soft VRAM and 90% device-use hard limits
remain. All files and caches stay in the checkout and its Python venv. There is
no pending-job cap and no cancellation of other projects.

## Evaluation and evidence

Retain mean/centered/RMS/maximum/spectral error, physical diagnostics,
uncertainty, failures, trainability, parameter movement, gradient/clipping
records, effective rank and source response. Warm latency, warm-process first
call, transfers and actual fresh-process startup are different measurements.
Unmeasured energy, money, p99 or deployment economics remain unavailable.

The main reference banks and physical-intervention teachers run in CPU stages.
A bounded auxiliary rollout/tangent panel may acquire additional CPU teachers
inside its evaluation allocation. Their measured time is separate from inference
latency and remains part of the allocation cost; accepted matching references
are reused. Representative post-hoc tolerance frontiers are not a deployable
schedule policy. Conditional amortization estimates do not establish complete
campaign break-even when allocation, attempted-work or coverage evidence is
missing.

Paired cluster analyses keep each parent's repeated queries and seeds together.
The primary candidate/base RMS point target is at most `.90`; the proposed
peak-error noninferiority margin is `1.05` and coverage margin `.02`. Their
registered uncertainty bounds matter: absence of significant harm is not proof
of noninferiority. Holm adjustment applies to the frozen primary comparisons.
RMS, tails, coverage and complete cost remain separate decisions.

Confirmation requires a separate immutable derivative freeze containing actual
checkpoint bytes and identities, normalization, target contracts, policies,
estimands and fresh parent-generation policy. It does not fit, select or revise
models. Changes informed by confirmation require a new cohort.

## Reporting

The atlas preserves raw tables and exact source pointers, distinguishes Ours,
Theirs and shared/analytic controls, and uses triangle markers for our learned
models. Dense training curves use continuous translucent observed low–high
bands with thin boundaries. Those bands are not confidence intervals. Missing
measurements stay missing; separate targets and grids are not collapsed into
one leaderboard. Images use 400 DPI and vector PDF companions.

Project-side Tower adapters register sealed root/subdirectory figures and
complete trial tables. Tower itself is unchanged. Compact review export is
lossless for retained evidence and explicitly lists omitted binary artifacts;
it is not a full checkpoint replay bundle.
