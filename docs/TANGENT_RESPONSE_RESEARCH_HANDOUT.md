# Research handout: learned spatially varying finite-time response

**Status:** proposed research and implementation contract, 12 October 2026
(America/Los_Angeles). Writing this document does not implement the method,
establish novelty, or produce new numerical evidence. The proposed scripts and
interfaces below are deliverables for the main implementation agent.

**Objective:** test whether the spatially varying tangent response along a
numerical anchor supplies a useful correction direction that the existing
homogeneous/mean response misses, and whether that response can be approximated
by a small learned representation at a useful complete cost.

**Scope:** a new staged program beside Response and Response Diagnostic.
Preserve their source contracts, completed runs, calibration records, negative
gates, and reports. Implement the complete staged software, but make numerical
expansion conditional on the evidence gates below. A negative early result must
still produce a complete, usable research report.

## 0. Copyable assignment for the main agent

> Read AGENTS.md and this entire handout. Implement this program in the separate
> tangent_response namespace, including mathematical operators, independent
> oracle checks, response-action compression, a learned response pilot,
> reference/precision contracts, training, validation, cost measurement,
> recoverable native Fedora Slurm execution, Raven reporting, and portable
> exports. Start by verifying the spatially varying tangent mechanism with the
> same anchor and residual as the mean-response control. Do not assume that
> useful low rank exists. Implement gated campaign scripts that stop numerical
> expansion when the mechanism, compression, or learning result is negative or
> unresolved; still finish reports and retain all artifacts. Use CPU work for
> implementation verification and actual allocated CUDA work for native claims.
> Deliver runnable Bash entry points and an exact runbook; keep proposed commands
> labeled until they exist and pass their declared checks. Preserve all earlier
> experiments and any active work. Freeze protocols before measurement, retain
> failures, and record substantive deviations. Audit related literature before
> claiming novelty. Follow the active main conversation's authorization for
> committing, pushing, or submitting external/native work; this handout does not
> inherit permission from archived conversations.

The handout requests implementation of the framework, not an unconditional
large experiment. Do not stop at a plan, pseudocode, or untested launch script.
Deliver code, available verification evidence, and concrete native commands.

## 1. Reading map and repository integration

Read these existing files before implementation:

- `AGENTS.md` and `docs/FEDORA_SLURM.md`.
- `docs/RESPONSE_RESEARCH.md`, `docs/RESPONSE_DIAGNOSTIC_RESEARCH.md`, and
  `docs/RESPONSE_DIAGNOSTIC_RUNBOOK.md`.
- `docs/TDN_NEXT_RESEARCH_HANDOUT.md`, especially the error identity, precision,
  compression, mathematical-bound, and identifiability sections.
- `results/resolve-native-review/details/interactions-review.md` and
  `results/resolve-native-review/details/fit-review.md`.
- `tdn/analysis/response/contracts.py`, `numerics.py`, `trajectory.py`,
  `features.py`, `compression.py`, `models.py`, and `training.py`.
- Current workflow, sealing, recovery, and Raven contracts in
  `scripts/response_workflow.py`, `scripts/response_diagnostic_campaign.sh`,
  `scripts/response_diagnostic_raven.py`, and `tdn/analysis/response/diagnostic_metrics.py`.

Reuse verified numerical primitives and reporting infrastructure where their
contracts match. Use adapters or small general-purpose refactors rather than
copying an entire old program. Test shared changes against the affected old
paths. Do not change scientific defaults in those programs to accommodate this
one, or edit the Raven application.

Suggested new interfaces, all **to be implemented**:

```text
tdn/analysis/tangent_response/
  contracts.py       # targets, path, precision, independent sampling units
  protocol.py        # immutable stages, defaults, work, decisions
  tangent.py         # matrix-free A_a action and tangent evolution
  oracle.py          # independent response and forced-error controls
  compression.py    # response actions, bases, ranks, conditioning
  models.py          # state-conditioned response factors and decoder
  training.py        # staged objectives, variable projection, exact resume
  evaluation.py      # physical errors, action errors, composition, shifts
  performance.py     # full-call and matched-tolerance measurements
  report.py          # Raven, CSV/JSON/HTML, decisions and omissions
scripts/tangent_response.py
scripts/tangent_response_workflow.py
scripts/tangent_response_local.sh
scripts/fedora_tangent_response.sh
scripts/tangent_response_campaign.sh
scripts/tangent_response_archive.sh
docs/TANGENT_RESPONSE_RESEARCH.md
docs/TANGENT_RESPONSE_RUNBOOK.md
tests/test_tangent_response*.py
```

Use separate `runs/.fedora-tangent-response-latest.json` and
`runs/.tangent-response-campaigns/` identities. Add a separate Raven recipe
`tangent-response`; retain the existing `response-diagnostic` recipe.

## 2. Evidence, hypothesis, and novelty target

Established motivation, with its original scope retained:

1. In Resolve's designed interaction panel, the filtered leading cubic opposed
   the missing GL4 correction in 149/216 conditions. This is finite-step
   representation evidence, not proof that every learned model has this defect.
2. Approximately 72–75% of squared error for one Resolve fitted candidate was
   mean error. Recompute this decomposition for every new candidate.
3. The completed Response pilot had incompatible eligibility/headroom cases and
   substantial non-floor fitting failures. Recent conditioning and architecture
   changes have implementation verification; their benefit must be measured.
4. Source-energy and response-weighted compression already exist in this
   project. Adding either alone is not a new contribution.

**Primary hypothesis:** moving spatially varying linear feedback from an
unconstrained learned source into a compact response operator improves the
correction's direction and conditioning enough to justify its cost.

**Proposed contribution:** a state-conditioned approximation of finite-time
error response, with a quantified rank/error/cost tradeoff, explicit nonlinear
remainder, and held-out numerical evidence. The approximation of response
actions on relevant residuals is the primary object; a uniformly low-rank
approximation of the complete evolution operator is not assumed.

Before publication claims, produce a cited related-work matrix covering
exponential/Rosenbrock and defect-correction methods, variational/tangent
equations, Krylov and low-rank evolution approximations, POD/DEIM and
goal-oriented reduction, neural reduced operators, and learned preconditioners
or propagators. Compare assumptions, information, causal structure, error
analysis, and complete cost. The error identity itself is established
mathematics. Neither attaching an encoder nor combining known ingredients
automatically establishes novelty. If a close predecessor exists, identify a
specific distinction or report replication rather than rename it.

## 3. Equation, target, and precision contract

Initial scope is the autonomous finite equation

\[
u'=f(u)=Lu+ru-rB(u,u),\quad r\geq0,\quad L=\kappa\Delta,\quad\kappa\geq0.
\]

Implement `fd_same_grid` first and `galerkin_same_grid` as a separately reported
extension. In FD, `B(a,b)` is nodal multiplication and `L` uses the declared
periodic difference symbol. In Galerkin, `B(a,b)=P_N(ab)` is the verified
dealiased projected product with the existing Fourier/Nyquist conventions.
Galerkin products are not generally associative.

Exclude `galerkin_projected_fine` from the initial error identity, training, and
rollout claims. Return an explicit unsupported-target reason. A future closure
extension needs an explicit hidden-state/lift contract and projected forcing;
coarse tangent dynamics alone do not solve that information problem.

Every record identifies geometry, domain length, grid, target, diffusion symbol,
product/projection version, path, substeps, time nodes, dtype, and source hash.
Parent fields are independent sampling units. Horizons, phases, grids, physics
interventions, model seeds, and probes attached to a parent are paired data.

Generate teachers for the **actual FP32 input**, promoted to FP64. Keep teachers,
physical residual labels, and subtraction in FP64. Disable TF32 under the native
model precision contract. Report FD nodal norms and Galerkin physical
interpolant norms correctly, including Nyquist mass weights. Label sampled
maximum errors as sampled maxima, not continuous supremum certificates.

### 3.1 Smooth mathematical path versus actual deployment

Let `a(t)` be the declared smooth numerical anchor path, with `a(0)=u0`.
Evaluate the mathematical path and its derivative in FP64 with fixed work.
Reuse `anchor_path` only after independently checking its derivative and
residual. Do not differentiate a rounded FP32 execution as though it were a
smooth mathematical trajectory. A horizon-dependent changing subdivision
count is not a smooth path at its transitions.

Store these separate vectors:

```text
physical_defect   = teacher64 - smooth_anchor64
arithmetic_bridge = smooth_anchor64 - deployed_anchor32.to(float64)
deployment_defect = teacher64 - deployed_anchor32.to(float64)
deployment_defect = physical_defect + arithmetic_bridge
```

The physical mechanism stages fit/compare `physical_defect`. Native deployment
stages separately measure physical prediction, arithmetic bridge, final FP32
addition, and final error. If the production model learns a numerical remainder,
name and ablate it; do not attribute that gain to the tangent mechanism.
If an implementation computes the FP64 bridge at inference, charge both anchor
evaluations and label that precision policy. No FP64 path, residual derivative,
teacher, probe construction, or interpolation may become a free inference
input through a dataset or cache.

## 4. Exact identities and meaningful controls

For `rho=a'-f(a)` and `e=u-a`, the same-equation error satisfies

\[
e'=A_a(t)e-rB(e,e)-\rho,\qquad e(0)=0,
\]

\[
A_a(t)z=Lz+rz-2rB(a(t),z).
\]

`A_a(t)` acts on the full field, including its mean. Do not center its output:
spatially varying reaction couples centered perturbations to the mean.
Implement this action matrix-free using the declared product.

Let `U_a(t,s)` solve

\[
\partial_t U_a(t,s)=A_a(t)U_a(t,s),\qquad U_a(s,s)=I.
\]

The exact integral identity is

\[
e(t)=-\int_0^t U_a(t,s)[\rho(s)+rB(e(s),e(s))]ds.
\]

The first linearized correction is

\[
e_U(t)=-\int_0^t U_a(t,s)\rho(s)ds.
\]

Define the **matched anchor-mean control**, with `mu_a(t)=mean(a(t))`, by

\[
G_a(t,s)=e^{(t-s)L}
 \exp\left(r\int_s^t[1-2\mu_a(q)]dq\right),
\]

\[
e_G(t)=-\int_0^t G_a(t,s)\rho(s)ds.
\]

Use the same anchor, residual, input, source quadrature opportunities, and
precision in `e_U` and `e_G`. A logistic-background response is a secondary
control, not a substitute for this matched anchor-mean comparison.

With `v_a=a-mu_a`, the omitted spatial feedback is

\[
H_a(t)z=-2rB(v_a(t),z),\qquad
U_a=G_a+\int_s^t G_a(t,q)H_a(q)U_a(q,s)dq.
\]

This time-ordered integral motivates the approximation; multiplication by a
single global positive gain does not represent it in general. Reaction can
amplify perturbations, so do not impose universal contraction or decay.

Optional one/two nonlinear correction iterations use

\[
e^{(0)}=0,\qquad
e^{(k+1)}(t)=-\int_0^t \widetilde U_a(t,s)
 [\rho(s)+rB(e^{(k)}(s),e^{(k)}(s))]ds.
\]

These require intermediate trajectory values and causal quadrature. Implement
them as separately costed controls. Do not claim convergence at large steps
without assumptions and evidence. A nonlinear full-error-equation integration
is an expensive oracle control, not a fast learned inference method.

## 5. Stage A: independently verify the mechanism

Implement these endpoint controls before training a network:

1. Smooth anchor alone.
2. Matched anchor-mean forced response `a+e_G`.
3. Spatial tangent forced response `a+e_U`.
4. One/two nonlinear correction iterations where numerically resolved.
5. Independently refined integration of the full error equation.
6. Strong existing analytic/classical controls, including the improved GL4
   implementation and an accuracy-qualified numerical solver.

Compute forced responses without dense propagators by solving
`z'=A_a(t)z-rho(t)`, `z(0)=0`, and its matched mean counterpart. For isolated
`U_a(t,s)z` probes, integrate the homogeneous tangent equation from `s`.
Dense matrix assembly is permitted only for explicitly small verification
problems, at most 64 spatial degrees of freedom. Production paths must reject
or avoid dense grid-squared allocation.

Use an independent NumPy/SciPy finite-equation implementation on small cases to
check the Torch operator, product, time derivative, and integration. Merely
calling the same product through two wrappers is not independence. Independently
refine tangent integration and anchor/source quadrature; teacher agreement does
not establish tangent convergence. Save all work and uncertainties separately.

Perform designed interventions on fixed fields: remove spatial variation from
`a` in the tangent operator; vary reaction and diffusion independently; change
relative spatial phase while retaining modal magnitudes. Track correction
direction, amplitude, mean/centered errors, and uncertainty. Interpret phase
interventions as designed probes, not additional independent fields.

**Entry result:** useful evidence is that spatial feedback improves the
same-equation correction on a prospectively declared nontrivial bank. If the
nonlinear remainder dominates, establish that with the nonlinear controls rather
than declare the tangent response a complete solution.

## 6. Data and reference defaults

Freeze the initial protocol before generating labels. Defaults below are
development choices, not an application-wide success standard.

| Profile | Independent parents | Geometry and purpose |
| --- | --- | --- |
| `cpu-smoke` | 2 train, 2 validation, 2 diagnostic | Reduced odd/even grids, all software paths, no scientific promotion |
| `mechanism` | 16 train, 8 validation, 8 diagnostic | Initially 16x16 FD; selected paired 32x32 refinements; designed physics interventions |
| `compression` | Same frozen mechanism parents | Same states, new predeclared forcing/probe descendants |
| `fit-pilot` | 8 train, 4 validation, 8 diagnostic | Selected by declared reference/headroom criteria, fresh parent IDs |
| `development` | 64 train, 16 validation, 24 diagnostic | Fresh parents, paired 16x16/32x32 and held-out 64x64 panel |

Use digest-based namespaced seeds distinct from all earlier programs. Parent
shape hashes, rather than seed labels alone, must guard split leakage and phase
twins. If old data are used for debugging, label them reused development data.

The CPU implementation smoke uses two optimizer updates, width 4, and rank 2
on reduced grids. It exercises each bounded implementation path and records
its scientific decisions as implementation-only; it cannot promote a native
stage or calibrate native work. Test genuine native gate stops separately with
controlled workflow fixtures. Reduced science that fails a gate must never be
reported as passing merely because the implementation smoke completed.

Declare low/mid/high means, varying amplitude, localized/mixed fields,
overlapping interactions, and grid-relative high-frequency stress. Include
constant states, zero time, zero reaction, and zero diffusion as separate null
controls. Enforce the bounded input assumption only in profiles relying on it;
do not infer Galerkin trajectory positivity from bounded initial nodes.

Initial horizons: train `[0.01, 0.03, 0.06, 0.12]`; validation
`[0.02, 0.045, 0.09]`; diagnostic `[0.015, 0.075, 0.15]`. Label `0.15`
extrapolation. Mechanism interventions target dimensionless active diffusion
numbers `[0.1, 1, 5]` and reaction numbers `[0.1, 1, 3]` at `h=0.12`.
Record the active-frequency definition and actual coefficients. Coefficients
matched separately at each grid are not a fixed-physics convergence test.

Reference acceptance must allocate one complete temporal/cross-solver budget,
not grant each component the entire budget. Use absolute RMS/maximum tolerance
`1e-10` as an initial FP64 teacher target and additionally resolve the smallest
effect being claimed. If attainable integration/path precision cannot meet the
requirements within declared caps, preserve `UNRESOLVED` rather than lower the
threshold after observing a result. Refinement differences are estimates.

Predeclare a finite candidate list for each fit regime. Screening uses reference
acceptance, signal, and deployment headroom; it must never use trained model
loss or speed. For mechanism comparisons require anchor physical-defect RMS at
least ten times the combined reference/path uncertainty estimate. Retain
requested denominators, every rejected candidate, and precision-null panels.
Fit-pilot selection also requires enough output resolution for its declared
accuracy target. Screened-bank results are conditional on that bank.

## 7. Stage B: test response-action compression before learning

The complete `U_a-G_a` need not be low rank. Test two separately named objects:

1. A reusable operator correction acting on independent source directions.
2. A task-conditioned approximation restricted to the declared residual family.

Do not call a rank-one representation of one endpoint vector evidence that an
operator is rank one. Do not build a dense tangent matrix at production grids.

Collect matrix-free actions for ranks `J in {2,4,8}`. Include actual anchor
residuals, independent Fourier-band probes, and signed random probes with
prospectively declared seeds. Use source times `s<h`, multiple endpoints, and
held-out source directions on the same states plus held-out parent states.
Construct probe distributions before action measurement. Probe count is work,
not independent population sample count.

Compare equal-rank controls:

- Fixed shared bases fitted on training actions only.
- Source-energy POD and response-action/endpoint-weighted bases where applicable.
- State-conditioned factors, initially oracle-fitted per state to measure
  representation headroom; per-state oracle fits are not deployable methods.
- Frozen/random encoder readouts and the eventual learned encoder.

Fit coefficients with stable FP64 least squares or variable projection where
the decoder is affine conditional on the factors. Report singular values,
retained rank, cutoff/ridge, norms, Gram matrices, and sensitivity in actual
deployment precision. Conditional linear solves do not certify a global optimum
of the nonlinear encoder or mean path.

Measure full-field action error, propagated physical residual error, mean and
centered error, and sampled maximum. A response-null source can have large
source norm and tiny endpoint effect; report absolute errors alongside relative
ones and use a predeclared nontrivial-action criterion. Quantify which source
directions were never tested.

## 8. Stage C: minimal learned response architecture

A transparent first parameterization is

\[
\widetilde U_\theta(t,s)z=G_a(t,s)z+
 \sum_{j=1}^{J} \alpha_j(u_0,x)\,
 c_j(t,s;u_0,L,r)\,
 \langle\beta_j(u_0,\cdot),z\rangle_{\rm phys}.
\]

Here `alpha_j` and `beta_j` are full spatial fields predicted by a periodic,
phase-aware encoder, `c_j` are signed temporal coefficients, and the inner
product is the target's physical one. The factors depend on physics and
geometry as well as `u0`; those arguments are abbreviated above. Mean content
is permitted in both factors. A purely diagonal Fourier multiplier cannot
generally represent this spatial mode mixing.

This is a hypothesis, not a guaranteed sufficient factorization. It separates
spatial directions from causal time dependence and may require more rank or
time channels than is useful. Record source rank, temporal-channel count,
width, parameters, and effective numerical rank separately.

Required structural properties:

- `c_j(s,s)=0`, so `U_theta(s,s)=I`; reject reversed time intervals.
- Exact zero correction to `G_a` when reaction is zero or the anchor is spatially
  constant. Zero diffusion alone is not a null of `U_a-G_a` on varying fields.
- Periodic translation covariance and real/conjugate-symmetric outputs.
- Correct declared physical wave numbers and product/projection conventions.
- No forced decay or positive-only coefficients; finite intervals must remain
  finite without concealing overflow by clipping.
- Nonzero factor/encoder initialization with only the final corrective
  coefficient readout zero. Several multiplied zero factors would kill learning.
- Measured conditioning and frozen train-only normalization of the complete
  decoded gain. Never divide labels by a tiny per-example gate or time step.

Initialize the response at `G_a`. With an explicit anchor residual, the initial
endpoint is consequently `a+e_G`, not the bare anchor; record both baselines.
Verify head gradients on the first update and factor/trunk gradients once the
corrective readout is active. Parameter movement alone is not fitting evidence.

Use rank 4, spatial width 16, depth 2 as the initial native learned prototype.
Rank 2 and 8 are targeted compression controls. Add width only after direct
coefficient fits distinguish factor capacity from conditioning/optimization.
Do not launch a width/rank/basis/optimizer Cartesian product.

### 8.1 Inference and cache contract

First implement a transparent decoder with explicit anchor path, residual, and
quadrature. Charge their complete inference work. If those operations erase
the possible cost advantage, investigate a separately named compact encoder
that predicts residual time channels and required path summaries from the
initial state and known physics. Train these summaries on train-only numerical
labels; assess each approximation separately. They are not teacher inputs at
deployment.

Cached encodings are valid only for identical current state, physics, geometry,
checkpoint, device, and dtype. Keep requested horizons outside the reusable
encoder or report horizon-dependent re-encoding. Every autonomous macrostep
re-encodes the new state. Charge cache creation, invalidation, and memory.

Null corrections and response identities are distinct. For constant `a`,
`U_a=G_a`; for zero reaction, `U_a` is heat evolution. Neither means the response
operator itself is zero. Exact-subflow FD splitting has physical-defect nulls
at zero diffusion, zero reaction, constant input, and zero time. Approximate
Galerkin reaction subflows and deployment rounding need their separately
measured numerical remainder; never mask them as if the ideal null applied.

## 9. Training, validation, and attribution

Train in stages: response-action fitting with explicit residuals; then endpoint
fitting; then, only if necessary and identified, compact residual/path prediction
and one nonlinear correction iteration. Preserve checkpoints and results from
each stage. Report additional source/action/path labels, acquisition cost,
storage, and optimizer work. Endpoint-only controls must not be described as
receiving identical supervision to an action-supervised model.

The initial fit pilot has six implementations: anchor, matched mean response,
oracle tangent response, fixed shared response basis, learned response rank 4,
and a properly conditioned matched-anchor residual FNO. Only learned arms use
the declared optimizer budget. Compare direct response coefficient fitting with
the learned response head on the tiny bank. Matched supervised controls and
nonlinear variants are conditional follow-ups, not compulsory extra sweeps.

Use full-parent deterministic updates with equal total parent weight. Select
checkpoints by validation-only physical loss; retain initialization, final
iterate, and best training-fit checkpoint separately. A training-fit checkpoint
is a diagnostic and cannot become the deployment selection.

Default native fit-pilot work: 300 committed updates; development work: 1,000,
three paired model seeds. Calibration: 64 updates using the actual stage's
query/parent layout, gradient horizon, auxiliary labels, rank, dtype, and
validation workload. Give each declared arm its opportunity; failed execution
is censored/unresolved rather than proof of architecture incapacity.

Start from the tested full-parent optimizer recipe (learning rate `0.001`,
Adam epsilon `1e-12`, five warmup/head updates, cosine decay to 5% of the initial
rate) while logging its exact behavior. For response factors, conditional
readout solves may replace head warmup if declared. Physical-loss gradients can
be small; measure before deciding to multiply objectives. Any changed optimizer
or objective is a declared treatment. Use at most one targeted follow-up when
an explicit failed-gradient or conditioning diagnostic justifies it.

Publish endpoint physical RMS and maximum, mean and centered errors, response
action error, residual reconstruction error, nonlinear-remainder size, learning
curves, unclipped gradient norms, clipping frequency, parameter movement, raw
and assembled gains, and FP32/FP64 assembly differences. Record every optimizer
commit privately for exact resume; bounded published curves retain full-history
summaries. Log identity plans once rather than repeating them in every update.

Use an explicit physical endpoint objective, plus declared action/source
auxiliaries with train-frozen scales. Report each component and its gradient
norm. If maximum error is a success criterion, declare a smooth peak penalty
or validation tie-break before fitting; do not substitute pooled MSE for that
criterion. Compare physical errors across arms, not differently normalized
losses. A global 100-fold improvement-from-initialization rule is not used.

## 10. Decisions and automatic campaign gates

Serialize each decision, evidence pointers, denominator, thresholds, and
uncertainties before downstream submission. Gate thresholds below are
prospective defaults for this new development program; they do not revise any
old gate. Record justified changes before measurements under a new protocol
hash. An inapplicable or precision-limited comparison is not a passing result.

| Stage | Default promotion criterion | Negative or unresolved result |
| --- | --- | --- |
| A: mechanism | On the declared nontrivial FD diagnostic bank, parent-weighted arithmetic mean physical RMS ratio `E_G/E_U >= 1.5`, with uncertainty-separated improvement for at least 75% of eligible parents and no greater than 2x sampled-maximum regression on an eligible endpoint | Finish mechanism/uncertainty report; do not submit native compression/learning expansion automatically |
| B: compression | At rank at most 8, held-out nontrivial response actions have relative physical error at most 0.1 for at least 90% of declared probes and worst at most 0.3; compressed endpoints retain at least 80% of the aggregate oracle improvement | Report operator/task-specific limitations, conditioning, and rank/cost curves; stop native learning expansion |
| C1: fitting | On the screened tiny bank, the best training-fit checkpoint reconstructs the oracle physical correction with relative RMS at most 0.05 on at least 90% of declared nontrivial training queries; no nonfinite outputs; all seeds receive their declared work | Localize factor/optimizer/arithmetic failure; do not launch broader development |
| C2: held-out learning | Validation-selected model retains at least 80% of aggregate oracle gain against `G_a` on diagnostic parents; disclose all action errors, failures, and tails | Preserve learning result; no automatic utility/confirmation expansion |
| D: utility | A preselected deployable policy has lower complete measured cost than the strongest qualifying control at the declared physical RMS/max tolerance, with requested-cohort coverage reported | Accuracy/representation contribution may remain valid; do not claim solver speedup |

Here `E_G`, `E_U`, and `E_L` use identical physical norms and parent weighting.
Retained gain is `(E_G-E_L)/(E_G-E_U)`, evaluated only where aggregate oracle
headroom is prospectively eligible; retain all requested coverage separately.
Teacher/path/tangent uncertainty must be below 10% of the reported improvement
for an uncertainty-separated comparison. This is an empirical estimate, not a
certified or probabilistic bound. Do not choose a favorable horizon or regime
after inspecting the diagnostic panel to rescue a pooled gate.

For utility use initial physical RMS/max tolerance `2e-5`, plus preregistered
`1e-4` and `1e-5` work-precision panels. Keep the tolerance rationale explicit.
Confirmatory success is a later frozen protocol with new uninspected parents;
the default campaign does not perform confirmation or treat these development
thresholds as a universal claim.

The full requested denominator, eligible denominator, teacher failures,
precision-null cases, missing measurements, initialization selections, and
censored trials accompany every decision. A failed FNO control challenges that
control's adequacy; it does not automatically invalidate a measured tangent
mechanism. Separate baseline fairness from candidate success.

## 11. Mathematical analysis deliverable

Derive and test a bound that separates response approximation from nonlinear
error feedback. For the first correction, the algebraic decomposition includes

\[
e(t)-e_\theta(t)=
 -\int_0^t [U_a(t,s)-\widetilde U_\theta(t,s)]\rho(s)ds
 -r\int_0^t U_a(t,s)B(e(s),e(s))ds,
\]

when `e_theta=-integral U_theta rho` and the residual is exact for the chosen
path. Add separate terms for residual/path approximation, numerical quadrature,
and deployment arithmetic where present. These terms can cancel; retain signed
cross terms in squared-error decompositions, alongside conservative norm sums.

Specify the norm, finite-dimensional product constant, admissible state bounds,
growth estimate, and time interval for any inequality. FD and Galerkin require
their own assumptions. Resolution-dependent constants are not grid-uniform
guarantees. Residual samples and numerical refinement do not yield a certificate
without controlling errors between samples and floating-point arithmetic.

Analyze the conditioning and scale/rotation/permutation freedom of learned
factors. Train-only whitening/normalization can stabilize coordinates but is
not a new accuracy mechanism. A conditional variable-projection solve is linear
only when all quantities determining the response and path are held fixed.

The theoretical deliverable may be a useful conditional approximation result
and empirically supported rank behavior. Do not promise a convergence theorem,
rigorous rank bound, or universal stability result that has not been derived.

## 12. Independent tests and numerical validation

Implement meaningful checks for the following contracts:

- JVP/finite-difference agreement of `f'(a)z` with `A_a z`; independently
  assembled FD and projected Galerkin products, including odd/even grids.
- Both sides of the full error identity and its mean/centered decomposition.
- Constant-background analytic response; `U(s,s)=I`; zero reaction; positive
  and negative signed probes; time and physics derivatives in FP64 tiny cases.
- Tangent composition `U(t,s)U(s,q)=U(t,q)` **along the same anchor path**.
  Nonlinear endpoint composition is a separate test with re-encoding.
- Independent convergence of forced response, source/path quadrature, and
  nonlinear controls. Resolve near-zero and stiff cases separately.
- Translation covariance, conjugate reality, physical norm/Nyquist conventions,
  and full-field mean coupling. A centered-only implementation must fail a
  designed mean-coupling counterexample.
- Stable conditional linear fitting, rank-deficient directions, zero action,
  declared cutoff/ridge, frozen training normalization, and FP32 deployment.
- Native head/trunk gradients, parameter movement, checkpoint reload, cached
  versus complete-call parity, and complete dtype/device identity checks.
- Exact continuation of model, optimizer, scheduler, RNG, query order, and
  checkpoint selection across allocation boundaries; all-attempt accounting.
- Planner/controller bindings, unavailable/unsupported target handling, gated
  stops, reports after failures, archive terminal-job checks, and Raven runtime.
- No future teacher/action labels in inference and no hidden dense operator
  allocation at production sizes. Cost instrumentation covers path/residual and
  nonlinear iteration work.

CPU tests establish CPU behavior. Enumerate the required CUDA cases for the
new namespace, execute them on the actual allocated device, and bind the
readiness receipt to source/software/profile. Existing CUDA primitives reused
by the new method retain their applicable checks. Do not equate skipped or
mocked CUDA cases with readiness, or blindly inherit an old case count.

## 13. Generalization and rollout validation

Keep three distinct evaluation panels:

1. Held-out parents within the declared development distribution.
2. Designed shifts in reaction, diffusion, mean, amplitude, spatial phase,
   frequency support, horizon, and paired physical resolution.
3. Same-grid autonomous rollouts, initially 2, 4, and 8 macrosteps, including
   different subdivisions of the same final time.

For rollouts compare endpoints and intermediate teachers, measure accumulated
physical errors, mean/variance balance, finite outputs, and nonlinear endpoint
composition differences. Record an independently checked tangent/Jacobian
sensitivity panel. Accuracy at one endpoint, a small composition residual, and
a numerically stable trajectory are different observations.

For FD under valid boundedness assumptions, record interval violations and the
appropriate dissipative diagnostics. Galerkin does not inherit the nodal maximum
principle. Never hide instability by clipping to the teacher's range.

## 14. Complete costs and work-based allocation

Measure on actual hardware, with source-bound calibration and the existing
Fedora limits: at most 35 minutes per allocation; CPU stages at most eight
physical cores/48 GiB; GPU stages at most four CPUs, 32/48 GiB host memory, and
one actual RTX 4090. Preserve 18 GiB/75% soft VRAM and 90% hard thresholds.
Keep caches, datasets, checkpoints, logs, reports, and temporary files inside
the checkout's own venv/project root. Never silently fall back from CUDA to CPU.

Freeze requested work and calibrated allowances per stage/variant. Count parent
queries, action probes, source/path nodes, tangent/forced solves, optimizer
commits, forward/backward calls, nonlinear iterations, and validation work.
There is no arbitrary summed campaign ceiling or new pending-job cap. Long
trials continue exactly across allocations; continuation is not a restart.

Benchmark batch one first and separately declared batched/multi-query workloads.
Full inference includes anchor, encoder, residual/path generation, contractions,
quadrature, transforms, nonlinear iterations, precision bridge if computed, and
output assembly. Report resident-input and full-process/transfer scopes
separately. Cached decoding is not full inference latency.

Compare locked learned checkpoints with strong analytic/classical solvers at
matched final time and tolerance, allowing numerical controls their own suitable
step schedule. Select deployment schedules from validation only. A retrospective
fastest-method oracle frontier is a diagnostic, not a deployed policy.
Charge estimator/rejection/fallback work if an acceptance policy is added.

Report offline reference/action-label cost, training cost, report/telemetry cost,
and inference separately. Only compute training-amortization break-even where
positive measured savings exist for an explicit workload; disclose omitted
costs. Do not report energy or monetary savings from latency alone.

## 15. Raven contracts, worker telemetry, and portability

Read and record the actual Raven source revision and its current
`docs/PROJECT_STANDARD.md`, `docs/PROJECT_REPORTING_V2.md`, and automated
reporting/export guide. Reuse the documented captured-script launch behavior:
the coordinator runs with `--local` on the submission host and scientific work
uses native Slurm workers. Validate original-script/captured-script context,
strip outer `RAVEN_*` state before worker submission, and retain independent
coordinator and allocated-worker lifecycles.

The new recipe selects Experiment, Evidence, and Artifacts. Do not claim Arrays
integration unless the workflow actually uses job arrays. Coordinator counts
describe infrastructure; research decisions live in the scientific reports.
Use project-owned relative paths under each run's `raven/`, bounded/paged detail
projects where required by Raven, and a complete overview retaining missing
methods and negative outcomes.

Implement worker telemetry for CPU utilization, process/child-process RAM, and
allocated-device GPU utilization/VRAM/power/temperature where supported. Read
Raven's actual sampling and reader contracts before choosing filenames or CSV
columns. Retain discoverable telemetry alongside logs and include it in portable
exports so analytics survive relocation. Enabling GPU sampling on the host
coordinator does not monitor GPU Slurm workers.

Provide a declared sampling period (initial default 5 seconds) and an explicit
`--worker-telemetry cpu,memory,gpu` option plus a disable option. Collect from
allocation start through terminal cleanup, retaining partial streams on failure.
Record process/device scope, identifiers, timestamps, sampling gaps, unsupported
counters, observer errors, and sampling overhead. Counter absence is NA, never
zero. Hardware telemetry failure need not invalidate science unless it removes
a measurement required for a claim; report it distinctly. Validate CPU/RAM and
GPU charts with the actual Raven reader after moving a small export.

Publish relevant research only: mechanism/action/endpoint errors, mean and
centered decomposition, uncertainty/headroom coverage, ranks/conditioning,
training curves, precision effects, rollout behavior, complete costs, and
worker resource timelines. Avoid generic evidence scores presented as accuracy
or a novelty score. Include a metric legend with definitions, units, weighting,
scope, uncertainty meaning, and missing-value conventions.

Produce HTML, CSV/JSON, plots with source pointers, and a portable Raven ZIP.
Retain exact journals/checkpoints in the run for recovery. A compact review
omits large replay payloads only with an explicit omission inventory and hashes;
it is not a full prediction replay. Bound repeated metadata and large tables
using existing paged export patterns; do not silently truncate measurements.

## 16. Required runnable interfaces

Implement and verify the following proposed command contract. Actual commands
must appear in the delivered runbook, with implemented `--help` and flags that
agree with the controller. Resolve naming conflicts once, before submission,
and update this interface consistently. The commands in this section do not
exist merely because this handout lists them.

Local CPU implementation smoke:

```bash
cd /home/rahel/TDN
bash scripts/tangent_response_local.sh --profile cpu-smoke \
  --run-id tangent-response-cpu-a
```

Read-only native planning, then the default **gated pilot**:

```bash
cd /home/rahel/TDN
bash scripts/fedora_tangent_response.sh doctor
bash scripts/fedora_tangent_response.sh plan --profile pilot \
  --target fd_same_grid --run-id tangent-response-pilot-a

raven launch scripts/tangent_response_campaign.sh \
  --local --project "$PWD" --recipe tangent-response \
  -- --profile pilot --target fd_same_grid \
  --run-id tangent-response-pilot-a \
  --worker-telemetry cpu,memory,gpu
```

The explicit run ID above must name a fresh campaign. A plan does not reserve
or execute a run: it prints the protocol/work specification, marks uncalibrated
downstream allocations as unavailable, and submits nothing. Without `--run-id`,
the campaign generates a fresh ID and prints it; use that exact ID for all
inspection commands instead of the example below. The controller binds actual
downstream allowances only after their required calibration is valid.

The campaign creates fresh stage IDs and a receipt before work, executes native
readiness/calibration as needed, runs Stage A, evaluates its gate, and submits
Stages B/C only after valid evidence and eligible decisions. A negative gate
finishes a verified report without pretending that unrun stages completed.
Enable utility measurements only for eligible learned models. Print the exact
child run IDs, research project paths, decision, and export location.

Inspect and export an exact run:

```bash
bash scripts/fedora_tangent_response.sh status tangent-response-pilot-a --json
bash scripts/fedora_tangent_response.sh validate tangent-response-pilot-a
bash scripts/fedora_tangent_response.sh decision tangent-response-pilot-a
bash scripts/fedora_tangent_response.sh compact tangent-response-pilot-a --mode review
```

Recovery and archive must operate on explicit retained identities:

```bash
bash scripts/fedora_tangent_response.sh recover tangent-response-pilot-a \
  --run-id tangent-response-recovery-a --plan
bash scripts/tangent_response_archive.sh --campaign CAMPAIGN_TOKEN --plan
```

Implement corresponding execute forms with terminal-job and stopped-coordinator
checks; preserve every original artifact. Never cancel unrelated work, overwrite
completed results, reuse an existing run ID, or silently retry an ambiguous
submission. Fresh launch means fresh IDs/calibration; exact recovery requires
the same frozen source/software/data and retains all attempts.

A larger `--profile development` launch requires a literal prerequisite pilot
decision and calibration identity, e.g. `--prerequisite RUN_ID`. Implement a
separate continuation/reconcile command if a coordinator dies between stages;
do not resubmit a fresh whole campaign under the pretense of recovery.
Confirmation is not included in the default runnable.

Use `set -euo pipefail`, validated argument arrays, and actual scheduler status.
Distinguish submission receipt from execution. Poll in bounded intervals, retain
logs/status, and let failure-aware final observers complete. Native infrastructure
failure returns nonzero. A complete negative scientific result can return zero
with an explicit `STOP` decision. Export verification failure is infrastructure
failure, even if numerical stages succeeded.

## 17. Completion and handoff checklist

The main agent's final delivery must include:

1. Implemented source, stage protocols, CLI/Bash scripts, config defaults, and
   a runbook tested against their real argument parsing.
2. Mathematical derivations, independently reproduced tiny checks, and
   source-traced mechanism/compression/learning reports where execution exists.
3. Appropriate regressions for the changed/shared paths, complete CPU smoke,
   declared native CUDA suite, and an explicit distinction between checks run
   here and checks awaiting the actual Fedora device.
4. A literature/novelty matrix with verified references and a precise claim
   candidate; no unsupported novelty promise.
5. Raven captured-script, worker-logging, telemetry discovery, and relocated
   portable-report verification, with limitations stated.
6. Frozen work/calibration/recovery contracts, all-attempt accounting, and
   complete negative/failure/unresolved routes.
7. Exact native launch/status/validate/decision/export/recovery commands. Link
   implemented files or the pushed commit when publication is authorized.
8. One recommended next numerical action selected by evidence, with no broad
   campaign expansion merely to exercise every implemented variant.

Code completion, CPU verification, native GPU readiness, mechanism benefit,
learning benefit, practical utility, and fresh confirmation are separate
statuses. Report each independently.

## 18. Legend

| Term or symbol | Meaning here |
| --- | --- |
| TDN | Temporal-defect network: a learned correction to a numerical anchor |
| FD | Finite difference; the declared same-grid nodal equation |
| Galerkin / `P_N` | Projection of the equation/product onto retained Fourier modes |
| Anchor `a(t)` | Declared smooth numerical reference path, not the true solution |
| `u`, `u0` | Solution field and actual initial input |
| `e=u-a` | Physical error of the smooth reference path |
| Residual `rho` | `a'-f(a)`, failure of that path to satisfy the finite equation |
| `L`, `kappa`, `r` | Diffusion generator, diffusion coefficient, reaction rate |
| `B(a,z)` | Target-specific quadratic product; projected in Galerkin |
| Tangent operator `A_a` | Linearization of the equation along the anchor path |
| Response `U_a(t,s)` | Propagation of a small perturbation from time `s` to `t` along that path |
| Matched mean response `G_a` | Heat propagation with the anchor's spatially averaged reaction feedback |
| `mu_a`, `v_a`, `H_a` | Anchor mean, centered anchor, and omitted spatially varying linear feedback |
| Nonlinear remainder | Quadratic error feedback `-rB(e,e)` absent from first linearized correction |
| Oracle | Numerically expensive diagnostic using information unavailable to a deployable method; never a free inference input |
| Response action | Applying a response operator to a specified field without constructing its full matrix |
| Rank `J` | Number of factor directions in the declared approximation; distinct from width and parameter count |
| `alpha`, `beta`, `c` | Learned output fields, input-projection fields, and signed temporal coefficients |
| Causal / time ordered | Earlier forcing affects later response; operators at different times cannot generally be reordered |
| POD / SVD | Proper orthogonal decomposition / singular-value decomposition for fitted compression and conditioning |
| DEIM | Discrete empirical interpolation, a related reduced nonlinear-evaluation method for literature comparison |
| JVP | Jacobian-vector product: directional derivative without a full Jacobian matrix |
| Variable projection | Solving an affine coefficient subproblem conditional on nonlinear features/path, then optimizing those features |
| Conditioning | Sensitivity to data/parameter/arithmetic perturbations; large cancelling factors may be unstable |
| RMS / MSE / maximum | Root mean squared / mean squared / largest sampled absolute error |
| Mean / centered error | Error in spatial average / remaining spatial pattern |
| FP32 / FP64 / ULP | 32-bit / 64-bit arithmetic / spacing of representable values near the output |
| Arithmetic bridge | Smooth FP64 anchor minus the actual FP32 deployed anchor |
| Headroom | Measurable room for improvement before uncertainty or output precision dominates |
| Parent / query / probe / seed | Independent field / paired evaluation / paired operator input direction / training or generation randomization |
| Macrostep / rollout | One model advance / repeated advances with re-encoding |
| Composition | Agreement between advancing over one interval and over its subdivisions, under the correct path/state contract |
| CPU / GPU / VRAM | Host processor / graphics processor / dedicated GPU memory |
| Calibration | Measured workload used to freeze allocations; not scientific success |
| Checkpoint / seal | Saved model state / verified artifact identity and completion contract |
| `STOP`, `UNRESOLVED`, `PASS`, `NOT_APPLICABLE` | Negative evidence, insufficient valid evidence, met declared criterion, unsupported comparison; never interchangeable |
