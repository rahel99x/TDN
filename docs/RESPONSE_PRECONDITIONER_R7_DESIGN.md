# R7 design memo: distinguish representation, conditioning and cost

Status: **prospective design, no new numerical evidence collected**. The
[independent review](../results/native-response-preconditioner-2936/README.md)
accepts native job 2936 at `261365b`; that acceptance establishes the revised
implementation, not scientific promotion. This is not yet a fully frozen or
executable R7 protocol: the source roster, exact source seed/family list, work
caps, timing design and decision requirements still require implementation and
prospective sealing.
The historical Tangent Response A PASS / B STOP and the inspected four-parent
implementation bank remain unchanged. The complete bounded R7 protocol must be
frozen, implemented and verified before native numerical expansion. This memo
is a recommendation for that protocol, not an executable launch contract.
Use the [standing research program](TDN_RESEARCH_PROGRAM.md) as the maintained
entry point; the current [runbook](RESPONSE_PRECONDITIONER_RUNBOOK.md) runs only
implementation verification and the old tiny smoke, not this proposed study.

## The smallest useful question

Can the *existing five operator branches*, at their measured complete cost,
approximate fixed-anchor actions on sources excluded from coefficient fitting?
If so, does a source-independent coefficient map predict those coefficients
across parents and time intervals? These are distinct questions. Endpoint,
trajectory, derivative-task, GPU and novelty claims remain separate later gates.

At context q = (u0, physics, geometry, anchor rule, s, t), define tau = t-s and

    K_q(c) z = P_q z + sum_{j=1}^5 c_j B_{q,j} z,
    (B_1,B_2,B_3) = tau^3 (C_1,C_2,C_3),
    (B_4,B_5) = (tau/t_ref)^3 (D_b H_tau D_c)_{1,2}.

Use exactly the current factors, commutators, anchor and two-panel P. At this
fixed context the eight scalar features eta(q) are source-independent and
c = W eta(q), with W having five rows and eight columns. There are five
coefficient degrees of freedom per context, not a rank-five physical operator:
each branch can have full matrix rank. The observed global design rank 25/40
belongs to the original tiny data matrix; it is not a universal model rank.

## Three fits that must not be conflated

1. **Local representation projection.** Stack B_j z for calibration sources z
   and fit one c shared by every source at q. The unregularized physical-norm
   least-squares residual estimates the best fit in this five-branch action
   span on those calibration sources. Use an independent FP64 QR/SVD and
   declare its numerical rank convention; report singular spectrum, fitted
   residual, normal residual, coefficient norm and projection sensitivity.
   A truncated solve has an optimum only in its declared numerical subspace.
   Its training residual is not a lower bound on unseen-source error.
2. **Local regularized oracle.** Solve the same source-shared five-coefficient
   problem with a strictly positive ridge, reporting its actual objective and
   sensitivity. This is a stability control, not the representational floor.
   The original 40-weight ridge penalty cannot be silently copied to c. For a
   single context, because eta includes a constant feature,

       min_{W eta=c} ||W||_F^2 = ||c||_2^2 / ||eta||_2^2,
       W_min = c eta^T / ||eta||_2^2.

   Thus lambda ||c||^2 is generally a different objective. With context data
   weight pi_q, a separately normalized local loss requires
   lambda_c = lambda / (pi_q ||eta(q)||^2) to reproduce the one-context induced
   penalty. Even this does not reproduce the joint global problem: independent
   c_q relax cross-context coupling. Label it as such. Do not declare an oracle
   superior to W solely because its penalty, constraints or data differ.
3. **Deployable global readout.** Fit the existing W only on training parents,
   using their calibration directions, preserving parent-balanced physical
   weights and the declared ridge menu.
   Evaluate initialization as a candidate. Use an action-only selection loss
   for this action-capacity question; retain the old mixed-loss selection as a
   historical comparator, rather than treating its unresolved endpoint rows
   as informative task supervision. Freeze the selected W before accessing
   diagnostic-parent truth. A label-informed oracle is never a deployable
   competitor and its teacher/fitting cost is always marked offline.

For an exact global-versus-relaxed *regularized objective* comparison, implement
the explicit constrained variable change C = W E, retaining the minimum-W-norm
penalty and cross-context feasibility; otherwise compare physical action errors
and report that objectives differ. This extra algebra is optional in the first
study. The independent local projection already answers the narrower question.

## Bounded cohort and withholding proposal

A conservative first R7 development bank can reuse the four inspected parents
and 5-by-5 FD grid. This prevents a smoke replay from being mislabeled fresh
confirmation. Reuse the exact two-train/one-validation/one-diagnostic split.
Here "held out" means excluded from the stated fit, not a previously unseen
parent: all four fields have already been inspected. Results cannot establish
fresh-population generalization. A future confirmation requires new parents.
Add source-time coverage without changing physics or extending beyond the
existing maximum physical time 0.08. Candidate (s,t) contexts are

    (0,.04), (0,.08), (.02,.04), (.02,.06), (.04,.06), (.04,.08).

The last four expose nonzero source times, with repeated lags .02 and .04.
This is six contexts per parent, 24 contexts total. Do not claim a full tensor
start/lag design. Avoid regenerating a path from a(s): every interval uses the
same original anchor a(q;u0). Reference integrators need an actual nonzero
initial integration time and z at s; subtracting s while also restarting the
anchor changes the target.

For each context, predeclare four normalized calibration sources and four
normalized withheld sources, giving at most 192 full-action source queries.
Use deterministic, separately hashed streams; normalization and seeds must be
fixed before teachers. For example, use random dense calibration sources and
withheld random/low-frequency/high-frequency/constant probes, with categories
reported separately; no source family may be selected after seeing errors.
The exact family/seed list, reference caps and resource ceiling belong in the
frozen machine-readable protocol. A chosen source count is development coverage,
not a reliability guarantee or universal operator-norm test. Descendants of
one field remain paired observations, not new independent parents.

Access order:

1. Freeze source family, contexts, task losses, weighting, ridge menu, controls,
   perturbations, reference procedure and maximum work. Validate nonzero-time
   integration and five-branch extraction on analytic/tiny-matrix fixtures.
2. Acquire training-calibration and validation references; fit W using only
   training calibration directions and select on the predeclared withheld
   validation directions. Those validation labels are model-selection data,
   not confirmation. Keep training withheld directions for a subsequent
   diagnostic report, without using them to fit or choose W.
3. Freeze checkpoint, source/software/protocol fingerprints and selection.
4. For each diagnostic context, expose calibration labels only, fit its local
   oracle, and seal its coefficients. No source-specific fitting is permitted.
5. Acquire withheld diagnostic sources and evaluate every frozen candidate.
   A diagnostic oracle is explicitly label-informed at its own context; this
   does not contaminate the separately sealed deployable W.
6. Preserve all unresolved/failed cases and denominators. Any adaptive choice
   based on these diagnostics requires a new development protocol, not reuse
   of the same observations as fresh confirmation.

Optional full-operator audit is a separate, later bounded check. On one or a
few predeclared tiny contexts, integrate the fundamental matrix independently
and form the five branch matrices on canonical inputs. Fit the matrix Frobenius
projection and report a physical-mass-weighted residual/operator norm. Fitting
all canonical columns consumes the full operator and is not a withheld-source
experiment. Report it as a representation audit only; run it after choices are
sealed and never use its diagnostic optimum to tune the deployed model. Its
25-column cost must be included in the explicit maximum-work inventory. A
verified least-squares projection supplies the minimum Frobenius residual for
that fixed tiny operator family, subject to reference and arithmetic uncertainty.
The spectral residual of that Frobenius optimum is not the minimum spectral
residual. For an N-dimensional square operator, min_c ||R(c)||_F / sqrt(N) is a
lower bound on min_c ||R(c)||_2 in exact arithmetic; ordinary FP64 estimates
and uncertain references are not automatically certified lower bounds.
Use the same mass-transformed residual M^(1/2) R M^(-1/2) in both matrix norms,
where M is the declared physical cell-mass matrix. State any Frobenius/RMS
normalization explicitly; changing normalization changes the numerical bound.

## Independent accuracy and perturbation controls

Use refined Lawson integration and an independently implemented time-dependent
matrix/action integrator on the *identical* anchor and interval. Freeze both
RMS and maximum acceptance conditions, refinement ceilings and failure handling.
Their agreement is a reference-uncertainty estimate, not an error certificate.
Keep the existing teacher acceptance policy where the estimand is unchanged.
Apply any correction-relative resolution floor to independent uncertainty,
retaining unresolved near-null responses as unresolved.

Record errors in U z, (U-P) z, and (U-G) z separately. K-P is the model's fitted
remainder; U-G is the older spatial-feedback estimand and is not interchangeable.
Report physical RMS/maximum, signed mean error and centered RMS. Never infer
matrix rank or kernel coverage from one successfully fitted vector.

For the regularized local/global fit, retain the linear map from weighted
labels to coefficients. In the local notation B = sqrt(Omega) A,

    S = (B^T B + lambda_c I)^(-1) B^T,
    delta c = S delta d,
    delta prediction_on_withheld = A_test S delta d.

Evaluate S with SVD, without forming unstable normal equations. Report
||S||, coefficient change, and held-out prediction amplification separately.
The exact-arithmetic bound ||S|| <= 1/(2 sqrt(lambda_c)) is a useful check,
not a substitute for the measured prediction sensitivity. For a fixed weighted
L2 label-radius epsilon, the worst-case prediction change for that norm is
||A_test S|| epsilon; exercise its leading singular direction and fixed seeded
perturbations, plus baseline/refined independent reference differences.
Include the output mass/RMS normalization in A_test when quoting gain in that
norm; an unnormalized Euclidean gain is not automatically an RMS gain.

Reference estimates define the scale of these sensitivity scenarios; they do
not prove that every teacher error lies inside that ball. A weighted aggregate
L2 uncertainty ball is also not the same as individual per-source bounds.
Declare the uncertainty set and report that interpretation. Perturb only the
training/calibration labels, never refit on withheld targets. Do not choose a
ridge after inspecting diagnostic perturbation outcomes.

## Classical controls and cost scope

Minimum mandatory measured action controls are ordered potential-heat products
with 1, 2, 4 and 8 panels on the identical anchor, and a deployable fixed-step
Lawson response control with predeclared steps. Refined adaptive Lawson/DOP853
used to establish labels is not automatically the strongest production solver.
Use predeclared step choices and compare the cheapest qualifying classical
method, preserving all accuracy failures. Include block/multiple-source execution
if the learned method receives equivalent reuse or batching.

Add a BCH analytic-coefficient control only after an independent derivation
and static/time-dependent tiny-matrix verification establish the signs and
nonautonomous term for this exact P. Do not copy coefficients from a different
splitting or interpret local order as stiff stability. Tiny static heterogeneous
expm is an implementation check, not the nonautonomous production target.
ETDRK4, Radau/BDF, Duhamel/cumulants and conditioned-wavelet baselines remain
important later, but implementing the entire ladder is not necessary before
this bounded capacity falsification. The old eight-panel accuracy advantage
must remain visible whether the new local oracle succeeds or fails.
Accuracy advantage is not Pareto dominance: in
the reviewed run the learned actions were faster in single-call observations
while less accurate. Include predeclared intermediate panel counts where needed
to resolve the classical work-precision frontier, without claiming matched-cost
superiority from those unrepeated observations.

Charge complete path/derivative construction, all potential quadrature calls,
feature construction, stencils, heat actions, output assembly, and all precision
bridges. Instrument anchor/autograd work or state explicitly which counters
remain incomplete; end-to-end measured time is still required. Separate:

- single fresh state/action calls, including setup every time;
- repeated sources on an identical context, setup plus every action;
- a changing-state forward endpoint/trajectory, which may not reuse a previous
  state's features or path.

Predeclare cold/warm handling and repeated paired randomized timing rounds,
thread settings, batch layout, memory measurement and timing uncertainty. Cache
keys must include state/path, physics, geometry, model, interval, dtype/device
and gradient contract. Never reuse detached inference state for a full-map
JVP. Report oracle acquisition/fitting separately; omit oracle timing from a
"deployable speedup" column. The first study can report work-precision curves
without claiming superiority; a utility gate requires a separately frozen
accuracy requirement and practical cost margin. Do not invent a favorable
threshold after inspecting these results.

## Mechanism decisions, without post-hoc thresholds

- Calibration oracle residual remains resolved and large relative to the
  predeclared target and uncertainty: first distinguish positive-ridge bias,
  numerical projection error and a genuinely resolved unregularized sampled
  fit residual. This does not establish a general full-operator limitation.
  A verified full-operator projection on a fixed tiny context can establish
  that context's Frobenius representation floor, with explicit uncertainty;
  it does not establish a universal limitation across states/grids. A scalar
  coefficient predictor cannot remove a confirmed span error with those fixed
  factors, but an encoder that changes the factors changes the tested family.
- Calibration oracle is good but withheld-source action is poor: source
  coverage, conditioning or an ill-constrained operator component is limiting.
  Examine source-family results and perturbation sensitivity; do not call it
  global five-branch sufficiency. A prespecified full-operator audit can clarify.
- Stable oracle works on withheld sources but sealed W fails there: there is a
  coefficient-map/coverage bottleneck. Distinguish within-parent time transfer
  from held-out-parent transfer; consider deterministic feature enrichment or
  a small conditioned model only under a new frozen protocol.
- W improves action error but loses complete matched-tolerance cost to the
  best classical control: action approximation is established without utility.
  Optimize charged work or choose a genuine repeated-source use case before
  adding model depth.
- The action and endpoint cost contracts must remain separate. The reviewed
  native endpoint has extra learned-branch work while its actual fitted change
  is below reference uncertainty; the repeated-response use case remains an
  eligible question because its single-call accuracy/cost tradeoff is different.
  Neither observation establishes a matched-tolerance speedup.
- Action improvement is resolved but endpoint change is below independent
  endpoint uncertainty: endpoint benefit is unresolved. Do not relabel a lower
  raw float as success or tune endpoint weights to magnify numerical noise.
- Endpoint/trajectory benefit later fails despite adequate action: decompose
  anchor, response, source quadrature, nonlinear remainder and arithmetic with
  paired vectors and signed cross terms. Only a measured nonlinear remainder
  supports a second correction stage. Local source-action oracles do not
  automatically define differentiable deployable endpoint maps.

No R7 result alone establishes R8, R9, R10 or R11. Native implementation
verification, scientific capacity, coefficient predictability, practical utility
and fresh confirmation require distinct statuses and evidence. The current
native success permits preparation and independent verification of this next
bounded study; it does not itself authorize a larger learned/GPU campaign.

## Useful outputs for external review

A source-bound protocol, parent/source/context index, immutable access journal,
reference refinements and uncertainty, separate oracle/global coefficients and
objectives, original and augmented spectra, original-design rank conventions,
withheld predictions, perturbation construction/results, per-method work and raw
paired timing rounds, failures and missing denominators. Publish portable JSON,
CSV and allow_pickle=False NPZ, plus readable terms and decision reasons. Raven
resource telemetry must be separate from science metrics. Any automated pass
should mean the stated contract completed, not that novelty or utility passed.

## Term guide

**Context** is the initial state, physics, geometry, anchor and time interval
that define one operator. **Source/forcing direction** is the vector z acted
on by that fixed operator; it must not affect its coefficients. **Oracle** is
a diagnostic fit allowed to see labels unavailable to a deployed model.
**Projection** is the smallest least-squares residual within a specified span;
**ridge** adds a positive coefficient-norm penalty. **Conditioning** describes
how perturbations can be amplified and is separate from representation error.
**Identifiability** concerns whether the data distinguish coefficient choices.
**Frobenius norm** sums squared matrix entries; **spectral norm** measures the
largest Euclidean gain. **BCH** means Baker–Campbell–Hausdorff, the expansion
used to analyze noncommuting exponential products. **Matched tolerance** means
every compared method meets the same predeclared physical accuracy requirement.
**Pareto dominance** would require no worse accuracy and no greater complete
cost, with a strict improvement in at least one. None is established here.
