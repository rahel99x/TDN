# TDN standing research program

**Maintained reference, started 15 October 2026.** Read this document before
changing the new response-preconditioner methods. Update the ledger and decision
log when implementation, assumptions or evidence change. Preserve dated
proposals and old results as historical snapshots.

The objective is an accuracy-controlled solver whose learned finite-time
response reduces complete cost while retaining trajectory and derivative
accuracy. The [full mathematical proposal and term legend](TDN_ARCHITECTURE_PROPOSAL_20261015.md)
records the derivation, competing methods, guarantees and novelty limitations.
The [runbook](RESPONSE_PRECONDITIONER_RUNBOOK.md) contains executable interfaces
and their verification status.

## Evidence that determines the current scope

- The [native Tangent Response review](../results/native-tangent-compression-20261013/README.md)
  retains A PASS and B STOP. Spatial feedback improved the inspected endpoints,
  but the existing compressed candidates did not meet the action/gain gates.
- The [13 October proposals](TANGENT_RESPONSE_PROPOSALS_20261013.md) distinguish
  conditioning, missing information and representation. The present branch
  tests a different representation without relabeling those negative results.
- The [15 October primary-source review](../results/tdn-architecture-proposal-20261015/primary-source-review.json)
  establishes close precedents for cumulants, conditioned compressed operators,
  and coefficient/Fourier constructions. Novelty and superiority remain open.
- Existing CPU tests, tiny coefficient fits and native CPU implementation
  checks cannot establish scientific promotion or GPU readiness.
- [Native job 2930](../results/native-response-preconditioner-2930/README.md)
  completed all 184 checks and the tiny smoke at `627de9f`. Its source and
  artifact hashes were independently verified. This closes native R6 for
  that original version.
- [Native job 2936](../results/native-response-preconditioner-2936/README.md)
  completes the bounded repair at `261365b`: all 202 expected checks, the
  frozen v2 protocol and the 24-reference/82-metric smoke are independently
  verified. The revised implementation now has native CPU evidence.

## Three products, three cost contracts

| Product | Required evidence | Cost boundary |
| --- | --- | --- |
| Forward solver | Accuracy throughout an evolving physical trajectory | Complete solve, including fresh state conditioning and rejected work |
| Repeated response engine | Held-out source actions on a declared fixed path | Path/encoding setup plus all actions, with legitimate reuse |
| Differentiable simulator | Accurate objective and derivatives; successful inference/control | Complete optimization to a common stopping target |

Keep same-grid FD, same-grid dealiased Galerkin and projected-fine dynamics
separate. The new implementation starts with periodic same-grid FD only.

## Method contract

For F(u)=Lu+r u(1-u), a smooth Strang anchor a has defect rho=a'-F(a).
The exact error satisfies

\[
e'=A_a e-r e^2-\rho,\qquad A_a=L+r\operatorname{diag}(1-2a).
\]

The first implementation combines two ordered potential–heat–potential panels,
three explicit sparse commutator actions, and two signed two-sided heat
branches. Coefficients depend on actual state/physics/time, not the forcing
direction. The source action remains linear. Full endpoint derivatives must
differentiate the anchor, features, coefficients and quadrature together.

The initial features are fixed deterministic functions, with an affine readout
fitted by parent-balanced ridge least squares. This is a globally solvable
conditional regression problem. It does not implement a trained nonlinear
encoder, variable-projection outer optimization or a nonlinear second stage.
Zero readout gives the classical response-corrected step, not the bare anchor.

For frozen features the implemented objective is
`J(w) = sum_i omega_i (A_i w - b_i)^2 + lambda ||w||^2`, with `lambda > 0`
and equal total weight per independent training parent. It has a unique
minimizer in exact arithmetic. The augmented FP64 SVD preserves this original
penalty through column scaling. The bound
`J(w)-J(w*) <= ||grad J(w)||^2/(4 lambda)` motivates its recorded numerical
optimality diagnostic; ordinary FP64 evaluation is not a certified bound.
This guarantee concerns the conditional regularized training objective,
not validation generalization, identifiable coefficients or future nonlinear
encoder training.

Following native review, scaling uses the norms of the original **augmented**
columns, `hypot(||sqrt(W) A_j||, sqrt(lambda))`, including the transformed ridge
penalty. Inference contracts the eight scalar features into five effective
coefficients before spatial multiplication. The 40-column design remains the
fitting interface. A separate fresh classical corrected endpoint evaluates
only the ordered response and GL3 residual quadrature, giving a usable complete
baseline cost instead of a cached attribution lookup.

A separate cubic reconstruction auditor checks the finite FD equation. Its
status must identify whether arithmetic and whole-interval enclosure support
a certificate or only an indicator. A panel certificate is neither a continuum
error bound nor a blanket guarantee for subsequent adaptive steps or gradients.

## Implementation ledger

Update this table from actual source and receipts, never from the intended plan.

| ID | Method or contract | Current status | Evidence / next gate |
| --- | --- | --- | --- |
| R0 | Equation, error identity, costs, baselines and literature | Documented | Dated proposal and primary-source record above |
| R1 | Ordered FD response and sparse commutators | Implemented, CPU verified | Independent tiny-matrix, analytic-limit and gradient checks in the receipt below |
| R2 | Fixed-feature two-sided response and complete endpoint map | Implemented, CPU verified | Source linearity, affine-design parity and complete-map JVP |
| R3 | Parent-balanced stable conditional ridge fitting | Implemented, CPU verified | Independent least-squares comparison and numerical optimality gap |
| R4 | Independent cubic finite-grid residual audit | Implemented, CPU verified | Exact Bernstein enclosure and outward bound; one panel, at most 64 spatial DOF |
| R5 | Bounded CPU verifier and tiny fit workflow | Implemented, CPU verified | Fresh output roots, retained failures, source hashes and held-out diagnostics |
| R6 | Raven native CPU verification wrapper | Native CPU verified at `261365b` | Job 2936: exact 202/202 checks, source/protocol integrity and smoke completed |
| R7 | Representation at fixed complete cost | Scientific study gated; bounded repair native CPU verified | Prospective identification design below; executable source/coverage/cost protocol still required |
| R8 | Learned multiscale encoder / variable projection | Proposed, scientifically gated | R7 capacity and stable fitting evidence |
| R9 | Second nonlinear correction or inexact Newton | Proposed, scientifically gated | Measured nonlinear remainder and charged causal work |
| R10 | Adaptive rollout/controller and derivative task | Proposed, scientifically gated | Whole-time accuracy, schedule derivative contract and independent task checks |
| R11 | Native GPU utility and fresh confirmation | Proposed, scientifically gated | Source-bound new CUDA roster, calibrated work, qualifying prior gates |

ETDRK4/Radau/BDF additions, conditioned-wavelet comparisons, independently
identified bridge-cumulant branches and online multi-panel certification remain
proposed. Existing numerical routines are reused only where their contracts
match; an imported routine does not establish a new comparison result.

The [15 October implementation receipt](../results/response-preconditioner-verification-20261015/README.md)
contains the exact commands, JUnit cases, source hashes and complete tiny
replay data. Its 178 mandatory checks and 27 optional Raven compatibility
checks overlap; those counts must not be added as distinct tests. The tiny
fit covers two training, one validation and one diagnostic parent.

The original local and native results do not justify expanding the architecture yet. Fitting
improved the four inspected response actions over the two-panel initialization,
but the classical eight-panel response remained more accurate on all four.
Endpoint differences from initialization were below reference uncertainty.
The selected augmented system's condition number was approximately 4.77e10;
a small objective gradient does not establish coefficient identifiability.
Two tiny cubic-panel upper bounds were obtained, with no requested tolerance
acceptance or matched-accuracy speedup established. R7 needs a separately
frozen, bounded scientific protocol and native prerequisites before expansion.

The [bounded repair protocol](RESPONSE_PRECONDITIONER_REPAIR_PROTOCOL_20261015.md)
keeps the original objective, ridge menu, selection and four development
parents. Its [retained-matrix replay](../results/response-preconditioner-repair-20261015/retained-fit-replay.json)
reduces augmented condition from 4.77e10 to 93.8 at the selected ridge without
changing original rank 25/40. Predictions change by at most 1.14e-13 RMS on
the retained queries. This is a numerical repair, not new scientific evidence.
The combined tiny-bank loss is action-dominated; endpoint signals are below
reference uncertainty. A future R7 design must resolve nonzero source-time
coverage, basis-versus-coefficient identification and complete repeated costs
before a learned encoder is justified. Independent per-state oracle and
classical-control derivations remain proposals, not implemented competitors.
The [repair verification receipt](../results/response-preconditioner-repair-20261015/README.md)
records 202 mandatory CPU passes, 33 overlapping Raven compatibility passes
and the completed 24-reference/82-metric development replay. Native job 2936
subsequently reproduced it: condition 93.8, rank 25/40 and unchanged teacher/
design arrays relative to native job 2930. All source files and both protocol
identities match the committed prelaunch receipts.

The new complete classical endpoint control exposes no resolved learned
endpoint benefit. Its native single calls took 8.6–8.7 ms versus 14.7–14.8 ms
for the learned endpoint; the learned corrections were far below reference
uncertainty. Response actions have a different tradeoff: fitted calls were
faster than eight panels in single observations but 1.45–1.63 times less
accurate. This is neither a matched-tolerance speedup nor classical Pareto
dominance. The combined validation loss is effectively all action error, so
increasing endpoint weights would amplify unresolved labels in this fixture.

The next eligible implementation is the [R7 identification design](RESPONSE_PRECONDITIONER_R7_DESIGN.md):
test source-independent five-coefficient oracles at nonzero source times,
withhold source directions, separate the global readout from representation
capacity, assess reference-sized label perturbations and compare complete
classical costs. A calibration oracle is not a deployable competitor; its
failure alone does not establish a full-operator representation limit. This
design is proposed, not a frozen executable scientific protocol. No further
repeat of the unchanged implementation smoke is required by this review.

### Source interfaces

| Interface | Purpose |
| --- | --- |
| [numerics.py](../tdn/analysis/response_preconditioner/numerics.py) | Ordered response, commutator actions and two-sided heat |
| [model.py](../tdn/analysis/response_preconditioner/model.py) | `FixedFeatureResponse`: forcing-linear action and full differentiable endpoint |
| [fitting.py](../tdn/analysis/response_preconditioner/fitting.py) | `fit_parent_balanced_ridge`: stable affine coefficient solve |
| [audit.py](../tdn/analysis/response_preconditioner/audit.py) | `audit_cubic_fd_panel`: independent whole-panel residual upper bound |
| [workflow.py](../tdn/analysis/response_preconditioner/workflow.py) | Frozen implementation-only protocol, verifier, fit and artifacts |

Use the runbook's wrappers for reproducible execution. The full derivation,
baseline ladder, term legend and prior-art boundaries remain in the dated
proposal linked above; this ledger records what has actually been built.

## Required metrics and acceptance

Record endpoint and intermediate physical RMS/maximum errors, mean/centered
decomposition, correction/action errors, teacher uncertainty, source coverage,
conditioning, ridge objective and numerical gap, JVP/finite-difference error,
residual enclosure and exact scope, initialization versus selected model,
operation counts, measured timing scope, and all failures/missing results.

Training parent weights and normalization are fixed before fitting. Validation
selects among declared choices; diagnostic data never fit features, weights or
hyperparameters. Paired descendants are not independent parents. Reference
refinement agreement is an estimate, not a certificate.

Compare against bare Strang, the ordered-response-corrected initialization,
and appropriately accurate classical controls. Eventually compare the fastest
qualifying method at equal physical output/tolerance requirements. Include
path, residual, encoding, quadrature, stencils, transforms, precision bridges,
auditing, rejection/fallback and offline acquisition/fitting in their declared
cost scopes. No speedup follows from a small model or a fast isolated kernel.

The existing gates stay unchanged. A new scientific protocol needs a new
frozen identity and fresh run roots; software verification cannot promote it.
The first runnable has no automatic larger-training, CUDA or confirmation
successor. Negative scientific observations remain visible when execution
finishes successfully.

## Research decision log

| Date | Decision | Reason / implications |
| --- | --- | --- |
| 2026-10-15 | Preserve the dated proposal; make this the stable entry point | Future work needs one maintained ledger without changing historical evidence |
| 2026-10-15 | Implement an isolated `response_preconditioner` namespace | Preserve old Tangent Response contracts, thresholds and failed outcomes |
| 2026-10-15 | Start with fixed features and an affine head | Separate representation and numerical conditioning from nonconvex optimization |
| 2026-10-15 | Add explicit commutator features | A bounded tau-cubed heat bank has a diagonal leading term and cannot generically cancel off-diagonal third-order splitting error |
| 2026-10-15 | Keep derivative and response objects distinct | The complete solver's Jacobian includes derivatives of its state-conditioned response |
| 2026-10-15 | Bound this milestone to software checks and tiny CPU fits | Existing B STOP does not authorize broader learning or a superiority claim |
| 2026-10-15 | Preserve initialization and reject numerically ineligible fits during selection | Optimization failure must remain visible and cannot become a deployed candidate |
| 2026-10-15 | Keep derivative-reference status separate from endpoint-reference acceptance | An independent variational solve without a refined uncertainty estimate is a diagnostic |
| 2026-10-15 | Retain the classical eight-panel advantage and poor coefficient conditioning | Passing software checks does not establish capacity, utility or novelty |
| 2026-10-15 | Guard interactive preparation and submission instead of using a bare shell exit | A preparation failure must remain visible without closing the user's terminal or submitting an empty contract; numerical code and evidence remain unchanged |
| 2026-10-15 | Deduplicate only identical installed-package metadata discoveries | Fedora path aliases may enumerate one package twice; distinct metadata locations or versions still fail preflight. See the [verification receipt](../results/response-preconditioner-package-identity-20261015/README.md); prepare a fresh source-bound contract after updating |
| 2026-10-15 | Record native job 2930 as CPU implementation verified | Exact source/artifact/roster checks pass; final independent scheduler accounting is absent and worker resource coverage is partial |
| 2026-10-15 | Repair augmented scaling and stream the fixed readout before expanding capacity | Preserve the original objective and full derivatives; rank and source-coverage limits remain; the existing bank stays inspected development evidence |
| 2026-10-15 | Add a fresh classical corrected endpoint cost | Cached zero-head attribution is retained but cannot be used as a complete solver timing |
| 2026-10-15 | Accept native job 2936 as verification of the bounded repair | 202 exact expected tests, frozen v2 protocols, 24 references and 82 metrics pass independent receipt checks; no scientific promotion |
| 2026-10-15 | Separate endpoint utility from the response-engine question | Fresh classical endpoint has no resolved accuracy disadvantage and less observed work; action accuracy/cost remains a tradeoff requiring repeated matched-tolerance comparison |
| 2026-10-15 | Prepare R7 identification before adding an encoder | Distinguish sampled-source projection, regularization, coefficient-map prediction and full-operator capacity; retain explicit prospective scope |

## Maintenance rule

For each follow-up, record the question, allowed information, target equation,
changed implementation, exact checks, outcome, unresolved limitations and next
eligible gate. Link the run's source-bound receipt and retain failed attempts.
Native verification is recorded only after actual native execution. A new
assumption or threshold must be declared before collecting the evidence used
to judge it.
