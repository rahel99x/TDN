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
- [Native R7 job 2939](../results/native-response-preconditioner-r7-2939/README.md)
  completed at `43837c5`: 298 expected checks, 192 accepted queries, 2,112
  metric rows and 336 complete timing bundles pass independent review. The
  learned response improves its base but has no established utility advantage;
  Lawson 2 is substantially more accurate at nearly the same observed cost.
- The [Phase 1 scaling protocol](RESPONSE_PRECONDITIONER_SCALING_PHASE1.md)
  starts the user-authorized process change after that review: independently
  vary physical-parent count and source-direction count while retaining the
  current architecture and objective. This is new bounded development, not
  an R8 encoder promotion or evidence that a larger bank will succeed.
- [Native P1 job 2944](../results/native-response-preconditioner-scaling-2944/README.md)
  completed the numerical study at `7d0c620`: 388 exact checks, 2,688 accepted
  reference pairs, nine sealed fits, 16,128 exactly replayed metric rows and
  1,008 complete timing bundles. The wrapper failed afterward during packing;
  preserve its FAILED status. More physical parents help, but Lawson 2 retains
  a large accuracy advantage at similar cost. Review now prioritizes the
  regularization/physical-prior question before further data or architecture.
- The user-authorized [P1R retained-data audit](RESPONSE_PRECONDITIONER_REGULARIZATION_AUDIT.md)
  is now implemented as the current phase. It compares weaker coefficient
  penalties and an action-energy shrinkage control on the reviewed P1 arrays,
  with unchanged architecture and validation-only selection. All 585 mandatory
  CPU checks and eight additional Raven compatibility cases passed in the
  [receipt](../results/response-preconditioner-regularization-verification-20261015/README.md).
  [Native job 2951](../results/native-response-preconditioner-regularization-2951/README.md)
  now completes P1R: 585 native checks, all 39 candidates and 19,584 exactly
  replayed metric rows, with source/input guards, reporting and packing complete.
  Largest-cell diagnostic RMS improves 15.44%, but remains 568 times Lawson two's.
  The next proposed gate separates local coefficient identification from the
  sampled five-branch span; it introduces no automatic architecture/data expansion.

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
| R7 | Representation at fixed complete cost | Native bounded study reviewed; utility unestablished | [Job 2939 review](../results/native-response-preconditioner-r7-2939/README.md): 298 checks, 192 queries, 2,112 replayed metric rows and 336 timing bundles; sampled-span and identification limits require further bounded audits |
| P1 | Independent-parent/source-direction scaling | Native numerical study reviewed; wrapper packaging FAILED | [Job 2944 review](../results/native-response-preconditioner-scaling-2944/README.md): exact 388 native checks, all references/metrics/timing complete; more parents help, stronger classical control wins near-cost accuracy. [Export repair](../results/response-preconditioner-scaling-export-fix-20261016/README.md) has 404 local required passes, distinct from this native roster. No unchanged rerun needed |
| P1R | Retained-bank coefficient-penalty and action-energy audit | Native completed and independently reviewed at `4d0e049`; utility unestablished | [Job 2951 review](../results/native-response-preconditioner-regularization-2951/README.md): 585 native checks, all 39 candidates and 19,584 exactly replayed metric rows; reporting/packing complete. Largest-cell RMS improves 15.44%, still 568× Lawson two; no unchanged rerun needed |
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
acceptance or matched-accuracy speedup established. The separate bounded R7
protocol below addresses identification; its completed native review still
does not justify architecture expansion.

The [bounded repair protocol](RESPONSE_PRECONDITIONER_REPAIR_PROTOCOL_20261015.md)
keeps the original objective, ridge menu, selection and four development
parents. Its [retained-matrix replay](../results/response-preconditioner-repair-20261015/retained-fit-replay.json)
reduces augmented condition from 4.77e10 to 93.8 at the selected ridge without
changing original rank 25/40. Predictions change by at most 1.14e-13 RMS on
the retained queries. This is a numerical repair, not new scientific evidence.
The combined tiny-bank loss is action-dominated; endpoint signals are below
reference uncertainty. R7 assesses nonzero source-time coverage,
basis-versus-coefficient identification and complete repeated costs;
the reviewed evidence does not yet justify a learned encoder. Independent
per-context projection/ridge oracles and fixed-step Lawson action controls are implemented by the separate
R7 protocol below. Their native comparison is reviewed below; BCH coefficient
derivations remain deferred.
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

The [R7 identification design](RESPONSE_PRECONDITIONER_R7_DESIGN.md) now has a
separate executable [protocol](../tdn/analysis/response_preconditioner/r7_protocol.py)
(`tdn.response_preconditioner.r7.v1`) and native Raven launcher. The [R7 verification receipt](../results/response-preconditioner-r7-verification-20261015/README.md) records 298 required CPU passes and 53 overlapping Raven
compatibility passes. Native job 2939 completed all 298 required checks and
the bounded study at `43837c5`; independent review verified its exact roster,
provenance and numerical artifacts.
The workflow preserves the inspected two-training/one-validation/one-diagnostic
parent split on the 5-by-5 FD grid. Six absolute-time intervals per parent and
four calibration/four withheld sources per interval give 24 contexts and at
most 192 reference queries. Nonzero source times use the original anchor,
without restarting the path from its intermediate state.

R7 fits the deployable global readout on training calibration sources, selects
on withheld validation directions, then seals it before diagnostic truth.
Each context separately receives a source-shared unregularized five-branch
projection and both predeclared positive-ridge oracles. The induced local
penalty accounts for feature norm and the training-equivalent context mass;
independent oracles relax the global coefficient-map constraints. They remain
offline label-informed diagnostics. Calibration fit quality alone establishes
neither withheld-source accuracy nor a full-operator representation limit.

Sensitivity uses weighted reference-sized label perturbations, a leading
prediction-gain direction, fixed seeded directions, and actual independent
reference differences. The estimated uncertainty ball is not a certificate.
Complete diagnostic action costs compare the sealed model against ordered
1/2/4/8-panel and fixed 2/4/8-step Lawson controls in five randomized paired
rounds. Every source rebuilds its path and features; this measures a no-reuse
bundle cost, not amortized or vectorized throughput. Raw failures and planned
denominators remain portable evidence.

The native study is bounded to 4 CPUs, 8 GiB and 20 minutes, with a 900-second
study budget and no automatic successor. It follows mandatory verification
within the same source-bound Raven attempt. The first R7 output is descriptive
work-precision evidence: no superiority threshold or matched-tolerance speedup
gate has been defined. Full-operator auditing, independently derived BCH
controls, new endpoints/rollouts, encoders, nonlinear corrections, GPU work and
fresh confirmation remain deferred. No repeat of the unchanged implementation
smoke or unchanged R7 study is needed.

### Native R7 findings and next gates

The [complete review and term legend](../results/native-response-preconditioner-r7-2939/README.md)
retains the original archive, independent audit/replay scripts, JSON aggregates
and work–precision figures. All 2,112 physical metrics replay exactly from
arrays; 49 sensitivity maps and 245 scenarios also reproduce. Final separate
Slurm accounting was not uploaded, so successful native/Raven application
completion is distinct from independently confirmed scheduler termination.

On the 24 diagnostic withheld queries, pooled physical RMS is `6.466e-8`
for the fitted response, `4.433e-7` for its ordered-two base, `2.771e-8` for
ordered eight and `8.637e-10` for Lawson two. Fitted four-source bundle medians
range from 15.52–15.97 ms versus Lawson two's 15.93–16.41 ms. That small time
difference has overlapping paired observations and no predeclared utility
margin; Lawson two's much smaller errors make it the essential near-cost
control. No matched-tolerance superiority follows.

The selected positive-ridge objective is minimized with augmented condition
65.60, but original rank remains 25/40. Five static features on only two
training parents have rank at most two; three varying time features raise
context-feature rank to five, bounding the five-block action design at 25.
More source directions on those same parents cannot fix this bound. Ridge is
83.5% of the selected objective and strongly shrinks short-lag local fits;
regularization remains confounded with coefficient-map prediction.

The local unregularized projection improves pooled error to `3.937e-8`, but
has a resolved calibration residual on the longest interval and loses there
to ordered eight. This is sampled-span evidence, not a full-operator lower
bound. Approximately 94.9% of global diagnostic squared error occurs on that
longest interval. Reference-sized positive-ridge perturbations are too small
to explain the current errors; no projection prediction-sensitivity conclusion
is available. The fitted error is 0.000425% of full spatial feedback `U-G`,
yet 14.585% of the fitted remainder `U-P`; retain both denominators.

The review recommended a tiny full-operator audit and an independently
derived BCH/Magnus coefficient control for the exact backbone, including its
time-dependent terms. Those remain useful proposed controls. The subsequent
user-authorized process change begins with the independent-parent/source
scaling phase below; it does not require those controls to be the sole next
experiment. The fixed family and objective are retained so physical coverage
can be tested separately. None of these results promotes R8–R11,
endpoint/trajectory utility, GPU readiness or novelty.

### P1: data scaling before joint architectural changes

The [Phase 1 design, launch commands and legend](RESPONSE_PRECONDITIONER_SCALING_PHASE1.md)
freeze `tdn.response_preconditioner.scaling.v1`, a parallel research phase
distinct from R8. It acquires 32 training, eight validation and eight
diagnostic parents from four balanced periodic discrete morphology families.
The 5-by-5 same-grid FD target, six R7 intervals, five response branches,
40-weight FP64 readout, positive-ridge menu and original physical action
objective remain unchanged. These are new development parents, not fresh
confirmation or a continuum sharp-interface benchmark.

P1 changes morphology and mean/amplitude sampling from R7, so cross-R7 pooled
differences do not isolate training-bank size. The parent/source-count
comparison is within P1's fixed population. Its private 48-query verification
fixture uses production-seed prefixes; some development fields are therefore
inspected during implementation checks and have no untouched-confirmation
status.

Nested 8/16/32-parent cohorts crossed with 2/4/8 calibration directions give
nine fits. All cells share the frozen parent order and exact source prefixes;
their common labels are acquired once. Each cell fits its training calibration
prefixes and selects only on the validation-withheld bank. All nine readouts
seal before acquiring training-withheld or diagnostic truth. The complete
planned bank has 2,688 U/G query pairs and 16,128 withheld metric rows. Parents
are the independent units; phases, sources and cells remain paired descendants.
Learning curves and descriptive paired-parent comparisons describe one finite
nested cohort; there are no confidence intervals or universal learning-law
claims. Training-withheld parents inside/outside each fitted cohort are
reported separately and are not treated as a fixed paired evaluation bank.

The largest 32-parent/eight-direction cell alone receives complete repeated
timing against ordered 2/8 and Lawson 2/4/8. The predeclared first diagnostic
parent of each family supplies six contexts: 1,008 bundles and 4,032 source
calls, including distinct first and warmup observations. Accuracy is retained
for all nine fits on all withheld parents. Report reference acquisition,
per-cell label requirements, fitting and complete action work separately.
`outputs/study/costs.json` charges each uniquely acquired label once globally,
retains per-cell label demand, and records reference, design, fitting, accuracy
and timing durations as nested components rather than additions to wall time.
The native contract remains one 4-CPU, 8-GiB, 20-minute Raven allocation with
mandatory verification followed by the 900-second-budget study in the same
attempt. It creates no automatic successor and defines no utility pass.

Later phases separately propose optimal experimental design, defect-source
supervision, generalized physical priors, learned stage elimination and
bounded comparisons that separate these interventions. P1 does not implement
those changes or preauthorize 64/128-parent expansion. The completed native
review below informs the next protocol. The dated R7 review and all older
negative outcomes remain unchanged.

### Native P1 findings and proposed next gate

The [complete review, figures and term legend](../results/native-response-preconditioner-scaling-2944/README.md)
retain the exact upload and independent provenance, numeric and cost replay.
The numerical study completed in 448.786 seconds. A 1,232,048-byte compact
packing inventory exceeded the original 1 MiB writer budget afterward; export
recovery preserved all evidence and the native FAILED status. Scientific Raven
series were not published because packing failed first. Numerical before/after
source snapshots match; the final native aggregate source check was not reached,
and independent final Slurm accounting remains absent.

At eight directions, 8 → 32 parents improves diagnostic RMS by 17.31%.
At 32 parents, 2 → 8 directions improves it by only 0.981% for four times the
training labels. At equal 384-query demand, 32 × 2 beats 8 × 8 by 16.49%.
These are within-bank descriptive comparisons, with only eight diagnostic
parents and one nested ordering. All nine designs now have rank 40, but ridge
shrinkage leaves linear-smoother effective degrees of freedom only 6.60–6.80.
Every selection chooses the lowest offered ridge, `1e-10`; its penalty is
59.2–67.6% of the objective. Numerical optimality gaps are negligible compared
with data loss. This separates solving the objective from choosing its bias.

Largest-fit pooled diagnostic RMS is `1.69111e-7`, versus `1.30447e-6` for
ordered two, `8.15400e-8` for ordered eight and `2.51812e-10` for Lawson two.
Lawson two's median paired complete-bundle time is about 1.027 times the
learned time on the fixed timing subset, while its accuracy is substantially
better. No matched-tolerance advantage is established. About 94.6% of learned
diagnostic MSE occurs on the longest interval. Learned errors are well above
estimated reference uncertainty; exact Lawson 4/8 gains near that uncertainty
remain unresolved.

The review additionally clarifies aggregate metric semantics: historical
`centered_rms` centers the pooled error and includes variation between action
means. New read-only review fields separately report RMS mean-component error
and within-action spatial error; these contribute 5.41% and 94.59% of largest-fit
diagnostic MSE. Historical files and primary RMS comparisons are unchanged.
Future reporting should explicitly name both decompositions.

The proposed next gate is a frozen retained-bank regularization/physical-prior
audit, moving the P4 question ahead of broad P2 acquisition. Compare a bounded
positive-ridge path and explicitly defined physical penalties under unchanged
data, architecture and validation-only selection. Do not infer that smaller
ridge must generalize. Preserve sampled/full-operator and nonautonomous
BCH/Magnus controls as separate bounded questions before adding an encoder.
This review executed no new fitting or reference solves. Subsequent inspected-bank
work is development; a fresh frozen evaluation is required for broader claims.
No P2–P6 launch, 64/128-parent expansion, R8–R11 promotion or novelty claim follows.

### P1R: reviewed retained-bank regularization audit

After the P1 review, the user authorized implementation of the bounded
[P1R protocol](RESPONSE_PRECONDITIONER_REGULARIZATION_AUDIT.md),
`tdn.response_preconditioner.regularization.v1`. This narrows the next question
to penalty choice before adding physical parents or model capacity. The exact
reviewed job-2944 archive supplies all inputs. P1 and its failed wrapper status
remain historical evidence; its experiment is not repeated.

The audit retains three original training cells: 8 parents × 8 directions,
32 × 2 and 32 × 8. The first two have equal training-label demand. Each cell
has zero readout, six positive coefficient ridges from `1e-6` through `1e-18`,
five positive action-energy strengths and one unregularized projection:
13 candidates per cell, 39 total. Projection is always an offline diagnostic.
Validation selects independently within each positive-penalty family, including
zero, and all six choices seal before this audit reads diagnostic or
training-withheld arrays. All data were already inspected during P1 review;
access ordering does not turn them into fresh confirmation.

The action-energy control penalizes `alpha ||sqrt(W) A w||²`, with a factor
constructed only from training designs and prior `w0=0`. For full original
column rank it is exactly `w_OLS/(1+alpha)`, uniformly shrinking the fitted
correction toward the ordered-two backbone. This is a declared regularization
control, not a derived physical law or BCH/Magnus prior. Its strength is not
numerically interchangeable with coefficient ridge. Independently derived
analytic priors remain proposed work.

The generalized fitter preserves the full declared penalty under column
scaling. Numerical objective-gap, nonregression and prediction-parity checks
determine eligibility. Estimated-reference sensitivity is reported separately
through a thin label-to-validation prediction map; neither a large coefficient
norm nor an arbitrary sensitivity threshold changes selection. Reports use
explicit per-action mean-component and spatial-centered errors, while all
historical P1 metric definitions remain unchanged.

Evaluation compares six selected, three original and three projection readouts
with five saved classical controls: 17 methods × 1,152 original withheld
queries = 19,584 metric slots. Predictions use retained designs and equivalent
five-branch contraction; this does not measure fresh full-response inference.
There are no new references, physical labels, architecture changes or timing
bundles. Reused acquisition costs and new offline computation remain distinct.

The `response-preconditioner-regularization` Raven recipe runs the mandatory
verifier and audit in one 4-CPU, 8-GiB, 20-minute native allocation, with a
600-second study budget. Local checks use synthetic retained-array fixtures,
not new fits on the P1 bank. The [verification receipt](../results/response-preconditioner-regularization-verification-20261015/README.md)
records 585 exact mandatory CPU passes, eight additional Raven compatibility
passes and a 32-query/144-metric synthetic integration.

[Native job 2951](../results/native-response-preconditioner-regularization-2951/README.md)
completed at `4d0e049`: all 585 required checks, 39 candidates, six selections
and 19,584 metrics. All five native source/input guards passed, including the
post-study guard before publication; research reporting and packing completed.
Independent review reproduces all physical metrics exactly and checks saved
objectives, spectra and sensitivity without refitting. No unchanged rerun is
needed. Final independent scheduler accounting is absent; recorded native and
Raven terminal outcomes agree at COMPLETED/exit zero.

Coefficient selections are `1e-18`, `1e-14`, `1e-12` for 8 × 8, 32 × 2 and
32 × 8. Diagnostic RMS improves 0.32%, 13.91% and 15.44%, respectively.
The eight-parent fit's maximum error worsens 55.07%; the largest fit improves
all eight diagnostic parents but still has 1.754 times ordered-eight and
567.87 times Lawson-two RMS. All action-energy selections are `1e-6`, nearly
the unregularized projection. Larger-cell interior ridge choices outperform
their projections; removing regularization wholesale is unsupported.

Saved objective gaps and fixed-design empirical label-perturbation effects are
too small to explain the remaining error. Neither is a stability certificate.
The largest fit's remaining MSE is 97.60% on `[0,.08]`, 81.09% on periodic
transitions, 51.32% on constant sources and 97.32% spatial rather than mean
error; these overlapping partitions cannot be added. Source linearity does
not imply that a constant forcing has a spatially constant response.

The [proposed structural-attribution gate](../results/native-response-preconditioner-regularization-2951/proposal-review.md)
uses the existing 32 training parents and 192 contexts only: 192 local
five-coefficient fits on eight calibration sources, 768 withheld predictions,
and 192 separately labeled evaluation-only projections jointly over each
context's four withheld sources. It distinguishes coefficient recovery from
sampled branch expressiveness, with rank/sensitivity checks and signed terms
when a truncated subspace cannot justify orthogonal decomposition. Diagnostic
parents lack calibration sources and must not be fitted on their withheld
truth. This gate is proposed, not implemented or run. Its label-informed local
coefficients cannot establish deployable conditioning, fresh generalization,
novelty, full-operator capacity or speedup. Encoder, new branches, full-operator
references and independently derived analytic priors remain separate decisions.

### Source interfaces

| Interface | Purpose |
| --- | --- |
| [numerics.py](../tdn/analysis/response_preconditioner/numerics.py) | Ordered response, commutator actions and two-sided heat |
| [model.py](../tdn/analysis/response_preconditioner/model.py) | `FixedFeatureResponse`: forcing-linear action and full differentiable endpoint |
| [fitting.py](../tdn/analysis/response_preconditioner/fitting.py) | `fit_parent_balanced_ridge`: stable affine coefficient solve |
| [audit.py](../tdn/analysis/response_preconditioner/audit.py) | `audit_cubic_fd_panel`: independent whole-panel residual upper bound |
| [workflow.py](../tdn/analysis/response_preconditioner/workflow.py) | Frozen implementation-only protocol, verifier, fit and artifacts |
| [r7_protocol.py](../tdn/analysis/response_preconditioner/r7_protocol.py) | Bounded development cohort, access order, source families, reference/work caps and timing contract |
| [r7_workflow.py](../tdn/analysis/response_preconditioner/r7_workflow.py) | Mandatory R7 verifier and separately authorized native identification study |
| [r7_fitting.py](../tdn/analysis/response_preconditioner/r7_fitting.py) | Sampled projection, correctly induced local ridge and label-to-prediction sensitivity |
| [r7_references.py](../tdn/analysis/response_preconditioner/r7_references.py) | Original-anchor absolute-time references and fixed-work Lawson actions |
| [r7_actions.py](../tdn/analysis/response_preconditioner/r7_actions.py) | Exact five-branch extraction and complete paired action timing |
| [scaling_protocol.py](../tdn/analysis/response_preconditioner/scaling_protocol.py) | P1 frozen cohort, two data axes, reference/work caps, access ordering and exact verifier roster |
| [scaling_data.py](../tdn/analysis/response_preconditioner/scaling_data.py) | Independent periodic discrete parent families, physical identities and nested source prefixes |
| [scaling_analysis.py](../tdn/analysis/response_preconditioner/scaling_analysis.py) | Per-parent/family learning curves, paired comparisons and complete accuracy/cost aggregation |
| [scaling_workflow.py](../tdn/analysis/response_preconditioner/scaling_workflow.py) | Mandatory P1 verifier, nine validation-only selections, retained denominators and native study |
| [regularization_protocol.py](../tdn/analysis/response_preconditioner/regularization_protocol.py) | P1R frozen cells, penalty paths, validation selection, work counts and exact verifier roster |
| [regularization_evidence.py](../tdn/analysis/response_preconditioner/regularization_evidence.py) | Pinned P1 archive identity, bounded numeric reads, copied inputs and protected access after a new selection seal |
| [regularization_fitting.py](../tdn/analysis/response_preconditioner/regularization_fitting.py) | Generalized Tikhonov with original-coordinate objective/curvature checks and ineligible projection diagnostics |
| [regularization_sensitivity.py](../tdn/analysis/response_preconditioner/regularization_sensitivity.py) | Thin prediction gain and empirical reference-perturbation diagnostics without a scientific eligibility threshold |
| [regularization_analysis.py](../tdn/analysis/response_preconditioner/regularization_analysis.py) | Per-parent physical metrics, per-action mean/spatial decomposition, portable charts and compact Raven series |
| [regularization_workflow.py](../tdn/analysis/response_preconditioner/regularization_workflow.py) | Same-attempt verifier, all retained-data candidates, six sealed choices, full evaluation denominators and cost scopes |
| [response_preconditioner_regularization.py](../scripts/response_preconditioner_regularization.py) | Explicit CPU verifier and guarded native-study entry point |
| [response_preconditioner_regularization_native.py](../scripts/response_preconditioner_regularization_native.py), [response_preconditioner_regularization.sh](../scripts/response_preconditioner_regularization.sh) | Prepared source/input contract, native Raven recipe, allocated worker telemetry and portable export |

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
selects among declared choices; diagnostic data never fit deployed features,
weights or hyperparameters. R7's separately labelled offline local oracles may
fit diagnostic calibration directions under their frozen contract, without
altering the sealed global model or seeing diagnostic withheld targets.
Paired descendants are not independent parents. Reference refinement agreement
is an estimate, not a certificate.

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
| 2026-10-15 | Implement the bounded R7 identification contract and native verifier → study launcher | Reuse four inspected parents with frozen source withholding and nonzero start times; retain local/global objective distinctions, uncertainty scenarios and complete no-reuse costs. 298 required CPU checks and 53 overlapping Raven compatibility checks pass in the retained receipt; native science remains pending and no architecture or utility gate is promoted |
| 2026-10-16 | Accept native R7 job 2939 as completed bounded identification evidence | Independent provenance and exact numerical replay pass; Lawson two is much more accurate at nearly the same complete cost. Retain structural rank, sampled-span, ridge-bias and denominator findings; require full-operator and analytic-control audits before architecture expansion |
| 2026-10-16 | Begin user-authorized P1 data-scaling phase with the existing architecture and objective | Separate the value of independent physical parents from source directions using one frozen 32/8/8 development bank and nine nested cells; retain the prior full-operator/analytic controls as proposals, not the sole next step. Native verification and evidence review remain required; no R8–R11 promotion or automatic larger cohort |
| 2026-10-16 | Verify P1 implementation and portable Raven execution contract | [Receipt](../results/response-preconditioner-scaling-verification-20261016/README.md): 388 exact mandatory passes, eight additional compatibility passes and the 48-query/288-metric private fixture. Retain inspected-prefix disclosure, fresh prediction replay and shared acquisition costs; full native study remains pending |
| 2026-10-16 | Repair full-scale P1 evidence packing and export | [Receipt](../results/response-preconditioner-scaling-export-fix-20261016/README.md): 404 required and 51 overlapping compatibility checks pass. Stream large files, give inventories explicit bounded writes and retain orphan bundles. Retry export of the reported completed attempt; its native scientific outcome awaits uploaded evidence, with no resubmission required for packaging recovery |
| 2026-10-15 PDT (2026-10-16 UTC) | Accept native P1 job 2944 as completed numerical development evidence while preserving FAILED wrapper status | [Review](../results/native-response-preconditioner-scaling-2944/README.md): immutable provenance, all 388 tests, 2,688 references, 16,128 metrics and 1,008 bundles verified. Packaging alone failed; no unchanged rerun. More parents help modestly; strong ridge shrinkage and near-cost Lawson accuracy motivate a bounded prior/regularization audit before broader acquisition or architecture |
| 2026-10-15 PDT (2026-10-16 UTC) | Distinguish pooled centering from per-action mean/spatial decomposition | Historical aggregate `centered_rms` includes between-action mean variation. Preserve it and add explicitly named review fields; full RMS comparisons are unaffected. Any production metric change needs a versioned contract |
| 2026-10-15 PDT (2026-10-16 UTC) | Implement user-authorized P1R before broader acquisition or architecture changes | [Frozen audit](RESPONSE_PRECONDITIONER_REGULARIZATION_AUDIT.md): three retained P1 cells, 39 candidates and six validation-only choices; compare coefficient penalties with a training-action-energy uniform-shrinkage control. No new labels or inference timing; derived physical priors remain deferred. [Receipt](../results/response-preconditioner-regularization-verification-20261015/README.md): 585 exact CPU passes, eight additional Raven compatibility passes and all 144 synthetic metric slots. Native execution remains pending |
| 2026-10-15 PDT (2026-10-16 UTC) | Accept native P1R job 2951 and propose structural attribution before another model change | [Review](../results/native-response-preconditioner-regularization-2951/README.md): 585 native checks, all 39 candidates and 19,584 exactly replayed metrics; final source/input guards, publication and packing succeed. Largest-cell diagnostic RMS improves 15.44%, still 568× Lawson two; small-cell worst error worsens 55.07%. Proposed retained-training-context local fits and separately labeled joint withheld projections distinguish coefficient recovery from sampled branch span. No new fits during review, no unchanged rerun, no automatic acquisition/encoder/confirmation |

## Maintenance rule

For each follow-up, record the question, allowed information, target equation,
changed implementation, exact checks, outcome, unresolved limitations and next
eligible gate. Link the run's source-bound receipt and retain failed attempts.
Native verification is recorded only after actual native execution. A new
assumption or threshold must be declared before collecting the evidence used
to judge it.
