# Tangent Response: diagnosis and staged research proposals

**Draft, 13 October 2026.** This document analyzes the
[native FD review](../results/native-tangent-compression-20261013/README.md)
and proposes the next research treatments. Recorded results, mathematical
deductions and untested proposals are distinguished throughout. The original
handout, A PASS, B STOP, failed workers, cohorts and thresholds remain unchanged.
This draft does not implement or execute the proposed numerical treatments.

The recommended direction is to **separate a known, structured spatial-feedback
response from a learned compact remainder**, after correcting the compression
diagnostics and coefficient conditioning. This preserves TDN's central object:
the response to a numerical anchor's defect, conditioned on the actual state
and equation. It does not require predicting the entire solution with a larger
generic network. The structured branch is established mathematics; the useful
learned remainder, its accuracy/cost tradeoff and its generalization are research
hypotheses, not established novelty.

## 1. What the native result establishes

| Observation | Supported conclusion | Conclusion it does not support |
| --- | --- | --- |
| A's parent-weighted physical RMS improves from `5.56352e-7` to `3.87704e-10`, approximately 1,435-fold, across 24 primary queries/eight parents. | Spatial feedback along the declared anchor contains valuable correction information on this development bank. | Learned fit, speedup, broad-distribution success or fresh confirmation. |
| Every reusable rank-2/4/8 candidate stops; rank-eight parent oracle retains 53.78% endpoint gain. | These fitted candidates do not meet the declared B contract. | Every possible state-conditioned or structured response is insufficient. |
| Rank-eight oracle passes 0/2,304 genuinely withheld directions. Its 268 passing non-training-parent probes were locally fitted directions. | The observed oracle fraction is representation evidence with label access, not deployable parent generalization. | Successful independent-source transfer or a trained encoder result. |
| Raw coefficient fits retain condition ratios near `1e10`, with zero ridge. | The inverse problem is highly sensitive and deserves a stable controlled comparison. | Regularization alone will recover the missing capacity or conditioning information. |
| The coarse shared action basis discards 43.84% weighted training energy. | That fixed pooled output space has a large projection floor. | A uniform rank lower bound for each individual state/time operator. |
| Two parent oracles pool 12 distinct input/physics combinations each. | Their name is narrower than their actual context; exact grouping is needed. | Fixing those two groups will rescue all cases; other fixed-context parents also fail. |
| Compression science completes before worker 2891 fails publication; report 2893 then fails on a missing summary. | There are two infrastructure failures as well as a substantive negative scientific result. | A completely sealed native phase, or a scientific PASS after repairing reporting. |

No native neural training occurred in this stopped phase. The
[implemented learned encoder](../tdn/analysis/tangent_response/models.py)
already receives phase-aware current-state, reaction, diffusion and geometry
features through [build_features](../tdn/analysis/response/features.py).
The failed shared coefficient control does not receive that information.
The result therefore does not diagnose Adam, training duration or the capacity
of an optimized conditioned encoder.

Using B's own matched baselines, 80% retained gain requires

\[
E_L\le E_G-0.8(E_G-E_U)=1.11580545\times10^{-7}.
\]

The rank-eight parent oracle's `2.57360340e-7` is 2.3065 times that ceiling.
These errors are arithmetic means of per-query physical RMS with equal parent
weight, not pooled MSE. A 1,435-fold oracle improvement does not mean a compressed
model must achieve 80% of that multiplicative ratio.
The [derived evidence record](../results/tangent-response-proposals-20261013/evidence.json)
binds these scalar calculations to the original JSON hashes.

## 2. Separate three mechanisms of failure

Let `D=U_a-G_a`, `z` be a source, and `y=Dz` its true correction action. With a
physical-orthonormal output basis `Q`, define its physical adjoint `Q*` and
`P=QQ*`. If a decoder predicts `Q c_hat`, then

\[
\|y-Q\widehat c\|_{\rm phys}^2
=\|(I-P)y\|_{\rm phys}^2
+\|Q^*y-\widehat c\|_2^2.
\]

The first term is **output-space capacity**. The second is **coefficient
recovery**. This exact orthogonal decomposition requires the declared FD or
Galerkin physical inner product; ordinary nodal QR would change the Galerkin
question. It applies to each source action, not directly to a sum of endpoint
errors after different source times can cancel.

For the existing coarse shared action basis, the weighted aggregate training
projection floor is `sqrt(0.43841348)=0.662128`. This says that even perfect
coefficients cannot remove that aggregate error inside that frozen basis. It
does **not** by itself prove that the 90%-of-probes gate is mathematically
impossible: energy weighting and a pass fraction are different statistics.
Per-probe projection errors must accompany the aggregate floor. Conversely,
the fine-grid endpoint-weighted basis loses only about `7.5e-12` fitting energy
while its decoder remains unstable. That observation isolates a coefficient
problem that a larger output basis need not solve.

There is a third term before either of these: **missing conditioning
information**. The current shared map groups by target/grid/domain and predicts
from `z,s,h`, pooling different backgrounds, reaction and diffusion. If exactly
identical predictor inputs have different labels `y_i`, their best deterministic
least-squares prediction is the weighted mean `y_bar`; the irreducible empirical
error is `sum_i w_i ||y_i-y_bar||^2`. Compute this only on exact repeated input
identities, with uncertainty and actual supplied features recorded. Nearby
feature vectors are not an exact information-theoretic lower bound.

The short-interval identity explains the missing information. In FD,

\[
\frac{D(s+\tau,s)z}{\tau}\longrightarrow
H_a(s)z=-2r\,\operatorname{center}(a(s))\,z.
\]

The same constant source and times require different output fields for different
backgrounds. Scaling, longer optimization and a tighter SVD cannot reconstruct
a variable that the predictor never sees. Equally, adding state information
does not remove an insufficient output basis or an unstable regression.

## 3. Proposal P1: repair coefficient recovery with a bounded comparison

Use existing sealed **training** source/action vectors at rank eight, retaining
the legacy fit as a control. First inventory whether those vectors, basis tensors
and path identities are actually present. The uploaded compact review omits
binary replay payloads; its JSON statistics do not permit an honest refit here.
An absent vector yields `UNRESOLVED` with a missing-replay-data reason, not a
reconstructed synthetic label.

For sample-weighted source/time design `Z` and physical output coordinates `C`,
define training column scales `s_k=||sqrt(W) Z[:,k]||`, remove exact zero columns
with an explicit inventory, and set `Z_e=Z S^-1`. Solve

\[
\min_T\|\sqrt W(Z_eT-C)\|_F^2+\lambda\|T\|_F^2,
\quad T_{\rm physical}=S^{-1}T.
\]

Use an FP64 SVD or QR/SVD route, never normal equations. Apply `S^-1` exactly
once: fold it into physical coefficients/input factors, or retain transformed
features with the equilibrated coefficients. Applying both changes the map.
With SVD
singular values `sigma_i`, the ridge filter is
`sigma_i/(sigma_i^2+lambda)` and a declared relative cutoff removes unsupported
directions. Ridge penalizes these equilibrated coordinates, so it is not the
same prior as penalizing raw physical coefficients. Sample weighting and
physical field whitening are distinct transformations.

A proposed finite first menu is:

| Treatment | Coordinates | Relative cutoff | Ridge |
| --- | --- | ---: | ---: |
| T0 | Original raw design | `1e-10` | 0 |
| T1 | Training-equilibrated | `1e-6` | 0 |
| T2 | Training-equilibrated | `1e-4` | 0 |
| T3 | Training-equilibrated | `1e-6` | `1e-6*sigma_max^2` |

These values are a **draft**, not calibrated remedies. Freeze this menu before
evaluating it. Select by four deterministic parent-blocked inner folds of the
training cohort, rebuilding bases, weights and normalization inside each fit
fold, including a new T0 fit within each fold. The recorded full-cohort legacy
fit is an attribution comparator, not an inner-fold scorer. Check available
parent coverage for every geometry/context: the fine-grid training context has
only one parent, so four-fold within-context selection there is not possible.
Predeclare coarse-grid selection with a separately reported fine-grid transfer
panel, or retain unsupported fold/context combinations as unresolved. Never fit
a fine-grid basis on the held-out parent to fill that gap.
Use the full-training frozen-basis decomposition separately for attribution;
do not let an inner-fold held-out label enter its selection basis. Refit the
selected treatment once on all training parents before any validation comparison.
Validation cannot fit a factor or normalize a gain; already inspected diagnostic
parents remain reused development evidence.

Publish singular/filter spectra, effective source-design rank, coefficient and
factor norms, weighted projection error, decoder error, unseen-direction error,
FP32 application difference and original requested coverage. Lower coefficient
norm or better training residual is not success by itself. A treatment advances
only to a bounded representation diagnosis when it reduces withheld training
error without a hidden capacity/objective change and has finite, reproducible
FP32 application. Original B promotion criteria remain the scientific bar.
If all treatments fail, stop expanding the solver menu and inspect representation
and information instead. Train-only scaling/TSVD/ridge are numerical repairs,
not novelty claims.

## 4. Proposal P2: make the representation oracle answer a precise question

Replace the ambiguous parent-level oracle *in a new diagnostic* with two named
objects:

1. A **fixed-state/time action oracle**, keyed by actual input hash, target,
   geometry, diffusion/reaction, anchor/path version and parameters, and exact
   source/endpoint times. This measures a sampled fixed operator's representation.
2. A **fixed-state time-family oracle**, with the same exact state/physics/path
   identity and an explicitly tested causal time dictionary. This measures the
   extra burden of sharing factors over times.

Parent IDs remain the independent statistical clusters; exact context keys are
not new independent parents. Fit these oracles only on directions designated for
local fitting; withheld directions must never construct the output basis or
decoder. A per-state diagnostic with label access on validation/diagnostic parents
is nondeployable oracle evidence, even if some directions are withheld.

Count independent fit directions within each exact operator before interpreting
rank. In this menu, residual quadrature nodes and independent-source fractions
can be different times. A fixed-time group can have fewer directions than rank
eight. Source generalization is then underidentified; report effective span,
missing coverage and `UNRESOLVED`, rather than declaring the full operator high
rank or filling missing directions with fitted labels. More teacher acquisition
requires a separately frozen, bounded request after this inventory.

Also disentangle two choices that the endpoint-weighted method currently changes
together: output-basis weighting and decoder weighting. After selecting one
stable solver on training data, a four-fit diagnostic can cross action versus
endpoint-weighted bases with broad-action versus endpoint-only decoder weights,
at the same rank and label bank. Preserve zero weights explicitly. Broad reusable
operator fitting should give independent probe families positive weight; task-only
fitting can retain endpoint weights but receives a narrower task claim. Exact
endpoint error contains cross-source-time inner products, so squared quadrature
weights alone are an output-sensitive heuristic, not the full endpoint objective.

## 5. Proposal P3: a structured response plus a compact learned remainder

The failed rank requirement may be imposed on the wrong part of the operator.
Multiplication by a nonconstant field is generically full rank on an FD grid,
yet costs `O(M)` for `M` spatial degrees of freedom. Full rank does not imply
expensive inference. At zero diffusion the exact FD response along a known path is

\[
D(t,s)z=G_a(t,s)
\left[\exp\!\left(-2r\int_s^t v_a(q,x)dq\right)-1\right]z.
\]

This is a full-rank diagonal action, not a rank-eight operator. Evaluating the
path/integral is still charged. This pointwise exponential identity is **FD-only**:
projected Galerkin multiplication is not associative, and its zero-diffusion
operator exponential cannot be replaced by exponentiating nodal values.

For the initial FD proposal, use the same-anchor Duhamel relation and define
the first spatial interaction

\[
Q_1(t,s)=\int_s^tG_a(t,q)H_a(q)G_a(q,s)dq,
\quad H_a(q)z=-2rB(v_a(q),z).
\]

Then target

\[
\widetilde U_\theta(t,s)z
=G_a(t,s)z+Q_{1,m}(t,s)z+R_\theta(t,s)z.
\]

`Q_{1,m}` is a prospectively specified quadrature approximation: heat evolution,
target multiplication, heat evolution, with positive-duration subintervals and
the matched mean factors. This branch captures spatial mode mixing and full-field
mean coupling without a dense `M x M` matrix. The learned finite-rank branch now
targets `U-G-Q_{1,m}`, rather than all multiplication-like feedback. Both the
structured branch alone and the same-rank old learned response are required
controls. Any improvement already supplied by `Q1` belongs to the classical
component, not to the encoder.

With zero learned readout, the new initial response is `G+Q1` and its endpoint
uses that structured response. It is not the legacy `G` initialization. Retain
both initial baselines in all fitting/gain reports.

For an exactly integrated `Q1`, the time-ordered expansion's remaining terms
have at least two insertions of `H`. With `kappa>=0` and the exact same-path
matched propagator, heat is contractive and the scalar mean factors multiply
across every partition. Therefore the **product of G-segment norms** is bounded by
`M0(t,s)=exp(integral_s^t r*(1-2*mean(a(q))) dq)`. Finite numerical mean
quadrature has a separately controlled error. Inserting H multiplies this bound
by the product of its operator norms; ordered integration gives `K^n/n!`. With
`K=integral_s^t ||H(q)||_phys dq`, a conditional bound is

\[
\|U-G-Q_1\|\le M_0(t,s)(e^K-1-K).
\]

This follows from ordered integration bounds `K^n/n!`; it requires valid path
and operator-norm bounds throughout the interval. It is not a certificate from
sampled extrema, and becomes loose or large when interaction strength `K` is
large. Grid/time/source resolution and target assumptions remain explicit.
For FD, `||H|| <= 2r||v||_infinity`; Galerkin requires a bound on the physical
interpolant and its projected product. Finite empirical probes cannot establish
a uniform operator bound.

The useful hypothesis is **better residual compressibility**, not a theorem that
the remainder has rank eight. A local principal term can be inexpensive while
its remaining action still has many important directions. Test the weighted
training spectra and per-probe projection/decoder errors of the remainder using
the same labels and exact state/time grouping before proposing learning.

There is a useful counterexample: for short FD intervals the leading tail is
`(h-s)^2 H(s)^2/2`, again generally a full-rank diagonal action. At zero diffusion
the exact tail after Q1 is a pointwise exponential minus its constant and linear
terms. Its smaller magnitude does not establish better *relative* low rank.
Require uncertainty/output-resolution headroom in the remainder and measurable
improvement over Q1 alone. Passing full-D accuracy because the classical branch
already explains almost everything is not evidence for useful learned compression.

### A forward structural issue in the existing learned branch

The current learned gate is proportional to
`r*t_ref*variance/(variance+floor^2)`. With a fixed positive training floor and
input `u=m+epsilon*v`, bounded factors/head give an `O(epsilon^2)` correction
near a constant state, whereas `U-G` generally has a nonzero `O(epsilon)` term.
This is a mathematical limit concern, not an observed cause of B: compression
does not use this gate and the network did not train. Relevant amplitudes may
also be well above that small floor.

An exact `Q1` captures the `O(epsilon)` interaction, making a quadratic gate
compatible with the two-insertion remainder. But ordinary fixed quadrature leaves
an `O(epsilon)` first-interaction error. A quadratic learned tail cannot be assumed
to repair it. Independently control the structured branch's path/time quadrature
error, or retain an explicit first-order corrective channel. Merely replacing
variance by its square root does not solve arbitrary multiplication-like action
capacity, smoothness or factor degeneracy.

If the first-interaction branch is too expensive or its interaction-strength
tail too large, a separately frozen **FD potential-splitting response** is an
alternative, rather than more nested quadrature by default. Factoring out the
same scalar mean response, use positive-duration heat/potential/heat substeps
for `L+H(q)` with the known path potential `-2r*v(q,x)`. This carries local
feedback at all potential orders while approximating its noncommuting/time
transport. It is related to classical Strang/exponential tangent propagation,
so its local-only accuracy and complete cost must be reported as a classical
control. The learned tail would target `U-S_m`. Spatially constant anchor-path and
zero-reaction correction nulls hold mathematically with the common matched
scalar and target heat;
time quadrature, deployment arithmetic and noncommutation remain errors to
resolve. A nodal potential exponential does not implement the Galerkin target.
Repeated FFT heat applications can differ from the common G branch by
roundoff; a future implementation needs the common zero-interaction branch for
literal null equality. This alternative is not an automatic successor to a
failed Q1 diagnostic.

### Architecture and complete-cost implications

Keep the encoder's actual current state, phase-aware fields, target, physical
wave numbers, diffusion/reaction and geometry. Use signed causal coefficients,
exact zero at `t=s`, zero corrective response for spatially constant paths or
`r=0`, and translation covariance. Do not enforce a zero-diffusion response null: varying
reaction still has a nonzero response correction. Do not center the output:
for example, `mean(H v)=-2r <v,v>_phys` is generally nonzero, with the physical
product/mass convention retained in Galerkin.

Audit the **right time derivative** as well as `U_theta(s,s)=I`. At fixed grid
and a smooth path, normalized consistent Q1 quadrature gives
`Q1,m(s+tau,s)=tau*H(s)+O(tau^2)`. A tail with an explicit `tau^2` prefactor and
bounded coefficients preserves that instantaneous generator; the existing
`tau`-gated tail could change it. This is a separately declared structural
treatment, not a stiff/grid-uniform order guarantee. It also does not remove
the fixed-interval amplitude/quadrature issue above. Applying the resulting
operator to `z` must be linear: nonlinearities belong in its state-conditioned
coefficient encoder. Check additivity and signed source scaling.

The first diagnostic should use one fixed quadrature design and the existing
rank-eight ceiling; its refinement comparison checks error, not a node-count
hyperparameter sweep. A changed structure requires a **new B protocol** that
reports residual rank, number of structured nodes, transforms, path evaluations,
memory and full-call work. It cannot be described as an old rank-eight PASS:
the total correction now contains a full-rank structured branch.

For a single action, a direct `m`-node branch costs approximately
`O(m M log M)` heat/product work plus charged path/mean construction; the low-rank
tail costs `O(JM)` contractions in addition to its encoder. For endpoint decoding,
nesting `m` interaction nodes inside every residual-source node can multiply
path/FFT work. Freeze and count that layout. Sharing genuinely identical path
nodes or integrating the equivalent first correction through the forced mean
response can reduce duplicate work, but each resulting numerical method needs
its own error and timing check. No cost saving is assumed from asymptotics on
16x16/32x32 grids.

Static source-derived counts illustrate the alternatives, without estimating
runtime. For the current two-dimensional depth-two, modes-four encoder,
stored parameter count is `198*W^2+27*W`; the diagonal factor/head count adds
`2*J*(W+1)+J*T*(W+1)` for width `W`, rank `J` and time channels `T`.

| Architecture | Stored parameters | Interpretation |
| --- | ---: | --- |
| Current `W=16,J=4,T=4` | 51,528 | Existing minimal conditioned prototype. |
| Current `W=16,J=8,T=4` | 51,936 | More spatial factors; does not fix missing information or unstable labels. |
| Full core `W=16,J=4,T=4` | 52,344 | Cross-factor time coupling, without raising spatial rank. |
| Current `W=32,J=4,T=4` | 204,408 | Approximately four times stored capacity; utility unmeasured. |

These counts include stored spectral redundancy and are not independent degrees
of freedom. A naive Q1 implementation has `1+2*m` heat applications per source:
eight endpoint source nodes and four interaction nodes give 72 versus eight
for G alone. This is a construction count, not a minimum or measured latency.
The scalar mean factors combine to `g(t,s)` and source/output spectra can be
reused. Freeze the optimized numerical layout and validate its error/cost;
parameter count or fewer API calls cannot substitute for measured full inference.

## 6. Proposal P4: stabilize learned factors before increasing width

Retain rank four as the first learned prototype if a new representation gate
passes; rank eight remains a diagnostic ceiling. The implemented feature channels
already include the missing state/physics, so start by auditing their information
and output rather than doubling width/depth.

Any learned local/stencil treatment must predict coefficients from the state
and known physics and remain linear in the source. If explicit FP64 path/residual
work dominates after accurate reconstruction is established, a separately named
compact path/residual-summary encoder is a later option. Charge its labels and
encoder work, and isolate its path, residual, quadrature and arithmetic errors;
no future teacher or saved oracle path becomes a free deployment input. This
follow-up is conditional, rather than a simultaneous architecture sweep.

For a proposed full-core response
`R_theta=Alpha C_theta(t,s) Beta*`, physical QR gives
`Alpha=Q_A R_A`, `Beta=Q_B R_B` and

\[
R_\theta=Q_A\,(R_A C_\theta R_B^*)\,Q_B^*.
\]

This exposes factor conditioning and bounds the action through the small core.
General QR/rotations change a diagonal core into a full `J x J` core; the current
diagonal temporal head cannot be left unchanged while claiming equivalence.
A full-core treatment adds `J^2` coefficients per temporal channel and charged
physical orthogonalization, with rank-loss and differentiability checks. It is
a separate option, not a silent gauge repair.

A smaller first treatment is train-frozen factor scale balancing and Gram/core
regularization, preserving the current diagonal representation. Measure the
actual assembled operator's sensitivity rather than only raw parameter norms.
Regularize excessive gain relative to declared train scales; universal
contraction, positive-only coefficients or clipping would impose unsupported
physics. Zero only the final corrective readout at initialization; keep spatial
factors nonzero and verify head gradients followed by trunk gradients.

Consider new time features only if the fixed-state/time oracle succeeds and the
shared time-family oracle fails. Use one declared dimensionless-time treatment
with the actual target diffusion symbol and reaction scale; do not divide labels
by tiny per-example `h-s`, amplitude or gates. A scalar mean/energy descriptor
cannot replace phase-aware spatial input. New horizon-dependent factors also
change cache invalidation and encoder cost.

## 7. Training and metrics must follow the repaired research object

Train first on complete response actions, then on endpoints, preserving the
action-stage checkpoint. If a structured branch is used, the learning target is
its remainder, but promotion still evaluates the **complete predicted `U-G`**
against the true `U-G`. A tiny remainder denominator cannot substitute for the
existing full-correction gate.

Use parent-balanced losses with positive independent Fourier/random weights for
the reusable-operator claim, train-frozen physical scales and explicit
mean/centered components. Endpoint-only or residual-only optimization can be a
separate task treatment. Preserve the different supervision budgets when
comparing the matched-anchor residual FNO. A task-specialized defect decoder can
remain interesting without passing a general operator gate; it needs a separately
declared task success claim and cannot open the original reusable-operator route.

Audit objective scale before changing the optimizer. Current
[training normalization](../tdn/analysis/tangent_response/training.py) computes
the action scale from `target_action`, the full U action. Subtracting the same
G in prediction and target leaves the same physical correction error, but a
scale dominated by the mean/identity response can make a tiny D correction's
gradients weak. A proposed treatment freezes a parent-weighted training scale
from D, or from the explicitly named structured remainder, and reports the
unscaled physical errors and gradient norms alongside it. The latter requires
resolved remainder headroom. Do not divide by each example's tiny action or
silently compare differently normalized loss values. This is a forward
optimization concern, not a measured native training failure.

| Measurement | Definition and interpretation |
| --- | --- |
| Full correction relative action error | `RMS((D_hat-D)z) / RMS(Dz)`, only on predeclared resolved nontrivial actions. |
| Absolute action error / source gain error | The physical error itself, and error divided by source norm. Distinguishes small denominators from large incorrect output. |
| Projection / decoder error | The orthogonal physical decomposition above, per probe and weighted aggregate. |
| Mean / centered / sampled maximum | Preserve spatial coupling and tails; Galerkin uses its physical mass and explicitly sampled interpolant maximum. |
| Direction/phase alignment | A descriptive cosine on non-null actions, with norms and sign retained; not a replacement for physical accuracy. |
| Endpoint retained gain | `(E_G-E_L)/(E_G-E_U)` on the identical eligible requested cohort and parent weighting. |
| Precision attribution | FP64 fit/action, FP32 application, arithmetic bridge and final FP32 output rounding are separate. |
| Conditioning / coverage | Source singular/filter spectra, effective rank, factor Grams, fitted direction span, missing/null/unresolved slots and exact fit access. |
| Complete utility | Accuracy-qualified full-call latency, batch-one and declared batches, offline work, resident/transfer/fresh-process scopes, all attempts. |

Keep the old nontrivial rule
`max(1e-9,1e-8*source_RMS,10*action_uncertainty)`. Relative errors on tiny resolved
actions can be large, but the observed absolute errors of 0.910 and 505.917
cannot be removed by renaming the denominator. Report null actions' absolute
errors and retain their requested counts. Neither exclude old stress failures
after inspection nor soften the worst-error criterion to rescue a candidate.

The 43.84% energy loss is an MSE-style training diagnostic; the endpoint gate
uses arithmetic mean RMS; the action gate uses per-probe fractions and a worst
case. Publish all three rather than interchanging them. Retain every prospectively
primary endpoint even when its individual oracle gain is unfavorable; aggregate
gain eligibility cannot filter those horizons. Probe totals are paired
work, not independent population samples. Parent-level distributions and
cluster summaries must accompany pooled probe fractions.

Native C0 must still resolve output precision/headroom on a fresh screened fit
bank. An accurate FP64 correction can disappear in final FP32 assembly. No
regularized coefficient fit, CPU test or successful chart authorizes CUDA
training. Preserve validation-only deployment checkpoint selection, initialized,
final and best-training checkpoints, and the declared opportunity per seed.
Retain two/four/eight-step re-encoded rollouts and intermediate references only
after the appropriate learned gate; endpoint fit is not rollout stability.

## 8. A finite sequence with explicit stop decisions

| Step | Requested work | Decision before the next step |
| --- | --- | --- |
| D0: inventory | Existing train vector/seal availability; exact predictor/context identities; fit-direction counts; inherited uncertainty and omitted-data inventory. | Missing vectors or incompatible contracts stay unresolved; do not reacquire teachers automatically. |
| D1: stable fit | One rank and frozen label bank; the four declared solver treatments, selected inside train-only parent folds; physical projection/decoder attribution. | If instability is reduced but representation error remains large, move to representation diagnosis rather than more solver tuning. No B promotion from a training statistic. |
| D2: oracle/objective | Exact state/time and state/time-family oracles; at most the four basis/weight diagnostic fits with one selected solver. | If action coverage is underidentified, report unresolved. If exact-context projection fails, do not spend optimizer work fitting that same basis. |
| D3: structured tail | One declared first-interaction branch/refinement check; compare full-D versus remainder spectra, per-probe errors and complete work using the retained training labels. | If it does not improve useful remainder representation or costs too much, retain the negative result. No extra interaction orders or width sweep automatically. |
| B-new: prospective qualification | Freeze actual structure, work, sample/probe roles and thresholds before new evaluation; distinguish seen-state/new-direction, unseen-state/same-physics and unseen-physics panels. | Require literal source-bound eligible PASS for the newly declared reusable scope. Keep oracle headroom distinct from deployable generalization and task-only evidence. |
| C0/CAL/C | Fresh screened bank, actual native calibration, one conditioned response prototype plus the matched FNO under their declared work. | Preserve the existing fit and held-out learning criteria; stop on insufficient headroom, failed fit, unresolved execution or held-out failure. |
| D: utility | Frozen validation-selected checkpoints versus strongest qualifying classical controls at matched physical tolerances. | Claim speedup only for positive complete measured savings and requested coverage; otherwise report accuracy/representation alone. |

Freeze B-new's `qualification_role` as `ORACLE_HEADROOM` or
`DEPLOYABLE_PREDICTOR` before evaluation. The existing decision code can select
a passing nondeployable oracle, with that scope recorded. Headroom qualification
requires disjoint local fit and evaluation directions and can motivate bounded
learning, but cannot be called learned transfer across states. A deployable
predictor must receive no evaluated state's action labels at fit or inference.
Record these roles separately, even when their numerical thresholds match.

D3 must also inventory retained path snapshots and quadrature opportunities.
Reusing D labels does not imply that Q1 can be computed from already saved path
values. Charge new structured path/FFT/quadrature evaluations separately, and
preregister any teacher/action reacquisition rather than treating it as free.

D0–D3 are proposed diagnostics, not currently implemented launcher stage names.
Their detailed counts/calibration must be frozen before execution; this draft
does not alter the existing launch contract. Respect existing allocations,
source guards and project-contained files. New code uses fresh output roots,
never exact-resume bypasses or edits to the original failed workers.

All proposals are informed by inspected development results. Any new comparison
on those parents is reused development evidence. Before a confirmatory claim,
freeze the selected architecture, solver, loss, work, checkpoint policy and
success criteria, then acquire genuinely fresh uninspected parents under a
separate protocol. Passing a redesigned development gate cannot retrospectively
turn the old B STOP into a pass.

## 9. Preserve a precise contribution rather than claim novelty for repairs

The [existing related-work audit](TANGENT_RESPONSE_LITERATURE.md) already places
TDN beside variational propagation, defect correction, exponential/Krylov
methods, output-sensitive reduction and neural operators. Duhamel/Dyson terms,
splitting, POD, equilibration, TSVD, ridge and variable projection are established.
Local plus nonlocal neural operators and factorized operator learning also have
precedents. Their use can make the research better without making each ingredient
new.

The [targeted related-work review](../results/tangent-response-proposals-20261013/related-work.md)
checks eight close references with individually declared scope:

| Prior art | Relevance and verification limit |
| --- | --- |
| [Blanes et al., 2009](https://doi.org/10.1016/j.physrep.2008.11.001) | Dyson/time-ordering and interaction picture; accessible manuscript sections checked. |
| [Hansen, 1998](https://doi.org/10.1137/1.9780898719697) | Rank-deficient/regularized fitting; literal metadata and author tools page, not full-book audit. |
| [Li et al., ICLR 2021](https://arxiv.org/abs/2010.08895) | Fourier integral-kernel learning; author abstract checked, no benchmark claims inherited. |
| [Lu et al., 2021](https://doi.org/10.1038/s42256-021-00302-5) | DeepONet factor contraction; metadata and author implementation text checked read-only. |
| [Kovachki et al., 2023](https://jmlr.org/papers/v24/21-1524.html) | Neural operators, including low-rank representations; official abstract checked. |
| [Fan et al., 2019](https://doi.org/10.1016/j.jcp.2019.02.002) | BCR-Net/wavelet operator architecture; metadata only, precise local/compact theorems unverified. |
| [Niesen and Wright, 2012](https://doi.org/10.1145/2168773.2168781) | Adaptive Krylov phi-function actions; metadata and author abstract checked. |
| [Güttel, 2013](https://doi.org/10.1002/gamm.201310002) | Rational Krylov/pole selection; metadata and deposited abstract, not full-paper audit. |

The [verification receipt](../results/tangent-response-proposals-20261013/literature-verification.json)
retains source hashes, exact checked excerpts and access limitations. This
targeted comparison does not replace a systematic priority search or proof-level
comparison. A frozen-Jacobian exponential/phi control also does not automatically
solve the time-varying anchor response; its operator actions, shifted solves,
orthogonalization, restarts and time discretization must be accuracy-qualified
and costed if added as a comparator. Formal inverse heat evolution can magnify
high frequencies; prefer stable positive-time integral/action forms.

A defensible **claim candidate** is:

> On a disclosed family of the same-grid reaction–diffusion equation, explicitly
> separating known spatial feedback from a state/physics-conditioned learned
> response remainder improves independent action and anchor-defect correction
> at a measured residual rank, precision and complete cost, while preserving
> mean coupling and exposing the nonlinear remainder.

This claim needs positive evidence against the structured branch alone, the
original rank-matched conditioned response, a strong classical action/defect
method and a competent matched-anchor FNO. If the known branch supplies all the
gain, the result is a classical improvement with an unnecessary network. If
the learned tail only memorizes residual directions, narrow the claim rather
than call it an operator. If no complete cost advantage exists, a useful accuracy
or representation study is still possible, but solver-speed novelty is absent.

The strongest possible technical distinction would be a **derived, target-specific
statement about the sampled remainder's approximability and conditional error**,
paired with actual out-of-parent/physics and matched-tolerance evidence. No
rank bound, mesh-uniform convergence or universal stability theorem is supplied
by the present native review. A literature search can identify close predecessors;
it cannot guarantee priority or novelty.

## 10. Reporting and the expanded term legend

The reporting fix in `d02f5ef` already handles the actual sealed 539 MB source
with exact segments and retains incomplete attempts as failure evidence. Its
86 passing tests and default Raven checks are infrastructure verification.
Preserve compact source-traced tables for the proposed attribution and keep
native payloads, omissions, partial resource streams and measured observer cost
separate. No generic reader-limit increase, truncated denominator or fabricated
successful inventory is part of the proposal.

| Term | Meaning in these proposals |
| --- | --- |
| `D=U-G` | Spatial response correction missing from the matched mean response. It is the action object measured by B. |
| `Q`, `P`, physical adjoint `*` | Orthonormal spatial basis, its physical projection, and transpose/adjoint using the target's physical mass. |
| Projection floor | Error that no coefficient choice in that fixed output basis can remove; not a universal operator-rank claim. |
| Decoder | Mapping source/time/state features into output-factor coefficients. It is distinct from selecting the output basis. |
| Equilibration | Training-coordinate scaling of design columns; inverse scales must be applied at inference. |
| TSVD | Truncated SVD: discarding design directions below a declared singular-value cutoff. |
| Ridge / Tikhonov | Quadratic penalty on declared coefficient coordinates; changes bias and stability. |
| Effective source rank | Retained design dimension after filtering; distinct from spatial factor rank `J`. |
| Exact context | Actual input, equation, target, geometry, anchor/path and source/endpoint time identities defining an operator. |
| Identifiability | Whether the supplied fitting information can distinguish candidate maps; few fitted directions cannot determine unseen actions without assumptions. |
| `Q1`, first Born/Dyson term | One insertion of known spatial feedback between matched propagators; an established time-ordered approximation. |
| Structured / compact remainder | Cheap known full-rank actions / the remaining response targeted by finite-rank factors. Their costs and ranks are reported separately. |
| `K`, `M0` | Integrated spatial-feedback norm and matched-propagator growth bound; conditional quantities, not measured certificates. |
| Physical QR / full core | Orthonormalization in the declared spatial norm / a small matrix coupling all input and output factors. |
| Gauge | Different factor coordinates representing the same operator when every core transformation is also allowed. |
| Inner fold | A training-only selection split; it must not leak held-out parents into bases or normalization. |
| B-new | Proposed separately frozen qualification of the changed response object; not a rename of the old B outcome. |
| Supervision budget | Labels and their acquisition cost given to a method, separate from optimizer-update budget. |
| Novelty hypothesis | A specific potentially distinct contribution that requires literature comparison and positive evidence. |

The [native review legend](../results/native-tangent-compression-20261013/README.md#provenance-and-terms),
[mathematical analysis](TANGENT_RESPONSE_ANALYSIS.md) and original
[handout legend](TANGENT_RESPONSE_RESEARCH_HANDOUT.md#18-legend) define the
remaining equation, precision, workflow and metric terms.
