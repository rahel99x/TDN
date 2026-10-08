# Five-gate physical-interaction research program

This program tests one focused question:

> Can a small, physically structured nonlinear-interaction correction deliver a
> better accuracy–cost or data-efficiency tradeoff than a competitive neural
> operator, and in which regimes does that translate into practical solver savings?

Implementation and successful tests do not answer that question. The new full
cohort must run on the allocated Fedora desktop. Previously inspected roadmap
results are development evidence; they are not reused as untouched confirmation.
The launch instructions are in [FRONTIER_RUNBOOK.md](FRONTIER_RUNBOOK.md).

Two claims are kept separate. The **neural claim** concerns learning and
architecture against the declared trained controls. The **solver claim** concerns
complete runtime at the same requested output and accuracy against optimized
classical controls. The program can support the first while rejecting the second.
A hypothesis can also be falsified: if an untrained analytic or frozen correction
matches the learned branch, the learned contribution has not earned its cost.

## Executable questions and evidence

The Q labels below organize this program's questions. They are document labels;
canonical experiment rows carry the corresponding G1–G5 gate IDs, stable
experiment IDs and their complete individual checks. They do not relabel the
older agenda Q1–Q7 or roadmap M00–M23 experiments.

| Question | Gap or failure mode addressed | Executable comparison and retained evidence | Principal implementation / tests |
|---|---|---|---|
| Q1: Are speed comparisons for the same physical output? | Mixing final times, grids or spatial equations creates false wins. | Strict parent/grid/track/final-time/RMS-target/maximum-target keys; missing and infeasible arms remain explicit. | `measurement.paired_frontiers`; `test_frontier_measurement.py` |
| Q2: Is measured speed robust to invocation order and setup? | Cold startup and favorable ordering can masquerade as acceleration. | Separate first-call and warmup costs; randomized interleaved repeated whole-call timing; raw order and samples retained. | `measurement.measure_paired`; measurement tests |
| Q3: Are teachers independently refined? | A tolerance-only rerun can return the same trajectory and a false zero uncertainty. | Independent FP64 Lawson RK4 with actual three-level step refinement; actual spatial refinement for continuum estimates; rounding allowance and acceptance diagnostics. | `data`; `test_frontier_data.py`; G1 scalar/reference audit |
| Q4: Are we learning temporal error above a spatial floor? | Better time integration cannot correct a mismatched spatial generator. | Separate FD/nodal and refined dealiased spectral targets; common continuous fields over paired grids; spatial-difference and refinement records. | `screening.run`; `test_frontier_screening.py` |
| Q5: Is there classical work available to save? | A task solved accurately by one cheap classical step leaves little acceleration opportunity. | DF and ETDRK4 step grids; easy/headroom/infeasible/unresolved strata; all declared tolerances, without removing easy cases. | `screening.summarize_headroom`; screening tests |
| Q6: Does learning improve the compact physical correction? | A trained-looking result may actually be an initialization or classical-method result. | Rank one versus its same-seed frozen initialization and analytic quadratic/quadratic+cubic controls; validation selections and complete learning curves. | `models`, `neural.train/confirm`; model and neural tests |
| Q7: Does forming nonlinear interactions before compression help? | Input compression can erase high-frequency pairs that produce retained low modes. | Rank one versus a pure input-compression ablation with the same full-state physical core and matched training protocol. | `PhaseRankOne`, `rank1_postcompression`; model structural/gradient tests |
| Q8: Is the neural competitor a credible trained control? | Weak or untrained FNO controls cannot support a literature-level victory. | Small and larger FNO hybrids retain full-grid local paths; direct FNO has no physical core. Same physical inputs, labels and loss; validation tuning and initialization-improvement diagnostics remain visible. | `neural.train`, `confirmation_gate`; neural tests |
| Q9: Does the compact model use data or computation efficiently? | Equal update counts are not equal training cost; parameter count is not runtime. | Nested 8/32-field cohorts, three paired seeds, training examples, elapsed/update time, parameter counts and validation/confirmation metrics. | `expected_model_specs`, `nested_subset`, training ledger; protocol/neural tests |
| Q10: Does behavior survive different time schedules? | A single training step can hide ordering or rollout errors. | Unequal and reversed schedules, common final times, uniform-step controls and reference-audited prescribed intermediate times. | `neural.rollout/confirm`; data horizon registry and neural tests |
| Q11: Where do normal, favorable and stressful cases fail? | Reporting only aggregate mean error hides regime-dependent failures. | Frozen low/mixed/high-pair/rough/near-Nyquist/boundary-mean/reaction-stiff/diffusion-stiff strata, plus fresh coefficient-shift fields. | Protocol parent registry; confirmation rows; protocol/neural tests |
| Q12: Does resolution transfer preserve accuracy? | A fixed spectral cutoff and a spatial floor can limit apparent resolution generalization. | Same trained checkpoints at both confirmation grids; independent per-track references and mean/centered/spectral error decomposition. | `neural.endpoint_errors`; numerics/data/neural tests |
| Q13: Does the throughput crossover survive real batched accuracy? | Repeated copies and missing references inflate apparent large-grid value. | Distinct field batches, fixed seed/data selection, 64²/128²/256² workloads, accepted independent references, input-transfer cost, paired timing and workload memory. | `scaling`; `test_frontier_scaling.py` |
| Q14: Is there room to pay for a policy? | Estimator and rejected work may cost more than the solver work saved. | Gate deployment by development headroom and the frozen candidate's confirmation cost margin; otherwise record BLOCKED/NA without inventing deployment results. | `confirmation_gate`, `policy.run`; neural/policy tests |
| Q15: Does an operational policy deliver accurate, cheaper outputs? | A reference-selected frontier is not a deployable decision rule. | Calibration-only temporal and temporal-plus-spatial estimators; fresh policy fields; charged proposal, estimation, rejected work and classical fallback. Truth audits decisions afterward. | `policy.proposal/deploy/classical_fallback`; policy tests |
| Q16: What reliability is actually established? | Repeated queries are not independent calibration functions; empirical envelopes are not formal conditional-risk guarantees. | Field-cluster scores, acceptance/false-acceptance outcomes, unavailable conformal quantiles and explicit NA formal guarantees. | `policy._envelope`; statistics and policy tests |
| Q17: Do savings repay offline work? | All-classical routing or timing noise can manufacture finite amortization. | Preparation/training/calibration/loading costs, matched accuracy, accepted trained-neural queries and a positive field-cluster saving lower bound. | `supported_amortization`, policy summaries; measurement/policy tests |
| Q18: Can we reconstruct failures and every plotted conclusion? | Summary scores can conceal dropped arms, partial stages or changed artifacts. | Sealed complete ledgers, exact checkpoint/source hashes, missing-stage reports, every-check maps, chart-source data and separate numerical/scheduler accounting. | `engine`, `report`, Fedora workflow; engine/reporting/CLI/workflow tests |

The stress cases are specified before measuring outcomes. They probe plausible
weaknesses; they do not prove universal worst-case behavior or establish a flaw in
every neural-operator architecture. FNO's full-grid local path is retained
specifically because it can carry information and nonlinear interactions beyond
a truncated Fourier branch.

## Equations and spatial target

The field is a scalar periodic reaction–diffusion state on a two-dimensional
unit-length domain unless its declared geometry says otherwise:

\[
\partial_t u=\kappa\Delta u+r(u-u^2),\qquad u(0,x)=u_0(x).
\]

The `discrete` track uses the periodic finite-difference diffusion symbol

\[
\lambda_{\mathrm{FD}}(k)
=-4\kappa\sum_{j=1}^{2}\Delta x_j^{-2}
\sin^2\!\left(\frac{\pi k_j}{N_j}\right),
\]

and nodal multiplication for the reaction. The `continuum` model track uses

\[
\lambda_{\mathrm{S}}(k)=-\kappa\sum_j(2\pi k_j/L_j)^2,
\qquad B_N(u)=r\bigl(u-P_N(u^2)\bigr),
\]

with dealiased products. A model on this track still advances a finite spectral
Galerkin approximation. Its comparison teacher uses larger spatial grids and
projection; the name is not a claim that a finite model or teacher is continuum
truth.

Write \(E_h=\exp(hA_N)\). The common diffusion-first core is

\[
S_h(u)=E_{h/2}\,R_h\!\left(E_{h/2}u\right).
\]

For FD/nodal reaction, \(R_h\) is the exact pointwise logistic flow. For the
Galerkin target, reaction evolves through projected RK4, with a declared minimum
subdivision and additional subdivision when \(rh\) is large. Applying exact nodal
logistic reaction would advance a different spatial equation. The Galerkin
reaction approximation and its work are included for every hybrid family; it is
not an unconditionally stable exact flow.

The error decomposition motivating G2 is

\[
\|\hat u_N-P_Nu\|
\le\underbrace{\|\hat u_N-u_N\|}_{\text{time/model error}}
+\underbrace{\|u_N-P_Nu\|}_{\text{spatial error}}.
\]

Teacher uncertainty adds another observational limitation. Improving the first
term cannot remove the second. Discrete and continuum-reference claims therefore
remain in separate panels and comparison cells.

## Compact phase-aware architecture

Let \(c=\langle u\rangle\), \(v=u-c\), and let the homogeneous logistic response
and its initial-state derivative be

\[
C(t;c)=\frac{c}{c+(1-c)e^{-rt}},\qquad
q(t;c)=\partial_cC(t;c)
=\frac{e^{-rt}}{[c+(1-c)e^{-rt}]^2}.
\]

For the declared product \(\mathcal P(a,b)\), define

\[
\mathcal B_s(v)=E_{h-s}\mathcal P(E_sv,E_sv),\qquad
\mathcal B_{\mathrm{split}}(v)
=E_{h/2}\mathcal P(E_{h/2}v,E_{h/2}v).
\]

The compact correction uses the quadratic finite-step defect form

\[
D_\theta(u,h)=P_K\!\left[
-rq(h;c)h\sum_{i=1}^{2}w_i(u,h)q(s_i;c)
\bigl(\mathcal B_{s_i}(v)-\mathcal B_{\mathrm{split}}(v)\bigr)
\right],\qquad \hat S_h=S_h+D_\theta.
\]

Its symmetric node pair is \(s_1=ah,\ s_2=(1-a)h\), where
\(a=.05+.4\,\sigma(\theta_a)\). The common weight is learned from a bounded
amplitude and a small conditioner of

\[
\left[c,\ \log(1+rh),\ \log(1+h\mu)\right],\qquad
\mu=\frac{-\langle v,A_Nv\rangle}{\langle v^2\rangle},
\]

with a safe zero-variance branch. The 3→8→1 conditioner plus node/amplitude scalars
stores **43 real parameters**. “Rank one” means one symmetric physical node pair;
it is not a Tucker rank, matrix rank or a one-term arithmetic cost claim. The
expanded separable term bound is separately reported.

Fourier phase is preserved through propagation and physical multiplication.
For example, two retained complex amplitudes can contribute through
\(\hat v_p\hat v_q\) at \(p+q\), including high-frequency pairs whose sum is low.
Projection \(P_K\) happens after the interaction is formed. The input-compression
ablation replaces \(v\) by the centered low-pass input in the correction branch;
both arms retain the same full-state physical core. It has no hidden correction
bypass that restores the removed input information.

Subtracting the split contribution *inside* every node term preserves important
nulls independent of learned weights: the correction vanishes for constant
fields, zero reaction, zero diffusion and zero step. At fixed mean and active-rate
features it is quadratic in centered amplitude. These constraints are checked
with nonzero weights, gradients, phase/translation tests and CPU/CUDA parity.
They do not imply invariant-interval preservation, global stability or full
nonlinear exactness. Outputs are not clipped to hide failures.

The untrained `analytic_quad` control uses fixed quadrature of the quadratic
Volterra defect; `analytic_quad_cubic` adds a causal nested cubic correction.
Their quadrature is an approximation to variational coefficients. Correct
amplitude order is different from a proof of a complete nonlinear method's time
order, and neither explicit RK behavior nor a successful sampled residual test
establishes A-stability. These controls are present to distinguish learning from
known physical computation.

## Honest controls and training loss

| Family | Physical core | Learned correction / role |
|---|---|---|
| `rank1` | Track-matched diffusion-first | Proposed 43-parameter symmetric interaction correction |
| `rank1_postcompression` | Same full-state core | Input-compression ablation of the same correction |
| `rank1_frozen` | Same core | Same-seed untrained rank-one initialization |
| `analytic_quad` | Same core | Fixed quadratic finite-step defect quadrature |
| `analytic_quad_cubic` | Same core | Fixed quadratic plus causal cubic quadrature |
| `fno_small` | Same core | Two blocks, width 16, four modes in the full profile |
| `fno_standard` | Same core | Four blocks, width 32, eight modes in the full profile |
| `direct_fno` | No PDE core | Four-block/width-32/eight-mode residual neural operator |
| `df` | Diffusion-first | Classical split control, including valid cross-step diffusion fusion |
| `etdrk4` | ETDRK4 | Independent evaluated classical family with stable phi functions and coefficient reuse |

Each FNO block retains both its spectral mixer and a full-grid 1×1 local path,
followed by GELU. Inputs include the state, centered state, normalized time,
reaction/diffusion coefficients, physical lengths through their coefficient
scaling, and grid-spacing information. Hybrid corrections use an explicit
\((h/t_{\mathrm{ref}})^3\) multiplier. Direct FNO uses
\(u+(h/t_{\mathrm{ref}})F_\theta\). The temporal multipliers enforce a declared
residual scaling and exact zero-time identity; they do not prove trained-model
accuracy at large or unseen times.

The rank branch has stronger physical restrictions and a much smaller hypothesis
class. The neural controls receive the same physical problem information and
accepted labels, but the models are not parameter-matched or compute-matched.
Their measured cost, parameter count, learning curves and examples seen are
reported separately. These are project-adapted FNO implementations, **not an
official reproduction of the published FNO benchmark or its best tuned model**.

For a training example with teacher \(y\), common DF baseline \(S_h(u)\), and
reference RMS uncertainty \(\eta\), training uses

\[
\mathcal L=
\frac{\operatorname{mean}(\hat y-y)^2+0.1\max_x(\hat y-y)^2}
{\max\{\operatorname{mean}(S_h(u)-y)^2,\eta^2,10^{-12}\}}.
\]

The same scale is used for every architecture on that example. The denominator
reduces domination by large absolute-error examples while preventing division
by a near-zero base defect. It does not remove the need to report unnormalized
RMS and maximum error. Validation averages within independent fields first, then
across fields. Checkpoint and learning-rate selection use validation only;
initialization is always an eligible checkpoint. Final-update, selected-update,
validation and failure records remain distinct.

The full profile contains 20 tuning trials and 80 final model records. The 80
records comprise 60 trained-model instances, 12 paired frozen-rank instances and
8 deterministic track/control instances. Two learning rates are tuned for each
trainable family and track for 40 updates; final trainable instances receive 200
updates. The declared total is **12,800 optimizer updates**, or **51,200 sampled
examples** at batch size four, before any failure or budget interruption. This is
a workload ceiling, not evidence that every update finished.

A five-percent validation gain over initialization is an adequacy diagnostic for
a local FNO control. It does not establish convergence, competitive published
hyperparameter optimization or external state-of-the-art performance. An
initialization-selected or otherwise inadequate control remains visible and
limits the claim; it is never discarded to make TDN look better.

## Cohorts, schedules and stress matrix

There are **116 distinct continuous fields** in the full declaration. Their
coefficients, phases, means and variances are fixed before execution. Grid samples
of the same field are paired observations, not new independent functions.

| Split | Independent fields | Use |
|---|---:|---|
| Development | 12 | Spatial/headroom screening only |
| Train | 32 | Nested 8- and 32-field training cohorts |
| Validation | 16 | Learning-rate and checkpoint selection |
| Calibration | 16 | Empirical deployment envelope fitting only |
| Confirmation | 24 | Frozen-checkpoint evaluation; 16 declared-physics and 8 coefficient-shift fields |
| Scaling | 4 | Distinct batch members for workload scaling |
| Policy | 12 | Fresh deployment outcomes after frozen selection/calibration |

Smoke and development profiles have separate field-seed ranges. They contain 14
and 36 fields respectively and cannot become fresh full evidence by relabeling a
run. Model seeds repeat across protocols for reproducibility; the continuous-field
cohorts do not. Within the full training design, three seeds and two data fractions
are retained rather than selecting the best seed after confirmation.

The full confirmation schedule bank has three paths to \(T=.12\):
\([.02,.04,.06]\), its reverse, and four \(.03\) steps; and two paths to \(T=.24\):
\([.04,.08,.12]\) and its reverse. Uniform classical schedules also enter the
fixed-horizon comparison: DF, ETDRK4 and both analytic corrections each receive
uniform 1/2/4/8-step endpoint-only trials at every final time. This prevents
unnecessary classical subdivisions from creating an apparent neural speed win.
Those extra trials request only an endpoint; all positive interior times in the
primary shared schedule bank have their own independently accepted teachers.
Finite sampled times do not
establish continuous-time accuracy between them.

| Dimension | Full declaration | Interpretation |
|---|---|---|
| Spatial targets | FD/nodal; dealiased continuum estimate | Never pooled as the same equation |
| Confirmation grids | 32² and 64² | Same fields/physical units, frozen trained checkpoints |
| Screen grids | 32², 64², 128² | Paired spatial refinement and teacher uncertainty |
| Training fractions | 8 and 32 independent fields | Nested balanced field cohorts |
| Seeds | Three paired seeds | Repeated training, not independent physical fields |
| Error targets | 2×10⁻⁴, 2×10⁻⁵, 2×10⁻⁶ | Joint RMS and maximum requirements |
| Normal/favorable | Low and mixed frequencies | Easy cases remain in the denominator |
| Spectral stress | High-pair, rough, near-Nyquist | Tests retained interactions and unresolved fine scales |
| Physical stress | Boundary mean, reaction stiff, diffusion stiff | Tests nonlinear amplitude and time-scale sensitivity |
| Coefficient shift | Last eight full confirmation fields use genuinely unseen coefficient pairs | Numerical coefficients change, not merely their label |
| Scaling | 64², 128², 256²; batches 1 and 4 | Real distinct fields, same equation within each batch |
| Policy | Temporal; temporal-plus-spatial | Fresh field outcomes and charged verification |

The 512²/1024² sweep is deliberately deferred until the bounded accuracy-backed
range warrants it. Larger allocations, more seeds, unbounded training or another
mechanism combinatorial campaign are not triggered automatically.

## Reference uncertainty, metrics and statistics

The independent teacher uses an integrating-factor/Lawson RK4 implementation
separate from the evaluated split and ETDRK4 model implementations. Temporal
acceptance compares actual \(m,2m,4m\) substep trajectories, requires decreasing
differences or a declared rounding regime, and retains a nonzero rounding
allowance. Continuum estimates additionally compare projected results from
actual spatially refined grids and include both temporal estimates in the
uncertainty accounting. Unresolved references remain unresolved; they cannot
silently become zero-error labels.

If \(e\) is observed error and \(\eta\) the accepted reference estimate, the
reported conservative score uses \(e+\eta\) for each norm. This is a useful
empirical margin, **not a certified mathematical upper bound**. Joint eligibility
requires accepted references, finite positive complete cost, finite predictions,
and both RMS and maximum margins below their respective thresholds.

Alongside endpoint RMS and maximum error, confirmation records mean error,
centered-field RMS, spectral low/high components, trajectory information,
feasibility, model/seed/data identity, parameters, raw timing and uncertainty.
The identities

\[
\mathrm{RMS}^2=e_{\mathrm{mean}}^2+\mathrm{RMS}_{\mathrm{centered}}^2,
\qquad
\mathrm{RMS}^2=\mathrm{RMS}_{\mathrm{low}}^2+\mathrm{RMS}_{\mathrm{high}}^2
\]

make it possible to identify whether gains concern the mean, resolved structure
or high frequencies. No clipping conceals invalid values.

A schedule frontier selects the cheapest accurate declared schedule using the
reference after the fact. It is labeled **post hoc and reference-informed**. It
is useful for assessing potential, but is not itself an operational controller.
Comparisons always match final time, target, grid and independent field; missing
and infeasible methods remain visible. Practical speed uses the declared 1.2×
threshold. Field-clustered descriptive uncertainty preserves pairing across
repeated schedules, grids and training seeds. It does not turn a large endpoint
row count into an equally large sample of independent functions, nor correct all
multiple comparisons automatically.

## Cost-gated deployment and reliability limits

The architecture/seed/data instance used by deployment is fixed before
confirmation. Confirmation can decide whether there is enough potential to test
it; it cannot select a new best model. The policy stage additionally requires
resolved development headroom in the relevant target track/grid. If these
necessary conditions fail, the stage records BLOCKED with explicit unmeasured
checks and performs no calibration or deployment computation.

For an enabled track, the temporal estimator compares one proposed step with two
half steps. The spatial variant also compares a doubled-grid prediction sampled
from the same declared continuous initial field. Calibration fits empirical
multipliers from **calibration fields only**, using accepted teacher uncertainty.
Fresh policy truth is read only after deployment decisions. Classical fallback
uses bounded ETDRK4 temporal refinement; its convergence flag is not a continuum
spatial-accuracy guarantee, so fallback accuracy is also audited.

The cost requirement follows directly from

\[
C_{\mathrm{deploy}}=C_{\mathrm{proposal}}+C_{\mathrm{estimator}}
+(1-p_{\mathrm{accept}})C_{\mathrm{classical}},
\qquad
C_{\mathrm{proposal}}+C_{\mathrm{estimator}}<p_{\mathrm{accept}}C_{\mathrm{classical}}.
\]

Actual reporting uses complete repeated calls, with estimator/rejection/fallback
components tied to one recorded representative warm sample. Component totals
must not be added to the complete-call total a second time. Cold invocation,
warmup, input transfer, setup and offline calibration remain distinguishable.

The 16 calibration fields cannot provide a finite standard 1% conformal
quantile: \(\lceil(n+1)(1-\alpha)\rceil=17>n\). More importantly, field
heterogeneity and coefficient shift do not establish exchangeability or a
conditional-risk guarantee among accepted calls. Formal claims remain NA. The
program reports empirical acceptance, possible false acceptance, fresh accuracy,
field-cluster uncertainty and the scope of every risk statistic.

Amortization requires matched deployed/classical accuracy, a positive saving
lower bound and actual accepted trained-neural queries. A policy that always
falls back to the same classical solver does not establish neural amortization.
Offline preparation, training, calibration and loading are charged conservatively;
no electricity, tariff or billing cost is invented.

## Graphs, audit trail and reading negative outcomes

The report produces a multipage PDF, individual high-resolution panels, an
HTML graph browser, complete compressed chart data and an every-check lookup.
The analytical views include:

- Training/validation loss by update, time and examples seen; gradients and effective trial settings remain in the complete training ledger.
- Model parameter counts, checkpoint/initialization outcomes and data-efficiency observations.
- RMS/maximum work–precision, fixed-target paired speed distributions and regime feasibility.
- Mean, centered and high-frequency error across resolution, with reference uncertainty retained.
- Accuracy-backed scaling latency, batch throughput and workload-level peak GPU memory.
- Proposal/estimator/fallback costs, acceptance/false acceptance, reliability limitations and amortization.
- Stage numerical time, frozen budgets and terminal scheduler-allocation time; complete experiment/check verdict maps.

Every plotted source is hashed. The accounting snapshot consumed by the report
is copied into its sealed science directory; later collection can refresh the
external scheduler snapshot without rewriting the plot's evidence. Allocation
and task-step rows are not summed. GPU memory belongs to the measured paired
workload unless explicitly isolated; host/shared RAM is not GPU VRAM.

GOOD/BAD/NA and 1–100 scores describe attainment of declared checks. They are not
model-quality probabilities or theorem proofs. A mathematically correct and
accurate method can be BAD because it costs too much. NA can mean an unresolved
reference, a missing stage, an unexecuted deployment gate or an unsupported formal
claim. Those meanings remain in the raw checks rather than being collapsed into
a single favorable headline.

Structural failures block successor science; scientific negatives remain valid
outcomes. Numerical failure, OOM, interruption and incomplete stages preserve
partial evidence. The final report runs after any outcome and does not replace
missing measurements with successful defaults. Full work requires real Fedora
Slurm allocations, capped at 45 minutes per stage with the existing dedicated
VRAM limits. See the [runbook](FRONTIER_RUNBOOK.md) for the exact resource table,
commands, artifacts and collection procedure.
