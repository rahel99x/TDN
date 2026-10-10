# Adjacent interaction study: initial research assessment

**2026-10-10 · CPU development evidence · separate from the active portfolio**

The strongest current result is a numerical diagnosis, not a new learned winner.
On the tested finite equations, missing higher-order spatial response and output
compression matter much more than additional quadrature nodes or separately
fitted LL/LH/HH gains. The next useful question is whether we can retain the
needed cubic response **cheaply enough to beat strong classical schedules**.
Neither learning nor complete solver acceleration has earned that claim yet.

The [handoff](ADJACENT_INTERACTION_STUDY_PROMPT.md) was implemented in an isolated
checkout on `research/adjacent-interactions`. The active main checkout, its
protocols and budget were unchanged. Primary measurements use source
`2d04dbf804ca9298078a7f7a4db8ff4953bc6c94`; the earlier `fd02415` run is preserved
and superseded where described below. No architecture is promoted into main.

Open the [67-panel HTML atlas](../results/adjacent-development/atlas/index.html),
[PDF atlas](../results/adjacent-development/atlas/adjacent-atlas.pdf),
[artifact guide](../results/adjacent-development/README.md),
[evidence map](ADJACENT_EVIDENCE.md), and [Fedora runbook](ADJACENT_RUNBOOK.md).

## What was implemented and actually executed

| Component | Implemented | Measured here | Remaining boundary |
|---|---|---|---|
| D01–D08 | Correction-space floors, feature collisions, grid/step/horizon cost, signed interactions, amplitude order, rollout stability, temporal reuse, favorable/stress regimes | All eight in the sealed 8×8 CPU smoke; selected grid/cost and signed-interaction cases use larger grids | These are development diagnostics, not population effects or GPU results |
| Controlled fields | All ten α values 0.1–1.0; canonical ridge and genuinely 2D spectra; phase, bandwidth, grid and heat-smoothing controls | 480 field-statistic records | Finite trigonometric fields are smooth; α is a generator label, not measured fractal dimension or solver accuracy |
| LL/LH/HH architecture | Fixed recombination, global fit, affine fit, neural gains, output-band/scalar/capacity controls | 26 instances: 13 families × two equation tracks; two updates, two training and two validation parents | Tiny smoke checks execution; meaningful optimization and native precision/cost need the bounded desktop program |
| Fresh evaluation | Frozen selection, parent identities, shared 1/2/4/8 schedules, paired complete-call timing, multiplicity and reference gates | 208 endpoint observations from **two independent fields**; six primary comparisons all NA | Repeated methods/schedules are not independent samples; no credible FNO superiority claim |
| Larger mechanism check | Same immutable development protocol, selected audit/D01/D05 | 16×16: 204 D01 records and 236 D05 records; 12 and 36 accepted refined references respectively | Six deterministic field constructions in D01, not twelve independent random fields |
| Additional prototypes | E01 pre-work selective cubic; E02 fitted signed cubic modulation | Both executed; neither advances | Protected 20% discretionary exploration ceiling remains explicit; no threshold retuning on held-out results |
| Execution/reporting | Seals, frozen inputs, recovery journals, cost ledger, Tower sidecars, separate atlas, main-job deferral | All 15 smoke units and three selected development units completed; 33 old/new execution and science seals verified | Native Slurm submission and CUDA execution are not available in this cloud environment |

CPU validation: 213 tests passed in the complete adjacent suite before the final
two plotting regressions; the final 16-test reporting suite then passed, including
those regressions and rendering. An earlier 69-test infrastructure pass is also
retained. All 92 mandatory CUDA identities match collection, but **zero CUDA tests
ran here**. Native GPU jobs execute the entire suite without skipped cases.

The complete validation record, raw logs and environment/source identities are
in [validation.json](../results/adjacent-development/validation.json) and
[`checks/`](../results/adjacent-development/checks/). The cloud used Python
3.12.14 and PyTorch 2.10.0+cpu; it does not substitute for Fedora Python 3.13.13
and the RTX 4090. Local numerical measurements used one Torch CPU thread.

## What these models are

The candidate remains a physics-based splitting-defect model. A physical split
provides a complete baseline endpoint. Fixed-background transport and signed
quadratic products supply spatial correction fields; a small state-dependent
conditioner can change their gains. The temporal dependence is in the physical
transport and quadrature, not merely an MLP receiving a timestamp.

Write the centered state as `v=L+H` and its quadratic defect as `Q(v)`, keeping
the **same background** in all evaluations. The implemented channels are

\[
Q_{LL}=Q(L),\qquad Q_{HH}=Q(H),\qquad
Q_{LH}=Q(L+H)-Q(L)-Q(H).
\]

Their unit-weight sum equals the original normalized two-node correction.
The shared implementation transports L and H together and forms the signed
products `L²`, `2LH`, `H²`; it does not create a third spatial dimension or
discard phase. All controls use the declared equation/product and matched
output support where attribution requires it. A scalar model has one effective
response; channels permit three responses identified by **input origin**.
Output-band gains instead distinguish **where the product lands**.

Global and affine fits test whether a neural conditioner is necessary. The
strong analytic quadratic+cubic control adds a genuinely different spatial
response, with additional work. FNO retains its local and spectral paths; Fourier
mode truncation does not mean it loses all high-frequency interaction information.
The neural comparator is a local adapted implementation, not a reproduction of
a published FNO benchmark.

The two tracks are distinct: FD/nodal dynamics and a finite dealiased Galerkin
equation. The code label `continuum` identifies the latter operator family;
same-grid time-refined measurements alone do **not** establish continuum
accuracy. D03 separately measures projected spatial refinement. The Galerkin
split also uses an approximate reaction substep; its near-constant baseline
error is not necessarily a learnable quadratic defect.

## Decisive observations

### 1. Higher-order shape has substantially more headroom than channel gains

At N=16, κ=0.004, r=3 and h=0.12, full-output four-node quadratic and full-output
quadratic+cubic give the following complete-state RMS errors. These are matched
output/node controls; the cubic method performs extra work.

| Field | FD: quadratic → quadratic+cubic | Galerkin: quadratic → quadratic+cubic |
|---|---:|---:|
| Low frequency | 3.274e−7 → 3.778e−8 | 3.620e−7 → 4.448e−8 |
| High pair | 5.483e−6 → 2.258e−7 | 7.284e−6 → 2.156e−7 |
| Mixed | 4.393e−6 → 2.257e−7 | 5.734e−6 → 2.383e−7 |
| Localized | 3.926e−6 → 3.865e−7 | 3.446e−6 → 4.673e−7 |
| Phase cancellation | 4.238e−6 → 2.003e−7 | 5.275e−6 → 1.706e−7 |

This is 7.37–33.79× lower RMS across these ten deterministic cases, **not** a
matched-cost or statistical advantage. In contrast, the bounded channel oracle
improves RMS over the bounded scalar oracle by at most **0.340%**. Output-band
oracles do better on the mixed fields. The Galerkin localized channel improvement
also worsens maximum error by 1.89% and the declared MSE + 0.1 maximum-squared
objective by 1.77%. A slightly smaller RMS is not an unqualified improvement.

Source: [N16 D01 rows](../results/adjacent-development/selected-development-v2/D01/diagnostic_rows.json),
particularly full-Q4/full-cubic pairs 00003/00006, 00020/00023, 00037/00040,
00054/00057, 00071/00074, 00088/00091, 00105/00108, 00122/00125,
00139/00142 and 00156/00159 (all prefixed `D01-`).

### 2. Output compression imposes a real floor, but removing it alone is unsafe

For desired correction `d = reference − baseline` and the exact implemented
orthogonal Fourier projector P, every correction c with c=Pc obeys

\[
\|d-c\|_2^2 = \|(I-P)d\|_2^2 + \|Pd-c\|_2^2.
\]

Thus `||(I−P)d||` is an RMS/L2 floor for that output support, regardless of
conditioner capacity. It is reference-informed, not a deployable estimator;
the maximum error of this projection is **not** a minimax lower bound.

For the N16 high-pair case, the floors are 4.188169e−6 FD and 8.185582e−6
Galerkin. The matched-cutoff quadratic+cubic oracles reach 4.188499e−6 and
8.185679e−6. Across high-pair, mixed, localized and cancellation cases they are
within 0.0012–0.218% of the floor. Full-output cubic corrections escape it.

However, on the N8 mixed case, removing compression from quadratic alone raises
RMS from 1.959e−6 to 2.000e−6 FD and 2.719e−6 to 3.073e−6 Galerkin. Filtering
can accidentally cancel other errors. The proposed follow-up therefore pairs
**interaction order and output retention**, rather than assuming that more
retained modes always help.

### 3. More nodes and a bigger conditioner are weak explanations here

In N16 D05, four-node versus 32-node quadratic quadrature differs by about
9.52e−15 FD and 2.47e−14 Galerkin, below the roughly 1.3e−12 reference
uncertainty, while endpoint defects remain much larger. Resolved amplitude
slopes are approximately 2 for the even defect, 3 for the odd defect, 4 after
quadratic correction and 5 after cubic correction. These finite-range fits
support the expansion; they are not an asymptotic theorem.

The cubic response has perpendicular fractions 0.951 FD and 0.974 Galerkin
relative to the scalar quadratic direction. Scalar rescaling cannot reproduce
most of that shape. D02's four smoke feature-pair comparisons have overlapping
acceptable gain intervals; they do not justify a larger conditioner. This is a
limited fixed-rule observation, not proof that the existing features suffice
for arbitrary nodes, states or parameter regimes.

Source: [N16 D05](../results/adjacent-development/selected-development-v2/D05/diagnostic_rows.json)
and [smoke D02](../results/adjacent-development/smoke-v2/D02/diagnostic_rows.json).

### 4. Classical cost controls remain decisive

D03's descriptive accuracy-qualified point minima are classical in all 24
sampled track/slice/tolerance cells: ETDRK4 in 21, DF in two, adaptive DOP853
in one. This selection is post hoc on the declared grid. Most methods have
three randomized paired warm observations; DOP853 has only a single warm
observation. The [derived frontier](../results/adjacent-development/d03-descriptive-frontier.json)
therefore provides no timing confidence interval or deployment policy.

D06 retains 32 paired autonomous/teacher-forced rollouts, mean and variance
behavior, finite-state and range checks, and error trajectories. Finite bounded
rollouts are not long-time stability proofs. D08 retains easy, near-constant,
small-step, finite-amplitude and phase/roughness cases; initialized neural paths
in those diagnostic rows are explicitly not trained competitor evidence.

### 5. The protected alternatives failed their current utility questions

| Prototype/workload | Measured v2 result | Decision |
|---|---|---|
| D07, 16 complete time queries, FD | Ours 12.06 ms; classical dense output 3.18 ms | BAD utility on this workload |
| D07, same query count, Galerkin | Ours 81.78 ms; classical dense output 12.22 ms | BAD utility on this workload |
| E01 selective cubic, Galerkin | 2.18× faster than always cubic; worst uncertainty-adjusted RMS regression 47.25× against a 1.05 limit | Stop current amplitude-only selector |
| E02 signed modulation, Galerkin | 2.89× faster; worst RMS regression 48.51× against a 1.10 limit; no median improvement over quadratic alone | Stop current one-coefficient cubic surrogate |

D07 beats direct quadrature at enough queries, but classical dense output is
both cheaper and more accurate here. Encoding refresh during autonomous rollout
removes the favorable reuse premise. Its kernel-only accuracy is not a complete
PDE speedup. These timings use one field per track and two paired rounds.

Both FD prototype aggregates remain NA because only three of four cases have
resolved error ratios. Other resolved FD cases already fail the accuracy gate;
refining the remaining easy reference cannot rescue these versions. No new
maximum-error failures occurred at the easy 2e−5 target, but that does not waive
the predeclared relative-regression gate. Four disjoint E02 fitting parents and
their reference/fitting costs are retained. No held-out threshold was retuned.

## Corrections found during review

1. **Oracle identifiability:** v1's unbounded multi-basis fit amplified near-zero
   directions with coefficients as large as 10¹²–10¹³. V2 retains the raw
   algebraic fit for audit but fits only singular directions with response RMS
   above five times reference uncertainty. Algebraic and reference-resolved
   rank are separate; coefficient identity is not inferred in null directions.
   This threshold is a conservative diagnostic choice, not a theorem about
   the exact analytic rank.
2. **Atlas inventories:** embedded summaries doubled evaluation observations.
   V2 plots the canonical 208 rows, separates validation, and selects canonical
   temporal/prototype measurements. Raw tables can retain both measurements and
   derived summaries; their row counts are not statistical sample sizes.
3. **Plot semantics:** heat-only fluctuation RMS was mislabeled as solver error.
   It now has separate amplitude/scaling panels. Positive losses use log axes;
   methods have distinct colors, Ours/Theirs/analytic labels and correct markers.
4. **Native CPU tests:** inherited Slurm IDs combined with a cleared execution
   mode could misclassify test paths as CARC. Only the CPU unit-test child
   environment is isolated. The allocated worker and every GPU test retain
   their actual scheduler identity and native checks.

V1 is preserved, not overwritten. Its oracle inference and affected charts are
superseded. V2 timings are primary; v1 partly overlapped engineering tests and
should be treated as integration evidence. Observed-range learning bands remain
distinct from statistical intervals; two-update smoke data are sparse, so gaps
are visible rather than filled with invented curves.

## Costs and remaining uncertainty

The [cost ledger](../results/adjacent-development/cost-ledger.json) records 139.79 s
of v1 per-stage CLI time, 133.20 s for v2 and 11.69 s for the selected N16 units:
**284.68 s of recorded CLI work**, including superseded work and reports.
Scientific timers are nested and must not be added again. This excludes some
unlogged engineering/ad hoc tests and is not total project cost, scheduler
billing, energy, or a cold-process deployment measurement. Monetary and energy
cost remain NA. Native collection includes actual `sacct` accounting and failed
or recovered allocations.

References refine actual numerical solutions, but refinement agreement is an
uncertainty estimate, not a certificate. The larger diagnostics are deterministic
examples. The learned pilot has two parents, two updates, one seed and FP32
effects small enough for several models to have identical measured endpoints.
FNO training adequacy remains unresolved. Equal data and update caps do not mean
equal training compute or inference cost. All six primary comparisons remain NA.
No newly completed systematic literature search or publication novelty claim
is made; the [prior-art boundaries](ADJACENT_EVIDENCE.md) identify remaining work.

## Portfolio decision

| Direction | Judgment | Most informative next condition |
|---|---|---|
| More quadrature nodes | Stop on these resolved workloads | Revisit only where measured quadrature error dominates the endpoint error |
| Larger LL/LH/HH conditioner | Low priority | First exhibit a bounded channel oracle that materially beats scalar and output-band controls in full errors, not just RMS |
| Higher-order correction + output retention | Continue as a numerical research question | Compare exact cubic, cheaper structured approximations and strongest classical schedules at matched accuracy and measured full cost |
| Selective cubic and signed modulation as implemented | Retire current variants | A changed mechanism must explain the demonstrated phase/direction failures before another pilot |
| Reusable temporal encoding | Retain negative result; change workload before expansion | Demonstrate a realistic query/refresh pattern with a margin over equally cached classical dense output |
| Learned channel superiority / FNO / GPU acceleration | Unresolved | Native checks, adequate bounded optimization, credible comparator selection and enough fresh independent parents |

The ready Fedora smoke is the next operational check, using its own checkout,
venv and queue dependencies. The full finite pilot exists for an explicit later
decision; successful smoke completion does not automatically launch it or justify
expanding the neural branch. Nothing here changes the running main portfolio.
