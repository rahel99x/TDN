# Full native Advance campaign: independent scientific review

Run: `fedora-advance-full-20261011T001715`, jobs **708–756**. Recorded source: `a7646702ff1061d78a722153ef2548d9d831de0d`. This document reviews the completed, frozen experiment. It does not change its models, thresholds, selection decisions or observations. Analyses motivated by inspecting confirmation are explicitly exploratory.

## Research judgment

**The two-coefficient model establishes a modest calibration benefit over its own fixed parent at the longer horizon. It does not establish an overall solver advantage.** At final time 0.24, fitting reduces geometric-mean RMS error by **13.79% on the discrete track and 12.53% on the continuum track**. The corresponding shorter-time comparisons do not meet the declared success criterion. More importantly, the four-node analytic interaction rule is **2.5–3.1 times more accurate** than the fitted model across all four primary track/time comparisons, and **none of the 20 accuracy-qualified cost claims succeeds**.

The useful next question is therefore whether we can obtain the finite-step accuracy of the strongest analytic interaction rule with less work. Increasing the network size or repeating this whole campaign would not resolve the current bottlenecks. First isolate quadrature error, cubic truncation and optimization; then give a revised candidate a credible neural competitor and a fresh confirmation cohort.

The strongest defensible contribution at present is a structured numerical interaction representation with a small, reproducible calibration effect. A useful learned solver, a general FNO advantage, lower complete deployment cost, and literature novelty remain unestablished.

[High-resolution results figure](../../results/advance-full-review/advance-full-primary-results.png) · [Vector PDF](../../results/advance-full-review/advance-full-primary-results.pdf) · [Machine-readable figure data](../../results/advance-full-review/primary-results-data.json)

## 1. What was verified

All **12 upload parts**, the concatenated archive, and all **7,448 retained files** passed content-hash verification. The archive contains 2.417 GB of retained exact bytes compressed to 291.2 MB. All **49 stage seals** are consistent with their source, protocol and prerequisites; 3,481 retained scientific artifact references matched the verified export manifest. There were no detected mismatches.

The compact archive intentionally omits 1,282 arrays, checkpoints and rendered figures. Their declarations and recorded hashes are preserved, but their absent payloads cannot be independently rehashed or replayed from this export. This review verifies retained evidence and reproduces statistical summaries; it is not an independent rerun of omitted model weights or teachers.

| Outcome | Verified result | What it establishes |
|---|---|---|
| Native execution | 49/49 allocations completed, exit 0:0 | The declared workflow finished |
| CPU checks | 301 passed | Implementation checks passed on their cases |
| GPU checks | 40 distinct cases, repeated in 23 GPU jobs: 920 successful executions | Native CUDA checks passed; these are not 920 distinct tests |
| Mathematical correctness failures | 0 reported across stages | Checked identities hold on tested inputs |
| Reference acceptance | 1,184/1,184 | Recorded refinement and acceptance criteria passed |
| Primary claims reproduced | 40/40, including bootstrap endpoints | Published aggregation is reproducible from retained data |
| Scientific accuracy claims | 14 GOOD, 6 BAD | Scoped comparisons, not an overall victory |
| Scientific cost claims | 0 GOOD, 14 BAD, 6 NA | No declared cost advantage established |

The source of this archive is **a764670**, regardless of the earlier `c5c9a85` campaign label. The separate Adjacent diagnostic failure and unrelated `cv-*` jobs are not part of this 49-job experiment.

## 2. What the models actually compute

The physical task is periodic reaction–diffusion, schematically

\[
u_t=Au+r(u-u^2),\qquad A=\kappa\Delta.
\]

The discrete track uses the central-difference Laplacian and nodal multiplication. The continuum track uses spectral propagation and projected, dealiased products. These are different target equations; their results remain separate.

Our primary candidate has the form

\[
\widehat u(h)=S_h^{DF}(u)+g_2 Q_h(u)+g_3\widetilde C_h(u),
\qquad g_i=1+0.75\tanh\theta_i.
\]

Here `DF` is diffusion-first Strang splitting, `Q` is the full-output normalized two-node quadratic defect, and the filtered cubic term is an analytic expression. **Only two global scalar gains are fitted.** The full input field, step, physical coefficients and spatial operator determine the analytic responses; the gains do not depend on the input field. Inference does not receive a reference solution or future state. This is a fitted numerical time-stepper, not a temporal MLP or a state-conditioned neural operator. Repeated time queries currently recompute the response; a reusable temporal encoding is not established by this model.

| Family | Role and learned content | Review outcome |
|---|---|---|
| `df` | Theirs: classical diffusion-first splitting; no learned parameters | Important cheap baseline; succeeds on easy scaling cases |
| `etdrk4` | Theirs: classical exponential Runge–Kutta; no learned parameters | Loses fixed coarse-schedule accuracy comparisons, but can be strong after schedule selection |
| `quad2_full` | Analytic control: normalized full-output GL2 | More accurate overall than our fitted cubic augmentation in exploratory matched-schedule comparisons |
| `quad4_full` | Analytic control: normalized full-output GL4 | Strongest practical target for the next accuracy/cost study |
| `analytic_quad_cubic` | Analytic control: nested quadratic and cubic interactions | Strong accuracy reference, with greater computational work |
| `quad2_conditioned` | Ours historical: compressed two-node correction with 43 fitted parameters | All six fits selected trained checkpoints; node, amplitude and conditioning effects remain combined |
| `commutator_raw` | Ours numerical: GL2 plus raw leading cubic term | Useful mechanism control; finite-step regressions remain |
| `commutator_cubic` | Ours numerical: GL2 plus heat-filtered cubic, unit gains | Clean same-component learning-attribution control, but weaker than GL4 |
| `two_basis` | Ours: the same two analytic bases, two fitted gains | All 12 fits selected trained checkpoints; modest longer-time parent improvement |
| `fno_legacy` | Theirs adapted: local hybrid FNO, nonzero output initialization | Learns from very poor initialization but remains much worse than DF on validation |
| `fno_zero` | Theirs adapted: same FNO with zero residual head | All six select baseline-equivalent initialization |
| `fno_scaled` | Theirs adapted: zero head plus physical residual scaling | All 12 select initialization, including both data sizes |
| `fno_mean` | Theirs adapted: scaled FNO with separate mean output | One of six selects a fitted checkpoint; no robust gain established |

There are 62 model instances: 14 frozen, 25 selecting fitted checkpoints and 23 selecting initialization. The local FNO trunk has width 16, two blocks and cutoff four, with 47,057 parameters, or 47,345 for the mean-separated variant. These hybrid controls are **not authoritative reproductions of the published FNO benchmark**.

## 3. Primary accuracy and statistical strength

Confirmation contains **64 independent fields**, three paired training seeds, grids 32² and 64², and two final times. Its 31,488 endpoint records are repeated measurements, not independent samples. Each learned primary contrast has 384 queries; grids are averaged within each field/seed before crossed field/seed bootstrap resampling. The 128² measurements belong to an eight-field scaling cohort, not to this main confirmation.

The following entries are geometric-mean **control RMS / our two-basis RMS**. Higher than one favors Ours. Brackets are descriptive 95% intervals, unadjusted for multiple comparisons.

| Track/time | Fixed filtered parent | GL4 | Historical conditioner | Scaled local FNO | ETDRK4 |
|---|---|---|---|---|---|
| Discrete / .12 | 1.045 [.921,1.179] BAD | .399 [.279,.543] BAD | 1.721 [1.320,2.249] GOOD | 7.519 [5.618,10.165] GOOD | 4.096 [2.749,6.571] GOOD |
| Discrete / .24 | 1.160 [1.023,1.314] GOOD | .327 [.248,.431] BAD | 1.466 [1.075,2.087] GOOD | 7.803 [5.801,10.722] GOOD | 16.500 [9.648,30.774] GOOD |
| Continuum / .12 | 1.053 [.898,1.265] BAD | .400 [.289,.538] BAD | 1.667 [1.262,2.183] GOOD | 6.645 [4.851,9.339] GOOD | 4.184 [2.876,6.488] GOOD |
| Continuum / .24 | 1.143 [1.026,1.272] GOOD | .330 [.228,.448] BAD | 1.332 [1.017,1.784] GOOD | 6.689 [4.987,8.944] GOOD | 16.910 [10.002,30.689] GOOD |

The implemented success gate requires ratio ≥1.10, interval lower bound >1, and no more than five percentage points of target-pass regression. A ratio of 1.10 means a **9.09% error reduction**, not literally 10%. BAD means the gate was unmet, not necessarily that every observation deteriorated. The prose table in `docs/ADVANCE_RESEARCH.md` describes coverage as separate for this claim; the implemented gate includes it. This review follows the actual code and reproduced outcomes, and flags that documentation discrepancy without retroactively changing the protocol.

At .24, 48/64 discrete fields and 47/64 continuum fields improve over the fixed parent after averaging repeated contrasts. The interval lower bounds are only modestly above one. Twenty comparisons and only three training seeds warrant restraint even though the frozen criteria pass.

Two retained analytic alternatives reveal an important limitation. In explicitly **post-hoc** contrasts, our fitted model has **3–21% more RMS error than plain GL2** and **3.9–8.0 times more than the nested quadratic+cubic rule**. Learning partly repairs the chosen parent; that does not establish that adding the cubic approximation is better than omitting it.

All endpoints are finite, use accepted references, and show no sampled physical-interval violation or endpoint energy increase above 1e−12. These finite checks do not establish positivity between samples, long-rollout stability, a full Jacobian contraction bound or a continuous-time guarantee. RMS, sampled maximum error, mean bias and accuracy-qualified cost remain distinct metrics; [all-model metrics](../../results/advance-full-review/all-model-metrics.csv) retain their separate summaries.

There is a concrete typical-error versus tail tradeoff: at .24 our worst sampled spatial maximum is **4.10e−4 discrete / 4.32e−4 continuum**, versus **1.99e−4 / 2.50e−4** for the historical conditioner, despite our better geometric-mean RMS. Mean bias accounts for about **40–46%** of the average per-query error-energy fraction in two-basis, versus **23–38%** for GL4 and **68–74%** for the zero/scaled FNO controls. Those fractions diagnose error structure; a smaller fraction alone does not imply smaller absolute error. The raw archive also preserves spectral-band, centered, variance and gradient errors. No endpoint derivative was supplied for a measured mean-balance residual, and no learned-model Jacobian or intermediate-time reference curve was measured in this campaign.

## 4. Why the mathematical diagnosis changes

For centered state `v`, diffusion generator `A`, and the target's symmetric product `B`, the implemented leading cubic defect is

\[
C_3(v,h)=\frac{r^2h^3}{6}\left[
B(Av,B(v,v))-2B(v,B(v,Av))+2B(v,AB(v,v))-AB(v,B(v,v))\right].
\]

With unprojected continuum products this reduces to `−r²κh³ v|∇v|²/3`. **That simplification is invalid for finite Galerkin products**, which are nonassociative. The implementation correctly retains the nested products. Symmetric heat filtering forms `H(h/2) C3(H(h/2)v,h)` with `H=exp(hA)`. It preserves the leading fixed-grid small-step coefficient, but is not the exact cubic interaction kernel at a finite step.

The audit verifies structural nulls, quadrature moments, translation behavior, the reaction mean identity and appropriate energy inner products. Small-step remainder slopes near four support an `O(h⁴)` remainder for this coefficient on the tested fixed grids. They do not prove a fourth-order full solver or a stiffness-uniform estimate.

The cheap diagnostic was **16², four constructed fields**, repeated over tracks and horizons. Filtered cubic beat GL2 in all 32 paired cases, with median error ratio 7.32; GL4 offered almost no gain there. However, the quadratic and cubic mode-sum supports are disjoint for **all four chosen spectra**, even after aliasing. Their near-unit basis condition numbers therefore do not establish generic identifiability.

In confirmation, GL4 is approximately **7–12 times more accurate than our fitted model** in the high-pair and near-Nyquist regime averages at .24. One continuum near-Nyquist field has a roughly **294-fold** regression against GL4; its initial-mode `hλ≈4.95`. A reaction-shift field is about **17.5-fold** worse than GL2, with `hr=2.72`. These are exploratory counterexamples, not causal proofs. They show why a small-step expansion can behave well in an easy diagnostic yet fail when omitted finite-step effects matter.

The next diagnostic must measure vector error alignment and separate quadratic quadrature, cubic truncation, heat filtering and reaction variation on the **same fields**. Include overlapping frequency sums, closed triads, independent phases and nearly collinear bases. A finite-step interaction representation deserves priority over an unconditional leading-order cubic addition.

## 5. Learning and competitor fairness

All 48 final fits completed 800 updates. Every two-basis fit selected the final update, so convergence to the optimum is unproven. Across the twelve fits, quadratic gains are approximately **1.023–1.071** and cubic gains **.734–.862**. The repeatable adjustment slightly strengthens the quadratic correction and suppresses the cubic one; it does not prove a universal physical coefficient.

For frozen one-step basis arrays, prediction error is affine in the two gains. The actual normalized squared-error plus maximum-squared-error objective is convex in those physical gains over a box. **A direct bounded fit is the decisive missing control.** It can separate limitations of the basis from limitations of the nonlinear gain parameterization and Adam optimization, without adding network capacity.

FNO failures remain partly confounded. All zero/scaled FNO optimizer choices tied at initialization during 40-update tuning. Deterministic first-choice selection then gave the clipped setting the 800-update opportunity. Zero-head FNO clipped every update; scaled variants clipped about 94–98%. Finite gradients and twenty-update single-example improvements establish that learning can start, but do not demonstrate near-zero memorization or a competent mixed-regime training configuration. Frequent clipping is a symptom, not proof of causality; equal long opportunities for tied settings are missing.

Consequently, our 6.6–7.8-fold RMS advantage over the selected scaled-FNO control largely compares against a model that retained the physical baseline. It cannot establish superiority over a successfully trained FNO or the paper. Running an unused residual trunk also measures redundant inference work, not an unavoidable cost of useful neural prediction.

The 16-field two-basis variants are modestly better on several summaries than the 64-field fits. But they get more exposure per sample, and the smaller subset covers different physics. This is not a clean sample-complexity result. Future data-efficiency comparisons need repeated, stratified subsets and explicitly separated equal-update and equal-compute budgets.

## 6. Reference quality and untested generalization

The reference audit found no split contamination, unexpected confirmation schedules or metadata inconsistency. It independently reproduced **372 frozen schedule choices from 19,840 validation rows**. Actual temporal refinement and continuum spatial refinement were performed. All endpoint reference metadata matches the original records.

Reference estimates are usually small compared with errors, but reach 23–32% of error in the easiest cases. A post-hoc triangle-inequality sensitivity using those estimates leaves all 20 primary accuracy verdicts unchanged. The estimates are not certified bounds, so this is sensitivity analysis rather than proof.

Several design limits remain material:

| Limitation | Consequence | Required control |
|---|---|---|
| Fine-grid continuum teacher versus model-grid dynamics | Total error includes both solver error and coarse-to-fine closure error | Converged same-grid Galerkin teacher alongside the projected fine teacher |
| Both reaction and diffusion multiplied by 1.7 | This is exact overall timescale change, `Fshift=1.7 Fbase`, not altered relative stiffness | Vary reaction and diffusion independently |
| Shift cohorts use different fields and physical groups | Distribution differences do not isolate causal shift effects | Paired interventions on common initial fields |
| Spectral shifts retain old regime labels | A row called “low” need not remain low frequency | Actual spectral descriptors and crossed labels |
| Near-Nyquist mode defined relative to N=32 | Resolution transfer to N=64/128 is not a new cutoff-relative stress test | Separate fixed-physical and grid-relative-frequency cohorts |
| Schedule selection uses N=32 validation | N=64 frontiers test transfer, not independently optimized schedules | Distinguish grid-specific tuning from schedule transfer |
| Only final times .12 and .24 have main teachers | Intermediate trajectories and long-time stability remain unvalidated | Focused rollout and dense-output reference panels |
| Only eight scaling fields at N=128 | Scaling does not establish broad N=128 accuracy generalization | Fresh larger-grid cohort if that becomes the claim |
| Eight fields per regime, two per regime×distribution cell | Subgroup rankings have weak statistical resolution | Treat current slices as exploratory; power the next specific claim |

Now that confirmation failures have informed these recommendations, revised models must use **new confirmation fields**. The original outcomes remain intact.

## 7. Complete cost and scaling

| Resource or work | Native measurement |
|---|---:|
| Summed allocation walltime | 3 h 32 min 08 s |
| GPU allocation time | 3 h 04 min 48 s |
| Confirmation allocations | 1 h 48 min |
| Training allocations | 1 h 15 min 30 s |
| Reference/data preparation allocations | 20 min 51 s |
| Largest individual job | 24 min 35 s |
| Peak recorded tensor allocation / reservation | 71.04 / 92.00 MiB |
| Largest step host RSS | 4.663 GiB, during reporting |
| Money and electrical energy | Unmeasured |

These totals count allocation records once; worker steps and parallel reference task-seconds are not added again. Sampled physical GPU use includes other resident memory. Low tensor allocation does not measure GPU utilization. There is no evidence that larger RAM or VRAM budgets are currently needed.

At the primary RMS/max target `2e−5`, field coverage requires **every paired seed/grid cell to pass** under the frozen validation-selected schedule:

| Track/time | Ours passing fields | GL4 passing fields | Interpretation |
|---|---:|---:|---|
| Discrete / .12 | 53/64 | 62/64 | Ours fails the 95% coverage threshold |
| Discrete / .24 | 62/64 | 64/64 | GL4 is about 2.12× faster on eligible paired queries |
| Continuum / .12 | 57/64 | 61/64 | Ours fails coverage; ETDRK4 is also a useful alternative |
| Continuum / .24 | 61/64 | 57/64 | GL4 is about 4.26× faster on eligible pairs, but has lower coverage itself |

Discrete .24 GL4 has median eligible latency about **8.55 ms** versus **18.26 ms** for Ours. This example is a meaningful operational loss. Conversely, conditional speedups against other controls cannot rescue failed coverage. Missing schedules mean no feasible choice on the declared validation menu, not that a method can never solve the task. Median latencies and paired geometric speed ratios are different summaries.

Confirmation performs **251,904 timed or probe solver calls**. Measurement itself dominates a substantial part of the research cost. Scaling shows roughly 3.6–4.2× batching throughput improvement across methods, not a unique TDN advantage. In its easy fixed two-step cases, DF and two-basis both pass; DF is approximately seven times faster on the discrete track and twice as fast on the continuum track.

Five confirmation timing repeats and twenty scaling repeats do not estimate production p95/p99 reliably. “Cold” means the first call in a warm process, excluding complete process/model startup. Native paired randomized timing and synchronization are sound, but concurrent desktop work can still interfere. Kernel-level cost, power and long-rollout service timing remain unmeasured.

The atlas's 179,712 amortization rows are conditional scenarios, not independent deployment successes. Most charge training/tuning without a complete model-specific allocation of references, preparation, validation and deployment costs. Full offline amortization remains unresolved. A break-even formula is meaningful only when a valid matched-accuracy latency saving exists; against the strongest analytic alternatives it generally does not here.

## 8. Protected exploratory paths

| Prototype | Actual evidence | Decision |
|---|---|---|
| XH: causal coarse memory | Clean fitted memory has median RMS about .049× coarse RK4 and .787× deterministic secant; under startup noise 1e−4 it is about 15.4× worse than coarse RK4 | Retain a small noise-aware pilot; add deterministic damped secant before crediting learning |
| XR: rational propagation | Worst relative errors versus exponential ETD1 are about 47×, 237× and 13× for the tested substitutions | Retire these substitutions on this Fourier-diagonal workload |
| XR: amortized queries | Candidate query representations miss the declared accuracy tolerance; query count also changes maximum requested horizon | No break-even claim; hold workload/horizon fixed before another study |
| XT: tangent and feature twins | 24 pairs with equal resolved state and unresolved variance produce different resolved futures | Retain as concrete information-loss evidence; scalar variance alone is insufficient for these examples |

For XH, the learned secant block is almost the identity. A simple damped rule could explain much of the fitted benefit. Noise in successive observations creates an `O(noise/h)` contamination in the inferred closure. Its clean median gain coexists with worse maximum error than deterministic memory; deployment is premature.

For XR, changing the linear response does not raise ETD1's nonlinear order. Padé(1,1) also lacks strong stiff damping. The single-query amortization test asks an earlier final time while classical dense output still integrates to the longer horizon, so it cannot isolate the effect of query count.

For XT, identical supplied features imply identical deterministic predictions; if the actual futures are separated by `d`, at least one prediction error is ≥`d/2`. This is a useful counterexample for the tested finite ODE, not a theorem that every closure fails or a demonstrated new solver.

All three native exploratory runs reuse previously inspected development parents and reproduce their numerical metrics. They add reproducibility and native timings, **not independent confirmation**.

## 9. Prioritized next portfolio

These are recommendations for a new development phase, not newly executed experiments or retroactive changes to this campaign.

| Priority | Experiment and falsifiable question | Advance / stop criterion |
|---|---|---|
| 1: attribution | Direct bounded two-gain fit, quadratic-only fit, unit gains, GL2 and GL4; is remaining error optimization or representation? | Compare identical loss/data and verify optimizer accuracy; stop adding capacity if an optimal two-gain fit still loses to the strongest contained analytic rule |
| 2: finite-step physics | Same fields, controlled `hλ≈.1–10` and `hr≈.1–3`; compare GL2/GL4, raw/filtered cubic and nested cubic, with same-grid/fine teachers | Advance only a correction that removes an identified error component without relying on accidental cancellation |
| 3: credible neural control | Continue tied optimizer settings under matched bounded long opportunities; require decisive tiny-set fitting and inspect spectral/mean/max errors | A trained competitor must survive held-out validation; initialization ties cannot support a literature-level neural claim |
| 4: physical representation | Cheap approximation to the finite-time interaction kernel, potentially constrained nodes or exponential response bases | Match GL4 accuracy while saving transforms/products; include construction and cache-refresh costs |
| 5: execution | Profile a few frozen easy, hard and borderline workloads, then optimize all methods comparably | Numerical parity plus a repeatable complete-call margin; no deployment policy before such a margin exists |
| Protected exploration | Damped/noise-aware memory and phase-sensitive closure tests | Beat inexpensive deterministic controls on clean and noisy cases; stop if the information or noise floor dominates |

Use the current cohort to diagnose these questions, then freeze any promising new claim before fresh confirmation. Keep all easy cases: they expose unnecessary model cost. Keep the failed and inconclusive branches visible. A smaller numerical rule is a successful research outcome if it earns its accuracy, robustness and cost advantages.

## Supporting audits and traceability

- [Independent statistics and subgroup tables](../../results/advance-full-review/statistics.md)
- [Training, coefficients and competitor fairness](../../results/advance-full-review/training.md)
- [Mathematical identities and all three prototypes](../../results/advance-full-review/mathematics.md)
- [Reference refinement, cohorts and frozen selection](../../results/advance-full-review/references.md)
- [Complete costs, scaling and amortization](../../results/advance-full-review/costs.md)
- [Compact review bundle README](../../results/advance-full-review/README.md)
- [Verification summary](../../results/advance-full-review/provenance-summary.json)

Archive SHA-256: `11e6edbf6fa709f17b5479ee9aa9f066dad05b1ce98950dddabe9c2932b6b3a0`.
Scientific source SHA-256: `ae810a101203ddd4cb26dcd75253ef62ebb0cea19f5a36f15a8c347d81d01466`.
