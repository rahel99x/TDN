# Native full Advance: complete-cost, scaling, and operational review

Run `fedora-advance-full-20261011T001715`, source `a7646702ff1061d78a722153ef2548d9d831de0d`. This review reads preserved native outputs and does not modify observations, models, selection, or claims. Recompute with `costs-review.py`; `costs-review.json` retains grouped numerical tables and raw-source names.

## Operational outcome

All **49/49** submitted allocations completed with exit `0:0`; there are **23 GPU allocations**. Their 23 JUnit suites contain the same 40 distinct CUDA case names, each passing with no errors or skips: **920 GPU case executions**, not 920 distinct tests. Audit supplies **301 CPU cases**. The workflow stayed within its declared allocation envelopes: largest request 35 minutes, actual longest job 24m35s (continuum training seed 0). No failed or interrupted allocation occurred in this campaign; failures from unrelated `cv-*` jobs are not TDN costs. This does not establish scientific success.

| Quantity | Measured value |
| --- | --- |
| Summed allocation walltime | 3h 32m 08s |
| GPU allocation walltime | 3h 04m 48s |
| Submission to terminal report state | 3h 32m 11s |
| Allocated CPU core-time | 15h 32m 56s |
| Actual CPU core-time | 4h 04m 14s |
| Science-stage timers | 3h 18m 23s |
| Allocation minus science-stage timers | 0h 13m 45s |
| Largest step MaxRSS | 4.663 GiB (report) |
| Peak sampled physical GPU use | 2.716 GiB (includes other resident use) |
| Maximum campaign tensor allocator peak | 71.04 MiB |
| Maximum tensor allocator reserved peak | 92.00 MiB |
| Lowest effective soft VRAM cap | 15.878 GiB |
| Money / electrical energy | NA: no tariff or power measurement |

Accounting uses only the 49 parent allocation records from terminal `state/scheduler-accounting.json`. Step rows are used for MaxRSS, never added again to time/cost totals. All allocation MaxRSS fields are absent, making a per-allocation memory sum invalid. The stage timers exclude most test, launch, verification, and serialization work. The 825 seconds outside those timers are not all scheduler overhead. Actual CPU/core reservation efficiency is 26.18%; that is **not GPU utilization**. No kernel profiler or GPU activity time establishes a device utilization percentage. The close sum-of-allocation and workflow elapsed times support near-serial execution; they do not alone distinguish queue waits from small overlaps.

| Stage | Jobs | Allocation | GPU allocation | Science timer |
| --- | --- | --- | --- | --- |
| aggregate | 1 | 0h 01m 14s | 0h 00m 00s | 0h 00m 34s |
| audit | 1 | 0h 00m 48s | 0h 00m 00s | 0h 00m 00s |
| confirm | 16 | 1h 48m 00s | 1h 48m 00s | 1h 42m 22s |
| confirm_prepare | 16 | 0h 15m 55s | 0h 00m 00s | 0h 12m 47s |
| diagnose | 1 | 0h 00m 15s | 0h 00m 00s | 0h 00m 08s |
| explore | 3 | 0h 00m 21s | 0h 00m 00s | 0h 00m 03s |
| freeze | 1 | 0h 00m 19s | 0h 00m 00s | 0h 00m 06s |
| prepare | 1 | 0h 03m 01s | 0h 00m 00s | 0h 02m 54s |
| report | 1 | 0h 03m 32s | 0h 00m 00s | 0h 02m 50s |
| scaling | 1 | 0h 01m 18s | 0h 01m 18s | 0h 00m 59s |
| scaling_prepare | 1 | 0h 01m 55s | 0h 00m 00s | 0h 01m 43s |
| train | 6 | 1h 15m 30s | 1h 15m 30s | 1h 13m 56s |

## Where time went

Confirmation consumed **1h48m** of allocation time (50.91% of the campaign), versus **1h15m30s** for training allocations (35.59%). Preparing train/validation, confirmation and scaling references consumed **20m51s** of allocations. Aggregation/reporting consumed another 4m46s. The campaign has no model or job failure charge, but intentionally rejected parameter candidates and trials are charged in the training totals.

The 62 frozen catalog models record **2637.9517 seconds (43m57.95s)** of training/tuning/equal-time/tiny-overfit/frozen-validation trial timers. These timers include their own validation and setup, so they are not pure optimizer time. Their sum is below 4436.3947 seconds of training-stage numerical time: approximately **1798.443 seconds** remains in schedule-frontier validation and other stage work. Do not present the catalog training number as complete offline cost. All 48 final fits reached 800 updates. Equal-time diagnostics are separate training opportunities, not the comparison catalog.

| Reference split | Track | Accepted / requested | Summed task seconds |
| --- | --- | --- | --- |
| confirmation | continuum | 256/256 | 1328.923 |
| confirmation | discrete | 256/256 | 12.859 |
| scaling | continuum | 16/16 | 288.916 |
| scaling | discrete | 16/16 | 1.570 |
| train | continuum | 192/192 | 580.883 |
| train | discrete | 192/192 | 4.262 |
| validation | continuum | 128/128 | 535.111 |
| validation | discrete | 128/128 | 3.127 |

Reference task seconds sum work performed by parallel CPU workers; they can exceed preparation elapsed time and must not be added to those allocation totals. They are retained to expose continuum-reference cost, not to double-charge it. There are 1184 accepted reference records. Spatial refinement, parent data generation, file sealing, and references are real offline work even though inference timing intentionally excludes them.

| Timing scope | Groups | Methods/workloads | Warm calls | First + warmup + memory probes | Timed work |
| --- | --- | --- | --- | --- | --- |
| aggregate | 2432 | 31488 | 157440 | 94464 | 1h 38m 51s |
| scaling | 36 | 216 | 4320 | 648 | 0h 00m 55s |

Confirmation makes **251904 full solver calls**: 157440 warm measurements plus 31488 each of first invocation, warmup and separate memory probes. Their summed durations are **5930.955 seconds**, 96.56% of confirmation science-stage walltime. Five repetitions make its timing experiment expensive while providing no meaningful p95/p99 service-tail estimate. Next time, reallocate some broad repeated timing to fewer representative but more deeply sampled workloads; retain paired randomized order and all accuracy measurements.

## Scaling: fixed two-step workloads, not an optimized solver frontier

The protocol requests 44 physics/grid/batch cells. **36 are measured** (216 model rows), and **8 cells remain explicitly `INSUFFICIENT_DISTINCT_FIELDS`**. Batch-one uses all eight scaling fields; batch-four uses only the first four fields sharing coefficients. The other four are not padded or duplicated. Batch-four therefore does **not** establish stiff/near-Nyquist/boundary-regime throughput. Both tracks use horizon 0.12, two steps of 0.06, FP32 and the first trained seed. No scaling-truth model selection occurred.

Every table entry is the median warm full-call latency in milliseconds across eight fields (batch-one) or a single four-field batch (batch-four). Reported ranges and per-field throughput remain in JSON. Methods and grids are paired through the same physical field; these are not independent regime replicates.

### discrete

| Method | 64² B1 ms | 128² B1 ms | 64² B4 ms | 128² B4 ms | Qualified B1 at each grid |
| --- | --- | --- | --- | --- | --- |
| df | 0.682 | 0.679 | 0.727 | 0.700 | 8/8; 8/8 |
| etdrk4 | 1.011 | 1.006 | 1.094 | 1.041 | 7/8; 7/8 |
| quad4_full | 4.380 | 4.287 | 4.361 | 4.709 | 8/8; 8/8 |
| commutator_cubic | 4.673 | 4.653 | 4.888 | 4.783 | 8/8; 8/8 |
| two_basis | 4.812 | 4.853 | 5.058 | 4.613 | 8/8; 8/8 |
| fno_scaled | 3.138 | 3.149 | 3.200 | 3.265 | 8/8; 8/8 |

### continuum

| Method | 64² B1 ms | 128² B1 ms | 64² B4 ms | 128² B4 ms | Qualified B1 at each grid |
| --- | --- | --- | --- | --- | --- |
| df | 12.562 | 13.625 | 13.808 | 14.531 | 8/8; 8/8 |
| etdrk4 | 3.954 | 4.175 | 4.182 | 4.516 | 7/8; 7/8 |
| quad4_full | 22.973 | 24.350 | 24.672 | 25.625 | 8/8; 8/8 |
| commutator_cubic | 25.312 | 26.853 | 27.520 | 28.541 | 8/8; 8/8 |
| two_basis | 25.512 | 27.098 | 27.339 | 28.476 | 8/8; 8/8 |
| fno_scaled | 15.366 | 16.428 | 16.669 | 17.512 | 8/8; 8/8 |

All methods pass the one batch-four workload at each grid/track. ETDRK4 misses the joint 2e−5 RMS/max target on the reaction-stiff batch-one field at both grids/tracks (RMS ≈2.50–2.52e−5; max ≈3.44–3.46e−5); it must not be advertised as universally qualified from these two-step results. DF, two-basis, filtered cubic, GL4, and scaled FNO pass all eight fixed-schedule fields. DF is about **7× faster than two-basis on the discrete track and 2× faster on the continuum track**. GL4 also beats two-basis at fixed schedule. No 64²/128² speed advantage is shown for two-basis.

On exactly the same four fields, batching raises throughput by **3.63–4.18×** across methods/grids/tracks (single-field calls summed then divided by one four-field call). This is general batching benefit, not a TDN-only mechanism. Two-basis obtains 3.83×/4.09× in discrete 64²/128² and 3.72×/3.82× in continuum. The near-flat 64²→128² latency despite 4× grid points is consistent with launch/Python/synchronization overhead dominating at these small grids; there is no profile proving that attribution. Larger grids alone have not yet moved the workload into a demonstrated compute-bound regime.

## Matched-accuracy economics

All **20 primary locked-cost claims remain unsuccessful: 14 BAD, 6 NA, 0 GOOD**. At t=0.12, two-basis has joint-field coverage 53/64 discrete and 57/64 continuum; at t=0.24 it has 62/64 discrete and 61/64 continuum. The 95% coverage rule therefore fails the short-horizon cases. Relative timing on surviving eligible rows cannot erase that failure.

| Track/horizon | Control | Predeclared speed ratio control / two-basis | Candidate coverage | Control coverage | Verdict |
| --- | --- | --- | --- | --- | --- |
| continuum 0.12 | commutator_cubic | NA | 89.06% | 98.44% | BAD |
| continuum 0.12 | quad4_full | NA | 89.06% | 95.31% | BAD |
| continuum 0.12 | quad2_conditioned | NA | 89.06% | 95.31% | BAD |
| continuum 0.12 | fno_scaled | NA | 89.06% | 0.00% | BAD |
| continuum 0.12 | etdrk4 | NA | 89.06% | 96.88% | BAD |
| continuum 0.24 | commutator_cubic | NA | 95.31% | 0.00% | NA |
| continuum 0.24 | quad4_full | 0.235 | 95.31% | 89.06% | BAD |
| continuum 0.24 | quad2_conditioned | NA | 95.31% | 0.00% | NA |
| continuum 0.24 | fno_scaled | NA | 95.31% | 0.00% | NA |
| continuum 0.24 | etdrk4 | NA | 95.31% | 0.00% | NA |
| discrete 0.12 | commutator_cubic | 1.995 | 82.81% | 89.06% | BAD |
| discrete 0.12 | quad4_full | 0.949 | 82.81% | 96.88% | BAD |
| discrete 0.12 | quad2_conditioned | 3.704 | 82.81% | 96.88% | BAD |
| discrete 0.12 | fno_scaled | NA | 82.81% | 0.00% | BAD |
| discrete 0.12 | etdrk4 | 0.430 | 82.81% | 82.81% | BAD |
| discrete 0.24 | commutator_cubic | 1.001 | 96.88% | 96.88% | BAD |
| discrete 0.24 | quad4_full | 0.473 | 96.88% | 100.00% | BAD |
| discrete 0.24 | quad2_conditioned | NA | 96.88% | 0.00% | NA |
| discrete 0.24 | fno_scaled | NA | 96.88% | 0.00% | NA |
| discrete 0.24 | etdrk4 | 0.213 | 96.88% | 82.81% | BAD |

The discrete t=.12 candidate is about 1.99× faster than filtered cubic and 3.70× faster than the old conditioner on eligible pairs, but it sacrifices coverage. Against GL4 it is slightly slower; against ETDRK4 it is much slower. At discrete t=.24 it is 2.12× slower than GL4 and 4.69× slower than ETDRK4 on eligible paired cases. At continuum t=.24 it is 4.26× slower than GL4. These published ratios are crossed-field geometric estimates, so they differ slightly from the JSON exploratory medians. None establishes complete solver-cost improvement.

Many control cells have **no globally validation-admissible locked schedule** because the frozen selection rule requires every validation field to qualify. Thus DF and scaled FNO have zero primary eligible locked pairs despite some useful per-case accuracy. An NA locked-cost result is not a physical impossibility theorem for those solvers. Introducing a regime-conditioned deterministic selector or adaptive controller would be a new experiment requiring fresh validation/confirmation, including its selection and fallback costs.

### Break-even: training-only lower bounds versus full campaign burden

The atlas has 179712 conditional amortization rows, largely repeated field/grid/seed/target scenarios. They are not 179712 independent positive observations. Upstream per-model complete offline costs are absent; the current atlas mostly uses catalog training/tuning only. Reference generation, frozen schedule selection, preparation, audit, verification and deployment loading are not zero. A mathematically appropriate per-use estimate is Q=(attributed offline cost)/(matched-accuracy control latency − candidate latency), and is undefined when the denominator is nonpositive or no matched eligible control exists.

| Illustrative discrete t=.12 comparator | Eligible rows | Positive margins | Median training-only queries | Median queries repaying whole campaign |
| --- | --- | --- | --- | --- |
| analytic_quad_cubic | 324/384 | 324 | 9,450 | 4,549,249 |
| commutator_cubic | 324/384 | 324 | 5,946 | 2,852,226 |
| quad2_conditioned | 324/384 | 324 | 2,144 | 1,028,223 |

Those are **exploratory conditional scenarios**, not deployable or population-guaranteed break-even counts. The last column charges the whole 12728-second serial workstation campaign to one repeated workload; it is a transparent campaign-repayment scenario, not an assertion that all competitor/reference/report research was necessary to deploy that model. A true method-specific complete offline numerator remains NA until shared expenses are allocated explicitly. Against primary GL4/ETDRK4 there is generally no useful saving to amortize; random positive near-tie margins do not support deployment. No dollars or joules may be inferred from these seconds.

## Measurement limitations and next actions

1. **Tail latency remains NA.** Confirmation has only five warm repeats and scaling twenty. Nested `tail_statistics.p95_seconds/p99_seconds` are correctly null; legacy `p95_seconds` is only an order statistic. Do not label it a service p95.
2. **First-call is not cold-process latency.** `cold_seconds` means first invocation in a timing group with warm process/libraries; process startup, CUDA context, model loading and transfers outside the callable are excluded. Both controls receive the same synchronization and randomized paired timing opportunities, but production cold start is unmeasured.
3. **Memory attribution is limited.** Separate warm probes include process-resident models/cache in absolute peaks; incremental peaks are not full standalone deployment footprint. Native GPU budget checks passed, but absence of GPU-memory pressure is not proof of high GPU occupancy.
4. **Operational coexistence is not isolation.** Other project CPU jobs appear in the user's scheduler listing during confirmation. Paired randomized order mitigates slow drift but cannot certify an interference-free benchmark. Preserve these native measurements; profile and repeat a small locked timing panel under controlled load before fine speed claims.
5. **Kernel/profile, CPU standalone memory, long-rollout service time and energy remain unmeasured.** No precision/compilation/fusion optimization is certified by this run. Profile representative frozen methods first, then give all methods comparable caching, batching and compilation.
6. **Capacity is not the immediate issue.** Peak tensor allocation was about 71 MiB; raising VRAM or host RAM limits is unjustified. Fix the work/accuracy margin and profile host overhead before increasing resources.
7. **No operational blocker requires rerunning the entire campaign.** Retain the successful measurement set and negative utility findings. Next experiments should test cheaper strong analytic rules and credible fitted corrections on harder work-qualified workloads, with explicit coverage/fallback costs. Native zero-head FNO baselines that retained initialization spend redundant network work to reproduce DF; such timings cannot establish useful neural speedup.
