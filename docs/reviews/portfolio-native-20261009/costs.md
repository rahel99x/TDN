# Native portfolio cost and scaling review

Source: exact canonical retained bytes for `fedora-portfolio-20261009T183307014599Z`, source `c5c9a85`. Machine-readable recomputation: `costs-analysis.py` → `costs-summary.json`. No model inference rerun and no original artifact changed.

## Accounting, complete costs, and operational coverage

All **40 allocation records completed with exit 0**. The final scheduler snapshot contains 40 allocations plus 3,291 task-step records, all COMPLETED. Allocation and step times are deliberately **not summed together**. The longest requested job limit was 40 minutes, below the 45-minute cap; the longest observed allocation was 1,336 s = 22m16s. The accounting describes only this run; the accidental later rerun is absent and cannot be charged from this package.

| Work | Jobs | Allocated elapsed | Fraction of all allocated elapsed |
|---|---:|---:|---:|
| Audit and diagnosis | 2 | 56 s | 0.26% |
| Development reference preparation | 1 | 105 s | 0.48% |
| Exploratory prototypes | 3 | 17 s | 0.08% |
| Training, tuning, validation schedules | 6 | 3,462 s = 57m42s | 15.79% |
| Freeze | 1 | 31 s | 0.14% |
| Confirmation references | 12 | 1,961 s = 32m41s | 8.94% |
| Confirmation measurements | 12 | 15,632 s = 4h20m32s | 71.28% |
| Aggregate | 1 | 253 s | 1.15% |
| Scaling | 1 | 220 s | 1.00% |
| Report | 1 | 194 s | 0.88% |
| Total | 40 | **21,931 s = 6h05m31s** | 100% |

GPU-reserved time: **19,314 s = 5h21m54s** across 19 GPU allocations. Confirmation uses 80.94% of GPU allocation time. Reserved CPU time: 96,028 core-seconds = 26.674 core-hours. Reported TotalCPU: 23,288.266 CPU-seconds = 6.469 core-hours. CPU reservation utilization is **24.25%**, not GPU utilization. The worker explicitly sets BLAS/OpenMP to one thread while reserving 4 CPUs on GPU jobs, so ~25% CPU allocation utilization is partly intentional.

Scientific engine timers total 20,747.370 s; another 1,183.630 s (=19m43.630s) of recorded allocations lies outside those engine timers. This includes startup, prerequisite checks, tests, packing/report publication, and other worker work; it is not all identified as avoidable overhead. The aggregate job is especially verification-heavy: 253 s allocated versus 31.995 s engine time. Report costs 194 s, of which 163.214 s is engine time.

Workflow creation at 2026-10-09 18:33:07.019887 UTC to final report state at 2026-10-10 09:48:21.192941 UTC is 54,914.173 s = 15h15m14.173s. This is an observed calendar envelope, not allocated compute: the 9h09m43.173s difference from allocation sums cannot be attributed solely to queueing from these records. No scheduler start/end timestamps are requested by the accounting collector.

Memory does not appear to constrain this campaign. Maximum observed per-task-step host RSS is **8,071.691 MiB**, in reporting; training tops out at 2,306.957 MiB and confirmation at 1,413.227 MiB. All allocation-level MaxRSS values are blank, so these are observed step maxima, not a certified simultaneous whole-node RSS. Maximum CUDA allocator peak is **302.191 MiB** and maximum reserved peak **390 MiB**, both at scaling. Physical device-use peaks also include display/other users of the GPU. Group peaks are not per-model memory attribution. No GPU utilization, power, energy, carbon, or price/rate was measured, so these stay NA.

## Training versus total offline work

The catalog contains **2,341.632 s** of summed trial walltime across all families, versus **1,202.880 s** of optimizer timing. Trial time includes validation and checkpoint/parameter bookkeeping as implemented; optimizer time does not cover least-squares fits. Training-stage engine time is 3,305.883 s, leaving 964.251 s outside the per-trial totals for sample preparation, validation schedule scans, catalog construction and verification. Neither trial time nor optimizer time alone is complete offline cost.

Across both tracks, all data sizes, and all seeds:

| Family | Models | Trials | All-trial wall seconds | Optimizer seconds |
|---|---:|---:|---:|---:|
| Conditioned two-node | 12 | 36 | 214.00 | 116.03 |
| Fixed-amplitude fitted control | 12 | 24 | 36.14 | 0 (not an optimizer-time measurement of least-squares) |
| Node-only | 12 | 36 | 200.65 | 106.94 |
| Joint node/amplitude | 12 | 36 | 198.25 | 107.58 |
| Linear-feature fit | 12 | 24 | 41.25 | 0 (same caveat) |
| Rich conditioned | 12 | 36 | 216.88 | 117.53 |
| Four-node conditioned | 12 | 36 | 266.00 | 151.77 |
| Residual quadratic | 12 | 36 | 191.91 | 92.57 |
| Small hybrid FNO | 12 | 60 | 205.15 | 104.11 |
| Standard hybrid FNO | 12 | 60 | 246.36 | 129.49 |
| Direct FNO | 12 | 60 | 105.09 | 65.84 |

The FNO controls got more tuning trials, but similar measured total walltime to conditioned quadrature. This is useful fairness evidence; it does not establish a fixed compute-matched frontier or adequate optimization of each family. The nominal 55/25/20 percent A/B/C budgets were ceilings. Exploratory prototype engine time was only 0.570 s combined, versus ~3,306 s training. The high-risk branch was executed but was a set of cheap toy falsification probes, not 20% of actual campaign compute.

For the genuinely positive **discrete T=.12, n32 training** comparison, conditioned quadrature takes 6.394 ms on average versus 10.678 ms for small hybrid FNO and 15.149 ms for standard hybrid FNO on the 22 fields where both frozen schedules qualify (132 paired seed/grid rows, not 132 independent fields). Mean savings are 4.283 and 8.755 ms. Per-seed conditioned training+tuning took 7.100–7.648 s; dividing only that by per-seed savings gives **1,631–1,806 queries** against small FNO and **812–870 queries** against standard FNO. These newly derived quantities are deliberately labeled optimistic *training-only* scenarios, not outputs of the broken amortization panel: they omit teachers, tuning other mechanisms, preprocessing, validation schedules, verification, startup, rejected queries and fallback; they also ignore comparator training costs and uncertainty in the mean saving. Classical/fixed controls remain faster. The full derivation is in `costs-optimistic-amortization.json` using `statistics-followup.json` and `freeze/catalog.json`.

## Scaling is useful but narrow

The 42 scaling rows are **36 measured method/workload cells** (6 methods × 3 grids × 2 tracks), all accuracy-qualified, plus **6 explicitly omitted batch-one workload cells** due to the declared case cap. There are only **four independent fields**: low, mixed, high-pair and rough, all κ=.001, reaction rate=1, T=.06, two equal steps. Three grids reuse these four continuous fields. One prescribed seed and training count 32 are used. Thus 144 field–method–grid–track evaluations are not 144 independent examples.

Every method passes RMS and maximum-error targets 2e-5 in these mild-physics short-horizon cases. The spectral 'stress' labels do not make this a universal worst-case or demanding time-to-solution test. The continuum/discrete spatial targets remain separate.

Warm complete-call latency per **batch of four fields**, milliseconds:

| Track / grid | Ours conditioned | Fixed analytic two-node | DF | ETDRK4 | Small hybrid FNO | Standard hybrid FNO |
|---|---:|---:|---:|---:|---:|---:|
| Discrete 64² | 4.093 | 3.292 | **0.667** | 1.010 | 2.604 | 3.614 |
| Discrete 128² | 4.083 | 3.300 | **0.671** | 0.989 | 2.566 | 3.674 |
| Discrete 256² | 4.149 | 3.347 | **0.678** | 1.013 | 2.852 | 5.843 |
| Continuum 64² | 20.549 | 19.570 | 13.229 | **4.049** | 15.098 | 16.215 |
| Continuum 128² | 22.460 | 21.432 | 14.556 | **4.416** | 16.633 | 17.736 |
| Continuum 256² | 21.018 | 20.068 | 13.615 | **4.183** | 15.735 | 18.634 |

Conditioned quadrature is ~6.1× slower than DF on discrete workloads and ~5.0–5.1× slower than ETDRK4 on continuum workloads at the shared target. ETDRK4 is faster **and** has smaller worst-field RMS/max error in all six panels. Normalized fixed quadrature is also faster and has lower worst-field RMS/max error in all six panels. These results do not support a large-grid solver advantage for the learned conditioner.

Grid increases from 64² to 256² multiply unknowns by 16 but barely increase quadrature latency; GPU launch/Python overhead is therefore a plausible bottleneck. It is not proven by a CUDA trace. Standard FNO latency grows more with grid. The capped batch-one rows prevent an honest batch-scaling speedup curve.

Scaling generated **185.738 s** of unique-case CPU reference work inside its GPU allocation, versus 189.680 s engine time; teacher generation occupies about 97.9% of that stage. Precomputing/sealing those teachers in a CPU stage can cut reserved GPU time without altering measured inference latency. Do not sum `teacher_seconds` across method rows: the same reference preparation total is repeated six times per case.

## Measurement and reporting defects or limits

1. **Scaling timing configuration is ignored.** The immutable protocol has `scaling.repeats=5` and `timing_repeats=5`, but `learning.scaling` reads only optional `timing.repeats`, falling back to 3. The actual output has 3 warm timing samples per method/cell. Confirmation correctly uses the five-repeat fallback. These three observations support an observed median, not a reliable p95/p99; the recorded p95 equals the observed maximum.
2. **The amortization graph is wired to absent columns.** `report.amortization` looks for `candidate_seconds`/`model_seconds` and `comparator_seconds`/`baseline_seconds` in `locked_frontiers`, which instead hold per-model `cost_seconds`. The exact original `report/tables/amortization.csv` was separately extracted and hash-verified: **50,688/50,688 rows are NA**, all have offline-training values, and **0** have inference-saving or break-even values. Thus the missing economic result is confirmed in the original output and explained by the code; it is not necessarily evidence of no positive margin. Any repaired calculation needs an explicit comparator join, the same workload/accuracy, uncertainty, all offline costs and acceptance/fallback costs; the matched cost gates remain a separate valid calculation.
3. `cold_seconds` means first invocation in a timing group, with warm process/libraries and possibly warm coefficients. It does **not** measure process startup, CUDA context creation, checkpoint loading, host-to-device transfer or service cold-start latency.
4. GPU peak memory is measured for a whole interleaved model group. Comparing it as though it were a per-model memory requirement is invalid.
5. CPU component profiling uses initialized FP64 models at 16², not frozen trained FP32 models on the 4090. CPU profiler categories do not establish the GPU bottleneck. They do show median whole-call CPU latency ~0.895 ms for conditioned quadrature versus ~0.717 ms fixed; isolated feature extraction is ~0.09–0.10 ms, while learned dense operators are only ~0.23% of instrumented self-CPU time. Reducing MLP parameters alone is unlikely to erase the measured several-fold complete-solver gap.
6. Validation chooses schedules using per-field amortized matching-physics batches, while confirmation measures single-field complete calls. Selection is frozen fairly, but it optimizes a different deployment workload. Publish matched batch-one and batched selection views, both frozen before fresh confirmation, before claiming a best deployable frontier.
7. Successful stages can contain no generic scored experiment rows: scaling has zero rows in `rows.jsonl` but 36 valid method cells in its dedicated table. Counting GOOD/BAD/NA rows alone therefore omits real experiments and cannot serve as universal success rate.
8. Research-budget safety calls execute inside rollout timing; they include occasional memory checks. Timings are valid for the instrumented implementation, but a deployment-kernel claim should separate and quantify that safety overhead without silently deleting it from one method only.

## Highest-value next operational changes

Keep original observations intact. First repair scaling-repeat and amortization reporting, then freeze a focused protocol rather than rerun every family. Profile frozen GPU models at batch1/4/16 (if memory allows), separately benchmark host overhead, FFTs, pointwise kernels, feature extraction and transfer, and expose 20–50 randomized warm rounds where timing matters. Retain matched optimizations for ETDRK4/DF/FNO. Move large-grid teacher generation out of GPU allocations. Measure CPU-versus-GPU crossover because small-grid complete calls may favor CPU. Reduce duplicated materialization of endpoint/timing tables and rendered assets while preserving raw sources and exact hashes: report memory and storage already exceed model memory by orders of magnitude. Spend additional confirmation compute on distinct fields/regimes and informative hypotheses, not extra replicas of already-dominated methods.
