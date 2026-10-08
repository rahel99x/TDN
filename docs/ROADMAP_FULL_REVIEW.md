# TDN full roadmap: native results review

The full run completed successfully, but it does **not** establish a general TDN efficiency advantage or a win over the FNO paper. The clearest learned signal is rank-one discrete accuracy. The clearest engineering gains are coefficient reuse and larger-workload throughput. Spatial inconsistency, expensive estimation, weak headroom and harmful combinations remain the main obstacles.

Run: `fedora-roadmap-20261008T022526329292Z`, jobs **197–206**, source **e46ae9f**, native Fedora/RTX 4090. This review uses the archived measurements only: no retraining, solver reruns, uploaded-code execution or checkpoint deserialization. Original scientific artifacts and scores remain unchanged. Corrected fixed-horizon comparisons are separate derived review results; the production frontier code has not been changed by this review.

[HD atlas and lookup guide](../results/figures/roadmap-full-20261008/README.txt) · [PNG](../results/figures/roadmap-full-20261008/TDN-complete-experiment-atlas.png) · [zoomable PDF](../results/figures/roadmap-full-20261008/TDN-complete-experiment-atlas.pdf) · [review data](../results/roadmap-full-review/README.md)

The atlas includes **every one of the 74,150 logged experiment rows**, **619,697 check occurrences**, all **642 native software-test executions**, and all **29 mechanism/combination summaries**. One row is report inventory, leaving 74,149 scientific records. These are repeated observations and overlapping checks, not that many independent samples. Coordinate lookup files identify every plotted record.

## Integrity and execution

All nine uploaded parts and the full 238,848,419-byte archive match the supplied index SHA-256. Gzip integrity and all 46,824 member paths pass. The ten stages completed; science/worker seals, source/protocol lineage and the separate Tower projection audit are recorded in the supporting JSON.

Native audit tests: **449 passed, one skipped** because the optional unchanged Tower checkout was unavailable. Each of the four GPU stages passed all **48 mandatory CUDA cases**. That is 192 GPU test executions, not 192 distinct test identities. The independent review subsequently validated all **45,277 Tower pages in 361 contracts** using unchanged Tower `356861e`, including every experiment/check projection and 5,023,520 scalar/config/evidence values. There are no missing or unindexed pages or reporting omissions. Root `metrics.jsonl` has only 795 progress records; full scientific results live in the indexed pages.

The jobs consumed **77 min 59 s** of summed allocation elapsed time, including **62 min 21 s** in GPU allocations. Allocated CPU time was **6.086 CPU-hours**. These totals count allocation rows once; batch/step rows are not added again. They exclude queue wait and do not imply currency cost. No monetary rate was supplied. Peak task RSS was about 9.13 GiB; peak GPU reserved memory was about 8.03 GiB, within the declared bounds.

| Stage | Job | Numerical seconds | Allocation seconds |
|---|---:|---:|---:|
| audit | 197 | 0.77 | 28 |
| headroom | 198 | 109.97 | 118 |
| prepare | 199 | 104.65 | 114 |
| train | 200 | 519.24 | 536 |
| confirm_prepare | 201 | 138.22 | 148 |
| confirm | 202 | 1485.58 | 1757 |
| policy | 203 | 767.49 | 1066 |
| transfer | 204 | 92.22 | 390 |
| scaling | 205 | 79.93 | 382 |
| report | 206 | 75.73 | 140 |

Allocation elapsed includes tests, prerequisite verification, reporting and process overhead; numerical timers cover stage computations. All ten allocations ended `COMPLETED`, exit `0:0`.

## The comparisons that must be corrected

**Different final times were mixed.** The original matched frontier grouped by parent/grid/target but searched schedules ending at both 0.27 and 0.81. Later smoothing can make the long-time state easier to approximate; it cannot substitute for the requested earlier state. In **1,412/22,032 original matched rows**, at least one comparison crosses final times. This includes 203 apparent Cheap-FNO wins and 671 Deep-FNO wins. The review reconstructed **44,064 separate-horizon rows** from the existing endpoints. They remain post-hoc schedule frontiers, not deployment policies.

The appropriate comparison is between candidate and control at the same `(parent, grid, spatial target, final time, RMS target, maximum target)`. After enforcing this, only **one of 28,275 jointly feasible neural/classical comparisons** is faster than the classical control, by just **1.007×**. A solitary 0.7% difference is not robust performance evidence. The original scores in the atlas stay visible for traceability; the PDF supplement uses corrected comparisons.

**A policy break-even is a timing artifact.** The only logged finite break-even, 49,015 queries, belongs to a policy accepting no neural proposals and executing classical fallback in every case. Of its apparent savings, 97.36% comes from a 217.247 ms initial classical call versus a 9.890 ms repeated fallback. The other cases give cost ratio 0.99097 and the paired cost-difference interval includes zero. No neural acceleration or useful amortization is established. Benchmark warm and cold execution separately, and randomize/repeat candidate/control order.

**Transfer teacher agreement is not demonstrated refinement.** All 592 timed transfer cases report zero teacher difference and 196 RHS evaluations. The two DOP853 solves tighten tolerances while preserving the same maximum step. Identical capped steps can produce identical answers. This does not prove the teachers wrong, but near-roundoff improvements require actual step refinement, a roundoff floor and an independent cross-check.

## What the neural experiments show

All **64 tuning trials and 51 final trials** completed **22,980 updates** without numerical failure. Final selection retained **42 trained checkpoints and nine initializations**. Cheap FNO and Embedded select initialization in all three seeds; learned efficacy cannot be credited to those selected models. All **40,320 confirmation endpoints** are finite. The independent units are **24 discrete field clusters**, with **12 on the continuum subset**, repeated over physics, grids, schedules and seeds.

At final time 0.27 and joint RMS/max tolerance 2e-5, rank one and rank two each cover 288/288 repeated discrete cells, versus 192/288 for Cheap/Deep FNO. But rank one is about 1.79× slower than Cheap FNO on their mutually feasible frontier cells and 4.62× slower than the strongest classical control. These are conditional descriptive ratios; accuracy coverage and latency eligibility must stay separate.

Rank one has **43 parameters**, is trained in all seeds, and improves discrete RMS in **1395/1440 same-schedule comparisons** against Cheap FNO, with **2.344× median error reduction**. At the identical schedule it costs about 1.82× more. Its continuum median paired error ratio is 0.999989, essentially no broad gain. Rank two costs roughly 1.35× rank one without a commensurate accuracy advantage.

C1 rank 0/1/2 has median same-schedule RMS ratio **1.0 versus untrained analytic `df_quad2`**, at about **1.39×/2.34×/2.79×** its runtime. Embedded's strong discrete accuracy comes from its initialization-selected classical RK4 backbone. Source-before-compression has a clear information-preservation diagnostic but only a **1.00215× median discrete error ratio** over its learned postcompression ablation here. The direct neural FNO fails every stated target, so it is not a credible basis for claiming superiority over the published method.

The main continuum bottleneck is spatial. Embedded median RMS errors are about 3–4e-8 on the discrete target but 4.21e-5 at grid 32 and 1.06e-5 at grid 64 on the continuum target. Approximately fourfold reduction when doubling resolution is consistent with a second-order spatial floor. At 2e-6, no trained family passes every schedule in any continuum group.

## Deployment and combination findings

Policy review verifies all **984 endpoints**, 20 calibration envelopes and 82 variant/target summaries. There are **900 accurate outputs, 572 proposal accepts, 412 fallbacks and 84 false accepts**. All false accepts are continuum cases without spatial checking; their failures remain resolved after accounting for estimated reference uncertainty. Across paired empirical variants, spatial checking improves continuum accuracy from **156/240 to 240/240**, consisting of **70 accepted proposals and 170 fallbacks**. Only six independent fields underlie this subset.

No empirical variant delivers both complete matched accuracy and lower total cost. The cheapest accurate continuum variant is initialization-selected Cheap FNO at **1.192× classical cost**; trained Source costs **1.202×**, C2 **4.162×**. Estimation consumes **422.619/460.551 s = 91.8%** of deployment time. Rejected work is an overlapping subset of this time and must not be added twice. At alpha=0.01, 24 calibration functions require unavailable conformal rank 25. Zero failures among four accepted independent functions still gives a 52.71% one-sided 95% upper conditional-risk bound under iid sampling, not 1%.

Combining mechanisms is not automatically beneficial. Quadratic+cubic correction reduces RMS by **103–3742×** in three numerical diagnostics, at **50–56× CPU cost**. Adding approximate mean replacement then worsens that stronger combination by **54–2348×** while adding more work. It can still pass a weak comparison against the original DF base. Because mean and centered error are orthogonal, `RMS² = mean_error² + centered_RMS²`; a less accurate replacement mean creates an unavoidable floor. Future combination gates should compare against their strongest contained submodel.

Fixed fractional filtering similarly spoils Richardson cancellation: filtering a correction that cancels a leading `h³ a` defect leaves `(I−F) h³ a`. The Gray–Scott and Burgers diagnostics show much worse errors after filtering; none of their guards activates. Precise near-roundoff ratios remain provisional until teacher refinement is repaired. The 394 transfer BAD rows fail only same-step cost checks, not reported accuracy or physical checks.

Coefficient reuse is the clearest engineering improvement: **2.426× ETDRK4 and 1.497× GL3**, with identical outputs and fewer preparations. At grid 512²/batch 16, Source and Rank one are **1.28× and 2.25× faster than Cheap FNO**, respectively. These large-grid cases lack independent accuracy references and use repeated copies of one engineering field; they are throughput evidence only.

## Every mechanism and combination

These are the original, independently recomputed **required-check attainment** scores. `BAD` means at least one required check failed, possibly cost alone. `NA` remains in the score denominator. The scores are not accuracy percentages, probabilities, theorem proofs or rankings of model quality. All expected stage coverage is present; 26 aggregates are BAD and three NA.

| ID | Recorded verdict | Score /100 | Review interpretation |
|---|---|---:|---|
| M00 | BAD | 61 | One bare DF step already meets the development target in 70/72 conditions; little normal-case headroom. |
| M01 | BAD | 68 | Spatial checking removes observed continuum false accepts; temporal accuracy alone leaves a grid-error floor. |
| M02 | BAD | 61 | Quadratic coefficient diagnostics pass; C1 accuracy largely comes from this untrained analytic term. |
| M03 | BAD | 60 | Rank one improves discrete RMS; rank two adds cost without a comparable benefit. |
| M04 | BAD | 60 | High-high-to-low information preservation is established on the diagnostic; learned endpoint advantage is small. |
| M05 | BAD | 82 | Retained Fourier products pass near machine precision; optional filtering must preserve temporal order. |
| M06 | BAD | 59 | Trust costs more without materially improving main confirmation coverage. |
| M07 | BAD | 68 | Routing and fallback execute correctly but no empirical policy beats accurate classical total cost. |
| M08 | BAD | 68 | Estimators consume 91.8% of measured policy deployment time. |
| M09 | BAD | 68 | 24 calibration fields and only six policy fields do not establish 1% risk or exchangeability. |
| M10 | BAD | 61 | Temporal supervision gives no meaningful broad continuum or endpoint improvement here. |
| M11 | BAD | 60 | Independent/shared tuning selected the same settings; tuning attribution is unresolved. |
| M12 | NA | 93 | Moment identities pass, but mean replacement severely degrades the strongest C3 submodel. |
| M13 | BAD | 59 | Multiband features do not improve main coverage; fixed filtering can spoil Richardson cancellation. |
| M14 | BAD | 82 | Geometry checks pass on bounded numerical controls; no trained geometry-generalization claim. |
| M15 | BAD | 67 | Classical coefficient reuse is useful; large-grid neural throughput crosses over with accuracy still NA. |
| M16 | BAD | 44 | Classical controls remain stronger; current FNO controls do not reproduce a published benchmark. |
| M17 | BAD | 84 | Stress families reveal grid/target sensitivity; transfer cases are too easy to expose guard behavior. |
| M18 | BAD | 69 | Field clusters remain paired; confirmation and policy subsets contain only 24 and six independent fields. |
| M19 | BAD | 82 | Cross-PDE tests exercise numerical correction, not learned TDN transfer; teacher refinement needs repair. |
| M20 | BAD | 65 | Fourth-order diagnostics pass; the strongest selected embedded models are untrained RK4 backbones. |
| M21 | BAD | 81 | Sampled bounds pass; guards never activate in transfer and no global certificate follows. |
| M22 | NA | 93 | Provenance and exports verify; two comparison/timing interpretation defects still require correction. |
| M23 | BAD | 87 | Cubic coefficients pass diagnostics; quadratic+cubic helps accuracy at high cost; cubic alone can regress. |
| C0 | NA | 74 | Screen documents weak headroom; it does not authorize a claim that neural correction is useful. |
| C1 | BAD | 61 | Useful accuracy is largely analytic; learned stacked branches add runtime. |
| C2 | BAD | 62 | Spatial checks improve observed safety, but estimation/rejection costs erase utility. |
| C3 | BAD | 94 | Quadratic+cubic is the useful subcombination; approximate mean replacement damages it. |
| C4 | BAD | 82 | Geometry/portability computations work, but fixed filtering harms error cancellation and guards are untested. |

## Recommended bounded next work

1. **Repair evaluation before another campaign.** Group frontiers by final time; add a mismatch regression; separate warm/cold timing and use repeated randomized paired calls; make transfer teachers actually refine time steps. Reaggregate existing endpoints before spending more GPU time.
2. **Keep the small physical branch.** Compare rank one, analytic quadratic, quadratic+cubic and their simplest shared-computation controls. Require matched-tolerance cost against optimized classical controls and a no-harm comparison against the strongest contained component. Avoid compulsory mean replacement and fixed filtering.
3. **Address the spatial target explicitly.** Use identical continuum-consistent operators/teachers for candidates and comparators, paired grid refinement, rough/high-mode fields and domains with measurable classical work. The present headroom screen already passes 70/72 conditions in one bare DF step.
4. **Test the observed large-workload crossover with accuracy.** Add accepted references on manageable larger grids, fresh fields and measured end-to-end latency. Preserve the useful throughput signal without treating accuracy-NA rows as solver wins.
5. **Strengthen external comparisons only after these gates.** Use a competitive trained FNO and an official public benchmark protocol. Any architecture selected using this full run needs a new independent confirmation cohort; these inspected fields are now development evidence.

Detailed neural, numerical and policy analyses, archive hashes, complete corrected comparison records and all figure lookup tables are linked from the supporting-data README. This review leaves the archived scientific evidence unchanged and identifies which production comparisons require a subsequent correction.
