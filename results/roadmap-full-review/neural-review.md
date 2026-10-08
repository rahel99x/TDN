# Native full roadmap: independent neural/training review

Read-only analysis of the uploaded run `fedora-roadmap-20261008T022526329292Z`. No training was performed, no checkpoint was deserialized, and no original scientific artifact was edited. Aggregates below were recomputed from `train/catalog.json` and `confirm/confirmation_rows.json` and cross-checked against the native log schema/source.

## Main conclusion

The bounded program executed successfully and found a real discrete-target accuracy signal in the tiny learned physical pair branch. It did not establish an efficiency advantage over strong classical methods, or a general learned precompression or continuum advantage. Rank one merits more focused investigation; additional pair rank and stacking it onto the analytic quadratic response mostly add cost here. The best discrete accuracy arm is an initialization-selected classical RK4 backbone, so that accuracy must not be credited to learning.

## Execution and attribution

- 64 tuning trials + 51 final trials completed all 22,980 declared updates. No numerical training failure occurred. Training-stage numerical time was 519.24 s; summed per-trial time was 516.18 s.
- Final checkpoint selection: 42 trained, 9 initialization. All three Cheap FNO and Embedded selections are initializations; the other three are one each of Multiband, C1 rank one and C1 rank two.
- 56 evaluated model instances = 17 neural families × 3 paired seeds + 5 untrained controls. 40,320 finite endpoints, 8,064 all-schedule groups, and no failed rollout. All endpoint teacher references were marked accepted.
- The independent sampling units are 24 field clusters for discrete evidence and 12 for continuum evidence. Physics variants, two grids, five schedules and three training seeds remain repeated paired observations. They do not produce 40,320 independent tests.
- Confirmation has only periodic logistic reaction–diffusion, fixed variance within the declared styles and six sparse field regimes. It includes coefficient shift, but does not reproduce an official FNO paper benchmark.

## Horizon bug in existing matched summaries

The original frontier code groups only by parent/grid/reference track and then minimizes over all schedules. Four schedules end at t=0.27; the fifth ends at t=0.81. A later, more smoothed state is a different target and cannot substitute for the requested earlier endpoint.

- 1,412 of 22,032 original matched rows contain at least one candidate/control horizon mismatch.
- Eligible non-self mismatches: 1,163/14,904 versus Cheap FNO; 1,165/14,904 versus Deep FNO; 1,372/16,811 versus best classical.
- 203 reported Cheap-FNO speed wins and 671 Deep-FNO speed wins compare different endpoints.
- Original any-schedule discrete 2e-6 feasibility: Source 168/288, Rank one 232/288, C1 variants 288/288. Restricted to t=0.27 these become 72/288, 112/288 and 192/288 respectively.
- I rebuilt 44,064 separate-horizon rows from the retained endpoint records, so no GPU rerun is needed to correct this report. These remain post-hoc schedule frontiers, not deployable adaptive policies.
- Across the rebuilt .27/.81 frontiers and all declared target levels, only one of 28,275 jointly feasible neural/classical comparisons wins on latency: postcompression at one t=.81 case with ratio 1.0071 (0.7%). This isolated margin does not establish a practical classical advantage.

## Same-horizon coverage and latency

The following table uses t=0.27 and requires both upper RMS and upper maximum error ≤2e-5. There are 288 discrete method/seed/parent/grid cells and 144 continuum cells; these are repeated measurements over 24/12 independent fields. Speed ratios use only jointly feasible cells; >1 means candidate faster. Classical timings are reused across paired neural seeds. No training amortization is included.

| Family | Trained selections /3 | Discrete feasible /288 | Continuum feasible /144 | Cheap-FNO / candidate latency ratio, discrete | Best-classical / candidate latency ratio, discrete |
|---|---:|---:|---:|---:|---:|
| source | 3 | 192 | 34 | 1.025× | 0.322× |
| rank1 | 3 | 288 | 48 | 0.557× | 0.217× |
| rank2 | 3 | 288 | 48 | 0.413× | 0.163× |
| source_postcompression | 3 | 192 | 36 | 0.917× | 0.286× |
| source_trust | 3 | 192 | 34 | 0.802× | 0.250× |
| source_consistency | 3 | 192 | 34 | 1.025× | 0.322× |
| source_df_loss | 3 | 192 | 34 | 1.023× | 0.319× |
| source_df_loss_shared | 3 | 192 | 34 | 1.024× | 0.321× |
| source_multiband | 2 | 192 | 33 | 0.743× | 0.231× |
| cheap_fno | 0 | 192 | 33 | 1.000× | 0.314× |
| deep_fno | 3 | 192 | 36 | 0.539× | 0.169× |
| direct_fno | 3 | 0 | 0 | NA | NA |
| c1_rank0 | 3 | 288 | 36 | 0.505× | 0.256× |
| c1_rank1 | 2 | 288 | 34 | 0.300× | 0.153× |
| c1_rank2 | 2 | 288 | 34 | 0.252× | 0.129× |
| c2_rank2 | 3 | 288 | 36 | 0.236× | 0.101× |
| source_embedded | 0 | 288 | 30 | 0.267× | 0.085× |

The established classical control set reaches 288/288 discrete and 42/144 continuum cells at this endpoint/target (continuum count repeats 14/48 controls across three neural seeds). Source has ~2.5% descriptive latency advantage over Cheap FNO on the mutually feasible discrete subset and ~1.90× over Deep FNO, but its own coverage is no better, and Cheap FNO selected its untrained zero correction. This is narrow implementation/cost evidence, not a trained FNO superiority claim.

## Mechanism attribution and shortcomings

| Mechanism | What the raw paired results support | What they do not support |
|---|---|---|
| Learned physical pair | Rank one has 43 parameters, is trained in all three seeds, improves RMS on 1395/1440 same-schedule discrete comparisons versus Cheap FNO, median 2.344× RMS reduction. | It is ~1.82× slower at the same schedule; continuum paired median error ratio is 0.999989, essentially no broad gain. |
| More pair rank | Rank two has 54 parameters and comparable discrete accuracy/coverage to rank one. | Rank two costs ~1.35× rank one and does not show a consistent accuracy advantage worth this cost. |
| Source before compression | Source has median discrete RMS ratio 1.00215 versus the matched postcompression arm and ~1.1155× latency advantage (the latter pays extra projection work). | A 0.2% median discrete error gain, continuum median ratio 1.0, and mixed peak-error outcomes do not show a decisive learned information-preservation advantage. |
| Analytic quadratic + source/pair C1 | C1 improves discrete errors substantially over DF/weak FNO controls. | Its median paired RMS ratio versus untrained df_quad2 is 1.0 at ranks 0/1/2, with ~1.39×/2.34×/2.79× latency penalty; much of the useful effect is analytic. |
| Embedded RK | All discrete endpoints meet 2e-5; 1344/1440 meet 2e-6. All-schedule strict passes are 264/288. | All three selected checkpoints are update zero: the stable subdivided RK4 accounts for its strong accuracy. It is slower, and continuum strict all-schedule coverage remains zero. |
| Enhanced loss and independent tuning | All enhanced-loss trials ran; the shared-setting control was retained. | Independent tuning happened to choose the same setting as Source, and shared/independent enhanced-loss accuracy is identical. This run cannot attribute a benefit to different optimizer settings. |
| Temporal consistency | Source and consistency selected updates 40/60/80; there is only a two-endpoint gain at the middle discrete target. | Extra training work did not produce a meaningful endpoint or continuum advantage. |
| Trust / multiband | Both were trained and tested without numerical failure; multiband retains one initialization. | Their aggregate discrete and continuum coverage does not improve materially on Source. |
| Direct FNO | All three final arms trained and reached update 300. | 0/1440 discrete and 0/720 continuum endpoints pass even 2e-4. This is an inadequately competitive baseline for any claim of beating the published FNO method. |

## Grid target and hard regimes

Rank one improves the discrete hard cases: at 2e-5 its high-pair regime passes 132/240 endpoints versus Cheap FNO 24/240, rough 168/240 versus 120/240, and near-Nyquist 156/240 versus 72/240. Each denominator consists of four independent fields repeated across grids/physics/seeds/schedules, not 240 independent fields.

The continuum target is different from the finite-difference discretization the hybrid model learns. Embedded median discrete RMS is approximately 3.09e-8 at grid32 and 3.87e-8 at grid64, while continuum medians are 4.21e-5 and 1.06e-5. The ~4× continuum decrease on doubled grid is consistent with a spatial-discretization bottleneck. This is not fixed by reducing temporal errors further. Teacher uncertainty is roughly 4.91e-9 median RMS on the continuum track, much smaller than these errors.

Across all five schedules, at the tightest 2e-6 tolerance every trained family has 0/144 continuum groups passing all schedules. Long-final-time successes cannot substitute for intermediate-time accuracy. Classical controls also have 0/48 continuum all-schedule strict passes.

## Recommended bounded follow-up

1. Correct frontier grouping by terminal horizon and regenerate summaries from the existing endpoints first; add a regression that prohibits cross-horizon cost comparisons.
2. Focus on rank one vs analytic df_quad2/GL3, retaining C1 rank-zero as attribution control. Optimize transform/launch cost and measure equal-accuracy cost before adding more branches.
3. Separate continuum-consistent spatial correction from discrete time correction. Preserve identical physical targets and grids between methods; use high-mode/rough cases and paired grid refinement.
4. Calibrate a credible trained FNO baseline on sufficient independent training fields/steps and a public benchmark protocol before claiming superiority over recent papers. Keep bounded exploratory performance and official-paper reproduction distinct.
5. Freeze a new independent confirmation cohort for any architecture selected after reviewing these results; the current full cohort is now development evidence.

## Artifacts

- `neural-review.json`: recomputed trial counts, endpoint/pass/error/cost summaries by family/track/regime/schedule/physics/grid, same-schedule paired comparisons, and corrected t=.27 frontiers.
- `neural-fixed-horizon-review.json`: mismatch inventory, all 44,064 corrected fixed-horizon comparisons and conditional summaries.
- `analyze-neural.py` and `analyze-neural-horizons.py`: reproducible JSON-only offline analysis scripts.
