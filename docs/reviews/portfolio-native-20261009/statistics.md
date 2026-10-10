# Statistical and numerical review of the native portfolio cohort

Run: `fedora-portfolio-20261009T183307014599Z`. Source: exact, hash-verified retained `aggregate/endpoint_rows.json`, `aggregate/comparisons.json`, `aggregate/claims.json`, and sealed protocol. This review adds descriptive analyses; it does not retroactively alter primary claims or original artifacts.

## Findings that change the interpretation

1. The primary learned contribution is **not established beyond correctly normalized analytic quadrature**. All eight predeclared normalized-control comparisons (two tracks × two training sizes × two horizons) are BAD. At 32 fields the fixed/conditioned RMS ratios are 0.966/0.971 for discrete t=0.12/0.24 and 0.947/0.896 for continuum; values below one favor the analytic rule. This is failure to demonstrate improvement, not a proof that no learned correction can help.
2. Normalization and physical representation explain much more than the learned conditioner. Across shared schedules at 32 fields or frozen controls, normalizing historical half weights reduces paired RMS by 3.00× discrete / 2.68× continuum; preserving interactions before input compression gives 2.23× / 2.08×. Increasing frozen quadrature from two to four nodes gives 1.99× / 2.13× while increasing same-schedule cost by about 34% / 18%. These are stronger controlled mechanisms than a generic claim that learning causes the benefit.
3. There is real but narrow success against the **locally trained hybrid FNOs**: every primary n=32 FNO RMS comparison is GOOD, by 11.6–43.9×, and discrete t=0.12 validation-locked cost is 1.67× / 2.37× lower than small/standard FNO at observed 24/24 candidate field coverage. It remains a bounded optimizer/data/control comparison, not a reproduction or defeat of published FNO benchmarks.
4. The conditioned model does have a tail-coverage advantage in one useful sense: at n=32 its discrete validation-selected schedules pass all seeds/grids on 24/24 fields at both endpoints; fixed quadrature passes 24/24 then 23/24. An average-error failure does not erase this narrower robustness observation. The candidate pays more than analytic/classical controls and needs further independent safety evidence.
5. Complete solver superiority is not established. Its discrete locked cost is 1.23–1.24× that of fixed quadrature, 2.91× DF at t=0.12, and 3.22× / 4.31× ETDRK4 at t=0.12/0.24 on jointly eligible pairs. Continuum locked coverage is only 20/24 and 21/24, below the declared 95% requirement, so apparent conditional speedups cannot be promoted to a successful deployment claim.
6. All 101,376 endpoint rows and 67,584 intermediate observations are finite and have accepted references. Numerical finiteness is not accuracy or physical admissibility. All 1,887 endpoint and 454 intermediate [0,1] violation rows belong to direct FNO, whose maximum is 1.54375. Re-evaluating every frozen confirmation input at 32 and 64 gives minimum 0.0411549 and maximum 0.8003631, so these violations are not inherited from invalid initial data. No other tested family violates the interval in this cohort.

## Every model, with the same primary schedule set

Ownership labels: **Ours** covers the learned quadrature/conditioner families and historical TDN. **Theirs (local adaptation)** covers FNO controls, and **Theirs (classical)** covers DF/RF/ETDRK4. **Analytic control** distinguishes unfitted quadrature and quadratic+cubic formulas; their inclusion is not a novelty claim.


These tables use only the four declared primary schedules, grids 32/64 and 24 independent fields. Learned rows use all three paired seeds at the 32-field training size (576 endpoint observations); frozen controls use one fixed model (192). **Lower RMS, maximum error and latency are better; higher joint passes are better. Error fractions describe error composition and are not independently better when lower.** Pass requires both RMS and maximum error plus recorded reference uncertainty ≤2e-5. Repeated observations are not independent fields. These same-schedule tables are not accuracy-qualified cost frontiers: a classical method may lose on an overly coarse common schedule yet win after choosing its own accurate step count. Mean fraction is median(mean_error²/RMS²).

### Discrete FD/nodal target

| Family | Joint passes | Median RMS | Median maximum error | P99 maximum error | Median mean-error fraction | Median warm ms |
|---|---:|---:|---:|---:|---:|---:|
| **Analytic control** · `analytic_quad_cubic` | 191/192 | 1.21e-07 | 3.57e-07 | 1.55e-05 | 35.1% | 22.185 |
| **Ours** · `band_gain` | 525/576 | 7.56e-07 | 1.66e-06 | 0.000171 | 50.0% | 6.834 |
| **Ours** · `conditioned_rich` | 528/576 | 5.93e-07 | 1.46e-06 | 0.000158 | 39.1% | 6.580 |
| **Theirs, classical** · `df` | 122/192 | 7.72e-06 | 1.14e-05 | 0.000318 | 76.9% | 0.930 |
| **Theirs, local FNO** · `direct_fno` | 0/576 | 0.0436 | 0.0962 | 0.423 | 57.3% | 4.384 |
| **Theirs, classical** · `etdrk4` | 106/192 | 6.71e-06 | 9.28e-06 | 0.0031 | 95.0% | 1.498 |
| **Theirs, local FNO** · `fno_small` | 182/576 | 3.01e-05 | 3.94e-05 | 0.000378 | 98.8% | 3.970 |
| **Theirs, local FNO** · `fno_standard` | 270/576 | 1.65e-05 | 2.18e-05 | 0.000324 | 97.1% | 5.643 |
| **Ours, historical** · `historical_half` | 147/192 | 4.01e-06 | 5.94e-06 | 0.000253 | 73.7% | 6.059 |
| **Ours** · `quad2_amplitude` | 519/576 | 7.73e-07 | 1.67e-06 | 0.000179 | 55.7% | 5.077 |
| **Ours** · `quad2_conditioned` | 526/576 | 6.62e-07 | 1.56e-06 | 0.000157 | 36.1% | 6.304 |
| **Ours** · `quad2_conditioned_full` | 532/576 | 6.26e-07 | 1.5e-06 | 0.000157 | 49.2% | 5.906 |
| **Analytic control** · `quad2_fixed` | 173/192 | 4.22e-07 | 1.16e-06 | 0.00018 | 15.1% | 5.065 |
| **Analytic control** · `quad2_full` | 175/192 | 4.02e-07 | 1.1e-06 | 0.00018 | 20.6% | 4.693 |
| **Analytic control** · `quad2_input` | 145/192 | 2e-06 | 3.28e-06 | 0.000317 | 53.6% | 5.457 |
| **Ours** · `quad2_joint` | 522/576 | 8.13e-07 | 1.75e-06 | 0.000175 | 57.6% | 5.073 |
| **Ours** · `quad2_linear` | 513/576 | 5.84e-07 | 1.32e-06 | 0.000135 | 26.9% | 6.150 |
| **Ours** · `quad2_nodes` | 522/576 | 7.61e-07 | 1.66e-06 | 0.000173 | 56.4% | 5.076 |
| **Ours** · `quad4_conditioned` | 546/576 | 3.35e-07 | 8.71e-07 | 5.44e-05 | 17.5% | 8.057 |
| **Analytic control** · `quad4_fixed` | 183/192 | 2.2e-07 | 6.38e-07 | 5.31e-05 | 8.1% | 6.812 |
| **Analytic control** · `quad4_full` | 187/192 | 2.05e-07 | 5.85e-07 | 4.97e-05 | 11.7% | 6.438 |
| **Ours** · `residual_quad2` | 524/576 | 7.34e-07 | 1.66e-06 | 0.000174 | 49.7% | 6.279 |
| **Theirs, classical** · `rf` | 102/192 | 9.85e-06 | 1.61e-05 | 0.00191 | 80.1% | 1.373 |

### Spatially refined continuum-estimate target

| Family | Joint passes | Median RMS | Median maximum error | P99 maximum error | Median mean-error fraction | Median warm ms |
|---|---:|---:|---:|---:|---:|---:|
| **Analytic control** · `analytic_quad_cubic` | 190/192 | 1.47e-07 | 4.37e-07 | 2.03e-05 | 54.2% | 108.117 |
| **Ours** · `band_gain` | 528/576 | 1.18e-06 | 2.07e-06 | 0.000162 | 69.3% | 30.877 |
| **Ours** · `conditioned_rich` | 534/576 | 1.1e-06 | 2.05e-06 | 0.000156 | 65.7% | 30.647 |
| **Theirs, classical** · `df` | 125/192 | 7.44e-06 | 1.06e-05 | 0.000274 | 79.2% | 18.918 |
| **Theirs, local FNO** · `direct_fno` | 0/576 | 0.0504 | 0.094 | 0.421 | 53.5% | 4.443 |
| **Theirs, classical** · `etdrk4` | 104/192 | 9.17e-06 | 1.15e-05 | 0.0031 | 95.7% | 5.805 |
| **Theirs, local FNO** · `fno_small` | 196/576 | 2.69e-05 | 3.67e-05 | 0.000288 | 98.7% | 22.065 |
| **Theirs, local FNO** · `fno_standard` | 240/576 | 1.97e-05 | 2.72e-05 | 0.000305 | 97.9% | 23.894 |
| **Ours, historical** · `historical_half` | 150/192 | 3.97e-06 | 6.32e-06 | 0.000175 | 76.0% | 30.140 |
| **Ours** · `quad2_amplitude` | 522/576 | 1.24e-06 | 2.16e-06 | 0.000164 | 67.5% | 28.961 |
| **Ours** · `quad2_conditioned` | 534/576 | 1.22e-06 | 2.12e-06 | 0.000155 | 65.4% | 30.373 |
| **Ours** · `quad2_conditioned_full` | 534/576 | 1.21e-06 | 2.01e-06 | 0.000156 | 68.9% | 30.008 |
| **Analytic control** · `quad2_fixed` | 172/192 | 7.63e-07 | 1.53e-06 | 0.000161 | 40.8% | 28.893 |
| **Analytic control** · `quad2_full` | 172/192 | 7.43e-07 | 1.34e-06 | 0.000161 | 49.0% | 28.541 |
| **Analytic control** · `quad2_input` | 146/192 | 2.07e-06 | 3.05e-06 | 0.000274 | 71.0% | 29.329 |
| **Ours** · `quad2_joint` | 522/576 | 1.25e-06 | 2.2e-06 | 0.000161 | 68.0% | 28.883 |
| **Ours** · `quad2_linear` | 510/576 | 9.95e-07 | 1.82e-06 | 0.000143 | 64.0% | 30.218 |
| **Ours** · `quad2_nodes` | 522/576 | 1.14e-06 | 2.08e-06 | 0.000161 | 68.6% | 28.919 |
| **Ours** · `quad4_conditioned` | 544/576 | 5.72e-07 | 1.17e-06 | 5.17e-05 | 36.7% | 35.889 |
| **Analytic control** · `quad4_fixed` | 186/192 | 3.19e-07 | 8.22e-07 | 5.64e-05 | 18.6% | 34.460 |
| **Analytic control** · `quad4_full` | 186/192 | 2.71e-07 | 6.31e-07 | 5.59e-05 | 22.4% | 34.030 |
| **Ours** · `residual_quad2` | 522/576 | 1.1e-06 | 1.97e-06 | 0.000163 | 67.3% | 30.325 |
| **Theirs, classical** · `rf` | 100/192 | 1.14e-05 | 1.73e-05 | 0.00196 | 83.8% | 36.584 |

## Controlled effects and uncertainty

Ratios below are **control RMS / candidate RMS**: >1 favors the candidate. Intervals are posthoc descriptive 95% field-and-seed crossed bootstrap intervals; there are 24 independent fields, not the row count. They are not multiplicity-adjusted. Frozen-control effects use one deterministic model; learned effects use all three seeds at n=32. Both endpoints, grids and all shared schedules are averaged inside each field/seed cell.

| Candidate vs control | Discrete ratio [95% interval] | Continuum ratio [95% interval] |
|---|---:|---:|
| `quad2_fixed` vs `historical_half` | 2.997 [2.244, 4.111] | 2.685 [2.009, 3.607] |
| `quad2_fixed` vs `quad2_input` | 2.230 [1.484, 3.428] | 2.083 [1.390, 3.162] |
| `quad4_fixed` vs `quad2_fixed` | 1.992 [1.395, 2.919] | 2.135 [1.403, 3.320] |
| `quad2_full` vs `quad2_fixed` | 1.102 [1.011, 1.244] | 1.050 [1.002, 1.115] |
| `quad4_full` vs `quad4_fixed` | 1.186 [1.021, 1.429] | 1.073 [1.006, 1.166] |
| `analytic_quad_cubic` vs `quad4_full` | 1.685 [1.404, 2.065] | 1.433 [1.245, 1.700] |
| `quad2_conditioned` vs `quad2_fixed` | 0.969 [0.850, 1.107] | 0.921 [0.799, 1.054] |
| `quad2_amplitude` vs `quad2_fixed` | 0.903 [0.802, 1.023] | 0.907 [0.796, 1.041] |
| `quad2_nodes` vs `quad2_fixed` | 0.912 [0.806, 1.037] | 0.940 [0.830, 1.075] |
| `quad2_joint` vs `quad2_fixed` | 0.898 [0.786, 1.025] | 0.905 [0.791, 1.042] |
| `quad2_linear` vs `quad2_fixed` | 0.912 [0.775, 1.060] | 0.875 [0.759, 1.011] |
| `conditioned_rich` vs `quad2_conditioned` | 1.014 [0.976, 1.054] | 1.010 [0.972, 1.051] |
| `residual_quad2` vs `quad2_conditioned` | 0.970 [0.917, 1.019] | 1.048 [0.986, 1.121] |
| `band_gain` vs `quad2_conditioned` | 0.934 [0.885, 0.970] | 0.984 [0.935, 1.026] |

The richer conditioner gives only ~1% paired improvement over the current conditioner, with intervals spanning no gain; it is not a justified increase in complexity. Band gain is about 7% worse in discrete RMS and also slower. The analytic quadratic+cubic rule improves on full-output four-node quadrature by 1.68× / 1.43× RMS, but costs about 3.41× / 3.15× more on the same schedule. This is an accuracy/cost tradeoff, not universal superiority. Output compression offers no measured inference win in the current implementation: removing it makes frozen Gauss-2 roughly 8.3% / 1.3% faster and modestly more accurate; it must earn a role in a different storage/reuse workload if retained.

## Selection and deployment failures versus architecture failures

For each entry below, the fraction is the number of **fields on which every evaluated seed/grid passes**, out of 24. Locked = validation-selected fixed schedule; oracle = cheapest accurate schedule selected after inspecting confirmation truth. The oracle is a diagnostic ceiling, never deployable evidence.

| Family | Target track | t | Locked fields | Oracle fields |
|---|---|---:|---:|---:|
| **Ours** · `quad2_conditioned` | discrete | 0.12 | 24/24 | 24/24 |
| **Ours** · `quad2_conditioned` | discrete | 0.24 | 24/24 | 24/24 |
| **Ours** · `quad2_conditioned` | continuum | 0.12 | 20/24 | 24/24 |
| **Ours** · `quad2_conditioned` | continuum | 0.24 | 21/24 | 24/24 |
| **Analytic control** · `quad2_fixed` | discrete | 0.12 | 24/24 | 24/24 |
| **Analytic control** · `quad2_fixed` | discrete | 0.24 | 23/24 | 23/24 |
| **Analytic control** · `quad2_fixed` | continuum | 0.12 | 22/24 | 24/24 |
| **Analytic control** · `quad2_fixed` | continuum | 0.24 | 22/24 | 23/24 |
| **Analytic control** · `quad4_full` | discrete | 0.12 | 20/24 | 24/24 |
| **Analytic control** · `quad4_full` | discrete | 0.24 | 18/24 | 24/24 |
| **Analytic control** · `quad4_full` | continuum | 0.12 | 20/24 | 24/24 |
| **Analytic control** · `quad4_full` | continuum | 0.24 | 18/24 | 24/24 |
| **Analytic control** · `analytic_quad_cubic` | discrete | 0.12 | 22/24 | 24/24 |
| **Analytic control** · `analytic_quad_cubic` | discrete | 0.24 | 21/24 | 24/24 |
| **Analytic control** · `analytic_quad_cubic` | continuum | 0.12 | 22/24 | 24/24 |
| **Analytic control** · `analytic_quad_cubic` | continuum | 0.24 | 21/24 | 24/24 |
| **Theirs, classical** · `df` | discrete | 0.12 | 22/24 | 22/24 |
| **Theirs, classical** · `df` | discrete | 0.24 | 0/24 | 20/24 |
| **Theirs, classical** · `df` | continuum | 0.12 | 22/24 | 22/24 |
| **Theirs, classical** · `df` | continuum | 0.24 | 0/24 | 20/24 |
| **Theirs, classical** · `etdrk4` | discrete | 0.12 | 20/24 | 24/24 |
| **Theirs, classical** · `etdrk4` | discrete | 0.24 | 22/24 | 22/24 |
| **Theirs, classical** · `etdrk4` | continuum | 0.12 | 20/24 | 24/24 |
| **Theirs, classical** · `etdrk4` | continuum | 0.24 | 22/24 | 22/24 |
| **Theirs, local FNO** · `fno_small` | discrete | 0.12 | 22/24 | 22/24 |
| **Theirs, local FNO** · `fno_small` | discrete | 0.24 | 0/24 | 10/24 |
| **Theirs, local FNO** · `fno_small` | continuum | 0.12 | 22/24 | 22/24 |
| **Theirs, local FNO** · `fno_small` | continuum | 0.24 | 0/24 | 18/24 |
| **Theirs, local FNO** · `fno_standard` | discrete | 0.12 | 22/24 | 22/24 |
| **Theirs, local FNO** · `fno_standard` | discrete | 0.24 | 0/24 | 18/24 |
| **Theirs, local FNO** · `fno_standard` | continuum | 0.12 | 22/24 | 22/24 |
| **Theirs, local FNO** · `fno_standard` | continuum | 0.24 | 0/24 | 20/24 |

Continuum conditioned coverage rises from 20/24 and 21/24 locked to 24/24 oracle: accurate options exist, but the fixed validation policy does not transfer safely to all shifted cases. Four-node full-output quadrature similarly has 24/24 oracle coverage yet only 20/24 and 18/24 locked. Conversely, DF/FNO at t=0.24 can have **no globally feasible validation-selected schedule** even though a posthoc per-field option is feasible for many fields; NA here is not the claim that the architecture cannot solve any field. This distinguishes model quality, menu sufficiency and deployment schedule design.

## Data efficiency, generalization and neglected diagnostics

* Increasing training from eight to 32 fields improves conditioned paired RMS by 1.084× discrete [1.026,1.176] and 1.119× continuum [1.053,1.209], but changes all-schedule joint passes only 1531→1535/1728 and 1550→1559/1728. This is a modest data benefit, not a scaling law or a reason to blindly multiply updates. Nested fields add phases, means and physics coverage as well as count.
* FNO data-size effects are unstable: small FNO paired RMS ratio(8/32) is 0.891 discrete and 0.850 continuum; standard FNO 1.197 and 0.802; all four intervals include one. More fields do not cure the bounded optimization issue by themselves.
* Shifted coefficients expose a real weakness. Conditioned all-schedule passes on the eight shifted fields are 429/576 discrete and 426/576 continuum, versus 1106/1152 and 1133/1152 on the 16 declared-distribution fields. These are repeated-query counts and the shift also changes physical difficulty; they are not independent binomial trials or a causal pure-shift experiment.
* Difficult signed-interaction cases still limit accuracy: conditioned passes 138/216 discrete and 157/216 continuum high-pair observations, versus low-frequency 216/216 and 212/216. Only three independent fields per regime support these percentages. Four-node full-output quadrature markedly helps high-pair and near-Nyquist errors, identifying quadrature/compression headroom more clearly than richer conditioning.
* Mean error is a distinct failure channel: primary-schedule median squared-error fraction from mean is ~36%/65% for conditioned discrete/continuum, ~15%/41% fixed quadrature, and ~97–99% for hybrid FNOs. Logistic reaction does **not** conserve the mean; the appropriate target is its evolving mean, not zero mean drift or conservation. Fixing a mean term must be paired with centered and total errors to avoid destructive error cancellation.
* Maximum error cannot be inferred from RMS. The pass criterion charges both; conditional average RMS gains can coexist with bad tails. Every-model median maximum and P99 maximum are listed above; worst observed rows including field identities are retained in `statistics-followup.json`.
* Spectral metric caveat: the implementation defines low frequency by radial wave number ≤N/4. Thus the physical cutoff changes from eight at N=32 to 16 at N=64. A mode near 14 is counted high on one grid and low on the other. Current low/high values are correct for relative-grid bands, but cross-grid spectral conclusions need fixed physical-frequency bands.
* Short-rollout audit is real (67,584 independently referenced intermediate observations), but horizons end at 0.24 and have at most eight steps. No long-time stability, invariant measure, many-refresh amortization or robustness over hundreds/thousands of steps is established.
* No-clipping bounds compliance in this cohort is useful but not a proof of positivity/maximum-principle preservation under arbitrary fields, steps or coefficients. Direct FNO fails it in the actual finite sample; the other models only have observed compliance.

## Statistical and fairness limits

* 101,376 endpoint rows are 24 independent continuous fields crossed with grids, schedules, data sizes, methods and three training seeds. Regime slices have only three fields. Original 14,752 pairwise subgroup comparisons are exploratory and unadjusted; visually nonoverlapping comparisons are not a familywise result.
* A 24/24 observed success rate is not proof of ≥95% deployment reliability: even under an ideal IID field model the one-sided exact 95% lower bound is about 88.3%. At least 59 independent zero-failure fields would be needed to put that bound above 95%; broader deployment heterogeneity can demand more. The current 95% gate is an empirical-cohort requirement.
* Fixed/learned primary effect intervals average repeated schedules and grids inside a field/seed cell. This is appropriate pairing, but it averages heterogeneous easy and hard effects. Always retain the grid/regime/horizon failure map and effect-size denominator.
* Primary accuracy uses paired geometric RMS ratios, a ≥1.1 effect, lower interval >1 and ≤5 percentage-point joint-pass regression. Primary cost uses ≥1.2 speedup, lower interval >1 and ≥95% all-seed/grid field coverage. Failing one is not a proof of mathematical incorrectness; conversely, passing all software checks does not establish any of these claims.
* Cost ratios are conditional on **both** methods having an eligible frozen schedule; record candidate/control field coverage beside them. Incomplete crossed designs receive no interval. They are not all-workload speedup or fallback-inclusive economics.
* Validation schedule selection uses batched per-field timing whereas confirmation measures single-field latency. Selection transfers unchanged, which prevents truth leakage, but validation should reflect the intended deployment batch and workload if the goal is efficient scheduling.
* The fields comprise eight fixed two/three-mode templates, a small mean/variance/physics set and fresh phases. This is far narrower than arbitrary rough fields, large-amplitude interfaces, anisotropy, boundary conditions, forced dynamics or a full established neural-operator benchmark.
* The present direct FNO learns a substantially harder task without the exact physical split; count it as a deliberately different-information control, not proof that generic neural operators cannot work. Fairness requires information, data, actual tuning/training compute and matched-tolerance cost views separately.
* Accepted references are refinement-based uncertainty estimates, not rigorous continuum error certificates. Both error norms charge the recorded uncertainty; source observations preserve separate FD/nodal and continuum-estimate targets. Relative errors, energy estimates, monetary charges, precise long-time failure rates and actual compiled deployment speed are not supplied by these endpoint metrics.

## What should continue

Prioritize an analytic interaction rule (normalized four-node/full-output, with quadratic+cubic as an accuracy control), reliable schedule selection and implementation profiling. Preserve the conditioned model as a tail-coverage/compact-adaptation candidate, but require it to beat those stronger matched controls on a newly declared cohort. Retire automatic increases in conditioner richness/bands pending a diagnostic that demonstrates missing identifiable information. Retain the separate speculative representations; this cohort neither validates nor invalidates their different workload claims.

Artifacts: `statistics-review.json` contains every-family/data-size/grid/regime/distribution/horizon/seed metric summary, all 96 unchanged primary claims, frontier coverage, and selected whole-cohort contrasts; `statistics-followup.json` contains paired data-efficiency, violations, worst cases and conditional savings; `statistics-endpoints-compact.csv.gz` is an exact-value compact projection of all 101,376 endpoint rows; `statistics-all-pair-subgroups.json.gz` retains every original subgroup comparison.
