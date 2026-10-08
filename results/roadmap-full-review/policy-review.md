# Native full roadmap policy review

Policy completed and all 984 detail records passed the trusted local semantic verifier. Recomputed all 20 calibration envelopes directly from 1,440 raw calibration rows and all 82 variant/target summaries. No uploaded code or pickle was executed.

## Scope

24 independent calibration fields; six independent fresh field clusters crossed with two physics settings; grid32, horizon0.27, RMS and maximum targets both2e-4. Four neural families use only seed2600011. 41 variants ×12 parents ×two targets =984 endpoints, not984 independent samples.

## Main observations

- 572 accepted proposals, 412 fallbacks;900 accurate endpoints and84 false accepts. All84 are resolved failures under the numerical-reference refinement estimate, all on the continuum target without a spatial guard. No unresolved accuracy labels. All412 fallback outputs pass.
- Classical adaptive controls pass24/24 target/parent endpoints (12 discrete,12 continuum).
- Spatial guard improves continuum empirical accuracy from156/240 to240/240, with70 accepted and170 fallback outputs. Temporal-only accepts216/240 and makes84 false accepts. These pooled repeated variants are descriptive, not an independent statistical trial.
- No empirical variant achieves both complete matched accuracy and lower total cost. All60 accurate empirical variant/target groups have no positive mean saving;20 other groups fail complete accuracy.

| Empirical track/estimator/spatial guard | Accurate / endpoints | Accepted | False accepts | Fallback | Total cost / classical |
|---|---:|---:|---:|---:|---:|
| continuum / interior_residual / False | 77/120 | 96 | 43 | 24 | 6.859 |
| continuum / interior_residual / True | 120/120 | 20 | 0 | 100 | 18.229 |
| continuum / step_doubling / False | 79/120 | 120 | 41 | 0 | 0.198 |
| continuum / step_doubling / True | 120/120 | 50 | 0 | 70 | 2.741 |
| discrete / interior_residual / False | 120/120 | 96 | 0 | 24 | 215.069 |
| discrete / interior_residual / True | 120/120 | 20 | 0 | 100 | 579.324 |
| discrete / step_doubling / False | 120/120 | 120 | 0 | 0 | 6.515 |
| discrete / step_doubling / True | 120/120 | 50 | 0 | 70 | 80.264 |

## Practical cost comparisons

Cheapest fully accurate empirical continuum variant: cheap-FNO +doubling +spatial guard, router off. It selects initialization, so this is no trained neural gain.12/12 outputs pass,5 accepted and7 fallback. Mean83.375ms, median104.330ms; control mean69.951ms, median64.530ms. Total cost ratio1.192. Accepted-only mean13.844ms versus paired control112.757ms; fallback queries mean133.040ms versus39.375ms. Accepted-only speed excludes the expensive rejected queries and cannot substantiate overall benefit.

Trained Source with the same settings costs84.089ms mean (1.202×control). C2 costs291.123ms (4.162×). The cheapest fully accurate empirical discrete variant (Source bank, doubling, temporal-only, router on) costs5.322ms mean versus2.124ms control (2.505×).

Across984 standalone invocations,460.551s was measured:24.906s proposal,422.619s estimators including spatial discrepancy,12.671s fallback, with explicit small decision/loop overhead. Rejected attempts number2,686 and consume404.297s; rejected time overlaps proposal/estimator time and must not be added again. Estimation is91.8% of total, rejected work87.8%. Calibration adds245.913s offline. Shared preparation/training/calibration/loading totals869.912s. These are numerical times, not scheduler billed hours.

## Apparent break-even is a measurement confound

The only logged finite break-even is49,015 queries for the conformal continuum policy, nominal ratio0.746. It accepts zero neural proposals and uses the identical classical fallback on12/12 queries.97.36% of its apparent saving comes from the first continuum control (217.247ms) versus its repeated fallback (9.890ms). Remaining11 queries give ratio0.99097. The paired bootstrap difference interval spans zero: [-52.240ms,+0.029ms]. This is evidence of baseline-first/variant-last timing and first-call overhead, not neural acceleration or credible amortization. Excluding the first endpoint is a diagnostic, not a replacement benchmark.

## Guarantees and math

Calibration stays disjoint and frozen. Each independent function receives the maximum error/indicator ratio over the declared model/query bank. With24 fields and alpha0.01, the required conformal rank is25, which exceeds24; quantile remains unavailable.99 fields is the arithmetic minimum for a finite rank, conditional on exchangeability, which is not established under the coefficient shift. Refinement uncertainty remains an estimate rather than a certified physical error bound.

The spatial guard is a useful mechanism result: temporal agreement compares an approximation with a refined solve of the same discrete generator and cannot see the FD-to-continuum spatial bias. A fresh2N evaluation adds an independent discrepancy that catches this floor; it remains empirical and expensive.

The interior residual uses a2/4-node quadrature of the flow residual with exp(reaction_rate×remaining_time) weighting. It probes wrong-generator consistency that step doubling can miss, but does not bound between-node reconstruction or quadrature remainders. On this cohort it has no accepted-accuracy advantage without spatial checking (43 continuum false accepts versus41 for doubling), and much larger cost.

Conditional risk is separate from marginal conformal coverage. The best accurate empirical spatial-doubling variant accepts five queries from four independent functions, with zero bad accepted functions. Under iid assumptions its one-sided95% upper risk bound is52.71%, far above1%.299 independent zero-failure accepted functions would be needed for the requested1% bound under those assumptions. No policy seed-generalization statement is supported from one training seed.

## Recommended next work

- Keep independent spatial discrepancy and distinct FD/continuum targets.
- Profile and reduce rejected estimator work before additional training; stage cheap screens before full refinement.
- Prevent neural/amortization claims for identical all-classical fallback paths; compare warmed, randomized, repeated independent-process deployments.
- Repair weak small-grid utility against2.124ms mean adaptive discrete control; any learned benefit must include all estimator/reject/fallback costs.
- Test first promising deployable variant on fresh independent field clusters and remaining paired training seeds; do not treat existing inspected cohort as fresh.
- Expand calibration/risk cohorts only after a reliable useful policy exists; more calibration alone does not prove exchangeability or certify reference errors.
