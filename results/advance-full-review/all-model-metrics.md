# Absolute metrics for every Advance family

[all-model-metrics.csv](all-model-metrics.csv) contains all **60 track × family × training-count × final-time summaries**, covering all 13 families. These are absolute primary-schedule measurements from the sealed native run `fedora-advance-full-20261011T001715`; they are not a universal leaderboard.

## Denominators and interpretation

Each row includes 64 independent fields, both 32² and 64² grids, and one declared two-step schedule: `[.04, .08]` at final time .12 or `[.08, .16]` at .24. A frozen analytic/classical method therefore has 128 endpoint observations. A learned model includes three training seeds and has 384 observations. Repeated grids and seeds do not multiply the number of independent fields. `training_seeds=0` denotes a deterministic frozen control; `train_count=0` means no training. The 16-field variants are secondary controls, not independent replications of sample complexity.

- RMS geometric mean, median, observed p90 and maximum summarize endpoint RMS errors. The balanced query design gives equal weight to each field in the geometric mean. Observed p90 is not a population quantile guarantee, and none of these columns is a confidence interval.
- `sampled_spatial_max_error_worst_endpoint` is the largest recorded spatial maximum error over every endpoint in that row. It is not a bound between grid points or on unseen fields.
- `mean_bias_mse_fraction_mean` averages `(spatial mean error / endpoint RMS error)²` over queries. For each endpoint, squared RMS equals squared mean error plus squared centered RMS. The reported mean of fractions is **not** the ratio of pooled mean-squared errors. Lower fractions do not by themselves mean lower absolute error.
- Endpoint coverage requires accepted reference, finite prediction, and uncertainty-adjusted RMS **and** maximum error each ≤2e−5. Joint field coverage requires every grid and every trained seed for that field to pass. Field coverage, not the number of successful queries, protects the solver-cost claim.
- `median_warm_complete_call_seconds` measures all primary-schedule queries, including inaccurate ones. It excludes training, reference generation, process startup, model loading and transfers outside the callable. It is **not accuracy-qualified time to solution**. Use the separately reported frozen validation-selected frontiers and campaign accounting for that question.
- The zero physical-bound and energy-increase columns describe sampled endpoints only. `energy_increase_above_1e_minus12_endpoint_count` uses a **1e−12** threshold, and `rms_below_1e_minus7_endpoint_count` counts RMS **below 1e−7**.

## Tradeoffs that a single RMS summary hides

At T=.24 the fitted two-basis model improves geometric-mean RMS over the historical conditioned model, but its **worst spatial maximum is worse**: discrete 4.10e−4 versus 1.99e−4; continuum 4.32e−4 versus 2.50e−4. A typical-error improvement does not establish worst-case robustness.

GL4 has good typical accuracy but still has outliers: discrete T=.12 geometric-mean RMS is 2.70e−7, yet the worst spatial maximum is 2.27e−5, just above the 2e−5 threshold. Four of 128 endpoint queries fail the joint gate. The nested quadratic+cubic control is especially accurate on the discrete track (all 128 primary queries pass at T=.12), while its continuum T=.12 worst maximum reaches 1.18e−4. Separate maximum-error and spatial-track reporting remains necessary.

Mean bias accounts for roughly **84–86%** of mean per-query error fractions in the legacy FNO, **68–74%** in the zero/scaled FNO controls, **40–46%** in the 64-field two-basis model, and **23–38%** in GL4. The safer FNO controls mostly preserve their initial physical backbone; these fractions diagnose the tested implementations and selected checkpoints, not a fundamental limit of Fourier neural operators.

ETDRK4 reaches a worst spatial error near .132 at T=.24 on this deliberately coarse two-step schedule. Its validation-selected schedules can be much more accurate and fast. Judging it only by the equal-step table would discard the very scheduling freedom needed for a fair solver comparison.

## Additional metrics and unmeasured scope

The canonical endpoint rows also retain low/middle/high spectral RMS contributions, signed/absolute mean error, centered RMS, variance and gradient errors, Lyapunov energies, sampled spatial extrema and physical moments. Those are retained in the compact archive; this CSV deliberately summarizes the most interpretable absolute endpoint measures. Low/middle/high bands use fixed physical frequencies, not grid-relative bins.

Primary rows are final endpoints of **two-step compositions**, not long-trajectory or dense-output error curves. They contain no full spatial error maps, intervening-time error history or learned endpoint tangent response. The XT prototype separately tests tangent-flow identities and coarse-information ambiguity; those results do not validate each learned model's Jacobian. `mean_balance_residual` is unavailable because no actual endpoint time derivative was supplied; an expected physical mean rate is not a measured derivative. GPU timing/memory probes and scaling rows exist separately. Kernel attribution, true process-cold latency and energy/currency costs are not established by this table.

Provenance: generated from the independent `primary_method_summary` reanalysis of exact original `aggregate/endpoint_rows.json`, without rerunning inference or changing selection. All 60 CSV rows were round-trip checked against that summary. Reconstruction script: `.runtime/advance-full-native-analysis/statistics-publish-table.py`.
