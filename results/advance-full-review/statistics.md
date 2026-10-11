# Independent confirmation and statistical review — native Advance full

Run: `fedora-advance-full-20261011T001715`. Sources: exact original `aggregate/endpoint_rows.json`, `aggregate/locked_frontiers.json`, `aggregate/claims.json`, and sealed protocol. This review does not alter models, schedules, thresholds or original outcomes. Reproduce by running `statistics-analysis.py` with the adjacent `review_io.py` and verified object store.

## Assessment

The experiment demonstrates a modest learned calibration benefit over its immediate fixed filtered-cubic parent at the longer final time. It does **not** demonstrate an accuracy-qualified solver-cost advantage, nor superiority over the strongest analytic alternative. The fixed four-node quadratic rule is more accurate than the fitted candidate on the aggregate same-schedule comparison in every track/time cell, and it often provides better accuracy-qualified cost. The expensive nested quadratic+cubic rule is still more accurate. These distinctions must precede any count of successful claims.

## Verification and denominators

All 20 primary accuracy claims and all 20 primary cost claims were independently reconstructed, including exact 1,000-replicate crossed-field/training-seed bootstrap endpoints. No discrepancies were found. There are 31,488 measured endpoint rows, 15,872 primary-schedule endpoint rows, 47,616 locked-frontier rows (including unavailable schedules), 62 model instances and 13 families. These are **64 independent fields**, not 31,488 independent observations. Each learned primary comparison contains 384 paired queries = 64 fields × 3 seeds × 2 grids; each field/seed averages its two grid queries before bootstrap. The two final times are separate claims. Frozen controls are correctly paired against each seed; they do not become three independently trained controls.

All measured endpoints were finite, used accepted references, had no sampled physical-interval violations, and no positive endpoint energy increase above 1e-12. This is successful finite checking, not a proof of stability or between-grid bounds. No numerical failure was removed. An accuracy target failure is distinct from numerical failure.

## Locked accuracy claims

Ratio below is geometric mean RMS(control)/RMS(two-basis), so **higher than 1 favors Ours**. Brackets are unadjusted descriptive 95% paired crossed-cluster intervals. GOOD requires ratio ≥1.1, lower endpoint >1 and no more than five percentage points of pass-coverage regression. The ratio threshold 1.1 corresponds to at least 9.09% relative error reduction, rather than literally 10%. BAD means the declared success criterion was not met; it does not automatically mean every field worsened.

| Track | Final time | Fixed filtered cubic | GL4 quadratic | Historical conditioned GL2 | FNO scaled | ETDRK4 |
|---|---:|---:|---:|---:|---:|---:|
| discrete | 0.12 | 1.045 [0.921, 1.179] BAD | 0.399 [0.279, 0.543] BAD | 1.721 [1.320, 2.249] GOOD | 7.519 [5.618, 10.165] GOOD | 4.096 [2.749, 6.571] GOOD |
| discrete | 0.24 | 1.160 [1.023, 1.314] GOOD | 0.327 [0.248, 0.431] BAD | 1.466 [1.075, 2.087] GOOD | 7.803 [5.801, 10.722] GOOD | 16.500 [9.648, 30.774] GOOD |
| continuum | 0.12 | 1.053 [0.898, 1.265] BAD | 0.400 [0.289, 0.538] BAD | 1.667 [1.262, 2.183] GOOD | 6.645 [4.851, 9.339] GOOD | 4.184 [2.876, 6.488] GOOD |
| continuum | 0.24 | 1.143 [1.026, 1.272] GOOD | 0.330 [0.228, 0.448] BAD | 1.332 [1.017, 1.784] GOOD | 6.689 [4.987, 8.944] GOOD | 16.910 [10.002, 30.689] GOOD |

Fourteen of twenty accuracy claims are GOOD; six are BAD. The central attribution success occurs only at T=0.24: 12.53% continuum and 13.79% discrete geometric-mean RMS reduction relative to the fixed filtered-cubic parent. Respectively 47/64 and 48/64 fields improve after averaging the repeated seed/grid contrasts. T=0.12 does not establish a calibration benefit. The two longer-time intervals are only just above 1; they are not adjusted for twenty simultaneous claims, and three training seeds provide limited seed uncertainty.

All four comparisons against GL4 are BAD, with control/candidate ratios 0.327–0.400: the candidate has approximately 2.5–3.1 times the RMS error. Its endpoint pass coverage is 12.5–19.5 percentage points lower. This is more consequential than beating the historical conditioner or initialization-retaining local FNO controls. An authoritative published FNO reproduction has not been supplied.

## Additional analytic controls (exploratory contrasts)

Plain normalized GL2 and nested quadratic+cubic were retained but not among the five primary contrasts. These comparisons use the same frozen primary schedules and were computed after seeing results; they are explicitly exploratory.

| Track | Time | GL2/candidate RMS | Nested quadratic+cubic/candidate RMS |
|---|---:|---:|---:|
| discrete | 0.12 | 0.941 | 0.191 |
| discrete | 0.24 | 0.829 | 0.126 |
| continuum | 0.12 | 0.973 | 0.257 |
| continuum | 0.24 | 0.875 | 0.191 |

The fitted candidate is 3–21% worse in geometric-mean RMS than uncorrected normalized GL2, and about 3.9–8.0 times worse than the expensive nested cubic control. Thus learning repairs some weakness of its chosen fixed parent, but the parent is not the strongest available backbone. A successful small calibration must not be presented as evidence that adding this correction is better than omitting it.

## Where the differences arise

Post-hoc T=0.24 regime table: each cell has eight independent fields, six repeated queries per field (three seeds × two grids), 48 queries total. Values are geometric mean ratio, with median field ratio in parentheses; higher favors the fitted candidate. Each regime mixes the four declared distribution strata; this is not a causal comparison of distribution shifts.

| Track | Regime | Fixed filtered cubic | Plain GL2 | GL4 |
|---|---|---:|---:|---:|
| discrete | low | 1.237 (1.207) | 0.703 (0.877) | 0.550 (0.743) |
| discrete | mixed | 0.934 (1.064) | 0.792 (1.022) | 0.680 (0.720) |
| discrete | high_pair | 1.245 (1.171) | 1.313 (1.192) | 0.142 (0.191) |
| discrete | broadband | 1.222 (1.264) | 1.128 (1.147) | 0.360 (0.472) |
| discrete | near_nyquist | 1.418 (1.365) | 1.378 (1.303) | 0.125 (0.155) |
| discrete | boundary_mean | 0.953 (0.990) | 1.110 (1.179) | 0.969 (1.137) |
| discrete | reaction_stiff | 1.246 (1.212) | 0.169 (0.117) | 0.134 (0.097) |
| discrete | diffusion_stiff | 1.106 (1.066) | 1.049 (1.048) | 0.417 (0.418) |
| continuum | low | 1.101 (1.113) | 0.654 (0.763) | 0.480 (0.535) |
| continuum | mixed | 1.012 (1.049) | 0.849 (0.998) | 0.697 (0.848) |
| continuum | high_pair | 1.190 (1.281) | 1.160 (1.186) | 0.085 (0.158) |
| continuum | broadband | 1.273 (1.288) | 1.190 (1.194) | 0.313 (0.368) |
| continuum | near_nyquist | 1.308 (1.387) | 1.278 (1.387) | 0.094 (0.242) |
| continuum | boundary_mean | 0.939 (1.014) | 1.075 (1.193) | 0.982 (1.220) |
| continuum | reaction_stiff | 1.202 (1.168) | 0.314 (0.352) | 0.262 (0.345) |
| continuum | diffusion_stiff | 1.171 (1.106) | 1.040 (1.087) | 0.650 (0.736) |

The candidate improves on GL2 in several high-frequency regimes, yet loses particularly on low/mixed states and reaction-stiff fields. GL4 improves on the candidate in every regime geometric mean. In high-pair/near-Nyquist regimes, GL4 is roughly 7–12 times more accurate than the candidate and 9–14 times more accurate than GL2. Quadrature is therefore no longer exhausted on these finite-step workloads, even though it looked exhausted in the earlier cheap small-step diagnostic.

Concrete high-frequency counterexample: continuum field `advance-confirmation-83020012`, near-Nyquist, declared distribution, T=0.24, has candidate RMS up to 9.61e-6 versus GL4 up to 3.43e-8, a geometric-mean regression of about 294× across its six seed/grid queries. The largest step has h·λ≈4.95 for the most rapidly diffusing initial mode. The cubic small-h expansion is not uniform in that parameter. Concrete reaction counterexample: discrete field `83020030`, coefficient shift, T=0.24, has about 17.5× worse RMS than GL2, with h·r=2.72. The latter suggests nonlinear/reaction variation matters in addition to diffusion stiffness. These are hypotheses to diagnose, not causal proofs from a single field.

Grid-specific comparisons preserve the main result: at T=0.24 the fixed-parent gain is 1.15–1.17× on 32² and 1.11–1.16× on 64², while the candidate remains substantially worse than GL4 on both. This is confirmation at 32²/64²; the separate 128² scaling stage does not create a 128² confirmation cohort.

## Accuracy-qualified complete-call cost

None of the 20 primary cost claims is GOOD: 14 are BAD and 6 NA. Coverage means a field passed the frozen tolerance on **every** seed and grid for that method, not merely that one successful query was fast. Missing validation schedules retain their denominators.

| Track | Time | Candidate passing fields / 64 | GL4 passing fields / 64 | Paired GL4/candidate speed ratio | Verdict |
|---|---:|---:|---:|---:|---|
| discrete | 0.12 | 53 | 62 | 0.9492142606590366 | BAD |
| discrete | 0.24 | 62 | 64 | 0.4726218445379293 | BAD |
| continuum | 0.12 | 57 | 61 | NA: incomplete crossed design | BAD |
| continuum | 0.24 | 61 | 57 | 0.23486735356882857 | BAD |

At discrete T=0.12, the candidate appears 1.995× faster than the fixed filtered parent and 3.704× faster than the historical conditioner **only on jointly eligible rows**, but it passes just 53/64 fields; those speedup claims correctly fail the 95% coverage rule. At discrete T=0.24 it passes 62/64, but has essentially equal cost to its fixed parent (1.001× ratio) and is 2.116× slower than GL4, which passes 64/64 fields. At continuum T=0.24 it passes 61/64 but is 4.258× slower than GL4 on common eligible queries; GL4 itself passes only 57/64, so its shorter time is not a complete replacement at the chosen threshold.

The best current operational choice is regime- and target-dependent. Continuum T=0.12 ETDRK4 achieves 62/64 fields with a median eligible latency near 15.6 ms, whereas the candidate reaches 57/64 and near 98.9 ms. Discrete T=0.24 GL4 is a particularly strong fixed option at about 8.55 ms and 64/64 fields. Do not infer a universal ranking from these examples. Median eligible cost and paired geometric ratios differ and must not be substituted for one another.

The cost intervals are unavailable when the successful-pair subset lacks a complete field×seed crossing; that conservatively avoids silently discarding unmatched seeds. A future descriptive missing-cell estimator could improve reporting, but cannot erase failed coverage or retroactively change this campaign’s primary gate. Missing feasible schedules also mean “not established on the tested validation menu,” not mathematical impossibility for the comparator.

## Uncertainty and data-size sensitivity

Reference uncertainty was not propagated by the original ratio claims; they rejected only pairs at/below the estimate. A post-hoc conservative triangle-inequality sensitivity replaces each ratio by `(control_error-u_control)/(candidate_error+u_candidate)` and the reverse upper expression. Assuming the reported estimates bound reference error, all twenty primary verdicts remain unchanged. Longer-time parent-comparison mean ratios remain at least 1.139 continuum and 1.156 discrete, and their descriptive lower bootstrap endpoints remain 1.022 and 1.020. These are sensitivity calculations, **not certificates**, because the uncertainty estimates are not rigorous bounds and the shared target induces correlation. Small per-field gains can still be reference-sensitive.

A separate post-hoc filter requiring both errors ≥1e-6 retains 38–39 fields for the longer-time fixed-parent contrast and yields ratios about 1.35–1.36. GL4 remains about 3.2–4.0 times more accurate on corresponding retained subsets. Hence the major conclusion is not simply driven by FP32-sized errors. This outcome-conditioned filter is exploratory and cannot replace the original complete-cohort comparison.

The secondary 16-field fitted model is only 3.5–9.2% better in geometric-mean RMS ratio than the 64-field model across four track/time cells; median field ratios improve by only 0.9–3.7%, and pass coverage sometimes regresses. The sixteen-field and sixty-four-field FNO-scaled endpoint errors are exactly identical on the primary comparisons. With one nested subset and equal update caps, this is not a sample-complexity curve or proof that less data is preferable.

## Limits and most informative next steps

- Keep the 64-field cohort locked as completed confirmation. Any revised interpretation or architecture derived from its failures is development work and needs new fields for a new claim.
- Treat the 14 GOOD labels as scoped tests, not a publication leaderboard. The strongest analytic comparator wins; the FNO controls require a successful-learning audit before any literature-level claim.
- Use GL4 as the default accuracy/cost target for the next residual design. Test whether one can approximate its finite-h interaction kernel cheaply rather than correcting a low-order expansion whose omitted terms dominate.
- Before learning more parameters, separate quadratic quadrature error from cubic truncation, reaction resummation and compression. Test the specific hλ≈1–10 and hr≈1–3 failures above with nested-reference error vectors and controlled ablations.
- Require a new correction to beat its strongest contained analytic component, not only its own fixed initialization. Preserve simple GL2, GL4, nested cubic and ETDRK4 alternatives.
- Continue reporting easy states; they reveal unnecessary correction cost and roundoff-limited ratios. Additional hard workloads need mathematical justification and must not conceal those losses.
- The eight-field regime slices, sixteen-field distribution slices, three training seeds and one GPU session support descriptive evidence, not universal population or tail-latency guarantees. Shift cohorts differ in states and physical coefficients; between-cohort differences are confounded.
