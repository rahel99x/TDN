# Full native Advance: training, model attribution and competitor fairness

Evidence: verified archive for `fedora-advance-full-20261011T001715`, source commit `a7646702ff1061d78a722153ef2548d9d831de0d`; source hash `ae810a101203ddd4cb26dcd75253ef62ebb0cea19f5a36f15a8c347d81d01466`. Exact inputs: `freeze/catalog.json`; six `train-*/catalog.json`, `learning_curves.jsonl`, and `fairness_diagnostics.json`; `audit/protocol.json`. Reproduction: `training-analysis.py`, with its compact output `training-analysis.json`. This review analyzes existing measurements; no models were refitted and no confirmation information changed model selection.

## Findings that are now established

The campaign contains **62 deployed model instances**, comprising 14 frozen controls and 48 trainable instances. Of those trainable instances, **25 selected fitted checkpoints and 23 selected their initialization**. This is different from the native smoke: both coefficients of the two-basis model genuinely moved during full training. All 48 declared final fits completed **800/800 updates**, with no numerical failures or time-cap truncations. A selected catalog row showing 40 updates means an initialization from a 40-update tuning trial won the selection; it does **not** mean the later 800-update experiment was omitted.

| Model | Trainable parameters | Full selection across two equation tracks and three seeds |
|---|---:|---|
| Historical conditioned GL2 | 43 | 6/6 fitted |
| Two-basis, 16 fields | 2 | 6/6 fitted |
| Two-basis, 64 fields | 2 | 6/6 fitted |
| Legacy hybrid FNO | 47,057 | 6/6 fitted, but validation errors remain very large |
| Zero-head hybrid FNO | 47,057 | 0/6 fitted; all select DF-equivalent initialization |
| Physically scaled hybrid FNO, 16 fields | 47,057 | 0/6 fitted; all select DF-equivalent initialization |
| Physically scaled hybrid FNO, 64 fields | 47,057 | 0/6 fitted; all select DF-equivalent initialization |
| Scaled mean-separated FNO | 47,345 | 1/6 fitted; one discrete seed improves validation by 2.70% |

The two-basis model is a **fitted numerical method**, not a temporal MLP or a state-conditioned neural network. It computes a diffusion-first Strang step plus two analytic state-dependent corrections, with **two global fitted gains**. Its coefficients do not change with the input field. The historical 43-parameter model really does have a small neural conditioner, but its node, amplitude and conditioner adjustments are combined in this comparison and cannot be credited separately.

## What the coefficients say

For the two-basis family,

`prediction = DF(state,h) + g2 × normalized_full_GL2_defect(state,h) + g3 × heat_filtered_cubic_defect(state,h)`.

The bounded parameterization is `g_i = 1 + 0.75 tanh(theta_i)`, initialized at `(1,1)`, with open admissible interval `(0.25,1.75)`. The analytic quantities carry all the spatial and temporal structure. A fixed-unit-gain control performs the same response calculation, making it the clean learning-attribution comparison.

The following are medians over three paired training seeds; the ranges are seed ranges, not confidence intervals.

| Track / training fields | Quadratic gain | Cubic gain | Validation objective, initial → selected | Median reduction |
|---|---:|---:|---:|---:|
| Discrete / 16 | 1.0334 [1.0227,1.0371] | 0.7379 [0.7351,0.7436] | 0.12480 → 0.10618 | 14.92% |
| Discrete / 64 | 1.0463 [1.0460,1.0548] | 0.8312 [0.8260,0.8384] | 0.12480 → 0.11114 | 10.95% |
| Continuum / 16 | 1.0611 [1.0418,1.0644] | 0.7381 [0.7340,0.7417] | 0.19421 → 0.17268 | 11.08% |
| Continuum / 64 | 1.0580 [1.0571,1.0715] | 0.8557 [0.8501,0.8619] | 0.19421 → 0.18107 | 6.77% |

The direction is reproducible: slightly amplify the quadratic correction and reduce the cubic correction. This is evidence for calibrating the combination on this development distribution. It is not proof that the cubic formula has a universal coefficient below one: the coefficient compensates for finite-step approximation, filtering, the chosen distribution and the loss together. Seed agreement does not establish independent identifiability of both basis directions.

All twelve two-basis fits selected update **800**, the final allowed checkpoint. That means convergence to the optimal two coefficients was not demonstrated. The system has only two degrees of freedom, so a more decisive next control is a direct constrained fit, not a larger neural network.

Historical conditioned GL2 reduced its own initial validation objective by median **19.69% discrete / 14.70% continuum**. Its left node moved from the GL2 location ≈0.2113 to 0.1921–0.1960 discrete and 0.1893–0.1906 continuum. Node movement, effective gain and output compression all matter. The logged global amplitude is not the effective field-conditioned gain: the network output and amplitude logit share an additive gauge. The run does not isolate a uniquely necessary neural effect.

## Validation is not confirmation or a universal ranking

The selected loss is an equal-independent-field average of

`(mean(error²) + 0.1 max(error²)) / max(mean(DF_error²), reference_uncertainty_rms², 10^-12)`.

There are four validation horizons for each of 32 fields. This objective balances errors relative to the baseline, and includes a peak-error penalty; it is not an RMS error with physical units. The baseline denominator and reference are used during training/selection, not supplied at deployment. RMS, maximum error, target eligibility and complete cost must remain separate reporting views.

For perspective, the frozen GL4 and full quadratic+cubic controls have validation objectives **0.04028 / 0.01298 discrete** and **0.07989 / 0.04363 continuum**. Thus the fitted two-basis family is **not** the most accurate option even on the validation criterion used to choose it. Comparing only against its fixed-unit-gain initialization would conceal this.

The statistics audit finds the primary 64-field two-basis accuracy improvement over its fixed-unit-gain control meets the predeclared criterion at **T=0.24**, not T=0.12; comparison with GL4 goes against the two-basis model. That supports a narrow calibration effect, not a general breakthrough. Complete cost conclusions require the paired cost analysis, not this training table.

## What the FNO failures do and do not establish

All FNO variants use a 16-wide, two-block local/spectral trunk with cutoff four. They receive the state, centered state, dimensionless step and physical coefficients, and reciprocal grid sizes. Their prediction is a known DF step plus a learned correction. These are **adapted local hybrid FNO controls**, not reproductions of the FNO paper or a tuned authoritative FNO benchmark. The comparison does not justify claims that TDN beats FNO in general.

The legacy output scale is `(h/0.2)^3`. Zero-head initialization preserves the physical solver exactly. Scaled models use the deployable `h³ r Var(u) active_rate (r+active_rate)` scale; the mean-separated model gives the mean its own output and centers the spatial output. All are meaningful controls. The zero head intentionally blocks first-step trunk gradients, but allows the output head to learn; tests establish that trunk gradients then activate.

The native gradients were finite and nonzero. However, optimization remains poorly conditioned:

| Final-training branch | Fraction of updates clipped at norm 1 |
|---|---:|
| FNO-zero, both tracks | 100% |
| FNO-scaled / 16 fields | 93.96% discrete, 97.38% continuum |
| FNO-scaled / 64 fields | 97.33% discrete, 97.71% continuum |
| FNO-mean using clipped setting | 96.69% discrete, 97.92% continuum |
| Two-basis, both data sizes/tracks | 0% |
| Historical conditioner | 0% discrete, 0.042% continuum |

FNO-zero median pre-clipping norm is approximately 263,634 discrete / 213,355 continuum. Physical scaling reduces that markedly: FNO-scaled/64 has median 17.27 / 49.85, but clipping still dominates. Gradients from the unclipped control use a loss scale of 0.01, so their raw norms cannot be directly compared with unscaled norms. Frequent clipping is a diagnostic, not proof that clipping caused the failure: Adam partly normalizes uniform gradient scaling, while changing clipping factors can alter its moments and the relative influence of batches. Causality requires the matched long control.

**A consequential selection limitation remains:** the four 40-update tuning opportunities for every zero/scaled FNO all retain their identical initialization. The lowest-loss selection therefore ties, and deterministic ordering selects the first setting: learning rate 0.001 with norm-1 clipping. Only that setting gets 800 final updates and the ten-second audit. Hence the run did test the unclipped, scaled-loss alternative briefly, but did **not** establish that it fails when given the same long opportunity. This is a limitation of the declared tuning design, not retrospective evidence that another setting would succeed.

Thirty single-example diagnostics all decreased their own objective. Safe zero/scaled/mean heads obtain roughly 2.4–4.4× reductions after twenty updates on the selected difficult example. That demonstrates usable gradients and some local fitting ability. It is not a successful near-zero overfit test: the selected objective remains roughly 0.35–0.72, and only one deliberately high-baseline-error example is used. Learning one hard example can coexist with failure on a mixed distribution containing nearly exact baseline cases. It does not identify whether the remaining problem is scale imbalance, clipping, unsuitable loss, train–validation horizon shift or insufficient training.

The legacy family makes large relative improvements from very bad initialization but finishes with median validation objective **4,051.5 discrete / 2,534.0 continuum**, versus DF at **0.96097 / 0.94731**. A dramatic descending learning curve alone is therefore misleading.

Five of six mean-separated FNOs still choose initialization. One discrete seed reaches 0.93501 versus initial 0.96097, a 2.70% reduction, using the unclipped scaled-loss setting. This is a modest single-seed observation, not stable evidence that mean separation solves the problem.

When an initialized FNO produces the same output as DF, timing its full unused neural trunk measures redundant work. Such timing is not evidence that a useful FNO representation is intrinsically slower. The cost report must say that explicitly.

## Budget and data-efficiency accounting

| Phase | Trials | Completed updates | Total measured trial seconds | Outcome |
|---|---:|---:|---:|---|
| Frozen-control audits | 14 | 0 | 15.53 | No fitted parameters |
| Tuning | 156 | 6,240 | 403.71 | Every trial completed 40 updates |
| Final fitting | 48 | 38,400 | 1,700.21 | Every trial completed 800 updates |
| Equal-time diagnostic | 48 | 17,471 | 502.26 | All reached time cap; 100–699 updates each |
| Single-example fitting | 30 | 600 | 16.24 | Every trial completed 20 updates |

Total recorded trial time is **2,637.95 seconds**, and the optimizer-only total is **1,556.68 seconds**. These exclude teacher generation, sample preparation, exhaustive validation schedule measurements and other stage overhead; they are not total campaign cost. The six native training allocations total 4,530 seconds according to the independent accounting audit. No trial failed numerically.

The equal-time trials ran **10.031–11.455 seconds**, median 10.311, against a ten-second cap. Their stopping points include validation/checkpoint boundary overhead; “equal time” here is a bounded achieved budget with explicit overshoot, not exact equality. All reached the wall cap and none were update-cap censored. No equal-time checkpoint ultimately displaced the selected final fit/initialization. The main confirmation is not an equal-training-compute comparison: it carries each model's best validation-selected result from the declared opportunities.

Both 16-field and 64-field fits use 800 updates × four samples, drawn with replacement. At three training horizons, this is approximately **66.7 versus 16.7 sample passes**, respectively. The small subset therefore receives four times the exposure per example. It is one nested subset per size, not multiple independent sample-size experiments.

Secondary fresh-confirmation comparisons from the statistical audit give field-averaged RMS ratios `error_n64/error_n16` of **1.035 / 1.052 continuum** and **1.092 / 1.046 discrete** at T=.12/.24. The small fit is modestly better on that error summary, but wins only 34–38 of 64 independent fields and target-pass changes are mixed. FNO-scaled predictions/errors are identical at both sizes because both select initialization. Neither observation supports a general sample-efficiency claim.

## Most informative next actions

1. **Fit the two gains directly with a bounded convex objective.** For frozen basis arrays, every error is affine in `(g2,g3)`. Mean squared error plus the maximum squared error is convex in these gains, and the admissible rectangle is convex. A deterministic bounded fit to the identical training loss eliminates the nonlinear logit/Adam optimization confound. Validate on the existing validation set; treat all previously inspected confirmation fields as development for any revised method. Compare against `(1,1)`, a fitted quadratic-only gain, GL4 and the full quadratic+cubic rule.
2. **Audit the gain basis Gram matrix and asymptotic response.** Stable seeds do not prove independent coefficients. If the basis directions are nearly collinear, prefer a one-parameter rule. Non-unit gains can sacrifice cancellation of leading defect terms; no higher-order or stiff-stability claim follows from a lower finite-step loss.
3. **Repair the FNO tuning gate before claiming a fair neural victory.** If short-trial best checkpoints all tie at initialization, carry more than the first setting into an equal, bounded long trial or use a predeclared tie handling rule. Compare larger learning-rate reduction, unclipped/scaled loss and matched-budget alternatives with independently frozen choices. Do not simply lengthen the same heavily clipped run.
4. **Make the overfit diagnostic decisive and regime aware.** Require a declared residual-error target on a tiny fixed set containing both hard and near-baseline-exact cases; inspect mean and spectral residuals, parameter updates and layerwise gradients. Merely improving after twenty updates is too weak. Keep this entirely in development data.
5. **Preserve the narrow claim.** A two-parameter analytic correction can benefit from calibration on this task. It has not displaced the strongest analytic method, and the broad neural comparison remains affected by a demonstrable optimization-selection limitation. Larger architecture claims should wait for that gap to close.
