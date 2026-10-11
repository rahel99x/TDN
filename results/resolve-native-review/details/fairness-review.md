# Independent review of the full native Resolve fairness experiment

## Judgment

The original optimizer-opportunity defect is repaired: every predeclared setting received 800 updates, and no short initialization tie removed an alternative. This changed the evidence: 8/12 selected FNO family/track/seed instances now retain trained checkpoints. Their held-out development gains remain small, every FNO instance is substantially less accurate than GL2 and GL4, and only 4/36 FNO tiny-fit trials meet the explicit memorization target. This supports continued diagnosis of the training formulation, not a claim that competitive FNOs have been defeated.

A particularly informative new diagnosis is the enormous variation in the normalized physical output scale. Its derivation is exact for the declared initial Fourier fields, and its extreme values match an independent CPU reconstruction to less than 2.5e-7 relative difference. Its role in the poor neural results is still observational: this campaign did not intervene on that scale.

## Evidence and denominators

Target clarification: the prepared `continuum` teachers use spatially refined 2N/4N Galerkin solves projected to N; predictions operate at N. These errors include unresolved-scale closure differences as well as integration/representation error. They are not the same-grid Galerkin targets used in the kernel microbenchmarks. The interaction panel separates those contributions.

Six stages cover two equation tracks and three paired training seeds. Each uses the same 64 training fields and 32 validation fields on a 32² grid. Training horizons are 0.02, 0.06 and 0.12; validation horizons are 0.03, 0.06, 0.12 and 0.24. The last horizon therefore tests extrapolation within the **validation selection set**, not untouched confirmation. The same validation fields select optimizer, phase and checkpoint and supply the reported performance. Reported comparisons are descriptive development results; confidence intervals would not remove this selection bias.

The retained evidence contains:

- 162 trials: 54 long-update, 54 equal-time and 54 tiny-memorization trials.
- All 54 long trials completed 800 updates. All 54 tiny trials completed 200 updates. None was numerically failed or time-censored.
- All 54 equal-time trials reached their nominal 10-second cap; elapsed time includes validation/checkpoint overhead and overshoot. None reached its 4,000-update ceiling.
- 72,460 optimizer updates, 289,840 sample presentations and 72,622 curve rows. Presentations are not independent examples: each update uses four samples, and the tiny trial repeats its single field four times.
- 8,064 finite selected-model/control metric rows, each associated with an accepted teacher. These are repeated models, horizons and seeds, not 8,064 independent fields.
- 18/18 selected-trial identities independently reconstructed exactly from validation objective and tie break. All 54 tiny gate outcomes also recomputed exactly.

Classical controls repeat identically across the three seeds. The independent validation unit remains the field; it is 32 fields, not 96.

## What the repaired opportunity actually establishes

The three settings were:

| Setting | Learning rate | Gradient clipping | Objective multiplier |
|---|---:|---:|---:|
| clipped | 0.001 | norm 1 | 1 |
| unclipped_scaled | 0.001 | none | 0.01 |
| lower_rate_unclipped | 0.0001 | none | 0.01 |

All settings received their full long trial from the same family/seed initialization. Equal-time trials also restart from that initialization; they are overlapping deterministic prefixes of long trials when update counts coincide, not independent replications. Tiny-fit data are training-only, and none of those checkpoints was selection eligible.

This resolves the previous first-tied-setting-only issue. It does **not** isolate clipping causally: removing clipping and multiplying the objective by 0.01 occur together. Idealized Adam is approximately invariant to a positive objective multiplier, but its epsilon, finite precision and clipping break exact invariance. A factorial clipping × objective-scale comparison remains missing. The lower-rate comparison does isolate learning rate within the unclipped, scaled branch.

All clipped long FNO trials retained initialization, and their pre-clipping gradients exceeded the threshold on roughly 92–96% of updates. The successful selected FNO fits came from unclipped settings. That is evidence of an association with optimization settings, not proof that clipping was the unique cause.

Selected states were trained in 5/6 `fno_scaled` instances and 3/6 `fno_mean` instances. All 6 two-basis instances were trained. Mean-separated continuum FNO retained initialization in two seeds despite succeeding at tiny memorization; single-example capacity and multi-regime learning are distinct gates. A zero parameter-movement entry refers to the **selected checkpoint**; it does not mean the optimizer never changed or explored the weights.

These are small local hybrid FNOs (width 16, depth 2, modes 4; 47,057 and 47,345 parameters) sharing the DF physical backbone. No architecture-capacity sweep, authoritative implementation reproduction, or broad task-level FNO claim is established.

## Accuracy, objectives and regimes

Below is the median across three seeds of the **arithmetic field/horizon mean RMS** on validation; lower is better. The values are not confidence intervals.

| Method | Discrete RMS | Continuum RMS |
|---|---:|---:|
| DF analytic backbone | 3.8934e-5 | 3.7724e-5 |
| FNO scaled — Theirs adapted control | 3.7383e-5 | 3.6203e-5 |
| FNO mean-separated — Theirs adapted control | 3.7796e-5 | 3.7724e-5 |
| Two-basis — Ours | 1.0073e-5 | 1.0583e-5 |
| GL2 analytic control | 1.0302e-5 | 1.1181e-5 |
| GL4 analytic control | 3.1572e-6 | 3.6982e-6 |

Across paired seeds/horizons, the arithmetic mean RMS improvements of the FNO families over DF are only about 1.0–2.8%. Geometric paired improvements are also modest (ratios DF/FNO of 1.007–1.050). Neither family wins the geometric mean over any of the 32 individual fields against GL2 or GL4 on either track. This finding concerns the tested trained adaptations, not the general capability of FNO.

The two-basis rule improves the arithmetic mean RMS over GL2 by about 2.1% on the discrete track and 5.5% on the continuum track. The paired geometric ratios GL2/two-basis are larger, 1.131 and 1.114, because geometric and arithmetic summaries weight the error distribution differently. It wins the field-averaged geometric comparison in 26/32 fields on each track. Against GL4, however, its arithmetic mean RMS is 3.19× and 2.86× as large; its geometric RMS is about 1.44× and 1.43× as large. It wins only 16/32 and 15/32 field-geometric comparisons. A universal leaderboard would hide this distribution.

The fitted numerical rule increases the quadratic coefficient to approximately 1.035–1.071 and decreases the cubic coefficient to 0.918–0.972. Five of six selected two-basis fits occur at update 800; the other at update 150. This does not establish an optimizer optimum and should be compared with the separate direct coefficient fit.

Regime-specific evidence remains small: only four validation fields per regime. Nonetheless, it identifies coherent failure patterns. Two-basis versus GL2 has geometric gains in high-pair, near-Nyquist and boundary-mean fields; reaction-stiff fields regress. Against GL4, its arithmetic mean RMS is about 5.1× worse on discrete reaction-stiff fields and 4.4× worse on continuum reaction-stiff fields. GL4 also strongly improves high-pair, near-Nyquist and diffusion-stiff regimes. Those are precisely the regimes where finite-step interaction resolution deserves priority.

The joint normalized objective and absolute RMS do not always rank methods the same way. For example, the two-basis continuum validation objective is 0.240–0.244, worse than GL2's 0.225, despite its smaller arithmetic mean RMS. The objective normalizes each example by its DF residual (with a floor) and includes a maximum-error term. This is a real tradeoff, not a bookkeeping contradiction.

No selected prediction in the 8,064 rows violates the sampled [0,1] interval or exhibits a resolved energy increase. These finite sampled checks are not stability theorems or long-rollout evidence.

## Tiny memorization and what it does not prove

The gate requires **both** an objective below 0.01 and a reduction to at most 1% of that trial's initialization objective. It is intentionally stricter than merely improving. The result is 4 GOOD and 50 BAD; no NA is caused by budget censoring.

All four successes are the mean-separated FNO on the continuum track: all three unclipped-scaled seeds and one clipped seed. The best objective reaches 1.12e-4. No scaled-only FNO tiny trial passes. No discrete FNO trial passes. No two-basis trial passes.

These denominators represent only **one chosen field/horizon per track**, repeated across three seeds and settings. The probe maximizes the DF residual normalization among training samples. It is a high-pair field at horizon 0.12 in each track; it is not a diverse memorization benchmark. Thus 4/36 FNO successes are not a population trainability rate.

The two-basis failures need a distinct interpretation. Its discrete initialization objective is already 0.00909, below the absolute threshold. Fitting reaches roughly 0.000375, a 95.9% improvement, but misses the additional 99% relative criterion. Do not use its BAD label to imply that its absolute one-field error is worse than the FNO's. The inherited initializations are different, so the gate is a within-method optimization test, not an across-method accuracy ranking.

## The main unresolved conditioning mechanism

The FNO correction has the form

\[
\hat u=S_h(u)+s(u,h)f_\theta(u,h,\kappa,r),\qquad
s=h^3rV\lambda_{\rm active}(r+\lambda_{\rm active}).
\]

Here `S_h` denotes the DF backbone, `V` is the spatial variance, and `lambda_active` is the state-weighted diffusion decay rate. Training divides squared error by

\[
d_j^2=\max(\operatorname{MSE}(S_h(u_j),y_j),\;\widehat\sigma_j^2,\;10^{-12}).
\]

A unit perturbation in the raw neural output therefore has normalized sensitivity `s/d_j`. In the exact declared Fourier input fields this sensitivity spans:

| Track | Training examples only | Train + validation examples |
|---|---:|---:|
| Discrete | 0.00131 to 4,982 | 0.00131 to 20,587 |
| Continuum | 0.00131 to 26,794 | 0.00131 to 87,652 |

These are 192 training and 128 validation queries per track. They are algebraic reconstructions from immutable field coefficients and recorded normalization. The four extreme cases were independently reconstructed using the actual CPU FP32 scale implementation; relative differences were below 2.5e-7. Their square gives the raw-output loss-curvature multiplier, spanning many orders of magnitude. This is **not** the condition number of the complete network Hessian.

The largest factors occur in diffusion-stiff fields. The scale follows a small-step polynomial, whereas heat transport damps finite-step high-frequency interactions. Thus the network may need a raw output extremely close to zero in one regime and a far larger response in another, even though both are nominally normalized training cases. The observed validation curves are consistent with severe sensitivity: the largest long-FNO validation excursion reaches approximately 3.88e5× its initial objective on the discrete track and 4.92e5× on the continuum track. Across the 18 long FNO trials per track, the median worst excursion is about 118× and 2,105×, respectively. This association does not identify the sole cause; representational capacity, stochastic sampling and time extrapolation remain alternatives.

Mean behavior deserves its own treatment. About 69–70% of discrete FNO validation squared error and 78% of continuum error is in the constant spatial mode. Low-frequency bands carry about 90% and 99% of error energy. Merely adding a mean head did not solve the multi-regime optimization problem. Both mean and spatial outputs still inherit the same scalar multiplier.

## Cost and measurement limits

All optimizer trials together used 2,806.1 measured seconds (46m46s), of which 1,845.1 seconds (30m45s) are optimizer-loop time. Long trials used 1,942.7 seconds, equal-time trials 559.9 seconds and tiny trials 303.6 seconds. These totals include every attempted setting, not just winners. They exclude teacher generation, final metric loops and scheduler overhead; complete campaign accounting is separate.

Equal-data and equal-requested-update opportunities are now documented and attained for long trials. Equal walltime gives different update counts: continuum FNO roughly 150–152, continuum two-basis 100, versus about 500–605 for the discrete methods. The expensive physical backbone influences these counts; parameter count alone is not training-compute fairness. There are no deployable latency or matched-accuracy frontiers in this stage.

## Priority decisions

1. **Advance a scale-conditioning ablation before increasing network size.** Compare the present small-step polynomial scale with a finite-step semigroup/mean-quadratic response scale. For a constant background and linear heat response, a useful schematic mean-discrepancy magnitude involves `r*h*sum |vhat_k|²[phi1(-2*h*lambda_k)-exp(-h*lambda_k)]`, with sign and reaction-background factors derived from the actual correction. Its stiff decay differs fundamentally from an exploding polynomial. This is not exact nonlinear truth; preserve small-step order and exact nulls, and never use teacher values during inference.
2. **Use inexpensive alternative explanations as controls.** Compare fixed development normalization, a bounded deployable scale, and separate mean/spatial scales. A target normalization may use training labels for training, but inference scaling must use state/physics only. Charge feature extraction and any extra transforms. Factor clipping and loss multiplier independently; keep learning-rate and initialization opportunities paired.
3. **Expand tiny diagnostics efficiently.** Use a small declared panel containing low, high-pair, diffusion-stiff and reaction-stiff fields on both tracks. Evaluate more frequently during the first 50 updates. Include a direct residual-output fit or last-layer regression to separate feature limitations from end-to-end optimization. Preserve the current one-field failures.
4. **Retain GL4 as the primary accuracy control.** Two-basis improvements over GL2 do not establish an advantage over the strongest analytic method. Compare each improvement with GL4 and with the direct two-coefficient optimum before creating a larger campaign.
5. **Require a fresh cohort for claims.** This validation cohort has now influenced model and future design decisions. Freeze the next protocol and selection before new independent confirmation. No additional conclusions about larger-grid generalization or long-rollout stability follow from this 32² fairness experiment.

## Reproduction

Run `.venv/bin/python .runtime/resolve-native-analysis/fairness-analysis.py` from the repository root with the verified canonical store present. The script writes `fairness-review.json`, reconstructs every selection and tiny gate, and retains track/seed/method/phase and regime-specific denominators. The four scale cross-checks are post-hoc CPU arithmetic checks, not new GPU measurements. No source or historical result files were changed by this review.
