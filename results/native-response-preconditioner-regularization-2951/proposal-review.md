# Proposed next gate: local coefficient identification and sampled branch capacity

**Proposal only; no new numerical fits, array evaluations or architecture changes.**
The metadata review found 192 training-parent/context combinations: 32 original
training parents, six intervals each, eight calibration sources and four withheld
sources per context. All are previously inspected development data. This proposal
does not access diagnostic-parent numeric data or revise the historical B STOP.

The useful next question is whether the five fixed response branches can represent
the required action with coefficients specific to the actual state, physics and
interval. The existing global model restricts those five coefficients to a shared
affine function of eight features. A local fit relaxes that restriction, but does
not demonstrate that a deployable encoder can infer the resulting coefficients.

## Bounded experiment to preregister

1. For each of the 192 training contexts, fit **one shared five-coefficient vector**
   to all eight calibration sources. Use the retained five branch actions and
   target `teacher - ordered_2`. This is 192 small five-column fits, with no new
   references or architecture. Each calibration system is 200-by-5 on the retained
   5-by-5 grid. Smaller-source treatments remain deferred.
2. Use FP64 QR/SVD, explicit physical row weights and training-only column scaling.
   Freeze the numerical rank rule and any regularization treatment before scoring;
   never select them using withheld targets. Record singular spectra, coefficient
   norms, residuals, numerical rank and affine/five-branch replay differences.
3. Seal all calibration-fitted coefficients before reading the four withheld
   targets. Score the local fit on every corresponding withheld source: 768
   planned local-prediction rows. Retain failed, unresolved and missing rows.
4. Separately compute an **evaluation-only sampled projection diagnostic** using
   the stacked four withheld sources in each context: 192 five-column projections.
   This diagnostic system is 100-by-5. Its coefficient vector is never selected,
   deployed or reused as a prediction
   model. All four sources must share it; fitting each source separately would
   allow source-dependent coefficients and answer the wrong operator question.
5. Reuse frozen global-40 and classical predictions, including zero correction,
   ordered-two/eight and Lawson-two/four/eight. The primary fitted comparison is
   the frozen global `p032-s08` model, paired on identical parents, contexts and
   sources. Any descriptive subgroup or other old-cell comparison must disclose
   its different training information and cannot count reused rows independently.

This amounts to at most 384 small five-column decompositions for one declared
solver treatment. Runtime, extraction, retention and memory must be budgeted and
approved through the existing bounded native workflow before execution. It is
not an authorization to expand the current experiment or acquire more labels.

## What the projection diagnostic actually identifies

Within one context, let `A` stack the five retained branch columns for the four
withheld sources, `b` stack their correction targets, and let `A_w`, `b_w` include
equal source mass and the declared normalized physical cell mass. For a candidate
coefficient vector `c`, write `v=A_w c` and let `P` be an orthogonal projector.

If `P` projects onto the **entire** column space of `A_w`, then

`||b_w-v||^2 = ||(I-P)b_w||^2 + ||P b_w-v||^2`.

The first term is the minimum error within that sampled, shared-coefficient
branch span. The second is coefficient recovery/transfer error. This is a finite
source-bank statement, not a full-operator bound or a population guarantee.

A TSVD cutoff instead defines a resolved numerical subspace. The actual predictor
may retain components outside it. In that case record the exact expanded identity

`||b_w-v||^2 = ||P(b_w-v)||^2 + ||(I-P)b_w||^2 + ||(I-P)v||^2`
`                - 2 <(I-P)b_w, (I-P)v>`.

Do not omit the signed cross term or call a truncated-span residual an unavoidable
full-span floor. Report reconstruction error, discarded singular values and rank
sensitivity. Coefficient magnitudes alone are not interpretable when the branches
are nearly dependent; compare their assembled actions as well.

## Decision consequences

| Observation | Supported next action |
| --- | --- |
| Shared withheld projection residual remains large and numerically resolved | Investigate branch expressiveness; a richer encoder alone cannot repair this sampled span. Test a specific missing operator feature against equal-cost classical refinement. |
| Projection residual is small but calibration-fitted local predictions fail | Diagnose source identifiability, conditioning and coefficient sensitivity before adding neural capacity. A smaller-source comparison would require its own later frozen treatment. |
| Local calibration fits transfer, while frozen global-40 predictions fail | State/time coefficient conditioning is a plausible limitation. A new deployable conditioner merits a separate frozen proposal; local label-informed coefficients do not establish out-of-parent generalization. |
| Local fits improve on global-40 but remain far behind qualifying classical controls | Do not infer a practical solver advantage or automatically add an encoder. Reconsider the work/accuracy target and physical backbone. |

Teacher/refinement differences remain uncertainty estimates, not certificates.
Retain absolute errors and uncertainty-resolved correction-relative errors; small
projection residuals can fit reference noise. Parent-balanced summaries use 32
independent training parents, not 192 independent contexts. Native speedup and
deployable coefficient acquisition remain untested: local oracle label costs are
not free inference, and the retained evaluation introduces no new timing evidence.
