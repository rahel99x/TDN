# Response evidence map and research priorities

This map separates inherited native evidence from the new CPU implementation
checks. The historical numbers below were checked against the repository's
[Resolve review](reviews/resolve-full-native-20261011.md) and
[Advance review](reviews/advance-full-native-20261011.md), which link the retained
measurements and their reconstruction. This release does not independently
replay omitted historical checkpoint bytes. Those inspected cohorts are
development evidence for Response.

| Classification | Evidence and its limit | Consequence for this program |
|---|---|---|
| Established, scoped native result | Resolve's two-gain fit improves its parent by 7.85% / 10.99% in arithmetic mean RMS, but retains 3.14× / 2.81× GL4 error. Direct and Adam optimization nearly agree. | Retain the fitted rule; stop treating more gain optimization as the principal opportunity. |
| Established, scoped native result | Shared-transform GL4 is 1.47–2.26× faster than sequential GL4 on the measured 4090 conditions, with numerical parity. | Use the improved analytic implementation and equivalent schedules; this is not proof of fastest complete solution. |
| Mechanistically plausible bottleneck | Resolve's fitted candidate has 71.8% / 74.6% mean contribution to pooled squared error. Leading cubic response opposes the remaining GL4 defect in 149/216 constructed conditions. | Learn the full finite-time source and coupled mean response; remeasure the decomposition after changing the anchor. |
| Confounded neural comparison | Historical adapted FNOs have poor tiny-fit behavior, highly varying output scales and initialization selections. They are not paper reproductions. | Repair direct/full-state and residual objectives, run the optimizer factorial and six-regime gate, and compare capacity/data/compute explicitly. |
| Negative practical result | Advance has no successful declared accuracy-qualified cost claim. Resolve's GL3 node reduction does not establish a speed advantage over shared GL4. | Keep neural accuracy and solver-cost claims separate. Count all reference, tuning, inference and failed work. |
| Negative exploratory result | Historical mixed-noise memory selected zero gain. The new local noise-aware observer probe also found no regime reaching its declared 10% gain: its RMS ratios versus coarse RK4 range from 0.968 to 1.089; see the saved receipt. | Protect a bounded observer study, preserve zero/deterministic controls and stop expansion without new evidence. |
| Mixed exploratory result | The separate local clean-memory fit ranges from 0.454× to 7.174× coarse-RK4 rollout RMS across the 12 declared regimes. Some deterministic controls improve more. | Treat apparent gains as confounded until the temporal-only / closure-only / combined panel separates integration and closure effects; preserve harmful noise regimes. |
| Correctness/measurement defects repaired here | Full-target FNO loss used an unsuitable residual scale; Galerkin physical and nodal metrics could mix; a phase-twin label was not sufficient evidence of paired inputs; feature reuse alone did not eliminate repeated source FFTs. | Separate objective contracts and Nyquist-weighted norms, generate genuine paired fields, and compare transparent versus cached-spectral decoding with parity checks. These repairs are not scientific wins. |
| Implemented, unconfirmed opportunity | Coupled learned sources, cached response kernels, fixed/interpolated source representations, response-weighted compression, numerical remainder, FP64 assembly and physical interventions. | Execute registered development and fresh confirmation. CPU checks establish implementation support only. |

The independent Appendix A audit reproduces exact identities and explicit
counterexamples, including all 54 declared full-path response cases. Its results
support the mechanism and its limitations, not a trained neural advantage.
See [verification receipts](../results/response-verification/README.md).

The portfolio priorities are:

1. **Attribution and fairness:** accepted references, properly scaled direct and
   residual FNOs, original temporal control, fixed-feature fits, matched source
   supervision, anchor and schedule controls. A weak comparator cannot support
   an architecture claim.
2. **Grounded improvement:** full coupled mean/source dynamics and source
   compression judged by downstream error. Identify whether representation,
   optimization, rounding, closure or execution dominates before expanding.
3. **Protected exploration:** same-observation shadow dynamics and noisy causal
   observers, including temporal-only/closure-only/combined attribution. A
   useful negative result completes this question without consuming the main
   training budget.

The frozen primary default is candidate/base geometric RMS at most 0.90, with
registered uncertainty evidence against parity, peak-error noninferiority and
coverage protection. Data efficiency, grid/parameter transfer, reusable-query
cost and numerical-method contributions have separate measurements. There is
no universal model leaderboard. A new analysis prompted by these results must
be labeled exploratory and cannot reuse inspected confirmation as fresh data.
