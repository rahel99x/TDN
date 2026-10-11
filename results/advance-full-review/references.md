# Independent reference, cohort and selection audit

Run: `fedora-advance-full-20261011T001715`; recorded source
`ae810a101203ddd4cb26dcd75253ef62ebb0cea19f5a36f15a8c347d81d01466`, protocol
`5226c377c6f097d6821b49d6011ee6d77fd039d4336377406d1e6d46c6a2b3c4`.
This review reads the exact retained JSON bytes through `review_io.py`. It does
not load omitted checkpoints, reconstruct omitted arrays, train anything or
change a frozen result. `references-audit.py` reproduces the counts and checks;
`references-audit.json` retains their machine-readable results.

## Integrity and independent units

| Split | Independent continuous fields | Grids | Accepted endpoint references |
|---|---:|---|---:|
| Training | 64 | 32² | 384/384 |
| Validation | 32 | 32² | 256/256 |
| Confirmation | 64 | 32² and 64² | 512/512 |
| Scaling | 8 | 64² and 128² | 32/32 |
| Total | 168 | | 1,184/1,184 |

No overlap was found across splits in parent IDs, field seeds, continuous field
IDs, field clusters or complete frozen Fourier terms. The 64 confirmation fields
are the statistical units. Three model seeds, two grids, multiple schedules,
thousands of Fourier coefficients and repeated timing rounds do not multiply
the sample size. Each confirmation regime has eight fields; each distribution
quarter has 16; a regime-by-distribution cell has only two.

Every prepared metadata hash matches the data manifest and every field
declaration matches the frozen protocol. All 17 fresh reference banks bind the
same exact checkpoint/selection freeze. All 72 fresh parent metadata files have
reported modification times after the freeze; the earliest is 19.8 seconds
afterwards. Runtime prerequisites and hash bindings are stronger evidence than
timestamps alone. Actual reference arrays and checkpoint payloads are declared
omissions from this compact export: the review verifies retained metadata and
recorded content hashes, not the omitted numerical arrays themselves.

## Teacher mathematics and actual refinement

The independent FP64 Lawson-RK4 teacher uses exact linear propagation and an
independently coded RK4 update for the nonlinear term. It does not invoke the
evaluated split or ETDRK4 solver. The discrete target uses the central-difference
Laplacian and nodal logistic multiplication. The continuum target uses dealiased
spectral Galerkin evolution on doubled and quadrupled grids and projects the
finer answer to the model grid.

The metadata records actual three-level temporal trials, not a configured
promise to refine. Final counts range from `(8,16,32)` to `(128,256,512)`, with
`(36,72,144)` for some rate-dependent starting counts. Counts are doubled, fine
differences contract or are at the roundoff floor, and acceptance requires both
RMS and maximum estimates below the frozen `2e-8` threshold. Continuum teachers
also actually compare 64²/128² for an N=32 answer, 128²/256² for N=64, and
256²/512² for N=128 scaling. Shared higher-grid calculations are cached; repeated
metadata entries must not be counted as independent teacher solves.

| Confirmation target | Largest RMS uncertainty estimate | Largest maximum uncertainty estimate |
|---|---:|---:|
| Discrete, N=32 | 1.867e-8 | 1.972e-8 |
| Discrete, N=64 | 1.518e-8 | 1.592e-8 |
| Continuum, N=32 | 9.493e-9 | 1.201e-8 |
| Continuum, N=64 | 9.493e-9 | 9.634e-9 |

Median confirmation RMS uncertainty is approximately `1e-9`. Continuum spatial
differences alone have maxima `1.393e-9` RMS / `1.098e-8` maximum at N=32 and
`1.419e-15` RMS / `3.220e-15` maximum at N=64. The latter indicates resolution of
this smooth, fixed physical-frequency family on those fine grids, not arbitrary
PDE resolution independence.

The uncertainty is deliberately conservative in combining observed projected
temporal differences and a spatial difference, but it is **not a rigorous error
certificate**. Two spatial levels and one time integrator can share systematic
error. Finite independent solver cross-checks in the audit strengthen the
implementation evidence but do not independently cross-check every one of the
1,184 targets.

All 31,488 measured confirmation endpoints carry reference metadata identical
to their original accepted teachers; no endpoint RMS error is at or below its
reference uncertainty. Median uncertainty/error is approximately 0.0203% for
discrete and 0.0186% for continuum, but the easiest individual cases reach
32.3% and 23.3%. Small gains in those cases need uncertainty sensitivity, even
though their absolute errors exceed the estimated uncertainty. The separate
statistics audit propagated conservative triangle-inequality envelopes using
these estimates and found the 20 primary verdicts unchanged; this remains a
sensitivity calculation, not certification.

### Two different error questions remain mixed in the continuum track

For discrete comparisons, refinement targets the same spatial ODE solved by
the discrete models. For continuum comparisons, the teacher evolves a finer
Galerkin ODE before projection whereas the models generally evolve the N-grid
Galerkin equation. In notation,

`model_N(t) - P_N u_fine(t)`

contains both the model/time approximation to `u_N(t)` and the truncation/closure
gap `u_N(t) - P_N u_fine(t)`. Actual fine-grid convergence validates the target,
but does not isolate this coarse-to-continuum gap. A new *converged same-N
Galerkin reference* is needed to partition it. The cheap 16² mechanism diagnostic
already uses same-grid targets; it cannot by itself establish that every later
continuum gain comes from reducing temporal error. This distinction matters
especially when an otherwise strong coarse solver cannot reach a strict target.

## Selection and schedule opportunity

The audit recomputed all **372 frozen schedule choices** directly from
**19,840 raw validation rows**, with **zero mismatches**. Each of the 620
model/schedule opportunities has all 32 distinct validation fields. Every model
receives both primary unequal-step schedules plus uniform 1-, 2-, 4- and 8-step
endpoint schedules at final times 0.12 and 0.24. A schedule is eligible only if
all 32 fields satisfy both RMS and maximum targets, including uncertainty.

| Track | Selected schedule cells | No feasible validation schedule |
|---|---:|---:|
| Discrete | 93 | 93 |
| Continuum | 69 | 117 |
| Total | 162 | 210 |

These infeasible cells are important results rather than missing measurements.
They mean that this method, target and finite schedule menu did not meet the
conservative validation criterion. They do not mean no finer step count or
adaptive method could solve the task.

The 31,488 unique confirmation cells exactly follow the primary schedules plus
the frozen validation-selected schedules for each model. No duplicate endpoint
cells or unexpected schedule access were found. Testing only selected schedules
avoids a large timing Cartesian product, but means the confirmation results are
**validation-selected schedule transfer**, not a complete post-hoc optimum over
every method/schedule combination. Selection uses N=32 validation, so N=64 is a
resolution-transfer test rather than an independently optimized N=64 frontier.
Previously inspected confirmation should now be treated as development evidence
if a schedule or model is revised.

References were generated for final times 0.12 and 0.24 only. Unequal-step
composition evaluates intermediate model states, but those states do not have
independent accepted teachers in this campaign. This is not a long-rollout or
continuous-time dense-output validation.

## Design limitations that should shape the next experiment

1. **Shift quarters are not paired interventions.** Independent phases and
   differing baseline physics/mean groups occur in each quarter. Comparing
   quarter averages cannot isolate an amplitude, coefficient or spectrum effect
   causally. Models are still fairly paired with one another within each field.

2. **The coefficient shift preserves the physical ratio.** Both diffusivity and
   reaction rate are multiplied by 1.7. Thus `F_shift = 1.7 F_base` and
   `u_shift(t) = u_base(1.7 t)` for a fixed initial field. This tests overall
   timescale transfer, not an independently changed diffusion/reaction balance.
   A future factorial test should vary the two coefficients separately.

3. **Some regime labels stop describing the spectrum after a shift.** The
   spectral-shift quarter replaces the original modes with 16 random modes but
   retains original labels such as `low`/`favorable`. Interpret regime and
   distribution jointly, or report the actual spectrum descriptors. Calling
   every row labeled `low` an easy low-frequency field would be incorrect.

4. **Near-Nyquist is relative to the training grid.** The fixed physical mode
   14 is near the N=32 cutoff; it is much farther from the N=64/128 cutoffs.
   Retaining fixed physical frequencies is correct for resolution transfer, but
   a separate constant-relative-cutoff stress cohort is needed for a large-grid
   near-Nyquist robustness claim.

5. **Data efficiency changes coverage as well as count.** The first 16 training
   fields contain the first two standard physics groups; the 64-field bank adds
   another group and additional phases. A performance difference is a joint
   data-count/distribution-coverage effect, not an isolated sample-complexity law.
   Future subsampling should be stratified and repeated.

6. **N=128 is an eight-field scaling test.** It should not be described as a
   64-field, three-seed fresh N=128 scientific confirmation. Small subgroup
   samples cannot substantiate universal robustness or tail-probability claims.

7. **Teacher uncertainty is small relative to operational tolerances, not
   always relative to a tiny performance difference.** Propagate or stress-test
   it when interpreting near-roundoff model rankings, and keep numerical target
   uncertainty separate from field-sampling intervals.

No evidence of split contamination, reference metadata corruption, acceptance
misreporting or confirmation-driven schedule selection was found. The main
remaining issues are the scope of the claim and causal attribution, rather than
an invalidated campaign.
