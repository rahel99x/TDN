# CARC interaction screen: accuracy gained, cost still unresolved

CPU job **12690663**, using source **806db03**, completed the frozen screen.
The field-derived interaction correction consistently improves Strang on this
bank. Its current implementation costs substantially more per step, and
ETDRK4 wins every two-dimensional case. This supports investigating a cheaper
correction and matched-tolerance performance; it does not establish a neural
solver advantage. No new numerical experiment or training was run for this review.

The [machine-readable review](../results/carc-interaction-screen-review.json)
records the uploaded archive hash, integrity checks, paired comparisons,
timings, limitations and next-experiment proposal.

## Verified execution and reporting

- **188 actual allocated tests passed**, with no failures, errors or skips.
- **502/502 records** completed: 48 passed correctness controls and 454
  observations. Numerical work took **3.272 seconds**; the recorded worker
  elapsed time was **49.896 seconds**, including tests and coordination.
- All **32 references** were accepted. All **416 method trajectories** completed
  with finite, bounded outputs. There were 28 one-step problems and four
  five-step rollouts, each evaluated by 13 methods.
- All five science-file hashes, the completion seal, protocol/configuration
  digests and execution-source fingerprints verify against source `806db03`.
  The archive's dirty-checkout flag does not establish a globally clean checkout;
  the verified execution-source content nevertheless matches the committed code.
- All **10,882 scalar metrics** retain exact values, types, inputs and source
  pointers; all **502 notes** appear exactly once. The three exporter omissions
  intentionally exclude pytest cache/work directories and Tower's own directory.
- Unmodified native Tower readers validated 19 output contracts, 38 metric
  records and 17 log entries. Independent schema validation passed 42 objects.
  All 40 uploaded files remain byte-identical.

The recorded request was four CPUs, 16 GiB, 30 minutes, account `anakano_81`,
user `aadaniel`, partition `main`, and no GPUs. Requested resources are not
measured utilization; the archive does not include independent final `sacct`
accounting.

## Accuracy at the same declared step schedule

Ratios below are medians of paired L2 errors, not ratios of aggregate medians.
A resolved win or loss requires error separation greater than twice the
accepted reference's recorded uncertainty. That uncertainty is a numerical
refinement diagnostic, not a rigorous continuum error certificate.

| Method | Wins/losses vs Strang | Median error / Strang | Wins/losses vs ETDRK4 |
| --- | ---: | ---: | ---: |
| ETDRK4 | 29 / 3 | 0.05627 | — |
| GL3 additive | 32 / 0 | 0.02742 | 23 / 9 |
| GL5 additive | 32 / 0 | 0.02739 | 23 / 9 |
| GL3 mean-only | 32 / 0 | 0.27664 | 6 / 26 |
| GL3 zero-mean | 32 / 0 | 0.96346 | 3 / 29 |

GL3 reduces median paired error by about 97.3% relative to Strang. The benefit
survives the four rollouts: their median error ratio is 0.01250. Against ETDRK4,
GL3 wins 20/28 one-step problems and 3/4 rollouts. ETDRK4 is more accurate and
faster in all three two-dimensional cases; GL3 has 3.60–5.50 times their error.
That limitation must remain in subsequent experiments.

In the historical finite-amplitude regime (`kappa=.02`, `r=6`, `h=.4`), L2 error
is 0.00176174 for Strang, 0.000425069 for GL3 and 0.0101150 for ETDRK4. This
16-cell field test is not the earlier 64-cell per-pair quadratic oracle.

The 32 problems are diagnostic conditions, not 32 independent statistical
parents. Some share fields across parameter/time settings, and the two remote
patterns are periodic translations. No significance or population-wide ranking
follows from these counts.

## Mechanism and cost findings

Strang's squared mean error accounts for a median **92.54%** of its total
squared error across the bank. The mean-only ablation removes much of this
bias, while the full correction improves on both mean-only and zero-mean
ablations in every case. The mechanism must represent reaction-driven mean
change and spatial redistribution; total mass conservation would be incorrect.

| Paired median timing ratio | One step | Five-step rollout |
| --- | ---: | ---: |
| GL3 / Strang | 5.846 | 5.905 |
| GL3 / ETDRK4 | 1.789 | 1.794 |
| GL5 / GL3 | 1.236 | 1.229 |

These are single sequential CPU samples with uncached coefficients. They do
not establish warmed performance, physical cold-cache timing or GPU speed.
The current ETDRK4 implementation recomputes phi coefficients on each call;
subsequent performance comparisons need equal caching opportunities and a
separate setup cost.

GL3 and GL5 use 18 and 26 transforms per active step, versus nine for ETDRK4
and two for Strang. GL5 changes median paired error negligibly while costing
about 23% more time. Its greater-than-1% accuracy gains occur in only four
one-step cases and the remote rollout. Capacity changes are small and often
unresolved; projected and additive variants have identical reported scientific
endpoints. The current bank does not establish an advantage from active clipping.

For context, **all Strang endpoints already meet L2 error 0.002**. This is a
post-hoc check using an earlier development tolerance, not this screen's
preregistered performance endpoint. With max-norm error 0.002, Strang passes
31/32 instead: the historical case fails. Both distinctions matter when defining
the next accuracy target. Error reductions alone do not establish useful speed.

## Next bounded experiment

First measure time to a fixed accuracy over the same final horizon. Compare
GL3/GL5, Strang with smaller steps, and ETDRK4 with smaller steps; use repeated
warmed measurements, account for setup, and give all methods equivalent
coefficient caching. Keep RMS and maximum error separate, use references
resolved at the stricter targets, and predeclare fresh one- and two-dimensional
fields. Preserve the observed small-step and 2D losses.

The clearest implementation opportunity is a cheap mean correction. Let
`v=u-mean(u)`, `N` be the number of cells, `P_k=|FFT(v)_k|^2/N^2`, and
`lambda_k<=0` the diffusion eigenvalue including `kappa`. Parseval's identity gives

\[
D(s)=\sum_k P_k\operatorname{expm1}(2\lambda_k s),
\qquad
\overline{d_h^{GL}}=-rJ_c(h)
\left[\sum_j\omega_jJ_c(s_j)D(s_j)-W_2D(h)\right].
\]

This reproduces the mean of the existing raw GL quadratic correction in exact
arithmetic using one input FFT and no inverse transforms. Including Strang,
that suggests **three transforms instead of the current mean-only branch's
18**. This shortcut is not implemented or timed by this review. It is not the
exact nonlinear mean, and it does not commute with nonlinear bounding.

Verify this identity numerically before measuring speed, then investigate a
compact spatial correction for the remaining error. Fresh amplitude sweeps and
2D fields should distinguish missing higher-amplitude interactions from
quadrature error. Those interactions can already contribute at order `h^3`;
do not force a learned remainder to begin at `h^4`. Retain the full unlearned
GL3 branch and ETDRK4 as controls in later neural comparisons.

This review changes no solver, frozen protocol, Tower application, completed
run or training budget. The next experiment remains a proposal.
