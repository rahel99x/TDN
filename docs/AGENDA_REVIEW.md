# Fedora full agenda review

The full run `fedora-agenda-20261007T170704067870Z` completed computationally,
but **no trained candidate passed the complete preregistered solver gate**.
This review uses uploaded native RTX 4090 evidence from source `31a607b`.
It performs no new numerical solve, training, checkpoint selection or GPU run.

## Verified execution

The complete archive was reconstructed from its index and two upload-sized
parts. Its SHA-256 is
`edd6eb13fa841699901810880ffd39d1c0dd44ae28041676337d50ed3efe7782`.
All 731 archive members, all eight prerequisite chains, and all 341 sealed
scientific files verify against the committed full protocol and executable
source fingerprint. The originally uploaded unsplit archive was truncated;
the complete parts resolve that transport issue.

The native CPU suite passed 531 checks with one optional upstream-Tower fixture
skipped. Each of the six GPU stages passed the complete 152-case mandatory
suite, including all 114 agenda and 38 consistency cases without skips.
These are repeated executions of the same suite, not 912 distinct tests.
All 222 structural cases completed, including all 34 required checks. Eight
teacher workers prepared 744 accepted references: 696 discrete and 48
continuum-estimate references. Normalization uses training parents only.

All 69 bounded training trials completed: 64 selected trained checkpoints and
five selected initialization. All 276 recorded checkpoint physical-limit checks
passed. The recorded numerical stages totaled 2,009 seconds, approximately
33.5 minutes; this is not independent scheduler accounting or billed wall time.
The policy stage used 1,133 of its 1,200-second numerical budget. There is no
reason to expand its budget automatically.

All eight reports pass unmodified native Tower contracts and readers. All
51,957 canonical rows are preserved across 472 bounded table pages; 1,098,824
displayed scalar cells, 125,190 nested source pointers and 9,372 learning metric
values match their original records. All 372 inventory files and 240 log paths
verify without omissions. Scientific files, archived reports and Tower source
were unchanged during review.

## What the results establish

The strongest improvement in the minimal controls comes from **established
diffusion-first Strang orientation**, before learning. The normalized
validation objective falls from `1.2275e-3` to `5.1829e-5`, a 23.68-fold
reduction. That objective combines normalized one-step MSE with half the
two-step MSE; it is not the endpoint RMS/maximum error.

Source transport beats precompression and the parameter-matched four-layer
FNO in two of three paired validation seeds. The equal-width one-layer FNO has
similar cost and often lower validation loss. The precompression experiment
retains a full-state physical-source and Strang bypass, so it does not isolate
a complete information bottleneck. This run does not settle a universal
precompression advantage or reproduce the published FNO benchmark.

Independent recomputation reproduces all 32,640 confirmation frontiers, 96
per-norm/classical comparison gates and 24 joint gates, with zero discrepancies.
All 12 planned confirmation slots are present. The source-time and closure
slots select initialization, so their results do not establish learned temporal
or closure benefits. The only two descriptive comparison passes are the
initialization-selected source-time arm against GL3 alone.

| Discrete reference-informed frontier | Source, three seeds | Matched FNO, three seeds | Rank-2 kernel, one seed |
| --- | ---: | ---: | ---: |
| RMS coverage at `2e-4` | 574/576 | 576/576 | 192/192 |
| Maximum-error coverage at `2e-5` | 477/576 | 533/576 | 176/192 |
| Median speed relative to matched FNO at `2e-4` | about 1.74x | 1x | about 1.68x |

These are paired counts across physical cases, grids, horizons and training
seeds. The full cohort has eight controlled field variants and one underlying
phase-seed cluster, not 32 independent random fields. The two norms have
separate frontiers, and feasible schedules are selected with reference errors;
these frontiers are not a deployable step-size policy.

**The bare diffusion-first Strang base reaches the primary discrete frontier
target in every case and is about 3.9 times faster than Source/rank-2.**
Median gains against ETDRK4/GL3 do not demonstrate an advantage over the best
available classical choice. At maximum error `2e-5`, rank-2 covers 176/192
cases versus 174/192 for its bare base; at `2e-6`, the counts are 90/192 versus
89/192. Its additional accuracy coverage is small relative to its added cost.

## Worst-case schedules and spatial target

Frontier coverage conceals failures on particular step orders. For the reversed
unseen schedule `[0.17, 0.07, 0.03]`, joint RMS/maximum accuracy at `2e-4` is:

| Method | Passing reversed-schedule cases |
| --- | ---: |
| Source, three seeds | 193/288 |
| Matched FNO, three seeds | 213/288 |
| Rank-2, one seed | 86/96 |
| Bare diffusion-first Strang | 90/96 |

Source's worst discrete maximum error is `0.04098`, versus `3.53e-5` for its
paired physical base on that case. Matched FNO reaches `0.04525`. Rank-2's
worst discrete maximum error is approximately `3.38e-4`, making it the strongest
compact robustness lead, but it still fails 8–9 spatial-regression comparisons
and has only one training seed in this confirmation.

Continuum-estimate coverage is nearly identical for the learned and classical
methods: 23/48 RMS and 14/48 maximum-error frontier cases at `2e-4`. The errors
decrease roughly fourfold with each doubling of resolution. This is evidence
of a dominant shared spatial discretization bias, rather than evidence that
all the neural architectures independently have the same temporal defect.
The same-grid FD/nodal and dealiased continuum targets must remain separate.

## Acceptance and complete cost

The reported `FALSE_ACCEPTANCE_GATE_PASS` applies to the **discrete target**:
1,179 accepted endpoints, 549 fallbacks and zero observed false accepts. It is
an empirical result on this paired cohort, not a certificate.

The continuum diagnostics reuse the discrete-target decisions. Of 410
accepted continuum endpoints, 306 fail the continuum accuracy target (74.63%).
All 306 are definitely resolved failures after accounting for the recorded
reference uncertainty. A small temporal estimator cannot certify a different
spatial discretization target. The classical controls also exhibit the shared
continuum coverage floor.

For Source with step doubling, the median attributed cost ratio is 0.851
against ETDRK4 and 0.405 against GL3. However, its total paired attributed cost
is 6.168 seconds against ETDRK4's 4.415 seconds: **about 40% more work**.
Rejected-work tails reverse the favorable median. These controls use a fixed
eightfold refinement, not the fastest accuracy-matched classical frontier.
Costs are measured component-attributed policy estimates, not independently
timed standalone deployments; the actual shared attempt benchmark is reported
separately. No deployed solver speed advantage follows from those median ratios.

## Next bounded experimental question

Prioritize a **headroom and spatial-target screen** before another broad
training campaign. Measure where the cheapest accurate diffusion-first base
actually fails, distinguish FD and continuum targets, and retain worst-order
accuracy and total/tail cost alongside reference-informed frontiers.

If that screen establishes headroom, the most useful architecture comparison
is bare DF, compact source transport, rank-1/rank-2 phase-aware kernels and the
cheap one-layer source-constrained FNO. Replicate the kernel across three
paired training seeds. Tune on the chosen DF orientation: current downstream
DF models inherit learning-rate/batch choices optimized on RF. Investigate
finite-step/stiffness damping and residual-aligned per-regime loss before
increasing network width or training length. Keep the current closure separate
until independent mean/spatial evidence supports it.

The stored trial IDs retain their initially declared RF/batch-1 labels even
when effective metadata records DF/batch-8. Use the effective orientation,
learning rate and batch fields when reading these results. This labeling issue
does not alter the verified checkpoints or scientific configuration.

The inspected full cohort is now development evidence. A subsequent
confirmation must freeze new disjoint fields, phase seeds and parent IDs before
inspection. Repeating this protocol is a paired replication, not a fresh cohort.

Machine-readable evidence is in
[results/fedora-agenda-review.json](../results/fedora-agenda-review.json).
