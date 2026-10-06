# Compact spatial corrections: bounded CPU experiment

The [CARC work–precision review](WORK_PRECISION_REVIEW.md) found that the
spectral mean correction is algebraically correct and cheaper than its full
evaluation, but does not win a complete-solver comparison at matched accuracy.
The full spatial GL3 correction has useful wins, alongside many losses to
Strang and ETDRK4. This experiment tests whether implementation fusion and a
compact spatial representation can retain its useful accuracy at lower cost.

The program is training-free. It measures the implemented numerical methods
on a fixed diagnostic bank; it does not establish neural generalization,
trainability, an advantage over FNO, or GPU performance. Negative results and
unattained targets remain part of the report.

[Cloud CPU validation](../results/compact-spatial-validation.json) completed all
**2,760 candidates in 399.55 seconds**, with all **166 individual references**
accepted. The **392 required parity checks** passed; **232 approximation
diagnostics** were retained as observations. The regression suite passed
**923 tests**, with one optional schema-dependency test skipped. Separate native
Tower and schema checks passed for both the completed run and a real SIGUSR1
interruption. All 8,082 scientific CSV rows were retained, with their cells and
provenance pointers checked against the canonical report. This validates the
cloud implementation; CARC timing still requires the allocated run below.

## Run on CARC

One CPU allocation runs correctness tests and then the experiment:
**4 CPUs, 16 GiB, 30 minutes**, charged to **anakano_81** for **aadaniel**.
The numerical experiment has a **1,200-second cap** and uses one intra-op
and one inter-op thread. All files remain under the project, using its existing
Python `.venv`.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main
bash scripts/carc_compact_spatial.sh
bash scripts/carc_compact_spatial.sh --submit
```

The first invocation previews the request. The second submits one CPU job;
it has no GPU successor, neural training, pending-job cap or automatic
resubmission. Do not source `common.sh` or execute numerics on the login node.

Save the printed run ID. These commands monitor and collect that run:

```bash
bash scripts/carc_compact_spatial.sh status latest
bash scripts/carc_compact_spatial.sh logs latest --lines 200
bash scripts/carc_compact_spatial.sh collect latest
```

Replace `latest` with the printed run ID when another research workflow is
active: the research launchers share their latest pointer. Upload the resulting
review archive for analysis. Existing runs are preserved; changed source or
configuration requires a fresh run.

For a smaller integration run, add `--smoke` to the submission command. Smoke
is a frozen subset, retains five measured timing repetitions, and has a
240-second numerical cap. It is not an alternate parameter search.

## Questions and ablations

The physical system remains the periodic discrete logistic reaction–diffusion
equation, `u' = kappa*Delta_d u + r*u*(1-u)`. The experiment separates these
questions:

1. Does batching transform operations reduce GL3's complete-solver cost while
   preserving its quadratic correction to numerical roundoff?
2. Can a few input Fourier modes represent the useful spatial correction,
   while the full input spectrum supplies the raw GL3 mean?
3. Does combining the mean and spatial corrections outperform either part
   alone, and when does discarded high-frequency interaction destroy that gain?
4. Do any improvements survive stricter error targets, larger amplitudes,
   retained historical failures, and changes in grid and batch size?

| Method | Role |
| --- | --- |
| `strang` | Smaller-step exact-subflow splitting control |
| `etdrk4` | Prepared higher-order exponential control |
| `gl3` | Original full quadratic interaction correction |
| `gl5` | Quadrature refinement; retains the earlier narrow success |
| `gl3_mean_spectral` | Full-spectrum mean-only ablation |
| `gl3_fused` | Same full GL3 correction with batched transform operations |
| `compact_gl3_m2` | Compact spatial correction with a two-mode cutoff per axis, plus GL3 mean |
| `compact_gl3_m4` | Four-mode cutoff per axis, plus GL3 mean |
| `compact_gl3_m8` | Eight-mode cutoff per axis, plus GL3 mean |
| `compact_gl3_m4_no_mean` | Four-mode zero-mean spatial correction only |

All cutoffs, schedules and comparisons are declared before execution. No
teacher state, per-case fitted coefficient or measured-error-based mode choice
enters a candidate solver.

## What the compact method computes

For each sample independently, write `c=mean(u)` and `v=u-c`. Let `P_K` keep
input Fourier modes with `|k_i| <= K` on each spatial axis. The compact branch
evaluates the quadratic GL3 interaction defect using `P_K v`; the background
mean remains that of the full input. It uses the **original fine-grid discrete
diffusion eigenvalues**, not a different coarse-grid Laplacian.

Products of retained modes can reach twice the cutoff. A compact transform
grid of `min(N_i, 4*K+1)` cells per axis retains those products without adding
aliasing beyond the original discrete grid. Spectral extraction and injection
account for the different FFT normalizations. The compact spatial result is
lifted to the original grid before being added to its Strang base.

The combined candidates replace the compact correction's mean with the
full-spectrum spectral GL3 mean. In mathematical notation,

\[
u_{\mathrm{new}}=S_h(u)+\mu^{GL3}_h(v;c)
 + \left[d^{GL3}_{h,K}(v;c)
 - \overline{d^{GL3}_{h,K}(v;c)}\right].
\]

Here `mu` is the mean of the **raw quadratic GL3 defect**. It is not the exact
nonlinear solution mean and imposes no fictitious mass conservation. The
`no_mean` ablation omits `mu` and still removes the compact defect's mean, so
it adds only the zero-mean spatial correction to Strang. The
compact spatial term is an approximation to full GL3, which is itself a
finite-amplitude approximation to the true defect. Missing amplitude terms
can already enter at local order `h^3`; fourth-order accuracy is not assumed.

Retaining a low output mode does not preserve all interactions that produce
it. Two discarded high input modes can generate a low difference mode; the
full-spectrum mean only repairs the zero mode. High-frequency stress cases
therefore matter even if a smooth field is represented almost exactly.

Fusion changes dispatch and reduction overhead. It does not make several
transformed fields into one field's mathematical work. Counters distinguish
FFT calls from transformed fields and cells. Compact methods still transform
the full input and return a full-grid output, so this is not a claim of a
complete solver with cost independent of the original grid size. Cached
coefficient bytes describe stored tensors, not measured peak process memory.

## Frozen experiment panels

The full protocol declares **2,760 candidate slots** and **4,640
method/norm/tolerance frontier entries** across 58 case groups. These are
repeated comparisons, not 2,760 independent problems.

### Accuracy and amplitude: 40 single-sample conditions

The first panel retains all **24 conditions** from the previous work–precision
bank, including its three historical 2D failures and finite-amplitude
counterexample. These reused conditions are regression evidence, not fresh
independent confirmation. Another **16 declared conditions** cross one- and
two-dimensional fields with four patterns and amplitudes **0.04 and 0.28**:
smooth, broad-band, a cutoff-straddling pair of modes 9 and 10, and near-Nyquist
content. They use 64 cells in one dimension and a 24×28 grid in two dimensions.
Within each pattern/dimension pair, only amplitude changes. The saved protocol
distinguishes these roles and records the exact phase and state construction.

All ten methods use **1, 2, 4, 8, 16 and 32 steps** to the same final time for
each condition: **2,400 candidates**. Small and large amplitudes probe how
well the quadratic representation handles finite-amplitude interactions.
New phase and frequency patterns probe the compact cutoff's limitations.

### Grid and batch scaling: 18 groups

The second panel crosses these grids with batch sizes **1, 4 and 16**:

- One dimension: **64, 256 and 1,024 cells**.
- Two dimensions: **16×16, 32×32 and 64×64 cells**.

It fixes the physical coefficients at `kappa=0.0005`, `r=2` and final time
`T=0.1`; coefficients are not rescaled to make larger grids artificially
easy. Within each dimension, the same smooth continuum pattern, with per-axis
frequencies at most five, is sampled at each resolution. This scaling bank
measures cost on that declared spectrum; the accuracy panel supplies the
separate high-frequency stress tests. Batch members have distinct means and
phases; smaller batches are prefixes of larger ones, and their identities
are retained. The ten methods use **2 and 8 steps**, giving **360 candidates**.

This panel measures whole-batch latency and per-member throughput. Its
accuracy requirement applies to the worst member, rather than allowing easy
members to dilute one large error. Each member gets its own reference
acceptance decision and mean/spatial error decomposition. A larger batch is
not a collection of independent timing trials.

Every reference solves the same spatially discrete system as its candidate.
Comparing grids here does not establish convergence to the continuum PDE.
The deliberately bounded coefficients and horizon also limit which larger
grid or long-time conclusions can follow.

## Reference, precision and cost rules

Refined FP64 coupled RK4 references retain the previous `1e-8` precision
criterion, including the `sqrt(N)` scaling used to control the maximum-norm
uncertainty estimate. The new protocol allows **five bounded refinement
attempts** rather than changing the acceptance threshold. It leaves the
previous run and its rejected reference unchanged. Reference uncertainty is a
refinement diagnostic, not a rigorous certificate.

RMS and maximum error define separate targets: **0.002, 0.0002, 0.00002 and
0.000002**. Feasibility includes accepted reference uncertainty. Batch
feasibility uses the worst-member error and uncertainty conservatively;
unaccepted member references prevent an eligible batch frontier. The report
retains rejected references and unattained targets.

Timed candidates use **FP64** throughout. Separate FP32/FP64 numerical checks
test implementation identities, exact limits, and batch isolation; their
results are not FP32 throughput measurements. Arithmetic and relative-error
limitations near zero remain explicit.

Preparation can retain only coefficients determined by step size, geometry,
dtype and device. Every call recomputes state means and mean-dependent reaction
terms. One complete warmup precedes **five measured complete rollouts** in
frozen rotating method orders. Raw durations and work counters are retained.

Two costs are compared independently: warmed median rollout time, and
preparation plus that median. The setup-inclusive measure is not physical
cold-cache timing and a single setup measurement does not estimate setup
variance. Neither metric includes reference generation or post-hoc selection.
Raw timing ranges are descriptive, not confidence intervals.

For each method, case, norm and target, the report selects the fastest feasible
candidate from the fixed step grid. This is **post-hoc work–precision analysis**,
not an adaptive step-size controller. A candidate may win on one panel and
lose on another; panel denominators and batch sizes should not be pooled into
a single universal ranking.

## Correctness and interpretation

Tests cover exact zero-time, reaction-only, diffusion-only and uniform-state
limits; mixed batches; per-sample means; prepared-cache independence; and
periodic-grid transform behavior. Fused/full and full-cutoff compact parity
distinguish implementation correctness from deliberate mode truncation.
Discarded modes, finite amplitudes and higher quadrature are separate controls.

Reports retain final RMS, maximum, signed mean and zero-mean spatial errors,
so another reduction in mean bias cannot hide a spatial failure. An invalid
or interrupted trajectory cannot win by stopping early. No clipping or
reference-based rescue silently repairs a candidate. Signals and budget
exhaustion retain partial reports without a successful completion seal.

Raw correction diagnostics compare each compact defect with full GL3. Their
relative RMS value divides the worst member's RMS difference by the worst
member's full-defect RMS, only when that denominator exceeds the declared
roundoff scale. It is a ratio of batch norms, not the worst individual relative
error; endpoint accuracy is evaluated separately against the PDE reference.

Computational completion does not imply a scientific advantage. The useful
next architecture is one that retains accuracy with a meaningful measured
complete-solver cost reduction in the intended regimes, while preserving its
counterexamples. ETDRK4 and Strang remain the controls if compression loses.
Any later neural comparison needs its own parent-disjoint data,
validation-only selection and matched full-rollout accuracy/cost experiment.

## Tower and review artifacts

The worker prints the exact `TDN_TOWER_DIR`. Use that directory to inspect this
run's metrics rather than an unrelated latest report:

```bash
REPORT=/home1/aadaniel/projects/TDN/runs/REPLACE_RUN_ID/tower/REPLACE_REPORT_ID
bash scripts/tower.sh validate "$REPORT"
bash scripts/tower.sh show "$REPORT"
```

The files under that exact report directory are:

| Path relative to `$REPORT` | Contents |
| --- | --- |
| `metrics.jsonl` | Live progress and computational status |
| `outputs/compact_spatial.csv` | Candidate accuracy, cost, work and status |
| `outputs/compact_spatial_frontiers.csv` | Separate warmed/setup selections or explicit unattained targets |
| `outputs/compact_spatial_checks.csv` | Reference evidence, identity checks and observed approximation errors |

Scientific tables retain exact `source_path` and `source_record` pointers into
the canonical `experiment/compact-spatial.json`; raw timing repetitions and
per-member errors remain available there. Their analytics IDs are
`analytics.compact_spatial`, `analytics.compact_spatial_frontiers` and
`analytics.compact_spatial_checks`. The experiment directory also retains its
protocol, effective configuration, summaries and cryptographic completion
manifest. Tower itself is unchanged. `collect` verifies sealed experiments and
packages their protocol, metrics, logs and failures for review.

For development in a separate non-CARC checkout, use:

```bash
bash scripts/compact_spatial_local.sh --smoke --run-dir runs/compact-local-001
```

The local wrapper requires the project's existing venv and a fresh path under
the checkout. It rejects CARC login-node use and Slurm/desktop overrides.
Cloud CPU validation remains distinct from actual allocated CARC evidence.
