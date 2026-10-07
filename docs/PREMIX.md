# Nonlinear mixing before compression: full hypothesis experiment

The [compact spatial review](COMPACT_SPATIAL_REVIEW.md) identified a concrete
failure: discarding input modes 9 and 10 removes their contribution to output
mode 1. Restoring the mean only repairs mode zero. This program tests whether
preserving nonlinear interactions before selecting output frequencies improves
accuracy, whether that representation can be economical, and whether a learned
version improves complete neural-solver accuracy and cost.

The numerical mechanism and trained architecture are separate experiments.
Agreement with GL3 tests the mechanism; independent PDE labels test the learned
solver. Neither agreement nor successful execution establishes an advantage.
The full program includes favorable, typical and adverse regimes declared
before execution. They are mechanistic expectations within periodic logistic
reaction–diffusion, not rankings chosen after observing results or universal
best/worst cases.

The [implementation validation record](../results/premix-validation.json)
records complete cloud CPU accuracy, scaling and three-seed neural runs,
artifact integrity, interruption handling, regression rechecks and native
Tower validation. Those checks establish implementation readiness; actual
A100 correctness and performance are measured by the allocated workflow below.

## Run the complete program on CARC

Use the existing project Python venv and the fixed project directory. The
launcher submits three independent CPU jobs and one dependent A100 job:

| Stage | Work | Allocation | Dependency |
| --- | --- | --- | --- |
| `accuracy` | Numerical identity checks and demanding accuracy comparisons | 4 CPUs, 16 GiB, 30 minutes | None |
| `scaling` | Larger-grid and native-batch comparisons | 4 CPUs, 16 GiB, 30 minutes | None |
| `prepare` | FP64 independent PDE teachers and shared training normalization | 4 CPUs, 16 GiB, 30 minutes | None |
| `neural` | Train, select and evaluate five families across three seeds | 1 A100 40 GB, 4 CPUs, 16 GiB, 30 minutes | All three CPU jobs succeed and their artifacts verify |

Each full stage has a **1,200-second numerical budget**. Scheduler queue time
is additional. The account is **`anakano_81`**, user **`aadaniel`**. There is no
pending-job cap, automatic expansion or automatic resubmission. The neural job
is submitted with an `afterok` dependency; a successful numerical comparison
means computational completion, not that our method must win before training.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main

RUN="carc-premix-$(date -u +%Y%m%dT%H%M%SZ)"

# Read-only preview: no scheduler calls or numerical work.
bash scripts/carc_premix.sh plan --run-id "$RUN"

# Submit the complete four-stage experiment.
bash scripts/carc_premix.sh run --run-id "$RUN"
```

Save the printed run ID. Inspect the same run throughout:

```bash
bash scripts/carc_premix.sh status "$RUN"
bash scripts/carc_premix.sh logs "$RUN" --lines 200
bash scripts/carc_premix.sh paths "$RUN"

# After completion, collect every stage for review.
bash scripts/carc_premix.sh collect "$RUN"
```

`latest` can replace the run ID; this program uses its own latest-run pointer.
The printed review archive includes checkpoints, references, protocols,
metrics, logs and failures. It excludes pytest temporary directories.

For a small integration run, use a fresh ID and append `--smoke` to both
`plan` and `run`. That profile still exercises the dependency chain and actual
GPU execution; its tiny data and training budgets do not test the scientific
hypothesis. Do not reuse a completed directory or pull changes into the checkout
while these jobs run: source, protocol, dataset and software fingerprints must
remain consistent across stages.

The wrappers configure caches, temporary files, logs, datasets and checkpoints
under `/home1/aadaniel/projects/TDN`. They require its Python `.venv`, real Slurm
allocations and the correct identity/account. Do not source `common.sh` or run
numerical stages on a login node. Tower itself is unchanged.

If an older checkout failed at `startup` with `Tower reports must be outside
science directories`, update `main` and submit `run` again to create a fresh
workflow. Preserve the failed run and its review archive. The corrected launcher
binds each report to its stage's science directory while storing reports in
`tower/` alongside the science stages; the frozen workflow `protocol.json`
remains intact. A startup failure before report creation has no metrics path,
so `paths` has nothing to print for that job. A dependency-cancelled neural job
does not need manual resubmission: the fresh workflow submits all dependencies.

## The numerical mechanism

The equation remains the same spatially discrete periodic system:

\[
u'=L u+r u(1-u),\qquad L=\kappa\Delta_d.
\]

For each parent separately, write `c=mean(u)`, `v=u-c`, and `E_s=exp(sL)`.
The GL3 quadratic correction contains

\[
Q_s(v)=(E_s v)^2-E_s(v^2).
\]

With normalized Fourier coefficients and the original discrete diffusion
eigenvalues, its output coefficient is

\[
\widehat Q_m(s)=\sum_k\widehat v_k\widehat v_{m-k}
\left[e^{s(\lambda_k+\lambda_{m-k})}-e^{s\lambda_m}\right].
\]

Indices wrap on the original grid. A retained low output `m` can therefore
depend on high input `k` and `m-k`. The older compact method uses `P_K v` inside
this expression. The new representation computes the interaction from the
full input, then projects the complete correction: `P_K d_GL3(v;c)`. Reaction
Jacobians, quadrature weights, diffusion transport and the endpoint subtraction
remain part of that correction; projecting `v²` alone is insufficient.

| Numerical method | Role |
| --- | --- |
| `strang`, `etdrk4` | Established complete-solver controls |
| `gl3`, `gl3_fused` | Original and fused versions of our existing correction |
| `compact_gl3_m4`, `compact_gl3_m8` | Previous input-truncation controls |
| `output_gl3_m4`, `output_gl3_m8` | Full nonlinear interaction followed by output projection |
| `gl3_fused_chunked` | Full fused correction evaluated in parent chunks of four |
| `selected_output_gl3_m4` | Direct selected-output convolution on two declared small cutoff-pair cases |

Output projection retains the full raw GL3 mean, because mode zero is retained.
That mean is not the exact nonlinear PDE mean or a mass-conservation law.
Output projection still pays for the full fused interaction; it is a
representation control, not a claim of inexpensive compression.

Direct selected-output convolution sums every original-grid source pair for
each retained output. Its work is `O(batch × cells × retained_outputs)`;
chunking eight output modes bounds forward workspace but does not remove that
work. Pair kernels are evaluated inside measured calls. This direct method is
therefore restricted to the 128-cell 1D and 32×32 2D cutoff-pair accuracy cases.
Its purpose is to test an alternative exact implementation of selected
coefficients, not to imply a speed advantage.

All products use the original collocation grid and retain its cyclic aliases.
There is no claim of continuum dealiasing. Missing output frequencies and
finite-amplitude terms remain possible errors even when every retained
quadratic interaction is exact.

## Expanded numerical panels

The `accuracy` panel has **24 conditions**: four grids—1D 128 and 512 cells,
2D 32×32 and 64×64—crossed with six patterns:

- **Favorable:** smooth, small-amplitude fields.
- **Typical:** broadband fields.
- **Adverse:** cutoff-straddling pairs, near-Nyquist content, localized
  structure and steep fronts.

It evaluates **1, 2, 4, 8, 16 and 32 steps** at the same final time within each
condition, for **1,308 candidates**. Pattern-specific amplitudes, reaction
rates, diffusion and horizons are frozen in the saved protocol. These coupled
stress regimes are not a full factorial capable of isolating every factor.
Near-Nyquist cases explicitly use grid-scaled diffusion; the other mechanism
patterns keep their physical parameters across resolutions.

The `scaling` panel has **36 groups**: 1D 128/512/2,048 cells and 2D
32×32/64×64/128×128, batches 1/4/16, and smooth/cutoff-pair fields. It evaluates
**two and eight steps**, for **648 candidates**. Here the amplitude is 0.28,
reaction rate is 6 and final time is 0.5. Diffusivity is `5e-6` in 1D and
`2.5e-5` in 2D, fixed across resolutions within each dimension. These choices
lengthen the horizon and strengthen reaction relative to the preceding scaling
bank without redefining old results.

Both panels report separate RMS and maximum targets:
**`2e-3`, `2e-4`, `2e-5`, `2e-6`, `2e-7`, `2e-8`**. Every batch member must meet
the target after adding its own accepted reference uncertainty. Smaller
batches are prefixes of larger ones. Reused member references retain their
state/equation fingerprint and provenance; they are not fresh observations.

References use independent same-grid FP64 coupled RK4 at `n,2n,4n`, with up to
seven refinement attempts and a `1e-9/sqrt(cells)` tolerance passed to the
reference acceptance rule. The maximum-norm uncertainty estimate multiplies
the RMS estimate by `sqrt(cells)`. These are refinement estimates, not rigorous
certificates or continuum-error bounds. Unresolved references and infeasible
targets remain visible.

Every numerical solver uses FP64. One complete warmup precedes five rotated
timing repetitions. Report warmed median rollout time separately from one
preparation measurement plus that median. Preparation includes only
state-independent coefficients; each call recomputes state-dependent terms.
Work counters separate FFT calls, transformed fields/cells and direct pair
products. Cached coefficient bytes are not measured peak memory. Batch latency
and per-member throughput are distinct quantities.

## The learned hypothesis

The learned model tests an economical inductive bias suggested by the
numerical mechanism. It does not execute full GL3 inside its encoder. With
shared physical features `f(u,h)`, width `w` and learned pointwise maps:

\[
z=\mathrm{lift}(f(u,h)),\qquad
q=z+\frac{A(z)\odot B(z)}{\sqrt w},\qquad
g=\mathrm{head}(S_K(q)).
\]

The pointwise product creates phase-dependent interactions while the full input
is available. `S_K` performs learned channel mixing on a symmetric retained
Fourier output box. The main raw correction is bandlimited, but the complete
physical update is not: all families retain the full-state Strang base and
the common signed-capacity output map.

| Family | What it tests |
| --- | --- |
| `premix` | Full-state nonlinear features and quadratic latent mixing before output compression |
| `precompress` | Same learned parameter shapes, but `P_K(u)` enters feature extraction before nonlinear mixing |
| `premix_local` | Premix plus a learned high-output-frequency bypass `head_local(q-P_K(q))` |
| `fno` | Representative four-layer spectral-plus-local FNO hybrid control |
| `cnn` | Representative periodic residual-CNN hybrid control |

The `precompress` ablation projects the state **before** computing reaction
features or nonlinear latent products, preventing full-state nonlinear
features from leaking the excluded interaction. It still uses the full input
for the physical Strang base. Encoder projection can overshoot the physical
interval; those features are not silently clipped.

The physical increment has the common factor `U_ref*(h/t_ref)^3`, with
`U_ref=1` and `t_ref=0.2`, followed by the same signed-capacity update for every
family. A cubic prefactor does not prove a learned method's global order.
Output heads start at zero, so initialization reproduces the physical base.

FNO already has nonlinear activations and full-grid local branches; the shared
physical features also contain a nonlinear reaction feature. It can form
nonlinear interactions. The experiment tests whether our explicit product and
compression order use the bounded training budget more effectively. The FNO
control is an adapted repository implementation, not a reproduction of the FNO
paper or a comparison against every published neural solver.

## Data, training and held-out evaluation

The full learned experiment declares **48 training, 24 validation and 48
diagnostic parents**. Initial-state hashes and parent identities are disjoint
across splits. Descendant states and horizons remain attached to their parent.
Normalization is fitted only on full training initial fields and shared across
families and seeds.

Development fields at 32×32 cover smooth, mixed-frequency, in-band pairs and
rough patterns with declared coefficient variation. Training uses one-step
horizons 0.08 and 0.16, together with two-step endpoints. The common objective is
physical endpoint MSE plus 0.5 times two-step MSE, divided by `(2e-4)^2`.
Teachers solve the coupled PDE, so the learner is not limited to reproducing
GL3's finite-amplitude approximation.

Each of **five families × three paired seeds** receives **300 Adam updates**,
learning rate `0.001`, width 16 and a nominal cutoff of four. Premix and
precompress match learned parameter shapes and initialization; their input
ordering differs. Other families do not match parameter count, spectral
support, FLOPs or training wall time. Those costs are recorded rather than
treated as matched capacity.

The validation horizon is 0.12 with its two-step endpoint at 0.24. Validation
selects the minimum-loss checkpoint at initialization and every 50 updates.
All checkpoints are frozen before diagnostics are evaluated. A selected
initialization is retained as `SELECTED_INITIALIZATION`; it is not evidence of
a learned improvement. Numerical failures and missing checkpoints remain in
the candidate plan.

The diagnostic bank has four independent parents in each of twelve regimes:
smooth, mixed, in-band pairs, held-out high pairs, near-Nyquist, rough, near
zero, near one, stronger reaction, stronger diffusion, longer rollout and
held-out resolution. Ordinary endpoint time is 0.24 with 1/2/4 steps. The longer
rollout ends at 0.96 with 4/8/16 steps. Resolution transfer uses a held-out 64×64
cohort. These cases examine interpolation, coefficient shifts, spectrum shifts,
bounds, accumulated error and resolution sensitivity separately.
The 64×64 cohort uses independent initial fields; it is not a paired
discretization comparison of the same parents at two resolutions.

FP64 teacher preparation allows five attempts and a finest RK4 count of 32,768,
using tolerance `2e-7/sqrt(cells)`. Any unresolved dataset reference withholds
training. Actual training and inference use FP32 with TF32 disabled. Neural
RMS and maximum targets are `2e-3`, `2e-4`, `2e-5`, `2e-6`; they differ from the
FP64 numerical panel's strictest targets. Strang, ETDRK4 and fused GL3 also run
on the same inference device and precision in this panel. One warmup and three
measured complete rollouts retain raw timing and memory evidence.
GPU memory measurements include the comparison's resident models and allocator
state; they are not isolated per-model memory requirements.

Every diagnostic parent is reused across training seeds and step counts;
repeated classical measurements are also paired. Do not count them as new
independent parents or pool neural, numerical and scaling panels into one
universal win rate. Frontiers choose the fastest eligible candidate from a
fixed step grid after observing reference errors. They are post-hoc
work–precision comparisons, not an adaptive deployed controller.

## What the results will answer

1. **Accuracy:** Does output selection preserve the important low-frequency
   correction where input truncation fails, and when do discarded outputs or
   finite-amplitude terms still dominate?
2. **Scaling:** Does parent chunking help at larger batches, and can any complete
   corrected solver beat Strang or ETDRK4 at the declared active targets?
3. **Preparation:** Are every training and diagnostic label independently
   resolved, with split, normalization and source provenance intact?
4. **Neural:** Does a trained premix checkpoint improve on its paired
   precompression ablation and competitive neural/classical controls at matched
   accuracy? Does the local bypass help rough or shifted conditions enough to
   justify its cost?

Correctness checks distinguish exact identities from deliberate approximation.
They include projection/full-cutoff parity, direct convolution, mixed-parent
means, exact subflow limits, gradients and independent dense physical-space
quadrature checks. GPU tests must actually execute inside the allocated GPU
task. Cloud CPU tests cannot substantiate A100 correctness or performance.

This remains one scalar periodic reaction–diffusion system and a bounded
development experiment. It does not test different PDE families, nonperiodic
boundaries, arbitrary geometry, continuum convergence, an exhaustive
architecture search or a published benchmark reproduction. Favorable and
adverse labels describe the frozen cases; an unexpected loss remains a useful
result. Completion alone is not a scientific pass.

## Tower and review artifacts

`paths` prints the exact `TDN_TOWER_DIR` and `TDN_TOWER_METRICS` for each stage.
Use that report path rather than another workflow's latest report:

```bash
bash scripts/carc_premix.sh paths "$RUN"

# Substitute one exact TDN_TOWER_DIR printed above.
REPORT=/home1/aadaniel/projects/TDN/runs/REPLACE_RUN_ID/tower/REPLACE_REPORT_ID
bash scripts/tower.sh validate "$REPORT"
bash scripts/tower.sh show "$REPORT"
```

Each report contains live `metrics.jsonl` and applicable scientific tables:

| Report-relative path | Evidence |
| --- | --- |
| `outputs/premix_candidates.csv` | Candidate status, error decomposition, timing and work |
| `outputs/premix_frontiers.csv` | Matched-target selections and unattained targets |
| `outputs/premix_checks.csv` | Reference acceptance, numerical identities and approximation diagnostics |
| `outputs/premix_training.csv` | Training progress, validation history and checkpoint selection |
| `outputs/premix_comparisons.csv` | Paired model comparisons, feasibility, speed and initialization-selected controls |
| `outputs/premix_groups.csv` | Separate results by declared regime/category, seed, norm and target |

Tables retain source paths, hashes and record pointers into canonical stage
JSON. The run also preserves execution metadata, immutable protocols, dataset
and checkpoint seals, raw timings, incomplete records and failure reasons.
Do not infer missing values or scientific success from a terminal Tower state.
The neural stage also writes `summary.txt` and `comparisons.json`. These retain
every declared comparison, distinguish coverage from speed among feasible
solvers, and separately count comparisons in which both selected checkpoints
were trained. Separation of raw timing ranges is descriptive, not a confidence
interval or a significance test.

In a separate non-CARC development checkout, the CPU integration wrapper is:

```bash
bash scripts/premix_local.sh --run-dir runs/premix-local-001
```

It defaults to smoke and executes all four stages sequentially on CPU. Its
results are local CPU validation; use the allocated CARC workflow above for
the declared A100 experiment.
