# Bounded M00–M23 experiments on Fedora

This program turns the [research roadmap](RESEARCH_ROADMAP_20261007.html) into
executable numerical mechanisms, trained controls, five combination panels,
and checked experiment logs. It uses the existing Fedora Slurm profile at
`/home/rahel/TDN`: Ryzen 7800X3D, 128 GB physical RAM, **110000 MiB Slurm memory**,
one RTX 4090 with 24 GB dedicated VRAM, and the project's Python `.venv`.

The implementation provides experiments, not established improvements. A job
can complete with scientifically BAD or NA results. Numerical checks support
or refute statements on their declared cases; they do not establish a general
theorem, official-paper benchmark win, or population risk guarantee. Native
CUDA results and full-budget completion must come from actual allocated runs.

The [implementation validation record](../results/roadmap-validation.json)
contains the CPU test results, complete ten-stage smoke, all-ID evidence
coverage, artifact checks and unchanged Tower validation. A separate synthetic
70,862-row reporting test preserves every record with zero omissions; it is
reporting-capacity evidence, not a scientific result.

## Run the program

For the already configured Fedora checkout:

```bash
cd /home/rahel/TDN
git pull --ff-only origin main
bash scripts/fedora_roadmap.sh doctor

# Inspect the complete allocation plan without submitting or computing.
bash scripts/fedora_roadmap.sh plan --smoke

# Submit all ten smoke stages, with their dependencies and final report.
bash scripts/fedora_roadmap.sh run --smoke
```

After the smoke jobs finish:

```bash
bash scripts/fedora_roadmap.sh status latest
bash scripts/fedora_roadmap.sh logs latest --lines 120
bash scripts/fedora_roadmap.sh validate latest
bash scripts/fedora_roadmap.sh paths latest
bash scripts/fedora_roadmap.sh collect latest
```

Once the smoke's runtime and artifact checks pass, submit the full bounded
program. Scientific BAD/NA outcomes do not invalidate an otherwise sound smoke.

```bash
bash scripts/fedora_roadmap.sh plan --full
bash scripts/fedora_roadmap.sh run --full
```

`run --full` schedules the whole program; do not manually submit its workers.
The default generated run ID is unique. `--run-id YOUR_UNIQUE_ID` is available
on `plan` and `run`. All monitoring commands accept that ID instead of `latest`.
Save the printed full run ID if you later start another workflow. Keep the
checkout, venv and profile unchanged while jobs are queued or running: their
fingerprints are prerequisites. Existing runs are preserved; these scripts do
not resume a changed experiment or silently expand an exhausted budget.

If a policy run fails with `Fourier teacher resampling requires a matching CPU
field`, update the checkout and start a fresh `run --smoke`. That failure came
from the continuum classical fallback invoking the CPU-only reference solver
with CUDA states. The deployment fallback now uses device-native Torch Fourier
products and Lawson RK4, checked against the independent CPU reference. Its
work stays on the requested device and remains included in policy costs. The
independent FP64 teacher is unchanged. Preserve the old run; changing source
invalidates its prerequisite fingerprints, so do not rerun individual workers
against its existing stage directories. After the new smoke finishes and
validates, use `run --full` as above.
The [device-fix validation record](../results/roadmap-policy-device-fix-validation.json)
records CPU regression checks and the complete smoke; the new CUDA cases still
require execution in your native allocation.

If this Fedora checkout has not been configured, use the same wrapper's setup
commands first. Existing working profiles need no replacement:

```bash
bash scripts/fedora_roadmap.sh configure --python python3.13 \
  --cpu-partition local --gpu-partition local --gpu-gres gpu:1
bash scripts/fedora_roadmap.sh setup
```

The saved profile supplies the real user, checkout, partitions, optional local
account and driver-compatible Torch installation. It does not inherit CARC's
`anakano_81` account or A100 requirements. See [Fedora setup](FEDORA_SLURM.md)
for profile and venv troubleshooting. All datasets, temporary files, caches,
checkpoints and reports stay under the checkout; `/tmp` and `/scratch1` are not
project storage. Run wrappers with `bash`; do not source their worker scripts.

For development in a separate CPU checkout outside Slurm:

```bash
bash scripts/roadmap_local.sh --smoke
# Or a larger development cohort:
bash scripts/roadmap_local.sh --development
```

The local wrapper executes all stages on CPU and rejects `--full`. Its results
are CPU development evidence and cannot satisfy native CUDA readiness or fresh
full confirmation. `--development` is also available to the Fedora `plan` and
`run` commands.

## Jobs and resource ceilings

The authoritative declaration is
[`tdn/analysis/roadmap/protocol.py`](../tdn/analysis/roadmap/protocol.py).
Stages run sequentially. Scientific jobs depend on successful prerequisite
execution; the final report uses `afterany` so failures and missing stages can
still be inventoried. A required structural failure stops the audit. A negative
accuracy, mathematical diagnostic or utility outcome remains visible and does
not by itself cancel other authorized mechanisms.

| Order | Stage | Device | CPUs | Host RAM | Slurm limit | Full numerical budget | Purpose |
|---|---|---|---:|---:|---:|---:|---|
| 1 | `audit` | CPU | 8 | 48 GiB | 20 min | 10 min | Correctness tests, physical coefficients, closure, order, C3 |
| 2 | `headroom` | CPU | 8 | 48 GiB | 45 min | 30 min | Classical work–accuracy, fusion and spatial consistency; C0 |
| 3 | `prepare` | CPU | 8 | 48 GiB | 45 min | 30 min | Train/validation/calibration teachers |
| 4 | `train` | CUDA | 4 | 32 GiB | 45 min | 30 min | DF tuning, all trained families, checkpoint freeze |
| 5 | `confirm_prepare` | CPU | 8 | 48 GiB | 45 min | 30 min | Fresh confirmation teachers after checkpoint verification |
| 6 | `confirm` | CUDA | 4 | 48 GiB | 45 min | 30 min | Frozen models, schedules, classical/FNO comparisons |
| 7 | `policy` | CUDA | 4 | 32 GiB | 30 min | 20 min | Actual routing, estimation, rejection and fallback; C2 |
| 8 | `transfer` | CPU | 8 | 48 GiB | 45 min | 30 min | Anisotropy, geometry, nonlinear PDEs and C4 |
| 9 | `scaling` | CUDA | 4 | 32 GiB | 30 min | 20 min | Larger grids/batches and charged coefficient reuse |
| 10 | `report` | CPU | 4 | 32 GiB | 20 min | 10 min | Seal validation, all 29 IDs, condensed scores and evidence index |

No job requests more than 45 minutes. The sum of requested limits is 6 h 10 min,
including 2 h 30 min of GPU-stage limits; these are ceilings, not predicted
runtime or billed consumption. Full numerical budgets total 4 hours, including
100 minutes for GPU science. Allocation limits leave room for setup, tests,
validation and reporting. Smoke numerical budgets are at most 240 seconds per
stage; development budgets are at most 900 seconds. Slurm limits remain explicit
and bounded for all profiles.

GPU work uses one actual task-visible 4090. The memory policy is the lower of
18 GiB, 75% of initially available memory and 75% of total dedicated memory,
with a 90% device-use hard threshold. Host/shared RAM is not additional VRAM.
FP32 models, TF32 off, and FP64 teachers remain explicit. CPU teachers use at
most eight worker processes, bounded by the actual allocation; numerical
kernels use single-thread settings for reproducible measurement.

Each of the four GPU jobs runs the **48 mandatory CUDA cases**: limits and
gradient checks for each of 17 families, four numerical parity cases, physical
fractional scaling, visible-allocation verification, four FP32/FP64 discrete and
continuum classical-controller cases, and four empirical-rejection/conformal-NA
policy fallback cases across both targets. JUnit validation
requires every exact case once, with no skipped, filtered, failed or duplicate
cases. Repeating these readiness tests in four jobs is not 192 independent
scientific demonstrations. There is no pending-job cap or automatic requeue.

## Cohorts, training and comparisons

The full main bank contains 24 independent train fields, 12 validation fields,
24 calibration fields and 24 fresh confirmation fields. Confirmation crosses
each field with two physics choices, producing 48 parents while retaining
**24 independent field clusters**. Each field has its own frozen phases and
coefficients. Grids, schedules, physics variants and training seeds remain
paired observations; they do not multiply the independent sample count.

The six main field regimes are low frequencies, mixed frequencies, high-mode
pairs, rough fields, near-Nyquist modes and a low-mean boundary regime.
Training uses grid 32; confirmation uses matched grids 32 and 64. All 24
calibration fields receive continuum-estimate references. The selected
confirmation continuum subset has 12 independent fields, covers all six
regimes, and includes both physics choices. Unselected continuum cases are
explicitly unavailable, not successful. Temporal teachers are independently
refined FP64 coupled solves; continuum estimates use dealiased spectral
references at grids 128 and 256 and retain spatial uncertainty.

The full program tunes 16 families on four learning-rate/batch settings
(`0.0003`/`0.001` crossed with batch 1/4), for **64 tuning trials of 120 updates**.
The shared-loss control deliberately reuses the Source optimizer choice. All
17 families then run **three paired seeds, 300 updates each: 51 final trials**.
Validation selects checkpoints, including update-zero initialization. Every
selected initialization remains labeled `SELECTED_INITIALIZATION`; it cannot
be credited as learned efficacy. The trial log records effective settings,
attempted/selected updates, gradients, parameter changes and fitting costs.
Planned, attempted, completed and selected counts remain separate. A numerical
trial failure is retained as BAD, with its validated pre-failure checkpoint
where available; an ineligible trial cannot win optimizer selection. If all
tuning trials fail, the initialized fallback remains explicitly labeled.
Invalid initialization is cataloged but not silently loaded as a usable model;
dependent comparisons become NA. Other declared arms continue, so a failed
architecture does not disappear or erase the rest of the experiment.

| Family | Actual arm | Role |
|---|---|---|
| `source` | Signed sources before spectral output compression, with DF base | Primary source control |
| `rank1` | One learned symmetric nonmidpoint pair of physical factors | Compact pair control |
| `rank2` | Two learned symmetric pairs | Rank comparison |
| `source_postcompression` | Compress input before constructing every learned source | Pure information-loss ablation; no unequal source bypass |
| `source_trust` | Source correction with validated stiffness/time envelope | Trust-on arm |
| `source_consistency` | Source plus generator/intermediate/composition supervision | Temporal supervision ablation |
| `source_df_loss` | Regime/peak/no-harm loss, independently tuned on DF | Loss and tuning arm |
| `source_df_loss_shared` | Same enhanced loss using Source's optimizer setting | Shared-setting loss attribution |
| `source_multiband` | Physical-frequency fractional low/high bands | Feature/spacing hypothesis |
| `cheap_fno` | One-layer source-informed hybrid FNO | Cheap neural comparator |
| `deep_fno` | Deeper source-informed hybrid FNO; nearest integer parameter-matched width | Depth/parameter control with actual cost reported |
| `direct_fno` | Time/physics-conditioned neural-only FNO, without DF base | Distinguishes neural-only and physical-hybrid comparisons |
| `c1_rank0` | Analytic quadratic DF response plus source branch; no learned pair | Same-combination rank-zero control |
| `c1_rank1` | Analytic response, sources, rank one and DF loss | C1 |
| `c1_rank2` | Same combination with rank two | C1 |
| `c2_rank2` | C1 rank two plus trust and temporal supervision | Trained component of C2 deployment |
| `source_embedded` | Stability-subdivided RK4 with bounded fifth-order vanishing learned residual | M20 order-constrained arm |

Analytic `df_base`, `df_quad2`, reaction-first splitting, ETDRK4 and fused GL3
controls are additional comparisons, not members of the 51 trained trials.
Nominal learned rank means symmetric quadrature-node pairs; it is not Tucker
rank or the number of FFTs. Parameter matching does not imply matching compute.
The FNO arms are project implementations and do not reproduce an external
paper's official dataset, training schedule or benchmark protocol.

Frozen confirmation retains RMS, maximum, mean and centered-spatial errors,
reference uncertainty, initialization status and each prescribed schedule.
The full schedules are `[.03,.07,.17]`, its reverse, `[.025,.055,.075,.115]`,
six `.045` steps, and three repetitions of `[.03,.07,.17]`. The first four end
at `.27`; the last ends at `.81`. Targets `2e-4`, `2e-5` and `2e-6` apply
separately to RMS and maximum error. One discrete intermediate prefix per
schedule is prepared; it is labeled discrete evidence even in a report also
containing continuum endpoints. Best feasible cost selections are post-hoc
frontiers, not deployed controllers. Every-schedule and transient results
remain separate so those frontiers cannot hide reversed-step failures.

## Mechanism implementation map

Source modules are under [`tdn/analysis/roadmap`](../tdn/analysis/roadmap):
[`numerics.py`](../tdn/analysis/roadmap/numerics.py) and
[`numerical_audit.py`](../tdn/analysis/roadmap/numerical_audit.py) implement the
physical experiments; [`models.py`](../tdn/analysis/roadmap/models.py) and
[`neural.py`](../tdn/analysis/roadmap/neural.py) implement learned arms;
[`classical.py`](../tdn/analysis/roadmap/classical.py),
[`policy.py`](../tdn/analysis/roadmap/policy.py),
[`statistics.py`](../tdn/analysis/roadmap/statistics.py),
[`portability.py`](../tdn/analysis/roadmap/portability.py),
[`data.py`](../tdn/analysis/roadmap/data.py),
[`core.py`](../tdn/analysis/roadmap/core.py) and
[`engine.py`](../tdn/analysis/roadmap/engine.py) supply comparisons and evidence.
The following IDs also appear in the condensed logs and aggregate report.

| ID | Module / experiment | Executed upgrade and identifying checks | Meaning and remaining limit |
|---|---|---|---|
| M00 | `classical`; `classical/*`, `df-fusion/*`; confirmation controls | Both Strang orientations, ETDRK4, GL3, step grid and variable-step fusion; accepted-reference headroom | Headroom counts only resolved conditions with a feasible comparator. Classical dominance is a valid outcome. |
| M01 | `classical`, `policy`; `spatial/*`, total-error variants | Same fields at N/2N/4N; observed refinement regime; actual two-grid deployment indicator | Temporal agreement cannot establish spatial accuracy. Refinement estimates are not continuum certificates. |
| M02 | `numerics`, `numerical_audit`, learned C1 arms | Zero-subtracted finite-step quadratic DF response; independent amplitude coefficients, quadrature refinement, exact nulls, midpoint-zero and stiff cases | Quadrature approximates the exact quadratic coefficient; finite-amplitude flow is not exact. |
| M03 | `models`, `neural`, scaling | Rank 1/2 physical-factor fitting and C1 ranks 0/1/2; gradients, selected checkpoints, all schedules and transform-aware costs | Extra rank must earn accuracy or useful work savings; training and parameter count alone establish neither. |
| M04 | `numerics`, `models`; `m04/*`, C3 compression arms | Full signed nonlinear products before output cutoff versus compressed-input products, identical base/support and no hidden source bypass | Tests high-high-to-low/mean response and quadratic amplitude scaling. Representation alone is not solver utility. |
| M05 | `numerics`, `classical`, `portability` | Differentiable padded products, analytic aliasing counterexamples, grid-scaled filters and declared Galerkin targets | Padding a quadratic RHS does not de-alias the exact logistic subflow's infinitely many harmonics. Projection need not preserve positivity. |
| M06 | `models`, `neural`, `policy` | Actual trust envelope, validation-only support calibration, trust/combined arms and no-harm/growth checks | Constant/null/interval checks do not prove stable sensitivity. Switching correction off everywhere is not a neural benefit. |
| M07 | `policy`; `policy-*`, `classical-*` | Frozen solver-bank order, actual propose/estimate/reject/accept loop and adaptive classical fallback; router-on/off | Truth audits decisions afterward. Total and tail cost include every rejected proposal and fallback. |
| M08 | `policy`, numerical embedded audit | Step doubling versus 2/4-node stability-weighted interior residual, actual spatial discrepancy and estimator costs | Quadrature and sampled reconstruction remain empirical; the residual indicator is not a certified bound. |
| M09 | `statistics`, `policy`; calibration and risk summaries | Joint function-level maxima over the full frozen model/query bank, conformal rank arithmetic, selective-risk bounds and shift detection | Current sample sizes and coefficient shift leave formal coverage/risk guarantees NA. |
| M10 | `neural`, combined C2 arm | Generator and intermediate supervision, independent errors at unequal/reversed steps, composition and abrupt-step tests | Self-consistency alone permits a wrong flow; independent teacher errors determine support. |
| M11 | `neural`; `source_df_loss*` | Actual DF optimizer search, defect floor, regime/peak/base-harm losses, shared versus separately tuned comparison | Gains only in surrogate loss do not pass confirmation. Initialization selections remain distinct. |
| M12 | `numerics`; dynamic moment and C3 panels | Feasible evolving mean/variance closure, instantaneous identities, binary-boundary inward flux, mean-only and spatial decomposition | Exact initial rates do not close the future moment hierarchy; logistic mean is not conserved. |
| M13 | `models`, `portability` | Actual physical-frequency fractional low/high features; constant nulls; physical versus grid-index filter spacing | A learned fractional feature is not the known PDE generator; resolution transfer and feature cost are measured separately. |
| M14 | `portability`; `transfer/*`, `geometry/*` | Anisotropic tensor transport/covariance; nonuniform finite-volume Dirichlet operator, weighted filtering, quadrature and remeshing | Bounded periodic/one-dimensional geometry branches do not establish arbitrary-geometry or nonnormal-transport performance. |
| M15 | `classical`, `policy` | Identical-map DF fusion, charged fresh versus reused ETD/GL3 coefficients, physical-key invalidation, batching and standalone policy cost | Cache creation, estimator/reject work and amortization are visible. Large-grid throughput is accuracy-NA outside teacher-bank grids. |
| M16 | `neural`, `classical` | Cheap/deep hybrid and direct neural-only FNO, prepared classical controls, all targets and matched work comparisons | Local adapted controls cannot substantiate an official-paper win; post-hoc frontiers are descriptive. |
| M17 | `data`, `portability`; factor bank | Independent phases, exact same-field grid sampling, single-factor changes and amplitude×roughness interaction | Paired variants are not independent draws; the sparse factor bank does not exhaust every physical regime. |
| M18 | `data`, `statistics`, `neural`, `policy` | Split and checkpoint freeze checks, fresh fields, paired field/seed bootstrap, independent-unit counts | Inspected development cohorts and repeated grids/seeds cannot become fresh independent confirmation. Exploratory multi-arm intervals are not multiplicity-corrected discoveries. |
| M19 | `portability`; `pde/*`, geometry | Independent coupled teachers for Gray–Scott, viscous Burgers/advection and variable-coefficient Dirichlet logistic RD; real nonlinear corrector controls | Equation-specific balances replace logistic assumptions. These are numerical portability controls, not trained cross-PDE superiority. |
| M20 | `numerics`, `models`, `neural`; `source_embedded` | Learned residual vanishing at fifth order, fixed-grid RK order, charged step-doubling and stiff-mode/nonlinear checks | Fixed RK4 stages cannot acquire arbitrary fourth-order weights. Stability subdivision costs work and does not imply unconditional A-stability. |
| M21 | numerical audit, frozen neural probes, transfer | Sampled monotonicity, concavity, directional growth, bounds and near-zero expansive control | For cooperative FD logistic flow the comparison envelope is exp(rh), not universal contraction. Sampled probes are not global certificates. |
| M22 | `core`, `engine`, launchers, Tower adapter | Effective settings, required check IDs, score recomputation, hashes, source/software/allocation lineage, failures and all-ID inventory | Infrastructure evidence is distinct from mathematical model evidence; numerical timers are not billing/utilization. |
| M23 | `numerics`; cubic and C3 panels | Causal cubic Volterra response minus same-base cubic split response; independent odd-amplitude extraction and charged finite-amplitude arms | Cubic improvement must exceed teacher uncertainty and earn its cost; a second quadratic gate would produce the wrong fourth-power response. |

## Combinations and mathematical interpretation

The five bundles are implemented with their compatible controls and ablations.
They do not mean an exhaustive power-set search over 24 mechanisms. Boundaries,
operators and output constraints must be compatible before branches are
composed; the transfer branch uses its own derived operators and teachers.

| Bundle | Actual composition | Separating controls |
|---|---|---|
| C0 | Target/product consistency + strongest classical controls + independent fields + provenance | Discrete versus continuum estimates; fused versus unfused; resolved headroom versus unresolved/infeasible conditions |
| C1 | DF + analytic finite-step quadratic response + signed source branch + learned rank remainder + DF loss | Same-combination ranks 0/1/2, source-only, rank-only, analytic-only, both hybrid FNO controls and direct FNO |
| C2 | Trained trust/temporal-consistent rank-2 arm in a calibrated operational bank, temporal/spatial indicators, routing and actual fallback | Source/rank2/cheap-FNO/combined groups; doubling versus residual; temporal-only versus spatial guard; router on/off; empirical versus formal-quantile fallback |
| C3 | Full nonlinear causal sources + dynamic moments + cubic response with a single mean replacement | DF, quadratic, cubic-only, quadratic+cubic, mean-only, mean+quadratic, full combination, zero-mean spatial, pre/post source compression |
| C4 | Dealiased nonlinear products + physical fractional filtering + compatible transport/geometry and invariant guard | Nodal/dealiased, unfiltered/physical-filter/grid-index-filter, guarded/unguarded; tensor rotations, periodic versus declared Dirichlet branch; coupled RD and Burgers controls |

For the default FD/nodal logistic equation, write
`u' = A(u) + B(u)`, with `A(u)=L_N u` and `B(u)=r(u-u²)`.
DF is diffusion-half → reaction → diffusion-half. The key finite-step upgrade
uses `u=c+v`, the exact homogeneous logistic background and its sensitivity
`q(t)`. Its quadratic defect contains

\[
-r q(h)\int_0^h q(s)
\left[D_{h-s}\big[(D_s v)^2\big]
      -D_{h/2}\big[(D_{h/2}v)^2\big]\right]ds.
\]

Subtracting the DF response inside the integral preserves commuting limits
independently of quadrature. A single midpoint gives zero, so the physical
response uses at least two nonmidpoint nodes. The learned rank arms vary
symmetric node pairs and signed gains. Their phases and zero output mode are
retained; an extra amplitude-squared gate would destroy the leading response.

The cubic hierarchy adds
`d'=(L+r(1-2c))d-2r a b` and subtracts the same-base cubic response. The moment
branch uses

\[
c'=r(c-c^2-V),\qquad
V'=2\langle v,Lv\rangle+2r(1-2c)V-2r\langle v^3\rangle.
\]

Its later energy/skewness evolution is a closure approximation. Replacing the
mean once prevents double-counting the kernel's zero mode. The logged mean
and centered-spatial errors show whether a mean improvement damages the field.

Deployment distinguishes temporal, spatial and teacher errors. In the bounded
cooperative FD setting, the motivating residual inequality is

\[
\|e(T)\|\le e^{rT}\|e(0)\|
 +\int_0^T e^{r(T-s)}\|\partial_s\Psi_s-F_N(\Psi_s)\|ds.
\]

The implemented interior-node quadrature does not bound every reconstruction
or quadrature remainder. Its refinement indicator is therefore empirical.
Similarly, observed second-order spatial refinement permits a labeled
Richardson estimate, not a rigorous continuum certificate. The
[math review](RESEARCH_ROADMAP_20261007.md) records the underlying assumptions
and derivations in more detail.

## Scores and what a log establishes

Each experiment is written immediately to `rows.jsonl`, then summarized in
`review.csv`. Every required check retains its measured operand, target,
relation, category, reason, applicability and evidence class. Missing math or
gap evidence is explicitly inserted as NA. The validator recomputes individual
checks, required-check inventory, category assessments, total score and row
hash. Definitions are frozen with the source/protocol and sealed artifacts.

| Verdict | Interpretation |
|---|---|
| `GOOD` | Every required check for that row passed on its declared evidence. |
| `BAD` | At least one required check failed; passing checks and costs remain visible. |
| `NA` | No required failure was observed, but at least one required check is unavailable or inapplicable. |

The category weights are correctness 2, mathematical evidence 2, gap 3 and
utility 3. The score is

```text
1 + round(99 × passed required check weight / all required check weight)
```

Required NA checks remain in the denominator and earn no credit. Entirely
unavailable evidence scores 1; a mixed-evidence NA row can score higher. Because
of rounding, the number must always be read with the verdict and coverage.
Optional checks do not inflate the score. Scores measure **evidence attainment**,
not accuracy percentages, probability of superiority, confidence or novelty.
Gap and math assessments are stored separately. A high aggregate cannot erase
a hard required failure, a failed stage or an unavailable claim.

The 29-ID report pools the relevant recorded checks and adds NA for a required
stage without verified ID-specific evidence. Its score is a descriptive
inventory, not an independent statistical test: repeated rows remain paired.
The report's own `review.csv` contains infrastructure inventory checks; use
`experiment_summary.csv` or `mechanism_summary.json` for mechanism conclusions.

For reliability, calibration maximizes over declared queries **within each
independent function**, then across the frozen solver-bank choices. Per-model
quantiles alone would not justify an adaptively selected model. At alpha .01,
a finite split-conformal rank needs at least 99 independent calibration
functions. A zero-failure one-sided 95% binomial upper risk bound below .01
needs at least 299 independent accepted functions under the stated sampling
assumptions. The current bounded bank supplies neither. Coefficient shift also
prevents an exchangeability claim, and teacher refinement estimates are not
certified physical error bounds. Formal guarantees remain NA; the formal
policy actually executes and charges classical fallback when unavailable.

Empirical policies use a calibration-only error/indicator envelope with the
declared safety factor. Confirmatory references audit the already completed
decision. Logs distinguish accuracy supported by a refinement estimate,
resolved failure under that estimate, and uncertainty-overlapping cases.
Possible false accepts and resolved false accepts are reported separately.
Marginal conformal coverage must never be presented as conditional-on-accept
coverage.

## Reading costs and outputs

Every run lives at `runs/fedora-roadmap-.../`. The roadmap keeps its own
`.fedora-roadmap-latest.json` pointer and does not overwrite agenda, consistency
or premix runs. Useful files are:

| Path inside the run | What to read |
|---|---|
| `protocol.json`, `roadmap-workflow.json`, `slurm-profile.json`, `jobs.json` | Frozen experiment declaration, actual site profile and job/dependency IDs |
| `<stage>/summary.txt` | Short execution result, timing and review paths |
| `<stage>/review.csv` | One row per experiment: IDs, GOOD/BAD/NA, score, evidence coverage, parameters, time, memory, failed/NA checks and effective settings |
| `<stage>/rows.jsonl` | Immediate complete check operands, metrics, evidence pointers and hashes; useful after interruption |
| `<stage>/rows.json` | Same experiment ledger in ordinary JSON |
| `<stage>/execution.json`, `science_manifest.json`, `workflow-seal.json` | Source/software/task lineage and science/execution digests |
| `train/training_plan.json`, `catalog.json`, `freeze.json`, `trials/`, `checkpoints/` | Tuning/final selections, actual parameters, learning curves, gradients, initialization attribution and model hashes |
| `prepare/`, `confirm_prepare/` | Sealed FP64 fields, references, uncertainty and independent parent identities |
| `confirm/confirmation_rows.json` | Individual schedules, all-schedule outcomes and target comparisons |
| `policy/calibration.json`, `policy-details.jsonl`, `policy-summaries.json`, `policy-summary.json` | Frozen fit, every proposal/rejection/fallback, risk classes, complete cost, paired summaries and amortization |
| `report/summary.txt` | Condensed table of every M00–M23 and C0–C4, including missing stages |
| `report/experiment_summary.csv`, `mechanism_summary.json` | Primary 29-ID assessments and every failed/NA check reference |
| `report/experiment_index.json` | Links to stage experiment tables; keeps aggregate and per-experiment evidence distinct |
| `logs/`, `state/`, `reporter-tests/` | Scheduler stdout/stderr, worker/software state and mandatory test evidence |
| `state/scheduler-accounting.json` | Latest collection-time `sacct` snapshot; dated copies preserve earlier snapshots and unavailable-field reasons |
| `tower/<report-id>/metrics.jsonl` | Task-specific Tower metrics; use the exact path printed by `paths` |

Timing fields distinguish whole-stage elapsed time, training/tuning time,
coefficient preparation, warmed repeated inference, estimator/route/reject work,
fallback and standalone deployment. GPU timing synchronizes the device;
memory fields distinguish allocator measurements from sampled device usage.
CPU-only rows have no CUDA memory measurement. Dedicated memory exhaustion or
budget interruption is retained, not converted into successful completion.

Compare costs only when both methods satisfy the same accuracy target and
teacher scope. Policy summaries include total and tail costs, not only the
median. Component times are checked against the inclusive standalone timer;
unattributed loop overhead is explicit. Shared preparation/training/calibration
cost is separate from per-query latency. Break-even counts require a positive
`accurate_classical_cost - deployment_cost` denominator; otherwise no finite
amortization gain is established. All such counts are workload estimates.
Numerical timers are not Slurm billed hours or utilization; unavailable
scheduler accounting must remain unknown.

Collection snapshots only the frozen workflow's jobs and their explicit steps,
retaining `JobIDRaw`, `State`, `ExitCode`, `ElapsedRaw`, `TotalCPU`, `AllocTRES`,
`MaxRSS` and `CPUTimeRAW` when available. Allocation and step rows are distinct
and must not be summed together. Missing `sacct` or missing fields remain
unavailable. No monetary rate was supplied, so no currency cost is inferred.
`collect latest --no-accounting` explicitly omits the optional scheduler query.

Full engineering scaling uses grids 32, 64, 128, 256, 512 and 1024, batches 1,
4 and 16, up to 18 workload cases and five timing repetitions. These are
bounded allocations under the existing VRAM policy. Accuracy can be checked
against the declared teacher bank at its grids; larger-grid throughput does
not become a large-grid accuracy claim.

## Tower and collecting a review

The project emits Tower-compatible reports and complete source-linked pages
for experiments, checks, scalars, mechanisms and artifacts. The existing Tower
application is used unchanged. Each stage/job has its own report; its
`outputs/roadmap-tables.json` indexes the generated tables. Metric-point pages
retain the canonical values rather than treating one console summary as all
experimental evidence.

```bash
bash scripts/fedora_roadmap.sh paths latest
bash scripts/tower.sh list
bash scripts/tower.sh validate latest
# Also run Tower's own validator when it is installed:
bash scripts/tower.sh validate latest --native
```

`tower.sh latest` selects the latest project Tower report, which could belong
to another workflow. Use the exact roadmap report path from `paths` when that
distinction matters. The main experiment remains reviewable through its
project CSV/JSON files independently of the panel.

```bash
bash scripts/fedora_roadmap.sh validate latest
bash scripts/fedora_roadmap.sh collect latest
```

Collection preserves checkpoints, references, raw metrics, logs and failures,
and excludes pytest temporary directories. It creates a dated review archive
and SHA-256 index under `runs/`. When the archive exceeds the upload threshold,
the collector also creates numbered parts of at most 28 MiB. Send the index
and every part together; do not upload only the first part. Partial or failed
workflows can still be collected. The final report retains missing/invalid
stages as unavailable evidence instead of assigning them successful scores.

For the first review, inspect complete coverage and correctness, then
same-target accuracy, worst schedules/fields, learned-versus-initialization
selections, and complete cost. Promote mechanisms only when those observations
support their intended gap. Reaching the end of ten jobs establishes execution
of this program, not an advantage over the classical controls or FNO.
