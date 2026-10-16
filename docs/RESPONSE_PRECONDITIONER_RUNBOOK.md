# Response-preconditioner implementation runbook

Read the [standing research program](TDN_RESEARCH_PROGRAM.md) and
[mathematical proposal](TDN_ARCHITECTURE_PROPOSAL_20261015.md). This namespace
implements the first bounded FD milestone, the separate R7 identification
study and the new [P1 data-scaling phase](RESPONSE_PRECONDITIONER_SCALING_PHASE1.md).
The original verifier/smoke remains `IMPLEMENTATION_ONLY`; R7 and P1 are
`BOUNDED_DEVELOPMENT`. None promotes the historical Tangent Response B STOP,
establishes native GPU readiness, or starts an automatic successor.

The first milestone implements 40 affine coefficients over three commutator
and two two-sided heat branches, a classical ordered-response backbone,
parent-balanced ridge fitting and a separate cubic residual auditor. The model
uses FP64 arithmetic on actual FP32 initial fields. P1 holds that model and
its objective fixed while comparing independent parents and source directions.
Learned encoders, adaptive rollouts and expansion beyond its frozen bank
remain later gated milestones.

[Native job 2930](../results/native-response-preconditioner-2930/README.md)
passed all 184 checks and the tiny smoke at `627de9f`. The next
[bounded repair](RESPONSE_PRECONDITIONER_REPAIR_PROTOCOL_20261015.md) preserves
the objective and development fixture, improves ridge scaling, streams the
five effective inference coefficients, and adds a fresh classical endpoint
baseline. Its protocol is `implementation.v2`, producing 82 metric rows;
the old run remains immutable. Prepare a new contract after pulling changes.
The [repair receipt](../results/response-preconditioner-repair-20261015/README.md)
records 202 mandatory CPU passes, 33 overlapping Raven compatibility passes,
and the completed implementation replay. [Native job 2936](../results/native-response-preconditioner-2936/README.md)
now verifies the revised source at `261365b`: 202 checks and the complete smoke
passed. This closes the repair milestone. No repeat of the unchanged smoke is
needed. The [R7 design](RESPONSE_PRECONDITIONER_R7_DESIGN.md) now has its own
implemented protocol and launcher below. The [R7 verification receipt](../results/response-preconditioner-r7-verification-20261015/README.md) records
298/298 required CPU checks and 53/53 overlapping Raven compatibility checks.
[Native R7 job 2939](../results/native-response-preconditioner-r7-2939/README.md)
completed at `43837c5`: all 298 checks, 192 accepted queries, 2,112 metric rows
and 336 timing bundles pass independent review. The retained review includes
figures, a term legend and read-only reproduction commands. The current model
has no established practical advantage over Lawson two; structural feature
coverage, regularization and a resolved sampled-span residual remain relevant.
Do not repeat the unchanged study. The launch commands below remain the R7 v1
interface, not an instruction to expand the bank or proceed to an encoder.

## P1: independent-parent and source-direction scaling

The new [Phase 1 design](RESPONSE_PRECONDITIONER_SCALING_PHASE1.md) is the
current implementation phase; it is separate from the old R7 protocol and
the proposed R8 encoder. The [verification receipt](../results/response-preconditioner-scaling-verification-20261016/README.md)
records 388 mandatory CPU passes, eight additional Raven compatibility passes,
and a completed 48-query implementation fixture. Native P1 remains pending. Its frozen
[`tdn.response_preconditioner.scaling.v1`](../tdn/analysis/response_preconditioner/scaling_protocol.py)
uses 32 training, eight validation and eight diagnostic parents across four
balanced discrete morphology families. Nine nested fits cross 8/16/32 training
parents with 2/4/8 calibration directions. All nine validation selections seal
before training-withheld or diagnostic truth. The model, physical loss and
ridge menu remain unchanged. These are new development data, not confirmation.
The private 48-query verification fixture uses production-seed prefixes, so
some development fields are inspected during checks. P1 also changes the
morphology and mean/amplitude distribution from R7: only comparisons within
P1 isolate parent/source count, not pooled differences against the old bank.

The complete planned work is 2,688 U/G query pairs, 16,128 withheld metric
rows and 1,008 timing bundles. Only the largest data cell is timed, against
ordered 2/8 and Lawson 2/4/8 on the fixed family-balanced diagnostic subset.
No full bank is acquired during preparation or local verification. P1's
native receipt remains pending; local checks cannot establish native execution
or a scientific benefit.

After older jobs using this checkout have finished, prepare and launch:

```bash
if cd /home/rahel/TDN &&
   git pull --ff-only origin main &&
   contract="$(.venv/bin/python scripts/response_preconditioner_scaling_native.py prepare)"; then
  raven launch scripts/response_preconditioner_scaling.sh \
    --project "$PWD" --recipe response-preconditioner-scaling \
    --sbatch=--partition=local --sbatch="--chdir=$PWD" \
    -- --contract "$contract"
else
  unset contract
  printf 'Preparation failed; no job submitted. See the error above.\n' >&2
fi
```

Use the configured CPU partition/account from `.tdn/fedora-slurm.json` when
it differs. Insert `--check` before `--` to preview. Leave `--local` out: the
study requires a genuine Raven/Slurm allocation. The guarded preparation
block leaves the terminal open on errors and never submits an empty contract.
Do not source the launcher or update the source/venv while its job is active.

The allocation is **4 CPUs, 8 GiB, 20 minutes**, with a **900-second study
budget**. It runs the exact mandatory verifier and then the study in the same
attempt. A release from another attempt is invalid. `--mode verify` after the
contract selects verification only; no study-only mode or automatic successor
exists. A changed source/software/profile needs a fresh prepared contract.

To verify implementation locally without acquiring this scientific bank:

```bash
if cd /home/rahel/TDN; then
  mkdir -p .runtime/response-preconditioner-scaling-local &&
  TMPDIR="$PWD/.runtime/response-preconditioner-scaling-local" \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python scripts/response_preconditioner_scaling.py verify --device cpu
fi
```

Once the native attempt is terminal, use its exact Raven run directory:

```bash
if cd /home/rahel/TDN; then
  .venv/bin/python scripts/response_preconditioner_scaling_native.py export \
    --run runs/launch-EXACT-a1
fi
```

The exporter prints a fresh folder beneath
`.runtime/response-preconditioner-scaling-exports/`. Upload its `README.txt`,
`index.json` and ZIP or every numbered ZIP part; each part is at most 30 MiB.
Export retains raw scientific arrays and requires terminal status. It does
not run science or require the checkout still to match the historical run.

Raven's `response-preconditioner-scaling` recipe enables experiment, evidence,
artifacts and CPU/memory observation. One job carries the entire bounded
study; arrays, GPU collection and tracing are disabled. Compact research
series expose learning-curve cells and parent aggregates, with complete
query-level CSV/JSON evidence under `outputs/study/`. Stage streams remain
under `logs/`; scheduler streams also remain in checkout
`logs/response-preconditioner-scaling/`. Raw study arrays are registered in
`outputs/native/portable-study-evidence.zip` with a packing manifest; the
exporter verifies and avoids duplicate payloads.

Read `outputs/native/status.json`, `outputs/verification/summary.json` and
`outputs/study/summary.json` first. Then review the parent manifest, all nine
sealed selections, reference/evaluation denominators, family and parent
learning curves, fit spectra and complete timing.
`outputs/study/costs.json` records unique acquisition once globally, per-cell
label demand, and reference/design/fitting/accuracy/timing durations. These
are nested elapsed components, not costs to add again to wall time. Resource
telemetry under
`outputs/native/telemetry/` observes actual worker CPU/RSS with stated coverage
limits; it is separate from supervisor graphs and per-method timing.
Passing execution does not promote a utility gate. There is no automatic
64/128-parent extension, new objective, new architecture or GPU successor.

## R7: bounded identification through native Raven

The source-bound contract is
[`tdn.response_preconditioner.r7.v1`](../tdn/analysis/response_preconditioner/r7_protocol.py).
It reuses the four inspected parents on the 5-by-5 same-grid FD problem,
preserving the two-training/one-validation/one-diagnostic split. Six intervals
per parent and four calibration/four withheld sources per interval produce
24 contexts and at most 192 reference queries. All intervals use the original
anchor at absolute time, including nonzero source times. These observations
are development evidence, not fresh confirmation.

R7 compares one source-shared five-coefficient projection and two local ridge
oracles with the deployable global 40-weight readout. Global fitting sees only
training calibration labels; withheld validation directions select among the
two frozen ridges and initialization. That choice seals before diagnostic
truth. Diagnostic local oracles then fit calibration sources only and seal
before withheld diagnostic sources are acquired. Both local ridges remain
reported; diagnostic outcomes do not select between them. Their induced
penalties account for context mass and feature norm, and their objectives
relax the shared global coefficient map.

The native entrypoint runs **verification, then the study**, in one real
Raven/Slurm allocation: **4 CPUs, 8 GiB, 20 minutes**, with a 900-second study
budget. The study cannot run alone or consume a verification release from
another attempt. Preparation freezes source, installed software, Fedora profile
and both protocols; it creates no scientific labels. Keep this checkout and
its venv unchanged while its jobs are queued or running.

After earlier jobs using the checkout have finished, prepare and submit:

```bash
if cd /home/rahel/TDN &&
   git pull --ff-only origin main &&
   contract="$(.venv/bin/python scripts/response_preconditioner_r7_native.py prepare)"; then
  raven launch scripts/response_preconditioner_r7.sh \
    --project "$PWD" --recipe response-preconditioner-r7 \
    --sbatch=--partition=local --sbatch="--chdir=$PWD" \
    -- --contract "$contract"
else
  unset contract
  printf 'Preparation failed; no job submitted. See the error above.\n' >&2
fi
```

This guarded block leaves the interactive terminal open on a preparation
failure. Use the CPU partition in `.tdn/fedora-slurm.json` if it differs from
`local`; add its required account before `--` when applicable. To preview the
prepared command, insert Raven's `--check` before `--`. The default mode runs
both stages; `--mode verify` after the contract selects verification only.
There is no `study`-only mode. Omit Raven `--local`: genuine allocation,
ownership, captured-script and private-runtime checks gate scientific work.

Local implementation verification is available without acquiring the bank:

```bash
if cd /home/rahel/TDN; then
  mkdir -p .runtime/response-preconditioner-r7-local &&
  TMPDIR="$PWD/.runtime/response-preconditioner-r7-local" \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python scripts/response_preconditioner_r7.py verify --device cpu
fi
```

Verification records the exact collected test roster and rejects failures,
errors, skips, duplicates or missing cases. Cloud checks and scheduler fixtures
establish implementation behavior only. Job 2939 supplies the actual native
receipt for `43837c5`; any changed source/protocol needs a fresh prepared
contract and its own execution evidence.

### R7 measurements and outputs

The measured classical controls are ordered 1/2/4/8-panel responses and fixed
2/4/8-step Lawson responses. Five paired randomized timing rounds cover each
diagnostic context's four withheld sources, retaining separate first-call and
warmup observations. Every source pays fresh path/setup/feature work; this is
a sequential bundle without cross-source caching or vectorized amortization.
The first-call label is benchmark-local: accuracy acquisition has already run,
so it is not a cold-process timing. Reported variation is empirical
min/median/max, not an independent-sample confidence interval. Local oracles
are offline fits and never enter a deployable speedup comparison.

Physical RMS, maximum, signed mean and centered errors retain source-family,
context, split and method identity. Full actions `U z`, fitted remainders
`(U-P) z`, and matched-mean feedback `(U-G) z` have separate interpretation.
The resolution floor is ten times the estimated reference uncertainty;
near-null corrections remain unresolved. Sensitivity reports coefficient
changes separately from normalized withheld prediction amplification. Its
reference-sized aggregate label ball is an empirical scenario, not a certified
or per-source teacher error bound. See the [R7 term guide](RESPONSE_PRECONDITIONER_R7_DESIGN.md#term-guide)
and the [original mathematical legend](TDN_ARCHITECTURE_PROPOSAL_20261015.md).

Raven owns the outer run; registered artifact paths are relative and portable.
The `response-preconditioner-r7` recipe enables experiment, evidence, artifacts
and CPU/memory observation, with no array, GPU collection or tracing. Research
metrics live under `r7/` phases. Actual worker process-tree telemetry remains
separate from Raven's supervisor graphs and from science metrics. Its sampled
RSS is not per-method or allocation-wide peak memory.

| Path within the Raven run | Evidence |
| --- | --- |
| `outputs/native/status.json`, `contract.json`, `profile.json`, `verification-release.json` | Native execution, original prepared contract/profile bytes and exact verification release |
| `outputs/native/telemetry/` | Actual worker CPU/RSS samples and coverage limits |
| `outputs/verification/` | Collected case roster, JUnit, check summary and source-bound verifier artifacts |
| `outputs/study/protocol.json`, `source.json`, `source_after.json`, `software.json` | Frozen workload and actual code/software identity |
| `outputs/study/query_index.json`, `denominators.json`, `references.json` | Every planned, attempted, accepted, unresolved, failed or missing query and reference refinements |
| `outputs/study/evaluation_denominators.json` | Expected, attempted, failed, unresolved and missing method evaluations and timing bundles |
| `outputs/study/parents.json`, `parents.npz`, `queries/` | Parent identities and numeric source/reference/design/prediction arrays; load NPZ with `allow_pickle=False` |
| `outputs/study/selection.json`, `fits/` | All global candidates, sealed selection, local projections/ridges, spectra and objectives |
| `outputs/study/sensitivity/` | Label perturbations, coefficient/prediction gains, actual independent-reference differences and replay arrays |
| `outputs/study/metrics.json`, `metrics.csv`, `work.json` | Portable science errors and explicit operation-counter scope |
| `outputs/study/timing.json`, `timing_records.json` | Completed context timing summaries and individually retained raw observations, including partial rounds |
| `outputs/study/work_precision.json`, `work_precision.csv` | Identical-context withheld accuracy joined with complete bundle timing; no fitted acceptance threshold or speedup claim |
| `outputs/study/summary.json`, `decisions.json`, `journal.jsonl`, `artifacts.json` | Execution separately from descriptive research decisions, access order and artifact hashes |
| `logs/` | Registered stage stdout/stderr; scheduler streams also reside in the checkout's `logs/response-preconditioner-r7/` |

Unresolved training or selection references block fitting; failed acquisition
and unattempted queries stay in the denominator. Reference uncertainty combines
refined Lawson disagreement and independent DOP853 disagreement, with both RMS
and maximum acceptance. A source query includes `U` and `G` and shares a
three-second allowance, at most 1,016 Lawson steps and 8,192 independent RHS
calls. Exhausted budgets retain the attempt; they do not expand the workload.

A completed study provides descriptive work-precision evidence, with no
predefined practical-utility pass threshold. Review the source-family residuals,
calibration/withheld behavior, perturbations and complete costs before choosing
another protocol. Full-operator audits, independently derived BCH controls,
new endpoint/trajectory evidence, nonlinear corrections, learned encoders,
GPU work and fresh confirmation remain deferred. Historical A PASS / B STOP
and the unresolved native 2936 endpoint benefit stay unchanged. The completed
2939 review identified a bounded full-operator audit and an independently
derived nonautonomous BCH/Magnus control as useful next questions. The user
subsequently chose the P1 process phase above to test data coverage first.
Those audits remain proposed controls; P1 preserves the architecture and
objective and requires its own frozen protocol and verification before new
numerical acquisition.

## Original implementation-only local CPU checks

Use the checkout's existing standalone `.venv`, outside a Slurm allocation:

```bash
cd /home/rahel/TDN
bash scripts/response_preconditioner_local.sh verify
bash scripts/response_preconditioner_local.sh smoke
```

The default output root is fresh and printed by the command. An explicit
`--output runs/response-preconditioner-my-check` must also be fresh. Existing
results are preserved, not reused or overwritten. Run sequentially; the wrapper
locks its own local workflow and takes a shared environment lock.

`verify` runs the declared existing/new CPU test roster and preserves its
JUnit receipt. `smoke` exercises a fixed tiny parent-disjoint bank, independent
references, conditional fitting, selection and held-out diagnostics. Neither
is a performance benchmark or a representative scientific study.

The interpreter and all caches/temp/logs/output stay inside the checkout.
CUDA is not silently substituted with CPU: this runnable explicitly supports
CPU only. Larger training and unsupported target arguments must be rejected.

## Native Raven CPU verification

Use the existing checkout `.venv`, configured `.tdn/fedora-slurm.json` and
installed Raven. The [Fedora setup guide](FEDORA_SLURM.md) covers a missing
profile/environment. Preserve source and the venv while any jobs using this
checkout are queued or running; update only after those jobs are terminal.

The native entrypoint is a single allocated worker: **4 CPUs, 8 GiB, 20
minutes**, with `verify` followed by `smoke`. Prepare a fresh source/software/
profile contract on the submission host, then preview:

```bash
if cd /home/rahel/TDN &&
   contract="$(.venv/bin/python scripts/response_preconditioner_native.py prepare)"; then
  raven launch scripts/response_preconditioner_verify.sh \
    --project "$PWD" --recipe response-preconditioner \
    --sbatch=--partition=local --sbatch="--chdir=$PWD" \
    --check -- --contract "$contract" --mode all
else
  unset contract
  printf 'Preparation failed; no job was submitted. See the error above.\n' >&2
fi
```

Submit that prepared contract:

```bash
if [[ -n "${contract:-}" && -f "$contract" ]]; then
  raven launch scripts/response_preconditioner_verify.sh \
    --project "$PWD" --recipe response-preconditioner \
    --sbatch=--partition=local --sbatch="--chdir=$PWD" \
    -- --contract "$contract" --mode all
else
  printf 'No prepared contract; run the preparation block above first.\n' >&2
fi
```

Use the CPU partition recorded in the profile if it differs from `local`.
If the profile requires an account, add `--sbatch=--account=YOUR_ACCOUNT`
before `--`. `--mode verify` or `--mode smoke` selects one bounded stage.
This entrypoint requires Raven's native Slurm execution; omit `--local`.
Options after `--` belong to TDN. There is one job, without an array or an
automatic successor. Source/software/profile changes require a new prepared
contract and a fresh launch, preserving the rejected attempt.

The recipe enables `experiment`, `evidence`, `artifacts` and CPU/memory
observation. Research series retain endpoint/action, split and method identity;
Raven records RMS, maximum, mean, centered and correction-relative errors.
Its built-in resource graphs observe the supervisor process tree, whose scope
can exclude Slurm task descendants. Actual allocated-worker CPU/RSS samples
are separately retained under `outputs/native/telemetry/` as portable CSV,
JSONL and a coverage index. Inspect the index for unavailable/partial samples.
CPU utilization is per logical core; RSS can count shared pages repeatedly.
GPU collection and tracing are disabled for this CPU milestone.

Raven owns the outer attempt and terminal status. TDN verifies the exact
captured script, original checkout, native allocation and private reporting
runtime. It registers stage stdout/stderr under `logs/`, research results
under `outputs/verification/` and `outputs/smoke/`, and execution/contract/
telemetry evidence under `outputs/native/`. All registered paths are relative
to the Raven run directory and survive moving that directory. Scheduler
streams are additionally retained in `logs/response-preconditioner/` in the
checkout. Preparation creates that parent before submission.

Actual Slurm/Raven execution was verified by job 2930 for `627de9f` and by
job 2936 for the revised implementation at `261365b`. The commands above remain
the implementation verifier, not an R7 scientific launcher. Cloud CPU tests
and scheduler fixtures alone do not establish native readiness. Preserve
source and the venv while relevant jobs are active; use fresh output roots
after numerical/source-contract changes. Both receipts have partial worker CPU
coverage; outer supervisor resource graphs are not allocation usage.

### If preparation fails or the terminal closes

An earlier version of these instructions ended the preparation line with
`|| exit`. Pasting that into an interactive terminal executes the terminal
shell's `exit` command when preparation fails. The guarded blocks above keep
the preparation error visible and prevent submission without a contract.
The underlying preparation error still needs to be resolved; a terminal
closing does not identify which prerequisite failed.

To inspect preparation alone, without submitting a job:

```bash
if cd /home/rahel/TDN; then
  if .venv/bin/python scripts/response_preconditioner_native.py prepare; then
    printf 'Preparation succeeded; the contract path is printed above.\n'
  else
    printf 'Preparation failed; retain the error printed above. No job was submitted.\n' >&2
  fi
fi
```

Run shell launchers with `bash ...` or `raven launch ...`, as documented.
Do not source them into the interactive shell: their `exec`, `exit` and shell
options are intended for their own process. Preparation requires the existing
standalone project venv and Fedora profile, an unchanged readable checkout,
and a submission shell outside an active Slurm/Raven attempt. Diagnose the
printed error before changing those settings.

The [six shell recovery checks](../results/response-preconditioner-terminal-exit-20261015.json)
reproduce the original exit and verify the guarded failure, successful argument
transport, missing contract/checkout and diagnostic paths. These use a recording
Raven fixture and submit no jobs.

### If preparation reports duplicate installed distributions

Python can discover the same installed distribution twice through repeated
search paths or Fedora's `lib`/`lib64` directory aliases. The initial launcher
rejected every repeated package name, including this harmless case. The
launcher now merges repeated discoveries only when the normalized package
name, version and resolved metadata directory all match. The software snapshot
still records one version per package.

Update the checkout after its queued/running jobs are terminal and prepare a
fresh contract using the guarded block above. No package reinstall is needed
for a repeated discovery of one installation. A remaining duplicate error now
prints the conflicting versions and metadata locations; preserve that output
and inspect it before changing the environment. Distinct installations, even
with equal versions, and unverifiable duplicate locations remain rejected.

The [package-discovery verification](../results/response-preconditioner-package-identity-20261015/README.md)
records the actual checks. Existing native contracts retain their original
source identities and cannot be reused across this code update.

## Reading results

Read execution status separately from research metrics. A completed tiny fit
can retain an initialization selection, poor held-out accuracy, conservative
residual bounds or no measured speed advantage. Missing references remain
unresolved. The fixed bank has two training parents, one validation parent
and one diagnostic parent, each on a 5 by 5 grid at horizons 0.04 and 0.08.
Each has two independent source directions. These are development fixtures,
not an untouched scientific confirmation cohort.

| File | Meaning |
| --- | --- |
| `summary.json` | Execution outcome, implementation-only decision, unresolved counts |
| `protocol.json`, `source*.json`, `software.json` | Frozen workload, source before/after, actual environment |
| `verification.xml`, `verification.log`, `checks.json` | Declared CPU test cases, failures and skips; any skip blocks this verifier |
| `selection.json`, `fitting.json` | Train-only fit diagnostics, validation criterion, all candidates including initialization and failures |
| `references.json` | Independent/refined teachers, matched mean response, uncertainty and unresolved status |
| `metrics.json`, `metrics.csv` | Physical endpoint/action errors; correction-relative denominator is RMS of `(U-G)z` on the identical anchor |
| `derivatives.json` | Whole-model JVP versus independent variational solve and a finite-difference sweep |
| `work.json`, `timing.json`, `architecture.json` | Explicit response counters, separately counted path/feature work, acquisition/fitting timings and model contract |
| `audit.json` | Two single-panel same-grid FD infinity-error upper bounds, with arithmetic and assumption scope |
| `arrays.npz`, `array_index.json` | Tiny numeric replay data and parent/query identities; load with NumPy `allow_pickle=False` |
| `journal.jsonl`, `artifacts.json`, `failure.txt` | Ordered events, terminal artifact hashes, and traceback when present |

Response operation counts exclude uninstrumented internal path/autograd
transforms; complete-call timing includes those costs. Single-call timings
are diagnostics, not a controlled speed benchmark. The ridge gradient/gap
is a numerical optimality diagnostic, not a rounding-certified proof.
`ordered_response_corrected_2` is the fresh classical endpoint; `zero_readout`
remains cached attribution. `fixed_feature_fitted` uses streamed inference.
The augmented condition number describes the scaled regularized system;
`original_design_numerical_rank` still describes the available data directions.
Improving the former does not repair missing information in the latter.

Derivative records separately identify endpoint-reference acceptance and the
unqualified uncertainty of the independent variational solve. Agreement with
that solve and the finite-difference sweep checks implementation; it does not
provide a derivative accuracy certificate.

The residual auditor uses exact rational cubic/Bernstein coefficients and
outward Decimal bounds. Read decimal-string bounds without rounding downward.
`CERTIFIED_FINITE_GRID_PANEL` proves an upper bound for the declared single
panel, not satisfaction of a requested tolerance. It does not certify continuum
error, later panels or derivatives; the public audit currently caps the panel
at 64 spatial degrees of freedom.

## Optional Raven compatibility checks

The original [15 October receipt](../results/response-preconditioner-verification-20261015/README.md)
records 178 mandatory cases, 27 overlapping optional compatibility cases,
and the completed tiny smoke. Compatibility was checked against Raven
`c34f0892d7bddc685c5af10489ecb114fd6adab6`. Job 2930 subsequently verified native
execution at `627de9f`; job 2936 subsequently verified the revised implementation
at `261365b`, including its expanded 202-case mandatory roster.

The mandatory `verify` roster includes launcher policy checks that need no
Raven installation. To also check materialization against a trusted Raven
source checkout, run this separately (set its actual path):

```bash
mkdir -p .runtime/response-preconditioner-compat
TDN_RAVEN_TEST_ROOT=/home/rahel/Raven \
TMPDIR="$PWD/.runtime/response-preconditioner-compat" \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_response_preconditioner_launch.py \
  -o cache_dir=.runtime/response-preconditioner-compat/pytest-cache
```

Tests requiring the trusted checkout skip if it is unavailable. Such a skip
is not compatibility evidence. Fixtures validate plans, source bindings,
failure retention and transport; actual native Slurm execution remains a
separate check for each changed source contract.
