# Response-preconditioner implementation runbook

Read the [standing research program](TDN_RESEARCH_PROGRAM.md) and
[mathematical proposal](TDN_ARCHITECTURE_PROPOSAL_20261015.md). This namespace
implements the first bounded FD milestone. All current workflow outcomes are
`IMPLEMENTATION_ONLY`, with no promotion of the historical Tangent Response
B STOP, no native GPU claim and no automatic scientific expansion.

The first milestone implements 40 affine coefficients over three commutator
and two two-sided heat branches, a classical ordered-response backbone,
parent-balanced ridge fitting and a separate cubic residual auditor. The model
uses FP64 arithmetic on actual FP32 initial fields. Learned encoders, adaptive
rollouts and larger scientific comparisons remain later gated milestones.

## Local CPU checks

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
cd /home/rahel/TDN
contract="$(.venv/bin/python scripts/response_preconditioner_native.py prepare)" || exit

raven launch scripts/response_preconditioner_verify.sh \
  --project "$PWD" --recipe response-preconditioner \
  --sbatch=--partition=local --sbatch="--chdir=$PWD" \
  --check -- --contract "$contract" --mode all
```

Submit that prepared contract:

```bash
raven launch scripts/response_preconditioner_verify.sh \
  --project "$PWD" --recipe response-preconditioner \
  --sbatch=--partition=local --sbatch="--chdir=$PWD" \
  -- --contract "$contract" --mode all
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

Actual Slurm/Raven execution remains pending until run on the configured native
machine. Cloud CPU tests and scheduler fixtures do not establish native
readiness. Preserve source and the venv while relevant jobs are active; use
fresh output roots after an update.

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

The [15 October receipt](../results/response-preconditioner-verification-20261015/README.md)
records 178 mandatory cases, 27 overlapping optional compatibility cases,
and the completed tiny smoke. Compatibility was checked against Raven
`c34f0892d7bddc685c5af10489ecb114fd6adab6`. Actual native execution is pending.

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
separate pending check.
