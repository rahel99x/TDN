# Response runbook

This is the separate M0–M7 + MX program. It preserves earlier runs. The desktop
profile is `/home/rahel/TDN`, Fedora, RTX 4090, 110000 MiB scheduler memory.
Use the repository `.venv`; do not use Conda. Each allocation is at most 35
minutes. Hours of training use exact optimizer continuations.

Do not pull code or change the venv while a campaign or recovery is active.
Source, software and data changes invalidate reuse. A source change after
calibration requires a new calibration. Record `git rev-parse HEAD` with the run.

## 1. Prepare and measure the native workload

```bash
cd /home/rahel/TDN
git pull --ff-only
git rev-parse HEAD
bash scripts/fedora_response.sh doctor
```

If the existing Fedora profile is already correct, reuse it. If configuring a
new checkout, the supported command is:

```bash
bash scripts/fedora_response.sh configure \
  --cpu-partition local --gpu-partition local --gpu-gres gpu:1
bash scripts/fedora_response.sh setup
```

Choose fresh IDs. These examples are literal usable IDs; change the suffix if
one already exists. The third target is included explicitly so its throughput
and reference behavior are measured, too:

```bash
bash scripts/fedora_response.sh plan --smoke --run-id response-smoke-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
bash scripts/fedora_response.sh run --smoke --run-id response-smoke-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
```

Native smoke runs the mandatory CUDA checks, 64-update calibration trials and a
tiny integration comparison. It cannot establish trainability or superiority.
The two-update local CPU smoke is never accepted as an RTX throughput estimate.

## 2. Check completion, then run the trainability pilot

```bash
bash scripts/fedora_response.sh status response-smoke-a
bash scripts/fedora_response.sh logs response-smoke-a --lines 200
bash scripts/fedora_response.sh validate response-smoke-a
bash scripts/fedora_response.sh paths response-smoke-a

bash scripts/fedora_response.sh plan --pilot --calibration response-smoke-a \
  --run-id response-pilot-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
bash scripts/fedora_response.sh run --pilot --calibration response-smoke-a \
  --run-id response-pilot-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
```

Inspect the printed work plan before submission. It separates measured native
throughput, extrapolation factors, requested updates, numerical allowances and
summed allocation ceilings. The total ceiling is not a runtime forecast.
The plan also counts declared reference-array bytes. Complete worst-case disk
use is explicitly unavailable: checkpoints, histories, plots, exports and
additional recovery attempts add storage. Keep adequate project-local space.
The exact per-component estimates and source hashes are also retained in the
submitted `protocol.json`. Each reference unit reserves time for all declared
phase descendants and any extra source-supervision solves; failure to converge
within its displayed allowance remains an unresolved reference.

The pilot executes the optimizer/scaling factorial and both individual and
joint six-regime fits. Null and precision-floor cases remain separate. A tiny
training-fit diagnostic does not select a deployment checkpoint; validation
does. Unresolved teachers, missing trials and time censoring cannot pass a gate.

## 3. Development follows the recorded gate

```bash
bash scripts/fedora_response.sh validate response-pilot-a
bash scripts/fedora_response.sh gate response-pilot-a

bash scripts/fedora_response.sh plan --development --pilot response-pilot-a \
  --calibration response-smoke-a --run-id response-development-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
bash scripts/fedora_response.sh run --development --pilot response-pilot-a \
  --calibration response-smoke-a --run-id response-development-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
```

A negative gate prevents this expansion; it does not erase the pilot or stop
the separately budgeted compression and closure diagnostics. Inspect `gate.json`
and the atlas before revising the research question or training opportunity.
An experimental revision needs a new protocol and run ID.

If diagnostics justify a larger trial, declare both work and allowance. For
example, these **optional** flags grant the temporal family 25000 updates and a
six-hour trial ceiling, split into bounded resumable jobs:

```bash
# Add to BOTH plan and run, for the same new run ID:
# --updates temporal=25000 --trial-seconds temporal=21600
```

Do not apply six hours to every family by default. Different capacities,
auxiliary objectives, validation banks and source ranks receive explicit cost
factors. Censoring means the allotted experiment is unresolved; it is not proof
that a model cannot learn.

## 4. Freeze before fresh confirmation

After development finishes, review its claims and costs. No automatic command
turns development observations into confirmation. The explicit freeze copies
selected checkpoints into a separate immutable package and binds the protocol,
models, source, software and fresh parent seed policy.

```bash
bash scripts/fedora_response.sh validate response-development-a
bash scripts/fedora_response.sh freeze response-development-a \
  --output-dir /home/rahel/TDN/outputs/response-freeze-a

bash scripts/fedora_response.sh plan --confirmation \
  --freeze /home/rahel/TDN/outputs/response-freeze-a/freeze.json \
  --run-id response-confirmation-a
bash scripts/fedora_response.sh run --confirmation \
  --freeze /home/rahel/TDN/outputs/response-freeze-a/freeze.json \
  --run-id response-confirmation-a
```

Freeze requires complete registered development evidence and credible controls;
it does not require favorable results. A valid negative comparison can be
confirmed. Confirmation counts, targets and estimands come from the freeze and
cannot be replaced on the command line. No training or selection reads its truth.

## 5. Monitor, recover and send review evidence

```bash
bash scripts/fedora_response.sh status response-development-a
bash scripts/fedora_response.sh logs response-development-a --lines 200
bash scripts/fedora_response.sh paths response-development-a
bash scripts/fedora_response.sh validate response-development-a

# After an interruption, keep the original attempt:
bash scripts/fedora_response.sh recover response-development-a \
  --run-id response-recovery-a --plan
bash scripts/fedora_response.sh recover response-development-a \
  --run-id response-recovery-a

# Once jobs are terminal:
bash scripts/fedora_response.sh collect response-development-a
bash scripts/fedora_response.sh compact response-development-a --mode review
```

Use the printed Tower/atlas paths. The project exports its own files and does
not modify Tower. Accounting may initially be `PENDING_ACCOUNTING`; a later
collection can refresh it without changing sealed science. `collect` creates a complete review archive, including binary replay files.
`compact --mode review` creates the smaller transfer package: exact metrics,
histories, manifests and logs, with an omission inventory for large replay
arrays/checkpoints and rendered figures. That compact package is not a full
binary replay package. Upload the printed `index.json` and every numbered part; do not
use XZ-9 for a routine multi-gigabyte review.

### Rerun after the native smoke fixes

The initial `response-smoke-a` exposed two implementation defects: dependency
verification used JSON object order instead of the declared stage order, and
the fresh checkpoint-check process omitted the parent's TF32-off setting.
These were execution defects, not training-budget or VRAM failures. The fix
retains exact checkpoint-output parity and the source/protocol seal checks.
The mandatory CUDA suite now has 54 cases, including a fresh-process direct-FNO
checkpoint check for each target. `checkpoint_reload.json` records both
processes' arithmetic settings and checkpoint/model hashes.

The next attempt, `response-smoke-b`, stopped at audit job 938: all 12 new
dependency-order test cases inherited `SLURM_JOB_ACCOUNT` after clearing the
job/step IDs and execution mode. The real storage guard then rejected their
mocked local CLI as a CARC invocation. The test now isolates inherited Slurm
and TDN variables and explicitly seeds an account to cover this regression.
Production runtime checks are unchanged. The failed audit blocked the scientific
successor jobs before they could exercise the corrected CUDA path; the final
report still ran.

After all old jobs are terminal, pull the fix and use a fresh smoke ID:

```bash
cd /home/rahel/TDN
git pull --ff-only
bash scripts/fedora_response.sh plan --smoke --run-id response-smoke-c \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
bash scripts/fedora_response.sh run --smoke --run-id response-smoke-c \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
```

Keep `response-smoke-a` and `response-smoke-b` unchanged. Exact-source `recover`
cannot reuse their artifacts with revised code. Once the new smoke finishes,
run `bash scripts/fedora_response.sh validate response-smoke-c` and use
`--calibration response-smoke-c` for the pilot. Do not use a failed smoke as
calibration for the revised source.

### Rerun after the cost allowance and Tower attachment fixes

At `3c092e1`, `response-smoke-c` passed the complete 54-case CUDA suite in
both Galerkin cost allocations, then hit their internal 240-second research
allowances. Those failures did not report CUDA or VRAM errors. Native smoke
cost allocations now allow 1800 seconds of scientific work plus reporting
headroom, for a 33-minute Slurm limit. This is a ceiling, not a completion-time
forecast. The declared models, schedules, fresh-process probes, query/batch
panels and correctness tests remain required. Local CPU smoke keeps its
240-second cost allowance.

The final report sealed its scientific atlas before its Tower export failed:
the adapter attempted to register more than Tower's 256 permitted log
attachments. Direct attachments are now capped, with the remaining entries
explicitly counted and accessible through the complete figure/table catalog.
The Tower application and scientific artifact seals are unchanged.

Keep all three earlier smoke attempts. An unchanged-source recovery would
retain the same cost allowances; the fixes change source and protocol identity,
so use a fresh smoke and calibration after the revised code is available in
the native checkout and all old jobs are terminal:

```bash
cd /home/rahel/TDN
git pull --ff-only
git rev-parse HEAD
bash scripts/fedora_response.sh plan --smoke --run-id response-smoke-d \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
bash scripts/fedora_response.sh run --smoke --run-id response-smoke-d \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
```

After every job is terminal, check both scheduler completion and artifact
validation:

```bash
bash scripts/fedora_response.sh status response-smoke-d
bash scripts/fedora_response.sh validate response-smoke-d
bash scripts/fedora_response.sh paths response-smoke-d
```

Use `--calibration response-smoke-d` for the pilot only after the revised
native smoke completes successfully. Local regression tests cannot establish
that the expanded allowance completes on the RTX 4090.

## Local implementation verification

These commands are appropriate for a separate cloud checkout without Slurm.
They do not provide native GPU or scientific confirmation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_response*.py -m 'not gpu' \
  --basetemp=.runtime/pytest-response-release
bash scripts/response_local.sh --smoke \
  --run-dir runs/response-local-release-a
```

Check `results/response-verification/README.md` for what was actually run in the
release. Mathematical correctness, successful execution, trainability and a
scientific advantage are four different outcomes.
