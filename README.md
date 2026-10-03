# Order-Anchored Temporal-Defect Networks

A runnable research pipeline for learning the finite-time error of a symmetric split solver. It includes exact noncommuting linear controls, periodic logistic reaction–diffusion, refined FP64 coupled teachers, six local-input learned models, deterministic datasets/training, checkpoint resume, and complete-solver accuracy/cost reports.

CARC deployment is fixed to **`/home1/aadaniel/projects/TDN`**, user **`aadaniel`**, account **`anakano_81`**, and a Python **`.venv`**. All temporary files, caches, data, logs and checkpoints stay under that directory. The user's storage and venv requirements replace the uploaded document's environment examples.

## Run on a Windows desktop

The native Windows edition supports a standalone Python `.venv`, PowerShell launchers, explicit CPU or NVIDIA CUDA execution, and a bounded sequential smoke adapted for an RTX 4090 with 24 GB dedicated VRAM. Extract the source ZIP to a writable local directory and begin with [WINDOWS_START_HERE.md](WINDOWS_START_HERE.md). The [Windows runbook](docs/WINDOWS.md) includes driver discovery, installation, the 12-step run, reports, resume and a prompt for a new local chat. Desktop execution keeps all generated files inside the extracted repository; it requires no Slurm allocation or lab account.

## Run on CARC

For a short CPU-only scientific check, use the new
[light-test workflow](docs/CARC_LIGHT_TESTS.md):

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main
bash scripts/carc_light_tests.sh             # preview
bash scripts/carc_light_tests.sh --submit    # one CPU job; no GPU
bash scripts/carc_status.sh
bash scripts/carc.sh logs latest --lines 200
```

This reuses your verified `.venv`, requests 2 CPUs and 8 GiB for 15 minutes,
and charges `anakano_81`. It runs focused regression tests followed by six
predeclared numerical/temporal screening cases. Reports retain failed scientific
checks; completion does not authorize a pilot. No training or GPU job is queued.

The [automated CARC workflow](docs/CARC_AUTOMATION.md) submits two sequential
allocations: CPU setup/tests/audit/data, then A100 tests/calibration/training/
evaluation/benchmark. Defaults already match the observed account, Python module
and driver-compatible PyTorch build. You do not need to source a helper or set
environment variables before starting.

```bash
cd /home1/aadaniel/projects/TDN
bash scripts/carc_start.sh             # preview; no jobs or files created
bash scripts/carc_start.sh --submit    # one CPU job and one dependent A100 job
bash scripts/carc_status.sh            # latest workflow, stages and scheduler
bash scripts/carc.sh logs latest       # relevant logs and report paths
```

The CPU job creates the project `.venv` when needed and verifies a compatible
existing venv before reusing it. Every substantial stage runs inside an actual
`srun` task. Failed tests, failed gates and checkpoint pauses stop later stages;
there are no automatic retries.

After updating code or fixing a failed workflow, preserve its files and restart
with one command. It cancels only verified pending jobs owned by that workflow:

```bash
bash scripts/carc_restart.sh --submit
```

The default [selected smoke config](configs/carc-smoke.yaml) has 12 optimizer
steps, an 8×8 grid, eager FP32 and FP64 teachers. Four bounded CPU candidates were
compared using validation parents; [the selection report](docs/CARC_CONFIG.md)
records the settings and measured limitations. Actual A100 performance tuning
remains unmeasured.

The strict pilot is a separate explicit opt-in:

```bash
bash scripts/carc.sh start --profile pilot --pilot-budget 1000
```

This is a dry run. Failed G2/G4 gates continue to block the larger pilot.
The [detailed CARC runbook](docs/CARC.md) retains discovery, individual-stage
submission and manual dependency commands for advanced use.

## Local CPU implementation check

In an existing isolated cloud checkout, use its project venv and an explicit CPU-only checkout override:

```bash
export TDN_LOCAL_TEST_ROOT="$(pwd -P)"
export TORCH_VERSION=2.10.0
export TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cpu
bash scripts/setup_venv.sh
bash scripts/smoke.sh configs/smoke.yaml "$(pwd -P)/runs/local-smoke"
```

The smoke runs the CPU suite, generates checksummed multi-horizon FP64 targets, trains a tiny model, evaluates independent diagnostic parents and writes runtime/error frontiers. Reuse of a completed run directory is rejected. On CARC, substantial CPU work also requires an allocated `srun` task.

After a successful smoke, compare all six learned controls serially on the same dataset:

```bash
bash scripts/compare_local.sh configs/smoke.yaml \
  "$(pwd -P)/runs/local-smoke/dataset" "$(pwd -P)/runs/learned-controls"
```

This helper has an explicit small CPU development budget and preserves failed/infeasible controls. The generic h-conditioned baseline and Taylor/hypersolver-inspired adaptation are distinct from the h-independent temporal model. Optional KAN and oscillatory branches are gated.

## Evidence and limits

[results/validation.json](results/validation.json) records actual prepared-machine checks; [results/gates.json](results/gates.json) records scientific status. CPU and mocked scheduler tests do not establish CARC authorization or A100 behavior. GPU parity tests are skipped locally and are required to execute in the allocated GPU stage.

The current small development regime has no demonstrated TDN efficiency advantage over cheap classical controls. A working training pipeline does not establish scientific merit. The smoke tolerance is a proposed screening value; confirmation remains blocked pending a reviewed tolerance/observable, physical ranges, untouched parents, and measured GPU gates.

See [SOURCE_LEDGER.md](SOURCE_LEDGER.md), [REPO_AUDIT.md](REPO_AUDIT.md), [implementation map](docs/IMPLEMENTATION_MAP.md), [WORK_LOG.md](WORK_LOG.md) and [NEXT_ACTIONS.md](NEXT_ACTIONS.md). The original [implementation specification](docs/IMPLEMENTATION_SPEC.md) is retained for provenance. Gray–Scott, advection, distributed HALO, KAN, custom kernels and a confirmatory campaign are later extensions, not validated features of this initial RD pipeline.
