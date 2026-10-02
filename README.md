# Order-Anchored Temporal-Defect Networks

A runnable research pipeline for learning the finite-time error of a symmetric split solver. It includes exact noncommuting linear controls, periodic logistic reaction–diffusion, refined FP64 coupled teachers, six local-input learned models, deterministic datasets/training, checkpoint resume, and complete-solver accuracy/cost reports.

CARC deployment is fixed to **`/home1/aadaniel/projects/TDN`**, user **`aadaniel`**, account **`anakano_81`**, and a Python **`.venv`**. All temporary files, caches, data, logs and checkpoints stay under that directory. The user's storage and venv requirements replace the uploaded document's environment examples.

## Run on a Windows desktop

The native Windows edition supports a standalone Python `.venv`, PowerShell launchers, explicit CPU or NVIDIA CUDA execution, and a bounded sequential smoke adapted for an RTX 4090 with 24 GB dedicated VRAM. Extract the source ZIP to a writable local directory and begin with [WINDOWS_START_HERE.md](WINDOWS_START_HERE.md). The [Windows runbook](docs/WINDOWS.md) includes driver discovery, installation, the 12-step run, reports, resume and a prompt for a new local chat. Desktop execution keeps all generated files inside the extracted repository; it requires no Slurm allocation or lab account.

## Run on CARC

The [CARC runbook](docs/CARC.md) gives the complete sequential procedure and per-stage allocation table. Begin on the login node with read-only discovery:

```bash
cd /home1/aadaniel/projects/TDN
bash scripts/probe_carc.sh
cp scripts/user.env.example.sh scripts/user.env.sh
chmod 600 scripts/user.env.sh
# Select the live standalone Python module if needed, then:
source scripts/user.env.sh
```

Review a five-minute allocated driver audit, submit it explicitly, then choose the compatible official CUDA wheel and exact PyTorch release in `user.env.sh`. Install the venv through a CPU allocation:

```bash
bash scripts/submit.sh driver-audit configs/smoke.yaml   # dry run
# Repeat with --submit only after reviewing the request.
# After its report, set TORCH_VERSION and TORCH_WHEEL_INDEX and source user.env.sh.
bash scripts/submit.sh setup configs/smoke.yaml          # dry run
# Repeat with --submit, then wait for successful setup.
```

Review the dependency-aware readiness sequence:

```bash
bash scripts/pipeline.sh configs/smoke.yaml --pipeline-id smoke-review
```

CPU tests → numerical audit → dataset generation → allocated A100 parity/signal tests → A100 memory/precision/compiler calibration. Add `--submit` explicitly with a fresh identifier to execute it. Every real submission verifies live account and partition limits; every allocation charges `anakano_81`. GPU work requests one full A100 40 GB and preserves Slurm's GPU visibility.

The headroom-required development pilot is separate:

```bash
bash scripts/pipeline.sh configs/pilot.yaml \
  --pipeline-id pilot-review --pilot-budget 1000
```

This remains a dry run. Failed scientific gates and exit75 pauses stop `afterok` successors. No full campaign, array, multi-node run, 128³ training or confirmatory experiment is launched automatically. The runbook explains explicit submission, local config, manual dependencies and checkpoint resume.

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
