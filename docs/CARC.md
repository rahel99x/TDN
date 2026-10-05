# CARC runbook

The [Tower integration runbook](TOWER.md) explains live metrics, grouped log
navigation, exported scientific tables, historical imports and scheduler-state
reconciliation. Each allocated workflow prints `TDN_TOWER_DIR`; pass that exact
directory as Tower's `--workdir`. CPU, GPU and resumed jobs use separate reports.

The next bounded study is [neural replication](NEURAL_REPLICATION.md), started
with `bash scripts/carc_neural_replication.sh --submit`. One 30-minute CPU job
compares five fixed families across three paired training seeds and 27 fresh
diagnostic parents. Optional A100 inference timing uses frozen CPU checkpoints
in a separate explicit 30-minute job.

For representative neural competitors, use the [neural benchmark workflow](NEURAL_BENCHMARKS.md)
with `bash scripts/carc_neural_benchmarks.sh --submit`. It compares a generic
time-conditioned MLP, periodic residual CNN, U-Net and FNO with all TDN controls.
One 30-minute CPU job trains/evaluates the initial screen; optional frozen A100
inference timing is a separate 30-minute job.

The [architecture research workflow](RESEARCH.md) runs the three proposed models
and their controls with `bash scripts/carc_research.sh --submit`. It uses one
30-minute CPU allocation for correctness tests, reference generation, bounded
neural training and diagnostic comparisons. GPU timing is a separate gated
30-minute action. The [architecture document](RESEARCH_ARCHITECTURES.md) records
the mathematical constraints and unresolved limitations.

For light CPU-only scientific screening in an existing verified venv, use
`bash scripts/carc_light_tests.sh --submit`, then
`bash scripts/carc_status.sh`. The [light-test runbook](CARC_LIGHT_TESTS.md)
describes its fixed six-case budget and retained reports.

For the default 12-step smoke, use the [automated two-job workflow](CARC_AUTOMATION.md):
`bash scripts/carc_start.sh --submit`, then `bash scripts/carc_status.sh`.
It supplies the observed CARC defaults and performs venv setup inside the CPU
allocation. The detailed commands below remain available for individual stages.

Deploy this checkout at `/home1/aadaniel/projects/TDN`, use the Linux identity `aadaniel`, and charge every allocation to `anakano_81`. All venv files, package caches, compiler caches, temporary files, datasets, logs, results and checkpoints stay inside that project. The document's environment and temporary-storage templates are replaced by this explicit user requirement: Python `venv` only and no use of `/tmp/` or `/scratch1/`.

The CARC commands below are prepared instructions. Cloud CPU tests and mocked scheduler tests do not establish live CARC access, GPU correctness, allocation success or A100 performance. A full campaign is never submitted automatically.

Use a Bash login shell. `scripts/common.sh` is a sourceable library: it preserves
the caller's shell options and returns an error for interactive policy failures.
Each executable submission or batch script enables its own strict mode and exits
on failed checks. You do not need to source `common.sh` to run the discovery or
submission commands below; the wrappers prepare the project-local environment.
If an older checkout closed your login session when sourcing it, reconnect and
update `main` before retrying. Keep strict mode inside executable scripts rather
than enabling it in the login shell.

## 1. Read-only discovery on login

From the existing project checkout, run:

```bash
cd /home1/aadaniel/projects/TDN
bash scripts/probe_carc.sh
cp scripts/user.env.example.sh scripts/user.env.sh
chmod 600 scripts/user.env.sh
```

Edit `scripts/user.env.sh` to select an available non-Conda Python 3.11–3.13 module if needed. Leave `TDN_CPU_PARTITION` unset to discover a live authorized CPU-only partition, or set it to a partition you verified. Source the file before any submission. Do not put credentials in it.

```bash
source scripts/user.env.sh
```

The probe inspects the current user, account associations, quota, partitions, GPU features, limits and Python/CUDA module inventory. It does not install packages, allocate hardware or execute numerics. The wrapper separately verifies the exact account association and live partition limits at each real submission. It prefers read-only `sacctmgr` evidence; if unavailable, it checks the complete account token in `myaccount` output for the current user. An ambiguous or missing account is a blocker. CPU partition discovery includes the requested CPU count, host memory and walltime. A visible condo partition is not evidence of authorization.

## 2. Observe the driver, then install the venv on CPU

A five-minute A100 request only observes the installed driver through Bash and `nvidia-smi`. It does not run Python or execute numerical kernels. Inspect the dry run first, then submit explicitly:

```bash
bash scripts/submit.sh driver-audit configs/smoke.yaml \
  --run-dir /home1/aadaniel/projects/TDN/runs/driver-audit
bash scripts/submit.sh driver-audit configs/smoke.yaml \
  --run-dir /home1/aadaniel/projects/TDN/runs/driver-audit --submit
```

After the job completes, read `runs/driver-audit/driver_audit.csv` and its `driver_audit_metadata.txt`. Choose an exact tested PyTorch version and its official CUDA wheel index from the current [PyTorch installation selector](https://pytorch.org/get-started/locally/), compatible with that installed driver. Store these choices in `scripts/user.env.sh`, then source it again. A toolkit module version is not the wheel runtime or the driver's compatibility level. Do not install or modify drivers. `TORCH_WHEEL_INDEX` must be an official HTTPS index at `https://download.pytorch.org/whl/cuNNN`; CPU-only local checks may use `https://download.pytorch.org/whl/cpu`.

Submit installation as a CPU task. These commands fail until `TORCH_VERSION` and `TORCH_WHEEL_INDEX` are set:

```bash
bash scripts/submit.sh setup configs/smoke.yaml \
  --run-dir /home1/aadaniel/projects/TDN/runs/setup
bash scripts/submit.sh setup configs/smoke.yaml \
  --run-dir /home1/aadaniel/projects/TDN/runs/setup --submit
```

`setup_venv.sh` creates `.venv` using `python3 -m venv`, installs an exact PyTorch release, pinned `requirements.txt`, and the local package without dependency or build isolation. It preserves TLS and package verification, checks dependency consistency, and writes `requirements/environment-freeze.txt` and `requirements/environment-versions.json`. Reruns reuse a compatible real project venv; an incompatible existing interpreter is reported and preserved. No substantial package installation happens on login. If compute nodes cannot download packages, use CARC's documented download route to a wheelhouse inside this project; do not bypass cluster policy.

## 3. Sequential readiness and bounded pilot

The default pipeline is a dry run of the readiness stages:

```bash
bash scripts/pipeline.sh configs/pilot.yaml --pipeline-id readiness-review
```

Its serial DAG is:

```text
CPU mathematical/package tests
  -> CPU numerical/information/representation audit
  -> CPU paired dataset generation
  -> short allocated A100 GPU parity and signal/checkpoint tests
  -> allocated A100 memory/time calibration
```

Add `--submit` explicitly to submit that readiness DAG. The supplied pilot manifest requires scientific headroom; a failed or unresolved audit prevents later stages. Every successor uses `afterok` with an actual parsable job ID returned by `sbatch`. An unfinished run exits 75 and cannot satisfy `afterok`. Each pipeline gets deterministic `runs/PIPELINE_ID/STAGE` directories and a shared `runs/PIPELINE_ID/dataset`; use a fresh identifier to protect existing work. The audit and calibration paths are exported to their dependent stages, and their configuration hashes must match.

A pilot includes training, evaluation and a matched-tolerance benchmark only when an explicit optimizer-step budget is supplied. The wrapper allows at most 1,000 steps per development pilot and rejects a config whose `training.max_steps` exceeds that budget:

```bash
# Review the entire proposed request before adding --submit.
bash scripts/pipeline.sh configs/pilot.yaml \
  --pipeline-id pilot-review --pilot-budget 1000
```

To run that reviewed pilot, invoke it with a new identifier and `--submit`. No large arrays, multi-node jobs, full 3-D training, hyperparameter campaign or confirmatory study is authorized by this wrapper. The measured gates must justify expansion. The default scientific tolerance in the development manifests is a proposed screening value; it is not a confirmed application tolerance.

CARC evaluation and benchmark stages use the one validation-selected `train/checkpoints/best.pt`. If no feasible checkpoint was produced, they fail instead of quietly replacing it. All failed runs remain in the reports.

## 4. Per-stage allocations and explicit dependencies

| Stage | Resource class | CPUs | Host RAM | Walltime |
|---|---|---:|---:|---:|
| `driver-audit` | one A100 40 GB | 2 | 4 GiB | 5 min |
| `setup` | CPU only | 4 | 16 GiB | 45 min |
| `cpu-tests`, `audit` | CPU only | 4 | 16 GiB | 30 min |
| `generate` | CPU only | 8 | 32 GiB | 1 h |
| `gpu-tests` | one A100 40 GB | 4 | 16 GiB | 15 min |
| `calibrate` | one A100 40 GB | 8 | 64 GiB | 30 min |
| `train` | one A100 40 GB | 8 | 64 GiB | 4 h |
| `evaluate`, `benchmark` | one A100 40 GB | 8 | 64 GiB | 1 h |

These are proposed starting allocations. Live partition/account/QOS reports are saved as `slurm_policy.json`; Slurm determines final admission. GPU requests always include `--partition=gpu --gpus-per-task=a100:1 --constraint=a100-40gb`. Host `--mem` is not VRAM. Log directories exist before `sbatch`. Account and absolute output paths are passed as shell-array arguments, not variable expressions in `#SBATCH` directives.

The manual interface exposes the same safe dependency options. Substitute real numeric IDs returned by the preceding submission:

```bash
bash scripts/submit.sh generate configs/pilot.yaml \
  --dependency afterok:ACTUAL_AUDIT_JOB_ID \
  --gate-report /home1/aadaniel/projects/TDN/runs/PILOT/audit/audit.json \
  --dataset /home1/aadaniel/projects/TDN/runs/PILOT/dataset \
  --run-dir /home1/aadaniel/projects/TDN/runs/PILOT/generate
```

The literal placeholder is intentionally rejected until replaced by a numeric job ID. For manual training add `--pilot-budget 1000`, the matching `--gate-report`, and `--calibration-report /home1/aadaniel/projects/TDN/runs/PILOT/calibrate/calibration.json`. Configs, reports, datasets and output/checkpoint paths must resolve inside the project, including when symlinks are present.

## 5. GPU visibility, memory and stop/resume

GPU preflight and the actual Python command run inside the **same `srun` task**. Preflight rejects absent allocation/account evidence, multiple visible GPUs, wrong hardware, MIG devices, an incorrect memory class or exhausted memory. The scripts preserve Slurm's `CUDA_VISIBLE_DEVICES`; Python uses `cuda:0`. Preflight records the actual total/free/device-used memory and the project soft budget `min(30 GiB, 0.80 * actual_total)`.

Compilation, Triton, Torch extensions, CUDA, Matplotlib, pip, Python bytecode and temporary files are redirected into `.cache` inside the project. Pytest receives a project-local `--basetemp`. Writable path symlink escapes are rejected. Check quota before preparing data or compiling; persistent storage is not an off-cluster backup.

`--signal=USR1@180` signals the job step. The main Python handler sets a stop flag; at a completed optimizer boundary the loop atomically saves a complete checkpoint and writes `PAUSED_NEEDS_RESUME`, then exits 75. Workers ignore USR1. No automatic requeue/resubmission is enabled. Resume deliberately with the same config, dataset and matching reports:

```bash
bash scripts/submit.sh train configs/pilot.yaml --pilot-budget 1000 \
  --dataset /home1/aadaniel/projects/TDN/runs/PILOT/dataset \
  --gate-report /home1/aadaniel/projects/TDN/runs/PILOT/audit/audit.json \
  --calibration-report /home1/aadaniel/projects/TDN/runs/PILOT/calibrate/calibration.json \
  --resume /home1/aadaniel/projects/TDN/runs/PILOT/train/checkpoints/last.pt \
  --run-dir /home1/aadaniel/projects/TDN/runs/PILOT/train
```

Inspect the dry run, then add `--submit`. Rebuild dependencies around the resumed job's new actual ID; paused predecessors never become successful automatically. Monitor with `squeue -u aadaniel` and `sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode`. `MaxRSS` measures host memory; use the application report for VRAM.

## 6. Cloud CPU smoke without Slurm claims

An explicit local CPU override permits validation in an existing cloud checkout. It cannot submit jobs or bypass CARC identity/path rules. From that checkout, with its venv already installed:

```bash
export TDN_LOCAL_TEST_ROOT="$(pwd -P)"
bash scripts/smoke.sh
```

The script runs CPU tests and sequential audit, generation, tiny training, evaluation and benchmark stages. It sets no artificial Slurm allocation or GPU visibility. A last checkpoint may be used solely for implementation smoke when no feasible best checkpoint exists; the resulting report is not a scientific success or CARC/GPU validation. CARC calls to this script require a real allocated CPU task.

The mocked scheduler suite checks argument quoting, exact account matching, live CPU discovery, resource requests, valid job IDs, dependency chaining, storage confinement and default dry-run behavior without calling a real scheduler. A GPU suite is never counted as successful when it only skipped tests.
