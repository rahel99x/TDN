# Automated CARC workflow

Run from `/home1/aadaniel/projects/TDN` as `aadaniel` in a Bash login shell.
Every allocation charges `anakano_81`; every generated file stays inside this
checkout. No Conda, `/tmp` or `/scratch1` is used for project storage.

## Start from the top

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main
bash scripts/carc_start.sh
bash scripts/carc_start.sh --submit
```

The first launcher call previews the allocations and stage order without
creating files or submitting jobs. The second creates a fresh workflow and
queues exactly two jobs:

| Job | Sequential work | Requested resources |
|---|---|---|
| CPU | Verify/create `.venv`, CPU tests, scientific audit, dataset generation | `main`, 4 CPUs, 16 GiB host RAM, 1 hour |
| GPU | GPU tests, calibration, 12-step training, evaluation, benchmark | `gpu`, one A100 40 GB, 4 CPUs, 16 GiB host RAM, 45 minutes |

The GPU job depends on successful completion of the actual CPU job ID. It uses
`--kill-on-invalid-dep=yes` so a failed predecessor cannot leave it indefinitely
waiting on an impossible dependency. Submitting two grouped jobs reduces eight
separate scheduler requests; queue duration still depends on CARC scheduling.
Resource limits are proposed margins for this tiny smoke, not measured queue or
throughput optima.

The defaults in `scripts/carc_defaults.sh` are grounded in the uploaded driver
audit and live probe: Python module `python/3.11.9`, exact
`torch==2.10.0+cu126`, the official CUDA 12.6 wheel index, CPU partition `main`
and account `anakano_81`. The observed A100 driver was `580.159.04`. The script
does not install or alter drivers. Live account and partition checks still run
before submission; genuine GPU preflight runs inside the GPU task.

An existing venv with the required interpreter and package versions is reused.
Otherwise installation takes place inside the allocated CPU job. No package
installation or numerical work runs on the login node. `--setup always` requests
an allocated reinstall; `--setup never` requires an already compatible venv.
An incompatible existing interpreter is preserved and reported.

Grouped jobs hold a project-local lock for the entire CPU or GPU phase so
another grouped workflow cannot install packages into the shared venv while
work is running. A conflicting phase stops with a clear message. Wait for the
active workflow to finish before submitting another one in this checkout.
Submission and recovery commands use a separate short-lived project lock so
simultaneous commands cannot lose recorded job IDs or submit duplicate resumes.

## Status and logs

```bash
bash scripts/carc_status.sh
bash scripts/carc.sh status latest
bash scripts/carc.sh logs latest
```

`latest` means the most recently submitted automated workflow in this checkout.
Pass the printed run identifier instead to inspect a specific run. Status joins
the local stage records with Slurm state, including cancellation or timeout when
a process could not publish its final record. Phase logs contain clearly marked
stage boundaries; individual scientific reports remain in their stage folders.

Each workflow keeps a config snapshot, source fingerprints, submitted job IDs,
stage status and software identity under `runs/`. Completed and failed work is
preserved. Editing execution code or configuration while jobs are queued causes
their source checks to fail rather than mixing evidence from different versions.

## Recover deliberately

After pulling a code fix, use a fresh workflow:

```bash
git pull --ff-only origin main
bash scripts/carc_restart.sh
bash scripts/carc_restart.sh --submit
```

Restart verifies scheduler ownership before canceling the old workflow's pending
jobs and submits a fresh two-job workflow. It preserves old files and does not
cancel running jobs. A running workflow must finish before you restart it.
To cancel pending jobs without restarting:

```bash
bash scripts/carc.sh cancel latest --submit
```

A training warning signal produces an atomically certified
`PAUSED_NEEDS_RESUME` checkpoint and exit 75. Resume is a separate explicit
action with the same source, config, dataset and software:

```bash
bash scripts/carc.sh resume latest
bash scripts/carc.sh resume latest --submit
```

Resume is allowed only for a certified training pause after all prerequisites
completed. It submits a new GPU allocation, preserves verified completed GPU
stages, resumes training from its last checkpoint, and then runs evaluation and
benchmark. It never silently resumes after a source or config change. Failed
tests, failed scientific gates and unverified checkpoints require investigation
and a fresh restart; there is no automatic retry loop.

## Configuration and scope

The default `configs/carc-smoke.yaml` is the best validation-error candidate
among four bounded CPU experiments, described in [CARC_CONFIG.md](CARC_CONFIG.md).
It retains the proposed tolerance `0.002`, disjoint parent splits, validation-only
checkpoint selection, FP64 teachers, eager FP32, TF32 off, zero workers and the
existing VRAM caps. It is a 12-step implementation diagnostic.

Actual A100 throughput and memory calibration are measured by the GPU job.
CPU selection is not an A100 optimum or a TDN efficiency claim. Historical G2/G4
failures still block the larger pilot; its config retains required headroom:

```bash
bash scripts/carc.sh start --profile pilot --pilot-budget 1000
```

That command previews a separately budgeted development pilot. Adding `--submit`
does not disable its gates. No confirmatory experiment, array, multi-node or
large-grid campaign is launched by these helpers. Individual-stage tools remain
available in [CARC.md](CARC.md).
