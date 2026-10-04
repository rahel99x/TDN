# Automated architecture hypothesis experiments

This workflow implements the three research proposals and their paired controls
in a separate development experiment. It trains real neural encoders on fresh,
parent-disjoint training states, selects checkpoints using validation states,
then measures one-step predictions and complete rollouts on diagnostic states.
Teacher-fitted per-parent coefficients are not used as trained-model evidence.

The default protocol in [`configs/research.yaml`](../configs/research.yaml)
uses 32×32 periodic states, FP32 learned/classical solvers, accepted FP64 coupled
references, three reaction–diffusion regimes, interior and boundary states,
eight model families, and at most 96 optimizer steps per family. A two-step
rollout loss supplements the one-step objective. The scientific tolerance stays
at `0.002`. The experiment is bounded to 1,200 seconds of numerical work and
keeps failures. These are development hypotheses, not a confirmatory campaign.

## Implementation validation

The [2026-10-04 cloud CPU record](../results/research_validation_20261004.json)
contains the exact source/configuration fingerprints, commands, software,
training outcomes and accuracy/cost comparisons. The full CPU regression suite
passed 496 tests; focused checks cover the subsequent report changes. The final
default experiment completed in 86 seconds, accepted all 135 references across
27 parents, and verified all 23 sealed artifacts. These timings describe this
cloud CPU environment, not CARC or CUDA execution.

Seven of eight families completed training. `transport` produced nonfinite
gradients and no eligible checkpoint; `temporal_mlp` selected its initialization
because its trained checkpoints failed validation admissibility. Five of nine
diagnostic cases had classical headroom, but none of the 49 eligible learned
comparisons achieved a CPU speed advantage. This short optimization budget does
not establish converged performance. The workflow preserves these negative
results and runs real CUDA correctness tests before any allocated A100 timing.

## Start on CARC

Run from the existing checkout and existing verified Python venv. The new
scripts do not install packages. If that venv needs repair, use the separate
allocated setup described in [CARC_AUTOMATION.md](CARC_AUTOMATION.md).

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only

# Preview the exact request; this makes no run directory and submits no job.
bash scripts/carc_research.sh

# One CPU allocation: focused tests, accepted references, training, evaluation.
bash scripts/carc_research.sh --submit

# These commands only inspect reports and scheduler metadata on the login node.
bash scripts/carc_research.sh status latest
bash scripts/carc_research.sh logs latest --lines 200

# After completion, package everything needed for review inside runs/.
bash scripts/carc_research.sh collect latest
```

Every submission uses a fresh `carc-research-*` run ID. Save the printed
`TDN_RESEARCH_RUN_ID` to distinguish simultaneous runs. Supply `--run-id NAME`
when a human-readable name is useful. Existing directories are rejected.
`latest` means the newest **research** workflow and does not change the existing
`carc.sh` workflow pointer. An explicit ID is safer when submitting several jobs.

The fixed request is **4 CPUs, 16 GiB host RAM and 30 minutes**, charged to
`anakano_81` for `aadaniel`. CPU work requests no GPU. The wrapper verifies the
live account association and partition limits before submission; workers verify
their running job identity and exact pinned venv. All logs, temporary files,
caches, data and checkpoints stay under `/home1/aadaniel/projects/TDN`.

No pending-job cap is used, and this workflow does not alter other projects'
submissions. Research workers share a read lock on the checkout's venv so they
can run concurrently. A simultaneous installer or older workflow holding the
same lock exclusively causes the research worker to stop safely; it does not
hold queued jobs. Avoid modifying code, configs or the venv during a run.

For a very small end-to-end installation check, use a fresh smoke workflow:

```bash
bash scripts/carc_research.sh --smoke --submit
```

Smoke reduces the grid and optimizer budget. Its timings and results are
implementation diagnostics, not evidence for the architecture hypotheses.

## What is compared

| Family | Role |
|---|---|
| `confluent_decay` | Repeated-rate temporal responses with equilibrium decay |
| `fixed_decay_r` | Ordinary fixed-rate modes with the same physical decay as confluent modes |
| `fixed_decay` | Ordinary fixed-rate modes with twice the physical reaction decay |
| `fixed_undamped` | Isolates the effect of the equilibrium envelope |
| `reaction_clock` | Bounded correction inside the final reaction map |
| `reaction_additive` | Paired additive-output control for the reaction basis |
| `transport` | Early correction transported by diffusion plus a late correction |
| `temporal_mlp` | Current learned-rate temporal model control |

The optional `reaction_hybrid` boundary extension can be appended to the full
family list in a project-local config and passed using `--config PATH`. It
remains experimental. Do not select only favorable models or change the fixed
tolerance, split definitions or horizon protocol to manufacture a pass.

Classical controls include Strang splitting, Richardson extrapolation, adaptive
splitting, coupled RK4 and the exact leading discrete commutator correction. Reports retain
the complete error-versus-cost comparison, rejected/infeasible trajectories,
reference uncertainty, first-use timing, and steady-state timing. The parent
seeds are disjoint across train, validation and diagnostics. A single bounded
development run is not a multi-seed statistical study or proof of novelty.

## Optional short A100 benchmark

GPU work is a separate explicit action. It reuses the frozen CPU-selected
checkpoints and performs no long GPU training. Replace `CPU_RUN_ID` with the
printed ID of the completed CPU research workflow:

```bash
# Preview first. This refuses a source with no eligible numerical headroom.
bash scripts/carc_research.sh benchmark CPU_RUN_ID

# Only after the completed CPU screen is eligible:
bash scripts/carc_research.sh benchmark CPU_RUN_ID --submit
bash scripts/carc_research.sh status latest
bash scripts/carc_research.sh logs latest --lines 200
bash scripts/carc_research.sh collect latest
```

The headroom criterion requires both Strang and Richardson error, after
subtracting reference uncertainty, to exceed `0.002` on a declared coarse-step
diagnostic case. This is a necessary screening condition, not a model victory.
A false result blocks submission before any A100 allocation is requested.
Do not disable this check; review the CPU reports to decide the next scientific
experiment. GPU superiority requires a whole-solve accuracy/cost comparison
against the best feasible classical control, including all model overhead.

The GPU request is **one full A100 40 GB, 4 CPUs, 16 GiB host RAM and 30 minutes**.
Allocated GPU preflight, actual CUDA parity/gradient tests and the benchmark
share the same `srun` task and Slurm GPU visibility. Skipped GPU tests are not
accepted as GPU validation. No automatic GPU successor, requeue, long campaign
or 20-hour request exists in this workflow.

Submission verifies the CPU phase completion, immutable raw config, source
fingerprint, protocol, dataset and all checkpoint/report hashes. The GPU worker
repeats those checks and requires the same software versions. Pulling new code
or changing the config after the CPU experiment requires a fresh CPU run;
old outputs remain readable and collectible.

## Reports, interruption and review

Each workflow has this layout under `runs/RUN_ID/`:

```text
research-workflow.json     immutable allocation/config/source description
config.yaml               immutable copy of the submitted config
jobs.json                 real parsable Slurm job ID
logs/                     stdout and stderr
state/                    allocation policy, verified software, current phase
experiment/
  protocol.json           declared states, horizons, losses and comparison rules
  dataset.pt              accepted reference pairs and their provenance
  checkpoints/            validation-selected learned models
  training/               training/validation records
  summary.json            results, headroom and complete comparisons
  summary.txt             readable results and numerical failures, also in logs
  manifest.json           hashes binding the artifacts
  COMPLETED               present only after successful final verification
```

On a numerical/test failure, signal or walltime limit, the workflow records the
stopping stage and preserves partial outputs. No successor starts after a
failure. This bounded research workflow does not implement exact training
resume: inspect and collect the failed run, then submit a fresh one. Never
delete a failure or reuse a completed directory to make a check pass.

`collect` writes a new dated `.tar.gz` beneath `runs/`, including data,
checkpoints, protocol, metrics, software/allocation metadata and logs. For a GPU
workflow it also includes the associated CPU workflow. It excludes pytest work
directories and refuses symlinks. Collection also works for failed or
interrupted runs; that archive is a snapshot, not a completion certificate.

## Local/cloud CPU development

In a non-CARC checkout with its standalone Python venv already installed:

```bash
bash scripts/research_local.sh --smoke
# Full bounded CPU development protocol:
bash scripts/research_local.sh
```

The script sets the explicit local CPU root, keeps storage in this checkout and
creates a fresh run directory. Optional `--config PATH` and `--run-dir PATH`
must remain within the project. It refuses CARC login-node use and does not
invent allocation variables or silently substitute CPU for a requested GPU.
These measurements must be labeled local/cloud CPU; mocked scheduler tests do
not establish live CARC or A100 behavior.
