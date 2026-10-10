# Order-Anchored Temporal-Defect Networks

A runnable research pipeline for learning the finite-time error of a symmetric split solver. It includes exact noncommuting linear controls, periodic logistic reaction–diffusion, refined FP64 coupled teachers, six local-input learned models, deterministic datasets/training, checkpoint resume, and complete-solver accuracy/cost reports.

This branch also contains the **separate [adjacent interaction study](docs/ADJACENT_RESEARCH.md)**:
D01–D08, controlled rough fields, LL/LH/HH correction channels, two bounded
alternatives, fitted/neural controls and a separate analytical atlas. Use the
[isolated-checkout runbook](docs/ADJACENT_RUNBOOK.md). Its source and budgets
must remain separate from live portfolio jobs; it makes no automatic change
to TDN's main architecture or research objective.

The current [three-path research portfolio](docs/PORTFOLIO_RESEARCH.md) resolves
normalization and conditioning attribution, tests three grounded improvements,
and protects 20% of its discretionary budget for three distinct exploratory
prototypes. It includes 23 method families, matched endpoint schedules,
independent discrete/continuum references, recoverable jobs and a source-linked
analytical atlas. Start with the [portfolio runbook](docs/PORTFOLIO_RUNBOOK.md):
`bash scripts/fedora_portfolio.sh plan --smoke`, then
`bash scripts/fedora_portfolio.sh run --smoke`. Full confirmation is a separately
declared 40-job campaign; every allocation is at most 45 minutes.
The [initial portfolio review](docs/PORTFOLIO_INITIAL_REVIEW.md) reports the
completed CPU diagnostics and integration run, including negative prototype
outcomes. The [52-page CPU atlas](results/portfolio-development/smoke/portfolio-atlas.pdf)
preserves missing evidence; native GPU confirmation remains outstanding.

The [tabbed research roadmap](docs/RESEARCH_ROADMAP_20261007.html) reviews ten
recent experiment groups, 62 supported work items, 24 proposed mechanisms and
14 related primary-source work records. It includes mathematical derivations,
falsifiable projections and a bounded next program for the Fedora desktop.
[Markdown](docs/RESEARCH_ROADMAP_20261007.md) and
[structured data](results/RESEARCH_ROADMAP_20261007.json) are also available.

Run the implemented M00–M23 mechanisms and C0–C4 combinations with the
[bounded Fedora roadmap experiment program](docs/ROADMAP_EXPERIMENTS.md).
It schedules ten stages and records per-experiment costs, parameters,
mathematical/gap checks, GOOD/BAD/NA verdicts and evidence-attainment scores.

CARC deployment is fixed to **`/home1/aadaniel/projects/TDN`**, user **`aadaniel`**, account **`anakano_81`**, and a Python **`.venv`**. All temporary files, caches, data, logs and checkpoints stay under that directory. The user's storage and venv requirements replace the uploaded document's environment examples.

## Run on Fedora with local Slurm

Use the [Fedora Slurm runbook](docs/FEDORA_SLURM.md) for the current premix
experiment on `/home/rahel/TDN`, standalone Python 3.13, a Ryzen 7800X3D,
128 GB RAM and an RTX 4090 with 24 GB dedicated VRAM. Start with
`bash scripts/fedora_slurm.sh doctor`, then configure and set up the local
profile. The launcher submits CPU accuracy, scaling and preparation sequentially,
followed by one GPU job, each limited to 30 minutes. It uses a project `.venv`,
an 18 GiB GPU soft cap, local Slurm partitions and project-owned Tower reports.
The profile does not inherit CARC billing or request an A100. Historical CARC
launchers retain their own environment requirements.

## Run on a Windows desktop

The native Windows edition supports a standalone Python `.venv`, PowerShell launchers, explicit CPU or NVIDIA CUDA execution, and a bounded sequential smoke adapted for an RTX 4090 with 24 GB dedicated VRAM. Extract the source ZIP to a writable local directory and begin with [WINDOWS_START_HERE.md](WINDOWS_START_HERE.md). The [Windows runbook](docs/WINDOWS.md) includes driver discovery, installation, the 12-step run, reports, resume and a prompt for a new local chat. Desktop execution keeps all generated files inside the extracted repository; it requires no Slurm allocation or lab account.

## Run on CARC

For the current full hypothesis test, use the [premix program](docs/PREMIX.md):
`bash scripts/carc_premix.sh plan`, then `bash scripts/carc_premix.sh run`.
It tests nonlinear spatial interactions before compression, larger-grid/batch
scaling, and five trained neural models over predeclared favorable, typical and
adverse conditions. Three CPU stages precede one A100 training/evaluation stage;
every allocation is capped at 30 minutes. Tower paths, logs and review collection
are automated. This is a bounded research comparison, with no established FNO
advantage or reproduction of the FNO paper.

The earlier [time-to-accuracy comparison](docs/WORK_PRECISION.md) remains available:
`bash scripts/carc_work_precision.sh --submit`. One bounded CPU job tests the
cheap spectral mean correction against full GL3/GL5, smaller-step Strang and
ETDRK2/4, with repeated timings and separate RMS/maximum-error frontiers.
It retains the earlier 2D failures and adds fresh amplitude stress tests.

The preceding [runtime interaction screen](docs/INTERACTION_SCREEN.md) remains available:
`bash scripts/carc_interactions.sh --submit`. One bounded CPU job tests
field-derived nonlinear corrections, bounded/mean ablations, Strang and
ETDRK2/4 with accepted FP64 references and Tower metrics. It has no training or
GPU successor.

The preceding [training-free mechanism audits](docs/MECHANISM_AUDITS.md) remain available:
`bash scripts/carc_mechanisms.sh --submit`. One bounded CPU job checks all eight
mechanism areas, coordinate/gate combinations and temporal representations,
with a dedicated Tower table. It does not train models or request an A100.

TDN now emits [Tower 2.3.1-compatible reports](docs/TOWER.md) for its workflows:
live metrics, grouped logs, scientific tables, artifact inventories and explicitly
selected planning histories. The integration is project-owned and uses your
existing Tower installation. Each job/attempt keeps a separate report; original
scientific results and checkpoint manifests remain intact.

For comparisons against representative neural solvers, use the
[neural benchmark workflow](docs/NEURAL_BENCHMARKS.md):
`bash scripts/carc_neural_benchmarks.sh --submit`. It adds a time-conditioned
MLP and direct/hybrid residual CNN, U-Net and FNO controls, with matched data and
optimizer updates and separate neural accuracy/cost reports.

The [three-seed neural replication](docs/NEURAL_REPLICATION.md) remains available:
`bash scripts/carc_neural_replication.sh --submit`. It repeats reaction-clock,
the generic MLP and three spatial hybrid controls on 27 fresh diagnostic parents.
All fifteen checkpoints freeze before diagnostics. Separate final-rollout and
transient endpoints retain missing, invalid and initialization-selected results.
It uses one 30-minute CPU allocation; optional frozen A100 timing is a separate
30-minute request.

For the new architecture hypotheses, use the [research workflow](docs/RESEARCH.md):

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main
bash scripts/carc_research.sh --submit
bash scripts/carc_research.sh status latest
bash scripts/carc_research.sh logs latest --lines 200
bash scripts/carc_research.sh collect latest
```

This runs focused correctness tests, accepted FP64 references, bounded neural
training and complete-solve comparisons in one CPU allocation: 4 CPUs, 16 GiB,
30 minutes, charged to `anakano_81`. Eight learned families include the three
proposals and matched controls. Training, validation and diagnostic parents are
disjoint. A separate, headroom-gated A100 benchmark loads the frozen checkpoints;
it has a 30-minute limit and submits no additional training. See the
[architecture definitions and known limitations](docs/RESEARCH_ARCHITECTURES.md).
These development comparisons do not replace the pilot's scientific gates.

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

The current small development regime has no demonstrated TDN efficiency advantage over cheap classical controls. A working training pipeline does not establish scientific merit. The original RD pipeline's smoke tolerance remains a proposed screening value; its historical confirmation gates require reviewed tolerances, physical ranges, untouched parents and measured GPU evidence. The separately declared research agenda below implements the user's authorized bounded confirmation program.

See [SOURCE_LEDGER.md](SOURCE_LEDGER.md), [REPO_AUDIT.md](REPO_AUDIT.md), [implementation map](docs/IMPLEMENTATION_MAP.md), [WORK_LOG.md](WORK_LOG.md) and [NEXT_ACTIONS.md](NEXT_ACTIONS.md). The original [implementation specification](docs/IMPLEMENTATION_SPEC.md) is retained for provenance. Gray–Scott, advection, distributed HALO, KAN, custom kernels and a confirmatory campaign are later extensions, not validated features of this initial RD pipeline.

## Next bounded consistency experiment

The physical gate and mean/spatial ablations are described in
[docs/CONSISTENCY.md](docs/CONSISTENCY.md). On the configured Fedora desktop,
start with `bash scripts/fedora_consistency.sh run --smoke`, then use `run`
for the full three-seed, 300-update experiment after the smoke completes.

## Full bounded research agenda

The uploaded research audit is implemented as an eight-stage program with
source-before-transport, nonlinear-time, mean–variance and mode-pair controls,
matched compression, fresh paired-grid confirmation and an audited fallback
policy. See [docs/RESEARCH_AGENDA.md](docs/RESEARCH_AGENDA.md) for every question,
Fedora budgets and the runbook. Start with
`bash scripts/fedora_agenda.sh run --smoke`, then use the separate
`--development` and full profiles. All stage allocations fit the desktop's
110000 MiB Slurm memory ceiling and end within 20–45 minutes each.

[results/agenda-validation.json](results/agenda-validation.json) records the
completed eight-stage CPU smoke, 222-case structural audit and independent Tower
checks. The [native full agenda review](docs/AGENDA_REVIEW.md) records all eight
completed Fedora stages and their verified scientific artifacts. No trained
complete joint solver gate passed; the roadmap above sets out the next research
questions without changing the archived gates.

The isolated adjacent study now includes [reviewed CPU evidence](docs/ADJACENT_INITIAL_REVIEW.md)
and a [67-panel atlas](results/adjacent-development/atlas/index.html). Its
[separate Fedora runbook](docs/ADJACENT_RUNBOOK.md) preserves the main portfolio.
These development results favor testing higher-order response and output
retention together; they do not establish a learned or GPU solver advantage.
