# Five-gate frontier program on Fedora

This program asks whether a compact physical correction can improve accuracy–cost
or data efficiency against competitive neural operators, and which regimes also
permit practical classical-solver savings. Those claims remain separate. All
results, including initialization selections, infeasibility and failures, remain
in the reports. A completed scheduler job is not evidence of superiority.
The equations, executable question matrix, cohort counts and claim boundaries are
documented in [FRONTIER_RESEARCH.md](FRONTIER_RESEARCH.md).

Use the existing `/home/rahel/TDN` checkout and its own Python `.venv`. The launchers
use the project interpreter without activation. The saved Fedora profile must
match user `rahel`, the checkout and the actual `local` partition. Do not use the
CARC account, fabricate Slurm variables or copy a venv from another machine.

## Start with the allocated smoke

```bash
cd /home/rahel/TDN
git pull --ff-only
bash scripts/fedora_frontier.sh doctor
bash scripts/fedora_frontier.sh plan --smoke
bash scripts/fedora_frontier.sh run --smoke
```

`doctor` checks the existing desktop configuration. On a fresh checkout, use the
existing `scripts/fedora_setup.sh configure` and `setup` procedures in
[FEDORA_SLURM.md](FEDORA_SLURM.md) first. Do not reconfigure an already verified
profile or reinstall the venv merely to launch this experiment.

Planning is read-only and makes no scheduler calls. `run` freezes the protocol,
source fingerprint, software and resource profile, then queues every stage. The
smoke exercises all five gates with a small, disjoint cohort; it is not fresh
scientific confirmation and cannot establish a competitive advantage.

```bash
bash scripts/fedora_frontier.sh status latest
bash scripts/fedora_frontier.sh logs latest --lines 200
bash scripts/fedora_frontier.sh paths latest
```

After the smoke has finished:

```bash
bash scripts/fedora_frontier.sh validate latest && \
  bash scripts/fedora_frontier.sh run --full
```

`validate` verifies science and execution seals. It does not require a favorable
scientific verdict. Failed or incomplete stages make validation fail. Preserve
that run, collect its evidence, and fix the cause before making a fresh run.

## Stage chain and budgets

| Stage | Device | CPUs / RAM | Allocation limit | Purpose |
|---|---|---|---|---|
| `audit` | CPU | 4 / 32 GiB | 20 min | Gate 1: structural correctness and measurement semantics |
| `screen` | CPU | 8 / 48 GiB | 45 min | Gate 2: spatial target, teacher refinement and classical headroom |
| `prepare` | CPU | 8 / 48 GiB | 45 min | Parent-disjoint training, validation and calibration teachers |
| `train` | RTX 4090 | 4 / 32 GiB | 45 min | Gate 3: bounded tuning/training and frozen model selection |
| `confirm_prepare` | CPU | 8 / 48 GiB | 45 min | Fresh teachers created after model/checkpoint freeze |
| `confirm-part-000` … `005` (full) | RTX 4090 | 4 / 48 GiB each | 45 min each | Gate 3: four independent parents per bounded part, all paired model arms |
| `confirm` | RTX 4090 | 4 / 48 GiB | 45 min | Verify every part and recompute whole-cohort confirmation comparisons |
| `scaling` | RTX 4090 | 4 / 48 GiB | 30 min | Gate 4: larger-grid accuracy, throughput and memory |
| `policy` | RTX 4090 | 4 / 32 GiB | 30 min | Gate 5: cost-gated end-to-end policy evaluation |
| `report` | CPU | 4 / 32 GiB | 20 min | Complete analytical graphs, costs, outcomes and retained failures |

Each scientific stage depends on successful execution of all its predecessors.
The report uses `afterany`, so it runs after success, failure or cancellation and
makes absent evidence explicit. A scientific negative is an outcome, not an
execution failure. A failed structural audit blocks subsequent science. The
policy stage can complete with a scientific blocked/NA outcome when its measured
cost-margin prerequisites do not justify deployment work.

CPU stages request four to eight CPUs and 32/48 GiB; GPU stages request four CPUs,
32/48 GiB and exactly one GPU. The frozen protocol contains each stage's explicit
walltime, at most 45 minutes. These allocations fit the scheduler's **110000 MiB**
host-memory cap. Dedicated VRAM limits remain **18 GiB and 75%** soft limits and a
90% device-use hard threshold. Host/shared memory does not enlarge GPU VRAM.
The chain runs sequentially; there is no pending-job cap or automatic expansion.
There are nine logical scientific stages and 15 physical jobs for full execution
(11 for smoke; 13 for development). Confirmation's scientific compute timer stays
at 1,800 seconds per full part. Splitting the work does not guarantee that every
hardware/workload combination finishes within that bound. A timeout stays visible;
the launcher never increases resources or resubmits indefinitely.

Full confirmation retains all 24 parents, 80 frozen model records, 21,888 endpoints
and 1,152 randomized paired timing groups. One complete parent, with all grids,
tracks and schedules, belongs to exactly one part. No paired timing group is split.
The aggregate copies original measurements and experiment rows unchanged, then
recomputes comparisons and gates across the entire cohort. It performs no model
inference; its allocation and aggregation overhead are recorded separately. Each
part starts a new process, so first-invocation timings describe that process rather
than the cache state of the old monolithic run.

Every GPU stage executes all 45 declared native CUDA cases and rejects
filtered, skipped, duplicated, failed or missing cases. CPU tests and mocked
scheduler tests cannot establish GPU readiness.

## Recover a confirmation timeout without retraining

Exit code 75 with `Bounded research walltime exceeded` means the internal compute
timer stopped the stage. A completed `report` job only means it recorded the
partial outcome. It does not make confirmation, scaling or policy complete.

Once the original workflow's jobs have all stopped:

```bash
cd /home/rahel/TDN
git pull --ff-only
bash scripts/fedora_frontier.sh recover latest --plan
bash scripts/fedora_frontier.sh recover latest
```

Use the original run ID instead of `latest` if a different workflow was submitted
since the failure. `--plan` verifies compatibility and prints the finite job plan
without submitting. Recovery creates a fresh coordinator and leaves the original
run intact. Its immutable `recovery.json` points to the verified audit, screen,
prepared banks, training/checkpoints and confirmation teachers. It queues six full
confirmation parts, their aggregate, scaling, policy and report: ten new jobs,
with no retraining or teacher regeneration.

The compatibility bridge accepts the original `aa5760c` release or the exact
current implementation. It verifies scientific source pins, protocol, frozen
checkpoints, teacher banks, execution lineage, project venv and Slurm profile.
It is not a general source-check bypass. Changed model code, data, dependencies
or site configuration require a fresh experiment. Unsealed measurements from the
old failed job remain forensic evidence and are excluded from new scientific
comparisons. A later interrupted recovery can reuse its complete sealed parts
under the same source/environment and queue only its missing parts and successors.

```bash
bash scripts/fedora_frontier.sh status latest
bash scripts/fedora_frontier.sh logs latest --lines 200
bash scripts/fedora_frontier.sh paths latest
# After all jobs finish:
bash scripts/fedora_frontier.sh validate latest
bash scripts/fedora_frontier.sh collect latest
```

Reports distinguish inherited costs, new allocations, partition measurements and
merge overhead. Incomplete parts remain descriptive and cannot pass a cohort gate.
Collection includes the origin artifacts and original failure evidence as well as
the new workflow. Do not delete the origin directories before collecting/reviewing.

## Review the graphs and exact evidence

```bash
bash scripts/fedora_frontier.sh paths latest
bash scripts/fedora_frontier.sh logs latest --lines 200
bash scripts/fedora_frontier.sh validate latest
bash scripts/fedora_frontier.sh collect latest
```

The printed run directory contains:

- `report/figures/frontier-atlas.pdf`: 33-page analytical atlas, including partition coverage and partial diagnostics.
- `report/figures/frontier-overview.png`: high-resolution overview.
- `report/figures/index.html`: local graph browser.
- `report/figures/chart-data.json.gz` and `manifest.json`: plotted data and figure inventory.
- `report/analysis.json`: machine-readable analytical summaries.
- Stage `rows.jsonl`, `summary.json`, manifests and execution seals: underlying observations and provenance.
- `reporter-tests/<stage>/tests.xml`: exact CPU/GPU test executions.
- `tower/<report>/metrics.jsonl`: Tower progress/analytics integration; `paths` prints the relevant directories.
- `state/scheduler-accounting.json`: allocation and task-step resource records
  captured before reporting and refreshed by `collect`.

The loss and compute views distinguish tuning from final training, initialization
selection from learning, cold startup from warmed execution, and individual
fields from paired grids/seeds/schedules. Matched-accuracy comparisons use the
same physical final time and spatial target. Large-grid accuracy without an
accepted reference remains NA. Public-paper superiority is not inferred from
these in-repository FNO controls.

The training stage's Tower report also contains bounded native metric pages for
every observed loss/validation/compute update, with exact trial lineage. Missing
validation measurements remain absent. The final report links the complete HD
figures, PDF and HTML browser; Tower itself is unchanged. Use `paths` to select
the training report for loss-series inspection and the reporting stage for the
analytical atlas.

G4 includes separate measured neural/classical throughput checks at the frozen
1.2× threshold. Passing its accuracy checks alone does not pass that utility
criterion. Even a favorable ratio is a descriptive result for the declared paired
workload, not a robust or universal speedup claim.

Scheduler elapsed time and experiment timers have different scopes. Allocation
and task-step records must not be summed together. Missing accounting is unknown,
not zero. No electricity, monetary or cluster-credit cost is assumed without a
supplied rate. `collect --no-accounting` disables the optional scheduler snapshot
but still transports the science artifacts.

The pre-report snapshot covers completed predecessor allocations. The reporting
job itself can still be running in that snapshot; its recorded elapsed time is
partial, and accounting can lag. Collection refreshes the scheduler record without
rewriting sealed scientific plots or observations.

Collection creates one archive plus a SHA-256 index. If it exceeds 28 MiB, it also
creates ordered `.part001`, `.part002`, ... files. Upload the index and **all**
parts, or the complete archive when supported. The archive includes failed and
partial stages, references, checkpoints, plots, metrics and logs. Pytest scratch
files are excluded; scientific evidence is not.

## Preserve earlier runs

The frontier workflow uses its own `frontier-workflow.json`, run ID and
`.fedora-frontier-latest.json` pointer. It does not reuse the roadmap, agenda,
consistency or premix latest pointers. To inspect a specific run after newer ones
exist, replace `latest` with its `fedora-frontier-...` run ID.

Do not pull code, edit the scientific protocol or replace dependencies while
queued/running stages depend on them. Source and software drift intentionally
invalidate successor stages. Completed and interrupted runs are never overwritten;
start a new run ID after a fix. Submission failures preserve submitted jobs and
attempt a separate report job without retrying scientific work or cancelling
other projects.

For development on a separate CPU checkout only:

```bash
bash scripts/frontier_local.sh --smoke
```

This local wrapper runs the same partition-and-merge sequence on CPU with project-contained caches and
a disjoint smoke/development cohort. It refuses `--full`, CUDA, CARC login use and
scheduler/desktop impersonation. Use the allocated Fedora launcher on the desktop
for native readiness and scientific confirmation.

To exercise the compatibility bridge on an existing local CPU smoke only, use
`bash scripts/frontier_local.sh --smoke --recover runs/ORIGIN_RUN`; it creates a
new local run. This cannot recover native CUDA evidence as CPU measurements.

## Repository validation and preview

The implementation passed a 3,470-test CPU regression run (one optional schema
test skipped), then 472 final frontier/Tower tests after the last scaling-score
change. The final-source nine-stage CPU smoke completed 752 experiment records
and verified all 253 scientific files and execution seals. Unmodified Tower
accepted all nine reports and 108 indexed pages; a separate strict schema audit
passed 903 objects. The complete 31-panel CPU preview is in
[results/frontier-cpu-smoke](../results/frontier-cpu-smoke/README.md); exact
validation and test provenance are in
[results/frontier-validation.json](../results/frontier-validation.json).

These are engineering checks. The tiny smoke retained BAD/NA scientific outcomes
and blocked deployment. It ran alongside regression tests, so its timing is not
an isolated performance measurement. Native Fedora Slurm, the 45 CUDA tests and
the full scientific cohort still require the allocated desktop run above.

The subsequent confirmation recovery fix passed 674 CPU regression tests. A real
CPU recovery reused the pinned release's five original stages, preserved all 496
smoke predictions, and completed partitioning, aggregation, scaling, policy and
33-panel reporting. A second recovery reused both sealed parts without repeating
their model measurements. Unmodified Tower accepted the new reports. See
[frontier-recovery-validation.json](../results/frontier-recovery-validation.json)
for exact source fingerprints, commands, JUnit and integration evidence. Native
Slurm/CUDA validation of the recovery change remains a desktop task.
