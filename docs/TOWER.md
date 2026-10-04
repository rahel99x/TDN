# TDN reports in Slurm Tower

TDN writes the formats documented by Slurm Tower 2.3.1. The Tower application,
its installation and its user settings remain unchanged. Project-owned
`.tower/config.json` binds the metrics, exact log index, output contract and
explicit planning aggregate. Its CARC profile identifies `aadaniel` and
`anakano_81`; reporting does not submit jobs or alter allocation limits.

The [integration validation record](../results/tower_integration_validation.json)
records 592 passing CPU tests and a complete default research run. Unmodified
Tower 2.3.1 accepted all 14 contract checks, 860 metric records and 17 grouped log
entries; the exact six schemas validated 865 objects. All 23 scientific artifact
hashes still matched. These are cloud CPU and native-reader checks; live CARC
display and native Windows execution remain separate environment checks.

## Start with a report

Continue submitting the existing TDN workflows. New executions automatically
produce Tower sidecars alongside the original scientific artifacts. Each
execution or restart gets a fresh report identity. Standalone stages use
`<source-name>-tower/<report-id>/`; grouped jobs use their workflow's
`tower/<report-id>/`. Child stages contribute to their job's report.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main

# Existing submission commands still apply, for example:
bash scripts/carc_research.sh --submit
bash scripts/carc_research.sh status latest
bash scripts/carc_research.sh logs latest --lines 200

# List concrete report directories, then print a command for the newest report.
bash scripts/tower.sh list
bash scripts/tower.sh show latest

# Explicitly launch your installed Tower with the selected report attached.
bash scripts/tower.sh open latest
```

`latest` selects the newest report creation found by bounded discovery and
prints its path, source, workload, job ID and state. When several jobs run in
parallel, select a concrete directory from `list` instead. You can limit the
listing to one workflow with `bash scripts/tower.sh list runs/RUN_ID`.
Discovery visits known report locations, not datasets or caches.

`show REPORT_DIR` only prints the command. `open REPORT_DIR` executes it using
the existing `tower` executable on `PATH`, from the project root:

```bash
tower --profile carc --config /home1/aadaniel/projects/TDN/.tower/config.json \
  --workdir /home1/aadaniel/projects/TDN/runs/RUN_ID/tower/REPORT_ID \
  --tab research --research-view experiment
```

If your `tower` command is an interactive shell alias or function, paste the
command printed by `show` into that shell. The Python launcher cannot resolve
shell-only aliases; no Tower installation or alias changes are needed.

The **absolute report directory** is essential: Tower does not discover the
active report from TDN's `run.json`, its workflow manifest or the selected job's
name. `--view artifacts`, `--view predict` and `--view tradeoffs` are also
accepted by `show` and `open`. Their relative contract and planning filenames
resolve from the project root; metrics and the log index resolve from the
selected report directory.

## Existing completed or failed runs

Historical runs need no rerun. Import an explicit science stage or experiment
directory containing terminal `stage.json` evidence:

```bash
# Use the actual historical path from your checkout.
bash scripts/tower.sh import runs/carc-light-20261003T224656758825Z/light-screen

# For a research workflow, select its experiment subdirectory.
bash scripts/tower.sh import runs/RESEARCH_RUN_ID/experiment
bash scripts/tower.sh show latest
```

The import prints a new sidecar directory and leaves every source file intact.
It records the original outcome, including failures and paused runs, and copies
only a recorded elapsed duration into the historical summary. Unrecorded
execution timestamps, allocations, memory, script fingerprints and job IDs stay
absent. Recorded source/configuration hashes remain labeled historical evidence;
the import does not revalidate old source code or scientific results. Its lone
metrics event marks the report export, with no synthetic training values or
historical ETA. Repeated imports are derivative snapshots of the same
observation and cannot become independent planning samples.

## Logs, live measurements and artifacts

In Tower's **Jobs**, **Recents** or **History**, select the desired job and press
`l`. Press uppercase `O` in Logs for the grouped file catalog, then select a file
with the arrows and Enter. Esc returns to the list. Lowercase `o` cycles files;
`e` switches scheduler stdout/stderr. TDN registers exact paths for scheduler
output, application output and readable analytics. A recorded `job_id` must
match the selected job. Old scheduler accounting may lack WorkDir or log paths;
the explicit report workdir and retained `logs.json` preserve project evidence.
Missing logs remain visibly missing rather than attaching another job's files.

The report contains:

| File | Contents and Tower use |
| --- | --- |
| `run.json` | Execution identity, lifecycle, known request facts and declared report paths |
| `logs.json` | Exact grouped log locations; references may point to sibling source files |
| `metrics.jsonl` | Finite timestamped measurements and phase progress for live charts and conditional ETA |
| `summary.json` | Final application outcome, measured runtime, compact scientific results and provenance |
| `outputs/results.json` | Scientific conclusions, exclusions and failures kept distinct from pipeline completion |
| `outputs/artifacts.json` | File inventory, source paths, bounded hashes, omissions and model/checkpoint locations |
| `outputs/tables.json` | Canonical table descriptions and projection limitations |
| `outputs/*.csv` | Canonical accuracy/cost, training, held-out, reference, gate, stage and test rows when available |

Live metrics are observations from the executing coordinator. Phase progress
resets for distinct work; Tower derives an ETA only when sufficient consistent
progress exists. Reporting does not rewrite losses, tolerances, validation
selection, failed trajectories or checkpoints. `COMPLETED` describes execution,
not an efficiency advantage or permission for a larger scientific campaign.

Analytics are bounded derivative views. The exporter visits at most 4,096
directory entries and 1,024 files, reads at most 16 MiB in total and 1 MiB per
source file, and limits projected rows to 10,000. Oversized, unreadable, malformed
or unrecognized artifacts stay in the source and appear as omissions or inventory
entries. Model checkpoints and arrays are inventoried; the reporter never
unpickles a checkpoint or loads numerical arrays. The output contract names exact
files under the sidecar; an inventoried source path is not an artifact-contract
permission to traverse outside that root.

## Validate, reconcile and export evidence

```bash
# Project structural checks; safe without Tower or numerical dependencies.
bash scripts/tower.sh validate latest

# Also invoke the existing Tower application's native artifact validator.
bash scripts/tower.sh validate REPORT_DIR --native

# After a job is terminal, reconcile its report with its exact sacct allocation.
bash scripts/tower.sh reconcile REPORT_DIR

# Explicitly choose independent reports for resource analysis.
bash scripts/tower.sh export REPORT_DIR_1 REPORT_DIR_2 \
  --reference REPORT_DIR_2 --output reports/planning.json

# Inspect Tower's evidence assessment for that aggregate.
tower run predict --file reports/planning.json
```

The default validator states that it performs TDN structural checks; it is not
presented as a full JSON Schema or native Tower compatibility check. `--native`
uses your installed Tower with the project's exact contract and workdir. Final
contracts may report a missing summary while a job is still running.

Reconciliation queries `sacct` for the report's recorded exact job or array-task
ID, ignores batch/extern step rows, and requires an unambiguous terminal state.
It retains application evidence and records scheduler evidence separately. A
termination signal alone never establishes timeout or out-of-memory. Whole-job
elapsed duration never replaces a narrower stage's measured runtime. Missing
accounting is reported as missing evidence, with the report preserved.

Planning export is explicit and bounded to 256 terminal reports and 1 MiB.
Failures remain included. Duplicate identities, repeated imports and duplicate
observations of the same job/workload are rejected. Choose comparable independent
jobs: source/configuration/data identities and real allocation counts matter.
Unknown resources stay absent, not zero; host CPU count and application worker
count do not establish a Slurm allocation. No host RSS or GPU reservation is
relabeled as allocation-wide peak memory. Tower may abstain from predictions
when evidence is insufficient; TDN does not fill missing facts to force a fit.
The integration does not invent controlled scaling observations, queue forecasts,
dependency recipes or Tower passports.

The tools use Python's standard library and bounded file inspection on the
login node. Training and numerical tests remain inside the existing allocations
and `.venv`. Generated files and temporary writes remain under the project;
no system temporary directory or scratch filesystem is used.

Continue using the workflow's normal `collect` command to produce a review
archive, for example `bash scripts/carc_research.sh collect RUN_ID`. Keep source
artifacts with their sidecars when moving a bundle: relative source log links
retain their meaning only with the same directory layout. Absolute scheduler
paths retain their original machine meaning and may need an explicit rebind.
