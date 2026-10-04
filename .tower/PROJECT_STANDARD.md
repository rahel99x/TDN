# Tower project reporting standard, version 1

Follow this convention when building a project you want to inspect with Tower.
It works with the current terminal application and does not require your
application to import Tower. Keep your existing source and build tools; add the
reporting files alongside them.

The [copyable project template](../examples/project-template/README.md) includes
a dependency-free reporter, a small working application, a batch script, and
configuration. [JSON Schemas](schemas/README.md) describe the interchange files.
Tower's native readers perform the final compatibility and evidence checks.

## Standard directory layout

```text
my-project/
├── README.md
├── src/                         application code; use your own source layout
├── configs/                     scientific parameters and input declarations
├── jobs/
│   └── run.sbatch                batch entry point
├── .tower/                      reusable, versioned Tower integration
│   ├── config.json
│   ├── contracts/
│   │   └── outputs.v1.json
│   └── definitions/             optional experiment and workflow recipes
│       ├── scaling.json
│       └── workflow.json
├── runs/
│   └── <run_id>/                one execution attempt, created exclusively
│       ├── run.json             identity, lifecycle, and declared file paths
│       ├── logs.json            grouped index of exact log locations
│       ├── metrics.jsonl        live numeric measurements and progress
│       ├── summary.json         final measurements and scientific results
│       ├── outputs/             tables, models, checkpoints, other results
│       ├── logs/
│       │   ├── stdout.log
│       │   └── stderr.log
│       └── passports/           immutable Tower provenance records
├── reports/
│   └── planning.json            bounded aggregate of explicitly selected runs
└── logs/                        optional Slurm output opened before job startup
```

`logs/`, `outputs/`, and `passports/` can stay empty until they have real data.
Recipes are optional: add them when the project actually has a controlled
scaling experiment or a dependency graph.

Track integration configuration, contracts, recipes, schemas, and reporting
code in Git. Keep generated run directories, reports, scheduler logs, and
private passports out of Git by default. Share a selected, reviewed run bundle
when someone needs to reproduce or inspect its results.

Tower's normal submission receipt may store passports under `.tower/passports/`
in the batch workdir. Ignore that generated subdirectory. An explicit
`--passport-dir` can associate the passport with a particular run directory.

## Identity and path rules

Use these identities consistently:

| Field | Meaning | Example |
| --- | --- | --- |
| `run_id` / summary `id` | Unique execution attempt; never reuse it | `fit-20261004T120000Z-a1` |
| `experiment_id` | Logical experiment shared by its attempts | `regression-v3` |
| `attempt` | Positive retry number within that experiment | `1` |
| `name` | Stable workload name used for comparisons | `my-project/regression-v3` |
| `job_id` | Actual Slurm scheduling identity, when known | `12345` or `12345_7` |

A project-scoped prefix or a UUID keeps run IDs distinct when you combine
projects. Use 1–128 characters from letters, digits, `.`, `_`, and `-`, starting
with a letter or digit. Keep a new directory for each retry, restart, or array
task. A changing job ID or run ID belongs in its identity field; putting it into
the stable workload name would prevent comparison of repeated work.

For an array task, use the array's parent job ID plus task index as its Slurm
identity. Keep every task's reports separate. Batch/extern steps and repeated
snapshots are not independent experiments.

Use relative, forward-slash paths in the shared project files. Their bases are:

| Path | Resolved relative to |
| --- | --- |
| `run.json.paths` entries | That run's directory |
| `run.json.provenance.script` | The project root |
| Output contract entries | The run root passed as `--workdir` or to `validate` |
| Relative `research.metrics_file` | `research.workdir`, including a CLI override |
| Relative `logs.manifest_file` | Selected `research.workdir` (`--workdir`), otherwise the selected job's actual WorkDir |
| Relative `logs.json` entries | The directory containing that log index |
| Contract/config/planning filenames supplied to Tower | The shell's current directory |
| A recipe's relative batch script | The explicit planning workdir |

Launch the documented commands from the project root. Use an absolute run root
when passing `--workdir`. An artifact contract names exact files, with no globs,
`..`, absolute paths, or symlink traversal.

Choose a run explicitly. Tower supports `{job_id}` substitution in configured
metric and log-index paths; it does not expand that token in an artifact workdir or discover
an active run from `run.json`. For a project whose directory names are actual
job IDs, `runs/{job_id}/metrics.jsonl` can follow the selected job with the
project root as the metric workdir. To validate that job's outputs, select its
concrete run directory as the artifact workdir.

## The files Tower consumes

| File | Producer | How Tower uses it |
| --- | --- | --- |
| `.tower/config.json` | Project author | `--config` attaches explicit report paths and preferences |
| `metrics.jsonl` | Application | Experiment graphs, latest values, progress, and conditional ETA |
| `.tower/contracts/outputs.v1.json` | Project author | Artifacts checks exact declared result files |
| `summary.json` | Application, then optionally scheduler reconciliation | Exported records become prediction/scaling evidence |
| `reports/planning.json` | Explicit aggregation step | Resources, Tradeoffs, Scaling, and optional captured scheduler/workflow evidence |
| Recipe JSON | Project author | Scaling/workflow analysis and script preflight |
| Passport JSON | Tower's provenance API | Passport view and immutable evidence comparisons |
| `logs.json` | Application's run coordinator | Native grouped Logs file browser; explicitly bound through `logs.manifest_file` |
| Job stdout/stderr | Application and batch launcher | Logs uses paths reported by Slurm or retained from actual controller evidence |
| `run.json` | Application | Project-owned identity and inventory; declared artifact checks can validate its presence/keys |

`run.json` and `summary.json` are project conventions. Tower reads metrics
directly, checks declared files through contracts, and analyzes summaries after
they are placed in a supported planning bundle. The template supplies that
explicit aggregation step.

## Log locations: `logs.json`

Keep a portable index beside each run's `run.json`. The index describes log
locations rather than copying or merging their contents. For example:

```json
{
  "schema": "tower.logs/v1",
  "run_id": "fit-20261004T120000Z-a1",
  "job_id": "12345",
  "logs": [
    {"id": "application.stdout", "path": "logs/stdout.log", "label": "Application stdout", "group": "Application"},
    {"id": "application.stderr", "path": "logs/stderr.log", "label": "Application stderr", "group": "Application"},
    {"id": "worker.rank-0", "path": "logs/rank-0.log", "label": "Rank 0", "group": "Workers"},
    {"id": "batch.stderr", "path": "/scratch/my-project/batch/12345.err", "label": "Batch stderr", "group": "Scheduler"}
  ]
}
```

Use the actual `run_id` and scheduler `job_id`, or omit `job_id` for a local run.
When `job_id` is present, Tower checks it against the selected job; a mismatch
does not attach another job's logs. `run_id` is descriptive project identity.
Declare `"log_index": "logs.json"` under `run.json.paths`; the template does so
automatically. Tower still requires the explicit native configuration binding
and does not follow the inventory's paths automatically:

```json
{"logs":{"manifest_file":"logs.json"}}
```

With `--workdir "$RUN_DIR"`, that index is read from the selected run. Otherwise
a relative configured index uses the selected scheduler job's actual WorkDir.
Use `runs/{job_id}/logs.json` from a known project workdir when run directories
use real scheduler IDs. If no real WorkDir is available, bind the concrete run
directory rather than relying on the shell's current directory.

| Index field | Rule |
| --- | --- |
| `schema` | Exactly `tower.logs/v1` |
| `logs` | Required array, at most 256 exact file entries |
| Entry `id` | Unique stable ASCII ID, 1–128 letters/digits/`.`/`_`/`-`, starting with a letter or digit |
| Entry `path` | Required nonempty printable path, at most 4,096 characters; no globs or backslashes |
| Entry `label` / `group` | Optional printable text, at most 160 characters each |
| Entry `description` | Optional printable text, at most 512 characters |
| `run_id` / `job_id` | Optional actual identities; use the selected job's exact scheduler ID |

Relative entry paths resolve from the **index directory**. Explicit `../`
locations can name sibling logs; unlike artifact-contract outputs, read-only
log attachments are not confined to the run root. Prefer relative paths within
the shared bundle. An absolute path may refer to an actual external file, but
keeps its source-machine meaning when the project moves. Copy that file into
the bundle and update its entry, or rebind the external path explicitly.

Publish UTF-8 JSON atomically, with finite values and unique keys, within the
256 KiB native read budget. Use one coordinator to register entries. The template's
`register_log(run_dir, id, path, label=..., group=..., description=...)` adds one
location atomically and refuses duplicate IDs or normalized paths. It never
reads log contents or scans directories. A declared log may not exist yet;
Tower reports missing files. Existing local files must be regular files with
real directory ancestors. Do not point entries at symlinks, pipes, or devices.

In **Jobs**, move to an active row or a **Recents** row and press `l`. In
**History**, select a job in any state and press `l`. Logs stays attached to
that exact job rather than switching to a currently running one. Press `O` to
open the grouped file list, move with arrows/PgUp/PgDn/Home/End, and press Enter
to open a file. Esc returns from the file to its list, then closes the list;
select another entry without changing jobs. Lowercase `o` remains quick file
cycling, and `e` switches scheduler stdout/stderr.

The list combines scheduler stdout/stderr, bounded job-ID-matching files in
those output directories, and explicit index entries. Different directories
and groups stay visible. Duplicate resolved locations produce one catalog
entry. There is no recursive filesystem scan. Catalog work is cached and
performed by the shared background worker to keep terminal input responsive.

Older Slurm accounting installations may omit stdout/stderr or WorkDir for
finished jobs. Tower uses actual retained controller paths when available and
reports missing evidence otherwise. A concrete run workdir and its index allow
project-owned logs to remain accessible even after scheduler metadata expires.
Preserve indexes and failure logs when retaining a failed, cancelled, timed-out,
or otherwise incomplete attempt.

## Live metrics: `metrics.jsonl`

Write UTF-8 JSON Lines, one complete object followed by a newline:

```json
{"t":1791115200.25,"step":12,"phase":"fit","metrics":{"loss":0.032,"throughput_samples_per_second":142.5},"progress":{"completed":12,"total":100,"unit":"steps"}}
```

| Field | Rule |
| --- | --- |
| `t` | Finite, nonnegative Unix epoch seconds, UTC; fractions are allowed |
| `metrics` | Object of numeric observations; at most 64 names |
| Metric name | Printable, nonempty string, at most 96 characters |
| Metric value | Finite JSON number; negative values are valid |
| `step` | Optional integer from 0 through 2^63−1 |
| `phase` | Optional printable string, at most 160 characters |
| `progress.completed` | Finite number, at least zero and no greater than total |
| `progress.total` | Finite number strictly greater than zero |
| `progress.unit` | Optional printable string, at most 64 characters |

Use stable metric names, with units in names when useful: `runtime_seconds`,
`throughput_samples_per_second`, `max_task_rss_bytes`, `accuracy_fraction`.
Report a fraction as 0–1 consistently, or name an explicit percentage metric.
Put experiment labels and parameters in run metadata rather than generating a
new metric name for each sample or worker.

Omit an unavailable metric. `null`, strings such as `"4 GiB"`, Boolean values,
NaN, and Infinity are not numeric metric observations. Zero is a real observed
value. Keep each line below 65,536 bytes.

Append complete lines and flush at a useful cadence, such as every few seconds
or every meaningful iteration. Use one coordinator to write the run's stream,
or Tower's locked writer for cooperating processes. Give independent ranks and
array tasks their own streams. Each record should describe measurements from
that timestamp; avoid combining unsynchronized peaks into an allocation total.

Report increasing timestamps and steps within a phase. Reset progress when a
phase starts. Tower computes a progress rate and ETA from several observations;
the writer supplies `completed`, `total`, and `unit`. A measurement's wall-clock
timestamp and its elapsed duration are separate quantities: use a monotonic
clock to measure runtime, and an epoch clock for `t`.

If Tower is installed in the application's Python environment:

```python
from tower.metrics import write_metric

write_metric("metrics.jsonl", {"loss": 0.032}, step=12, phase="fit",
             completed=12, total=100, unit="steps")
```

Other languages can append the same JSONL. The template's `reporting.py` writes
the format using only Python's standard library.

## Run inventory: `run.json`

The inventory's minimum fields are:

```json
{
  "schema": "tower.run/v1",
  "run_id": "fit-20261004T120000Z-a1",
  "experiment_id": "regression-v3",
  "attempt": 1,
  "state": "RUNNING"
}
```

Add declared relative paths, actual job identity, stable parameters, code/input
identity, and request metadata as needed by your application. The schema gives
the supported vocabulary. Create the directory exclusively before work starts;
an existing directory belongs to an earlier attempt.

Write a new complete JSON document to a temporary file in the same directory,
then atomically replace the inventory. Use the same publication pattern for
the final summary and aggregate report. Readers should see a complete previous
or next version rather than a half-written JSON document.

The application can report its observed success or failure. A lost process may
leave a run marked `RUNNING`; a downstream reconciliation step should use actual
scheduler evidence to finalize it. A termination signal alone does not establish
an OOM or a timeout. Preserve interrupted measurements and record the verified
terminal state when it becomes known.

## Final report: `summary.json`

Use flat allocation and measurement fields so the same record can feed the
existing prediction and scaling readers. A completed Slurm-backed report might
look like this:

```json
{
  "schema": "tower.summary/v1",
  "id": "fit-20261004T120000Z-a1",
  "job_id": "12345",
  "name": "my-project/regression-v3",
  "state": "COMPLETED",
  "partition": "main",
  "account": "research",
  "qos": "normal",
  "cpus": 4,
  "nodes": 1,
  "gpus": 0,
  "gpu_type": "",
  "mem_bytes": 8589934592,
  "time_seconds": 3600,
  "runtime_seconds": 91.2,
  "cpu_seconds": 310.4,
  "memory_bytes": 2684354560,
  "memory_scope": "max_task_rss",
  "script_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "input_size": 100000,
  "parameters": {"dataset": "v3", "iterations": 100, "code_revision": "release-17"},
  "start": 1791115200,
  "end": 1791115291.2,
  "results": {"loss": 0.018, "accuracy_fraction": 0.94}
}
```

These values illustrate the shape; the producer must report its actual
measurements and fingerprint. `schema`, `id`, `name`, and `state` form the minimum
summary. Optional unknown measurements can be omitted or `null`. The minimum
summary provides an inventory even when there is insufficient evidence for an
analysis.

| Field | Meaning |
| --- | --- |
| `cpus`, `nodes`, `gpus` | Known total allocation counts; `gpus: 0` means a known CPU-only allocation |
| `mem_bytes` | Total allocation memory request, in bytes |
| `time_seconds` | Requested walltime, in seconds |
| `runtime_seconds` | Measured execution duration; exclude time waiting for allocation |
| `cpu_seconds` | Observed CPU time summed across the declared scope |
| `memory_bytes` | Observed memory measurement, paired with `memory_scope` |
| `script_sha256` | Actual 64-hex fingerprint of the declared executable/batch entry point |
| `input_size` | Explicit workload-size measure with a stable definition |
| `parameters` | Small stable JSON object describing comparable scientific work |
| `results` | Project-specific final scientific results |
| `start`, `end`, `submit` | Optional epoch timestamps for the corresponding actual events |

The observed memory scopes are `job_peak`, `per_node_peak`, and `max_task_rss`.
They describe different measurements. Ordinary `sacct` MaxRSS belongs to
`max_task_rss`; it is not total job memory. Record a job peak only when you
actually measured a simultaneous allocation-wide peak.

Slurm's `--mem` is normally a per-node request. A `--mem-per-cpu` request needs
the known allocation CPU count. Normalize these into a total `mem_bytes` only
when the counts and request semantics are known. Submission recipes retain
Slurm's own per-task/per-node option vocabulary. Do not put a per-task CPU count
into the total `cpus` field.

Use omission or `null` for an unknown allocation. Application worker count and
`os.cpu_count()` do not prove allocated CPU count. An unknown GPU request is
different from a verified zero-GPU allocation; an empty GPU type means a known
generic GPU request where that distinction matters.

### Preserve comparability and failure evidence

Resource predictions compare `name`, `partition`, total CPU/node/GPU counts,
and any supplied account, QoS, request, script, input-size, or parameter
identities. Keep code and input identities in small, stable `parameters` fields
as well as their detailed provenance records. A batch-script hash alone may not
identify the application it launches or the dataset it reads.

Use explicit data hashes or versions, a code revision/fingerprint, and the
parameters that change scientific work. Keep parameters shallow: the predictor
accepts depth at most four and a bounded 128-value tree. Record a repetition's
identity separately. If different random seeds belong to the same experimental
protocol, document that protocol; if the seed defines different work, include
it in the cohort parameters.

Preserve `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`, and other observed
terminal states. A timed-out duration and OOM memory observation are incomplete
measurements. Include those reports in aggregates instead of discarding them
or relabeling them as completed runs. Unavailable measurements remain unknown.
Tower can then withhold unsupported intervals and scaling claims.

For controlled scaling, add explicit `workers`, `problem_size`, and a positive
`repeat` number. Use three or more comparable repeats per worker configuration
to obtain observed repeat spread. Keep fixed work for strong scaling or a fixed
work-per-worker ratio for weak scaling. Retain the same code and parameter
identity across the comparison. These are empirical measurements, not assumed
speedups from requesting more CPUs.

For resource comparisons across clusters, include the relevant platform or
hardware identity in cohort parameters and select comparable runs explicitly.
Keep a worker's CPU time separate from a measured sum across an allocation;
record its source and scope in report metadata.

## Output declaration: `.tower/contracts/outputs.v1.json`

Example contract:

```json
{
  "version": 1,
  "outputs": [
    {"path": "run.json", "required": true, "format": "json", "max_bytes": 65536,
     "required_keys": ["schema", "run_id", "experiment_id", "attempt", "state"]},
    {"path": "logs.json", "required": true, "format": "json", "max_bytes": 262144,
     "required_keys": ["schema", "logs"]},
    {"path": "summary.json", "required": true, "format": "json", "max_bytes": 262144,
     "required_keys": ["schema", "id", "name", "state"]},
    {"path": "metrics.jsonl", "required": true, "format": "text", "min_bytes": 1,
     "max_bytes": 4194304},
    {"path": "outputs/results.csv", "required": false, "format": "csv",
     "columns": ["step", "value"], "min_rows": 1, "max_bytes": 4194304}
  ]
}
```

Declare your project's actual result names and required columns. Mark a result
required when successful completion depends on it. Use bounded byte limits and
appropriate row/key/hash checks. Tower validates JSON top-level keys and CSV
structure; it does not execute custom validation commands or recursively parse
a scientific model.

An artifact check passing establishes the declared file/structure checks.
Read the run's state and scientific results to assess its outcome. During a
running job, a required final file may legitimately be absent. Full JSON Schema
validation can be added to the producing project's own CI; Tower's artifact
`required_keys` check is intentionally narrower.

## Aggregate report: `reports/planning.json`

Export explicitly selected run summaries to the native planning format:

```json
{
  "version": 1,
  "kind": "tower.planning",
  "coverage": 0.8,
  "query": {"name": "my-project/regression-v3", "partition": "main",
            "cpus": 4, "nodes": 1, "gpus": 0},
  "history": [],
  "scaling": []
}
```

`history` contains the selected summary objects, retaining successes and
failures. `query` describes the particular comparable workload/allocation you
want to predict. Include the known scientific identity fields used by those
summaries and an explicit memory scope when predicting a memory measurement.
`scaling` contains the subset with declared scaling metadata, including failed
attempts. A single completed run is useful evidence but cannot establish a
calibrated prediction interval or a repeat spread.

The template collector accepts explicit run directories and an optional
reference run, validates its inputs, and publishes a bounded aggregate. It does
not recursively search storage or infer missing metadata. Keep normal planning
bundles at most 1 MiB and at most 10,000 input records. Archive older reports
and select a coherent analysis cohort when the dataset grows. Scaling-only
inputs can use Tower's 8 MiB reader; shared bundles should fit the smaller limit.

Optional `candidates`, `jobs`, `observations`, `queue_history`, `workflow`, and
captured blocker metadata follow the native formats in the
[wave-two guide](WAVE_TWO.md). Publish captured scheduler evidence with its
actual timestamp, connection identity, and known resource fields. An explicit
`jobs` snapshot must carry its own evidence; Tower will not fill it from an
unrelated live snapshot. Let Tower collect connection-scoped pre-start forecast
observations for live jobs rather than generating queue outcomes from an
application's runtime.

Keep forecast observations separate from experiment measurements. Never derive
a queue start from CPU utilization, a synthetic date, or a completed-only
runtime estimate. Use actual submission/start events and issued predictions.

## Open and inspect a project

From the project root, after creating a run:

```bash
RUN_DIR="$PWD/runs/my-run"
tower --config .tower/config.json --workdir "$RUN_DIR" \
  --tab research --research-view experiment
tower run validate .tower/contracts/outputs.v1.json "$RUN_DIR"
```

On a development machine without Slurm, add `--fake --no-state --no-plugins`
to the dashboard command. Its scheduler data is simulated; the attached project
files are actual reports. On CARC, use the site's Python/Slurm environment and
your account, partition, and polling preferences from the [runbook](runbook.md).

After exporting a planning bundle:

```bash
tower run predict --file reports/planning.json
tower run scaling analyze reports/planning.json --mode strong --baseline 1
tower --config .tower/config.json --workdir "$RUN_DIR" \
  --tab research --research-view predict
```

The two `tower run` analyses above are offline. Missing allocation or repetition
metadata produces an explicit insufficient/partial result. Supply real metadata
when it is known; keep local development reports honest when it is unavailable.

For provenance, use Tower's API or command to create a genuine passport:

```bash
tower run passport capture jobs/run.sbatch --workdir "$PWD" \
  --input experiment.py --input reporting.py \
  --output-dir "$RUN_DIR/passports"
```

Pass the printed file path to `--passport` or the Passport command. Keep
submission receipts and actual returned job IDs in separate metadata; saved
passports remain immutable. A `run.json` inventory or hand-written hash record
does not have the checksum/identity contract of a Tower passport.

Copy passports intact when moving a run. Their captured host and absolute paths
describe the original observation. Use the relocated run's relative report
paths for viewing; preserve the passport's contents and checksum.

Batch stdout/stderr directories must exist before submission because Slurm
opens them before application startup. Use precreated run log directories when
the run ID is known before submission, or a precreated project `logs/` directory
with Slurm job/task tokens. Tower follows the selected job's actual scheduler
`StdOut`/`StdErr` paths and the explicitly bound `logs.json` index. Declare
application logs in that index for browsing, and optionally in the artifact
contract as exact text files for checks. Preserve failure output in either case.

## Adoption checklist for every new project

1. Copy the integration template and keep its source/parameter layout suited to
   your application.
2. Create a unique run directory and inventory before execution; preserve old
   attempts.
3. Emit complete, finite JSONL metric records with stable names and units.
4. Publish actual final measurements/results atomically and retain failures.
5. Declare exact output files and meaningful bounds/checks in the contract.
   Index application, worker, and external logs in `logs.json` with stable groups
   and actual job identity, then verify browsing a retained failed attempt.
6. Record real allocation counts, request semantics, memory scope, and
   scientifically comparable code/input/parameter identities when known.
7. Export selected run summaries into a bounded planning bundle for analysis.
8. Verify an actual metric stream and contract with Tower before submitting
   a production job; keep scripts, local report production, and schemas in CI.

Version new reporting contracts when their meaning changes. Preserve the
version-1 native field names and units when extending reports; put additional
scientific results under `results` and keep raw large artifacts in `outputs/`.
The producing project owns measurement definitions and scientific validation.
Tower presents the declared evidence and its limits.
