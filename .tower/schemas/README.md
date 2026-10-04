# Tower project reporting schemas

These JSON Schemas describe the portable project convention in
[PROJECT_STANDARD.md](../PROJECT_STANDARD.md). They use JSON Schema Draft 2020-12 and
relative references so projects can copy this directory without changing URLs.
Tower's runtime remains standard-library-only; it does not load these schema
files or require a JSON Schema package.

| File | Applies to | Current Tower reader |
| --- | --- | --- |
| [metrics.v1.schema.json](metrics.v1.schema.json) | Each object in `runs/<run_id>/metrics.jsonl` | Native incremental `MetricReader`; Experiment view |
| [run.v1.schema.json](run.v1.schema.json) | `runs/<run_id>/run.json` | Project-owned descriptive manifest; explicit path binding is required |
| [logs.v1.schema.json](logs.v1.schema.json) | `runs/<run_id>/logs.json` | Native grouped Logs catalog, bound by `logs.manifest_file` |
| [summary.v1.schema.json](summary.v1.schema.json) | `runs/<run_id>/summary.json` | Flat fields consumed after aggregation into a planning bundle |
| [output-contract.v1.schema.json](output-contract.v1.schema.json) | `.tower/contracts/outputs.v1.json` | Native artifact contract loader and bounded validation |
| [planning.v1.schema.json](planning.v1.schema.json) | `reports/planning.json` | Native planning-file views and file-based analysis commands |

`planning.v1.schema.json` also includes native workflow and scaling recipe
definitions as `$defs.workflowRecipe` and `$defs.scalingRecipe`. Standalone
recipes retain their native `{"version": 1, "kind": "tower.workflow"}` or
`{"version": 1, "kind": "tower.scaling"}` header. The planning bundle uses
`{"version": 1, "kind": "tower.planning"}`. A run manifest and summary use the
distinct `schema` markers `tower.run/v1` and `tower.summary/v1`; metric records
carry no version or schema marker.
The separate log index uses `tower.logs/v1`; inventory `paths.log_index` describes
its location, while native `logs.manifest_file` binds it to the selected workdir.

`run.json` and scientific `results` are project-owned metadata. Tower 2.3 does
not scan for manifests, follow their paths, display arbitrary result objects, or
generate aggregate reports automatically. A project reporter creates
`reports/planning.json` from explicitly selected summaries; Tower reads that
file through its existing planning commands and flags. Tower passports retain
their native schema and should be created through Tower's passport commands.

## Meaning of the shared fields

Use stable workload `name`, scientific `parameters`, `script_sha256`, and known
input size to distinguish comparable work. `id` identifies one run attempt;
`job_id` identifies the scheduler job or scheduling episode. A retry gets a new
run directory and `id`. Duplicating one observation does not create another
independent repeat.

`cpus`, `nodes`, and `gpus` describe recorded allocation totals. `mem_bytes` is
the requested allocation-wide memory total and `time_seconds` is the requested
walltime limit. `runtime_seconds`, `cpu_seconds`, and `memory_bytes` are measured
quantities. An observed memory value requires `memory_scope`:

| Scope | Meaning |
| --- | --- |
| `job_peak` | Measured peak for the whole job |
| `per_node_peak` | Peak for the relevant node-level measurement |
| `max_task_rss` | Maximum individual task RSS, including Slurm MaxRSS |

These scopes are not interchangeable. Document the measurement source and
scope in `metadata`; do the same for CPU seconds. Python `process_time()`
excludes child processes and is not an allocation-wide measurement unless the
workload's scope actually matches it. Unknown data is omitted or `null`, never
an invented zero. Explicit `gpus: 0` means known zero GPUs. An empty `gpu_type`
means known generic GPUs; an omitted/null type means unknown.

Native prediction needs concrete `name`, `partition`, `cpus`, `nodes`, and
`gpus`. Known optional identities further constrain the cohort. Missing fields
or incompatible memory scopes cause abstention. Scaling additionally needs
explicit `workers`, `problem_size`, `parameters`, `script_sha256`, terminal
state, and a unique `job_id` or `repeat`; it does not infer CPU counts from
workers. Failed, timed-out, or OOM runs remain in reports as incomplete/censored
observations rather than being relabeled as successful measurements.

`UNKNOWN` and `INTERRUPTED` are descriptive application states when a verified
scheduler terminal cause is unavailable. A signal alone does not establish a
timeout or OOM. Preserve that uncertainty and reconcile with scheduler evidence
later; the analysis readers exclude unsupported states from completed fits.

## Structural and native validation

A schema validator checks types, keys, and local ranges. The native readers
also check semantics and bounded I/O. Both apply to portable projects:

- JSON is UTF-8, with no duplicate object keys or nonfinite numbers. Numeric
  unknowns use `null` or omission; a string such as `"NaN"` is not a numeric
  measurement. Native printable-text checks are stricter than the schemas'
  control-character patterns for some Unicode characters.
- A metric line ends with a newline and stays within 65,536 bytes. Each record
  contains at most 64 named metrics; the stream also retains at most 64 distinct
  series. Native progress validation enforces `completed <= total` and
  `total > 0`. The reader keeps a bounded recent sample for responsive plotting.
- A planning document stays within the default 1 MiB command/UI read budget,
  32 nesting levels, and 100,000 JSON values. Individual analysis input lists
  stay within 10,000 records; candidate lists stay within 64. Split larger
  histories into explicitly chosen cohort/report files.
- Shared prediction parameters stay within four nesting levels and 128 total
  JSON values, with at most 64 members per object/list. Schemas enforce depth
  and local limits; native validation additionally checks the total-value
  budget and byte limits.
- Output contracts stay within 256 KiB, use exact relative paths, and contain
  no symlinks or glob expansion. Native validation checks unique paths,
  ordered min/max sizes or row counts, and conflicting exact row counts.
  Passing the contract schema does not mean the outputs exist or passed checks.
- A log index stays within 256 KiB and 256 entries. Native validation checks
  unique entry IDs and actual selected `job_id` when recorded. Relative paths
  resolve from the index directory, including explicitly named sibling files;
  exact absolute paths may point to other locations. Neither expands globs.
  Catalog construction deduplicates normalized paths and shows groups/labels;
  schema validation alone does not establish file availability. These read-only
  log paths have a different policy from confined artifact-contract outputs.
- Workflow validation checks unique IDs, known dependencies, no cycles, at
  most 4,096 edges, and `lower <= estimate <= upper`. Resource/topology and
  walltime checks still apply when scripts are prepared.
- Scaling validation checks unique configurations, experiment size limits,
  concrete independent repeat identity, matching scientific work, and
  controlled strong/weak scaling. A valid recipe prepares reviewable plans;
  it does not execute the workload or establish that it consumed the controls.
- Forecast observations must contain actual pre-start scheduler predictions
  and later observed starts. Calibration checks chronology, resource cohorts,
  distinct submitted jobs, and forecast lead. A planning `now` value freezes
  analysis for historical reports; omit it for a current queue forecast.

Local native file readers reject symlinks/devices and detect files that change
while being read. Producers should append complete metric records and replace
manifests, log indexes, summaries, and aggregates atomically. Keep logs, credentials,
datasets, personal shell configuration, and generated run outputs out of the
public project template.

For optional development validation, use a Draft 2020-12 validator and resolve
the sibling schema references from this directory. Native Tower readers remain
the final authority for runtime compatibility; the runnable project template
demonstrates those checks without adding a dependency to Tower.
