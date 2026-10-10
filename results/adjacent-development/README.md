# Adjacent CPU development evidence

Read the [research assessment](../../docs/ADJACENT_INITIAL_REVIEW.md) before
interpreting the plots. This is an isolated adjacent study, not confirmation
or a change to the main architecture.

- [HTML atlas](atlas/index.html) and [67-panel PDF](atlas/adjacent-atlas.pdf).
- `smoke-v2/`: readable JSON copies of the corrected 15-unit CPU run.
- `selected-development-v2/`: audit, D01 and D05 at 16×16; this is not the entire
  development workflow.
- `superseded-smoke-v1/`: retained earlier measurements with known oracle and
  plotting defects. Do not substitute these for corrected primary evidence.
- `archives/`: exact complete runs, including checkpoints, references, science
  and execution seals, raw arrays, reports and Tower sidecars.
- `archive-index.json` and `publication-index.json`: byte counts and SHA-256
  values. Readability directories omit non-JSON payloads and are **not** complete
  independently verifiable stage directories; use the archives for verification.
- `checks/`, `validation.json`, `cost-ledger.json`: executed validation,
  source/environment identities, limitations and recorded costs.
- `d03-descriptive-frontier.json`: explicitly post-hoc CPU timing point minima;
  not statistical confidence intervals or a deployable policy.

Verify the archive hash against its index before extracting it under a fresh
project checkout. Every member is under `runs/`; original artifacts remain
unchanged. The records identify the measured source commits; do not attempt to
resume them using a different checkout, environment or revision.

The atlas JSON source pointers refer to original paths in those archives.
Observed ranges are not confidence intervals. Raw tables retain derived
summaries as well as measurements where useful, but plots use canonical
observations. Repeated tracks, schedules and phases do not enlarge the number
of independent field parents.

To replay the selected development subset on CPU, use a checkout of the measured
source and its venv, set project-local temporary/cache paths as in
`scripts/adjacent_local.sh`, and run `selected-development-command.py` from the
project root. It refuses to overwrite its run path. For routine local or native
execution, use the [runbook](../../docs/ADJACENT_RUNBOOK.md); the native workflow
supports fresh full evaluation, recovery and actual CUDA checks.
