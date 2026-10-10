# Native portfolio review evidence

Start with [the comprehensive review](../../docs/PORTFOLIO_FULL_REVIEW.md). This directory contains **post-hoc descriptive analyses** of `fedora-portfolio-20261009T183307014599Z`, source commit `c5c9a85e5cc92cf4f361e57a74626c8a0a5c2fc3`. It does not contain a new training run, a revised confirmation protocol, or complete replay inputs.

## Contents and interpretation

| File | Contents |
|---|---|
| `native-protocol.json`, `native-claims.json`, `native-aggregate-summary.json` | Exact bytes of the original declarations and aggregate claim records. Original GOOD/BAD/NA outcomes remain unchanged. |
| `statistics-review.json.gz` | Every family/data size/grid/regime/distribution/horizon/seed; RMS, maximum, centered, mean, spectral, uncertainty, time, bounds, coverage and selected paired effects. |
| `statistics-endpoints-compact.csv.gz` | Exact-value derived column projection of all 101,376 endpoints. This is not the original JSON serialization and does not replace its provenance. |
| `statistics-all-pair-subgroups.json.gz` | Original 14,752 exploratory pairwise subgroup comparisons in compressed JSON form; not multiplicity-adjusted. |
| `statistics-followup.json.gz` | Derived data-size effects, physical violations, worst cases and conditional economic comparisons. |
| `training-summary.json.gz` | All 176 selected model instances, 536 trials, curve diagnostics, fitting behavior and data-size comparisons. |
| `training-response-audit.json.gz` | Post-hoc reconstruction of small models from retained parameter reports; initial-state gain responses, not new PDE inference accuracy or GPU timing. |
| `costs-summary.json.gz` | Allocation/step accounting without double-counting; trial versus campaign cost; scaling and timing limitations. |
| `costs-optimistic-amortization.json` | Explicitly incomplete training-only break-even derivation on mutually qualifying FNO comparisons. Not deployment economics. |
| `costs-amortization-artifact-verification.json` | Original atlas amortization CSV: 50,688 NA rows and missing savings/break-even columns, confirming the reporting defect. |
| `provenance-summary.json` | Forty stage metadata/lineage/ledger checks, split disjointness, all 1,952 reference metadata records and native JUnit counts. |
| `statistics-initial-fields.json`, `statistics-timing-counts.json` | Reconstructed input interval audit and actual timing-group inventory. |
| `source-index.json.gz` | Canonical source-relative paths and original exact-byte hashes used by the analysis. |
| `publication.json` | SHA-256 inventory of this curated publication, original archive identity and scientific source identity. |
| `*.py` | Analysis programs retained for inspection and reproduction with the original archive. |

The main archive has 17 parts and SHA-256 `979fd362ad3c0dd7bc456707cc667875d6f0f3e461edf8965b0c4751d4403591`. Its 10,034 retained files were byte-verified. The compact preset omitted 1,402 numerical arrays, checkpoints and rendered assets while retaining their declared original hashes. This review cannot independently verify omitted bytes or rerun every prediction without them. No original artifact was changed.

Reference uncertainty comes from actual refinement, not a certificate. Primary intervals and additional review intervals are descriptive; repeated schedules/grids/seeds are paired within independent fields. All 24 confirmation fields have now been inspected. New selection or architectural decisions require new confirmation data.

## Reproduce the analyses

Use this repository's Python venv and the original reviewed scientific source. No GPU or Slurm allocation is needed to parse retained measurements. The algebraic response audit is CPU-only and does not establish new prediction accuracy.

Place the uploaded `index.json`, `manifest.json` and all 17 original parts in `.runtime/portfolio-upload-20261009/`, preserving filenames. In a fresh `.runtime/portfolio-analysis/` working directory, copy the analysis programs from this folder. `extract_canonical.py` verifies SHA-256 while producing compressed exact-byte canonical working objects; it intentionally refuses to overwrite existing objects.

```bash
cd /home/rahel/TDN
mkdir -p .runtime/portfolio-analysis
cp results/portfolio-full-review/*.py .runtime/portfolio-analysis/
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .runtime/portfolio-analysis/extract_canonical.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .runtime/portfolio-analysis/provenance-review.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .runtime/portfolio-analysis/training-analysis.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .runtime/portfolio-analysis/costs-analysis.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .runtime/portfolio-analysis/statistics-review.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .runtime/portfolio-analysis/statistics-followup.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .runtime/portfolio-analysis/training-response-audit.py
```

The extractor reads the archive sequentially without writing a multi-gigabyte uncompressed tree. Analysis materializes some JSON in memory. All work stays inside the checkout. Keep original completed runs and archives intact. These are reproduction commands for the review, not instructions to rerun the expensive native experiment.

Several small supporting audits were computed directly during review; their source scopes and values are recorded in their JSON. The main analysis scripts and source index provide the broad reconstruction path. Native GPU measurements remain those from the uploaded run.
