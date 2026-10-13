# Response diagnostic implementation verification

The implementation adds a separate trainability diagnostic with 34 architecture
and training arms, independent teacher/screening contracts, validation-only
checkpoint selection, and source-bound research metrics. The native default
declares 97 calibration trials at 64 updates followed by 669 diagnostic trials
at 2,000 updates. Its required CUDA suite contains 97 cases.

The checks below ran on 2026-10-13 in the managed cloud checkout, using its
standalone Python 3.12.14 venv, PyTorch 2.10.0+cpu, NumPy 2.2.6, and SciPy
1.15.3. CUDA and a native Fedora Slurm allocation were unavailable. These
checks establish implementation behavior and artifact integrity, not native
calibration, scientific benefit, or an efficiency advantage.

## Final regression

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TDN_LOCAL_TEST_ROOT=/workspace/TDN \
  .venv/bin/python -m pytest -q \
  tests/test_response*.py tests/test_frontier_reporting.py \
  tests/test_desktop_slurm_runtime.py tests/test_tower_reporting.py \
  -m 'not gpu' \
  --basetemp .runtime/onboarding/diagnostic-final-regression-tests \
  -o cache_dir=.runtime/onboarding/diagnostic-pytest-cache
```

Result: **715 passed, two optional checks skipped, 97 CUDA cases deselected**
in 86.97 seconds. The skipped checks require an optional unchanged Tower
checkout and an optional schema-validation development dependency. Mocked
native lifecycle tests cover preflight failures, Tower startup failures, and
failures after a completed science body; none represents actual GPU execution.
Python compilation, Bash syntax checks for the diagnostic launchers, and
`git diff --check` also passed.

## Reduced end-to-end CPU workflows

Two fresh reduced CPU workflows completed in separate directories:

| Check | Targets | Completed units | Completed trials | Logical updates |
| --- | --- | ---: | ---: | ---: |
| CPU A | All three target contracts | 138 / 138 | 97 / 97 | 194 |
| CPU B | FD same-grid | 51 / 51 | 34 / 34 | 68 |

CPU A exercised the complete reduced three-target roster before the final
correction diagnostic/public-query-plan export change. CPU B exercised the
final training/report contract, including measured correction diagnostics,
whole-history optimization summaries, selected-checkpoint inference timings,
compacted query plans, CSV exports, plots, and the portable Raven project.
All 34 CPU B trials retained their correction and inference measurements.
Their numerical values and timings have CPU scopes.

The original execution/science artifact digests were checked for every unit
in both runs. CPU B also passed the final execution/science contract validator,
using its recorded source identity. CPU A retains its earlier measurement
contract; it was not reinterpreted using the final contract. Subsequent native
worker reporting and test changes passed the final regression above.

The portable archives contain copied research metadata and reports rather
than checkpoints or array replay payloads. Archive integrity, confined relative
paths, log indices, metrics, moved-project usability, and actual Raven reader
compatibility are recorded in [verification.json](verification.json).

The default 669-trial compiled-metadata size projection is 177.5 MiB, including
an 8 MiB reserve, below the 256 MiB per-file export limit. This is a metadata
shape projection from CPU templates, not a native runtime or accuracy result.

The source hashes and archive hashes in the receipt identify the exact tested
trees. Runs were executed against uncommitted changes based on `3c092e1`; the
tree hashes, rather than that base commit alone, identify the implementation.
Large generated run directories remain local and are not committed.

## Native validation and execution

Use the configured Fedora checkout and its real Slurm allocations:

```bash
cd /home/rahel/TDN
git pull --ff-only
bash scripts/response_diagnostic_campaign.sh
```

This creates fresh IDs, runs native CUDA checks and calibration, binds the
full diagnostic to verified measurements, preserves failures and scientific
negative outcomes, and exports the final portable Raven report. Read the
[runbook](../../docs/RESPONSE_DIAGNOSTIC_RUNBOOK.md) for receipts, recovery,
Raven registration, detail pages, and upload paths.

The historical Response pilot and its `STOP_OR_UNRESOLVED` gate remain
unchanged. These implementation checks do not authorize confirmation or
deployment.
