# Response implementation verification

These are CPU implementation checks from the managed cloud checkout, not new
RTX 4090 measurements or evidence of neural superiority. The native target is
Fedora / Python 3.13.13 / RTX 4090; this environment uses Python 3.12.14 and
PyTorch 2.10.0+cpu. Each receipt records its source identity. Intermediate
receipts are retained when the implementation changes.

Audit test-isolation correction after native job 938: **460 passed, 54 CUDA
cases excluded**, in 42.39 seconds. See `pytest-audit-isolation.txt` and
`audit-isolation.json`. This uses the native worker's exact CPU audit selection
and environment-preparation helper with synthetic Slurm job/step/account
metadata confined to the test subprocess. The 12 dependency-order cases now
isolate all inherited Slurm/TDN variables, including the account variable that
caused the native failure. Production runtime guards are unchanged. This is
CPU infrastructure verification; the corrected native CUDA run remains pending.

`pytest-audit-invalid-fixture.txt` retains an initial verification invocation
with a nonnumeric synthetic job ID: 457 passed and three reporting tests
correctly rejected that invalid identity. Using numeric synthetic IDs corrected
the invocation; no production reporting checks were changed.

Native-smoke failure correction: **502 passed, 54 CUDA cases excluded**, in
65.50 seconds. See `pytest-native-smoke-fixes.txt` and `native-smoke-fixes.json`.
The corrected local smoke uses the native controller's sorted JSON format and
completed all 42 stages with verified execution/science seals. All 16 training
checkpoints reproduced exactly in fresh processes. The three new native CUDA
tests exercise fresh-process direct-FNO reload for each target; their desktop
execution remains pending. This correction preserves exact parity and source
checks rather than relaxing them.

Original release regression result: **486 passed, 51 CUDA cases excluded**, in 49.34 seconds.
See `pytest-final.txt` and `release-verification.json`.

The release validation includes the Response regression suite, adjacent export
and reporting regressions, and a complete local smoke workflow. The smoke uses
8×8 fields and two updates per trial. It must not pass the native trainability
gate or be accepted as a GPU throughput calibration.

`local-smoke-a-receipt.json` records the first complete integration: 42 completed
and verified execution/science seals, 115 analytical figures (PNG and PDF), and
no missing stages. Its measured tables retain negative and unresolved outcomes.
The atlas reports four unresolved/no-advantage comparisons; this tiny run is
not an experiment with enough training or independent parents to establish a
scientific effect.

`local-smoke-final-receipt.json` records the subsequent complete release
integration: **42 verified stages and 119 figures**, with Tower recognizing all
238 PNG/PDF files and reporting no export omissions. Both selected-decoder
quadrature panels completed all six requested rules. Architecture and selected
mathematical/diagnostic figures were visually checked for readable labels and
absence of overlapping annotations. The [architecture diagram](architecture/architecture.png)
is informational; it is not a measured scientific success.

The independent numerical checks are in `math/` and `numerical-panels/`. Their
receipts bind exact source versions and distinguish numerical identities,
constructed counterexamples and representation diagnostics. They do not show
that a neural model learned those identities. The protected observer probe in
`closure/` retains its negative result rather than selecting favorable regimes.

`pytest-oversubscribed-interrupted.txt` records an intentionally interrupted
verification process after 209 passing tests: restoring a high default Torch
thread count made the tiny tests inefficient. The release command fixes
OMP/MKL/OpenBLAS to one thread. No interrupted run is counted as a passing suite.

`pytest-caption-assertion-failure.txt` retains a stale plotting-test failure:
the warning moved from beneath the axes into the caption to remove overlap.
The assertion was updated to verify its new visible location. This one-line
test change occurred after the final integration; production source remained
identical. `integration-source-files.json` and `release-verification.json`
record that distinction. The complete regression suite was rerun afterward.

`compact-review-receipt.json` verifies 2646 byte-identical restored files and
unchanged originals. The 73,254,639-byte first integration run produced a
4,890,424-byte review upload. Omitted replay arrays/checkpoints and rendered
figures are explicitly inventoried; this is transport verification, not full
binary replay. All retained numerical records keep their original bytes.

Reproduce the relevant regression checks with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_response*.py \
  tests/test_compact_review.py tests/test_resolve_tower.py \
  tests/test_frontier_reporting.py tests/test_resolve_report.py \
  -m 'not gpu' --basetemp=.runtime/pytest-response-release
```

Native CUDA cases are deliberately excluded from this CPU command. The Fedora
worker requires all 54 declared CUDA cases, without skipped or duplicate cases,
before native calibration. Use the [runbook](../../docs/RESPONSE_RUNBOOK.md).
Fresh confirmation remains a separate frozen protocol after development.
