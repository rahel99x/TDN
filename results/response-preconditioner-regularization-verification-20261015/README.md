# P1R implementation verification — 15 October 2026 PDT

**CPU verified; native retained-bank execution pending.** This receipt was
produced on 16 October UTC (15 October in America/Los_Angeles). It verifies the
[bounded regularization audit](../../docs/RESPONSE_PRECONDITIONER_REGULARIZATION_AUDIT.md),
not a new result on the P1 physical bank. Historical experiments remain unchanged.

The public verifier passed **585/585 exact mandatory cases** in 90.70 seconds,
with zero failures, errors, skips, duplicates or missing cases. This includes
404 inherited checks and 181 new checks: 45 generalized-fitting, 19 evidence,
41 sensitivity, 33 analysis, 25 integrated workflow and 18 native lifecycle
cases. A separate **8/8 optional Raven compatibility cases** passed against
clean Raven `c34f0892d7bddc685c5af10489ecb114fd6adab6`; these eight do not overlap
the required roster. They verify compatibility, not an actual Slurm allocation.

The synthetic integration completed all **32 input queries, 13 candidates,
two validation selections and 144 metric rows**. It used generated algebraic
arrays, without PDE solves or the retained P1 values. All 11 positive fits
solved; the projection remained deliberately ineligible. Raw predictions and
inputs, parent aggregates, sensitivity, costs, selection seals and PNG figures
are retained under `synthetic/`. These fabricated test controls are not measured
solver comparisons and must not be interpreted as scientific superiority.

Both runs have source fingerprint
`ff37f88c9dcbdf394a562c0da1ef1765e4f1c1dd8b191940179786de67c66358`.
The exact source-file hashes, software versions, test roster, JUnit, logs and
before/after source identity are in `verification/`. The verify-protocol digest
is `a1ea8d521c5fb9ff0ac4f05082347f7a4307a1bb212d6daf376e7c9bb437f39c`.

## What was checked

- Generalized Tikhonov against independent SciPy/closed-form solutions,
  coordinate reparameterization, nonzero priors, nullspaces, extreme scaling,
  original-coordinate optimality and nonregression. Action-energy full-rank
  solutions match uniform shrinkage of OLS; this is not a derived PDE prior.
- Thin-SVD prediction sensitivity against explicit maps and perturbations,
  with uncertainty scenarios separated from certified error bounds.
- Pinned source archive, safe numeric loading, exact new-run selection seals,
  protected-read ordering, copied-input hashes and changed-input rejection.
- Validation-only selection, zero fallback, every failed/missing denominator,
  exact mean/spatial decomposition, raw prediction replay, partial-query
  interruption, reporting-budget enforcement and primary-error preservation.
- Native source/input/software binding, same-attempt verifier requirement,
  finite named research series before packing, failed-stage preservation,
  bounded inventories exceeding 1 MiB, terminal export and mutation refusal.
- Trusted Raven accepts a 65 MiB required table without a false content-format
  claim and accepts binary PNGs. Format checks have a 1 MiB budget per output
  folder; full native file hashes remain separate. Only named caches/test
  fixtures are excluded; `.raven` evidence is preserved.

A separate read-only integration review compared real archive metadata and
NPY headers with the workflow: all three training prefixes, validation order,
40-weight layouts, shapes, dtypes, historical methods and planned counts match.
No real retained-bank fit or inference evaluation ran in this environment.

## Reproduce and run

From the repository, with its standalone venv and a fresh output directory:

```bash
mkdir -p .runtime/regularization-verification/tmp \
  .runtime/regularization-verification/matplotlib \
  .runtime/regularization-verification/cache
TMPDIR="$PWD/.runtime/regularization-verification/tmp" \
  MPLCONFIGDIR="$PWD/.runtime/regularization-verification/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/regularization-verification/cache" \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/response_preconditioner_regularization.py verify \
  --device cpu --output .runtime/regularization-verification/recheck
```

Use the same environment assignments with
`results/response-preconditioner-regularization-verification-20261015/reproduce_fixture.py
--output .runtime/regularization-verification/synthetic-recheck` to replay the
private algebraic integration fixture. This requires the test dependency set
and never opens the real P1 array bank. `receipt.json` retains the actual commands.

The [runbook](../../docs/RESPONSE_PRECONDITIONER_RUNBOOK.md#p1r-current-bounded-retained-data-regularization-audit)
has guarded native prepare/launch/export commands. That one native attempt
repeats mandatory verification before fitting the real retained bank. Review
its selected penalties, sensitivity, original/control comparisons and missing
slots before proposing new acquisition or architecture. No larger experiment,
fresh confirmation, GPU readiness, novelty or speedup follows from these checks.

`retention.json` records copied files and the two omitted SVGs (their equivalent
PNGs are retained). The original fixture artifact manifest is preserved without
rewriting its SVG entries. Temporary pytest fixtures and caches are omitted.
`manifest.json` hashes this retained receipt, excluding only itself.
