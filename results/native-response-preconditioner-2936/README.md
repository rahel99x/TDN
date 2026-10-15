# Native response-preconditioner repair: job 2936

**Native CPU verification passed at `261365b`.** This closes the bounded
implementation repair; another repetition of the same smoke is not needed.
The [standing research program](../../docs/TDN_RESEARCH_PROGRAM.md) records the
next gate. No larger scientific campaign or learned encoder was run here.

## Execution and independent integrity review

The user supplied `launch-ba0ccf6660874b2f826b1d7fd9e25886-a1.zip`, retained
unchanged alongside this review. Job **2936** ran through Raven's native Slurm
contract on 15 October 2026, 23:12:01–23:12:45 UTC. Worker elapsed time was
44.62 seconds; allocation was four CPUs, 8 GiB, 20 minutes. Both the worker
and Raven report completion with application exit code zero.

The [integrity review](integrity-review.json) independently verifies:

- The exact **202 expected JUnit cases**, with no failures, errors, skips or
  duplicates, against the prelaunch receipt committed at `261365b`.
- Both frozen v2 protocols, all 239 stage source files, the 619-file native
  source aggregate, unchanged source snapshots, and 30 stage artifact hashes
  and sizes.
- All 38 required portable Raven artifacts and all 82 research rows across
  raw JSON, CSV and Raven's metric stream.
- All 54 retained evidence files against the original ZIP bytes.

The source fingerprint is
`c5adaf6a929bd583efe0beb92de65b723b0c2492fb67836ddaa86efdbb6b1dbe`.
The original ZIP SHA256 is
`f1da910e7075ff7c569c931adbbf22c905e04f6a1f5be57735d75d05ecee149d`.

The ZIP has 220 entries, including macOS metadata, test fixtures and 19 test
fixture symlinks. The [extraction manifest](extraction-manifest.json) records
the regular evidence files retained for inspection and every omitted entry.
No symlink or uploaded code was executed. Deliberately failed test fixtures
are not failures of the actual run.

Final independent Slurm accounting is absent; `scheduler_outcome` is unknown
and the allocation's RUNNING field is a startup snapshot. Original prepared
contract/profile bytes are absent, so their raw hashes cannot be independently
rechecked from the reserialized copies. Four stage logs are portable; external
scheduler stdout/stderr paths are not bundled. These limitations are retained
alongside the verified application outcome.

## What the repair establishes

The [scientific review](scientific-review.json) uses descriptive arithmetic on
retained arrays only: no new references, fitting, model execution or tuning.
The smoke completed all 24 reference queries, 82 metric rows, two derivative
diagnostics and two finite-grid panel audits, with no unresolved reference or
ineligible fit. Selected ridge remains `1e-10`.

| Quantity | Native result | Interpretation |
| --- | --- | --- |
| Augmented condition number | 93.8237, previously 4.76694e10 | Avoidable numerical imbalance is repaired |
| Original design rank | 25/40 | Information and identifiability limits remain |
| Retained arrays versus job 2930 | All 96 arrays exactly equal | The repair replay used the same data and fitting design |
| Largest retained prediction change versus job 2930 | 1.14e-13 RMS | Objective-preserving scaling produces essentially the same predictions |
| Largest retained prediction difference versus revised cloud run | 7.46e-16 RMS | Native and cloud numerical results closely agree |

The exact-positive-ridge objective has a unique minimizer; the available data
still do not independently identify all 40 coefficient directions. Reference
agreement and ordinary FP64 optimality diagnostics are not rounding-certified
proofs of prediction accuracy.

## Endpoint and response products give different answers

The fresh classical response-corrected endpoint exactly reproduces the cached
zero-head error metrics. It includes actual path, residual and quadrature work.

| Horizon | Learned correction RMS | Reference uncertainty RMS | Classical call | Learned call |
| --- | --- | --- | --- | --- |
| 0.04 | 9.79e-17 | 6.03e-12 | 8.72 ms | 14.75 ms |
| 0.08 | 5.98e-15 | 1.19e-11 | 8.60 ms | 14.82 ms |

The learned endpoint effect is far below reference uncertainty. On these
inspected cases the learned branch adds work without demonstrated accuracy
benefit. The timing ratios, 1.69–1.72, describe single observations, not a
controlled performance benchmark. The classical corrected endpoint is the
appropriate current forward-solver control.

For response actions, the fitted model improves RMS error **9.84–11.05 times**
over the two-panel initialization. Its error remains **1.448–1.626 times**
the eight-panel classical error on all four diagnostic actions. However,
fitted calls took 3.70–4.14 ms versus 6.43–6.55 ms for eight panels in single
observations. Neither method therefore establishes accuracy/cost dominance:
a matched-tolerance, repeated complete-cost comparison is still required,
including the strongest qualifying classical work level.

The combined validation objective is **99.9999999998615% action error**.
Endpoint differences are numerically unresolved and effectively cannot affect
selection. Increasing their weight would amplify unresolved labels. Separate
the forward-solver and response-engine assessments before changing the loss.

Complete-map derivatives remain consistent with finite differences and the
independent variational solve, whose uncertainty is unqualified. The two
certified cubic-panel infinity-error bounds remain approximately 1.17e-7
(fitted) and 1.37e-7 (Strang). They do not certify the much smaller observed
endpoint errors, requested-tolerance acceptance, subsequent panels, continuum
accuracy or gradients.

## Resources and next decision

Worker telemetry contains 45 samples: 23 partial from process churn, 21 usable
CPU-percent intervals, zero missed samples and no observer errors. Maximum
observed summed RSS is 1,220,833,280 bytes, approximately 1.14 GiB. It is not
an allocation peak; shared pages can be counted repeatedly and departed
processes are absent. Raven's supervisor graphs have a separate scope. GPU
collection was disabled and this run establishes no GPU readiness.

The next task is [R7 representation identification](../../docs/RESPONSE_PRECONDITIONER_R7_DESIGN.md):
separate the best capacity of the five current operator branches from the
ability to identify their coefficients, generalize to unseen sources, and
deliver useful complete cost. Nonzero source times, withheld directions,
properly qualified oracles, uncertainty sensitivity and classical controls
must be declared before collecting a new bank. The design is not yet a
frozen executable scientific protocol. Historical Tangent Response
**A PASS / B STOP** and all later training/confirmation gates remain unchanged.

`audit_receipt.py` reproduces the integrity review against immutable Git
objects; `review_science.py` reproduces the descriptive comparisons. Their
arguments identify the extracted evidence, original ZIP, old native ZIP and
committed cloud receipt. `bundle-manifest.json` hashes the retained review.

To reproduce the read-only review from the repository root, choose a fresh
output directory and run:

```bash
python results/native-response-preconditioner-2936/extract_evidence.py \
  --output .runtime/native-preconditioner-2936-reproduction
python results/native-response-preconditioner-2936/audit_receipt.py \
  --repository . --evidence .runtime/native-preconditioner-2936-reproduction \
  --archive results/native-response-preconditioner-2936/launch-ba0ccf6660874b2f826b1d7fd9e25886-a1.zip \
  --extraction-manifest results/native-response-preconditioner-2936/extraction-manifest.json \
  --output .runtime/native-preconditioner-2936-reproduction/integrity-review.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  results/native-response-preconditioner-2936/review_science.py \
  --run .runtime/native-preconditioner-2936-reproduction \
  --cloud-smoke results/response-preconditioner-repair-20261015/smoke \
  --old-zip results/native-response-preconditioner-2930/response-preconditioner-2930-review-20261015T225818448985Z.zip \
  --output .runtime/native-preconditioner-2936-reproduction/scientific-review.json
```

The extractor checks the archive and retained payload hashes before writing
to a fresh project-contained directory. It restores only the reviewed regular
files. These commands acquire no teachers and submit no jobs.
