# Tangent Response implementation verification

Verified in the managed Linux CPU environment on **13 October 2026 UTC**.
These are the preserved initial delivery records for commit `9bbc5b8`.
The subsequent [native worker bootstrap repair](native-bootstrap-fix/README.md)
has its own verification record; it changes transport, not the numerical models,
datasets, scientific gates or observations recorded here.
This record accompanies the [implemented framework](../../docs/TANGENT_RESPONSE_RESEARCH.md),
[native runbook](../../docs/TANGENT_RESPONSE_RUNBOOK.md),
[analysis and terminology](../../docs/TANGENT_RESPONSE_ANALYSIS.md) and
[literature audit](../../docs/TANGENT_RESPONSE_LITERATURE.md).
The original uploaded handout is unchanged: SHA256
`10b181143de34f4e6698d6095c04196469a77c512b1f20273f7e1a520cc17545`.

## Checks and claims

| Observation | Result | Scope |
| --- | --- | --- |
| Full available CPU regression | **5,354 passed**, four optional checks skipped, 572 GPU cases deselected. | Mathematical, architecture, training, sealing, native-controller and existing-program regressions. |
| Final export CLI adapter | **92 affected checks passed**, no failures or skips. | Actual campaign/child argument forwarding, replay/full transport, failures and workflow controls after the full suite. |
| Final FD CPU rehearsal | **76/76 units sealed and independently verified**. | Reduced six-parent mechanism cohort and a separate six-parent fit cohort; two updates per learned arm. |
| Final Galerkin CPU rehearsal | **76/76 units sealed and independently verified**. | Separate same-grid physical norm/product contract; CPU implementation evidence. |
| Model deployment | Both arms output FP32, retain FP64 attribution, pass production fresh-process inference and complete 2/4/8-step autonomous rollouts. | CPU behavior under the initial delivery's recorded producer hash, including final rounding. |
| Raven launch contract | Actual `--check` and captured `--help` launches succeeded; drained logs and source/plan identities were checked. | Pinned Raven source CLI, local coordinator; **no Slurm science submitted**. |
| Portable reports | Both final exports were checksum-verified, relocated and read with Raven's default contracts/budgets. | Actual scientific citations, tables, logs and CPU/RSS streams; GPU samples are zero. |
| Required native CUDA roster | **29 skipped because CUDA is unavailable**. | Not native GPU verification; native workers reject missing/skipped required cases. |
| Native gates, learned benefit, solver speedup, confirmation | **Not established**. | CPU decisions cannot promote a native phase. |

The four optional skips cover an unchanged upstream Tower checkout and its
optional schema-validation dependency. The full suite ran before the final
isolated `replay` → `full` exporter adapter; all numerical/model code was already
present. The 92 affected checks and both final D rehearsals ran after that fix.
The [JUnit receipts](tangent-full-regression-delivery.xml),
[export/workflow receipt](tangent-compact-delivery.xml),
[CUDA-unavailable receipt](tangent-gpu-unavailable-delivery.xml) and
[structured verification record](verification.json) retain exact counts and
source identities. `python -m pip check` found no broken dependencies.

Initial delivery's final producer source fingerprint:
`aa12fea81573c181602ccf49a0aa3642df65a2c37374cc387c0ecf2134bb2dde`.
The working-tree runs record their base Git commit before publication; their
byte fingerprint identifies that initial delivery. No checkpoints or calibration
from these CPU runs can authorize native work.

## Final research records

The final runs are `tangent-response-cpu-fd-20261013-d` and
`tangent-response-cpu-galerkin-20261013-d`. Each completes two calibration fits
and two research fits, all receiving 2/2 updates. There is **one paired research
seed**, not four independent trials. Each mechanism and fit cohort has two
parents per train/validation/diagnostic split. Rehearsals reuse the prospectively
declared CPU seeds and are not additional independent scientific cohorts.

All reported gates remain `IMPLEMENTATION_ONLY`, with eligibility false.
Their separately retained toy outcomes are A `PASS`, B `UNRESOLVED`, C0 and
calibration `PASS`, C and D `STOP`. The unresolved compression means no qualified
fixed shared basis is available: its endpoint/rollout rows stay missing,
including all three unavailable 2/4/8-step rollouts. Both learned models complete
their rollout rows, but successful execution does not rescue negative held-out
gain or establish useful cost. Native Galerkin is a separate extension and cannot
borrow the primary FD mechanism gate.

The [FD summary](fd/run-summary.json) and [Galerkin summary](galerkin/run-summary.json)
contain all 76 manifest hashes, work/selection counts, dtype/rounding coverage,
fresh-process identities and reader counts. Their adjacent `decision.json`,
`fit_screen.json` and `performance_results.json` preserve exact outcomes.
The [FD reader audit](fd/raven-verification.json) and
[Galerkin reader audit](galerkin/raven-verification.json) retain every issue,
coverage count and missing-method observation. Actual CPU/RSS telemetry is
available; no native GPU, energy or monetary result is inferred.

## Inspect the portable Raven evidence

The final [FD zip](fd/tangent-response-raven.zip) and
[Galerkin zip](galerkin/tangent-response-raven.zip) contain relative links,
source-traced CSV/JSON, learning/physical/resource plots, standard worker/data
pages and omission inventories. The adjacent manifests record their original
byte hashes. Exact teachers, action arrays, checkpoints and journals remain in
the preserved source runs and are explicitly omitted from this review transport.

Extract into fresh project-contained directories:

```bash
unzip results/tangent-response-verification/fd/tangent-response-raven.zip \
  -d results/generated/tangent-response-fd-review
unzip results/tangent-response-verification/galerkin/tangent-response-raven.zip \
  -d results/generated/tangent-response-galerkin-review
```

Register the extracted root with Raven's `:project ABSOLUTE_PATH`, or open
`runs/<original-run-id>-analysis/outputs/index.html` below that root in a browser.
The overview's `artifact_projects.json` lists independent data projects;
`tangent_research.json` maps every source segment/table chunk. All default
8 MiB reader budgets and bounded project pages were checked after relocation.
The [captured native-launch records](native-launch-contract/run.json) and
[bibliographic receipt](literature-verification.json) are also retained here.

## Preserved failures and substantive implementation corrections

- CPU A completed all units but references/source qualification left its fit
  bank unresolved. Its original bytes remain unchanged. Qualification now
  measures the **actual analytic-path Gauss–Legendre source integral** rather
  than requiring an unused Hermite path approximation. The reduced CPU source
  refinement cap increased from 16 to 64 under a new protocol; the native cap
  was already 64. The `1e-10` uncertainty budget and scientific thresholds did
  not change.
- CPU B exposed a real deployment mismatch: the bridge decoder returned an FP64
  field. Strict fresh-process probes correctly failed. Final assembly now emits
  FP32, retains FP64 physical terms and final-rounding vectors, and preserves
  zero-time state/time derivatives. B remains preserved as failed probe evidence.
  Corrected C and final D probes succeed.
- The early large overview exceeded Raven's default 8 MiB contract budget.
  Lossless standard artifact pages now retain every source byte/table row while
  keeping each reader contract bounded; an independent forensic re-export of
  unchanged A verified this repair without rewriting its original science.
- Native public `compact --mode replay` now maps to the existing exporter's
  `full` mode for both campaign and child paths. No old export default changed.
- A shared checkpoint-reload check found mismatched parent/child backend flags.
  The child now receives the parent's deterministic/cuDNN execution policy while
  retaining IEEE FP32. Earlier six-argument internal calls remain compatible;
  existing research defaults and observations were preserved.

The structured record independently verifies the five earlier A/B/C campaigns
against their own sealed historical protocols/source hashes. None was deleted,
overwritten, merged into final results or promoted. Earlier Response programs,
native runs and pointers remain separate, and the Raven application is unchanged.

The recommended next numerical action is **the primary native FD mechanism
pilot** using the runbook. Its actual allocated audit and prospective A result
decide whether compression is warranted. Further learning, utility and
development remain conditional on literal eligible gates and measured native
calibration; there is no automatic confirmatory expansion.
