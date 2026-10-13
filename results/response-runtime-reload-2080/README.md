# Response checkpoint reload incident: job 2080

The user supplied the stderr and [native receipt](native-receipt.json) from
`response-diagnostic-smoke-20261013T032059-fd157c33d389`,
`trial-9b2cb697421ddc9a-000`. The JSON is transcribed from the user's paste;
the native tensor files, selected checkpoint and allocation were not available
in this cloud checkout. Its reported device is CUDA. This incident is separate
from the [Tangent worker bootstrap repair](../tangent-response-verification/native-bootstrap-fix/README.md).

## Finding

The parent and child runtime dictionaries have exactly one difference: only
the child includes `deterministic_warn_only: false`. Every shared runtime value
is identical. Exact dictionary comparison therefore produces `runtime_equal:
false` even though the receipt reports exactly equal predictions, a maximum
output difference of zero, identical model state and identical checkpoint bytes.

| Receipt field | Meaning | Reported result |
| --- | --- | --- |
| `exact_equal` / `maximum_difference` | Exact tensor output equality / largest absolute output difference for this probe. | `true` / `0.0` |
| `state_equal` | Equality of loaded model-state digests. | `true` |
| `checkpoint_equal` | Equality of selected checkpoint-file digests. | `true` |
| `runtime_equal` | Exact equality of the recorded precision, thread and deterministic settings. | `false`, due to unequal dictionary keys |
| `deterministic_warn_only` | Whether PyTorch warns rather than raises when requested deterministic algorithms are unavailable. | Recorded only by the child, as `false` |

The parent matches the older runtime-recording schema; the child matches the
schema delivered in `9bbc5b8`. The user subsequently confirmed that the checkout
was updated during the running jobs. This explains the mixed versions: an
imported parent keeps its old implementation, while a new interpreter reads the
newer files. The receipt contains no source fingerprints that can reconstruct
the exact pair of source trees.

There is no observed output discrepancy in this probe and no evidence here of
a formula, model-architecture, loss or numerical-threshold failure. This single
probe does not establish training success, scientific benefit or native-suite
readiness. The original attempt remains **FAIL**: a missing parent setting is
unrecorded, and must not be filled with an assumed default to grant a pass.

## Repair and verification

The Response engine now passes its frozen entry source fingerprint to training.
Reload checks it before loading the selected checkpoint, before launching the
child and after the child returns. The child independently checks it before
loading and after inference, and records its actual source fingerprint. A source
change raises an explicit `SourceChangedError`; all exact output, runtime, state,
checkpoint and source comparisons must pass. The existing final stage and
recovery source checks remain in force. Both historical internal child CLI forms
remain accepted. No missing metadata default or tolerance relaxation was added.

All **104 focused CPU checks passed**, with no failures, errors or skips:
reload, training, CLI, dependency-order and workflow checks. The tests exercise
real child interpreters, stale-source rejection before checkpoint loading,
engine-to-training source binding, successful exact source/runtime/output parity,
and rejection of output, state, runtime and source-receipt corruption. See the
[JUnit receipt](source-guard-tests.xml) and [verification record](verification.json).
No numerical campaign or native CUDA rerun was performed for this repair.

## Native recovery

Keep the original run, logs, journal and receipt. Do not invoke the probe again
inside that original directory: it writes new reload files. Leave the checkout
and venv unchanged while any related workers or coordinator remain active.
After they are terminal, pull the fix and launch a fresh campaign with new IDs
and calibration. Updated files do not replace code already imported by a
running parent. Source-changed attempts cannot use exact-source recovery.

Use the [Response diagnostic runbook](../../docs/RESPONSE_DIAGNOSTIC_RUNBOOK.md)
for the older experiment or the [Tangent native runbook](../../docs/TANGENT_RESPONSE_RUNBOOK.md)
for the staged program. Their independent evidence gates remain required.
