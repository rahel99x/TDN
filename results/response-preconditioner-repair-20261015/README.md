# Bounded implementation repair after native job 2930

**202 mandatory CPU checks passed; revised native execution pending.** This
implements the [prospective repair contract](../../docs/RESPONSE_PRECONDITIONER_REPAIR_PROTOCOL_20261015.md)
following the [native review](../native-response-preconditioner-2930/README.md).
The [standing program](../../docs/TDN_RESEARCH_PROGRAM.md) remains the maintained
research reference. The four parents remain inspected development fixtures.

## Changes and evidence

- Augmented-column ridge scaling preserves the original positive-ridge
  objective and penalty. The [retained native matrix replay](retained-fit-replay.json)
  agrees with an independent original-coordinate SVD solution; selected
  augmented condition improves from 4.76694e10 to **93.8237**. Original design
  rank remains **25/40**. Maximum retained-query prediction change is
  **1.14e-13 RMS**. The larger ridge's condition improves to 1.17144.
- Inference contracts eight scalar features into five branch coefficients
  before multiplying spatial actions. Training still uses the 40-column
  design. Full state/source/time/physics JVPs and readout gradients agree with
  that design, including batched zero-time and zero-readout cases.
- `classical_corrected_endpoint` computes a fresh ordered-response/GL3 endpoint
  without encoding or learned branches. Its two diagnostic rows have
  `complete_uncached_call` timing; cached zero-head attribution remains labeled
  separately. Logical tensor counters are not peak-memory measurements.

The [verification receipt](verification.json) records exact commands, summaries,
source identity and limitations. The actual local wrapper passed **202 cases**,
zero failures/errors/skips, and completed the tiny smoke: 24 accepted reference
queries, 82 metrics, two derivative diagnostics and two finite-grid panel
audits. All fitted candidates remained eligible. All **33 overlapping** optional
Raven launcher tests passed against `c34f0892d7bddc685c5af10489ecb114fd6adab6`;
these counts must not be added as distinct tests. The source fingerprint is
`c5adaf6a929bd583efe0beb92de65b723b0c2492fb67836ddaa86efdbb6b1dbe`.
All 30 stage-manifest entries and source-before/source-after equality were
checked independently after execution.

The revised smoke still selects ridge `1e-10`. Fitted diagnostic action error
is 1.448–1.626 times the eight-panel classical error on the four inspected
actions. Endpoint benefit remains below reference uncertainty. Better scaling
does not resolve deficient source-time coverage, provide new information,
establish identifiability, or justify a learned encoder. Single-call timings,
partial operation counters and a tiny bank do not establish speedup.

## Reproduction and next step

Use fresh roots with the [runbook](../../docs/RESPONSE_PRECONDITIONER_RUNBOOK.md).
Prepare a fresh contract and run its native Raven `--mode all` command after
updating the native checkout. Job 2930 verified the previous source only.
There is no automatic scientific, training or GPU successor.

`verification/` and `smoke/` retain unmodified top-level stage artifacts,
including the source/protocol snapshots, logs, JUnit and replay arrays.
`prospective-freeze.json` records the contract hash before the revised smoke.
`replay_retained_fit.py` re-solves only the original ZIP's training matrix,
without acquiring teachers. It preserves existing receipts by refusing to
overwrite its output; use `--output runs/response-preconditioner-retained-replay-new.json` for a
new development replay while preserving this receipt.

The focused unit receipts overlap the mandatory roster. The first fitting-test
receipt is retained as `initial-test-rounding-failure.xml`: one test assertion
incorrectly required bit-exact equality of a computed FP64 penalty to one.
It was corrected to a rounding tolerance before the successful focused and
full runs; no objective, science threshold or solver check was weakened.

All current outcomes remain **IMPLEMENTATION_ONLY**. Historical Tangent Response
**A PASS / B STOP** is unchanged. A future R7 scientific study still requires
its own frozen source-time/direction/oracle/control/cost protocol and gates.

**Terms:** augmented system = data rows plus ridge-penalty rows; condition
number = largest/smallest singular value of that numerical system; design
rank = resolved independent data-column directions; RMS = root mean square;
JVP = Jacobian-vector product of the complete implemented map; GL3 = three-node
Gauss–Legendre quadrature. A unique regularized minimizer does not imply all
coefficients can be identified from the data.
