# Response-preconditioner implementation verification — 15 October 2026

**CPU implementation verified; native execution pending.** This is the first
bounded implementation of the [standing research program](../../docs/TDN_RESEARCH_PROGRAM.md)
and its [dated architecture proposal](../../docs/TDN_ARCHITECTURE_PROPOSAL_20261015.md).
Historical Tangent Response A PASS/B STOP is unchanged. No scientific campaign,
native Slurm/CUDA run, learned encoder or fresh confirmation was performed.

The [receipt](verification.json) records exact commands, source fingerprint,
versions and limitations. The [runbook](../../docs/RESPONSE_PRECONDITIONER_RUNBOOK.md)
provides local and native Raven launch commands.

## Verification

- The actual local wrapper's mandatory verifier passed **178 cases**, with
  zero failures, errors or skips. See [JUnit](verification/verification.xml)
  and [log](verification/verification.log).
- The optional trusted-Raven compatibility suite passed **27 cases**, with
  zero skips, against Raven `c34f0892d7bddc685c5af10489ecb114fd6adab6`.
  See [JUnit](raven-compatibility.xml). It overlaps the mandatory roster;
  the two counts are not distinct tests. Scheduler identities in these tests
  are fixtures, not native execution evidence.
- The actual local wrapper's tiny smoke completed: two training parents,
  one validation parent, one diagnostic parent; 24 accepted reference queries,
  80 metric rows, two derivative diagnostics and two audited cubic panels.
  All matched-mean references were resolved. Selection froze before diagnostic
  generation; initialization and every fitted candidate remain recorded.
- Independently verified the complete terminal artifact inventories and exact
  source-before/source-after equality for both retained attempts. The final
  source fingerprint is `0ab5fd82b5f9fedfa9c3d008c304ebd597a825b0e4327a7a6d78734af1482751`.

Execution used the project Python 3.12.14 venv, PyTorch 2.10.0+cpu, NumPy
2.2.6 and SciPy 1.15.3. The fixed-feature model and teachers used explicit
FP64 arithmetic on initial fields first represented in FP32.

## Observations and limits

The validation-selected ridge was `1e-10`; its combined endpoint/action
validation MSE was approximately `9.15e-15`, versus `8.67e-13` for initialization.
Both fitted candidates passed their numerical optimality/nonregression checks.
The selected augmented system's condition number was approximately `4.77e10`:
a small objective gradient does not demonstrate well-identified coefficients.
The recorded gradient/gap uses ordinary FP64, not validated arithmetic.

The fitted response improved over the two-panel initialization on the four
diagnostic actions. The classical eight-panel response was more accurate on
all four. Fitted endpoints showed no resolved improvement over initialization:
their tiny differences were below reference uncertainty. This bank has only
one diagnostic parent and two short horizons; it cannot establish generalization
or scientific superiority.

The complete-model JVP was checked against finite differences and an independent
NumPy/SciPy variational solve. The latter has no separately refined uncertainty
estimate. Final records explicitly separate endpoint-reference acceptance from
this unqualified derivative diagnostic. The internal forcing response is not
the full model Jacobian.

The single-panel infinity-error upper bounds were approximately `1.17e-7`
for the fitted endpoint's cubic reconstruction and `1.37e-7` for one Strang
step. The full decimal strings, rational coefficients and assumptions are in
[audit.json](smoke/audit.json). These certify conservative bounds for the
specified exact-real finite FD equation, not tolerance acceptance, continuum
error, subsequent panels or gradients. They do not certify the much smaller
reference-comparison errors.

Single-call timings establish no speedup. `zero_readout` attribution timings
reuse cached designs and are explicitly labeled `cached_attribution_only`;
they are not complete solver timings. Explicit response counters do not
include all internal path/autograd transforms. Complete measured utility and
offline amortization remain open.

## Retained files

`verification/` and `smoke/` contain unmodified top-level outputs of the final
source-bound runs, including terminal hash manifests and full tiny NumPy replay
arrays. Original runtime paths in receipts are provenance; artifacts within
these directories use relative filenames. Load `arrays.npz` with
`numpy.load(..., allow_pickle=False)`.

`initial-metadata-review/` preserves the first passing implementation attempt.
Review found that derivative records inherited the endpoint's acceptance flag;
the subsequent change made the two reference statuses explicit and reran the
complete verifier and smoke with fresh roots. Neither original attempt nor
historical scientific evidence was overwritten. `raven-compatibility.xml`
records checks of the unchanged launcher files; no broader native claim follows.

`bundle-manifest.json` hashes this portable receipt bundle. The larger original
run directories remain in the checkout's ignored `runs/` tree. Native jobs,
GPU work, adaptive rollout and scientific promotion remain pending their gates.
