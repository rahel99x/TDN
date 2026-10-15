# Native response-preconditioner job 2930

**Native CPU implementation verified at `627de9f`. Scientific promotion remains
blocked.** The user supplied the original ZIP retained here. The run is
`launch-440e1ccfae26476ebf4b4dcaf4165ed0-a1`; it used an actual Fedora Slurm
allocation of four CPUs, 8 GiB and 20 minutes with Raven's launch contract.
It ran on 15 October 2026 at 20:00 UTC, with application exit code zero.
Raven elapsed time was about 45 seconds. Final independent Slurm accounting is
absent: the allocation snapshot is from startup and `scheduler_outcome` is
`unknown`. Neither is evidence of a failed application.

## Receipt integrity

[integrity-review.json](integrity-review.json) records the independently checked
54 ZIP entries, 30 stage-manifest hashes/sizes, 239 stage source hashes,
619-file native aggregate, 184 unique expected JUnit cases, 38 portable Raven
artifact paths and 80 research metric rows. All agree. No uploaded code was
executed. The [audit script](audit_receipt.py) reads data and immutable Git
objects at the recorded source revision, not the changing working tree.

Archive SHA256:
`242904177750e2c8fc5e6b2d6852980e6003aef94380207ebbfde76a98b3bfb5`.
The stage fingerprint is
`1b7c7ef69bd3e3948a378d64c37fd7c8d9a6fd4adf3349a3df71f49a52ca2983`.

The 184 mandatory checks passed with no failures, errors or skips. The smoke
completed 24 resolved reference queries, 80 metrics, two complete-model JVP
diagnostics and two single-panel residual audits. Native software was Python
3.13.13, PyTorch 2.10.0+cu126, NumPy 2.2.6 and SciPy 1.15.3. Computation was
explicitly CPU; an installed CUDA wheel does not establish GPU execution.

The native contract copy was reserialized by the worker. Original prepared
contract/profile bytes are absent, so their raw hashes cannot be rechecked
from this bundle. Four stage stdout/stderr logs are portable; scheduler stream
paths in `logs.json` refer to host files absent from the ZIP.

## Scientific interpretation

The [complete numerical review](scientific-review.json) preserves the exact
comparisons, costs, derivative limits and the next gate.

| Observation | Meaning |
| --- | --- |
| Four diagnostic action errors reduced roughly tenfold from initialization | Evidence that the fixed basis can improve these inspected actions |
| Eight-panel classical RMS error is 1.45–1.63 times smaller than fitted error on all four | No superiority over the accurate classical response |
| Fitted endpoint RMS 3.78e-13 / 7.34e-13 at time .04 / .08; reference uncertainty 6.03e-12 / 1.19e-11 | Endpoint benefit is unresolved |
| Original design rank 25/40; augmented condition 4.77e10 | Unique ridge minimizer does not mean identifiable coefficients; numerical scaling needs repair |
| Training endpoint MSE 3.46e-25 versus action MSE 1.31e-14 | The combined raw loss is effectively action-dominated on this tiny fixture |

All independent action queries start at zero. Consequently, five source-time
feature columns receive only tiny endpoint-defect signals. More encoder
capacity cannot supply missing training information. Augmented-column scaling
can remove avoidable numerical imbalance while preserving the objective; it
cannot increase the original rank or establish generalization.

Complete-map JVPs agree closely with finite differences and the independent
variational solve, whose uncertainty was not separately refined. The certified
finite-grid cubic-panel bounds are approximately 1.17e-7 for the fitted model
and 1.37e-7 for Strang. They do not certify the much smaller measured endpoint
errors, requested tolerance, continuum accuracy, later panels or derivatives.

Single recorded fitted calls took about 4 ms per action and 15–16 ms per
endpoint. Initialization timings reuse cached designs and cannot serve as
solver-cost comparisons. The next repair adds a fresh backbone-only corrected
endpoint and contracts scalar coefficients before spatial inference. Repeated
matched-tolerance timing, full internal transform counts and amortization remain
future work. No speedup claim follows from these observations.

## Resource telemetry and gate

The worker records 44 samples, 21 marked partial and 22 usable CPU-utilization
values; process-tree churn makes some intervals unavailable. Maximum observed
worker RSS is about 1.137 GiB, not a guaranteed process/allocation peak. Raven's
outer monitor sees approximately 9.8 MiB supervisor RSS and must not be read as
worker usage. GPU monitoring was disabled. Raw CSV/JSONL and coverage indexes
remain in the ZIP.

This completes the native R6 milestone at the recorded source. The
[bounded repair protocol](../../docs/RESPONSE_PRECONDITIONER_REPAIR_PROTOCOL_20261015.md)
permits implementation replay of the same inspected development fixture.
R7's broader scientific protocol, learned encoders, GPU work and confirmation
remain gated. Historical Tangent Response **A PASS / B STOP** is unchanged.
