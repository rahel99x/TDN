# Implementation map and scientific limits

| Specification area | Code / evidence |
|---|---|
| Stable temporal reference | `reference/temporal_core.py`, `tests/test_temporal_core.py` preserve Appendix A/B; extended independent tests in `tests/test_models.py`. |
| Tier-A exact controls / anchors | `tdn/numerics/linear.py`, `tests/test_numerics.py`. |
| Discrete RD, subflows, teacher, invariants | `tdn/numerics/{operators,subflows,splitting,reference,invariants}.py`. |
| Local features and information access | `tdn/features/local.py`, `tdn/analysis/influence.py`; full-domain FFT rollout, radius-one feature support. |
| Primary / matched learned baselines | `tdn/models/temporal.py`, `tdn/models/solver.py`; no queried h in primary encoding. |
| Immutable physical parents / shards | `tdn/data/{dataset,generation,provenance}.py`; paired FP64 labels, checksums, quantization audit and atomic completion markers. |
| Training, full-domain 1/2/4-window losses | `tdn/train/loop.py`; train-only normalization, synchronous parent-balanced exact-replay sampler and cell-weighted microbatches. |
| Serialization / signals / committed resume | `tdn/train/checkpoint.py`, `tdn/runtime/signal_handling.py`, `tests/test_data_training.py`. |
| Numerical / temporal / locality gates | `tdn/analysis/workflow.py`; matrix-free JVP/VJP, finite-difference sweep, exact feature-equality pairs, temporal-fit controls labeled as teacher-informed oracles. |
| Full solver frontiers / failures | `tdn/analysis/{workflow,profiling,statistics}.py`; validation locks one policy per method, all diagnostic parents and sequence failures retained. |
| Allocation, venv, storage, dependencies | `scripts/`, `tdn/runtime/{storage,preflight}.py`, [CARC runbook](CARC.md). |
| FP32 → BF16 → compiler ladder | `tdn/runtime/calibration.py`; forward and gradient comparisons, compile-first-use memory, pointwise training, teacher/feature/FFT/full-domain backward measurements. A real A100 task is required. |
| Strict configuration and factual stages | `tdn/config.py`, `tdn/cli.py`; unknown fields rejected, completed artifacts preserved, paused signal exits75, stale prerequisite reports rejected. |

The Python module names differ slightly from the illustrative layout in the specification; the table provides the mapping. Optional branches are explicitly rejected rather than populated with placeholders.

Training starts in eager FP32. Derivative and composition loss mechanisms are not part of the initial loss; derivative validity and mixed-step composition are independently audited. Short rollout losses recompute features and encoding at each student state with gradients through the base solver. Full-grid evolution avoids an unproved patch-guard assumption. No worker prefetch is used; exact CPU resume is tested and CUDA determinism is a separate allocated check.

The scalar norm is cell-volume weighted and resolution independent. Teacher uncertainty, label subtraction, state quantization and teacher method/counts are stored separately. Scientific failure records include inadmissible and nonfinite states; there is no hidden physical-state clipping or removal of difficult parents.

The supplied development gate is empirical screening on declared probes. It is not a nonlinear error certificate. A failed G2/G4 means the protocol stops before expensive pilot expansion, even when the implementation tests pass. The smoke manifest explicitly permits tiny diagnostic runs to test the software without asserting scientific merit. G7 confirmation and optional G8 branches remain closed.
