# Response-preconditioner bounded implementation repair

This prospective contract follows native job **2930** at source `627de9f`.
It is frozen before running the revised implementation smoke. It authorizes
software repair and replay of the existing tiny development fixture only.
It does not open R7's scientific campaign, a learned encoder or GPU training.
The standing entry point remains [TDN_RESEARCH_PROGRAM.md](TDN_RESEARCH_PROGRAM.md).

## Question and allowed changes

Can we preserve the declared positive-ridge objective and full differentiable
response while removing avoidable numerical scaling and inference overhead?

1. Equilibrate the **augmented** ridge columns using
   `s_j = hypot(||sqrt(W) A[:,j]||_2, sqrt(lambda))`. Solve in `x = s*w`,
   retaining `sqrt(lambda)/s` in the penalty block and every singular value.
   This changes coordinates, not physical row weights, regularization or the
   unique exact-arithmetic minimizer. Compare against an independent solve
   of the original objective, including tiny and zero columns.
2. At each state/physics/time query, contract the eight scalar features with
   the readout into five branch coefficients before applying them to spatial
   actions. Keep the 40-column design for fitting and parity checks. Preserve
   all state, time, physics and readout derivatives; do not detach features.
3. Time an actual fresh classical response-corrected endpoint that computes
   the ordered backbone and GL3 defect quadrature without learned branches.
   Preserve the cached zero-head attribution with its original scope label.

## Fixed information and checks

Keep the existing four parent seeds, 5-by-5 grid, two horizons, two directions,
FP32 initial data, FP64 arithmetic, reference acceptance, parent weights,
ridge menu and validation-only selection. These parents have been inspected:
every replay is development evidence, never fresh confirmation. The new
protocol has a distinct version and fresh output roots; job 2930 is immutable.

Required software evidence is the entire mandatory CPU roster, independent
ridge-objective checks, affine-design/primal/full-derivative parity, batched
time and physics checks, baseline independence, source guards and native
launcher compatibility. Existing numerical tolerances remain in force.
Retain failed attempts. The revised smoke must produce the same 24 reference
queries, all resolved, two derivative diagnostics and the existing audit scope;
adding two diagnostic baseline rows raises the metric inventory from 80 to 82.
Inventory counts establish execution coverage, not model quality.

Report augmented conditioning alongside the original design rank, numerical
objective/gap and prediction differences. Better equilibration cannot create
missing information. Five source-time columns are poorly exercised when all
independent actions begin at zero; this replay does not repair that coverage.
Preserve the eight-panel classical action comparison and reference uncertainty
on endpoint differences, whether the new outcome is favorable or unfavorable.

## Costs and gate

Each displayed inference time must state whether it is a complete fresh call
or a cached attribution. Single-call timings remain diagnostic; no speedup,
memory saving in bytes, or matched-tolerance utility claim is permitted.
Work counters describe explicit branches and path calls; uninstrumented
internal anchor/autograd transforms remain a stated limitation.

Passing these checks permits a fresh **native CPU implementation verification**
with the existing 4-CPU, 8-GiB, 20-minute Raven contract. It does not authorize
larger numerical work. After its review, R7 still needs a separately frozen
bounded protocol specifying nonzero source times, withheld directions,
per-state five-coefficient oracles, independently verified classical controls,
reference-sized perturbation sensitivity and complete repeated cost accounting.
No oracle may become a deployable competitor or use diagnostic truth to tune
the model. Historical Tangent Response **A PASS / B STOP** remains unchanged.
