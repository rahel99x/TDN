# Training-free mechanism test program

This is the first stage of the architecture pivot: small, deterministic tests
that can eliminate weak mechanisms before another neural training campaign.
It covers all eight proposed mechanism areas, with six coordinate/gate
combinations and four temporal representations. It does not change the trained
TDN model, reproduce the original FNO paper, or select a winning architecture.
The full protocol contains 600 cases: 397 coordinate, 165 temporal and 38
structural probes. The smaller smoke contains 278 cases. Cases and scalar
metric rows are not independent statistical replicates.

The [recorded cloud CPU validation](../results/mechanism-audit-validation.json)
completed all 600 cases: 155 passed controls, 416 observations and 29 expected
limitations. All 10,173 scalar metrics exported with zero omissions and passed
native Tower validation. This is development evidence; the new workflow has
not yet run on CARC.

## Run on CARC

Use the existing verified project venv. The launcher performs identity,
account, allocation, source/configuration and venv-lock checks. It submits one
CPU job with **4 CPUs, 16 GiB and 30 minutes**, billed to `anakano_81`; the
numerical audit itself has a **1,200-second cap** and uses one numerical thread.
There is no GPU successor, automatic training, pending-job cap or automatic
resubmission.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main

# Inspect the planned job, then submit it once.
bash scripts/carc_mechanisms.sh
bash scripts/carc_mechanisms.sh --submit

# These commands do not submit more jobs.
bash scripts/carc_mechanisms.sh status latest
bash scripts/carc_mechanisms.sh logs latest --lines 200
bash scripts/carc_mechanisms.sh collect latest
```

Submission prints a `carc-mechanisms-...` run ID. Use that ID instead of
`latest` when other research workflows are active: the research workflows share
the same latest pointer. The worker first runs focused mechanism correctness
tests, then executes the full frozen audit. `--smoke` selects a smaller declared
case set and a 180-second numerical cap; it is not needed before this small
full CPU audit. Preserve a failed run and submit a fresh ID after fixing its
cause. Completed artifacts cannot be overwritten.

All project, cache, temporary, test and run files stay beneath
`/home1/aadaniel/projects/TDN`. Do not source scripts manually or run the
numerical entrypoint on a CARC login node.

For development in a separate non-CARC checkout, the equivalent local CPU run
is below. Run once with a fresh directory name; this uses that checkout's venv
and labels the evidence as local CPU.

```bash
.venv/bin/python scripts/mechanism_audit.py --config configs/mechanism-audit.yaml \
  --run-dir runs/mechanism-audit-local-001
```

## What is tested

| Mechanism | Current tests and combinations | What they can establish |
|---|---|---|
| Diffusion-conditioned temporal response | Stable FP32/FP64 phi functions; independent quadrature and matrix exponential; exact first/second variations; scalar, transported, output-phi and reaction/pair-informed two-coefficient fits | Implementation identities and temporal representation limitations on fixed discrete problems |
| Nonlocal spatial context | Remote perturbations with local versus global gates; required correction coordinates computed after accounting for each physical diffusion intermediate | Whether a local gate excludes a needed correction despite the already-global physical base |
| Bounded output coordinates | Clock/local, post-capacity/local, clock/global, post-capacity/global, pre-capacity/global, additive/global | Exact limits, boundary reachability, gradients, bounds, cubic-order obstructions and placement effects |
| Anti-aliasing / resolution | Analytic sinusoid squares, coarse nonlinear evaluation versus oversampling/filtering, restriction/interpolation and translation checks | Aliasing and resampling identities; not trained cross-resolution generalization |
| Composition and PDE consistency | Unequal split compositions, coupled references, identity and diffusion-only negative controls | Why exact composition alone can reward the wrong dynamics |
| Spectral factorization | Additive-axis and rank-1/rank-3/dense linear fits to separable and oblique responses | Fixed linear representation/cost tradeoffs; not the expressivity of a deep F-FNO |
| Equation-specific physics | RD source balance, shared-face flux telescoping and compatible divergence projection | Algebraic controls; not an implemented advection or Navier–Stokes benchmark |
| Adaptive error control | Step-doubling asymptotics, exact-but-wrong-generator shared-bias control and counted scalar adaptation | Estimator failure modes and actual accepted/rejected work; not certified global error or a neural controller |

Coordinate probes cross diffusion and reaction independently. Same-grid
refined FP64 teachers record acceptance and uncertainty; a rejected reference
is inconclusive. Temporal fits use disjoint fitting, validation and test times.
Validation/test labels never fit coefficients or select a model. Sum, doubled,
DC and difference-frequency interactions distinguish input-pair information
from output frequency alone. Independent finite-amplitude PDE checks verify
the perturbation coefficients and amplitude scaling.

The coordinate combinations use fixed coefficients, not trained heads. Oracle
temporal fits are per-case representation diagnostics with extra information
explicitly recorded for the reaction/pair-aware basis. Better oracle fit does
not establish generalization or a speed advantage.

## Read the results in Tower

The allocated worker prints `TDN research: Tower report: ...`. Copy that exact
directory and use it as `REPORT` below; it owns the job's live metrics, logs and
analytics. Tower itself is unchanged.

```bash
REPORT=/home1/aadaniel/projects/TDN/runs/REPLACE_RUN_ID/tower/REPLACE_REPORT_ID
bash scripts/tower.sh validate "$REPORT"
bash scripts/tower.sh show "$REPORT"
```

The metrics path is `$REPORT/metrics.jsonl`. The detailed table is
`$REPORT/outputs/mechanisms.csv`, also registered as `analytics.mechanisms` in
the log catalog. Each scalar retains panel, case, mechanism, test, variant,
kind, outcome, inputs and an exact JSON pointer to the scientific record.
The optional table leaves existing Tower contracts compatible with old runs.
Its dedicated export limit is 12,000 scalar rows and 8 MiB; existing tables
keep their previous limits. Overflows remain explicit in the artifact report.

Scientific artifacts live in `runs/RUN_ID/experiment/`:

- `protocol.json`: frozen case IDs, parameters, time splits, source and software.
- `config.json`: effective bounded configuration.
- `coordinates.json`, `temporal.json`, `structure.json`: panel results and limitations.
- `mechanism-audit.json`: canonical combined case/metric records.
- `summary.json` / `summary.txt`: coverage, outcomes, runtime and scope.
- `manifest.json` / `COMPLETED`: hashes binding all seven scientific JSON artifacts.

`PASS` means a declared control met its threshold. `OBSERVED` records a probe
without a win claim. `EXPECTED_LIMITATION` confirms an intended counterexample.
`FAIL` prevents completion even for a negative control. `INCONCLUSIVE`, an
interruption or budget exhaustion also prevents sealing. Completed panels are
retained; an interrupted panel lists its unreported case IDs. Partial rows
inside a panel that has not returned are not checkpointed.

Collect the run after completion and upload the resulting review archive. That
includes failures as well as successful evidence, and excludes pytest work
directories. Native Tower validation can also be requested using the existing
`tower.sh validate "$REPORT" --native` option when Tower is installed.

## Decisions after this stage

Inspect per-case errors and controls before selecting a trained experiment.
In particular, output-frequency phi filters can fail to represent nonlinear
pair interactions; interval-preserving capacity updates can distort the cubic
coefficient when available capacity is also cubic-sized. Neither mechanism is
automatically an improvement.

The next stage remains separate: matched training of coordinate/context
combinations, then a temporal × context panel with identical data and budgets.
Only successful candidates should proceed to resolution-transfer tests,
factorized spatial encoders, composition/PDE objectives and full adaptive
complete-solve comparisons. Independent larger PDE benchmarks and faithful FNO
baselines are still required before claiming an advantage over that paper.

Background implementations informing the probes include the authors'
[FNO guide](https://neuraloperator.github.io/dev/theory_guide/fno.html),
[F-FNO](https://github.com/alasdairtran/fourierflow),
[CNO](https://github.com/camlab-ethz/ConvolutionalNeuralOperator), and
[PINO](https://github.com/neuraloperator/physics_informed). These diagnostics
are TDN-specific; they are not reproductions of those papers' benchmark results.
