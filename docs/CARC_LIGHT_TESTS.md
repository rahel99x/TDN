# Light CPU tests on CARC

Run from `/home1/aadaniel/projects/TDN` as `aadaniel`. This workflow charges
`anakano_81` and keeps its configuration, logs, reports, caches and temporary
files inside that project. It uses the existing verified Python `.venv`.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main
bash scripts/carc_light_tests.sh
bash scripts/carc_light_tests.sh --submit
bash scripts/carc_status.sh
bash scripts/carc.sh logs latest --lines 200
```

The first launcher call previews without creating run files or submitting work.
The second queues exactly one job on the live-validated CPU partition `main`:
**2 CPUs, 8 GiB host RAM and 15 minutes**, with no GPU request or dependency.
The allocated `srun` task performs these stages sequentially:

1. Verify and reuse the existing venv, without installing packages.
2. Run focused regression tests for the light screen, temporal oracle and CLI,
   plus the existing analysis tests.
3. Collect the fixed numerical and temporal screening results.

Failed regression tests stop the screen. A fully collected screen can complete
with failed scientific checks; those failures are results for review. There is
no dataset generation, training, pilot, automatic retry or pending-job limit.
The shared-checkout phase lock prevents concurrent modification of this venv;
it does not limit submissions from other projects.

Your completed smoke already verified the venv. If it is missing or incompatible,
the light job stops and preserves it. Run the separate allocated setup workflow
first if installation is needed:

```bash
bash scripts/carc.sh setup
bash scripts/carc.sh setup --submit
```

## Fixed scientific screen

The launcher pins `configs/carc-light.yaml`, including the unchanged proposed
tolerance `0.002` and required headroom. Before teacher computation, `plan.json`
records the initial arrays, seeds, physical regimes, horizons, caps and decisions.

| Regime | Diffusivity | Reaction rate | Initial states |
|---|---:|---:|---|
| Baseline | 0.01 | 2 | Seeded mixed-frequency and bounded random |
| Intermediate | 0.03 | 6 | The same two states |
| Stiff | 0.10 | 12 | The same two states |

All six cases use an 8×8 periodic grid, FP64 states and refined FP64 references.
The two states are reused across regimes; these are not six independent parents.
References use at most 324 substeps at the finest level. The numerical screen
has a ten-minute budget within the job's fifteen-minute allocation.

Headroom requires both split and Richardson rollout errors, after subtracting
reference uncertainty, to exceed `0.002` at `h=0.32`, `T=0.64`. Costs include
first use, one warmup and three steady-state repeats of each complete solver.

Temporal fits use eight fitting horizons and four amplitude columns, followed
by six disjoint held-out horizons. Held-out labels never enter the rate optimizer.
The fixed and learned temporal models are compared with polynomial and rational
controls. Learned-rate evidence additionally requires successful, informative,
locally identifiable fitting; the rate optimizer has a 40-evaluation budget.
Reported residual calls include Jacobian work and can exceed 40.

The temporal check requires a 20% held-out improvement after accounting for
reference uncertainty. The full audit now also uses more fitting observations
than temporal coefficients; a saturated interpolation cannot count as learned
rate evidence. Configurations and source fingerprints are checked during execution.

A joint passing case is only a candidate for a full G1–G4 audit and a short,
representative CUDA comparison. This screen does not fit a deployable encoder,
test G1/G3, establish a TDN efficiency advantage or authorize a larger pilot.
Every failed case stays in the output; no tolerance or gate is weakened.

The recorded [cloud CPU validation](../results/carc-light-validation.json)
completed the numerical screen in 5.59 seconds (7.67 seconds including CLI
startup). All six cases had accepted references; five met the representation
comparison, and none passed headroom. Richardson already stayed below `0.002`
in every case, leaving no joint candidate. These measurements use the cloud CPU
and do not establish CARC queue time, runtime or A100 behavior.

## Reports and recovery

The launcher prints the fresh `carc-light-...` run identifier. Use it instead of
`latest` if you later submit another workflow. Stage logs print the concise
scientific summary. The run directory contains:

- `workflow.json`, `config.yaml` and `jobs.json`: submission and fingerprints.
- `logs/cpu-JOB_ID.out` and `.err`: regression results and scientific summary.
- `light-screen/plan.json`: the protocol written before numerical measurements.
- `light-screen/cases/*.json`: every declared case and its diagnostics.
- `light-screen/summary.json` and `summary.txt`: combined results for review.
- `light-screen/stage.json`: actual environment, source identity and completion.

To collect the completed light reports for this chat, paste the run identifier
printed by the launcher into the variable below. The archive stays in the project.

```bash
TDN_LIGHT_RUN='carc-light-REPLACE_WITH_PRINTED_RUN_ID'
tar -czf "runs/${TDN_LIGHT_RUN}-review.tar.gz" \
  "runs/${TDN_LIGHT_RUN}/workflow.json" \
  "runs/${TDN_LIGHT_RUN}/config.yaml" \
  "runs/${TDN_LIGHT_RUN}/jobs.json" \
  "runs/${TDN_LIGHT_RUN}/logs" \
  "runs/${TDN_LIGHT_RUN}/state" \
  "runs/${TDN_LIGHT_RUN}/light-screen"
```

After an interruption or a code update, preserve the old evidence and start a
fresh light run:

```bash
bash scripts/carc.sh restart carc-light-REPLACE_WITH_PRINTED_RUN_ID
bash scripts/carc.sh restart carc-light-REPLACE_WITH_PRINTED_RUN_ID --submit
```

Restart keeps the light profile and verifies ownership before canceling pending
jobs. It does not cancel a running job. Light screening has no resumable training
checkpoint; a signal or timeout retains partial output with no completion proof.
