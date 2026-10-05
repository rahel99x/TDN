# Paired neural replication study

This bounded study repeats the screened `reaction_clock` model against
`generic_mlp`, `residual_cnn_split`, `unet_split` and `fno_split`. It tests whether
the earlier accuracy/cost patterns survive new training seeds and fresh
diagnostic parents, and whether acceptable final-time rollouts also have
acceptable transient predictions. The configuration is fixed before execution;
there is no hyperparameter search or automatic campaign expansion.

The models are local adaptations with a maximum of 96 Adam updates each. This
experiment does not establish convergence or reproduce a published benchmark.
An advantage over these controls is evidence about this declared problem and
budget, rather than a claim to outperform every neural solver.

## Implementation validation

The [cloud CPU validation record](../results/neural-replication-validation.json)
contains the full fixed study, source/configuration fingerprints and per-seed
endpoints. The CPU suite passed 861 tests with one optional schema dependency
skip and 43 GPU tests deselected. The full numerical run completed in 487.60
seconds, accepted all 225 reference targets and verified all 73 sealed artifacts.
All fifteen selections completed; fourteen selected changed parameters after
training, while `generic_mlp` seed `74011` selected initialization. That distinction
remains explicit in every comparison.

Reaction-clock had some feasible rollout step on all 27 parents in each seed,
but only 5–6 parents passed every joint transient check. FNO-split passed those
transient checks on 9–10 parents. Clock had 48–54 passing rollout horizons out
of 108 per seed, versus the generic MLP's 51; broader parent coverage does not
mean better accuracy at every horizon. Its CPU timing advantages over the
spatial controls describe the declared post-hoc feasible frontier. These results
support further comparison and expose transient weaknesses; they establish no
general neural superiority or A100 timing result.

The fresh states also retain a smooth-case regression: clock loses to the generic
MLP on all 36 mixed-frequency rollout error comparisons in each seed. Although
clock lowers boundary-state errors relative to the generic control, no boundary
parent passes the joint transient endpoint. Those weaknesses should guide future
architecture work alongside the spatial controls' stronger transient results.

Tower exported all 3,240 frontier and 1,620 held-out rows, all fifteen endpoint
rows and twelve paired summaries, with zero reporting omissions. Unmodified
pinned Tower accepted all eighteen contract outputs and 1,803 live metric
records; independent JSON Schema checks validated 1,807 objects. No new CARC
submission or GPU execution occurred during this validation.

## Start on CARC

Use the existing verified project Python venv. Pull the current source before
starting a fresh workflow; completed experiments bind their source,
configuration, software and artifacts to that checkout.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only

# Preview; no numerical work, directory creation or scheduler call.
bash scripts/carc_neural_replication.sh

# Submit one bounded CPU job for correctness tests and the whole study.
bash scripts/carc_neural_replication.sh --submit

bash scripts/carc_neural_replication.sh status latest
bash scripts/carc_neural_replication.sh logs latest --lines 200
bash scripts/tower.sh list
bash scripts/tower.sh validate latest
bash scripts/tower.sh show latest

# Preserve and collect the complete review archive after completion.
bash scripts/carc_neural_replication.sh collect latest
```

Save the printed `TDN_RESEARCH_RUN_ID` and use it in place of `latest` if other
research jobs run concurrently. The research wrappers share a latest pointer.
Tower selects the most recently created report independently; use the exact
`TDN_TOWER_DIR` printed by the worker or research status to inspect this job.

The CPU request is **4 CPUs, 16 GiB host RAM and 30 minutes**, billed to
`anakano_81` for `aadaniel`. The complete numerical study has a shared
1,200-second budget. Logs, data, temporary files, caches and checkpoints remain
under `/home1/aadaniel/projects/TDN`; neither `/tmp` nor `/scratch1` is used.
The existing controller verifies account, identity, storage, venv, allocation,
source and artifact fingerprints. There is no pending-job cap, automatic GPU
successor or automatic requeue. Failed and completed runs are preserved.

For a tiny pipeline diagnostic, use a separate fresh workflow:

```bash
bash scripts/carc_neural_replication.sh --smoke --submit
```

Smoke uses an 8×8 grid, two optimizer updates per model, one timing repeat and
three diagonal regime/state strata per parent block. It retains all three
training seeds and all three fresh diagnostic blocks, but cannot substantiate
the full study's performance claims.

## Fixed protocol and interpretation

[`configs/neural-replication.yaml`](../configs/neural-replication.yaml) declares
protocol version 2, suite `neural-replication`. Its full configuration uses the
32×32 periodic reaction–diffusion problem, tolerance `0.002`, learning rate
`0.001`, width 16 for the spatial and clock models, and the generic MLP's
existing width-32 body. Parameter count and training walltime are reported;
equal update budgets do not imply equal capacity or compute.

The three model initialization seeds are `74011`, `74021`, `74031`, paired with
sample schedule seeds `74012`, `74022`, `74032`. Each five-family replicate
uses the same schedule and the same nine training and nine validation parents.
Normalization uses training initial states only, and checkpoint selection uses
validation only. All fifteen checkpoint selections finish before diagnostics
are evaluated; fresh diagnostic results cannot influence model selection.

There are **27 unique diagnostic parents**, comprising three fresh blocks of
the same three regimes and three state classes. All three training replicates
are evaluated on those same parents. This produces repeated measurements on
27 parents, **not 81 independent diagnostic observations**. The full shared
dataset has 45 parents and 225 accepted-reference targets when reference
generation succeeds. Failed acceptance or incomplete execution remains visible.

Read the two accuracy endpoints separately:

- Final-rollout accuracy at time `0.32`, tested with step sizes `0.04`, `0.08`,
  `0.16`, `0.32`; feasibility uses the RMS error plus reference uncertainty and
  the same tolerance and admissibility checks for every method.
- Transient one-step and two-step accuracy at held-out horizons `0.03`, `0.11`;
  an accurate final state can coexist with unacceptable intermediate states.

Per family and training seed, the full study declares 108 rollout cases and
54 held-out rows. Missing, invalid, initialization-selected and failed training
outcomes remain distinct. A robust transient parent passes both held-out
horizons for both the one-step and two-step endpoints. Comparison summaries
report paired accuracy/cost counts and speedup ranges per training seed, with
classical methods retained as practical controls. Repeated classical timings
do not create additional independent parents.

The fastest feasible step is selected from the declared diagnostic grid after
reference errors are known. Its cost comparison describes that grid's
accuracy/cost frontier; it is not a deployed adaptive step-selection algorithm.
Per-seed descriptive results do not imply statistical significance or authorize
a confirmatory campaign.

## Optional A100 inference timing

After a complete CPU study, request frozen-checkpoint inference explicitly.
Replace `CPU_RUN_ID` with its actual printed workflow ID:

```bash
# Preview the separate A100 request.
bash scripts/carc_neural_replication.sh benchmark CPU_RUN_ID

# Submit only when you want the optional timing run.
bash scripts/carc_neural_replication.sh benchmark CPU_RUN_ID --submit
bash scripts/carc_neural_replication.sh status latest
bash scripts/carc_neural_replication.sh logs latest --lines 200
bash scripts/carc_neural_replication.sh collect latest
```

The request is **one A100 40 GB, 4 CPUs, 16 GiB host RAM and 30 minutes**. GPU
preflight, real CUDA correctness tests and inference share the same allocated
`srun` task. Inference has a 600-second shared numerical budget and performs no
GPU training. It uses the sealed CPU dataset, protocol and selected checkpoints
with matching source, configuration and software.

Eligibility requires at least one training seed containing both a completed,
validation-selected `reaction_clock` checkpoint and a completed neural control
checkpoint with a positive selected optimizer step and changed parameters.
The selected checkpoints must be present in the sealed artifact manifest.
A trained clock from one seed and a trained baseline from another cannot satisfy
this gate. Classical headroom is not required for this neural comparison;
the original architecture workflow's headroom gate remains in force. The
eligibility gate establishes trained checkpoints for timing, not an accuracy
or efficiency win.

An old CPU run cannot be benchmarked after changing execution source. Preserve
it and start a fresh CPU study when source or scientific configuration changes.
Cloud CPU tests and fake scheduler tests provide no evidence of actual CARC
admission, GPU correctness or A100 performance.

## Artifacts and Tower

The root experiment retains `protocol.json`, `dataset.pt`, `references.json`,
`normalization.json`, `summary.json`, `replication.json` and its completed
artifact manifest. Each `replicates/seed-*/` directory holds the five training
records and selected checkpoints. Inference rows are split into
`blocks/block-*/` directories to keep each scientific table within the bounded
reporter's file limits. Every block records its training seed, sample schedule
seed and diagnostic block identity.

Tower's project-side exports retain seed and block identities in the training,
held-out, frontier and neural comparison tables. Replication endpoint and
comparison tables summarize the declared denominators and per-seed outcomes.
Use those tables alongside the original JSON artifacts, validation histories,
failures and runtime provenance. The project supplies this reporting; Tower
itself is unchanged. See [TOWER.md](TOWER.md) for exact report selection and
structural validation commands.

## Local/cloud CPU development

In a separate non-CARC checkout with the project `.venv` installed:

```bash
bash scripts/neural_replication_local.sh --smoke
bash scripts/neural_replication_local.sh --run-dir runs/neural-replication-review
```

The wrapper delegates to `research_local.sh`, keeps all storage in the checkout
and refuses CARC login-node use. Label these measurements local/cloud CPU.
