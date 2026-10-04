# Matched neural architecture benchmarks

This bounded experiment compares TDN against a residual CNN, U-Net and Fourier
neural operator (FNO) on the existing periodic reaction–diffusion problem. Each
backbone has a direct learned time-stepper and a hybrid corrector around the
same Strang step used by TDN. A generic time-conditioned MLP adds a control
for TDN's structured temporal representation. These comparisons separate the
contribution of the physical step from the learned architecture. Classical methods remain practical
accuracy/cost controls; a neural architecture claim uses the declared neural
competitors as its primary comparison.

These implementations are problem-specific adaptations of established model
families. They do not reproduce a published benchmark or establish performance
against every neural PDE solver. The nine training parents and 96 optimizer
steps per family form an initial screen, not a convergence study.

The spatial backbones follow the residual-block idea in
[He et al.](https://arxiv.org/abs/1512.03385), the encoder/decoder skip structure in
[Ronneberger et al.](https://arxiv.org/abs/1505.04597), and the spectral/local
operator blocks in [Li et al.](https://arxiv.org/abs/2010.08895). Their periodic
boundary handling, physical feature inputs and bounded output heads are
adaptations for this experiment, rather than the original published models.

## Start the complete CPU screen on CARC

Use the existing verified project venv. No package installation is performed by
these scripts. After pulling code, start a fresh CPU workflow: source hashes
bind the resulting checkpoints, reports and optional GPU benchmark to that
checkout. An old CPU workflow cannot be benchmarked against changed source.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only

# Preview: no allocation or run directory is created.
bash scripts/carc_neural_benchmarks.sh

# Submit one bounded CPU job, including correctness tests and all families.
bash scripts/carc_neural_benchmarks.sh --submit

# Inspect the shared research workflow pointer.
bash scripts/carc_neural_benchmarks.sh status latest
bash scripts/carc_neural_benchmarks.sh logs latest --lines 200

# Once the worker starts, inspect its Tower report.
bash scripts/tower.sh list
bash scripts/tower.sh validate latest
bash scripts/tower.sh show latest

# After completion, collect the review archive printed by this command.
bash scripts/carc_neural_benchmarks.sh collect latest
```

Save the printed `TDN_RESEARCH_RUN_ID`; use that explicit ID instead of `latest`
when several research jobs are submitted. These wrappers share
`carc_research.sh`'s research pointer, including earlier architecture workflows.
Tower's `latest` selects the most recently created report, which can belong to
a different simultaneous job; use the exact `TDN_TOWER_DIR` printed by the
worker or research status command to select a particular report.

The CPU request is **4 CPUs, 16 GiB host RAM and 30 minutes**, billed to
`anakano_81` for `aadaniel`. Numerical work has a 1,200-second limit. All files,
including logs, checkpoints, temporary files and caches, stay under
`/home1/aadaniel/projects/TDN`; there is no `/tmp` or `/scratch1` use. Account,
allocation, venv, storage, source and artifact checks remain owned by the
existing controller. There is no pending-job cap or automatic GPU successor.
Completed and failed runs are preserved.

For a tiny installation diagnostic, use a fresh run:

```bash
bash scripts/carc_neural_benchmarks.sh --smoke --submit
```

Smoke reduces the problem and optimizer budget; it cannot substantiate neural
performance claims. A custom project-local config is accepted with
`--config PATH`; changes to the declared protocol must be reported explicitly.

## Models and matching rules

The default [`configs/neural-benchmarks.yaml`](../configs/neural-benchmarks.yaml)
retains all eight TDN research/control families and adds seven neural baselines:

| Family | Learned map | Physical step |
|---|---|---|
| `residual_cnn` | Residual convolutional backbone | None |
| `unet` | Multiscale U-Net backbone | None |
| `fno` | Fourier spectral backbone | None |
| `residual_cnn_split` | Same residual CNN capacity as its direct control | Strang |
| `unet_split` | Same U-Net capacity as its direct control | Strang |
| `fno_split` | Same FNO capacity as its direct control | Strang |
| `generic_mlp` | Generic time-conditioned MLP correction | Strang |

The six spatial baselines receive the same twelve physical-state features and
the channel `h/t_ref`. Their direct controls use the input state as the base
and learn the raw increment `U_ref * (h/t_ref) * network(features, h)` without
a physical Strang step. Hybrid controls use Strang as the base and the raw
increment `U_ref * (h/t_ref)^3 * network(features, h)`. Both apply the same
bounded signed-capacity output map `base + m*delta/(m+abs(delta))`, where `m`
is the available capacity toward the corresponding boundary. This head changes
the raw learned increment; it is not post-hoc clipping. The same full-rollout
checks reject invalid or nonfinite states for every model, and their failures
remain visible.

The original `temporal_mlp` is a structured TDN control: its time-independent
encoder predicts amplitudes and rates for exponential temporal modes. The new
`generic_mlp` instead receives `h` as an input and learns its time dependence
directly, adding `h^3 * network(features, h)` to Strang without exponential-mode
constraints. It uses the existing generic MLP body with width 32 and two hidden
layers. Its raw additive correction has different bounds from the six spatial
baselines; comparisons retain actual admissibility failures.

Spatial baselines and the new TDN research models use width 16. The original
`temporal_mlp` retains its width-32, two-hidden-layer, four-mode body, matching
the generic MLP body width and depth. Every family uses FP32, a maximum budget
of 96 optimizer steps and learning rate `0.001`. Numerical failures can stop
training earlier and remain recorded.
The common data protocol has 27 parents on a periodic 32×32 grid: nine train,
nine validation and nine diagnostic parents spanning the declared regimes and
state types. Checkpoints are selected on validation parents only. Diagnostics
use the same held-out horizons (`0.03`, `0.11`), final rollout time `0.32`, and
step sizes (`0.04`, `0.08`, `0.16`, `0.32`). Accepted coupled references use
FP64. The tolerance remains `0.002`.

Equal data and optimizer updates are not equal parameter counts, FLOPs or
training walltime. Parameter counts and training time are recorded so the
comparison can be interpreted accordingly. Width 16 also does not imply equal
capacity across these architectures. This first screen needs repeated seeds
and adequate convergence checks before a strong superiority claim.

Review one-step and held-out errors, feasible rollouts, admissibility failures,
validation-selected checkpoints and whole-rollout inference cost together.
Record training cost separately from inference cost, and include failed or
initialization-selected models rather than silently dropping them. The
error/cost tables retain the classical methods as an application comparison.

## Optional frozen-checkpoint A100 timing

GPU timing is a separate explicit request after a completed, verified CPU
workflow. For this neural-comparison config, lack of classical numerical
headroom does not prevent measuring neural relative costs. At least one TDN
and one neural baseline must each have a trained, validation-selected
checkpoint: a positive selected step with changed parameters. This does not
declare an accuracy or speed victory, and the original architecture config's
classical headroom gate is unchanged. Ineligible or missing neural checkpoints
remain recorded rather than being replaced.

Replace `CPU_RUN_ID` with the actual ID printed by the CPU submission:

```bash
# Preview a separate GPU job using the immutable CPU config and checkpoints.
bash scripts/carc_neural_benchmarks.sh benchmark CPU_RUN_ID

# Explicit submission; no GPU training is performed.
bash scripts/carc_neural_benchmarks.sh benchmark CPU_RUN_ID --submit
bash scripts/carc_neural_benchmarks.sh status latest
bash scripts/carc_neural_benchmarks.sh logs latest --lines 200
bash scripts/carc_neural_benchmarks.sh collect latest
```

The GPU request is **one A100 40 GB, 4 CPUs, 16 GiB host RAM and 30 minutes**.
Actual GPU preflight, CUDA correctness tests and inference timing share the
same allocated `srun` task. The controller verifies the CPU completion marker,
config, source, sealed artifact hashes and software before benchmarking. There
is no automatic requeue, campaign or long GPU training request. CPU or mocked
scheduler tests cannot establish A100 correctness or performance.

## Local/cloud CPU development

In a separate non-CARC checkout with the project `.venv` already installed:

```bash
bash scripts/neural_benchmarks_local.sh --smoke
bash scripts/neural_benchmarks_local.sh

# Optional fresh path, still inside the checkout:
bash scripts/neural_benchmarks_local.sh --run-dir runs/neural-review
```

These commands delegate to `research_local.sh` and preserve its project-local
storage rules. They refuse CARC login-node use and create fresh directories.
Label their measurements local/cloud CPU. The ordinary research report layout,
sealed artifacts, collection and Tower integration are described in
[RESEARCH.md](RESEARCH.md) and [TOWER.md](TOWER.md).

## Initial implementation validation

The full cloud CPU screen completed in 118.95 seconds: all 135 references were
accepted, 14/15 families selected trained checkpoints, and `transport` retained
its numerical validation failure. All seven new baselines trained. The full
CPU correctness suite passed 736 tests; actual A100 execution remains unrun.
Unmodified pinned Tower APIs validated all 16 artifact checks and read 1,705
metric records; independent schema validation passed 1,709 objects.

The original `temporal_mlp` had lower held-out one-step RMS error than
`generic_mlp` in 13/18 cases and lower two-step RMS error in 10/18. It was slower
in all seven feasible matched-tolerance parent comparisons. The three direct
spatial controls had no tolerance-feasible diagnostic rollout in this small
screen. These outcomes motivate further testing; they do not establish a
universal architecture ranking. The bounded CPU measurement, configuration,
training evidence, comparison aggregates and artifact hashes are recorded in
[`results/neural-benchmark-validation.json`](../results/neural-benchmark-validation.json).
