# Bounded physical-consistency experiment

The full Fedora premix review found that a learned correction can damage the
mean even where the physical Strang update is already accurate. This program
tests physically vanishing corrections and separate mean/spatial treatment
with the same changes applied to premix and FNO. It preserves the nonlinear
interaction-before-compression hypothesis and its input-truncation ablation.

This is a fresh, bounded development experiment on periodic logistic
reaction–diffusion. Its completion establishes coverage and implementation
checks; superiority must be assessed from the held-out measurements.

## Run on Fedora

Reuse the verified local profile and project `.venv` created by
[FEDORA_SLURM.md](FEDORA_SLURM.md). For this desktop the project is
`/home/rahel/TDN`, both partitions are `local`, the GPU resource is `gpu:1`,
and the optional scheduler account is unset. Wait for active workflows to finish
before pulling changes into the checkout.

```bash
cd /home/rahel/TDN
git pull --ff-only origin main

# Read-only preview, then an allocated integration check.
bash scripts/fedora_consistency.sh plan --smoke
bash scripts/fedora_consistency.sh run --smoke
bash scripts/fedora_consistency.sh status latest
bash scripts/fedora_consistency.sh logs latest --lines 200
```

After all three smoke stages complete, run the full protocol:

```bash
bash scripts/fedora_consistency.sh run
bash scripts/fedora_consistency.sh status latest
bash scripts/fedora_consistency.sh logs latest --lines 200
bash scripts/fedora_consistency.sh paths latest

# After completion, make the review archive.
bash scripts/fedora_consistency.sh collect latest
```

The new launcher has its own latest-run pointer; old premix workflows and
results remain available through their original launcher. Save the printed run
ID and use it in place of `latest` when several submissions exist.

The dependency chain is **audit → prepare → neural**:

| Stage | Work | Allocation |
| --- | --- | --- |
| Audit | Unit tests and an arbitrary-weight structural audit | 4 CPUs, 16 GiB, 30 minutes |
| Prepare | Independent accepted FP64 teachers and training-only normalization | 4 CPUs, 16 GiB, 30 minutes |
| Neural | Actual allocated GPU checks, paired training and diagnostics | 1 RTX 4090, 4 CPUs, 16 GiB, 30 minutes |

Every numerical stage has a 1,200-second cap. Dependencies use `afterok` and
verify predecessor seals; a required structural failure prevents preparation
and GPU work. No pending-job cap, automatic retry or extended training is
introduced. The existing desktop GPU policy applies: at most 18 GiB, further
limited to 75% of initially free/total VRAM, with a 90% device-use hard threshold.
All caches, temporary files, datasets, reports and archives remain inside the
project. Execution uses its Python venv and real Slurm tasks.

## Seven learned arms

| Arm | Backbone | Correction treatment |
| --- | --- | --- |
| `premix` | Current premix | Current cubic signed-capacity correction |
| `fno` | Current adapted FNO | Current cubic signed-capacity correction |
| `premix_gated` | Premix | Physical commutator gate |
| `fno_gated` | FNO | The identical physical gate |
| `premix_moment` | Premix | Gate plus separate mean/spatial heads and final-mean calibration |
| `fno_moment` | FNO | The identical gate and mean/spatial treatment |
| `precompress` | Parameter-matched premix ablation | Input projection before nonlinear feature construction |

Strang, ETDRK4 and fused GL3 are complete-solver controls on the same device and
precision. Unmodified learned controls reuse their original model factory.
Gated arms share their backbone parameter initialization with their current
control. Moment arms add a scalar mean head; their extra capacity and inference
work are recorded. Premix and FNO retain different parameter counts and FLOPs.

For the discrete diffusion operator
`(Lu)_i = sum_j w_ij*(u_j-u_i)` and reaction `R(u)=r*u*(1-u)`,

\[
C_i = R'(u_i)(Lu)_i-(LR(u))_i
    = r\sum_j w_{ij}(u_j-u_i)^2.
\]

Here the two neighbors per axis have weight `w=kappa/dx²`. The gate is
`tanh(t_ref²*C_i/U_ref)`, and the gated increment is
`U_ref*(h/t_ref)³*gate*network`. It vanishes for homogeneous input, `r=0`,
`kappa=0`, and at zero step size, for arbitrary finite learned weights. The
commutator is exactly quadratic in spatial perturbation amplitude; the gate
has that leading behavior. This single commutator is a structural factor,
not the complete leading Strang defect or a global order guarantee. Multiplying
by the spatial gate can create output frequencies beyond the spectral box.

Moment arms form a bounded learned scalar mean target from the physical base
mean and a separate pooled-latent head. The gated spatial proposal is centered
before its signed-capacity map. A bounded per-parent redistribution then
calibrates the **final output** to that scalar target. Centering only the raw
proposal is insufficient because the capacity map can introduce a mean shift.
The redistribution is piecewise differentiable and adds global per-parent
reductions; its cost is included in complete-rollout timing.

The calibrated target is learned, not the exact physical mean. Logistic
reaction obeys

\[
\bar u'=r(\bar u-\bar u^2-\operatorname{Var}(u)),
\]

so no mean-conservation assumption is imposed. These constraints also do not
establish a nonlinear stability theorem.

## Frozen coverage and selection

The full plan uses **48 training, 24 validation and 48 fresh diagnostic
parents**, with new seed ranges 910000/920000/930000. Previously reviewed
diagnostic fields are excluded from this fresh assessment. Training and
validation cover smooth, mixed, in-band pairs, rough, near-zero and near-one
fields, with varied means, amplitudes, reaction and diffusion coefficients.
Dimensionless reaction and diffusion step parameters are saved in the protocol.

Training uses horizons **0.04, 0.08, 0.16 and 0.24** and two-step endpoints.
Validation uses **0.06, 0.12 and 0.20**, also with two-step endpoints. Teacher
generation requires all **456** declared references to be accepted before
training. Normalization uses only training initial fields and is shared across
arms. Every arm receives **300 Adam updates**, learning rate 0.001, width 16,
cutoff four, and paired seeds **940011, 940021, 940031**. The same schedule and
physical endpoint objective apply across arms; validation selects among
initialization and checkpoints every 50 updates.

All checkpoints are frozen before diagnostics. A selected initialization stays
explicit and does not count as learned superiority. The same 48 diagnostic
parents are reused across seeds, norms and schedules; those repeated rows are
paired descriptive measurements, not independent trials.

The diagnostic bank has four parents in each of twelve declared regimes:
smooth, mixed, in-band pairs, held-out high-frequency pairs, near-Nyquist,
rough, both bounds, stronger reaction, stronger diffusion, long rollout and
64×64 resolution transfer. Ordinary time is 0.24 with 1/2/4 steps; long rollout
ends at 0.96 with 4/8/16 steps. Horizon 0.24 is now in training, so ordinary
endpoints are not a completely held-out-horizon test. Resolution parents are
independent fields, not paired discretizations.

Inference/training use FP32 with TF32 off; same-grid teachers use FP64 RK4
refinement. RMS and maximum targets remain **2e-3, 2e-4, 2e-5, 2e-6**.
Post-hoc frontiers select from the fixed step grid using errors plus reference
uncertainty; they are not a deployable adaptive policy or error certificate.

## Structural checks and interpretation

[The cloud validation record](../results/consistency-validation.json) records
the passing full CPU regression, sealed smoke and full experiments, and checks
using unmodified Tower readers. Those CPU diagnostics already inspect this
frozen diagnostic bank; the native desktop run repeats the same protocol for
device validation and measured timing. It is not a second untouched cohort.
The new 38 allocated-CUDA tests still require the actual desktop GPU.

The audit declares **128 checks**, including **110 required checks** and
18 observations of unconstrained controls. It tests arbitrary nonzero weights
in FP32/FP64, exact commuting limits, finite gradients, bounds, discrete
commutator identities, amplitude scaling, batch independence through model
tests, and final physical mean calibration. A control that lacks the new
constraint remains an observation; any required constrained failure blocks the
workflow. Trained checkpoint consistency is measured again after selection.

Assess coverage at matched accuracy before timing. Inspect per-seed and
per-regime comparisons, mean/spatial error decomposition, regressions against
the physical base, and whether both selected checkpoints are trained. A lower
validation loss or a fast initialization-selected model alone does not establish
the hypothesis. This remains one scalar periodic PDE and an adapted FNO
comparison, without a published-paper reproduction claim.

`paths` prints exact Tower report and metrics locations. Reports include the
structural checks, training histories, candidates, frontiers, paired comparisons
and groups, with canonical provenance. Allocated CPU/GPU JUnit files are
published from a dedicated stage-specific reporting directory, so test tables
are populated without importing neighboring science stages.

For native Tower validation, pass an exact report path from `paths`:

```bash
bash scripts/tower.sh validate /home/rahel/TDN/runs/RUN/tower/REPORT --native
```

The full tables exceed Tower's default aggregate 8 MiB read budget. This project
uses Tower's unchanged Python API with a fixed 32 MiB allowance only for
consistency reports; native table pages retain their existing bounded reads.
Custom Tower launchers may require `TDN_TOWER_PYTHON` pointing to the absolute
Python interpreter that imports your existing Tower installation.

For cloud development, the separate CPU-only integration wrapper is:

```bash
bash scripts/consistency_local.sh --run-dir runs/consistency-local-check
# Optional bounded full CPU development validation:
bash scripts/consistency_local.sh --full --run-dir runs/consistency-local-full
```

These local checks establish CPU implementation behavior. The allocated Fedora
GPU smoke is required to verify native CUDA execution of the new models.
