# Bounded TDN research agenda

This program implements all six stages and all experimental questions in the
user's research audit, plus teacher preparation and the deployment-policy
extension. The research question is whether a compact correction can preserve
the correct quadratic spatial variation and stiff temporal response, and then
improve complete-solver accuracy and cost. Structural zero limits and a faster
adapted FNO comparison are useful checks; neither establishes a solver benefit.

The exact requirement inventory is
[AGENDA_REQUIREMENTS.json](AGENDA_REQUIREMENTS.json). The six supplied literature
links, their executable questions Q1–Q6, and the deployment question Q7 are in
[AGENDA_LITERATURE.md](AGENDA_LITERATURE.md). Source retrieval was blocked by this
cloud environment's proxy, so bibliographic metadata and paper text are not
presented as independently verified. This program does not claim a published
FNO reproduction or architecture novelty.

## Run on the current Fedora desktop

Use `/home/rahel/TDN`, its existing verified `.venv` and Fedora Slurm profile.
Both CPU and GPU partitions are `local`; the GPU resource is `gpu:1`. The
optional scheduler account remains whatever the local profile actually records.
CARC's account and A100 checks stay separate.

Wait for any active TDN workflow to finish before changing source in this
checkout. The launch freezes source, software, profile, protocol, prerequisites
and success criteria. Changing them during a workflow invalidates its seals.

```bash
cd /home/rahel/TDN
git pull --ff-only origin main

# Inspect all allocations and scientific budgets without submitting work.
bash scripts/fedora_agenda.sh plan --smoke

# Run the eight-stage allocated integration check.
bash scripts/fedora_agenda.sh run --smoke
bash scripts/fedora_agenda.sh status latest
bash scripts/fedora_agenda.sh logs latest --lines 200
bash scripts/fedora_agenda.sh paths latest
```

After all smoke stages complete successfully, run the bounded development
cohort. It exercises the questions without viewing the full confirmation bank:

```bash
bash scripts/fedora_agenda.sh plan --development
bash scripts/fedora_agenda.sh run --development
bash scripts/fedora_agenda.sh status latest
bash scripts/fedora_agenda.sh logs latest --lines 200

# After completion, preserve the review evidence.
bash scripts/fedora_agenda.sh collect latest
```

The full preregistered run is a separate fresh workflow:

```bash
bash scripts/fedora_agenda.sh plan
bash scripts/fedora_agenda.sh run
bash scripts/fedora_agenda.sh status latest
bash scripts/fedora_agenda.sh logs latest --lines 200
bash scripts/fedora_agenda.sh paths latest

# After all stages finish:
bash scripts/fedora_agenda.sh collect latest
```

Save the printed run identifier. Substitute it for `latest` when reviewing an
older run or when several workflows have been submitted. There is no
pending-job cap. Slurm reserves the single GPU and the declared CPU/RAM for each
stage; unrelated projects remain free to queue through the scheduler.

Full confirmation is fresh for the first uninspected campaign. Repeating the
same frozen full protocol reuses its parents and produces paired replications;
it does not create another independent confirmation cohort.

`collect` retains the complete archive. If it exceeds the upload limit, it also
produces 28 MiB parts with a SHA-256/index record. Preserve the index and all
parts when transferring a split archive for review.

The local cloud-development wrapper is deliberately separate:

```bash
bash scripts/agenda_local.sh --smoke --run-dir runs/agenda-local-smoke
bash scripts/agenda_local.sh --development --run-dir runs/agenda-local-development
```

Local checks use CPU and development-only parents. They do not establish native
CUDA readiness, measured RTX 4090 speed, or fresh full-confirmation results.
The local wrapper rejects `--full`; the full confirmation bank stays in the
allocated desktop workflow.

## Budgets scaled to the machine

The desktop has eight physical CPU cores, 128 GB physical host RAM, and 24 GB
**dedicated** GPU VRAM. Slurm exposes `110000` MiB host memory; the program uses
that scheduler ceiling rather than assuming all installed RAM is allocatable.
A 48 GiB request is 49,152 MiB and fits the ceiling. Host/shared memory is not
additional GPU VRAM.

All eight TDN stages run in a strict dependency sequence. This avoids assigning
multiple full-CPU teacher jobs to the eight-core machine and keeps the TDN
workflow's own simultaneous memory request below the scheduler ceiling.

| Stage | Device | CPUs | Host RAM | Allocation wall limit | Numerical cap |
| --- | --- | ---: | ---: | ---: | ---: |
| `structure` | CPU | 8 | 48 GiB | 20 minutes | 600 seconds |
| `prepare` | CPU | 8 | 48 GiB | 45 minutes | 1,800 seconds |
| `controls` | RTX 4090 | 4 | 32 GiB | 30 minutes | 1,200 seconds |
| `optimize` | RTX 4090 | 4 | 32 GiB | 45 minutes | 1,800 seconds |
| `compression` | RTX 4090 | 4 | 32 GiB | 30 minutes | 1,200 seconds |
| `kernel` | RTX 4090 | 4 | 32 GiB | 30 minutes | 1,200 seconds |
| `confirm` | RTX 4090 | 4 | 48 GiB | 45 minutes | 1,800 seconds |
| `policy` | RTX 4090 | 4 | 32 GiB | 30 minutes | 1,200 seconds |

Smoke and development profiles have smaller frozen workloads and numerical
budgets.
The smoke policy uses two attempted step sizes; development and full use all
four. Both retain all three estimators, independent calibration and fallback.
`plan` prints their exact allocations. Full stages do not expand automatically
when a cap is reached, retry with changed settings, or become a 20-hour training
job. A scientific failure or inconclusive result remains visible; correctness,
missing prerequisites and budget failures are explicit computational failures.

The existing GPU policy remains an 18 GiB soft ceiling, further limited to 75%
of initially free/total VRAM, and a 90% device-use hard threshold. FP32 training
and inference run with TF32 off; independent teachers run in FP64. CUDA work
requires a real allocation and actual GPU checks, with no CPU fallback hidden
under a GPU label. All caches, temporary files, datasets, checkpoints, logs,
reports and archives stay inside the project. Python runs from a venv, never
Conda.

## Physical model and four architectural mechanisms

The declared target remains periodic logistic reaction–diffusion,

\[
 u'=Lu+r u(1-u),\qquad
 (Lu)_i=\sum_j w_{ij}(u_j-u_i),\quad w_{ij}=\kappa/\Delta x_d^2.
\]

A1 changes **where the physical source enters**. The historical treatment
multiplies the backbone output by the local source/gate. The new source arm
multiplies features by the physically vanishing source before a zero-preserving
spatial operator. Bias-free post-source maps ensure zero source gives zero
correction for arbitrary finite weights. The physical source uses the full
input even in the precompression control; that information bypass is recorded.

For a uniform interior background plus a small perturbation, the historical
quadratic gate has a fixed leading spatial profile. An equivariant backbone is
constant at the background, so its leading correction is a scalar multiple of
that profile. The endpoint mean head adds a constant direction. Source-first
transport can redistribute the quadratic field and reach flat receiver regions.
The structural panel compares oracle gate, gate-plus-constant, filtered-source
and signed-source-dictionary spans against an independently resolved defect.
Oracle fits diagnose representation; they are not trained solver results.

A separate nonlinear time decoder tests the temporal restriction independently
of gate placement. It receives `h/t_ref`, `h*r`, a diffusion stiffness scale and
the state mean through a SiLU MLP. Gate-only, time-only and combined controls
prevent improvements from being attributed to an unidentified combination.
The temporal audit evaluates proposals through the actual bounded capacity map.

A2 is a **phase-aware eigenvalue-conditioned mode-pair branch**. Real-even
learned filters act on complex phase-bearing input spectra; filtered-field
products use the declared native-grid circular convolution. Input and output
filters receive the actual discrete diffusion eigenvalues scaled by the queried
step, reaction scale and mean. The physical commutator construction preserves
pair symmetry and zero limits. It is already quadratic in perturbation
amplitude and is not multiplied by another quadratic gate.

Learned nominal ranks 2, 4 and 8 are distinct from strict tensor rank: each
commutator factor expands into at most three separable terms. The implementation
records `1+4R` correction transforms and `3R` local Laplacian applications for
nominal rank `R`, in addition to the physical base. The offline study uses a
constructive quadrature-factor response approximation: `R` response terms plus
the analytic split terms. It is not an optimal tensor-rank fit, and its rank
label does not mean it is the same factorization as the learned branch. The
training-free response/rank panels are deliberately bounded one-dimensional
periodic diagnostics; the learned solver and fresh confirmation are
two-dimensional. A promising 1D oracle does not establish 2D rank transfer.

A3 compares the two **established Strang orientations**:

\[
 R_{h/2}D_hR_{h/2},\qquad D_{h/2}R_hD_{h/2}.
\]

For strong diffusion on a connected periodic grid, the coupled solution
approaches `R_h(mean(u0))`. Reaction-first Strang generally has a different
homogenized endpoint; diffusion-first has the correct limiting endpoint.
Both are retained as complete-solver controls. Highest-mode stiffness and
spectral-gap homogenization are measured separately. Complete rollout timing
counts the actual transforms; corrections can prevent half-step fusion.

A4 tests a **bounded dynamical mean–variance closure**, rather than an arbitrary
endpoint mean. With `c=mean(u)`, `v=u-c`, `V=mean(v²)`, the exact instantaneous
semidiscrete identities are

\[
 c'=r(c-c^2-V),\qquad
 V'=2\langle v,Lv\rangle+2r(1-2c)V-2r\langle v^3\rangle.
\]

The implemented closure uses `V=z*c*(1-c)`, an initial rate from these identities,
a learned elapsed-time remainder and four bounded substeps. It subtracts the
zero-learned integrated endpoint before calibrating the physical base's final
mean. This is an approximate frozen-moment evolution, not the exact future
variance law or a mean-conservation claim. Equal-mean/equal-variance fields with
different spectra and phases expose what energy summaries can and cannot
recover. The spatial kernel remains separate. The historical arbitrary mean
head and analytic spectral-mean correction are retained controls.

## Eight executable stages and every research question

| Stage | Decisive measurements | Questions |
| --- | --- | --- |
| `structure` | Quadratic gate-span obstruction; uncertainty after division by amplitude squared; actual bounded-map time response; strong-diffusion orientation; rank/phase/aliasing and moment identities | Q1, Q2, Q3, Q4, Q6 |
| `prepare` | Independent accepted teachers; disjoint train/validation/calibration/confirmation manifests; continuous-field provenance; training-only normalization | Q5 prerequisite |
| `controls` | Source placement and nonlinear time decoding separately; current local gate/mean anchors; both physical orientations; mean–variance closure | Q1, Q2 and A1–A4 controls |
| `optimize` | Bounded learning-rate/batch alternatives; gradients before/after clipping, correction sizes, physical defect scales and validation loss by regime | Optimization diagnosis |
| `compression` | Paired trained source-constrained premix/precompression; actual matched support; parameter-matched FNO and separately measured-cost comparisons; bypass inventory | Q4 |
| `kernel` | Offline rank evidence; conditional rank 2/4/8 learned study; phase and eigenvalue scale retention; added-transform cost | Q2, Q3 |
| `confirm` | Fresh paired grids, crossed factors, truly unseen equal/unequal steps, RMS/maximum coverage, spatial regressions and classical work–precision frontiers | Q1, Q2, Q3, Q4, Q6 |
| `policy` | Calibration-only estimator fit; frozen threshold; independent accepted accuracy; false accepts; rejected work and timed classical fallback | Q5, Q6, Q7 |

The structural plan declares 32 smoke, 76 development and 222 full cases.
Each stage writes question-bearing records, a summary and a sealed artifact
manifest. A stage's negative observation is evidence, not a reason to hide its
rows. The offline rank criterion determines whether the learned-kernel branch
is promising; an unfavorable result remains an explicit decision and retains
the structural evidence. Initialization-selected matched pairs remain
inconclusive about learned compression benefit.

The full frozen trial matrix declares 11 minimal controls, 36 learning-rate/
batch optimization trials, 18 matched compression trials across three paired
seeds, and up to four conditional kernel trials. This is at most 69 attempts,
with 120, 120, 300 and 180 updates per trial respectively. The optimization
choices are learning rates `1e-4`, `3e-4`, `1e-3` and physical minibatches
1, 4, 8. Each minibatch shares reaction, diffusion, grid and queried step;
parents are sampled with replacement inside its frozen physics bucket.
Matched downstream trials inherit the source control's validation-selected
orientation and optimization settings, with their exact lineage recorded.

Confirmation keeps at most 12 frozen variants: the three mandatory source/
precompression/parameter-matched-FNO families across all three seeds, two
additional source/time/closure controls, and the validation-selected eligible
kernel. Numerical training failures and initialization selections remain
explicit and do not count as learned superiority.

Seventeen available family IDs cover historical local controls, source controls,
closure/endpoint treatment, matched FNO and pair kernels:

| Group | Family IDs |
| --- | --- |
| Historical local gate controls | `local_gate`, `local_gate_time`, `local_endpoint` |
| Source placement and time controls | `source`, `source_time`, `source_endpoint` |
| Mean–variance treatment | `source_closure`, `source_time_closure` |
| Compression and FNO controls | `precompress_source`, `fno_source`, `fno_source_depth4`, `fno_source_matched`, `fno_anchor` |
| Structured pair branch | `pair_rank2`, `pair_rank4`, `pair_rank8`, `pair_rank4_closure` |

The matched FNO uses the inclusive Fourier-mixing support mask of the source
model. Its full-grid local branches and SiLU activations still carry frequencies
outside that box; their bypass is recorded. Matching learned Fourier support
does not make the complete function classes identical.
The historical four-layer anchor intentionally retains its old mask and is
labeled accordingly. Parameter matching uses the nearest actual stored count
at an integer width, with the remaining mismatch reported. Parameter matching
and inference-cost matching are different comparisons; neither is implied by
matching the integer `modes` option.

## Fresh confirmation and declared success

Training, validation, calibration and confirmation parents are disjoint. The
full confirmation bank consists of eight controlled continuous-field variants
crossed with four diffusion/reaction combinations: 32 physical cases sampled on
paired 32×32, 64×64 and 128×128 grids. These cases are not 32 independent
random fields. Several variants reuse phases/seeds to isolate one changed
factor. Field coefficients, phases, means,
amplitude ratios and physical parameters are frozen in the protocol. The stored
`amplitude` factor is the relative harmonic-weight ratio at fixed total initial
variance; it is not the state's supremum amplitude. A separate variance factor
changes the perturbation scale. The same field
is sampled on all three grids rather than drawing unrelated resolution cases.

The final-time 0.27 schedules include unequal steps `0.03,0.07,0.17`, their
reverse order, `0.025,0.055,0.075,0.115`, and six equal `0.045` steps. A separate
long-horizon 0.81 case tests accumulation. These steps must remain absent from
training and validation selection. Equal-time composition error and independent
endpoint accuracy are separate measurements.

Same-grid discrete teachers resolve the declared finite-difference/nodal
reaction–diffusion equation. The continuum-estimate track uses the predeclared eight physics-index-zero
confirmation cases. It uses a dealiased spectral teacher with independent
spatial and temporal refinements, then projects to the comparison grid. It is an estimate, not a mathematical
certificate. Unresolved reference uncertainty yields an inconclusive target,
not an artificial pass. Product conventions and Nyquist interpolation are
explicit; errors from the two targets are not pooled.

Success criteria are frozen before the fresh cohort is opened. The full
protocol's operational criteria require both RMS and maximum-error targets of
`2e-4`, at least 75% coverage in each regime and 90% overall, a spatial-error ratio no greater than 1.1 against the physical base above the
declared FP32 floor, and a
median 1.1× complete-solver speedup over ETDRK4/fused GL3 with accuracy coverage
retained. The deployment gate additionally permits at most 1% empirical false
acceptance; per-policy counts and denominators remain explicit. An aggregate
rate across candidate policies does not validate each policy separately. These are decision rules, not statistical significance or proof of
uniform superiority. Reference uncertainty, per-regime failures, initialization
selections and same-step spatial damage remain visible even when a median
improves. Repeated grids/seeds/norms/schedules are paired observations of the
same physical parents.

## Acceptance policy and reported cost

A held-out calibration bank fits empirical estimator envelopes for step
agreement, an autodifferentiated flow-defect residual, and their combined
envelope. The defect proposal sums substep-weighted norms of
`d_h Phi_h(u) - F(Phi_h(u))`. It is an empirical residual diagnostic, not a
rigorous stability-weighted defect integral. The
thresholds and selected checkpoints freeze before the fresh bank is examined.
Policy decisions may use estimators and declared physical features, not fresh
reference errors.

Every accepted prediction is independently checked against the same-grid
teacher at each RMS/maximum target. The report includes acceptance and false
acceptance counts, denominators/rates, rejected neural and estimator work,
classical fallback calls and measured fallback cost. The comparison benchmark shares estimator evaluations. Each policy receives
component-attributed costs for its own proposals, estimator, rejects and
fallback, while actual estimator-attempt benchmark work is reported separately. This attributed
sum is a deployment cost estimate from measured components, not an
independently timed standalone policy wall clock. Inspect both cost records;
unused but actually computed estimators must not disappear from the benchmark
attempt-work total. The stage elapsed time includes calibration, classical
controls, fallback and reporting work as well. Direct ETDRK4 and fused GL3
remain controls. An empirical low false-accept rate is not a certified bound;
composition agreement can coexist with wrong predictions.

Reference-informed fixed-grid work–precision frontiers remain useful
post-hoc diagnostics. They are labeled separately from deployable policy
frontiers so a policy advantage cannot come from looking at the true error to
select its step size.

## Evidence and Tower

`status` reports scheduler and scientific stage outcomes; `paths` prints the
exact stage report and metrics paths. `collect` includes protocols, data,
references, selected/initial checkpoints, histories, candidate/frontier tables,
question records, acceptance/rejection evidence, logs, failures and seals.
Stage-specific test reports populate Tower's test table without importing
neighboring stage science. The project adapts its output contracts; Tower's
application remains unchanged.

Review computational correctness first, then representation and optimization
findings, then fresh accuracy coverage, and finally complete-solver/policy cost.
An incomplete stage or failed audit preserves its evidence and blocks dependent
work. A fully completed workflow can still conclude that a hypothesis failed or
that a target remains inconclusive.

[Validation evidence](../results/agenda-validation.json) records the actual
cloud CPU checks, complete smoke seals, parallel teachers and unchanged native
Tower reader. Its execution scope explicitly separates these checks from the
remaining Fedora Slurm/CUDA validation.
