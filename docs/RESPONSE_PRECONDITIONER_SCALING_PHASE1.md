# Phase 1: independent-parent and source-direction scaling

**CPU implementation verified, 16 October 2026. Native results pending.** This is the
first phase of the process changes requested after the
[native R7 review](../results/native-response-preconditioner-r7-2939/README.md).
It is a new bounded development protocol, **P1**, separate from the proposed
R8 learned encoder. Read the [standing program](TDN_RESEARCH_PROGRAM.md) and
[runbook](RESPONSE_PRECONDITIONER_RUNBOOK.md) alongside this document.

The question is whether the current fixed response family benefits more from
additional independent physical parents or additional forcing directions on
each parent. The architecture, ridge objective and candidate menu remain
fixed so these two data axes have an interpretable comparison. This phase
does not implement the other proposed interventions simultaneously.

## Why this experiment

R7's two training parents constrained its 40-column design to rank 25. More
directions on those same parents cannot remove that structural bound. Its
selected ridge objective was already minimized, while a local five-branch
projection retained resolved error on the longest interval. Additional
optimization, increased physical coverage and increased representation are
therefore different hypotheses.

P1 tests the data-coverage hypothesis with one predeclared population and
ordering. Its learning curves are conditional on that population and ordering;
they are not a universal sample-complexity law. The previously suggested 32,
64 and 128-parent cohorts were examples, not an authorization to acquire all
three. This bounded first phase ends at 32 training parents. The review of
its actual costs and learning curves determines any subsequent proposal.

## Frozen comparison

| Component | Phase 1 contract |
| --- | --- |
| Equation and target | Periodic logistic reaction–diffusion, same-grid finite differences on a 5 by 5 grid |
| Model | Existing five response branches, eight deterministic state/physics/time features per branch, 40 affine weights |
| Precision | FP64 computation from the actual FP32 initial states |
| Independent parents | 32 training, 8 validation and 8 diagnostic; four field families balanced within each split |
| Training-parent axis | Nested prefixes of 8, 16 and 32 parents, balanced across the four families |
| Calibration-source axis | Nested prefixes of 2, 4 and 8 dense random directions per parent/context |
| Cells | Nine separately fitted and validation-selected models: each parent count crossed with each source count |
| Fixed intervals | `(0, .04)`, `(0, .08)`, `(.02, .04)`, `(.02, .06)`, `(.04, .06)`, `(.04, .08)` |
| Withheld directions | Four per context: dense random, low frequency, high frequency and constant, each unit physical RMS |
| Ridge candidates | Initialization, `lambda=1e-10`, `lambda=1e-6`; unchanged parent-balanced physical action MSE objective |
| Selection | Minimum validation-withheld physical action MSE, with initialization first on exact ties |
| Classical controls | Ordered 2 and 8 panels; fixed-step Lawson 2, 4 and 8 |
| Native envelope | One real Raven/Slurm allocation: 4 CPUs, 8 GiB, 20 minutes; 900-second study budget |

Parents are generated independently of the inspected R7 bank. Their complete
manifest, field construction, seeds, physics and deterministic family-balanced
ordering are frozen before reference acquisition. Split membership belongs
to the parent; all of its intervals, source directions and spatial cells stay
in the same split. Phase 1 is **fresh development**, not untouched scientific
confirmation, and uses no data-dependent parent selection.

The four families are `smooth_spectral` (low-frequency trigonometric mixture),
`localized_periodic` (periodic exponential bump), `periodic_transition`
(hyperbolic-tangent transition) and `multiscale_discrete` (a mixture including
the grid's higher Fourier frequencies). Each split interleaves those families
in that order. The 8/16/32 training prefixes therefore contain 2/4/8 parents
per family; validation and diagnostic each contain two per family.

Training seeds are `2026101601`–`2026101632`, validation seeds
`2026101701`–`2026101708`, and diagnostic seeds `2026101801`–`2026101808`.
Independent per-parent RNG streams generate morphology, state mean/amplitude
and physics. Each discrete morphology is centered and divided by its maximum
absolute value. Sampled mean is uniform in `[.25, .65]`, amplitude in
`[.04, .18]`, diffusion coefficient in `[.009, .017]`, and reaction rate in
`[1.7, 2.9]`. The resulting state is rounded to FP32 once. Parent identity
hashes the actual field and canonical physics/grid metadata; field identity
is separately retained to detect repeated inputs despite different labels.

These are periodic discrete morphology families, not continuum-resolved
sharp-interface tests. They sample a specified small development population,
not all possible PDE regimes. The grid is held fixed; a learning curve here
does not establish large-grid accuracy, resolution transfer or large-grid
speed.

P1 changes the morphology families and mean/amplitude sampling from R7.
Only the nested comparisons **within P1** hold that population construction
fixed while varying parent/source count. A pooled error difference between
R7 and P1 would also reflect the changed physical population and cannot be
attributed solely to the larger training bank.

### Work and access order

All 32 training parents receive eight calibration and four withheld sources
at each of six contexts. Validation and diagnostic parents receive only the
four withheld sources. Thus the complete bank has

`32 * 6 * (8 + 4) + (8 + 8) * 6 * 4 = 2,688`

source/context query pairs. Each pair acquires both the full response `U`
and the matched-mean response `G`, with their refinement and independent
reference checks. This is 2,688 query pairs, not 2,688 independent physical
experiments. The fixed acquisition/reference caps and retained denominators
apply even when some references fail or remain unresolved.

The larger cells reuse the exact smaller-cell parents and directions.
Training labels are acquired once and shared across the nine comparisons.
Unused calibration directions do not become training labels for a smaller
cell. Each fit gives equal total mass to its included parents, then equal
mass to contexts, included calibration directions and cells within a parent.
Withholding and nested membership are recorded explicitly.

Each cell fits only its training calibration labels and selects only on the
fixed validation-withheld bank. **All nine selections must seal before any
training-withheld or diagnostic truth is acquired.** The diagnostic bank evaluates every sealed
cell; it does not choose a winning model for deployment. There are no local
diagnostic oracle fits in this phase. Repeatedly inspecting these diagnostics
for later design decisions makes them development evidence permanently.

Nonzero source times use the same original anchor evaluated at absolute time.
The path is not restarted from its intermediate state. Reference uncertainty
continues to be an estimate from refined and independent computations, not a
certificate. Unresolved fitting or selection labels block the dependent work;
failed and unattempted slots remain in the planned denominator.

## What the measurements can decide

The primary output is a two-axis learning-curve table: independent training
parents versus calibration directions per parent. Preserve all nine cells,
all candidate objectives and validation outcomes, including initialization
selection or ineligible fits. Report original design rank/spectrum separately
from the regularized augmented condition number and numerical optimality
diagnostic. Positive ridge can make a fit unique without making its data
directions identifiable.

Physical RMS and maximum error, signed mean and centered error retain parent,
family, split, interval, source family and method identity. Report the full
response error alongside both relative denominators: the learned remainder
`(U-P)z` and full spatial feedback `(U-G)z`. Small error relative to full
feedback can coexist with substantial error in the much smaller remainder.
Reference-sized or near-zero denominators stay visibly unresolved.

The 1,152 withheld queries across all splits each retain all nine selected
fits and five classical controls: 16,128 planned metric rows. Missing or
unresolved rows remain explicit, even when later work is blocked.

Parent aggregates are the statistical units. Directions, times and grid cells
are paired descendants, not independent samples for error bars. A paired
parent difference compares two cells on the same withheld parent. Family
strata and individual-parent results remain visible alongside pooled curves.
Training-withheld curves distinguish parents inside each fitted cohort from
those outside it. Their membership changes across cohort sizes, so these
curves are not paired comparisons on a common test population. Validation and
diagnostic populations remain fixed and support the paired comparisons.

Report individual-parent differences, median/worst differences and the
fraction of parents improved. There is no bootstrap, confidence interval or
p-value in this phase. With only two diagnostic parents per family and one
nested cohort ordering, these summaries are descriptive evidence for the
fixed sample. They do not justify population-wide guarantees or a
statistically selected best architecture.

Interpret the curves jointly:

- Improvement with more parents at fixed source count supports expanding
  physical coverage under a new budgeted protocol.
- Improvement with more directions at fixed parent count supports buying
  better within-parent response coverage.
- Persistent calibration error suggests an approximation or penalty limit;
  it does not identify which one without separate controls.
- Better action accuracy without adequate complete cost leaves practical
  solver utility unresolved.

There is no predeclared superiority threshold in this phase. A complete job
or a downward learning curve is not a utility pass, a promotion of historical
B STOP, or evidence of architectural novelty.

### Timing and costs

Time only the predeclared largest cell, 32 parents by 8 calibration directions,
against the five classical controls. Timing uses the first diagnostic parent
of each of the four families and all six intervals: 24 contexts. This subset
is fixed before labels; accuracy remains available for all eight diagnostic
parents and all nine fitted cells.

Each context/method evaluates its four withheld sources in one complete
bundle with fresh setup for each source. Retain one first measured bundle,
one warmup and five randomized paired rounds:

`24 contexts * 6 methods * (1 + 1 + 5) = 1,008 bundles`

and `1,008 * 4 = 4,032` complete source calls. The first measured bundle is
benchmark-local after accuracy work, not a cold-process startup. There is no
cross-source cache or vectorized throughput claim. Keep raw paired rounds;
report their empirical variation rather than treating source calls as
independent timing experiments.

Record acquisition, fitting/selection, accuracy evaluation and timing work
in their declared scopes. Shared bank acquisition is a campaign cost, not a
fresh acquisition charged nine times. Conversely, each cell's label demand
must remain visible. `outputs/study/costs.json` records unique acquired labels
once globally and each cell's label demand separately, with reference,
design-construction, fitting, accuracy-evaluation and timing durations. These
are nested components of elapsed work, not independent amounts to add to
study or native wall time. Native worker CPU/RSS telemetry is distinct from
per-method timing and Raven's supervisor graphs. Sampled process-tree RSS is
neither per-method peak memory nor an allocation-wide measurement. This
phase has no GPU observations.

## Verification and native execution

The executable protocol and exact verifier roster are frozen in
[`scaling_protocol.py`](../tdn/analysis/response_preconditioner/scaling_protocol.py).
The entrypoint is
[`response_preconditioner_scaling.py`](../scripts/response_preconditioner_scaling.py).
The [verification receipt](../results/response-preconditioner-scaling-verification-20261016/README.md)
records 388/388 required CPU checks and eight additional pinned Raven
compatibility checks, all passing without skips. The retained private fixture
completes 48 queries, four fits, 288 fresh metric rows and 12 timing bundles.
Implementation checks exercise generation, nesting, split isolation, fitting,
aggregate metrics, failure retention and native launch boundaries. Local CPU
verification acquires no phase-wide scientific bank and is not a native
execution receipt. Consult the maintained ledger for the actual verification
record; native P1 results remain pending until an uploaded native receipt is
reviewed.

The private 48-query verification fixture uses prefixes of the production
parent seeds. Consequently some P1 development fields and their small fixture
queries are inspected during verification. This is intentional implementation
coverage, not untouched confirmation. The production bank still requires
its complete frozen workload and all-nine selection seals; a fixture cannot
release or stand in for that study.

Local verification:

```bash
if cd /home/rahel/TDN; then
  mkdir -p .runtime/response-preconditioner-scaling-local &&
  TMPDIR="$PWD/.runtime/response-preconditioner-scaling-local" \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python scripts/response_preconditioner_scaling.py verify --device cpu
fi
```

After all older jobs using this checkout have finished, prepare and submit
the new phase:

```bash
if cd /home/rahel/TDN &&
   git pull --ff-only origin main &&
   contract="$(.venv/bin/python scripts/response_preconditioner_scaling_native.py prepare)"; then
  raven launch scripts/response_preconditioner_scaling.sh \
    --project "$PWD" --recipe response-preconditioner-scaling \
    --sbatch=--partition=local --sbatch="--chdir=$PWD" \
    -- --contract "$contract"
else
  unset contract
  printf 'Preparation failed; no job submitted. See the error above.\n' >&2
fi
```

Use the CPU partition from `.tdn/fedora-slurm.json` if it differs from `local`;
include its required account before `--` if applicable. Insert Raven `--check`
before `--` to preview. The guarded block does not exit the interactive shell
on preparation failure. Do not source the launcher. Do not pass `--local`:
the study requires the actual native allocation and private Raven runtime.

Preparation freezes source, software, profile and both verification/study
protocols without acquiring scientific labels. The default worker runs the
complete mandatory verifier, then the study within the same attempt. A
verification release from another run is invalid. `--mode verify` after the
contract requests verification only; no study-only mode exists. Preserve the
checkout and venv while the job is queued or running. A code, package or
profile change requires a fresh contract and attempt, preserving the old one.

After terminal completion, replace the example path with Raven's exact run
directory to export its evidence:

```bash
if cd /home/rahel/TDN; then
  .venv/bin/python scripts/response_preconditioner_scaling_native.py export \
    --run runs/launch-EXACT-a1
fi
```

The exporter prints a fresh directory under
`.runtime/response-preconditioner-scaling-exports/`. Upload its `README.txt`,
`index.json` and ZIP or every numbered ZIP part. It caps each upload part at
30 MiB and preserves raw scientific files. Export requires terminal records;
it does not rerun science or require the current source to equal the old run.

The initial P1 exporter used a 1 MiB writer limit for file inventories and a
64 MiB reader limit for individual evidence files. A full 7,680-file bulk
inventory can exceed the first limit after its ZIP has already been written;
export can then encounter the second limit on that retained ZIP. The repaired
exporter streams files in 1 MiB chunks and uses explicit 16 MiB inventory
limits, retaining the 512 MiB packing and 1 GiB total export budgets. Errors
identify the offending path and byte budget. Symlink, regular-file, terminal
state, hash and mutation checks remain enforced.
The [repair receipt](../results/response-preconditioner-scaling-export-fix-20261016/README.md)
records 404 required CPU passes and 51 overlapping Raven compatibility passes,
including a real 65 MiB file and 7,680-file inventory regression.

For this packaging error, pull the repair after active jobs using the checkout
finish, then repeat **only the export command** for the original attempt.
Preserve its native failure status, completed scientific files and any partial
packaging files. An archive without `packing.json` is retained in the review
bundle. Export does not repair historical status or rerun the experiment;
the uploaded study evidence determines whether scientific work completed.

### Portable evidence and Raven analytics

Raven owns the outer run. The `response-preconditioner-scaling` recipe enables
experiment, evidence, artifacts and CPU/memory observation. This is one job,
without an array, GPU collection, tracing or automatic successor. Keep science
metrics separate from resource telemetry. The compact research series expose
learning-curve cells and parent aggregates; full query-level metrics remain
portable JSON/CSV artifacts rather than flooding the analytics interface.

| Path within the run | Evidence |
| --- | --- |
| `outputs/native/` | Status, original prepared contract/profile, verification release and actual worker telemetry |
| `outputs/verification/` | Exact collected roster, JUnit, summary and verification artifacts |
| `outputs/study/` | Frozen protocol/source/software, parents and queries, all selections and fits, learning curves, per-parent/query metrics, complete timing and failure denominators |
| `outputs/study/costs.json` | Unique acquisition charged once, per-cell label demand and nested component durations |
| `outputs/native/portable-study-evidence.zip`, `packing.json` | Registered raw study payload and its packing record; export verifies and avoids duplicating the payload |
| `logs/` | Portable verification/study stdout/stderr |

Scheduler streams also remain under checkout
`logs/response-preconditioner-scaling/`. Binary query arrays are retained;
load NPZ with `allow_pickle=False`. Packing occurs after science and is charged
to native elapsed time, not inserted into measured method timing.

## Phase roadmap and stop points

| Phase | Question and intervention | Current authorization/status |
| --- | --- | --- |
| P1 | Independent-parent versus source-direction scaling with the fixed response family | This bounded implementation; native evidence pending |
| P2 | D-optimal/E-optimal acquisition versus stratified/random physical coverage | Proposed; freeze a separate selection rule, acquisition budget and untouched evaluation bank |
| P3 | Actual-defect and full-quadrature correction supervision versus generic source supervision | Proposed; implement task-specific labels and preserve generic stress tests |
| P4 | Generalized Tikhonov/physical priors versus the unchanged coefficient ridge | Proposed; declare the new objective, penalty nullspace and candidate-selection rule |
| P5 | Replace costly classical stages with learned approximations | Proposed; profile representative grids, enforce appropriate consistency and charge complete work |
| P6 | Bounded comparisons separating data, representation, supervision and penalty | Proposed; reuse eligible references, freeze comparisons and independent-parent analysis |

The full-operator audit and independently derived analytic correction remain
useful controls, not sole prerequisites for this newly authorized data-scaling
phase. R8's nonlinear encoder, new response branches, adaptive endpoints,
rollouts, GPUs and fresh confirmation remain gated. No 64/128-parent cohort,
alternative loss, new architecture or follow-on launch is automatic.

Before the next phase, independently review source/artifact integrity,
reference acceptance, all nine access-order seals, fit diagnostics, family
coverage, learning curves and complete work. An exhausted budget or failed
reference is a retained outcome, not permission to enlarge the run silently.

## Term guide

| Term | Meaning here |
| --- | --- |
| Parent | Independently generated physical initial field with its sampled physics; the unit of split and uncertainty analysis |
| Family | A predeclared spatial field construction, used for balanced coverage and stratified reporting |
| Context | One parent at one absolute source/end time interval |
| Source / direction | A forcing vector applied to the linear response, normalized to unit physical RMS |
| Calibration source | A training direction eligible for the cell's coefficient fit |
| Withheld source | A direction excluded from fitting; validation withholding selects candidates, diagnostic withholding evaluates sealed candidates |
| Nested cell | One prefix of parents crossed with one prefix of calibration directions, sharing exact observations with larger cells |
| Physical RMS | Square root of cell-mass-weighted mean squared field error |
| `U` | Full finite-time tangent response along the declared anchor |
| `G` | Matched-mean response used to expose spatial feedback |
| `P` | Existing two-panel ordered classical response backbone |
| `(U-P)z` | Remainder targeted by the learned correction for source `z` |
| `(U-G)z` | Full spatial feedback; a different relative-error denominator |
| Ridge / Tikhonov | Quadratic regularization of the conditional coefficient fit; P1 retains the existing isotropic coefficient penalty |
| Design rank | Number of identifiable linear combinations in the unregularized sampled action design, not spatial operator rank |
| D-optimal / E-optimal | Later acquisition rules that respectively increase information determinant or its weakest eigenvalue |
| Defect | Difference between the anchor's time derivative and its governing vector field; not the same as a generic random source |
| Development | Evidence available for subsequent design changes; it cannot later be relabeled untouched confirmation |
