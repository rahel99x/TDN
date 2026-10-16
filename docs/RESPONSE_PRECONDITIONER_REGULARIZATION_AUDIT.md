# P1R: bounded retained-data regularization audit

**Native audit completed and reviewed at `4d0e049`.** This is a separate
source-bound development protocol, `tdn.response_preconditioner.regularization.v1`.
It implements the next bounded question selected in the
[P1 review](../results/native-response-preconditioner-scaling-2944/README.md),
without changing P1, R7 or the model architecture. Read the
[standing program](TDN_RESEARCH_PROGRAM.md) and
[runbook](RESPONSE_PRECONDITIONER_RUNBOOK.md) for current verification status.
The [implementation receipt](../results/response-preconditioner-regularization-verification-20261015/README.md)
records **585 exact mandatory CPU passes**, eight additional Raven compatibility
passes and the 32-query/144-metric algebraic fixture.
[Native job 2951](../results/native-response-preconditioner-regularization-2951/README.md)
passed its exact 585-case suite and completed all 39 candidates and 19,584
metric slots. Independent replay verifies source/input guards, selection,
numerical objectives, sensitivity, metrics and successful publication/packing.
Largest-cell diagnostic RMS improves 15.44%, but stays 568 times Lawson two's.
Read the review's figures, term guide and
[proposed structural-attribution gate](../results/native-response-preconditioner-regularization-2951/proposal-review.md)
before further work. No unchanged rerun is needed; the protocol below remains
the immutable implemented experiment, not a launch instruction for a successor.

## Question and scope

P1's original objective was accurately minimized, but every validation choice
landed on the lowest offered ridge, `1e-10`. All sampled designs had rank 40,
while effective degrees of freedom were only 6.60–6.80. More parents improved
accuracy modestly; more directions helped little. The learned action remained
less accurate than Lawson 2 at similar complete cost.

This audit asks whether the coefficient penalty suppresses useful corrections,
and distinguishes that penalty's directional shrinkage from an explicit
penalty on the physical correction action. It does not assume that weaker
regularization will generalize or that a new architecture is necessary.

The exact reviewed job-2944 archive is the input. It is committed at
`results/native-response-preconditioner-scaling-2944/native-review.zip`, with
SHA-256 `caaacb862df0e272763340b69f523e66884c40e32013cd1ec6f0ff46eb89b6fb`.
The matching upload index, source/protocol manifests and query/prediction hashes
are checked before use. The original numerical study completed, although its
wrapper failed during packing. Its original status and files remain unchanged.
Preparation inspects hashes and provenance; it does not load numeric arrays or
fit models. The worker rechecks inputs before and after execution.

All P1 data have already been inspected. Training/validation/diagnostic roles
are retained to prevent further fitting leakage, but this reuse is development,
not fresh confirmation. Input-file access controls document this audit's access
order; they cannot erase prior inspection or prove all external information
access.

## Frozen workload

| Component | Contract |
| --- | --- |
| Model/target | Unchanged five branches × eight fixed features, 40 FP64 readout weights; actual FP32 initial fields, 5 × 5 periodic same-grid FD |
| Cells | 8 parents × 8 directions; 32 × 2; 32 × 8, using original sealed prefixes |
| Rationale | First two have equal 384-query training demand; third is P1's largest cell |
| Coefficient ridge strengths | `1e-6, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18` |
| Action-energy strengths | `1e-6, 1e-4, 1e-2, 1, 100` |
| Controls per cell | Zero readout, saved original P1 readout, unregularized training projection |
| Candidate roster | 13 per cell: zero + six coefficient ridges + five action-energy penalties + one projection; 39 total, 33 positive fits and three projections |
| Selection | One choice per cell and positive-penalty family, including zero; six choices total |
| Validation | Original eight validation parents and 192 withheld queries only |
| Evaluation | All 1,152 original withheld queries; 17 methods and 19,584 metric slots |
| Methods | Six selected readouts, three original readouts, three projections, ordered 2/8 and Lawson 2/4/8 |
| New truth/timing | Zero teacher solves, new parents, physical labels or inference timing bundles |
| Resources | One native Raven/Slurm CPU allocation: 4 CPUs, 8 GiB, 20 minutes; one numerical thread; 600-second study budget including final reporting |
| Expansion | No automatic successor, larger cohort, encoder, GPU work or confirmation |

The numerical array bank comprises 2,688 original queries. Its union is reused
once; overlapping cell demands are not summed into new acquisition cost.
The unregularized projection is always an offline diagnostic and is never
eligible for selection. Diagnostic comparisons between cells or families do
not create an additional model-selection step.

## Objectives and guarantees

Let `A` be the retained action design, `b = (U-P)z` its target remainder and
`W` positive physical row weights normalized to total one and equal parent mass.
The original coefficient ridge minimizes

\[
J_\lambda(w)=\|\sqrt W(Aw-b)\|_2^2+\lambda\|w\|_2^2.
\]

Every listed `lambda` is strictly positive, so the exact objective has a unique
minimizer even if the original design is rank deficient. The augmented SVD uses
column scaling while carrying the original penalty through that coordinate
change. It does not penalize the scaled coefficients instead.

The implemented generalized fitter also supports

\[
J_{\lambda,L,w_0}(w)=\|\sqrt W(Aw-b)\|_2^2+
\lambda\|L(w-w_0)\|_2^2.
\]

For `lambda > 0`, strict convexity requires
`ker(sqrt(W) A) ∩ ker(L) = {0}`. With
`B = [sqrt(W) A; sqrt(lambda) L]` and `mu = sigma_min(B)^2`, the numerical
optimality diagnostic is

\[
J(w)-J(w_*)\leq \frac{\|\nabla J(w)\|_2^2}{4\mu}.
\]

The inequality is exact-arithmetic mathematics; its implementation uses ordinary
FP64 values, not certified outward rounding. Using `lambda` alone as curvature
would be wrong for a general `L`. Singular systems are explicitly ineligible;
their diagnostic minimum-norm result cannot silently become a selected model.

### Action-energy control

This audit fixes `w0 = 0` and chooses `L = diag(sigma) V^T` from the training-only
SVD `sqrt(W) A = U diag(sigma) V^T`. Therefore

\[
J_\alpha^{\rm action}(w)=\|\sqrt W(Aw-b)\|_2^2+
\alpha\|\sqrt W A w\|_2^2.
\]

This penalizes physical correction energy on the training distribution, rather
than coefficient size in an arbitrary feature coordinate system. It is a
changed prior toward the ordered-two backbone, not a numerical repair. It uses
no validation or diagnostic labels to construct `L`.

For a full-column-rank design,
`w_alpha = w_OLS / (1 + alpha)`: the control uniformly shrinks the unregularized
fitted action. This identity is retained as a numerical diagnostic. It is not a
richer PDE-derived prior, a BCH/Magnus correction or a novelty claim. The action
arm explicitly requires original rank 40. A rank-deficient arm remains ineligible
rather than silently acquiring a ridge or changing its nullspace convention.

The projection uses an unregularized SVD with rank cutoff
`eps64 * max(A.shape) * sigma_max`. Its residual and stability are diagnostic;
it neither defines a full-operator lower bound nor proves generalization.

## Selection, stability and protected access

For each cell/family, the selected candidate minimizes validation physical MSE
among zero and numerically eligible positive penalties. Zero wins an exact tie;
otherwise the listed strength order breaks ties. Eligibility requires finite
coefficients and predictions, a numerically full-rank augmented problem, the
original-objective gap and baseline nonregression checks, and agreement between
retained affine and five-branch prediction arithmetic. The relative objective
slack is `1e-8`, with `1e-28` absolute slack; these are numerical checks, not
scientific accuracy thresholds. Prediction arithmetic uses `128 * eps64` times
its declared accumulation scale. The `lambda=1e-10` fit is compared with the
saved original coefficients on the validation design before protected access.

All six selections and every candidate, including failures, are written and
sealed before diagnostic or training-withheld numeric arrays are accessible.
The input adapter requires that exact new-run selection file and checks it again
during protected reads. An old P1 selection cannot release this new audit.
Source and input guards are retained through the native lifecycle.

Sensitivity remains visible without introducing an arbitrary accuracy gate.
Let `E = sqrt(W_val) A_val`. For coefficient ridge, the label-to-validation
prediction derivative is

\[
E V\operatorname{diag}\left(\frac{\sigma_j}{\sigma_j^2+\lambda}\right)U^T.
\]

For the full-rank action-energy arm it uses
`1 / ((1 + alpha) sigma_j)`; projection uses its explicit rank cutoff.
A thin SVD computes the operator gain without creating a dense
validation-by-training matrix. The training estimated-reference radius is
`sqrt(sum_q parent_query_weight_q * uncertainty_q^2)`. Multiplying by the gain
estimates worst prediction RMS over that ball. The actual retained
`refined_U - independent_U` label difference is propagated separately.

Both changes are reported relative to validation error, target remainder and
reference uncertainty. The estimated ball is not a proven admissible-error set.
A small objective gradient does not imply stable coefficients or generalization.
Sensitivity does not secretly change the declared selection rule: unavailable
or large sensitivity is retained as evidence requiring review.

## Metrics, costs and interpretation

The audit scores the retained 40-column affine prediction and checks its
five-branch contraction against the declared arithmetic tolerance. That tolerance
scales with the absolute coefficient/design accumulation, so cancellation can
increase it. The saved discrepancy and tolerance remain visible for review.
It does not measure a newly deployed full response
call. Original classical controls use the saved native predictions. Their P1
timings remain historical context; no new timing or speedup is inferred from
unchanged architecture or coefficient-only execution.

For each action error `e`, report physical RMS, maximum, signed mean, reference
uncertainty, and explicit ratios to `U`, `(U-P)z` and `(U-G)z`. Feedback uncertainty
includes both the full and matched-mean reference estimates. A ratio is unresolved
when its signal does not exceed ten estimated uncertainties. Missing and failed
slots remain in all planned denominators.

The exact decomposition is

\[
\mathrm{MSE}=\mathbb E_q\langle e_q\rangle_x^2+
\mathbb E_q\langle(e_q-\langle e_q\rangle_x)^2\rangle_x.
\]

Queries are averaged within each parent, then parents equally. The new schema
publishes `mean_component_mse/rms` and `spatial_centered_mse/rms`; its centered
aliases mean within-action spatial error. P1's old pooled-centering field and
files are not rewritten. Training-withheld parents inside/outside each cell's
cohort are separate groups. Eight diagnostic parents remain only eight units;
there are no confidence intervals, significance tests or population claims.

Cost records distinguish reused original reference/design time from new input
reads, training penalty setup, candidate computation, sensitivity and evaluation.
`fitting_seconds` is inclusive candidate computation, including validation and
sensitivity; `sensitivity_seconds` is a subset. These are not additive to wall
time. Worker CPU/RSS and Raven supervisor telemetry have different scopes and
are not per-method peak memory or GPU measurements.

## Native operation and portable outputs

The following retains the launch interface for a separately authorized repeat.
Run only after earlier jobs using this checkout are terminal. The project venv,
Fedora Slurm profile and installed Raven are required. Preparation uses the
committed reviewed P1 archive, so no manual old-run path is needed.

```bash
if cd /home/rahel/TDN &&
   git pull --ff-only origin main &&
   contract="$(.venv/bin/python scripts/response_preconditioner_regularization_native.py prepare)"; then
  raven launch scripts/response_preconditioner_regularization.sh \
    --project "$PWD" --recipe response-preconditioner-regularization \
    --sbatch=--partition=local --sbatch="--chdir=$PWD" \
    -- --contract "$contract"
else
  unset contract
  printf 'Preparation failed; no job submitted. See the error above.\n' >&2
fi
```

Use the CPU partition/account from `.tdn/fedora-slurm.json` when different.
Insert `--check` before `--` for a launch preview. Do not pass `--local`: production
fitting requires the genuine allocation. The guarded block keeps an interactive
terminal open on errors. Do not source the launcher or update source/venv while
its job is active. Default mode verifies the exact mandatory roster, then runs
the audit in that same attempt. `--mode verify` is available; a study-only mode
or a verification release from another attempt is not.

For local implementation checks, without fitting the retained P1 bank:

```bash
if cd /home/rahel/TDN; then
  mkdir -p .runtime/response-preconditioner-regularization-local &&
  TMPDIR="$PWD/.runtime/response-preconditioner-regularization-local" \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python scripts/response_preconditioner_regularization.py verify --device cpu
fi
```

The recipe enables experiment, evidence, artifacts and CPU/memory observation;
there is one job, no array, GPU collection or tracing. Full post-stage source and
input checks are persisted before research publication and optional packing.
Compact series under `regularization/` show validation paths, fit diagnostics,
physical errors, mean/spatial components, counts and relevant costs. Complete
numeric tables and figures remain portable artifacts. An evidence-packing failure
retains the numerical stage's separate execution status and already published
research metrics, while the wrapper still records its failure.

Raven format-checks at most 1 MiB of content per output folder. Larger JSON/CSV
tables remain required downloadable artifacts without claiming Raven parsed
them; the native artifact inventories hash their full contents. The audit also
retains the original study error when reporting or input closing fails, with
secondary failures recorded separately.

| Run path | Meaning |
| --- | --- |
| `outputs/native/status.json`, `contract.json`, `input_identity.json`, `post-study-identity.json` | Native lifecycle and exact source/input bindings |
| `outputs/native/telemetry/` | Allocated worker process-tree CPU/RSS samples and coverage limits |
| `outputs/verification/` | Exact roster, JUnit, summary, source and verifier artifacts |
| `outputs/study/protocol.json`, `source*.json`, `software.json` | Frozen study contract and execution provenance |
| `outputs/study/inputs/` | Consumed original query/prediction arrays, provenance, access journal and file hashes; no old orphan ZIP |
| `outputs/study/fits/`, `selection.json`, `sensitivity.json` | Every candidate, selected readouts, seals, spectra, objectives and uncertainty scenarios |
| `outputs/study/predictions/`, `metrics.json`, `metrics.csv` | Numeric replay and every expected method/query result |
| `outputs/study/learning_curves.*`, `parent_metrics.*`, `figures/` | Parent-balanced comparisons and portable charts |
| `outputs/study/denominators.json`, `costs.json`, `summary.json`, `decisions.json`, `artifacts.json` | Completeness, costs, execution versus research status, file hashes |
| `logs/` | Verification and study streams; scheduler streams also remain under checkout `logs/response-preconditioner-regularization/` |

After the attempt is terminal, export its exact run directory:

```bash
if cd /home/rahel/TDN; then
  .venv/bin/python scripts/response_preconditioner_regularization_native.py export \
    --run runs/launch-EXACT-a1
fi
```

Upload the printed export directory's `README.txt`, `index.json` and ZIP or every
numbered part. Upload parts are at most 30 MiB. Export streams files and uses
explicit 16 MiB inventory budgets, preserving original data and terminal status.
No research is rerun during export. The new portable study contains roughly
62 MB of reused inputs before its own results, rather than the original redundant
96 MB packing ZIP. No Git-history access is required on the native machine to
validate the pinned archive.

## Review gate and term guide

First verify input/source identity, all candidate outcomes, selection ordering,
missing slots and numerical optimality. Then compare validation/diagnostic error
and sensitivity against the original fit and strongest classical controls.
Interpret outcomes conditionally:

- Stable improvement from weaker penalties supports revising the regularization
  rule before adding parameters.
- Improvement from action-energy shrinkage supports examining coordinate-dependent
  coefficient bias; it does not prove a new physical law.
- A good projection with poor selected positive fits points toward penalty or
  selection effects. A poor global projection can still reflect a restrictive
  coefficient map; it does not isolate the local branch span.
- Further structural claims require a separately frozen local/full-operator audit
  and an independently derived nonautonomous analytic control.

Any next data acquisition, physical prior, architectural change or confirmation
requires its own frozen question and budget. Historical B STOP and the absence
of demonstrated matched-tolerance utility remain unchanged.

| Term | Meaning |
| --- | --- |
| P1R | Retained-data regularization audit following P1; separate from R8 encoder work |
| `lambda`, `alpha` | Coefficient-ridge strength / action-energy strength; units differ |
| `L`, `w0` | Generalized penalty operator / preferred coefficient vector, fixed to zero in this audit |
| `Aw`, `bw`, `E` | `sqrt(W_train) A`, weighted training target, weighted validation design |
| `sigma`, `U`, `V` in SVD | Singular values and left/right singular vectors; SVD `U` differs from tangent response operator `U` |
| OLS / projection | Ordinary unregularized least squares on training labels, used only as a diagnostic |
| Curvature `mu` | Smallest eigenvalue of the quadratic normal matrix; half the Hessian's minimum eigenvalue |
| Effective degrees of freedom | Trace of the regularized linear smoother, not a count of physical laws |
| Prediction gain | Norm of the map from weighted training-label perturbations to validation-action changes |
| Estimated ball | Empirical uncertainty radius used for sensitivity, without a certified containment claim |
| Seal | Hash of persisted choices, required before protected numeric access |
| Retained replay | Computation from already saved arrays; distinct from new teachers, full inference timing or confirmation |

The [P1 term legend](../results/native-response-preconditioner-scaling-2944/README.md#9-term-and-abbreviation-legend)
provides the remaining physical, solver, metric and resource definitions.
