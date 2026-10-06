# Runtime interaction correction: bounded CPU screen

This implements the next experiment proposed in [the CARC mechanism
review](MECHANISM_REVIEW.md). It computes nonlinear interaction features from
the input field, without generating-frequency labels, per-case fitted
coefficients, or neural training. It does not change the frozen mechanism audit.

[Cloud CPU validation](../results/interaction-screen-validation.json) completed
502/502 records with all 32 references accepted in 3.61 seconds of numerical
work. The integrated regression suite passed 492 tests (one optional schema
test skipped); 53 affected reporting tests passed after export compaction.
Independent native Tower validation and 37 schema checks passed, and all
10,882 scalar values and 502 case notes verified with zero export omissions.
These are implementation results, not a learned-solver efficiency claim.

## Run on CARC

Use the existing verified project Python venv. One CPU allocation runs focused
correctness tests followed by the frozen comparison: **4 CPUs, 16 GiB, 30
minutes**, charged to **anakano_81**. Numerical work has a **1,200-second cap**.
There is no GPU successor, pending-job cap, automatic resubmission, or training.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main
bash scripts/carc_interactions.sh
bash scripts/carc_interactions.sh --submit

bash scripts/carc_interactions.sh status latest
bash scripts/carc_interactions.sh logs latest --lines 200
bash scripts/carc_interactions.sh collect latest
```

The first invocation previews the request; only `--submit` submits it. Record
the printed run ID and substitute it for `latest` when other research workflows
are active, since they share the latest pointer. Use a fresh run after a code
change. Completed experiments are never overwritten or relabeled.

The full protocol declares **502 case records**: 13 methods and one reference
for each of 32 diagnostic problems, plus 54 exact-limit/precision controls.
`--smoke` uses five predeclared problems and all controls (124 records) with a
180-second numerical budget. It is an integration subset, not a different fit.

The wrappers enforce the existing CARC identity, root, account, allocation,
venv and shared-lock checks. All caches, temporary files and outputs stay
under `/home1/aadaniel/projects/TDN`. Do not source `common.sh` or execute the
numerical entrypoint on a CARC login node.

## Construction and controls

For the same periodic central-difference reaction–diffusion system as the
previous audit, write `c=mean(u)`, `v=u-c`, `E_t=exp(t*kappa*Delta_d)` and
`J_c(t)=R'_t(c)`, where `R` is the exact logistic reaction flow. The quadratic
coupled response is

\[
Q_h(v)=-rJ_c(h)\int_0^h J_c(s)E_{h-s}[(E_s v)^2]\,ds.
\]

The implementation subtracts the quadratic response already present in
Strang. With `F_s=E_(h-s)[(E_s v)^2]`, `B=E_h(v^2)`, `C=(E_h v)^2`, and
`W2=integral_(h/2)^h J_c(s) ds`, its anchored form is

\[
d_h=-rJ_c(h)\left[\int_0^h J_c(s)(F_s-B)\,ds+W2(B-C)\right].
\]

Three- and five-node Gauss–Legendre rules approximate the integral. The
reaction weight is analytic, with stable exponential evaluation. No extra
variation gate multiplies this already-quadratic feature. Uniform states and
zero diffusion/reaction retain the physical split. The construction represents
the second variation around a homogeneous background; it is a truncated
approximation on finite-amplitude fields, not their exact finite-time defect.
Higher-amplitude terms can still contribute at cubic order in time; this
construction alone does not promote the full solver to fourth-order accuracy.

The screen compares Strang, ETDRK2, ETDRK4 and interaction-corrected Strang.
Correction ablations include additive, capacity-bounded and projected outputs,
and full, zero-mean and mean-only corrections. The mean ablations separate
reaction-driven mean change from spatial redistribution. Projection and
capacity are controls, not claims of novel architectures or long-time stability.

The frozen bank includes mixed Fourier fields, independently crossed reaction
and diffusion rates, several times and mean states, boundary/remote patterns,
multiple one-dimensional grids, small two-dimensional cases and rollouts.
It uses the discrete Laplacian consistently in the baseline, features and
teacher. An independent refined FP64 coupled RK4 teacher records uncertainty
and acceptance. Rejected or unresolved references remain inconclusive.

Per-case reports retain error, physical admissibility, mean error, actual
inference time and operation counts. Transform and quadrature costs belong to
the correction. Timing is local CPU implementation evidence, not A100 timing,
an asymptotic complexity proof, or a deployable tolerance controller.

For nondegenerate single calls, GL3/GL5 use 18/26 forward-plus-inverse Fourier
transforms, compared with 5 for ETDRK2 and 9 for ETDRK4. These counts are measured
by the implementation and tested against actual transform calls. Runtime work
accumulates across rollout steps; failed calls explicitly mark incomplete
counters. Remaining relative FP32 cancellation at very small steps is reported
separately from absolute state error.

## Tower and review artifacts

The worker prints its exact `TDN_TOWER_DIR`. Use that report directory rather
than an unrelated historical `latest` report:

```bash
REPORT=/home1/aadaniel/projects/TDN/runs/REPLACE_RUN_ID/tower/REPLACE_REPORT_ID
bash scripts/tower.sh validate "$REPORT"
bash scripts/tower.sh show "$REPORT"
```

Live metrics are in `$REPORT/metrics.jsonl`. Scientific scalar metrics are
in `$REPORT/outputs/mechanisms.csv`, registered as `analytics.mechanisms`.
Each row preserves the case, variant, inputs, metric type and exact JSON pointer
into `interaction-screen.json`. Summary duplicates do not generate duplicate
metrics. The case note appears on its first metric row; every row's case pointer
resolves the full original note. This avoids repeating long prose enough times
to exhaust Tower's shared read budget. Tower itself is unchanged.

The experiment directory contains its frozen protocol and effective config,
canonical `interaction-screen.json`, readable and machine-readable summaries,
and the manifest/completion marker binding its scientific artifacts. Stage
status and Tower separate computational completion from scientific advantage.
Negative comparisons remain visible; no outcome starts a training campaign.

## Interpretation and next decision

This study has no fitted model, training split or checkpoint selection. Its
different regimes are diagnostic coverage, not measured learned
generalization. The reference is the accurate flow of the specified discrete
system, not a spatially converged continuum-PDE solution. No result establishes
superiority to FNO, INC or a foundation model.

Advance only if field-derived corrections improve finite-amplitude accuracy
under explicit work accounting and the intended limits remain correct. Keep
the unlearned correction and ETDRK4 as controls in any subsequent matched
training comparison. A slow but accurate feature may justify approximation;
a poor finite-amplitude feature does not justify longer training by itself.

Recent prior art informs these controls: [INC](https://github.com/tum-pbs/INC)
already embeds learned sources in exponential integrators, and
[Exponax](https://github.com/Ceyron/exponax) implements ETDRK solvers for this PDE
family. The new experiment tests the specific nonlinear interaction structure;
it does not claim exponential integration or learned numerical correction is new.

For development in a separate non-CARC checkout, use
`bash scripts/interactions_local.sh --run-dir runs/interaction-local-001`.
The wrapper selects the existing project venv and a fresh project-local run.
It rejects CARC login-node use and Slurm/desktop overrides. Cloud CPU evidence
does not validate actual CARC execution or A100 performance.
