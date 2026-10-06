# Time to accuracy: bounded CPU experiment

The [CARC interaction review](INTERACTION_REVIEW.md) found a useful correction
with a substantial runtime cost. This experiment tests whether an inexpensive
mean correction retains enough accuracy to reduce time to a specified error,
and whether full GL3 or GL5 earns its additional work. ETDRK4 and smaller-step
Strang remain controls, including the earlier two-dimensional failures.

## Run on CARC

One allocation runs correctness tests and the complete frozen experiment:
**4 CPUs, 16 GiB, 30 minutes**, account **anakano_81**, user **aadaniel**.
The numerical program has a **1,200-second budget**. It uses the existing
project Python venv and retains all files beneath the project directory.

```bash
cd /home1/aadaniel/projects/TDN
git pull --ff-only origin main
bash scripts/carc_work_precision.sh
bash scripts/carc_work_precision.sh --submit

bash scripts/carc_work_precision.sh status latest
bash scripts/carc_work_precision.sh logs latest --lines 200
bash scripts/carc_work_precision.sh collect latest
```

The first invocation previews the allocation; `--submit` submits one CPU job.
Save its printed run ID and replace `latest` with that ID when using other
research workflows: their latest pointer is shared. There is no pending-job
cap, GPU successor, automatic resubmission or neural training. Run the wrappers
with `bash`; do not source `common.sh` or run numerical work on the login node.
Existing results are preserved; code changes require a fresh run.

## Hypotheses and controls

The periodic discrete system is
`u' = kappa*Delta_d u + r*u*(1-u)`. All seven methods solve this same system:

| Method | Purpose | Transforms per active step |
| --- | --- | ---: |
| Strang | Exact subflow splitting, with smaller steps allowed | 2 |
| ETDRK2 | Lower-order exponential control | 5 |
| ETDRK4 | Higher-order exponential control | 9 |
| GL3 | Full additive quadratic interaction correction | 18 |
| GL5 | Quadrature-refinement control | 26 |
| GL3 mean, full evaluation | Original mean ablation and parity control | 18 |
| GL3 mean, spectral evaluation | Inexpensive mean correction | 3 |

The GL counts include their Strang base and describe nonuniform fields with positive
reaction, diffusion and time. Degenerate exact limits perform less work.
Counts accumulate across actual steps and exclude coefficient preparation,
which performs no transforms. They are implementation costs, not FLOP counts.

For `c=mean(u)`, `v=u-c`, `N` spatial cells, diffusion eigenvalues
`lambda_k<=0` of `kappa*Delta_d` (including `kappa`), and
`P_k=|FFT(v)_k|^2/N^2`, Parseval's identity gives

\[
D(s)=\sum_k P_k\operatorname{expm1}(2\lambda_k s),\qquad
\overline{d_h^{GL}}=-rJ_c(h)
\left[\sum_j\omega_jJ_c(s_j)D(s_j)-W_2D(h)\right].
\]

Here `J_c(s)` is the derivative of the exact logistic reaction flow and
`W2=integral_(h/2)^h J_c(s) ds`. The correction needs one input FFT and no
inverse transform. Its scalar result is added to every spatial cell. It
reproduces the mean of the existing **raw quadratic GL correction** in exact
arithmetic. It does not reproduce its spatial component, the exact nonlinear
mean, or the result of nonlinear clipping. The experiment checks numerical
parity in FP32 and FP64, including complete rollouts.
For FP32 inputs, the spectral implementation uses FP64 correction coefficients
and reductions to limit small-step cancellation while retaining the FP32 FFT
and output. Its metadata records this mixed arithmetic. The timed comparison
uses FP64 states for every method.

The hypotheses are separate:

- Spectral evaluation preserves the original mean correction while reducing
  work. This is an algebraic correctness and implementation-cost question.
- Removing mean bias is sufficient for some accuracy targets. A cheap
  correction may still lose when spatial error dominates.
- Full GL3 earns its extra cost at tighter targets or larger amplitudes.
  ETDRK4 may remain preferable in two dimensions.
- GL5 improves the frontier when quadrature error matters. Otherwise its
  additional nodes only increase cost.

No hypothesis requires a positive result. Higher-amplitude terms can already
enter at order `h^3`; the full quadratic correction does not automatically
raise the solver's order to four.

## Frozen comparisons

The full bank has **24 diagnostic conditions**: 20 fresh one- and
two-dimensional fields and four historical controls. The historical controls
retain the finite-amplitude counterexample and all three earlier 2D regimes.
The fresh conditions vary grid, mean, amplitude, diffusion, reaction and final
time. This is a declared stress bank, not a full factorial design or an
independent statistical sample. The exact fields and schedules are saved in
`protocol.json` before numerical execution.

Every method integrates each condition to its fixed final time using
**1, 2, 4, 8, 16 or 32 steps**. RMS and maximum error use separate tolerance
tracks: **0.002, 0.0002, 0.00002 and 0.000002**. A refined FP64 coupled RK4
reference solves the same discrete system. Its numerical uncertainty is
added to error for feasibility; the maximum-norm estimate uses
`sqrt(N)*RMS uncertainty`. These are refinement estimates, not rigorous error
certificates or continuum convergence results. Rejected references cannot
produce feasible frontier entries.

Each method may cache state-independent coefficients for its fixed step,
grid, dtype and device. State means, reaction Jacobians and mean-dependent
weights are recomputed for every state. One preparation measurement is recorded
separately, in a predetermined rotating method order.
After one warmup rollout, five complete rollouts are timed in a predetermined
rotating method order. Timing includes step validation and work accounting;
reference construction, error evaluation and frontier selection are outside
the measured rollout. One CPU thread is used for numerical kernels.

Prepared median, minimum and maximum times and all raw repetitions are
retained. Setup-inclusive time means preparation plus the warmed median; it
excludes warmup and does not claim physical cold-cache timing. Repetitions
describe timing variability on this machine and are not independent problem
samples. Cloud CPU evidence must remain distinct from CARC CPU or A100 timing.

For each method, condition, error norm and tolerance, the report selects the
fastest feasible measured step count on the declared grid. This is a
**post-hoc diagnostic frontier**, not a deployable step-size controller.
The setup-inclusive winner is selected separately from the same feasible
candidates and reported alongside the prepared-time winner.
Missing feasible entries remain explicit. Invalid or interrupted trajectories
cannot acquire a final-horizon error or win by stopping early.

`--smoke` uses four predeclared conditions with the same methods, step counts,
tolerances and timing repetitions, under a 180-second numerical budget.
Signals, budget exhaustion and failures retain partial evidence without a
successful completion seal. Computational completion is separate from a
scientific advantage.

## Tower and review

The worker prints its exact `TDN_TOWER_DIR`. The report provides:

- `metrics.jsonl`: live progress and computational status.
- `outputs/work_precision.csv`: one row per measured candidate.
- `outputs/tolerance_frontiers.csv`: each feasible selected candidate or
  explicit absence of a feasible candidate.
- `outputs/work_precision_checks.csv`: reference and parity evidence.

CSV rows retain exact pointers to the authoritative `work-precision.json`,
which includes raw timing repetitions and work counters. Preparation costs,
error norms, infeasibility and the limits of post-hoc selection remain visible.
The project exports these files using Tower's existing features; Tower itself
is unchanged.

```bash
REPORT=/home1/aadaniel/projects/TDN/runs/REPLACE_RUN_ID/tower/REPLACE_REPORT_ID
bash scripts/tower.sh validate "$REPORT"
bash scripts/tower.sh show "$REPORT"
```

Collection includes the frozen protocol, configuration, canonical results,
summaries, integrity manifest, tests, logs and Tower report. Upload the printed
review archive to compare the CARC measurements. These results decide whether
to retain the mean correction, develop a compact spatial remainder, or use
ETDRK4 as the core for a later matched neural comparison. They cannot establish
an advantage over FNO without that subsequent comparison.

For development in a separate non-CARC checkout, run
`bash scripts/work_precision_local.sh --run-dir runs/work-precision-local-001`.
It selects the project venv, contains storage in the checkout and rejects
CARC login-node use or Slurm/desktop overrides.
