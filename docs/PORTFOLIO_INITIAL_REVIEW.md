# TDN portfolio: first evidence review

Review date: **2026-10-08, America/Los_Angeles**. This review separates the new CPU development diagnostics from the end-to-end CPU smoke run. Neither is the planned full desktop GPU comparison. The frozen research questions, claim thresholds and branch retirement rules remain in [the research design](PORTFOLIO_RESEARCH.md), [hypothesis register](PORTFOLIO_HYPOTHESES.json) and [full protocol](../results/portfolio-development/full-protocol.json).

**Research judgment.** Continue the controlled attribution campaign. Preserve nonlinear products before output compression. The most informative new numerical evidence points toward the *missing interaction order and spatial target*, rather than inadequate quadrature alone, as important sources of remaining error. There is no new evidence that a neural conditioner is necessary or that the complete TDN solver is faster. Of the speculative prototypes, coarse-state ambiguity provides the clearest reason for a further small experiment. The current temporal encoding and computation selector fail their cost gates and should not receive larger experiments without a changed computational premise.

This is a narrower and more useful direction than increasing network size. A successful result may be a small fitted correction, an improved analytic formula, or an efficient closure. The full controlled campaign will determine which interpretation survives.

## What was actually run

| Evidence set | Execution and scope | What it establishes | What it does not establish |
|---|---|---|---|
| Development diagnostics v2 | CPU; five sealed units: audit, diagnose, X01, X02, X03; no neural training or confirmation | Implemented formulas, targeted numerical comparisons and bounded prototype outcomes on their declared synthetic fields | GPU performance; a trained-model advantage; population-level superiority |
| Local smoke v2 | CPU; all 14 planned units completed, including report; 23 method families instantiated on two equation tracks; 46 selected instances; 2,024 endpoint rows from two independent fields | The full prepare/train/freeze/confirm/aggregate/scaling/report path executes, preserves controls and reports absent evidence honestly | Adequate optimization, data efficiency, robust generalization, or a credible reproduction of published FNO results |
| Full Fedora campaign | Prepared, frozen protocol; not executed in this environment | Reproducible next experiments with explicit budgets and fresh field identities | Any full or native GPU result |

The smoke run produced a [52-panel atlas](../results/portfolio-development/smoke/portfolio-atlas.pdf) and [28 accuracy/20 cost claim records](../results/portfolio-development/smoke/aggregate-claims.json). **All 48 primary claim records are NA.** Its two independent fields are below the declared minimum of five for these intervals. Thousands of repeated schedules, grids and model comparisons do not increase that denominator. The report's successful verification is a computational outcome. The [raw chart data](../results/portfolio-development/smoke/chart-data.json.gz) retain observations behind the figures; translucent observed learning ranges are not confidence intervals.

The smoke training cap is two updates, with two training parents and one training seed. Its selected catalog contains 20 frozen controls, 17 selected initializations and nine fitted checkpoints. Large error ratios against a two-update FNO are not evidence of a competitive neural advantage. Selecting initialization when training does not improve validation is the correct behavior; these counts do not diagnose the cause of training failure at a realistic budget.

### Provenance and scope of published copies

The authoritative source revision for both v2 runs is `4b6b5ec1e3055bb53b6507ae4775c7bed1dc91b6`, with scientific source-tree SHA-256 `a83f6a15a41d35ae17ccc08babc8304204a5f12e5c82d0d96399cb3982d45861`. The cloud environment used Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.2.6 and SciPy 1.15.3. These measurements do not describe the RTX 4090.

Original run roots are `runs/portfolio-development-diagnostics-v2` and `runs/portfolio-local-review-v2`. Curated development files and their original provenance are in [the publication index](../results/portfolio-development/development/publication-index.json); the [validation record](../results/portfolio-development/validation.json) identifies published artifacts and hashes. The published subset preserves source measurements and original seals but omits some execution inputs. It is not a standalone copy of every original execution bundle, and should not be presented as independently rerunnable solely from those copies.

The development protocol hash is `70c1bb2dbf778adad6342364ea919de667770da9fde4b6e9d9f3b337eeebd465`; the smoke protocol hash is `a334aa226f0981961372fb65075f6eee4d089550329e798876cd3604b22e0555`. Prototype numbers below come from the larger **development** diagnostics, not from the smaller smoke variants embedded in the atlas.

## The numerical bottleneck is now better localized

The diagnostic uses a 16×16 periodic grid, domain 1×1, diffusivity 0.004, reaction rate 3 and endpoint time 0.12. There is one diagnostic field per regime, not an independent statistical cohort. All errors in the next table are RMS endpoint errors against the refined reference of the specified same-grid equation. “Continuum track” here means the model's spectral Galerkin equation; a separate spatial refinement comparison is still required before calling an endpoint a continuum-accurate solution.

| Development field | Equation track | One-step DF | Full four-node quadratic correction | Quadratic + cubic correction |
|---|---|---:|---:|---:|
| Low frequency | Discrete | 1.58e−7 | 1.42e−7 | 1.24e−8 |
| Low frequency | Galerkin | 1.76e−7 | 1.57e−7 | 1.66e−8 |
| High pair | Discrete | 2.70e−5 | 3.27e−6 | 9.37e−8 |
| High pair | Galerkin | 4.94e−5 | 3.65e−6 | 7.77e−8 |
| Rough | Discrete | 6.73e−6 | 7.97e−7 | 8.15e−8 |
| Rough | Galerkin | 1.31e−5 | 7.05e−7 | 9.12e−8 |
| Near Nyquist | Discrete | 2.60e−5 | 1.32e−6 | 6.06e−8 |
| Near Nyquist | Galerkin | 5.33e−5 | 1.39e−6 | 4.03e−8 |

Lower is better. These are matched fields, equations and endpoints, **not matched inference costs**. Raw measurements and component labels are in [diagnostic_rows.json](../results/portfolio-development/development/diagnose/diagnostic_rows.json). The full quadratic and cubic rules have more work than the historical learned pair, so the table identifies attainable accuracy and a useful control rather than a learned or computational win.

The high-pair Galerkin field is particularly informative:

| Targeted comparison | Measured RMS | Interpretation |
|---|---:|---|
| DF with 1, 2, 4, 8 steps | 4.94e−5, 1.26e−5, 3.18e−6, 7.95e−7 | Temporal refinement behaves consistently with a second-order split on this case. |
| Two-node versus sixteen-node quadratic correction | 2.68e−7 | Quadrature is a measurable but smaller error than the full quadratic endpoint defect. |
| Four-node versus sixteen-node quadratic correction | 6.60e−13 | More nodes are unlikely to fix the remaining 3.65e−6 error of the four-node rule here. |
| Premature input compression versus output-only compression of the quadratic correction | 4.89e−5 | Discarding high-frequency inputs removes a useful interaction contribution. |
| Full versus output-projected two-node correction | 0 on this field | Output compression is harmless in this particular comparison; it is not universally lossless. |
| Same-grid Galerkin versus fine-grid projected reference | 4.69e−6 | Spatial discretization already exceeds the improved quadratic+cubic same-grid error. |
| 2N versus 4N projected spatial references | 1.69e−10 | Actual spatial refinement supports the spatial-error diagnosis for this field. |
| Finite-difference nodal equation versus fine continuum reference | 6.96e−3 | A difference between specified equations; it must not be charged as a temporal integration defect of the discrete solver. |

The final temporal reference difference is 7.53e−14 RMS for this field; the recorded uncertainty floor is 1e−12. This is measured refinement evidence and an uncertainty estimate, not a theorem or a certified error bound. The component contrasts are not additive: changing one approximation can alter cancellation with another.

### What the math suggests

Writing a solution around its evolving mean as `u = c + v`, the quadratic interaction correction accounts for terms of order `v²`. Its transport factors and signed split-defect subtraction are already known physical structure. A more accurate time quadrature computes that same quadratic object more accurately; it does not create the missing cubic interaction shape. Likewise, multiplying the quadratic object by one scalar gain can only change its amplitude. It cannot represent an arbitrary cubic residual whose spatial pattern is linearly independent of that object.

For this field, four-node quadrature error is near the reference floor while quadratic+cubic correction lowers same-grid RMS from 3.65e−6 to 7.77e−8, about 47-fold. This weakens the explanation that “we only need to learn better quadrature nodes.” It supports studying whether an inexpensive selected higher-order contribution can reproduce some of that improvement. It does **not** establish that a learned approximation is cheaper than evaluating the analytic cubic rule or that this ratio persists across fields.

The exact next discrimination is between: (i) a scalar correction to the existing quadratic term; (ii) the best bounded non-neural scalar/affine fit; (iii) a learned residual anchored to normalized quadrature; and (iv) the analytic quadratic+cubic control. These are already represented in the controlled campaign. An explicitly distilled or selectively computed cubic residual would be a **new development hypothesis**, requiring its own registered test and fresh confirmation if promoted.

For continuum claims, improving temporal accuracy below the spatial discrepancy is insufficient. Either retain a same-discrete-equation claim or add a separately controlled closure/refinement mechanism. Training a correction against a finer continuum target silently changes the learning problem and would confound temporal defect correction with spatial compensation.

## What the first profiling pass says

The diagnostic frozen CPU profiling uses small 16×16 fields and FP64, with five warm repeats per field. Median warm latency across the four field-specific medians is:

| Frozen method | Median warm latency |
|---|---:|
| Analytic DF | 0.330 ms |
| ETDRK4 | 0.224 ms |
| Normalized two-node correction | 1.019 ms |
| Conditioned two-node correction | 1.317 ms |
| Full four-node quadratic correction | 1.167 ms |
| Small FNO diagnostic configuration | 1.294 ms |

Lower is better. These timings use the diagnostic configurations and are **not accuracy-qualified solver rankings**. The FNO diagnostic width/depth differ from competitive full-training settings. See [profile_rows.json](../results/portfolio-development/development/diagnose/profile_rows.json) for individual repetitions, configurations and instrumented components.

For the high-pair conditioned case, the instrumented self-CPU-time distribution assigns roughly 16% to transforms, 16% to pointwise operations, 12% to copies, 6% to nonlinear operations and 5% to allocations. The learned dense layers account for only about 0.2%; other tensor operations account for the remainder. This supports measuring tensor traffic, transforms and orchestration before increasing network capacity. These percentages exclude Python time and profiling overhead and are not a complete deployment-time breakdown. They cannot identify the dominant RTX 4090 bottleneck.

The conditioned pair must beat the correctly normalized frozen rule and the strongest accuracy-qualified analytic endpoint menu before a deployment policy is warranted. Additional rejection, estimation and fallback work would consume an already unproven margin.

## Protected exploratory results

The prototypes deliberately test different representations and information assumptions. Their GOOD/BAD/NA checks are evidence for their stated local question; they are not three new verified PDE solvers. Their exact equations, controls and prior-art scope are documented in [the exploration notes](PORTFOLIO_EXPLORATION.md).

### X01: a reusable interaction-time kernel

This is a fixed one-dimensional, constant-generator **second-Picard quadratic response**, with 361 mode pairs and a degree-12 Chebyshev time representation. It is not the full logistic DF model. On the declared 20 horizons, including zero, the encoded response has maximum relative RMS difference 5.08e−16 against the exact cached pair response. The corresponding DOP853 dense-output difference is 1.10e−11; FFT four-node quadrature is 1.15e−12.

The representation is accurate, but the measured cost does not support the intended deployment. The compact representation takes about 746.9 μs to build and 24.93 μs per scalar query; the exact cached representation takes about 2.67 μs to build and 25.29 μs per query. The modeled break-even is **2,062 queries**, above the declared limit of 32. Even reusing the operator-only basis gives 319 queries. DOP853 dense output queries take about 18.54 μs, with a more expensive approximately 1.52 ms build.

These break-even values use small CPU timing differences and a scalar-query cost model, not a measured batched deployment curve or a timing confidence interval. [The raw amortization record](../results/portfolio-development/development/explore-X01/temporal_amortization.json) preserves assumptions. **Decision: retain the accurate representation result; stop the current cost claim.** Further investment needs a reason that structured pair reduction, fused evaluation, or thousands of real same-state queries changes the cost balance. A rollout refresh generally rebuilds state-dependent coefficients, so reuse does not automatically persist across steps.

### X02: coarse twins and useful history

This experiment constructs fields with identical retained coarse states but different unresolved components. On 20 held-out independent twin parents, the retained initial-state difference is zero while the minimum future target separation is 9.60e−6. An instantaneous map receiving only that coarse state cannot identify both answers. This is a direct information obstruction, not an optimization failure.

The target is one complex retained Fourier coefficient in a small one-dimensional Galerkin problem. Held-out RMS is 3.23e−5 for the instantaneous fit, 3.39e−7 for the history persistence control, and 9.73e−11 for the fitted memory rule. The fitted rule improves a deliberately favorable and narrow synthetic setting; it is not evidence that an arbitrary memory architecture solves closure. There are 20 independent held-out parents, not 40 independent twin states. An eight-/sixteen-step RK4 versus DOP853 subset cross-check is near machine precision, but does not certify continuum dynamics.

The short history is generated by a backward reference solve. Its measured acquisition cost is approximately 0.370 ms per state, compared with 1.37 μs for the fitted query. That cost must be charged if history is not naturally available from a prior forward rollout. See [closure_fit.json](../results/portfolio-development/development/explore-X02/closure_fit.json) and [closure_rows.json](../results/portfolio-development/development/explore-X02/closure_rows.json).

**Decision: advance one bounded causal-history pilot.** Generate histories forward, vary unresolved phases and physical parameters, predict more than one retained mode, and test rollout stability. Compare memory against inexpensive added instantaneous moments, deterministic persistence and numerical coarse-graining. Failure under these changes would retire the broad closure interpretation while preserving the useful coarse-state non-identifiability example. This follow-up is exploratory and has not been executed in the present campaign.

### X03: select work before forming all products

Both a deterministic tail heuristic and a fitted selector meet the declared 1% relative error threshold on 24 independent held-out fields: maximum errors are 0.470% and 0.631%, with zero violations. Their mean retained pair fractions are 34.6% and 39.9%. Nevertheless, they are slower than forming the full small vectorized pair set. Paired geometric full/selector speed ratios are **0.691 and 0.615**, where greater than one is faster and the advance threshold is 1.10.

The full-pair warm median is 49.73 μs; deterministic and fitted selection medians are 69.46 and 79.74 μs. Feature calculation, selecting and gathering a variable subset cost more than the saved arithmetic. The equal-energy phase audit exposes nearly identical energy features across different phases, but **does not** produce a tolerance violation in the tested cases. It should not be advertised as a demonstrated selector failure mode.

**Decision: stop the current selectors.** Reducing operation count is insufficient when the full operation is already small and vectorized. A larger interaction workload, compiled fixed-shape selection, or a rigorous cheap omission bound could change the premise, but that is a new experiment. [Raw selection measurements](../results/portfolio-development/development/explore-X03/selection_rows.json) retain successes, cost failures and field identities.

## What changes in the portfolio, and what stays fixed

| Path | Continue now | Explanation weakened by present evidence | Still unresolved | Next decisive observation |
|---|---|---|---|---|
| A: attribution and fairness | Execute normalized frozen, amplitude-only, node-only, joint, bounded affine and conditioned controls with identical loss, node support, compression and endpoint menus. Keep analytic cubic and credible neural/classical alternatives. | A historical gain over the half-normalized frozen rule alone demonstrates useful learning. That inference was confounded and is no longer admissible. | Whether a state-dependent neural correction beats normalization and simple fitted controls; credible FNO optimization and matched-cost frontier. | A fresh full paired contrast beyond 10% error improvement with its stated uncertainty/coverage conditions, followed by an accuracy-qualified complete-cost result. |
| B: grounded improvements | Keep rich features, residual anchoring and band gain in the registered bounded pilot; retain full quadratic+cubic as the attainable-accuracy control. Profile native execution before optimization. | More quadrature nodes alone explain the remaining error on the diagnosed high-pair field. A larger MLP alone is the obvious cost intervention. | How often higher-order spatial structure dominates; whether useful residual structure can be computed or learned cheaply; FP32/GPU behavior. | Improvement over the strongest contained component, with extra transforms, feature cost and train/reference cost charged. |
| C: distinct prototypes | Preserve protected capacity. Prefer a small forward-history closure follow-up; retain the temporal representation as a mathematical prototype. | Fewer selected pairs necessarily run faster; accurate compact time encoding necessarily amortizes quickly. | Valid reuse at realistic query counts; causal closure stability; cost at a genuinely expensive interaction scale. | A changed premise must first pass a cheap independent kill test. Do not enlarge the failed current selectors or claim the closure is deployment-ready. |

The full protocol remains **55% A, 25% B, 20% C** within its 7,200-second discretionary experiment budget. Shared teachers, validation, confirmation, profiling and reporting are additional, separately logged work. Current diagnostic evidence changes the interpretation and priority of future proposals; it does not alter the frozen full primary metrics or relabel a newly chosen workload as confirmation.

The prepared full program has 40 bounded jobs and 176 model instances: 20 frozen instances plus 13 fitted families × two tracks × three training seeds × two training-set sizes. Its 24 fresh confirmation parents yield 101,376 planned endpoint rows over two grids and 12 schedules. The 24 fields are the relevant independent denominator. Field-level intervals, descriptive regime effects and failure denominators must remain visible; repeated rows are not independent evidence.

Requests remain within the desktop's 110,000 MiB Slurm cap and one 24 GB RTX 4090, with no job longer than 40 minutes under the required 45-minute maximum. The summed full science-time ceiling is 15 h 40 min; the summed allocation walltime ceiling is 21 h 38 min. These are safety ceilings across many jobs, not a duration forecast or a resource bill. Run commands and recovery details are in [the runbook](PORTFOLIO_RUNBOOK.md).

## Defects, costs and limitations retained in the record

The initial local v1 scaling implementation evaluated a single full-horizon call even when the protocol declared a multi-step schedule, and used the first listed target instead of `primary_target`. This made the v1 scaling cells inconsistent with their declared schedule/target interpretation. Revision `4b6b5ec` now rolls out the actual schedule, selects the named primary target and stores the step count and schedule. The original v1 artifacts remain intact; they must not be used as evidence for the corrected scaling experiment.

The v1 sum of scientific-stage timers is 252.88 seconds. The corrected v2 smoke sum is 242.42 seconds, including 76.51 seconds for atlas reporting. These include different numerical/reporting work and should not be treated as a controlled speed comparison. They also exclude interpreter/startup and other workflow overhead. They are incurred development work, not free discarded trials. The complete resource ledger and [validation record](../results/portfolio-development/validation.json) identify the separately available operational measurements. Monetary cost is unavailable without an actual rate and accounting model; it is not zero.

The new CPU development diagnostics take approximately 7.59 seconds inside their five stage timers. Together with v2 smoke, these 19 verified units total 250.02 science-timer seconds. Recorded operational time across v1, v2 and these development units is 562.64 seconds; this excludes unit tests, engineering time and unrecorded entry/export overhead. It is not the total project labor or energy cost. This cheapness is useful for deciding which ideas to retire, but does not predict the full campaign.

The code validation record contains 291 passing CPU tests, 11 passing targeted learning tests and one final claim-semantics regression. The 71 declared CUDA cases were collected but not executed here. The 4090 preflight, CUDA checks, synchronization-aware timing, GPU memory ceilings, native Slurm accounting, credible training budgets and fresh confirmation still require the user's desktop. None is represented as locally completed.

The strongest defensible current contribution remains a physically motivated representation of nonlinear interactions, with explicit evidence about when compression placement and interaction order matter. A learned advantage over properly normalized analytic and fitted controls remains open. Complete solver acceleration remains open. The next valuable outcome is an honest attribution result—even if it removes the neural network—followed by a cost comparison on a workload whose required accuracy creates enough numerical work to be worth avoiding.
