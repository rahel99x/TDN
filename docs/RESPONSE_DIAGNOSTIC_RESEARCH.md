# Response diagnostic research contract

The completed native Response pilot is a negative or unresolved development
experiment. Its 261 trials completed all 406,800 requested updates, but the
recorded gate was `STOP_OR_UNRESOLVED`: none of nine family/target groups passed.
Four target/regime cells were floor-limited for the selected tiny parents, and
87 of 1,440 teacher queries remained unresolved. This diagnostic uses a new
protocol to separate case eligibility, numerical precision, representation, and
optimization. It neither changes the old gate nor relabels these measurements as
fresh confirmation.

Read [Response research](RESPONSE_RESEARCH.md) for the three target contracts,
physical norms, FP32 input/anchor and FP64 subtraction rules, parent independence,
and separate neural-architecture/complete-solver-cost claims. Read the
[diagnostic runbook](RESPONSE_DIAGNOSTIC_RUNBOOK.md) for native execution and Raven
report paths.

## Eligibility is experimental design

The pilot's chosen high-mean parent had teacher-minus-deployed-anchor defects
below its frozen normalization floor for all three targets. FD's strong-reaction
parent was also entirely floor-limited. Consequently the FD group had only four
eligible regimes against five required; longer optimization alone could not
satisfy that frozen gate. A floor-limited cell does not assert that the trained
model attained an irreducible error floor, and the normalization safeguard is
not a mathematical error bound.

The new protocol declares an ordered, finite candidate bank and screening query
menu before evaluating models. The screen uses teacher acceptance, signal size,
FP32 output resolution, and declared precision/headroom checks to select eligible
parent identities. It records every candidate/query, failure reason, denominator,
selected identity, and exhausted unresolved group. It does not rank parents by a
trained model's loss or replace failed candidates through unreported sampling.
Original high-mean and reaction labels remain a priori regime labels; they are
not universal worst-case claims.

Training binds the screening artifact and selected parent IDs to the trial
identity before normalization. Missing or exhausted coverage produces an explicit
unresolved trial/target result. No code path silently trains on fewer declared
regimes, treats omitted teachers as passing, or lowers a floor to manufacture
eligibility. Null and precision-floor controls remain visible separately from
resolvable fits. Horizons, grids, phase twins, and seeds remain descendants of
the same independent parent.

Screening defines a new development estimand conditional on the declared eligible
bank. It does not prove performance on unscreened fields. Report both requested
and accepted coverage; effect summaries using accepted subsets retain the full
requested denominator in the accompanying coverage panel.

## Teachers must fit the complete uncertainty budget

The pilot's 13 same-grid unresolved teachers contracted under Lawson refinement
but disagreed with DOP853 above the `1e-10` maximum requirement. Its 74
projected-fine failures had individually accepted temporal teachers whose
combined uncertainty estimates exceeded the total requirement. For 73/74,
projected spatial maximum differences were below `1e-12`; one validation query
had spatial difference `1.987e-10` and independently failed the spatial budget.
None of these failures was a timeout.

The diagnostic allocates temporal component budgets within the final projected
target tolerance and retains independent cross-solver checks. It preserves the
total acceptance contract, contracting refinement, package/TLS verification, and
unresolved outcomes when caps are reached. Tighter numerical solves are a
declared new experiment, not retrospective acceptance of the pilot's labels.
Agreement remains an uncertainty estimate, not an error certificate.

Projected-fine deployment teachers lift the actual FP32 input by the declared
Fourier interpolation with zero extra hidden coefficients, independently refine
fine grids, then project. Main-bank projected-fine teachers and protected
hidden-state closure answer different questions. Finite Galerkin reaction-subflow
error is tracked separately from splitting error and deployment roundoff.

## Compare physical errors and optimization opportunities

The roster contains **34 arms**, with seven primary comparisons. Their names
identify actual configuration variants rather than new scientific claims:

| Primary arm | Architecture and training contract |
| --- | --- |
| `direct_baseline` | Spatial Fourier neural operator predicting the full endpoint; legacy full-target normalization and stochastic training. |
| `residual_baseline` | Matched physical anchor plus learned mean/centered endpoint correction; legacy residual-channel normalization and stochastic training. |
| `temporal_baseline` | Adapted original `TemporalMLP` endpoint control with its legacy cubic time dictionary and residual-channel objective. |
| `residual_conditioned` | Residual FNO with frozen gain normalization, full-parent physical objective, scheduled learning rate, and head warmup. |
| `adaptive_mixed` | Pointwise state encoder and separate mean/centered heads over a mixed, target-order temporal dictionary; conditioned physical training. |
| `spatial_mixed` | Spatial Fourier encoder with the same mixed temporal dictionary and conditioned physical training. |
| `coupled_source` | Spatially encoded source fields and a learned mean path, integrated through the physical propagator by quadrature; full-parent physical training. |

Exploratory arms separate these mechanisms:

| Arm group | What changes |
| --- | --- |
| `residual_scaled`, `residual_full_batch` | Gain normalization alone, or the full-parent physical training recipe without gain normalization, respectively. |
| `adaptive_response` | Uses response-kernel temporal channels rather than the mixed dictionary. |
| `adaptive_width64/128/210`, `spatial_width32`, `residual_width32` | Explicit width/capacity controls; realized parameter counts accompany their measurements. |
| `adaptive_rank8`, `coupled_rank2/rank8` | Number of endpoint temporal channels or integrated source channels. |
| Residual/adaptive/coupled `fp64_assembly` | Promotes final anchor/correction assembly while retaining declared model precision; deployed output differences are reported. |
| Residual/adaptive/coupled `numerical_remainder` | Adds a separate numerical-remainder channel, without ideal-physics null masks, to represent roundoff/subflow defects. |
| `coupled_cached`, `coupled_exponential`, `coupled_direct_interpolation`, `coupled_fixed_physical`, `coupled_learned_rates` | Cached spectral decoding, exponential time basis, direct source interpolation, fixed physical source channels, or learned bounded time rates. |
| Residual/coupled `rollout2`, `source_supervised`, and `coupled_consistency` | Two-step rollout loss, additional teacher-source supervision, or mean/variance consistency penalty; these five arms exclude projected-fine dynamics. |

The default three-target smoke calibrates all **97 eligible arm/target cells**
with 64 updates each. The full diagnostic contains **669 trials**, defaulting to
2,000 updates each (**1,338,000 logical updates**): every primary arm has three
paired seeds and six individual-regime plus one joint fit per target; every
exploratory arm has one seed with joint, low-mean, and large-amplitude fits.
Same-grid-only exclusions are explicit `NOT_APPLICABLE` cells, not failed
competitors silently removed from a common claim. Every native GPU allocation
requires all **97 CUDA checks (54 original + 43 diagnostic)**. CPU smoke reduces
work and width and cannot calibrate this native workload.

The full-parent physical-objective recipe explicitly uses **Adam epsilon
`1e-12`**, versus **`1e-8` in the three legacy baselines**. It uses learning rate
`.001`, five updates of learning-rate warmup, five head-only updates, and cosine
decay to `.05` of the base learning rate. These are declared optimizer treatments,
not hidden changes: physical errors with unit objective scales can produce much
smaller gradients than normalized residual losses, and Adam epsilon affects
their update conditioning. `residual_full_batch` therefore changes a recipe
including objective, batching, schedule, head warmup, and epsilon; it is not a
pure batching-only ablation.

The following terms have distinct meanings:

| Term | Meaning in this protocol |
| --- | --- |
| FNO | Fourier neural operator: learned spatial mixing plus local channels; this local adaptation does not reproduce a published benchmark. |
| Physical anchor | Declared numerical endpoint used as the correction baseline; its computation remains part of the complete model call. |
| Frozen gain normalization | Positive train-only scale fitted from input/physics/time envelopes; it conditions a correction basis and is not covariance whitening or a model-error bound. |
| Full-parent update | All declared training queries contribute with equal total weight per independent parent to one optimizer commit. |
| Recording cadence | Native curves and live research metrics record every 40 updates, plus initialization, completed validation and attempt-end observations; CPU smoke records every update. Exact journals retain every committed update. |
| Whole-history summary | Gradient extrema/means, clipping frequency and committed query work computed from every private optimizer-history row, including updates absent from a plot. |
| Physical objective | FP64 endpoint mean/centered squared errors with both objective scales equal to one; different from family-normalized legacy losses. |
| Mixed temporal dictionary | Signed free coefficients on constant, decaying/rational, and localized channels; it does not force all learned responses to decay. |
| Target time order | Cubic onset for same-grid finite-equation correction, quadratic onset for the declared projected-fine closure dictionary. |
| Source rank | Number of source channels integrated by the decoder; it is distinct from width, parameter count, or effective fitted rank. |
| Head warmup | Output/readout parameters train before the nonzero encoder trunk is unfrozen; only final correction readouts start at zero. |
| Source supervision | Additional offline teacher labels and loss; its acquisition, storage, and training work are charged separately. |
| Numerical remainder | Explicit arithmetic/subflow correction channel without ideal interaction masks; it is not a new physical mechanism. |

The report's `metric_legend.md` defines every plotted research metric. These
additional abbreviations and symbols explain its underlying measurements:

| Term | Meaning |
| --- | --- |
| TDN | Temporal-defect network: learns a finite-time correction to a declared numerical anchor. |
| FD | Finite differences: the same-grid discrete spatial equation. |
| Galerkin | Fourier projection of the spatial equation; its physical interpolant norm includes the declared Nyquist weighting. |
| DF / Strang anchor | Symmetric diffusion–reaction–diffusion numerical split, including its actual deployment arithmetic. |
| FP32 / FP64 | 32-bit / 64-bit floating-point arithmetic. Teachers use FP64; ordinary model deployment uses FP32. |
| MSE / RMS | Mean squared error / root mean squared error. Their units are state squared / state. |
| ULP | Unit in the last place: spacing between adjacent floating-point outputs near the measured value. |
| CPU / GPU / CUDA | Central processor / graphics processor / NVIDIA GPU execution platform. CPU results do not establish CUDA readiness. |
| VRAM | Dedicated GPU memory, distinct from host RAM or scheduler memory requests. |
| Adam epsilon | Positive denominator safeguard in the adaptive optimizer; changes how very small gradients produce updates. |
| LR / warmup / cosine decay | Learning rate / initial scheduled increase or head-only training / smooth decrease toward a declared minimum rate. |
| SVD / $r_{\rm cond}$ | Singular-value decomposition / relative cutoff for retaining fitted directions in a conditioning-aware linear control. |
| Nyquist mode | Highest representable Fourier frequency on an even grid; its physical interpolant weighting differs from sampled nodal weighting. |
| TF32 | Reduced-mantissa NVIDIA tensor arithmetic; disabled for this declared FP32 execution contract. |
| DOP853 / Lawson | Independent adaptive Runge–Kutta / exponential-integrating-factor reference methods used to estimate temporal uncertainty. |
| Parent / query / seed | Independent continuous initial field / one paired grid–horizon–phase observation / deterministic training initialization. |
| $u$, $m$, $v$ | State field, its spatial mean, and centered field $v=u-m$. |
| $h$, $t_{\rm ref}$, $\tau$ | Endpoint horizon, fixed reference time, and dimensionless horizon $\tau=h/t_{\rm ref}$. |
| $\kappa$, $r$ | Diffusion coefficient and logistic reaction rate in $u_t=\kappa\Delta u+r u(1-u)$. |
| $p$, $J$ | Declared onset order and number of temporal/source channels; neither is a theorem about trained global accuracy. |
| JSON / JSONL / CSV | Structured metadata / one metric record per line / tabular export for portable analysis. |
| GOOD / BAD / NA | Satisfied / failed / unavailable or inapplicable evidence for the specific declared check. |

For response channels, the implementation evaluates
$K_p(x)=\int_0^1 e^{-x(1-s)}s^{p-1}\,ds$ and multiplies it by
$\tau^p$. The mixed dictionary instead uses a bounded onset envelope
$\tau^p/(1+\tau^p)$ with constant, decaying, rational and localized channels.
Both have the declared small-time onset and free signed learned coefficients.
Neither dictionary proves the target law or an error bound at a finite horizon.

Capacity and common physical-objective controls help separate model size and
training conditioning. They do not make all arm pairs matched comparisons.
Family-specific normalized losses are not cross-family accuracy rankings.
Unnormalized physical mean, centered/RMS, maximum, and reference-adjusted errors
are the comparable measurements.

Each variant receives its declared training opportunity. Record initialization,
validation-selected checkpoint, final iterate, and best observed training-fit
diagnostic separately; a training-fit checkpoint does not become a deployment
selection. Capture gradient norms before clipping, clipping frequency, parameter
movement, effective optimizer settings, update counts, numerical failures, and
censoring. A successful physical-target fit without the declared reduction, or a
large relative reduction with unacceptable absolute error, remains distinct from
a passing combined trainability criterion.

Correction diagnostics retain the norm and fraction of learned corrections that
survive FP32 final addition and their physically assembled error. Additional
FP64 assembly or finer frozen-weight quadrature probes are separately charged
numerical diagnostics. They cannot retrospectively improve a reported FP32
deployment prediction or serve as proof of architecture incapacity.

The selected-checkpoint timing panel measures a synchronized **complete model
call** against its same-query physical anchor on resident input, with one warmup
and **five repetitions for one prospectively chosen accepted validation query**.
Reusable encoders additionally record encoding and same-state cached decoding
time, checked against the complete call. Cached decoder latency is not total
inference latency. This small paired benchmark does **not establish solver
utility**: it lacks a matched-tolerance numerical frontier and complete rollout,
transfer, fresh-process startup, amortization, energy, or monetary evidence.

## Work, continuation, and decisions

Native diagnostic smoke measures the actual declared variants on the configured
RTX 4090. CPU measurements never supply native throughput. Planning freezes
calibration identities, requested updates, work factors, and allowances before
the diagnostic runs. Longer trials continue exactly with model, optimizer, RNG,
sampler, dataset, source, software, and precision identities intact across jobs
of at most 35 minutes. Include all original and recovery attempts in cost.

Computational completion, valid seals, teacher coverage, attainable training fit,
held-out generalization, and solver cost are separate results. A negative or
unresolved scientific outcome completes the diagnostic evidence when the
declared measurements and reports verify. A failed job or unsealed/missing result
cannot be promoted to success. The final report depends on `afterany` so failure
evidence is preserved. This diagnostic does not automatically unlock legacy
development, freeze, confirmation, or deployment.

Interpret optimizer evidence proportionally. The prior experiment had no
training time censoring; extending every family's GPU allowance is unsupported.
The new plan declares any additional work explicitly and measures its native
cost. A censored run remains unresolved, not proof that a model cannot learn.

## Portable reporting and preserved evidence

Publish the measured pipeline outcomes and raw screening, trial, endpoint,
learning-curve, correction, cost, and decision tables in the run-local Raven
project. Unit attempts retain exact actual Slurm identity, logs, terminal status,
and scientific outcome. The aggregate analysis is a manually selectable attempt
without a fabricated scheduler job. Continuous observed-range bands are not
confidence intervals; repeated seed/horizon measurements are not independent
parent replicates. Missing measurements and conditional denominators stay visible.

When a campaign declares more than 255 units, exact attempts are paged into
standalone `raven/page-000`, `page-001`, and subsequent projects with at most
255 unit attempts each. The root Raven project contains the aggregate overview;
`project_catalog.json` maps every declared unit, page, and attempt. Smaller smoke
runs may retain all attempts plus the overview directly under the root. This
preserves the full scientific matrix while respecting Raven's bounded discovery.

The portable ZIP has project-relative paths, native Raven contracts, an HTML
analysis, and raw CSV/JSON data. It is readable without a live scheduler and does
not certify provenance of omitted replay bytes. Compact exports retain exact
source pointers and omission inventories; a full replay or checkpoint freeze
requires its actual binary artifacts. Monetary, energy, deployment economics,
and scientific superiority remain unavailable without the corresponding
measurements and comparison contract.

File formats and discovery follow the pinned Raven
[project standard](https://github.com/rahel99x/Raven/blob/c34f0892d7bddc685c5af10489ecb114fd6adab6/docs/PROJECT_STANDARD.md)
and [V2 reporting guide](https://github.com/rahel99x/Raven/blob/c34f0892d7bddc685c5af10489ecb114fd6adab6/docs/PROJECT_REPORTING_V2.md).
