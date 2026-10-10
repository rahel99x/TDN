# TDN native portfolio: comprehensive results review

Reviewed 2026-10-10. Run `fedora-portfolio-20261009T183307014599Z`; source commit `c5c9a85e5cc92cf4f361e57a74626c8a0a5c2fc3`. This reviews the uploaded full Fedora/RTX 4090 campaign, not the earlier CPU smoke, other historical campaigns, or the separate adjacent-interaction study.

**The strongest result is presently numerical and representational. Correct normalization and preserving nonlinear interactions before compression help substantially. The learned conditioner has not established an improvement over correctly normalized frozen quadrature, and complete solver acceleration over strong classical alternatives remains unproved. Its substantial advantage over the tested local FNOs is real within this experiment, but those FNOs remain inadequately optimized for a claim about competitive neural operators.**

The experiment succeeds at attribution: it weakens explanations that previously made learning look necessary. That is useful progress. The most informative next directions are a cheaper higher-order interaction correction, explicitly separated spatial closure, and a small causal-history closure pilot—not a larger version of every current network.

Detailed supporting reviews:

- [Every family, error metric, regime, paired comparison and frontier](reviews/portfolio-native-20261009/statistics.md).
- [Architectures, training, optimization, data efficiency and learned responses](reviews/portfolio-native-20261009/training.md).
- [Mathematical mechanisms and all three exploratory prototypes](reviews/portfolio-native-20261009/mathematics.md).
- [Complete accounting, scaling, timing limitations and amortization](reviews/portfolio-native-20261009/costs.md).
- [Machine-readable review, source index and reproducible analysis scripts](../results/portfolio-full-review/README.md).

## Evidence quality and scope

| Item | Verified observation | Interpretation |
|---|---|---|
| Transport | 17/17 parts; archive SHA-256 verified; 10,034 retained files checked | Uploaded review evidence arrived intact. |
| Science metadata | All 40 completion markers, protocols, prerequisite/lineage hashes and streamed/compact ledger pairs agree | No inconsistent lineage or mixed protocol detected. |
| Omitted payloads | 1,402 numerical arrays/checkpoints/renderings omitted by the review preset, with original hashes retained | This review cannot independently replay all predictions or verify omitted bytes. Transport integrity is not independent scientific replication. |
| Execution | 40/40 allocations completed, exit zero | Pipeline completion; not 40 scientific successes. |
| Native tests | 366 CPU cases plus 71 CUDA cases in each of 19 GPU jobs: 1,715 executed test instances, zero failure/error/skip | Repeated CUDA checks are repeated executions, not 1,349 distinct hypotheses. |
| Models | 23 families, 176 selected instances, 536 completed trials | 150 fitted checkpoints, six selected initializations, 20 frozen controls. |
| Data | 32 training, 16 validation, 24 confirmation parents; no field-ID overlap between splits | Only 24 independent confirmation fields; three per regime. |
| References | 1,952/1,952 declared train/validation/confirmation references accepted after actual refinement | Acceptance is an empirical uncertainty assessment, not a mathematical certificate. |
| Endpoints | 101,376/101,376 finite outputs with accepted references; 67,584 intermediate observations | Repeated fields, grids, schedules and seeds do not increase the independent denominator. |
| Primary claims | Accuracy: 17 GOOD, 39 BAD. Cost: 4 GOOD, 30 BAD, 6 NA | These are predefined descriptive claim checks, not independent discoveries or an overall quality score. |

The 17 accuracy successes consist of 16 comparisons with the local hybrid FNOs and one with the affine control at eight training parents on the discrete long endpoint. None establishes the primary conditioner's superiority over normalized frozen quadrature. The four cost successes are discrete short-endpoint comparisons against the two hybrid FNOs at the two data sizes.

Confirmation teachers use FP64 Lawson RK4 with actual temporal refinements and, for continuum targets, projected spatial refinement. All 1,536 confirmation references are accepted. Maximum uncertainty estimates are approximately **1.89e−8 RMS / 2.00e−8 maximum** for discrete and **8.68e−9 / 9.98e−9** for continuum, small against the primary 2e−5 tolerance. These estimates do not resolve every near-roundoff error ratio. Independent solver cross-checks exist in the mathematical audit; they are not a second independently generated teacher for every confirmation endpoint.

The scientific source in `tdn/`, `reference/` and configurations is unchanged between the native run revision and the code examined for this review. Original artifacts and scientific verdicts remain untouched. Reconstructed parameter responses and additional whole-cohort contrasts below are explicitly post-hoc descriptive analyses, not newly locked confirmation.

## What our current architecture actually is

The task is periodic scalar logistic reaction–diffusion:

\[
u_t=L u+r u(1-u).
\]

The discrete track specifies a finite-difference Laplacian and nodal reaction. The continuum track compares coarse spectral models with projected spatially refined Galerkin solutions. These are different targets.

The primary model is approximately

\[
\widehat u(h)=S_{\rm DF}(h)u+
g_\theta(z(u,h))\,P_m D_{2,h}(u;\xi_\theta),
\]

where `S_DF` is a diffusion-first Strang step, `D2` is a transported quadratic interaction-defect formula, `P_m` is output compression, `xi` is a learned global symmetric node pair, and `g` is a small learned scalar gain. Diffusion is propagated analytically; reaction uses exact logistic flow on the discrete track and projected RK4 on the Galerkin track. The full field enters the physical interaction formula before compression. Three global conditioning features are mean, nondimensional reaction time and an energy-weighted diffusion rate. The conditioned model has **43 fitted parameters**: 41 in the network plus two global gain/node logits. Richer and band-dependent models have 75 and 59 fitted parameters.

**It is a time-step-conditioned hybrid numerical operator.** It is not a recurrent model, nor a reusable learned state encoding that cheaply answers arbitrary future-time queries. It reconstructs and propagates physical interactions at each step. Its small parameter count does not imply low inference cost.

The physical branch retains signed amplitudes and phases. The small conditioner does not retain all spatial information. Node locations are global learned parameters, not state-adaptive nodes. Analytic nulls for constant fields, zero reaction, zero diffusion and zero time are valuable structural properties; they do not prove positivity, long-time stability or a convergence order for arbitrary learned parameters.

## What the decisive controls now tell us

The following attribution ratios summarize paired whole-cohort endpoints with the same configurations. They are post-hoc descriptive effects, not universal rankings or matched-accuracy speedups. Greater-than-one RMS improvement means the changed method has lower error.

| Intervention | Discrete RMS improvement | Continuum RMS improvement | Judgment |
|---|---:|---:|---|
| Historical half-normalized rule → normalized frozen GL2 | 3.00× | 2.68× | A substantial old apparent learning benefit was normalization. |
| Input compression → interactions formed before output compression | 2.23× | 2.08× | Supports preserving useful high-to-low nonlinear interactions; benefit depends on spectrum. |
| Frozen GL2 → frozen GL4 | 1.99× | 2.13× | More work buys accuracy; same-schedule cost rises about 1.34× / 1.18×. |
| Output-compressed GL2 → full-output GL2 | 1.10× | 1.05× | Compression is not free: removing it is also slightly faster in this implementation. |
| Frozen GL2 → analytic quadratic+cubic | See full paired tables | See full paired tables | Strong accuracy control, but expensive; a target for efficient approximation, not an automatic solver winner. |

The primary 32-parent conditioner's **registered** comparisons with normalized frozen GL2 are more decisive:

| Track / final time | Frozen/learned geometric RMS ratio | Descriptive 95% interval | Joint RMS+max passing query cells: learned / frozen |
|---|---:|---:|---:|
| Discrete, 0.12 | 0.966 | 0.849–1.080 | 799 / 792 of 864 |
| Discrete, 0.24 | 0.971 | 0.832–1.124 | 736 / 723 of 864 |
| Continuum, 0.12 | 0.947 | 0.821–1.073 | 816 / 786 of 864 |
| Continuum, 0.24 | 0.896 | 0.754–1.047 | 743 / 720 of 864 |

Greater than one favors learning. All four fail the registered ≥1.10 improvement criterion with lower interval bound >1. Their point estimates correspond to approximately **3–12% worse geometric RMS**, but the intervals do not establish that learning is uniformly harmful. It improves some threshold crossings and tails. That tradeoff matters: better mean loss, better average RMS and higher feasible coverage are different objectives.

Amplitude-only, node-only, joint and affine fitting are essential controls. The state-dependent scalar gain does not earn a general accuracy claim against them either. Rich conditioning changes average performance only modestly; band gains add cost and worsen some comparisons. More expressive output coefficients do not guarantee a better residual direction.

Learned gains are not constant: post-hoc reconstruction gives initial-field gain ranges roughly **0.957–1.441 discrete / 0.959–1.542 continuum**, with larger corrections in diffusion-stiff and high-frequency regimes. That is interpretable adaptation, not proof of useful adaptation. Global gain logit and final network bias have an exact additive non-identifiability; individual weights should not be interpreted as unique physical discoveries.

## Accuracy, failures, generalization and the solver frontier

At the primary threshold, **both RMS and maximum error, including estimated teacher uncertainty, must be ≤2e−5**. The report also retains loose 2e−4 and strict 2e−6 targets. A pass fraction across the entire schedule menu is not a deployment success rate; deliberately coarse one-step alternatives remain in it.

The primary n32 model has all-menu median RMS **6.51e−7 discrete / 1.11e−6 continuum**, but its 99th-percentile RMS is **1.26e−4 / 1.10e−4**. Maximum observed pointwise errors reach **3.88e−4 / 3.36e−4**. Small medians therefore coexist with real hard-case failures. The full statistics appendix retains all 23 families, norms, tails, seeds, regimes, data sizes and horizons instead of hiding them behind a universal leaderboard.

Mean error matters. For the primary model, the median fraction of squared RMS error explained by mean error is approximately **39% discrete / 59% continuum**. For the hybrid FNOs it is about **97–99%**. Their problem is not simply an inability to resolve small spatial scales. Logistic reaction does not conserve the mean, so the correct target is the reference mean evolution, not forced mean conservation.

All **1,887 endpoint rows and 454 intermediate observations** leaving [0,1] belong to `direct_fno`. Native initial fields are inside [0,1]; this is not inherited invalid input. The largest predicted value is **1.544**. No other family exhibits a recorded interval violation. This establishes bounded observed behavior only over this short experiment; it is not a positivity or stability theorem. Direct FNO has zero primary-tolerance endpoint passes and should be treated as a failed bounded fit.

Validation-frozen schedule coverage for the primary n32 model:

| Scope | Learned conditioner | Frozen GL2 | Interpretation |
|---|---:|---:|---|
| Discrete 0.12 | 24/24 fields | 24/24 | Every tested field qualifies for both methods. |
| Discrete 0.24 | 24/24 | 23/24 | One additional observed field qualifies; population advantage remains unresolved. |
| Continuum 0.12 | 20/24 | 22/24 | Learned schedule misses the 95% observed coverage gate. |
| Continuum 0.24 | 21/24 | 22/24 | Same limitation. |

Here a field passes only if all its paired grids/seeds pass. Reference-informed post-hoc schedule selection finds a passing learned schedule for **24/24 fields in all four scopes**, allowing different selections for each model/field/grid cell. This does not establish one common successful schedule. Thus the continuum locked failures do not show that the method cannot reach accuracy within its menu. They show that the validation-selected schedule does not reliably transfer. Post-hoc truth-selected schedules are an oracle upper bound, not a usable policy.

Even 24/24 is not statistical assurance of 95% population reliability: under an IID binomial assumption its one-sided 95% lower bound is only about **88.3%**. At least 59 independent zero-failure samples would be needed for a lower bound above 95%, before considering regime balance, multiplicity or non-IID task construction.

At the locked discrete 0.12 endpoint, the conditioner is **1.67× faster than small FNO and 2.37× faster than standard FNO** on mutually qualifying paired cases, with better observed field coverage. It is simultaneously about **1.23× slower than frozen GL2, 2.91× slower than DF and 3.22× slower than ETDRK4** on their respective mutually qualifying cases. At 0.24 it is about **4.31× slower than ETDRK4** on their common qualifying subset. Classical coverage is not identical, so these conditional timings do not establish universal classical dominance. They do rule out a broad cost-win claim for our model.

Continuum's apparent conditional speed advantages over some methods cannot rescue its failed coverage gate. Some long-endpoint comparators have no validation-qualified schedule; their cost ratio is NA, not an infinite win for TDN.

## Training and competitor credibility

Training completed without failed or prematurely truncated trials. Six selected initializations are the three deterministic seeds each of eight-parent continuum amplitude and affine fits; fitting converged but validation preferred analytic initialization. Retaining those initializations was correct.

All 12 primary conditioners selected their last allowed update, **200**. The 120-second safety cap never bound training. This supports a small convergence diagnostic, not a presumption that more training will fix confirmation. Increasing parents from eight to 32 improves post-hoc paired confirmation RMS by about **1.084× discrete / 1.119× continuum**, while adding only four/nine passing cells out of 1,728. There are two nested data sizes and one cohort, not a sample-complexity curve.

The n32 primary model is **11.6–43.9× more accurate in registered geometric RMS comparisons than the tested hybrid FNOs**. However:

- Standard FNO's selected validation objective remains approximately **662× DF discrete / 1,194× DF continuum**. These are normalized loss ratios, not RMS ratios.
- Every default-clipped FNO update exceeds its clipping threshold; logged gradient norms are pre-clipping and loss-scale dependent. An unclipped alternative was also tried, so clipping alone is not established as the cause.
- Only `t_ref=0.2` was actually tested. No effective temporal-output scale sweep occurred.
- FNOs received more tuning trials and similar aggregate trial walltime, but equal update counts are not an equal-compute frontier or proof of convergence.
- The hybrid FNO's nonzero initial residual can overwhelm the tiny defect it needs to learn. A baseline-preserving initialization and properly scaled residual target deserve a controlled audit.
- Direct FNO learns the entire update while our method receives an analytic physical integrator. That is a legitimate system comparison, but it does not isolate neural representation quality.

The defensible statement is **“our physical hybrid beats these bounded local FNO adaptations.”** This campaign does not establish defeat of the FNO paper or of a well-tuned operator-learning baseline.

Training is one-step endpoint supervision from original states, not evolved-state or short-rollout training. Validation already includes time 0.24; it is outside training horizons but not untouched time extrapolation. Fields come from a small family of two/three-mode templates, with randomized phases and prescribed means/physics. “Rough” here is a declared spectral regime, not broadband turbulence or a universal worst case. No arbitrary geometry, boundary-condition transfer, heterogeneous coefficients, noisy inputs or long-horizon deployment claim is established.

## Where the remaining mathematical headroom is

The diagnostic high-pair Galerkin field gives a useful separation at 16², FP64, reaction rate 3, diffusivity .004 and time .12. The first three errors use the same-grid Galerkin teacher; the final row separately compares spatial targets:

| Quantity | RMS magnitude |
|---|---:|
| DF endpoint error | 4.94e−5 |
| Full four-node quadratic endpoint error | 3.65e−6 |
| Quadratic+cubic endpoint error | 7.77e−8 |
| Four-node versus sixteen-node quadratic correction difference | 6.60e−13 |
| Coarse Galerkin versus refined continuum discrepancy | 4.69e−6 |

This is a targeted one-field diagnostic, not a statistical claim. It nevertheless sharply weakens “more quadrature nodes” as the next explanation: quadrature is already much more accurate than the remaining solution. Missing finite-amplitude interaction shape matters, and spatial truncation can exceed the improved temporal error.

Writing the centered field as `v`, a scalar correction spans only `g D2(v)`. If the true residual contains a substantial cubic or different spatial component, no scalar conditioner can reproduce it. Learning a low-cost approximation to a strong higher-order rule is better motivated than simply widening this gain network. The analytic cubic rule itself is costly and sometimes leaves tail failures; it supplies an accuracy target, not a demonstrated economical method.

Continuum compensation must be separately labeled. Correcting a chosen discrete equation and compensating its missing fine scales are different mathematical problems. Keep a same-equation temporal track alongside the refined continuum track before crediting a learned gain to either mechanism.

## Exploratory path: explicit outcomes

| Prototype | Result | Cost/validity qualification | Decision |
|---|---|---|---|
| X01 compact continuous-time kernel | Maximum relative discrepancy 6.15e−16 for a second-Picard interaction response | 865-query break-even versus cached exact pairs, against a 32-query gate. At the registered tolerance, GL2 is cheaper to build and query. One synthetic state; not a complete nonlinear flow. | Preserve representation evidence; stop the present speed claim. |
| X02 coarse closure with history | Instantaneous fit RMS 2.42e−5; history persistence 2.45e−7; fitted history 5.99e−11 on 20 held-out parents | History generation costs ~241 μs/state versus ~1.16 μs/query. One mode, fixed physics and noiseless backward-generated histories. Stronger analytic history control needed. | Advance one small causal-forward-history pilot. |
| X03 compute selection | Both selectors meet 1% error while keeping ~35%/40% of pairs | Speed ratios 0.750× deterministic / 0.672× fitted: both slower after selection overhead. | Stop current implementations unless a changed workload or kernel design alters the premise. |

X02 contains the most interesting additional information result: identical resolved current states can have different futures because unresolved content differs. Any deterministic instantaneous closure receiving only those identical inputs must fail to distinguish them. History can restore information. This is a useful mechanism already related to established closure ideas, not a literature-novelty claim. The fitted rule is already a two-coefficient numerical formula; no neural network is needed for this prototype.

## Scaling and complete cost

Scaling measured six methods on grids **64², 128² and 256²**, both tracks, batch four: **36 measured cells**, all accuracy-qualified. Six declared batch-one workload cells were omitted by the cap. The same four fields are reused across sizes, with mild fixed physics, time 0.06 and two steps.

At the common tolerance, DF takes about **0.67 ms per four-field discrete batch**, versus **4.08–4.15 ms** for the conditioner. On continuum, ETDRK4 takes **4.05–4.42 ms**, versus **20.55–22.46 ms**. Frozen GL2 and ETDRK4 are faster and have lower worst-field errors than the conditioner in all six panels. Large grids alone have not created an advantage.

Almost flat learned latency despite 16× more grid points suggests launch/Python overhead, but a frozen-model GPU trace is still needed. Existing component profiles are CPU/FP64 initialized probes. They cannot prove the CUDA bottleneck.

| Resource measure | Native recorded cost |
|---|---:|
| Sum of 40 allocation elapsed times | **6h05m31s** |
| GPU-reserved time | **5h21m54s** |
| Confirmation allocations | **4h20m32s**; 71.3% of allocated elapsed |
| Training/tuning/schedule-selection allocations | **57m42s** |
| Recorded trial walltime alone | **39m02s** |
| Reserved / used CPU core-hours | **26.674 / 6.469** |
| Maximum observed step RSS | **7.88 GiB**, reporting stage |
| Maximum CUDA allocated / reserved peak | **302 / 390 MiB**, whole-group peaks |
| Power, energy, monetary cost | **Not measured / no rate supplied** |

The observed calendar envelope is about 15h15m, distinct from summed allocated compute. The record does not permit attributing its difference solely to queueing. The accidental later rerun and prior campaigns are not included in these costs. Longest actual job is 22m16s; longest request 40 minutes, within the 45-minute limit.

Low recorded CPU reservation utilization does not measure GPU utilization. GPU work was not VRAM-limited. Most expense went into exhaustive confirmation measurements, not learning: 1,152 paired groups × 88 models × seven first/warmup/warm invocations produced **709,632 timed complete solver calls**, excluding additional intermediate-audit rollouts. Scaling also spent about 98% of its scientific timer generating CPU teachers while reserving a GPU; these references should be prepared in a CPU stage.

Optimistic training-only amortization against qualifying FNO cases can be around 800–1,800 queries, depending on comparator and seed. This omits references, preprocessing, schedule selection, acceptance/fallback work and other offline costs. Faster analytic controls still need to be considered. It is not an established deployment break-even.

## Defects, missing metrics and easy-to-miss interpretation risks

1. **Amortization reporting defect:** all 50,688 original rows are NA because the calculator expects nonexistent candidate/comparator timing columns instead of joining `cost_seconds`. The missing curve is not a scientific finding of no margin.
2. **Timing protocol mismatch:** scaling performs three warm repeats despite five being declared. Confirmation uses five. Neither provides a stable tail-latency estimate; recorded p95 is essentially the observed maximum.
3. **Cold latency is mislabeled for deployment interpretation:** it means first invocation inside a warm process, not CUDA/context/checkpoint/service startup.
4. **Schedule-selection workload mismatch:** validation amortizes matching-physics batches, confirmation times individual fields. Batch opportunities should be compared explicitly with frozen selections for each deployment workload.
5. **Memory attribution:** measured CUDA peaks belong to interleaved groups, not individual architectures. Parameter count, checkpoint storage, activation memory and physical GPU use are distinct.
6. **Spectral-band semantics:** the diagnostic boundary is radial N/4. A physical mode can change from the “high” band to “low” when the grid doubles. Use common physical bands for cross-grid interpretation or retain explicit relative-Nyquist labels.
7. **Statistical scope:** 24 fields, three seeds, three fields per regime, unadjusted descriptive intervals and many contrasts. Deterministic fits repeated across seeds are not independent optimizations. Small regime samples cannot support broad population claims.
8. **Reference and precision floors:** FP64 teachers are cast to FP32 during training. Very small ratios can reflect roundoff or cancellation, even with well-converged references. Maximum-error uncertainty is estimated, not certified.
9. **Mean/shape decomposition:** larger RMS can be predominantly mean bias. The logistic PDE changes mean physically; “conservation repair” would be an incorrect objective.
10. **Training objective mismatch:** defect-normalized RMS-plus-peak loss, equal-query geometric confirmation RMS and deployed joint coverage are different objectives. An improvement in one may trade against another without any code bug.
11. **Repeated validation choices:** 536 trials and multiple schedule choices use just 16 validation parents. Selection uncertainty and possible validation overfitting deserve fresh development cohorts, not tuning on these inspected confirmation fields.
12. **Closed distribution:** limited Fourier templates, narrow amplitude regimes, short time horizons and small parameter shifts do not substantiate broad PDE generalization. Accuracy at 256² is four-field batch evidence, not a large independent test bank.
13. **No operational policy benefit established:** acceptance, rejection, fallback and safety-check overhead must all be charged. An oracle frontier does not implement one of these policies.
14. **Scoring is not science quality:** GOOD/BAD/NA and 1–100 evidence scores measure the declared checks. They are not theorem proofs, calibrated success probabilities, novelty scores or a universal model ranking. Some dedicated measurement tables have no corresponding generic scored row.
15. **Unknown complete project cost:** the archive contains this run, not engineering labor, every prior failure, every accidental rerun, electricity or external rates. Unknown cost is not zero.

Observed-range learning bands describe observed lows/highs, not confidence intervals. Smoothing should never remove failures, hide missing measurements, imply independent data, or change numerical summaries.

## Recommended next decisions

**First, repair interpretation and fairness before expanding comparisons.** Correct amortization joins, honor timing settings, retain batch-one scaling, use physical spectral bands and distinguish true cold starts. Give hybrid FNO a verified baseline-preserving initialization, appropriately scaled residual target and a small equal-compute convergence study. Keep all current results intact.

**Second, narrow the grounded science question:** can a cheap additional interaction basis capture the residual that a scalar multiple of quadratic defect cannot, at a cost below the strongest accuracy-qualified classical alternative? Use normalized full GL2/GL4, the cubic control, scalar/affine fits, and ETDRK4/DF. Measure residual direction/correlation and separate mean/centered errors before choosing a network. Distinguish temporal residual from unresolved spatial closure. Stop if the added basis has no useful cost margin.

**Third, study reliability through schedules, not hindsight.** The candidate can reach continuum tolerance somewhere in its current menu; the locked choice sometimes misses. Diagnose validation/confirmation physics and batching shifts, then test a cheap deterministic schedule rule or embedded estimate. Develop a learned policy only if its total cost can beat that deterministic rule. Use new confirmation after any revision.

**Fourth, protect one genuinely different pilot:** causal history closure with histories obtained by forward operation, a derived deterministic coefficient, noise/parameter shifts, multiple modes, rollout stability and charged startup/refresh cost. Its small prototype is promising enough for a bounded test, not a large campaign.

**Retire or freeze for now:** the current pair selectors, current temporal-encoding speed claim, direct FNO as a purported strong baseline, indiscriminate gain-network enlargement, and exhaustive confirmations of every dominated arm. Preserve all negative outcomes. Spend the next budget on informative independent fields and credible controls rather than another large Cartesian product.

**Strongest defensible contribution today:** a physically structured representation and experimental account of when nonlinear interactions must precede compression, which analytic interaction orders reduce error, and why those accuracy gains often fail to become end-to-end speedups. A learned advantage remains possible, but this campaign does not yet establish it over the strongest relevant analytic controls. The next contribution may reasonably be a small fitted rule, a better numerical method or a useful closure rather than a larger neural operator.
