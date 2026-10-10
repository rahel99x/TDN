# Mathematical, attribution and exploratory review

Scope: exact retained artifacts for `fedora-portfolio-20261009T183307014599Z`, scientific source tree `a83f6a15a41d35ae17ccc08babc8304204a5f12e5c82d0d96399cb3982d45861`, native run commit `c5c9a85e5cc92cf4f361e57a74626c8a0a5c2fc3`. These observations are from the uploaded run, not the older published CPU development pilot. Canonical source names below are relative to the run. Original bytes are available through `review_io.py` and their hashes in `evidence-index.json`.

## What the model actually computes

The specified equation is logistic reaction–diffusion, `u_t = L u + r u(1-u)`, on a periodic two-dimensional domain. The discrete track uses the periodic finite-difference symbol and nodal products. The continuum-labelled model uses a Fourier symbol and dealiased projected quadratic products. The latter's *labels* are projected spatially refined solutions, so correcting its coarse Galerkin equation and compensating for its spatial truncation are distinct effects.

For a current field write `u=c+v`, where `c=mean(u)` and `mean(v)=0`. Let `H_s=exp(sL)`, `C_s=logistic_s(c)`, and `q_s=d C_s/dc`. With the analytic scalar logistic background, the second-order centered-amplitude defect between the coupled flow and diffusion-first Strang has the representation

`D2_h(v) = -r q_h ∫_0^h q_s [H_(h-s)((H_s v)^2) - H_(h/2)((H_(h/2)v)^2)] ds`.

Products are interpreted in the declared equation. `quadratic_df_defect` approximates this integral by the specified nodes and weights; the midpoint-only rule is zero by construction. The physical nulls for constant fields, r=0, diffusion=0 and h=0 are structural, not learned. The cubic control adds the next causal nested Volterra coefficient; it is not merely a larger quadratic gain.

The main candidate computes `Strang_DF(u,h) + g_theta(features(u,h)) P_m D2_h(u; learned global nodes)`. The discrete reaction subflow is exact logistic; the Galerkin reaction subflow uses projected RK4, so not every component of the continuum backbone is an exact analytic flow. All transported full-field products are formed before output cutoff. The physical branch therefore receives all spatial amplitudes and phases. Only the *gain selector* receives the three global features: mean, log(1+h r), and log(1+h active_diffusion_rate). Rich conditioning adds variance, mean absolute centered amplitude, spectral-rate spread and a nonlinear-strength statistic. Band gain emits three scalar gains for DC/lower/upper retained output bands.

This is a time-step-conditioned hybrid numerical operator. It does not encode a state once and reuse a learned temporal latent representation over arbitrary future queries. Its small trainable parameter count is not its compute count: heat propagation, products, transforms, projection, feature extraction and validation all remain. The learned nodes are global parameters, not state-conditioned node locations. Physical phase information is preserved even though the scalar selector is phase-insensitive conditional on its features.

## Attribution checks that the implementation gets right

- `historical_half` preserves total frozen quadrature weight 0.5; `quad2_fixed` uses normalized GL2 weights 0.5+0.5. The historical control alone could never isolate the value of learning.
- Frozen, amplitude-only, nodes-only, joint, affine and full conditioned GL2 share the output cutoff, physical branch and two-node structure. Fitted effective gains share (0.25,1.75). The affine/scalar fits refine their initializer against the same defect-normalized RMS-plus-peak objective as neural fitting; they are not deliberately weak plain unscaled least squares controls.
- `quad2_input` changes only premature input compression; this is a sound intervention on information availability. Full vs output-cutoff controls and two- vs four-node controls are explicit.
- All applicable methods receive the same endpoint schedules, including inexpensive 1/2/4/8-step options. Validation-locked choices and reference-informed post-hoc frontiers are kept distinct.
- FNO hybrid controls keep a full-resolution local path and receive the same DF backbone. Fourier truncation alone does not imply that these FNOs cannot generate useful high-to-low interactions through local nonlinearities.
- Independent adaptive DOP853 cross-checks and quadrature polynomial moments pass in the mathematical audit. The audit has 60 records, no correctness failures, and all overall verdicts NA because predictive benefit is intentionally unestablished by algebra.

## Attribution and interpretation limits

1. `1+.75*tanh(global_logit+conditioner_output)` has an exact additive gauge between global logit and final network bias. Individual fitted parameters cannot be given independent physical meanings. At small h, node locations and gain can also compensate in the leading defect coefficient. `parameter_report` acknowledges this, but full state-dependent response-Jacobian identifiability is not computed; only the affine initializer gets singular values and matrix rank.
2. Symmetric learned nodes retain the intended small-step parity/consistency but no longer have all Gauss polynomial moments. A fitted node shift can compensate missing higher-order physics rather than improve quadrature accuracy. These are different explanations.
3. A scalar gain cannot rotate a residual outside the spatial span of its analytic defect. Better optimization cannot remove that representation ceiling. Three band gains relax the ceiling only within their three fixed components.
4. Portfolio continuum labels are refined-space targets while the physical backbone is coarse Galerkin. A continuum gain may partly learn unresolved-scale compensation; discrete gain is the cleaner temporal-attribution experiment.
5. The FNO training code has an optional `fno_t_refs` scale-search hook, but the full frozen protocol does not set it: there is one `t_ref=.2`. Its actual search is learning rate and optimizer/clipping/loss-scale controls, not a temporal-output scale sweep. This is an unresolved baseline opportunity, not evidence that the reported comparisons used an incorrect implementation.
6. Validation frontiers choose schedules using amortized matching-physics batches and transfer them to single-field confirmation. That transfer is honestly recorded but is not optimal single-field schedule tuning; performance changes can come from batching opportunities.
7. Primary intervals are descriptive crossed field/seed bootstrap, not calibrated simultaneous hypothesis tests. All-pair subgroup tables are explicitly exploratory; they do not exclude below-teacher-floor error ratios as strictly as the primary claims. Three parents per regime cannot support broad population-level statements.

## Native diagnostic bottlenecks

`diagnose/diagnostic_rows.json` uses one 16² field per regime, r=3, diffusivity=.004, endpoint h=.12, FP64. The reference is the *same-grid* finite equation even when the row track is named continuum. A separate 2N/4N comparison measures spatial discrepancy. Thus these figures explain mechanisms; they are not the 24-parent fresh GPU confirmation.

| Field / equation | One-step DF RMS | Full GL4 quadratic RMS | Quadratic+cubic RMS | GL4 / cubic improvement |
|---|---:|---:|---:|---:|
| Low / discrete | 1.579644e-7 | 1.417730e-7 | 1.243784e-8 | 11.4× |
| Low / Galerkin | 1.759140e-7 | 1.569721e-7 | 1.663524e-8 | 9.44× |
| High pair / discrete | 2.698379e-5 | 3.274513e-6 | 9.373322e-8 | 34.9× |
| High pair / Galerkin | 4.936738e-5 | 3.649243e-6 | 7.770859e-8 | 47.0× |
| Rough / discrete | 6.735429e-6 | 7.458299e-7 | 8.520045e-8 | 8.75× |
| Rough / Galerkin | 1.313756e-5 | 6.312368e-7 | 9.574163e-8 | 6.59× |
| Near Nyquist / discrete | 2.603605e-5 | 1.321836e-6 | 6.106057e-8 | 21.6× |
| Near Nyquist / Galerkin | 5.326211e-5 | 1.387494e-6 | 4.029314e-8 | 34.4× |

For the high-pair Galerkin diagnostic, GL2-to-GL16 correction RMS difference is 2.678729e-7, whereas GL4-to-GL16 is 6.600181e-13. The remaining GL4 endpoint error is 3.649243e-6. This strongly weakens “just use more accurate quadrature” for that case. Cubic shape/finite-amplitude effects provide much more accuracy headroom than extra quadrature nodes.

Premature input compression changes the correction by 4.888903e-5 RMS for high-pair Galerkin, close to the entire 4.936738e-5 one-step DF error. For the low-frequency field, the placement contrast is ~7e-18. This is an appropriately conditional representation result, not a universal benefit.

Output compression is not universally harmless: it changes the rough Galerkin correction by 2.371313e-6 RMS and discrete high-pair by 2.855027e-6. These changes can *improve* or worsen endpoint error because errors cancel; a component norm is not an additive error budget.

The high-pair same-grid Galerkin versus refined continuum discrepancy is 4.690780e-6 RMS, larger than the 7.770859e-8 quadratic+cubic same-grid error. The actual 2N/4N spatial reference difference is only 1.685626e-10. Near-Nyquist discrepancy is 1.315369e-6; rough is 7.606947e-7. A high-accuracy coarse temporal method is therefore not automatically a high-accuracy continuum solver. The discrete-versus-continuum discrepancy can reach 0.009400947 RMS on the near-Nyquist field; it is a different equation, not temporal failure.

Native CPU diagnostic warm medians across four fields: DF 0.2123 ms, ETDRK4 0.1823 ms, normalized GL2 0.7166 ms, conditioned GL2 0.8952 ms, GL4 full 0.8656 ms, small diagnostic FNO 0.6407 ms. These are configuration-specific unqualified latencies, not matched-accuracy rankings and not GPU predictions. In high-pair instrumented conditioned profiling, transforms are 23.83% of operator self-time, pointwise arithmetic 17.92%, copies 8.89%, allocations 5.68%, learned dense layers only 0.32%, remainder miscellaneous/nonlinear operations. Increasing MLP capacity addresses little of measured inference cost.

## Protected prototypes

### X01: accurate temporal representation, unfavorable economics

Source: `explore-X01/prototype_rows.json`, `temporal_query_rows.json`, `temporal_amortization.json`, `temporal_encoding.json`.

The degree-12 Chebyshev kernel encoding approximates the second-Picard response of a fixed 1D constant-generator semilinear problem, not the entire nonlinear solver. It has one synthetic state, 361 signed mode pairs, 20 correlated horizons. Its maximum relative RMS discrepancy is 6.150338e-16; independent 64-node quadrature differs by 4.470280e-19 absolute. That is excellent representation fidelity.

| Method | Build | Warm query | Max relative RMS |
|---|---:|---:|---:|
| Ours compact temporal encoding | 520.967 µs | 18.641 µs | 6.15e-16 |
| Analytic cached exact pairs | 2.270 µs | 19.241 µs | 0 |
| Analytic normalized GL2 | 199.781 µs | 17.771 µs | 2.86e-6 |
| Analytic normalized GL4 | 100.855 µs | 19.561 µs | 1.64e-12 |
| Theirs DOP853 dense output | 906.637 µs | 13.681 µs | 1.77e-11 |

The recorded cold-inclusive break-even versus exact cached pairs is 865 queries, far beyond the registered 32; with existing operator basis it is still 139. The compact state is 2,704 bytes versus 5,776 pair-product bytes, but its reusable operator basis adds 37,544 bytes. Array storage is not process peak memory.

An additional review inference: at the registered 0.001 relative-error requirement, GL2 already qualifies and has both lower build and lower query time than the compact encoding. The compact representation is therefore dominated by this control under the measured build-plus-linear-query cost model. Even demanding much higher fidelity, DOP853 overtakes the compact encoding after approximately 78 queries, before its 865-query crossover with the cached exact rule. Claims of amortization must compare the whole credible envelope, not one convenient comparator. Tiny ~0.6 µs differences have no timing interval and must not be treated as robust crossover estimates.

Decision: retain the representation result, stop the current speed claim. Only a new workload or structure reducing encoding/query costs merits further work. State changes require re-encoding; operator/horizon-range changes require a new basis.

### X02: information obstruction and narrow successful history fit

Source: `explore-X02/closure_rows.json`, `closure_fit.json`, `prototype_rows.json`.

Twenty independent held-out twin parents have identical resolved current coefficients but future target separation at least 8.803219e-6. Any deterministic method receiving only that identical coarse information must return the same output, so its pair RMSE cannot be below half the target separation. Aggregated lower bound is 2.421671e-5, exactly matching the instantaneous fitted control here. This is a meaningful information result independent of neural capacity.

This obstruction does not apply directly to current rank1/GL2 models or the full-grid FNOs: their physical/neural branches receive the unresolved full state. It supports a possible cheaper coarse-state architecture only if the unavailable information is supplied through valid history or extra state variables.

| Predictor | Held-out RMS of one complex retained mode |
|---|---:|
| Instantaneous fit | 2.421671e-5 |
| Deterministic history persistence | 2.445348e-7 |
| Ours two-coefficient history fit | 5.993303e-11 |
| Full-state instantaneous first-order oracle | 2.035679e-7 |

The learned numerical rule is `(0.00998051017 - 0.000178092509*mean) * past_derivative`, with shared real/imaginary coefficients. No network is required. Persistence already improves RMSE about 99×; the fitted rule improves persistence ~4,080× in this very narrow construction. Independent RK4/DOP853 subset agreement is 1.67e-16.

History is generated by a backward reference solve, costing 240.8 µs per state, while the fitted query is only 1.16 µs. Cheap inference alone is not deployment economy. The task predicts one mode at one future/history horizon, with a fixed pair of unresolved frequencies, fixed physics, and smooth noiseless histories. Coefficient-shift, history noise, causal startup, multi-mode outputs and long rollout drift remain untested. The full-state oracle is only first-order; outperforming it does not beat a strong full-state analytic integrator. The successful rule may approximate a simple known response coefficient; test a derived deterministic history coefficient before ascribing value to learning.

Decision: advance one bounded causal-forward-history pilot with an analytic history comparator and charged acquisition cost. Do not promote this to a universal learned closure.

### X03: work reduction fails to become time reduction

Source: `explore-X03/selection_rows.json`, `selection_fit.json`, `selection_phase_audit.json`.

| Selector, 24 held-out parents | Maximum relative error | Mean retained pairs | Geometric full/selector speed ratio |
|---|---:|---:|---:|
| Deterministic L1 tail | 0.5762% | 35.09% | 0.7497× |
| Ours fitted rank rule | 0.6160% | 39.95% | 0.6719× |

Both have zero violations of the 1% requirement, but neither approaches the 1.10× speed gate. Full-pair median across parents is 42.69 µs; deterministic and fitted medians are 50.86 and 57.54 µs. Geometric paired ratios and ratios of marginal medians are different summaries, so the latter should not be substituted for the registered result. Feature extraction, sorting/masking/gathering and kernels erase arithmetic savings.

One amplitude parent with eight phase variants demonstrates phase-blind features (maximum difference 1.11e-16), but not an actual tolerance failure: deterministic errors 0.1446–0.2889%, fitted errors zero. The eight phases are not eight independent parents. Magnitude-tail control is not a relative error certificate under cancellation.

Decision: stop both current selectors. A substantially larger interaction workload, fixed-shape fused blocks, or a cheaply certified omission bound would be a new premise, not a reason to rerun the same failed efficiency design at greater expense.

## Metric gaps relevant to the main review

- Physical high/low spectral-error cutoff is radial `N/4`; it changes between 32² and64² and differs from the model's fixed box cutoff. Rebin to common physical wavenumbers before interpreting resolution effects as the same spectral-band task.
- FP64 teachers are converted to FP32 training targets, and defect normalization uses an absolute 1e-6 RMS floor. The extremely easy error regime can be limited by model precision/roundoff even when teacher uncertainty is much lower.
- Analytic nulls, positivity observations and short-schedule accuracy do not establish stability, positivity or monotonicity of the corrected method under arbitrary rollouts. Long-time and perturbation robustness are open.
- “Upper RMS/max” adds an empirical reference uncertainty estimate. It is not a certified mathematical bound.
- A parameter reduction is not an inference-cost or information reduction when the full physical state and expensive formula remain available.
- Equal data, equal updates, capped trial time and optimized validation frontiers are distinct fairness objectives. This program improves transparency without proving that every competing model reached a competitive optimum.
- Descriptive geometric ratios can hide absolute irrelevance on easy workloads, tail failures or absence of a deployment margin. Always pair them with absolute errors, both norms, field coverage, failures, uncertainty and cost.
- Excellent score/check completion for an analytic reference can be tautological; scores measure declared evidence attainment, not mathematical novelty or model quality.

## Research judgment

The defensible mechanism remains preservation of signed nonlinear interactions before projection, with a useful finite-amplitude expansion and exact physical nulls. The diagnostic evidence points to missing higher-order interaction shape and, on continuum targets, spatial closure as more important than increasing quadrature order. The causal-history prototype gives the clearest protected new formulation worth a small next test. The reusable-kernel and adaptive-selection prototypes provide valuable negative cost evidence that should be respected. None of these mathematical observations substitutes for the main native learned-attribution and matched-accuracy cost results.
