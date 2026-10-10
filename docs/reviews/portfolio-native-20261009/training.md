# Native portfolio training, architecture, and fairness review

This is a post-hoc review of `fedora-portfolio-20261009T183307014599Z`, using exact verified archive bytes through `review_io.py`. Sources: `freeze/catalog.json`, all six `train-{A,B}-*/learning_curves.jsonl`, and `train-A-000/protocol.json`. Computed tables are in `training-summary.json`; the reproducible parser is `training-analysis.py`. No source, checkpoint, protocol, or original result was changed.

## What was actually trained

There are **176 selected model instances, 536 completed trials, and 40,424 learning-curve rows**. Selection retained 150 fitted checkpoints, six initializations, and 20 frozen controls. No trial records a numerical failure, and **no gradient trial stopped before its declared update count**. Total recorded trial time is **2,341.63 s (39.03 min)**, of which **1,202.88 s (20.05 min)** is synchronized optimizer-loop time. Neither number includes all teacher preparation, validation-frontier timing, scheduler overhead, or campaign analysis; they are not total campaign cost.

The six initialization selections are exactly the three paired seeds of `continuum/quad2_amplitude/n8` and `continuum/quad2_linear/n8`. All six fit optimizers reported convergence; their fitted training solution did not beat normalized analytic initialization on validation. This is a generalization/selection result, not a crashed fit. The scalar and affine fits are deterministic for the same data, so their three identical seed instances are not three independent fit replications.

All 12 primary `quad2_conditioned` instances, and all 12 `conditioned_rich` instances, selected update **200**, the last permitted update. This does not prove that more training would help confirmation, but it rules out describing these runs as demonstrated optimization convergence. Conversely, FNO instances all selected trained checkpoints, not initialization, so their poor final behavior cannot be explained away by saying they were never trained.

## What these architectures are

The core equation is periodic logistic reaction–diffusion. Except for direct FNO, models use a known numerical physical backbone; discrete reaction is exact logistic while Galerkin reaction is integrated with projected RK4. The main learned model adds an analytically constructed transported quadratic interaction defect, formed from the full state before output lowpass compression. A tiny network chooses its scalar gain from current-state mean, nondimensional reaction time, and active diffusion rate. The symmetric quadrature node pair is trained globally per model, not predicted afresh from each input. This is a learned physical time-step correction, not a generic MLP predicting a whole future field.

The global features contain no future state or teacher information. Their mean and spectral-energy-derived active rate do, however, discard phase/layout distinctions; the analytic interaction formula still retains those distinctions. Increasing feature richness cannot compensate for an omitted spatial residual shape if the output remains just one scalar times the same analytic field.

| Family | Stored/fitted parameters | What changes |
|---|---:|---|
| `historical_half` | 43 stored, **0 fitted** | Unchanged frozen historical half-normalized pair |
| `quad2_fixed`, `quad2_full`, `quad2_input` | 0 | Normalized GL2 with output cutoff, no cutoff, or premature input cutoff |
| `quad2_amplitude` | 1 fitted | Bounded global scalar amplitude, fixed nodes |
| `quad2_nodes` | 1 fitted | One global symmetric node pair, unit gain |
| `quad2_joint` | 2 fitted | Global nodes plus global gain |
| `quad2_linear` | 4 fitted buffer coefficients | Bounded affine gain of three physical features |
| `quad2_conditioned`, `quad2_conditioned_full` | 43 fitted | Three-feature 3→8→1 conditioner, global gain logit and node logit |
| `quad4_fixed`, `quad4_full` | 0 | Normalized four-node analytic controls |
| `quad4_conditioned` | 44 fitted | Two global symmetric node pairs plus scalar conditioner |
| `analytic_quad_cubic` | 0 | Known quadratic and cubic interaction formulas |
| `residual_quad2` | 41 fitted | Fixed GL2, small gain adjustment bounded to (0.75, 1.25) |
| `conditioned_rich` | 75 fitted | Seven-feature conditioner, global gain and node logits |
| `band_gain` | 59 fitted | Three separate DC/lower/upper retained-band gains |
| `fno_small` | 47,057 fitted | Width 16, depth 2, four spectral modes, full-grid local path; physical hybrid |
| `fno_standard` | 1,258,977 fitted | Width 32, depth 4, eight modes, local path; physical hybrid |
| `direct_fno` | 1,258,977 fitted | Same large network, no DF backbone; learns full update |
| DF, RF, ETDRK4 | 0 | Established numerical controls |

FNO has a real full-grid local branch and nonlinear layers, so it is not artificially reduced to a lowpass linear operator. It receives eight channels including full and centered fields, step, reaction/diffusion parameters and grid size. Hybrid output is scaled by `(h/t_ref)^3`; direct output by `h/t_ref`. These are local adaptations, not reproductions of a published FNO benchmark.

## The most consequential fairness result: FNO remains inadequate

The primary selection loss is dimensionless `(mean(error²) + 0.1 max(error²)) / max(DF MSE, reference uncertainty², 1e−12)`, field-weighted on validation. Lower is better. The following values are median selected validation objectives across three seeds at **32 training parents**, retaining tracks separately:

| Method | Discrete target | Galerkin/continuum target |
|---|---:|---:|
| DF | 0.90099 | 0.87222 |
| Normalized GL2 | 0.16248 | 0.16726 |
| Scalar amplitude | 0.15853 | 0.16428 |
| Affine gain | 0.15152 | 0.15471 |
| Primary conditioner | **0.15040** | **0.15249** |
| Analytic quadratic+cubic | **0.01876** | **0.04111** |
| FNO small | 3,142.80 | 912.48 |
| FNO standard | 596.34 | 1,041.13 |
| Direct FNO | 57,832,502 | 68,363,440 |

For example, the standard FNO's selected loss is about **662× DF** on the discrete track and **1,194× DF** on the Galerkin track. These are loss ratios, not RMS ratios or confirmation results. Median selected validation RMS is approximately 4.9e−5–8.8e−5 for hybrid FNO versus approximately 4.9e−6–5.4e−6 for the primary conditioner. Direct FNO remains around 0.027–0.030 RMS. A model can improve thousands-fold from a poor starting point and remain unsuitable as the basis for a scientific superiority claim.

Every recorded default-clipped FNO update exceeded its unit clipping threshold: 100% of 2,360 clipped small-FNO updates, 100% of 2,360 clipped standard-FNO updates, and 100% of 3,360 clipped direct-FNO updates. The logs report **pre-clipping** gradient norms. Medians across all optimizer controls are roughly 78,277, 54,580, and 1.47e9 respectively. These values span differently scaled losses; they are a conditioning diagnostic, not comparable physical sensitivities. The alternate unclipped, 0.01-loss-scale control was implemented and sometimes selected. Therefore clipping alone is not established as the cause.

All three FNO families searched two learning rates and two optimizer controls, then retrained from the same seeded initialization for 200 updates using the selected setting. The better tuning checkpoint could be retained. There was **no actual temporal-scale search**: every recorded `t_ref` is 0.2 despite a source comment describing configurable scale search. There was no optimizer/scheduler/capacity sweep beyond the declared alternatives or authoritative implementation reproduction.

The 120-second per-trial cap never bound these trials. Final FNO training took approximately 3.4–15.5 seconds per hybrid trial and 4.1–4.5 seconds per direct trial; update count, not equal walltime, limited work. Equal update caps are not equal training compute. Strong analytic initialization is a material advantage of our formulation, but a fair hybrid FNO should receive a baseline-preserving residual initialization and a well-scaled target before declaring it a strong alternative. The nonzero FNO head was deliberately chosen to activate all layers initially; its residual can nevertheless dwarf the tiny physical defect being learned.

## Data size, loss, and learned behavior

Increasing training parents from eight to 32 gives modest validation improvements for the primary conditioner: paired-seed geometric loss ratios `loss(n8)/loss(n32)` are **1.0172 discrete** and **1.0424 Galerkin** (approximately 1.7% and 4.1% loss reductions). Affine fits give 1.0471 and 1.0811. Node-only and four-node conditioned controls show almost no additional validation benefit. FNO data trends are inconsistent: small FNO 0.3359 discrete / 2.3647 Galerkin; standard FNO 4.3689 discrete / 0.5691 Galerkin. Several apparent improvements are strongly seed-dependent. These are two points on one regime-balanced nested training subset, not a fitted sample-complexity law, and there are no independently resampled training cohorts. Increasing size also reduces effective passes per example at fixed updates.

The independent endpoint parser's **post-hoc** matched confirmation data-size contrasts (`statistics-followup.json:data_efficiency`) are more encouraging for the primary model: RMS(n8)/RMS(n32) is **1.0842 [1.0262, 1.1759] discrete** and **1.1190 [1.0526, 1.2088] Galerkin**, using its paired 24-field/three-seed bootstrap and all declared schedules. Joint eligible cells increase only **1,531→1,535 of 1,728** and **1,550→1,559 of 1,728** respectively. Those row denominators include repeated grids, schedules and seeds; there are still only 24 independent fields. This improvement over the smaller-data version does not imply improvement over normalized analytic quadrature. FNO data-size intervals cross one: small FNO discrete 0.891 [0.561,1.420], Galerkin 0.850 [0.349,2.364]; standard FNO discrete 1.197 [0.728,1.693], Galerkin 0.802 [0.510,1.229]. These exploratory intervals are unadjusted for multiple comparisons.

Lower selected validation loss does **not** imply lower equally weighted confirmation RMS, better maximum error, or more feasible solver schedules. The objective weights samples by the DF defect (with a floor), emphasizes maximum error, and is computed on one-step original-state predictions. Training contains four endpoint times (.02, .04, .08, .12), no intermediate evolved-state or rollout loss. Validation adds .03, .06, .10, .12 and .24. Consequently the .24 endpoint is held out from training but **not** an untouched time-extrapolation test: model selection saw it. Multi-step confirmation remains a genuine composition test within the declared short horizons, not long-time stability evidence.

The three n32 primary node pairs shift the GL2 left node from 0.211325 to approximately **0.2051–0.2061 discrete** and **0.2027–0.2041 Galerkin**, symmetric about 0.5. Scalar-only n32 fits choose amplitudes **1.06630 discrete** and **1.09585 Galerkin**, identically across the three nominal seeds. This provides a possible route to a compact numerical rule, but the more flexible primary model is not simply a constant gain.

The original catalog records only three training and three validation response probes per model, generally three horizons of one parent. To avoid inferring whole-cohort behavior from them, `training-response-audit.py` reconstructed 32 selected tiny models from their fully retained parameter-report tensors, regenerated declared initial fields, and evaluated gains on CPU at grids 32 and 64 and horizons .03/.06/.10/.12/.24. It performed **no PDE solve, training, timing comparison, or new accuracy test**. Reconstructed saved-probe gains agree within 1.19e−7. This is explicitly post-hoc descriptive analysis of existing model parameters.

At grid 32 on confirmation initial fields, primary gain ranges are **0.9568–1.4409 discrete** and **0.9586–1.5419 Galerkin** across three seeds and the five queried times. Mean gains are about 1.052 and 1.090. Diffusion-stiff fields receive the largest average corrections (1.208 / 1.306); high-pair and near-Nyquist regimes also receive stronger gain. Low-frequency fields average 0.984 / 0.991. This is interpretable input adaptation, but usefulness must come from measured errors, not from the existence of variation.

A strict identifiability limitation remains: the conditioner final bias and global amplitude logit enter the same sum before `tanh`, so an equal-and-opposite shift produces exactly the same correction. Their separate values have no unique physical interpretation. Node/gain compensation may create further practical ambiguities. The affine least-squares design has full recorded rank four; that does not establish full-network identifiability. Its fit log's `fields=128` at n32 actually counts **32 parents × four horizons**, not 128 independent fields. Report statistical units using parent IDs, not that incidental label.

## Decisions

1. Retain normalized analytic and scalar/affine fitted controls as the attribution standard. The new controls successfully expose improvements that should not be credited to neural state dependence.
2. Keep the primary/rich variants as bounded diagnostic models, not the default deployment winner. Last-update selection supports one controlled optimization-convergence check, but confirmation regression should take priority over simply adding updates.
3. Repair the FNO baseline before making broad neural-method claims: verify a near-baseline initialization, scaled residual target, optimization curves, equal-compute opportunities, and implementation parity against a credible reference. Preserve all present results as the bounded local adaptation comparison they actually are.
4. Evaluate objectives explicitly against the intended claim: mean/RMS, maximum error, mean drift, and joint feasible coverage must remain separate. Add evolved-state/short-rollout supervision only as a new declared treatment, with fresh confirmation if it changes selection.
5. Prefer investigating missing physical residual shape and lower-order numerical structure to increasing conditioner width indiscriminately. The analytic cubic control's validation advantage is large; the neural gain's is small and need not survive another metric.
6. Do not call the 8→32 experiment a demonstrated data-efficiency law, the three deterministic fitted-control seeds independent replications, or the global node values uniquely identified physics. Keep the response audit exploratory and source-linked.
