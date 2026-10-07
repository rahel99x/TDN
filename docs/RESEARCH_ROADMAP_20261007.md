# TDN research roadmap — 7 October 2026

**Recommendation: pursue a compact, physically derived finite-step defect correction only where an optimized classical core leaves useful headroom. First repair the continuum target/acceptance mismatch, tune the actual diffusion-first core, and reserve genuinely independent fields for confirmation.** More updates or a larger network do not address the dominant observed failures.

This review combines ten recent experiment groups, a complete ledger of **62 supported work items** (51 active and 11 addressed regression controls), **24 proposed mechanisms**, and **14 closely related primary-source work records**. The tabs cover proposed mechanisms, gaps and required work, projected results, mathematical review, and evidence/papers. The Markdown and JSON companions preserve the same information for review and future protocol design.

The core opportunity is to learn the **finite-time interaction defect of a known physical solver**, preserving signed nonlinear source interactions before compression. A low-dimensional temporal MLP can parameterize part of this kernel, but the hypothesis is in the physically constrained source/transport composition, its finite-step damping, and its accuracy–cost behavior. A generic time-conditioned MLP alone is not a distinctive operator architecture.

### What the recent results establish

| Observation | Interpretation | Immediate consequence |
| --- | --- | --- |
| All eight native Fedora agenda stages completed and their 341 scientific files were sealed/verified; 0/24 predeclared joint comparisons passed. | Execution correctness and reproducible negative results are strong; a trained complete solver advantage is unestablished. | Retain the gates and report failed scientific endpoints explicitly. |
| Diffusion-first Strang improved the recorded validation objective 23.68× before learning; bare DF is about 3.9× cheaper than Source/rank-2 and reaches all primary discrete frontier targets. | Orientation and the physical core account for a large part of the apparent gain. | Establish headroom over optimized DF before training a corrector. |
| Source is about 1.74× faster than the deep matched FNO at the primary discrete frontier, but strict maximum-error coverage is 477/576 versus FNO 533/576 at 2e-5. | A neural speed comparison can coexist with weaker strict accuracy and no classical utility. | Include cheap one-layer FNO, every schedule, both norms, and total deployment work. |
| Source's worst discrete maximum error is 0.04098 versus its paired physical base about 3.53e-5; rank-2's worst is about 3.38e-4. | Finite-step and ordering failures can overwhelm a very accurate base; rank-2 is a promising compact robustness lead. | Test exact finite-step response and trust envelopes; replicate rank-2 beyond its single confirmation seed. |
| Discrete policy: 0/1179 observed false accepts. Continuum policy: 306/410 accepted outputs definitely fail the recorded target. | Temporal agreement is not an estimate of spatial bias; success on the discrete equation does not transfer automatically. | Separate temporal, spatial and reference uncertainty before accepting continuum outputs. |
| Eight controlled fields share one underlying random phase cluster, crossed with physics/grid factors. | Thousands of rows do not provide thousands of independent fields. | Treat inspected data as development; freeze new independent phases/coefficients for confirmation. |
| Source doubling has a favorable median attributed cost but about 1.397× the summed ETDRK4 cost. | Tails/rejects matter, and component-attributed policy work is not standalone deployment timing. | Measure complete decisions, estimates, rejects, fallback, totals and p95 against eligible operational classical controls. |

### Scope and interpretation

The literature window is **2025-10-07 through 2026-10-07**, inclusive. First-publication date, venue/version date and code-release date are different. Older first preprints with recent editions, and manuscripts without a verified first-publication date, are visibly labeled as context. The original FNO work is background; a local adapted `fno_split` comparison does not reproduce or defeat that original paper.

The ledger is exhaustive for the supported shortcomings identified in the inspected artifacts and source review. It is not a claim to enumerate all possible model flaws or all papers published worldwide. Every projected outcome is an **untested directional hypothesis** with a proposed falsifier. Any new numerical success threshold or stage scope below must be frozen in a new protocol; it does not rewrite historical gates.

Strong foundations to preserve: native allocated-CUDA tests, physical-null controls, training-only normalization, validation-only checkpoint selection, refined independent FP64 teachers, immutable source/protocol/file seals, explicit initialization/failure outcomes, and complete Tower lineage. Scientific negative results remain completed computational stages.


## Proposed mechanisms

P0 = prerequisite or blocking correction; P1 = main hypothesis after prerequisites; P2 = conditional extension. Established mathematics is identified explicitly. The integrated finite-step, phase-aware low-call hypothesis is promising, but neither novelty nor superiority is proven.

| ID / mechanism | Priority | TDN gap targeted | Related work | Projected direction |
| --- | --- | --- | --- | --- |
| [M00 — Headroom map and fused classical reference](#mechanisms-m00) | P0 | D01, D02, D03, D22 | [P01](#paper-p01), [P09](#paper-p09), [P14](#paper-p14) | Find regimes where extra correction accuracy can actually replace classical steps; otherwise establish that the classical solver is the right answer. |
| [M01 — Consistent spatial target and two-resolution discrepancy](#mechanisms-m01) | P0 | D04, D09, D26, D27, D28 | [P02](#paper-p02), [P10](#paper-p10), [P12](#paper-p12) | Explain and reduce the common continuum floor; expose continuum errors before an otherwise small temporal indicator accepts them. |
| [M02 — Exact finite-step quadratic DF defect](#mechanisms-m02) | P1 | D05, D17, D19, D21, D22, D29, D34, D51 | [P01](#paper-p01), [P05](#paper-p05), [P07](#paper-p07), [P08](#paper-p08) | Replace stiffness-growing cubic extrapolation with a response that is exact at quadratic amplitude and damped at large diffusion/long times. |
| [M03 — Physical factors plus learned rank-1/rank-2 remainder](#mechanisms-m03) | P1 | D13, D14, D15, D20, D21, D34, D35, D37 | [P01](#paper-p01), [P10](#paper-p10), [P11](#paper-p11) | Retain phase-aware robustness with fewer transforms than a generic high-rank branch; rank 1 may offer a useful cost point. |
| [M04 — Signed nonlinear sources before output compression](#mechanisms-m04) | P1 | D16, D17, D24, H01, H02 | [P01](#paper-p01), [P11](#paper-p11), [P12](#paper-p12) | Preserve high-high-to-low/zero interactions and remote receiver response that an input-only cutoff or post-transport point gate erases. |
| [M05 — Product-consistent anti-aliasing and grid-scaled filters](#mechanisms-m05) | P0 | D04, D24, D26, D28, D37 | [P02](#paper-p02), [P10](#paper-p10), [P12](#paper-p12) | Remove spurious retained-mode folding and enforce physical scale when resolution changes. |
| [M06 — Stiffness-aware trust envelope for the correction](#mechanisms-m06) | P0 | D05, D08, D20, D21, D29, D51 | [P01](#paper-p01), [P04](#paper-p04), [P05](#paper-p05) | Suppress harmful corrections where the base is already accurate or the finite-step response is outside validated support. |
| [M07 — Conditional solver bank with a classical safe path](#mechanisms-m07) | P1 | D01, D02, D05, D10, D20, D30, D33, D49 | [P03](#paper-p03), [P06](#paper-p06), [P07](#paper-p07) | Skip unnecessary neural work on easy states; spend correction/refinement work only where the expected accuracy benefit pays for it. |
| [M08 — Cheap embedded estimator and stability-weighted residual](#mechanisms-m08) | P1 | D09, D10, D26, D31, D32, D33, D49, D50 | [P01](#paper-p01), [P02](#paper-p02), [P07](#paper-p07), [P08](#paper-p08) | Replace expensive repeated JVP work or blind step agreement with a target-correct, shared-computation error signal. |
| [M09 — Function-level calibrated total-error and selective risk](#mechanisms-m09) | P1 | D09, D11, D12, D26, D27, D30, D47, D49, D50 | [P02](#paper-p02), [P06](#paper-p06) | Turn empirical envelopes into explicit distributional statements where assumptions hold; detect spatial/grid shift instead of silently transferring coverage. |
| [M10 — Unequal-step generator consistency and intermediate supervision](#mechanisms-m10) | P1 | D05, D08, D19, D29, D34, H04 | [P01](#paper-p01), [P05](#paper-p05), [P07](#paper-p07), [P08](#paper-p08) | Reduce order sensitivity and long-rollout drift while preserving genuine dynamics at arbitrary unseen individual steps. |
| [M11 — DF-tuned defect loss with regime and maximum-error control](#mechanisms-m11) | P0 | D06, D07, D20, D39, D42, D48, D52 | [P01](#paper-p01), [P03](#paper-p03), [P07](#paper-p07) | Reduce harmful corrections in already-solved cases and focus capacity on true defects rather than aggregate normalized MSE. |
| [M12 — Dynamic feasible mean–variance closure](#mechanisms-m12) | P2 | D17, D18, D23, H03 | [P01](#paper-p01), [P04](#paper-p04), [P05](#paper-p05) | Improve a verified mean-dominated regime while repairing the frozen closure binary-field boundary defect. |
| [M13 — Dimensionally consistent multi-band/fractional features](#mechanisms-m13) | P2 | D21, D24, D25, D28, D35, D37 | [P11](#paper-p11), [P12](#paper-p12) | Represent unresolved spectral bands economically without grid-dependent derivative units or unintended constant leakage. |
| [M14 — Geometry-aware and anisotropic transport](#mechanisms-m14) | P2 | D25, D28, D38 | [P10](#paper-p10), [P13](#paper-p13), [P14](#paper-p14) | Extend phase/interaction structure to geometry only when anisotropic or irregular-domain headroom is independently demonstrated. |
| [M15 — Cost-aware algebraic reuse, fusion and batching](#mechanisms-m15) | P1 | D10, D14, D15, D31, D32, D35, D36, D37, D40, H06, H07 | [P03](#paper-p03), [P04](#paper-p04), [P14](#paper-p14) | Lower real inference/estimator overhead through identical-map reuse and equitable optimized baselines. |
| [M16 — Faithful neural comparison and strongest classical utility](#mechanisms-m16) | P0 | D01, D03, D15, D41, D48, H08 | [P09](#paper-p09), [P11](#paper-p11), [P14](#paper-p14) | Determine whether gains belong to the physical solver, interaction bias, implementation or a genuine neural benchmark advantage. |
| [M17 — Factor-isolated stress and coefficient-transfer bank](#mechanisms-m17) | P2 | D25, D27, D28, D29, D38, D39 | [P01](#paper-p01), [P03](#paper-p03), [P06](#paper-p06) | Identify where finite-amplitude, roughness, stiffness, initial state or resolution actually causes failure. |
| [M18 — Fresh independent fields and hierarchical inference](#mechanisms-m18) | P0 | D11, D12, D13, D25, D27, D30, D39, D40, D47, D48, D52, H04 | [P02](#paper-p02), [P06](#paper-p06) | Give uncertainty intervals with a real independent unit and protect confirmation from architecture-selection reuse. |
| [M19 — PDE and boundary portability ladder](#mechanisms-m19) | P2 | D38 | [P01](#paper-p01), [P03](#paper-p03), [P10](#paper-p10), [P13](#paper-p13), [P14](#paper-p14) | Assess usefulness beyond favorable scalar periodic logistic reaction–diffusion. |
| [M20 — Order-constrained learned embedded stages](#mechanisms-m20) | P1 | D19, D29, D34, D51 | [P07](#paper-p07), [P08](#paper-p08) | Learn a useful estimator/stage extension without silently sacrificing order or relying on an RK4 name. |
| [M21 — Physical monotonicity and growth-envelope diagnostics](#mechanisms-m21) | P1 | D05, D20, D29, D51, H03 | [P01](#paper-p01), [P05](#paper-p05) | Expose or constrain unstable learned sensitivity beyond the existing null/interval tests. |
| [M22 — Effective configuration, provenance and reproducible measurement](#mechanisms-m22) | P0 | D43, D44, D45, D46, D52, H05, H09, H10 | [P09](#paper-p09), [P14](#paper-p14) | Remove human-label ambiguity, preserve repaired workflows and make cost/scientific attribution reproducible. |
| [M23 — Dedicated cubic Volterra remainder](#mechanisms-m23) | P2 | D17, D18, D21, D34, H01 | [P01](#paper-p01), [P07](#paper-p07) | Address finite-amplitude failures remaining after the quadratic physical kernel is accurate. |

### Feasible combinations with isolated controls

| Combination | Mechanisms | Purpose | Required ablation | Projection |
| --- | --- | --- | --- | --- |
| C0 — Prerequisite screen | [M00](#mechanisms-m00), [M01](#mechanisms-m01), [M05](#mechanisms-m05), [M16](#mechanisms-m16), [M18](#mechanisms-m18), [M22](#mechanisms-m22) | Resolve headroom, target and evidence before architecture training. | Classical only; fixed independent development fields; compare fusion and target choices separately. | Could terminate a neural utility branch correctly; no guaranteed profitable regime. |
| C1 — Primary compact finite-step residual | [M02](#mechanisms-m02), [M03](#mechanisms-m03), [M04](#mechanisms-m04), [M11](#mechanisms-m11) | Known DF core + exact quadratic factors + low-rank learned remainder. | Ranks0/1/2, analytic2node response, transportedsource, cheapFNO; each component alone before combination. | Best supported architectural hypothesis for stiff/order robustness; speed remains contingent on larger admissible steps. |
| C2 — Operational error/cost-controlled hybrid | [M01](#mechanisms-m01), [M06](#mechanisms-m06), [M07](#mechanisms-m07), [M08](#mechanisms-m08), [M09](#mechanisms-m09), [M10](#mechanisms-m10), [M15](#mechanisms-m15) | Add deployment only after the model passes independent prescribed schedules. | Cheapdoubling versus interiorresidual; temporalonly versus temporal+spatial; routeron/off; exact standalone time. | Fewer harmful accepts and lower tails are testable; expensive reliability may erase utility. |
| C3 — Finite-amplitude side branch | [M04](#mechanisms-m04), [M12](#mechanisms-m12), [M23](#mechanisms-m23) | Separate mean closure from higher-amplitude causal corrections. | Meanonly, zero-mean spatial, cubic only, combined with zeromode double-count prevention. | Conditional on diagnosed mean/cubic headroom; current evidence does not support making closure default. |
| C4 — Later transfer/geometry branch | [M05](#mechanisms-m05), [M13](#mechanisms-m13), [M14](#mechanisms-m14), [M17](#mechanisms-m17), [M19](#mechanisms-m19), [M21](#mechanisms-m21) | Broaden physics only after scalar mechanism evidence. | Physicalspacing versus learnedspacing; isotropic versus anisotropic; periodic versus declaredboundary. | Potential portability; no current evidence of cross-PDE dominance. |

Combination effects should be reported on independent paired fields, with a declared lower-is-better response such as log(error + floor) or log(cost):

\[
I_i=Y_{AB,i}-Y_{A,i}-Y_{B,i}+Y_{0,i}.
\]

A negative interaction is a descriptive improvement beyond the additive log effects; uncertainty and complete accuracy/cost gates are still required. Run single mechanisms first, then only supported sparse combinations. Spatial upgrades must be applied to both learned and classical arms with the same new target-specific teacher.

<a id="mechanisms-m00"></a>

### M00 — Headroom map and fused classical reference

**Priority:** P0. **Targets:** D01, D02, D03, D22. **Related records:** [P01](#paper-p01), [P09](#paper-p09), [P14](#paper-p14).

Implement/profile fused variable-step DF, both orientations, ETDRK4 and GL3; retain all-domain and screened-stratum reports separately; no new training needed first.

**Why it could help:** Find regimes where extra correction accuracy can actually replace classical steps; otherwise establish that the classical solver is the right answer.

**Status:** established numerical-analysis and benchmarking control. [Mathematical requirements](#math-m00).

<a id="mechanisms-m01"></a>

### M01 — Consistent spatial target and two-resolution discrepancy

**Priority:** P0. **Targets:** D04, D09, D26, D27, D28. **Related records:** [P02](#paper-p02), [P10](#paper-p10), [P12](#paper-p12).

Declare FD/nodal and spectral/Galerkin tracks, teachers and projections; add bounded N/2N/4N checks, error-order detector and separate spatial cost.

**Why it could help:** Explain and reduce the common continuum floor; expose continuum errors before an otherwise small temporal indicator accepts them.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m01).

<a id="mechanisms-m02"></a>

### M02 — Exact finite-step quadratic DF defect

**Priority:** P1. **Targets:** D05, D17, D19, D21, D22, D29, D34, D51. **Related records:** [P01](#paper-p01), [P05](#paper-p05), [P07](#paper-p07), [P08](#paper-p08).

Use zero-subtracted stable quadrature or scaled divided differences, analytic logistic background, centered phases and independent amplitude/time teachers. One-node midpoint DF defect is identically zero.

**Why it could help:** Replace stiffness-growing cubic extrapolation with a response that is exact at quadratic amplitude and damped at large diffusion/long times.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m02).

<a id="mechanisms-m03"></a>

### M03 — Physical factors plus learned rank-1/rank-2 remainder

**Priority:** P1. **Targets:** D13, D14, D15, D20, D21, D34, D35, D37. **Related records:** [P01](#paper-p01), [P10](#paper-p10), [P11](#paper-p11).

Distinguish learned nominal rank, expanded commutator terms and coordinate Tucker rank; share only algebraically identical transforms; tune the actual DF base.

**Why it could help:** Retain phase-aware robustness with fewer transforms than a generic high-rank branch; rank 1 may offer a useful cost point.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m03).

<a id="mechanisms-m04"></a>

### M04 — Signed nonlinear sources before output compression

**Priority:** P1. **Targets:** D16, D17, D24, H01, H02. **Related records:** [P01](#paper-p01), [P11](#paper-p11), [P12](#paper-p12).

Build a signed discrete source dictionary; remove undeclared learned-branch bypass in the pure ablation; retain exact constant, zero-h, zero-r and zero-kappa checks.

**Why it could help:** Preserve high-high-to-low/zero interactions and remote receiver response that an input-only cutoff or post-transport point gate erases.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m04).

<a id="mechanisms-m05"></a>

### M05 — Product-consistent anti-aliasing and grid-scaled filters

**Priority:** P0. **Targets:** D04, D24, D26, D28, D37. **Related records:** [P02](#paper-p02), [P10](#paper-p10), [P12](#paper-p12).

Separate 3/2-padded quadratic RHS from exact pointwise logistic flow, which has infinitely many harmonics; test both target conventions and actual spacing rather than learned effective spacing.

**Why it could help:** Remove spurious retained-mode folding and enforce physical scale when resolution changes.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m05).

<a id="mechanisms-m06"></a>

### M06 — Stiffness-aware trust envelope for the correction

**Priority:** P0. **Targets:** D05, D08, D20, D21, D29, D51. **Related records:** [P01](#paper-p01), [P04](#paper-p04), [P05](#paper-p05).

Use analytic response/stiffness features and development-only trust calibration; test output-zero conjugate pairs and physical growth rather than universal contraction.

**Why it could help:** Suppress harmful corrections where the base is already accurate or the finite-step response is outside validated support.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m06).

<a id="mechanisms-m07"></a>

### M07 — Conditional solver bank with a classical safe path

**Priority:** P1. **Targets:** D01, D02, D05, D10, D20, D30, D33, D49. **Related records:** [P03](#paper-p03), [P06](#paper-p06), [P07](#paper-p07).

Compare bare DF, corrected DF and classical refinement with actual decision costs; freeze features and selection before fresh fields.

**Why it could help:** Skip unnecessary neural work on easy states; spend correction/refinement work only where the expected accuracy benefit pays for it.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m07).

<a id="mechanisms-m08"></a>

### M08 — Cheap embedded estimator and stability-weighted residual

**Priority:** P1. **Targets:** D09, D10, D26, D31, D32, D33, D49, D50. **Related records:** [P01](#paper-p01), [P02](#paper-p02), [P07](#paper-p07), [P08](#paper-p08).

Use two/three interior nodes with refinement validation, a valid continuous reconstruction and separate spatial budget; share stages only when the numerical maps match exactly.

**Why it could help:** Replace expensive repeated JVP work or blind step agreement with a target-correct, shared-computation error signal.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m08).

<a id="mechanisms-m09"></a>

### M09 — Function-level calibrated total-error and selective risk

**Priority:** P1. **Targets:** D09, D11, D12, D26, D27, D30, D47, D49, D50. **Related records:** [P02](#paper-p02), [P06](#paper-p06).

Do not inflate the cohort to meet statistics blindly: begin bounded reliability diagnostics; stronger guarantee is a separately justified scope with fresh calibration and confirmation.

**Why it could help:** Turn empirical envelopes into explicit distributional statements where assumptions hold; detect spatial/grid shift instead of silently transferring coverage.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m09).

<a id="mechanisms-m10"></a>

### M10 — Unequal-step generator consistency and intermediate supervision

**Priority:** P1. **Targets:** D05, D08, D19, D29, D34, H04. **Related records:** [P01](#paper-p01), [P05](#paper-p05), [P07](#paper-p07), [P08](#paper-p08).

Add generator/defect supervision and physical finite-h response; keep held-out individual h values and dense transients distinct from validation schedules.

**Why it could help:** Reduce order sensitivity and long-rollout drift while preserving genuine dynamics at arbitrary unseen individual steps.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m10).

<a id="mechanisms-m11"></a>

### M11 — DF-tuned defect loss with regime and maximum-error control

**Priority:** P0. **Targets:** D06, D07, D20, D39, D42, D48, D52. **Related records:** [P01](#paper-p01), [P03](#paper-p03), [P07](#paper-p07).

Tune LR/batch on effective orientation; include initialization selection and uncertainty floor; keep shared and per-arm tuning as separate fairness experiments.

**Why it could help:** Reduce harmful corrections in already-solved cases and focus capacity on true defects rather than aggregate normalized MSE.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m11).

<a id="mechanisms-m12"></a>

### M12 — Dynamic feasible mean–variance closure

**Priority:** P2. **Targets:** D17, D18, D23, H03. **Related records:** [P01](#paper-p01), [P04](#paper-p04), [P05](#paper-p05).

Dynamic spectral energy/third moment is required for an exact future law; enforce a=0 at constant fields, inward flux at z=1 and avoid double-counting the kernel zero mode.

**Why it could help:** Improve a verified mean-dominated regime while repairing the frozen closure binary-field boundary defect.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m12).

<a id="mechanisms-m13"></a>

### M13 — Dimensionally consistent multi-band/fractional features

**Priority:** P2. **Targets:** D21, D24, D25, D28, D35, D37. **Related records:** [P11](#paper-p11), [P12](#paper-p12).

Restore actual spacing/symbol explicitly or label dimensionless learned features; compare ordinary derivatives/physical factors before adding fractional orders.

**Why it could help:** Represent unresolved spectral bands economically without grid-dependent derivative units or unintended constant leakage.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m13).

<a id="mechanisms-m14"></a>

### M14 — Geometry-aware and anisotropic transport

**Priority:** P2. **Targets:** D25, D28, D38. **Related records:** [P10](#paper-p10), [P13](#paper-p13), [P14](#paper-p14).

Use operator-aware Krylov/rational/multilevel transport or weighted geometry operators; HNO/Radon are comparison directions, not justified default upgrades here.

**Why it could help:** Extend phase/interaction structure to geometry only when anisotropic or irregular-domain headroom is independently demonstrated.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m14).

<a id="mechanisms-m15"></a>

### M15 — Cost-aware algebraic reuse, fusion and batching

**Priority:** P1. **Targets:** D10, D14, D15, D31, D32, D35, D36, D37, D40, H06, H07. **Related records:** [P03](#paper-p03), [P04](#paper-p04), [P14](#paper-p14).

Profile FFTs, allocations, padding, host synchronization and reduction dispatch; cache only invariant prepared data; benchmark small and larger grids separately.

**Why it could help:** Lower real inference/estimator overhead through identical-map reuse and equitable optimized baselines.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m15).

<a id="mechanisms-m16"></a>

### M16 — Faithful neural comparison and strongest classical utility

**Priority:** P0. **Targets:** D01, D03, D15, D41, D48, H08. **Related records:** [P09](#paper-p09), [P11](#paper-p11), [P14](#paper-p14).

Keep neural-only and hybrid comparisons separate; add recent operator candidates only after faithful usable implementations and target match are verified.

**Why it could help:** Determine whether gains belong to the physical solver, interaction bias, implementation or a genuine neural benchmark advantage.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m16).

<a id="mechanisms-m17"></a>

### M17 — Factor-isolated stress and coefficient-transfer bank

**Priority:** P2. **Targets:** D25, D27, D28, D29, D38, D39. **Related records:** [P01](#paper-p01), [P03](#paper-p03), [P06](#paper-p06).

Use sparse factorial designs and bounded shards; preserve exact continuous fields when sampling paired grids.

**Why it could help:** Identify where finite-amplitude, roughness, stiffness, initial state or resolution actually causes failure.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m17).

<a id="mechanisms-m18"></a>

### M18 — Fresh independent fields and hierarchical inference

**Priority:** P0. **Targets:** D11, D12, D13, D25, D27, D30, D39, D40, D47, D48, D52, H04. **Related records:** [P02](#paper-p02), [P06](#paper-p06).

Cluster bootstrap or exact field-level binomial methods under stated assumptions; account for multiple arms/tolerances and report device selection differences.

**Why it could help:** Give uncertainty intervals with a real independent unit and protect confirmation from architecture-selection reuse.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m18).

<a id="mechanisms-m19"></a>

### M19 — PDE and boundary portability ladder

**Priority:** P2. **Targets:** D38. **Related records:** [P01](#paper-p01), [P03](#paper-p03), [P10](#paper-p10), [P13](#paper-p13), [P14](#paper-p14).

Implement and validate each future PDE branch separately, including source structure, operator/boundary flow and diagnostic cost; no automatic expansion now.

**Why it could help:** Assess usefulness beyond favorable scalar periodic logistic reaction–diffusion.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m19).

<a id="mechanisms-m20"></a>

### M20 — Order-constrained learned embedded stages

**Priority:** P1. **Targets:** D19, D29, D34, D51. **Related records:** [P07](#paper-p07), [P08](#paper-p08).

If retaining fixed RK4 stages, learn a higher-order vanishing residual or extra stages/nullspace rather than free b; account for every nonlinear evaluation.

**Why it could help:** Learn a useful estimator/stage extension without silently sacrificing order or relying on an RK4 name.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m20).

<a id="mechanisms-m21"></a>

### M21 — Physical monotonicity and growth-envelope diagnostics

**Priority:** P1. **Targets:** D05, D20, D29, D51, H03. **Related records:** [P01](#paper-p01), [P05](#paper-p05).

Use cooperative FD assumptions explicitly; convex-monotone semigroup theory does not automatically apply to logistic concave reaction or spectral projections. Add pointwise Jensen/concavity diagnostics for the cooperative FD logistic target; do not transfer a convex-semigroup theorem without checking its assumptions.

**Why it could help:** Expose or constrain unstable learned sensitivity beyond the existing null/interval tests.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m21).

<a id="mechanisms-m22"></a>

### M22 — Effective configuration, provenance and reproducible measurement

**Priority:** P0. **Targets:** D43, D44, D45, D46, D52, H05, H09, H10. **Related records:** [P09](#paper-p09), [P14](#paper-p14).

Retain historical regression controls and split-archive verification; independently validate paper dates/proofs where currently unavailable; do not alter Tower.

**Why it could help:** Remove human-label ambiguity, preserve repaired workflows and make cost/scientific attribution reproducible.

**Status:** research/reproducibility infrastructure, not an architectural novelty. [Mathematical requirements](#math-m22).

<a id="mechanisms-m23"></a>

### M23 — Dedicated cubic Volterra remainder

**Priority:** P2. **Targets:** D17, D18, D21, D34, H01. **Related records:** [P01](#paper-p01), [P07](#paper-p07).

Retain causal phase products and physical limits; add only after an amplitude-series diagnostic isolates a cubic bottleneck.

**Why it could help:** Address finite-amplitude failures remaining after the quadratic physical kernel is accurate.

**Status:** proposed integration of established mathematics. [Mathematical requirements](#math-m23).



## Gaps and complete work inventory

**51 active items; 11 addressed items retained as regression controls.** Every item has inspected evidence, required work, mathematical significance and a mechanism crosswalk. Evidence pointers use `path:/JSON/pointer`; native archive evidence is distinguished from committed cloud replay evidence. P0/P1/P2 rank current urgency, not likelihood of success.

### Active shortcomings and required work

| ID / area | Status / priority | Shortcoming | Work required | Mathematical significance | Mechanisms | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| D01 — research objective | active / P0 | No trained candidate passes the complete joint performance gate | Predeclare the neural comparison goal and the best-classical utility goal separately; do not infer one from the other. | Accuracy, coverage, robustness, and full cost must hold for the same frozen trained arm rather than selected subgates. | [M00](#mechanisms-m00), [M07](#mechanisms-m07), [M16](#mechanisms-m16) | `results/fedora-agenda-review.json:/scientific_outcome` |
| D02 — headroom | active / P0 | The current primary target is often too easy for bare diffusion-first Strang | Screen frozen cases where the cheapest accurate base truly fails at the same horizon before training; quantify added coverage against lost coverage. | If e_base <= tau and c_base < c_model, a correction cannot improve utility without improving a different predeclared endpoint. | [M00](#mechanisms-m00), [M07](#mechanisms-m07) | `results/fedora-agenda-review.json:/confirmation/bare_diffusion_first_matched_frontier_comparisons`; `results/carc-compact-spatial-review.json` |
| D03 — classical comparator | active / P0 | The agenda joint gate compares ETDRK4 and GL3, while the cheaper DF base dominates primary cases | Benchmark the best matched classical frontier, including RF/DF Strang and prepared ETDRK4/GL3, in every performance claim. | The minimum accurate cost is min_method,min_schedule {cost : error <= tau}; beating expensive controls alone is insufficient. | [M00](#mechanisms-m00), [M16](#mechanisms-m16) | `results/fedora-agenda-review.json`; `docs/AGENDA_REVIEW.md` |
| D04 — spatial target | active / P0 | Temporal accuracy and continuum spatial accuracy diverge | Create a consistent continuum-native generator/control or an independent spatial error budget; keep FD and continuum endpoints separate. | &#124;&#124;u_hat - U&#124;&#124; <= &#124;&#124;u_hat - u_H&#124;&#124; + &#124;&#124;u_H - U&#124;&#124;; a same-grid temporal estimator observes the first term only. | [M01](#mechanisms-m01), [M05](#mechanisms-m05) | `results/fedora-agenda-review.json:/acceptance_policies/tracks/continuum` |
| D05 — finite-step accuracy | active / P0 | Source/FNO has finite-step reverse-order catastrophes despite exact physical null limits | Add stable stiffness and elapsed-time scaling; train and validate the finite-step residual and worst step orders. Retain physical null checks with arbitrary weights. | Source maximum error 0.04098 and FNO 0.04525 contrast with the paired base error 3.53e-5. O(h^3) behavior at a fixed grid is not uniform in h*lambda_max. | [M02](#mechanisms-m02), [M06](#mechanisms-m06), [M07](#mechanisms-m07), [M10](#mechanisms-m10), [M21](#mechanisms-m21) | `results/fedora-agenda-review.json:/confirmation/worst_cases` |
| D06 — training design | active / P0 | RF-optimized learning rate and batch size were applied to downstream DF models | Tune the actual chosen DF orientation with a paired, short trial budget and fresh development fields. | The correction target changes with splitting orientation; optimizer choices need not transfer. | [M11](#mechanisms-m11) | `results/fedora-agenda-review.json:/training/interpretation` |
| D07 — training objective | active / P0 | Aggregate normalized one/two-step validation loss improves while strict or per-regime coverage can regress | Use a predeclared residual-aligned mean/spatial/peak/per-regime objective with an explicit base-harm penalty and validation Pareto constraints. | One-step MSE + 0.5 * two-step MSE is not maximum-cell error or joint endpoint coverage; its weighting can reward a harmful tradeoff. | [M11](#mechanisms-m11) | `results/fedora-agenda-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json` |
| D08 — worst schedules | active / P0 | The post-hoc feasible frontier hides failed prescribed schedules | Report success on any schedule and on every schedule, worst reversals, and intermediate errors independently of the work–precision optimum. | Source passes 481/576 cases on every schedule versus 574/576 on some schedule; rank-2 passes 182/192 on every schedule versus 192/192 on some schedule. | [M06](#mechanisms-m06), [M10](#mechanisms-m10) | `results/fedora-agenda-review.json:/confirmation/same_schedule_joint_norm_accuracy_summary` |
| D09 — policy target | active / P0 | A discrete-target policy cannot approve continuum predictions | Add a spatial criterion before continuum acceptance; use a separate continuum calibration bank and a consistent generator. | 306/410 accepted continuum predictions are resolved failures. Zero temporal defect does not bound spatial model bias. | [M01](#mechanisms-m01), [M08](#mechanisms-m08), [M09](#mechanisms-m09) | `results/fedora-agenda-review.json:/acceptance_policies/tracks` |
| D10 — policy cost | active / P0 | Median apparent speed reverses when total and tail costs are included | Time each standalone deployed policy end to end, including rejected attempts, calibration/setup amortization, fallback, percentiles, and totals. | Source with step doubling costs 6.168 seconds versus ETDRK4 4.415 seconds, a ratio of 1.397 despite the favorable median ratio 0.851. E[cost] differs from the median cost ratio. | [M07](#mechanisms-m07), [M08](#mechanisms-m08), [M15](#mechanisms-m15) | `results/fedora-agenda-review.json:/acceptance_policies` |
| D11 — independence | active / P0 | The full agenda has eight controlled field variants but one phase-seed cluster | Generate independently sampled, disjoint continuous fields and phase clusters, with variants nested within each field; use paired seed/grid analysis. | 32 physics cases multiplied by grids, seeds, and targets remain correlated observations rather than an independent and identically distributed sample. | [M09](#mechanisms-m09), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json:/confirmation/paired_design` |
| D12 — cohort status | active / P0 | All inspected diagnostic and full banks are now development evidence | Freeze new field/phase/parent IDs, checkpoints, and criteria before fresh confirmation; reuse the old bank as an explicit regression cohort. | Repeating the protocol replicates an inspected cohort; a new run ID cannot restore untouched status. | [M09](#mechanisms-m09), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json`; `docs/AGENDA_REVIEW.md`; `.runtime/review-consistency-20261007/accuracy-review.json`; `.runtime/review-fedora-premix-20261007/neural-review.json` |
| D13 — rank replication | active / P1 | The best rank-2 kernel has only one training seed | Compare rank-1/rank-2 and a zero-weight baseline across at least three paired seeds before increasing rank. | A nominal rank-2 kernel with 120 parameters can win validation through inductive bias or optimization rather than universal rank capacity. | [M03](#mechanisms-m03), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json:/training` |
| D14 — rank theorem | active / P1 | Constructive rank-4/rank-8 accuracy does not imply learned nominal-rank accuracy | Measure the actual learned kernel response, expanded terms, and operator error across phase, eigenvalue, and coefficient ranges. | Offline quadrature has R+2 terms versus up to 3R commutator terms in the learned branch. Offline maximum errors are 10.84% at rank-2, 1.917% at rank-4, and 0.003% at rank-8. | [M03](#mechanisms-m03), [M15](#mechanisms-m15) | `results/fedora-agenda-review.json:/references_and_structural`; `results/carc-mechanism-audit-review.json` |
| D15 — cost matching | active / P1 | Parameter matching does not match compute, depth, or the most useful comparator | Keep the one-layer, equal-width source-constrained FNO as a frozen confirmation comparator alongside the deep parameter-matched FNO; report transformed fields, latency, and parameters. | Source uses 2 correction + 4 base transforms; rank-2 uses 9 + 4; the matched four-layer FNO uses 8 + 4. Parameter counts alone do not count work. | [M03](#mechanisms-m03), [M15](#mechanisms-m15), [M16](#mechanisms-m16) | `results/fedora-agenda-review.json:/training/interpretation`; `.runtime/review-consistency-20261007/reporting-review.json` |
| D16 — compression inference | active / P1 | The early-compression arm retains a full physical-source and base bypass | Cross nonlinear source construction before/after compression with identical bypasses and a true bottleneck control; state exactly which information is removed. | A full-state source bypass can transport discarded encoder information; the current ablation does not isolate complete compression. | [M04](#mechanisms-m04) | `results/fedora-agenda-review.json:/training/interpretation` |
| D17 — higher amplitude | active / P1 | The quadratic defect is accurate perturbatively but misses finite-amplitude h^3 terms | Evaluate amplitude sweeps and separately add cubic or higher interaction features with correct limits; avoid forcing the remainder to start at h^4. | GL3 residual amplitude order is approximately 3 and late time order approximately 2, consistent with cubic-amplitude terms entering local order h^3. | [M02](#mechanisms-m02), [M04](#mechanisms-m04), [M12](#mechanisms-m12), [M23](#mechanisms-m23) | `results/carc-work-precision-review.json`; `results/carc-compact-spatial-review.json`; `results/carc-mechanism-audit-review.json` |
| D18 — mean closure | active / P1 | The current closure has weak improvement and selects initialization in full confirmation | Pause universal closure promotion; independently audit exact instantaneous moment terms, evolved moments/skewness, and mean/spatial error. | m_prime = r*(m - m^2 - V); V_prime = 2<v,Lv> + 2r*(1 - 2m)*V - 2r*<v^3>. Freezing initial moments does not give the future moment law. | [M12](#mechanisms-m12), [M23](#mechanisms-m23) | `results/fedora-agenda-review.json:/training/interpretation` |
| D19 — time decoder | active / P1 | The nonlinear time decoder benefit is not robust across tuning or confirmation | Pair linear/nonlinear time maps on the actual DF base with identical features/trials; test finite off-grid h and zero/small h. | The source_time full-confirmation slot selects the DF initialization, so it cannot establish a trained temporal benefit. | [M02](#mechanisms-m02), [M10](#mechanisms-m10), [M20](#mechanisms-m20) | `results/fedora-agenda-review.json:/training/interpretation` |
| D20 — spatial regression | active / P1 | Learned corrections damage same-step physical-base spatial accuracy | Predeclare the centered spatial harm rate, peak error, and an absolute floor; use a safe correction constraint or reversion policy based on a deployable residual. | Rank-2 still fails 8–9 spatial-regression comparisons. Pointwise bounds alone do not imply accuracy. | [M03](#mechanisms-m03), [M06](#mechanisms-m06), [M07](#mechanisms-m07), [M11](#mechanisms-m11), [M21](#mechanisms-m21) | `results/fedora-agenda-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json` |
| D21 — stiffness | active / P1 | Input-pair and output stiffness conditioning is incompletely tested at scale | Cross h*lambda_p, h*lambda_q, h*lambda_output and h*r independently (lambda already includes kappa); include fixed-physical-mode and fixed-grid-fraction refinement sweeps. | Pairs with the same output frequency have different input decay; a scalar output-frequency phi basis cannot represent all responses. | [M02](#mechanisms-m02), [M03](#mechanisms-m03), [M06](#mechanisms-m06), [M13](#mechanisms-m13), [M23](#mechanisms-m23) | `results/carc-mechanism-audit-review.json`; `results/fedora-agenda-review.json` |
| D22 — strong diffusion | active / P1 | Classical orientation can obscure the learned mechanism advantage | Retain both RF and DF orientations plus the strong-diffusion homogenized limit; compare corrections relative to the appropriate base. | The 23.684-fold validation gain comes from the classical DF orientation rather than a new neural feature. | [M00](#mechanisms-m00), [M02](#mechanisms-m02) | `results/fedora-agenda-review.json:/training/interpretation` |
| D23 — mean target | active / P1 | A learned mean head can redistribute spatial shape without reducing true mean error | Score signed mean, absolute mean, centered spatial error, and RMS separately; avoid imposing mass conservation. | Logistic reaction changes the mean. Lower moment-head validation loss need not lower physical mean endpoint error. | [M12](#mechanisms-m12) | `.runtime/review-consistency-20261007/accuracy-review.json`; `.runtime/review-consistency-20261007/reporting-review.json` |
| D24 — aliasing | active / P1 | Same-grid cyclic products and continuum dealiased products solve different targets | Keep nodal FD and dealiased spectral controls; perform padded-product, common-band, and total-energy diagnostics with a nonlinear bounded-output check. | Low output frequencies generated by high input pairs are physical; alias folding is a distinct effect. Filtering can reintroduce overshoots. | [M04](#mechanisms-m04), [M05](#mechanisms-m05), [M13](#mechanisms-m13) | `results/carc-compact-spatial-review.json`; `results/carc-mechanism-audit-review.json`; `results/fedora-agenda-review.json` |
| D25 — coefficient generalization | active / P1 | The full learned training physics range is sparse | Add independently held-out coefficient cells and interpolation/extrapolation tests with the same field pairing and fixed budgets. | Crossing kappa 0.001/0.03 with reaction rate 0.5/8 does not establish continuous-parameter operator generalization. | [M13](#mechanisms-m13), [M14](#mechanisms-m14), [M17](#mechanisms-m17), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json:/confirmation/paired_design` |
| D26 — reference scope | active / P1 | Refinement uncertainty is not a rigorous certificate | Distinguish numerical estimates from certificates; maintain independent time/space refinement and add residual norm bounds only where their assumptions are proved. | Summed temporal/spatial estimates describe observed refinement rather than certifying arbitrary continuum truth. | [M01](#mechanisms-m01), [M05](#mechanisms-m05), [M08](#mechanisms-m08), [M09](#mechanisms-m09) | `results/carc-work-precision-review.json`; `results/carc-compact-spatial-review.json`; `results/fedora-agenda-review.json` |
| D27 — continuum coverage | active / P1 | The continuum confirmation subset has only eight fields under physics_index 0 | Expand a separately bounded continuum physics cross after aligning the spatial target; retain the current limited scope explicitly. | Spatial bias evidence under one physics choice cannot automatically cover other diffusion/reaction combinations. | [M01](#mechanisms-m01), [M09](#mechanisms-m09), [M17](#mechanisms-m17), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json:/confirmation/paired_design` |
| D28 — resolution transfer | active / P1 | Earlier held-out grid tests used different fields; agenda pairing improves this, but tested sizes remain small | Continue using the same continuous field at grids 32/64/128 with new independent phase clusters; add spectral-band and spatial-bias breakdowns. | Independent 64-square field performance confounds resolution changes with field changes. | [M01](#mechanisms-m01), [M05](#mechanisms-m05), [M13](#mechanisms-m13), [M14](#mechanisms-m14), [M17](#mechanisms-m17) | `.runtime/review-fedora-premix-20261007/neural-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json`; `results/fedora-agenda-review.json` |
| D29 — long horizon | active / P1 | Endpoint success gives limited evidence about intermediate transients and long rollouts | Audit dense intermediate horizons, longer matched physical times, and perturbation growth separately from endpoint feasibility. | Fast reaction can erase earlier error; an accurate final state can conceal an inaccurate trajectory. | [M02](#mechanisms-m02), [M06](#mechanisms-m06), [M10](#mechanisms-m10), [M17](#mechanisms-m17), [M20](#mechanisms-m20), [M21](#mechanisms-m21) | `results/carc-gpu-benchmark-review.json`; `results/carc-neural-replication-review.json`; `results/fedora-agenda-review.json` |
| D30 — policy reliability | active / P1 | Zero discrete false accepts on a paired cohort does not establish a population bound | Use independent accepted trajectory units, stratified regimes, and a fixed test target; compute an uncertainty bound without treating repeated rows as independent. | With zero errors in n independent and identically distributed Bernoulli accepted units under a fixed frozen policy, the one-sided 95% binomial upper bound is 1 - 0.05^(1/n). Requiring this bound to be at most 1% needs n >= 299. | [M07](#mechanisms-m07), [M09](#mechanisms-m09), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json:/acceptance_policies/interpretation` |
| D31 — policy estimator | active / P1 | Autodiff flow-defect/JVP and combined policies cost too much | Measure each estimator’s standalone cost and calibrated failure rate; use the cheapest stage first and escalate only when needed. | A small composition or step-doubling difference can miss wrong-generator bias. The actual PDE residual checks this bias but adds JVP cost. | [M08](#mechanisms-m08), [M15](#mechanisms-m15) | `results/fedora-agenda-review.json:/acceptance_policies/interpretation` |
| D32 — policy amortization | active / P1 | Policy costs are shared-component attributions rather than independent deployment timings | Create an independent-process benchmark for each frozen policy; report construction/loading, setup, warmup, and training amortization separately. | Training break-even Q = C_train / (C_baseline - C_learned) exists only with positive per-query savings. The bare DF base is often cheaper. | [M08](#mechanisms-m08), [M15](#mechanisms-m15) | `results/fedora-agenda-review.json:/acceptance_policies`; `.runtime/review-consistency-20261007/reporting-review.json` |
| D33 — policy fallback control | active / P1 | Fixed eightfold-refined classical policies are not the fastest accuracy-matched controllers | Compare deployable step control and accuracy-matched classical choices; retain fixed-refinement diagnostics explicitly. | A slow fallback control cannot justify a speed claim against the best classical solver. | [M07](#mechanisms-m07), [M08](#mechanisms-m08) | `results/fedora-agenda-review.json:/acceptance_policies/interpretation` |
| D34 — numerical order | active / P1 | The quadratic GL3 correction does not provide uniform fourth-order time integration | Test local/global order at a fixed grid and along a stiff mesh sequence; target missing commutator or higher-amplitude terms, retaining ETDRK4 as a control. | Observed GL3 late orders are 1.88/1.93 versus approximately 4 for ETDRK4. Quadrature order 7 does not set the order of the complete method. | [M02](#mechanisms-m02), [M03](#mechanisms-m03), [M10](#mechanisms-m10), [M20](#mechanisms-m20), [M23](#mechanisms-m23) | `results/carc-work-precision-review.json`; `results/carc-compact-spatial-review.json` |
| D35 — GPU scaling | active / P2 | Native learned GPU workloads are narrow: batch size 1, grids up to 128-square, and no isolated per-model VRAM peaks | Profile bounded larger grids and batches with accuracy headroom, isolated model memory, and synchronized standalone costs; never infer VRAM from 128 GB host RAM. | Low parameter counts do not bound FFT activations, physical-base cost, or available display-GPU memory. | [M03](#mechanisms-m03), [M13](#mechanisms-m13), [M15](#mechanisms-m15) | `.runtime/review-consistency-20261007/reporting-review.json`; `results/fedora-agenda-review.json` |
| D36 — fusion scaling | active / P2 | Fusion dispatch gains reverse in large groups; chunking benefits depend on the workload | Retain original, fused, and chunked variants; profile memory/cache behavior and choose chunking from measured development workloads. | FFT dispatches decrease from 18 to 7, but transformed fields decrease only from 18 to 17. Fewer launches do not imply a proportional arithmetic reduction. | [M15](#mechanisms-m15) | `results/carc-compact-spatial-review.json`; `.runtime/review-fedora-premix-20261007/numerical-review.json` |
| D37 — selected-output algorithm | active / P2 | Preserving low outputs from the full input does not automatically save work | Replace O(N*retained_outputs) direct pair enumeration with separable or structured FFT features and test complete solver costs. | Bounded native direct selected convolution costs 2.03 times the full FFT output method in 1D and 10.06 times in 2D. Output projection still computes full GL3. | [M03](#mechanisms-m03), [M05](#mechanisms-m05), [M13](#mechanisms-m13), [M15](#mechanisms-m15) | `.runtime/review-fedora-premix-20261007/neural-review.json`; `.runtime/review-fedora-premix-20261007/numerical-review.json` |
| D38 — benchmark coverage | active / P2 | Evidence covers only scalar periodic logistic RD, without other PDE/domain/boundary tests | After RD gates pass, add one justified separate benchmark with a teacher and physical constraints, such as variable-coefficient/anisotropic RD or nonperiodic boundaries. Do not infer an advantage over all neural solvers. | Fourier translation structure and exact subflows do not automatically transfer to other geometries, flux laws, incompressible systems, or multiple species. | [M14](#mechanisms-m14), [M17](#mechanisms-m17), [M19](#mechanisms-m19) | `AGENTS.md`; `results/fedora-agenda-review.json`; `.runtime/review-fedora-premix-20261007/neural-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json` |
| D39 — factor disentanglement | active / P2 | Earlier stress banks couple amplitude, spectrum, grid, and horizon factors | Use controlled one-factor variants, independent field clusters, and a factorial subset; separate variance-matched rough fields from amplitude shifts. | Observed rough-field failures can combine amplitude distribution shift and high-frequency effects; covariance does not establish a mechanism. | [M11](#mechanisms-m11), [M17](#mechanisms-m17), [M18](#mechanisms-m18) | `.runtime/review-fedora-premix-20261007/neural-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json`; `results/carc-compact-spatial-review.json` |
| D40 — timing uncertainty | active / P2 | Small repeated timing sets and one setup sample do not establish confidence across runs | Interleave or randomize repeats across methods; use several independent processes/sessions with a bounded repetition budget and preserve all raw samples. | Overlapping or disjoint observed ranges are descriptive rather than confidence intervals; setup variance is unknown. | [M15](#mechanisms-m15), [M18](#mechanisms-m18) | `results/carc-gpu-benchmark-review.json`; `results/carc-work-precision-review.json`; `results/carc-compact-spatial-review.json`; `.runtime/review-consistency-20261007/reporting-review.json` |
| D41 — best baseline fairness | active / P2 | The current FNO adapters are not published-paper reproductions; convergence and search budgets are unmatched | Define the exact claim against the project’s FNO hybrid. For a published-paper claim, reproduce its public benchmark, tasks, training, and normalization with a fixed matched search budget. | 96/300 updates cannot establish a converged neural ceiling. Equal update counts do not imply equal FLOPs or optimization opportunity. | [M16](#mechanisms-m16) | `results/carc-gpu-benchmark-review.json`; `results/carc-neural-replication-review.json`; `results/fedora-agenda-review.json`; `.runtime/review-consistency-20261007/reporting-review.json` |
| D42 — training efficiency | active / P2 | Current learning curves and early checkpoint selection do not support indiscriminately increasing training | Keep bounded architecture ablations; inspect gradients, residuals, learning-curve saturation, sampling, and exposures before increasing the budget. | 69 trials completed with no obvious gradient failure and rare clipping. Additional updates alone do not repair missing representation or an inconsistent target. | [M11](#mechanisms-m11) | `results/fedora-agenda-review.json:/training/interpretation` |
| D43 — effective labels | active / P2 | Trial IDs retain RF/b1 labels despite effective DF/b8 configuration | Name future trials from effective frozen metadata; display the ID alongside actual orientation, learning rate, and batch size. Preserve original hashes. | The labeling issue does not alter checkpoints, but it can mislead mechanism attribution. | [M22](#mechanisms-m22) | `results/fedora-agenda-review.json:/training/interpretation` |
| D44 — resource accounting | active / P2 | Reviewed archives lack independent final sacct accounting | Export job/step state, wall time, CPU, MaxRSS, and TRES accounting when available; distinguish resource requests from measurements. | Recorded 2009 seconds of science work is not billed GPU wall time; sampled physical memory is not a per-kernel peak. | [M22](#mechanisms-m22) | `results/fedora-agenda-review.json:/independent_scheduler_accounting`; `results/carc-neural-replication-review.json`; `results/carc-work-precision-review.json`; `results/carc-compact-spatial-review.json` |
| D46 — literature scope | active / P2 | Earlier agenda literature links were unverified; this primary-source review only partially closes that gap | This dated primary-source review addresses accessible papers within 2025-10-07 through 2026-10-07. Verify remaining inaccessible full text and novelty claims; distinguish publication, preprint, version date, and project inference. | A new combination is not automatically a new mathematical mechanism. Novelty requires checking the exact assumptions and claims against accessible primary sources, while unresolved access limitations remain explicit. | [M22](#mechanisms-m22) | `docs/AGENDA_LITERATURE.md` |
| D47 — statistics | active / P2 | Pooled frontier, cell, and test counts overstate evidence if treated as independent samples | Separate fields, coefficient cases, training seeds, schedules, grids, norms, and repeated tests; use paired or clustered summaries. | 6 × 152 GPU test executions represent 152 distinct tests; 51957 rows do not represent 51957 experiments. | [M09](#mechanisms-m09), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json`; `results/carc-neural-replication-review.json`; `results/carc-mechanism-audit-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json` |
| D48 — initialization attribution | active / P2 | An initialization-selected arm is still sometimes the strongest outcome | Retain zero-weight and bare-base controls at all stages; require a selected checkpoint with positive updates and changed parameters for a learning claim. | The agenda has 5/69 initialization selections and consistency has 7/21. Initialization passes do not demonstrate the benefit of a trainable feature. | [M11](#mechanisms-m11), [M16](#mechanisms-m16), [M18](#mechanisms-m18) | `results/fedora-agenda-review.json`; `.runtime/review-fedora-premix-20261007/neural-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json`; `results/carc-neural-replication-review.json` |
| D49 — policy comparison | active / P1 | Policy comparisons omit the strongest rank-2 kernel and the matched FNO | Apply identical frozen calibration/accept/reject/fallback methodology to rank-1/rank-2, Source, and the cheap FNO, with matched controls and paired seeds. | Current policies cover Source and initialization-selected source_time/closure arms rather than a learned kernel/FNO controller comparison. | [M07](#mechanisms-m07), [M08](#mechanisms-m08), [M09](#mechanisms-m09) | `results/fedora-agenda-review.json:/acceptance_policies/calibration` |
| D50 — calibration shift | active / P1 | The maximum calibration-ratio envelope with safety factor 2 is empirical and nonuniform | Test independent phase, coefficient, and variance shifts; quantify estimator bias and coverage by regime, and justify any probabilistic bound assumptions explicitly. | 16 calibration parents and step-doubling ratio multipliers 11.2–86.7 show that estimator scale/quality varies sharply by arm. A maximum fitted ratio is not a theorem outside the bank. | [M08](#mechanisms-m08), [M09](#mechanisms-m09) | `results/fedora-agenda-review.json:/acceptance_policies/calibration` |
| D51 — stability proof | active / P1 | Pointwise bounds and exact physical limits do not establish dissipative or stiff stability | Derive a suitable local Lipschitz or one-sided contraction condition for a correction; test perturbation growth, frequency-wise response, and the finite-h stiffness envelope. | For Phi = S + delta, Lip(Phi) <= Lip(S) + Lip(delta). Invariance of [0,1] alone still permits sensitive error amplification. | [M02](#mechanisms-m02), [M06](#mechanisms-m06), [M20](#mechanisms-m20), [M21](#mechanisms-m21) | `results/carc-mechanism-audit-review.json`; `results/fedora-agenda-review.json` |
| D52 — device reproducibility | active / P2 | Native and cloud checkpoint selections can differ near validation thresholds | Retain device-specific selected hashes and losses; use frozen checkpoints for device timing and repeat bounded seed runs without assuming bitwise equality. | Native consistency selects 14 trained checkpoints and 7 initializations versus cloud 15/6. Earlier CNN rollout decisions changed six rows across CARC/cloud without changing other pass decisions. | [M11](#mechanisms-m11), [M18](#mechanisms-m18), [M22](#mechanisms-m22) | `results/carc-neural-replication-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json`; `results/consistency-validation.json` |

### Addressed findings to retain as regression controls

| ID / area | Status / priority | Shortcoming | Work required | Mathematical significance | Mechanisms | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| D45 — report transport | addressed_keep_regression / P2 | The original full archive was truncated; complete parts resolved this review | Keep verified split indexes, part SHA values, and dataset/checkpoint inventories; reject incomplete uploads clearly. | Local hashes detect differences but do not independently authenticate execution or provide digital signatures. | [M22](#mechanisms-m22) | `results/fedora-agenda-review.json:/archive` |
| H01 — clock coordinate | addressed_in_agenda_keep_regression / P1 | A clock shift cannot supply the required cubic state variation near a zero receiver | Retain the historical clock control and structural source/pair-correction tests. | w = O(h) and delta_t = O(h^3) give delta_u = O(h^4); a state correction can reach order h^3. | [M04](#mechanisms-m04), [M23](#mechanisms-m23) | `results/carc-mechanism-audit-review.json`; `results/carc-neural-replication-review.json` |
| H02 — gate placement | addressed_in_agenda_keep_regression / P1 | A pointwise quadratic gate after transport removes a required remote leading direction | Keep a zero-preserving signed physical source before transport; do not apply another postgate to a branch that is already quadratic in spatial amplitude. | All 24 remote bump cases have a zero receiver gate. An extra quadratic gate suppresses an amplitude^2 response to amplitude^4. | [M04](#mechanisms-m04) | `results/carc-mechanism-audit-review.json`; `results/fedora-agenda-review.json` |
| H03 — physical nulls | addressed_in_agenda_keep_regression / P1 | Ungated premix/FNO exhibits learned mean drift in constant or commuting cases | Keep uniform-field tests with arbitrary weights, exact r=0/kappa=0/h=0 limits, and amplitude-order tests. | Source construction before transport and explicit subtractions enforce the known operator nullspace; they do not guarantee finite-h accuracy. | [M12](#mechanisms-m12), [M21](#mechanisms-m21) | `.runtime/review-fedora-premix-20261007/neural-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json`; `results/fedora-agenda-review.json` |
| H04 — unseen steps | addressed_in_agenda_keep_regression / P1 | All consistency diagnostic h queries were used during development | Keep the new agenda’s unequal, unseen h queries and new fresh diagnostic seeds. | Composition consistency at training h does not establish unseen-h generalization; an identity map can satisfy composition while solving the wrong dynamics. | [M10](#mechanisms-m10), [M18](#mechanisms-m18) | `.runtime/review-consistency-20261007/accuracy-review.json`; `results/fedora-agenda-review.json` |
| H05 — reference refinement | addressed_in_agenda_keep_regression / P2 | An earlier reference was excluded by the precision budget despite correct convergence | Allow a separately declared fourth bounded refinement without relaxing precision; retain the valid historical exclusion. | The old RMS uncertainty was 1.14488e-10 against budget 1.02062e-10; the fourth attempt reaches 7.0614e-12 and passes the unchanged threshold. | [M22](#mechanisms-m22) | `results/carc-work-precision-review.json`; `results/carc-compact-spatial-review.json` |
| H06 — FFTfusion | addressed_in_agenda_keep_regression / P2 | Original GL3 coefficient/reduction dispatch imposes unnecessary small-grid cost | Keep fused identity and cost regression checks, including the largest-batch speed reversal. | Fused FP64 parity error is 3e-15 and complete-solver median speedup is 1.587 times. This is an implementation gain rather than an accuracy change. | [M15](#mechanisms-m15) | `results/carc-interaction-screen-review.json`; `results/carc-compact-spatial-review.json` |
| H07 — teacher/timing caching | addressed_in_agenda_keep_regression / P2 | The earlier single uncached ETDRK4 comparison gave uneven preparation opportunities | Retain separate prepared and setup-inclusive frontiers with all raw timing samples. | One-time setup changes winner counts; cached and uncached solver latency cannot be compared fairly without accounting for preparation. | [M15](#mechanisms-m15) | `results/carc-interaction-screen-review.json`; `results/carc-work-precision-review.json`; `results/carc-compact-spatial-review.json` |
| H08 — parameter comparator | addressed_in_agenda_keep_regression / P2 | Earlier premix/FNO differences in depth, parameters, and support confounded backbone attribution | Retain agenda capacity/depth controls; add the cheap FNO to fresh confirmation. | Earlier premix has 23825 parameters versus FNO 67137. The agenda records width, depth, and support explicitly; matching one quantity does not match every cost. | [M16](#mechanisms-m16) | `.runtime/review-fedora-premix-20261007/neural-review.json`; `.runtime/review-consistency-20261007/accuracy-review.json`; `results/fedora-agenda-review.json` |
| H09 — workflow | addressed_keep_regression / P2 | Tower location, flock, partition, signal, and reader-cap failures previously blocked runs | Retain strict source, seal, JUnit, and reporting tests. The current full workflow verifies; these historical failures should not be presented as ongoing failures. | Reporting and training correctness are necessary but insufficient for scientific utility. | [M22](#mechanisms-m22) | `results/premix-startup-fix-validation.json`; `results/fedora-partition-fix-validation.json`; `results/carc-gpu-benchmark-review.json`; `results/fedora-agenda-review.json` |
| H10 — policy failure completeness | addressed_in_agenda_keep_regression / P2 | Earlier policy designs omitted complete rejection/fallback cost or wrong-generator bias | Keep the independent PDE JVP defect, charged rejection/fallback work, and decisions fitted only on frozen calibration. Add standalone cost measurement next. | The mechanism audit’s wrong-generator example has zero composition error but true error 0.1484. Step doubling alone cannot certify model bias. | [M22](#mechanisms-m22) | `results/carc-mechanism-audit-review.json`; `results/fedora-agenda-review.json` |


### Questions that the current record cannot answer

The record cannot establish population success, statistically bounded 1% conditional false acceptance, trained rank-2 replication, benefit against cheap FNO on fresh fields, official FNO-paper reproduction, arbitrary-grid/geometry transfer, universal semigroup/stability guarantees, or a standalone solver-policy speedup. These are explicit tasks in the ledger, rather than positive conclusions inferred from structural/oracle tests.

Recent-paper gap statements below distinguish (a) explicit author limitations, (b) properties of an inspected released implementation, and (c) our proposed opportunity. A code-level observation does not disprove a paper theorem; absence from a README is not evidence of absence from a paper.


## Projected results and decisions

These are proposed directions and next-run decision criteria, **not numerical forecasts or measured gains**. Some successful tests may demonstrate that the classical solver or a spatial upgrade is the correct practical choice. Nothing in this document authorizes automatic model/budget escalation.

| Mechanism | Projected effect | Proposed success evidence | Falsifier / stop condition |
| --- | --- | --- | --- |
| [M00](#mechanisms-m00) | Find regimes where extra correction accuracy can actually replace classical steps; otherwise establish that the classical solver is the right answer. | Before training, identify a prespecified hard stratum on independent development fields where even the cheapest accurate optimized classical core requires additional work. | Reject a neural solver-utility campaign if bare DF dominates the relevant cost–accuracy region; a tighter tolerance chosen after viewing confirmation does not rescue the old gate. |
| [M01](#mechanisms-m01) | Explain and reduce the common continuum floor; expose continuum errors before an otherwise small temporal indicator accepts them. | On new matched fields, verify spatial refinement behavior and require separate RMS/max total budgets below the declared target. | Reject a continuum claim if only the discrete temporal indicator is small, or if refinement order is unresolved; a solver-core change is not a neural gain. |
| [M02](#mechanisms-m02) | Replace stiffness-growing cubic extrapolation with a response that is exact at quadratic amplitude and damped at large diffusion/long times. | Recover amplitude-squared teacher response, fixed-grid small-h order, exact commuting limits and strong-diffusion limit; then reduce prescribed-order failures at charged cost. | Reject if quadrature breaks zero diffusion/reaction limits, if finite-amplitude gains disappear, or if required transforms cost more than the saved steps. |
| [M03](#mechanisms-m03) | Retain phase-aware robustness with fewer transforms than a generic high-rank branch; rank 1 may offer a useful cost point. | Three paired training seeds; compare rank 0/1/2 and analytic two-node kernel against bare DF and the cheap FNO at every prescribed schedule. | Reject if low-rank training cannot recover physical response, if worst-order benefit is seed-specific, or if classical/FNO Pareto dominance persists. |
| [M04](#mechanisms-m04) | Preserve high-high-to-low/zero interactions and remote receiver response that an input-only cutoff or post-transport point gate erases. | Pure before/after-product ablation with identical learned support, physical base and cost; amplitude-squared defect above reference uncertainty. | Reject a bottleneck conclusion if a full-source bypass remains in only one branch; structural span improvement without finite-amplitude utility is insufficient. |
| [M05](#mechanisms-m05) | Remove spurious retained-mode folding and enforce physical scale when resolution changes. | Demonstrate common-field grid transfer with resolved nonlinear products and independent continuum uncertainty; report padding and memory costs. | Reject if higher-harmonic energy is silently discarded, if positivity fails after projection, or if padding costs exceed any useful step savings. |
| [M06](#mechanisms-m06) | Suppress harmful corrections where the base is already accurate or the finite-step response is outside validated support. | Remove large reverse-order damage and enforce a predeclared same-step no-regression envelope outside the numerical floor, without losing hard-case coverage. | Reject if the envelope simply switches all corrections off, if low/mean modes are erased by a global lambda-max scale, or if routing uses held-out truth. |
| [M07](#mechanisms-m07) | Skip unnecessary neural work on easy states; spend correction/refinement work only where the expected accuracy benefit pays for it. | A frozen router and fallback policy must lower total and p95 standalone cost at matched accuracy/coverage against the cheapest operational classical policy. | Reject if route/JVP/reject overhead erases savings, if rare tails dominate totals, or if an oracle truth-based router is used as the deployed method. |
| [M08](#mechanisms-m08) | Replace expensive repeated JVP work or blind step agreement with a target-correct, shared-computation error signal. | Compare cheap embedded/doubling and interior-residual indicators on rank-2, Source and FNO; require accuracy, rejects and full standalone cost to pass together. | Reject a deterministic certificate without quadrature/reconstruction bounds; self-agreement can share the same spatial/model bias. |
| [M09](#mechanisms-m09) | Turn empirical envelopes into explicit distributional statements where assumptions hold; detect spatial/grid shift instead of silently transferring coverage. | Use independent function clusters and a declared joint query score; audit conditional-on-accept risk separately. At 1%, finite conformal quantile needs n>=99; zero-failure 95% test bound needs 299 independent accepted units. | Reject formal coverage claims under reused phases, distribution shift, unbounded spatial score, adaptively selected policies, or inadequate independent sample count. |
| [M10](#mechanisms-m10) | Reduce order sensitivity and long-rollout drift while preserving genuine dynamics at arbitrary unseen individual steps. | Freeze intermediate targets, all selected triple permutations and abrupt step changes; pass independent RMS/max errors, not only composition agreement. | Reject if a semigroup regularizer favors a physically wrong flow or if continuous h claims only reflect integer-frame forecast multiples. |
| [M11](#mechanisms-m11) | Reduce harmful corrections in already-solved cases and focus capacity on true defects rather than aggregate normalized MSE. | Same architecture/data/update budget, actual DF tuning, validation-only weights; promote on per-regime max accuracy, every-schedule robustness and complete cost. | Reject if improvement exists only in the surrogate loss, if tiny defect division amplifies reference noise, or if confirmation is reused for loss selection. |
| [M12](#mechanisms-m12) | Improve a verified mean-dominated regime while repairing the frozen closure binary-field boundary defect. | Separate mean/centered-spatial errors, exact initial rate tests and fields with equal c,V but different phase/spectrum; compare analytic spectral mean and endpoint head. | Reject if closure remains initialization-selected, spatial damage rises, or extra reductions cost more than the mean-error benefit. |
| [M13](#mechanisms-m13) | Represent unresolved spectral bands economically without grid-dependent derivative units or unintended constant leakage. | Fixed physical frequency transfer across N, constant/null tests, support-matched multiband ablation and charged derivative cost. | Reject if learned effective spacing absorbs training-grid units and fails transfer; a useful fractional feature is not a proof of the PDE generator. |
| [M14](#mechanisms-m14) | Extend phase/interaction structure to geometry only when anisotropic or irregular-domain headroom is independently demonstrated. | Check quadrature measures, boundary lifting, remeshing, tensor rotation and interaction truncation against accepted teachers. | Reject a direct Fourier-pair transplant outside periodic constant coefficients; eigenvalues alone do not control nonnormal advection transients. |
| [M15](#mechanisms-m15) | Lower real inference/estimator overhead through identical-map reuse and equitable optimized baselines. | Verify identical predictions within a declared precision tolerance, warmup/repeated matched-order timings, total/tail work, memory and deployment amortization. | Reject if batching favors only one method, if reused filters depend on changed h/operator, or if fusion changes products/output constraints. |
| [M16](#mechanisms-m16) | Determine whether gains belong to the physical solver, interaction bias, implementation or a genuine neural benchmark advantage. | Include cheap one-layer and deep matched FNO in local confirmation; any paper-level claim requires official target/data/config/version reproduction separately. | Reject a paper win inferred from adapted FNO only, omitted cheap controls, unequal tuning/convergence or initialization-selected efficacy. |
| [M17](#mechanisms-m17) | Identify where finite-amplitude, roughness, stiffness, initial state or resolution actually causes failure. | Prespecified one-factor changes plus selected two-factor interactions; include more than the current continuum physics-index-0 subset. | Reject causal claims from banks that change grid, amplitude and horizon together or count shared-field crossings as independent draws. |
| [M18](#mechanisms-m18) | Give uncertainty intervals with a real independent unit and protect confirmation from architecture-selection reuse. | New disjoint phases/coefficients and parent IDs; freeze protocol, candidate, policy and primary endpoint; distinguish exploratory comparisons and selected primary test. | Reject population confidence from one phase cluster, repeated grids/seeds or post-selection slices; do not call a replay a new confirmation. |
| [M19](#mechanisms-m19) | Assess usefulness beyond favorable scalar periodic logistic reaction–diffusion. | After scalar headroom, bounded coupled RD and advection-diffusion/Burgers with converged independent teachers and actual invariants; later geometry/fluid benchmarks. | Reject cross-PDE superiority extrapolated from current scalar experiments; nonperiodic/variable-coefficient physics needs new derivations. |
| [M20](#mechanisms-m20) | Learn a useful estimator/stage extension without silently sacrificing order or relying on an RK4 name. | Check order on exact/smooth problems, stiff response on known linear modes and actual nonlinear target; enforce consistency before arbitrary-h claims. | Reject arbitrary softmax final weights as fourth-order by default; A-stability is also not implied by positive/sum-one weights. |
| [M21](#mechanisms-m21) | Expose or constrain unstable learned sensitivity beyond the existing null/interval tests. | Compare Jacobian-direction diagnostics with physical growth envelopes, perturbation rollouts and hard stiffness states; any certificate needs an actual upper bound. | Reject universal contraction<=1 near u=0, where physical logistic growth is expansive; a sampled JVP is not a global Jacobian certificate. |
| [M22](#mechanisms-m22) | Remove human-label ambiguity, preserve repaired workflows and make cost/scientific attribution reproducible. | Use effective DF/batch-8 fields instead of RF/batch-1 ID labels; archive scheduler records, actual source hashes, reader contracts and dated citation evidence. | Reject replacing native selection counts with cloud counts, relabeling old immutable artifacts, or reporting numerical timers as billed/resource utilization. |
| [M23](#mechanisms-m23) | Address finite-amplitude failures remaining after the quadratic physical kernel is accurate. | Demonstrate an epsilon-cubed residual distinct from quadratic teacher uncertainty and reduced finite-amplitude max errors at affordable cost. | Reject extra amplitude-squared gating of a quadratic branch, which gives epsilon^4; reject added cubic cost without new useful coverage. |

## Robust bounded next program

All stages remain serial on Fedora `local`, under `/home/rahel/TDN`, Python
3.13.13 `.venv`, no system-temp/scratch outputs, no pending cap, no hidden
reruns/extended budgets. CPU ≤8 physical cores and ≤48 GiB per stage; GPU stages four CPUs,
32/48 GiB host RAM and one 4090, soft GPU ≤18 GiB and 75% of initially free/total,
hard physical device-use fraction 90%. The Slurm cap is 110000 MiB, not all
128 GiB physical RAM. Keep 20/30/45-minute allocations and existing science
ceilings; scientific negative results are successful completed stages.

The counts below are design proposals, not implemented protocol or measured
runtime. A stage stops and preserves partial evidence if its frozen cap is
reached. Any scope adjustment belongs to a newly frozen protocol, not a silent
midrun change.

| Stage | Bounded program | Outputs / predeclared decision | Allocation and science ceiling |
| --- | --- | --- | --- |
| D0 Provenance and target audit | Fix effective-configuration labeling in new records; frozen PDE/symbol/product/teacher/precision/estimator declaration; structural null, reality, symmetry, amplitude, alias and bounds tests. | Old records immutable. Exact declared/resolved fingerprints; no inaccurate target labels. | CPU8,48GiB,20min; ≤600 science seconds. |
| D1 Headroom and spatial screen | 12 independent random phase/coefficient development fields, six fixed shape strata, four κ/r combinations; primary grids32/64 with bounded128 subset; classical RF/DF/ETD/GL3, schedules forward/reverse/equal/long. Full all-domain table and separately defined headroom stratum. | Error/cost Pareto map, same-step mean/spatial and two-resolution errors. Freeze any hard regime from this development bank only. | CPU8,48GiB,45min; ≤1800 science seconds. Teacher acceptance prerequisite. |
| D2 Pure mathematical ablations | Rank0/1/2 kernel approximations; full-source versus projected-input versus product-before-projection; nodal/dealiased product; stiffness-envelope basis without neural fit. Phase/opposite-phase, high-high-to-low and remote source cases. | Correctness/representation uncertainty and complete primitive operation cost. Stop unsupported branch. | CPU8,48GiB,20min; ≤600 science seconds. |
| D3 Tune the actual DF core | Four learned families (Source, rank1, rank2, cheap source-FNO), two LRs × two physical batches, 120 updates, one paired seed:16 trials. Record exposures, histories, best initialization and gradient/correction scales. | Freeze same shared-tuning comparison; if arm-specific tuning used, give every arm equal budget and declare separately. | GPU1,CPU4,32GiB,45min; ≤1800 science seconds. |
| D4 Mechanism singles and sparse combinations | Baseline Source/rank1/rank2/cheapFNO ×3 seeds=12 final trials at300 updates. Add only mechanism supported by D2: envelope and residual-loss single ablations on Source and rank2, then one combined arm only if each single helps. Cap total final learned trials at24. | Matched data/updates/modes/core; no width/rank expansion. Record failure/init; do not silently increase the cap to finish all potential combinations. | GPU1,CPU4,32GiB,45min; ≤1800 science seconds. |
| D5 Frozen independent confirmation | At least24 new independent phase/coefficient field clusters, balanced across six shape strata, crossed physics; paired32/64 and bounded128; grids512/1024 only in later throughput-only branch. Three training seeds remain crossed, not counted as fields. New phases, IDs and normalization fixed before loading. | All schedules retained; primary both-norm τ=2e-4 plus2e-5/2e-6 secondary; all-domain coverage/worst-regime/nonregression and optimistic frontier separate from deployed policy. Cluster intervals descriptive at this size. | CPU reference prep8,48GiB,45min/1800s separately; GPU confirm4,48GiB,45min/1800s. Reduce bounded continuum subset beforehand if teacher cap requires it. |
| D6 Standalone acceptance/cost test | Only best selected learned arm, cheap FNO and bare/operational classical controls; separate field-level calibration. Begin step doubling; add embedded pair only after algebraic parity. Spatial audit separately for continuum. Randomized/block-interleaved timings with10 repeats on a fixed small representative subset; enough setup repeats to display range. | Total+median+p95/worst cost, rejected work and fallback, independent accepted accuracy. Reject if total cost above fastest eligible classical deployment or unacceptable worst-regime failures. | GPU1,CPU4,32GiB,30min; ≤1200 science seconds. Narrow scope instead of expanding expensive JVP trial matrix. |
| D7 Portability and scaling after D5/D6 | Coupled RD then advection-diffusion/Burgers, independent teacher validation first. Throughput sweep grids128/256/512/1024 and batches1/4/16 under VRAM cap; accuracy on agreed physical fields, not an easy smooth bank only. | Distinguish throughput-only, resolution transfer and new-PDE claims. Native official FNO benchmark reproduced as separate track. | Newly frozen20–45min task allocations; no automatic promotion or20h jobs. |

The D1 12-field screen and D5 24-field confirmation are bounded diagnostic
choices, not powered population-success studies. If a statistically strong
≤1% false-acceptance statement is a research goal, freeze a separate
field-independent audit with an explicit sample-size/power calculation; do
not overstate these smaller studies.

## A factorial strategy that avoids an unmanageable combination matrix

1. Keep core, source representation, temporal envelope, spatial operator,
   loss and estimator as separate factors. A changed core or teacher is not a
   pure neural architecture ablation.
2. D0–D2 test physical/basis mechanisms without training. Use these to eliminate
   mechanisms that fail correctness or have no representational headroom.
3. The primary D4 contrast is **compact rank1/2 versus Source versus cheapFNO**
   with the same DF core. Test time envelope and residual loss as two single
   additions on the one or two supported backbones. Include a single combined
   envelope+loss arm only after the singles have independent development
   support. Do not run all subsets of every proposed upgrade.
4. To quantify combination interaction, on a paired development field i use
   a prespecified response Y, e.g. log(error+floor) or log(cost), and

       I_i = Y_AB,i − Y_A,i − Y_B,i + Y_0,i.

   I<0 denotes additional benefit on a lower-is-better log response. This is
   not automatically statistical synergy; uncertainty is at independent field
   cluster level and success still needs complete deployed cost and accuracy.
5. For the spatial operator/dealiasing contrast, every learned AND classical
   arm receives the same operator/product, with a new target-specific teacher.
   Report the classical-core improvement first, then the incremental learned
   improvement. A spatial upgrade may be the correct practical answer even
   if there is no neural gain.
6. Moment closure remains an independent optional subexperiment after a
   verified mean-dominated failure; do not combine it by default with a kernel
   whose mean component already exists. Avoid double counting mean response.

## Statistical and benchmark reporting plan

The independent experimental unit is the new random continuous field draw
(phase/coefficient seed), not a grid, physics setting, schedule, tolerance,
norm, estimator or optimization seed. Use the same fields/schedules within
each arm; cross training seed with field to separate optimization and field
variation. Controlled shape variants from one random draw form one cluster.

Primary endpoint is prespecified both-norm tolerance at all required queried
schedules, compared to fastest eligible frozen operational classical control
and cheap FNO. The post-hoc fastest feasible frontier is an oracle descriptive
upper-bound comparator, not the deployed operational control. Report all
infeasible cases without dropping them from coverage. Report speed only on
predeclared jointly feasible comparisons and show coverage separately.

For each independent field cluster, aggregate within-cluster results by the
declared estimand: mean paired log speed ratio, worst required schedule error,
joint-success fraction or all-schedule success. Resample whole field clusters,
keeping all paired arms and repeated observations together. To represent
training randomness, use a crossed/hierarchical summary over the three seeds,
or explicitly show per-seed intervals; three seeds alone give weak variance
estimation. With one cluster, the historical campaign has no meaningful field
bootstrap. No existing row-level confidence interval should be manufactured.

Publish paired total-cost ratio sum_i C_ours,i / sum_i C_control,i as well as
median_i(C_ours,i/C_control,i), p95 and maximum; they answer different questions.
Retain cold/setup amortization

    C_total(M)=C_setup+M C_warm,

and give crossover M only when setup and warm cost are measured repeatedly.
Charge model/teacher training separately for amortization claims. Timing
repeats are measurement replicates, not independent PDE samples. FP32/TF32-off
model execution versus FP64 teachers is intentional, but both learned and
classical compared operational methods use the same declared device/precision
unless a separate precision-Pareto benchmark is explicitly stated.

Performance gates must be frozen before next confirmation. An illustrative
operational gate, consistent with the current project's scale, is:

- both RMS and maximum upper errors ≤2e-4 on ≥90% overall and ≥75% each
  predeclared regime, plus a separate stringent all-required-schedule rate;
- no more than1.1× same-step spatial error when base error is resolved, with
  absolute floor1e-6 explicitly retained; harmful>10× regressions always listed;
- primary chosen model selected as a trained checkpoint in all three paired
  seeds; initialization remains a separate physical control;
- ≥1.1× median **and** favorable total standalone deployment cost against the
  cheapest eligible classical policy and cheapFNO, without reducing coverage;
- observed false acceptance≤1% in declared target/each required domain,
  empirical at this sample size; continuum passes require spatial estimation;
- no claim of statistical significance or a population bound unless an
  independently justified cluster/sample-size analysis supports it.

The all-schedule and total-cost clauses are stronger new criteria and therefore
must be preregistered for the next run; they must not be retroactively claimed
as original gates for the already inspected native archive.

## Priority and stop rules

**First:** provenance label fix, target separation, training-free headroom,
cheapFNO inclusion and genuinely independent fields. These enable an honest
decision and may show that classical spatial/core work is the best route.

**Second, conditional on headroom:** rank1/rank2 phase-aware residual,
finite-stiffness damping and residual-aligned loss with actual DF tuning.
Worst-order failures make this a more direct research question than simply
increasing model width, layers, rank or updates.

**Third, conditional on accuracy:** standalone estimator/reject/fallback
optimization with consistent spatial target; this is a solver mechanism and
must be measured operationally. Cost sharing or a zero-failure empirical cohort
is insufficient.

**Later:** moment closure, multiscale/new-PDE extensions and official large
FNO benchmarks. They can be scientifically useful but are presently less
supported by the audited advantage. Stop a mechanism if it cannot beat the
cheapest accurate physical core on the declared fresh domain; report the
negative result rather than escalating its training budget.



## Mathematical review

All diffusion eigenvalues below use the **nonpositive** convention. Discrete nodal, dealiased Galerkin and continuum targets are distinct. The derivations state their validity domain; small-amplitude and fixed-operator asymptotics do not establish uniform stiff, boundary, geometry or large-amplitude behavior.

## 2. Operator, bracket, and target conventions

The current problem is periodic scalar logistic reaction–diffusion,

\[
\dot u=A(u)+B(u),\qquad A(u)=L_Nu,\quad B(u)=r(u-u^2),\quad L_N=\kappa\Delta_N,
\]

with nonnegative \(\kappa,r\), second-order central differences, nodal products, and physical states in \([0,1]\). The grid has spacing \(d_j\). The exact **discrete** diffusion eigenvalues are

\[
\lambda_N(k)=-4\kappa\sum_j\frac{\sin^2(\pi k_j/N_j)}{d_j^2}\le0.
\]

The continuum eigenvalues are \(\lambda_\infty(k)=-\kappa|\xi_k|^2\), with \(\xi_{k,j}=2\pi k_j/\ell_j\). They are not interchangeable at high frequency. Nodal multiplication implements cyclic convolution, \(m=k+l\pmod N\); dealiased Galerkin multiplication is a different semidiscrete equation.

Use the repository-compatible vector-field bracket

\[
[X,Y](u)=Y'(u)X(u)-X'(u)Y(u).
\]

Consequently

\[
C(u)=[A,B](u)=B'(u)L_Nu-L_N B(u)
=r\{L_N(u^2)-2uL_Nu\}
=r\sum_jw_{ij}(u_j-u_i)^2\ge0.
\]

Here the neighbor weights already include \(\kappa/d_j^2\). This is exactly quadratic in perturbation amplitude, has units state/time², and vanishes for constant fields, \(r=0\), and \(\kappa=0\). In the continuum it is \(2\kappa r|\nabla u|^2\). Bracket conventions with the opposite sign give the same consistently nested double brackets; a paper or implementation must still state its convention before reporting first-bracket signs.

For rightmost action first,

\[
S_h^{DF}=D_{h/2}R_hD_{h/2},\qquad S_h^{RF}=R_{h/2}D_hR_{h/2}.
\]

At fixed resolved spatial operator, smooth bounded states, and sufficiently small \(h\), the BCH modified vector fields are

\[
\log S_h^{DF}=h(A+B)+h^3E_{DF}+O(h^5),\qquad
E_{DF}=-\frac1{24}[A,C]-\frac1{12}[B,C],
\]

\[
\log S_h^{RF}=h(A+B)+h^3E_{RF}+O(h^5),\qquad
E_{RF}=\frac1{12}[A,C]+\frac1{24}[B,C].
\]

The leading **exact-flow-minus-split** correction is \(-h^3E\), not \(+h^3E\). The flow map's endpoint expansion includes higher-order terms even though the formal symmetric logarithm has odd powers. These are fixed-operator asymptotics, not uniform estimates as \(h|\lambda|\) or spatial derivatives grow.

In the smooth continuum constant-diffusion case,

\[
[A,C]=-4\kappa^2r\|\nabla^2u\|_F^2,\qquad
[B,C]=2\kappa r^2(1-2u)|\nabla u|^2.
\]

Thus

\[
\Phi_h-S_h^{DF}=\frac{h^3}{6}\bigl[-\kappa^2r\|\nabla^2u\|_F^2
+\kappa r^2(1-2u)|\nabla u|^2\bigr]+O(h^4),
\]

whereas RF has different coefficients. These continuum identities must not replace independently evaluated discrete brackets in the nodal model. The sign formula was independently checked on the scalar exactly solvable example \(A(x)=1,B(x)=x^2\): ABA minus exact flow has coefficient \(x^2/6-1/12\).

**Consequence:** a single nonnegative local gate can ensure physical nulls but does not encode the sign-changing double-commutator defect. A field-level signed source dictionary and output transport are required for complete leading-order expressive coverage.

## 3. Exact quadratic response: the strongest physically grounded candidate

Expand about an interior constant initial background \(c_0\):

\[
u(t;c_0+\varepsilon v)=c(t)+\varepsilon a(t)+\varepsilon^2b(t)+O(\varepsilon^3),
\qquad c'=r(c-c^2),\quad 0<c_0<1.
\]

The quadratic coefficient convention is half the second directional derivative. Let

\[
z(t)=e^{-rt},\quad d(t)=c_0+(1-c_0)z(t),\quad
c(t)=c_0/d(t),\quad q(t)=z(t)/d(t)^2.
\]

Then \(q'=r(1-2c(t))q\), \(a(t)=q(t)D_tv\), and variation of constants gives the exact semidiscrete quadratic response

\[
b(h)=-rq(h)\int_0^h q(s)D_{h-s}\big[(D_sv)^2\big]\,ds.
\]

Each product contains the input Fourier phases. With normalized Fourier coefficients its pair kernel is

\[
K^{coupled}_{k,l,m}=-rq(h)\int_0^hq(s)
\exp\{(h-s)\lambda_m+s(\lambda_k+\lambda_l)\}\,ds,
\qquad m=k+l\pmod N.
\]

It is symmetric in \(k,l\), respects complex conjugacy, and depends on **both** input diffusion rates and the actual output rate. Multiplying by real-even learned functions of the eigenvalues retains real output and translation equivariance; keeping only power magnitudes destroys phase-sensitive pair interactions.

The exact DF quadratic split response has scalar background coefficient

\[
b_R(h)=-q(h)\frac{1-z(h)}{d(h)}=-rq(h)\int_0^hq(s)\,ds,
\]

and pair kernel

\[
K^{DF}_{k,l,m}=b_R(h)\exp\bigl\{\tfrac h2(\lambda_k+\lambda_l+\lambda_m)\bigr\}.
\]

Therefore the exact DF quadratic **defect** is

\[
\boxed{K^{defect}_{k,l,m}=-rq(h)\int_0^hq(s)
\left[e^{(h-s)\lambda_m+s(\lambda_k+\lambda_l)}-
 e^{h(\lambda_k+\lambda_l+\lambda_m)/2}\right]\,ds.}
\]

This is a particularly useful constructive upgrade: all diffusion exponent arguments inside the integral are nonpositive; the response is finite-step damped; \(\kappa=0\) cancels inside the integrand exactly; \(r=0\) and \(h=0\) vanish; constant input directions are handled through centering. It already transports an amplitude² branch and must **not** receive an extra quadratic output gate, which would turn the leading correction into amplitude⁴ and erase the needed response.

A numerical quadrature must preserve the cancellation. Ordinary quadrature of \(q(s)\) has a nonzero error at \(\kappa=0\) if the analytic split scalar is subtracted separately. Either use the boxed zero-subtracted integrand directly or normalize positive quadrature weights to the exact known integral

\[
I(h)=\int_0^hq(s)ds=\frac{1-e^{-rh}}{r\{c_0+(1-c_0)e^{-rh}\}},\quad I(h)\to h\text{ as }r\to0.
\]

Use `expm1` and a deliberate zero-rate limit. The quadratic DF defect also vanishes whenever \(\lambda_k+\lambda_l-\lambda_m=0\), because the two exponentials coincide for every integration time. This is a useful paired-mode null test; it does not imply all higher-amplitude defects vanish. A one-node midpoint quadrature of the boxed DF defect is identically zero, because its exponential difference vanishes at s=h/2. A nontrivial symmetric physical defect needs at least two nodes or explicitly integrated higher commutator moments; a learned nominal rank-1 branch is a different representation and must not be equated with one midpoint quadrature node. Compare the numerical kernel with independent symmetric-amplitude teachers at several amplitude and time refinements. Do not infer cubic consistency just because an overall \(h^3\) multiplier exists.

### Finite-step φ functions and their limits

If \(q(s)\) is frozen to one, the diffusion part integrates analytically. With \(\nu=\lambda_k+\lambda_l-\lambda_m\),

\[
\int_0^h e^{(h-s)\lambda_m+s(\lambda_k+\lambda_l)}ds
=e^{h\lambda_m}h\varphi_1(h\nu)
=\frac{e^{h(\lambda_k+\lambda_l)}-e^{h\lambda_m}}{\nu},
\quad\varphi_1(z)=\frac{e^z-1}{z}.
\]

At \(\nu=0\), the limit is \(he^{h\lambda_m}\). **\(\nu\) can be positive even though every diffusion eigenvalue is nonpositive.** Computing \(\varphi_1(h\nu)\) and \(e^{h\lambda_m}\) separately can overflow and lose a finite result. Use stable divided differences, nonpositive-exponent quadrature, or an `expm1` form scaled by the larger endpoint exponent. Exact background evolution or a matched small-time expansion remains necessary: freezing \(q\) indiscriminately can introduce missing lower-order reaction terms. A cheap φ kernel is a proposed approximation, not the exact nonlinear background response.

### Zero mode, stiffness, and amplitude caveats

For a conjugate pair \(k,-k\), output \(m=0\) has \(\lambda_m=0\). Its mean response is still generated by damped inputs. It must not be suppressed merely because the output has zero diffusion rate. Conversely, a global factor based only on the most negative grid eigenvalue can erase physically relevant low-frequency or mean corrections.

At fixed positive \(h\), as \(\kappa\to\infty\), the actual solution with mean-centered perturbation approaches \(R_h(c_0)\). DF approaches the same value; RF approaches \(R_{h/2}(\langle R_{h/2}(u_0)\rangle)\), which retains an initial nonlinear averaging bias. The conjugate-pair mean response is typically \(O(1/\kappa)\), whereas a blind \(h^3\kappa\) or \(h^3\kappa^2\) formula grows. This explains why finite-step damping is a stronger direction than a more flexible unconstrained decoder multiplying a cubic polynomial.

For fixed interior c₀ and r>0, q(h) and the homogeneous quadratic coefficient tend to zero as h→∞. The exact quadratic defect consequently decays as the solution approaches its stable logistic equilibrium; an interval-preserving capacity map applied to an unbounded cubic correction does not by itself enforce the correct equilibrium. This is an additional independent long-time falsifier.

The spectral gap controls homogenization; the largest-magnitude eigenvalue controls the fastest resolved decay and explicit stability. They are different. The quadratic expansion is rigorous locally around interior constants and small amplitude; its usefulness for large rough fields requires higher-amplitude experiments. Background derivative bounds are not uniform as \(c_0\) approaches 0 or 1.

## 4. Target-consistent spatial upgrades and estimators

For resolved low modes,

\[
\lambda_N(k)=-\kappa|\xi_k|^2+
\frac\kappa{12}\sum_jd_j^2\xi_{k,j}^4+O(d^4|\xi|^6).
\]

FD consequently under-diffuses relative to the continuum heat operator. Linear propagation differs by approximately

\[
e^{h\lambda_N}-e^{h\lambda_\infty}
\simeq e^{h\lambda_\infty}\frac{h\kappa}{12}\sum_jd_j^2\xi_{k,j}^4.
\]

This explains an approximately fourfold reduction under grid doubling for smooth modes. Nonlinear aliasing adds a separate error: resolving the initial modes does not ensure that their quadratic products are resolved. At \(N=32\), initial frequencies up to 15 can generate sums up to 30, which fold on the native nodal grid. A 3/2-padded quadratic RHS defines a dealiased Galerkin target; it does not retroactively change the current FD/nodal target.

For a continuum claim the error budget must be explicitly split:

\[
\|\widehat u_N-P_Nu\|\le
\underbrace{\|\widehat u_N-u_N\|}_{\text{temporal/model}}
+\underbrace{\|u_N-P_Nu\|}_{\text{spatial/operator}}
+\underbrace{\eta_{reference}}_{\text{reference uncertainty}}.
\]

Here \(P_N\) and the continuous initial field must be fixed consistently, and RMS/maximum budgets are separate. Accept only if independently estimated upper budgets sum below the requested target. Existing empirical temporal acceptance cannot certify the second term.

A three-grid detector can compare propagated solutions at \(N,2N,4N\), projected to the same grid and norm. In a verified asymptotic second-order regime, \(d=\|u_{2N}-P_{2N}u_{4N}\|\) gives \(d/3\) as an estimate of the 4N error and \(4d/3\) as an estimate of the 2N error, after restriction to the common comparison space. This is not a certificate when order changes, aliasing dominates, initial fields change under resampling, or cancellation is possible. Report observed refinement ratios and inflate or reject outside the asymptotic regime; do not silently divide by three. In maximum norm, an RMS/L² estimate requires a valid conversion, which can be very pessimistic or resolution-dependent.

Spatial upgrade choices require different guarantees:

- Keep the current nodal target and measure its spatial floor honestly; this is immediately testable and needs no solver rewrite.
- Implement a separate spectral/dealiased target with exponential integration and converged oversampling. Exact pointwise logistic subflows generate arbitrarily high harmonics; 3/2 padding of the **quadratic RHS** is not a proof that one sampled exact logistic map is fully dealiased.
- Consider a higher-order positivity-preserving conservative discretization. Ordinary fourth-order central differences have signed off-diagonal weights and lose the M-matrix maximum-principle argument.
- A finite Galerkin projection is not automatically pointwise interval-preserving. Spectral diffusion and projected reactions need measured interval excursions and possibly a separately proved limiter; the FD interval proof must not be carried over unchanged.

## 5. A stability-weighted residual estimator instead of an endpoint proxy

For a differentiable family \(\Psi_s(u_n)\) with \(\Psi_0(u_n)=u_n\), construct a continuous local numerical path \(y(s)=\Psi_s(u_n)\). Its residual is

\[
\rho(s)=\partial_s\Psi_s(u_n)-F_N(\Psi_s(u_n)).
\]

A horizon JVP computes the actual derivative. For the FD problem on bounded states, the Jacobian is

\[
J_N(u)=L_N+r\,\mathrm{diag}(1-2u).
\]

Because \(L_N\) is symmetric negative semidefinite and a Metzler zero-row-sum generator,

\[
\mu_2(J_N)\le r,\qquad \mu_\infty(J_N)\le r.
\]

Thus a valid stability comparison gives

\[
\|e(T)\|\le e^{rT}\|e(0)\|+
\int_0^T e^{r(T-s)}\|\rho(s)\|ds.
\]

A tighter bound can use certified state lower bounds to replace \(r\) by \(r(1-2u_{min})\). Do not claim that diffusion's large negative \(\lambda_{max}\) causes positive \(e^{|\lambda_{max}|T}\) amplification: a generic Lipschitz bound is needlessly pessimistic here. Nor can the spectral-gap decay simply be assigned to every error, since the mean mode and heterogeneous reaction coupling remain.

The current proxy sums \(h\|\rho(h)\|\), using one endpoint sample per substep. It is not the integral above, not stability weighted, and may miss interior peaks. Upgrade to two/three interior quadrature nodes with residual-refinement tests, appropriate time weights, and independent calibration. A **rigorous** certificate additionally needs a bound on quadrature error and the admissible reconstruction trajectory, not only autodiff. Calibration envelopes, conformal bounds, or empirical maximum ratios must be labeled empirical or distributional under their stated assumptions. They cannot be called deterministic certificates.

Step doubling \(\|\Psi_h(u)-\Psi_{h/2}^2(u)\|\) can share systematic spatial or model error. A Richardson denominator is justified only after demonstrating the relevant asymptotic order; stiff finite-step schedules cannot assume it. Keep both independent residual and refinement diagnostics, and ensure estimates target the same operator as deployment.

The exact PDE does not contract all perturbations: near zero logistic growth has derivative approximately \(e^{rh}\). A universal neural Jacobian penalty \(\|J\Psi_h\|\le1\) would contradict the physics. Use the physical growth envelope and separated mean/fluctuation analysis. A few power-iteration/JVP directions estimate sensitivity but do not prove an upper operator-norm bound.

## 6. Guarding a learned correction and the unavoidable cost algebra

Propose \(\Psi_h=S_h^{DF}+g_h\,\delta_\theta\), followed by the declared physical output map, where \(g_h\) is decided without fresh truth. Cheap measured features can include predicted residual/headroom, stiffness, spectrum, and a calibrated uncertainty budget. A rejected proposal uses a cheaper validated classical step/refinement. This is a conditional deployment design, not a guarantee that every accepted correction beats the base.

The existing signed-capacity map is

\[
P(b,d)=b+\frac{m\,d}{m+|d|},\quad m=1-b\ (d\ge0),\ m=b\ (d<0).
\]

It preserves \([0,1]\) when the base is in the interval and has derivative one in \(d\) at zero for interior states. It does not prove accuracy, monotonicity of the entire network, contractivity, or exact mean evolution. Near saturated capacity its behavior is not a uniform high-order approximation. Mean recalibration after this map redistributes the full spatial field and must be included in spatial-error assessment.

For proposal stages reached with probabilities \(p_j\), actual deployed expected cost is

\[
C_{deploy}=\sum_jp_jC_j+p_{fallback}C_{fallback}.
\]

Every rejected attempt, encoder, transform, JVP, synchronization, refinement, and fallback counts. For one proposal including all decision/estimator work with cost \(C_A\), the general utility condition is \(C_A+p_fC_f<C_C\). If the fallback is exactly the same classical solver, \(C_f=C_C\), this simplifies to

\[
C_A<(1-p_{fallback})C_C.
\]

If the proposal already computes a complete same-step DF base plus a correction, it cannot outperform that same-step base. Its only solver-cost route is reducing steps or removing/reusing work. Accuracy headroom must therefore be identified **before** training. Against a cheap fused DF rollout, even perfect acceptance may be uneconomic.

Report sums, p50/p90/p95/p99, worst cost, peak memory, failure/retry distribution, and separately timed standalone policies. Reference-informed frontiers are useful upper potential estimates, but cannot choose deployment steps using unknown errors. Training/teacher/preparation costs should be amortized explicitly: break-even deployment count is \(T_{offline}/(C_C-C_{deploy})\) only if the denominator is positive.

A particularly important classical benchmark is fusion of consecutive DF heat stages. An n-step bare DF rollout contains adjacent \(D_{h_j/2}\) and \(D_{h_{j+1}/2}\), which combine to \(D_{(h_j+h_{j+1})/2}\). This needs n+1 diffusion calls rather than 2n. An arbitrary nonlinear learned correction at every completed endpoint prevents this exact fusion. Arranging a learned correction inside the middle stage could retain fusion and end-step damping, but exact endpoint defects may then require inverse diffusion to represent; inverse heat is unstable and must not be introduced casually. Benchmark the fusion benefit before proposing such a topology.

## 7. Variable-step composition, order sensitivity, and training targets

For an autonomous exact flow,

\[
\Phi_b\circ\Phi_a=\Phi_{a+b}=\Phi_a\circ\Phi_b.
\]

Proposed regularizers compare \(\Psi_b(\Psi_a(u))\) with \(\Psi_{a+b}(u)\), and compare reversed unequal-step sequences, while retaining supervised independent teachers. If \(\Psi_h=\Phi_h+h^{p+1}E+\cdots\), the leading composition defect is proportional to

\[
a^{p+1}+b^{p+1}-(a+b)^{p+1}.
\]

Modified generators \(F+h^2E\) for a second-order map also produce reversed-order commutator terms proportional in magnitude to \(ab(b^2-a^2)[F,E]\). Large unequal steps and state-dependent nonlinear corrections can make the failure severe outside the small-time regime. A semigroup loss alone is insufficient: the identity map is a perfect semigroup but solves the wrong dynamics. Require \(\Psi_0=I\), \(\partial_h\Psi_0=F\), physical limits, and independent accuracy supervision.

Use genuinely unseen individual h values, all permutations of selected triples, abrupt large-to-small and small-to-large changes, long mixed schedules, dense intermediate outputs, and reference convergence. Reversed [0.17,0.07,0.03] must remain a retained development counterexample rather than disappear into a new favorable frontier.

Optimize the actual DF model rather than carrying over RF learning rates/batches. Train the **defect relative to the same effective base**, separately for mean and centered fluctuation, with noise-aware scaling. A candidate loss is

\[
\mathcal L_\delta=\sum_{regime}w_{regime}
\frac{\|\Psi_\theta-S^{DF}-(u_{teacher}-S^{DF})\|^2}
{\tau_{floor}^2+\|u_{teacher}-S^{DF}\|^2},
\]

plus independent maximum-error, composition, and physically allowable sensitivity diagnostics. \(\tau_{floor}\) must include reference uncertainty and declared target scales; dividing by \(h^3\) at tiny h amplifies subtraction and teacher noise. Loss normalization must not reweight confirmation after inspection. Actual RMS/maximum gates remain distinct; smooth max surrogates optimize but do not certify maximum accuracy.

## 8. Mean–variance closure: what is exact and what cannot close

For \(c=\langle u\rangle\), \(v=u-c\), \(V=\langle v^2\rangle\), \(M_3=\langle v^3\rangle\), and \(E=\langle v,L_Nv\rangle\le0\), the exact instantaneous identities are

\[
c'=r(c-c^2-V),\qquad
V'=2E+2r(1-2c)V-2rM_3.
\]

For \(u\in[0,1]\), \(0\le V\le c(1-c)\) and \(-cV\le M_3\le(1-c)V\). Mean is nondecreasing because \(c'=r\langle u(1-u)\rangle\ge0\); it is not conserved.

The present closure freezes initial moment rates (and regularizes degenerate variance fractions). Two fields can share \(c,V\) but differ in \(M_3\), spectrum, and diffusion energy; therefore a universal autonomous two-moment closure cannot be exact. Learned future \(E,M_3\) need state/spectral/phase information or a justified restricted distributional model. An exact identity evaluated at the initial state does not establish an exact future closure.

For \(0<c<1\), the fraction \(z=V/[c(1-c)]\) lies in \([0,1]\). At \(c=0\) or \(c=1\), bounded fields have \(V=0\); use the exact constant branch instead of dividing by zero. A logistic-coordinate rate locks \(z=1\), although diffusion must reduce the variance of a nonconstant binary field. The present regularized zero fraction rate fails to represent this limiting behavior. A proposed inward-flux closure

\[
z'=a(1-z)-bz,\qquad a,b\ge0,
\]

can move \(z=1\) inward when \(b>0\); constant-field \(z=0\) requires \(a=0\). For frozen rates it has an exact bounded exponential update. These rates must match the initial exact moment derivative where resolved and then be tested against independently evolved moments. Couple \(c'=rc(1-c)(1-z)\) with a positivity-preserving integrator. This is a proposed repair, not proof that the closure improves costs or full-field accuracy. Keep it a side experiment until it beats analytic spectral-mean and learned endpoint controls on separate mean/spatial diagnostics.

## 9. Rank, compression, and cost tradeoffs

A quadrature node in the exact quadratic response is a separable input–input–output term: diffusion-filter both inputs, multiply, diffusion-filter the output. R quadrature nodes plus analytic split terms give an **offline constructive representation**. They do not prove that the current learned parameterization of nominal rank R can achieve that response. Its discrete commutator multiplies a pair by \(\lambda_m-\lambda_k-\lambda_l\), expanding each nominal rank into up to three separable terms.

Dense pair matrices are \(O(N^2)\) in the number of spatial modes (and very expensive in 2D). Separable physical quadrature is \(O(RN\log N)\), but constants matter. Current pair correction transform count is \(1+4R\), versus Source's two correction transforms. At small grids FFT launch overhead and host control may dominate. At larger grids memory traffic, transforms, and batching matter; no universal ranking follows from parameter count.

Test rank 0, 1, 2 and the exact low-node physical kernel before increasing rank. The present implementation lacks rank 1, so that is genuinely new work. Include three paired training seeds, measured effective approximation error, explicit analytic terms, and actual full rollout transforms. For the DF physical defect, distinguish rank-1 learned pair factors from the identically zero one-node midpoint quadrature. Try analytic fixed factors plus a small learned remainder rather than asking an unconstrained network to learn all diffusion damping. Reuse the input FFT where mathematically identical and test exact zero-initialization/source-null behavior after fusion. A nonlinear capacity map depends on separately reconstructed base and increment, so simply adding their spectra does not remove an inverse transform without changing the map.

Source-before-compression is worth retaining because physical high–high pairs can create low/zero output modes before an encoder truncates them. But a fully local commutator source is only one quadratic combination: it does not encode every input/output diffusion-rate kernel. Test a signed dictionary of discrete double brackets, time-integrated source features, and phase-aware pair products. Multi-scale cutoff/rank must be chosen on development only and accompanied by aliasing/translation/phase/null tests. Removing the full source bypass should be a separately labeled information-bottleneck ablation.

Large amplitude can require cubic and higher responses. A recursive variational hierarchy supplies structured higher terms, e.g. the cubic coefficient is forced by \(-2r\,a(s)b(s)\). Its products need phase and causal ordering. Additional amplitude-dependent heads may learn such terms but cost and physical-null guarantees require explicit analysis. Do not turn the entire corrector into amplitude⁴ by gating an already quadratic branch; a dedicated cubic residual is a different, testable mechanism.



### Calibration, selection and the independent statistical unit

For an exchangeable calibration set of **independent functions** and a joint query score \(S_i\), split conformal uses

\[
k=\lceil(n+1)(1-\alpha)\rceil,\qquad q_\alpha=S_{(k)}.
\]

If \(k>n\), the usual valid convention is an infinite threshold. At \(\alpha=0.01\), the first finite rank requires \(n\ge99\). Grid crossings, tolerances, schedules, training seeds and phase-related shape variants do not create new independent function draws. A joint score can maximize over a **fixed declared** set of queries/norms; an RMS score is not automatically a maximum-norm guarantee.

Even a valid marginal guarantee must be distinguished from selective accuracy. If acceptance entails a certified/calibrated upper bound below the target and the marginal probability of a bound miss is at most \(\alpha\), then

\[
P(\mathrm{bad}\mid\mathrm{accept})\le
\frac{P(\mathrm{bound\ miss})}{P(\mathrm{accept})}
\le\frac{\alpha}{P(\mathrm{accept})}.
\]

Thus 99% marginal coverage does not itself imply 99% correctness conditional on accepting. Nor does marginal coverage imply coverage for every realized calibration set or physics stratum. Exchangeability and target assumptions are essential; distribution shift can invalidate them. The recent CUQ paper already treats continuum scores under dominating regularity bounds, so the opportunity is a sharp **usable** spatial/temporal bound and selective controller with complete cost, rather than first inventing continuum conformal prediction.

For a fixed policy evaluated on \(n\) independent comparable accepted units with zero failures, the one-sided 95% binomial upper bound is

\[
p_{\mathrm{upper}}=1-0.05^{1/n}.
\]

It falls below 1% at \(n=299\). This calculation assumes genuinely independent Bernoulli units and a fixed policy/distribution; it is not applicable to 1179 dependent historical row outcomes. Multiple policy selection, simultaneous stratum claims and correlated queries require an adjusted design. The proposed 12-field development and 24-field confirmation stages are bounded diagnostics, not powered 1% population guarantees.

### Learned time integration: consistency and RK order are different

TI-DeepONet's inspected final softmax weights depend on the current state and do not depend on h; they are positive and sum to one, which supplies a first-order weight consistency condition. It does not alone supply the rooted-tree order conditions of its fixed classical RK4 stages. For fixed stage matrix \(A\), \(c=A\mathbf1\), order through three requires

\[
b^T\mathbf1=1,\quad b^Tc=\tfrac12,\quad
b^T(c\odot c)=\tfrac13,\quad b^TAc=\tfrac16.
\]

For the classical four stages, \(c=(0,\tfrac12,\tfrac12,1)\) and \(Ac=(0,0,\tfrac14,\tfrac12)\). These four linear constraints uniquely give

\[
b=(\tfrac16,\tfrac13,\tfrac13,\tfrac16).
\]

There is no nontrivial free final-weight nullspace preserving even third order for that fixed tableau. A defensible upgrade therefore learns additional stage structure, a constrained larger tableau, a useful embedded estimator, or a residual that vanishes at sufficiently high order. This is a released-code/order deduction; the journal article's complete theoretical claims were not accessible. Positive weights do not establish A-stability, stiff order, positivity of nonlinear stages, or accuracy under arbitrary unequal steps.

### Monotonicity, concavity and physical growth

For the cooperative FD logistic target, \(J(t)=L_N+r\operatorname{diag}(1-2u(t))\) has nonnegative off-diagonal entries. Its fundamental matrix \(U(t,s)\) is componentwise nonnegative. A first variation \(a\) and the quadratic coefficient \(b\) satisfy

\[
a'=J(t)a,\qquad b'=J(t)b-r a^2,\qquad b(0)=0.
\]

Therefore \(b(t)=-r\int_0^t U(t,s)a(s)^2\,ds\le0\) componentwise: the exact flow is pointwise concave in its initial field, and is also monotone. Positive linear heat flow and the increasing concave logistic reaction flow preserve these properties under RF/DF composition. A useful diagnostic is the pointwise Jensen inequality

\[
\Phi_h(\theta u+(1-\theta)v)\ge
\theta\Phi_h(u)+(1-\theta)\Phi_h(v),\qquad 0\le\theta\le1.
\]

Interval-preserving learned corrections need not satisfy it. This is a new proposed diagnostic for M21, not a proof that the referenced **convex** monotone semigroup theorem applies to this concave flow. Any transformed-variable theorem needs its own domain/assumption check. The physical growth envelope is \(e^{rh}\), not universal contraction by one; logistic evolution near zero is expansive.

### Fractional, tensor and geometric extensions: what the symbols mean

A truncated Grünwald–Letnikov feature has the discrete symbol

\[
G_{\beta,K}(\omega)=s^{-\beta}\sum_{j=0}^{K-1}(-1)^j{\beta\choose j}e^{-ij\omega}.
\]

At \(\omega=0\), the finite sum generally does not vanish for noninteger \(\beta\). Holding a learned \(s\) fixed while changing actual spacing does not enforce physical derivative units. The inspected delNO implementation includes learned effective-spacing scaling; it does **not** omit scaling altogether. Test raw feature leakage and whole-model behavior separately, and compare actual physical symbols before promoting fractional orders.

With physical wavevector \(\xi\), a real spatial fractional feature can use \(|\xi|^\beta\) or \((-\lambda/\kappa)^{\beta/2}\) for \(\kappa>0\), both with units length to the power \(-\beta\) and a separately defined \(\kappa=0\) convention. Since our diffusion eigenvalue \(\lambda\) already contains \(\kappa\), it has units inverse time: \((-t_{ref}\lambda)^{\beta/2}\) is instead a dimensionless diffusion-rate feature. Raw \((-\lambda)^{\beta/2}\) is not dimensionally interchangeable with a spatial derivative. Physical coordinates, reference scales and branch/reality conventions must be declared.

Coordinate Tucker rank compresses an output tensor, channel-weight Tucker rank compresses FNO parameters, and separable mode-pair rank compresses a bilinear interaction kernel. These are different mathematical objects. Nominal parameter savings need not reduce transforms or wall time. In geometry, Fourier convolution selection \(m=k+l\) becomes overlaps \(\langle\psi_m,\psi_k\psi_l\rangle\) with the correct integration measure and boundary operator. A learned latent near/far routing is not automatically an error-controlled multipole expansion. Fixed-kernel quadrature consistency is not automatically learned-mesh or remeshing robustness.

### Upgrade requirements and significance for every mechanism

<a id="math-m00"></a>

### M00 — Headroom map and fused classical reference

Compare $\min_{a,h,N}\{C(a,h,N):e_{\rm RMS}\le\tau_R,\ e_\infty\le\tau_\infty\}$. Adjacent DF heat stages fuse: $D_aD_b=D_{a+b}$.

**What this signifies:** Find regimes where extra correction accuracy can actually replace classical steps; otherwise establish that the classical solver is the right answer.

**Required implementation/verification:** Implement/profile fused variable-step DF, both orientations, ETDRK4 and GL3; retain all-domain and screened-stratum reports separately; no new training needed first.

**Validity/decision limit:** Reject a neural solver-utility campaign if bare DF dominates the relevant cost–accuracy region; a tighter tolerance chosen after viewing confirmation does not rescue the old gate.

<a id="math-m01"></a>

### M01 — Consistent spatial target and two-resolution discrepancy

$e_{\rm total}\le e_{\rm time}+e_{\rm space}+\eta_{\rm ref}$; $\lambda_N(k)=-4\kappa\sum_j\sin^2(\pi k_j/N_j)/d_j^2$, whereas $\lambda_\infty=-\kappa|\xi|^2$.

**What this signifies:** Explain and reduce the common continuum floor; expose continuum errors before an otherwise small temporal indicator accepts them.

**Required implementation/verification:** Declare FD/nodal and spectral/Galerkin tracks, teachers and projections; add bounded N/2N/4N checks, error-order detector and separate spatial cost.

**Validity/decision limit:** Reject a continuum claim if only the discrete temporal indicator is small, or if refinement order is unresolved; a solver-core change is not a neural gain.

<a id="math-m02"></a>

### M02 — Exact finite-step quadratic DF defect

$K^{\rm defect}_{k,l,m}=-rq(h)\int_0^h q(s)[e^{(h-s)\lambda_m+s(\lambda_k+\lambda_l)}-e^{h(\lambda_k+\lambda_l+\lambda_m)/2}]\,ds$.

**What this signifies:** Replace stiffness-growing cubic extrapolation with a response that is exact at quadratic amplitude and damped at large diffusion/long times.

**Required implementation/verification:** Use zero-subtracted stable quadrature or scaled divided differences, analytic logistic background, centered phases and independent amplitude/time teachers. One-node midpoint DF defect is identically zero.

**Validity/decision limit:** Reject if quadrature breaks zero diffusion/reaction limits, if finite-amplitude gains disappear, or if required transforms cost more than the saved steps.

<a id="math-m03"></a>

### M03 — Physical factors plus learned rank-1/rank-2 remainder

$\widehat\delta_m=\sum_{k+l=m}\sum_{a=1}^R f_a(k)g_a(l)o_a(m)\widehat v_k\widehat v_l$ with symmetry/reality constraints and physical defect factors.

**What this signifies:** Retain phase-aware robustness with fewer transforms than a generic high-rank branch; rank 1 may offer a useful cost point.

**Required implementation/verification:** Distinguish learned nominal rank, expanded commutator terms and coordinate Tucker rank; share only algebraically identical transforms; tune the actual DF base.

**Validity/decision limit:** Reject if low-rank training cannot recover physical response, if worst-order benefit is seed-specific, or if classical/FNO Pareto dominance persists.

<a id="math-m04"></a>

### M04 — Signed nonlinear sources before output compression

$P_KQ(u)\ne P_KQ(P_Ku)$ in general; $[A,C]$ and $[B,C]$ provide signed sources with the required leading amplitude-squared terms before zero-preserving transport.

**What this signifies:** Preserve high-high-to-low/zero interactions and remote receiver response that an input-only cutoff or post-transport point gate erases.

**Required implementation/verification:** Build a signed discrete source dictionary; remove undeclared learned-branch bypass in the pure ablation; retain exact constant, zero-h, zero-r and zero-kappa checks.

**Validity/decision limit:** Reject a bottleneck conclusion if a full-source bypass remains in only one branch; structural span improvement without finite-amplitude utility is insufficient.

<a id="math-m05"></a>

### M05 — Product-consistent anti-aliasing and grid-scaled filters

Quadratic products require the declared projected convolution, not endpoint filtering. For low resolved modes, $\lambda_N-\lambda_\infty=\kappa\sum_jd_j^2\xi_j^4/12+O(d^4|\xi|^6)$.

**What this signifies:** Remove spurious retained-mode folding and enforce physical scale when resolution changes.

**Required implementation/verification:** Separate 3/2-padded quadratic RHS from exact pointwise logistic flow, which has infinitely many harmonics; test both target conventions and actual spacing rather than learned effective spacing.

**Validity/decision limit:** Reject if higher-harmonic energy is silently discarded, if positivity fails after projection, or if padding costs exceed any useful step savings.

<a id="math-m06"></a>

### M06 — Stiffness-aware trust envelope for the correction

$\Psi_h=S_h^{DF}+g(h,r,\lambda_{\rm input},\lambda_{\rm output},u)\delta_\theta$; bounds/nulls alone do not bound $D_u\Psi_h$.

**What this signifies:** Suppress harmful corrections where the base is already accurate or the finite-step response is outside validated support.

**Required implementation/verification:** Use analytic response/stiffness features and development-only trust calibration; test output-zero conjugate pairs and physical growth rather than universal contraction.

**Validity/decision limit:** Reject if the envelope simply switches all corrections off, if low/mean modes are erased by a global lambda-max scale, or if routing uses held-out truth.

<a id="math-m07"></a>

### M07 — Conditional solver bank with a classical safe path

$C_{\rm deploy}=\sum_jp_jC_j+p_fC_f$; one proposal including all decision/estimator work requires $C_A+p_fC_f<C_C$. If $C_f=C_C$, this reduces to $C_A<(1-p_f)C_C$.

**What this signifies:** Skip unnecessary neural work on easy states; spend correction/refinement work only where the expected accuracy benefit pays for it.

**Required implementation/verification:** Compare bare DF, corrected DF and classical refinement with actual decision costs; freeze features and selection before fresh fields.

**Validity/decision limit:** Reject if route/JVP/reject overhead erases savings, if rare tails dominate totals, or if an oracle truth-based router is used as the deployed method.

<a id="math-m08"></a>

### M08 — Cheap embedded estimator and stability-weighted residual

$\rho(s)=\partial_s\Psi_s-F_N(\Psi_s)$; $\|e(T)\|\le e^{rT}\|e(0)\|+\int_0^Te^{r(T-s)}\|\rho(s)\|ds$ for the bounded FD target.

**What this signifies:** Replace expensive repeated JVP work or blind step agreement with a target-correct, shared-computation error signal.

**Required implementation/verification:** Use two/three interior nodes with refinement validation, a valid continuous reconstruction and separate spatial budget; share stages only when the numerical maps match exactly.

**Validity/decision limit:** Reject a deterministic certificate without quadrature/reconstruction bounds; self-agreement can share the same spatial/model bias.

<a id="math-m09"></a>

### M09 — Function-level calibrated total-error and selective risk

$k=\lceil(n+1)(1-\alpha)\rceil$ for conformal ranks. Marginal coverage gives only $P(\mathrm{bad}\mid\mathrm{accept})\le\alpha/P(\mathrm{accept})$ under a valid acceptance implication.

**What this signifies:** Turn empirical envelopes into explicit distributional statements where assumptions hold; detect spatial/grid shift instead of silently transferring coverage.

**Required implementation/verification:** Do not inflate the cohort to meet statistics blindly: begin bounded reliability diagnostics; stronger guarantee is a separately justified scope with fresh calibration and confirmation.

**Validity/decision limit:** Reject formal coverage claims under reused phases, distribution shift, unbounded spatial score, adaptively selected policies, or inadequate independent sample count.

<a id="math-m10"></a>

### M10 — Unequal-step generator consistency and intermediate supervision

$\Phi_b\Phi_a=\Phi_{a+b}=\Phi_a\Phi_b$, $\Psi_0=I$, $\partial_h\Psi_0=F$. Composition loss alone also accepts the identity or a wrong autonomous flow.

**What this signifies:** Reduce order sensitivity and long-rollout drift while preserving genuine dynamics at arbitrary unseen individual steps.

**Required implementation/verification:** Add generator/defect supervision and physical finite-h response; keep held-out individual h values and dense transients distinct from validation schedules.

**Validity/decision limit:** Reject if a semigroup regularizer favors a physically wrong flow or if continuous h claims only reflect integer-frame forecast multiples.

<a id="math-m11"></a>

### M11 — DF-tuned defect loss with regime and maximum-error control

$\mathcal L_\delta=\sum_gw_g\|\delta_\theta-(u_* -S^{DF})\|^2/(\tau_{floor}^2+\|u_*-S^{DF}\|^2)$ plus independent mean/spatial/max diagnostics.

**What this signifies:** Reduce harmful corrections in already-solved cases and focus capacity on true defects rather than aggregate normalized MSE.

**Required implementation/verification:** Tune LR/batch on effective orientation; include initialization selection and uncertainty floor; keep shared and per-arm tuning as separate fairness experiments.

**Validity/decision limit:** Reject if improvement exists only in the surrogate loss, if tiny defect division amplifies reference noise, or if confirmation is reused for loss selection.

<a id="math-m12"></a>

### M12 — Dynamic feasible mean–variance closure

$c^\prime=r(c-c^2-V)$, $V^\prime=2\langle v,Lv\rangle+2r(1-2c)V-2r\langle v^3\rangle$. Feasible $z=V/[c(1-c)]$ can use $z^\prime=a(1-z)-bz$. The variance fraction is defined only for $0<c<1$; endpoints use the exact constant-field branch.

**What this signifies:** Improve a verified mean-dominated regime while repairing the frozen closure binary-field boundary defect.

**Required implementation/verification:** Dynamic spectral energy/third moment is required for an exact future law; enforce a=0 at constant fields, inward flux at z=1 and avoid double-counting the kernel zero mode.

**Validity/decision limit:** Reject if closure remains initialization-selected, spatial damage rises, or extra reductions cost more than the mean-error benefit.

<a id="math-m13"></a>

### M13 — Dimensionally consistent multi-band/fractional features

Physical derivatives use $(i\xi)^\beta$ or a stated discrete symbol. A truncated GL stencil has constant response $\sum_{j=0}^{K-1}(-1)^j{\beta\choose j}$, generally nonzero.

**What this signifies:** Represent unresolved spectral bands economically without grid-dependent derivative units or unintended constant leakage.

**Required implementation/verification:** Restore actual spacing/symbol explicitly or label dimensionless learned features; compare ordinary derivatives/physical factors before adding fractional orders.

**Validity/decision limit:** Reject if learned effective spacing absorbs training-grid units and fails transfer; a useful fractional feature is not a proof of the PDE generator.

<a id="math-m14"></a>

### M14 — Geometry-aware and anisotropic transport

For non-Fourier eigenfunctions, interactions use $\langle\psi_m,\psi_k\psi_l\rangle$; anisotropic diffusion has symbol $-\xi^TK\xi$.

**What this signifies:** Extend phase/interaction structure to geometry only when anisotropic or irregular-domain headroom is independently demonstrated.

**Required implementation/verification:** Use operator-aware Krylov/rational/multilevel transport or weighted geometry operators; HNO/Radon are comparison directions, not justified default upgrades here.

**Validity/decision limit:** Reject a direct Fourier-pair transplant outside periodic constant coefficients; eigenvalues alone do not control nonnormal advection transients.

<a id="math-m15"></a>

### M15 — Cost-aware algebraic reuse, fusion and batching

$C_{offline}/(C_C-C_{deploy})$ is a break-even count only if the denominator is positive. Parameter rank is not transform or latency cost.

**What this signifies:** Lower real inference/estimator overhead through identical-map reuse and equitable optimized baselines.

**Required implementation/verification:** Profile FFTs, allocations, padding, host synchronization and reduction dispatch; cache only invariant prepared data; benchmark small and larger grids separately.

**Validity/decision limit:** Reject if batching favors only one method, if reused filters depend on changed h/operator, or if fusion changes products/output constraints.

<a id="math-m16"></a>

### M16 — Faithful neural comparison and strongest classical utility

Use a multiobjective Pareto comparison of accuracy, latency, memory and offline cost; equal parameters do not imply equal depth/FLOPs or useful support.

**What this signifies:** Determine whether gains belong to the physical solver, interaction bias, implementation or a genuine neural benchmark advantage.

**Required implementation/verification:** Keep neural-only and hybrid comparisons separate; add recent operator candidates only after faithful usable implementations and target match are verified.

**Validity/decision limit:** Reject a paper win inferred from adapted FNO only, omitted cheap controls, unequal tuning/convergence or initialization-selected efficacy.

<a id="math-m17"></a>

### M17 — Factor-isolated stress and coefficient-transfer bank

Cross amplitude, mean, phase, spectrum, r, kappa, h and N with independent field clusters; eigenvalue gap and maximum rate are separate axes.

**What this signifies:** Identify where finite-amplitude, roughness, stiffness, initial state or resolution actually causes failure.

**Required implementation/verification:** Use sparse factorial designs and bounded shards; preserve exact continuous fields when sampling paired grids.

**Validity/decision limit:** Reject causal claims from banks that change grid, amplitude and horizon together or count shared-field crossings as independent draws.

<a id="math-m18"></a>

### M18 — Fresh independent fields and hierarchical inference

Aggregate outcomes at independent field/function clusters, with paired solver differences and crossed training-seed effects; cell counts do not determine effective n.

**What this signifies:** Give uncertainty intervals with a real independent unit and protect confirmation from architecture-selection reuse.

**Required implementation/verification:** Cluster bootstrap or exact field-level binomial methods under stated assumptions; account for multiple arms/tolerances and report device selection differences.

**Validity/decision limit:** Reject population confidence from one phase cluster, repeated grids/seeds or post-selection slices; do not call a replay a new confirmation.

<a id="math-m19"></a>

### M19 — PDE and boundary portability ladder

Transfer $\partial_tu=Au+N(u)$ only with equation-specific balance, positivity and boundary assumptions; logistic $0\le u\le1$ is not a universal invariant.

**What this signifies:** Assess usefulness beyond favorable scalar periodic logistic reaction–diffusion.

**Required implementation/verification:** Implement and validate each future PDE branch separately, including source structure, operator/boundary flow and diagnostic cost; no automatic expansion now.

**Validity/decision limit:** Reject cross-PDE superiority extrapolated from current scalar experiments; nonperiodic/variable-coefficient physics needs new derivations.

<a id="math-m20"></a>

### M20 — Order-constrained learned embedded stages

For RK, $b^T1=1$, $b^Tc=1/2$, $b^Tc^2=1/3$, $b^TAc=1/6$. Fixed classical RK4 stages already force classic b by third order.

**What this signifies:** Learn a useful estimator/stage extension without silently sacrificing order or relying on an RK4 name.

**Required implementation/verification:** If retaining fixed RK4 stages, learn a higher-order vanishing residual or extra stages/nullspace rather than free b; account for every nonlinear evaluation.

**Validity/decision limit:** Reject arbitrary softmax final weights as fourth-order by default; A-stability is also not implied by positive/sum-one weights.

<a id="math-m21"></a>

### M21 — Physical monotonicity and growth-envelope diagnostics

For bounded cooperative FD logistic flow, $D_u\Phi_h\ge0$ entrywise and $\|D_u\Phi_h\|\le e^{rh}$ in appropriate norms; interval bounds alone do not imply either.

**What this signifies:** Expose or constrain unstable learned sensitivity beyond the existing null/interval tests.

**Required implementation/verification:** Use cooperative FD assumptions explicitly; convex-monotone semigroup theory does not automatically apply to logistic concave reaction or spectral projections. Add pointwise Jensen/concavity diagnostics for the cooperative FD logistic target; do not transfer a convex-semigroup theorem without checking its assumptions.

**Validity/decision limit:** Reject universal contraction<=1 near u=0, where physical logistic growth is expansive; a sampled JVP is not a global Jacobian certificate.

<a id="math-m22"></a>

### M22 — Effective configuration, provenance and reproducible measurement

A canonical resolved-config digest must include actual base, batch, LR, target, products, precision and software; proof/evidence classes remain explicit.

**What this signifies:** Remove human-label ambiguity, preserve repaired workflows and make cost/scientific attribution reproducible.

**Required implementation/verification:** Retain historical regression controls and split-archive verification; independently validate paper dates/proofs where currently unavailable; do not alter Tower.

**Validity/decision limit:** Reject replacing native selection counts with cloud counts, relabeling old immutable artifacts, or reporting numerical timers as billed/resource utilization.

<a id="math-m23"></a>

### M23 — Dedicated cubic Volterra remainder

For $u=c+\epsilon a+\epsilon^2b+\epsilon^3d+\cdots$, $d^\prime=(L+r(1-2c))d-2rab$; subtract the same-base cubic split response.

**What this signifies:** Address finite-amplitude failures remaining after the quadratic physical kernel is accurate.

**Required implementation/verification:** Retain causal phase products and physical limits; add only after an amplitude-series diagnostic isolates a cubic bottleneck.

**Validity/decision limit:** Reject extra amplitude-squared gating of a quadratic branch, which gives epsilon^4; reject added cubic cost without new useful coverage.


### Independent algebra checks made during this review

The BCH sign was checked on the exactly solvable scalar fields \(A(x)=1\), \(B(x)=x^2\). The coupled-minus-DF quadratic coefficient, divided by \(h^3\), was compared on a small FD grid against independently evaluated discrete BCH double-commutator coefficients using autodiff JVPs and symmetric-amplitude bracket extraction. The recorded coefficient discrepancy decreased approximately 0.00115 to 0.00028 as h decreased from 0.001 to 0.00025. These are small mathematical consistency checks, not a separately integrated full nonlinear teacher, new native GPU experiments, training evidence or uniform-order proofs. The kernel still needs reference-uncertainty-controlled amplitude/time quadrature convergence, large-amplitude tests and full charged deployment evaluation.


## Evidence and related papers

### Recent experiment timeline

| Experiment | Evidence class / source | Verified or explicitly classified observations | Limits |
| --- | --- | --- | --- |
| E01 — Initial/light screens | native CARC CPU output plus committed cloud validation; `results/gates.json`, `results/carc-light-validation.json` | Historical G2/G4 failed; native light output reported six accepted references, zero numerical-headroom cases and five temporal-oracle advantages. No joint candidate. | Six cases reuse two states; the committed light validation is cloud CPU, not the native job. Tiny temporal oracle fits cannot rank deployed solvers. |
| E02 — First frozen A100 neural benchmark | reviewed uploaded actual CARC A100 execution; `results/carc-gpu-benchmark-review.json` | 35 GPU tests; 9 diagnostic parents and one training initialization. Reaction-clock has 0 speed wins against the best classical method; the original temporal MLP loses all 7 eligible speed comparisons against generic MLP. Clock stiff-boundary one-step RMS is 0.04872 against the 0.002 target. | 96 updates, unequal capacities, a 32-square grid with batch size 1, and three timing repeats. Reference-informed horizon choice can mask transients. |
| E03 — Three-seed neural replication | reviewed uploaded actual CARC CPU execution; separate A100 replication log in user conversation; `results/carc-neural-replication-review.json` | 386 CPU tests; 225/225 references accepted; 15 checkpoint selections, 14 trained. Clock jointly passes transient tests on 6/6/5 of 27 parents across the seeds; FNO-split passes 9/10/9. No learned family jointly passes boundary parents. Clock speed wins against the best classical method are 0/0/1. | The committed reviewed JSON records CPU execution. The user A100 log similarly reported 6/6/5 and 9/10/9, but a complete independently reviewed uploaded A100 replication archive is absent here. Three seeds share 27 parents; these are not 81 independent and identically distributed samples. |
| E04 — Training-free mechanism audit | reviewed uploaded actual CARC CPU execution; `results/carc-mechanism-audit-review.json`, `docs/MECHANISM_REVIEW.md` | 600/600 cases, 103 tests, and 40 accepted coupled references. Clock cannot generate the required cubic boundary correction; 24 remote cases have a zero receiver gate. The pair-aware oracle is best in 21/24 cases, with median normalized held-out kernel error 0.151 against scalar error 0.672. | Case-specific oracle coefficients and privileged pair/background labels make this a representation probe rather than a learned comparison. Capacity attenuates leading coefficients and does not guarantee accuracy. |
| E05 — Field-derived interaction screen | reviewed uploaded actual CARC CPU execution; `results/carc-interaction-screen-review.json`, `docs/INTERACTION_REVIEW.md` | 502 records, 188 tests, and 32 references. GL3 improves on Strang in 32/32 cases, with median L2 error ratio 0.02742; it wins 23/32 accuracy comparisons against ETDRK4. ETDRK4 is more accurate and faster in all 3 historical 2D cases. GL3 per-step cost is 5.846 times Strang and 1.789 times ETDRK4. | Single sequential, uncached timing samples. Mean error accounts for 92.54% of split squared error, but mean-only loses most ETDRK4 comparisons. Every Strang L2 error is already below 0.002, so that target gives no efficiency headroom. |
| E06 — Prepared work–precision | reviewed uploaded actual CARC CPU execution; `results/carc-work-precision-review.json`, `docs/WORK_PRECISION_REVIEW.md` | 1008 candidates, 302 tests, and 240 parity checks. There are 23 accepted conditions; one frozen reference exceeds its budget by 12.17%, excluding 56 frontiers. Warmed fastest-method counts across 184 targets are Strang 102 / ETDRK4 73 / GL3 8 / GL5 1; setup-inclusive counts are 135/40/8/1. The spectral mean identity agrees to approximately 1e-15 but yields no global wins. | The mean-only correction leaves 99.9433% of its remaining error energy in the spatial component at one step. Full GL3 late apparent order is 1.88 versus ETDRK4 3.98. Five repeats and a single setup estimate do not establish timing uncertainty. Same-grid reference uncertainty is not a continuum certificate. |
| E07 — Compact-spatial accuracy/scaling | reviewed uploaded actual CARC CPU execution; `results/carc-compact-spatial-review.json`, `docs/COMPACT_SPATIAL_REVIEW.md` | 2760 candidates, 281 tests, 166 references, and 392 parity checks. Fused GL3 is a median 1.587 times faster than original GL3, with parity error 3e-15. Input cutoffs are never the fastest complete solver; the high input modes 9/10 generate interaction mode 1, which truncation discards. K=8 high-pair error is 20.7 times GL3, and near-Nyquist error is approximately 156 times higher. Scaling Strang errors are already 7.7 times below the tightest target. | Fusion reverses its speed advantage at a 64-square grid and batch size 16. Small-grid cutoff methods cost approximately 1.89 times fused GL3. The quadratic branch remains approximately second order in time. Smooth scaling tests measure throughput without demonstrating accuracy headroom. |
| E08 — Fedora premix | uploaded actual Fedora RTX4090 run; independently rehashed now; `.runtime/review-fedora-premix-20261007/neural-review.json`, `.runtime/review-fedora-premix-20261007/numerical-review.json`, `.runtime/research-review-native-prerequisites.json` | 61 scientific file hashes verified; 434 CPU / 27 GPU tests; 240 accepted neural references. There are 15 trials, 14 trained selections, 2592 candidates, and 48 diagnostic parents. At RMS 2e-4, premix coverage is 16/12/12 versus FNO 32/24/16; classical ETDRK4 and GL3 each cover 48/48. Output projection preserves low-frequency interaction coefficients, but direct selected convolution costs approximately 2 times the full FFT output method in 1D and 10 times in 2D. | Mixing before compression improves validation but does not establish held-out dominance. Full output projection still computes full GL3. FNO capacity is unequal: 23825 versus 67137 parameters. Different fields at different grids do not isolate resolution transfer. Mean drift and missing exact null limits motivate the consistency experiment. |
| E09 — Fedora physical consistency | uploaded actual Fedora RTX4090 run; independently rehashed now; `.runtime/review-consistency-20261007/accuracy-review.json`, `.runtime/review-consistency-20261007/reporting-review.json`, `.runtime/research-review-native-prerequisites.json` | 69 scientific files verified; 304 CPU / 38 GPU tests; 456 references. There are 21 trials, 14 trained selections, 7 initializations, and 3456 candidates. Constrained moment backbones each cover 40/48 at RMS 2e-4; all neural arms cover 36/48 at maximum error 2e-4. Premix moment coverage at strict RMS 2e-6 is 22/28/12 versus FNO 24/28/12. GL3 covers 44/48 at RMS 2e-6. | The mean head adds approximately 22% same-step overhead to premix and 13.5% to FNO. Stiff premix cases require more steps. Constraints benefit both backbones. All diagnostic step queries were used during development, and grids are not paired discretizations. Seven initialization selections cannot establish learning. |
| E10 — Full Fedora research agenda | reviewed uploaded actual Fedora RTX4090 full run; `results/fedora-agenda-review.json`, `docs/AGENDA_REVIEW.md` | 8 complete, sealed stages and 341 scientific files; 222 structural cases with 34 required passes; 744 references; 69 trials with 64 trained selections and 5 initializations. All 152 GPU tests pass at each of 6 stages. None of 24 joint criteria passes. Diffusion-first orientation improves validation by a factor of 23.68. Source RMS 2e-4 coverage is 574/576 versus FNO 576/576; maximum-error 2e-5 coverage is 477/576 versus 533/576; rank-2 covers 176/192. Bare diffusion-first Strang is approximately 3.9 times faster than Source/rank-2. Discrete false accepts are 0/1179; continuum failures are 306/410 accepted predictions. | 8 controlled fields share 1 phase-seed cluster. Rank-2 has only one training seed; the cheap one-layer FNO is not confirmed. The continuum subset contains 8 parents under one physics choice. Policy costs are attributed components rather than independently timed standalone deployment. RF-optimized hyperparameters were applied to DF models. Reversed-step catastrophes remain. The inspected full bank is now development evidence. |


### Latest full native audit and provenance

The native source run is `fedora-agenda-20261007T170704067870Z`, collected as two uploaded parts and its index. Reassembly verified 58,242,523 compressed bytes, SHA-256 `edd6eb13fa841699901810880ffd39d1c0dd44ae28041676337d50ed3efe7782`, 731 members and 237,072,468 expanded bytes. All eight source/protocol/file lineages and all 341 sealed scientific-file hashes were checked. Frontier/gate recomputation had zero mismatches. The archived source commit is `31a607b`; the executable tree matches this document's baseline fingerprint `a5605ac67aac5737a51873825e2054ec997e79206133328dfdd69d17b0f6e5f3`.

The committed [full agenda review](AGENDA_REVIEW.md) and [machine-readable audit](../results/fedora-agenda-review.json) retain the detailed findings. There were 531 CPU test passes and one optional Tower-fixture skip. The same 152 mandatory GPU tests passed at six stages: 912 executions, not 912 different tests. The 744 accepted references comprised 696 discrete and 48 continuum references. All 69 trials remained visible (64 trained, 5 initialization selections); 32,640 frontier rows and 96 classical gates per norm plus 24 joint gates were independently recomputed.

Native premix and consistency artifacts were independently rehashed again for this synthesis: 61 and 69 sealed science files. Native consistency selected 14 trained and 7 initialization slots; a committed cloud replay selected 15 and 6. The native counts describe the user's experiment. CPU cloud validation cannot stand in for native GPU evidence. The companion JSON records all manifest hashes/source commits and marks each timeline evidence class.

Tower data were reviewed as reporting/provenance, not as additional scientific evidence. All eight native reports were clean at the inspected version; 356,861 report checks, 51,957 canonical metric rows, 472 paginated pages, 9,372 learning values, 1,098,824 scalar cells and 125,190 nested source pointers had no reported omissions. This document changes no Tower application behavior.

### Literature selection and access protocol

Selection favors work directly related to stiff spectral evolution, finite/variable time integration, nonlinear source interaction, operator compression, spatial transfer, uncertainty/acceptance, geometric transport and real operator cost. Fourteen work records are retained, rather than claiming fourteen independently verified peer-reviewed new publications. Search terms, attempts, primary URLs, repository revisions, source-level classifications and SHA-256 snapshots are recorded in the companion JSON.

The environment allowed author GitHub repositories/raw files but blocked direct arXiv, publisher and OpenReview reads. Full text was read only where authors legitimately supplied it; otherwise the review is limited to author metadata and released code. No inaccessible theorem or experimental table is presented as independently checked. An author-reported venue is labeled accordingly; a repository commit date is not substituted for first publication. P04/P08 have older first preprints; P13 has a repository initial commit one day outside the window; P06 has only an author-reported submitted manuscript/year. These contextual records inform experiments without being counted as confirmed in-window first publications.

| Record / title | Date / window status | What was read | Gap type / opportunity | Linked mechanisms |
| --- | --- | --- | --- | --- |
| [P01 — Stable spectral neural operator for learning stiff PDE systems from limited data](#paper-p01) | 2025-12 — month from author-supplied arXiv identifier | Author README and released SSNO implementation; full paper unavailable | project hypothesis, release evidence: Linear dissipativity alone does not prove nonlinear stability/invariant domain; compare low-call known-physics correction at matched full work | [M00](#mechanisms-m00), [M02](#mechanisms-m02), [M03](#mechanisms-m03), [M04](#mechanisms-m04), [M06](#mechanisms-m06), [M08](#mechanisms-m08), [M10](#mechanisms-m10), [M11](#mechanisms-m11), [M12](#mechanisms-m12), [M17](#mechanisms-m17), [M19](#mechanisms-m19), [M21](#mechanisms-m21), [M23](#mechanisms-m23) |
| [P02 — Conformal Uncertainty Quantification Guarantees for Neural Operators](#paper-p02) | 2026-08-28 — day verified on author-hosted PDF titlepage | Author-hosted full paper main text, guarantees, limitations and experiments verified; calibration source inspected | explicit paper limitation: No distribution-shift guarantee or target coverage conditional on every realized calibration set; explicit paper limitation: Fixed-grid coverage not automatically resolution invariant; dominating continuum score may be conservative/problem-dependent; project hypothesis: Develop sharp task-consistent spatial+temporal error score with selective/fallback total work | [M01](#mechanisms-m01), [M05](#mechanisms-m05), [M08](#mechanisms-m08), [M09](#mechanisms-m09), [M18](#mechanisms-m18) |
| [P03 — Walrus: A Cross-Domain Foundation Model for Continuum Dynamics](#paper-p03) | 2025-11 — month from author-supplied arXiv identifier | Author README and isotropic model implementation; full paper unavailable | project hypothesis, release evidence: Assess compact-kernel phase/translation robustness and adaptive compute at same physical accuracy/full work | [M07](#mechanisms-m07), [M11](#mechanisms-m11), [M15](#mechanisms-m15), [M17](#mechanisms-m17), [M19](#mechanisms-m19) |
| [P04 — Light-Weight Diffusion Multiplier and Uncertainty Quantification for Fourier Neural Operators](#paper-p04) | 2025 (NeurIPS edition) — author-reported conference edition, exact day unverified; first release 2025-08, exact day unverified; first preprint outside window | Author README/model multiplier/Bayesian decoder/fitting script; full paper unavailable | project hypothesis, release evidence: Parameter uncertainty does not automatically bound spatial bias or selected-error risk; compare this very cheap multiplier as established baseline | [M06](#mechanisms-m06), [M12](#mechanisms-m12), [M15](#mechanisms-m15) |
| [P05 — Neural operators approximate strongly continuous convex monotone semigroups](#paper-p05) | 2026-09 — month from author-supplied arXiv identifier | Author README/shared neural-operator source; full theorem text unavailable | project hypothesis, theoretical applicability unverified: A semigroup penalty alone does not establish truth or finite-step accuracy; verify logistic assumptions and quantitative cost/error | [M02](#mechanisms-m02), [M06](#mechanisms-m06), [M10](#mechanisms-m10), [M12](#mechanisms-m12), [M21](#mechanisms-m21) |
| [P06 — Calibrate on the Physics: Stress-Stratified Conformal Prediction for Reliable Neural-Operator Stress Surrogates](#paper-p06) | 2026 — author citation year, exact publication date unknown | Author README/CITATION.cff/evaluation harness | explicit author release statement: Acceptance/defer deployment analysis excluded from submitted manuscript; measure actual accepted-error risk and full costs in TDN; project hypothesis: Average calibration can hide worst stiffness/spectrum/schedule regimes | [M07](#mechanisms-m07), [M09](#mechanisms-m09), [M17](#mechanisms-m17), [M18](#mechanisms-m18) |
| [P07 — TANTE: Time-adaptive operator learning via neural Taylor expansion](#paper-p07) | 2026 — author-reported journal citation year; exact day unknown | Author README bibliographic metadata and pinned released models/tante.py/r_evaler.py; full article unavailable | verified released-code limitation: Inspected path predicts integer frame-multiple horizons; arbitrary unequal-step calibrated acceptance not established by inspected release; project hypothesis: Stiff semigroup-filtered arbitrary h with independent truth and rejection/full-cost audit | [M02](#mechanisms-m02), [M07](#mechanisms-m07), [M08](#mechanisms-m08), [M10](#mechanisms-m10), [M11](#mechanisms-m11), [M20](#mechanisms-m20), [M23](#mechanisms-m23) |
| [P08 — TI-DeepONet: Learnable time integration for stable long-term extrapolation](#paper-p08) | 2026 (journal edition) — author-reported journal citation year; exact day unknown; first release 2025-05, exact day unverified; first preprint outside window | Author README/slides and pinned Burgers TI(L) notebook; full journal article unavailable | verified released-code implementation fact: Softmax normalization does not impose higher RK order conditions; fixed classical stages plus variable final weights cannot retain generic third order unless weights become classic; project hypothesis: Order-controlled defect correction with stiff finite-h and generator truth tests | [M02](#mechanisms-m02), [M08](#mechanisms-m08), [M10](#mechanisms-m10), [M20](#mechanisms-m20) |
| [P09 — Fourier Neural Operators Explained: A Practical Perspective](#paper-p09) | 2025-12 — month from official library author-supplied arXiv identifier | Official library README verifies bibliographic metadata; paper text unavailable | project baseline requirement, not paper gap: Use optimized official-style FNO/Tucker controls before efficiency claim | [M00](#mechanisms-m00), [M16](#mechanisms-m16), [M22](#mechanisms-m22) |
| [P10 — Hyperbolic Neural Operator](#paper-p10) | venue: ICML 2026 / PMLR 306, author paper and BibTeX; first preprint: not verified; public code release: 2026-05-13; full text release: 2026-09-07; latest checked repository commit date: 2026-09-29; first publication in window: unknown; 2026 venue and releases are in window | complete author-posted text, including appendices, README, structured metadata and commit history; publisher record not independently checked | Explicit author limits: Section 4.2: HNO does not build an explicit FMM tree, compute multipole expansions, or provide an error-controlled truncation rule.; Appendix J.6: theorem does not apply to arbitrary resampling or severe subsampling violating quadrature accuracy; aggressive remeshing and distribution-shifted point sampling left as future work.. An error-budgeted near/far representation of the physical splitting residual, retaining a local full-resolution source path, could couple scalable compression to a measured residual tolerance. Separate fixed-grid, continuous-field refinement and irregular-sampling targets. | [M01](#mechanisms-m01), [M03](#mechanisms-m03), [M05](#mechanisms-m05), [M14](#mechanisms-m14), [M19](#mechanisms-m19) |
| [P11 — Tucker-FNO: Tensor Tucker-Fourier Neural Operator and its Universal Approximation Theory](#paper-p11) | venue: ICLR 2026, author README; first preprint: not verified; public repository initial commit: 2026-02-04; public implementation release: 2026-02-06; first publication in window: unknown; 2026 venue and releases are in window | official author README and released plasticity code; full paper and universal approximation proof not read | Separate coordinate tensor compression, channel-weight tensorization and bilinear source-kernel compression, then test which actually preserves phase-dependent interactions at fixed measured cost. Use physical-frequency axis encoders or quadrature-weighted pooling if transfer across grids is required. | [M03](#mechanisms-m03), [M04](#mechanisms-m04), [M13](#mechanisms-m13), [M16](#mechanisms-m16) |
| [P12 — Fractional is Better: Learnable Derivative Orders in Neural Operator Learning](#paper-p12) | venue: ICML 2026, author README; first preprint: not verified; README arXiv badge is a placeholder; public repository initial commit: 2026-05-28; public notebook release: 2026-05-29; first publication in window: unknown; 2026 venue and releases are in window | author README and full first Burgers notebook code; paper proof and claimed cross-backbone benchmark tables not independently read | Use dimensionally consistent fractional or rational spectral features of the physical source, with multiplier zero at k=0 and actual continuum/discrete eigenvalue choice recorded. Preserve exact physical nulls before feature transport and test noise sensitivity separately from deterministic teacher error. | [M01](#mechanisms-m01), [M04](#mechanisms-m04), [M05](#mechanisms-m05), [M13](#mechanisms-m13) |
| [P13 — Solving Partial Differential Equations via Radon Neural Operator](#paper-p13) | venue: NeurIPS 2025, author README; first preprint: not verified; public repository initial commit: 2025-10-06 (one day outside the rolling window); public implementation release: 2026-03-18; first publication in window: not established; include as recent venue/version context, not as newly discovered in-window work | author README, repository layout and commit history only; full paper/discretization proof unread | A directional or angular mode-pair representation could help future anisotropic or irregular-domain cases, but first test angular quadrature, orientation transfer and target consistency. Keep current periodic Fourier source pair representation as the cheap direct baseline. | [M14](#mechanisms-m14), [M19](#mechanisms-m19) |
| [P14 — EqGINO: Equivariant Geometry-Informed Fourier Neural Operator for 3D PDEs](#paper-p14) | venue: ICML 2026; ICLR 2026 AI&PDE Workshop, author README; linked preprint: arXiv:2606.03260 (June 2026 ID; page/first submitted day not verified); implementation release: v1.1.0 2026-06-21, author release notes; first publication in window: linked June 2026 preprint and release within window; exact first-day unverified | author README and detailed dated release notes; paper theory unread; author benchmark claims not independently reproduced | Audit optimized implementations and inference costs before concluding an architecture advantage. Implement source/pair FFT reuse and avoid per-stage CUDA synchronization or allocation, then remeasure identical predictions against equally optimized one-layer and deeper FNO/classical controls. | [M00](#mechanisms-m00), [M14](#mechanisms-m14), [M15](#mechanisms-m15), [M16](#mechanisms-m16), [M19](#mechanisms-m19), [M22](#mechanisms-m22) |

<a id="paper-p01"></a>

### P01 — Stable spectral neural operator for learning stiff PDE systems from limited data

**Authors:** Rui Zhang, Han Wan, Yang Liu, Hao Sun

**Date and status:** 2025-12 — month from author-supplied arXiv identifier. preprint

**Verification level:** Author README and released SSNO implementation; full paper unavailable

**Primary repository:** [https://github.com/optray/SSNO](https://github.com/optray/SSNO)

**Author-supplied arXiv identifier:** [2512.11686](https://arxiv.org/abs/2512.11686); direct archive page was not accessible.

**Verified scope:**

- Equation-free spectral linear/nonlinear split with integrating-factor RK, explicit dt argument
- Released linear lambda=-softplus(a)+i*b
- Released step evaluates nonlinear function four times; quadratic product is anti-aliased
- 2–5 trajectory OOD/accuracy gains are author README claims, not reproduced

**Targeted gap/opportunity and its evidence type:** project hypothesis, release evidence: Linear dissipativity alone does not prove nonlinear stability/invariant domain; compare low-call known-physics correction at matched full work

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/optray/SSNO/49a0a487a39bc090da9109b22c7d0ddcb4b7cce6/README.md](https://raw.githubusercontent.com/optray/SSNO/49a0a487a39bc090da9109b22c7d0ddcb4b7cce6/README.md) — SHA-256 `c8437a8ca249a3fa06fa16e0c78f08c2e0d1a7355c9e5b7f491bae1ee17da4b4`
- [https://github.com/optray/SSNO/blob/49a0a487a39bc090da9109b22c7d0ddcb4b7cce6/kuramoto_sivashinsky/neural_operator/model/ssno.py](https://github.com/optray/SSNO/blob/49a0a487a39bc090da9109b22c7d0ddcb4b7cce6/kuramoto_sivashinsky/neural_operator/model/ssno.py) — release source inspected; detailed inventory in JSON

<a id="paper-p02"></a>

### P02 — Conformal Uncertainty Quantification Guarantees for Neural Operators

**Authors:** Tom Stent, Nicolas Boullé

**Date and status:** 2026-08-28 — day verified on author-hosted PDF titlepage. preprint

**Verification level:** Author-hosted full paper main text, guarantees, limitations and experiments verified; calibration source inspected

**Primary repository:** [https://github.com/tom-stent/conformal-neural-operator](https://github.com/tom-stent/conformal-neural-operator)

**Author-supplied arXiv identifier:** [2608.28515v1](https://arxiv.org/abs/2608.28515v1); direct archive page was not accessible.

**Verified scope:**

- Function-level spatial quantile conformal score under exchangeability
- Fixed-grid and ideal continuum marginal guarantees Theorems2.3/2.5
- Conditional coverage across calibration sets Beta(k,n+1-k) under i.i.d. atomless scores
- Remark2.6 dominating-score framework; released continuum_calibrate adds spatial/data/Lipschitz terms
- Figure4: Darcy zero-shot fine-grid transfer coverage zero in the evaluated experiment

**Targeted gap/opportunity and its evidence type:** explicit paper limitation: No distribution-shift guarantee or target coverage conditional on every realized calibration set; explicit paper limitation: Fixed-grid coverage not automatically resolution invariant; dominating continuum score may be conservative/problem-dependent; project hypothesis: Develop sharp task-consistent spatial+temporal error score with selective/fallback total work

**Existing contribution that must not be claimed as our novelty:** Continuum conformal bridging already exists; proposed contribution cannot claim its invention

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/tom-stent/conformal-neural-operator/5508719638527e81f637e9715234b9beb278e272/README.md](https://raw.githubusercontent.com/tom-stent/conformal-neural-operator/5508719638527e81f637e9715234b9beb278e272/README.md) — SHA-256 `959bdcfb30636e787d86145176eccbaa61718cbb75c681374be5c1ef6c14d2b9`
- [https://raw.githubusercontent.com/tom-stent/conformal-neural-operator/5508719638527e81f637e9715234b9beb278e272/report/2608.28515v1.pdf](https://raw.githubusercontent.com/tom-stent/conformal-neural-operator/5508719638527e81f637e9715234b9beb278e272/report/2608.28515v1.pdf) — SHA-256 `3f2912d8e26f5c03b1bf3ea7046833ca388adceede642167741ffd1c1e42dade`

<a id="paper-p03"></a>

### P03 — Walrus: A Cross-Domain Foundation Model for Continuum Dynamics

**Authors:** Michael McCabe et al.

**Date and status:** 2025-11 — month from author-supplied arXiv identifier. preprint

**Verification level:** Author README and isotropic model implementation; full paper unavailable

**Primary repository:** [https://github.com/PolymathicAI/walrus](https://github.com/PolymathicAI/walrus)

**Author-supplied arXiv identifier:** [2511.15684](https://arxiv.org/abs/2511.15684); direct archive page was not accessible.

**Verified scope:**

- Factorized space-time attention with encoder-processor-decoder
- Patch jittering suppresses repeated downsampling error patterns
- Stride modulation adjusts internal resolution/compute
- 19 physical scenarios/63 variables, 2D/3D is author README claim

**Targeted gap/opportunity and its evidence type:** project hypothesis, release evidence: Assess compact-kernel phase/translation robustness and adaptive compute at same physical accuracy/full work

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/PolymathicAI/walrus/73f2cf6b4c5871c8f5434802725b55666c66b5ab/README.md](https://raw.githubusercontent.com/PolymathicAI/walrus/73f2cf6b4c5871c8f5434802725b55666c66b5ab/README.md) — SHA-256 `ff93f9009112fdeeef101897d6e213dc6f60c9eea04d3433269eb22f6107d2b0`
- [https://github.com/PolymathicAI/walrus/blob/73f2cf6b4c5871c8f5434802725b55666c66b5ab/walrus/models/isotropic_model.py](https://github.com/PolymathicAI/walrus/blob/73f2cf6b4c5871c8f5434802725b55666c66b5ab/walrus/models/isotropic_model.py) — release source inspected; detailed inventory in JSON

<a id="paper-p04"></a>

### P04 — Light-Weight Diffusion Multiplier and Uncertainty Quantification for Fourier Neural Operators

**Authors:** Albert Matveev, Sanmitra Ghosh, Aamal Hussain, James-Michael Leahy, Michalis Michaelides

**Date and status:** 2025 (NeurIPS edition) — author-reported conference edition, exact day unverified; first release 2025-08, exact day unverified; first preprint outside window. author-reported NeurIPS2025 spotlight; first preprint outside window

**Verification level:** Author README/model multiplier/Bayesian decoder/fitting script; full paper unavailable

**Primary repository:** [https://github.com/PhysicsXLtd/DINOZAUR](https://github.com/PhysicsXLtd/DINOZAUR)

**Author-supplied arXiv identifier:** [2508.00643](https://arxiv.org/abs/2508.00643); direct archive page was not accessible.

**Verified scope:**

- Channelwise diffusion time constrained positive, exp(-eigenvalue*time)
- Bayesian diffusion multiplier and Laplace fitting workflow
- In inspected deterministic multiplier one time parameter per input channel

**Targeted gap/opportunity and its evidence type:** project hypothesis, release evidence: Parameter uncertainty does not automatically bound spatial bias or selected-error risk; compare this very cheap multiplier as established baseline

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/PhysicsXLtd/DINOZAUR/8ce47fd8c19c791c1f892751adcca8ed1c56c7b6/README.md](https://raw.githubusercontent.com/PhysicsXLtd/DINOZAUR/8ce47fd8c19c791c1f892751adcca8ed1c56c7b6/README.md) — SHA-256 `b004150f78d5cf51941b4d8ba1c094c83da397dd25751f50e4b19a773ef5e11c`
- [https://github.com/PhysicsXLtd/DINOZAUR/blob/8ce47fd8c19c791c1f892751adcca8ed1c56c7b6/src/dinozaur/models/layers/multipliers.py](https://github.com/PhysicsXLtd/DINOZAUR/blob/8ce47fd8c19c791c1f892751adcca8ed1c56c7b6/src/dinozaur/models/layers/multipliers.py) — release source inspected; detailed inventory in JSON

<a id="paper-p05"></a>

### P05 — Neural operators approximate strongly continuous convex monotone semigroups

**Authors:** Jonas Blessing, Philipp Schmocker, Alessandro Sgarabottolo

**Date and status:** 2026-09 — month from author-supplied arXiv identifier. preprint

**Verification level:** Author README/shared neural-operator source; full theorem text unavailable

**Primary repository:** [https://github.com/sgarale/chernoff_neural](https://github.com/sgarale/chernoff_neural)

**Author-supplied arXiv identifier:** [2609.02727](https://arxiv.org/abs/2609.02727); direct archive page was not accessible.

**Verified scope:**

- Learned Chernoff one-step operators composed toward horizons
- Envelope readout includes max of neuron features
- Semilinear PDE splitting and stochastic control/uncertainty examples

**Targeted gap/opportunity and its evidence type:** project hypothesis, theoretical applicability unverified: A semigroup penalty alone does not establish truth or finite-step accuracy; verify logistic assumptions and quantitative cost/error

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/sgarale/chernoff_neural/ac12e7b8db54b7cbc970babd88c73848e571ee32/README.md](https://raw.githubusercontent.com/sgarale/chernoff_neural/ac12e7b8db54b7cbc970babd88c73848e571ee32/README.md) — SHA-256 `c5ed366264af79802323a0054ed8c99d71d20ee5dad260aee5acc2671642c54c`
- [https://github.com/sgarale/chernoff_neural/blob/ac12e7b8db54b7cbc970babd88c73848e571ee32/neural_operators.py](https://github.com/sgarale/chernoff_neural/blob/ac12e7b8db54b7cbc970babd88c73848e571ee32/neural_operators.py) — release source inspected; detailed inventory in JSON

<a id="paper-p06"></a>

### P06 — Calibrate on the Physics: Stress-Stratified Conformal Prediction for Reliable Neural-Operator Stress Surrogates

**Authors:** Raqeeb M. Al-Naqeeb, Bader Alserhan, Mohammed Muhi Faris

**Date and status:** 2026 — author citation year, exact publication date unknown. Submitted manuscript, author reports IJSS submission; no acceptance verified

**Verification level:** Author README/CITATION.cff/evaluation harness

**Primary repository:** [https://github.com/global-prog/neural-operator-stress-reliability](https://github.com/global-prog/neural-operator-stress-reliability)

**Verified scope:**

- Physics-stratified conformal calibration and resolution transfer release
- FNO/DeepONet with analytic/FEM reference data release
- README explicitly removes accept/defer analysis from submitted paper; file retained for completeness
- Reported stress concentration coverage/uncertainty correlations are author claims, not reproduced

**Targeted gap/opportunity and its evidence type:** explicit author release statement: Acceptance/defer deployment analysis excluded from submitted manuscript; measure actual accepted-error risk and full costs in TDN; project hypothesis: Average calibration can hide worst stiffness/spectrum/schedule regimes

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/global-prog/neural-operator-stress-reliability/8f81d69204838a9d87f9f1f9af7fb0222329723d/README.md](https://raw.githubusercontent.com/global-prog/neural-operator-stress-reliability/8f81d69204838a9d87f9f1f9af7fb0222329723d/README.md) — SHA-256 `26c583f0dcfc7d43591110a29ee43b719f6d73c1bd64289b690327208747c3c4`
- [https://github.com/global-prog/neural-operator-stress-reliability/blob/8f81d69204838a9d87f9f1f9af7fb0222329723d/eval_harness.py](https://github.com/global-prog/neural-operator-stress-reliability/blob/8f81d69204838a9d87f9f1f9af7fb0222329723d/eval_harness.py) — release source inspected; detailed inventory in JSON

<a id="paper-p07"></a>

### P07 — TANTE: Time-adaptive operator learning via neural Taylor expansion

**Authors:** Zhikai Wu, Sifan Wang, Shiyang Zhang, Sizhuang He, Min Zhu, Anran Jiao, Lu Lu, David van Dijk

**Date and status:** 2026 — author-reported journal citation year; exact day unknown. Author-reported JCP562(2026)115041

**Verification level:** Author README bibliographic metadata and pinned released models/tante.py/r_evaler.py; full article unavailable

**Primary repository:** [https://github.com/zwu88/TANTE](https://github.com/zwu88/TANTE)

**DOI:** [10.1016/j.jcp.2026.115041](https://doi.org/10.1016/j.jcp.2026.115041).

**Verified scope:**

- Neural Taylor derivatives with learned validity horizon/forecast length
- Code sums derivatives*(i*frame_interval)^order/factorial(order)
- Released path uses floor(R_t[0]) to determine batch output length; line labeled temporary/TODO
- Released evaluation tests Well dataset rollouts; reported paper metrics not reproduced

**Targeted gap/opportunity and its evidence type:** verified released-code limitation: Inspected path predicts integer frame-multiple horizons; arbitrary unequal-step calibrated acceptance not established by inspected release; project hypothesis: Stiff semigroup-filtered arbitrary h with independent truth and rejection/full-cost audit

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/zwu88/TANTE/d271d9aaf3c86343bc9b0671df3bf34ce37d753c/models/tante.py](https://raw.githubusercontent.com/zwu88/TANTE/d271d9aaf3c86343bc9b0671df3bf34ce37d753c/models/tante.py) — SHA-256 `ee3092517641b030652dde80e18b2fce5e1f4143833cdaa1cc5fda2c832e03f2`
- [https://raw.githubusercontent.com/zwu88/TANTE/d271d9aaf3c86343bc9b0671df3bf34ce37d753c/trainer/r_evaler.py](https://raw.githubusercontent.com/zwu88/TANTE/d271d9aaf3c86343bc9b0671df3bf34ce37d753c/trainer/r_evaler.py) — SHA-256 `de103e9d0499489d659ab47d7fc5f3028a93ad8496d4504abd71f6690549605b`
- [https://raw.githubusercontent.com/zwu88/TANTE/d271d9aaf3c86343bc9b0671df3bf34ce37d753c/README.md](https://raw.githubusercontent.com/zwu88/TANTE/d271d9aaf3c86343bc9b0671df3bf34ce37d753c/README.md) — SHA-256 `e71c20a885b7093bda61e3c879a6ac2ed19d9666b1e89a0802d4c90d72beb7a8`

<a id="paper-p08"></a>

### P08 — TI-DeepONet: Learnable time integration for stable long-term extrapolation

**Authors:** Dibyajyoti Nayak, Somdatta Goswami

**Date and status:** 2026 (journal edition) — author-reported journal citation year; exact day unknown; first release 2025-05, exact day unverified; first preprint outside window. Author-reported CMAME456(2026)118960; first preprint outside window

**Verification level:** Author README/slides and pinned Burgers TI(L) notebook; full journal article unavailable

**Primary repository:** [https://github.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation](https://github.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation)

**DOI:** [10.1016/j.cma.2026.118960](https://doi.org/10.1016/j.cma.2026.118960).

**Author-supplied arXiv identifier:** [2505.17341](https://arxiv.org/abs/2505.17341); direct archive page was not accessible.

**Verified scope:**

- Learns instantaneous time derivative, integrated with classical numerical schemes
- Learnable final RK4 slope coefficients are softmax-positive and sum to one
- Released fixed classicRK4 stages with state-dependent learned final b weights
- Numerical headline extrapolation improvement claims are author claims, not reproduced

**Targeted gap/opportunity and its evidence type:** verified released-code implementation fact: Softmax normalization does not impose higher RK order conditions; fixed classical stages plus variable final weights cannot retain generic third order unless weights become classic; project hypothesis: Order-controlled defect correction with stiff finite-h and generator truth tests

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation/ddea09251940ebf8ce9721fef0c47f14e6ec5a2b/Slides_TI-DeepONet_updated.pdf](https://raw.githubusercontent.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation/ddea09251940ebf8ce9721fef0c47f14e6ec5a2b/Slides_TI-DeepONet_updated.pdf) — SHA-256 `d8909b671155cd5de07e5e61146f4d5dd3ef75876cad0e3404bf5fc7b5202e6e`
- [https://raw.githubusercontent.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation/ddea09251940ebf8ce9721fef0c47f14e6ec5a2b/README.md](https://raw.githubusercontent.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation/ddea09251940ebf8ce9721fef0c47f14e6ec5a2b/README.md) — SHA-256 `5c490275f29d73ad6f0b84560db7a88edd34eb37001acc4bd139a7370242eeb4`
- [https://raw.githubusercontent.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation/ddea09251940ebf8ce9721fef0c47f14e6ec5a2b/Codes/1D_Burgers/TI%28L%29-DeepONet.ipynb](https://raw.githubusercontent.com/Centrum-IntelliPhysics/TI-DeepONet-for-Stable-Long-Term-Extrapolation/ddea09251940ebf8ce9721fef0c47f14e6ec5a2b/Codes/1D_Burgers/TI%28L%29-DeepONet.ipynb) — release source inspected; detailed inventory in JSON

<a id="paper-p09"></a>

### P09 — Fourier Neural Operators Explained: A Practical Perspective

**Authors:** Valentin Duruisseaux, Jean Kossaifi, Anima Anandkumar

**Date and status:** 2025-12 — month from official library author-supplied arXiv identifier. preprint, tooling/background reference

**Verification level:** Official library README verifies bibliographic metadata; paper text unavailable

**Primary repository:** [https://github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)

**Author-supplied arXiv identifier:** [2512.01421](https://arxiv.org/abs/2512.01421); direct archive page was not accessible.

**Verified scope:**

- Official NeuralOperator bibliography verifies title/authors/eprint
- Library README describes direct Tucker-factor contraction

**Targeted gap/opportunity and its evidence type:** project baseline requirement, not paper gap: Use optimized official-style FNO/Tucker controls before efficiency claim

**Primary inspected snapshots:**

- [https://raw.githubusercontent.com/neuraloperator/neuraloperator/main/README.rst](https://raw.githubusercontent.com/neuraloperator/neuraloperator/main/README.rst) — SHA-256 `b05da95483cb73cc965b2d3ee793d7d9de678cac8e1a086fa0c3fa6cea1012c4`

<a id="paper-p10"></a>

### P10 — Hyperbolic Neural Operator

**Authors:** Jieyuan Pei, Zhuoxuan Li, Wei Li, Haobo Zhang, Jiawei Jiang, Jianwei Zheng

**Date and status:** venue: ICML 2026 / PMLR 306, author paper and BibTeX; first preprint: not verified; public code release: 2026-05-13; full text release: 2026-09-07; latest checked repository commit date: 2026-09-29; first publication in window: unknown; 2026 venue and releases are in window. Venue metadata is author-reported; see date evidence.

**Verification level:** complete author-posted text, including appendices, README, structured metadata and commit history; publisher record not independently checked

**Primary repository:** [https://github.com/GuobaPei/Hyperbolic-Neural-Operator](https://github.com/GuobaPei/Hyperbolic-Neural-Operator)

**Verified scope:**

- Lorentz-hyperboloid distance Gibbs kernel and learned near/far latent routing
- Weighted softmax Nyström quadrature
- Point-cloud representative tokens reduce interaction cost from N^2 to NM+M^2; fixed-size grid patch tokens still yield quadratic token mixing
- Matched routing ablations and forward/train/VRAM costs on Darcy; additional representative AirfRANS density scan
- Appendix G fixed-kernel consistency requires bounded Lipschitz query/key/input, positive quadrature and temperature bounded away from zero

**Targeted gap/opportunity and its evidence type:** Explicit author limits: Section 4.2: HNO does not build an explicit FMM tree, compute multipole expansions, or provide an error-controlled truncation rule.; Appendix J.6: theorem does not apply to arbitrary resampling or severe subsampling violating quadrature accuracy; aggressive remeshing and distribution-shifted point sampling left as future work.. An error-budgeted near/far representation of the physical splitting residual, retaining a local full-resolution source path, could couple scalable compression to a measured residual tolerance. Separate fixed-grid, continuous-field refinement and irregular-sampling targets.

**Required mathematical extension:** Normalized weighted kernel Kq; radius/temperature-dependent Lipschitz constants; quadrature and compression remainder bounds; exact q=0 implies Kq=0; discrete-to-continuum error separated from temporal residual; row-stochastic kernel does not by itself enforce the logistic mean evolution.

**Additional primary sources:**

- [https://raw.githubusercontent.com/GuobaPei/Hyperbolic-Neural-Operator/c7a9e8128e5cb6507f5e0b818141640f465a221c/docs/paper.txt](https://raw.githubusercontent.com/GuobaPei/Hyperbolic-Neural-Operator/c7a9e8128e5cb6507f5e0b818141640f465a221c/docs/paper.txt)
- [https://openreview.net/forum?id=CUQwYTTNu8](https://openreview.net/forum?id=CUQwYTTNu8)
- [https://icml.cc/virtual/2026/poster/65554](https://icml.cc/virtual/2026/poster/65554)

Local snapshot SHA-256 values are retained in the JSON source inventory; they identify the exact inspected bytes.

<a id="paper-p11"></a>

### P11 — Tucker-FNO: Tensor Tucker-Fourier Neural Operator and its Universal Approximation Theory

**Authors:** Full author list not verified; the GuanchengZhou repository identifies itself as official code. Do not infer the complete paper author list from the account name.

**Date and status:** venue: ICLR 2026, author README; first preprint: not verified; public repository initial commit: 2026-02-04; public implementation release: 2026-02-06; first publication in window: unknown; 2026 venue and releases are in window. Venue metadata is author-reported; see date evidence.

**Verification level:** official author README and released plasticity code; full paper and universal approximation proof not read

**Primary repository:** [https://github.com/GuanchengZhou/Tucker-FNO](https://github.com/GuanchengZhou/Tucker-FNO)

**Verified scope:**

- Separate signal denoising/inpainting and plasticity PDE code
- Released FNO3d constructs axis factors with one-dimensional spectral blocks then reconstructs with a trainable Tucker core
- Axis-collapsing Conv1d input channel counts explicitly depend on s1, s2, s3; identical weights cannot directly accept arbitrary grid sizes through these layers

**Targeted gap/opportunity and its evidence type:** Separate coordinate tensor compression, channel-weight tensorization and bilinear source-kernel compression, then test which actually preserves phase-dependent interactions at fixed measured cost. Use physical-frequency axis encoders or quadrature-weighted pooling if transfer across grids is required.

**Scope limits:** The grid-shaped implementation finding does not disprove a continuum approximation theorem.; Output-coordinate Tucker rank is not our bilinear mode-pair kernel rank.; README does not provide paper theorem assumptions or complete bibliography.

**Required mathematical extension:** Coordinate Tucker output U(x,y,t)=sum G_ijk a_i(x)b_j(y)c_k(t), versus bilinear Fourier residual B(u)_k=sum_{p+q=k} K(p,q) u_p u_q. These compress different tensors. Use singular-value tails and physical spectral weights to bound the kernel residual; preserve K(p,q)=K(q,p) and reality relations.

**Additional primary sources:**

- [https://raw.githubusercontent.com/GuanchengZhou/Tucker-FNO/10bc429cdada1121fe71ea11b2eb182cd668810b/PDE/Tucker_FNO_plas.py](https://raw.githubusercontent.com/GuanchengZhou/Tucker-FNO/10bc429cdada1121fe71ea11b2eb182cd668810b/PDE/Tucker_FNO_plas.py)

Local snapshot SHA-256 values are retained in the JSON source inventory; they identify the exact inspected bytes.

<a id="paper-p12"></a>

### P12 — Fractional is Better: Learnable Derivative Orders in Neural Operator Learning

**Authors:** Fares B. Mehouachi, Saif Eddin Jabari

**Date and status:** venue: ICML 2026, author README; first preprint: not verified; README arXiv badge is a placeholder; public repository initial commit: 2026-05-28; public notebook release: 2026-05-29; first publication in window: unknown; 2026 venue and releases are in window. Venue metadata is author-reported; see date evidence.

**Verification level:** author README and full first Burgers notebook code; paper proof and claimed cross-backbone benchmark tables not independently read

**Primary repository:** [https://github.com/FaresBMehouachi/delNO](https://github.com/FaresBMehouachi/delNO)

**Verified scope:**

- Learnable fractional derivative feature orders and scales augment unchanged backbones
- Raw Grünwald-Letnikov convolution uses a finite K<=16 stencil
- Notebook full feature multiplies raw stencil by h_eff^(-beta), with learned h_eff initialized to 2/modes rather than actual grid spacing
- Author README reports beta*<PDE order under its spectral bias/variance model; no universal theorem claim is made here

**Targeted gap/opportunity and its evidence type:** Use dimensionally consistent fractional or rational spectral features of the physical source, with multiplier zero at k=0 and actual continuum/discrete eigenvalue choice recorded. Preserve exact physical nulls before feature transport and test noise sensitivity separately from deterministic teacher error.

**Released-code deductions:** Holding learned h_eff fixed as actual grid spacing changes does not by itself enforce physical-unit derivative equivalence.; For a noninteger beta, a finite truncated GL coefficient sum need not vanish, so the raw feature can have constant leakage. This is a feature-level deduction; it does not imply all delNO models must violate a PDE constraint.

**Required mathematical extension:** GL D_beta,dx u=dx^(-beta) sum_j (-1)^j binomial(beta,j)u(x-j dx). A real periodic spatial alternative uses |xi|^beta or (-lambda_k/kappa)^(beta/2) for kappa>0, with actual physical wavevector/units and a declared kappa=0 convention. Our lambda contains kappa and has units 1/time; (-t_ref*lambda)^(beta/2) is instead a dimensionless diffusion-rate feature. Derivative order and scaling must be separated. For beta>0, k=0 is explicitly zero; at beta=0 use a declared identity or mean-subtracted identity convention.

**Additional primary sources:**

- [https://openreview.net/forum?id=V4FDM692AZ](https://openreview.net/forum?id=V4FDM692AZ)
- [https://raw.githubusercontent.com/FaresBMehouachi/delNO/871fec3a7ae7ae1955c421e66cbe1ac1ae70535a/notebooks/Notebook1_del_NO.ipynb](https://raw.githubusercontent.com/FaresBMehouachi/delNO/871fec3a7ae7ae1955c421e66cbe1ac1ae70535a/notebooks/Notebook1_del_NO.ipynb)

Local snapshot SHA-256 values are retained in the JSON source inventory; they identify the exact inspected bytes.

<a id="paper-p13"></a>

### P13 — Solving Partial Differential Equations via Radon Neural Operator

**Authors:** Wenbin Lu, Yihan Chen, Junnan Xu, Wei Li, Junwei Zhu, Jianwei Zheng

**Date and status:** venue: NeurIPS 2025, author README; first preprint: not verified; public repository initial commit: 2025-10-06 (one day outside the rolling window); public implementation release: 2026-03-18; first publication in window: not established; include as recent venue/version context, not as newly discovered in-window work. Venue metadata is author-reported; see date evidence.

**Verification level:** author README, repository layout and commit history only; full paper/discretization proof unread

**Primary repository:** [https://github.com/wenbin-lu/Radon-Neural-Operator](https://github.com/wenbin-lu/Radon-Neural-Operator)

**Verified scope:**

- Radon forward/inverse projections, sinogram convolutions, angle-aware reweighting and physics attention
- README lists Darcy, Navier-Stokes, Airfoil, Pipe, Plasticity and Allen-Cahn data
- README claims theoretical discretization guarantees; assumptions not verified here

**Targeted gap/opportunity and its evidence type:** A directional or angular mode-pair representation could help future anisotropic or irregular-domain cases, but first test angular quadrature, orientation transfer and target consistency. Keep current periodic Fourier source pair representation as the cheap direct baseline.

**Scope limits:** Do not describe 'no aliasing guarantee', 'no temporal tests' or 'no cost study' as proven paper omissions based on the README.; Radon transforms of compact Euclidean fields do not automatically preserve a periodic torus target without a declared extension/convention.

**Required mathematical extension:** Fourier slice theorem F_s(Ru)(omega,theta)=u_hat(omega theta); finite angle and detector quadrature errors; inverse-filter regularization; boundary/periodic extension; rotated tensor/eigenvalue symmetries and invariant physical source.

**Additional primary sources:**

- [https://github.com/wenbin-lu/Radon-Neural-Operator/commit/c07523208cbc799e5d8219ca2ed472d8bbde1d45](https://github.com/wenbin-lu/Radon-Neural-Operator/commit/c07523208cbc799e5d8219ca2ed472d8bbde1d45)

Local snapshot SHA-256 values are retained in the JSON source inventory; they identify the exact inspected bytes.

<a id="paper-p14"></a>

### P14 — EqGINO: Equivariant Geometry-Informed Fourier Neural Operator for 3D PDEs

**Authors:** Complete author list not verified in accessed README; repository identifies itself as official code.

**Date and status:** venue: ICML 2026; ICLR 2026 AI&PDE Workshop, author README; linked preprint: arXiv:2606.03260 (June 2026 ID; page/first submitted day not verified); implementation release: v1.1.0 2026-06-21, author release notes; first publication in window: linked June 2026 preprint and release within window; exact first-day unverified. Venue metadata is author-reported; see date evidence.

**Verification level:** author README and detailed dated release notes; paper theory unread; author benchmark claims not independently reproduced

**Primary repository:** [https://github.com/sung-won-kim/EqGINO](https://github.com/sung-won-kim/EqGINO)

**Verified scope:**

- Equivariant GINO on 3D AhmedBody and ShapeNetCar CFD
- Author v1.1.0 A/B: unchanged model, sample and weights on L40S, 70,661 nodes; 2.87x inference speed and 1.79x less peak memory
- Optimization includes avoiding dense cdist, query-row chunking/checkpointing, FFT full-mode fast path, on-device metrics, cached latent grid
- Release notes distinguish bit-exact FFT/chunk results from small boundary-edge numerical differences in the neighbor backend

**Targeted gap/opportunity and its evidence type:** Audit optimized implementations and inference costs before concluding an architecture advantage. Implement source/pair FFT reuse and avoid per-stage CUDA synchronization or allocation, then remeasure identical predictions against equally optimized one-layer and deeper FNO/classical controls.

**Required mathematical extension:** Equivariance commuting diagrams G(T_g u)=T_g G(u), quadrature measures and discrete group limitations; numerical equivalence of reused transforms; cost counts include memory traffic/synchronization/launches, not only FLOPs/parameters.

**Additional primary sources:**

- [https://github.com/sung-won-kim/EqGINO/blob/main/RELEASES.md](https://github.com/sung-won-kim/EqGINO/blob/main/RELEASES.md)
- [https://arxiv.org/abs/2606.03260](https://arxiv.org/abs/2606.03260)

Local snapshot SHA-256 values are retained in the JSON source inventory; they identify the exact inspected bytes.


### Reproduction limits and next document update

This is a research/design deliverable. It adds no trained result, changes no executable solver/configuration, and makes no new paper-level performance claim. The JSON contains the complete data tables, source provenance, work-to-mechanism crosswalk and bounded proposed workplan. Future updates should append frozen native outcomes, teacher uncertainty, effective parameters, independent field IDs and deployed costs; they should preserve historical protocols/gates and label exploratory findings separately.

