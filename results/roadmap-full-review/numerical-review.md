# Numerical, headroom, transfer and scaling review

Run: `fedora-roadmap-20261008T022526329292Z`, full native Fedora run. Source fingerprint `2527656149023f9e89959fb6d3c6b3534a68388fa9a8126b8f13455d74897d17`.

This review reads uploaded JSON and local source; it does not execute uploaded code, load pickles/checkpoints, train models, or rerun solvers. Derived ratios below are arithmetic on recorded native results. Detailed operands and groups are in `numerical-review.json`.

## What survived the numerical tests

| Mechanism | Observed evidence | Interpretation and limitation |
|---|---|---|
| M02 quadratic DF defect | All three 1D/2D coefficient extractions passed; coefficient discrepancies 4.03e-12 to 5.16e-11. Six-node versus twelve-node differences at most 6.49e-15. Midpoint defect zero. | Strong finite diagnostic support for the implemented coefficient, including nulls. This is not finite-amplitude exactness or a speed advantage. |
| M04 sources before compression | Retained source RMS 0.68479 versus 3.44e-32 after input compression; doubling perturbation multiplies response exactly by four. | The high-high-to-low information loss is real. Signed physical products should precede compression. Representation advantage alone does not establish solver utility. |
| M05 product de-aliasing | Three grids: retained-product max errors 2.55e-15–6.33e-15 versus 0.5 nodal alias error. Gradient checks pass. | Correct Fourier-Galerkin quadratic product construction. It does not de-alias the infinitely nonlinear exact logistic map or prove positivity. |
| M12 moment closure | Initial mean/variance rate errors 3.83e-9/7.73e-10; binary normalized variance falls to 0.8272. Equal initial moments but different spectra give actual future mean gap 0.001340 and predicted gap 0.001312. | Spectral information matters beyond mean and variance. The future closure is approximate: its mean error creates a damaging floor when combined with a more accurate spatial response. |
| M20 order-constrained RK residual | Observed global orders 4.010 and 4.005 on the fixed mild grid. Embedded estimate 7.427e-11 versus measured error 7.440e-11. | Fourth-order consistency and one embedded-effectivity example supported. No A-stability or stiffness-independent cost; this panel uses a fixed residual, not evidence of successful learned residual training. |
| M21 physical inequalities | All sampled monotonicity, concavity, growth and interval checks pass, including three stressed physics settings. | Sampled diagnostics only. The general cooperative envelope is exp(rh), and at r h=4 it is 54.6: a permissive growth check is not contraction. |
| M23 cubic response | Independent cubic coefficient discrepancy 4.41e-8; five-/ten-node quadrature difference 3.75e-13; signal-resolution check passes. | Good finite numerical support for cubic response; uncertainty after amplitude extraction is 8.87e-6, so do not treat 4.41e-8 as a certified coefficient error. Cubic utility depends on including the quadratic term. |

The audit has 52 rows: 33 GOOD, 17 NA, 2 BAD, zero correctness failures. The only BAD rows are cubic-only centered-spatial regressions at amplitudes 0.04 and 0.12. These are small (~0.08% and ~0.12%) but real under the declared teacher estimates. NA often records an unsupported global certificate or missing category rather than a numerical malfunction.

## C3: composition is not monotone improvement

Same 1D family, h=0.15, amplitudes 0.04/0.12/0.25. RMS improvements are relative to the same-step DF base; cost is measured CPU median divided by base time. These individual timings use one sample and should not establish work–precision superiority.

| Arm | RMS improvement, amplitude .04 / .12 / .25 | Cost multiplier, .04 / .12 / .25 |
|---|---|---|
| Quadratic | 66.98 / 23.26 / 11.90 | 8.39 / 7.75 / 8.40 |
| Cubic only | 1.005 / 1.017 / 1.037 | 50.34 / 44.72 / 49.30 |
| Quadratic + cubic | 3742 / 428.33 / 103.35 | 54.30 / 50.33 / 56.27 |
| Mean only | 1.115 / 1.151 / 1.228 | 56.54 / 50.49 / 55.52 |
| Mean + quadratic + cubic | 1.594 / 1.683 / 1.916 | 109.61 / 100.53 / 110.00 |

The largest useful accuracy signal is quadratic+cubic without the approximate mean replacement. Its RMS errors are 1.47e-10, 1.17e-8, 2.13e-7. Adding the moment closure worsens these to 3.45e-7, 2.97e-6, 1.15e-5—roughly 2348×, 254×, 54× worse—while adding cost. It still passes a weak “no worse than DF” check. Future composition checks need a no-harm comparison against the strongest contained submodel, not only the original base.

The algebra explains this: if m and s denote orthogonal mean and centered-field errors, RMS² = m² + ||s||². A mean replacement cannot preserve a better existing zero mode unless its own mean estimate is at least as accurate. Replacing the zero mode once prevents double counting but does not ensure accuracy. Make the closure optional or choose it only with a deployable error criterion; do not hard-wire it into every combination.

Source-before-compression beats input-before-source in all three paired C3 cases by roughly 12–17% RMS, but both remain dominated by the mean-closure floor. Cubic-only does not compensate for an omitted leading quadratic defect.

## C0 and headroom: the current target is mostly already easy

- 1,152 classical candidates, 72 conditions, all with accepted teachers and at least one feasible candidate. These are 12 development field clusters crossed with grids and horizons, not 72 independent fresh fields.
- At the joint RMS/max target 2e-4, **70/72 conditions already pass one bare DF step**. Only two require extra bare-DF work. The faster feasible method on the measured step grid is reaction-first splitting in 49 conditions and diffusion-first in 23; no ETDRK4 or GL3 wins latency at this target.
- Candidate passes: DF 285/288, reaction-first 260/288, ETDRK4 273/288, fused GL3 288/288. GL3's reliability costs work; it is not the fastest default.
- Identical-map DF fusion agrees to 8.88e-16, but measured speedup is only 1.043× median, with 51/72 wins and range 0.872–1.232×. Removing two heat calls on this short schedule is correct but not a large speed breakthrough.
- Of 48 separate RMS/max spatial rows, 38 show the declared second-order regime and 26 meet the 2N spatial budget. The largest coarse-grid continuum discrepancy is 0.02485, far above 2e-4. Several high-mode fields are pre-asymptotic (difference ratios about 19–24 rather than 4). More temporal accuracy cannot repair the wrong spatial target.

Before expensive neural training, require a measured region in which classical work is genuinely nontrivial and spatial error is resolved. Use stricter targets or carefully expanded horizons/regimes with independently accepted teachers. Do not combine semidiscrete accuracy with a continuum advantage claim.

## C4, M13, M14, M19: portability works computationally, optional mechanisms can harm accuracy

Transfer has 595 rows over six independent phase clusters. All **394 BAD rows fail only the “complete same-step cost no larger than base” check**; there are no failed reported accuracy, finite-output or physical-math checks. Therefore “394 failures” does not mean 394 incorrect trajectories. Conversely, all tasks already sit far below the loose 2e-4 target, leaving little utility headroom.

| Periodic anisotropic arm | Rows | Median error improvement over base | Median same-step cost/base |
|---|---:|---:|---:|
| Nodal unfiltered | 96 | 1.961× | 0.0955× |
| Dealiased unfiltered | 96 | 1.961× | 1.2720× |
| Dealiased fractional | 96 | 1.973× | 1.2722× |
| Grid-index fractional | 96 | 1.983× | 1.2722× |
| C4 guarded | 96 | 1.973× | 1.2721× |

The nodal arm uses an exact logistic subflow whereas the dealiased base integrates a Galerkin reaction using four RK4 substeps. Its ~10.5× speed advantage is an established solver/computation choice on these mild resolved fields, not a new neural result or a general reason to ignore aliasing. All periodic base max errors lie below 7.43e-8—already thousands of times below the stated tolerance. Larger h and near-cutoff energy are needed to distinguish de-aliasing/filter value.

Every guard recorded zero fallback steps. Guard-on equals the corresponding fractional correction numerically here; the tests do not identify guard effectiveness in a regime that actually violates an invariant.

The cross-PDE correction is a charged three-split Richardson control, not trained TDN transfer:

| Equation | Base max error (roughly) | Unfiltered correction | Fractional+guarded correction | Cost/base |
|---|---:|---:|---:|---:|
| Gray–Scott | 8.05e-9 | 1.85e-13 | 9.72e-10 | ~3× |
| Viscous Burgers | 2.82e-7 | 2.20e-11–2.80e-11 | 3.80e-8 | ~3× |

**Fractional filtering worsens the unfiltered correction by ~5250× in Gray–Scott and 1359–1731× in Burgers.** No fallback activates, so this is the filtering ablation. If a symmetric second-order split has S_h = exact + h³a + …, the Richardson delta cancels h³a. Applying a fixed spatial filter F to delta leaves a leading (I−F)h³a term. An h-independent filter can therefore destroy the cancellation and leave second-order behavior. The source code explicitly acknowledges this issue; this run does not establish an improved order for C4. Keep unfiltered correction as the reference, measure convergence order, and test h-dependent filtering or an error-aware guard before combining it by default.

Nonuniform Dirichlet correction improves max error by ~3.25–6.65× at ~2.98–4.54× cost. Weighted symmetry and boundary reconstruction pass. Independent analytic-heat remeshing and quadrature show orders 1.989 and 1.999 on grids16/32. Tensor rotations/translations agree to 3.33e-16. These are bounded numerical geometry checks, not learned arbitrary-domain generalization.

### Important reference-validation gap

All **592 transfer timed rows report exactly zero teacher uncertainty and 196 DOP853 RHS evaluations**. `independent_reference` runs two tolerances differing by 10× while holding `max_step=horizon/8` fixed. On these mild problems the capped step sequence can be identical, yielding identical answers. Agreement of two identical trajectories is not actual temporal refinement. This does not prove the references are wrong; it means near-roundoff improvement ratios, especially Gray–Scott's 1e-13 result, need stronger verification before strong claims. Require different maximum steps (and preferably a second high-order integrator/check), preserve a roundoff floor, and record actual time grids/evaluation counts.

## M15 scaling and amortized classical work

GPU scaling completes 18 workloads (six square grids32–1024 crossed with batches1/4/16) for eight families: 144 neural timing rows plus three cache checks. It uses one repeated engineering field and the first training seed. Batch copies are not independent data.

- 48 rows have accepted same-grid references at32/64; 42 hybrid/rank rows pass the joint target but all are slower than bare DF at that same schedule. The six direct-FNO rows fail both teacher-backed RMS and max. There is no classical speed victory here.
- Remaining 96 rows at128–1024 have no independent same-grid teacher and correctly retain accuracy NA. Throughput comparisons are useful engineering data only.
- At512², batch16, Source is 1.280× and Rank1 is 2.250× faster than cheap FNO; at1024², batch16, Source is1.193× and Rank1 is1.385× faster. These are promising large-workload throughput signals that must be paired with new teacher-backed accuracy.
- On tiny32/64 grids, Source only gains ~2–4% over cheap FNO and Rank1 is slower; at large batches the compact branch becomes useful. Parameter count alone does not predict kernel-launch/transform cost: Rank1 has43 parameters but is not fastest at small grids.
- Scaling peaks: 6.048 GiB allocated, 8.029 GiB reserved, 9.590 GiB sampled whole-device usage—below the declared VRAM budget. This is scaling-stage memory, not whole-program peak memory.
- Charged cache reuse for three complete rollouts reduces coefficient preparations9→2 with seven hits and exactly matching outputs. ETDRK4 gains2.426× and GL3 gains1.497×. Key changes in h, physics, lengths, grid and dtype all pass fresh-parity checks. This is the clearest robust engineering gain and belongs in every fair classical comparison.

## Suggested priorities from this panel

1. Preserve source-before-compression and analytic quadratic response; retain quadratic+cubic as a high-accuracy reference. Compress/approximate their work only after measuring matched-tolerance cost against the strongest classical method.
2. Remove compulsory moment replacement and fixed fractional filtering from default combined architectures until they beat their own contained controls. Add checks against strongest submodel and convergence-order preservation.
3. Fix portability teacher refinement and choose genuinely difficult target/horizon regimes. Include cases where guards activate and fail, with independently resolved reference errors.
4. Add accepted large-grid references for the compact/source versus cheap-FNO crossover. Reuse the current frozen checkpoints for engineering characterization; use fresh parents for confirmatory claims.
5. Keep coefficient reuse in baselines, measure total work including cold preparations, and retain wall-clock variance. New architecture complexity has to earn fewer solver steps, not only a smaller same-step error that was already far below tolerance.
