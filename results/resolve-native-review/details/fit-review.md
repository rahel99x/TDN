# Native full Resolve: coefficient fitting review

The direct optimizer and Adam reach essentially the same two-coefficient solution. Further optimization of this fixed global-gain model is not the promising bottleneck. Calibration modestly improves its fixed parent, but strong analytic rules retain a much larger accuracy advantage.

## Evidence and scope

Target clarification: the prepared `continuum` teachers use spatially refined 2N/4N Galerkin solves projected to N; predictions operate at N. These errors include unresolved-scale closure differences as well as integration/representation error. They are not the same-grid Galerkin targets used in the kernel microbenchmarks. The interaction panel separates those contributions.

Sources are the exact sealed `fit-discrete/` and `fit-continuum/` artifacts from `fedora-resolve-20261011T052313346218Z`, jobs 849 and 850, source tree `5d73ce536ff1bc8a354d68e641b83f4f5afa46481dadb6b853fb86b26480be2e`. Native execution was Fedora CPU, Python 3.13.13, Torch 2.10.0+cu126. The presence of a CUDA wheel does not make these CPU fits GPU measurements.

There are **64 independent training fields**, each at h=0.02/0.06/0.12, and **32 separate development-validation fields**, each at h=0.03/0.06/0.12/0.24. Both tracks use these same field identities at 32². Nine methods give 1,152 validation endpoint rows per track, 2,304 total; the statistical unit remains the 32 fields, not the rows, tracks, or four repeated horizons. Each regime has only four validation fields. There is one deterministic full-batch fit per track; the seed field is bookkeeping, not a replicated fit. No fresh confirmation or rollout is established.

I independently recomputed all 18 method-level mean RMS and normalized validation-loss summaries from endpoint rows; they match. Derived regime/horizon tables and an explicitly post-hoc 10,000-resample field bootstrap are in `fit-review.json`. Original objective cuts, gains, curves, reference uncertainties and endpoints remain unchanged.

## What was optimized

The prediction is $\hat u=S_h^{DF}(u)+g_2 Q_{GL2,h}(u)+g_3 C_{3,h}(u)$, where the two physical fields are frozen FP32 responses, and the cubic term is the existing heat-filtered leading cubic defect. The direct fit varies only two global gains in $[0.25,1.75]$. The loss is equal-field averaged $(\operatorname{mean} e^2+0.1\max e^2)/s$, with the same fixed DF-derived normalization as the Adam control. Its real-affine arithmetic is FP64; native endpoint assembly and the FP64 reference metrics are separately measured.

For each max-square active component, the selected gradient is a valid convex subgradient. A convex mixture of supporting planes supplies a numerical lower bound over the gain box. The known nonnegativity of squared loss supplies another lower bound. These floating-point bounds diagnose convergence; they are not interval-arithmetic certificates. The normalized MSE part has Gram matrix G and Hessian 2G, giving positive curvature for these observed data.

| Quantity | Discrete | Continuum |
|---|---:|---:|
| Direct gains (g2, g3) | 1.056391, 0.911835 | 1.072458, 0.845623 |
| Adam gains (g2, g3) | 1.057115, 0.912497 | 1.072383, 0.843705 |
| Direct loss / fixed-parent loss | 0.101912433 / 0.104613008 | 0.156482167 / 0.160342416 |
| Direct objective gap | 8.77e-07 | 6.44e-07 |
| Adam objective minus direct objective | -3.01e-07 | -1.46e-07 |
| Direct iterations / Adam updates | 24 / 800 | 21 / 800 |
| Direct optimizer time / Adam optimizer time | 60.3 ms / 2.906 s | 52.8 ms / 2.800 s |
| Total observed fit-stage attempt | 6.460 s | 14.000 s |

All eight direct subproblems converge under the declared 1.01e-6 gap tolerance. Adam is slightly better than the returned direct iterate, by less than either gap; this is expected with a finite stopping tolerance and is not a contradictory optimum. The two-gain fit reduces training loss only 2.58%/2.41%. Its gains are interior, so active box bounds do not explain the remaining residual. Direct fitting is around 48–53 times faster for this tiny offline solve, but that is **not an inference-speed claim**. Basis preparation, references and deployment costs are different quantities.

This resolves the new controlled objective: optimization is nearly exhausted. It does not retroactively establish convergence of the older stochastic FP32 training, whose distribution, rate, sampling and arithmetic differed.

## Held-out accuracy and coverage

RMS below is the arithmetic mean over fields and then their four horizons; maximum is the largest spatial endpoint error, not a population tail. A joint pass requires both uncertainty-inflated RMS and maximum error ≤2e-5. Fields passing means every prescribed horizon passes.

| Method | Discrete mean RMS | Continuum mean RMS | Discrete joint queries /128 | Continuum joint queries /128 | Fields passing all horizons D/C /32 |
|---|---:|---:|---:|---:|---:|
| analytic_quad_cubic | 1.699e-06 | 2.72e-06 | 124 | 125 | 28/30 |
| cubic_gain_direct | 1.058e-05 | 1.158e-05 | 95 | 101 | 14/16 |
| fixed_unit | 1.076e-05 | 1.169e-05 | 95 | 103 | 14/16 |
| quad2_full | 1.03e-05 | 1.118e-05 | 95 | 103 | 16/19 |
| quad4_full | 3.155e-06 | 3.699e-06 | 112 | 114 | 18/23 |
| quadratic_gain_direct | 1.005e-05 | 1.055e-05 | 97 | 108 | 14/19 |
| quadratic_only_direct | 9.739e-06 | 1.02e-05 | 98 | 107 | 17/21 |
| two_gain_adam | 9.91e-06 | 1.041e-05 | 98 | 108 | 15/19 |
| two_gain_direct | 9.913e-06 | 1.041e-05 | 98 | 108 | 15/19 |

The primary fitted rule lowers mean RMS by **7.85% discrete and 10.99% continuum** relative to its fixed parent. Estimated-reference-uncertainty-adjusted ratios are 1.08480 and 1.12322; the preregistered ratio≥1.10 parent gate is BAD/GOOD, respectively. A ratio of 1.10 represents a 9.09% error reduction, not exactly 10%.

Against GL4, the fitted rule has **3.14×/2.81× larger mean RMS**. Against full nested quadratic+cubic, it has **5.84×/3.83× larger mean RMS**. GL4 gives 112/114 joint passes, versus 98/108 for ours; the nested rule gives 124/125. The strong-control gate is BAD on both tracks. All methods are finite, have no sampled [0,1] violations and no measured endpoint Lyapunov-energy increases; those finite observations are not stability proofs.

The no-cubic fitted control is especially informative. With gains (1.04095,0)/(1.06583,0), it has 1.76%/1.96% smaller mean RMS than the two-gain fit, even though the two-gain fit wins on 24/32 and 26/32 individual field means. Difficult tails dominate the aggregate; neither count alone resolves utility. Its normalized objective is worse on the discrete track, but slightly better on the continuum track. More capacity in this cubic channel is not currently justified by its average absolute-error utility.

The arithmetic-RMS objective, normalized max/MSE training objective, worst maximum error and field coverage do not rank every ablation identically. For example, fitting only g2 while keeping g3=1 reduces average RMS, but slightly worsens validation normalized loss relative to fixed unit gains on both tracks. This is an observable objective/distribution tradeoff, not an optimizer bug.

## Uncertainty and heterogeneity

The following intervals are additional, post-hoc descriptive paired-field bootstrap intervals for a ratio of arithmetic mean RMS. They condition on the frozen fitted coefficients, do not refit models in resamples and are not adjusted for multiple comparisons.

| Comparator / our direct fit | Discrete ratio [95% interval] | Continuum ratio [95% interval] | Fields with ours lower RMS D/C |
|---|---:|---:|---:|
| fixed_unit | 1.085 [1.039, 1.148] | 1.123 [1.074, 1.222] | 22/22 out of32 |
| quad2_full | 1.039 [0.866, 1.161] | 1.075 [0.919, 1.223] | 23/25 out of32 |
| quad4_full | 0.318 [0.203, 0.594] | 0.355 [0.230, 0.643] | 15/15 out of32 |
| analytic_quad_cubic | 0.171 [0.107, 0.289] | 0.261 [0.158, 0.418] | 1/0 out of32 |

There is evidence of modest fixed-parent calibration, but the continuum point gate does not imply a population improvement of at least 9.09%: its post-hoc lower interval endpoint is only 1.074. The apparent small average advantage over plain GL2 has intervals crossing one on both tracks. The GL4 and nested advantages remain substantial.

All 128 unique reference queries per track are accepted. Maximum RMS uncertainty is 1.39e-8/9.30e-9; uncertainty is at most 1.82%/0.81% of the candidate error. The GL4 failure does not disappear under the recorded uncertainty sensitivity. Reference agreement remains empirical, not certified.

Many easy endpoints are near FP32 precision: 51/128 candidate queries per track have RMS below 1e-7; native affine-assembly changes reach about 2.98e-8 maximum, up to 84%/79% of an individual tiny RMS. Small easy-case wins should not be interpreted as mechanisms. In contrast, high-pair and near-Nyquist regimes account for 68%/75% of the sum of candidate RMS, and their strong-control differences are much larger.

| Regime (4 fields; 16 queries per track) | Fixed / direct RMS D/C | GL4 / direct RMS D/C | Interpretation |
|---|---:|---:|---|
| low | 0.910/0.891 | 1.125/1.086 | Calibration regresses; tiny absolute errors |
| mixed | 1.074/1.054 | 1.016/1.032 | Small GL4 differences; nested remains stronger |
| high_pair | 1.104/1.125 | 0.263/0.306 | Calibration helps parent; GL4 is much stronger |
| overlapping_triads | 0.842/0.782 | 1.017/0.936 | Calibration regresses; mostly tiny errors |
| near_nyquist | 1.091/1.154 | 0.171/0.203 | Calibration helps parent; large GL4 deficit |
| boundary_mean | 1.076/1.124 | 1.174/1.219 | Ours modestly beats GL4 here; nested still much better |
| reaction_stiff | 1.069/1.122 | 0.203/0.247 | Cubic fit loses strongly to omitting the cubic term |
| diffusion_stiff | 1.044/1.060 | 0.482/0.662 | Quadrature accuracy remains limiting |

In reaction-stiff fields, the candidate has 4.40×/3.58× larger mean RMS than plain GL2. Two fixed gains cannot reliably repair the leading cubic formula across this regime. Worst relative GL4 regressions reach about 66× discrete and 98× continuum, including physically nontrivial candidate errors around 1e-5–1e-4. A average improvement should never hide those examples.

At h=0.24, beyond the largest training horizon0.12, the parent/direct RMS ratio is1.071 discrete and1.162 continuum. GL4/direct is only0.343/0.400. Calibration extrapolates modestly; it does not rescue the finite-step representation. The experiment contains single-step endpoints only, not multiple refreshed encodings or long-rollout stability.

## Identifiability and what is actually wrong

The aggregate normalized Gram correlation is−0.0362/−0.0387 on training and0.00974/0.01120 on validation. Both matrices have numerical rank2; raw condition numbers are16.55/15.78 (training) and7.71/6.55 (validation). These are adequate for two **global** coefficients on this mixture. They cannot establish identifiability of a state-dependent conditioner.

The new overlapping-triad fields address the earlier artificially orthogonal spectrum: absolute within-query quadratic/cubic cosine reaches0.536/0.516 (median0.283/0.263), rather than identically zero. Still, five of16 queries in this regime have both responses below1e-8. Other stiff cases can be much less informative about one coefficient. For example, a continuum near-Nyquist h=.24 case has cosine−.9998 but cubic RMS1.18e-19 against quadratic RMS8.04e-8. The spectacular cosine is a vanishing-response degeneracy, not useful evidence that a cubic gain is learnable there. Norms must accompany correlation plots.

The mean mode is a large overlooked contributor: summed mean-error squares account for **71.8%/74.6%** of candidate squared RMS across the validation queries. GL4 lowers mean absolute error by approximately4.9×/3.7×, while centered RMS improves by2.35×/2.13×. This points toward resolving nonlinear mean/variance interaction accurately, not merely adding spatial expressiveness. Logistic reaction does not conserve mean: d(mean u)/dt = r(mean u − (mean u)² − variance u).

## Next informative experiment

1. Retire more global-gain optimizer sweeps for this fixed basis; retain direct fitting as a cheap diagnostic/control. The new evidence substantially weakens optimization failure as the explanation for this model's analytic-control deficit.
2. On matched stiff and overlapping fields, swap the GL4 mean correction and centered correction independently against GL2/cubic. Measure vector alignment, mean-balance defect and maximum error, preserving exact nulls. This localizes whether zero-mode interaction or full spatial redistribution creates the gain.
3. Test a direct finite-step zero-mode interaction kernel from Fourier energy before a new neural mean head. For the quadratic response, the mean contribution reduces to a weighted sum over |v_k|² of the one-dimensional time integral q(s)[exp(2sλ_k)−exp(hλ_k)], multiplied by−r q(h). Retain the correct Galerkin/Nyquist mass convention and charge the kernel evaluation. It may avoid repeated spatial transport for this component; this is a falsifiable new prototype, not an established speedup.
4. Evaluate any replacement against frozen GL4 and nested controls at matched absolute RMS/max targets; include the fitted no-cubic control. Use a new development panel now, then lock a separate untouched confirmation cohort after selecting the mechanism.

No complete solver-cost advantage, published-FNO superiority, convergence order, continuum guarantee, statistical stability guarantee or training-distribution generalization follows from this coefficient unit. The strongest established outcome is that simple calibrated gains can modestly repair their parent, and that optimization of those gains is no longer the major obstacle.
