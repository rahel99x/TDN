# Native Resolve memory review

**The protected memory path does not justify promotion.** The validation-selected deterministic answer is zero memory; the noise-aware fitted model has worse median paired RMS than coarse RK4 in every reported scenario. Clean-only fitting has a real clean-history benefit, but amplifies small observation errors and fails a sampled boundedness check in the shifted, noisier case.

## Scope and verification

Native job 863 used Python 3.13.13 in the project venv on CPU. Source `5d73ce536ff1bc8a354d68e641b83f4f5afa46481dadb6b853fb86b26480be2e` / commit `f07f9d56ccaee85b73b8d090aec3bc22ac067cf9`. The result is independent of the earlier cloud-sized probe: phase seed base 96200000, 48 train parents, 16 validation parents, 16 development-test parents. All splits are disjoint. Full means coverage, not confirmation.

The 30,720 rows are 16 fields × two noise draws × three noise levels × two reaction rates × five methods × 32 steps. The zero-noise draws repeat identical states. Independent evidence remains 16 test fields, not 30,720 samples. The six scopes reuse those fields and are paired.

Each method receives two noisy coarse observations at t=0.08 and t=0.10, then predicts without new measurements for 32 steps of h=0.02, to t=0.74. The fine/coarse Galerkin cutoffs are 8/2 (17/5 real degrees of freedom), on a periodic one-dimensional domain. κ=0.02 and r=0.7 or 2.1 are tested. Thus hr=0.014/0.042 and the predicted interval spans 0.448/1.344 reaction times: this is not a long-time or stiff spatial stress guarantee.

Noise σ is per independent real Fourier coordinate, not physical pointwise RMS. At cutoff 2 its root-mean-square over noise and physical space is 3σ. The two history perturbations are independent; methods and reaction scenarios share the same standardized draws. Noisy histories are not refreshed from future truth.

All 30 native summary cells were independently reconstructed with 0 mismatches. The selection-file SHA-256 matches. Zero-gain trajectories are bitwise identical to coarse RK4. All 32 reference trajectories pass the declared threshold; uncertainty estimates range 1.11e-12–5.96e-12, versus 1e-9 allowed. The largest uncertainty/RMS ratio in any recorded point is 5.88e-05. References use a tighter DOP853 solve plus independent 512-step RK4 endpoint; this addresses temporal error in the finite ODE, not continuum closure certification.

## Endpoint comparisons

Values below are the native median of paired candidate/coarse-RK4 RMS ratios across parent/noise realizations; **lower is better, 1 is a tie**. They are not ratios of unpaired median errors. The JSON also provides independent-parent geometric ratios and descriptive bootstrap intervals.

| Reaction | Coordinate noise σ | Plain secant | Clean fitted | Noise-aware fitted |
|---|---:|---:|---:|---:|
| reaction-1 | 0 | 0.325 | 0.128 | 12.057 |
| reaction-1 | 0.0001 | 37.267 | 29.075 | 6.549 |
| reaction-1 | 0.001 | 47.340 | 39.211 | 1.195 |
| reaction-3 | 0 | 0.637 | 0.236 | 5.093 |
| reaction-3 | 0.0001 | 22.762 | 18.844 | 5.367 |
| reaction-3 | 0.001 | 54.465 | 45.550 | 1.433 |

Clean fitted memory reduces paired RMS by 87.2% at the base rate and 76.4% after tripling reaction, but not under noise. At σ=1e-4 its RMS is 18.8–29.1× worse than coarse; at σ=1e-3 it is 39.2–45.6× worse. Noise-aware fitting reduces much of this amplification, yet still loses to coarse by 1.195–12.057× in native median paired RMS. Its improvement over a broken noisy-history model is not a solver advantage.

This is not uniform worsening on every parent. At base rate and σ=1e-3, the parent-clustered geometric noise-aware/coarse ratio is 1.109, with a descriptive unadjusted 95% bootstrap interval [0.950, 1.297]; six of 16 parents improve. At 3r and σ=1e-3 the ratio is 1.331 [1.116, 1.587], with four of 16 improving. These small development samples do not establish a benefit or a universal theorem of failure.

Worst sampled spatial error agrees with the clean-history benefit: clean fit reaches 1.08e-4/4.19e-4 at r/3r, versus coarse 1.51e-3/3.83e-3. At σ=1e-3 it instead reaches 0.383/0.284, while coarse remains at 0.00875/0.00622. Noise-aware worst errors are 0.00993/0.00774 there. Mean errors likewise expose damage: at the base rate and σ=1e-3, clean fit has median mean error 0.0352 versus coarse 0.000693.

## Boundedness and what the score misses

All states remain finite and all reference checks pass. That does **not** establish numerical stability. The required scoring checks currently test finiteness, but interval violations are only reported metrics, so zero correctness failures can coexist with severe overshoot.

| Scenario | Method | Parents violating [0,1] / 16 | Noise trajectories / 32 | Largest sampled violation |
|---|---|---:|---:|---:|
| reaction-1/noise-0.001 | plain_secant | 1 | 1 | 0.077108 |
| reaction-3/noise-0.001 | clean_fitted_memory | 3 | 3 | 0.089902 |
| reaction-3/noise-0.001 | plain_secant | 8 | 11 | 0.160180 |

The scored projected references remain within [0.308612, 0.871480] on the same 257 sampling points. These are observed physical-interval violations, not a general maximum-principle theorem for finite Galerkin truncations; unsampled maxima may be larger.

## Mathematical interpretation

With S_h the coarse RK4 map, plain memory advances x[n+1] = S_h(x[n]) + x[n] − S_h(x[n−1]). In the locally neutral limit S_h≈I this becomes x[n+1]=2x[n]−x[n−1], with a repeated unit characteristic root. Startup noise propagates as η[n]=(n+1)η[0]−nη[−1]; its standard deviation grows as σ√((n+1)²+n²). At n=32 the factor is 45.97. This simple limit explains why a method with plausible clean closure can amplify history noise dramatically; it is a mechanistic approximation, not a proof for the nonlinear problem.

Fixed damping α changes the neutral-limit roots to 1 and α. The validation objective is the average log of whole-trajectory RMS relative to coarse across equal-weight noise scenarios. Every tested nonzero gain loses: even the best nonzero option, α=0.25 and decay=4, has log ratio 0.1723 (about 1.188×). The selected α=0 has exactly coarse states. This selection concerns a mixed clean/noisy workload; it does not say that memory is useless when observations are certified clean.

Effective clean-fit secant coefficients have diagonal 0.9844–0.9879, singular values 0.9821–0.9925: they remain close to persistent secant memory. The noise-aware fit instead learns a negative secant block, diagonal −0.5002 to −0.4314. A secant-only, small-step Gaussian calculation predicts −1/2: the noisy derivative feature contains (η[0]−η[−1])/h, while its supervised correction target contains −η[0]/h. The covariance ratio is −σ²/(2σ²). The full model also uses state and squared-state features, so this is an explanatory limit, not a derivation of every fitted coefficient.

This reveals an important learning problem: noise augmentation teaches a one-step denoising response and then reuses it autonomously as a dynamical closure. New measurement noise does not arrive after startup, while the regression was trained on noisy teacher snapshots at every example. Both fits are linear-in-coefficients regressions over nonlinear state/secant features, with 80 coefficients plus 16 feature normalizers; increasing network size would not repair this information/target mismatch by itself. The noisy fit also has a different validation objective from the historical clean recipe, so their differences cannot be attributed solely to augmentation.

## Costs

Scheduler allocation: 14 s, billing=4,cpu=4,mem=32G,node=1, state COMPLETED; do not sum allocation and step rows.
The numerical stage took 8.381s; the experiment worker took 10.051s. Unique offline teacher/reference work was 0.559s, startup solves 0.034s, and fitting/selection 3.311s (mostly the damping grid: 2.604s). Repeated per-method rows reuse startup/reference costs and must not be summed as independent solves.

At base rate with clean inputs, observed median complete 32-step prediction cost including startup is 1.209 ms for coarse, 2.071 ms for plain secant, 2.093 ms for selected zero-gain damping, and approximately 2.75 ms for either fitted method. Zero-gain damping still computes the unused secant, so it reproduces coarse dynamics with avoidable overhead. A future implementation should dispatch α=0 directly to coarse. These fixed-order CPU timing observations are descriptive, not randomized deployment speed estimates; no accuracy-qualified cost claim or monetary claim follows.

## Decision

1. Stop promotion of the present noise-aware affine closure and stop treating noisy secant persistence as a deployment candidate. Preserve the useful negative evidence.
2. Keep the clean-only fit as a narrow control when clean causal state history is genuinely available, and require it to beat simple lightly damped secant on an explicitly clean validation workload before adding capacity.
3. Any next protected prototype should estimate state/noise jointly: test a deterministic two-observation filter or causal observer, then add learned closure only if the filter leaves a resolved residual. Supply known noise level or estimate uncertainty causally; gate memory when it is below its observation-noise floor. Compare against direct coarse and a deterministic observer at equal information and complete cost.
4. Use fresh development parents for this revision, train on autonomous rollouts or a correctly modeled observation process, and keep clean/noisy/rate-shift objectives separate. Advance only with at least 20% error reduction over the best validation-selected deterministic control, no reference-resolved maximum/mean regression, no new interval violations, and a plausible complete-cost margin. Require new locked confirmation before a comparative claim.
5. Make boundedness, failure rate, mean drift and long-horizon growth explicit required acceptance checks. Verify over longer physical horizons and independent noise structures, not just more steps at a single small h.

No new experiments or source changes were made in this review. All native outcomes, including failed utility checks and overshoots, remain intact.
