# Native Advance full: mathematical and exploratory review

Reviewed exact canonical artifacts from `fedora-advance-full-20261011T001715`, source commit `a7646702ff1061d78a722153ef2548d9d831de0d`. The original archive verification is owned by the main review. This subreview makes no source changes, trains no models, and does not reinterpret these development probes as fresh main confirmation. Reproducible numeric aggregation: `math-summary.py`, `math-summary.json` beside this file.

## Outcomes and denominators

| Stage | Logged experiments | Stage verdicts | Correctness failures | Actual scope |
|---|---:|---|---:|---|
| audit | 36 | 36 NA | 0 | 26 model/track cases, Gauss moments, teacher cross-checks, mean/cubic/energy identities |
| diagnose | 256 | 253 GOOD, 3 BAD | 0 | 8 methods × 4 constructed fields × 2 equations × 4 horizons |
| XH history | 30 | 12 NA, 18 BAD | 0 | 5 methods × 6 scenarios; 24 development test parents reused across scenarios |
| XR rational | 4 | 1 NA, 3 BAD | 0 | 24 parents × 2 diffusion values × 4 schedules × 4 responses = 768 endpoints |
| XT tangent | 2 | 1 NA, 1 BAD | 0 | 24 parents × 3 horizons = 72 tangents; 24 feature-equivalent twins |

NA in the audit does not denote an identity failure: a mathematical check alone cannot provide deployment or gap-closure evidence. Likewise, a correct implementation can produce a BAD scientific result. The 3 diagnostic BADs are DF target misses at h=0.12: high-pair discrete (maximum error only), high-pair continuum, near-Nyquist continuum. They are not the 3 raw-cubic regressions, which still satisfy the loose target.

Prototype check-level counts (not independent samples): XH 60 mathematical GOOD, 9 gap GOOD, 21 gap BAD, 30 gap NA, 30 utility NA; XR 12 mathematical GOOD, 4 gap GOOD, 4 gap BAD, 4 utility NA; XT 6 mathematical GOOD, 1 gap GOOD, 1 gap BAD, 1 utility NA. All check scores are evidence-attainment summaries, not probabilities or theorem validity scores.

## What the cubic mathematics establishes

Let A be the diffusion generator, v=u−mean(u), and B the target's symmetric quadratic product. The implemented leading centered-amplitude cubic defect is

C3(v,h) = r²h³/6 { B(Av,B(v,v)) − 2B(v,B(v,Av)) + 2B(v,A B(v,v)) − A B(v,B(v,v)) }.

For nodal multiplication, associativity reduces the braces to −v²Av + 2v A(v²) − A(v³). For an unprojected continuum Laplacian, the differential product rule further gives C3 = −r²κh³ v|∇v|²/3. That last identity is NOT valid after a finite Galerkin projection: projected multiplication is nonassociative. The code retains nested B operations on the continuum/Galerkin track, and a regression explicitly detects the invalid nodal simplification.

The filtered candidate is H(h/2) C3(H(h/2)v,h), H=exp(hA). It preserves the leading fixed-grid h³ coefficient and centered-amplitude cubic homogeneity. It is a heuristic finite-step regularization, not the full cubic Volterra coefficient, a higher-order proof for the entire solver, or a stiffness-uniform estimate. The raw and filtered versions are separately retained. Physical corrections vanish at h=0, κ=0, r=0, and constant states; randomized trained weights do not destroy those structural nulls in the physical families.

The native audit's 26 model/track cases have exact zero-time identities, finite outputs, and maximum translation discrepancy 2.22e−16. GL2/GL4 polynomial-moment errors are 5.55e−17 and 1.11e−16. Same-grid Lawson teachers agree with independently integrated DOP853 to at most 1.11e−15 in the small audit. The separate mean identity is m′=r[m−m²−Var(u)], not mean conservation. The continuum variance/energy use the Fourier-interpolant inner product with half-weight at each even-grid Nyquist axis. The gradient-flow energy identity dE/dt=−||rhs||²_M is checked with that norm; a nodal norm would misstate it at Nyquist modes.

In 24 small-step comparisons against a six-node nested cubic rule, observed remainder slopes are 3.98796–3.99928 as h varies 0.0005,0.001,0.002. This supports C3 approximating that coefficient with an O(h⁴) remainder on these fixed grids. Independent DOP853 odd-amplitude defect extraction in the test suite provides an additional check. None of this is a global fourth-order solver result or proof of stable behavior under grid refinement.

## Diagnostic accuracy: a missing direction, not a confirmed speedup

These are FP64 CPU development probes at 16², κ=.004, r=3, four specific phase realizations, h=.015,.03,.06,.12. The same-grid 64/128-step reference uncertainty is at most 6.95e−14 RMS and 8.73e−14 sampled max; it does not assess continuum spatial refinement. Tables give medians of paired GL2 error / method error, so >1 is better.

| Method | Paired median RMS reduction vs GL2 | Better than GL2 | Joint RMS/max ≤2e−5 |
|---|---:|---:|---:|
| DF | 0.0774× | 0/32 | 29/32 |
| GL2 | 1× | reference | 32/32 |
| GL4 | 0.999998× | 13/32, mostly tiny differences | 32/32 |
| Raw cubic + GL2 | 4.4482× | 29/32 | 32/32 |
| Filtered cubic + GL2 | 7.3157× | 32/32 | 32/32 |
| Two-basis initialization | identical to filtered cubic | 32/32 | 32/32 |
| Full nested quadratic+cubic | 64.8436× | 32/32 | 32/32 |
| ETDRK4 | 1.8195× | 17/32 | 32/32 |

The filtered candidate's paired range is 1.167–35.337×. Raw cubic loses at h=.12 on continuum high-pair, broadband and near-Nyquist fields; its worst ratio is 0.6574 (about 52% greater error). Full nested cubic remains a stronger accuracy reference. GL4's near-unit median weakens the hypothesis that merely adding quadrature nodes resolves the observed residual. Filtered C3 plausibly captures missing interaction structure more cheaply than a full nested integral, but almost all cases already pass the target, and diagnostic timings are unreplicated CPU calls. Main confirmation and cost analysis must decide practical value.

There are no endpoint energy increases or sampled [0,1] violations in these 256 cases. The 64 composition probes use one finite-difference perturbation direction per field/track/method; physical amplification is approximately .381–.585, below the conditional exact-flow bound exp(3*.12)=1.4333. A single direction is not a Jacobian norm, and finite endpoint checks do not certify positivity/stability along the trajectory. The exact-flow bound presumes nonnegative states/interpolants; Galerkin positivity is not guaranteed merely by this observation.

## A consequential identifiability blind spot

All 32 same-field, unconstrained two-basis oracles have rank 2 and normalized condition number approximately 1. Their fitted GL2 gain ranges .9970–1.0445, cubic gain .6416–1.5014, and median one-basis-oracle / two-basis-oracle error ratio is 16.9715×. Median residual norm left outside the two-dimensional span is .00638 (range .000147–.05991). These fits use each evaluation teacher directly, so they are representation diagnostics rather than deployable learned performance.

Crucially, condition≈1 is designed into these particular spectra. An independent signed-frequency enumeration shows that the quadratic and cubic mode-sum supports are disjoint for ALL FOUR diagnostic field families, even after modulo-16 aliasing. Quadratic and cubic corrections are therefore spectrally orthogonal in this diagnostic design. The result establishes a useful second direction on these fields; it does not establish identifiability or absence of destructive cancellation on generic broadband spectra. The 32 cases are four constructed spatial fields with repeated tracks/horizons, not 32 independent realizations.

The next informative diagnostic should add closed-triad/overlapping sum supports, many independent phases, relative basis norms and near-collinearity. Examine how often raw/filtered cubic aligns against the existing error, and whether global gains can fit the observed regime- and h-dependent optimal gains. Do not use the current oracle's excellent conditioning as a general property of the learned architecture.

## XH: causal memory helps clean states but is not yet robust

This is a separate one-dimensional non-aliased finite Fourier ODE (K=8 fine, m=2 coarse) with r=.7. Twenty-four training parents and six validation parents each provide three repeated observation times; 24 test parents produce 12 autonomous steps. A linear-in-features ridge model has 80 parameters, not a neural network. Previous and current observations are integrated FORWARD; future evaluation teachers never refresh the prediction trajectory. Coarse RK4, persistence, instantaneous 55-parameter fit and deterministic secant memory are controls. Acquisition and unique teacher/fit costs are recorded.

The coarse increment residual is η̂_n=[c_n−Ψ_h(c_{n−1})]/h, and fitted memory predicts Ψ_h(c_n)+h g(c_n,η̂_n). Its six scenario results are:

| Scenario | Fitted-memory / coarse-RK4 RMS ratio | Fitted-memory / deterministic-memory ratio | Interpretation |
|---|---:|---:|---|
| Normal | .04902 | .78690 | Large gain over no closure; modest additional fitted gain |
| Doubled step | .08488 | .73011 | Clean-history benefit persists |
| Tripled diffusion | .11258 | .75176 | Clean coefficient-shift benefit persists |
| Startup noise 1e−4 | 15.3712 | .95521 | Worse than no closure; misses 10% extra gain |
| Startup noise 1e−3 | 16.5643 | .95035 | Same failure, larger absolute error |
| Combined step/coefficient/noise shift | 15.2830 | .95494 | Same failure |

Ratios are paired medians across 24 parents, not ratios of independently aggregated medians. The normal fitted-memory median RMS is 3.071e−6 versus deterministic memory 3.641e−6 and coarse RK4 5.760e−5. But worst sampled maximum error is 1.476e−5 versus deterministic memory 1.199e−5: a median gain coexists with a worse tail. The instantaneous fit fails the declared 20% reduction against coarse RK4 in every scenario. Every 8,640 rollout row remains finite and within [0,1] on the 257-point probe; absence of blow-up does not make its noise error acceptable.

Normal complete trajectory medians, including startup, are .713 ms coarse RK4, 1.030 ms secant memory, 1.318 ms fitted memory. These are descriptive, nonrandomized per-trajectory CPU timings. Ridge fitting and validation for both fitted models total .001078 s; teacher generation is separately itemized. No matched-accuracy complete-cost advantage is established, and the 30 utility checks properly remain NA.

An effective-coefficient inspection gives a useful competing explanation: after undoing feature standardization, the learned secant block has diagonal .98989–.99696 and singular values .98476–.99963, very near identity. Most of the clean improvement may be a mild modification of deterministic memory rather than a powerful state-dependent closure. A deterministic damped/decaying secant, with damping selected only on development validation, is the decisive missing control.

Noise amplification has an explicit cause: taking a difference of independently noisy observations and dividing by h produces an O(σ/h) contaminated closure estimate. A memory architecture trained only on clean history need not suppress it; recurrence can compound the contamination. Stop direct deployment of this clean-history fit. A small noise-aware/damped-memory pilot is justified only if it retains the clean gain and beats a deterministic filter under noise at matched error and complete cost.

## XR: rational responses do not earn advancement here

All four controls have the ETD1 structure a_{n+1}=E(hL)a_n+hΦ(hL)N(a_n), E=1+zΦ. Substituting a higher-order rational response does not raise the nonlinear integrator's order. The backward-Euler resolvent and Padé(0,2) are damped on the large negative real axis; Padé(1,1) approaches −1 and lacks L-stability. Scalar consistency/reference checks pass, but all rational competitors fail the worst-case 5% error-noninferiority criterion.

| Response | Median paired RMS / exponential-ETD1 | Worst paired ratio | Magnitude of E(−10⁶) |
|---|---:|---:|---:|
| Backward-Euler resolvent | 3.3970 | 46.855 | ~1e−6 |
| Padé(1,1) | .99988 | 236.518 | .999996 |
| Padé(0,2) | 1.06344 | 12.599 | ~2e−12 |

A pooled median near one would conceal Padé(1,1)'s severe tail. No 768 endpoint row is nonfinite or outside sampled [0,1], but error still rules out the tested substitutions. Descriptive per-call speed ratios around 1.06–1.11 are not accuracy-qualified deployment gains.

Repeated-query tests include equal caching of nonlinear products and response kernels plus a classical dense-output control. However, they use only ONE field, and their Q=1 case asks t=.005 while Q≥4 reaches t=.2; DOP853 setup integrates to .2 regardless. This is not a pure fixed-horizon query-count comparison and unfairly overcharges dense setup in the single-early-query case. All exponential/rational candidate representations fail the declared 1e−3 relative tolerance for Q=4,16,64 (errors about1.87–1.95e−3); classical dense output passes. Their low cached query-only latencies cannot establish a break-even advantage. Before revisiting amortized temporal operators, keep the final horizon fixed across Q, refresh encodings appropriately, give classical solvers only the requested horizon, and compare quality-qualified costs across independent fields.

Retire these simple rational substitutions for this workload. Rational/Krylov methods may be worthwhile when the linear operator is not cheaply Fourier-diagonalizable, but that would be a distinct, explicitly labeled workload.

## XT: a real information-loss result, not a new solver

The tangent equation v′=(r−κk²)v−2rP_K(a*v) is integrated with the state. Across72 parent/horizon cases the worst epsilon-sweep finite-difference relative error is2.9844e−7; the best-per-case maximum is3.0571e−11. Maximum semigroup-chain discrepancy is5.8720e−13. Mean/tangent-mean identities agree to5.55e−17 and6.94e−18. An initially unresolved perturbation reaches the resolved modes: minimum resolved tangent norm1.1580e−4. Full sampled tangent norms are≤.9946; this is one-direction contraction, not an all-direction theorem.

All24 paired fields share resolved coefficients and unresolved energy (feature mismatch≤1.39e−17) and have identical instantaneous mean derivatives to rounding. Their resolved endpoints at t=.2 nevertheless differ by5.5513e−6–8.0134e−5. Any deterministic predictor receiving only those identical features must return the same answer for each pair. By the triangle inequality, at least one member has error≥half the pair separation. This is a concrete finite-ODE counterexample to scalar-variance sufficiency for the full spatial future, not a claim that every moment-based closure fails or that TDN's full-field physical products are inadequate.

The proof direction suggests richer phase/interaction information or causal memory; merely adding one global variance feature cannot universally fix missing spatial information. The 696 unique state/tangent/finite-difference/twin solves total .2756 s. There is no deployed learned tangent solver or speed claim.

## Independence and next decisions

All three native full exploratory artifacts have exactly the same parent IDs and settings as the previously inspected `.runtime/advance-exploration-full-development` artifacts. Comparing every selected numerical error metric gives maximum difference0 across all8,640 XH rows,768 XR rows and72 XT rows. These native executions demonstrate reproducibility and supply native timings; they are not new independent confirmation. The 16² diagnostic likewise repeats the earlier local development design. Only the separate sealed main confirmation cohort may support the frozen main comparative claims.

Continue: cheap filtered cubic as a numerical interaction hypothesis, conditional on main accuracy-qualified cost; robust damped causal memory as a small development pilot; phase-sensitive representation diagnostics motivated by XT.

Change: add overlapping spectral supports/closed triads and independent phases before declaring two-basis identifiability; compare learned gains against fixed/low-dimensional deterministic rules; hold query horizon fixed in amortization tests; separate mean, spatial maximum and median RMS outcomes.

Stop for now: scalar-variance-only closure as a generally sufficient state, clean-history memory deployment under noise, and the tested rational substitutions as performance candidates. Preserve their failures. None of the above establishes literature novelty, FNO-paper superiority, a global stability theorem, or lower complete solver cost.
