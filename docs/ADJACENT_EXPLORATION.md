# Adjacent exploration and temporal reuse

This study has separate source, data, costs and decisions. It does not promote
an architecture into TDN or alter its running portfolio. Twenty percent of the
**adjacent discretionary pilot budget** is reserved for the two alternatives
below; this is independent of the main portfolio's protected allocation.
The executable protocol is authoritative for actual seconds and stage limits.

## Evidence motivating the designs

The verified CPU portfolio development diagnostic found that four-node
quadrature already resolved the quadratic integral to approximately
`6.6e-13` on its high-pair Galerkin example, while the complete quadratic
correction had about `3.65e-6` RMS error. Adding the analytic cubic lowered
that case to about `7.77e-8`. These are same-grid, same-schedule development
observations, not a matched-cost win or proof that learning is necessary.

The previous X01 compact temporal kernel represented its lifted quadratic
response accurately but required about **2062 queries** to repay encoding,
against a registered realistic limit of 32. The response was not a complete
nonlinear PDE solution. The adjacent D07 program therefore retains the
negative finding and tests a changed computational premise explicitly: actual
groups of queries, complete build-and-query costs, and a complete finite-PDE
correction with a separately timed physical backbone. It does not extrapolate
a kernel-only latency into an end-to-end speedup.

## Two registered alternatives

| ID | Mechanism and challenged assumption | Added or avoided computation | Largest risk | Cheapest falsifier | Advance criterion |
|---|---|---|---|---|---|
| E01 | Compute the analytic cubic only when `h r RMS(v) > 0.012`, with `u=m+v`. Challenges uniform interaction order on every state. | One mean/variance statistic, followed by **no cubic evaluation** on skipped states. Selection occurs before nested transport/products. | A scalar amplitude indicator can miss phase-specific cubic relevance. | Held-out mixed-phase parents; compare complete PDE errors against always-quadratic and always-quadratic+cubic. | Median paired complete speedup at least 1.10, worst held-out RMS ratio at most 1.10 against always-cubic, at least 1.10 median RMS improvement over quadratic alone, and no new maximum-error target failures. |
| E02 | Use `gamma P[v Q(v)]` as a signed cubic direction, with one globally fitted coefficient. Challenges the need to evaluate nested cubic transport to obtain useful cubic structure. | One target-consistent product after the quadratic correction, replacing nested cubic quadrature. Analytic cubic labels and fitting are charged offline. | The true cubic spatial direction may not align with the surrogate, even when amplitude order is correct. | Fit on disjoint parents, then inspect held-out cubic residual and complete PDE error; retain failures. | Same full-cost and accuracy gates as E01; no per-query reference-informed coefficient is allowed. |

E02 is the additional idea generated from the amplitude-order diagnostic:
it combines an already available quadratic spatial direction with the signed
fluctuation. It is not an extra input band, extra quadrature node, or an
absolute-energy gain. Neither proposal claims literature novelty.

At a fixed background and physical operator, homogeneity gives

\[
Q_{m,h}(\epsilon v)=\epsilon^2 Q_{m,h}(v),\qquad
P[(\epsilon v)Q_{m,h}(\epsilon v)]
=\epsilon^3 P[vQ_{m,h}(v)].
\]

The surrogate is odd under `v -> -v`; its signed phase is preserved. Here
`P` means the product operator of the declared target: nodal multiplication
for the FD/nodal equation, sequential dealiased Galerkin multiplication for
the finite spectral equation. It is not an arbitrary output filter.
Homogeneity does **not** establish equality with the cubic Volterra term,
stability, positivity, or improved timestep order. The zero-time,
zero-diffusion, zero-reaction and constant-field limits follow only because
the quadratic factor vanishes in those limits; explicit numerical tests
check the implementation.

The coefficient is fitted by bounded global least squares on fit parents:

\[
\gamma_* = \operatorname{clip}_{[-8,8]}
\frac{\sum_i\langle P[v_iQ_i], C_i\rangle}
{\sum_i\|P[v_iQ_i]\|^2},
\]

where `C_i` is the analytic cubic correction. A vanishing denominator is
reported as unresolved rather than an informative fitted coefficient.
Fit parents and held-out development parents have different seeds and IDs.
The fixed bound and selector threshold are not retuned on held-out results.
The frozen protocol can impose a stricter regression allowance: E01 currently
allows 5% RMS regression, while E02 allows 10%. The actual thresholds are
written into each decision; the protocol takes precedence over the broad
10% pilot envelope in the table above.
Every parent must also have an accepted reference and errors resolved above
five times its refinement uncertainty. Unresolved cases retain their full
denominator and produce `NA / REFINE_REFERENCE`, even if measured latency is
excellent; nonfinite predictions remain `BAD / NUMERICAL_FAILURE`. Ratios
used for decisions include the measured uncertainty in the unfavorable
direction. These are sensitivity checks, not certified error bounds.
References use time refinement of an independent Lawson solver; agreement
is an uncertainty estimate, not a certificate or a continuum limit.

These are **development** experiments. Parent counts are deliberately small;
the output has no confirmatory confidence interval. All related methods on
one field are paired. A successful pilot would justify a separately frozen,
independent comparison, not immediate incorporation into TDN.

## D07: temporal reuse with its complete costs

Two tasks remain separate in every output:

1. A finite one-dimensional, non-aliased lifted quadratic interaction kernel.
   Controls include cached exact pair sums, batched FFT four-node quadrature,
   and SciPy DOP853 dense output for exactly the same lifted equation.
2. A complete two-dimensional finite reaction–diffusion equation. Controls
   include the DF backbone plus direct four-node quadratic correction and
   DOP853 dense output for the complete finite equation. The reusable method
   interpolates the physical quadratic correction, then still computes its
   DF backbone and reconstructs the complete field at each requested time.

The second encoding fits a Chebyshev representation of `Q(h)/h^3` on the
declared interval, so decoding multiplies by `h^3` and retains the exact
zero-time correction. This structural factor alone is not a proof of the
approximation error near zero. Off-node numerical parity is measured, and
complete PDE errors are compared against tighter adaptive dense output and
an independent endpoint Lawson refinement. The `continuum` operator label
means a **finite Galerkin equation** in D07; no spatially refined continuum
accuracy claim is made by this diagnostic.

Four workloads are distinguished:

- **Same-state queries:** one unchanged state/operator, several horizons.
- **Trajectory dense output:** the same computation interpreted as samples
  along one initial-value trajectory. These rows share measurement IDs with
  same-state queries; they are not additional independent evidence.
- **Parameter queries:** changed diffusion parameters require rebuilding both
  state encodings and classical dense trajectories. All rebuild work is timed.
- **Autonomous rollout:** each step consumes its own predicted state, so every
  encoding is refreshed. These results cannot claim one-time-encoding cost.

Cache keys include exact state bytes, shape/grid, precision, spatial target,
domain lengths, physical parameters and valid time interval. Any change
requires refresh. Validation hashing and reconstruction costs are measured.
The current diagnostic is CPU-only: transfer is zero because no host/device
transfer occurs, compilation is unavailable, and neither value implies a GPU
measurement. Stored array bytes are reported; they are not peak process RSS.

For `C(Q)=E+Qq` versus `Qd`, a finite nominal break-even exists only when
`d>q`. Because the inequality is strict, the smallest integer is
`floor(E/(d-q))+1`. If `d<=q`, the reported count is **NA**, even when an
optimistic division would produce a negative answer. Build/query extrapolation
is secondary to measured finite groups of queries. Advancement additionally
requires the measured complete-PDE maximum error target and complete cost
below both direct correction and classical dense output at a realistic query
count no larger than 32. One synthetic parent is insufficient for a comparative
scientific claim.

Timings are repeated in randomized paired order. First invocation is recorded
separately from warm repeats and is explicitly not a cold-process startup.
Complete measurements construct a new encoding/dense solution; query-only
measurements allow equivalent reuse to every applicable method. The DOP853
adapter uses the same Torch spatial RHS through NumPy: this is a reproducible
control, not an optimized authoritative implementation or literature benchmark.
No compatible trained historical temporal-MLP checkpoint is silently replaced
by a random model; its applicability remains explicit NA.

## Prior work and novelty boundaries

The starting mathematics is established: the Volterra/Picard expansion of
nonlinear evolution, polynomial interaction order, exponential propagators,
interpolatory temporal bases, adaptive work allocation, and Galerkin products.
Relevant anchors include:

- [Hochbruck and Ostermann, *Exponential integrators* (2010)](https://doi.org/10.1017/S0962492910000048):
  variation of constants, stiff order and exponential integration.
- [Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*](https://doi.org/10.1007/978-3-540-78862-1):
  embedded integration and dense output as established competing capabilities.
- [Gupta, Xiao and Bogdan, *Multiwavelet-based Operator Learning* (2021)](https://arxiv.org/abs/2109.13459):
  multiscale operator representations already exist; extra scale channels alone
  cannot substantiate novelty.
- [Li et al., *Fourier Neural Operator*](https://arxiv.org/abs/2010.08895):
  spectral layers include local paths and nonlinearities; their ability to form
  fine-scale interactions cannot be dismissed from a Fourier cutoff alone.

These citations delimit the concepts, rather than establish an exhaustive
novelty search. A distinct signed-modulation combination can be new to this
repository while remaining closely related to established polynomial/Volterra
closures. Any eventual novelty claim requires a focused literature review
after the actual useful mechanism has been identified.

## Artifact and decision contract

`temporal_rows.json` and `temporal_cost_rows.json` retain task/workload labels,
raw randomized timings, errors and actual query counts. `temporal_build_costs.json`
contains preparation, encoding, reference, cache-validation, reconstruction
and storage costs, plus measured and modeled break-even results.
`temporal_cache_checks.json` records actual invalidation checks.
`temporal_applicability.json` preserves unsupported comparisons as NA.

`prototype_measurements.json` retains every held-out field/method observation;
`prototype_fits.json` records the offline fit cost, identities and coefficient;
`prototype_rows.json` records mathematical, gap and utility checks;
`prototype_decisions.json` states continue/stop outcomes per track.
No energy or monetary figure is invented when actual measurement/rates are
absent. Computation completing is separate from mathematics passing and
utility succeeding. Negative results are retained unchanged.
