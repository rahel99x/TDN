# Adjacent interaction study: research contract

This program asks whether retaining the **origin, sign, scale and temporal response of nonlinear interactions** adds useful information at an acceptable complete cost. It supports TDN's original objective—approximating the finite-time splitting defect with useful temporal structure—without replacing that objective or modifying the active portfolio. Its source, protocol, budgets, run identifiers, reference banks, model selection and conclusions are separate. Incorporating a result into the main architecture requires a later explicit decision.

The supplied [handoff](ADJACENT_INTERACTION_STUDY_PROMPT.md) defines the study. The machine-readable [hypothesis register](ADJACENT_HYPOTHESES.json), immutable `tdn.analysis.adjacent.protocol`, executable diagnostics, local/native launchers and atlas implement its questions. See the [exploration notes](ADJACENT_EXPLORATION.md) for the two bounded alternatives. Source implementation and measured artifacts take precedence over aspirations in the handoff.

## Evidence boundary and original objective

The [original specification](IMPLEMENTATION_SPEC.md) proposed reliable physical subsolvers plus an order-anchored temporal approximation to the difference between split and coupled evolution. The current benchmark is periodic logistic reaction–diffusion. It is not a completed general ADR or HALO reproduction. The [portfolio review](PORTFOLIO_INITIAL_REVIEW.md) and its source-linked artifacts provide the relevant development evidence; pending native portfolio results are not available evidence.

| Evidence class | Verified observation | Implication for this adjacent study |
|---|---|---|
| Mathematical implementation | Current normalized two- and four-node rules have weights summing to one; historical frozen rank1 has total weight one half. | Preserve historical control, but do not attribute a normalization repair to learning. |
| Measured development observation | On the prior high-pair Galerkin field, GL4 versus GL16 quadratic correction differed by about `6.60e-13`, while the GL4 endpoint residual was `3.65e-6`. | More nodes cannot explain the dominant residual on that field. D01/D05 test whether another spatial direction matters. |
| Measured development observation | The analytic quadratic+cubic rule reduced that same-grid endpoint RMS to `7.77e-8`; premature input compression caused a `4.89e-5` correction contrast. | Higher-order shape and retained input interactions deserve controls. These are not matched-cost or learned wins. |
| Different target equations | Same-grid Galerkin versus projected finer reference differed by `4.69e-6`; FD versus fine continuum differed by `6.96e-3`. | Discrete temporal correction and spatial compensation are different tasks. Keep both targets named. |
| Negative cost result | Prior X01 accurate kernel encoding required about 2,062 nominal queries to amortize; X03 selectors reduced arithmetic but slowed execution. | D07 requires changed reuse economics and actual multi-query measurements. E01 must select before cubic work. |
| Confounded comparison | Historical learned versus half-normalized frozen comparisons and strong analytic rules with different node counts/cutoffs do not isolate neural necessity. | Read actual gain ranges; match node rules, projection and physical operators. |
| Unresolved neural comparison | Prior CPU smoke used two updates, two training parents and one seed; all primary smoke claims remained NA. | A short failed FNO fit is not evidence against FNO. Native pilots report optimization adequacy and full costs. |
| Untested hypothesis | Three origin channels can distinguish LL, LH and HH contributions before recombination. | They may expand the useful correction span, but can be collinear, poorly identifiable or too expensive. |

The diagnostic values above are small CPU development measurements, not 4090 results or independent cohort estimates. The prior profile found dense learned layers around 0.2% of instrumented self-CPU time in one small conditioned example; transforms, products, copies and orchestration dominate that example. It does not locate the native GPU bottleneck.

## What is analytic and what is learned

For the mean `m` and fluctuation `v=u-m`, TDN's physical branch computes a diffusion-first Strang backbone `S_h(u)` and a signed finite-time quadratic defect `Q_{m,h}(v)`. It uses the full field, physical parameters, diffusion symbols, logistic background response, transported products, quadrature and a split-defect subtraction. Those transforms and products remain the main work even when the conditioner has few parameters.

The existing scalar conditioner receives mean and dimensionless reaction/diffusion-step information. Richer controls additionally use variance, centered amplitude and spectral summaries. Its output changes a gain or node response; it does not receive a future reference. Existing output-band gains distinguish **where an interaction lands**. The new channels distinguish **which input scales produced it**.

Actual portfolio bounds differ by family: the scalar/affine conditioned gain is `1+0.75*tanh(z)` with open range `(0.25,1.75)`; the residual anchor uses `(0.75,1.25)`; output-band gain uses `(0.5,1.5)`; fixed normalized gains are one. Oracle bounds are read from the implementation. A bounded gain alone establishes neither stability nor positivity. Raw node/amplitude/conditioner parameters can be non-identifiable; reports inspect effective responses and basis sensitivity.

### Three input-origin channels

At one fixed background, timestep, equation, operator and quadrature, write `v=v_L+v_H` and define

\[
C_{LL}=Q_{m,h}(v_L),\quad C_{HH}=Q_{m,h}(v_H),\quad
C_{LH}=Q_{m,h}(v)-C_{LL}-C_{HH}.
\]

Quadratic homogeneity gives exact recombination `C_LL+C_LH+C_HH=Q(v)` up to numerical rounding. Recomputing a new background for each term invalidates this attribution. The transparent polarization reference and shared-transport implementation are checked against each other. The latter forms transported low/high fields once, then uses the physical products `v_L²`, `2v_Lv_H`, `v_H²` with the same transport and split subtraction. It avoids an explicit all-mode-pairs production tensor.

The candidate is

\[
u_{n+1}=S_h(u_n)+P_K\big(g_{LL}C_{LL}+g_{LH}C_{LH}+g_{HH}C_{HH}\big).
\]

All gains initialize at one. Fixed recombination must match the normalized baseline before fitting. An extra channel index is not a third spatial coordinate, and no FFT is applied along it. Reality, signed phase, cross-input symmetry and physical nulls are structural requirements. Small-step order and autonomous stability need separate measurements.

The control ladder retains fixed recombination, global fitted channel gains, affine-feature fitted channel gains, a small neural mixer, the scalar conditioner, output-band adaptation, ordinary feature-channel capacity, normalized GL2/GL4, full quadratic+cubic, DF/RF/ETDRK4 and local-path FNO variants where applicable. The ordinary feature control tests capacity, but comparable parameters alone are not comparable measured compute. Fitted constants and affine rules remain meaningful successful outcomes if they remove the network.

## Diagnostic decisions and mathematics

| Program | Controlled question and mathematics | Primary observation | Advance/change/stop decision |
|---|---|---|---|
| D01 | For `d=u_ref-S_h(u)` and `q=Q_h(u)`, `a*=<q,d>/<q,q>`. Clip to the actual admissible scalar interval. Solve channel/band/cubic projections with rank-revealing least squares and bounded optimization. | Residual RMS/max/mean, combined training objective, effective rank, condition number, sensitivity and spectral/spatial patterns. | Large scalar floor but materially lower channel floor supports a new basis. A low scalar floor and bad deployed response support feature/fit work. Near-zero correction/reference uncertainty makes inferred gains unresolved. |
| D02 | Match actual conditioner features across distinct spectral/phase arrangements. Compare a shared response to separately fitted responses and their acceptable gain sets. | Feature mismatch, excess shared-response error, interval overlap and extraction cost. | Add features only if ambiguity causes a resolved error penalty. Different futures alone do not establish insufficiency because the physical branch sees the full field. |
| D03 | Separate paired fixed-bandwidth grid refinement, added scales, changes in h and changes in physical T. | Exact step sequences, endpoint/trajectory errors, convergence, latency, memory and qualified cost. | Locate a region worth confirmation; stop broad acceleration claims where spatial floors or execution overhead consume headroom. |
| D04 | For modes p,q with retained k=p+q, quadratic response contains signed complex products `v_hat(p)v_hat(q)` rather than only their energy magnitudes. | Signed Fourier coefficient and phase error with a near-zero absolute floor. | Mechanism evidence requires advantage over matched compression and local-path neural controls, not an assumption that FNO discards every high mode. |
| D05 | For `u=m+epsilon*v`, separate the even and odd residual responses using positive and negative epsilon. Hold h fixed for amplitude order; hold amplitude fixed for temporal order. | GL2/GL4/refined quadrature error, residual slopes above uncertainty, cubic direction parallel/orthogonal to quadratic span. | More nodes only when integration of the quadratic formula dominates. Missing cubic shape justifies higher order; learning must still beat analytic and fitted controls. |
| D06 | Apply each method to its own outputs; use dense early accepted reference times and longer horizons. | Worst/integrated trajectory error, mean/variance/modes, threshold crossings, range violations, perturbations and unequal-step behavior. | Classify accurate/stable, inaccurate/stable and unstable separately. Late equilibrium cannot erase earlier failure. Logistic mean is not conserved. |
| D07 | Evaluate `E+Qq` versus measured direct/cached/classical totals, with build and refresh costs. Nominal break-even is finite only if `d>q`. | Actual multi-query totals, error, cache validity, encoding/query/reconstruction/storage costs and accuracy-qualified break-even. | Advance only inside a realistic query workload. Kernel approximation and complete-PDE accuracy are separate outcomes. |
| D08 | Retain smooth, weak, near-constant, small-step, tight-tolerance and latency-favorable controls; stress finite amplitude, localization and phase cancellation. | Per-regime error/cost/failure with initialization and validation curves. | Publish narrow wins/losses/unresolved cases. Equal data, train compute, inference cost and selected frontier are different comparison definitions. |

The scalar projection minimizes squared L2 only. The pilot's RMS/peak composite objective is evaluated separately; it is not made optimal by the projection formula. Similarly, component-error contrasts need not add because error cancellation can change when one approximation is replaced.

## Roughness benchmark and its limits

For `alpha=0.1,...,0.9`, set `H=1-alpha` and consider the ideal ridge

\[
W_H(x)=\sum_{j=0}^{\infty}2^{-Hj}\cos(2\pi2^jx).
\]

The classical cosine Weierstrass graph-dimension result applies to integer frequency multiplier `b=2` and amplitude factor `lambda=2^{-H}` with `1/b<lambda<1`; the graph has Hausdorff dimension `2+log(lambda)/log(b)=2-H`. Extrusion along an independent interval yields surface dimension `3-H=2+alpha`. See [Shen, *Hausdorff dimension of the graphs of the classical Weierstrass functions*](https://arxiv.org/abs/1505.03986). This assumption does not automatically cover arbitrary randomized phases or independently two-dimensional generators.

Every numerical input is a finite trigonometric sum and therefore smooth, with graph dimension two. The nominal dimension refers to an ideal limiting construction, not the finite solver input or the network's dimensionality. At `alpha=1`, `H=0` loses decay and the infinite series no longer converges in the same manner. This requested endpoint is a separately named finite-bandwidth stress case. It says nothing about the possibility of dimension-three graphs under other constructions.

The study implements a theorem-linked ridge for mathematical checks and independently two-dimensional multiscale fields for performance. It records physical modes, shell energies, increments, empirical scaling fit ranges, mean, amplitude, physical range, phases and parent identities. Mean/amplitude/range are controlled without clipping; incompatible simultaneous controls are recorded. Structured/random phases can have matched power but different nonlinear signed products. Translations and roughness/grid variants of a parent remain paired, not new independent samples. Fixed-bandwidth refinement and adding scales have separate labels. Diffusion rapidly smooths the initial field, so nominal initial dimension is not assumed to predict later difficulty.

## Exploration and novelty boundary

Exactly two alternatives receive the protected 20% of the adjacent discretionary pilot budget. E01 decides whether cubic work is needed using an inexpensive dimensionless amplitude indicator **before** constructing the cubic correction. E02 fits a single bounded coefficient multiplying the signed cubic-homogeneous direction `v*Q(v)`, using the target equation's product and independent fit parents. E02 follows the existing missing-cubic-direction diagnostic: it challenges the need to construct every nested Volterra transport. Both compare against always-quadratic and always-quadratic+cubic complete solves. Reducing arithmetic without reducing measured complete time is a cost failure.

These are new-to-TDN experiments. Multiscale products, bilinear/Volterra expansions, quadrature, exponential integrators and nonlinear coarse-graining are established ideas. [Multiwavelet operator learning](https://arxiv.org/abs/2109.13459) already represents operator interactions across scales; origin channels alone do not establish literature novelty. The prior exploration notes document related temporal kernels and closure. No priority claim or claim to defeat a published FNO paper follows from this implementation. A later successful claim needs an up-to-date focused prior-art check and, where necessary, authoritative competitor reproduction.

## Finite budget, execution and statistical gates

The immutable adjacent protocol publishes every unit's scientific timer, CPU/GPU requirement and allocation ceiling before execution. The main portfolio receives no budget transfer and remains unchanged. The discretionary split is 80% architecture pilots and 20% E01/E02; references, diagnostics, evaluation, failures/recovery and reporting are separately enumerated. The launcher must keep the eight-core desktop, 110000 MiB scheduler cap, at most 48 GiB per job, one 24 GiB 4090, project-local venv/caches, existing 18 GiB/75% soft GPU limits and at most 45-minute allocations. A ceiling is not a duration forecast or a bill. Substantial desktop work runs in this isolated checkout and does not alter the source used by active main jobs.

Run mathematical checks, then D01/D05, then D02/D04; other diagnostics refine the interpretation. Smoke is a bounded development/program check. Full is a finite pilot plus separately frozen fresh evaluation; it is not automatically a successful confirmatory experiment. References actually refine and retain uncertainty estimates. A refined same-grid equation is not silently a continuum reference. Fitting/selection inspect development training and validation only; fresh field generation follows the frozen decision.

The three predeclared primary contrasts are channel neural versus global channel fit, scalar conditioner and output-band gain. Confirmation sample planning uses independent validation-parent paired log-error variability and a desired log-effect half-width of `log(1.15)`; Bonferroni accounts for six track-specific primary contrasts (three methods across two equation tracks). A maximum-parent cap binds the run. Insufficient precision, reference quality or optimizer adequacy yields NA and no automatic budget expansion. Related phases, grids, schedules, roughness variants and training seeds stay paired at the parent level. Subgroups are descriptive unless separately locked. The reference-adjusted interval is the bootstrap interval of pessimistic parent log effects, not an interval enclosing every possible true effect; the raw-effect interval is reported separately.

A useful accuracy claim requires the registered error effect without unacceptable maximum-error/coverage regression. A useful solver claim additionally requires the registered accuracy-qualified time margin with repeated randomized paired timings, device synchronization and a complete cost ledger. The baseline's optimization and frozen selection must be adequate. Deployment policy is not created merely because a model can be trained.

## Reading the adjacent atlas

The atlas is separate from the portfolio report and retains exact source hashes and JSON pointers, raw values, unknown values, selected checkpoints and failed units. Every panel states what is better or worse. Ours uses triangles, Theirs circles and analytic/oracle controls squares. Dense loss plots use thin continuous low/high borders and translucent observed-range interiors; missing windows stay gaps. Observed ranges are not confidence intervals, and pooled experiment-row counts are not sample sizes.

Reports distinguish implemented, mathematically checked, CPU-tested, native GPU-tested and scientifically supported results. A plot with no measurement is NA. Costs without an energy measurement or actual monetary rate remain NA. A reference-informed oracle is a representation diagnostic, never a deployment method. Final conclusions answer which limitation was demonstrated, which explanation weakened, whether channels added identifiable useful information, whether learning improved on fitting/analytics and whether it repaid complete cost. Each branch receives continue/change/stop with its evidence; promotion into main TDN remains a separate proposal.
