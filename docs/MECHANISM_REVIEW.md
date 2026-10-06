# CARC mechanism audit: evidence and next direction

The evidence favors **nonlocal nonlinear interaction response with bounded
state corrections** as the next direction. It identifies restrictions in the
current clock coordinate and local gate, and a promising temporal oracle. It
does not select a trained architecture or establish an advantage over FNO.

This review concerns CPU job **12678403**, workflow
`carc-mechanisms-20261006T010026502981Z`, source `d1992d3`. The machine-readable
record is [carc-mechanism-audit-review.json](../results/carc-mechanism-audit-review.json).
The original archive and all scientific artifacts were preserved.

## Verified execution and reporting

- All seven sealed scientific artifact hashes, the completion marker, protocol,
  configuration, execution/dependency fingerprints and software-record hash verify.
- All **103 allocated tests passed**. The audit completed **600/600 cases**:
  155 passed controls, 416 observations and 29 expected limitations.
- Numerical time was **2.810 seconds**; the whole allocated worker took
  **16.792 seconds**, excluding queue time. There was no training or GPU work.
- Every one of **10,173 scalar metrics** matches its original JSON value,
  case, inputs and note. Native unchanged Tower validated 19 output contracts,
  nine live metric records and 20 log entries. Independent schema validation
  passed 13 objects.
- Tower's three reported omissions are intentional exclusions of pytest cache,
  pytest work and the report tree. No scientific metric is missing.

Cloud and CARC case identities, inputs, kinds and outcomes agree. The values
are not bitwise identical: the known rank-one DC dictionary has unstable huge
condition-number estimates, while FP32 and small residual diagnostics also
vary. These differences did not change any classification. The repeated
`experiment: COMPLETED` status line displays worker state and experiment
summary; the logs contain one tests stage and one experiment stage.

## What the coordinate/context probes establish

All 40 distinct coupled reference cases were accepted.

**The clock coordinate misses a needed boundary-order direction.** In the
two-site example with initial state `[0, 0.7]`, diffusion 0.05 and reaction rate
2, the receiver needs a correction `0.0653333 h^3 + O(h^4)`. Its physical
intermediate is `O(h)`, so an `O(h^3)` clock shift changes the state only at
`O(h^4)`. Both local and global clock variants therefore have zero cubic state
correction in this example. Capacity and additive coordinates can reach that
direction, although the prescribed global head overshoots the required cubic
coefficient by about fourteenfold.

**A local gate can exclude needed remote corrections.** All 24 remote-bump
cases have a zero receiver gate despite resolved required clock shifts from
`2.31e-5` to `4.94e-2`. These shifts are computed after accounting for the
already-global physical diffusion intermediate. This is not a claim that the
complete split solver has a local receptive field.

A global scalar gate removes the exclusion but cannot encode spatial layout.
For diffusion 0.03, reaction 1 and time 0.1, bumps at distances three and five
have the same global gate, 0.193548, but require shifts 0.000630703 and
0.0000231243. Nonlocal support and useful nonlocal information are separate
ablation questions.

**Capacity preserves bounds but can distort the leading coefficient.** For
available directional capacity `m` and raw state increment `d`,

\[
P(v,d)=v+\frac{m d}{m+|d|}.
\]

When `m=0.4 h^3` and `|d|=0.2 h^3`, the retained correction is exactly two
thirds of the requested one at every tested step. The map is generally C1,
not C2, at zero correction. Pre/post-reaction capacity placement shares the
same cubic coefficient here and differs at fourth order; placement is a
secondary ablation.

Fixed coefficients do not rank trained models. Global post-capacity worsened
two-site RMS error in all 16 cases, with a median 9.86 times split error;
global clock improved four cases but had a median 1.66 times split error.
The same raw coefficients are not equal state corrections across coordinates.
These observations motivate calibrated, matched comparisons instead of
promoting the capacity model based solely on reachability.

## Temporal response is the strongest positive lead

Each of 24 fixed regime/pair cases fits its own coefficients using six fitting
times. Validation and four test times are disjoint from fitting. The last test
time, 0.4, extrapolates beyond the largest fitting time, 0.3.

The following error divides each case's maximum held-out quadratic-defect
coefficient error by its maximum true defect amplitude across sampled times.
It is not pointwise relative error of a full PDE solution.

| Representation | Median normalized error | Worst normalized error | Lowest-error cases |
|---|---:|---:|---:|
| Scalar time basis | 0.672 | 436.28 | 0/24 |
| Transported time basis | 0.795 | 43.72 | 1/24 |
| Output-frequency phi basis | 0.238 | 272.46 | 2/24 |
| Reaction/pair-aware oracle | 0.151 | 0.306 | 21/24 |

For the three interpolating test times alone, the pair-aware oracle is lowest
in 14/24 cases and has median normalized error 0.00757; output-phi is 0.01570.
Thus the signal extends beyond the extrapolation test, but is not universal.
For diffusion 0.02, reaction 6 and pair `(3,-1)`, output-phi wins with 0.0184
versus pair-aware 0.2718.

The pair-aware oracle receives explicit generating-pair and background
information. Its coefficients are fit separately for each case. The scalar
basis fits a state defect directly and is not the full deployed reaction-clock
model. Also, the transported DC dictionary has only one effective coefficient
in all four DC cases. These are representation probes, not fair trained-solver
or FNO comparisons, and not independent statistical replications.

The quadratic kernel contains

\[
J_c(h)J_c(s)e^{-\kappa\lambda_{p+q}(h-s)}
e^{-\kappa(\lambda_p+\lambda_q)s}.
\]

Input-pair diffusion and reaction sensitivity both affect the response. Pairs
`(1,3)` and `(2,2)` share output mode four but have different temporal responses.
Merely assigning a phi filter to output frequency misses this information.
The worst phi example has a well-conditioned fit; its failure is not explained
by poor least-squares conditioning. FP32 evaluation disagreements are much
smaller than these representation errors, although cancellation and stable
small-step evaluation still need attention.

## Next proposed CPU experiment

First replace the oracle's pair labels with a branch computed from the input
field. This proposal is **not implemented by this review**.

Let `c=mean(u)`, `v=u-c`, `E_t=exp(tA)` and `J_c(t)=R'_t(c)`, where `R` is the
logistic reaction flow. The quadratic coupled response is

\[
Q_h(v)=-rJ_c(h)\int_0^h J_c(s)E_{h-s}[(E_s v)^2]\,ds.
\]

Strang already contains a quadratic response. Subtract it before applying any
correction. An equivalent anchored defect is

\[
d_h=-rJ_c(h)\left[\int_0^h J_c(s)(F_s-B)\,ds+W_2(B-C)\right],
\]

where `F_s=E_{h-s}[(E_s v)^2]`, `B=E_h(v^2)`, `C=(E_h v)^2`, and
`W_2=integral_(h/2)^h J_c(s) ds`. Compute that last weight analytically. For
positive reaction rate, its antiderivative from zero is

\[
W(t)=\frac{1-e^{-rt}}{r[c+(1-c)e^{-rt}]}.
\]

Use stable exponential evaluation and an explicit zero-reaction branch.
Compare three- and five-node Gauss–Legendre rules for the anchored integral.
The differences vanish in the diffusion-free limit; uniform inputs vanish
through `v=0`. At fixed grid with fixed smooth parameters the quadrature errors
are seventh and eleventh order, preserving the cubic defect anchor. This
does not give mesh-uniform or stiff accuracy, and remaining cancellation
requires FP32 tests.

The construction is exact for the quadratic variation around a constant
background. On a finite-amplitude field it is a truncated approximation.
Do not multiply this already-quadratic branch by the existing quadratic
variation gate: that would suppress its second variation to fourth order in
amplitude. Compare finite, nonsingular gains or use its intrinsic zero limits.

The bounded CPU screen should retain the scalar and output-phi controls, use
unseen signed pairs, mixed fields, several mean states, independently crossed
diffusion/reaction rates and genuinely held-out times. Cross additive versus
capacity application, test near-boundary states and all exact limits, retain
the observed counterexamples, and count every FFT and quadrature evaluation.
Success means better finite-amplitude reference error under explicit work
accounting, not merely better perturbative oracle fit.

Then run a matched clock/post-capacity by local/nonlocal-context panel, with a
global-gate-only ablation and an additive control. Follow it with scalar,
output-phi and runtime-interaction temporal branches using identical data,
selection rules, paired seeds, encoder budgets and complete inference costs.
Do not equate a scalar global gate with a learned multiscale encoder.

## Where the remaining mechanisms fit

| Mechanism | Finding and priority |
|---|---|
| Anti-aliasing | Removes the artificial resolved-band harmonic: RMS 0.353553 to below 9e-16. It discards unresolved physical energy 0.125. Add a later multiscale/resolution ablation, with separate common-band and converged-PDE tracks. Filtering a bounded output can reintroduce overshoot. |
| Factorization | Oblique multiplier error is 66.44% at rank one and 17.56% at rank three. Preserve cross-axis capacity; investigate factorization as a later cost tradeoff. These linear projections do not bound deep F-FNO expressivity. |
| Composition/PDE objectives | Identity has zero composition discrepancy but true RD error 0.1711. Compare data, data+composition, data+PDE and both only after the base architecture; never promote by composition alone. |
| Equation-specific constraints | RD reaction increases mean by 0.192436 in its control, so total mass conservation is wrong. Retain source-aware balance. Flux and projection architectures require separately implemented conservation-law/incompressible benchmarks. |
| Adaptive stepping | A wrong generator has estimated error 3.7e-17 but true error 0.1484. Defer neural adaptation until accuracy and estimator bias are understood; charge rejected work. The scalar demonstration used 11 accepts, three rejects and 84 RHS calls. |

No long A100 training run follows from this review. The immediate proposal is
a bounded CPU mechanism test; matched training and larger FNO comparisons are
separate later experiments.
