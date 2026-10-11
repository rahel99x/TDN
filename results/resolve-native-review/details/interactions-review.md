# Native full Resolve R2: finite-step interaction review

The strongest finding is that the heat-filtered leading cubic correction is frequently pointing in the wrong direction, not merely using an imperfect amplitude. A finite-time nested interaction control substantially improves accuracy across this designed panel, but it also has counterexamples and has not established lower complete solver cost.

This is a read-only analysis of the verified canonical `interaction_cases.json`, `interaction_rows.json`, stage summaries and protocols. `interactions-review.py` reproduces the numerical tables in `interactions-review.json`; that JSON includes the SHA-256 of every source artifact used. No new experiment, source edit or model selection was performed for this review.

## Coverage, completion and uncertainty

- **12/12 stages completed**, covering two tracks, grids 32²/64², and three spectral templates; each has two phase variants crossed with nine independently varied diffusion/reaction settings.
- **216 cases and 1,512 method endpoints**, all finite, with no component exceptions or mathematical correctness failures.
- All **216 same-grid and 216 same-N Galerkin teachers accepted**. The projected fine-grid teacher accepted **203/216**, leaving 13 unresolved target triples. All numerical evidence and failures remain present.
- These are **eight distinct continuous fields**, producing 12 unique sampled arrays: four fixed-physical-frequency fields shared across grids and four explicitly grid-relative fields. They come from three designed templates with two phase interventions each. They are **not eight IID draws**, and neither the 216 cases nor the 1,512 endpoints are independent samples. A genuinely independently sampled population count is unavailable by design. No confidence intervals or population success rates are inferred here.
- The methods in R2 run in **CPU FP64**, with common full-output corrections and one endpoint step at horizon 0.12. These results do not establish FP32 GPU performance, rollout performance or validation-selected accuracy–cost superiority.
- Stage numerical elapsed time totals **1,127.54 s (18 min 47.5 s)**, of which reference generation accounts for **1,092.27 s**. Single diagnostic calls share precomputed components; their timing is not a fair deployment benchmark.

A win below means that the control error minus estimated reference RMS uncertainty exceeds the candidate error plus that uncertainty. These are descriptive uncertainty-separated observations on paired probes. The refinement uncertainty is an estimate, not a certificate. The error norm is sampled nodal RMS/max, not a Nyquist-weighted continuous physical L2 norm.

The 13 unresolved references are all **grid-relative, weak-diffusion (`hλ_active = 0.1`) cases**: eight continuum cases and five discrete cases, mostly at `hr = 3` and some at `hr = 1`. Both fine temporal solves accepted; their projected spatial refinement exceeded the frozen `2e-8` maximum uncertainty allowance. For example, the continuum 32² phase-zero `hr = 3` case has projected spatial difference `3.926e-7` and total maximum uncertainty `3.973e-7`. They do not support accepted continuum accuracy claims. Their independently accepted same-grid targets still support same-grid comparisons.

## Controlled accuracy comparisons

RMS ratios are `GL4 error / candidate error`; larger values favor the candidate. A median across this fixed panel is descriptive, not a universal model ranking. Counts use all 216 accepted same-grid targets. “Joint pass” uses the existing `2e-5` RMS **and** maximum thresholds, adding estimated uncertainty to both errors.

| Method | Wins / losses / unresolved versus GL4 | Median RMS ratio | Joint passes / 216 | Largest sampled maximum error |
|---|---:|---:|---:|---:|
| Theirs: DF Strang | 0 / 216 / 0 | 0.0490 | 76 | 5.883e-4 |
| Theirs: one-step ETDRK4 | 24 / 192 / 0 | 0.0164 | 64 | 8.903e-2 |
| Analytic control: GL2 | 58 / 158 / 0 | 0.551 | 132 | 2.091e-4 |
| Analytic control: GL4 | self-comparison | 1.000 | 168 | 1.470e-4 |
| Ours numerical: GL2 + raw leading cubic | 44 / 172 / 0 | 0.0879 | 92 | 1.004e-2 |
| Ours numerical: GL2 + filtered leading cubic | 48 / 168 / 0 | 0.216 | 104 | 3.024e-4 |
| Analytic control: GL4 + nested cubic | **204 / 8 / 4** | **6.360** | **212** | **3.439e-5** |

All 1,512 predictions stayed within the sampled physical interval and showed no energy increase greater than `1e-12`. These reassuring diagnostics do **not** establish accuracy: one-step ETDRK4 still reached about 0.089 maximum error. Its one-step failure is not a verdict on a classical solver allowed to use its own smaller-step accuracy–cost frontier.

The filtered cubic's wins have a sharply delimited location: **all 48 occur at `hλ_active = 0.1` and `hr` equal to 0.1 or 1**. It loses every comparison at `hr = 3`, or `hλ_active >= 1`.

| `hλ_active` | `hr` | Filtered-leading-cubic median GL4/candidate ratio | Nested-cubic median GL4/candidate ratio |
|---:|---:|---:|---:|
| 0.1 | 0.1 | 3.058 | 55.301 |
| 0.1 | 1 | 3.277 | 5.356 |
| 0.1 | 3 | 0.166 | 5.918 |
| 1 | 0.1 | 0.0746 | 47.193 |
| 1 | 1 | 0.487 | 18.324 |
| 1 | 3 | 0.365 | 6.968 |
| 5 | 0.1 | 0.0549 | 1.033 |
| 5 | 1 | 0.0810 | 1.385 |
| 5 | 3 | 0.172 | 2.289 |

Each table cell contains 24 repeated field/grid/track cases, not 24 independent fields. The two tracks also need not have identical diffusivity: the experiment matches dimensionless active stiffness using each declared track/grid's generator. Within every intervention, the same continuous field is retained and diffusion and reaction are varied independently. Finite-grid FD differences across grids should therefore not be interpreted as a fixed-κ convergence study.

A strong counterexample is the continuum overlapping-triad phase-one case at `hλ_active = 5`, `hr = 3`, appearing on both grids. GL4 RMS is `1.6573e-6`; filtered-leading cubic gives `3.4538e-5`, **20.84× worse**; raw-leading cubic gives `1.1261e-3`, **679.47× worse**. The repeated grids are the same physical field and essentially the same outcome, not independent replications.

## Direction, not only amplitude

Let `R = teacher − DF`, let `Q` be the quadratic correction and let `C` be the filtered cubic basis. The desired additional correction after the quadratic term is `R − Q`. Its signed alignment with `C` is reconstructed independently from the logged residual projection, normalized Gram matrix and endpoint error norms:

\[
\cos(C,R-Q)=\frac{\cos(C,R)\,\|R\|-\cos(C,Q)\,\|Q\|}{\|R-Q\|}.
\]

All norms can be RMS because the common number-of-points factor cancels. `||R||` is the measured DF RMS error; `||R−Q||` is the measured GL2 or GL4 RMS error. This uses the **refined teacher's actual missing defect**, not an assumption that the nested cubic is exact.

The filtered leading cubic has negative alignment with the missing defect in **132/216 cases after GL2** and **149/216 after GL4**. The latter comparison diagnoses the basis even when the more accurate quadratic correction is present. Examples of median cosine to the missing GL4 defect are **−0.958** at `hλ_active = 0.1, hr = 3`, **−0.890** at `(1,1)`, and **−0.972** at `(1,3)`.

These are not angles formed from defects at the estimated noise floor:

- All 216 GL4 defects are at least **93.94×** the estimated reference RMS uncertainty; the corresponding GL2 minimum is **640.59×**.
- The minimum filtered-basis RMS is **4.94e-10**, above a conservative `1e-12` absolute basis threshold and approximately 17,380 times `128 × FP64 epsilon`.
- Requiring defect RMS greater than ten times uncertainty and basis RMS greater than `1e-12` retains **all 149/132 negative-angle cases**.
- The stronger sign criterion `cos(C, defect) < −uncertainty / ||defect||` also retains all 149/132. This means the estimated target uncertainty is too small to explain the observed opposing projection under the reported estimate; it is not a rigorous bound on the unknown exact teacher error.

For `E(g)=||R−Q−gC||²`, the derivative at zero is `E'(0)=−2<C,R−Q>`. Negative alignment makes that derivative positive, and the squared-error function is convex in `g`. Thus, on the measured target, adding **any positive amount of this direction while fixing Q** worsens RMS when the inner product is negative. Jointly changing the quadratic gain can change the remaining defect, so this statement is not a theorem about every jointly conditioned architecture.

This gives a concrete reason to **stop treating a positive-gain-only leading-cubic conditioner as a broadly applicable upgrade**. Retain its weak-diffusion, nonstiff successes, but do not fix the failure solely by increasing neural capacity or training duration.

## What the asymptotic formula does—and does not—say

With `v = u − mean(u)`, diffusion generator `A`, and the target's symmetric quadratic product `B`, the leading cubic defect is

\[
C_{3,0}=\frac{r^2h^3}{6}\{B(Av,B(v,v))-2B(v,B(v,Av))+2B(v,AB(v,v))-AB(v,B(v,v))\}.
\]

For the nodal product this reduces to

\[
\frac{r^2h^3}{6}[-v^2Av+2vA(v^2)-A(v^3)].
\]

Products in a truncated Galerkin algebra must retain their nesting; importing a continuum differential product identity into that algebra would change the equation. This formula is the leading **fixed-grid, small-time cubic-amplitude term**, not the full finite-time cubic interaction kernel and not a stiff-order guarantee. A heat filter preserves the leading small-step coefficient while changing higher terms heuristically. It does not resum all homogeneous reaction responses or the nested causal transport geometry.

The relevant small parameters include `hA` on active interactions and `hr`, not only the nominal horizon 0.12. The present grid reaches active diffusion number 5 and reaction number 3, where truncating that temporal expansion is not justified uniformly. Sign reversal of a finite-time response there does not disprove the original small-step derivation.

The filtered basis opposes the six-node nested cubic in 157/216 cases. At `(hλ_active,hr)=(0.1,3)`, its median norm is 5.89 times the nested basis and their median cosine is −0.975; at `(5,3)`, raw leading-cubic norm is approximately 275 times the nested basis at the median. These are additional finite-quadrature diagnostics: **the six-node nested cubic itself was not separately quadrature-converged and must not be called the exact cubic coefficient**.

## GL4 and nested cubic also have limits

At diffusion number 5, the GL4-versus-GL8 quadratic difference is a median **0.778 times the complete GL4 error**, with range **0.175–1.936**. Quadratic quadrature error remains material even after moving from two to four nodes. GL8-versus-GL16 differences are retained, but successive-rule agreement is not an integration-error certificate.

All eight uncertainty-separated nested-cubic losses occur in the **aligned-harmonic, phase-zero, diffusion-number-five** cases: `hr = 0.1` or `hr = 1`, across both tracks and grids. At `hr = 1`, continuum GL4 RMS is `1.7260e-6`, while GL4 + nested cubic is `3.4390e-6`, **1.99× worse**. The GL4-versus-GL8 quadratic difference is `3.3421e-6`, comparable to these errors. The nested direction points against the actual missing GL4 defect in precisely eight cases. This is consistent with removing favorable cancellation between the inaccurate quadratic rule and omitted interaction terms. It does not prove that nested-cubic quadrature is converged or isolate the entire loss to one mechanism.

The four remaining unresolved nested comparisons are overlapping-triad phase-one cases at diffusion number 5 and `hr = 0.1`, repeated across track/grid. The apparent improvement is only about 0.032%, below estimated reference uncertainty. They must remain unresolved.

## Spatial, temporal and closure errors are different learning problems

The logged decomposition is the vector identity

\[
p-y_f=(p-y_d)+(y_d-y_g)+(y_g-y_f),
\]

where `y_d` solves the declared same-grid equation, `y_g` solves the same-N Galerkin equation, and `y_f` is the projected spatially refined Galerkin estimate. The three terms represent numerical temporal error, FD-versus-Galerkin discretization error, and unresolved-scale closure difference. Squared norms require cross inner products; summing component norms would conceal cancellation. Maximum vector identity residual is zero and maximum squared-RMS reconstruction residual is `6.51e-19`.

On accepted target triples:

| Track/template | Accepted cases | Discretization RMS range | Closure RMS range | Closure larger than summed uncertainty |
|---|---:|---:|---:|---:|
| Discrete / overlapping triads | 36 | 9.47e-7–4.61e-5 | 5.76e-17–1.24e-15 | 0/36 |
| Discrete / aligned harmonics | 36 | 1.94e-6–1.41e-4 | 6.36e-17–2.76e-13 | 0/36 |
| Discrete / grid-relative | 31 | 3.62e-5–7.22e-3 | 1.99e-10–2.15e-5 | 27/31 |
| Continuum / overlapping triads | 36 | zero by definition | 5.98e-17–9.22e-16 | 0/36 |
| Continuum / aligned harmonics | 36 | zero by definition | 6.47e-17–2.99e-13 | 0/36 |
| Continuum / grid-relative | 28 | zero by definition | 9.81e-10–7.31e-6 | 24/28 |

Smooth fixed-frequency fields show no resolved closure signal. Grid-relative inputs do, and FD discretization can be much larger still. A learned correction trained against fine continuum targets may compensate spatial discretization or closure rather than improve the specified discrete integrator. Such an outcome must not be attributed entirely to temporal architecture. Unresolved fine references are excluded from the table rather than treated as favorable evidence.

## Conditioning and oracle limits

The new design fixes the old forced orthogonality: Q2/filtered-C3 normalized-pair condition numbers range **1.15–2.74** for overlapping triads, **1.43–7.43** for aligned harmonics, and **1.16–1.92** for grid-relative probes. The harmonic cosine reaches 0.964. This is useful designed correlation, although it is not extreme singularity or generic identifiability across unknown fields.

The teacher-informed, unconstrained Q2/C3 oracle requests a negative cubic gain in **119/216 cases** and lies outside the historical `[0.25,1.75]²` box in **159/216**. Cubic coefficients can range from roughly −89.6 to +55.8. These are per-case label-informed RMS optima, not deployable fitted laws. Large coefficients can compensate quadratic quadrature error while the filtered cubic has become tiny. A well-conditioned *column-normalized* Gram matrix does not remove that absolute scale problem. Simply widening the gain bounds would not establish a safe or identifiable physical correction.

## Research judgment and next decisive controls

1. **Retain finite-time interactions as the primary grounded path.** The nested control's 204 wins and 212/216 joint passes establish substantial accuracy headroom on this designed panel. Require measured complete cost before making a practical advantage claim.
2. **Refine the quadratic and nested rules together.** Compare GL4/GL8 and independently converged nested cubic on the aligned-harmonic counterexamples. Do not attribute every gain or loss to cubic order while leaving quadrature error comparable to the residual.
3. **Replace the broad positive-gain leading-cubic hypothesis with a finite-time representation hypothesis.** Candidate representations must retain background reaction response, causal time ordering and the changing direction of interaction corrections. A compact approximation or distillation of the finite-time kernel is more defensible than multiplying the current leading term by a larger network.
4. **Separate same-grid dynamics from closure.** Keep the fixed-frequency panel, and add independently accepted fine references for the unresolved grid-relative cases before extending continuum conclusions. Any learned closure needs its own controls and target definition.
5. **Do not expand the claim beyond this development panel.** There is one mean, one amplitude, one endpoint horizon, two designed phases, no learned method in R2, no long rollout and no held-out statistical confirmation. Weak-diffusion benefits and high-stiffness failures both remain visible. New frozen selection and fresh independently drawn fields are required for a comparative scientific claim.
