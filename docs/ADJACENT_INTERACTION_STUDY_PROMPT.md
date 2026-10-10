# TDN Adjacent Interaction Study — Research Prompt

This should remain an **adjacent research study, with its own protocol, budget, results, and conclusions**. It should inform TDN without changing the original objective or interfering with the campaign already running.

The work is feasible, with three qualifications:

- The eight diagnostic programs are practical. Credible long-rollout references and well-trained neural competitors will require more compute than the initial diagnostics.
- The interaction-channel architecture is implementable. Its main uncertainty is whether the additional information improves accuracy enough to repay the extra computation.
- “2.1-D through 2.9-D” can describe ideal rough surfaces whose finite approximations become test inputs. The requested 3.0 endpoint requires separate treatment. None of these labels automatically describes the neural network’s dimension.

The current [model implementation](../tdn/analysis/portfolio/models.py) and [portfolio review](PORTFOLIO_INITIAL_REVIEW.md) already include normalized quadrature, richer scalar conditioning, and output-band gains. The new architectural question is whether retaining **which input scales produced an interaction** adds something useful beyond those controls.

---

Act as a numerical analyst, scientific machine-learning researcher, and computational experimentalist. Develop an **adjacent experimental program for TDN** that investigates representation limits, multiscale nonlinear interactions, rough input fields, and temporal reuse.

The motivating question is:

> When does preserving the origin, sign, scale, and temporal response of nonlinear interactions provide useful information that existing corrections or neural operators cannot exploit as accurately or economically?

Treat this as a hypothesis to investigate. A useful outcome may be an improved learned correction, a simpler numerical rule, a clear limitation, or evidence that the proposed representation does not justify its cost.

**Preserve the main project.** TDN’s original research objective, temporal-model ambitions, active portfolio, frozen comparisons, and existing budgets remain authoritative. This adjacent study must not redefine the project, replace its main architecture, change an active protocol, or consume compute already reserved for the main campaign.

1. **Establish the boundary and reconstruct the relevant evidence.**

   Begin by reading the original project specification, current implementation, research reviews, hypothesis register, and available measurements. Record the original objective and explain how this study supports it.

   Inspect the actual implementations before deciding what needs to be built. In particular, verify the historical rank1 correction, normalized two- and four-node controls, scalar and affine fitted controls, node fitting, state-dependent conditioning, output-band gains, analytic quadratic+cubic correction, and existing temporal-reuse prototypes.

   Produce a compact evidence map identifying established observations, confounded comparisons, negative findings, and untested hypotheses. Do not treat pending results as available evidence.

   Work in an isolated checkout outside the source tree used by active jobs. Give this study separate configurations, run identifiers, manifests, checkpoints, results, and reports. Reuse existing artifacts only through explicit provenance and compatibility checks.

   Previously inspected confirmation data may support development diagnostics. New confirmatory claims require fresh data and a separately frozen protocol.

   Implement all eight diagnostic programs below with bounded smoke cases. Expand only the comparisons that answer a consequential question.

2. **D01 — Measure the best possible correction within the current representation.**

   For a specified field, equation, grid, step, physical backbone, node rule, and cutoff, define

   \[
   d=u_{\mathrm{reference}}-S_h(u),
   \qquad
   q=Q_h(u).
   \]

   For squared \(L^2\) error, compute the unconstrained scalar optimum

   \[
   a^*=\frac{\langle q,d\rangle}{\langle q,q\rangle},
   \qquad \langle q,q\rangle>0.
   \]

   Also compute the optimum within the architecture’s actual admissible gain range. Read those bounds from the implementation; do not assume all variants share them.

   Compare the uncorrected backbone, normalized fixed correction, fitted global amplitude, deployed conditioner, bounded scalar oracle, and unrestricted scalar oracle.

   Extend the diagnostic to output-band gains, interaction-channel gains, and a quadratic+cubic basis. Use stable least squares or constrained optimization, and report conditioning, effective rank, coefficient sensitivity, and reference uncertainty.

   Record absolute residual error, improvement relative to the backbone, spatial and spectral residual patterns, maximum error, and mean error. A small RMS residual does not guarantee a small maximum error.

   Handle vanishing corrections explicitly. When \(\|q\|\) or the desired defect is near numerical or reference uncertainty, mark ratios and inferred gains as unresolved.

   The projection formula optimizes squared \(L^2\) error. If training uses a combined RMS/maximum-error objective, evaluate that objective separately rather than calling the projection optimal for it.

   **Decision:** a large scalar-oracle residual supports changing the spatial correction basis. A small oracle residual combined with poor learned performance points toward features, constraints, data, or optimization.

   These are reference-informed diagnostics, not deployable methods.

3. **D02 — Test whether the conditioner receives enough information.**

   Construct paired states that closely match the current conditioner’s features while differing in spectral arrangement, phase, localization, or distribution across scales.

   Verify feature matching numerically in the actual target equation. Where possible, hold physical parameters, step, grid, mean, and fluctuation amplitude fixed.

   The physical branch still receives the complete field. Therefore, different future solutions alone do not demonstrate conditioner insufficiency.

   Instead, determine whether feature-matched states require materially different gains or node choices. Compare the best shared response against separately optimized responses and the deployed model.

   Report acceptable gain intervals, not just a single optimizer output: apparently different optima may belong to nearly flat objectives. Require disagreement that exceeds reference uncertainty and numerical sensitivity.

   Compare the original features with existing richer features, inexpensive spectral moments, selected band energies, and phase-sensitive interaction summaries. Charge feature extraction.

   **Decision:** enlarge the conditioner’s information only when feature ambiguity explains a measurable part of the error. More features without this evidence remain exploratory.

4. **D03 — Map grid, timestep, duration, and cost separately.**

   Build controlled experiments that distinguish four changes: refining the same physical field, adding finer-scale content, changing integration step, and extending physical simulation duration.

   Keep discrete-equation and continuum-reference tracks separate. Use identical continuous parent fields across paired grids, with correct restriction or projection. When increasing bandwidth, label it as a changed workload.

   Vary \(N\), \(h\), and final time \(T\) through targeted slices rather than a full Cartesian product. Record the exact step sequence and final endpoint.

   Give applicable methods the same inexpensive one-, two-, four-, and eight-step opportunities. Permit further refinement and suitable adaptive classical methods where necessary to reach tight tolerances; an arbitrary schedule ceiling must not manufacture an advantage.

   Measure endpoint and trajectory error, convergence behavior, failure frequency, latency, peak memory, and accuracy-qualified total solve cost. Distinguish physical-frequency cutoffs from cutoffs fixed as a fraction of grid resolution.

   Use a small pilot to locate transitions, then freeze the regions selected for independent confirmation.

   **Decision:** identify a reproducible region of useful performance, or show that spatial error, temporal error, or execution overhead removes the available advantage.

5. **D04 — Test signed high-frequency interactions.**

   Construct mode pairs whose products contribute to retained low frequencies. Sweep input frequencies, separation, phase, orientation, amplitude, diffusion strength, and requested time.

   Include constructive and destructive interference, mixed spectra, and localized examples. Compare fields with similar energy spectra but different phase-dependent nonlinear responses.

   Include normalized analytic correction, premature input compression, output-only compression, the current learned correction, output-band adaptation, the proposed interaction channels, and credible FNO controls with their local paths intact.

   Inspect signed complex Fourier coefficients and phase errors as well as aggregate norms. Near a vanishing target coefficient, use absolute error and an explicit scale floor.

   Keep the declared nodal product and dealiased Galerkin product distinct. Aliasing can be part of a specified discrete equation; it must not silently enter a continuum comparison.

   **Decision:** establish whether performance follows the proposed information-preservation mechanism and whether it survives a meaningful competing architecture.

   Do not assume that FNO loses all high-frequency information. Its local paths and nonlinearities must be accounted for.

6. **D05 — Separate quadrature error from missing interaction order.**

   For controlled states \(u=m+\epsilon v\), vary fluctuation amplitude \(\epsilon\) and step \(h\).

   Compare normalized two-node, four-node, and sufficiently refined quadrature of the same quadratic formula. Separately compare quadratic and quadratic+cubic corrections, with matched compression where attribution requires it.

   Use positive and negative perturbations around the same admissible background to help isolate even and odd amplitude contributions. Check observed scaling only over ranges above roundoff and reference uncertainty.

   Examine residual directions and spectra. Determine whether a cubic contribution supplies a new spatial direction or mainly rescales an existing quadratic one.

   Keep amplitude order, timestep order, and quadrature convergence separate. State which quantity is held fixed in each experiment.

   **Decision:** additional nodes are justified when quadrature error matters. Higher-order structure is justified when refining the quadratic integral leaves the dominant error intact. Learning is justified only if it improves on the appropriate analytic and fitted controls.

7. **D06 — Measure transient fidelity and autonomous rollout behavior.**

   Evaluate repeated application to the model’s own outputs over short, intermediate, and long horizons. Separate these autonomous results from teacher-forced one-step tests.

   Measure worst trajectory error, time-integrated error, mean and variance evolution, selected spectral modes, threshold-crossing times, relaxation rates, physical-range violations, and failure frequency.

   Logistic reaction does not conserve the mean; test the correct mean dynamics rather than imposing false conservation.

   Compare accurate transient evolution with merely reaching the correct equilibrium. Diffusion and reaction may erase early errors, so a good final endpoint cannot substitute for trajectory evidence.

   Sample early times densely enough to observe the decay and interaction of fine scales. At late times, report absolute errors alongside relative measures that can become misleading near equilibrium.

   Include unequal steps, unfamiliar step sequences, perturbations of initial conditions, and horizons beyond those used in training.

   **Decision:** distinguish accurate stable evolution, stable but inaccurate evolution, and unstable evolution. Report each with its denominator.

8. **D07 — Test genuine temporal amortization.**

   Investigate whether the original temporal-model idea becomes useful when many outputs are requested from the same state.

   Separate repeated queries from one state, dense outputs along one trajectory, parameter queries, and autonomous rollouts that require refreshing the representation. These workloads have different reuse opportunities.

   Compare the existing temporal model where applicable, direct physical corrections, reusable interaction encodings, cached analytic kernels, and classical dense-output methods.

   Measure encoding, operator preparation, compilation, refresh, query, reconstruction, transfer, and storage costs. Give competitors equivalent caching and batching opportunities.

   For a simple repeated-query cost model,

   \[
   C_{\mathrm{ours}}(Q)=E+Qq,
   \qquad
   C_{\mathrm{direct}}(Q)=Qd,
   \]

   the nominal break-even count is

   \[
   Q>\frac{E}{d-q}
   \]

   only when \(d>q\). Validate this model against actual multi-query measurements, and compare against the complete classical build-and-query cost as well.

   Define cache validity precisely. A changed state, operator, parameter, grid, precision, or time interval can invalidate some or all cached quantities.

   Evaluate approximation of the interaction kernel separately from accuracy of the complete PDE solution.

   **Decision:** advance only when the accuracy-qualified break-even point lies within a realistic workload. Prior cost failures require a changed computational premise before expansion.

9. **D08 — Include competitor-favorable and assumption-breaking workloads.**

   Retain smooth fields, weak nonlinear interactions, nearly constant states, small steps, stringent tolerances, single-field latency, larger batches, and sufficiently resourced neural training.

   Stress TDN’s own assumptions through strong finite-amplitude effects, localized structures, phase cancellation, unfamiliar spectral arrangements, and cases where low output modes alone are insufficient.

   Audit competing implementations, supervision, normalization, optimization, capacity, selected checkpoints, and validation curves. A failed short training run cannot establish architectural inferiority.

   Distinguish equal-data, equal-training-compute, equal-inference-cost, and best validation-selected frontier comparisons. Equal parameter counts or update counts are insufficient definitions of fairness.

   Include a relevant multiscale neural comparator if the new claim concerns multiscale representation and the budget permits a credible implementation.

   **Decision:** publish where each method wins, loses, or remains unresolved. The intended contribution may occupy a narrow domain.

10. **Build the controlled \(2.X\)-D input benchmark.**

    Interpret the requested sweep as

    \[
    D=2+\alpha,
    \qquad
    \alpha\in\{0.1,0.2,\ldots,1.0\}.
    \]

    For \(0<\alpha<1\), set \(H=1-\alpha\). A canonical mathematical reference is

    \[
    W_H(x)=\sum_{j=0}^{\infty}
    2^{-Hj}\cos(2\pi\,2^jx).
    \]

    Its graph has dimension \(2-H\); extending it along an independent interval gives a surface of dimension \(3-H=2+\alpha\). Use the established [Weierstrass graph-dimension result](https://arxiv.org/abs/1505.03986), and state its assumptions.

    Implement finite truncations for numerical work. A finite trigonometric sum is smooth: describe these as finite-resolution approximations to the ideal rough surface, with measured scaling over a declared range.

    At \(\alpha=1\), the construction gives \(H=0\), and its infinite series no longer converges in the same way. Include the requested endpoint as a separately labeled finite-bandwidth stress case. This limitation concerns this construction; it is not a claim that dimension-three graphs are mathematically impossible.

    Use the anisotropic ridge construction for mathematical checks. Performance experiments must also include genuinely two-dimensional, independently generated multiscale fields. Justify any dimension interpretation for each generator rather than transferring the ridge theorem automatically.

    Control mean, fluctuation amplitude, physical range, bandwidth, orientation, and phase statistics. Avoid clipping that silently changes the spectrum. Report any quantities that cannot be matched simultaneously.

    Keep fixed-bandwidth grid refinement separate from increasing the number of represented scales. Use physical wavenumbers and paired parent identities.

    Compare randomized and structured phases at matched spectral power. Global translations of one field must not masquerade as many independent examples.

    Record empirical roughness, shell energies, interaction strength, and their evolution. Diffusion can rapidly smooth rough initial data; nominal initial dimension alone may not predict later difficulty.

    The primary question is:

    > As fine-scale content persists across more scales, does preserving nonlinear interactions improve accuracy per unit of complete computation, and which measurable property explains the change?

11. **Implement the interaction representation inside the physical correction.**

    Preserve the distinction between an additional information channel and an additional spatial dimension.

    Write

    \[
    u=m+v,
    \qquad
    v=v_L+v_H.
    \]

    At fixed background \(m\), physics, timestep, quadrature, and spatial operator, denote the homogeneous quadratic correction by \(\mathcal Q_{m,h}(v)\).

    Define

    \[
    C_{LL}=\mathcal Q_{m,h}(v_L),
    \]

    \[
    C_{HH}=\mathcal Q_{m,h}(v_H),
    \]

    \[
    C_{LH}
    =\mathcal Q_{m,h}(v_L+v_H)
    -\mathcal Q_{m,h}(v_L)
    -\mathcal Q_{m,h}(v_H).
    \]

    These must satisfy

    \[
    C_{LL}+C_{LH}+C_{HH}
    =\mathcal Q_{m,h}(v).
    \]

    Hold the same background fixed in every term. Recomputing a different background for each channel changes the intended decomposition.

    First implement this identity as a transparent reference. Then derive an efficient implementation using shared transported fields and products while retaining the correction’s physical subtractions.

    The spectral representation may be

    \[
    Z(k,c)=\mathcal F(C_c)(k),
    \qquad c\in\{LL,LH,HH\}.
    \]

    Its extra axis identifies interaction origin. It is not a third spatial coordinate, and it does not justify applying an FFT along the channel axis.

    A minimal learned version is

    \[
    u_{n+1}
    =
    S_h(u_n)
    +
    P_K\!\left[
    g_{LL}C_{LL}
    +g_{LH}C_{LH}
    +g_{HH}C_{HH}
    \right].
    \]

    Initialize all gains to one. The fixed-recombination model must reproduce the normalized baseline before training.

    Compare fixed recombination, global fitted channel gains, simple feature-dependent fitted gains, and a small neural mixer. Retain scalar conditioning and output-band gains as separate controls: output bands distinguish where an interaction lands; these channels distinguish where it came from.

    Preserve reality, signed phase information, symmetry under swapping inputs, and the appropriate zero-time, zero-reaction, zero-diffusion, and constant-field limits. Check small-step order and autonomous stability rather than inferring them from bounded gains.

    Use oracle projections to establish whether the channels actually enlarge the useful response space. Measure collinearity and effective rank.

    Start with three channels. Introduce more bands, output-dependent mixing, or spatial adaptation only when a diagnostic identifies the missing information. Avoid explicit all-pairs tensors at production grid sizes.

    The fixed decomposition has the same representational output as the baseline when recombined. Its purpose is an attribution control, not an accuracy improvement by itself.

12. **Compare physical interaction channels against simpler explanations.**

    At minimum, retain the normalized two-node baseline, fixed channel recombination, fitted channel gains, learned channel gains, current scalar conditioner, output-band model, and analytic quadratic+cubic correction.

    Add an ordinary feature-channel model with comparable measured compute. This tests whether any gain comes from physical channel meaning or merely additional capacity.

    If time-response channels become promising, compare them against extra quadrature nodes and inexpensive analytic temporal approximations.

    Assess conditioning and training independently: a representational advantage may exist even when one training recipe fails, while lower training loss may conceal poorer rollout behavior.

    Investigate distillation whenever learned gains exhibit a simple pattern. A constant rule, affine formula, lookup table, or inexpensive numerical approximation is a valuable outcome if it retains the benefit.

    Require each combination to improve on its strongest contained component. More branches must earn their cost.

13. **Keep a small, distinct exploration portfolio.**

    Beyond the mandatory three-channel prototype, examine a few alternatives and select at most two for initial implementation.

    Possible directions include grouping interactions by decay rate rather than input band; retaining signed interaction moments for repeated time queries; adding a selectively computed cubic component; using wavelets for localized interactions; or advancing a coarse interaction state with causal memory.

    Generate at least one additional idea from the diagnostics rather than only following this list.

    For each proposal, state the challenged assumption, mathematical mechanism, likely computation saved or added, principal failure mode, cheapest falsification experiment, and advance criterion.

    A memory prototype must obtain history causally from forward evolution and charge acquisition or startup cost. A selector must save work before the expensive computation occurs. A temporal representation must include refresh costs.

    Reserve 20% of this adjacent study’s discretionary pilot budget for these alternatives. This allocation is additional to, and independent of, the main project’s existing budget.

    Review relevant prior work before novelty claims: bilinear and Volterra operators, multiscale product decompositions, exponential integrators, wavelet operators, learned quadrature, and coarse-grained dynamics. For example, [multiwavelet operator learning](https://arxiv.org/abs/2109.13459) already studies learned operator representations across scales.

    “New to TDN,” “a new combination,” and “new in the literature” must remain distinct statements.

14. **Use staged execution, credible statistics, and complete costs.**

    Execute mathematical checks first, then cheap diagnostics, focused fitting, bounded architecture pilots, and finally independently confirmed comparisons.

    Start with D01 and D05 because they distinguish missing spatial structure from insufficient quadrature or conditioning. Run the feature-collision and signed-interaction tests next. Use their findings to choose the smallest useful architecture.

    Implement the full roughness sweep, but use a few predeclared representative levels for early development. Freeze the expanded comparison before evaluating fresh confirmation data.

    Publish a finite budget for this adjacent study before substantial execution. Include references, training, tuning, profiling, failed attempts, recovery, and reporting. Do not silently expand it or divert the main campaign’s allocations.

    Keep jobs below the existing 45-minute maximum and preserve current host-memory and RTX 4090 limits. Partition expensive work into recoverable units with sealed inputs and verified aggregation. Prepare exact desktop commands for work unavailable locally.

    Do not interfere with the jobs already running. Schedule substantial desktop experiments separately.

    Use independent parent fields as the statistical unit. Keep related phases, grids, roughness variants, schedules, and seeds together when they are paired constructions. Choose confirmation sample size from the desired precision and pilot variability, not a convenient row count.

    Predeclare primary effects, acceptable accuracy or coverage regressions, and uncertainty procedures. Correct for multiple primary comparisons or clearly label subgroup analyses exploratory.

    Measure repeated randomized paired timings with proper device synchronization. Separate cold and warm measurements, latency and throughput, and profiling from deployment timing.

    Charge feature extraction, transforms, products, allocations, transfers, encoding, verification, rejection, and fallback. Report accuracy-qualified cost, not latency of an incomplete subroutine.

    Record energy or monetary cost only when actual measurements or rates exist. Otherwise retain NA.

15. **Deliver an adjacent-study report that makes decisions possible.**

    Produce a hypothesis register, executable diagnostic programs, validated field generators, architecture diagrams, raw measurements, reproducible configurations, bounded launch and recovery commands, and a separate analytical atlas.

    The atlas should show scalar and channel error floors, feature-collision evidence, grid–step–duration maps, signed interaction errors, quadrature and amplitude-order behavior, roughness-dependent performance, trajectory observables, learned responses, runtime breakdowns, memory, and temporal break-even curves.

    Preserve the project’s visual conventions: clear Ours/Theirs labels with analytic controls distinguished; triangles for our models; thin lines; consistent colors; explicit better/worse directions; continuous translucent observed-range bands for dense curves; and visible gaps for absent measurements.

    Observed ranges are not confidence intervals. Show statistical uncertainty separately and retain traceability to source observations.

    Distinguish implemented, mathematically checked, CPU-tested, GPU-tested, and scientifically supported outcomes. Successful jobs do not establish successful hypotheses.

    End by answering: Which limitation was demonstrated? Which explanation weakened? Which representation added useful information? Did learning add value beyond fitted or analytic rules? Did any improvement repay its complete cost? Which workloads remain unfavorable?

    Recommend whether each branch should continue, change, or stop. Preserve useful negative results.

    **Any recommendation to incorporate a result into TDN’s main architecture or objectives must be presented as a separate proposed decision. Completing this adjacent study does not authorize that promotion.**
