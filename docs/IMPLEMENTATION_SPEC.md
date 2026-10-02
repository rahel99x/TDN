# Order-Anchored Temporal-Defect Networks
## Prompt-ready research and implementation specification for one NVIDIA A100 40 GB on USC CARC

**Document status:** proposed implementation protocol, not a completed solver or a prediction of acceptance.  
**Prepared:** October 1, 2026. Cluster and software facts below were checked against the primary sources in §24; recheck the live cluster before submission.  
**Primary deliverable:** a tested, reproducible ADR solver that learns the finite-time splitting defect using explicitly evaluable temporal units.  
**Default deployment:** a regular USC CARC user, one Discovery GPU allocation, one full A100 40 GB, no root privileges.  
**Evidence boundary:** the included temporal-core tests ran on CPU. No A100, Slurm, CARC account, private HALO repository, learned model, or new PDE experiment was executed in preparing this document.

### Navigation

[Agent mandate](#0-copy-ready-instruction-to-the-implementing-agent) · [Gates](#2-execution-map-and-stopping-gates) · [Architecture](#4-primary-architecture-order-anchored-temporal-units) · [Stable numerics](#5-stable-temporal-evaluation-correctness-before-speed) · [Model variants](#6-model-families-and-exact-architectural-controls) · [Training](#10-training-objectives-and-schedule) · [40-GB memory plan](#11-a100-40-gb-memory-budget) · [GPU optimization](#13-gpu-implementation-and-profiling-sequence) · [CARC deployment](#15-usc-carc-deployment-for-a-regular-user) · [Milestones](#21-milestone-by-milestone-work-orders) · [Sources](#24-source-references-and-provenance)

---

## 0. Copy-ready instruction to the implementing agent

You are implementing a research-grade numerical method, not trying to produce a favorable result at any cost. Read this specification before editing. Work in the existing repository when it is available; otherwise create an isolated new package with explicit adapters. Do not replace audited HALO operators while claiming to reproduce their results. The supplied support files are references and launch templates, not a completed training pipeline.

Implement the following objective:

> Keep reliable advection, diffusion, and reaction subsolvers. Approximate the error between their split composition and a more accurate coupled evolution with a small, order-anchored temporal model. Its parameters depend on the initial local state and fixed physical metadata, while the requested step enters only through an analytic temporal decoder. Determine whether this improves the complete accuracy–cost frontier under a 40 GB GPU constraint.

### 0.1 Mandatory operating rules

1. **Separate four categories in every report:** source-reported observation, mathematical deduction under stated assumptions, proposed configuration, and newly measured result. Do not invent benchmarks, theorem guarantees, hardware performance, access rights, or job outcomes.
2. **Do not revive the closed codec/basis/repair branches by default.** Reuse their tested numerical and measurement infrastructure only where appropriate.
3. **Start with CPU/tiny-grid correctness and a one-GPU pilot.** No multi-node work, large arrays, or 3-D full-field training before the gates below pass.
4. **Treat 40 GB as the hardware class, not all usable VRAM.** Read actual device memory. Use a soft budget `min(30 GiB, 0.80 * device_total)` and record peak allocated, reserved, and device-used memory. These are conservative project limits, not NVIDIA requirements.
5. **Use normal-user mechanisms only.** No `sudo`, driver installation, power/clock changes, MIG reconfiguration, privileged containers, MPS daemons, or unallocated GPU work. Do not train or compile substantial workloads on login nodes.
6. **Verify the authorized Slurm account, partition, GPU feature, environment, and writable paths.** The username is not necessarily the Slurm charging account. A regular login does not by itself prove that the requested allocation is authorized.
7. **Do not submit a full campaign automatically.** Prepare scripts and dry runs; perform only the explicitly authorized smoke/pilot budget. The included submission wrapper defaults to dry-run.
8. **Protect existing work.** Inspect `git status`; do not reset, delete, or overwrite unrelated work. Never put credentials in configs or logs. Never delete another user's files/processes to recover resources.
9. **Use actual input information.** If a proposed feature requires a larger stencil, receiver-private data, or communication, account for it. No future-state or teacher-label leakage into the deployed model.
10. **Fail closed on ambiguity in scientific claims.** A missing application tolerance, unknown discretization, or absent reference-validation result is a recorded blocker to confirmation, not a value to fabricate.
11. **Keep the agent execution log compact and actionable.** Update `WORK_LOG.md`, `NEXT_ACTIONS.md`, and a machine-readable gate file after each milestone. Record the exact command, commit, config hash, result path, and whether the command actually ran.
12. **Do not equate implementation with discovery.** A new architecture earns a claim only by beating appropriate numerical and neural controls at meaningful total cost.

### 0.2 Required final outputs from the implementing agent

- A working `tdn` package or equivalent documented repository integration.
- Unit, derivative, convergence, rollout, checkpoint-resume, serialization, and GPU parity tests.
- An immutable parent-level dataset split and exact teacher provenance.
- A memory/time calibration report for the allocated A100 40 GB.
- Reproducible Slurm scripts, environment locks, stage manifests, logs, and result schemas.
- Accuracy–time and accuracy–memory frontiers, not just training losses.
- A clear decision: continue, narrow the claim, or stop; identify the failed/satisfied gates.
- No claim of GPU validation when only CPU support tests ran.

---

## 1. What is already known, and what is not

### 1.1 Current project basis

The user reports three later conclusions: the codec affects approximately 1.6% of the relevant traffic; a frozen acoustic basis beats tested learned/adaptive alternatives at lower cost; and a reach rule distinguishes tested stability/accuracy regimes, with tests on both sides at 32³. These are **user-reported results in the conversation**. The later raw logs are not included in the supplied iteration-4 export. Preserve that provenance; do not attribute those three facts to an older file.

The attached `iteration4_results_export.md` reports earlier 16³ CPU/float64 pilots, the audited operator/derivative infrastructure, partial-rank stale-basis results, and failed repair/whitened-loss branches. In particular, its §6 limits the small-grid interpretation. It is historical background, not a license to claim the subsequent 32³ experiments were independently checked. [PROJECT-1]

The previous response proposed the temporal-unit model developed below. Its equations are proposed mathematics and architecture. No trained temporal model or ADR speedup was supplied.

### 1.2 Consequence for this implementation

Target the expense of accurate **coupled time evolution**, not the already-small auxiliary codec payload. The initial ADR package can run on one GPU without a distributed solver. Reuse HALO's operators, JVP audits, counters, or boundary code only after checking their applicability to the new equation. A periodic global-FFT prototype is not evidence of a local or distributed halo algorithm.

### 1.3 Novelty guardrail

Neural numerical corrections, explicit continuous-time neural models, exponential-integrator functions, and KAN parameterizations all have prior art. The candidate distinction is the tested combination of:

- known-order anchoring of a splitting defect;
- an h-independent local encoder and analytic finite-time response;
- a measured information-radius restriction;
- interpretable amplitudes/timescales where identifiable;
- a complete accuracy–cost improvement against tuned classical methods.

Do not claim that `dE/dt`, a learned residual, a KAN, or an exponential function alone is new. [P1–P5]

---

## 2. Execution map and stopping gates

| Gate | Implement first | Evidence required to proceed |
|---|---|---|
| G0: access and audit | Repository, account, paths, environment, numerical definition | No guessed permissions/settings; existing relevant tests pass |
| G1: mathematical core | Stable temporal function, derivatives, anchors | CPU FP64 tests and independent reference comparisons pass |
| G2: numerical headroom | Accurate coupled teacher and strong split/classical baselines | A material finite-step error/cost gap remains |
| G3: learnability/locality | Legal features and hidden-information tests | Defect is sufficiently predictable from deployed inputs |
| G4: temporal representation | Fixed-rate, learned-rate, polynomial/rational fits | Temporal units improve a meaningful fit/transfer/cost tradeoff |
| G5: one-GPU learning | Small MLP parameter encoder, short rollouts | Validation improvement without numerical failures or memory excess |
| G6: optimized implementation | FP32 baseline → BF16 net → compile → optional fusion | Accuracy and gradient tolerances retained after each change |
| G7: confirmation | New parents, unseen horizons, stiff regimes, resolution transfer | Fixed-tolerance full-solver benefit with uncertainty and costs |
| G8: optional branches | KAN, anchors, oscillations, constraints, distributed reach | Separate ablations; no merger of unvalidated mechanisms |

A default practical screening target is an approximately 2× larger accurate macrostep **and** at least 20% less end-to-end time at a fixed tolerance. These are proposed development targets, not forecasted gains or conference thresholds. Freeze a real tolerance before confirmatory evaluation. A larger macrostep alone is insufficient if reaction substeps, wider halos, or model evaluation erase the benefit.

---

## 3. Canonical mathematical problem

### 3.1 State and operators

Let `u` denote the semidiscrete physical state, with all spatial discretization choices fixed. Use

\[
\dot u=F(u)=\mathcal A(u)+\mathcal D(u)+\mathcal R(u).
\]

For an autonomous problem, `Phi_h(u)` is the exact flow of this finite-dimensional system. A numerical teacher approximates `Phi`, so keep its error explicit. For time-dependent forcing, augment the state with the clock or use `Phi(t0,h,u)` consistently; do not impose an autonomous composition identity on a nonautonomous problem.

Choose one audited symmetric split map, for example

\[
S_h=\Phi_{\mathcal R}^{h/2}\circ\Phi_{\mathcal D}^{h/2}
\circ\Phi_{\mathcal A}^{h}\circ\Phi_{\mathcal D}^{h/2}
\circ\Phi_{\mathcal R}^{h/2},
\]

where the rightmost map acts first. For the initial reaction–diffusion problem use

\[
S_h=\Phi_{\mathcal R}^{h/2}\circ\Phi_{\mathcal D}^{h}
\circ\Phi_{\mathcal R}^{h/2}.
\]

When subflows are only approximate, record their method, substep rules, tolerances, and cost. Symmetric notation alone does not make an inaccurate/non-symmetric subsolver a second-order method.

### 3.2 Defect and exact evolution identity

Define

\[
E(h;u)=\Phi_h(u)-S_h(u),\qquad E(0;u)=0.
\]

Holding the initial state fixed,

\[
\partial_hE=F(S_h+E)-\partial_hS_h
=[F(S_h+E)-F(S_h)]+[F(S_h)-\partial_hS_h].
\]

The first bracket propagates/couples the error. The second creates it. The proposal approximates this response; it does not integrate this full expensive equation online.

Adaptive substep counts can make `S_h` nonsmooth as a function of `h`. For temporal-derivative audits use analytic subflows or freeze a documented substep count over the horizon interval. Report count-change discontinuities separately instead of interpreting their derivative as a smooth flow derivative.

### 3.3 Required norms and error budget

Specify a weighted discrete norm

\[
\|v\|_W^2=v^TWv,
\]

including cell volume, species/field scaling, and any output mask. Do not allow resolution-dependent cell counts to change the metric silently.

Distinguish:

\[
\|\widehat\Phi_h-\Phi_{\rm continuum}\|
\le \|\widehat\Phi_h-\Phi_{\rm discrete}\|
+\|\Phi_{\rm discrete}-\Phi_{\rm continuum}\|.
\]

Inside the first term distinguish learned error, subsolver error, teacher uncertainty, floating-point error, and any local-patch/boundary approximation. Tiny learned errors are not meaningful if reference subtraction or state quantization is larger.

---

## 4. Primary architecture: order-anchored temporal units

### 4.1 Differential unit and closed form

For each mode `j`, use

\[
z'_j(s)=-\lambda_jz_j(s)+s^2a_j,\qquad z_j(0)=0.
\]

The encoder produces amplitudes and nonnegative rates from the initial state and fixed metadata. They stay constant while querying this macrostep:

\[
(a_j,\lambda_j)=N_\theta(\xi(u,\mu)),\qquad
\partial_ha_j=\partial_h\lambda_j=0.
\]

Here `mu` includes equation parameters, grid spacing, geometry, and an initial clock for nonautonomous extensions. **The queried `h` must not enter this encoder in the primary architecture.** Including `h`, `h*r'(u)`, or horizon-dependent features would invalidate the frozen-parameter temporal derivatives. Such an h-conditioned variant is an ablation, not the same differential model.

The exact frozen-parameter response is

\[
z_j(h)=a_j\int_0^h e^{-\lambda_j(h-s)}s^2\,ds
=h^3a_j\Psi(\lambda_jh),
\]

\[
\Psi(x)=\frac{x^2-2x+2-2e^{-x}}{x^3},\qquad \Psi(0)=\tfrac13.
\]

The update is

\[
\widehat\Phi_h(u)=S_h(u)+\widehat E(h;u),\qquad
\widehat E=\sum_jz_j(h).
\]

This is related to the standard exponential-integrator function `Psi(x)=2 phi_3(-x)`. The integration formula is not new; it is the architectural building block. [P3]

### 4.2 Temporal derivatives without an inner solve

For frozen encoder outputs,

\[
z'_j=-\lambda_jz_j+h^2a_j,
\]
\[
z''_j=-\lambda_jz'_j+2ha_j,
\]
\[
z'''_j=-\lambda_jz''_j+2a_j.
\]

Therefore

\[
\widehat E(0)=\widehat E'(0)=\widehat E''(0)=0,
\qquad \widehat E'''(0)=2\sum_ja_j.
\]

These derivatives are with respect to step length with `u` fixed. The **state** Jacobian must still differentiate through features, amplitudes, rates, and the base solver; freezing those derivatives would implement another model.

### 4.3 Leading-order anchor

If the audited smooth split has

\[
E(h;u)=h^3e_3(u)+O(h^4),
\]

impose

\[
\sum_ja_j=3e_3.
\]

One implementation predicts `M-1` amplitudes and sets the last to `3*e3 - sum(free)`. This gives the correct leading local defect. With the usual uniform smoothness and stability assumptions, reducing local error to `O(h^4)` supports third-order global convergence; it is not automatically fourth order or uniformly accurate at arbitrary stiffness.

For a smooth autonomous vector field,

\[
\partial_h^3\Phi_h(u)|_{h=0}
=D^2F(u)[F(u),F(u)]+DF(u)DF(u)F(u),
\]

and

\[
e_3=\tfrac16\left(\partial_h^3\Phi_h|_0-\partial_h^3S_h|_0\right).
\]

Audit a discrete `e3` on tiny systems before implementing matrix-free actions. Do not copy continuum commutator identities where the stencil chain rule does not hold. Numerical small-h estimates require FP64 subtraction and a measured truncation/roundoff plateau. Do not divide noisy labels by arbitrarily small `h^3`.

Start unanchored with a bounded `O(h^3)` correction; add the exact-leading anchor as a separately costed ablation. An anchor that costs more than the saved numerical work is not a practical win.

### 4.4 Nondimensionalization

Use fixed per-problem reference scales `U_ref` and `t_ref`, never `t_ref=h`. Define

\[
\tau=h/t_{\rm ref},\qquad
\alpha_j=a_jt_{\rm ref}^3/U_{\rm ref},\qquad
\beta_j=\lambda_jt_{\rm ref}.
\]

Then

\[
\widehat E=U_{\rm ref}\tau^3\sum_j\alpha_j\Psi(\beta_j\tau).
\]

The nth temporal derivative gains `U_ref/t_ref^n`. Species-specific state scales are allowed if logged and consistently used. A parameter-dependent `t_ref` is permitted when fixed for that problem; do not change it between horizons to make extrapolation look easier.

### 4.5 Expressivity and identifiability limits

Nonnegative real rates yield relaxation-shaped units. They do not span every response efficiently. In particular, as `h` becomes very large, a fixed positive-rate unit grows polynomially rather than reproducing arbitrary exponential amplification. Restrict the validated horizon or add a separately tested architecture; do not claim universal stiff extrapolation.

Mode permutations and cancellation make parameters nonunique. Record coefficient/rate ranges and cancellation ratios such as

\[
\frac{\sum_j\|z_j\|}{\|\sum_jz_j\|+\epsilon}.
\]

A learned rate is a model parameter, not automatically a physical timescale. Do not sort modes discontinuously in the hot path merely to give plots an attractive ordering.

---

## 5. Stable temporal evaluation: correctness before speed

### 5.1 Required formula

For small nonnegative `x`, use

\[
\Psi(x)=\sum_{n=0}^{\infty}\frac{2(-x)^n}{(n+3)!}
=\tfrac13-\tfrac{x}{12}+\tfrac{x^2}{60}-\tfrac{x^3}{360}+\cdots.
\]

The supplied reference uses degree 14 on `[0,0.5]`. For larger `x`, use the inverse form

\[
\Psi(x)=\frac1x\left[1-\frac2x-\frac{2\operatorname{expm1}(-x)}{x^2}\right].
\]

Both branches must stay finite even when evaluated eagerly under `torch.where`. An unselected `0/0` expression can still create invalid gradients. Clamp the *internal branch inputs*, not the physical model output, to avoid this.

Evaluate `Psi`, rate positivity, modal summation, anchors, and physical state addition in FP32 or FP64. Do not evaluate the cancellation-prone quotient in BF16/FP16. Preserve FP64 in validation. At a branch threshold verify function and derivative agreement; use a polynomial degree/cut selected by measured error, not hand-tuned benchmark outcomes.

### 5.2 Numerical tests

Required before GPU training:

- Compare against high-precision scalar evaluation on `x=0`, logarithmic grids from `1e-12` to `1e6`, and points around the branch cut.
- Check `Psi(0)=1/3`, positivity for `x>=0`, and large-x scaling.
- Compare the integral solution with independent quadrature or a tiny exact linear ODE.
- `gradcheck` and `gradgradcheck` for amplitude, rate, and h away from arbitrary domain constraints.
- Check the first three time derivatives against the recurrence, including h=0.
- Check that the unanchored zero-head model reproduces the base solver exactly in the chosen arithmetic.
- Check `sum(a)=3e3` and `E'''(0)=6e3` for the anchored model.
- Check convergence order on an exact noncommuting linear split system; do not use commuting controls as the only test.
- Repeat after autocast, compilation, and any custom kernel.

The reference suite uses FP64 relative function tolerance `3e-13` and FP32 `5e-6` over its declared grid. These are helper tolerances, not an end-to-end PDE error specification.

### 5.3 Hot-path optimization options

After correctness, profile a fused feature/parameter/temporal pipeline. Start with `torch.compile` fusion before writing Triton. A custom temporal decoder can fuse rate positivity, stable `Psi`, amplitude multiplication, and mode reduction without materializing `[cells, horizons, species, modes]` intermediates.

Preserve support for derivatives required by the experiment. A forward-only Triton kernel is not a drop-in replacement for a function used in JVP, gradgrad, or state-Jacobian training. Keep a functional reference backend and test every custom backward. Use inference-only fusion if that is the only audited derivative coverage, and label it accordingly.

---

## 6. Model families and exact architectural controls

### 6.1 Primary parameter encoder

Use audited physical features, a pointwise two-hidden-layer MLP, and separate amplitude/rate heads.

Proposed initial configuration:

| Quantity | Starting value | Controlled alternatives |
|---|---:|---|
| Audited features | up to 16, zero-pad only for alignment | minimal sufficiency subset |
| Width | 64 | 32, then 128 only with evidence |
| Hidden layers | 2 | 1 or 3 |
| Activation | SiLU | one prespecified alternative if needed |
| Modes | 4 | 2 and 8 |
| Rates | shared across species at each local context | species-specific, tile-level, global fixed dictionary |
| Rate initialization | logarithmic 0.1 to 100 in reference-time units | fixed measured dictionary |
| Rate positivity | softplus | bounded positive rates if explicit range is part of method |
| Amplitude initialization | zero | anchored initialization when e3 is supplied |
| Dropout / BatchNorm | none | not part of the primary numerical map |

Zero amplitudes imply zero initial gradients for rate parameters. The amplitude head should learn first; this is expected. Verify that rate gradients become nonzero once amplitudes depart from zero. Avoid diagnosing this as broken autograd without checking the algebra.

For `F=16`, width `w=64`, `M=4`, and `C=4`, a pointwise encoder costs roughly

\[
N(Fw+w^2+wM(C+1))=6400N\ \text{MACs},
\]

before nonlinearities and decoder work. On 128³ this is about 13.4 billion MACs. Small parameter count does **not** mean negligible fieldwise work. Measure whether this cost can plausibly be amortized by larger accurate steps. A tiny classical stencil may be cheaper than the network by a large factor.

### 6.2 Spatial structure

A concrete initial feature menu is the scaled state, scaled gradient components, Laplacian, reaction value/first derivative, local transport components, fixed equation parameters, and geometry ratios formed with `t_ref` rather than queried h. For scalar RD, examples include `u/U_ref`, `dx_d * D_d u/U_ref`, `dx_d^2 * D_dd u/U_ref`, `t_ref*r(u)/U_ref`, `t_ref*r'(u)`, and `kappa*t_ref/dx_d^2`. Choose a minimal set after sufficiency tests; this is not a requirement to compute expensive second derivatives for every problem. Feature derivatives remain in the graph during rollout training. Do not store an offline feature tensor and accidentally use it at a changed student state.

Start with pointwise maps of local derivative/invariant features. Do not begin with a 3-D U-Net, transformer, global attention, or a whole-field neural ODE. Those obscure the information and runtime hypotheses.

If local features prove insufficient, add a narrowly specified depthwise/local convolution with a counted stencil radius. Include the larger support in reach tests and patch guards. A wide neural receptive field is not free halo information.

Rates may be shared on a tile while amplitudes vary per cell. This can reduce parameter evaluation, but it changes the model and needs accuracy, seam, and gradient tests. Do not use global pooling if the intended distributed implementation cannot obtain those statistics without communication.

### 6.3 KAN branch

Only after the MLP temporal model is working, replace its parameter encoder with a small KAN. Compare identical legal features, outputs, losses, parent splits, and tuning budgets. The temporal decoder remains unchanged. [P4,P5]

Start with cubic B-splines, 5–9 intervals per normalized feature, widths 16 or 32, and two small layers. These are proposed settings. Fit input normalization and knot ranges on training data only. Freeze knot structure during confirmatory evaluation. Handle out-of-range values explicitly; clipping can hide extrapolation failures and change derivatives.

For a shared knot grid per input feature, evaluate a basis tensor of shape `[points, F, K]`, flatten it to `[points, F*K]`, and multiply by weights. Do not materialize `[points, F, O, K]`. If a library uses independent per-edge grids, this optimization changes its architecture; document that choice rather than claiming mathematical identity.

Stream chunks so spline activations fit VRAM. Disable symbolic regression, plotting, activation-history capture, and automatic knot refinement in the hot path. Run symbolic analysis only on exported small samples after training. Do not rely on an undocumented library method name; inspect the installed version and write a local adapter.

KAN variants can be slower than small GEMM-based MLPs. Profile in isolation and end-to-end. A discovered polynomial/rational/spline formula may replace the trained KAN for deployment after independent validation.

### 6.4 Required baseline ladder

1. Tuned split solver with appropriate adaptive/classical error control.
2. A competitive coupled numerical method such as an appropriate IMEX/exponential/RK scheme.
3. Classical leading-defect or commutator correction where practical.
4. Polynomial/rational temporal correction with the same feature access.
5. Fixed-rate temporal dictionary with linear/nonlinear amplitude regression.
6. `h^3 * MLP(features, h)` generic time-conditioned correction.
7. `h^3 * KAN(features, h)` when comparing KAN benefits.
8. Temporal units with h-independent MLP encoder.
9. The same temporal units with KAN encoder.
10. Order-anchored variants, with anchor computation fully costed.

Include an appropriate hypersolver/Taylor-style learned numerical baseline after reviewing its assumptions. No baseline should be left in slow eager FP64 while the proposed model receives compiled BF16 acceleration; compare matched-precision and best-valid optimized frontiers separately. [P1,P2]

---

## 7. Numerical problems and teacher construction

### 7.1 Tier A: tiny exact controls

Use two- or three-state noncommuting linear operators. Compute exact matrix exponentials and a symmetric split. Obtain the h³ coefficient by multiplying power-series matrices through order three. Include commuting operators, zero reaction, zero diffusion, and zero advection as limits.

Purpose: test the architecture, leading anchor, time jets, and convergence without spatial discretization ambiguity. These are controls, not sufficient evidence of a useful PDE method.

### 7.2 Tier B: scalar logistic reaction–diffusion

Use periodic

\[
u_t=\kappa\Delta_hu+\lambda u(1-u),
\]

with initial states in a documented admissible range. Start at 1-D/2-D small grids. The reaction flow for `u in [0,1]` is

\[
R_h(u)=\frac{u}{u+(1-u)e^{-\lambda h}}.
\]

Implement endpoints, underflow, and derivatives deliberately: `u=0` must remain zero even when `exp(-lambda*h)` underflows. Avoid an unselected `0/0` branch. Do not quietly clamp an erroneous PDE state to `[0,1]` and count it as accurate. Outside the validated interval, use an explicit failure/fallback policy.

For constant periodic diffusion on the central-difference grid, the exact semidiscrete Fourier multiplier is

\[
\widehat D_h(k)=\exp\left[-4\kappa h\sum_d
\frac{\sin^2(\pi k_d/N_d)}{\Delta x_d^2}\right].
\]

Use the discrete eigenvalue, not the continuum `-|k|^2`, when claiming same-discretization comparison. Cache geometry-dependent eigenvalues. The h-dependent exponential is recomputed or cached by a declared horizon bucket. FFTs are global operations: do not infer a finite distributed halo or a communication saving from this single-device implementation.

Alternative: a positive/energy-compatible local stencil integrator with verified substep stability. Count every substep. Keep teacher/reference and deployed subsolver distinctions explicit.

### 7.3 Tier C: nonlinear two-species coupling

Use a declared two-species problem, for example Gray–Scott:

\[
u_t=D_u\Delta_hu-uv^2+f(1-u),
\qquad
v_t=D_v\Delta_hv+uv^2-(f+k)v.
\]

Choose parameter ranges through a stability and reference audit; the manifest deliberately leaves them unset. There is no general linear total-mass invariant for these reaction terms. Do not impose one by habit. Positivity bounds, front/pattern metrics, and stiffness limits must match the chosen equations.

Begin at 32²/64², expand to 128², then 32³/64³. Full 128³ training is optional and gated by memory and scientific need. Use unequal diffusivities to expose coupling. Hold out physical parameter combinations, not only random snapshots.

### 7.4 Add advection only after the RD method is understood

Use an explicitly specified material or conservative form, velocity field, and flux. Material advection with prescribed velocity and a homogeneous scalar pointwise reaction commutes under smooth continuum assumptions, but conservative transport with nonzero divergence or an actual discrete chain rule can differ. Treat these as controls, not interchangeable equations.

For periodic constant velocity, an exact discrete linear transport subflow is possible, but centered transport need not preserve positivity. For nonlinear fronts use a suitable classical flux/integrator and report limiter differentiability. Do not attribute spatial-shock errors to time splitting.

### 7.5 Teacher acceptance

Generate a coupled teacher with sufficient refinement, independently of the split map. For explicit diffusion, respect the semidiscrete stability restriction; for stiff reaction, use a method/tolerance that actually resolves its timescale. A large fixed RK4 step is not an accurate stiff teacher just because it is FP64.

For each representative regime, compare at least two refinements/tolerances and require the measured difference to be small relative to the intended defect target. Proposed screening: reference uncertainty below 5% of the defect norm and below a declared fraction of the final numerical tolerance. Use three levels or an independent method when convergence is unclear. Exclude roundoff-dominated horizons from relative-defect fitting or weight them by a noise floor.

Compute `teacher - split` in FP64 **before** casting the defect to FP32. Storing two FP32 states and subtracting them later can destroy an h³-sized label. Save the defect separately, plus teacher uncertainty and normalization. If input states are downcast, quantify how that input perturbation changes the target; do not pretend FP64 labels remove FP32 input error.

---

## 8. Information/reach audit

Define the actual model observation `O_r(u)` at radius `r`, including all derivative features, intermediate states, histories, and normalization inputs.

For exact indistinguishable states with

\[
O_r(u^+)=O_r(u^-),
\]

any deterministic local correction obeys

\[
\max\{\|\widehat E-E_h(u^+)\|,\|\widehat E-E_h(u^-)\|\}
\ge \tfrac12\|E_h(u^+)-E_h(u^-)\|.
\]

For linearized analysis, let `N_r` span observation-invisible perturbations and define

\[
C_{\rm int}(r,h)=L_yDE_h(u)N_r,
\qquad C_{\rm full}(r,h)=L_yD\Phi_h(u)N_r.
\]

For a symmetric hidden ball of radius `rho`, the linear worst-case hidden target radius is `rho*||C_int||`. Compare defect versus full-map influence. This asks whether the residual is genuinely easier to infer, not just smaller in absolute amplitude.

Use matrix-free JVP/VJP and a few independent probes. Never build a dense 128³ Jacobian. Audit JVP/VJP dot products and finite differences on tiny grids. For nonlinear observations, `DO*v=0` is only first-order invisibility; validate exact feature equality or label the result as a sensitivity study.

For diffusion, there is generally no exact continuum finite-speed cone. Use tolerance-dependent tails or the actual discrete dependency graph. For an explicit radius-one RK4 stencil, one substep can grow exact dependence by up to four cells; nested feature derivatives and a multi-step rollout expand it further. A physical advection reach estimate alone does not justify cropping or patch boundaries.

---

## 9. Dataset, horizons, and patch contracts

### 9.1 Parent-level provenance

A parent consists of the initial-condition realization, physical coefficients, forcing, domain, and discretization family. All derived windows, patches, horizons, perturbations, and augmentations stay in its split. Use independent test parents untouched by previous pilot choices.

Proposed small development allocation: 64 training, 16 validation, and 16 diagnostic parents per cheap family, with a separate sealed confirmatory set selected after pilot variance is known. These are planning counts, not a power guarantee. Keep all exposed historical HALO parents in development unless their provenance proves otherwise.

### 9.2 Paired multi-horizon data

At each anchor save several h values, the split result, defect, reference uncertainty, and physical metadata. A starting horizon set is `h_base * {0.25, 0.5, 1, 2}`. Evaluate unseen/intermediate and larger horizons only when teacher stability and reach permit them; do not force a numerical instability just to label a point OOD.

`h_base` is a sampling scale, not the model's `t_ref=h` trick. Encode one initial state once and reuse its amplitudes/rates across its horizons. Store h in FP32/FP64 tensor form. Avoid host-scalar specializations creating one compiled graph per numerical h value.

For rollout evaluation fix total physical time and observation times. Use a declared final shorter step where needed. Mixed h sequences are independent tests, not an excuse for comparing different final times.

### 9.3 Patch regression versus patch rollout

For one-step regression, generate targets from a trusted full-domain solve, extract a core and sufficient input guard, and test whether the target is inferable from that input. A cropped teacher field does not prove locality.

For rollout training on a patch, the evolving guard must be adequate for **every** substep and learned feature stencil. Choose one of:

- evolve the full small domain and compute loss on a cropped region;
- enlarge the patch by a verified discrete dependence margin;
- impose a declared external boundary schedule and label that training protocol explicitly.

Do not use future teacher boundary values in training and then claim autonomous local rollout without measuring the change at deployment. For FFT subflows, a finite cropped patch does not reproduce a global periodic solve by default.

### 9.4 Storage format and I/O

Use moderate contiguous shards, not millions of per-cell files. Reasonable starting shard size: 128–512 MiB, adjusted to measured I/O. Use `.npy` memmaps or a documented chunked format with independent reader handles. Never share an unsafe inherited HDF5 handle across DataLoader workers. Use one writer per shard, atomic completion markers, checksums, and immutable indices.

Store metadata separately from tensors: parent ID, anchor time, h, grid, equation, parameters, reference settings, precision, feature version, split hash, and generation commit. Normalization uses training data only.

Keep authoritative manifests and checkpoints in persistent storage. Use scratch for replaceable shards and performance caches, with a second protected copy or regeneration route for irreplaceable data. CARC scratch is not backup. [C3]

---

## 10. Training objectives and schedule

### 10.1 Primary supervised loss

Train against the coupled defect with a fixed, documented weighted norm:

\[
L_{\rm defect}=\|\widehat E-E\|_W^2.
\]

Avoid dividing every sample by its own almost-zero defect norm. Use a training-derived scale with a noise floor, or stratified horizon weighting. Also report absolute state error and unnormalized defect error so normalization does not hide bad cases.

For anchored models, measure what remains beyond the known correction. Do not make the neural model learn the same easy leading term and claim the anchor adds new information.

### 10.2 Temporal derivative loss

With an independently accurate teacher derivative,

\[
L_{\partial h}=\|\partial_h\widehat E-\partial_hE\|_W^2.
\]

Obtain the teacher derivative by a documented differentiable reference, the exact defect identity where applicable, or an audited centered h-difference. Choose h-difference size via convergence, not a single arbitrary value. Normalize derivative units consistently.

Start with zero derivative-loss frequency. Add it on a small fixed fraction of batches after ordinary regression works; JVP/double-backward can dominate memory and time. Compare benefit per GPU-hour, not just per optimizer step.

### 10.3 Composition and defect consistency

For autonomous dynamics the exact flow satisfies

\[
\Phi_{h+k}(u)=\Phi_k(\Phi_h(u)).
\]

The defect satisfies

\[
E_{h+k}(u)=S_k(S_h(u)+E_h(u))-S_{h+k}(u)
+E_k(S_h(u)+E_h(u)).
\]

Re-encode at the corrected intermediate state for the second step. Do not reuse the initial state's coefficients across multiple physical macrosteps unless explicitly defining a different model. A composition loss is useful but cannot substitute for correct targets: an inaccurate map may be self-consistent. Semigroup-informed learning is prior work. [P6]

### 10.4 Rollout training

Start with one-window regression, then 2-window and 4-window losses after the base method is stable. Keep longer free rollouts for validation. Do not begin with dozens of differentiable 3-D windows.

Checkpointing recomputes forward work but preserves the intended gradient when implemented correctly; detaching between windows changes the objective. If using truncated backpropagation, label and ablate it. Never call `inference_mode` on a base subsolver that must propagate gradients from the correction to later states.

### 10.5 Suggested optimization budget

Initial AdamW setup: learning rate `1e-3`, weight decay `1e-5`, gradient-norm clipping at 1.0, at most 5,000 optimizer steps per prespecified run. These are proposed defaults, not tuned claims. Use validation and a bounded learning-rate search shared across families.

Stage a pilot approximately as follows: 1,000 steps one-step regression; up to 3,000 steps with selected short-rollout/multi-horizon training; up to 1,000 steps for a prespecified refinement/precision stage. Do not keep training until a favorable test statistic appears.

Use `optimizer.zero_grad(set_to_none=True)`. Fused AdamW is optional after parity and memory tests; optimizer state is small here, so it may not be the main bottleneck. Loss accumulation must weight microbatches by actual valid cells/parents, not average unequal chunk means blindly.

### 10.6 Checkpoint selection

Select on validation feasibility first (no forbidden failures, tolerance satisfied), then the declared accuracy–cost objective. Save last and best checkpoints, config, normalization, knots/dictionaries, precision policy, optimizer/scheduler, RNG, and sampler cursor. Never choose a different checkpoint for each test case.

---

## 11. A100 40 GB memory budget

### 11.1 What to assume

NVIDIA documents A100's Ampere architecture, tensor-core formats, and memory system. CARC lists both 40 GB and 80 GB A100 variants. This project deliberately targets the smaller device. Do not assume SXM versus PCIe, exact sustained bandwidth, full boost clocks, or a guaranteed share of node resources; record the allocated device. [C1,C2,N1]

Use actual `torch.cuda.get_device_properties(0).total_memory`. Do not assume `40e9` equals `40*2**30`. Do not mistake `#SBATCH --mem=64G` for GPU memory: that request is host RAM.

### 11.2 Useful field sizes (arithmetic, not a measurement)

For a single physical state with 4 channels in FP32:

| Grid | Physical state | One 64-channel BF16 hidden field | One 16-channel FP32 feature field |
|---|---:|---:|---:|
| 32³ | 0.5 MiB | 4 MiB | 2 MiB |
| 64³ | 4 MiB | 32 MiB | 16 MiB |
| 128³ | 32 MiB | 256 MiB | 128 MiB |
| 256³ | 256 MiB | 2 GiB | 1 GiB |

These omit batch multiplicity, guards, saved activations, base-solver stages, modal arrays, FFT workspaces, gradients, copies, allocator reservation, and compiler graph pools.

At 128³, `C=4,M=8` FP32 amplitudes alone occupy 256 MiB; rates and decoder intermediates add more. At 256³, even a few full hidden fields and unrolled windows can exhaust 40 GB despite a tiny model.

### 11.3 Budget formula

Estimate

\[
M_{\rm total}\approx M_{\rm parameters+optimizer}
+M_{\rm live\ physical\ states}
+M_{\rm features}
+M_{\rm retained\ activations}
+M_{\rm modal\ intermediates}
+M_{\rm FFT/solver\ workspaces}
+M_{\rm compile/graph\ pools}
+M_{\rm allocator\ slack}.
\]

For FP32 parameters with FP32 gradients and two FP32 Adam moments, a rough persistent parameter-state estimate is `16 * n_parameters` bytes; actual optimizer temporaries may add more. This term is usually small for these models.

Use a soft limit `min(30 GiB, .8 * actual_total)` after warmup. Treat 90% device-used memory as a warning/stop threshold, not a utilization target. The usable limit must also reflect other driver/library allocations and the observed initial free memory.

### 11.4 Starting execution profiles

| Profile | Initial shape / work | Precision / strategy |
|---|---|---|
| Mathematical audit | tiny arrays / 1-D small grids | FP64 CPU, then FP64 GPU parity |
| Regression | 32,768 encoded cells per microbatch | FP32 reference; BF16 encoder after audit |
| 2-D rollout | 64² full domains or valid guarded cores | 1→2→4 windows; checkpoint only if needed |
| 3-D fine-tune | 32³ cores, batch 1 initially | 2 windows, then 4; measured guards |
| Full 128³ inference | one domain | stream 32,768–65,536-cell encoder chunks |
| Full 128³ training | not a default | only after memory calibration proves need and feasibility |

These are proposed launch points, not guaranteed-fitting batches. Sweep microbatch sizes in a bounded calibration job after the teacher/feature path exists.

### 11.5 Correct chunking

For independent pointwise regression, accumulate gradients chunk by chunk, dividing by the full effective sample weight. This frees each chunk's graph.

For a loss after a globally coupled solver step, chunking the encoder alone does not automatically free all training activations: concatenated outputs keep graphs alive. Use non-reentrant activation checkpointing around each deterministic encoder chunk or around audited solver segments. Do not detach chunk outputs to make the memory meter look better.

For inference, preallocate the correction/output buffer and stream chunks under `inference_mode`. Avoid stacking all horizons: decode one horizon at a time or use a small bounded horizon bucket. Share encoded parameters when querying several h values at the same initial state.

### 11.6 OOM recovery order

1. Verify that loss histories and retained references are not holding graphs.
2. Reduce microbatch/cell chunk size while preserving the effective batch with gradient accumulation.
3. Reduce simultaneous horizon decoding and DataLoader prefetch.
4. Apply activation checkpointing with gradient parity tests.
5. Reduce permitted rollout length only as a recorded new training configuration.
6. Revisit architecture width/rank with a new config and fair baseline comparison.

Do not reduce numerical precision of the physics state silently, remove validation, detach gradients, or change the target tolerance to make a job fit. After OOM, discard the incomplete optimizer step and restart from a complete checkpoint; arbitrary in-process recovery can retain memory and corrupt accumulation state.

---

## 12. Precision policy: do not erase the defect being learned

| Component | Correctness reference | Optimized candidate |
|---|---|---|
| Coupled teacher and small defect subtraction | FP64 | keep FP64 unless independent error audit allows otherwise |
| Base solver physical state | FP64 initially | FP32 only after tolerance-level comparison |
| Cached defect labels | FP64 source | FP32 stored after subtraction and noise-floor audit |
| Neural hidden GEMMs | FP32 | BF16 autocast |
| Rate positivity / Psi / amplitude sum / anchor | FP64 tests | FP32 minimum |
| State addition and weighted reductions | FP64 diagnostics | FP32 with FP64 evaluation reductions where required |
| Derivative-label and high-order audits | FP64 | optional separate FP32 robustness check |

BF16 has favorable range but limited precision; it is not appropriate for subtracting nearly equal teacher/split states. AMP applies per-operation rules and does not prove numerical equivalence. [T2]

Keep model master parameters and optimizer states in FP32. BF16 autocast usually does not require FP16-style dynamic loss scaling, but check finite gradients and the installed backend. FP16 is a separately audited alternative, not a default.

TF32 can affect FP32 GEMMs/convolutions. Keep it disabled for reference runs. Test it only as an explicitly recorded candidate; it need not affect pure stencil elementwise arithmetic. PyTorch's precision-control APIs are version-sensitive: inspect the installed version, use its documented controls, and do not mix legacy `allow_tf32` with incompatible newer `fp32_precision` settings. [T1]

Always cast and accumulate the correction before adding it to the physical state. If the correction is smaller than FP32 state resolution, FP32 evolution may be the limiting error regardless of network quality; retain a higher-precision state for that target or relax the task only with explicit scientific justification.

---

## 13. GPU implementation and profiling sequence

### 13.1 Tensor layout

Use a consistent physical state layout such as `[B,C,Z,Y,X]` (or 2-D equivalent) for the solver. For pointwise encoders, form a contiguous `[points,F]` chunk once, run GEMMs, and write correction outputs back with an audited map.

Do not permute/contiguize entire 3-D fields repeatedly inside every layer or horizon. Include layout conversion in timing. Channel-last/`channels_last_3d` may help a 1×1 convolution implementation but is not universally faster; benchmark against the matrix representation.

Feature computation should fuse common derivatives and reuse them. A naive sequence of `torch.roll` operations allocates full temporary fields. Keep that functional version as a correctness baseline, then test a fused stencil or explicit padded-slice implementation. In-place overwrites that invalidate autograd are forbidden in the training backend.

### 13.2 First optimize the measured bottleneck

Profile representative teacher, baseline, encoder, decoder, and rollout phases separately. Record CPU wait, kernel launch count, GEMM/stencil/FFT time, H2D/D2H, and peak memory. Use short profiler windows with warmup, not a whole training run. CUDA events and synchronization define meaningful GPU timings; raw Python wall time around asynchronous launches does not. [T1,T4,N2]

An A100 has no native H100-style FP8/TMA path. Do not import Hopper-specific kernels. For a tiny parameter model, attention optimizers, FlashAttention, FSDP, ZeRO, and optimizer offload are not the first solutions. This workload is likely to be dominated by physical fields, stencil/FFT work, and launch or activation cost, but measure rather than assume. [N1]

### 13.3 Compilation ladder

1. Establish eager FP32 correctness and timing.
2. Audit BF16 neural operations with FP32 physical arithmetic.
3. Compile the stable encoder/temporal decoder with `torch.compile(..., mode='default')`.
4. Compile audited pure tensor base-solver segments, not the entire Slurm/data/checkpoint loop.
5. Benchmark `max-autotune-no-cudagraphs` for repeated shapes if supported.
6. Benchmark `reduce-overhead` or `max-autotune` only when static-shape reuse and memory permit their CUDA-graph behavior.
7. Use manual CUDA graphs only after a simple path is demonstrably launch-bound and static. Do not stack manual capture onto automatic capture without understanding the pools and ownership.

Mode names and behavior depend on the installed PyTorch release. The current documentation describes CUDA-graph/private-workspace tradeoffs; they are not guaranteed speedups. [T3,T5]

Warm every shape/precision bucket, perform a backward and optimizer step, then reset peak memory counters and measure steady state. Report compile time and first-use memory separately. Keep an uncompiled model handle for checkpoint state dictionaries. Capture/compile derivative-loss paths separately if they are incompatible with the ordinary path; falling back must be logged.

### 13.4 Static-shape discipline

Bucket resolution, patch shape, batch size, channels, and mode count. Use `drop_last` for training only if the sampling/weighting implications are controlled; pad/mask validation instead of dropping test data. Pass h as a tensor rather than changing a Python float that triggers specialization. No `.item()`, `.cpu()`, printing, or Python branching on CUDA tensor values in a captured hot path.

A CUDA graph reuses addresses and shapes. Keep inputs/outputs alive and copy data into fixed buffers. CPU I/O, checkpointing, dynamic failure handling, and arbitrary adaptive loops remain outside capture. [T1,T3]

### 13.5 Custom kernel escalation

Only implement a Triton/CUDA kernel when profiling shows a material bottleneck remaining after compilation. Candidate kernels: derivative-feature stencil, stable Psi/modal reduction, KAN basis evaluation, or layout-fused pointwise projection.

Target A100 compute capability 8.0 and the actual compiler/runtime compatibility. Implement reference comparisons for forward, first derivatives, and any higher derivative used. Avoid register-heavy fusion that spills more than it saves. Sweep a few launch configurations on the allocated GPU; do not copy an H100 tuning table.

### 13.6 Host-side settings

A starting 64-GB host-memory budget should leave room for Python/compilation, 8–16 GiB of active shard/page cache, a bounded set of worker batches, at most a few GiB of pinned staging buffers, and a substantial safety margin. These are planning allocations; inspect actual RSS and cgroup accounting during compile and training. Do not load the entire dataset separately in each worker. Memory-mapped read-only shards and lightweight index objects are preferable to a large Python list copied into worker processes.

Start with 8 allocated CPUs, two DataLoader workers, and one intra-op/inter-op thread per training process. Set worker thread pools to one. Increase workers only when measured input starvation justifies it. `pin_memory=True` and nonblocking copies can overlap transfer when used correctly; excessive pinned memory competes with host RAM. [T4,T6]

Use `optimizer.zero_grad(set_to_none=True)`. Never call `empty_cache()` each iteration; it does not free live tensors and can destroy allocator reuse. Do not run `nvidia-smi` or query Slurm every training step. Log device summaries periodically and use buffered metrics.

---

## 14. Checkpointing, exact-enough resume, and failure handling

### 14.1 Save complete state at a safe boundary

Save after a completed optimizer step: original uncompiled model state, optimizer/scheduler, scaler if any, Python/NumPy/Torch RNG state, CUDA RNG states, normalization, feature definition, anchor/version, KAN knots, dataset/split hashes, parent/sample cursor, training phase, precision/compile settings, and global step.

For interrupted gradient accumulation, either finish the declared accumulation if time allows or resume from the previous committed optimizer step and replay it. Do not save half an update as though it were complete. DataLoader prefetch must not advance the authoritative committed sample cursor. Exact replay requires a sampler design that accounts for worker RNG; otherwise report statistical rather than bitwise resume.

### 14.2 Atomic and persistent writing

Write a temporary checkpoint in the **same persistent directory**, flush/close, then use atomic rename. Maintain a previous known-good checkpoint and a small `latest.json`. Include checksums/metadata and test loading in a fresh process. A temporary file on scratch followed by a cross-filesystem rename is not the same atomic operation.

Start with checkpoint cadence of 500 optimizer steps or 10 minutes, whichever triggers first; adjust after measuring I/O. Save immediately at a clean boundary after a stop flag. These are proposed defaults.

### 14.3 Slurm signals

The template requests `--signal=USR1@180`. Without the `B:` prefix, Slurm signals job steps rather than only the batch shell; implement the main Python handler before training starts. The handler sets a flag only—no CUDA, serialization, or locks inside it. DataLoader workers should ignore `SIGUSR1` so the warning does not kill their processes unexpectedly. [S1]

At the next safe boundary, checkpoint and write status `PAUSED_NEEDS_RESUME`. Exit with a documented nonzero code such as 75, so an `afterok` dependency does not mistake an unfinished run for success. Resubmit with `--resume` explicitly. No automatic requeue or infinite self-resubmission is enabled by default.

Register `SIGTERM` for best-effort stop as well, but do not rely on enough time after termination or on catching SIGKILL. Periodic persistent checkpoints are the primary protection. Test signal handling on a cheap allocated smoke job before a long run.

### 14.4 Scientific failures

If the model produces nonfinite states, violates the declared admissible set, or exceeds the reference tolerance, record a failure. A fallback to two classical half-steps can be a legitimate algorithm if its trigger and cost are declared. It is not acceptable to discard fallback cases from the speed/error table.

Do not silently clip physical states, truncate reaction rates, or skip difficult parents. These actions change the numerical method.

---

## 15. USC CARC deployment for a regular user

### 15.1 Verified public defaults versus live checks

CARC's GPU guide documents the Discovery `gpu` partition, typed requests such as `--gpus-per-task=a100:1`, and `--constraint=a100-40gb` to distinguish the 40 GB model. Its running-jobs guide uses `myaccount` for authorized associations. Treat these as documented request forms, not proof of current availability or your authorization. [C1,C4]

Begin on Discovery unless `myaccount` explicitly authorizes another selected partition. Do not presume Endeavour condo access. Read current `sinfo` and partition settings. Use your account's association rather than a guessed project ID. If only the default account is authorized and verified, omitting `--account` is valid; the template supports that. [C4,C5]

### 15.2 Read-only login discovery

Run the included `scripts/probe_carc.sh`. It asks for `myaccount`, `myquota`, `sinfo`, `scontrol show partition gpu`, and module inventory. It does not submit a job or train a model.

Check exact writable persistent and scratch paths from `myquota` and filesystem permissions. Do not use a guessed `/project/<name>` path. The current storage guide lists `/home1`, `/project2`, and `/scratch1`; an official January 2026 maintenance announcement says the old `/project` mount was being removed and `/scratch1` rebuilt. Prefer current paths and live checks over old examples. [C3,C8]

### 15.3 Critical storage distinction

CARC documents `/tmp` and `/dev/shm` as RAM-backed filesystems constrained by requested/node host memory. **Do not assume local NVMe, or copy a large dataset there.** The default script puts temporary compilation files on the user's verified scratch path. [C3]

Scratch is temporary and not backed up. Persistent project storage is also not automatically an off-cluster backup. Keep an appropriate second copy of irreplaceable checkpoints/results. Use a modest number of large shards rather than metadata-heavy millions of files.

### 15.4 Environment creation

Use a separate user environment. CARC documents `module load conda`; inspect the available module and initialize the shell using the module's Conda base, without requiring persistent `.bashrc` edits in every job. [C6]

Example interactive setup sequence, after choosing authorized values:

```bash
# Read-only discovery on login first.
bash scripts/probe_carc.sh

# Fill and source a private copy of user.env.example.sh.
source /absolute/path/to/user.env.sh

# Obtain an allocation for GPU/environment verification; account is optional ONLY
# when the default account is known to be authorized.
args=(salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8
      --gpus-per-task=a100:1 --constraint=a100-40gb --mem=64G --time=01:00:00)
[[ -n "${CARC_ACCOUNT:-}" ]] && args+=(--account="$CARC_ACCOUNT")
"${args[@]}"
# CARC may place you on the allocated node; if it leaves a login shell, use srun
# within the allocation. Verify hostname and allocation before any GPU work.
```

Inside the allocated context:

```bash
module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
# One-time creation; do not run again over an existing environment without review.
# Choose a tested Python version compatible with the selected PyTorch build.
conda create --prefix "$TDN_ENV_PREFIX" python=3.11 pip -y
conda activate "$TDN_ENV_PREFIX"

# REQUIRED choices: inspect the driver and official PyTorch package selector.
# Do NOT paste an old CARC cuda=11.8 recipe and call it current by assumption.
# Set TORCH_VERSION and TORCH_WHEEL_INDEX to a compatible, explicitly selected build.
: "${TORCH_VERSION:?Select and record a compatible tested version}"
: "${TORCH_WHEEL_INDEX:?Select its official CUDA wheel index}"
python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_WHEEL_INDEX"
python -m pip install numpy scipy pyyaml pytest mpmath
# After implementation/package metadata exists:
python -m pip install --no-deps -e "$TDN_REPO_ROOT"
python -m pip freeze > "$TDN_PROJECT_ROOT/tdn-environment-freeze.txt"
conda list --explicit > "$TDN_PROJECT_ROOT/tdn-conda-explicit.txt"
```

Record driver, wheel CUDA runtime, compiler, and PyTorch versions. The toolkit reported by `nvcc` is not the same object as the wheel runtime or the driver's supported CUDA level. Do not install/modify the driver. A working standard wheel does not imply that custom CUDA extensions will compile; inspect a compatible compiler/toolkit only when needed. [C7,N2]

Do not perform large environment solves, code compilation, or benchmarks on login nodes. If package access is restricted from compute nodes, use CARC's documented permitted download/setup route or an offline wheelhouse; do not bypass policy.

### 15.5 GPU request and host budget

The primary script requests one task, one A100 40 GB, 8 CPUs, 64 GB host RAM, and four hours. These are proposed starting requests; GPU partition wall-time and account/QOS limits must be queried. Do not overrequest 256 GB host RAM merely because a node has it. CARC resource charging depends on requested resources. [C1,C4]

A short A100 smoke test can use the GPU partition with a short time limit. Do not assume that Discovery's generic `debug` partition contains A100s.

Keep `CUDA_VISIBLE_DEVICES` as Slurm assigns it. Inside a one-GPU task, use `cuda:0`; do not map it to physical GPU index 0 by overwriting the variable. The preflight rejects a non-A100, a non-full 40 GB-class device, or unexpected multi-GPU visibility for this primary pipeline.

### 15.6 Job launch and monitoring

After the package has implemented the specified CLI and passed smoke tests:

```bash
source /absolute/path/to/user.env.sh
bash scripts/submit.sh audit /absolute/path/to/audit.yaml             # dry-run
bash scripts/submit.sh audit /absolute/path/to/audit.yaml --submit    # explicit

# Later, after required config fields and gates are filled:
bash scripts/submit.sh train /absolute/path/to/pilot.yaml             # dry-run
bash scripts/submit.sh train /absolute/path/to/pilot.yaml --submit

squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

`MaxRSS` is a host-memory accounting metric, not peak VRAM. GPU accounting may be configured differently; keep application-side CUDA memory logs.

Use Slurm job IDs actually returned by `sbatch`. Do not claim submission success from a constructed command string. The wrapper creates log directories **before** submission. Shell variables in `#SBATCH` directives are not expanded; variable paths/account options are passed from the wrapper's command-line array. [S1]

### 15.7 Arrays and dependencies

After a pilot budget is approved, run a small explicit trial manifest with a concurrency cap such as `%1` or `%2`, consistent with the account limits. Each array element selects exactly one manifest row using `SLURM_ARRAY_TASK_ID`; it does not independently launch a whole hyperparameter sweep.

Use separate generation, training, and evaluation stages. `afterok` can depend on successful completion of the prior stage, but do not chain a paused/incomplete run as successful. The job-script template does not auto-submit dependent jobs.

One GPU per training trial is the default. Multi-GPU data parallelism is optional only after single-GPU saturation and large enough batches. A model with few parameters may gain more from independent serial trials than DDP communication. Domain-decomposed physical simulation is a different implementation from DDP training; never conflate their scaling claims.

---

## 16. Code organization and interface contracts

Proposed new package layout (adapt names only with a documented mapping):

```text
tdn/
  cli.py                   # audit/generate/train/evaluate/benchmark stages
  config.py                # strict schema; reject unknown and missing required fields
  numerics/
    operators.py           # actual ADR discrete operators
    subflows.py            # trusted A/D/R steps, their cost and differentiability
    splitting.py           # fixed split composition and provenance
    reference.py           # independently refined coupled teachers
    invariants.py          # equation-specific admissibility, not blanket clipping
  models/
    temporal.py            # reference and optimized Psi/mode evaluation
    parameter_mlp.py       # h-independent encoder
    parameter_kan.py       # optional audited KAN adapter
    anchors.py             # audited discrete leading error coefficients
    oscillatory.py         # optional; not enabled by default
  features/
    local.py               # value/gradient/laplacian feature actions
    provenance.py          # stencil, units, shared/private/history information
  data/
    parents.py             # immutable physical parent specification
    shards.py              # safe indexed readers/writers
    sampling.py            # parent-balanced, resumable samplers
  train/
    losses.py              # defects, derivatives, composition, rollouts
    loop.py                # accumulation, AMP, checkpoints, validation
    checkpoint.py          # atomic full-state resume
  analysis/
    influence.py           # matrix-free full-vs-defect information tests
    convergence.py         # local/global order and horizon interpolation
    statistics.py          # parent-level comparisons and failures
    profiling.py           # events, timing, CUDA memory and host I/O
  runtime/
    precision.py           # reference vs speed policies, version-aware
    signal_handling.py     # USR1/TERM flags; worker signal policy
    metadata.py            # filtered reproducibility fields, no secrets
scripts/
  probe_carc.sh
  gpu_preflight.py
  submit.sh
  train_a100.sbatch
configs/
tests/
results/                   # generated, immutable per run; do not type table numbers
```

Core contracts:

```python
# These are interfaces to implement, not functions already provided by HALO.
def split_step(u, h, equation, geometry, *, differentiable: bool): ...
def reference_step(u, h, equation, geometry, tolerance): ...
def extract_features(u, equation, geometry, feature_spec): ...
def encode_modes(features, metadata, leading_defect=None): ...
def decode_modes(encoded, h): ...
def corrected_step(u, h, equation, geometry, model, policy): ...
def audit_dependency(feature_spec, solver_spec, horizon_spec): ...
```

`encode_modes` must not receive h in the primary model. `corrected_step` recomputes encoding at each new physical state. `reference_step` is not callable from the deployment model except as an explicitly costed audit/fallback.

Use dataclasses/typed configs with shape and units documentation. Keep Python control, I/O, and failure recovery outside pure tensor kernels. Add range checks at data/config boundaries and validation intervals rather than expensive `.item()` assertions inside every compiled pointwise operation.

---

## 17. Detailed acceptance tests

| Test class | Required check | What it does NOT establish |
|---|---|---|
| Temporal algebra | high-precision Psi, jets, anchors, grad/gradgrad | PDE speed or accuracy |
| Numerical baseline | split order and reference convergence | correct learned extrapolation |
| Discrete derivatives | JVP/VJP dot test and step-size FD sweep | uniform nonlinear error certificate |
| Information | same-input pairs and influence tail | universal finite diffusion cone |
| Zero correction | exact base equivalence within dtype | useful learning |
| Precision | eager FP32 vs BF16-net/FP32-state/compiled parity | full-range numerical equivalence |
| Chunking | full vs chunked forward and gradients | free activation-memory reduction |
| Checkpoint | interrupted vs uninterrupted committed steps | bitwise identity under uncontrolled worker RNG |
| Physical invariants | equation-specific constraints/failure policy | every field's accuracy |
| Temporal transfer | new h and unequal-step sequences | arbitrary-stiffness generalization |
| Deployment | full solver timing at matched tolerance | distributed speedup from a one-GPU test |

For convergence plots, fit slopes only over a range where discretization/reference/roundoff errors are smaller than the measured temporal error. Show the range and raw points. A visually straight line over two favorable points is not enough.

For full rollouts, report all failed cases and maximum/quantile errors as well as means. Compute comparisons over physical parents. With zero failures, a one-sided binomial upper bound is `1 - alpha**(1/n)`; 16 clean examples do not establish a 1% failure rate.

---

## 18. End-to-end benchmark and inference claims

Primary evaluation: a fixed physical horizon and a prespecified tolerance/physical observable. Tune each baseline on validation to meet the same target. Include initialization and first-use costs separately from reusable steady-state costs.

Record:

- split/teacher method and number of calls/substeps;
- model feature, encode, decode, layout-copy, and fallback time;
- H2D/D2H and data I/O where part of the intended use;
- accepted/rejected macrosteps and actual simulated time;
- peak allocated/reserved/device-used VRAM and host RSS;
- training and teacher-generation GPU/CPU hours;
- compiled graph/kernel count and compile warmup time;
- source/installed software versions and GPU identity.

If a learned method is slower at matched tolerance, report that. Accuracy alone may remain scientifically informative, but do not label it an efficiency result.

For repeated use, define training amortization

\[
N_{\rm break}=\left\lceil\frac{C_{\rm data}+C_{\rm train}+C_{\rm setup}}
{t_{\rm baseline}-t_{\rm learned}}\right\rceil
\]

when the denominator is positive and units/workloads are consistent. If no positive denominator exists, there is no runtime break-even to report.

Do not compare FP64 references against BF16 neural paths without providing an accuracy-matched practical classical baseline. Do not hide that model inference at every cell can exceed the arithmetic of a simple split step.

---

## 19. Optional avenues, in priority order

**Ablation discipline:** change only one of the temporal basis, encoder family, leading anchor, physical correction basis, constraint mechanism, and step controller at a time. Use equal information and validation budgets. Maintain a named `method_id` for each deployed composition, including precision, fallback and solver variants.

### 19.1 Fixed versus learned relaxation rates

First fit amplitudes over a fixed logarithmic rate dictionary. If learned rates do not improve transfer/cost beyond this baseline, keep fixed rates. That simplifies gradients, identifiability, and inference.

### 19.2 Exact-leading anchor

Add an independently audited e3 constraint. Count commutator/JVP cost. Compare local/global order and finite-stiffness accuracy. The anchor is not a proof of uniform high order when e3 grows with stiffness.

### 19.3 KAN parameterization and formula extraction

Use KAN only as an alternative encoder. Prune and simplify after the full numerical experiment works. Test any extracted formula on untouched parents, h values, and physical regimes. A fitted expression is a hypothesis, not a discovered law by visual appeal. [P4,P5]

### 19.4 Structured interaction basis

Predict scalar coefficients multiplying audited spatial interaction terms, instead of full cellwise amplitudes. This can cut inference/output work. Its basis must span the relevant defect; measure the representation gap before attributing failure to the network. Nested brackets and their feature stencils may be expensive or nonlocal.

### 19.5 Oscillatory temporal units

Use a 2×2 block

\[
M_j=\begin{pmatrix}-\gamma_j&-\omega_j\\\omega_j&-\gamma_j\end{pmatrix},
\qquad z'_j=M_jz_j+s^2b_j,
\]

with `gamma>=0`. Then

\[
z_j(h)=2h^3\varphi_3(hM_j)b_j.
\]

First validate against a small augmented matrix exponential on CPU. A batched per-cell generic matrix exponential is an oracle, not an efficient A100 decoder. Derive/test real arithmetic formulas and small-frequency limits before deployment. Oscillatory modes add parameters and identifiability issues; add them only if real-rate fits fail in an interpretable way.

### 19.6 Higher-order forcing and generalized order

A forcing `s^p*a` gives

\[
z(h)=p!\,h^{p+1}\varphi_{p+1}(-\lambda h)a.
\]

Use this to construct an architecture matching a different base order, or a correction beyond a known anchor. Do not add unrestricted lower powers that destroy the base solver's consistency.

### 19.7 Conservation/positivity constraints

Where a linear invariant is genuinely present, use a flux-form amplitude or stoichiometric subspace. For positivity use a stated limiter/projection/fallback and assess how it changes order, gradients, and cost. Projection activity must be reported. Do not impose mass conservation on a reaction system that does not conserve it.

### 19.8 Tolerance/step controller

Only after the correction is accurate, train or calibrate a cheap error estimator and choose among admissible h values. Compare with ordinary step doubling or splitting-defect estimators. A model confidence score is not a numerical certificate. Evaluate mixed-step composition and count rejected/fallback steps.

### 19.9 Differentiable inverse problems

After forward success, test parameter recovery/control using the same discrete objective and exact-enough gradient audits. Derivatives must include state-dependent encoder outputs. Quantization and adaptive switches require explicit differentiation semantics. Verify the final parameters with the trusted solver rather than reporting only a surrogate objective.

### 19.10 Halo/distributed integration

Return to HALO only after single-device merit. Keep the frozen acoustic basis and audited communication path unless the new equation requires another method. Count the full operator/feature reach, all ranks, actual messages, and synchronization. Global FFT pilots cannot be relabeled local distributed methods. A one-GPU training job is not an HPC scaling result.

---

## 20. Common failure diagnoses

| Symptom | Investigate before changing architecture |
|---|---|
| Tiny h targets look random | FP64 subtraction, teacher uncertainty, state quantization, noisy h³ normalization |
| Lower h no longer improves error | spatial/reference floor, state precision, incorrect anchor sign/order |
| Temporal derivatives disagree | queried h leaked into encoder, time-dependent substep count, unsafe Psi branch |
| Rate gradients are zero initially | zero amplitude initialization; verify they activate later |
| Rates saturate / amplitudes cancel | target not represented efficiently, poor scales, insufficient temporal range |
| Great one-step fit, failed rollout | state sensitivity, wrong feature/history access, insufficient guards, physical constraints |
| Larger macrostep, no speedup | neural per-cell cost, internal reaction steps, FFT/stencil cost, fallback overhead |
| Low reported VRAM but crash on compile | workspace/capture pools, not only allocated tensors, host compile memory |
| GPU mostly idle | too-small point batches, Python loops, `.item()`, slow shared I/O, excessive workers |
| KAN is slow | materialized edge bases, symbolic hooks, per-cell Python, scatter/gather overhead |
| Slurm OOM with spare GPU memory | host RAM, tmpfs, worker replication, pinned prefetched batches |
| No GPU in job | environment/build mismatch, wrong request, missing allocation, visibility overwritten |
| Resume loses reproducibility | incomplete optimizer step, RNG/sampler/normalization not restored |
| Apparent locality gain disappears at scale | different discrete reach, FFT global dependence, uncounted feature stencil |

---

## 21. Milestone-by-milestone work orders

### M0 — Establish the implementation boundary

Inventory the repo, relevant tests, uploaded historical exports, and the latest user-reported decisions. Write `SOURCE_LEDGER.md` and `REPO_AUDIT.md`. Identify adapters rather than editing existing PDE semantics. Record unresolved physical tolerances and parameter ranges. Run read-only CARC discovery when actually connected.

### M1 — Implement numerical temporal core

Copy the reference implementation from Appendix A, then test it independently. Add shape/unit checks, safe-rate validation, and explicit h-independent encoder contracts. Preserve the CPU reference permanently. Produce `math_validation.json` with errors and software versions.

### M2 — Establish a trusted ADR baseline

Implement the Tier-A exact system and Tier-B RD baseline. Generate teacher-refinement plots and split-order plots. Profile classical corrections before neural work. Stop if the proposed workload has no useful temporal/cost gap.

### M3 — Establish information and temporal headroom

Generate a few paired multi-horizon parents. Measure defect shape, fixed-rate fits, classical fits, and hidden influence. Check that features are sufficient without future/private data. Decide whether the temporal representation should remain real-rate, be enriched, or be abandoned.

### M4 — Implement dataset and small learned pilot

Freeze parent splits and physical ranges. Implement shard readers and a deterministic sampler. Train the small MLP temporal model under eager FP32. Compare a time-conditioned MLP and fixed-rate baseline. Do not add KAN until this experiment is interpretable.

### M5 — Calibrate and optimize on the A100

Run the GPU preflight, memory calibration, short profiler, and precision/compile ladder. Log matched outputs and gradients at each change. Establish soft VRAM and host-memory margins. Validate checkpoint/resume and the pre-time-limit signal.

### M6 — Add one scientific extension

Choose either leading-order anchoring, KAN parameterization, or oscillatory units based on the preceding diagnosis. Run the matching ablation, not all branches at once. Keep the same test set unopened.

### M7 — Confirm the full solver result

Run untouched parents, mixed horizons, stiffness ranges, physical observables, and resolution transfer. Select one deployed policy on validation. Generate statistics by independent parent and record every failure. Benchmark total runtime at fixed tolerance.

### M8 — Decide the paper scope

Classify the outcome as an efficiency method, an accuracy/generalization method, an information/approximation finding, or a failed avenue. Do not imply a strong ML result if a cheap classical formula matches it. If GPU optimization consumed the gain, report it and stop before distributed expansion.

---

## 22. Status of the support package

The support package accompanying this Markdown contains:

- `reference/temporal_core.py`: an h-independent parameter MLP and stable scalar temporal decoder;
- `tests/test_temporal_core.py`: five CPU tests, including values, jets, grad/gradgrad, anchors, and global order on an exact linear system;
- `scripts/probe_carc.sh`: read-only cluster/account discovery;
- `scripts/gpu_preflight.py`: requires an actual Slurm GPU allocation;
- `scripts/submit.sh` and `train_a100.sbatch`: dry-run-first launch templates for the **to-be-implemented** `tdn.cli`;
- `configs/pilot.yaml`: proposed strict schema with required scientific fields intentionally unset;
- `results/support_validation.json`: results actually measured while preparing this package.

The five CPU tests passed in the preparation environment. Bash syntax and YAML parsing were checked. The GPU preflight, Slurm script execution, CUDA compilation, training, and ADR solvers were **not run**. No `tdn.cli` training implementation is supplied; the implementing agent must build it before submitting the template as a training job.

To rerun the CPU helper tests after extracting the package:

```bash
cd temporal_defect_carc
PYTHONPATH=. python -m pytest -q tests/test_temporal_core.py
```

These tests require PyTorch, NumPy, SciPy, mpmath, and pytest. They do not require a GPU. In CARC, obey login-node limits even for larger CPU tests; use an allocated CPU/GPU node as appropriate.

---

## 23. Final agent completion checklist

Before declaring the implementation ready, answer each item with a path to evidence:

- Which exact equation, discretization, split ordering, and teacher were implemented?
- Is the primary encoder independent of queried h, including indirect features?
- Does the temporal decoder pass small/large-argument and derivative tests?
- Does zero correction recover the trusted base solver?
- Is the anchor exact for the implemented discrete method, or only an approximation?
- Are train/validation/test parents independent and immutable?
- Are labels accurate above the reference and precision floor?
- Are feature and rollout dependency radii explicitly accounted for?
- Does the model beat fixed-rate and classical temporal corrections?
- Is KAN an independently justified improvement or just an optional adapter?
- Does the optimized path preserve the specified error/gradient tolerances?
- Is peak warm VRAM below the chosen budget with both reserved and device memory recorded?
- Are host RAM, tmpfs, pinned data, and compiler caches bounded?
- Did checkpoint/resume and USR1 handling work in an actual allocated smoke job?
- Are account, path, module, GPU, and environment choices verified rather than guessed?
- Are all timing comparisons at matched tolerance and physical horizon?
- Are failed runs and fallbacks included in results?
- Which parts remain untested, and what exactly is the next authorized command?

---

## 24. Source references and provenance

The linked primary sources below support hardware, scheduler, software, and prior-art statements. The proposed architectures, budgets, thresholds, schemas, and launch workflow are design choices in this document, not claims that these sources endorse or validate the method. Avoid copying old version-specific commands without live verification.

### Project material

- **[PROJECT-1]** User-provided `iteration4_results_export.md`, especially §§1, 2, 6, and 7. Historical 16³ CPU pilot findings and code provenance; not the later 32³ reach campaign.
- **[PROJECT-2]** User's later conversation summary: approximately 1.6% traffic coverage; frozen acoustic basis dominates; reach rule tested at 32³. Raw later logs not provided here.
- **[PROJECT-3]** Previous conversation's proposed time-differential/temporal-defect architecture. A proposal, not an experiment.

### USC CARC and Slurm

- **[C1]** [CARC GPU Programming](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming.html): typed GPU requests and 40 GB A100 constraint.
- **[C2]** [Discovery Resource Overview](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery): documented hardware/partitions, subject to live changes.
- **[C3]** [Storage File Systems](https://www.carc.usc.edu/user-guides/research-data-management/storage-file-systems): path roles, backup distinctions, RAM-backed `/tmp` and `/dev/shm`.
- **[C4]** [Running Jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs): account discovery, limits, job management, login-node restrictions.
- **[C5]** [Getting Started with Discovery](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery): default-account behavior and batch workflow.
- **[C6]** [Using Conda](https://www.carc.usc.edu/user-guides/hpc-systems/software/conda): module-based user environment setup.
- **[C7]** [PyTorch Installation](https://www.carc.usc.edu/user-guides/data-science/pytorch-installation): cluster installation workflow; its historical package versions are not automatic current choices.
- **[C8]** [January 2026 CARC maintenance notice](https://hpc-discourse.usc.edu/t/important-scheduled-carc-maintenance-postponed-jan-30-feb-1-2026/1199): announced removal of old `/project` and scratch rebuild; verify actual present mounts.
- **[S1]** [Slurm `sbatch` manual](https://slurm.schedmd.com/sbatch.html): literal directive parsing, signal semantics, resource/dependency options.

### NVIDIA and PyTorch

- **[N1]** [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/): A100 architecture and memory/tensor-core characteristics.
- **[N2]** [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): measurement, memory access, transfer, and kernel optimization principles.
- **[T1]** [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda): device visibility, allocator, precision, and CUDA graph constraints.
- **[T2]** [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html): operation-specific autocast and gradient scaling; inspect the installed release.
- **[T3]** [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile): modes and their workspace/capture behavior.
- **[T4]** [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): transfer, gradient, fusion, layout, and CPU/thread guidance.
- **[T5]** [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint): recomputation, RNG/state restrictions, explicit `use_reentrant=False` recommendation.
- **[T6]** [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html): workers, pinning, prefetch, iterable/map-style semantics.

### Mathematical and model prior art

- **[P1]** [Hypersolvers: Toward Fast Continuous-Depth Models, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f1686b4badcf28d33ed632036c7ab0b8-Abstract.html).
- **[P2]** [Closed-form continuous-time neural networks, Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00556-7).
- **[P3]** [A Krylov subspace algorithm for evaluating the phi-functions appearing in exponential integrators](https://arxiv.org/abs/0907.4631): matrix-function actions and phi-function context.
- **[P4]** [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756).
- **[P5]** [KAN 2.0: Kolmogorov–Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205).
- **[P6]** [Deep-OSG / semigroup-informed operator learning](https://arxiv.org/abs/2302.03358): inspect the current paper before finalizing the baseline.
- **[P7]** [Analysis of operator splitting in the non-asymptotic regime for nonlinear reaction–diffusion equations](https://arxiv.org/abs/1402.1828): stiff/non-asymptotic splitting limitations.

**End of the main specification.** Appendices below include the exact support files so this Markdown can be used independently as an implementing-agent prompt.


---

## Appendix A. Temporal-core reference

**File to create:** `reference/temporal_core.py`.

Also create an empty `reference/__init__.py`. This reference is not the full PDE model.

```python
"""CPU-testable temporal units, not an implemented ADR solver or a CUDA benchmark.

Shapes: amplitudes [..., C, M], rates broadcastable to amplitudes,
        h scalar or broadcastable after the caller adds channel/mode axes.
Rates and amplitudes MUST be independent of the queried h for the stated jets.
All quantities may be nondimensional. A wrapper must restore physical units.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def psi3(x: Tensor) -> Tensor:
    """Integral int_0^1 exp(-x*(1-r))*r**2 dr for finite x >= 0.

    Preserve float64; promote lower precision to float32. The caller validates
    nonnegative finite inputs outside a compiled hot path. Both torch.where
    branches are finite, including at x=0, to avoid NaN gradients.
    This is a correctness reference; profile before replacing it with a kernel.
    """
    if not x.is_floating_point():
        raise TypeError("psi3 expects a real floating-point tensor")
    y = x if x.dtype == torch.float64 else x.float()
    cut = 0.5
    s = torch.clamp(y, min=0.0, max=cut)
    degree = 14
    value = torch.full_like(s, 2.0 * (-1.0)**degree / math.factorial(degree + 3))
    for n in range(degree - 1, -1, -1):
        value = value * s + 2.0 * (-1.0)**n / math.factorial(n + 3)
    # Inverse form avoids x**3 overflow at large x. Its small-x cancellation
    # is bypassed by the polynomial branch. Do not use an unclamped 0/0 branch.
    safe = torch.clamp(y, min=cut)
    inv = safe.reciprocal()
    large = inv * (1.0 - 2.0 * inv - 2.0 * torch.expm1(-safe) * inv.square())
    return torch.where(y <= cut, value, large)


def mode_response(amplitudes: Tensor, rates: Tensor, h: Tensor) -> Tensor:
    """Return each mode's value without reducing the last (mode) axis."""
    dtype = torch.float64 if amplitudes.dtype == torch.float64 else torch.float32
    a = amplitudes.to(dtype)
    lam = rates.to(device=a.device, dtype=dtype)
    step = h.to(device=a.device, dtype=dtype)
    return step.pow(3) * a * psi3(lam * step)


def temporal_defect(amplitudes: Tensor, rates: Tensor, h: Tensor) -> Tensor:
    return mode_response(amplitudes, rates, h).sum(dim=-1)


def temporal_jets(amplitudes: Tensor, rates: Tensor, h: Tensor) -> tuple[Tensor, ...]:
    """E, dE/dh, d2E/dh2, d3E/dh3, holding encoded parameters fixed."""
    z = mode_response(amplitudes, rates, h)
    a = amplitudes.to(z.dtype)
    lam = rates.to(device=z.device, dtype=z.dtype)
    step = h.to(device=z.device, dtype=z.dtype)
    first = -lam * z + step.square() * a
    second = -lam * first + 2.0 * step * a
    third = -lam * second + 2.0 * a
    return tuple(t.sum(dim=-1) for t in (z, first, second, third))


def anchored_amplitudes(free: Tensor, leading_defect: Tensor) -> Tensor:
    """Given M-1 free modes, enforce sum_m a_m = 3*e3 exactly in arithmetic.

    free [..., C, M-1]; leading_defect [..., C]. Floating-point cancellation
    must still be measured. This does not compute the correct e3 for a solver.
    """
    last = 3.0 * leading_defect - free.sum(dim=-1)
    return torch.cat((free, last.unsqueeze(-1)), dim=-1)


@dataclass(frozen=True)
class EncodedModes:
    amplitudes: Tensor
    rates: Tensor


class TemporalParameterMLP(nn.Module):
    """Small h-independent parameter generator used as an architectural reference.

    Input [..., F]. Outputs [..., C, M] amplitudes and [..., 1, M] rates.
    Zero amplitude initialization gives zero correction. Initial rate gradients
    are consequently zero; this is expected, not a broken autograd graph.
    No batch statistics, dropout, coordinates, or queried step enter by default.
    """
    def __init__(self, features: int, channels: int, modes: int = 4,
                 width: int = 64, anchored: bool = False) -> None:
        super().__init__()
        if min(features, channels, modes, width) <= 0:
            raise ValueError("all dimensions must be positive")
        if anchored and modes < 2:
            raise ValueError("this reference's learned anchor uses at least 2 modes")
        self.channels, self.modes, self.anchored = channels, modes, anchored
        self.body = nn.Sequential(nn.Linear(features, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        count = modes - 1 if anchored else modes
        self.amplitude = nn.Linear(width, channels * count)
        self.rate = nn.Linear(width, modes)
        nn.init.zeros_(self.amplitude.weight)
        nn.init.zeros_(self.amplitude.bias)
        nn.init.zeros_(self.rate.weight)
        desired = torch.logspace(-1, 2, modes)
        # Stable inverse softplus, desired = softplus(bias).
        with torch.no_grad():
            self.rate.bias.copy_(desired + torch.log(-torch.expm1(-desired)))

    def encode(self, features: Tensor, leading_defect: Tensor | None = None) -> EncodedModes:
        hidden = self.body(features)
        count = self.modes - 1 if self.anchored else self.modes
        # Preserve float64 for audits; explicit FP32 head output under BF16 AMP.
        raw_a = self.amplitude(hidden)
        raw_lam = self.rate(hidden)
        if raw_a.dtype != torch.float64:
            raw_a = raw_a.float()
            raw_lam = raw_lam.float()
        amplitudes = raw_a.reshape(*features.shape[:-1], self.channels, count)
        if self.anchored:
            if leading_defect is None:
                raise ValueError("anchored model requires the independently audited e3")
            amplitudes = anchored_amplitudes(amplitudes, leading_defect.to(amplitudes.dtype))
        rates = F.softplus(raw_lam).unsqueeze(-2)
        return EncodedModes(amplitudes, rates)

    def forward(self, features: Tensor, h: Tensor,
                leading_defect: Tensor | None = None) -> Tensor:
        modes = self.encode(features, leading_defect)
        return temporal_defect(modes.amplitudes, modes.rates, h)
```


---

## Appendix B. CPU mathematical tests

**File to create:** `tests/test_temporal_core.py`.

```python
from __future__ import annotations
import math
import numpy as np
import mpmath as mp
import torch
from scipy.linalg import expm
from reference.temporal_core import (psi3, mode_response, temporal_defect,
    temporal_jets, anchored_amplitudes, TemporalParameterMLP)


def oracle(x: float) -> float:
    with mp.workdps(90):
        z = mp.mpf(float(x))
        if z == 0:
            return 1.0/3.0
        return float((z*z - 2*z + 2 - 2*mp.exp(-z)) / (z*z*z))


def test_values():
    grid = np.r_[0.0, np.logspace(-12, 6, 221), 0.5-1e-8, 0.5, 0.5+1e-8]
    for dtype, tolerance in [(torch.float64, 3e-13), (torch.float32, 5e-6)]:
        x = torch.tensor(grid, dtype=dtype)
        ref = np.array([oracle(v) for v in x.double().numpy()])
        out = psi3(x).double().numpy()
        assert np.max(np.abs((out-ref)/ref)) < tolerance
        assert np.all(out > 0)
    assert psi3(torch.tensor([0.,1.], dtype=torch.bfloat16)).dtype == torch.float32


def test_jets():
    a = torch.tensor([[[0.3,-0.4,0.6]]], dtype=torch.float64)
    rates = torch.tensor([[[0.0,1.2,8.0]]], dtype=torch.float64)
    for hv in [0.0,1e-3,0.04,0.5,2.0]:
        h = torch.tensor(hv, dtype=torch.float64, requires_grad=True)
        jets = temporal_jets(a,rates,h)
        f = jets[0].sum()
        for j in range(1,4):
            f, = torch.autograd.grad(f,h,create_graph=True)
            assert torch.allclose(f,jets[j].sum(),atol=2e-11,rtol=2e-10)


def test_gradcheck():
    a = torch.tensor([[0.4,-0.3,0.1]],dtype=torch.float64,requires_grad=True)
    lam = torch.tensor([[0.03,0.4,8.0]],dtype=torch.float64,requires_grad=True)
    h = torch.tensor(0.2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(temporal_defect,(a,lam,h),atol=2e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(temporal_defect,(a,lam,h),atol=3e-6,rtol=1e-4)


def test_anchor_and_zero():
    e3 = torch.tensor([[0.2,-0.7]],dtype=torch.float64)
    free = torch.tensor([[[0.3,-0.1],[0.9,0.2]]],dtype=torch.float64)
    a = anchored_amplitudes(free,e3)
    assert torch.allclose(a.sum(-1),3*e3)
    rates = torch.tensor([[[0.1,1.,10.]]],dtype=torch.float64)
    jets = temporal_jets(a,rates,torch.tensor(0.,dtype=torch.float64))
    assert all(torch.count_nonzero(t)==0 for t in jets[:3])
    assert torch.allclose(jets[3],6*e3)
    net = TemporalParameterMLP(8,2,4,64).double()
    x = torch.randn(16,8,dtype=torch.float64)
    assert torch.count_nonzero(net(x,torch.tensor(.4,dtype=torch.float64)))==0
    # A single encoder evaluation reused at several h is the intended contract.
    encoded = net.encode(x)
    assert encoded.amplitudes.shape == (16,2,4)
    assert encoded.rates.shape == (16,1,4)


def leading_matrix(A, B):
    result = np.zeros_like(A)
    for p in range(4):
        for q in range(4-p):
            r=3-p-q
            result += (np.linalg.matrix_power(A/2,p)/math.factorial(p)
                       @ (np.linalg.matrix_power(B,q)/math.factorial(q))
                       @ (np.linalg.matrix_power(A/2,r)/math.factorial(r)))
    return np.linalg.matrix_power(A+B,3)/6-result


def test_global_order_on_exact_linear_system():
    A = np.array([[-.3,.8],[-.1,-.2]])
    B = np.array([[-1.0,.1],[.4,-.8]])
    u0 = np.array([.7,-.4]); T=1.
    E3=leading_matrix(A,B)
    errs=[]
    for n in [8,16,32,64]:
        h=T/n
        S=expm(.5*h*A)@expm(h*B)@expm(.5*h*A)
        # lambda=0, amplitude=3*e3 gives h**3*e3.
        M=S+h**3*E3
        got=np.linalg.matrix_power(M,n)@u0
        errs.append(np.linalg.norm(got-expm(T*(A+B))@u0))
    orders=np.log2(np.array(errs[:-1])/np.array(errs[1:]))
    assert np.min(orders[-2:]) > 2.8
    assert np.max(orders[-2:]) < 3.2
```


---

## Appendix C. Proposed pilot manifest

**File to create:** `configs/pilot.yaml`.

Null scientific fields are intentionally blocking. The agent must fill them from an audited pilot/design, not submit this file unchanged as a confirmatory run.

```yaml
# Proposed strict schema for an implementing agent, NOT an existing HALO config.
schema_version: 1
experiment: tdn_rd_pilot
seed: 0
stage: regression
problem:
  family: scalar_logistic_reaction_diffusion
  dimension: 2
  channels: 1
  grid: [64, 64]
  periodic: true
  reference_time_scale: 1.0  # fixed per problem, never chosen from requested h
  reference_state_scale: 1.0
  split: reaction_half_diffusion_full_reaction_half
  diffusion_backend: exact_discrete_periodic_fft
  teacher_backend: audited_refined_coupled_rk4
  teacher_precision: float64
  teacher_error_fraction_of_defect: 0.05
  parameter_manifest: null  # REQUIRED: freeze physical ranges after stiffness audit
  split_manifest: null      # REQUIRED: parent-level train/validation/test lists
model:
  family: temporal_parameter_mlp
  feature_count: 16          # audited feature construction; zero-pad to 16 if needed
  width: 64
  hidden_layers: 2
  modes: 4
  shared_rates_across_species: true
  h_independent_encoder: true
  anchored_leading_defect: false
  rate_parameterization: softplus
  amplitude_initialization: zero
training:
  max_optimizer_steps: 5000
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.00001
  grad_clip_norm: 1.0
  encoded_points_per_microbatch: 32768
  gradient_accumulation_steps: 1
  horizons_per_anchor: 4
  rollout_windows: 1
  patch_core: [32, 32]
  guard_cells: null          # REQUIRED for patch-rollout stage, based on dependency audit
  derivative_loss_fraction_of_batches: 0.0
  composition_loss_fraction_of_batches: 0.0
  num_workers: 2
  prefetch_factor: 2
  pin_memory: true
precision:
  teacher: float64
  state: float32
  psi_and_accumulation: float32
  network_autocast: bfloat16 # enable ONLY after FP32 baseline agreement
  amp_enabled: false
  tf32_enabled: false
performance:
  compile_mode: eager       # advance to default, then benchmark alternatives
  cuda_graphs: false
  checkpoint_non_reentrant: false
  fixed_shape_buckets: true
  soft_vram_gib_cap: 30
  soft_vram_fraction: 0.80
  hard_warning_fraction: 0.90
  cell_chunk_size: 32768
  intraop_threads: 1
  interop_threads: 1
  empty_cache_in_hot_loop: false
validation:
  every_steps: 250
  checkpoint_every_steps: 500
  checkpoint_every_seconds: 600
  primary_metric: full_rollout_error_at_fixed_physical_horizon
  tolerance_manifest: null  # REQUIRED before confirmatory evaluation
  report_all_failed_runs: true
```


---

## Appendix D1. Read-only CARC discovery

**File to create:** `scripts/probe_carc.sh`.

```bash
#!/usr/bin/env bash
# Read-only account/queue discovery. No job submission, installation or GPU work.
set -u
printf '\n== Identity and scheduler ==\n'
hostname; id; command -v sbatch || true
printf '\n== Authorized accounts and quotas ==\n'
command -v myaccount >/dev/null && myaccount || true
command -v myquota >/dev/null && myquota || true
printf '\n== GPU partition and actual node features ==\n'
command -v sinfo >/dev/null && sinfo -p gpu -N -o '%N %G %f %m %c' || true
command -v scontrol >/dev/null && scontrol show partition gpu || true
printf '\n== Module inventory ==\n'
if type module >/dev/null 2>&1; then
  module avail conda 2>&1 || true
  module avail cuda 2>&1 || true
else
  printf 'Module shell function is unavailable in this shell; use the CARC login environment.\n'
fi
printf '\nDo not infer access to Endeavour/condo partitions from their visibility.\n'
printf 'Choose writable existing project and scratch roots from myquota; do not guess them.\n'
```


---

## Appendix D2. User configuration template

**File to create:** `scripts/user.env.example.sh`.

```bash
# COPY to a private local config; replace all CHOOSE_ values after probe_carc.sh.
# Do not put passwords, tokens, SSH keys or other secrets in this file.
export TDN_REPO_ROOT="CHOOSE_ABSOLUTE_PATH_TO_IMPLEMENTED_REPOSITORY"
export TDN_PROJECT_ROOT="CHOOSE_EXISTING_WRITABLE_PERSISTENT_DIRECTORY"
export TDN_SCRATCH_ROOT="CHOOSE_EXISTING_WRITABLE_SCRATCH_DIRECTORY"
export TDN_ENV_PREFIX="CHOOSE_ABSOLUTE_CONDA_ENV_PREFIX"
# Exact authorized project association from myaccount, not your username.
# Leave empty only after confirming that your default account works on gpu.
export CARC_ACCOUNT=""
export TDN_PARTITION="gpu"
export TDN_GPU_REQUEST="a100:1"
export TDN_GPU_CONSTRAINT="a100-40gb"
export TDN_CPUS="8"
export TDN_HOST_MEM="64G"
export TDN_WALLTIME="04:00:00"
export TDN_NUM_WORKERS="2"
export TDN_RESUME="none"
```


---

## Appendix D3. Dry-run-first submission wrapper

**File to create:** `scripts/submit.sh`.

```bash
#!/usr/bin/env bash
# Submission TEMPLATE for the tdn.cli interface to be implemented by the agent.
# Dry run by default. Only --submit calls sbatch. No conda installation here.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 STAGE CONFIG_PATH [--submit]" >&2; exit 2
fi
stage="$1"; config="$(realpath "$2")"; submit="${3:---dry-run}"
case "$submit" in --submit|--dry-run) ;; *) echo 'Unknown option' >&2; exit 2;; esac
case "$stage" in audit|generate|train|evaluate|benchmark) ;; *) echo 'Unknown stage' >&2; exit 2;; esac
for name in TDN_REPO_ROOT TDN_PROJECT_ROOT TDN_SCRATCH_ROOT TDN_ENV_PREFIX; do
  value="${!name:-}"
  if [[ "$value" != /* || "$value" == *CHOOSE_* || ! -d "$value" ]]; then
    echo "$name must be an existing absolute directory chosen on CARC" >&2; exit 2
  fi
done
[[ -f "$config" ]] || { echo 'Config missing' >&2; exit 2; }
[[ -x "$TDN_ENV_PREFIX/bin/python" ]] || { echo 'Environment Python missing' >&2; exit 2; }
[[ -w "$TDN_PROJECT_ROOT" && -w "$TDN_SCRATCH_ROOT" ]] || { echo 'Roots not writable' >&2; exit 2; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TDN_STAGE="$stage" TDN_CONFIG="$config"
export TDN_RUN_DIR="${TDN_RUN_DIR:-$TDN_PROJECT_ROOT/runs/${stage}_$(date +%Y%m%dT%H%M%S)_${RANDOM}}"
export TDN_RESUME="${TDN_RESUME:-none}"
logroot="$TDN_PROJECT_ROOT/logs"
args=(sbatch --parsable
      --partition="${TDN_PARTITION:-gpu}"
      --gpus-per-task="${TDN_GPU_REQUEST:-a100:1}"
      --constraint="${TDN_GPU_CONSTRAINT:-a100-40gb}"
      --cpus-per-task="${TDN_CPUS:-8}"
      --mem="${TDN_HOST_MEM:-64G}"
      --time="${TDN_WALLTIME:-04:00:00}"
      --output="$logroot/%x-%j.out"
      --error="$logroot/%x-%j.err")
if [[ -n "${CARC_ACCOUNT:-}" ]]; then args+=(--account="$CARC_ACCOUNT"); fi
args+=("$script_dir/train_a100.sbatch")
printf 'Stage: %s\nConfig: %s\nRun directory: %s\n' "$stage" "$config" "$TDN_RUN_DIR"
printf 'Command: '; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$submit" != --submit ]]; then
  echo 'DRY RUN: no directories created and no job submitted.'; exit 0
fi
command -v sbatch >/dev/null || { echo 'sbatch unavailable' >&2; exit 2; }
mkdir -p "$logroot" "$TDN_RUN_DIR"
# No eval; explicit argument array. Requires tdn.cli to exist in the selected env.
job_id="$("${args[@]}")"
printf '%s\n' "$job_id" | tee "$TDN_RUN_DIR/slurm_job_id.txt"
```


---

## Appendix D4. One-A100 Slurm batch template

**File to create:** `scripts/train_a100.sbatch`.

The `tdn.cli` target must be implemented and installed first. Bash syntax was checked; this job has not been submitted. Copy `scripts/gpu_preflight.py` into the implemented repository at the path used below.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=tdn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@180
#SBATCH --open-mode=append
# Account and absolute output paths are supplied by submit.sh, not shell variables
# inside SBATCH directives. These defaults do not prove access to this partition.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm, not bash on a login node}"
: "${TDN_REPO_ROOT:?}" "${TDN_PROJECT_ROOT:?}" "${TDN_SCRATCH_ROOT:?}"
: "${TDN_ENV_PREFIX:?}" "${TDN_RUN_DIR:?}" "${TDN_STAGE:?}" "${TDN_CONFIG:?}"
module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TDN_ENV_PREFIX"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Shared scratch on CARC, NOT a presumed local NVMe disk. /tmp is RAM-backed.
work="$TDN_SCRATCH_ROOT/tdn-work/$SLURM_JOB_ID"
mkdir -p "$work/tmp" "$work/inductor" "$work/triton" "$work/extensions" "$TDN_RUN_DIR"
export TMPDIR="$work/tmp"
export TORCHINDUCTOR_CACHE_DIR="$work/inductor"
export TRITON_CACHE_DIR="$work/triton"
export TORCH_EXTENSIONS_DIR="$work/extensions"
# Set only for compiled extensions, never as a claim that the installed wheel works.
export TORCH_CUDA_ARCH_LIST="8.0"
cd "$TDN_REPO_ROOT"
python -u "$TDN_REPO_ROOT/scripts/gpu_preflight.py" --output "$TDN_RUN_DIR/gpu_preflight.json"
# Default Slurm --signal (without B:) signals job steps. Main Python must catch
# USR1 and write an atomic checkpoint at a safe boundary. Worker processes should
# ignore USR1. No automatic requeue, resubmission, or filesystem deletion.
exec srun --ntasks=1 --unbuffered python -u -m tdn.cli "$TDN_STAGE" \
  --config "$TDN_CONFIG" --run-dir "$TDN_RUN_DIR" --resume "${TDN_RESUME:-none}"
```


---

## Appendix D5. Allocated-GPU preflight

**File to create:** `scripts/gpu_preflight.py`.

```python
"""Run on an allocated GPU node only; this script does not submit jobs."""
from __future__ import annotations
import argparse, json, os, platform, subprocess
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Refusing GPU preflight outside a Slurm allocation')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable in selected environment; do not start training')
    if torch.cuda.device_count()!=1:
        raise SystemExit('Primary pipeline requires exactly one visible GPU')
    props=torch.cuda.get_device_properties(0)
    total=props.total_memory; free,_=torch.cuda.mem_get_info()
    if 'A100' not in props.name or (props.major,props.minor)!=(8,0):
        raise SystemExit(f'Wrong device: {props.name}, capability {props.major}.{props.minor}')
    if not (35*1024**3 <= total <= 45*1024**3):
        raise SystemExit(f'Expected full A100 40 GB-class device, found {total/1024**3:.2f} GiB')
    x=torch.arange(1024,device='cuda',dtype=torch.float32)
    assert torch.isfinite((x*x).sum()).item()
    torch.cuda.synchronize()
    smi=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                         '--format=csv,noheader'],capture_output=True,text=True,check=False)
    data={'python':platform.python_version(),'torch':torch.__version__,
          'torch_cuda_runtime':torch.version.cuda,'gpu_name':props.name,
          'compute_capability':[props.major,props.minor],
          'total_bytes':total,'free_bytes_at_start':free,
          'soft_reserved_budget_bytes':int(min(30*1024**3,.80*total)),
          'bfloat16_supported':torch.cuda.is_bf16_supported(),
          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'slurm_job_id':os.environ['SLURM_JOB_ID'],
          'nvidia_smi_note':'May list physical GPUs beyond the task-visible mapping.',
          'nvidia_smi':smi.stdout.strip()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
if __name__=='__main__': main()
```


---

## Appendix E. Measured support-validation record

**File to create:** `results/support_validation.json`.

```json
{
  "scope": "CPU support checks only; no GPU, CARC, Slurm execution, ADR training or PDE benchmark.",
  "environment": {
    "python": "3.13.5",
    "torch": "2.10.0+cpu",
    "cuda_available": false,
    "numpy": "2.3.5",
    "scipy": "1.17.0"
  },
  "pytest_exit_code": 0,
  "pytest_stdout": ".....                                                                    [100%]\n5 passed in 1.84s",
  "psi_max_relative_error": {
    "torch.float32": 7.410736190010583e-07,
    "torch.float64": 2.066338914451926e-15
  },
  "linear_anchor_global_errors_n8_16_32_64": [
    1.4374923893458662e-05,
    1.737205462864861e-06,
    2.13514455289362e-07,
    2.6464894680720746e-08
  ],
  "linear_anchor_observed_orders": [
    3.0487140181148313,
    3.024362743494126,
    3.0121819282736464
  ],
  "bash_syntax": [
    {
      "path": "scripts/probe_carc.sh",
      "bash_syntax_exit": 0
    },
    {
      "path": "scripts/submit.sh",
      "bash_syntax_exit": 0
    },
    {
      "path": "scripts/train_a100.sbatch",
      "bash_syntax_exit": 0
    },
    {
      "path": "scripts/user.env.example.sh",
      "bash_syntax_exit": 0
    }
  ],
  "yaml_parse_success": true,
  "unimplemented": [
    "tdn.cli",
    "ADR operators/teachers/training loop",
    "GPU optimized kernels"
  ],
  "not_run": [
    "gpu_preflight.py",
    "sbatch",
    "CUDA compilation",
    "A100 benchmark"
  ]
}
```


---

## Appendix F. Reproducible support-validator

**File to create:** `reference/validate_support.py`.

```python
"""Re-run CPU helper tests and write factual preparation metrics, never GPU claims."""
from __future__ import annotations
import json, math, platform, subprocess, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import expm
import torch
import mpmath as mp
from reference.temporal_core import psi3


def high_precision(x: float) -> float:
    with mp.workdps(90):
        y=mp.mpf(float(x))
        return float(mp.mpf(1)/3 if y==0 else
                     (y*y-2*y+2-2*mp.exp(-y))/(y*y*y))


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_temporal_core.py'],
                        cwd=root,capture_output=True,text=True,check=False)
    metrics={}
    grid=np.r_[0.,np.logspace(-12,6,221),.5-1e-8,.5,.5+1e-8]
    for dtype in [torch.float32,torch.float64]:
        x=torch.tensor(grid,dtype=dtype)
        y=psi3(x).double().numpy()
        ref=np.array([high_precision(float(v)) for v in x.double().numpy()])
        metrics[str(dtype)]=float(np.max(np.abs((y-ref)/ref)))
    A=np.array([[-.3,.8],[-.1,-.2]])
    B=np.array([[-1.,.1],[.4,-.8]])
    coeff=np.zeros_like(A)
    for p in range(4):
        for q in range(4-p):
            r=3-p-q
            coeff += ((np.linalg.matrix_power(A/2,p)/math.factorial(p))
                     @ (np.linalg.matrix_power(B,q)/math.factorial(q))
                     @ (np.linalg.matrix_power(A/2,r)/math.factorial(r)))
    e3=np.linalg.matrix_power(A+B,3)/6-coeff
    u0=np.array([.7,-.4]);errs=[]
    for n in [8,16,32,64]:
        h=1/n;S=expm(h*A/2)@expm(h*B)@expm(h*A/2)
        errs.append(float(np.linalg.norm(np.linalg.matrix_power(S+h**3*e3,n)@u0-expm(A+B)@u0)))
    scripts=[]
    for f in sorted((root/'scripts').glob('*')):
        if f.suffix in ('.sh','.sbatch'):
            p=subprocess.run(['bash','-n',str(f)],capture_output=True,text=True)
            scripts.append({'path':str(f.relative_to(root)),'bash_syntax_exit':p.returncode})
    import yaml
    config=yaml.safe_load((root/'configs/pilot.yaml').read_text())
    result={'scope':'CPU support checks only; no GPU, CARC, Slurm execution, ADR training or PDE benchmark.',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,
                           'cuda_available':torch.cuda.is_available(),'numpy':np.__version__,
                           'scipy':scipy.__version__},
            'pytest_exit_code':test.returncode,'pytest_stdout':test.stdout.strip(),
            'psi_max_relative_error':metrics,
            'linear_anchor_global_errors_n8_16_32_64':errs,
            'linear_anchor_observed_orders':np.log2(np.array(errs[:-1])/np.array(errs[1:])).tolist(),
            'bash_syntax':scripts,'yaml_parse_success':isinstance(config,dict),
            'unimplemented':['tdn.cli','ADR operators/teachers/training loop','GPU optimized kernels'],
            'not_run':['gpu_preflight.py','sbatch','CUDA compilation','A100 benchmark']}
    out=root/'results/support_validation.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if test.returncode or any(v['bash_syntax_exit'] for v in scripts):
        raise SystemExit(1)
if __name__=='__main__':main()
```


---

## Closing execution instruction

Start with M0 and M1. Preserve the separation between CPU reference correctness and GPU-optimized correctness. Do not launch long jobs until the numerical teacher, available information, and temporal representation each show useful headroom. Produce a working small method, an honest cost profile, and a reproducible decision before expanding the architecture or cluster footprint.
