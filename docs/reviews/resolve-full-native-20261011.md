# Resolve full native review — 11 October 2026

**The strongest positive result is a faster implementation of the same analytic interaction rule. The strongest scientific diagnosis is that finite-step interaction representation, including its mean response, now matters more than optimizing two global gains. The neural comparison remains unresolved as a comparison with competitive FNOs.**

Run: `fedora-resolve-20261011T052313346218Z`; native commit `f07f9d56ccaee85b73b8d090aec3bc22ac067cf9`. This is a comprehensive **development** campaign. It contains no untouched confirmation cohort. This review adds explicitly post-hoc analysis; it does not alter the original results or scores.

Detailed source-traced reviews, numerical tables, audit scripts and a high-resolution overview are in [the review package](../../results/resolve-native-review/README.md). The interpretation below distinguishes the specified finite equation from a spatially refined continuum estimate. Files call the spectral track `continuum`; where only its finite Galerkin system was verified, this review says **Galerkin**.

## What completed, and what was actually verified

All **28 jobs, 847–874, completed**. The CPU suite passed 252 cases. The 20 distinct mandatory CUDA cases passed in each of ten GPU allocations: **200 executions**, not 200 different scientific tests. No required mathematical/correctness check failed.

All four upload parts, their concatenated 86,547,084-byte archive and the optional manifest match their hashes. Transport verification covered all **3,716 retained files**. An independently rehashed canonical subset contains 681 files; 28 stage seals, dependencies, identities and execution bindings agree. Native software was Python 3.13.13 in the project venv and PyTorch 2.10.0+cu126, on the RTX 4090 where CUDA was requested.

The compact export intentionally omitted 626 arrays, checkpoints and rendered figures, totaling 157.7 MB. Their declared hashes are retained, but their actual bytes, checkpoint reload behavior and original rendered appearance were **not independently verified here**. This review is an artifact audit and numerical reconstruction from retained measurements, not a complete binary replay.

The 2,048 experiment rows contain 198 GOOD, 409 BAD and 1,441 NA verdicts, with 11,058 check rows. These are heterogeneous evidence checks and repeated conditions. Their proportions are neither model accuracy nor a scientific success rate. Successful jobs, mathematical correctness and useful scientific improvement remain separate outcomes.

## Evidence map and decisions

| Path | What the measurements support | Main remaining limit | Decision |
|---|---|---|---|
| R1: direct coefficient fitting | Two global gains modestly improve their fixed parent; Adam and the direct convex solve reach essentially the same objective | Both remain substantially less accurate than GL4 and nested quadratic+cubic | Stop more optimizer sweeps for this fixed basis; retain the direct fit as a control |
| R2: finite-step interactions | The leading filtered cubic term frequently opposes the missing finite-step correction; nested response usually helps much more | Some fine-grid targets are unresolved; nested time quadrature is not independently converged | Replace or selectively omit the unsuitable response before increasing its conditioner |
| R3: FNO opportunity | Every setting received the declared long trial; 8/12 selected FNO instances are trained | Tiny memorization remains weak; physical output scaling varies enormously; no authoritative FNO reproduction | Repair and causally test scaling and objective conditioning before interpreting neural rankings |
| R4: midpoint-null GL3 | Its mathematical construction is valid | No material speed benefit over optimized GL4; fewer accuracy-qualified conditions | Retain as a quadrature diagnostic; stop promoting its current 4090 latency case |
| R5: shared-transform GL4 | The same rule runs about 1.5–2.3× faster with small measured FP32 disagreement | Narrow field population; no fastest-classical or end-to-end deployment claim | Promote it to the strengthened analytic baseline |
| RX: memory | Clean fitted memory can help clean histories; validation selects zero gain for mixed noise conditions | Noise amplification, interval violations, no robust advantage over coarse RK4 | Stop the present noisy closure; protect a small observer-first exploratory test |

The campaign substantially weakens “we simply have not optimized the two gains enough” and “GL3 will be faster because it uses fewer active nodes.” It strengthens a finite-step representation explanation. It does **not** prove that learning cannot help, that the scalar scale is the sole FNO problem, or that nested cubic is an exact target.

## What these models are

The common physical problem is periodic logistic reaction–diffusion,

\[
u_t=Lu+r u(1-u),\qquad L=\kappa\Delta,
\]

with separately specified finite-difference/nodal and projected Galerkin operators. The analytic diffusion–reaction–diffusion map is denoted \(S_h^{DF}\). The Galerkin reaction part requires projected numerical work; its cost is not that of the simple pointwise logistic formula.

The current two-basis model predicts

\[
\widehat u=S_h^{DF}(u)+g_2Q_{GL2,h}(u)+g_3C_{\mathrm{lead},h}(u).
\]

Its spatial responses are computed from the full current state and known physics. In this campaign, only **two global coefficients** are fitted. It is a physics-based corrected time-stepper, not a large neural state representation. The direct solver and two-parameter neural parameterization are alternative ways to fit the same response family. The historical state-dependent conditioners are not all re-evaluated in this campaign.

GL2, GL3, GL4 and nested quadratic+cubic are analytic controls. Batched GL4 changes transform reuse and execution, not its mathematical response or weights. The FNO controls are locally adapted hybrid correctors sharing the physical backbone: width 16, depth 2, four retained modes, approximately 47,000 parameters. They are labeled **Theirs — adapted FNO**, not reproductions of the published FNO benchmark. RX is a separate one-dimensional coarse-state memory study, not a 2D version of the same architecture.

## R1: fitting is largely solved, accuracy is not

The direct objective is convex in the two coefficients: a positive sum of squared affine residuals and maximum squared residuals on the box \([0.25,1.75]^2\). Supporting planes yield numerical lower bounds, not interval-arithmetic certificates. Final gaps are **8.77e−7 / 6.44e−7**, within the declared tolerance. Adam is slightly better than the returned direct iterate by less than these gaps. The resulting gains are approximately **(1.0564, 0.9118)** and **(1.0725, 0.8456)**.

The following values are arithmetic field/horizon mean RMS; lower is better. There are 32 independent validation fields and four repeated horizons per track, on 32². These are held-out from fitting but remain development data. In R1 and R3, the `continuum` target is a **spatially refined 2N/4N Galerkin reference projected to N**, whereas the model operates at N. Its error therefore includes unresolved-scale/closure discrepancy as well as time-integration and representation error. It is not the same-grid finite-equation target used by R4–R5. R2 supplies the separate decomposition needed to attribute these terms.

| Method | Discrete RMS | Continuum-estimate RMS | Joint RMS/max passes, discrete /128 | Continuum /128 |
|---|---:|---:|---:|---:|
| Fixed two-basis parent | 1.076e−5 | 1.169e−5 | 95 | 103 |
| **Ours: direct two-gain fit** | **9.913e−6** | **1.041e−5** | **98** | **108** |
| Ours: fitted quadratic only, cubic omitted | 9.739e−6 | 1.020e−5 | 98 | 107 |
| Analytic GL2 | 1.030e−5 | 1.118e−5 | 95 | 103 |
| Analytic GL4 | 3.155e−6 | 3.699e−6 | 112 | 114 |
| Analytic nested quadratic+cubic | 1.699e−6 | 2.720e−6 | 124 | 125 |

The joint pass uses uncertainty-inflated RMS and maximum error, both at most 2e−5. Across all four horizons, ours passes on only **15/32 and 19/32 fields**; GL4 passes on 18/32 and 23/32; nested passes on 28/32 and 30/32.

Calibration improves its parent by **7.85% / 10.99%** in mean RMS. Yet its error remains **3.14× / 2.81× GL4** and **5.84× / 3.83× nested**. Omitting the cubic term slightly improves average RMS further. Counts of field wins, normalized training objective, mean RMS and worst maximum error do not all rank these arms identically; difficult tails matter.

Post-hoc field-bootstrap parent/ours ratios are 1.085 [1.039, 1.148] and 1.123 [1.074, 1.222]. These are conditional, unadjusted descriptive intervals for a frozen fit. The smaller apparent advantage over plain GL2 has intervals crossing one. Reference uncertainty is at most 1.82% / 0.81% of the candidate RMS, so it does not explain the large GL4 deficit. Several easy endpoints are near FP32 rounding and should not carry mechanistic claims.

**The mean is a major bottleneck:** mean-mode error contributes **71.8% / 74.6%** of pooled candidate squared error. GL4 reduces mean absolute error about 4.9× / 3.7×. Logistic reaction does not conserve the mean:

\[
\frac{d\bar u}{dt}=r\,[\bar u-\bar u^2-\operatorname{Var}(u)].
\]

Thus an apparently spatial correction can improve accuracy chiefly through the variance-driven mean response. A larger spatial network is not the first conclusion to draw.

## R2: the cubic response has the wrong finite-step direction surprisingly often

The interaction panel has **216 physical cases and 1,512 method endpoints**, with independent diffusion and reaction stiffness interventions, overlapping triads, aligned harmonics and grid-relative frequencies on 32² and 64². Repeated phases, grids and physics are paired interventions, not 216 independent random fields.

All 216 same-equation temporal references pass. The full same-grid/fine-grid target set passes on **203/216**; 13 unresolved spatial comparisons remain unavailable. There are eight distinct deterministic continuous fields, not 216 independent field samples. For a prediction \(p\), finite-difference reference \(u_D\), same-grid Galerkin reference \(u_G\), and projected fine reference \(Pu_f\), the measured identity is

\[
p-Pu_f=(p-u_D)+(u_D-u_G)+(u_G-Pu_f).
\]

The squared norm includes signed cross terms. Adding component error norms would misattribute cancellation and cannot diagnose a continuum gain.

Relative to GL4, adding the nested cubic response gives **204 uncertainty-separated wins, eight losses and four unresolved comparisons**; the median GL4/corrected RMS ratio is about 6.36. At the joint 2e−5 RMS/maximum threshold, nested passes 212/216, GL4 168/216, plain GL2 132/216, and GL2 plus filtered cubic only 104/216. The losses cluster in aligned-phase cases where correction disrupts cancellation. Nested six-node quadrature has not independently converged, so “nested response” must not become shorthand for exact cubic truth.

More decisively, the current filtered leading cubic response has negative alignment with the **remaining refined-teacher defect** after GL4 in **149/216** cases, and after GL2 in **132/216**. All survive the diagnostic filters: remaining defect exceeds ten times reported reference uncertainty, and the cubic RMS exceeds 1e−12. The smallest remaining-GL4-defect/reference-uncertainty ratio is about 94.

For \(R=u_{ref}-S^{DF}\), a retained quadratic response \(Q\) and cubic response \(C\), the relevant direction is \(\langle C,R-Q\rangle\), not just \(\langle C,R\rangle\). For a fixed Q and positive gain, negative alignment means moving along C initially increases the squared residual. It does not prove that every coupled change is harmful. Teacher-informed unconstrained coefficients are diagnostics only; their sometimes enormous values near tiny responses are a reason to inspect identifiability, not to remove constraints indiscriminately.

These are **finite-step** findings. Large \(h\lambda\) or \(hr\) at a fixed horizon is not a small-step asymptotic experiment. Neither GL4 nor the nested response should be assumed converged at high stiffness. The appropriate next control is quadrature refinement or a stiffness-aware kernel, followed by selective higher-order response—not additional amplitude fitting of an unsuitable direction.

## R3: more credible opportunity exposes a conditioning problem

All **54 long trials reached 800 updates**, all 54 tiny trials reached 200 updates, and 54 equal-time trials reached their time cap. None numerically failed. Every predeclared optimizer setting received its opportunity; 18 selected checkpoints and 54 tiny-gate outcomes were independently reconstructed exactly.

Eight of twelve selected FNO family/track/seed instances now use trained weights. Their overall gains over the DF backbone remain small, roughly 1–2.8% when averaging the selected families' arithmetic RMS across seeds. The per-track median-seed summaries below use the native fairness evaluation, which differs slightly in assembly from the direct-fit unit.

| Method | Discrete RMS | Continuum-estimate RMS |
|---|---:|---:|
| Analytic DF | 3.8934e−5 | 3.7724e−5 |
| Theirs: adapted scaled FNO | 3.7383e−5 | 3.6203e−5 |
| Theirs: adapted mean-separated FNO | 3.7796e−5 | 3.7724e−5 |
| Ours: trained two-basis | 1.0073e−5 | 1.0583e−5 |
| Analytic GL4 | 3.1572e−6 | 3.6982e−6 |

The FNO families lose the field-geometric comparison against GL2 and GL4 on every one of the 32 fields on both tracks. Nevertheless, this is **not a defensible defeat of FNO as a method**: only four of 36 FNO tiny trials meet the stringent memorization gate, all mean-separated Galerkin trials, on one selected training example per track. Capacity, optimization, scaling and supervision still need disentangling.

The raw neural correction is multiplied by

\[
s=h^3rV\lambda_{active}(r+\lambda_{active}).
\]

If the squared-loss denominator is \(d^2\), the normalized sensitivity of the physical output to raw network output is \(s/d\). It spans **0.00131–20,587** on discrete train+validation queries and **0.00131–87,652** on Galerkin queries. Training alone already spans millions-fold. Squaring this quantity gives the raw-output loss-curvature multiplier; it is not the full network Hessian condition number.

A small-step polynomial scale can grow where the finite-step heat semigroup actually suppresses the response. Large recorded validation excursions are consistent with this diagnosis, but an intervention is required to establish causality. Adding a mean head did not remove the shared scale. Mean error still accounts for approximately 69–70% / 78% of FNO squared error.

The optimization comparisons also change clipping and objective multiplier together. A proper factorial control is still required. Equal updates are not equal compute; the Galerkin backbone makes training more expensive. All attempted trials cost **46m46s**, including optimization, validation and trial overhead; this is already included in campaign allocation cost, not an extra amount to add.

## R4–R5: a useful execution improvement, a negative node-count result

Four 64²/128² kernel stages include 256 endpoints, 20 randomized paired warm timing rounds per condition, and same-grid reference refinement. All 16 references pass. There are only four phase draws across two fixed two-mode shapes. For independent wavevectors these phase changes correspond to translations of the same continuous shape, especially redundant for Galerkin error norms. High-frequency physical wavelengths change with resolution. This is not a broad field-population or cross-grid generalization experiment.

| Track/grid | Sequential GL4 / batched GL4 latency: observed condition medians | Accuracy-qualified conditions /8 |
|---|---:|---:|
| Discrete 64² | **2.029–2.256×** | 8 |
| Discrete 128² | **2.017–2.121×** | 6 |
| Galerkin 64² | **1.481–1.507×** | 8 |
| Galerkin 128² | **1.473–1.483×** | 8 |

Batched GL4 wins **640/640 paired timing rounds**. Maximum endpoint disagreement with the sequential rule is **1.788e−7**, below the declared 3e−6 parity limit. The two discrete-128 one-step accuracy failures occur for both implementations. These are resident-device complete-call measurements; caller-side transfers, process startup and production service tails were not measured.

GL3 uses two active off-midpoint responses because its midpoint defect vanishes. The full weights remain \(5/18,4/9,5/18\); the weight of the zero-response midpoint is not redistributed. Degree-five polynomial quadrature exactness does not imply a fifth-order full solver. Here GL3 reaches a 1.2× speed gain over batched GL4 in **0/32 conditions**, while accuracy passes on **24/32 versus 30/32**. Thus node count alone is a poor execution-cost model.

Profiling explains part of this. Discrete GL4 reduces FFT dispatch calls from 24 to eight. Galerkin GL4 reduces them from 276 to 212, but the shared DF backbone already takes **196**. Batched calls do not imply scalar-transform arithmetic costs, and allocator/copy dispatch counts are not physical allocation or PCIe transfer counts. The major Galerkin opportunity is now the projected reaction backbone, subject to parity controls.

An exploratory reference-informed one/two-step comparison selects DF in ten of 16 cells and batched GL4 in six; GL3 never wins. This is not a deployable policy. DF is unfused and stronger classical alternatives are absent from this micro-panel. No fastest-solver or learned speedup claim follows.

## RX: useful negative evidence about memory

Sixteen independent development-test fields, repeated noise realizations and two reaction rates support this separate 1D finite-Galerkin study. Two startup observations are followed by 32 autonomous steps. There is no later access to reference truth. All references pass and their uncertainty is negligible relative to prediction error.

Clean fitted memory has median candidate/coarse endpoint RMS ratios **0.128** at the base reaction rate and **0.236** at three times that rate. With noisy histories, the same fit becomes roughly **19–46× worse**. Noise-aware fitting suppresses much of that amplification but has worse median paired RMS than coarse RK4 in all six scenarios. One small-sample interval includes parity; uniform failure on every field is not established.

At the higher rate and largest noise, plain secant leaves the sampled physical interval on **8/16 parents**, and clean fitted memory does so on **3/16**. All states remain finite. Therefore the existing finiteness gate does not establish stable or physically acceptable deployment. Historical scores should remain unchanged, while future acceptance explicitly includes interval violations, mean drift and horizon growth.

The neutral-limit secant recurrence is \(x_{n+1}=2x_n-x_{n-1}\). Its repeated unit root amplifies independent startup noise by \(\sqrt{(n+1)^2+n^2}\), about **45.97 at 32 steps**. Noise-aware one-step fitting can instead learn a denoising response that is then wrongly reused as autonomous dynamics. More network capacity does not repair that observation-process mismatch.

Validation selects **zero memory gain**, reproducing coarse RK4 exactly while still paying unnecessary secant overhead. Retain clean memory as a narrow control; stop promotion of the current noisy closure. A causal observer/noise-floor gate is the next protected experiment, with direct coarse and deterministic filtering at equal information as controls.

## Complete cost, resource use and reporting defects

The campaign used **1h18m51s summed allocation time**, including **53m23s of GPU reservation**, **6.8567 allocated CPU-core hours** and **1.6224 consumed CPU-core hours**. Allocations and Slurm steps were not double counted. Numerical stage timers, optimizer timers and allocation time overlap and must not be added together.

Fairness used 63.1% of allocation time; interactions 25.4%; kernel experiments 4.6%. The largest sampled Slurm RSS was **2.79 GiB**. Recorded PyTorch peaks were **68.10 MiB allocated / 70 MiB reserved**. Whole-device use includes the desktop and other processes. Low host CPU utilization is not a measurement of GPU utilization. Serial fitting/interactions reserve eight CPUs while consuming about one; smaller reservations deserve a separate operational check, with preparation retaining parallel capacity.

Energy, electricity cost, actual monetary charges, production cold latency, p99, deployment-volume amortization and full transfer/fallback costs are **unmeasured**. This campaign supplies no reason to enlarge VRAM or host-memory limits. All jobs stayed within the existing 35-minute maximum and desktop envelope.

One presentation defect was found: Tower received all canonical tables and 72,622 learning observations, but its figure index reported zero figures although the science report declared 22. The project adapter only recognized a `figures/` subdirectory, whereas Resolve writes PNG/PDF pairs at the report root. A separate project-only patch repairs declared, hash-checked figure discovery. It changes no Tower application, scientific artifact or historical measurement. The detailed provenance review retains the original failure.

## Prioritized next portfolio

These are proposed next experiments, **not completed results or a silently revised primary hypothesis**. Use fresh development fields now; choose and freeze candidates before a new untouched confirmation cohort. Keep the original benchmark and its failures visible.

| Priority | Falsifiable mechanism and cheapest test | Controls and primary outcomes | Advance / stop decision |
|---|---|---|---|
| 1: resolve the mean contribution | Independently replace GL2's mean and centered correction with GL4's, on matched fields | Original GL2, full GL4, no-cubic fit; RMS, maximum, mean-balance error and full-call latency | Continue a cheap mean rule only if it recovers most of GL4's resolved gain without a maximum-error or cost penalty |
| 2: finite-step neural conditioning | Replace the divergent polynomial scale with a semigroup-informed response scale; separate mean/spatial scaling | Current scale, fixed development scale, training-target normalization; independent clipping × loss-scale controls; tiny panel before full training | Require reliable tiny fits across several regimes, stable optimization and held-out improvement over strong controls; otherwise stop capacity expansion |
| 3: interaction kernel | Compare GL4/GL8 and early-time-aware quadrature at independently varied diffusion/reaction stiffness | Same operator, no output compression, nested-refinement control; angles, signed cross terms, uncertainty and matched-error cost | Continue only if quadrature refinement resolves the observed defect and the cheaper representation retains that gain |
| 4: comparable system optimization | Profile and improve the projected reaction backbone; allow DF fusion and equivalent caching to competitors | Frozen numerical outputs, paired randomized complete-call timing, one/two-step endpoints, host and device peaks | Numerical parity first; only then claim a cost change, including launch/transfer costs for the selected deployment workload |
| Protected exploration: causal observer | Filter two noisy observations before adding memory; gate correction against causal uncertainty | Coarse RK4, deterministic observer, clean secant; autonomous rollout, noise shifts, physical interval and mean error | Require ≥20% error reduction over the best deterministic control, no resolved maximum/mean regression or new interval violations, and a plausible cost margin |

A useful mean-only prototype follows directly from the quadratic variation-of-constants response. Write \(u=c+v\), \(\bar v=0\), \(L\hat v_k=-\lambda_k\hat v_k\), and let \(q(t)=\partial\Phi_t(c)/\partial c\) be the derivative of the logistic background flow with respect to its initial mean. With consistent Parseval/Nyquist weights,

\[
\overline{Q_h}=-r q(h)\sum_k |\hat v_k|^2
\int_0^h q(s)\,[e^{-2\lambda_k s}-e^{-\lambda_k h}]\,ds.
\]

This is the mean of the **quadratic-amplitude correction**, not an exact nonlinear mean law. It reduces the interaction to spectral energies and scalar time kernels, potentially avoiding repeated transported spatial products for this component. In the illustrative frozen-background limit \(q(s)=1\), the integral is

\[
h[\phi_1(-2h\lambda_k)-e^{-h\lambda_k}],\qquad
\phi_1(z)=(e^z-1)/z.
\]

The bracket behaves as \((h\lambda_k)^2/6\) at small steps and decays like \(1/(2h\lambda_k)\) under strong damping. That is fundamentally different from extrapolating an unbounded small-step polynomial. The real implementation must retain the background factor, stable small-argument evaluation, correct spatial symbols and mode weights, exact nulls, and all feature/transform costs. This is a proposed numerical mechanism; no novelty or speed claim is made.

An appropriate next allocation is approximately **40% attribution and competitor repair, 45% grounded interaction/execution improvements, and 15% protected exploration**, with a bounded pilot before expansion. Do not spend that budget on an exhaustive Cartesian product. Use independent field shapes, means, amplitudes, mode support and physical parameters—not just phase translations—and retain easy workloads alongside stress cases. The independent-field unit, failure denominators and reference acceptance must accompany every summary.

The defensible contribution today is a verified shared-transform implementation improvement plus a precise diagnosis of the finite-step representation and conditioning limitations. A useful learned correction, an improved purely numerical rule or a different temporal formulation remain possible. The next decisive question is whether we can retain the strong analytic interaction response at lower complete cost, while making the neural comparison genuinely trainable.
