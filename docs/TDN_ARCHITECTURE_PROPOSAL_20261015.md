# TDN: a response model inside an accuracy-controlled numerical solver

**Research proposal, 15 October 2026. Not an implemented architecture or a superiority claim.**

The recommendation is to develop TDN as a solver that uses a learned finite-time
response to correct a computable numerical defect, differentiates the complete
corrected step, and independently checks the resulting trajectory. The learning
problem should first have an exactly solvable conditional regression stage.
Further depth should address a measured nonlinear error, rather than add layers
without identifying the limiting mechanism.

This extends the [13 October proposal](TANGENT_RESPONSE_PROPOSALS_20261013.md)
and preserves the [native review](../results/native-tangent-compression-20261013/README.md):
A PASS, B STOP, old failed workers, thresholds and cohorts remain unchanged.
No new numerical campaign, learned training, or native GPU experiment was run
for this proposal. Existing CPU numerical checks were rerun to verify the
restored development environment; their scope is recorded in the
[verification record](../results/tdn-architecture-proposal-20261015/verification.json).

## 1. What a numerical-analysis specialist would decide first

TDN already has exact discrete diffusion/logistic splitting, independent
reference implementations, refinement, derivative checks, conditional readout
fitting and complete-cost reporting. These are retained controls, not new
contributions. See the [implementation map](IMPLEMENTATION_MAP.md),
[Tangent Response analysis](TANGENT_RESPONSE_RESEARCH.md), and
[model implementation](../tdn/analysis/tangent_response/models.py).

The first decision is which job TDN must make cheaper:

| Job | Required output | Main cost unit |
| --- | --- | --- |
| Forward simulation | Accurate states throughout a declared physical time interval | Complete accepted trajectory |
| Repeated response queries | Actions on genuinely unseen sources along one fixed state/physics/path | Path and encoding setup plus all queries |
| Inference or control | Reliable objective and gradients, followed by a successful optimization | Complete optimization to a common stopping target |

The second job may amortize an encoding over many sources. A changing-state
rollout must rebuild state-dependent information. Success in one job cannot
establish speedup in the others. Start with a declared workload, tolerances,
output times and hardware; a generic claim of a faster PDE model is too broad.

Separate the finite equation from the continuum claim. Same-grid finite
differences (FD), same-grid dealiased Galerkin, and projection of a finer
solution are different targets. This proposal's first architecture and
certificate analysis concern periodic same-grid FD. Galerkin requires its
projected product and physical inner product; projected-fine dynamics need an
explicit closure. Neither is silently obtained by changing a heat multiplier.

The observed 1,435-fold improvement from G to the full response is evidence of
useful spatial feedback on the inspected development bank. It establishes
neither a cheap approximation nor successful neural training. The failed
rank-eight oracle retained only 53.78% of endpoint gain, and passed none of
2,304 genuinely withheld directions. Stable fitting, conditioning information,
representation and utility must be diagnosed separately.

Maintain an error budget for spatial discretization, reference uncertainty,
anchor/path approximation, response approximation, source quadrature, nonlinear
remainder, coefficient fitting, rollout accumulation and arithmetic. Attribute
these with paired vector experiments and signed cross terms; their squared
errors do not simply add. A correction smaller than the final state's FP32
spacing may disappear during assembly. Measure that floor and compare an
explicit FP64/compensated assembly control before reducing precision. Earlier
large FP64 compression errors cannot be attributed to this precision floor.

## 2. The baseline ladder

For the FD equation, let

\[
F(u)=Lu+r u\odot(1-u),\qquad L=\kappa\Delta_{\rm FD},\quad r,\kappa\ge0.
\]

Let H_h=exp(hL) be the discrete heat action and
R_h(u)=u exp(rh)/[1+u(exp(rh)-1)] the exact logistic subflow. Actual code needs
stable exponential/endpoint evaluation. The continuous Fourier Laplacian must
not replace the FD symbol in a same-grid comparison.

| Baseline | Specific mechanism and purpose |
| --- | --- |
| Exact limits | H_h for zero reaction; R_h for zero diffusion; homogeneous logistic evolution; exact equilibrium linearizations. These detect errors without a learned or numerical teacher. |
| Lie and both Strang orientations | R_(h/2) H_h R_(h/2), H_(h/2) R_h H_(h/2), and fixed 1/2/4/8-step controls. Include adaptive step doubling with all rejected work. |
| Strong coupled integration | Retain coupled RK4 and Lawson RK4; add independently verified ETDRK4 with stable phi functions and a stiff Radau IIA or BDF implementation where appropriate. These are proposed additional competitors, not measured winners. |
| Static heterogeneous response | Dense expm on tiny grids; scaling-and-Taylor exponential action such as expm_multiply, or Arnoldi/Lanczos Krylov action on large grids. Block methods are relevant for multiple right-hand sides with shared setup. |
| Nonautonomous response | Ordered potential–heat–potential products and Lawson RK4 on the identical anchor. A frozen Jacobian exponential is not the time-dependent response. |
| Structured response approximation | First/second Duhamel actions, and first/second cumulants under the same discrete bridge law. Duhamel operator products and entrywise kernel cumulants are different approximations. |
| Leading splitting error | Independently derived Baker–Campbell–Hausdorff (BCH)/commutator correction, with nonautonomous terms where needed. Test whether a known leading term removes the work left for learning. |
| Stable compression | Physical-norm POD/SVD with training-only equilibration and ridge/TSVD, plus a state-conditioned nonstandard-wavelet operator following Feliu-Faba et al. |
| Neural comparisons | Properly trained residual FNO, U-Net and a generic conditioned operator with matched information and budgets. Include the same heat bank without the proposed correction/training structure. |

The strongest speed competitor is the fastest qualifying classical method,
not the expensive reference that produced labels. For a known equation,
integrating the known generator is the primary control; a learned-generator
comparison is a separate attribution experiment.

## 3. Accuracy over time and useful optimization

For a homogeneous logistic path a(t), the exact response is

\[
U(t,s)=R'_{t-s}(a(s))H_{t-s},\qquad
R'_h(a)=\frac{e^{rh}}{[1+a(e^{rh}-1)]^2}.
\]

Near equilibria zero and one, Fourier modes have rates lambda_FD(k)+r and
lambda_FD(k)-r, respectively. Tiny static heterogeneous matrices test spatial
coefficient placement; independently integrated time-dependent matrices test
ordering. Manufactured periodic solutions with analytically computed forcing
can test convergence, but belong to verification rather than the original
unforced performance population.

Growth near the unstable zero state is physical. Enforcing contraction on
every mode would change the dynamics; stability controls must use the correct
state-dependent growth allowance.

Track whole-trajectory RMS and maximum error, mean and centered errors,
front position/time-to-threshold, invariant-region violations, and step
composition. Late convergence toward u=1 can conceal inaccurate transients.
The infinite-domain Fisher front speed 2 sqrt(kappa r) is not an exact oracle
for a finite periodic experiment.

Two additional exact FD identities are useful independent checks. With a
normalized cell-mass matrix M and spatial mean mu,

\[
\mu'=r[\mu(1-\mu)-\operatorname{Var}(u)],
\]
\[
\mathcal E(u)=-\tfrac12u^TMLu-\tfrac r2u^TMu+
\tfrac r3\sum_i M_{ii}u_i^3,
\qquad \frac{d\mathcal E}{dt}=-\|F(u)\|_M^2.
\]

Logistic reaction does not conserve mean. Audit these balances without
postprocessing states to force a pass. A model should also recover temporal
convergence before reaching spatial or arithmetic floors, and spatial
convergence for the same continuous initial field.

For parameter inference, define an actual observation objective before training
for gradients. Compare classical adjoints, variational sensitivities and the
complete model's derivatives. Use central-difference sweeps, the mass-adjoint
identity, and the Taylor remainder

\[
J(p+\epsilon v)-J(p)-\epsilon\nabla J(p)^Tv=O(\epsilon^2).
\]

Benchmark L-BFGS with line search and Gauss–Newton/trust-region methods to the
same objective and stationarity target. Report parameter recovery, rejected
steps, forward/adjoint evaluations and total time. Accurate states alone do
not establish accurate gradients.

A useful conditional optimization guarantee is possible: if a nonzero
approximate gradient satisfies ||g_tilde-g|| <= eta ||g_tilde|| with eta<1, then
-g_tilde is a descent direction for the true objective. This requires an
actual gradient-error bound; an ensemble variance or small state MSE does not
supply it. Appropriate smoothness and true-objective backtracking are also
needed for a decreasing accepted step.

## 4. Proposed architecture: response-preconditioned defect correction

### 4.1 The exact object

Choose a smooth, computable Strang anchor a(t;u0), with a(0)=u0, and define
rho=a'-F(a). All path and derivative evaluations are charged. With e=u-a,

\[
e'=A_a(t)e-r e\odot e-\rho,\qquad
A_a(t)=L+r\operatorname{diag}(1-2a(t)),\qquad e(0)=0.
\]

This is an exact finite-equation identity. It identifies what the first
linear response correction omits: the nonlinear error feedback -r e^2.
Computing the anchor from the initial state is deployable; receiving a future
reference trajectory as an input is not.

### 4.2 A fixed, affordable-to-measure action

The first proposed action has two ordered classical panels, three sparse
commutator features, and two signed learned heat branches. This is a fixed
initial budget, not an assertion that four heat applications plus stencil
work suffice at all lags or resolutions.

For tau=t-s, divide [s,t] into two equal intervals I1,I2. Define

\[
V_j=\int_{I_j}r(1-2a(q))\,dq,\qquad
Q_j=D_{\exp(V_j/2)}H_{\tau/2}D_{\exp(V_j/2)},\qquad
P_a(t,s)=Q_2Q_1.
\]

D_b denotes multiplication by the field b. For the first implementation,
evaluate each V_j with two-point Gauss–Legendre quadrature on that panel;
its path/integration error remains separately measured. Where a subcase has
an analytic integral, use and verify that exact control. In particular,
zero-diffusion response is a local exponential of the integrated potential;
quadrature does not become exact merely because diffusion vanishes.

There is a subtle short-time limitation in the heat bank alone. For bounded
fields, tau^3 D_b H_tau D_c has a diagonal leading term because H_tau=I+tau L+....
Its off-diagonal contribution begins at order tau^4. A generic order-tau^3
splitting error already has off-diagonal double commutators. A heat-only
remainder of this form cannot generally cancel that leading error.

Supply three explicit sparse features instead. With the potential matrix
V=D_{r(1-2a(s))}, its derivative dot V=D_{-2r a'(s)}, and [A,B]=AB-BA, define

\[
C_1=[L,[L,V]],\qquad C_2=[V,[L,V]],\qquad C_3=[L,\dot V].
\]

Apply these matrix-free with the declared FD stencil; do not form dense
commutators or replace them with FFT Laplacians without charging that change.
Their classical leading coefficients must be derived independently for this
specific ordered backbone and checked on tiny matrices. The learned version
uses bounded dimensionless affine readouts xi_j; it does not presume those
coefficients or a higher order. The complete action is

\[
K_{\theta,a}(t,s)z=P_a(t,s)z+
\tau^3\sum_{j=1}^{3}\xi_{j,\theta} C_jz+
\left(\frac{\tau}{t_{\rm ref}}\right)^3
\sum_{\ell=1}^{2}\gamma_{\ell,\theta}(u_0,\kappa,r,s,t)
D_{b_\ell}H_\tau D_{c_\ell}z.
\]

All three C_j have units of inverse time cubed, so their tau^3 factors are
dimensionally consistent. They vanish when diffusion or reaction vanishes,
or when the anchor is spatially uniform. Undamped commutator actions can grow
with grid refinement; finite-lag attenuation, high-frequency stability and
actual usefulness must be tested. The local expansion does not establish
stiff stability. With shared intermediates the three features can use seven
FD Laplacian stencil actions per source, plus pointwise products; verify the
implemented count rather than treating them as free analytic corrections.

The coefficient fields and scalar coefficients depend on actual state,
physics, geometry, anchor summaries and times, never on source z. The action
is therefore exactly linear in z. Multiplication occurs on both sides of
diffusion. Ordered panels can represent temporal asymmetry; input-side
modulation removes the restrictive row shape of a single output-only
coefficient times a heat kernel. Neither property establishes universality.

Use a small periodic local/multiscale encoder, initially width 16 and depth 2,
with current state, its mean/fluctuation, known physics and geometry features.
Compute its state/physics/geometry backbone once per actual current state.
Inexpensive time-conditioned heads then produce b, c, xi and gamma for each (s,t).
These heads receive nondimensional reaction r tau and diffusion kappa tau/dx^2;
the backbone retains physical lengths and resolution. Charge all heads and
anchor queries. Any alternative that feeds source-time path states through
the full encoder must count those additional encodings explicitly.
Compare fixed finite-difference/heat-smoothed features before learning this
encoder. Spatial factors are nonzero at initialization; only the final affine
readout starts at zero, making the initial action exactly P_a. This avoids
the zero-gradient trap of initializing a squared learned factor at zero.
Bounded time features and factors are needed for the stated short-time scaling.

Each two-sided heat branch can have full physical matrix rank. The model
therefore avoids the old fixed small output subspace, but may still have an
insufficient kernel factorization. Measure its error versus lag, grid,
heterogeneity and actual cost. Do not infer rank from one successful vector.

The tau^3 factor protects the local response order of a second-order backbone
under fixed-grid smoothness and bounded-feature assumptions. It does not prove
a higher nonlinear solver order or a grid-uniform stiff order. Establish the
complete scheme's order including path and source quadrature.

Pure heat, pure reaction and homogeneous state evolution have exact anchors
and zero mathematical defect. Preserve these full-step nulls. Do not impose a
zero-diffusion gate on U-G: heterogeneous local reaction makes that response
difference nonzero. Any additional gate on K-P needs an independently verified
null for that particular remainder.

Two ordered panels plus two learned branches cost four heat actions, normally
eight FFTs per source action. Add the explicitly counted commutator stencil
actions and anchor derivatives. A proposed three-node source quadrature
therefore requires 12 heat actions/24 transforms plus that stencil work for
one endpoint correction before other path construction, encoding, assembly
and verification. With that stencil implementation, add 21 Laplacian actions
at three source nodes. Batching changes latency, not the work inventory.
This cost may already lose to a few extra
classical steps; measure that crossover before expensive training.

### 4.3 Controlled depth

Start with one correction. With a fixed declared quadrature Q,

\[
e^{(0)}=0,\qquad
e^{(k+1)}(t)=-Q_{[0,t]}
K_{\theta,a}(t,s)[\rho(s)+r e^{(k)}(s)^2],\qquad
\Psi_\theta(u_0,h)=a(h;u_0)+e^{(k)}(h).
\]

Only add a second stage when an oracle decomposition shows that omitted
nonlinear feedback limits first-stage accuracy. Values of e^(1) at inner
nodes need their own causal computation and accounting; a second stage is
not automatically twice the cost. No indefinite learned iteration is proposed.

For the exact U and exact integration, the Picard map has a standard sufficient
local convergence condition. In the path sup-infinity norm let B be the norm
of the linear forced correction, C=sup_t integral_0^t ||U(t,s)||_infinity ds,
and R an error-ball radius. Require both

\[
B+rCR^2\le R,\qquad 2rCR<1.
\]

The first condition makes the ball invariant; the second gives contraction.
Replacing U by a learned K changes the fixed point. These inequalities alone
do not establish correctness of the learned iteration. Check the final true
equation defect, or establish separate operator-error bounds.

An eventual alternative is an inexact Newton correction of a collocation
residual R(y). A learned update delta must satisfy the computable forcing test
||R(y)+DR(y)delta|| <= eta ||R(y)||, eta<1. Under the usual local regularity,
invertibility and initial-neighborhood assumptions this supports convergence;
otherwise refine or fall back. This is a later comparator, not part of the
first prototype and not novelty in Newton's method.

### 4.4 The complete derivative

Use JVP/VJP of the complete map Psi_theta for inference and optimization,
including anchor, encoder, coefficient and quadrature dependence. The
internal fixed-anchor K is not asserted to equal D_u Psi_theta. An arbitrary
state-conditioned operator need not be the Jacobian of any nonlinear map.

Adaptive accept/reject choices and fallback switches make the solver only
piecewise differentiable. A frozen accepted schedule gives a clearly defined
conditional derivative; it is not the derivative of changing the schedule.
Validate the chosen derivative contract with objective-level Taylor tests and
report schedule switches. Never silently detach state-dependent features.

## 5. What can be guaranteed about minimizing training loss

First freeze the nonlinear features, factors, path and one-correction layout.
The complete step is then affine in its final readout w:

\[
\Psi_w(x)=\Psi_0(x)+\sum_j w_j\phi_j(x),\qquad
D\Psi_w(x)=D\Psi_0(x)+\sum_j w_jD\phi_j(x).
\]

Stack physical-norm state errors, independent-source action errors and
derivative errors in a weighted design A and target b. Include initial-state
derivatives of every frozen feature: frozen parameters do not mean frozen
input dependence. Use fixed train-only scaling and parent-balanced weights:

\[
\min_w \|Aw-b\|_2^2+\lambda\|w\|_2^2,\qquad\lambda>0.
\]

The Hessian is 2(A^T A+lambda I), which is positive definite. This conditional
problem has a unique global minimizer; solve the augmented least-squares system
with FP64 QR/SVD rather than forming normal equations. Record singular values,
the ridge/cutoff, residuals and optimality tolerance. With zero correction
feasible, the attained aggregate regularized training objective is no worse
than the base value, up to the verified numerical solve error. Individual
metrics, validation and unseen data have no such automatic guarantee.

Here zero readout gives Psi_0=a-Q P_a rho, the classical response-corrected
step, not generally the raw Strang anchor a. The latter remains a separate
baseline. The signed commutator coefficients are included in the affine fit.

For this unhalved objective, strong convexity also gives an optimality-gap
bound: L(w)-L(w_star) <= ||gradient L(w)||^2/(4 lambda). Evaluate the residual
and gradient accurately enough to support the reported numerical gap. An
additional SVD truncation changes the feasible subspace; describe its optimum
as the optimum of that declared truncated problem.

Use an absolute floor tied to independently estimated label uncertainty for
near-null action normalization. Evaluate correction-relative error on (U-G)z
where appropriate, not only relative error on the much larger full Uz.
Keep source-action, endpoint, derivative and rollout objectives visible
individually. The historical parent-mean RMS gate is not interchangeable with
the squared training objective. Preserve vector cross terms in error attribution.

Only after this diagnostic, consider variable projection: optimize encoder
parameters while re-solving the conditional readout. AdamW followed by
deterministic L-BFGS is a candidate optimization schedule, with validation
selection, inner readout optimality gaps, and outer stationarity/multistart
diagnostics. The outer problem remains nonconvex and has no certified global gap.
Joint factor learning, multi-step student rollout and a second nonlinear
correction also destroy the simple convex guarantee. No global-optimum or
generalization theorem is claimed for them.

## 6. A result checker independent of network confidence

Construct a continuous piecewise cubic Hermite reconstruction y(t) from the
accepted candidate states and declared endpoint slopes. Its endpoints must
equal the actual returned states. For FD logistic dynamics,

\[
d(t)=y'(t)-Ly(t)-r y(t)(1-y(t))
\]

is polynomial of degree at most six on each panel. Bernstein-basis coefficient
bounds can enclose its value over the entire interval. A polynomial in this
basis lies in the convex hull of its control coefficients. Use outward-rounded
interval arithmetic for the coefficients, norms and subsequent bound if a
rigorous certificate is claimed. Ordinary FP64 estimates and sampled maxima
remain indicators.

This cubic audit reconstruction is distinct from the smooth Strang anchor
used to define rho. Even an exact heat/logistic solution sampled at endpoints
generally has a nonzero cubic-interpolant defect. Its interpolation error can
make the audit conservative; it does not invalidate the anchor's exact limits.

For true initial states in [0,1], the FD maximum principle supplies u>=0.
Suppose y>=m_y is independently enclosed and ML=L^T M is negative
semidefinite. The exact secant equation for e=y-u is

\[
e'=[L+r\operatorname{diag}(1-y-u)]e+d.
\]

Hence alpha=r(1-m_y) bounds its logarithmic norm. If D_n bounds the entire
panel defect in the physical norm, a valid error recurrence is

\[
E_{n+1}\le e^{\alpha h_n}E_n+
h_n\varphi_1(\alpha h_n)D_n,\qquad
\varphi_1(z)=\frac{e^z-1}{z},\quad\varphi_1(0)=1.
\]

Apply the recurrence at each elapsed time within a panel to bound the entire
trajectory. When maximum-error certification is required, use the corresponding
infinity-norm recurrence and residual bound as well; an M-norm bound alone
does not certify the maximum-error tolerance. The Markov FD generator has
zero row sums, supplying the required infinity logarithmic-norm bound.

Use expm1 near zero. An independently proved lower bound m_u on the true
solution can sharpen alpha to r(1-m_y-m_u). Substituting 2m_y would be invalid
without a true-solution bound. Include initial quantization/error and any
reconstruction jumps; a continuous matching reconstruction avoids the latter.
Account for rounding in evaluation of the actual discrete equation and bound.

This could certify the actual finite-grid candidate even if its internal
learned response is inaccurate. It does not certify continuum spatial error,
unseen tasks, or gradients. Full-interval residual enclosures can be costly and
the bound can be pessimistic. First implement a tiny-case independent auditor,
then measure whether it can be used online. A sampled/calibrated deployment
controller must retain an empirical label until it meets these requirements.

## 7. External evidence and acceptance gates

Evidence should be reproducible without trusting either author's judgment:

1. Analytic limits, manufactured solutions and independent numerical backends
   check implementation and temporal/spatial targets.
2. A separately implemented residual auditor checks bounded tiny FD cases.
3. Freeze source, dependencies, checkpoints, hyperparameters, targets and
   success criteria before generating a fresh confirmation bank.
4. Withhold complete parents, state/physics regimes and source directions;
   never count paired descendants as independent populations. Report failures,
   unresolved references and coverage denominators.
5. Run a complete inverse/control task against the classical objective and
   validate its gradients independently.
6. Publish portable raw metrics, source/data/checkpoint hashes and native
   commands so a separate person or machine can reproduce the result.

Eight successful independent parents do not establish high reliability. For
illustration, zero failures in eight independent identically distributed
Bernoulli trials still gives an exact one-sided 95% failure-probability upper
bound of 1-0.05^(1/8), about 31.2%. A new reliability claim needs a prespecified
population and adequate independent coverage; this calculation is not a request
to expand the current stopped campaign.

The proposed order of gates is:

| Gate | Advance only when |
| --- | --- |
| Specification and reference | Target equation, deployable information, all error terms and independent reference uncertainty are resolved. |
| Capacity at fixed work | The same four-heat-action plus commutator representation fits unseen directions sufficiently well; distinguish per-state label-informed oracle evidence from deployable prediction. |
| Conditional trainability | Stable readout improves held-out parents; optimization and representation floors are separated. |
| Structural accuracy | Full-step nulls, order, state admissibility, source linearity and actual model derivatives meet the frozen tests. |
| Native utility | Accepted trajectory/task accuracy beats the best qualifying classical frontier after all costs. |
| Fresh confirmation | Frozen model/control choices reproduce benefit on new independent parents with prespecified statistical uncertainty. |

Use the old gate criteria unchanged wherever the estimand is identical. A new
task may have a new protocol, frozen in advance, but cannot retroactively turn
old B into PASS. Report reference uncertainty below the margin used to decide
success; smaller error estimates do not constitute certificates by themselves.

## 8. Speed, hardware and deployment economics

Measure the complete required workload with the same output times, tolerance,
grid, device, precision policy and batch layout. Report cold start, warm latency,
throughput, peak memory, FFT/matvec counts and native CPU/memory/GPU telemetry.
Synchronize GPU timings and interleave paired randomized measurements. Include
encoding, path/residual construction, quadrature, derivative work, validation,
rejected candidates, fallback, compilation and precision bridges.

Cache reuse requires identical state/path, physics, geometry, model, dtype and
device. Count costs once only where the real workload permits reuse. Training
data, fitting, tuning and failures belong to the declared offline budget.

For a fixed repeated workload, the break-even query count is

\[
N_* = C_{\rm offline}/(C_{\rm classical}-C_{\rm TDN}),
\]

only if C_classical>C_TDN. State-changing and repeated-same-state workloads
have different costs. All-classical fallback does not show neural amortization.
For a superiority claim, predeclare a practical margin and require a paired
uncertainty interval to support it; a favorable mean alone is insufficient.

Raven should publish separate research and resource channels: accepted error,
whole-time bound/indicator and its status, action errors, gradient checks,
optimization objective, reference uncertainty, all-attempt latency, fallback
fraction and source-bound provenance alongside CPU/RAM/GPU traces. A Raven
successful job or GPU utilization graph is not scientific acceptance.

## 9. Novelty after primary-paper review

The following primary sources were retrieved and their relevant mathematical
sections read on 15 October. The [review record](../results/tdn-architecture-proposal-20261015/primary-source-review.json)
states exact scope; it is not a systematic novelty proof or a replication.

| Source | Established overlap and required comparator |
| --- | --- |
| [Kunikeev, Freeman and Doll, 2010](https://arxiv.org/abs/1006.1641), section II.B, equations 35–40 | Mean plus higher cumulants is classical. Compare first/second cumulants under TDN's FD bridge, not a different continuum path law. Their path-variable convergence rate does not prove convergence of TDN's learned cumulant expansion. |
| [Feliu-Faba, Fan and Ying, 2019/2020](https://arxiv.org/abs/1906.06782), section 3 | Nonlinear parameter conditioning with a forcing-linear compressed wavelet operator is prior art. Implement an information-matched conditioned wavelet comparator. |
| [Cheon, KAFlow, 9 October 2026](https://arxiv.org/abs/2610.13028), sections 2.5–2.7 and supplement S1 | Spatial coefficient/Fourier multiplier generators, gauge fixing, Lawson integration and two-sided factors in Strang evolution overlap. Section 2.6 explicitly warns that low generator separation rank need not yield low propagator rank. |

KAFlow's exact stiff linear exponential does not make its full nonlinear
integrator exact. Its statement makes the new implementation/models available
on request; without official assets, label a local adaptation KAFlow-inspired,
not a reproduction. No author was contacted for this proposal.

The earlier independently identified mean/cumulant branch remains an optional
research arm, not the meaning of K-P above. It requires independent labels for
the bridge mean m=E[X] and C=log E[exp X]-E[X]. Action supervision alone does not
separate them. A frozen approximate mean m_hat changes the required residual
to C+m-m_hat, which may be negative. Never clip that bias into an invented
nonnegative cumulant. Charge sampling/label uncertainty and all acquisition.

The candidate claim is consequently narrow: **a learned finite-time response
used inside a defect-correction solver can improve complete cost at matched
trajectory and derivative accuracy, while an independent residual test controls
acceptance.** The intended technical contribution is the measured mechanism and
its error/cost analysis. Heat factors, multiscale conditioning, deferred
correction, variable projection, Newton tests and residual bounds are existing
ideas. No exact duplicate was established by this targeted check, and novelty
or superiority has not yet been established either.

## 10. Term legend

| Term | Meaning in this proposal |
| --- | --- |
| TDN | Temporal-Defect Network; this repository develops order-anchored corrections to numerical evolution. |
| Anchor a | Deployable classical approximation generated from the actual input. |
| Defect rho | a'-F(a), the amount by which the anchor fails the differential equation. |
| U / G | Full fixed-anchor tangent propagator / its matched spatial-mean control. Neither is automatically the derivative of a corrected nonlinear solver. |
| P / K | Ordered classical response approximation / that approximation plus learned signed branches. |
| D_b / H_h | Spatial multiplication by b / exact finite-grid diffusion over h. |
| JVP / VJP | Jacobian-vector / vector-Jacobian product; evaluated through the complete deployed map. |
| M norm | Physical cell-mass norm; its target-specific definition must be preserved. |
| FD / FFT | Finite differences / fast Fourier transform. |
| ETDRK4 / BDF | Fourth-order exponential time-differencing Runge–Kutta / backward differentiation formula. |
| POD / SVD / TSVD | Proper orthogonal decomposition / singular-value decomposition / truncated SVD. |
| Ridge / variable projection | Quadratic regularization / elimination of linear coefficients inside a remaining nonlinear fit. |
| FNO / U-Net | Fourier neural operator / an encoder-decoder network with spatial multiscale features. |
| Picard / inexact Newton | Fixed-point iteration / a Newton update satisfying a controlled linearized residual. |
| Commutator / BCH | AB-BA measures operator-order dependence; the Baker–Campbell–Hausdorff expansion identifies leading splitting terms. |
| Phi_1 | Stable exponential-integrator function (exp(z)-1)/z with value 1 at zero. |
| Bernstein enclosure | Whole-interval polynomial bound from Bernstein control coefficients, with outward rounding needed for numerical rigor. |
| Certificate / indicator | A bound with verified assumptions and arithmetic / an empirical estimate without that guarantee. |
| Parent / confirmation | Independent field draw / a fresh cohort inspected only after decisions are frozen. |
| Work–precision frontier | Best achieved complete cost at each accuracy requirement. |

