# Tangent response: identities, conditional bounds and interpretation

These derivations apply to the declared finite equation, not an unspecified
continuum PDE or projected-fine closure. The implementation separately checks
FD and same-grid dealiased Galerkin operators on odd/even grids. See the
[research handout](TANGENT_RESPONSE_RESEARCH_HANDOUT.md) and
[related-work audit](TANGENT_RESPONSE_LITERATURE.md). Numerical refinement
differences and finite probe results are estimates, not certificates.

## Exact error equation and matched control

Let `B` be the symmetric bilinear product of the target contract and let

\[
f(u)=Lu+ru-rB(u,u),\qquad \rho=a'-f(a),\qquad e=u-a,
\]

where the smooth path satisfies `a(0)=u0`. Bilinearity gives

\[
f(a+e)-f(a)=Le+re-2rB(a,e)-rB(e,e),
\quad e'=A_a e-rB(e,e)-\rho,
\quad A_a z=Lz+rz-2rB(a,z).
\]

The mean is part of the tangent field. Projecting `A_a z` to zero mean would
remove the contribution `-2r mean(B(a,z))` and generally change the equation.
Projected Galerkin products are not generally associative; no step of this
derivation replaces `B(a,B(b,c))` by `B(B(a,b),c)`.

For the same-path evolution `U_a(t,s)` and zero initial error,

\[
e(t)=-\int_0^t U_a(t,s)\rho(s)\,ds
      -r\int_0^t U_a(t,s)B(e(s),e(s))\,ds.
\]

Write `mu_a=mean(a)` and `v_a=a-mu_a`. The matched mean response is

\[
G_a(t,s)=e^{(t-s)L}
\exp\!\left(r\int_s^t[1-2\mu_a(q)]dq\right),
\quad H_a(t)z=-2rB(v_a(t),z).
\]

Because constant multiplication is the target identity, `A_a=A_mean+H_a` and

\[
U_a(t,s)=G_a(t,s)+\int_s^tG_a(t,q)H_a(q)U_a(q,s)dq.
\]

This time-ordered relation explains what the mean control omits. It does not
say that a scalar positive gain or a diagonal Fourier multiplier represents
the omitted spatial mode mixing. At zero reaction the response is heat
evolution, not zero. At a spatially constant anchor `U_a=G_a`; zero diffusion
alone is not a null of their difference on varying fields.

## Finite physical norms and product constants

Use normalized nodal L2 for FD. If the grid has `M` nodes,

\[
\|B(x,y)\|_2\le\|x\|_\infty\|y\|_2
\le\sqrt M\,\|x\|_2\|y\|_2.
\]

For Galerkin, use normalized L2 of the real physical interpolant and the
verified Nyquist mass weighting. The target product is orthogonal projection
of the dealiased physical product. Therefore

\[
\|P_N(xy)\|_2\le\|x\|_\infty\|y\|_2
\le C_N\|x\|_2\|y\|_2.
\]

A conservative constant compatible with the stored Fourier/Nyquist
coordinates is

\[
C_N=\sqrt{\prod_i d_i},\qquad
d_i=\begin{cases}N_i,&N_i\text{ odd},\\N_i+1,&N_i\text{ even}.\end{cases}
\]

To see the even-grid factor, apply weighted Cauchy–Schwarz to the Fourier
coefficient sum. Ordinary coefficients have physical weight one; the
collapsed real Nyquist coefficient has weight one half, so the sum of inverse
weights on that axis is `N_i+1`. Tensor-product weights multiply over axes.
This is a conservative embedding constant, not an optimal one. Both FD and
Galerkin constants depend on resolution. A fixed-grid inequality must not be
advertised as grid-uniform stability or convergence.

`evaluation.finite_product_constant` records this dependence. CPU checks test
the product inequality for both targets on odd/even grids; sampled tests do
not independently prove the inequality for every field.

## Conditional growth and first-correction bound

For FD, the periodic diffusion generator is negative semidefinite in nodal
L2. If the entire smooth path has a verified nodal lower bound `a_min(t)`,

\[
\langle z,A_a(t)z\rangle
\le r[1-2a_{\min}(t)]\|z\|_2^2,
\quad \|U_a(t,s)\|\le M(t,s)
=\exp\!\left(\int_s^t\omega(q)dq\right),
\]

with `omega=r(1-2 a_min)`. On a genuinely bounded FD path in `[0,1]`, `omega<=r`.
It may be positive: universal contraction is not assumed. Bounded inputs alone
do not establish bounded outputs for a learned step.

For Galerkin the diffusion part is also dissipative in the declared physical
inner product. An independently valid bound on the *physical interpolant*
can be used; a conservative choice is
`omega(t)=r+2r C_N ||a(t)||_2`. Nodal bounds or sampled extrema do not supply
a continuous maximum principle. Do not infer Galerkin trajectory positivity
from bounded initial nodes.

Define the learned first correction
`e_theta=-integral U_tilde rho`. Its exact same-path error is

\[
e-e_\theta=
-\int_0^t(U_a-\widetilde U_a)\rho\,ds
-r\int_0^tU_aB(e,e)\,ds.
\]

On a finite interval with all required continuous-time bounds,

\[
\|e(t)-e_\theta(t)\|_2\le
\int_0^t\|(U_a-\widetilde U_a)\rho(s)\|_2ds
+rC_N\int_0^tM(t,s)\|e(s)\|_2^2ds.
\]

A uniform operator-error bound could bound the first integrand by
`epsilon(t,s)||rho(s)||`, but the experiment measures relevant actions instead
of assuming a uniformly low-rank operator. Finite action probes are not such
a uniform bound. The nonlinear term needs an admissible error/state bound or
independently measured trajectory; endpoint accuracy alone does not control it.
`evaluation.conditional_response_bound` records an explicit growth/product
constant and labels its output conditional, never certified.

## Path, residual, quadrature and arithmetic terms

Let the numerical decoder use an approximate path `a_hat`, learned response
`U_tilde_a_hat`, residual samples `rho_hat`, quadrature `Q`, and arithmetic
perturbation `d_arith`. Write

\[
y=\widehat a(t)+Q[-\widetilde U_{\widehat a}\widehat\rho]+d_{\rm arith}.
\]

The following separate vectors sum exactly to `u-y` when their integrals are
evaluated as mathematical quantities:

\[
\begin{aligned}
d_R&=-\int(U_a-\widetilde U_a)\rho,\\
d_N&=-r\int U_a B(e,e),\\
d_P&=a(t)-\widehat a(t)-\int(\widetilde U_a-\widetilde U_{\widehat a})\rho,\\
d_\rho&=-\int\widetilde U_{\widehat a}(\rho-\widehat\rho),\\
d_Q&=\int[-\widetilde U_{\widehat a}\widehat\rho]-Q[-\widetilde U_{\widehat a}\widehat\rho],\\
d_A&=-d_{\rm arith}.
\end{aligned}
\]

Response and residual/path approximations are distinct. A numerical path error
affects both its endpoint and its response. For true tangent propagators,
variation of constants bounds path sensitivity using
`||A_a-A_a_hat|| <= 2r C_N ||a-a_hat||`; a learned response needs its own
conditional sensitivity analysis, not automatic inheritance of that bound.
Quadrature control must include behavior between source samples. Refinement
agreement, FP64 arithmetic and accepted teachers do not by themselves certify
all these terms.

Terms can cancel. Retain

\[
\left\|\sum_i d_i\right\|_2^2=
\sum_i\|d_i\|_2^2+2\sum_{i<j}\langle d_i,d_j\rangle.
\]

`evaluation.signed_error_decomposition` publishes signed cross terms and checks
their reconstructed sum. A conservative sum of component norms is a different
quantity; unsigned squared components alone are not an error decomposition.

The precision vectors are also distinct:
`physical_defect=teacher64-smooth_anchor64`,
`arithmetic_bridge=smooth_anchor64-deployed_anchor32.double()`, and
`deployment_defect=physical_defect+arithmetic_bridge`. Casting a rounded FP32
prediction to FP64 for loss does not restore discarded correction bits. If a
method computes the bridge at inference, both anchors are charged.

## Factor conditioning and identifiability

For a factor representation `Alpha C(t,s) Beta*`, reciprocal rescaling of
matched columns and permutation are representation freedoms. General rotations
are freedoms only when the temporal coefficient representation transforms
consistently: `Alpha->Alpha R`, `Beta*->R^-1 Beta*`,
`C->R^-1 C R`. A strictly diagonal temporal coefficient family is not invariant
under arbitrary rotations unless those rotations commute with `C`, for example
inside equal-coefficient blocks. Do not claim unrestricted rotational symmetry
for an architecture that cannot represent the transformed coefficients.

Small/large cancelling factors can worsen sensitivity while leaving exact
actions unchanged. Report Gram matrices, singular values, cutoff/ridge,
retained rank, raw/assembled gain, perturbation sensitivity and actual FP32
deployment. Train-only whitening/rescaling changes coordinates; it is not a
new physical accuracy mechanism. Conditional variable projection is affine
only with every path, response factor and feature that determines the design
held fixed. It provides no global optimality certificate for the encoder.

## Evaluation and costs

Parent fields are the independent units. Horizons, shifts, grids, probes and
model seeds attached to them remain paired. Endpoints retain all requested
teachers and failures, with equal parent and within-parent query weighting.
Mean/amplitude/phase/frequency/horizon/physics shifts and paired resolution
panels are descriptive development evidence. Fixed coefficients across grids
measure a fixed-physics refinement; grid-matched active stiffness changes the
equation and is reported separately.

Autonomous rollouts use 2, 4 and 8 macrosteps, re-encoding the actual student
state each time. Intermediate teachers evolve the original actual input to
every cumulative time. Missing teachers remain missing. Endpoint moment
balance uses a labeled trapezoidal diagnostic; a small value neither proves
endpoint accuracy nor enforces the differential identity. Nonlinear endpoint
composition is distinct from `U(t,s)U(s,q)=U(t,q)` along one fixed anchor path.
JVP/central finite differences are independently compared on declared probes.

Full-call timing encloses anchor, path/residual generation, encoding,
contraction, quadrature, transforms, nonlinear iterations and output assembly,
including both anchors for an inference bridge. Cache creation/decoding/parity,
resident calls, complete transfers and fresh processes have separate scopes.
CUDA timings synchronize and use repeated randomized method order. Numerical
controls use the same equation, input, final time and physical tolerances
`2e-5`, with declared `1e-4`/`1e-5` panels. Schedules are frozen using validation
only before diagnostic parents are evaluated; retrospective per-query fastest
choices are not deployable policies.

Utility work requires eligible Stage C evidence and source-bound native
readiness/checkpoint identities. CPU smoke exercises bounded software paths
and remains implementation-only. Negative/unresolved learning does not
automatically launch utility. Energy and money stay unavailable without their
own measurements and rates. Break-even is computed only for positive measured
savings and an explicit workload, with included and omitted offline costs.

## Additional report terminology

The complete [handout legend](TANGENT_RESPONSE_RESEARCH_HANDOUT.md#18-legend)
defines the equation symbols and research terms. These implementation names
also appear in the reports:

| Name | Meaning and scope |
| --- | --- |
| PDE / RHS | Partial differential equation / right-hand side. This program measures the declared finite equation, with its specified spatial discretization. |
| FNO | Fourier neural operator. Here it is the local matched-anchor endpoint residual control, not a reproduced published benchmark. |
| FFT | Fast Fourier transform; used for periodic diffusion, factors and dealiased products. Transform work belongs in complete cost. |
| `DF` / `df` | Diffusion-first symmetric split: half diffusion, reaction, half diffusion. |
| GL4 / `shared_gl4` | Four-node Gauss–Legendre quadratic-defect quadrature added to the anchor, with shared transforms. The node count does not prove fourth-order accuracy of the complete flow. |
| DOP853 | SciPy's adaptive explicit Runge–Kutta solver used by the independent NumPy reference route. Refinement agreement is an empirical uncertainty estimate. |
| P1 / P2 / `nonlinear1` / `nonlinear2` | One or two Picard correction iterations for quadratic error feedback; separate from the first linearized response and the full nonlinear error oracle. |
| Nyquist | Highest representable frequency on an even grid; its real Fourier coefficient has the declared physical mass weight rather than an ordinary nodal weight. |
| TF32 | TensorFloat-32 backend arithmetic policy; disabled here. FP32 outputs are still rounded even when internal attribution uses FP64. |
| RSS | Resident set size: measured process/descendant host memory, with scope and coverage recorded. It is distinct from GPU memory. |
| CUDA / Slurm / Raven | NVIDIA compute runtime / allocation scheduler / launch, research and artifact viewer. Actual CUDA verification requires a real allocated device. |
| JUnit / seal / SHA256 | Machine-readable test receipt / completed-artifact identity contract / cryptographic byte hash. A matching hash does not establish scientific benefit. |
| C0 / C1 / C2 | Prospective fit-bank screening / best-training-checkpoint reconstruction / validation-selected diagnostic gain. C requires both C1 and C2. |
| Calibration / trial / seed | Workload measurement / one declared optimizer opportunity / paired randomization. Calibration fits and repeated CPU rehearsals are not extra independent research trials. |
| Resident / full transfer / fresh process | Already-resident complete calls / calls including input-output transfer / independent invocation including startup and checkpoint loading. Their timings have different scopes. |
| ULP / final rounding | Representable-output spacing / deployed output minus its retained FP64 physical assembly. Neither can be erased by converting the rounded output back to FP64. |
| `IMPLEMENTATION_ONLY` | A CPU software rehearsal with native promotion disabled, regardless of its separately displayed toy scientific outcomes. |
