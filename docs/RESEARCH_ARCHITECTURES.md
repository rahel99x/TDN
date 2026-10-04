# Implemented TDN research architectures

These are bounded development hypotheses for periodic scalar logistic
reaction–diffusion. They are not certified novel methods or demonstrated
efficiency improvements. Use the separate [research workflow](RESEARCH.md);
the established pilot schema and its scientific gates remain unchanged.

The common physical step is Strang composition
`R(h/2) D(h) R(h/2)`, using the exact logistic reaction and exact discrete
periodic FFT diffusion. Logistic reaction does not conserve mass. Every
spatially constant state, zero reaction and zero diffusion has zero splitting
defect, not merely the two equilibrium states.

## Confluent temporal decoder with equilibrium decay

For fixed reference time `t_ref`, let `tau=h/t_ref` and define

```text
J_k(rho,tau) = integral_0^tau exp[-rho*(tau-s)] (tau-s)^k s^2 ds / k!
K_k = exp(-gamma*tau) J_k,  k=0,1,2,3
u_next = S_h(u) + U_ref sum_k a_k(u) K_k
gamma = reaction_rate * t_ref
```

One positive learned rate is shared by the deployed model. The initial
experiment shares that rate across its training regimes; the earlier
teacher-informed probes fitted rates separately per parent, an easier task.
The four amplitudes come from a step-size-independent one-layer local encoder.
Temporal bases are shared across cells for each requested horizon.

Repeated-rate modes directly represent derivatives of exponential responses,
avoiding large subtractive amplitudes as ordinary rates coalesce. The extra
decay envelope makes the correction vanish at long times in the declared
positive-rate logistic equilibrium regime. This is not an assumption for
arbitrary PDEs and does not prove nonlinear rollout stability.

The fixed-rate controls use the same encoder, exact-limit factor and four
amplitudes, with rates `[0.1,1,10,100]`. `fixed_decay` uses `gamma=2*r*t_ref`,
`fixed_undamped` uses zero decay, and `fixed_decay_r` uses `r*t_ref` for a
comparison with exactly the same envelope as the confluent model.
A simple damped dictionary was competitive in the exploratory oracle probes;
it must remain in the comparison.

Tests independently check the integral, small-time cubic coefficient, FP32
evaluation, state/time/rate derivatives and repeated-rate cancellation.

## Correction transported through the existing split

```text
u_next = R_(h/2)(D_h(R_(h/2)(u) + q_in)) + q_late
```

The early correction gains spatial reach through the already required FFT.
The late bypass avoids learning an inverse of strongly contracting diffusion
or reaction. Both corrections depend on the initial state and an analytic
time decoder. The implementation uses one forward/inverse FFT pair.

With vector fields `A=kappa*Delta_d`, `B=r*u*(1-u)` and convention
`[F,G]=F'G-G'F`, define `C=[A,B]`, `ZA=[A,C]`, `ZB=[B,C]`. The discrete
leading defect is `e3=-ZA/12-ZB/24`. In particular,

```text
C_i = -kappa*r sum_j w_ij (u_j-u_i)^2
C'[v]_i = -2*kappa*r sum_j w_ij (u_j-u_i)(v_j-v_i)
```

`ZA` requires radius-two information. A continuum product-rule substitution
would change the numerical method. Two shared positive damping rates and five
encoder outputs construct

```text
q_in   = h^3 [alpha*e3 + tau*(a1*ZA+a2*ZB)] / (1+lambda_in*tau)^4
q_late = h^3 [(1-alpha)*e3 + tau*(b1*ZA+b2*ZB)] / (1+lambda_late*tau)^4
```

The zero-head model retains its exact cubic anchor; it is not plain Strang.
The parameter-free `e3_anchor` control adds `h^3*e3` after Strang. Tests verify
the coefficient using independent nested vector-field derivatives and coupled
FP64 reference solutions. Cubic local-error cancellation generally raises
global order from two to three, not four.

There is no positivity guarantee. Every transported intermediate is exposed
to the experiment's admissibility checks. Failed stages and large-horizon
fits remain in reports; there is no clipping or inverse-flow fitting.

## Bounded reaction coordinates

Let `w=D_h(R_(h/2)(u))`, `z=1-exp(-r*h)`, and let two bounded signed amplitudes
give `eta=a*z^3+b*z^4`. The model computes

```text
u_next = w / (w+(1-w)*exp[-r*h/2-eta])
```

The signed-clock implementation evaluates both signs stably, preserves exact
endpoints, and supplies analytic backward/JVP sensitivities. Zero amplitudes
reproduce the physical split. The output stays in `[0,1]` if the input to the
last reaction lies there. FFT roundoff is not silently clipped.

`reaction_additive` uses the same amplitudes and time basis as an additive
state correction. Optional `reaction_polynomial` retains the coordinate map
but uses unsaturated cubic/quartic time functions. These isolate the roles of
coordinates and time dependence.

A real limitation is retained as a regression test: at an initially zero cell
receiving diffusion, a cubic clock perturbation can produce only a quartic
state correction, while the required defect has a cubic term. Tests do not
pretend this limitation is resolved.

Optional `reaction_hybrid` is a separate experimental boundary extension.
It blends the clock result with a signed capacity update
`P(v,d)=v+m*d/(m+abs(d))`, where `m=1-v` for positive increments and `m=v`
otherwise. A fixed smooth initial-state gate selects the interior branch.
This is a Patankar-type construction, not a new positivity theorem. It can
supply a cubic correction when capacity scales as h, but not uniformly when
capacity scales as h cubed. Its earlier standalone temporal fit was weak;
it is excluded from the default experiment and can be enabled explicitly.

## Shared constraints and measurement

The small encoders receive live state/geometry/physics features and never the
queried horizon. Their normalization comes only from training parents. A
bounded radius-two variation factor enforces the exact commuting limits for
the confluent and reaction families; transport enforces them through its
analytic commutator fields. The original temporal baseline remains unmodified.

All learned physical maps must be differentiated as complete compositions.
The old additive decoder's frozen temporal jets cannot describe transported
or reaction-coordinate corrections.

Dense neural operation counts omit feature gathers, stencil passes,
exponentials, divisions, memory traffic and transforms. Only complete-solve
timing at matched error can substantiate efficiency. The workflow keeps
scientific failures distinct from successful execution and never promotes
this development screen into confirmatory evidence.

## Established components and prior work

- [Hypersolvers](https://github.com/DiffEqML/diffeqml-research/tree/master/hypersolver)
  establishes learned local truncation corrections.
- [Closed-form continuous-time networks](https://github.com/raminmh/CfC)
  establishes closed-form temporal neural units.
- [SciML exponential utilities](https://github.com/SciML/ExponentialUtilities.jl/blob/master/src/phi.jl)
  implements established exponential moments and augmented/Jordan machinery.
- [PositiveIntegrators.jl](https://github.com/NumericalMathematics/PositiveIntegrators.jl/blob/main/docs/src/scalar_pds.md)
  documents the Patankar rational update; its convergence documentation
  discusses order reduction near zero.

The specific integrations are project research proposals. Neither inspection
of these sources nor passing implementation tests establishes publication
priority or superiority over classical numerical methods.
