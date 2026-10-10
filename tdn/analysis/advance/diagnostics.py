"""Independent mathematical audits and residual-direction diagnostics.

Oracle projections use development teachers and are explicitly representation
diagnostics. They never select frozen deployment models or read confirmation.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import solve_ivp
import torch

from tdn.analysis.frontier.data import field_state, lawson_reference
from tdn.analysis.frontier.numerics import rhs, df_step, product, diffusion_symbol, apply_multiplier
from tdn.analysis.roadmap.numerics import quadrature, cubic_df_defect, quadratic_df_defect
from tdn.numerics import Equation, Geometry
from tdn.runtime.metadata import write_json
from .core import check, clean


def rms(x): return float(x.detach().square().mean().sqrt())


def probe_field(n, regime, amplitude=.04, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.arange(n, dtype=torch.float64) / n
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    k = max(2, n//2-2)
    modes = {"low":[(1,0),(0,1)], "high_pair":[(k-1,1),(k,1)],
        "broadband":[(1,0),(1,2),(2,-1),(k,1)], "near_nyquist":[(k,0),(0,k)]}[regime]
    v = sum(torch.cos(2*math.pi*(a*xx+b*yy)+float(rng.uniform(0,2*math.pi))) / math.sqrt(j+1)
            for j,(a,b) in enumerate(modes))
    v -= v.mean(); v /= v.square().mean().sqrt()
    return (.4 + amplitude*v)[None,None]


def residual_projection(residual, bases):
    """Unconstrained same-field oracle; no inference/deployment claim."""
    matrix = torch.stack([b.reshape(-1) for b in bases],1).double()
    target = residual.reshape(-1).double()
    norms = torch.linalg.vector_norm(matrix,dim=0)
    nonzero = norms > 1e-30
    normalized = torch.where(nonzero, matrix/norms.clamp_min(1e-30), torch.zeros_like(matrix))
    solution = torch.linalg.lstsq(normalized,target,driver="gelsd")
    coefficients = solution.solution/norms.clamp_min(1e-30)
    projected = matrix@coefficients
    singular = torch.linalg.svdvals(normalized)
    target_norm = torch.linalg.vector_norm(target)
    error = torch.linalg.vector_norm(target-projected)
    cosine = (normalized.T@target)/target_norm.clamp_min(1e-30)
    return dict(oracle_coefficients=coefficients.tolist(), basis_norms=norms.tolist(),
        singular_values=singular.tolist(), rank=int(solution.rank),
        condition_number=float(singular[0]/singular[-1]) if float(singular[-1])>1e-14 else None,
        relative_unrepresented_norm=float(error/target_norm) if float(target_norm)>1e-30 else None,
        residual_basis_cosines=cosine.tolist(),
        oracle_error_rms=float(error/math.sqrt(target.numel())),
        scope="unbounded teacher-informed development projection; not fitted or deployable inference")


def audit(ctx):
    from .models import FAMILIES, MODEL_SPECS, make_model
    from .numerics import cubic_commutator_defect as cubic_commutator
    from .metrics import energy_value, physical_inner_product
    results=[]
    for nodes in (2,4):
        x,w=quadrature(nodes)
        error=max(abs(sum(a*b**degree for a,b in zip(w,x))-1/(degree+1)) for degree in range(2*nodes))
        row=dict(nodes=nodes,maximum_polynomial_moment_error=error);results.append(row)
        ctx.record(f"quadrature/{nodes}",["A1"],metrics=row,checks=[
            check("gauss-moments",error,2e-14,category="correctness"),
            check("quadrature-math",error,2e-14,category="math")])
    eq=Equation(.004,3.);geom=Geometry((8,8),(1.,1.));u=probe_field(8,"low")
    for track in ctx.protocol["tracks"]:
        for family in FAMILIES:
            ctx.budget.check(); torch.manual_seed(85111)
            model=make_model(family,track,dict(width=4,modes=1,depth=2)).double().eval()
            with torch.no_grad():
                zero=model(u,0.,eq,geom);answer=model(u,.01,eq,geom)
                shifted=model(torch.roll(u,(1,2),(-2,-1)),.01,eq,geom)
            zero_error=float((zero-u).abs().max())
            translation_error=float((shifted-torch.roll(answer,(1,2),(-2,-1))).abs().max())
            row=dict(family=family,track=track,zero_time_max=zero_error,translation_max=translation_error,
                finite=bool(torch.isfinite(answer).all()),parameters=sum(p.numel() for p in model.parameters()))
            results.append(row)
            ctx.record(f"model/{track}/{family}",["A1","B2"],metrics=row,checks=[
                check("zero-time",zero_error,1e-13,category="correctness"),
                check("translation-equivariance",translation_error,2e-11,category="math"),
                check("finite",row["finite"],True,"eq",category="correctness")])
        # Independent time integrator, same spatial equation and exact requested endpoint.
        small=probe_field(4,"low",amplitude=.015);geo=Geometry((4,4),(1.,1.));h=.03
        def fun(t,y): return rhs(torch.from_numpy(y.copy()).reshape_as(small),eq,geo,track).numpy().ravel()
        ode=solve_ivp(fun,(0,h),small.numpy().ravel(),method="DOP853",rtol=2e-12,atol=2e-13)
        teacher=lawson_reference(small,h,eq,geo,64,track,check_budget=ctx.budget.check)
        error=float((teacher-torch.from_numpy(ode.y[:,-1]).reshape_as(small)).abs().max())
        row=dict(track=track,teacher_crosscheck_max=error,dop853_nfev=ode.nfev);results.append(row)
        ctx.record("teacher/"+track,["A1"],metrics=row,checks=[
            check("adaptive-completed",bool(ode.success),True,"eq",category="correctness"),
            check("independent-teacher",error,2e-9,category="math")])
        # The exact mean identity is instantaneous, not a mean-conservation law.
        derivative=rhs(u,eq,geom,track);mean=u.mean();variance=physical_inner_product(u-mean,u-mean,geom,track)
        expected=eq.reaction_rate*(mean*(1-mean)-variance)
        error=float(abs(derivative.mean()-expected))
        ctx.record("mean-law/"+track,["A1","B2"],metrics=dict(track=track,instantaneous_mean_identity_error=error),
            checks=[check("reaction-variance-mean-law",error,2e-12,category="math")])
        # Homogeneity and oddness of the derived cubic amplitude term.
        c=u.mean();double=c+2*(u-c)
        q=cubic_commutator(u,.01,eq,geom,track=track,transport="none")
        q2=cubic_commutator(double,.01,eq,geom,track=track,transport="none")
        parity=cubic_commutator(2*c-u,.01,eq,geom,track=track,transport="none")
        scale=rms(q2-8*q)/max(rms(q),1e-25);odd=rms(parity+q)/max(rms(q),1e-25)
        ctx.record("cubic-homogeneity/"+track,["A1","B1"],metrics=dict(track=track,amplitude_cubic_relative=scale,odd_relative=odd),
            checks=[check("cubic-amplitude",scale,1e-8,category="math"),check("odd-centered-parity",odd,1e-8,category="math")])
        # Random even-grid fields exercise Nyquist mass weights as well as
        # ordinary modes. The continuum norm is not generally the nodal norm.
        generator=torch.Generator().manual_seed(85513)
        state=(.4+.03*torch.randn(u.shape,dtype=u.dtype,generator=generator)).requires_grad_()
        flow=rhs(state,eq,geom,track)
        gradient,=torch.autograd.grad(energy_value(state,eq,geom,track),state)
        energy_error=float(abs((gradient*flow).sum()+physical_inner_product(flow,flow,geom,track)).detach())
        ctx.record("gradient-flow/"+track,["A1","B2"],metrics=dict(track=track,
            energy_dissipation_identity_error=energy_error,
            norm="nodal FD or Nyquist-aware Fourier interpolant inner product"),
            checks=[check("gradient-flow-energy-identity",energy_error,2e-11,category="math")])
    write_json(ctx.path/"audit_rows.json",dict(rows=clean(results)))
    return dict(model_cases=len(FAMILIES)*2,independent_teacher_tracks=2,
        meaning="finite mathematical/implementation tests; not a stability or scientific superiority proof")


def run(ctx):
    from .models import make_model
    from .numerics import cubic_commutator_defect as cubic_commutator
    from .metrics import endpoint_metrics, physical_inner_product
    p=ctx.protocol["diagnostics"];n=p["grid"];geom=Geometry((n,n),(1.,1.));eq=Equation(.004,3.)
    rows=[];projections=[];composition=[];orders=[];amplitudes=[]
    families=["df","quad2_full","quad4_full","analytic_quad_cubic","commutator_raw","commutator_cubic","two_basis","etdrk4"]
    for regime in p["regimes"]:
        u=probe_field(n,regime,.04,p["seed"])
        for track in ctx.protocol["tracks"]:
            models={f:make_model(f,track,dict(modes=2,width=4,depth=2)).double().eval() for f in families}
            for h in p["horizons"]:
                ctx.budget.check()
                coarse=lawson_reference(u,h,eq,geom,64,track,check_budget=ctx.budget.check)
                teacher=lawson_reference(u,h,eq,geom,128,track,check_budget=ctx.budget.check)
                uncertainty=rms(teacher-coarse);max_uncertainty=float((teacher-coarse).abs().max())
                ref=dict(accepted=max_uncertainty<=2e-7,uncertainty_rms=uncertainty,uncertainty_max_bound=max_uncertainty,
                    uncertainty_is_certificate=False, target="same-grid specified equation; not spatially refined continuum")
                base=df_step(u,h,eq,geom,track)
                quad=quadratic_df_defect(u,h,eq,geom,nodes=2,target=track)
                cubic=cubic_commutator(u,h,eq,geom,track=track,transport="symmetric_heat")
                projection=residual_projection(teacher-base,[quad,cubic])
                one=residual_projection(teacher-base,[quad])
                pr=dict(regime=regime,track=track,grid=n,horizon=h,**projection,
                    one_basis_oracle_error_rms=one["oracle_error_rms"],reference_accepted=ref["accepted"])
                projections.append(pr)
                for family,model in models.items():
                    ctx.budget.check();tick=time.perf_counter()
                    with torch.no_grad(): answer=model(u,h,eq,geom)
                    inference_seconds=time.perf_counter()-tick
                    metrics=endpoint_metrics(answer,teacher,u,eq,geom,track,reference=ref)
                    row={"family":family,"regime":regime,"horizon":h,
                        "unreplicated_inference_seconds":inference_seconds,
                        "timing_scope":"CPU development diagnostic; excludes metrics; not paired deployment timing",
                        **metrics,"grid":n}
                    rows.append(row)
                    ctx.record(f"error/{regime}/{track}/{h}/{family}",["B1","B2"],metrics=row,checks=[
                        check("reference-accepted",ref["accepted"],True,"eq",category="math"),
                        check("joint-rms",row.get("upper_rms"),ctx.protocol["primary_target"]),
                        check("joint-maximum",row.get("upper_max"),ctx.protocol["primary_target"])])
                if h==p["horizons"][-1]:
                    direction=torch.sin(torch.arange(n*n,dtype=u.dtype).reshape_as(u));direction/=direction.square().mean().sqrt()
                    epsilon=p["tangent_epsilon"]
                    for family,model in models.items():
                        ctx.budget.check()
                        with torch.no_grad():
                            whole=model(u,h,eq,geom);half=model(model(u,h/2,eq,geom),h/2,eq,geom)
                            difference=(model(u+epsilon*direction,h,eq,geom)-model(u-epsilon*direction,h,eq,geom))/(2*epsilon)
                        physical_amplification=float((physical_inner_product(difference,difference,geom,track)/
                            physical_inner_product(direction,direction,geom,track)).sqrt())
                        composition.append(dict(family=family,track=track,regime=regime,grid=n,horizon=h,
                            composition_defect_rms=rms(half-whole),directional_amplification=rms(difference),
                            physical_directional_amplification=physical_amplification,
                            conditional_exact_flow_physical_L2_bound=math.exp(eq.reaction_rate*h),
                            bound_condition="Exact flow only, assuming nonnegative nodal FD states or continuous Galerkin interpolants along the trajectory; positivity not certified here",
                            finite_difference_epsilon=epsilon,
                            scope="one sampled direction, not operator norm; composition defect is not an error certificate"))
            # Leading time behavior against full independently derived nested cubic rule.
            for h in (.0005,.001,.002):
                ctx.budget.check()
                exact=cubic_df_defect(u,h,eq,geom,nodes=6,target=track)
                raw=cubic_commutator(u,h,eq,geom,track=track,transport="none")
                orders.append(dict(track=track,regime=regime,horizon=h,grid=n,
                    cubic_reference_rms=rms(exact),raw_remainder_rms=rms(raw-exact),
                    remainder_over_h4=rms(raw-exact)/h**4,scope="fixed-grid small-step diagnostic, no stiff-uniform order claim"))
            for amplitude in p["amplitudes"]:
                ctx.budget.check();state=probe_field(n,regime,amplitude,p["seed"]);h=.03
                base=df_step(state,h,eq,geom,track)
                coarse=lawson_reference(state,h,eq,geom,64,track,check_budget=ctx.budget.check)
                teacher=lawson_reference(state,h,eq,geom,128,track,check_budget=ctx.budget.check)
                with torch.no_grad():
                    fixed=models["quad2_full"](state,h,eq,geom)
                    cubic=models["commutator_cubic"](state,h,eq,geom)
                amplitudes.append(dict(track=track,regime=regime,grid=n,horizon=h,amplitude=amplitude,
                    base_error_rms=rms(base-teacher),quad_error_rms=rms(fixed-teacher),cubic_error_rms=rms(cubic-teacher),
                    uncertainty_rms=rms(teacher-coarse),
                    reference_accepted=float((teacher-coarse).abs().max())<=2e-7,
                    scope="fixed time and spatial grid; observed amplitude dependence is not an asymptotic-order proof"))
    write_json(ctx.path/"diagnostic_rows.json",dict(rows=clean(rows)))
    write_json(ctx.path/"residual_projections.json",dict(rows=clean(projections)))
    write_json(ctx.path/"composition_rows.json",dict(rows=clean(composition)))
    write_json(ctx.path/"order_rows.json",dict(rows=clean(orders)))
    write_json(ctx.path/"amplitude_rows.json",dict(rows=clean(amplitudes)))
    return dict(endpoint_observations=len(rows),oracle_projections=len(projections),composition_probes=len(composition),
        teacher_target="same-grid equation; continuum compensation must be evaluated separately",
        no_training=True,no_confirmation_access=True)
