"""Cheap error-source contrasts and separately instrumented operation profiles.

Contrasts are not an additive error bound. In particular, the specified nodal
equation and the projected continuum equation have different solutions.
Optimization/deployment limits remain NA until their own evidence exists.
"""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from scipy.integrate import solve_ivp

from tdn.analysis.frontier.data import lawson_reference
from tdn.analysis.frontier.numerics import rhs
from tdn.analysis.roadmap.numerics import fourier_resample, quadratic_df_defect, quadrature
from tdn.numerics import Equation, Geometry
from tdn.runtime.metadata import write_json
from .core import check
from .models import FAMILIES, MODEL_SPECS, make_model, physical_features
from .protocol import digest


def _state(n, regime, seed):
    rng = np.random.default_rng(seed)
    x = torch.arange(n, dtype=torch.float64)/n
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    high = max(2, n//2-2)
    modes = {"low": [(1,0),(0,1)], "high_pair": [(high-1,1),(high,1)],
             "rough": [(1,0),(2,1),(high,-1)], "near_nyquist": [(high,0),(0,high)]}[regime]
    u = torch.full((n,n), .43, dtype=torch.float64)
    for j,(a,b) in enumerate(modes):
        u += .07/math.sqrt(j+1)*torch.cos(2*math.pi*(a*xx+b*yy)+float(rng.uniform(0,2*math.pi)))
    return u[None,None]


def _norms(value):
    return dict(error_rms=float(value.square().mean().sqrt()), error_max=float(value.abs().max()))


def audit(ctx):
    checks = []
    for count in (2,4):
        nodes, weights = quadrature(count)
        for degree in range(2*count):
            error = abs(sum(w*x**degree for x,w in zip(nodes,weights))-1/(degree+1))
            ctx.record(f"quadrature/{count}/{degree}", ["A1","A3"],
                metrics={"moment_error":error}, checks=[check("polynomial-moment",error,2e-14,category="correctness"),
                    check("normalized-gauss-exactness",error,2e-14,category="math"),
                    check("predictive-benefit",None,None,reason="Algebra does not establish an accuracy-cost benefit")])
    u = _state(8,"low",int(ctx.protocol["exploration"]["seed"])+100)
    eq,geo=Equation(.004,2.),Geometry((8,8),(1.,1.))
    for track in ctx.protocol["tracks"]:
        for family in FAMILIES:
            ctx.budget.check()
            model=make_model(family,track,dict(modes=2,width=3,depth=2)).double()
            with torch.no_grad():
                zero=model(u,0.,eq,geo)
                answer=model(u,.02,eq,geo)
            error=float((zero-u).abs().max());finite=bool(torch.isfinite(answer).all())
            checks=[check("zero-time-identity",error,0.,category="correctness"),
                    check("finite",finite,True,"eq",category="correctness"),
                    check("h-zero-limit",error,0.,category="math")]
            if MODEL_SPECS[family]["nodes"]:
                for null,neq in (("diffusion",Equation(0.,2.)),("reaction",Equation(.004,0.))):
                    increment=model.correction_components(u,.02,neq,geo)["increment"]
                    checks.append(check("null-"+null,float(increment.detach().abs().max()),0.,category="correctness"))
            ctx.record(f"model/{track}/{family}",["A1","A3","A4"],metrics={"family":family,"track":track},checks=checks,
                config=model.architecture_metadata())
        # Independent adaptive DOP853 cross-check of the same finite equation.
        n=4; small=_state(n,"low",int(ctx.protocol["exploration"]["seed"])+101)
        geom=Geometry((n,n),(1.,1.));h=.06
        def fun(t,y):
            return rhs(torch.from_numpy(y.copy()).reshape_as(small),eq,geom,track).numpy().ravel()
        ode=solve_ivp(fun,(0,h),small.numpy().ravel(),method="DOP853",rtol=2e-12,atol=2e-13)
        teacher=lawson_reference(small,h,eq,geom,128,track,check_budget=ctx.budget.check)
        discrepancy=float((teacher-torch.from_numpy(ode.y[:,-1]).reshape_as(small)).abs().max())
        ctx.record(f"teacher/{track}",["A4"],metrics={"track":track,"independent_discrepancy_max":discrepancy,"dop853_nfev":ode.nfev},
            checks=[check("independent-integrator",discrepancy,2e-9,category="correctness"),
                    check("adaptive-completion",bool(ode.success),True,"eq",category="correctness"),
                    check("finite-case-agreement",discrepancy,2e-9,category="math")])
    return dict(model_cases=2*len(FAMILIES),independent_teacher_tracks=2,
                purpose="implementation and finite mathematical checks; no comparative claim")


def _category(name):
    key=name.lower()
    if "fft" in key: return "transforms"
    if any(s in key for s in ("empty","alloc","resize")): return "allocations"
    if any(s in key for s in ("copy","to_copy","clone")): return "copies"
    if any(s in key for s in ("mul","add","sub","div","square","pow")): return "pointwise_arithmetic"
    if any(s in key for s in ("exp","tanh","sigmoid","silu")): return "nonlinear_functions"
    if any(s in key for s in ("mm","linear","convolution")): return "learned_dense_or_convolution"
    return "other_tensor_operations"


def _profile(ctx,u,eq,geo,regime):
    records=[]
    for family in ctx.protocol["diagnostics"]["profile_families"]:
        model=make_model(family,"discrete",dict(modes=4,width=8,depth=2)).double()
        with torch.no_grad():
            model.clear_cache(); start=time.perf_counter(); model(u,.12,eq,geo);first=time.perf_counter()-start
            _,timing=ctx.measure(lambda:model(u,.12,eq,geo),repeats=ctx.protocol["diagnostics"]["repetitions"],warmup=1)
            records.append(dict(regime=regime,family=family,component="warm_end_to_end",seconds=timing["median_seconds"],
                                samples_seconds=timing["samples_seconds"],measurement_scope="uninstrumented CPU wall latency"))
            records.append(dict(regime=regime,family=family,component="first_invocation",seconds=first,
                                measurement_scope="new model coefficients; warm process and libraries, not cold process"))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                model(u,.12,eq,geo)
            categories={}
            for event in profile.key_averages():
                category=_category(event.key)
                categories[category]=categories.get(category,0.)+event.self_cpu_time_total/1e6
            for category,seconds in categories.items():
                records.append(dict(regime=regime,family=family,component=category,seconds=seconds,
                    measurement_scope="instrumented self CPU operator time; Python/profiler overhead excluded; not deployment latency"))
            ctx.budget.check()
    return records


def run(ctx):
    declared=ctx.protocol["diagnostics"];n=declared["grid"];h=declared["horizon"]
    rows=[];profiles=[]
    eq,geo=Equation(.004,3.),Geometry((n,n),(1.,1.))
    seed=int(ctx.protocol["exploration"]["seed"])+200
    for index,regime in enumerate(declared["regimes"]):
        ctx.budget.check();u=_state(n,regime,seed+index)
        discrete_reference=None
        for track in ctx.protocol["tracks"]:
            start=time.perf_counter()
            levels=[lawson_reference(u,h,eq,geo,count,track,check_budget=ctx.budget.check) for count in (32,64,128)]
            differences=[_norms(levels[i+1]-levels[i]) for i in range(2)]
            uncertainty=max(differences[-1]["error_max"],1e-12)
            accepted=(uncertainty<=1e-8 and (differences[-1]["error_max"]<=differences[0]["error_max"]/4 or uncertainty<=1e-12))
            if track=="discrete":discrete_reference=levels[-1]
            def add(component,value=None,interpretation="",**extras):
                row=dict(diagnostic="bottleneck_contrast",regime=regime,track=track,component=component,
                         error_rms=None,error_max=None,reference_uncertainty=uncertainty,
                         reference_accepted=accepted,seconds=None,interpretation=interpretation)
                row.update(extras)
                if value is not None:row.update(_norms(value))
                rows.append(row)
            add("reference",levels[-1]-levels[-2],"Observed last time-refinement change; conservative estimate, not a certificate",
                seconds=time.perf_counter()-start,refinement_counts=[32,64,128],refinement_differences=differences)
            model=make_model("df",track,dict(modes=4)).double()
            for count in (1,2,4,8):
                answer=u.clone()
                with torch.no_grad():
                    for _ in range(count):answer=model(answer,h/count,eq,geo)
                add("temporal",answer-levels[-1],"Same-grid DF integration error; includes reaction-subflow approximation on continuum track",steps=count)
            increments={}
            for nodes in (2,4,16):
                positions,weights=quadrature(nodes)
                increments[nodes]=quadratic_df_defect(u,h,eq,geo,target=track,node_positions=positions,node_weights=weights)
            for nodes in (2,4):
                add("quadrature",increments[nodes]-increments[16],"Node refinement of identical quadratic formula; excludes higher Picard orders",nodes=nodes)
            outputs={family:make_model(family,track,dict(modes=4)).double()(u,h,eq,geo).detach()
                     for family in ("quad2_fixed","quad2_full","quad2_input","quad4_full","analytic_quad_cubic")}
            add("compression",outputs["quad2_full"]-outputs["quad2_fixed"],"Output cutoff contrast; signed contributions can cancel other errors")
            add("precompression",outputs["quad2_input"]-outputs["quad2_fixed"],"Input filtering destroys available interactions; matched output cutoff")
            for family in ("quad4_full","analytic_quad_cubic"):
                add("interaction_remainder",outputs[family]-levels[-1],"Endpoint error after analytic correction; higher-order interactions and finite-step terms remain",family=family)
            if track=="continuum":
                fine=[]
                for factor in (2,4):
                    # Resample the same coarse continuous polynomial, never regenerate different frequencies.
                    lifted=fourier_resample(u,(n*factor,n*factor))
                    fine.append(fourier_resample(lawson_reference(lifted,h,eq,Geometry((n*factor,n*factor),(1.,1.)),128,
                                             track,check_budget=ctx.budget.check),(n,n)))
                add("spatial",levels[-1]-fine[-1],"Galerkin truncation contrast at fixed fine temporal count; not a temporal-learning target")
                add("spatial_reference",fine[-1]-fine[-2],"Actual doubled-grid continuum reference change; time error reported separately")
                add("equation_mismatch",discrete_reference-fine[-1],"Specified nodal equation versus projected continuum; cannot count as correcting the same discrete dynamics")
            add("optimization",interpretation="NA here: use observed learning/validation curves and initialized controls in train units")
            add("deployment",interpretation="NA: no certified estimator or fallback policy evaluated; require a positive complete-cost margin first")
            ctx.record(f"diagnostic/{regime}/{track}",["A1","A3","A4","B4"],metrics={"regime":regime,"track":track,"reference_uncertainty":uncertainty},
                checks=[check("time-refinement",accepted,True,"eq",category="correctness"),
                        check("temporal-refinement-change",uncertainty,1e-8,category="math"),
                        check("net-learnable-headroom",None,None,reason="Contrasts are not independent additive error bounds; matched learned controls still required")])
        profiles.extend(_profile(ctx,u,eq,geo,regime))
        with torch.no_grad():
            for rich in (False,True):
                from tdn.analysis.frontier.numerics import validate
                _,timing=ctx.measure(lambda:physical_features(u,validate(u,h,geo,"discrete"),eq,geo,"discrete",rich=rich),
                                    repeats=declared["repetitions"],warmup=1)
                profiles.append(dict(regime=regime,family="conditioned_rich" if rich else "quad2_conditioned",
                    component="standalone_feature_extraction",seconds=timing["median_seconds"],
                    samples_seconds=timing["samples_seconds"],measurement_scope="uninstrumented isolated feature call; overlaps end-to-end work"))
    write_json(ctx.path/"diagnostic_rows.json",rows)
    write_json(ctx.path/"profile_rows.json",profiles)
    write_json(ctx.path/"error_budget_semantics.json",dict(additive_bound=False,
        error_terms=declared["terms"],unmeasured=["optimization until training","deployment until underlying cost margin"],
        target_tracks_separate=True,reference_uncertainty_certificate=False,
        cpu_profile_does_not_predict_gpu=True,protocol_sha256=digest(ctx.protocol)))
    return dict(diagnostic_contrasts=len(rows),profile_components=len(profiles),regimes=len(declared["regimes"]),
                error_terms_are_nonadditive=True,all_estimates_development=True)
