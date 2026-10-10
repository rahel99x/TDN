"""Two bounded alternatives; neither is promoted into TDN's main architecture.

E01 chooses whether to form the expensive cubic term *before* computing it.
E02 replaces nested cubic transport with a signed homogeneous cubic direction
suggested by the prior amplitude-order diagnostic. Its coefficient is fitted
on disjoint development parents, never selected using held-out outcomes.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from tdn.analysis.frontier.core import check
from tdn.analysis.frontier import numerics
from tdn.analysis.frontier.data import lawson_reference
from tdn.analysis.roadmap.numerics import quadratic_df_defect, cubic_df_defect
from tdn.numerics import Equation, Geometry
from tdn.runtime.metadata import write_json
from .temporal import paired_timing, _error, _measure


PROPOSALS = {
    "E01": dict(name="Pre-work cubic selection", role="Ours numerical prototype",
        challenged_assumption="Every field needs the same interaction order",
        mechanism="Compute cubic only when h*r*RMS(u-mean(u)) exceeds a fixed threshold",
        saved_work="Nested cubic transports and products on skipped fields",
        failure_mode="Amplitude statistic can miss phase-specific cubic relevance",
        falsification="Held-out complete-PDE error exceeds the frozen protocol regression allowance",
        advance="Use frozen protocol speed, regression and quadratic-improvement gates, with resolved accepted references and no new maximum-target failures",
        novelty="New to TDN implementation; deterministic adaptivity and order selection are established"),
    "E02": dict(name="Signed cubic modulation", role="Ours fitted numerical prototype",
        challenged_assumption="A missing cubic spatial direction requires nested cubic quadrature",
        mechanism="Fit gamma*P[(u-mean(u))*Q(u)] to the analytic cubic on independent fit parents",
        saved_work="Replace nested Volterra transports with one signed physical product",
        failure_mode="Cubic shape need not align with the modulated quadratic direction",
        falsification="Held-out cubic residual remains large or complete-PDE accuracy regresses",
        advance="Use frozen protocol speed, regression and quadratic-improvement gates, with resolved accepted references and no new maximum-target failures",
        novelty="Diagnostics-generated new-to-TDN combination; not a literature novelty claim"),
}


def selection_statistic(u, h, equation):
    """Cheap dimensionless finite-amplitude indicator; no target/cubic input."""
    if not math.isfinite(h) or h < 0:
        raise ValueError("Finite nonnegative step required")
    fluctuation = u-u.mean(tuple(range(2,u.ndim)),keepdim=True)
    return float(h*equation.reaction_rate*fluctuation.square().mean().sqrt())


def selective_correction(u,h,equation,geometry,track,*,threshold=.012,cubic_function=None):
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("Finite nonnegative selector threshold required")
    selected=selection_statistic(u,h,equation)>threshold
    q=quadratic_df_defect(u,h,equation,geometry,target=track,nodes=4)
    # This branch is intentionally before calling cubic_function; tests inject a
    # failing callback to demonstrate skipped expensive work does not occur.
    if selected:
        cubic_function=cubic_function or cubic_df_defect
        q=q+cubic_function(u,h,equation,geometry,target=track,nodes=4)
    return q,selected


def signed_modulation(u,quadratic,track):
    """Same spatial product as target; cubic in fluctuation at fixed background.

    No additional gate squares an already-quadratic term. No absolute values
    erase phase: Q(-v)=Q(v), so this direction changes sign under v->-v.
    """
    if u.shape!=quadratic.shape:
        raise ValueError("Quadratic direction and field must share the grid")
    return numerics.product(u-u.mean(tuple(range(2,u.ndim)),keepdim=True),quadratic,track)


def fit_modulation(pairs,*,bound=8.,floor=1e-28):
    """One bounded global least-squares coefficient, not an oracle per query."""
    if bound<=0 or floor<=0:
        raise ValueError("Positive fit bounds and numerical floor required")
    numerator=sum(float((direction*cubic).sum()) for direction,cubic in pairs)
    denominator=sum(float(direction.square().sum()) for direction,_ in pairs)
    if denominator<=floor:
        return dict(coefficient=0.,status="UNRESOLVED_ZERO_DIRECTION",denominator=denominator,
                    admissible_interval=[-bound,bound],training_pairs=len(pairs))
    unconstrained=numerator/denominator
    return dict(coefficient=max(-bound,min(bound,unconstrained)),unconstrained_coefficient=unconstrained,
                status="FITTED",denominator=denominator,admissible_interval=[-bound,bound],training_pairs=len(pairs))


def _parent(seed,index,n):
    rng=np.random.default_rng(seed+index)
    x=torch.arange(n,dtype=torch.float64)/n
    xx,yy=torch.meshgrid(x,x,indexing="ij")
    high=max(2,n//2-2)
    modes=[(1,0),(high,1),(high-1,-1)]
    phases=rng.uniform(-np.pi,np.pi,len(modes))
    wave=sum(torch.cos(2*torch.pi*(a*xx+b*yy)+float(p))/math.sqrt(i+1)
             for i,((a,b),p) in enumerate(zip(modes,phases)))
    wave=wave-wave.mean();wave=wave/wave.abs().max()
    amplitude=(.02,.07,.14,.22)[index%4]
    state=(.43+amplitude*wave)[None,None]
    h=(.04,.08,.12,.16)[(index//2)%4]
    return state,h,dict(parent_id=f"adjacent-exploration-{seed+index}",amplitude=amplitude,
                        phases=phases.tolist(),mode_vectors=modes,h=h)


def _emit(ctx,rows,prototype,identifier,metrics,checks,config):
    row=dict(schema="tdn.adjacent-prototype/v1",prototype_id=prototype,experiment_id=identifier,
             method=config["method"],role=config["role"],status="COMPLETED",metrics=metrics,
             checks=checks,config=config)
    rows.append(row)
    mechanism_ids=([prototype] if prototype!="CONTROL" else
                   getattr(ctx,"unit",{}).get("prototype_ids",["E01","E02"]))
    ctx.record(identifier,mechanism_ids,metrics=metrics,checks=checks,config=config,
               evidence=["prototype_rows.json","prototype_measurements.json","prototype_fits.json"])


def assess_cohort(prototype,track,cohort,control,quadratic_control,gate,maximum_target):
    """Full-denominator decision: unresolved accuracy can never pass a cost gate.

    Uncertainty-adjusted ratios are sensitivity checks against a measured
    refinement estimate, not certified error intervals. No denominator is
    inflated to turn an uncertainty-sized control error into a favorable ratio.
    """
    finite=all(r.get("finite_prediction",False) and control[r["parent_id"]].get("finite_prediction",False)
               and quadratic_control[r["parent_id"]].get("finite_prediction",False) for r in cohort)
    accepted=[r for r in cohort if r.get("reference_accepted",False)]
    resolved=[r for r in cohort if r.get("errors_resolved",False)
              and control[r["parent_id"]].get("errors_resolved",False)
              and quadratic_control[r["parent_id"]].get("errors_resolved",False)]
    all_resolved=bool(cohort) and len(accepted)==len(cohort) and len(resolved)==len(cohort) and finite
    speed=float(np.median([r["speedup_over_analytic"] for r in cohort])) if cohort else None
    worst=improvement=newly_failed=None
    if all_resolved:
        worst=max((r["error_rms"]+r["reference_uncertainty_max"])/
                  (control[r["parent_id"]]["error_rms"]-r["reference_uncertainty_max"]) for r in cohort)
        improvement=float(np.median([(quadratic_control[r["parent_id"]]["error_rms"]-r["reference_uncertainty_max"])/
                    (r["error_rms"]+r["reference_uncertainty_max"]) for r in cohort]))
        newly_failed=sum(r["error_max"]+r["reference_uncertainty_max"]>maximum_target and
            control[r["parent_id"]]["error_max"]+r["reference_uncertainty_max"]<=maximum_target for r in cohort)
    passed=all_resolved and speed>=gate["speedup"] and worst<=gate["rms_ratio"] and newly_failed==0 and improvement>=gate["quadratic_improvement"]
    utility,decision=("BAD","NUMERICAL_FAILURE") if not finite else (
        ("NA","REFINE_REFERENCE") if not all_resolved else
        ("GOOD","FOCUSED_PILOT") if passed else ("BAD","STOP_CURRENT_VARIANT"))
    return dict(prototype_id=prototype,track=track,independent_parent_count=len(cohort),
        reference_accepted_parent_count=len(accepted),resolved_parent_count=len(resolved),
        unresolved_parent_ids=[r["parent_id"] for r in cohort if r not in accepted or r not in resolved],
        finite_predictions=finite,full_denominator_required=True,
        median_paired_speedup=speed,worst_rms_ratio=worst,new_maximum_target_failures=newly_failed,
        median_accuracy_improvement_over_quadratic=improvement,frozen_thresholds=gate,
        ratio_scope="refinement-uncertainty-adjusted sensitivity, not a certified bound",
        skipped_cubic_parents=sum(not r["cubic_computed"] for r in cohort),
        utility=utility,decision=decision,scientific_status="DEVELOPMENT_ONLY",confidence_interval=None,
        qualification="Independent held-out development parents; unresolved cases retain denominator weight; no confirmatory claim")


def run(ctx):
    if str(getattr(ctx,"device","cpu"))!="cpu":
        raise ValueError("Alternative prototypes explicitly measure CPU")
    unit=getattr(ctx,"unit",{})
    ids=unit.get("prototype_ids",["E01","E02"])
    if not ids or any(p not in PROPOSALS for p in ids):
        raise ValueError("Only the two registered alternative prototypes are permitted")
    smoke=ctx.protocol.get("profile")=="smoke"
    options=dict(n=8 if smoke else 16,fit_parents=4 if smoke else 8,
                 heldout_parents=4 if smoke else 8,repeats=2 if smoke else 5,
                 threshold=.012,seed=944001,relative_rms_regression=1.10,
                 minimum_speedup=1.10,maximum_target=2e-5)
    options.update(ctx.protocol.get("prototypes",{}))
    exploration=ctx.protocol.get("exploration",{})
    gates={name:dict(rms_ratio=(1+exploration.get(name,{}).get("maximum_accuracy_regression",.10)
                               if name=="E01" else exploration.get(name,{}).get("error_ratio",1.10)),
                    speedup=exploration.get(name,{}).get("speedup",1.10),
                    quadratic_improvement=exploration.get(name,{}).get("minimum_accuracy_improvement_over_quadratic",1.10))
           for name in ids}
    tracks=ctx.protocol.get("tracks",["discrete","continuum"])
    n=options["n"];geo=Geometry((n,n),(1.,1.));eq=Equation(.004,3.)
    rows=[];measurements=[];fits={};decisions=[];rng=np.random.default_rng(options["seed"])
    with torch.no_grad():
        for track in tracks:
            fit_start=time.perf_counter();pairs=[];fit_ids=[]
            if "E02" in ids:
                for i in range(options["fit_parents"]):
                    ctx.budget.check()
                    u,h,parent=_parent(options["seed"],i,n)
                    q=quadratic_df_defect(u,h,eq,geo,target=track,nodes=4)
                    c=cubic_df_defect(u,h,eq,geo,target=track,nodes=4)
                    pairs.append((signed_modulation(u,q,track),c));fit_ids.append(parent["parent_id"])
                fit=fit_modulation(pairs)
            else: fit=dict(coefficient=0.,status="NOT_APPLICABLE",training_pairs=0)
            fit.update(parent_ids=fit_ids,fit_seconds=time.perf_counter()-fit_start,
                       objective="Analytic cubic squared L2 on fit parents only",track=track)
            fits[track]=fit
            for i in range(options["heldout_parents"]):
                ctx.budget.check()
                u,h,parent=_parent(options["seed"]+10000,i,n)
                base=numerics.df_step(u,h,eq,geo,track)
                q=quadratic_df_defect(u,h,eq,geo,target=track,nodes=4)
                c=cubic_df_defect(u,h,eq,geo,target=track,nodes=4)
                teacher,reference_seconds=_measure(lambda:lawson_reference(u,h,eq,geo,128,track,check_budget=ctx.budget.check))
                fine,refinement_seconds=_measure(lambda:lawson_reference(u,h,eq,geo,256,track,check_budget=ctx.budget.check))
                references_finite=bool(torch.isfinite(fine).all() and torch.isfinite(teacher).all())
                uncertainty=max(float((fine-teacher).abs().max()),1e-12) if references_finite else None
                reference_accepted=references_finite and uncertainty<=2e-8
                def classical():
                    return numerics.df_step(u,h,eq,geo,track)+quadratic_df_defect(u,h,eq,geo,target=track,nodes=4)+cubic_df_defect(u,h,eq,geo,target=track,nodes=4)
                def quadratic():
                    return numerics.df_step(u,h,eq,geo,track)+quadratic_df_defect(u,h,eq,geo,target=track,nodes=4)
                def selected():
                    return numerics.df_step(u,h,eq,geo,track)+selective_correction(u,h,eq,geo,track,threshold=options["threshold"])[0]
                def modulated():
                    quad=quadratic_df_defect(u,h,eq,geo,target=track,nodes=4)
                    return numerics.df_step(u,h,eq,geo,track)+quad+fit["coefficient"]*signed_modulation(u,quad,track)
                calls={"analytic_quad_cubic":classical,"quadratic":quadratic}
                if "E01" in ids:calls["selective_cubic"]=selected
                if "E02" in ids:calls["signed_modulation"]=modulated
                timings=paired_timing(calls,repeats=options["repeats"],rng=rng,budget=ctx.budget)
                truth=fine.numpy();control_answer=classical()
                control_finite=bool(torch.isfinite(control_answer).all())
                control_error=(_error(control_answer.numpy(),truth) if references_finite and control_finite else
                               dict(error_rms=None,error_max=None))
                for method,call in calls.items():
                    answer=call();finite_prediction=bool(torch.isfinite(answer).all())
                    error=(_error(answer.numpy(),truth) if references_finite and finite_prediction else
                           dict(error_rms=None,error_max=None))
                    prediction=answer-base-q
                    direction_error=(_error(prediction.numpy(),c.numpy()) if finite_prediction else
                                     dict(error_rms=None,error_max=None))
                    errors_resolved=bool(reference_accepted and finite_prediction and control_finite
                        and control_error["error_rms"]>5*uncertainty and error["error_rms"]>5*uncertainty)
                    ratio=error["error_rms"]/control_error["error_rms"] if errors_resolved else None
                    speedup=timings["analytic_quad_cubic"]["median_seconds"]/timings[method]["median_seconds"]
                    selected_bool=selection_statistic(u,h,eq)>options["threshold"]
                    measurement=dict(parent,track=track,split="heldout_development",method=method,
                        timing=timings[method],reference_seconds=reference_seconds+refinement_seconds,
                        reference_cost_id=f"{track}/{parent['parent_id']}",reference_cost_shared_across_methods=True,
                        reference_uncertainty_max=uncertainty,reference_is_certificate=False,
                        reference_accepted=reference_accepted,errors_resolved=errors_resolved,
                        finite_prediction=finite_prediction,
                        **error,cubic_direction_error=direction_error,rms_ratio_to_analytic=ratio,
                        speedup_over_analytic=speedup,selection_statistic=selection_statistic(u,h,eq),
                        cubic_computed=(selected_bool if method=="selective_cubic" else method=="analytic_quad_cubic"),
                        scalar_fit_coefficient=fit["coefficient"] if method=="signed_modulation" else None)
                    measurements.append(measurement)
                    prototype="E01" if method=="selective_cubic" else "E02" if method=="signed_modulation" else "CONTROL"
                    gate=gates[prototype] if prototype!="CONTROL" else dict(rms_ratio=None,speedup=None)
                    checks=[check("reference-refinement",uncertainty,2e-8,category="math"),
                            check("finite-prediction",finite_prediction,True,"eq",category="correctness"),
                            check("resolved-accuracy",True if errors_resolved else None,True,"eq",category="gap",
                                  reason="Both control and measured errors must exceed five times accepted reference uncertainty"),
                            check("rms-regression",ratio,gate["rms_ratio"]),
                            check("complete-speedup",speedup if prototype!="CONTROL" else None,
                                  gate["speedup"],"ge",category="utility",
                                  reason="Analytic rows provide the comparison; they do not claim prototype savings" if prototype=="CONTROL" else "")]
                    _emit(ctx,rows,prototype,f"{prototype}/{track}/{parent['parent_id']}/{method}",
                        dict(error,rms_ratio_to_analytic=ratio,speedup_over_analytic=speedup,
                             reference_accepted=reference_accepted,errors_resolved=errors_resolved,
                             finite_prediction=finite_prediction,
                             cubic_direction_error=direction_error,complete_seconds=timings[method]["median_seconds"]),
                        checks,dict(parent,method=method,track=track,split="heldout_development",
                                    role="Analytic control" if prototype=="CONTROL" else PROPOSALS[prototype]["role"]))
            for prototype in ids:
                method={"E01":"selective_cubic","E02":"signed_modulation"}[prototype]
                cohort=[r for r in measurements if r["track"]==track and r["method"]==method]
                control={r["parent_id"]:r for r in measurements if r["track"]==track and r["method"]=="analytic_quad_cubic"}
                quadratic_control={r["parent_id"]:r for r in measurements if r["track"]==track and r["method"]=="quadratic"}
                gate=gates[prototype]
                decisions.append(assess_cohort(prototype,track,cohort,control,quadratic_control,gate,options["maximum_target"]))
    write_json(ctx.path/"prototype_rows.json",rows)
    write_json(ctx.path/"prototype_measurements.json",measurements)
    write_json(ctx.path/"prototype_fits.json",fits)
    write_json(ctx.path/"prototype_decisions.json",dict(proposals={k:{**PROPOSALS[k],"frozen_thresholds":gates[k]} for k in ids},decisions=decisions,
        protected_budget_fraction=.20,budget_scope="Adjacent discretionary pilot budget only; independent of main portfolio",
        mathematical_status="Finite numerical checks, not a proof",energy_joules=None,monetary_cost=None))
    return dict(status="COMPLETED",prototype_ids=ids,row_count=len(rows),decisions=decisions,
                device="cpu",scientific_status="DEVELOPMENT_ONLY")
