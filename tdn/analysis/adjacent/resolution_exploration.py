"""Bounded 64²/128² follow-ups to negative E01 and D07 development findings.

No historical result is replaced. These two new variants inspect fixed
training-bank parents, never untouched evaluation parents. References are
consumed from sealed banks; missing intermediate truth remains explicit NA.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
from numpy.polynomial.chebyshev import chebfit
from scipy.integrate import solve_ivp
import torch

from tdn.analysis.frontier import numerics
from tdn.analysis.frontier.core import check
from tdn.analysis.roadmap.numerics import quadratic_df_defect, cubic_df_defect,background_response,quadrature
from tdn.numerics import Equation, Geometry
from tdn.runtime.metadata import write_json


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _timed(call, device):
    _sync(device); start=time.perf_counter(); answer=call(); _sync(device)
    return answer,time.perf_counter()-start


def interaction_selector(u,q,h,equation,*,absolute_rms_target,budget_fraction=.1):
    """A deployable heuristic, not a certified cubic bound or fitted policy.

    Q is required by both branches and is computed before this decision.
    Unlike the historical amplitude-only rule, this statistic includes the
    actual signed quadratic interaction's spatial norm. No cubic is evaluated
    to decide whether cubic work is needed.
    """
    if absolute_rms_target<=0 or not 0<budget_fraction<=1 or h<0:
        raise ValueError("Positive target, bounded budget fraction and nonnegative time required")
    fluctuation=u-u.mean(tuple(range(2,u.ndim)),keepdim=True)
    indicator=float(h*equation.reaction_rate*fluctuation.abs().amax()*q.square().mean().sqrt())
    threshold=absolute_rms_target*budget_fraction
    return indicator>threshold,dict(indicator=indicator,threshold=threshold,
        indicator_semantics="h*r*max|v|*RMS(Q); heuristic, not a mathematical error bound")


def selected_step(u,h,equation,geometry,track,*,target=2e-5,budget_fraction=.1,cubic_function=None,cache=None):
    base=numerics.df_step(u,h,equation,geometry,track,cache=cache)
    q=(cached_quadratic(u,h,equation,geometry,track,cache) if cache is not None else
       quadratic_df_defect(u,h,equation,geometry,nodes=4,target=track))
    selected,features=interaction_selector(u,q,h,equation,absolute_rms_target=target,
                                           budget_fraction=budget_fraction)
    if selected:
        cubic_function=cubic_function or cubic_df_defect
        q=q+cubic_function(u,h,equation,geometry,nodes=4,target=track)
    return base+q,dict(features,cubic_computed=selected)


def cached_quadratic(u,h,equation,geometry,track,cache):
    """The identical GL4 physical formula with cached transport coefficients.

    State-dependent transported fields/products are always recomputed. The
    cache contains operator multipliers only, granting the direct comparator
    its own valid reuse opportunity instead of withholding it from the control.
    """
    numerics.validate(u,h,geometry,track)
    if h==0 or equation.kappa==0 or equation.reaction_rate==0:return torch.zeros_like(u)
    axes=tuple(range(2,u.ndim));mean=u.mean(axes,keepdim=True);v=u-mean
    mean=torch.where(v.abs().amax(axes,keepdim=True)==0,torch.full_like(mean,.5),mean)
    _,qh,_=background_response(mean,h,equation.reaction_rate)
    def heat(field,t):return numerics.heat_step(field,t,equation,geometry,track,cache=cache)
    middle=heat(v,h/2);split=heat(numerics.product(middle,middle,track),h/2)
    result=torch.zeros_like(u);nodes,weights=quadrature(4)
    for node,weight in zip(nodes,weights):
        s=h*float(node);_,qs,_=background_response(mean,s,equation.reaction_rate)
        propagated=heat(v,s)
        shape=heat(numerics.product(propagated,propagated,track),h-s)
        result=result+float(weight)*qs*(shape-split)
    return -equation.reaction_rate*qh*h*result


def direct_step(u,h,equation,geometry,track,*,cubic=False,cache=None):
    q=(cached_quadratic(u,h,equation,geometry,track,cache) if cache is not None else
       quadratic_df_defect(u,h,equation,geometry,nodes=4,target=track))
    result=numerics.df_step(u,h,equation,geometry,track,cache=cache)+q
    return result+cubic_df_defect(u,h,equation,geometry,nodes=4,target=track) if cubic else result


def _fingerprint(u,equation,geometry,track,horizon):
    identity=repr((tuple(u.shape),str(u.dtype),str(u.device),equation.kappa,
        equation.reaction_rate,geometry.grid,geometry.lengths,track,float(horizon))).encode()
    return hashlib.sha256(identity+u.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


class DeviceCorrectionEncoding:
    """Immutable-state temporal correction, with charged FP64 preparation.

    Q(h)/h³ is sampled in FP64 on the declared device to avoid amplified FP32
    cancellation near zero. A CPU Chebyshev fit and both transfers are charged
    inside preparation. Stored coefficients, backbone and decoding use the
    original inference precision. No FFT is taken over polynomial channels.
    """
    def __init__(self,u,equation,geometry,track,horizon,*,degree=6,check_budget=lambda:None):
        if not math.isfinite(horizon) or horizon<=0 or type(degree) is not int or degree<1:
            raise ValueError("Positive finite horizon and polynomial degree required")
        numerics.validate(u,0.,geometry,track)
        self.state=u.detach().clone();self.equation=equation;self.geometry=geometry
        self.track,self.horizon,self.degree=track,float(horizon),degree
        self.identity=_fingerprint(u,equation,geometry,track,horizon)
        self.cache=numerics.CoefficientCache()
        x=np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1));times=horizon*(x+1)/2
        initial64=u.to(torch.float64);samples=[]
        with torch.no_grad():
            for h in times:
                check_budget()
                q=quadratic_df_defect(initial64,float(h),equation,geometry,nodes=4,target=track)
                samples.append((q/float(h)**3).cpu().numpy().ravel())
        coefficients=chebfit(x,np.stack(samples),degree)
        self.coefficients=torch.as_tensor(coefficients,dtype=u.dtype,device=u.device).reshape(degree+1,*u.shape)
        self.preparation_precision="float64";self.inference_precision=str(u.dtype)

    def validate(self,u,equation,geometry,track,horizon):
        if _fingerprint(u,equation,geometry,track,horizon)!=self.identity:
            raise ValueError("Changed state, physical operator, grid, precision or interval requires refresh")

    def correction(self,h):
        if not math.isfinite(h) or not 0<=h<=self.horizon:
            raise ValueError("Query outside stored temporal interval")
        x=2*h/self.horizon-1
        b1=torch.zeros_like(self.state);b2=torch.zeros_like(self.state)
        for coefficient in reversed(self.coefficients[1:]):
            b0=2*x*b1-b2+coefficient;b2,b1=b1,b0
        return h**3*(x*b1-b2+self.coefficients[0])

    def query(self,h):
        return numerics.df_step(self.state,h,self.equation,self.geometry,self.track,cache=self.cache)+self.correction(h)

    @property
    def storage_bytes(self):
        return (self.state.numel()+self.coefficients.numel())*self.state.element_size()


def classical_dense(initial,equation,geometry,track,horizon,*,rtol=1e-7,atol=1e-9,check_budget=lambda:None):
    """Complete finite-equation CPU DOP853 with charged device transfers.

    The solver supports many queries from one state through its established
    dense output. Its Torch/NumPy RHS is disclosed, not called an optimized
    authoritative implementation. Every RHS checks the enclosing wall budget.
    """
    u=initial.detach().cpu().to(torch.float64)
    def rhs(_,values):
        check_budget()
        return numerics.rhs(torch.from_numpy(values.copy()).reshape_as(u),equation,geometry,track).numpy().ravel()
    solution=solve_ivp(rhs,(0.,horizon),u.numpy().ravel(),method="DOP853",dense_output=True,rtol=rtol,atol=atol)
    if not solution.success:
        raise RuntimeError("Dense classical solve failed: "+solution.message)
    return solution


def paired_calls(calls,device,*,repeats,rng,check_budget):
    """Whole-call randomized pairs with synchronization and separate first use."""
    if repeats<1:raise ValueError("Positive timing count required")
    first={};samples={name:[] for name in calls};answers={};memory={name:[] for name in calls}
    for name in rng.permutation(list(calls)):
        check_budget();answers[name],first[name]=_timed(calls[name],device)
    for repeat in range(repeats):
        for order,name in enumerate(rng.permutation(list(calls))):
            check_budget()
            if torch.device(device).type=="cuda":torch.cuda.reset_peak_memory_stats(device)
            answers[name],elapsed=_timed(calls[name],device)
            samples[name].append(dict(round=repeat,order=order,seconds=elapsed))
            if torch.device(device).type=="cuda":
                memory[name].append(dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device)))
    timing={name:dict(median_seconds=float(np.median([r["seconds"] for r in values])),raw=values,
        first_invocation_seconds=first[name],device=str(device),
        peak_allocated_bytes=max((m["allocated"] for m in memory[name]),default=None),
        peak_reserved_bytes=max((m["reserved"] for m in memory[name]),default=None),
        memory_scope="whole process allocator during call, including resident comparator caches; CPU peak NA",
        scope="randomized paired complete call; first invocation in warm process is separate") for name,values in samples.items()}
    return answers,timing


def _errors(prediction,reference,metadata):
    finite=bool(torch.isfinite(prediction).all())
    uncertainty=float(metadata.get("uncertainty_rms",metadata.get("reference_uncertainty",1e-12)))
    maximum_uncertainty=float(metadata.get("uncertainty_max_bound",metadata.get("reference_uncertainty",uncertainty)))
    accepted=bool(metadata.get("reference_accepted",metadata.get("accepted",False)))
    if not (math.isfinite(uncertainty) and math.isfinite(maximum_uncertainty)):
        accepted=False;uncertainty=maximum_uncertainty=None
    if finite:
        difference=prediction.detach().cpu().double()-reference.detach().cpu().double()
        rms=float(difference.square().mean().sqrt());maximum=float(difference.abs().max())
    else:rms=maximum=None
    return dict(error_rms=rms,error_max=maximum,finite=finite,reference_accepted=accepted,
                uncertainty_rms=uncertainty,uncertainty_max=maximum_uncertainty,uncertainty_is_certificate=False)


def _qualified(error,target):
    return bool(error["finite"] and error["reference_accepted"] and error["error_rms"]+error["uncertainty_rms"]<=target
                and error["error_max"]+error["uncertainty_max"]<=target)


def _emit(ctx,rows,row,checks):
    row=dict(schema="tdn.adjacent-resolution-exploration/v1",status="COMPLETED",**row)
    rows.append(row)
    write_json(ctx.path/"resolution_exploration_rows.json",rows)
    ctx.record(row["experiment_id"],[row["mechanism_id"]],metrics=row["metrics"],checks=checks,
        config=row["config"],evidence=["resolution_exploration_rows.json","resolution_exploration_costs.json"])


def _selective(ctx,u,ref,metadata,equation,geometry,track,options,parent,rows,costs):
    target=options["target"];h=options["horizon"]
    caches={name:numerics.CoefficientCache() for name in ("quadratic","quadratic_cubic","interaction_selected_cubic")}
    calls={"quadratic":lambda:direct_step(u,h,equation,geometry,track,cache=caches["quadratic"]),
           "quadratic_cubic":lambda:direct_step(u,h,equation,geometry,track,cubic=True,cache=caches["quadratic_cubic"]),
           "interaction_selected_cubic":lambda:selected_step(u,h,equation,geometry,track,target=target,
                                          budget_fraction=options["selector_budget_fraction"],cache=caches["interaction_selected_cubic"])[0]}
    answers,timing=paired_calls(calls,ctx.device,repeats=options["repeats"],rng=options["rng"],check_budget=options["check"])
    errors={name:_errors(value,ref,metadata) for name,value in answers.items()}
    _,features=selected_step(u,h,equation,geometry,track,target=target,budget_fraction=options["selector_budget_fraction"])
    control=errors["quadratic_cubic"];ours=errors["interaction_selected_cubic"]
    resolved=bool(ours["finite"] and ours["reference_accepted"] and control["finite"]
        and control["error_rms"]>options["relative_error_floor_multiple"]*control["uncertainty_rms"])
    ratio=(ours["error_rms"]+ours["uncertainty_rms"])/(control["error_rms"]-control["uncertainty_rms"]) if resolved else None
    margin=timing["quadratic_cubic"]["median_seconds"]/timing["interaction_selected_cubic"]["median_seconds"]
    qualified={name:_qualified(error,target) for name,error in errors.items()}
    best_qualified=min((timing[name]["median_seconds"] for name in ("quadratic","quadratic_cubic") if qualified[name]),default=None)
    frontier_margin=(best_qualified/timing["interaction_selected_cubic"]["median_seconds"]
                     if best_qualified is not None and qualified["interaction_selected_cubic"] else None)
    for method in calls:
        metrics=dict(errors[method],complete_seconds=timing[method]["median_seconds"],
            accuracy_qualified=qualified[method],target=target,
            speedup_over_always_cubic=margin if method=="interaction_selected_cubic" else None,
            rms_ratio_to_cubic=ratio if method=="interaction_selected_cubic" else None,
            speedup_over_best_qualified_component=frontier_margin if method=="interaction_selected_cubic" else None)
        costs.append(dict(parent_id=parent,variant="resolution-E01-interaction-selector",method=method,
                          timing=timing[method],metrics=metrics))
        _emit(ctx,rows,dict(experiment_id=f"HR-E01/{parent}/{method}",mechanism_id="HR-E01",metrics=metrics,
            config=dict(parent_id=parent,grid=geometry.grid[0],track=track,horizon=h,method=method,
                kappa=equation.kappa,reaction_rate=equation.reaction_rate,domain=list(geometry.lengths),
                quad_nodes=4,cubic_nodes=4,output_compression="none",
                role="Ours numerical prototype" if method=="interaction_selected_cubic" else "Analytic control",
                selection=features,cohort="fixed training-bank parents; exploratory",ancestor="historical E01 negative retained")),
            [check("finite",errors[method]["finite"],True,"eq",category="correctness"),
             check("accepted-bank-reference",True if errors[method]["reference_accepted"] else None,True,"eq",category="math"),
             check("relative-cubic-regression",ratio if method=="interaction_selected_cubic" else None,
                   options["selector_rms_ratio_limit"]),
             check("complete-qualified-margin",frontier_margin if method=="interaction_selected_cubic" else None,
                   options["required_speedup"],"ge",category="utility")])
    return dict(parent_id=parent,resolved=resolved,finite=ours["finite"],reference_accepted=ours["reference_accepted"],
                rms_ratio=ratio,speedup=margin,frontier_speedup=frontier_margin,cubic_computed=features["cubic_computed"],
                qualified=qualified["interaction_selected_cubic"])


def _temporal(ctx,u,ref,metadata,equation,geometry,track,options,parent,rows,costs):
    h=options["horizon"];device=ctx.device;degree=options["temporal_degree"]
    def encode(state=u,interval=h):
        return DeviceCorrectionEncoding(state,equation,geometry,track,interval,degree=degree,check_budget=options["check"])
    encoding,encoding_seconds=_timed(encode,device)
    dense,dense_seconds=_timed(lambda:classical_dense(u,equation,geometry,track,h,
        rtol=options["temporal_rtol"],atol=options["temporal_atol"],check_budget=options["check"]),device)
    decisions=[]
    direct_cache=numerics.CoefficientCache(256)
    for count in options["query_counts"]:
        times=np.linspace(h/count,h,count)
        def encoded_queries():
            encoding.validate(u,equation,geometry,track,h)
            return torch.stack([encoding.query(float(t)) for t in times])
        def direct_queries():return torch.stack([direct_step(u,float(t),equation,geometry,track,cache=direct_cache) for t in times])
        def dense_queries():
            return torch.from_numpy(dense.sol(times).T.copy()).reshape(count,*u.shape).to(device=u.device,dtype=u.dtype)
        calls={"reusable_quadratic":encoded_queries,"direct_gl4":direct_queries,"classical_dense":dense_queries}
        answers,timing=paired_calls(calls,device,repeats=options["repeats"],rng=options["rng"],check_budget=options["check"])
        build={"reusable_quadratic":encoding_seconds,"direct_gl4":0.,"classical_dense":dense_seconds}
        for method,answer in answers.items():
            endpoint=_errors(answer[-1],ref,metadata)
            parity=float((answer-answers["direct_gl4"]).abs().max()) if method=="reusable_quadratic" else None
            total=build[method]+timing[method]["median_seconds"]
            # Q=1 only requests the bank endpoint. Larger Q currently lacks
            # independent interior bank truth: a plausible interpolant is not
            # itself a trajectory reference.
            all_query_accuracy=(_qualified(endpoint,options["target"]) if count==1 else None)
            metrics=dict(endpoint,query_count=count,build_seconds=build[method],query_seconds=timing[method]["median_seconds"],
                complete_seconds=total,complete_cost_scope="measured build plus measured entire query group; not Q times single-query latency",
                first_group_complete_seconds=build[method]+timing[method]["first_invocation_seconds"],
                operator_cache_policy="equivalent reusable DF coefficient caches; first query group and warm groups separate",
                all_query_accuracy=all_query_accuracy,endpoint_accuracy_qualified=_qualified(endpoint,options["target"]),
                interpolation_parity_max=parity,storage_bytes=encoding.storage_bytes if method=="reusable_quadratic" else None,
                peak_memory_bytes=None)
            costs.append(dict(parent_id=parent,variant="resolution-temporal-reuse",query_count=count,method=method,
                              timing=timing[method],metrics=metrics))
            _emit(ctx,rows,dict(experiment_id=f"HR-E02/{parent}/Q{count}/{method}",mechanism_id="HR-E02",metrics=metrics,
                config=dict(parent_id=parent,grid=geometry.grid[0],track=track,horizon=h,query_times=times.tolist(),method=method,
                    role="Ours numerical prototype" if method=="reusable_quadratic" else "Theirs" if method=="classical_dense" else "Analytic control",
                    kappa=equation.kappa,reaction_rate=equation.reaction_rate,domain=list(geometry.lengths),
                    quad_nodes=4,output_compression="none",
                    encoding_precision="float64; CPU fit and device transfers charged",inference_precision=str(u.dtype),
                    cohort="fixed training-bank parents; exploratory",ancestor="D07 CPU cost failure retained")),
                [check("finite",endpoint["finite"],True,"eq",category="correctness"),
                 check("accepted-bank-reference",True if endpoint["reference_accepted"] else None,True,"eq",category="math"),
                 check("all-requested-query-accuracy",all_query_accuracy,True,"eq",reason="Off-grid bank references absent for Q>1; endpoint alone cannot certify a trajectory")])
        group={name:build[name]+timing[name]["median_seconds"] for name in calls}
        ours_error=_errors(answers["reusable_quadratic"][-1],ref,metadata)
        competitor_errors={name:_errors(answers[name][-1],ref,metadata) for name in ("direct_gl4","classical_dense")}
        eligible=[group[name] for name,error in competitor_errors.items() if _qualified(error,options["target"])]
        margin=min(eligible)/group["reusable_quadratic"] if eligible else None
        decisions.append(dict(parent_id=parent,query_count=count,speedup_over_best_endpoint_qualified=margin,
            endpoint_qualified=_qualified(ours_error,options["target"]),all_query_accuracy=(count==1),
            reference_accepted=ours_error["reference_accepted"],finite=ours_error["finite"],
            complete_seconds=group["reusable_quadratic"],comparison_seconds=group))
    # Refresh cannot be free during autonomous use. This single two-step call
    # is a separate cost diagnostic, not another independent field observation.
    def refreshed():
        first=encode(interval=h/2).query(h/2)
        return encode(first,h/2).query(h/2)
    def direct_rollout():return direct_step(direct_step(u,h/2,equation,geometry,track),h/2,equation,geometry,track)
    refresh_answers,refresh_times=paired_calls({"refreshed_encoding":refreshed,"direct_gl4_two_step":direct_rollout},
        device,repeats=1,rng=options["rng"],check_budget=options["check"])
    for method,answer in refresh_answers.items():
        costs.append(dict(parent_id=parent,variant="autonomous_refresh",method=method,query_count=2,
            refresh_count=2 if method=="refreshed_encoding" else 0,timing=refresh_times[method],
            metrics=dict(_errors(answer,ref,metadata),complete_seconds=refresh_times[method]["median_seconds"])))
    return decisions


def _entry_value(entry,key,default=None):
    return entry.get(key,entry.get("parent",{}).get(key,default))


def temporal_group_decision(count,group,parent_denominator,required_speedup):
    """Endpoint-only cost observations cannot decide an unverified query task.

    In particular, an endpoint-qualified comparator may be inaccurate at an
    interior time. Its faster query group cannot reject an all-query claim
    whose accuracy qualification is still missing for both methods.
    """
    margin_available=(len(group)==parent_denominator and
                      all(r["speedup_over_best_endpoint_qualified"] is not None for r in group))
    cost_failed=margin_available and any(r["speedup_over_best_endpoint_qualified"]<required_speedup for r in group)
    endpoint_failed=any(not r["finite"] or (r["reference_accepted"] and not r["endpoint_qualified"]) for r in group)
    endpoint_outcome=("BAD" if endpoint_failed or cost_failed else "NA" if not margin_available else
                      "GOOD" if all(r["endpoint_qualified"] for r in group) else "NA")
    return dict(query_count=count,outcome="NA" if count>1 else endpoint_outcome,
        endpoint_only_outcome=endpoint_outcome,endpoint_accuracy_failure=endpoint_failed,
        descriptive_cost_margin_failed=cost_failed,parents_measured=len(group),parents_planned=parent_denominator,
        reason=("All-query accuracy missing for both candidate and controls; endpoint-only timing is descriptive"
                if count>1 else "Endpoint accuracy or finiteness failed" if endpoint_failed else
                "Endpoint-qualified cost margin failed" if cost_failed else "Endpoint qualified cost check"),
        minimum_speedup=min((r["speedup_over_best_endpoint_qualified"] for r in group
                             if r["speedup_over_best_endpoint_qualified"] is not None),default=None))


@torch.no_grad()
def run(ctx):
    from .resolution_diagnostics import iterate_bank_entries,load_bank_entry
    settings=ctx.protocol["resolution"];grid=int(ctx.unit["grid"]);track=ctx.unit["track"]
    options=dict(parent_limit=4,query_counts=[1,4,8],repeats=2,temporal_degree=6,selector_budget_fraction=.1,
        selector_rms_ratio_limit=1.05,required_speedup=1.10,relative_error_floor_multiple=5.,
        temporal_rtol=1e-7,temporal_atol=1e-9,maximum_wall_fraction=.90)
    options.update(settings.get("exploration",{}))
    if not options["query_counts"] or max(options["query_counts"])>8 or min(options["query_counts"])<1:
        raise ValueError("Bounded resolution exploration permits only one through eight queries")
    options["target"]=2e-5 if 2e-5 in settings["tolerances"] else min(settings["tolerances"])
    options["horizon"]=max(settings["training_horizons"])
    options["rng"]=np.random.default_rng(settings["seed"]+grid+(0 if track=="discrete" else 1000))
    started=time.perf_counter();limit=float(ctx.unit["seconds"])*options["maximum_wall_fraction"]
    def budget():
        ctx.budget.check()
        if time.perf_counter()-started>=limit:
            raise TimeoutError("Bounded resolution exploration diagnostic slice reached its frozen limit")
    options["check"]=budget
    bank_dirs=[ctx.prerequisites[name] for name in ctx.unit["bank_units"]]
    entries=[entry for bank in bank_dirs for entry in iterate_bank_entries(bank)]
    selected=[]
    for entry in entries:
        parent=_entry_value(entry,"parent_id")
        if parent and parent not in selected:selected.append(parent)
    selected=sorted(selected)[:options["parent_limit"]]
    if not selected:raise ValueError("No verified development parents for exploration")
    rows=[];costs=[];selection=[];temporal=[];interruption=None
    write_json(ctx.path/"resolution_exploration_scope.json",dict(grid=grid,track=track,parent_ids=selected,
        query_counts=options["query_counts"],reference_banks=[str(p) for p in bank_dirs],
        field_role="Previously prepared training parents, exploratory development only",
        references_regenerated=False,interior_query_references_available=False,
        frozen_options={k:v for k,v in options.items() if k not in ("rng","check")},
        protected_allocation="20% of this resolution study's discretionary pilot envelope, independent of historical budgets",
        historical_results="E01 amplitude-only selection and D07 temporal cost failures remain unchanged"))
    try:
        for parent in selected:
            budget()
            initial,reference,metadata=load_bank_entry(bank_dirs,parent,track,grid,options["horizon"])
            physical=metadata.get("parent",metadata)
            equation=Equation(physical["kappa"],physical["reaction_rate"])
            geometry=Geometry((grid,grid),tuple(physical["domain"]))
            u=initial.to(device=ctx.device,dtype=torch.float32)
            selection.append(_selective(ctx,u,reference,metadata,equation,geometry,track,options,parent,rows,costs))
            write_json(ctx.path/"resolution_exploration_costs.json",costs)
            temporal.extend(_temporal(ctx,u,reference,metadata,equation,geometry,track,options,parent,rows,costs))
            write_json(ctx.path/"resolution_exploration_costs.json",costs)
    except TimeoutError as error:
        interruption=str(error)
        # Previously completed parents remain visible. This is a scientific
        # incomplete outcome; the enclosing stage still reports its evidence.
    write_json(ctx.path/"resolution_exploration_costs.json",costs)
    complete=len(selection)==len(selected) and len(temporal)==len(selected)*len(options["query_counts"])
    selector_resolved=complete and all(r["resolved"] for r in selection)
    selector_good=selector_resolved and all(r["rms_ratio"]<=options["selector_rms_ratio_limit"] and
        r["qualified"] and r["frontier_speedup"] is not None and r["frontier_speedup"]>=options["required_speedup"] for r in selection)
    selector_outcome=("BAD" if any(not r["finite"] for r in selection) else
                      "NA" if not selector_resolved else "GOOD" if selector_good else "BAD")
    query_groups=[]
    for count in options["query_counts"]:
        group=[r for r in temporal if r["query_count"]==count]
        query_groups.append(temporal_group_decision(count,group,len(selected),options["required_speedup"]))
    decisions=dict(status="COMPLETED",evidence_status="COMPLETE_BOUNDED_DEVELOPMENT" if complete else "INCOMPLETE_BOUNDED_DEVELOPMENT",
        interruption=interruption,grid=grid,track=track,device=str(ctx.device),parent_denominator=len(selected),
        selective_cubic=dict(outcome=selector_outcome,parents=selection,advance=selector_good),
        temporal=dict(query_groups=query_groups,parents=temporal,advance=False,
            reason="No temporal expansion authorized without independently accepted interior references and favorable complete costs"),
        elapsed_seconds=time.perf_counter()-started,energy_joules=None,monetary_cost=None,
        hardware_claim="Actual current device only; no claim of native GPU execution from CPU tests")
    write_json(ctx.path/"resolution_exploration_decisions.json",decisions)
    return decisions
