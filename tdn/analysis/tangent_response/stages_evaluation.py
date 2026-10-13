"""Stage adapters for selected-checkpoint evaluation and gated full utility.

All auxiliary menus originate in the frozen protocol. Additional teachers are
independent CPU acquisitions, charged separately from complete inference.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from .contracts import digest, clean
from . import evaluation as ev
from . import performance as perf
from .gates import learning_decision, require_promotion


def _check(ctx):
    if getattr(ctx,"budget",None) is not None:ctx.budget.check()


def _save(ctx,name,value):
    destination=Path(ctx.path)/name
    destination.write_text(json.dumps(clean(value),sort_keys=True,indent=2,allow_nan=False)+"\n")


def _descriptors(methods,trials):
    rows=[t.get("result",t) for t in trials]
    output={}
    for name,value in methods.items():
        descriptor=dict(value) if isinstance(value,dict) else dict(call=value)
        function=descriptor["call"]
        metadata=dict(getattr(function,"metadata",{}))
        family=descriptor.get("family",getattr(descriptor.get("model"),"family",name.split("-seed-")[0]))
        matches=[t for t in rows if t.get("family")==family]
        seed=descriptor.get("seed")
        if seed is None and len(matches)==1:seed=matches[0].get("seed")
        trial=next((t for t in matches if t.get("seed")==seed),{})
        descriptor.update(family=family,seed=seed,checkpoint_sha256=descriptor.get("checkpoint_sha256",trial.get("selected_checkpoint_sha256")),
                          checkpoint_selected=bool(trial),metadata=metadata,trial=trial)
        output[name]=descriptor
    return output


def _prepared(samples,p):
    oversampling=int(p.get("evaluation",{}).get("maximum_oversampling",2))
    return [dict(s,record=dict(s["record"],maximum_oversampling=oversampling)) for s in samples]


def _primary(sample,p):
    r=sample["record"]
    return r.get("panel","primary")=="primary" and r.get("grid")==p.get("fit_geometry",{}).get("grid",r.get("base_grid",r.get("grid")))


def _endpoints(ctx,samples,descriptors):
    rows=[];comparisons=[]
    for name,descriptor in descriptors.items():
        function=descriptor["call"]
        for sample in samples:
            record=sample["record"]
            if record["split"]=="train":continue
            row=dict(ev._identity(record),method=name,family=descriptor["family"],seed=descriptor["seed"],
                     panel=record.get("panel","primary"),status="UNRESOLVED_REFERENCE",
                     reference_accepted=ev.reference_status(sample)[0])
            if ev.reference_status(sample)[0]:
                try:
                    _check(ctx)
                    _,eq,geom,target=ev.sample_context(sample)
                    with torch.no_grad():
                        output=function(sample["u0"],float(record["horizon"]),eq,geom,target)
                    parts=getattr(function,"last_parts",{})
                    row.update(ev.endpoint_metrics(output,sample),status="MEASURED" if bool(torch.isfinite(output).all()) else "FAILED_NONFINITE",
                               work=dict(getattr(function,"last_work",{})),output_dtype=str(output.dtype))
                    row.update(ev.final_rounding_metrics(output,parts,geom,target))
                    physical=parts.get("physical_prediction",output)
                    physical_metrics=ev.endpoint_metrics(physical,sample)
                    row["physical_prediction_error"]=physical_metrics
                    row["physical_prediction_available"]=parts.get("physical_prediction") is not None
                    if record["split"]=="diagnostic" and _primary(sample,ctx.protocol) and descriptor["family"] in ctx.protocol["training"]["arms"]:
                        g=sample.get("mean_endpoint64");u=sample.get("oracle_endpoint64")
                        om=sample.get("oracle_metadata",{})
                        uncertainty_values=(om.get("matched_mean_uncertainty_rms"),om.get("uncertainty_rms"))
                        uncertainty=sum(uncertainty_values) if all(isinstance(v,(int,float)) and v>=0 for v in uncertainty_values) else None
                        comparison=dict(row,selected_checkpoint=descriptor["checkpoint_selected"],checkpoint_sha256=descriptor["checkpoint_sha256"],
                            E_G=ev.endpoint_metrics(g,sample)["rms"] if g is not None else None,
                            E_U=ev.endpoint_metrics(u,sample)["rms"] if u is not None else None,
                            E_L=physical_metrics["rms"] if row["physical_prediction_available"] else None,
                            uncertainty_combined=uncertainty,physical_gain_scope="same smooth anchor; arithmetic/FP32 deployment error recorded separately")
                        comparisons.append(comparison)
                except (RuntimeError,ValueError,FloatingPointError,OverflowError,TimeoutError,InterruptedError) as exc:
                    row.update(status="TIME_CENSORED" if isinstance(exc,(TimeoutError,InterruptedError)) else "UNRESOLVED" if "UNRESOLVED" in str(exc) or "UNAVAILABLE" in str(exc) else "FAILED",reason=str(exc),
                               work=dict(getattr(function,"last_work",{})))
            rows.append(row)
    summaries={}
    for name,split,target,panel in sorted({(r["method"],r["split"],r["target_kind"],r["panel"]) for r in rows}):
        chosen=[r for r in rows if (r["method"],r["split"],r["target_kind"],r["panel"])==(name,split,target,panel)]
        summaries[f"{name}/{split}/{target}/{panel}"]=ev.parent_summary(chosen)
    return rows,comparisons,summaries


def _auxiliary(ctx,samples,descriptors):
    from .oracle import refined_teacher
    from .tangent import AnchorPath,response_action
    cfg=ctx.protocol.get("evaluation",{})
    count=int(cfg.get("auxiliary_parents",0))
    horizon=ctx.protocol.get("horizons",{}).get("diagnostic",[.075])
    chosen_horizon=horizon[min(1,len(horizon)-1)]
    selected=[s for s in samples if s["record"]["split"]=="diagnostic" and _primary(s,ctx.protocol)
              and abs(s["record"]["horizon"]-chosen_horizon)<1e-13]
    selected=sorted(selected,key=lambda s:(s["record"].get("slot",0),s["record"]["parent_id"]))[:count]
    cache={};acquisition=[]
    def provider(original,at):
        _check(ctx)
        r,eq,geom,target=ev.sample_context(original)
        key=digest(dict(input_sha256=r.get("input_sha256"),parent_id=r["parent_id"],grid=r["grid"],target=target,
                        kappa=r["kappa"],reaction_rate=r["reaction_rate"],physical_time=float(at)))
        if key not in cache:
            tick=time.perf_counter()
            value,reference=refined_teacher(original["u0"].detach().cpu(),at,eq,geom,target,
                config={**ctx.protocol["reference"],"max_seconds_per_query":ctx.protocol["reference"]["seconds_per_query"]},budget=ctx.budget)
            cache[key]=dict(teacher64=value.to(original["u0"].device),reference=reference,reference_accepted=reference.get("accepted",False))
            acquisition.append(dict(parent_id=r["parent_id"],physical_time=at,grid=r["grid"],teacher=reference,seconds=time.perf_counter()-tick,
                                    scope="CPU independent original-input teacher; never an inference feature",cache_identity=key))
        return cache[key]
    rollouts=[];composition=[];sensitivity=[];tangent=[]
    for sample in selected:
        r,eq,geom,target=ev.sample_context(sample)
        for name,descriptor in descriptors.items():
            raw_function=descriptor["call"]
            def function(*args,raw_function=raw_function):
                _check(ctx)
                return raw_function(*args)
            for steps in cfg.get("rollout_steps",[2,4,8]):
                try:
                    result=ev.evaluate_rollout(function,sample,int(steps),provider,method=name,
                                              bounded_fd=ctx.protocol.get("mechanism",{}).get("bounded_inputs",False))
                    rollouts.append(dict(method=name,parent_id=r["parent_id"],query_id=r["query_id"],**result))
                except (RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                    rollouts.append(dict(method=name,parent_id=r["parent_id"],query_id=r["query_id"],requested_steps=steps,
                                         measured_steps=0,status="UNRESOLVED",reason=str(exc)))
            try:
                composition.append(dict(method=name,parent_id=r["parent_id"],query_id=r["query_id"],
                    **ev.nonlinear_endpoint_composition(function,sample["u0"],r["horizon"],eq,geom,target)))
            except (RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                composition.append(dict(method=name,parent_id=r["parent_id"],status="UNRESOLVED",reason=str(exc)))
            if descriptor["family"] in ctx.protocol["training"]["arms"]:
                direction=torch.ones_like(sample["u0"])
                try:
                    sensitivity.append(dict(method=name,parent_id=r["parent_id"],query_id=r["query_id"],
                        **ev.sensitivity_panel(lambda u:function(u,r["horizon"],eq,geom,target),sample["u0"],direction,geom,target,
                                               epsilons=cfg.get("sensitivity_epsilons",[1e-4,1e-5]))))
                except (RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                    sensitivity.append(dict(method=name,parent_id=r["parent_id"],status="UNRESOLVED",reason=str(exc)))
        try:
            _check(ctx)
            path=AnchorPath(sample["u0"].double(),eq,geom,target,reaction_substeps=ctx.protocol["anchor"]["reaction_substeps"],budget=ctx.budget)
            action=lambda z,s,h:response_action(path,eq,geom,target,z,s,h,steps=ctx.protocol["mechanism"]["initial_steps"],budget=ctx.budget)
            probe=torch.ones_like(sample["u0"],dtype=torch.float64)
            tangent.append(dict(parent_id=r["parent_id"],query_id=r["query_id"],
                **ev.same_path_tangent_composition(action,probe,0.,r["horizon"]/2,r["horizon"],geom,target,
                                                   path_identity=digest({"record":r,"anchor":ctx.protocol["anchor"]}))))
        except (RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
            tangent.append(dict(parent_id=r["parent_id"],status="UNRESOLVED",reason=str(exc)))
    return dict(rollouts=rollouts,endpoint_composition=composition,sensitivity=sensitivity,same_path_tangent_composition=tangent,
                teacher_acquisition=acquisition,requested_auxiliary_parents=count,observed_auxiliary_parents=len(selected),
                selection="first declared diagnostic slots at the protocol's middle horizon; no accuracy/speed selection",
                paired_resolution_scope="fixed physics and grid-matched stiffness recorded separately in query plan")


def run_evaluation(ctx,samples,methods,training_trials):
    p=ctx.protocol
    original=list(samples)
    ev.assert_parent_disjoint(original)
    descriptors=_descriptors(methods,training_trials)
    rows,comparisons,summaries=_endpoints(ctx,_prepared(original,p),descriptors)
    plan=[copy.deepcopy(s["record"]) for s in original]
    try:
        auxiliary=_auxiliary(ctx,_prepared(original,p),descriptors)
    except (TimeoutError,InterruptedError) as exc:
        auxiliary=dict(status="TIME_CENSORED",reason=str(exc),requested_auxiliary_parents=p.get("evaluation",{}).get("auxiliary_parents",0),
                       observed_auxiliary_parents=0,scope="Remaining auxiliary measurements unavailable; endpoint evidence retained")
    result=dict(schema="tdn.tangent-response-evaluation/v1",rows=rows,learning_comparisons=comparisons,summaries=summaries,
                screened_query_plan=plan,screened_manifest_sha256=digest(plan),
                requested_train_query_ids=[r["query_id"] for r in plan if r["split"]=="train"],
                requested_primary_diagnostic_query_ids=[s["record"]["query_id"] for s in original if s["record"]["split"]=="diagnostic" and _primary(s,p)],
                requested_validation_query_ids=[r["query_id"] for r in plan if r["split"]=="validation"],
                auxiliary=auxiliary,
                method_metadata={name:{k:d[k] for k in ("family","seed","checkpoint_sha256","metadata")} for name,d in descriptors.items()},
                source_tree_sha256=ctx.sourcehash,evidence_scope="DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS")
    result["learning_decision"]=learning_decision([t.get("result",t) for t in training_trials],result,p)
    _save(ctx,"evaluation_results.json",result)
    return dict(evaluated_rows=len(rows),requested_primary_diagnostic_queries=len(result["requested_primary_diagnostic_query_ids"]),
                scientific_outcome=result["learning_decision"]["status"],auxiliary_requested_parents=result["auxiliary"]["requested_auxiliary_parents"])


def _learning_receipt(ctx):
    for value in getattr(ctx,"prerequisites",{}).values():
        path=Path(value) if isinstance(value,(str,Path)) else None
        if path is not None and (path/"decision.json").is_file():
            candidate=json.loads((path/"decision.json").read_text())
            if candidate.get("gate")=="C":return candidate
        if path is not None and (path/"evaluation_results.json").is_file():
            return json.loads((path/"evaluation_results.json").read_text()).get("learning_decision",{})
    return {}


def run_utility(ctx,samples,methods,training_trials):
    p=ctx.protocol;cfg=dict(p.get("performance",{}));cpu=p.get("profile")=="cpu-smoke"
    decision=_learning_receipt(ctx)
    if not cpu:require_promotion(decision,"C")
    descriptors=_descriptors(methods,training_trials)
    deployable={name:dict(d["metadata"],call=d["call"],method=name,schedule="frozen") for name,d in descriptors.items()
                if d["metadata"].get("deployable",False)}
    controls=perf.classical_controls(schedules=cfg.get("classical_schedules",[1,2] if cpu else [1,2,4,8]),
                                     adaptive_tolerances=cfg.get("adaptive_tolerances",[1e-8] if cpu else [1e-8,1e-10]),
                                     reaction_substeps=p["anchor"]["reaction_substeps"])
    all_methods={name:dict(method,_budget=ctx.budget) for name,method in {**deployable,**controls}.items()}
    samples=_prepared(list(samples),p)
    validation=[s for s in samples if s["record"]["split"]=="validation" and _primary(s,p)]
    diagnostic=[s for s in samples if s["record"]["split"]=="diagnostic" and _primary(s,p)]
    validation_result=perf.evaluate_schedules(validation,all_methods,repeats=cfg.get("repeats",5),warmup=cfg.get("warmup",1),
                                             seed=p.get("training",{}).get("seeds",[0])[0],device=ctx.device)
    frozen=perf.freeze_schedules(validation_result["rows"],tolerances=cfg.get("tolerances",perf.TOLERANCES),
                                  source_identity=dict(source_tree_sha256=ctx.sourcehash,protocol_sha256=digest(p),
                                                       checkpoint_identities={name:d["checkpoint_sha256"] for name,d in descriptors.items() if d["checkpoint_sha256"]}))
    _save(ctx,"frozen_schedules.json",frozen)
    # Freeze is saved before the first diagnostic accuracy/timing observation.
    diagnostic_result=perf.evaluate_schedules(diagnostic,all_methods,repeats=cfg.get("repeats",5),warmup=cfg.get("warmup",1),
                                             seed=p.get("training",{}).get("seeds",[0])[0],device=ctx.device)
    candidates=[name for name,d in descriptors.items() if d["family"]=="learned_response" and name in deployable]
    competitors=sorted({v["method"] for v in controls.values()}|{name for name,d in descriptors.items()
                      if name in deployable and d["family"]!="learned_response"})
    comparisons={name:perf.utility_comparison(diagnostic_result["rows"],frozen,tolerance=cfg.get("primary_tolerance",2e-5),
        candidate=name,controls=competitors) for name in candidates}
    work_precision={str(tolerance):{name:perf.utility_comparison(diagnostic_result["rows"],frozen,tolerance=tolerance,
        candidate=name,controls=competitors) for name in candidates} for tolerance in cfg.get("tolerances",perf.TOLERANCES)}
    transfer=perf.benchmark_calls(all_methods,diagnostic,repeats=cfg.get("repeats",5),warmup=cfg.get("warmup",1),device=ctx.device,scope="full_transfer")
    caches={}
    for name,d in descriptors.items():
        model=d.get("model")
        if diagnostic and callable(getattr(model,"encode",None)) and callable(getattr(model,"decode",None)):
            try:
                _check(ctx)
                caches[name]=perf.benchmark_cache(model,diagnostic[0],repeats=cfg.get("repeats",5),device=ctx.device)
            except (RuntimeError,ValueError,TimeoutError,InterruptedError) as exc:
                caches[name]=dict(status="UNRESOLVED",reason=str(exc))
    checkpoint_ids={name:d["checkpoint_sha256"] for name,d in descriptors.items() if d["checkpoint_sha256"]}
    readiness=getattr(ctx,"native_readiness",None) or cfg.get("native_readiness",{})
    combined=dict(outcomes=[dict(row,candidate=name,seed=descriptors[name]["seed"]) for name,value in comparisons.items() for row in value["outcomes"]],
                  frozen_sha256=frozen["sha256"],candidate=candidates[0],controls=competitors,
                  declared_candidates=[dict(method=name,seed=descriptors[name]["seed"],checkpoint_sha256=descriptors[name]["checkpoint_sha256"]) for name in candidates],
                  requested_targets=sorted({s["record"]["target_kind"] for s in diagnostic}),
                  policy_scope="Every selected seed evaluated; no favorable seed selection") if candidates else {}
    fresh={}
    repo=Path(__file__).resolve().parents[3]
    trusted_script=repo/"scripts"/"tangent_response_infer.py"
    if diagnostic:
        record={k:diagnostic[0]["record"][k] for k in ("parent_id","query_id","target_kind","grid","lengths","horizon","kappa","reaction_rate")}
        input_path=Path(ctx.path)/"fresh-process-input.pt"
        torch.save(dict(u0=diagnostic[0]["u0"].detach().cpu(),record=record),input_path)
        input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
        trial_paths={(t.get("result",t).get("family"),t.get("result",t).get("seed")):Path(t["path"]) for t in training_trials if "path" in t}
        for name,d in descriptors.items():
            path=trial_paths.get((d["family"],d["seed"]))
            if path is None:continue
            try:
                _check(ctx)
                checkpoint=path/"selected.pt"
                if not trusted_script.is_file() or not checkpoint.is_file():
                    fresh[name]=dict(status="UNRESOLVED",reason="Trusted fresh-process runner or selected checkpoint unavailable")
                    continue
                actual_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if actual_sha!=d["checkpoint_sha256"]:
                    fresh[name]=dict(status="UNRESOLVED",reason="Selected checkpoint identity changed")
                    continue
                remaining=float(cfg.get("fresh_process_timeout",60))
                if hasattr(ctx.budget,"seconds") and hasattr(ctx.budget,"start"):
                    remaining=min(remaining,ctx.budget.seconds-(time.monotonic()-ctx.budget.start))
                if remaining<1:raise TimeoutError("Stage budget exhausted before fresh-process measurement")
                fresh[name]=perf.benchmark_fresh_process([sys.executable,str(trusted_script),"--checkpoint",str(checkpoint),
                    "--input",str(input_path),"--device",str(ctx.device),"--source-sha256",ctx.sourcehash],cwd=repo,repeats=int(cfg.get("fresh_process_repeats",1 if cpu else 2)),timeout=remaining,
                    identity=dict(checkpoint_sha256=actual_sha,input_sha256=input_sha,source_tree_sha256=ctx.sourcehash),budget=ctx.budget)
            except (TimeoutError,InterruptedError) as exc:
                fresh[name]=dict(status="TIME_CENSORED",reason=str(exc))
    result=dict(schema="tdn.tangent-response-performance/v1",status="IMPLEMENTATION_ONLY" if cpu else "MEASURED",
                validation=validation_result,diagnostic=diagnostic_result,frozen_schedules=frozen,
                utility=combined,all_seed_utilities=comparisons,work_precision=work_precision,
                full_transfer=transfer,cache=caches,fresh_process=fresh,
                fresh_process_status="MEASURED_SEPARATELY" if any(value.get("summary",{}).get("observations",0)>0 for value in fresh.values()) else "UNRESOLVED_OR_UNAVAILABLE",
                native_readiness=readiness,checkpoint_identities=checkpoint_ids,source_tree_sha256=ctx.sourcehash,
                learning_decision=decision,energy_joules=None,monetary_cost=None,
                method_inventory={name:{k:d[k] for k in ("family","seed","checkpoint_sha256","metadata")} for name,d in descriptors.items()},
                timing_omissions=[dict(method=name,status=d["metadata"].get("status","UNAVAILABLE"),reason=d["metadata"].get("reason","Nondeployable oracle timing excluded from utility"))
                                 for name,d in descriptors.items() if name not in deployable],
                utility_scope="validation-frozen full-call resident latency; transfer/cache scopes separate; no energy/money claim")
    _save(ctx,"performance_results.json",result)
    return dict(measured_timing_rows=len(diagnostic_result["timing"]["rows"]),candidate_models=len(candidates),
                status=result["status"],utility_decision_available=bool(result["utility"]))
