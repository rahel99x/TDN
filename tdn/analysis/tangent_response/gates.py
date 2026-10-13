"""Prospectively registered scientific decisions, distinct from job completion.

Requested identities are checked before eligible cohorts are aggregated. CPU
implementation smoke never promotes native stages. Unknown uncertainties and
missing work remain unresolved rather than receiving zero-error credit.
"""
from __future__ import annotations

from collections import defaultdict
import math

from .contracts import digest

STATUSES = {"PASS","STOP","UNRESOLVED","NOT_APPLICABLE","IMPLEMENTATION_ONLY"}


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0


def _parent_mean(rows,key):
    groups=defaultdict(list)
    for row in rows:
        groups[row["parent_id"]].append(float(key(row)))
    return sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None


def _cpu(p):
    return p.get("profile")=="cpu-smoke" or p.get("execution_scope")=="CPU_IMPLEMENTATION_SMOKE"


def _decision(gate,stage,status,p,*,thresholds=None,evidence=None,denominators=None,uncertainties=None,reason=""):
    raw=status
    if _cpu(p):
        status="IMPLEMENTATION_ONLY"
    result=dict(schema="tdn.tangent-response-decision/v1",gate=gate,decision=status,eligible=status=="PASS",
                stage=stage,status=status,eligible_next=status=="PASS",thresholds=thresholds or {},
                evidence=evidence or {},denominators=denominators or {},uncertainties=uncertainties or {},reason=reason,
                profile=p.get("profile"),execution_scope=p.get("execution_scope"),
                underlying_scientific_decision=raw if _cpu(p) else None,
                evidence_scope="DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS",uncertainty_is_certificate=False)
    result["protocol_sha256"]=digest(p)
    return result


def validate_decision(value,*,gate=None):
    if not isinstance(value,dict) or value.get("status") not in STATUSES:
        raise ValueError("A serialized scientific decision is required")
    if value.get("decision")!=value["status"] or value.get("eligible")!=(value["status"]=="PASS") or value.get("eligible_next")!=value.get("eligible"):
        raise ValueError("Scientific decision aliases disagree")
    if gate is not None and value.get("gate")!=gate:
        raise ValueError("Wrong scientific prerequisite gate")
    if value.get("status")=="PASS" and (value.get("profile")=="cpu-smoke" or value.get("execution_scope")=="CPU_IMPLEMENTATION_SMOKE"):
        raise ValueError("CPU implementation evidence cannot promote native science")
    return value


def require_promotion(value,gate):
    validate_decision(value,gate=gate)
    if value["status"]!="PASS":
        raise ValueError("Downstream numerical expansion requires a literal eligible PASS")
    return value


def _primary_queries(p,*,fit=False,split="diagnostic"):
    plan=p.get("fit_queries" if fit else "queries",[])
    base=p.get("fit_geometry",{}).get("grid") if fit else (p.get("geometry",{}).get("grids") or [None])[0]
    return [q for q in plan if q.get("split")==split and q.get("panel","primary")=="primary"
            and (base is None or q.get("grid")==q.get("base_grid",base))]


def _indexed(rows):
    index={}
    duplicates=[]
    for row in rows:
        key=row.get("query_id")
        if key in index:
            duplicates.append(key)
        index[key]=row
    return index,duplicates


def _screened_fit_queries(p,result,split):
    manifest=result.get("screened_query_plan")
    if manifest is None:
        return _primary_queries(p,fit=True,split=split),True
    if result.get("screened_manifest_sha256")!=digest(manifest):return [],False
    parents={r.get("parent_id") for r in manifest}
    expected=[q for q in p.get("fit_queries",[]) if q.get("parent_id") in parents]
    if {q["query_id"] for q in expected}!={q.get("query_id") for q in manifest}:return [],False
    parent_counts={s:len({q["parent_id"] for q in expected if q["split"]==s}) for s in ("train","validation","diagnostic")}
    required=p.get("fit_parent_counts",{})
    if required and any(parent_counts[s]!=required.get(s) for s in parent_counts):return [],False
    selected={q["query_id"] for q in expected}
    return [q for q in _primary_queries(p,fit=True,split=split) if q["query_id"] in selected],True


def _mechanism_uncertainty(row):
    explicit=row.get("uncertainty_combined")
    if isinstance(explicit,dict):explicit=explicit.get("rms")
    if explicit is not None:
        return float(explicit) if _number(explicit) else None
    controls=row.get("controls",{})
    values=[controls.get(name,{}).get("uncertainty_rms") for name in ("G","U")]
    # These endpoint uncertainty estimates already contain reference, time
    # integration and source/path estimates. Summing them controls a difference
    # of errors conservatively; shared teacher uncertainty is not assumed to cancel.
    return sum(values) if all(_number(v) for v in values) else None


def mechanism_decision(rows,p):
    rows=list(rows.get("rows",[]) if isinstance(rows,dict) else rows)
    limits=dict(p.get("gates",{}).get("A",{}))
    if p.get("target")!="fd_same_grid" and not _cpu(p):
        return _decision("A","mechanism","NOT_APPLICABLE",p,thresholds=limits,
                         evidence={"extension_rows":len(rows)},reason="Initial mechanism promotion is FD only; Galerkin extension is reported separately")
    requested=_primary_queries(p)
    indexed,duplicates=_indexed(rows)
    missing=[q["query_id"] for q in requested if q["query_id"] not in indexed]
    eligible,excluded,unresolved=[],[],[]
    for query in requested:
        row=indexed.get(query["query_id"])
        if row is None:continue
        controls=row.get("controls",{})
        uncertainty=_mechanism_uncertainty(row)
        valid=(row.get("reference",{}).get("accepted") is True and row.get("primary_status")=="RESOLVED"
               and all(_number(controls.get(name,{}).get(k)) for name in ("anchor","G","U") for k in ("rms","maximum"))
               and uncertainty is not None)
        if not valid:
            unresolved.append(query["query_id"]);continue
        signal=controls["anchor"]["rms"]
        if signal < limits.get("minimum_signal_uncertainty_ratio",10.)*uncertainty or signal==0:
            excluded.append(dict(query_id=query["query_id"],parent_id=query["parent_id"],reason="PROSPECTIVE_PRECISION_OR_SIGNAL_LIMIT",anchor_rms=signal,uncertainty=uncertainty));continue
        eligible.append(dict(row,_uncertainty=uncertainty))
    parents=defaultdict(list)
    for row in eligible:parents[row["parent_id"]].append(row)
    separated=[]
    for parent,values in parents.items():
        gain=sum(r["controls"]["G"]["rms"]-r["controls"]["U"]["rms"] for r in values)/len(values)
        uncertainty=sum(r["_uncertainty"] for r in values)/len(values)
        separated.append(dict(parent_id=parent,gain=gain,uncertainty=uncertainty,
                              separated=gain>0 and uncertainty<limits.get("uncertainty_fraction_of_improvement",.1)*gain))
    eg=_parent_mean(eligible,lambda r:r["controls"]["G"]["rms"])
    eu=_parent_mean(eligible,lambda r:r["controls"]["U"]["rms"])
    unc=_parent_mean(eligible,lambda r:r["_uncertainty"])
    ratio=eg/eu if eg is not None and eu is not None and eu>0 else math.inf if eg and eu==0 else None
    fraction=sum(r["separated"] for r in separated)/len(separated) if separated else None
    regressions=[r["query_id"] for r in eligible if r["controls"]["U"]["maximum"]>limits.get("maximum_regression",2.)*r["controls"]["G"]["maximum"]]
    aggregate_separated=bool(eg is not None and eu is not None and eg>eu and unc<limits.get("uncertainty_fraction_of_improvement",.1)*(eg-eu))
    complete=bool(requested) and not missing and not unresolved and not duplicates
    passed=complete and bool(eligible) and ratio is not None and ratio>=limits.get("rms_ratio",1.5) and fraction>=limits.get("parent_improvement_fraction",.75) and not regressions and aggregate_separated
    status="UNRESOLVED" if not complete or not eligible else "PASS" if passed else "STOP"
    evidence=dict(E_G=eg,E_U=eu,rms_ratio=None if ratio is None or not math.isfinite(ratio) else ratio,
                  zero_oracle_error=bool(eu==0 and eg),ratio_definition="ratio of parent-weighted arithmetic mean RMS, not mean per-query ratios",
                  separated_parent_fraction=fraction,parent_rows=separated,maximum_regressions=regressions,
                  missing_queries=missing,unresolved_queries=unresolved,duplicate_queries=duplicates,precision_limited=excluded,
                  nonprimary_rows_reported=len(rows)-len(requested))
    return _decision("A","mechanism",status,p,thresholds=limits,evidence=evidence,
                     denominators=dict(requested_queries=len(requested),eligible_queries=len(eligible),requested_parents=len({q["parent_id"] for q in requested}),eligible_parents=len(parents),precision_limited_queries=len(excluded)),
                     uncertainties=dict(parent_mean_combined_rms=unc,aggregate_improvement_separated=aggregate_separated),
                     reason="Full requested primary bank retained; paired refinements and interventions never rescue primary promotion")


def compression_decision(result,p):
    limits=dict(p.get("gates",{}).get("B",{}))
    plans=result.get("probe_plan",[])
    expected=[r for r in plans if r.get("split")!="train" or r.get("probe_role")=="heldout_direction"]
    declared_ranks=p.get("compression",{}).get("ranks",[2,4,8])
    methods=sorted({(r.get("method"),r.get("requested_rank")) for r in result.get("method_rows",[]) if r.get("status")=="FITTED"}
                   |{(r.get("method"),r.get("requested_rank")) for r in result.get("action_rows",[])})
    candidates=[]
    for method,rank in methods:
        if not isinstance(rank,int) or rank not in declared_ranks or rank>limits.get("rank_cap",8):continue
        action_rows=[r for r in result.get("action_rows",[]) if r.get("method")==method and r.get("requested_rank")==rank]
        method_rows=[r for r in result.get("method_rows",[]) if r.get("method")==method and r.get("requested_rank")==rank]
        endpoint_rows=[r for r in result.get("endpoint_rows",[]) if r.get("method")==method and r.get("requested_rank")==rank]
        objects={r.get("compression_object") for r in action_rows+method_rows+endpoint_rows if r.get("compression_object")}
        if "task_residual_family" in objects or str(method).startswith("task_residual_"):
            # This method fits a declared residual family. Applying unrelated
            # independent directions remains a diagnostic panel, not evidence
            # that this narrower task representation failed an operator claim.
            candidates.append(dict(method=method,rank=rank,status="NOT_APPLICABLE",passed=False,
                                   promotion_eligible=False,compression_object="task_residual_family",
                                   reported_action_rows=len(action_rows),
                                   reported_residual_action_rows=sum(r.get("probe_kind")=="residual" for r in action_rows),
                                   reported_endpoint_rows=len(endpoint_rows),
                                   diagnostic_panels=dict(actions="action_rows",endpoints="endpoint_rows"),
                                   generalization_scope="Residual-family diagnostics only; reusable-operator promotion requires independent held-out directions and parents"))
            continue
        index={r.get("probe_id"):r for r in action_rows}
        duplicate=len(index)!=len(action_rows)
        missing=[r["probe_id"] for r in expected if r["probe_id"] not in index]
        nontrivial=[];unresolved=[];nulls=[]
        for plan in expected:
            row=index.get(plan["probe_id"])
            if row is None:continue
            if row.get("status")!="RESOLVED" or not _number(row.get("uncertainty")):
                unresolved.append(plan["probe_id"])
            elif row.get("nontrivial") is not True:
                nulls.append(plan["probe_id"])
            elif not _number(row.get("relative_error")):
                unresolved.append(plan["probe_id"])
            else:nontrivial.append(row)
        scopes={"heldout_parent" if r.get("split")!="train" else "heldout_direction" for r in nontrivial}
        fractions={}
        for scope in ("heldout_direction","heldout_parent"):
            values=[r for r in nontrivial if ("heldout_parent" if r.get("split")!="train" else "heldout_direction")==scope]
            fractions[scope]=sum(r["relative_error"]<=limits.get("action_relative_error",.1) for r in values)/len(values) if values else None
        worst=max((r["relative_error"] for r in nontrivial),default=None)
        endpoints=[r for r in result.get("endpoint_rows",[]) if r.get("method")==method and r.get("requested_rank")==rank and r.get("split")=="diagnostic"]
        requested_endpoints=_primary_queries(p)
        endpoint_index,endpoint_duplicates=_indexed(endpoints)
        endpoint_missing=[q["query_id"] for q in requested_endpoints if q["query_id"] not in endpoint_index]
        eligible_endpoints=[];endpoint_unresolved=[]
        for query in requested_endpoints:
            row=endpoint_index.get(query["query_id"])
            if row is None:continue
            if row.get("status")!="RESOLVED" or not all(_number(row.get(k,{}).get("rms")) for k in ("mean_error","oracle_error","compressed_error")):
                endpoint_unresolved.append(query["query_id"]);continue
            # Action and quadrature uncertainty alone is insufficient: require
            # separately supplied teacher/path estimates for a gain claim.
            uncertainty=row.get("combined_uncertainty",row.get("uncertainty_combined"))
            if uncertainty is None:
                extra=row.get("teacher_path_uncertainty")
                uncertainty=(row["uncertainty"]+extra if _number(row.get("uncertainty")) and _number(extra) else None)
            if not _number(uncertainty):
                endpoint_unresolved.append(query["query_id"]);continue
            # Retained improvement is a ratio of aggregate errors. Do not
            # remove an unfavorable horizon using its individual oracle gain.
            eligible_endpoints.append(dict(row,_uncertainty=uncertainty))
        eg=_parent_mean(eligible_endpoints,lambda r:r["mean_error"]["rms"])
        eu=_parent_mean(eligible_endpoints,lambda r:r["oracle_error"]["rms"])
        el=_parent_mean(eligible_endpoints,lambda r:r["compressed_error"]["rms"])
        unc=_parent_mean(eligible_endpoints,lambda r:r["_uncertainty"])
        retained=(eg-el)/(eg-eu) if eg is not None and eg>eu else None
        separated=bool(retained is not None and unc<.1*(eg-eu))
        complete=bool(expected) and not missing and not unresolved and not duplicate and not endpoint_missing and not endpoint_unresolved and not endpoint_duplicates
        pass_actions=(scopes=={"heldout_direction","heldout_parent"} and all(v is not None and v>=limits.get("probe_fraction",.9) for v in fractions.values()) and worst<=limits.get("worst_action_relative_error",.3))
        passed=complete and pass_actions and retained is not None and retained>=limits.get("retained_oracle_gain",.8) and separated
        candidates.append(dict(method=method,rank=rank,status="PASS" if passed else "STOP" if complete and nontrivial and eligible_endpoints and separated else "UNRESOLVED",
                               passed=passed,promotion_eligible=True,compression_object=next(iter(objects)) if len(objects)==1 else "reusable_operator",
                               probe_fractions=fractions,worst_relative_error=worst,retained_gain=retained,E_G=eg,E_U=eu,E_L=el,
                               requested_probes=len(expected),nontrivial_probes=len(nontrivial),null_probes=len(nulls),
                               missing_probes=missing,unresolved_probes=unresolved,missing_endpoints=endpoint_missing,unresolved_endpoints=endpoint_unresolved,
                               deployable=all(r.get("deployable",False) for r in action_rows),
                               generalization_scope="state oracle is representation headroom, not deployable held-out-parent generalization" if any(not r.get("deployable",False) for r in action_rows) else "training-only fitted reusable factors",
                               combined_uncertainty=unc))
    passed=sorted((r for r in candidates if r["passed"]),key=lambda r:(r["rank"],r["method"]))
    chosen=passed[0] if passed else None
    promotion_candidates=[r for r in candidates if r["promotion_eligible"]]
    status=("PASS" if chosen else "NOT_APPLICABLE" if candidates and not promotion_candidates else
            "UNRESOLVED" if not promotion_candidates or any(c["status"]=="UNRESOLVED" for c in promotion_candidates) else "STOP")
    return _decision("B","compression",status,p,thresholds=limits,evidence=dict(candidates=candidates,selected=chosen),
                     denominators=dict(requested_probes=len(expected),requested_primary_endpoints=len(_primary_queries(p))),
                     reason="Lowest passing reusable-operator rank/method; residual-family panels are retained separately without operator promotion")


def learning_decision(trial_results,evaluation_result,p):
    trials=trial_results.get("trials",[]) if isinstance(trial_results,dict) else list(trial_results)
    c1=dict(p.get("gates",{}).get("C1",{}));c2=dict(p.get("gates",{}).get("C2",{}))
    candidate=p.get("training",{}).get("candidate","learned_response")
    seeds=p.get("training",{}).get("seeds",[])
    wanted=[(candidate,seed) for seed in seeds]
    indexed={(r.get("family"),r.get("seed")):r for r in trials}
    duplicates=len(indexed)!=len(trials)
    c1rows=[];unresolved=[]
    for family,seed in wanted:
        trial=indexed.get((family,seed))
        if trial is None:
            unresolved.append(dict(family=family,seed=seed,reason="Missing declared candidate trial"));continue
        complete=trial.get("status")=="COMPLETED_UPDATES" and trial.get("updates")==trial.get("requested_updates")==p.get("training",{}).get("updates")
        fit=trial.get("criterion_train",{})
        rows=fit.get("rows",[])
        expected=trial.get("requested_train_query_ids",evaluation_result.get("requested_train_query_ids"))
        ids=[r.get("query_id") for r in rows]
        if expected is not None and set(ids)!=set(expected):complete=False
        if len(ids)!=len(set(ids)) or not rows:complete=False
        invalid=[r.get("query_id") for r in rows if not r.get("finite") or not r.get("reference_accepted")]
        eligible=[r for r in rows if _number(r.get("relative_oracle_correction_error"))]
        passes=sum(r["relative_oracle_correction_error"]<=c1.get("relative_oracle_correction_error",.05) for r in eligible)
        fraction=passes/len(eligible) if eligible else None
        passed=complete and not invalid and bool(eligible) and fraction>=c1.get("query_fraction",.9)
        c1rows.append(dict(family=family,seed=seed,completed_work=complete,requested_queries=len(rows),eligible_queries=len(eligible),
                           precision_limited_queries=len(rows)-len(eligible),passed_queries=passes,fraction=fraction,passed=passed,
                           criterion_update=trial.get("criterion_update"),selected_update=trial.get("selected_update"),invalid_queries=invalid))
        if not complete or invalid or not eligible:unresolved.append(dict(family=family,seed=seed,reason="Candidate work, labels, finite outputs or declared cohort incomplete"))
    baseline=[dict(family=t.get("family"),seed=t.get("seed"),status=t.get("status"),updates=t.get("updates"),requested_updates=t.get("requested_updates")) for t in trials if t.get("family")!=candidate]
    comparisons=evaluation_result.get("learning_comparisons",evaluation_result.get("comparison_rows",[]))
    by_seed=[]
    requested,manifest_valid=_screened_fit_queries(p,evaluation_result,"diagnostic")
    for seed in seeds:
        rows=[r for r in comparisons if r.get("family",r.get("method"))==candidate and r.get("seed")==seed]
        index,dups=_indexed(rows)
        missing=[q["query_id"] for q in requested if q["query_id"] not in index]
        valid=[];bad=[]
        for q in requested:
            row=index.get(q["query_id"])
            if row is None:continue
            expected_checkpoint=indexed.get((candidate,seed),{}).get("selected_checkpoint_sha256")
            if (row.get("status") not in ("MEASURED","RESOLVED") or row.get("selected_checkpoint") is not True
                    or expected_checkpoint is not None and row.get("checkpoint_sha256")!=expected_checkpoint
                    or not row.get("reference_accepted") or not row.get("finite")
                    or not all(_number(row.get(k)) for k in ("E_G","E_U","E_L","uncertainty_combined"))):
                bad.append(q["query_id"]);continue
            valid.append(row)
        eg=_parent_mean(valid,lambda r:r["E_G"]);eu=_parent_mean(valid,lambda r:r["E_U"]);el=_parent_mean(valid,lambda r:r["E_L"])
        unc=_parent_mean(valid,lambda r:r["uncertainty_combined"])
        retained=(eg-el)/(eg-eu) if eg is not None and eg>eu else None
        complete=manifest_valid and bool(requested) and not missing and not bad and not dups and bool(valid)
        headroom=bool(retained is not None and unc<.1*(eg-eu))
        passed=complete and retained is not None and retained>=c2.get("retained_oracle_gain",.8) and headroom
        by_seed.append(dict(seed=seed,requested_queries=len(requested),eligible_queries=len(valid),missing_queries=missing,unresolved_queries=bad,
                            retained_gain=retained,E_G=eg,E_U=eu,E_L=el,uncertainty=unc,passed=passed,complete=complete,aggregate_headroom=headroom))
    c1pass=bool(wanted) and len(c1rows)==len(wanted) and all(r["passed"] for r in c1rows)
    c2pass=bool(by_seed) and all(r["passed"] for r in by_seed)
    status="UNRESOLVED" if not manifest_valid or unresolved or duplicates or not by_seed or any(not r["complete"] or not r["aggregate_headroom"] for r in by_seed) else "PASS" if c1pass and c2pass else "STOP"
    return _decision("C","learning",status,p,thresholds=dict(C1=c1,C2=c2),
                     evidence=dict(C1_passed=c1pass,C2_passed=c2pass,training_criterion=c1rows,selected_heldout=by_seed,
                                   candidate_failures=unresolved,baseline_adequacy=baseline,duplicate_trials=duplicates,screened_manifest_valid=manifest_valid),
                     denominators=dict(declared_candidate_trials=len(wanted),observed_candidate_trials=len(c1rows),requested_primary_diagnostic_queries=len(requested)),
                     reason="Training-fit diagnostic and validation-selected held-out gain are distinct; FNO adequacy does not silently redefine candidate success")


def utility_decision(performance,p):
    limits=dict(p.get("gates",{}).get("D",{}))
    measured=performance.get("utility",performance.get("comparison",{}))
    outcomes=measured.get("outcomes",[])
    receipt=performance.get("native_readiness",{})
    source=performance.get("source_tree_sha256")
    native=receipt.get("verified") is True and bool(source) and receipt.get("source_tree_sha256")==source
    identities=performance.get("checkpoint_identities",{})
    identity_valid=bool(identities) and all(isinstance(v,str) and len(v)==64 and all(c in "0123456789abcdef" for c in v) for v in identities.values())
    valid=bool(outcomes) and bool(measured.get("frozen_sha256")) and (native and identity_valid or _cpu(p))
    candidate=measured.get("candidate")
    seeds=p.get("training",{}).get("seeds",[])
    declared=measured.get("declared_candidates")
    if declared is None:
        # Legacy single-candidate comparisons remain usable; a paired campaign
        # must identify each actual method and its frozen selected checkpoint.
        declared=[dict(method=candidate,seed=seeds[0],checkpoint_sha256=identities.get(candidate))] if candidate and len(seeds)==1 else []
    methods=[r.get("method") for r in declared]
    declared_seeds=[r.get("seed") for r in declared]
    registration_valid=(bool(declared) and len(set(methods))==len(methods) and len(set(declared_seeds))==len(declared_seeds)
                        and set(declared_seeds)==set(seeds) and all(isinstance(m,str) and m for m in methods)
                        and all(r.get("method") in identities and r.get("checkpoint_sha256")==identities.get(r.get("method")) for r in declared))
    targets=measured.get("requested_targets",[p.get("target")])
    expected={(method,target) for method in methods for target in targets if target}
    observed=[(r.get("candidate",candidate),r.get("target_kind",p.get("target"))) for r in outcomes]
    seed_by_method={r.get("method"):r.get("seed") for r in declared}
    complete=valid and registration_valid and bool(expected) and len(set(observed))==len(observed) and set(observed)==expected and all(
        r.get("status") in ("PASS","STOP") and r.get("seed",seed_by_method.get(r.get("candidate",candidate)))==seed_by_method.get(r.get("candidate",candidate))
        and r.get("methods",{}).get(r.get("candidate",candidate),{}).get("qualified") is True
        and r.get("methods",{}).get(r.get("strongest_qualifying_control"),{}).get("qualified") is True for r in outcomes)
    passed=complete and all(r.get("passed") is True and _number(r.get("seconds_saved")) and r["seconds_saved"]>0 for r in outcomes)
    status="PASS" if passed else "STOP" if complete else "UNRESOLVED"
    return _decision("D","utility",status,p,thresholds=limits,evidence=dict(comparison=measured,complete_cost=complete,
                     candidate_registration_valid=registration_valid,missing_candidate_targets=sorted(expected-set(observed))),
                     denominators=dict(targets=len(targets),declared_candidate_seeds=len(seeds),requested_candidate_targets=len(expected),observed_candidate_targets=len(outcomes)),
                     reason="Requires every declared seed's actual candidate method and validation-frozen accuracy-qualified full-cost coverage; no energy or monetary inference")


def data_decision(screen,p):
    required=screen.get("required_counts",p.get("fit_parent_counts",{}))
    if "candidates" in screen and "selected_parent_ids" in screen:
        candidates=screen["candidates"]
        selected_ids=screen["selected_parent_ids"]
        chosen=[r for r in candidates if r.get("selected") is True]
        selected=defaultdict(list)
        for row in chosen:selected[row["split"]].append(row["parent_id"])
        authority_complete=(screen.get("status")=="ACCEPTED" and set(selected_ids)=={r["parent_id"] for r in chosen}
            and len(selected_ids)==len(set(selected_ids)) and all(r.get("eligible") is True and r.get("requested_queries")==r.get("observed_queries") for r in chosen)
            and len({(r["split"],r["slot"]) for r in chosen})==len(chosen)
            and screen.get("accepted_parent_slots")==screen.get("requested_parent_slots")==len(chosen))
    else:
        selected=screen.get("selected",screen.get("selected_parents",{}))
        authority_complete=False
    if isinstance(selected,list):
        grouped=defaultdict(list)
        for row in selected:grouped[row["split"]].append(row.get("parent_id"))
        selected=dict(grouped)
    counts={split:len(selected.get(split,[])) for split in ("train","validation","diagnostic")}
    requirements={split:int(required.get(split,0)) for split in counts}
    identities=[parent if isinstance(parent,str) else parent.get("parent_id") for values in selected.values() for parent in values] if isinstance(selected,dict) else []
    complete=authority_complete and all(requirements[s]>0 and counts[s]==requirements[s] for s in counts) and len(identities)==len(set(identities))
    status="PASS" if complete else "UNRESOLVED"
    return _decision("C0","fit_data",status,p,evidence=dict(selected_counts=counts,required_counts=requirements,screening=screen),
                     denominators=dict(requested_candidates=len(screen.get("candidates",screen.get("rows",[]))),selected_parents=sum(counts.values())),
                     reason="Reference/signal/deployment screening only; trained loss and speed cannot select this bank")


def calibration_decision(results,p):
    from .protocol import training_workload
    data=results.get("data",results)
    measurements=data.get("measurements",data.get("trial_measurements",{}))
    if isinstance(measurements,list):measurements={r.get("family",r.get("arm")):r for r in measurements}
    missing=[];invalid=[]
    for arm in p.get("training",{}).get("arms",[]):
        row=measurements.get(arm)
        if row is None:missing.append(arm);continue
        expected=training_workload(p,arm)
        minimum=p.get("training",{}).get("calibration_updates",64)
        updates=row.get("completed_updates",row.get("updates",0))
        valid=(row.get("status","COMPLETED_UPDATES")=="COMPLETED_UPDATES" and updates>=minimum
               and _number(row.get("seconds_per_update")) and row["seconds_per_update"]>0
               and row.get("workload_sha256",row.get("workload_hash"))==expected)
        if not valid:invalid.append(arm)
    native=(data.get("native") is True or data.get("execution_mode")=="desktop-slurm") and data.get("device")=="cuda"
    source=data.get("source_tree_sha256")
    bound=bool(source) and (p.get("source_tree_sha256") is None or source==p["source_tree_sha256"])
    status="PASS" if not missing and not invalid and bool(measurements) and (native and bound or _cpu(p)) else "UNRESOLVED"
    return _decision("CALIBRATION","calibration",status,p,evidence=dict(missing_arms=missing,invalid_arms=invalid,native=native,source_bound=bound,measurements=measurements),
                     denominators=dict(requested_arms=len(p.get("training",{}).get("arms",[])),measured_arms=len(measurements)),
                     reason="Actual stage layout/rank/dtype/gradients/validation workload must match source-bound native calibration")
