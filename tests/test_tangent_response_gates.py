"""Controlled workflow fixtures test promotion logic, not native readiness."""
import copy

import pytest

from tdn.analysis.tangent_response.gates import (
    calibration_decision, compression_decision, data_decision, learning_decision,
    mechanism_decision, require_promotion, utility_decision, validate_decision,
)
from tdn.analysis.tangent_response.protocol import build_protocol, training_workload


def protocol(parents=2):
    queries=[dict(parent_id=f"p{i}",query_id=f"q{i}",split="diagnostic",panel="primary",
                  target_kind="fd_same_grid",grid=[5,5],base_grid=[5,5],horizon=.075) for i in range(parents)]
    return dict(profile="pilot",execution_scope="NATIVE_RESEARCH",target="fd_same_grid",
                geometry=dict(grids=[[5,5]]),fit_geometry=dict(grid=[5,5]),queries=queries,
                fit_queries=copy.deepcopy(queries),gates=dict(
                    A=dict(rms_ratio=1.5,parent_improvement_fraction=.75,maximum_regression=2.,
                           uncertainty_fraction_of_improvement=.1,minimum_signal_uncertainty_ratio=10.),
                    B=dict(rank_cap=8,action_relative_error=.1,probe_fraction=.9,worst_action_relative_error=.3,retained_oracle_gain=.8),
                    C1=dict(relative_oracle_correction_error=.05,query_fraction=.9),C2=dict(retained_oracle_gain=.8),
                    D=dict(primary_tolerance=2e-5,cost_ratio=1.)),compression=dict(ranks=[2,4,8]),
                training=dict(updates=300,seeds=[1,2,3],arms=["learned_response","residual_fno"]))


def mechanism_row(q,eg=1.,eu=.5,unc=1e-8):
    return dict(q,reference=dict(accepted=True),primary_status="RESOLVED",
                controls={name:dict(rms=rms,maximum=rms,uncertainty_rms=unc/2) for name,rms in [("anchor",2.),("G",eg),("U",eu)]})


def test_mechanism_ratio_of_weighted_rms_means_not_mean_ratios():
    p=protocol()
    rows=[mechanism_row(p["queries"][0],eg=1.,eu=.01),mechanism_row(p["queries"][1],eg=9.,eu=8.)]
    result=mechanism_decision(rows,p)
    assert result["decision"]=="STOP"
    assert result["evidence"]["rms_ratio"]==pytest.approx(10/8.01)
    assert result["evidence"]["separated_parent_fraction"]==1


def test_mechanism_requested_missing_uncertain_and_tail_outcomes():
    p=protocol()
    rows=[mechanism_row(q) for q in p["queries"]]
    assert mechanism_decision(rows,p)["status"]=="PASS"
    assert mechanism_decision(rows[:1],p)["status"]=="UNRESOLVED"
    uncertain=copy.deepcopy(rows);uncertain[0]["controls"]["G"]["uncertainty_rms"]=None
    assert mechanism_decision(uncertain,p)["status"]=="UNRESOLVED"
    tail=copy.deepcopy(rows);tail[0]["controls"]["U"]["maximum"]=3
    assert mechanism_decision(tail,p)["status"]=="STOP"
    unknown=copy.deepcopy(rows);unknown[0]["reference"]["accepted"]=False
    assert mechanism_decision(unknown,p)["status"]=="UNRESOLVED"


def test_floor_and_designed_intervention_do_not_rescue_primary():
    p=protocol()
    rows=[mechanism_row(q,eg=1.,eu=.8) for q in p["queries"]]
    rows.append(dict(mechanism_row(dict(p["queries"][0],query_id="intervention"),eg=100,eu=.001),panel="physics_intervention"))
    assert mechanism_decision(rows,p)["status"]=="STOP"
    floor=[mechanism_row(q,unc=1.) for q in p["queries"]]
    result=mechanism_decision(floor,p)
    assert result["status"]=="UNRESOLVED" and result["denominators"]["eligible_queries"]==0


def test_cpu_evidence_cannot_promote_and_galerkin_is_separate():
    p=protocol();rows=[mechanism_row(q) for q in p["queries"]]
    p["profile"]="cpu-smoke";p["execution_scope"]="CPU_IMPLEMENTATION_SMOKE"
    result=mechanism_decision(rows,p)
    assert result["status"]=="IMPLEMENTATION_ONLY" and result["underlying_scientific_decision"]=="PASS"
    assert not result["eligible"]
    with pytest.raises(ValueError,match="literal"):
        require_promotion(result,"A")
    p["profile"]="pilot";p["execution_scope"]="NATIVE_RESEARCH";p["target"]="galerkin_same_grid"
    assert mechanism_decision(rows,p)["status"]=="NOT_APPLICABLE"


def test_decision_alias_drift_rejected():
    p=protocol();result=mechanism_decision([mechanism_row(q) for q in p["queries"]],p)
    validate_decision(result,gate="A")
    with pytest.raises(ValueError,match="aliases"):
        validate_decision(dict(result,eligible_next=False))
    with pytest.raises(ValueError,match="Wrong"):
        validate_decision(result,gate="B")


def compression_fixture(p,deployable=True):
    plan=[dict(probe_id="d",split="train",probe_role="heldout_direction"),
          dict(probe_id="p",split="diagnostic",probe_role="fit")]
    actions=[dict(r,method="shared_response_action_pod_rank2",requested_rank=2,status="RESOLVED",nontrivial=True,
                  uncertainty=1e-10,relative_error=.05,deployable=deployable) for r in plan]
    endpoints=[dict(q,method="shared_response_action_pod_rank2",requested_rank=2,status="RESOLVED",deployable=deployable,
                    mean_error=dict(rms=1.),oracle_error=dict(rms=.5),compressed_error=dict(rms=.55),
                    uncertainty=1e-10,teacher_path_uncertainty=1e-10) for q in p["queries"]]
    return dict(probe_plan=plan,action_rows=actions,endpoint_rows=endpoints)


def test_compression_both_scopes_requested_coverage_and_uncertainty():
    p=protocol();data=compression_fixture(p)
    result=compression_decision(data,p)
    assert result["status"]=="PASS" and result["evidence"]["selected"]["retained_gain"]==pytest.approx(.9)
    missing=copy.deepcopy(data);missing["action_rows"]=missing["action_rows"][:1]
    assert compression_decision(missing,p)["status"]=="UNRESOLVED"
    unseparated=copy.deepcopy(data)
    for row in unseparated["endpoint_rows"]:del row["teacher_path_uncertainty"]
    assert compression_decision(unseparated,p)["status"]=="UNRESOLVED"
    bad=copy.deepcopy(data);bad["action_rows"][1]["relative_error"]=.4
    assert compression_decision(bad,p)["status"]=="STOP"


def test_state_oracle_headroom_has_no_deployable_generalization_claim():
    p=protocol();result=compression_decision(compression_fixture(p,deployable=False),p)
    assert result["status"]=="PASS"
    selected=result["evidence"]["selected"]
    assert not selected["deployable"] and "not deployable" in selected["generalization_scope"]


def test_task_residual_panels_do_not_assert_reusable_operator_failure_or_promotion():
    p=protocol();task=compression_fixture(p)
    for key in ("action_rows","endpoint_rows"):
        for row in task[key]:
            row.update(method="task_residual_response_pod_rank2",compression_object="task_residual_family")
    task["action_rows"][0]["relative_error"]=50.
    result=compression_decision(task,p)
    assert result["status"]=="NOT_APPLICABLE"
    diagnostic=result["evidence"]["candidates"][0]
    assert diagnostic["status"]=="NOT_APPLICABLE" and not diagnostic["promotion_eligible"]
    assert diagnostic["reported_action_rows"]==2 and diagnostic["reported_endpoint_rows"]==2
    mixed=compression_fixture(p)
    for key in ("action_rows","endpoint_rows"):
        mixed[key]+=task[key]
    result=compression_decision(mixed,p)
    assert result["status"]=="PASS"
    assert result["evidence"]["selected"]["method"]=="shared_response_action_pod_rank2"


def test_compression_cannot_drop_an_unfavorable_oracle_horizon():
    p=protocol();data=compression_fixture(p)
    data["endpoint_rows"][1].update(mean_error=dict(rms=1.),oracle_error=dict(rms=1.4),compressed_error=dict(rms=1.3))
    result=compression_decision(data,p)
    # Aggregate oracle gain is .05 and learned gain .075: retain both rows,
    # rather than quoting the first horizon's .9 retained-gain ratio.
    selected=result["evidence"]["selected"]
    assert selected["E_G"]==1 and selected["E_U"]==pytest.approx(.95)
    assert selected["E_L"]==pytest.approx(.925)
    assert selected["retained_gain"]==pytest.approx(1.5)


def learning_fixture(p):
    trials=[]
    comparisons=[]
    for seed in p["training"]["seeds"]:
        rows=[dict(query_id="train-q",parent_id="train-p",split="train",relative_oracle_correction_error=.03,
                   finite=True,reference_accepted=True)]
        trials.append(dict(family="learned_response",seed=seed,status="COMPLETED_UPDATES",updates=300,requested_updates=300,
                           requested_train_query_ids=["train-q"],criterion_train=dict(rows=rows),criterion_update=280,selected_update=40))
        for q in p["fit_queries"]:
            comparisons.append(dict(q,family="learned_response",seed=seed,status="MEASURED",selected_checkpoint=True,
                                    finite=True,reference_accepted=True,E_G=1.,E_U=.5,E_L=.55,uncertainty_combined=1e-8))
    trials.append(dict(family="residual_fno",seed=1,status="FAILED_NONFINITE",updates=10,requested_updates=300))
    return trials,dict(learning_comparisons=comparisons)


def test_learning_training_fit_vs_selection_baseline_and_no_init_rule():
    p=protocol();trials,evaluation=learning_fixture(p)
    before=copy.deepcopy(p)
    result=learning_decision(trials,evaluation,p)
    assert result["status"]=="PASS" and result["evidence"]["baseline_adequacy"][0]["status"]=="FAILED_NONFINITE"
    assert result["evidence"]["training_criterion"][0]["criterion_update"]!=result["evidence"]["training_criterion"][0]["selected_update"]
    assert p==before
    stopped=copy.deepcopy(trials);stopped[0]["criterion_train"]["rows"][0]["relative_oracle_correction_error"]=.08
    assert learning_decision(stopped,evaluation,p)["status"]=="STOP"


def test_learning_missing_work_teacher_or_selected_evaluation_unresolved():
    p=protocol();trials,evaluation=learning_fixture(p)
    partial=copy.deepcopy(trials);partial[0]["updates"]=299
    assert learning_decision(partial,evaluation,p)["status"]=="UNRESOLVED"
    unseen=copy.deepcopy(evaluation);unseen["learning_comparisons"][0]["selected_checkpoint"]=False
    assert learning_decision(trials,unseen,p)["status"]=="UNRESOLVED"
    assert learning_decision(trials,{},p)["status"]=="UNRESOLVED"


def test_complete_screening_slots_required():
    p=protocol();p["fit_parent_counts"]={"train":1,"validation":1,"diagnostic":1}
    candidates=[dict(parent_id=s,split=s,slot=0,candidate=0,requested_queries=3,observed_queries=3,eligible=True,selected=True)
                for s in p["fit_parent_counts"]]
    screen=dict(status="ACCEPTED",requested_parent_slots=3,accepted_parent_slots=3,candidates=candidates,
                selected_parent_ids=list(p["fit_parent_counts"]))
    assert data_decision(screen,p)["status"]=="PASS"
    bad=copy.deepcopy(screen);bad["candidates"][0]["eligible"]=False
    assert data_decision(bad,p)["status"]=="UNRESOLVED"


def test_calibration_workload_source_and_actual_device_required():
    p=build_protocol("pilot",phase="calibration");p["source_tree_sha256"]="source"
    measurements={arm:dict(status="COMPLETED_UPDATES",updates=64,seconds_per_update=.1,
                           workload_sha256=training_workload(p,arm)) for arm in p["training"]["arms"]}
    evidence=dict(native=True,device="cuda",source_tree_sha256="source",measurements=measurements)
    assert calibration_decision(evidence,p)["status"]=="PASS"
    assert calibration_decision(dict(evidence,device="cpu"),p)["status"]=="UNRESOLVED"
    wrong=copy.deepcopy(evidence);wrong["measurements"]["learned_response"]["workload_sha256"]="other"
    assert calibration_decision(wrong,p)["status"]=="UNRESOLVED"


def test_utility_strongest_qualified_control_not_every_failed_control():
    p=protocol();p["training"]["seeds"]=[1]
    outcome=dict(status="PASS",passed=True,seconds_saved=.01,strongest_qualifying_control="gl4",
                 methods=dict(learned=dict(qualified=True),gl4=dict(qualified=True),inaccurate=dict(qualified=False)))
    result=dict(native_readiness=dict(verified=True,source_tree_sha256="source"),source_tree_sha256="source",checkpoint_identities=dict(learned="a"*64),
                utility=dict(outcomes=[outcome],candidate="learned",frozen_sha256="frozen"))
    assert utility_decision(result,p)["status"]=="PASS"
    assert utility_decision(dict(result,native_readiness={}),p)["status"]=="UNRESOLVED"


def test_utility_requires_all_paired_seed_methods_with_stable_identities():
    p=protocol();names=["learned_response","learned_response-seed-2","learned_response-seed-3"]
    identities={name:str(seed)*64 for name,seed in zip(names,p["training"]["seeds"])}
    declarations=[dict(method=name,seed=seed,checkpoint_sha256=identities[name]) for name,seed in zip(names,p["training"]["seeds"])]
    outcomes=[dict(candidate=name,seed=seed,target_kind=p["target"],status="PASS",passed=True,seconds_saved=.01,
                   strongest_qualifying_control="gl4",methods={name:dict(qualified=True),"gl4":dict(qualified=True)})
              for name,seed in zip(names,p["training"]["seeds"])]
    result=dict(native_readiness=dict(verified=True,source_tree_sha256="source"),source_tree_sha256="source",checkpoint_identities=identities,
                utility=dict(candidate="paired_learned_seeds",declared_candidates=declarations,requested_targets=[p["target"]],outcomes=outcomes,frozen_sha256="frozen"))
    assert utility_decision(result,p)["status"]=="PASS"
    missing=copy.deepcopy(result);missing["utility"]["outcomes"].pop()
    assert utility_decision(missing,p)["status"]=="UNRESOLVED"
    duplicate=copy.deepcopy(result);duplicate["utility"]["outcomes"].append(outcomes[0])
    assert utility_decision(duplicate,p)["status"]=="UNRESOLVED"
    drift=copy.deepcopy(result);drift["utility"]["declared_candidates"][2]["checkpoint_sha256"]="f"*64
    assert utility_decision(drift,p)["status"]=="UNRESOLVED"
    failed=copy.deepcopy(result);failed["utility"]["outcomes"][2].update(status="STOP",passed=False,seconds_saved=-.01)
    assert utility_decision(failed,p)["status"]=="STOP"
