import json
from types import SimpleNamespace

import torch

from tdn.analysis.tangent_response.contracts import digest
from tdn.analysis.tangent_response.inference import EndpointMethod
from tdn.analysis.tangent_response.stages_evaluation import run_evaluation,run_utility


class Budget:
    def __init__(self):self.checks=0
    def check(self):self.checks+=1


def fixture(tmp_path):
    samples=[]
    for split in ("train","validation","diagnostic"):
        u=torch.full((1,1,5,5),.2,dtype=torch.float32)
        record=dict(parent_id=split,query_id=split+"-q",split=split,slot=0,candidate=0,panel="primary",grid=[5,5],
                    base_grid=[5,5],lengths=[1.,1.],horizon=.02 if split=="diagnostic" else .01,
                    target_kind="fd_same_grid",kappa=0.,reaction_rate=0.,shape_sha256=split)
        samples.append(dict(record=record,u0=u,teacher64=u.double(),mean_endpoint64=u.double()+.03,
                            oracle_endpoint64=u.double()+.01,
                            reference=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12),
                            oracle_metadata=dict(accepted=True,uncertainty_rms=1e-10,matched_mean_uncertainty_rms=1e-10)))
    p=dict(profile="cpu-smoke",execution_scope="CPU_IMPLEMENTATION_SMOKE",target="fd_same_grid",
           fit_queries=[s["record"] for s in samples],fit_geometry=dict(grid=[5,5]),fit_parent_counts=dict(train=1,validation=1,diagnostic=1),
           training=dict(arms=["learned_response","residual_fno"],seeds=[1],updates=2),
           gates=dict(C1=dict(relative_oracle_correction_error=.05,query_fraction=.9),C2=dict(retained_oracle_gain=.8)),
           evaluation=dict(auxiliary_parents=0,maximum_oversampling=4),horizons=dict(diagnostic=[.02]),anchor=dict(reaction_substeps=8),
           performance=dict(tolerances=[1e-4,2e-5,1e-5],primary_tolerance=2e-5,repeats=1,warmup=0,classical_schedules=[1],adaptive_tolerances=[1e-8]))
    def learned(u,h,*args):
        value=u.double()+.011
        return dict(output=value.float(),physical_prediction=value,work=dict(smooth_anchor_evaluations=1,deployed_anchor_evaluations=1))
    fn=EndpointMethod("learned_response",learned,dict(deployable=True,inference_teacher_inputs=False))
    methods={"learned_response":dict(call=fn,family="learned_response",seed=1,checkpoint_sha256="a"*64)}
    trial=dict(family="learned_response",seed=1,status="COMPLETED_UPDATES",updates=2,requested_updates=2,
               requested_train_query_ids=["train-q"],selected_checkpoint_sha256="a"*64,selected_update=1,criterion_update=2,
               criterion_train=dict(rows=[dict(query_id="train-q",parent_id="train",split="train",finite=True,reference_accepted=True,
                                               relative_oracle_correction_error=.01)]))
    ctx=SimpleNamespace(protocol=p,path=tmp_path,sourcehash="source",device="cpu",budget=Budget(),prerequisites={})
    return ctx,samples,methods,[dict(path="unused",result=trial)]


def test_evaluation_selected_seed_primary_manifest_and_separate_physical_deployment(tmp_path):
    ctx,samples,methods,trials=fixture(tmp_path)
    summary=run_evaluation(ctx,samples,methods,trials)
    result=json.loads((tmp_path/"evaluation_results.json").read_text())
    assert summary["scientific_outcome"]=="IMPLEMENTATION_ONLY"
    assert result["screened_manifest_sha256"]==digest(result["screened_query_plan"])
    assert result["requested_train_query_ids"]==["train-q"]
    row=result["learning_comparisons"][0]
    assert row["selected_checkpoint"] and row["seed"]==1
    assert row["E_L"]!=row["rms"]
    assert row["output_dtype"]=="torch.float32"
    assert row["final_rounding_available"] and row["final_rounding_rms"]>0
    assert result["learning_decision"]["underlying_scientific_decision"]=="PASS"
    assert ctx.budget.checks>0


def test_utility_freezes_validation_before_diagnostics_and_retains_negative_candidate(tmp_path):
    ctx,samples,methods,trials=fixture(tmp_path)
    summary=run_utility(ctx,samples,methods,trials)
    frozen=json.loads((tmp_path/"frozen_schedules.json").read_text())
    result=json.loads((tmp_path/"performance_results.json").read_text())
    assert summary["status"]=="IMPLEMENTATION_ONLY"
    assert frozen["selection_scope"].startswith("validation only")
    assert result["utility"]["outcomes"][0]["status"]=="UNRESOLVED"
    assert not result["fresh_process"] or all(r.get("status")=="UNRESOLVED" for r in result["fresh_process"].values())
    assert result["energy_joules"] is None and result["monetary_cost"] is None
    assert result["validation"]["rows"] and result["diagnostic"]["rows"]


def test_evaluation_expired_budget_preserves_requested_endpoint_rows(tmp_path):
    ctx,samples,methods,trials=fixture(tmp_path)
    class Expired:
        def check(self):raise TimeoutError("declared cap")
    ctx.budget=Expired()
    run_evaluation(ctx,samples,methods,trials)
    result=json.loads((tmp_path/"evaluation_results.json").read_text())
    assert len(result["rows"])==2
    assert all(row["status"]=="TIME_CENSORED" for row in result["rows"])
    assert result["learning_decision"]["underlying_scientific_decision"]=="UNRESOLVED"


def test_utility_expired_budget_preserves_requested_rows_and_freeze(tmp_path):
    ctx,samples,methods,trials=fixture(tmp_path)
    class Expired:
        def check(self):raise TimeoutError("declared cap")
    ctx.budget=Expired()
    run_utility(ctx,samples,methods,trials)
    result=json.loads((tmp_path/"performance_results.json").read_text())
    assert result["diagnostic"]["rows"]
    assert all(row["status"]=="TIME_CENSORED" for row in result["diagnostic"]["rows"])
    assert result["utility"]["outcomes"][0]["status"]=="UNRESOLVED"
