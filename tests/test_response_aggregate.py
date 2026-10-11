"""Aggregate scientific registries independently of plotting or Slurm."""
import copy
from pathlib import Path

import pytest

from tdn.analysis.response import report


def setup(monkeypatch, tmp_path, *, seeds=(11, 12, 13), scope="primary", pair_seeds=None):
    candidate, comparator = "coupled_source::primary::n32", "residual_fno::primary::n32"
    exploratory = "coupled_source::mean_only::n32"
    parent_ids = [f"parent-{j}" for j in range(3)]
    role = "galerkin_same_grid/grid-32x32/h-0.03/steps-1"
    pair = dict(candidate=candidate, comparator=comparator, scope=scope, seeds=list(pair_seeds or seeds))
    protocol = dict(profile="development", scientific_scope="DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS", dimension=2,
        train_grid=32, seeds=list(seeds), parent_counts={"train":32}, selection_source={"gate_decision":"CONTINUE"},
        evaluation=dict(frozen_schedules=[1], comparisons=[pair], rms_target=1., max_target=1.),
        units={"reference":dict(kind="references", grid=32, horizons=[.03], parent_ids=parent_ids),
               "evaluate":dict(kind="evaluate", target_kind="galerkin_same_grid", data_stages=["reference"]),
               "selection":dict(kind="select"), "aggregate":dict(kind="aggregate")}, trials={})
    models = []
    for family, method in (("coupled_source", candidate), ("residual_fno", comparator)):
        for seed in seeds:
            item = dict(target_kind="galerkin_same_grid", family=family, method_id=method, variant_id="primary", data_count=32, seed=seed, phase="main")
            protocol["trials"][method+str(seed)] = item
            models.append(item)
    registry = dict(target_kind="galerkin_same_grid", parents=list(parent_ids), seeds=list(seeds), methods=[candidate, comparator, exploratory],
        expected_seed_map={candidate:list(seeds),comparator:list(seeds),exploratory:[seeds[0]]},
        query_roles=[role], norm="trigonometric_interpolant_rms", frozen_before_evaluation=True)
    rows = [dict(parent_id=p, query_role=role, query_id="globally-unique-"+p, grid=[32,32], seed=s, method=m,
        target_kind="galerkin_same_grid", norm="trigonometric_interpolant_rms", rms=.1 if m==candidate else .2,
        maximum=.3, uncertainty_rms=1e-8, uncertainty_maximum=1e-8, reference_accepted=True,
        reference_bounded=True, bound_violation=False) for p in parent_ids for m in (candidate,comparator) for s in seeds]
    rows += [dict(r, method=exploratory) for r in rows if r["method"]==candidate and r["seed"]==seeds[0]]
    data = dict(sources=[], tables={
        "evaluate/query_registry.json":dict(filename="query_registry.json", payload=registry),
        "evaluate/endpoint_metrics.json":dict(filename="endpoint_metrics.json", payload={"rows":rows}),
        "selection/selection.json":dict(filename="selection.json",payload=dict(selection_contract="frozen-validation", selected_models=models, missing=[]))})
    monkeypatch.setattr(report, "collect", lambda ctx:data)
    monkeypatch.setattr(report, "verify_sources", lambda data:None)
    class Context:
        path=tmp_path
        def record(self,*args,**kwargs): pass
    ctx=Context();ctx.protocol=protocol
    return ctx,data,candidate,comparator,exploratory


def test_protocol_variant_identity_and_parent_relative_roles(monkeypatch,tmp_path):
    ctx,data,candidate,_,_=setup(monkeypatch,tmp_path)
    result=report.aggregate(ctx)
    claim=result["claims"]["comparisons"][0]
    assert claim["candidate"]==candidate
    assert claim["attempted_parents"]==3
    assert claim["complete_paired_parents"]==3
    assert claim["denominators"][candidate]["attempted"]==9
    assert result["freeze_eligible"]
    assert result["expected_selection_groups"]==6


def test_secondary_is_not_promoted_into_primary_family(monkeypatch,tmp_path):
    ctx,data,_,_,_=setup(monkeypatch,tmp_path,scope="secondary")
    result=report.aggregate(ctx)
    assert not result["claims"]["comparisons"]
    assert len(result["claims"]["exploratory_comparisons"])==1


def test_one_seed_exploration_allowed_but_primary_seed_dropping_rejected(monkeypatch,tmp_path):
    ctx,data,_,_,_=setup(monkeypatch,tmp_path,scope="secondary",pair_seeds=[11])
    result=report.aggregate(ctx)
    claim=result["claims"]["exploratory_comparisons"][0]
    assert all(v["attempted"]==3 for v in claim["denominators"].values())
    assert result["estimands"][0]["registry"]["seeds"]==[11]
    ctx.protocol["evaluation"]["comparisons"][0]["scope"]="primary"
    with pytest.raises(ValueError,match="full paired seed"):
        report.aggregate(ctx)


def test_missing_registry_retains_declared_parent_denominators(monkeypatch,tmp_path):
    ctx,data,candidate,_,_=setup(monkeypatch,tmp_path)
    del data["tables"]["evaluate/query_registry.json"]
    result=report.aggregate(ctx)
    claim=result["claims"]["comparisons"][0]
    assert claim["attempted_parents"]==3
    assert claim["complete_paired_parents"]==0
    assert claim["denominators"][candidate]["attempted"]==9
    assert not result["freeze_eligible"]


def test_missing_method_remains_unavailable_and_blocks_freeze(monkeypatch,tmp_path):
    ctx,data,candidate,_,_=setup(monkeypatch,tmp_path)
    reg=data["tables"]["evaluate/query_registry.json"]["payload"]
    reg["methods"].remove(candidate);reg["expected_seed_map"].pop(candidate)
    data["tables"]["evaluate/endpoint_metrics.json"]["payload"]["rows"]=[r for r in data["tables"]["evaluate/endpoint_metrics.json"]["payload"]["rows"] if r["method"]!=candidate]
    result=report.aggregate(ctx)
    assert not result["freeze_eligible"]
    assert result["claims"]["comparisons"][0]["denominators"][candidate]["failed_prediction_on_valid_reference"]==9


def test_selection_omission_cannot_disappear_when_no_trial_result_exists(monkeypatch,tmp_path):
    ctx,data,_,_,_=setup(monkeypatch,tmp_path)
    selected=data["tables"]["selection/selection.json"]["payload"]["selected_models"]
    selected.pop()
    result=report.aggregate(ctx)
    assert not result["freeze_eligible"]
    assert len(result["missing_selection_groups"])==1


def test_unregistered_method_or_query_role_rejected(monkeypatch,tmp_path):
    ctx,data,_,_,_=setup(monkeypatch,tmp_path)
    rows=data["tables"]["evaluate/endpoint_metrics.json"]["payload"]["rows"]
    rows[0]["method"]="coupled_source::unregistered::n32"
    with pytest.raises(ValueError,match="method/seed"):
        report.aggregate(ctx)
    ctx,data,_,_,_=setup(monkeypatch,tmp_path)
    data["tables"]["evaluate/query_registry.json"]["payload"]["query_roles"].append("unregistered")
    with pytest.raises(ValueError,match="query-role"):
        report.aggregate(ctx)


def test_changed_parent_or_equation_cannot_be_pooled(monkeypatch,tmp_path):
    ctx,data,_,_,_=setup(monkeypatch,tmp_path)
    data["tables"]["evaluate/query_registry.json"]["payload"]["parents"].pop()
    with pytest.raises(ValueError,match="parent denominator"):
        report.aggregate(ctx)


def test_phase_descendants_share_their_parent_weight_and_independence(monkeypatch,tmp_path):
    ctx,data,candidate,_,_=setup(monkeypatch,tmp_path)
    ctx.protocol["phase_variants"]=["original","phase_twin"]
    registry=data["tables"]["evaluate/query_registry.json"]["payload"]
    original=registry["query_roles"][0]
    registry["query_roles"].append(original+"/phase-phase_twin")
    rows=data["tables"]["evaluate/endpoint_metrics.json"]["payload"]["rows"]
    rows.extend([dict(r,query_role=original+"/phase-phase_twin",query_id=r["query_id"]+"-phase") for r in list(rows)])
    result=report.aggregate(ctx)
    claim=result["claims"]["comparisons"][0]
    assert claim["attempted_parents"]==3
    assert claim["denominators"][candidate]["attempted"]==18
    assert claim["rms_geometric_ratio"]==pytest.approx(.5)


def test_grid_relative_stress_keeps_raw_points_but_cannot_enter_primary_pool(monkeypatch,tmp_path):
    ctx,data,_,_,_=setup(monkeypatch,tmp_path)
    ctx.protocol["units"]["stress"] = dict(kind="evaluate", target_kind="galerkin_same_grid",
        descriptive_only=True, panel="grid_relative_stress", data_stages=["reference"])
    data["tables"]["stress/endpoint_metrics.json"] = dict(filename="endpoint_metrics.json",
        payload={"rows":[dict(parent_id="stress-special-parent",rms=.0000001)]})
    result=report.aggregate(ctx)
    assert result["claims"]["comparisons"][0]["attempted_parents"]==3
    assert len(result["excluded_panels"])==1
    assert result["excluded_panels"][0]["raw_endpoint_table_present"]
    assert result["excluded_panels"][0]["panel"]=="grid_relative_stress"
    assert data["tables"]["stress/endpoint_metrics.json"]["payload"]["rows"]
