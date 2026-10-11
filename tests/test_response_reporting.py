"""Synthetic corruption/denominator fixtures, never scientific measurements."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdn.analysis.response.core import Context, check, write_reviews
from tdn.analysis.response.protocol import build_protocol, digest
from tdn.analysis.response import report as atlas
from tdn.reporting import begin_report
from tdn.runtime.metadata import write_json
from tdn.tower_analytics import publish_outputs


def seal(path, protocol, artifacts):
    stage = path.name
    manifest = dict(schema="tdn.response-science/v1", stage=stage,
        kind=protocol["units"][stage]["kind"], unit_sha256=digest(protocol["units"][stage]),
        protocol_sha256=digest(protocol), source_tree_sha256="a"*64,
        software={"source_tree_sha256":"a"*64}, device="cpu")
    for name, value in artifacts.items():
        write_json(path/name, value)
    manifest["artifacts"] = {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*") if p.is_file() and p.name not in ("science_manifest.json","COMPLETED")}
    write_json(path/"science_manifest.json", manifest)
    (path/"COMPLETED").write_text(digest(manifest)+"\n")
    return manifest


def tower_fixture(tmp_path):
    p = build_protocol("smoke", local_smoke=True)
    source=tmp_path/"report";source.mkdir()
    ctx=Context(p,"report",source,{},"cpu",SimpleNamespace(check=lambda:None))
    ctx.record("synthetic",["M0"], metrics={"count":1},
        checks=[check("identity",0,0,"eq",category="math")])
    write_reviews(source,ctx.rows)
    (source/"figures").mkdir()
    (source/"figures/chart.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    seal(source,p,{"protocol.json":p,"summary.json":dict(schema=p["schema"],stage="report",status="COMPLETED",
        experiment_count=1,protocol_sha256=digest(p),source_tree_sha256="a"*64,device="cpu"),
        "learning_curves.json":{"rows":[dict(model_id="residual_fno",family="residual_fno",track="fd_same_grid",
            phase="factorial",trial_id=t,seed=1,update=0,train_loss=.1,elapsed_seconds=.1) for t in ("clip","no-clip")]}})
    tower=Path(begin_report(source,name="TDN/response/synthetic",script="scripts/response.py",report_parent=tmp_path/"tower"))
    return source,tower,p


def test_tower_recognizes_response_units_and_nested_verified_figures(tmp_path):
    source,tower,p=tower_fixture(tmp_path)
    result=publish_outputs(tower,[source])
    assert result["frontier"]["reporting_complete"],result
    assert result["frontier"]["figure_count"]==1
    index=json.loads((tower/"outputs/frontier-tables.json").read_text())
    assert index["sources"][0]["program"]=="response"
    assert index["sources"][0]["scientific_scope"]==p["scientific_scope"]


def test_tower_changed_figure_not_credited(tmp_path):
    source,tower,_=tower_fixture(tmp_path)
    (source/"figures/chart.pdf").write_bytes(b"changed")
    result=publish_outputs(tower,[source])
    assert not result["frontier"]["reporting_complete"]
    assert result["frontier"]["figure_count"]==0


def test_atlas_source_hash_change_and_exact_pointers(tmp_path):
    source,_,p=tower_fixture(tmp_path)
    ctx=SimpleNamespace(protocol=p,stage="aggregate",prerequisites={"report":source})
    data=atlas.collect(ctx)
    curves=list(atlas.records(data,["learning_curves.json"]))
    assert [r["json_pointer"] for r in curves]==["/rows/0","/rows/1"]
    (source/"learning_curves.json").write_text("{}")
    with pytest.raises(ValueError,match="changed"):
        atlas.verify_sources(data)


def aggregate_fixture(tmp_path, missing=False):
    target="fd_same_grid"
    role=f"{target}/grid-8x8/h-0.02/steps-1"
    p=build_protocol("smoke",local_smoke=True)
    p.update(profile="development",seeds=[1],train_grid=8,phase_variants=["original"],
        selection_source={"gate_decision":"CONTINUE"},
        evaluation=dict(rms_target=1.,max_target=1.,frozen_schedules=[1],
            comparisons=[dict(candidate="coupled_source",comparator="residual_fno")]))
    p["units"]={"reference":dict(id="reference",kind="references",grid=8,horizons=[.02],parent_ids=["p1","p2"]),
        "evaluate":dict(id="evaluate",kind="evaluate",target_kind=target,data_stages=["reference"]),
        "selection":dict(id="selection",kind="select"),"aggregate":dict(id="aggregate",kind="aggregate")}
    p["stages"]=list(p["units"])
    source=tmp_path/"evaluate";source.mkdir()
    rows=[dict(parent_id=parent,query_id=parent,query_role=role,seed=1,method=method,target_kind=target,grid=[8,8],
        norm="nodal_rms",rms=.1,maximum=.2,uncertainty_rms=1e-8,uncertainty_maximum=1e-8,reference_accepted=True,
        prediction_finite=True,reference_bounded=True,bound_violation=False)
        for parent in ["p1","p2"] for method in ["coupled_source","residual_fno"]
        if not (missing and parent=="p2" and method=="coupled_source")]
    seal(source,p,{"query_registry.json":dict(target_kind=target,parents=["p1","p2"],query_roles=[role],seeds=[1],frozen_before_evaluation=True,
        norm="nodal_rms",methods=["coupled_source","residual_fno"],expected_seed_map={"coupled_source":[1],"residual_fno":[1]}),
        "endpoint_metrics.json":{"rows":rows}})
    path=tmp_path/"aggregate";path.mkdir()
    ctx=SimpleNamespace(protocol=p,stage="aggregate",path=path,prerequisites={"evaluate":source},record=lambda *a,**k:None)
    return ctx


def test_aggregate_missing_model_does_not_shrink_registered_denominator(tmp_path):
    result=atlas.aggregate(aggregate_fixture(tmp_path,missing=True))
    claim=result["claims"]["comparisons"][0]
    assert claim["attempted_parents"]==2
    assert claim["complete_paired_parents"]==1
    assert claim["denominators"]["coupled_source"]["attempted"]==2
    assert not claim["primary_rms_supported"]
    assert not result["freeze_eligible"]


def test_aggregate_rejects_changed_expected_parent_menu(tmp_path):
    ctx=aggregate_fixture(tmp_path)
    source=ctx.prerequisites["evaluate"]
    registry=json.loads((source/"query_registry.json").read_text());registry["parents"]=["p1"]
    seal(source,ctx.protocol,{"query_registry.json":registry})
    with pytest.raises(ValueError,match="parent denominator"):
        atlas.aggregate(ctx)


def test_ours_theirs_shared_control_styles_are_distinct():
    assert atlas.style("coupled_source")["marker"]=="^"
    assert atlas.style("direct_fno")["marker"]=="o"
    assert atlas.style("residual_fno")["marker"]=="s"


def test_review_counts_latest_continuation_once_without_claiming_success():
    data=dict(scope="CPU_IMPLEMENTATION_SMOKE",stages=[dict(status="COMPLETED"),dict(status="FAILED")],
        omissions=[dict(stage="missing")],tables={
            "chunk1":dict(filename="training_results.json",payload={"trials":[dict(trial_id="a",updates=1,selected_update=0,status="INTERRUPTED")]}),
            "chunk2":dict(filename="training_results.json",payload={"trials":[dict(trial_id="a",updates=2,selected_update=0,status="COMPLETED")]}),
        })
    review=atlas.evidence_review(data)
    assert len(review["answers"])==7
    assert "Distinct recorded trials: 1; initialization selected: 1" in review["answers"][2]["answer"]
    assert "Missing or unverified stages: 1" in review["answers"][0]["answer"]
    assert "NOT_MEASURED" in review["answers"][5]["answer"]
    assert review["trial_sources"]==[dict(source_id="chunk2",json_pointer="/trials/0")]


def test_accounting_never_adds_steps_to_allocations_or_fills_missing_time(tmp_path):
    base=tmp_path/"campaign";(base/"state").mkdir(parents=True)
    ctx=SimpleNamespace(path=base/"report")
    data={"sources":[]}
    snapshot=dict(schema="tdn.scheduler-accounting/v1",status="RECORDED",all_allocations_terminal=True,
        requested_job_ids=["11","12"],records=[
            dict(record_kind="allocation",allocation_job_id="11",elapsed_seconds=10,allocated_cpu_seconds=20,State="FAILED"),
            dict(record_kind="step",allocation_job_id="11",elapsed_seconds=9,allocated_cpu_seconds=18),
            dict(record_kind="allocation",allocation_job_id="12",elapsed_seconds=None,allocated_cpu_seconds=None,State="COMPLETED")])
    write_json(base/"state/scheduler-accounting-20260101.json",snapshot)
    result=atlas.accounting_snapshot(ctx,data)
    assert len(result["rows"])==2
    assert result["observed_allocation_seconds"]==10
    assert result["complete_allocation_seconds"] is None
    assert result["energy_joules"] is None and result["currency_cost"] is None
    assert data["sources"][0]["sha256"]


def test_atlas_keeps_failed_operational_cost_without_crediting_science(tmp_path):
    p=build_protocol("smoke",local_smoke=True)
    ctx=SimpleNamespace(protocol=p,stage="report",prerequisites={},
        stage_failures={"audit":{"status":"FAILED","elapsed_seconds":17.,"cost_scope":"CLI timer","error":"broken"}})
    data=atlas.collect(ctx)
    audit=next(s for s in data["stages"] if s["stage"]=="audit")
    assert audit["seconds"]==17 and audit["timing_scope"]=="CLI timer"
    assert data["tables"]=={}
