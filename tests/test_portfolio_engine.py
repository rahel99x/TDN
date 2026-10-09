"""Recoverability and evidence safety using explicitly synthetic dispatcher rows."""
import json
from pathlib import Path

import pytest

from tdn.analysis.portfolio import engine
from tdn.analysis.portfolio.core import check
from tdn.analysis.portfolio.protocol import build_protocol, digest, validate_protocol
from tdn.runtime.metadata import write_json


@pytest.fixture
def dispatch(monkeypatch):
    metadata={"source_tree_sha256":"a"*64,"python":"unit-test"}
    monkeypatch.setattr(engine,"software_metadata",lambda:metadata.copy())
    def run(ctx):
        identity="synthetic-"+ctx.stage
        if identity not in ctx.identities:
            ctx.record(identity,["A1"],metrics={"unit_test_only":True},checks=[
                check("synthetic-math",True,True,"eq",category="math"),
                check("synthetic-no-result",None,None)])
        write_json(ctx.path/"payload.json",{"unit_test_only":True,"stage":ctx.stage})
        return {"unit_test_only":True}
    monkeypatch.setattr(engine,"_dispatch",run)
    return run,metadata


def campaign(tmp_path,p,stages):
    paths={}
    for stage in stages:
        path=tmp_path/stage
        engine.run_stage(p,stage,path,prerequisites={k:paths[k] for k in p["units"][stage]["dependencies"]})
        paths[stage]=path
    return paths


def test_all_declared_units_seal_exact_scope(tmp_path,dispatch):
    p=build_protocol("smoke");paths=campaign(tmp_path,p,p["units"])
    for stage,path in paths.items():
        manifest=engine.verify_science(p,path,source_tree_sha256="a"*64)
        assert manifest["unit_sha256"]==digest(p["units"][stage])
        assert manifest["device"]=="cpu"
    with pytest.raises(FileExistsError):engine.run_stage(p,"audit",paths["audit"])


@pytest.mark.parametrize("mode",["bytes","extra","missing","symlink","marker"])
def test_evidence_tampering_rejected(tmp_path,dispatch,mode):
    p=build_protocol();base=campaign(tmp_path,p,["audit"])["audit"]
    if mode=="bytes":(base/"payload.json").write_text("{}")
    elif mode=="extra":(base/"hidden.json").write_text("{}")
    elif mode=="missing":(base/"payload.json").unlink()
    elif mode=="symlink":(base/"link").symlink_to(base/"payload.json")
    else:(base/"COMPLETED").write_text("wrong")
    with pytest.raises((ValueError,OSError)):engine.verify_science(p,base)


def test_mixed_training_ancestors_rejected(tmp_path,dispatch):
    p=build_protocol()
    first=campaign(tmp_path/"first",p,["audit","prepare","train-A-000"])
    second=campaign(tmp_path/"second",p,["audit","prepare","train-B-000"])
    with pytest.raises(ValueError,match="lineages"):
        engine.run_stage(p,"freeze",tmp_path/"freeze",prerequisites={"train-A-000":first["train-A-000"],"train-B-000":second["train-B-000"]})


def test_changed_software_cannot_satisfy_prerequisite(tmp_path,dispatch):
    p=build_protocol();base=campaign(tmp_path,p,["audit"])["audit"]
    dispatch[1]["python"]="changed"
    with pytest.raises(ValueError,match="software"):
        engine.run_stage(p,"prepare",tmp_path/"prepare",prerequisites={"audit":base})


def test_math_failure_blocks_audit_but_not_research_negative(tmp_path,dispatch,monkeypatch):
    def fail(ctx):
        ctx.record("synthetic-failure",["A1"],checks=[check("wrong-math",2,1,"eq",category="math")])
    monkeypatch.setattr(engine,"_dispatch",fail)
    p=build_protocol()
    with pytest.raises(RuntimeError,match="mathematical checks"):
        engine.run_stage(p,"audit",tmp_path/"bad")
    assert not (tmp_path/"bad"/"COMPLETED").exists()


def test_interrupted_train_resumes_committed_ledger(tmp_path,dispatch,monkeypatch):
    p=build_protocol();parents=campaign(tmp_path,p,["audit","prepare"])
    prior={k:parents[k] for k in ("audit","prepare")};base=tmp_path/"train"
    def interrupted(ctx):dispatch[0](ctx);raise TimeoutError("unit-test interruption")
    monkeypatch.setattr(engine,"_dispatch",interrupted)
    with pytest.raises(TimeoutError):engine.run_stage(p,"train-A-000",base,prerequisites=prior)
    assert json.loads((base/"summary.json").read_text())["status"]=="INTERRUPTED"
    monkeypatch.setattr(engine,"_dispatch",dispatch[0])
    result=engine.run_stage(p,"train-A-000",base,prerequisites=prior,resume=True)
    assert result["experiment_count"]==1
    engine.verify_science(p,base)


def test_report_can_show_missing_science_without_accepting_it(tmp_path,dispatch):
    p=build_protocol();base=tmp_path/"report"
    result=engine.run_stage(p,"report",base,stage_failures={"train-A-000":{"status":"INTERRUPTED"}})
    assert result["status"]=="COMPLETED"
    assert engine.verify_science(p,base)["prerequisites"]=={}


def test_full_cannot_silently_run_on_local_cpu(tmp_path,dispatch,monkeypatch):
    monkeypatch.delenv("TDN_EXECUTION_MODE",raising=False)
    with pytest.raises(ValueError,match="native allocated"):
        engine.run_stage(build_protocol("full"),"audit",tmp_path/"full")


@pytest.mark.parametrize("profile",["smoke","development","full"])
def test_predeclared_capacity_and_teacher_endpoints(profile):
    p=build_protocol(profile);validate_protocol(p)
    assert set(p["evaluation_horizons"])<=set(p["validation_horizons"])
    assert len(p["gpu_test_cases"])==71
    assert p["portfolio_budget"]["fractions"]["C"]==.2
    from tdn.analysis.portfolio.models import MODEL_SPECS
    from tdn.analysis.portfolio.protocol import FROZEN
    assert set(FROZEN)=={name for name,spec in MODEL_SPECS.items() if spec["fit"]=="frozen"}
    p["units"]["audit"]["seconds"]+=1
    with pytest.raises(ValueError,match="declaration changed"):validate_protocol(p)
