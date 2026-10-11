import json

import pytest

from tdn.analysis.resolve import engine
from tdn.analysis.resolve.core import check
from tdn.analysis.resolve.protocol import build_protocol
from tdn.runtime.metadata import write_json


@pytest.fixture
def isolated_engine(monkeypatch):
    # Fixed metadata isolates the provenance contract from concurrent editor work.
    monkeypatch.setattr(engine,"software_metadata",lambda:dict(source_tree_sha256="a"*64,
        git_commit="b"*40,python="test",torch="test",numpy="test",scipy="test",venv=True,
        executable="project-venv",torch_cuda_runtime=None))
    def dispatch(ctx):
        ctx.record("measured",["A1"],metrics={"observed":1},checks=[
            check("identity",0,0,category="math"),check("scientific",None,True,"eq",category="gap")])
        write_json(ctx.path/"evidence.json",{"value":1})
        return {}
    monkeypatch.setattr(engine,"_dispatch",dispatch)
    return build_protocol("smoke")


def test_resolve_science_seal_rejects_changed_or_extra_artifacts(tmp_path,isolated_engine):
    path=tmp_path/"audit";engine.run_stage(isolated_engine,"audit",path)
    assert engine.verify_science(isolated_engine,path)["stage"]=="audit"
    write_json(path/"evidence.json",{"value":2})
    with pytest.raises(ValueError,match="inventory or bytes changed"):
        engine.verify_science(isolated_engine,path)
    write_json(path/"evidence.json",{"value":1})
    (path/"unexpected.txt").write_text("added")
    with pytest.raises(ValueError,match="inventory or bytes changed"):
        engine.verify_science(isolated_engine,path)


def test_resolve_failed_attempt_retains_rows_without_completion(tmp_path,isolated_engine,monkeypatch):
    original=engine._dispatch
    def interrupted(ctx):
        original(ctx);raise TimeoutError("bounded interruption")
    monkeypatch.setattr(engine,"_dispatch",interrupted)
    path=tmp_path/"audit"
    with pytest.raises(TimeoutError):engine.run_stage(isolated_engine,"audit",path)
    assert json.loads((path/"summary.json").read_text())["status"]=="INTERRUPTED"
    assert len((path/"rows.jsonl").read_text().splitlines())==1
    assert not (path/"COMPLETED").exists() and not (path/engine.MANIFEST).exists()
    with pytest.raises(FileExistsError):engine.run_stage(isolated_engine,"audit",path)


def test_resolve_journal_resume_binds_prerequisites_and_retains_ledger(tmp_path,isolated_engine,monkeypatch):
    p=isolated_engine;audit=tmp_path/"audit";prep=tmp_path/"prepare";fit=tmp_path/"fit"
    engine.run_stage(p,"audit",audit)
    engine.run_stage(p,"prepare",prep,prerequisites={"audit":audit})
    original=engine._dispatch
    def interrupted(ctx):
        original(ctx);raise InterruptedError("signal")
    monkeypatch.setattr(engine,"_dispatch",interrupted)
    with pytest.raises(InterruptedError):
        engine.run_stage(p,"fit-discrete",fit,prerequisites={"prepare":prep})
    def continued(ctx):
        assert ctx.resume and "measured" in ctx.identities
        ctx.record("after-resume",["R1"],metrics={},checks=[check("identity",0,0,category="math")])
        return {}
    monkeypatch.setattr(engine,"_dispatch",continued)
    engine.run_stage(p,"fit-discrete",fit,prerequisites={"prepare":prep},resume=True)
    assert engine.verify_science(p,fit)["stage"]=="fit-discrete"
    assert json.loads((fit/"summary.json").read_text())["experiment_count"]==2
    with pytest.raises(FileExistsError):
        engine.run_stage(p,"fit-discrete",fit,prerequisites={"prepare":prep},resume=True)


def test_resolve_full_refuses_unallocated_execution(tmp_path,monkeypatch):
    monkeypatch.delenv("TDN_EXECUTION_MODE",raising=False)
    with pytest.raises(ValueError,match="native allocated"):
        engine.run_stage(build_protocol("full"),"audit",tmp_path/"audit")


def test_resolve_partial_report_does_not_require_failed_science(tmp_path,isolated_engine):
    result=engine.run_stage(isolated_engine,"report",tmp_path/"report",
        prerequisites={},stage_failures={"audit":{"status":"FAILED"}})
    assert result["status"]=="COMPLETED"
    assert result["scientific_outcome"].startswith("DEVELOPMENT_EVIDENCE_ONLY")
    assert engine.verify_science(isolated_engine,tmp_path/"report")["prerequisites"]=={}
