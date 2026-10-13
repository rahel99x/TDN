"""Explicit scheduler stubs validate archive safety, never native readiness."""
from __future__ import annotations

import getpass
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archive_module():
    spec = importlib.util.spec_from_file_location("diagnostic_archive_test", ROOT/"scripts/response_diagnostic_archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path/"project with spaces"
    root.mkdir()
    token = "20261018T120000-abcdef123456"
    smoke, full = "response-diagnostic-smoke-"+token, "response-diagnostic-"+token
    profile = dict(schema_version=1,kind="desktop-slurm",root=str(root),user=getpass.getuser(),
        cpu_partition="local",gpu_partition="local",account=None,gpu_gres="gpu:1",
        expected_gpu_name="RTX 4090",gpu_vram_gib=24,torch_version="2.10.0+cu126",
        torch_wheel_index="https://download.pytorch.org/whl/cu126")
    receipt = root/"runs/.response-diagnostic-campaigns"/token/"receipt.json"
    write(receipt,dict(schema="tdn.response-diagnostic-campaign/v1",project_root=str(root),status="FAILED",
        created_at="2026-10-18T12:00:00+00:00",smoke_run_id=smoke,diagnostic_run_id=full))
    for name, kind, job in [(smoke,"diagnostic-smoke","101"),(full,"diagnostic","102")]:
        base=root/"runs"/name
        write(base/"response-workflow.json",dict(kind="desktop-slurm-response",run_id=name,run_dir=str(base),
            profile=kind,slurm_profile=profile,slurm_profile_sha256="a"*64,
            source_sha256="historical-source",source_tree_sha256="another-historical-source",
            scientific_protocol={"units":{"audit":{"kind":"audit"},"train":{"kind":"train"}}}))
        write(base/"jobs.json",[{"stage":"audit","job_id":job,"submitted_at":"2026-10-18T12:01:00Z"}])
        write(base/"state/audit.json",{"status":"INTERRUPTED","original_error":"cancelled"})
        (base/"original-checkpoint.bin").write_bytes(b"\0original checkpoint bytes\xff")
    write(root/"runs/.fedora-response-diagnostic-latest.json",{"run_id":full})
    write(root/"runs/.fedora-response-latest.json",{"run_id":"historical-pilot"})
    (root/"runs/unrelated").mkdir()
    (root/"runs/unrelated/evidence.txt").write_text("keep unrelated")
    return dict(root=root,token=token,smoke=smoke,full=full,profile=profile,receipt=receipt)


class Scheduler:
    def __init__(self, states=None, edit=None):
        self.states=states or {};self.calls=0;self.edit=edit

    def observe(self,jobs,*,user,root,run_ids):
        self.calls+=1
        if self.edit:self.edit(self.calls)
        return {job["job_id"]:dict(job_id=job["job_id"],state=self.states.get(job["job_id"],"CANCELLED"),
            user=user,account="",workdir=str(root),job_name=job["expected_name"],comment=job["expected_comment"])
            for job in jobs}


def run_archive(module,campaign,**kwargs):
    return module.archive(campaign["root"],site=campaign["profile"],scheduler=kwargs.pop("scheduler",Scheduler()),**kwargs)


def test_plan_is_confined_and_lists_both_selected_runs(archive_module,campaign):
    result=run_archive(archive_module,campaign,plan=True)
    assert result["action"]=="PLAN_ONLY" and result["submitted_jobs"]==2
    assert result["scheduler_state_counts"]=={"CANCELLED":2}
    assert campaign["receipt"].is_file()
    assert not (campaign["root"]/"runs/archived-response-diagnostics").exists()
    assert {m["kind"] for m in result["moves"]}=={"native_run","campaign_receipt_directory","diagnostic_latest_pointer"}


def test_cancelled_partial_science_archives_all_bytes_and_keeps_unrelated(archive_module,campaign):
    original={p.relative_to(campaign["root"]):p.read_bytes() for p in campaign["root"].rglob("*") if p.is_file()}
    result=run_archive(archive_module,campaign)
    assert result["status"]=="ARCHIVED"
    destination=campaign["root"]/result["archive_directory"]
    for move in result["moves"]:
        source=Path(move["original"]);target=destination/move["archived"]
        assert not (campaign["root"]/source).exists()
        if target.is_dir():
            for relative,raw in original.items():
                if relative.is_relative_to(source):assert (target/relative.relative_to(source)).read_bytes()==raw
        else:assert target.read_bytes()==original[source]
    assert (campaign["root"]/"runs/unrelated/evidence.txt").read_text()=="keep unrelated"
    assert (campaign["root"]/"runs/.fedora-response-latest.json").is_file()
    assert (destination/"scheduler-proof.json").is_file()
    assert result["native_source_fingerprints"]=="PRESERVED_WITHOUT_CURRENT_SOURCE_COMPARISON"


@pytest.mark.parametrize("state",["RUNNING","PENDING","COMPLETING","UNKNOWN"])
def test_live_or_unknown_actual_jobs_block_all_moves(archive_module,campaign,state):
    with pytest.raises(ValueError,match="terminal"):
        run_archive(archive_module,campaign,scheduler=Scheduler({"101":state}))
    assert campaign["receipt"].is_file()
    assert not (campaign["root"]/"runs/archived-response-diagnostics").exists()


def test_running_receipt_prevents_future_diagnostic_submission_race(archive_module,campaign):
    value=json.loads(campaign["receipt"].read_text());value["status"]="RUNNING";write(campaign["receipt"],value)
    with pytest.raises(ValueError,match="receipt must be terminal"):
        run_archive(archive_module,campaign)


def test_partially_submitted_only_smoke_preserves_missing_full_identity(archive_module,campaign):
    # Move the fixture full root out of the live path to model no submission.
    full=campaign["root"]/"runs"/campaign["full"]
    full.rename(campaign["root"]/"saved-fixture")
    result=run_archive(archive_module,campaign,run_id=campaign["smoke"])
    assert result["absent_run_ids"]==[campaign["full"]] and result["submitted_jobs"]==1
    assert (campaign["root"]/"saved-fixture").is_dir()


def test_empty_job_inventory_still_requires_live_scheduler_census(archive_module,campaign):
    for name in (campaign["smoke"],campaign["full"]):write(campaign["root"]/"runs"/name/"jobs.json",[])
    scheduler=Scheduler()
    result=run_archive(archive_module,campaign,scheduler=scheduler,plan=True)
    assert result["submitted_jobs"]==0 and scheduler.calls==1


def test_foreign_scheduler_ownership_cannot_archive(archive_module,campaign):
    class Foreign(Scheduler):
        def observe(self,*args,**kwargs):
            rows=super().observe(*args,**kwargs);rows["101"]["comment"]="another-project";return rows
    with pytest.raises(ValueError,match="ownership"):
        run_archive(archive_module,campaign,scheduler=Foreign())


def test_symlinked_metadata_and_receipt_identity_mismatch_block(archive_module,campaign):
    path=campaign["root"]/"runs"/campaign["smoke"]/"jobs.json"
    moved=path.with_name("original-jobs.json");path.rename(moved);path.symlink_to(moved.name)
    with pytest.raises(ValueError,match="symlink"):
        run_archive(archive_module,campaign,plan=True)


def test_other_diagnostic_latest_pointer_is_not_moved(archive_module,campaign):
    pointer=campaign["root"]/"runs/.fedora-response-diagnostic-latest.json"
    write(pointer,{"run_id":"a-different-diagnostic"})
    result=run_archive(archive_module,campaign)
    assert pointer.is_file() and all(m["kind"]!="diagnostic_latest_pointer" for m in result["moves"])


def raven(campaign,state="FAILED"):
    token="raven-token"
    name="launch-"+token+"-a1"
    receipt=json.loads(campaign["receipt"].read_text())
    receipt["raven_launch"]=dict(project_root=str(campaign["root"]),coordinator_run_id=name,
        coordinator_run_dir="runs/"+name,launch_token=token)
    write(campaign["receipt"],receipt)
    base=campaign["root"]/"runs"/name
    write(base/"run.json",dict(schema="raven.run/v1",run_id=name,state=state,
        provenance={"script":"scripts/response_diagnostic_campaign.sh"},metadata={"automation":{
            "execution":"local","project_root":str(campaign["root"]),"plan":{
                "token":token,"project_root":str(campaign["root"]),"script_relative":"scripts/response_diagnostic_campaign.sh"}}}))
    write(base/"summary.json",dict(schema="raven.summary/v1",id=name,state=state))
    return base


def test_exact_terminal_raven_coordinator_is_recorded_but_preserved(archive_module,campaign):
    outer=raven(campaign)
    before=(outer/"run.json").read_bytes()
    result=run_archive(archive_module,campaign)
    assert result["raven_coordinator"]["action"]=="PRESERVED"
    assert (outer/"run.json").read_bytes()==before


def test_nonterminal_raven_coordinator_blocks_native_moves(archive_module,campaign):
    raven(campaign,"RUNNING")
    with pytest.raises(ValueError,match="matching terminal"):
        run_archive(archive_module,campaign)


def test_second_inspection_detects_changed_receipt_before_mutation(archive_module,campaign):
    def edit(call):
        if call==2:
            value=json.loads(campaign["receipt"].read_text());value["events"]=[{"new":"changed"}];write(campaign["receipt"],value)
    # Mutation happens during second scheduler read; fingerprints are rechecked below.
    with pytest.raises(ValueError,match="changed"):
        run_archive(archive_module,campaign,scheduler=Scheduler(edit=edit))
    assert campaign["receipt"].is_file()


def test_move_failure_rolls_back_without_deleting_originals(archive_module,campaign,monkeypatch):
    original=archive_module.rename_noreplace;count=0
    def rename(root,source,target,expected_identity=None):
        nonlocal count
        count+=1
        if count==2:raise OSError("injected rename failure")
        return original(root,source,target,expected_identity)
    monkeypatch.setattr(archive_module,"rename_noreplace",rename)
    with pytest.raises(OSError,match="injected"):
        run_archive(archive_module,campaign)
    assert campaign["receipt"].is_file()
    assert (campaign["root"]/"runs"/campaign["smoke"]/"original-checkpoint.bin").is_file()
    result=list((campaign["root"]/"runs/archived-response-diagnostics").glob("*/archive-manifest.json"))
    assert len(result)==1 and json.loads(result[0].read_text())["status"]=="INTERRUPTED_PRESERVED"


def test_native_scheduler_reconciles_unknown_live_job_and_exact_accounting(archive_module,monkeypatch):
    scheduler=archive_module.NativeScheduler()
    monkeypatch.setattr(scheduler,"command",lambda _:"77|tdn-response:selected:audit:hash|tdn-response-audit|/home/rahel/project with spaces\n")
    with pytest.raises(ValueError,match="live submitted"):
        scheduler.observe([],user="rahel",root=Path("/home/rahel/project with spaces"),run_ids=["selected"])
    values=iter(["", "101|CANCELLED by 1000|rahel||tdn-response-audit|exact-comment|/home/rahel/project with spaces|\n101.batch|CANCELLED|||||\n"])
    monkeypatch.setattr(scheduler,"command",lambda _:next(values))
    row=scheduler.observe([{"job_id":"101"}],user="rahel",root=Path("/home/rahel/project with spaces"),run_ids=["selected"])
    assert row["101"]["workdir"]=="/home/rahel/project with spaces"


def test_absent_historical_comment_records_coverage_without_relaxing_owner(archive_module,campaign):
    class MissingComment(Scheduler):
        def observe(self,*args,**kwargs):
            rows=super().observe(*args,**kwargs)
            for row in rows.values():row["comment"]=""
            return rows
    result=run_archive(archive_module,campaign,plan=True,scheduler=MissingComment())
    assert result["scheduler_comment_coverage"]=={"NOT_RETAINED_BY_SCHEDULER":2}


def test_scontrol_fills_absent_comment_and_keeps_spaced_workdir(archive_module,monkeypatch):
    from types import SimpleNamespace
    scheduler=archive_module.NativeScheduler()
    values=iter(["","101|CANCELLED|rahel||tdn-response-audit||/home/rahel/project with spaces|\n"])
    monkeypatch.setattr(scheduler,"command",lambda _:next(values))
    monkeypatch.setattr(archive_module.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,
        stdout="JobId=101 JobName=tdn-response-audit UserId=rahel(1000) Account=(null) Comment=retained-comment WorkDir=/home/rahel/project with spaces JobState=CANCELLED"))
    row=scheduler.observe([{"job_id":"101"}],user="rahel",root=Path("/home/rahel/project with spaces"),run_ids=["selected"])
    assert row["101"]["comment"]=="retained-comment" and row["101"]["comment_coverage"]=="VERIFIED"


def test_retained_live_lifecycle_cannot_override_terminal_accounting(archive_module,monkeypatch):
    from types import SimpleNamespace
    scheduler=archive_module.NativeScheduler()
    values=iter(["","101|CANCELLED|rahel||tdn-response-audit||/home/rahel/project with spaces|\n"])
    monkeypatch.setattr(scheduler,"command",lambda _:next(values))
    monkeypatch.setattr(archive_module.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,
        stdout="JobId=101 JobName=tdn-response-audit UserId=rahel(1000) JobState=RUNNING Comment=retained-comment WorkDir=/home/rahel/project with spaces"))
    with pytest.raises(ValueError,match="lifecycle"):
        scheduler.observe([{"job_id":"101"}],user="rahel",root=Path("/home/rahel/project with spaces"),run_ids=["selected"])


def test_atomic_rename_does_not_overwrite_concurrently_created_target(archive_module,tmp_path):
    root=tmp_path/"root";root.mkdir()
    source=root/"source";target=root/"target"
    source.write_bytes(b"source");target.write_bytes(b"existing sentinel")
    with pytest.raises(FileExistsError):archive_module.rename_noreplace(root,source,target)
    assert source.read_bytes()==b"source" and target.read_bytes()==b"existing sentinel"


def test_rollback_preserves_recreated_source_and_retains_original_in_archive(archive_module,campaign,monkeypatch):
    original=archive_module.rename_noreplace;count=0
    def rename(root,source,target,expected_identity=None):
        nonlocal count
        count+=1
        if count==2:
            recreated=root/"runs"/campaign["smoke"]
            recreated.mkdir();(recreated/"new.txt").write_text("new sentinel")
            raise OSError("injected race")
        return original(root,source,target,expected_identity)
    monkeypatch.setattr(archive_module,"rename_noreplace",rename)
    with pytest.raises(OSError):run_archive(archive_module,campaign)
    assert (campaign["root"]/"runs"/campaign["smoke"]/"new.txt").read_text()=="new sentinel"
    manifests=list((campaign["root"]/"runs/archived-response-diagnostics").glob("*/archive-manifest.json"))
    value=json.loads(manifests[0].read_text())
    assert len(value["retained_archive_paths"])==1
    assert (campaign["root"]/value["retained_archive_paths"][0]/"original-checkpoint.bin").is_file()


def test_rollback_replacement_error_is_journaled_and_other_roots_restore(archive_module,campaign,monkeypatch):
    original=archive_module.rename_noreplace;count=0
    def rename(root,source,target,expected_identity=None):
        nonlocal count
        count+=1
        if count==3:raise OSError("stop after two moves")
        if count==4:raise ValueError("archived root identity replaced")
        return original(root,source,target,expected_identity)
    monkeypatch.setattr(archive_module,"rename_noreplace",rename)
    with pytest.raises(OSError):run_archive(archive_module,campaign)
    manifests=list((campaign["root"]/"runs/archived-response-diagnostics").glob("*/archive-manifest.json"))
    value=json.loads(manifests[0].read_text())
    assert value["status"]=="INTERRUPTED_PRESERVED" and len(value["retained_archive_paths"])==1
    assert (campaign["root"]/"runs"/campaign["smoke"]).is_dir()
    assert campaign["receipt"].is_file()


def test_shell_syntax():
    result=subprocess.run(["bash","-n",str(ROOT/"scripts/response_diagnostic_archive.sh")],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
