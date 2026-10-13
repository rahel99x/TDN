"""Tangent-only adaptation, negative reports, native readers and moved bundles."""
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from tdn.analysis.response import diagnostic_metrics as old
from tdn.analysis.response import diagnostic_report as transport
from tdn.analysis.tangent_response import report, report_artifacts
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json


def ctx(tmp_path):
    root = tmp_path/"tangent-response-cpu-a"
    protocol = dict(profile="cpu-smoke", stages=["mechanism", "compression", "report"],
        units={"mechanism":dict(kind="mechanism",device="cpu"),"compression":dict(kind="compression",device="cpu"),"report":dict(kind="report",device="cpu")})
    path = root/"report";path.mkdir(parents=True)
    return SimpleNamespace(path=path, stage="report", protocol=protocol, unit=protocol["units"]["report"],
                           device="cpu", sourcehash="a"*64, prereqs={})


def sealed(context, stage, payload):
    base = context.path.parent/stage;base.mkdir()
    payload = dict(payload, **{"summary.json":dict(status="COMPLETED",elapsed_seconds=.1,device="cpu")})
    hashes = {}
    for name, value in payload.items():
        write_json(base/name, value);hashes[name] = hashlib.sha256((base/name).read_bytes()).hexdigest()
    (base/"checkpoint.pt").write_bytes(b"binary replay is never loaded")
    hashes["checkpoint.pt"] = hashlib.sha256((base/"checkpoint.pt").read_bytes()).hexdigest()
    manifest = dict(schema="tdn.tangent-response-science/v1",stage=stage,profile=context.protocol["profile"],
        protocol_sha256=digest(context.protocol),source_tree_sha256="a"*64,artifacts=hashes)
    write_json(base/"science_manifest.json",manifest)
    (base/"COMPLETED").write_text(hashlib.sha256((base/"science_manifest.json").read_bytes()).hexdigest()+"\n")
    context.prereqs[stage] = base
    return base


def native(monkeypatch):
    root = Path(os.environ.get("TDN_RAVEN_TEST_ROOT", "/workspace/Raven"))
    if not (root/"raven/projects.py").is_file():pytest.skip("Trusted optional Raven checkout unavailable")
    monkeypatch.syspath_prepend(str(root))
    return [importlib.import_module("raven."+name) for name in ("projects", "reporting", "metrics", "artifacts")]


def test_negative_mechanism_report_retains_unrun_methods_and_all_source_rows(tmp_path, monkeypatch):
    context = ctx(tmp_path)
    sealed(context, "mechanism", {"mechanism_results.json":dict(rows=[dict(query_id="q",parent_id="p",status="UNRESOLVED",
        controls={"anchor":dict(rms=.1,maximum=.2),"G":dict(rms=.05,maximum=.1),"U":dict(rms=.06,maximum=.12)},
        reference=dict(accepted=False,uncertainty_rms=.03))],coverage=dict(requested_queries=2,recorded_queries=1,missing_queries=1)),
        "decision.json":dict(gate="A",decision="STOP",eligible=False,denominators=dict(requested=2,eligible=0))})
    # Binary loaders are forbidden even for project-owned checkpoint bytes.
    import torch
    monkeypatch.setattr(torch, "load", lambda *a,**kw:pytest.fail("Report loaded a checkpoint"))
    report.start_attempt(context)
    result = report.run(context)
    assert result["scientific_decisions"][0]["decision"] == "STOP"
    assert result["research"]["missing_units"] == 1
    project = old.raven_root(context)
    overview = next(p for p in (project/"runs").iterdir() if p.name.endswith("-analysis"))
    data = json.loads((overview/"outputs/tangent_research.json").read_text())
    methods = {r["method"]:r for r in report_artifacts.read_rows(overview/"outputs",data,"method_coverage")}
    assert methods["learned_response"]["status"] == "NOT_RUN_OR_MISSING"
    assert data["tables"]["endpoint_controls"]["rows"] == 3
    assert any(row["path"] == "checkpoint.pt" for row in report_artifacts.read_section(overview/"outputs",data,"omissions"))
    assert data["native_gpu_verification"] == "NOT_INFERRED_FROM_REPORTING"
    manifest = transport.verify_portable(context.path/report.BUNDLE)
    assert manifest["excluded_attempts"]
    assert not any(row["path"].endswith(".pt") for row in manifest["files"])


def test_actual_native_raven_parsers_read_moved_tangent_project(tmp_path, monkeypatch):
    projects, reporting, metrics, artifacts = native(monkeypatch)
    context = ctx(tmp_path)
    sealed(context,"mechanism",{"mechanism_results.json":dict(rows=[dict(query_id="q",rms=.01,maximum=.02)]),
        "decision.json":dict(gate="A",decision="IMPLEMENTATION_ONLY",eligible=False)})
    report.start_attempt(context);report.run(context)
    destination = tmp_path/"unrelated-machine";destination.mkdir()
    with zipfile.ZipFile(context.path/report.BUNDLE) as archive:
        for name in archive.namelist():
            old.relative_path(name);file = destination/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(archive.read(name))
    discovered = projects.discover_project(str(destination))
    assert not discovered["warnings"]
    for entry in discovered["runs"]:
        run = projects.validate_inventory(entry["inventory"]);attempt = destination/"runs"/run["run_id"]
        assert run["project_id"] == "TDN-tangent-response" and run.get("job_id") is None
        reporting._summary(json.loads((attempt/"summary.json").read_text()))
        for line in (attempt/"metrics.jsonl").read_text().splitlines():metrics._record(json.loads(line))
        assert artifacts.validate_contract(json.loads((attempt/"outputs/contract.json").read_text()),str(attempt))["valid"]
        output=attempt/"outputs"
        if not (output/"tangent_research.json").is_file():continue
        index=json.loads((output/"tangent_research.json").read_text())
        for page in index["artifact_projects"]:
            page_root=output/page["path"];inventory=projects.discover_project(str(page_root))
            assert not inventory["warnings"] and len(inventory["runs"])<=255
            for entry in inventory["runs"]:
                run=projects.validate_inventory(entry["inventory"]);child=page_root/"runs"/run["run_id"]
                assert run.get("job_id") is None
                assert artifacts.validate_contract(json.loads((child/"outputs/contract.json").read_text()),str(child))["valid"]


def test_tangent_configuration_preserves_response_defaults_and_pages(tmp_path, monkeypatch):
    context = ctx(tmp_path)
    units = {f"unit-{i:03}":dict(kind="mechanism") for i in range(600)}
    context.protocol.update(units=units,stages=list(units));context.stage="unit-256";context.path=context.path.parent/context.stage
    path = report.start_attempt(context,job_id="123",worker_managed=True)
    run = json.loads((path/"run.json").read_text())
    assert run["project_id"] == "TDN-tangent-response" and path.parent.parent.name == "page-001"
    assert "Response diagnostic" not in (old.raven_root(context)/"README.md").read_text()
    legacy = SimpleNamespace(path=tmp_path/"response"/"unit",stage="unit",device="cpu",
        protocol=dict(profile="diagnostic",units={"unit":dict(kind="train")},source_tree_sha256="b"*64))
    previous = old.start_attempt(legacy)
    assert json.loads((previous/"run.json").read_text())["project_id"] == "TDN-response-diagnostic"
    assert (old.raven_root(legacy)/"README.md").read_text().startswith("# Response diagnostic")


def test_tampered_science_source_is_visible_unresolved_not_promoted(tmp_path):
    context = ctx(tmp_path);base = sealed(context,"mechanism",{"decision.json":dict(gate="A",decision="PASS",eligible=True)})
    (base/"decision.json").write_text('{}')
    data, _ = report.collect(context)
    assert data["summary"]["unresolved_source_units"] == 1
    assert not data["decisions"]


def inherited(context, tmp_path):
    """Controlled native binding metadata; it does not establish CUDA readiness."""
    ancestor = ctx(tmp_path/"ancestor")
    ancestor.protocol["phase"] = "mechanism"
    base = sealed(ancestor, "mechanism", {"protocol.json":ancestor.protocol,
        "mechanism_results.json":dict(rows=[dict(query_id="q-A", controls={"G":dict(rms=.01,maximum=.02)})]),
        "decision.json":dict(gate="A",decision="PASS",eligible=True)})
    root = tmp_path;run = base.parent;state = run/"state";state.mkdir()
    worker = state/"mechanism-worker.json"
    write_json(worker, dict(status="COMPLETED",source_sha256="b"*64,
        protocol_sha256=digest(ancestor.protocol),slurm_job_id="controlled-fixture"))
    software = state/"submission-software.json";write_json(software, dict(scope="controlled metadata fixture"))
    relative = worker.relative_to(root).as_posix()
    write_json(base/"workflow-seal.json", dict(schema="tdn.tangent-response-workflow-seal/v1",stage="mechanism",
        status="COMPLETED",source_tree_sha256="a"*64,protocol_sha256=digest(ancestor.protocol),
        slurm_profile_sha256="c"*64,slurm_job_id="controlled-fixture",worker_state_path=relative,
        science_manifest_sha256=hashlib.sha256((base/"science_manifest.json").read_bytes()).hexdigest(),
        artifacts={relative:hashlib.sha256(worker.read_bytes()).hexdigest()}))
    workflow = run/"native-workflow.json"
    write_json(workflow, dict(root=str(root),run_dir=str(run),run_id=run.name,phase="mechanism",profile=ancestor.protocol["profile"],
        scientific_protocol=ancestor.protocol,source_sha256="b"*64,source_tree_sha256="a"*64,
        protocol_sha256=digest(ancestor.protocol),slurm_profile_sha256="c"*64))
    context.protocol["prerequisites"] = {"mechanism":dict(run_id=run.name,phase="mechanism",profile=ancestor.protocol["profile"],
        workflow_path=str(workflow),workflow_sha256=hashlib.sha256(workflow.read_bytes()).hexdigest(),
        protocol_sha256=digest(ancestor.protocol),source_sha256="b"*64,source_tree_sha256="a"*64,slurm_profile_sha256="c"*64,
        software_sha256=hashlib.sha256(software.read_bytes()).hexdigest(),stage_paths={"mechanism":str(base)},
        seals={"mechanism":hashlib.sha256((base/"workflow-seal.json").read_bytes()).hexdigest()})}
    context.prereqs["mechanism/mechanism"] = base
    return base, worker


def test_inherited_evidence_uses_ancestor_protocol_and_exact_native_seals(tmp_path):
    context = ctx(tmp_path/"child");base, worker = inherited(context, tmp_path)
    assert digest(context.protocol) != digest(json.loads((base/"protocol.json").read_text()))
    data, _ = report.collect(context)
    assert data["decisions"][0]["gate"] == "A"
    assert data["decisions"][0]["stage"] == "mechanism/mechanism"
    assert data["decisions"][0]["native_job_id"] == "controlled-fixture"
    row = next(r for r in data["tables"]["units"] if r["stage"] == "mechanism/mechanism")
    assert row["status"] == "COMPLETED"
    worker.write_text('{}')
    invalid, _ = report.collect(context)
    assert not invalid["decisions"]
    assert next(r for r in invalid["tables"]["units"] if r["stage"] == "mechanism/mechanism")["status"] == "UNRESOLVED_SOURCE"


def test_missing_inherited_stage_is_explicit_and_never_promoted(tmp_path):
    context = ctx(tmp_path/"child");inherited(context, tmp_path)
    context.prereqs.clear()
    data, _ = report.collect(context)
    assert data["summary"]["unresolved_source_units"] == 1 and not data["decisions"]


def test_terminal_telemetry_contract_and_physical_learning_resource_plots(tmp_path, monkeypatch):
    import time
    from tdn.analysis.tangent_response import telemetry
    _, _, _, artifacts = native(monkeypatch)
    context = ctx(tmp_path)
    context.stage = "mechanism";context.unit=context.protocol["units"]["mechanism"]
    context.path=context.path.parent/"mechanism";context.path.mkdir()
    attempt = report.start_attempt(context)
    handle = telemetry.start_recording(attempt, {"enabled":["cpu", "memory"],"sampling_seconds":.03}, environ={})
    time.sleep(.1);telemetry.stop_recording(handle, state="FAILED")
    report.finish_attempt(context, dict(status="FAILED",elapsed_seconds=.1))
    contract = json.loads((attempt/"outputs/contract.json").read_text())
    assert artifacts.validate_contract(contract,str(attempt),max_bytes=64<<20)["valid"]
    assert any(row["path"] == "telemetry/cpu-memory.csv" for row in contract["outputs"])
    logs = json.loads((attempt/"logs.json").read_text())
    assert any(row["group"] == "Resources" for row in logs["logs"])
    output = tmp_path/"plots";output.mkdir()
    figures = report._plots(output, dict(learning_curves=[dict(update=1,full_train_rms=.01,validation_rms=.02,
        full_train_maximum=.04,gradient_norm=2.,train_loss=.1)],
        worker_resource_samples=[dict(stage="controlled",elapsed_seconds=1.,cpu_utilization_percent=110.,rss_bytes=1000,
            utilization_percent=42.,memory_used_mib=100.,power_watts=90.,temperature_celsius=49.)]))
    assert {"observed-learning_curves-physical-rms.png","observed-learning_curves-gradient.png",
            "resource-cpu_utilization_percent.png","resource-power_watts.png","resource-temperature_celsius.png"} <= set(figures)


def recovery_lineage(tmp_path):
    """Controlled allocated-worker metadata; no scheduler/CUDA claim is made."""
    from tdn.analysis.tangent_response import telemetry
    root=tmp_path/"project";root.mkdir()
    protocol=dict(profile="pilot",execution_scope="NATIVE_RESEARCH",stages=["mechanism","report"],
        units={"mechanism":dict(kind="mechanism",device="cpu"),"report":dict(kind="report",device="cpu")})
    profile=dict(user="controlled-user",scope="Controlled native metadata fixture")
    software=dict(source_tree_sha256="a"*64,scope="Controlled software fixture")
    origin=None
    for index,state in enumerate(("FAILED","COMPLETED")):
        name="tangent-response-controlled-"+str(index);run=root/"runs"/name;run.mkdir(parents=True)
        (run/"state").mkdir();path=run/"mechanism";path.mkdir()
        write_json(run/"protocol.json",protocol);write_json(run/"slurm-profile.json",profile)
        write_json(run/"state/submission-software.json",software)
        job=str(101+index);write_json(run/"jobs.json",[dict(stage="mechanism",job_id=job)])
        write_json(run/"state/mechanism.json",dict(status=state,source_sha256="b"*64,
            protocol_sha256=digest(protocol),slurm_job_id=job))
        context=SimpleNamespace(path=path,stage="mechanism",protocol=protocol,unit=protocol["units"]["mechanism"],device="cpu",sourcehash="a"*64)
        attempt=report.start_attempt(context,job_id=job,worker_managed=True)
        handle=telemetry.start_recording(attempt,{"enabled":["cpu","memory"],"sampling_seconds":.02},environ={})
        import time
        time.sleep(.06);coverage=telemetry.stop_recording(handle,state=state)
        report.finish_attempt(context,dict(status=state,native_worker_finalized=True,native_worker_runtime_seconds=2.+index,
            native_worker_exit_code=0 if state=="COMPLETED" else 37,native_worker_error=None if state=="COMPLETED" else "Controlled failure",
            science_body_status="NOT_STARTED",worker_telemetry=coverage))
        workflow=dict(schema_version=1,kind="desktop-slurm-tangent-response",execution_mode="desktop-slurm",
            root=str(root),run_dir=str(run),run_id=name,phase="mechanism",profile="pilot",scientific_protocol=protocol,
            source_sha256="b"*64,source_tree_sha256="a"*64,protocol_sha256=digest(protocol),
            slurm_profile=profile,slurm_profile_sha256=digest(profile))
        if origin:workflow["recovery"]=dict(origin_run=origin.parent.name,origin_workflow=str(origin),
            origin_workflow_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),stage_paths={},resume_paths={},seals={})
        origin=run/"tangent-response-workflow.json";write_json(origin,workflow)
    path=run/"report";path.mkdir()
    return SimpleNamespace(path=path,stage="report",protocol=protocol,unit=protocol["units"]["report"],device="cpu",sourcehash="a"*64,prereqs={}),origin


def test_exact_recovery_lineage_retains_failed_attempt_costs_and_portable_readers(tmp_path, monkeypatch):
    projects, reporting, _, artifacts = native(monkeypatch)
    retained=importlib.import_module("raven.retained_telemetry_io")
    context,workflow=recovery_lineage(tmp_path)
    monkeypatch.setenv("TDN_TANGENT_WORKFLOW",str(workflow))
    report.start_attempt(context,job_id="103",worker_managed=True)
    report.run(context)
    overview=next(p for p in (old.raven_root(context)/"runs").iterdir() if p.name.endswith("-analysis"))
    data=json.loads((overview/"outputs/tangent_research.json").read_text())
    assert data["lineage_costs"]["workflows"] == 2 and not report_artifacts.read_rows(overview/"outputs",data,"lineage_errors")
    assert data["lineage_costs"]["observed_native_worker_seconds"] == 5.
    assert {r["worker_status"] for r in report_artifacts.read_rows(overview/"outputs",data,"native_attempts")} == {"FAILED","COMPLETED"}
    assert data["tables"]["worker_costs"]["rows"] == 2
    assert report_artifacts.read_section(overview/"outputs",data,"lineage_projects")[0]["attempts"][0]["job_id"] == "101"
    moved=tmp_path/"another-host";moved.mkdir()
    with zipfile.ZipFile(context.path/report.BUNDLE) as archive:archive.extractall(moved)
    imported=next((moved/"runs"/overview.name/"outputs/lineage").iterdir())
    inventory=projects.discover_project(str(imported))
    assert len(inventory["runs"]) == 1 and not inventory["warnings"]
    attempt=imported/"runs"/inventory["runs"][0]["inventory"]["run_id"]
    assert projects.validate_inventory(json.loads((attempt/"run.json").read_text()))["job_id"] == "101"
    assert reporting._summary(json.loads((attempt/"summary.json").read_text()))["state"] == "FAILED"
    assert artifacts.validate_contract(json.loads((attempt/"outputs/contract.json").read_text()),str(attempt),max_bytes=64<<20)["valid"]
    index=retained.validate_index(json.loads((attempt/"telemetry/index.json").read_text()),attempt.name)
    assert retained.parse_rows((attempt/index["sources"][0]["path"]).read_bytes(),index["sources"][0],attempt.name)


def test_changed_recovery_origin_is_reported_as_missing_cost_evidence(tmp_path, monkeypatch):
    context,workflow=recovery_lineage(tmp_path);monkeypatch.setenv("TDN_TANGENT_WORKFLOW",str(workflow))
    origin=Path(json.loads(workflow.read_text())["recovery"]["origin_workflow"]);origin.write_text('{}')
    data,_=report.collect(context)
    assert data["lineage_costs"]["cost_status"] == "INCOMPLETE"
    assert len(data["tables"]["lineage_errors"]) == 1
    assert data["lineage_costs"]["observed_native_worker_seconds"] == 3.


def test_bounded_artifact_pages_preserve_every_row_and_large_utf8_source(tmp_path, monkeypatch):
    projects,_,_,artifacts=native(monkeypatch)
    monkeypatch.setattr(report_artifacts,"PAGE_BYTES",2<<20)
    monkeypatch.setattr(report_artifacts,"CHUNK_BYTES",256<<10)
    context=ctx(tmp_path/"scientific");data,_=report.collect(context)
    raw=json.dumps(dict(text="é"*1200000),ensure_ascii=False,separators=(",",":")).encode()
    checksum=hashlib.sha256(raw).hexdigest();pointer="sources/"+checksum+".json"
    data["sources"]=[dict(path=pointer,sha256=checksum,bytes=len(raw),scope="Controlled serialization fixture")]
    rows=[dict(query_id="controlled-"+str(i),rms=i/10000,source_path=pointer,source_sha256=checksum,note="x"*500) for i in range(2500)]
    data["tables"]["controlled_rows"]=rows
    output=tmp_path/"publication";output.mkdir()
    index,published=report_artifacts.publish(output,data,{pointer:raw},report.CONFIG,"tangent-response-controlled")
    assert (output/"tangent_research.json").stat().st_size<=256<<10
    complete=report_artifacts.read_rows(output,index,"controlled_rows")
    assert len(complete)==len(rows)
    assert [(r["query_id"],r["rms"],r["note"]) for r in complete]==[(r["query_id"],r["rms"],r["note"]) for r in rows]
    source=report_artifacts.read_section(output,index,"sources")[0]
    assert (output/source["path"]).read_bytes()==raw
    segments=index["raw_source_segments"][source["path"]]
    assert b"".join((output/part["path"]).read_bytes() for part in segments)==raw
    assert all((output/part["path"]).read_bytes().decode("utf-8") for part in segments)
    for page in index["artifact_projects"]:
        root=output/page["path"];discovered=projects.discover_project(str(root));assert not discovered["warnings"]
        for record in discovered["runs"]:
            attempt=root/"runs"/record["inventory"]["run_id"]
            assert artifacts.validate_contract(json.loads((attempt/"outputs/contract.json").read_text()),str(attempt))["valid"]


@pytest.mark.parametrize("exit_code", [0, 37])
def test_actual_captured_tangent_bridge_preserves_raven_lifecycle_and_negative_gate(tmp_path, monkeypatch, exit_code):
    native(monkeypatch)
    launch = importlib.import_module("raven.launch")
    runtime = importlib.import_module("raven.launch_runtime")
    projects = importlib.import_module("raven.projects")
    reporting = importlib.import_module("raven.reporting")
    settings = importlib.import_module("raven.launch_settings")
    root = tmp_path/"TDN tangent project with spaces";scripts = root/"scripts";scripts.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    for name in ("response_diagnostic_raven.py", "tangent_response_raven.py"):
        shutil.copyfile(repository/"scripts"/name, scripts/name)
    (root/".raven").mkdir();shutil.copyfile(repository/".raven/automation.json", root/".raven/automation.json")
    assert settings.load_settings(root)["recipes"]["tangent-response"]["cpu_memory"] is False
    source = scripts/"tangent_response_campaign.sh"
    source.write_text('''#!/usr/bin/env bash
set -euo pipefail
"$RAVEN_REPORT_PYTHON" "$RAVEN_PROJECT_ROOT/scripts/tangent_response_raven.py" check --snapshot "${BASH_SOURCE[0]}"
"$RAVEN_REPORT_PYTHON" - "$RAVEN_PROJECT_ROOT" "$@" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);run=Path(os.environ['RAVEN_RUN_DIR']);plan=json.loads(Path(os.environ['RAVEN_LAUNCH_PLAN']).read_text())
directory=root/'runs/.tangent-response-campaigns/controlled';directory.mkdir(parents=True)
value=dict(schema='tdn.tangent-response-campaign/v1',run_id='tangent-response-controlled',project_root=str(root),
    run_ids={'mechanism':'tangent-response-controlled-mechanism'},gates={'A':{'decision':'STOP','eligible':False}},
    outcomes={},events=[],phase='MECHANISM_STOP',status='COMPLETED',argv=sys.argv[2:],
    raven_launch=dict(project_root=str(root),coordinator_run_id=run.name,coordinator_run_dir=run.relative_to(root).as_posix(),launch_token=plan['token']))
(directory/'receipt.json').write_text(json.dumps(value))
PY
receipt="$RAVEN_PROJECT_ROOT/runs/.tangent-response-campaigns/controlled/receipt.json"
"$RAVEN_REPORT_PYTHON" "$RAVEN_PROJECT_ROOT/scripts/tangent_response_raven.py" initialize --receipt "$receipt" --snapshot "${BASH_SOURCE[0]}"
printf 'controlled coordinator log; no native GPU work\n' > "$RAVEN_RUN_DIR/logs/campaign.stdout.log"
"$RAVEN_REPORT_PYTHON" "$RAVEN_PROJECT_ROOT/scripts/tangent_response_raven.py" finish --receipt "$receipt"
exit "${TANGENT_CONTROLLED_EXIT:-0}"
''')
    plan = launch.resolve_plan([str(source), "--project", str(root), "--local", "--recipe", "tangent-response",
        "--", "--profile", "pilot", "--worker-telemetry", "cpu,memory,gpu"], cwd=root, environ={})
    env = {k:v for k,v in os.environ.items() if not k.startswith(("RAVEN_", "SLURM_", "SBATCH_"))}
    env["TANGENT_CONTROLLED_EXIT"] = str(exit_code)
    receipt = runtime.execute_plan(plan, environ=env)
    assert receipt["exit_code"] == exit_code
    run = Path(receipt["run_dir"])
    inventory = projects.validate_inventory(json.loads((run/"run.json").read_text()))
    assert inventory["state"] == ("COMPLETED" if exit_code == 0 else "FAILED") and inventory.get("job_id") is None
    reporting._summary(json.loads((run/"summary.json").read_text()))
    scientific = json.loads((run/"outputs/decisions.json").read_text())
    assert scientific["gates"]["A"]["decision"] == "STOP"
    campaign = json.loads((run/"outputs/campaign.json").read_text())
    assert campaign["argv"] == ["--profile", "pilot", "--worker-telemetry", "cpu,memory,gpu"]
    paths = json.loads((run/"outputs/research_paths.json").read_text())
    assert paths["runs"]["mechanism"]["portable_research_bundle"].endswith("/report/tangent-response-raven.zip")
    assert (run/"logs/campaign.stdout.log").read_text().startswith("controlled coordinator log")
