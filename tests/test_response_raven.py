"""Raven native interchange, live identity, and portable relocated views."""
from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from tdn.analysis.response import diagnostic_metrics as dm
from tdn.analysis.response import diagnostic_report as dr
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json


def _ctx(tmp_path):
    path = tmp_path/"response-diagnostic-raven-example"/"training-unit"
    path.mkdir(parents=True)
    protocol = dict(profile="diagnostic", source_tree_sha256="a"*64,
        units={"training-unit":dict(kind="train",trial_id="one")},
        trials={"one":dict(family="temporal", variant_id="gain", seed=101, target_kind="fd_same_grid",phase="tiny_joint")})
    return SimpleNamespace(path=path, protocol=protocol, stage="training-unit", device="cpu")


def _completed(tmp_path, monkeypatch):
    monkeypatch.delenv("TDN_RESPONSE_RAVEN_DIR", raising=False)
    ctx = _ctx(tmp_path)
    path = dm.start_attempt(ctx, job_id="12345_7", sourcehash="a"*64)
    dm.emit_live({"train_loss":.125,"train_rms":None,"examples_seen":6}, phase="training", step=3,
        progress={"completed":3,"total":10,"unit":"optimizer updates"})
    summary = dict(status="COMPLETED", elapsed_seconds=1.5, experiment_count=1,
        scientific_outcome="Diagnostic only", details={"updates":3})
    (ctx.path/"summary.json").write_text(json.dumps(summary))
    dm.finish_attempt(ctx, summary)
    return ctx, path


@pytest.mark.parametrize("metrics,kwargs", [({"loss":float("nan")}, {}), ({"loss":True}, {}),
    ({"loss":1}, {"step":-1}), ({"loss":1},{"progress":{"completed":3,"total":2}}),
    ({"loss":1}, {"t":-1}), ({"bad\nname":1}, {})])
def test_invalid_records_never_emit(metrics, kwargs):
    with pytest.raises(ValueError):
        dm.metric_record(metrics, **kwargs)


def test_live_records_keep_exact_job_identity_and_missing_values(tmp_path, monkeypatch):
    ctx, path = _completed(tmp_path, monkeypatch)
    run = json.loads((path/"run.json").read_text())
    summary = json.loads((path/"summary.json").read_text())
    assert run["job_id"] == summary["job_id"] == "12345_7"
    assert run["state"] == summary["state"] == "COMPLETED"
    assert summary["runtime_seconds"] == 1.5
    assert "Numerical science stage body" in summary["metadata"]["runtime_scope"]
    records = [json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
    assert "train_rms" not in records[1]["metrics"]
    assert records[1]["progress"]["completed"] == 3
    assert "schema" not in records[1]
    assert (ctx.path/"summary.json").read_text().find("raven.summary") == -1
    assert "TDN_RESPONSE_RAVEN_DIR" not in os.environ


def test_local_attempt_has_no_invented_scheduler_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("TDN_RESPONSE_RAVEN_DIR", raising=False)
    ctx = _ctx(tmp_path)
    path = dm.start_attempt(ctx)
    assert json.loads((path/"run.json").read_text())["job_id"] is None
    assert "job_id" not in json.loads((path/"logs.json").read_text())


def test_changed_job_or_completed_attempt_requires_fresh_directory(tmp_path, monkeypatch):
    ctx, _ = _completed(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError, match="Preserve"):
        dm.start_attempt(ctx, job_id="new")


def _native_terminal(status="FAILED", runtime=9.5, code=1, **extra):
    # Controlled reporting metadata; this is not an allocation/CUDA readiness test.
    return dict(status=status, native_worker_finalized=True,
        native_worker_runtime_seconds=runtime, native_worker_exit_code=code,
        native_worker_error="worker guard failed" if code else None,
        science_body_status="NOT_STARTED", science_body_elapsed_seconds=None, **extra)


def test_native_preflight_failure_is_an_exact_attempt_without_science(tmp_path, monkeypatch):
    monkeypatch.delenv("TDN_RESPONSE_RAVEN_DIR", raising=False)
    ctx = _ctx(tmp_path)
    ctx.path.rmdir()
    ctx.protocol["source_tree_sha256"] = None
    path = dm.start_attempt(ctx, job_id="12345", worker_managed=True)
    assert not ctx.path.exists()
    dm.finish_attempt(ctx, _native_terminal(runtime=4.25, code=17))
    run = json.loads((path/"run.json").read_text())
    summary = json.loads((path/"summary.json").read_text())
    assert run["state"] == summary["state"] == "FAILED"
    assert run["job_id"] == summary["job_id"] == "12345"
    assert summary["runtime_seconds"] == 4.25 and summary["exit_code"] == 17
    assert summary["results"]["science_body_status"] == "NOT_STARTED"
    assert summary["results"].get("science_body_elapsed_seconds") is None
    assert summary["metadata"]["source_tree_sha256"] is None
    assert summary["metadata"]["native_gpu_verification"] == "NOT_INFERRED_FROM_REPORTING"
    assert not ctx.path.exists()


@pytest.mark.parametrize("worker_status,code", [("FAILED", 19), ("COMPLETED", 0)])
def test_native_terminal_worker_state_keeps_science_timer_and_refreshes_exact_logs(
        tmp_path, monkeypatch, worker_status, code):
    monkeypatch.delenv("TDN_RESPONSE_RAVEN_DIR", raising=False)
    ctx = _ctx(tmp_path)
    path = dm.start_attempt(ctx, job_id="12345", worker_managed=True)
    started = json.loads((path/"run.json").read_text())["start"]
    log = ctx.path.parent/"logs"/"training-unit-12345.out"
    log.parent.mkdir()
    log.write_text("worker preflight\n")
    dm.register_log(ctx, log, label="Scheduler stdout", group="Execution")
    before_logs = (path/"logs.json").read_bytes()
    before_metrics = (path/"metrics.jsonl").read_bytes()
    assert dm.start_attempt(ctx, job_id="12345", worker_managed=True) == path
    assert (path/"metrics.jsonl").read_bytes() == before_metrics
    assert dm.start_attempt(ctx, job_id="12345") == path
    assert (path/"logs.json").read_bytes() == before_logs
    assert json.loads((path/"run.json").read_text())["start"] == started
    science = dict(status="COMPLETED", elapsed_seconds=1.5, experiment_count=1,
        scientific_outcome="Diagnostic only", details={"updates":3})
    raw = json.dumps(science).encode()
    (ctx.path/"summary.json").write_bytes(raw)
    dm.finish_attempt(ctx, science)
    interim_run = json.loads((path/"run.json").read_text())
    interim = json.loads((path/"summary.json").read_text())
    assert interim_run["state"] == interim["state"] == "RUNNING"
    assert "end" not in interim_run and "end" not in interim
    assert interim.get("runtime_seconds") is None and interim.get("exit_code") is None
    assert interim["results"]["science_body_status"] == "COMPLETED"
    assert interim["results"]["science_body_elapsed_seconds"] == 1.5
    records = [json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
    assert not any("execution_completed" in row["metrics"] for row in records)
    log.write_text("worker preflight\nscience completed\nlate worker guard\n")
    dm.finish_attempt(ctx, _native_terminal(worker_status, runtime=8.0, code=code))
    final_run = json.loads((path/"run.json").read_text())
    final = json.loads((path/"summary.json").read_text())
    assert final_run["state"] == final["state"] == worker_status
    assert final["runtime_seconds"] == 8.0 and final["exit_code"] == code
    assert final["start"] == started
    assert final["results"]["science_body_status"] == "COMPLETED"
    assert final["results"]["science_body_elapsed_seconds"] == 1.5
    assert final["metadata"]["native_worker_finalized"] is True
    assert "preflight, tests, science and final checks" in final["metadata"]["runtime_scope"]
    assert (ctx.path/"summary.json").read_bytes() == raw
    index = json.loads((path/"logs.json").read_text())
    copied = next(row for row in index["logs"] if row["label"] == "Scheduler stdout")
    assert (path/copied["path"]).read_text() == log.read_text()
    assert "TDN_RESPONSE_RAVEN_DIR" not in os.environ


def test_worker_managed_reuse_rejects_changed_provenance(tmp_path, monkeypatch):
    monkeypatch.delenv("TDN_RESPONSE_RAVEN_DIR", raising=False)
    ctx = _ctx(tmp_path)
    dm.start_attempt(ctx, job_id="12345", worker_managed=True)
    with pytest.raises(ValueError, match="source identity changed"):
        dm.start_attempt(ctx, job_id="12345", sourcehash="b"*64)
    with pytest.raises(ValueError, match="scheduler identity"):
        dm.start_attempt(ctx, job_id="12346")
    ctx.protocol["trials"]["one"]["seed"] = 102
    with pytest.raises(ValueError, match="protocol changed"):
        dm.start_attempt(ctx, job_id="12345")


def test_moved_portable_bundle_resolves_all_pointers_and_hashes(tmp_path, monkeypatch):
    ctx, path = _completed(tmp_path, monkeypatch)
    project = dm.raven_project(ctx)
    archive = tmp_path/"view.zip"
    receipt = dr.portable_export(project, archive)
    assert receipt["included_files"] > 5
    dr.verify_portable(archive)
    moved = tmp_path/"unrelated-machine"/"research"
    moved.mkdir(parents=True)
    with zipfile.ZipFile(archive) as transport:
        # These members were produced and checksum-validated in this test.
        for name in transport.namelist():
            dm.relative_path(name)
            destination = moved/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(transport.read(name))
    run = json.loads((moved/"runs"/path.name/"run.json").read_text())
    for field, pointer in run["paths"].items():
        dm.relative_path(pointer)
        assert (moved/"runs"/path.name/pointer).exists()
    index = json.loads((moved/"runs"/path.name/"outputs/source_index.json").read_text())
    assert index["sources"][0]["path"] == "outputs/science/summary.json"


def test_export_rejects_symlink_and_unsafe_archive_members(tmp_path, monkeypatch):
    ctx, path = _completed(tmp_path, monkeypatch)
    (path/"outputs"/"escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        dr.portable_export(dm.raven_project(ctx), tmp_path/"bad.zip")
    with zipfile.ZipFile(tmp_path/"traversal.zip","w") as archive:
        archive.writestr("../escape","no")
    with pytest.raises(ValueError, match="relative"):
        dr.verify_portable(tmp_path/"traversal.zip")


def _raven_modules(monkeypatch):
    checkout = Path(os.environ.get("TDN_RAVEN_CHECKOUT", "/workspace/Raven"))
    if not (checkout/"raven/projects.py").is_file():
        pytest.skip("Optional compatibility check needs a trusted Raven checkout via TDN_RAVEN_CHECKOUT")
    monkeypatch.syspath_prepend(str(checkout))
    return [importlib.import_module("raven."+name) for name in
            ("projects","metrics","reporting","log_catalog","artifacts")]


def test_actual_raven_native_parsers_and_relocated_discovery(tmp_path, monkeypatch):
    projects, native_metrics, reporting, logs, artifacts = _raven_modules(monkeypatch)
    ctx, path = _completed(tmp_path, monkeypatch)
    run = json.loads((path/"run.json").read_text())
    assert projects.validate_inventory(run)["job_id"] == "12345_7"
    assert reporting._summary(json.loads((path/"summary.json").read_text()))["id"] == path.name
    for line in (path/"metrics.jsonl").read_text().splitlines():
        native_metrics._record(json.loads(line))
    logs._manifest_entries(json.loads((path/"logs.json").read_text()), "12345_7", str(path/"logs.json"))
    with pytest.raises(ValueError):
        logs._manifest_entries(json.loads((path/"logs.json").read_text()), "12345_8", str(path/"logs.json"))
    artifacts._contract(json.loads((path/"outputs/contract.json").read_text()))
    result = projects.discover_project(str(dm.raven_project(ctx)))
    assert not result["warnings"]
    assert result["runs"][0]["inventory"]["run_id"] == path.name
    selected = projects.select_run(str(dm.raven_project(ctx)), path.name)
    assert selected["binding"]["job_id"] == "12345_7"


def test_complete_analysis_export_keeps_failed_queries_and_native_contract(tmp_path, monkeypatch):
    projects, native_metrics, reporting, logs, artifacts = _raven_modules(monkeypatch)
    ctx = _ctx(tmp_path)
    ctx.stage = "report"
    ctx.path = ctx.path.parent/"report"
    ctx.path.mkdir()
    ctx.protocol["units"]["report"] = {"kind":"report","mechanism":"M3"}
    ctx.protocol["trials"]["one"]["updates"] = 3
    ctx.protocol["mechanisms"] = {"M3":{}}
    ctx.unit = ctx.protocol["units"]["report"]
    ctx.prerequisites = {}
    captured = []
    ctx.record = lambda *args,**kwargs:captured.append((args,kwargs))
    base = ctx.path.parent/"training-unit"
    endpoint = dict(query_id="q",parent_id="p",rms=.001,maximum=.002,mse=1e-6,
        target_defect_rms=.004,reference_accepted=True)
    fit = dict(mse=1e-6,rms=.001,maximum=.002,rows=[endpoint])
    trial = dict(trial_id="one",family="temporal",target_kind="fd_same_grid",variant_id="gain",seed=101,
        status="COMPLETED_UPDATES",updates=3,requested_updates=3,initial_train=dict(mse=.01,rows=[dict(endpoint,mse=.01)]),
        selected_train=fit,final_train=fit,attainable_train=fit,criterion_train=fit,
        requested_training_queries=2,accepted_training_queries=1,requested_validation_queries=1,accepted_validation_queries=1,
        reference_coverage_rows=[dict(query_id="missing",reference_accepted=False,reason="UNRESOLVED")])
    curves = [dict(trial_id="one",update=i,train_loss=.01/(i+1),train_rms=.1/(i+1),gradient_norm=.01)
              for i in range(4)]
    payload = {"training_results.json":dict(trials=[trial]),"learning_curves.jsonl":curves}
    hashes = {}
    for name, value in payload.items():
        raw = ("".join(json.dumps(r)+"\n" for r in value) if name.endswith(".jsonl") else json.dumps(value)).encode()
        (base/name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    manifest = dict(schema="tdn.response-science/v1",stage="training-unit",kind="train",source_tree_sha256="a"*64,
        protocol_sha256=digest(ctx.protocol),artifacts=hashes)
    write_json(base/"science_manifest.json",manifest)
    (base/"COMPLETED").write_text(digest(manifest)+"\n")
    ctx.prerequisites["training-unit"] = base
    dm.start_attempt(ctx,job_id="12346",sourcehash="a"*64)
    result = dr.run(ctx)
    assert result["eligible_for_development"] is False
    assert result["research_summary"]["complete_reference_coverage_trials"] == 0
    assert captured[0][0][0] == "diagnostic/portable-research-report"
    portable = dr.verify_portable(ctx.path/dr.BUNDLE)
    assert dm.attempt_path(ctx).relative_to(dm.raven_root(ctx)).as_posix() in portable["excluded_attempts"]
    analysis = dr._analysis_path(ctx)
    data = json.loads((analysis/"outputs/diagnostic_metrics.json").read_text())
    assert data["trials"][0]["reference_coverage_rows"][0]["query_id"] == "missing"
    assert (analysis/"outputs/learning-one.png").is_file()
    assert data["trials"][0]["initialization_mse_reduction"] == pytest.approx(10000)
    assert projects.validate_inventory(json.loads((analysis/"run.json").read_text()))["run_id"] == analysis.name
    assert reporting._summary(json.loads((analysis/"summary.json").read_text()))["state"] == "COMPLETED"
    contract = json.loads((analysis/"outputs/contract.json").read_text())
    artifacts._contract(contract)
    actual = artifacts.validate_contract(contract,str(analysis))
    assert actual["valid"] is True, actual


def test_large_campaign_pages_exact_attempts_before_discovery_cap(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    units = {f"unit-{index:03d}":dict(kind="train",trial_id="one") for index in range(600)}
    ctx.protocol["units"] = units
    ctx.stage = "unit-256"
    ctx.path = ctx.path.parent/ctx.stage
    ctx.path.mkdir()
    page = dm.raven_project(ctx)
    assert page.name == "page-001"
    attempt = dm.start_attempt(ctx,job_id="256")
    assert attempt.parent.parent == page
    catalog = json.loads((dm.raven_root(ctx)/"project_catalog.json").read_text())
    assert len(catalog["attempts"]) == 600
    page_counts = {}
    for row in catalog["attempts"]:
        page_counts[row["page"]] = page_counts.get(row["page"],0)+1
        dm.relative_path(row["inventory_path"])
    assert max(page_counts.values()) <= 255
    assert catalog["attempts"][256]["page"] == "page-001"


def test_physical_learning_plots_use_scheduled_full_bank_fields(tmp_path,monkeypatch):
    import matplotlib.axes
    original=matplotlib.axes.Axes.plot
    seen={}
    def capture(self,x,y,*args,**kwargs):
        seen[kwargs.get("label")]=list(y)
        return original(self,x,y,*args,**kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes,"plot",capture)
    trial=dict(trial_id="one",status="COMPLETED_UPDATES",family="temporal",variant_id="conditioned",seed=1,target_kind="fd_same_grid")
    data=dict(trials=[trial],learning_curves=[dict(trial_id="one",update=40,train_loss=.1,full_train_loss=.08,
        validation_loss=.09,full_train_rms=.01,full_train_maximum=.02,validation_rms=.03,validation_maximum=.04)])
    dr._plots(tmp_path,data)
    assert seen["full_train_rms"] == [.01]
    assert seen["full_train_maximum"] == [.02]
    assert seen["validation_maximum"] == [.04]
    assert seen["full_train_loss"] == [.08]
