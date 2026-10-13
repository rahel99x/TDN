"""Real Raven coordinator launches with controlled native-command fixtures.

These checks exercise capture, process execution and Raven's own readers. The
stubbed child commands are lifecycle fixtures, not Slurm or CUDA evidence.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
CAMPAIGN = "response_diagnostic_campaign.sh"
HOOK = "response_diagnostic_raven.py"


def _read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def raven(monkeypatch):
    checkout = Path(os.environ.get("TDN_RAVEN_TEST_ROOT",
        os.environ.get("TDN_RAVEN_CHECKOUT", "/workspace/Raven")))
    if not (checkout / "raven/launch_runtime.py").is_file():
        pytest.skip("Optional launch compatibility needs a trusted Raven checkout via TDN_RAVEN_TEST_ROOT")
    monkeypatch.syspath_prepend(str(checkout))
    return {name: importlib.import_module("raven." + name) for name in
        ("launch", "launch_runtime", "projects", "metrics", "reporting", "log_catalog", "artifacts")}


STUB = r'''from pathlib import Path
import json, os, sys

root = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
command = args[0]
run_id = args[args.index('--run-id') + 1] if '--run-id' in args else args[1]
with (root / 'native-calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'argv': args, 'cwd': str(Path.cwd()),
        'raven_environment': {k: v for k, v in os.environ.items() if k.startswith('RAVEN_')}}) + '\n')
base = root / 'runs' / run_id
base.mkdir(parents=True, exist_ok=True)
(base / 'raven').mkdir(exist_ok=True)
failed = bool(os.environ.get('TDN_TEST_INFRA_FAILURE')) and run_id.startswith('response-diagnostic-') and not run_id.startswith('response-diagnostic-smoke-')
units = {'training-one': {'job_id': '12345', 'scheduler_state': 'FAILED' if failed else 'COMPLETED',
    'worker_status': 'FAILED' if failed else 'COMPLETED',
    'artifact_validation': 'MISSING' if failed else 'SEALED_UNVERIFIED', 'scientific_outcome': 'NEGATIVE'},
    'report': {'job_id': '12346', 'scheduler_state': 'COMPLETED',
    'worker_status': 'COMPLETED', 'artifact_validation': 'SEALED_UNVERIFIED', 'scientific_outcome': 'NEGATIVE'}}
if command == 'status' and '--json' in args:
    print(json.dumps({'run_id': run_id, 'profile': 'diagnostic-smoke' if 'smoke-' in run_id else 'diagnostic',
        'terminal': True, 'completed': not failed, 'report_complete': True,
        'failed_units': ['training-one'] if failed else [], 'units': units}))
elif command == 'validate':
    print(json.dumps({'run_id': run_id, 'valid': not failed}))
    raise SystemExit(2 if failed else 0)
elif command == 'decision':
    print(json.dumps({'schema': 'tdn.response-diagnostic-decision/v1',
        'decision': 'STOP_DO_NOT_ADVANCE', 'eligible_for_development': False,
        'reason': 'Controlled negative science fixture'}))
elif command == 'compact':
    print('review archive retained ' + run_id)
    print('native stderr tail ' + run_id, file=sys.stderr)
elif command in ('plan', 'run', 'status'):
    print(command + ' controlled native fixture ' + run_id)
else:
    raise SystemExit('unexpected native fixture command ' + command)
'''


@pytest.fixture
def project(tmp_path):
    # Space-containing paths exercise shell and frozen-plan argument handling.
    root = tmp_path / "TDN checkout with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in (CAMPAIGN, HOOK):
        source = REPOSITORY / "scripts" / name
        assert source.is_file(), f"Required coordinator implementation is absent: {source}"
        shutil.copyfile(source, scripts / name)
    venv = root / ".venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    (scripts / "native_stub.py").write_text(STUB)
    (scripts / "fedora_response_diagnostic.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"\n'
        'exec "$root/.venv/bin/python" "$root/scripts/native_stub.py" "$@"\n')
    return root


def _clean_environment(**extra):
    env = {key: value for key, value in os.environ.items()
        if not key.startswith(("RAVEN_", "SLURM_", "SBATCH_", "TDN_RESPONSE_", "TDN_TEST_"))}
    env.update(extra)
    return env


def _plan(raven, root, *, local=True):
    return raven["launch"].resolve_plan([
        str(root / "scripts" / CAMPAIGN), "--project", str(root),
        *( ["--local"] if local else [] ), "--", "--targets", "fd_same_grid",
        "--updates", "temporal=4"], cwd=root, environ={})


def _calls(root):
    return [json.loads(line) for line in (root / "native-calls.jsonl").read_text().splitlines()]


def _campaign_receipt(root):
    files = list((root / "runs/.response-diagnostic-campaigns").glob("*/receipt.json"))
    assert len(files) == 1
    return files[0], _read(files[0])


def _assert_native_reader_compatibility(raven, root, run):
    manifest = _read(run / "run.json")
    assert raven["projects"].validate_inventory(manifest)["run_id"] == run.name
    raven["reporting"]._summary(_read(run / "summary.json"))
    for line in (run / manifest["paths"]["metrics"]).read_text().splitlines():
        raven["metrics"]._record(json.loads(line))
    logs = _read(run / manifest["paths"]["log_index"])
    raven["log_catalog"]._manifest_entries(logs, "", str(run / "logs.json"))
    contract = _read(run / manifest["paths"]["contract"])
    raven["artifacts"]._contract(contract)
    validated = raven["artifacts"].validate_contract(contract, run)
    assert validated["status"] == "valid", validated
    selected = raven["projects"].select_run(str(root), run.name)
    assert selected["run"]["inventory"]["run_id"] == run.name
    paths = [entry["path"] for entry in contract["outputs"]]
    assert len(paths) == len(set(paths)), "Coordinator artifacts must be registered once"
    log_paths = [entry["path"] for entry in logs["logs"]]
    assert len(log_paths) == len(set(log_paths)), "Coordinator logs must be registered once"
    return manifest, logs, contract


@pytest.mark.parametrize("infrastructure_failure", [False, True])
def test_actual_captured_campaign_records_lifecycle_and_separate_science(
        raven, project, infrastructure_failure, capfd):
    env = _clean_environment(**({"TDN_TEST_INFRA_FAILURE": "1"} if infrastructure_failure else {}))
    launch_receipt = raven["launch_runtime"].execute_plan(_plan(raven, project), environ=env)
    stdout, stderr = capfd.readouterr()
    assert launch_receipt["exit_code"] == (2 if infrastructure_failure else 0), stderr
    run = Path(launch_receipt["run_dir"])
    expected = "FAILED" if infrastructure_failure else "COMPLETED"
    assert _read(run / "run.json")["state"] == expected
    assert _read(run / "summary.json")["state"] == expected
    assert _read(run / "summary.json")["exit_code"] == (2 if infrastructure_failure else 0)
    path, receipt = _campaign_receipt(project)
    assert receipt["status"] == expected
    assert receipt["targets"] == ["fd_same_grid"]
    assert receipt["diagnostic_overrides"] == ["--updates", "temporal=4"]
    assert receipt["raven_launch"]["coordinator_run_id"] == run.name
    assert receipt["raven_launch"]["coordinator_run_dir"] == run.relative_to(project).as_posix()
    assert _read(path.parent / "diagnostic-decision.json")["eligible_for_development"] is False
    calls = _calls(project)
    assert calls and all(call["cwd"] == str(project) for call in calls)
    # The foreground native controller remains part of this coordinator.
    # Actual sbatch worker-environment isolation has its own workflow test.
    assert all(call["raven_environment"].get("RAVEN_RUN_ID") == run.name for call in calls), calls
    assert sum(call["argv"][0] == "run" for call in calls) == 2
    assert all(call["argv"][0] not in ("cancel", "recover") for call in calls)
    manifest, logs, contract = _assert_native_reader_compatibility(raven, project, run)
    assert manifest["provenance"]["script"] == "scripts/" + CAMPAIGN
    assert "job_id" not in manifest, "The local coordinator must not invent a native job identity"
    registered = {entry["path"] for entry in logs["logs"]}
    assert {"logs/campaign.stdout.log", "logs/campaign.stderr.log"} <= registered
    assert "Diagnostic run:" in (run / "logs/campaign.stdout.log").read_text()
    assert "native stderr tail " + receipt["diagnostic_run_id"] in (run / "logs/campaign.stderr.log").read_text()
    assert "native stderr tail " + receipt["diagnostic_run_id"] in stderr
    output_paths = {entry["path"] for entry in contract["outputs"]}
    assert {"outputs/campaign.json", "outputs/research_paths.json"} <= output_paths
    assert _read(run / "outputs/campaign.json")["status"] == expected
    research_paths = json.dumps(_read(run / "outputs/research_paths.json"))
    assert receipt["smoke_run_id"] in research_paths
    assert receipt["diagnostic_run_id"] in research_paths
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert metrics and all(isinstance(value, (int, float)) and not isinstance(value, bool)
        for row in metrics for value in row["metrics"].values())
    native_counts = [row["metrics"] for row in metrics if "registered_units" in row["metrics"]]
    assert len(native_counts) >= 2
    assert all(row["registered_units"] == row["submitted_units"] == row["terminal_jobs"] == 2
        for row in native_counts)
    assert native_counts[-1]["completed_units"] == (1 if infrastructure_failure else 2)
    assert native_counts[-1]["failed_units"] == int(infrastructure_failure)
    if infrastructure_failure:
        assert any(receipt["diagnostic_run_id"] in call["argv"] and call["argv"][0] == "compact" for call in calls)
    else:
        assert "Diagnostic campaign evidence is complete:" in stdout


@pytest.fixture
def frozen_context(raven, project):
    frozen = raven["launch_runtime"].materialize(_plan(raven, project))
    env = _clean_environment()
    run = raven["launch_runtime"]._initialize_run(frozen, env)
    folder = Path(frozen["runtime_bundle"]).parent
    env.update(RAVEN_RUN_DIR=str(run), RAVEN_RUN_ID=run.name,
        RAVEN_PROJECT_ROOT=str(project), RAVEN_LAUNCH_PLAN=str(folder / "plan.json"),
        RAVEN_SOURCE_SCRIPT=frozen["script"], RAVEN_REPORT_HELPER=str(folder / "report.py"),
        RAVEN_REPORT_PYTHON=frozen["report_python"], RAVEN_RUNTIME_BUNDLE=frozen["runtime_bundle"],
        RAVEN_SHELL_HELPER=str(folder / "shell.sh"))
    return frozen, run, env


def _check(project, context, *, env=None, snapshot=None):
    frozen, _run, baseline = context
    return subprocess.run([sys.executable, str(project / "scripts" / HOOK), "check",
        "--snapshot", str(snapshot or project / frozen["source_snapshot"])],
        cwd=project, env=baseline if env is None else env, capture_output=True, text=True)


def test_real_frozen_launch_context_validates_without_mutating_lifecycle(project, frozen_context):
    _frozen, run, _env = frozen_context
    before = (run / "run.json").read_bytes()
    result = _check(project, frozen_context)
    assert result.returncode == 0, result.stderr
    assert (run / "run.json").read_bytes() == before
    assert not (project / "native-calls.jsonl").exists()


@pytest.mark.parametrize("variable,replacement", [
    ("RAVEN_RUN_ID", "foreign-attempt"),
    ("RAVEN_RUN_DIR", "foreign-run"),
    ("RAVEN_PROJECT_ROOT", "foreign-project"),
    ("RAVEN_SOURCE_SCRIPT", "foreign-script"),
    ("RAVEN_REPORT_HELPER", "foreign-helper"),
    ("RAVEN_REPORT_PYTHON", "foreign-python"),
    ("SLURM_JOB_ID", "12345"), ("SLURM_STEP_ID", "batch"), ("SLURM_JOB_ACCOUNT", "foreign-account"),
])
def test_launch_hook_rejects_foreign_or_slurm_context_without_native_work(
        project, frozen_context, variable, replacement):
    _frozen, run, env = frozen_context
    env = dict(env, **{variable: replacement})
    before = (run / "run.json").read_bytes()
    result = _check(project, frozen_context, env=env)
    assert result.returncode != 0, variable
    assert (run / "run.json").read_bytes() == before
    assert not (project / "native-calls.jsonl").exists()


@pytest.mark.parametrize("changed", ["source", "snapshot", "snapshot_argument"])
def test_launch_hook_rejects_current_or_captured_source_drift(project, frozen_context, changed):
    frozen, run, _env = frozen_context
    snapshot = project / frozen["source_snapshot"]
    argument = None
    if changed == "snapshot_argument":
        argument = project / "scripts" / CAMPAIGN
    else:
        source = project / "scripts" / CAMPAIGN if changed == "source" else snapshot
        source.chmod(0o600)
        source.write_text(source.read_text() + "\n# changed after the frozen launch\n")
    before = (run / "run.json").read_bytes()
    result = _check(project, frozen_context, snapshot=argument)
    assert result.returncode != 0
    assert (run / "run.json").read_bytes() == before
    assert not (project / "native-calls.jsonl").exists()


def test_launch_hook_rejects_symlink_source_membership(project, frozen_context):
    frozen, run, _env = frozen_context
    original = project / "scripts" / CAMPAIGN
    alternate = project / "scripts/source-copy.sh"
    original.rename(alternate)
    original.symlink_to(alternate.name)
    before = (run / "run.json").read_bytes()
    result = _check(project, frozen_context)
    assert result.returncode != 0
    assert (run / "run.json").read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("project_root", "/foreign/project"),
    ("script_relative", "scripts/foreign.sh"),
    ("source_snapshot", "scripts/" + CAMPAIGN),
    ("script_sha256", "0" * 64),
    ("execution", "slurm"),
])
def test_launch_hook_rejects_changed_frozen_plan_membership(
        project, frozen_context, field, value):
    _frozen, run, env = frozen_context
    path = Path(env["RAVEN_LAUNCH_PLAN"])
    plan = _read(path)
    plan[field] = value
    path.chmod(0o600)
    path.write_text(json.dumps(plan))
    before = (run / "run.json").read_bytes()
    result = _check(project, frozen_context)
    assert result.returncode != 0, field
    assert (run / "run.json").read_bytes() == before
    assert not (project / "native-calls.jsonl").exists()


def _initialize_hook_receipt(project, context):
    frozen, run, env = context
    directory = project / "runs/.response-diagnostic-campaigns/controlled-receipt"
    directory.mkdir(parents=True)
    path = directory / "receipt.json"
    path.write_text(json.dumps({"schema": "tdn.response-diagnostic-campaign/v1",
        "project_root": str(project), "smoke_run_id": "response-diagnostic-smoke-controlled",
        "diagnostic_run_id": "response-diagnostic-controlled", "targets": ["fd_same_grid"],
        "diagnostic_overrides": [], "status": "RUNNING", "phase": "CREATED", "events": [],
        "raven_launch": {"project_root": str(project), "coordinator_run_id": run.name,
            "coordinator_run_dir": run.relative_to(project).as_posix(), "launch_token": frozen["token"]}}))
    command = [sys.executable, str(project / "scripts" / HOOK), "initialize",
        "--receipt", str(path), "--snapshot", str(project / frozen["source_snapshot"])]
    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return path, command


def test_initialize_registers_once_and_keeps_raven_lifecycle_owned(
        project, frozen_context):
    _frozen, run, env = frozen_context
    _receipt, command = _initialize_hook_receipt(project, frozen_context)
    run_before = (run / "run.json").read_bytes()
    logs_before = (run / "logs.json").read_bytes()
    manifest = _read(run / "run.json")
    contract_path = run / manifest["paths"]["contract"]
    contract_before = contract_path.read_bytes()
    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (run / "run.json").read_bytes() == run_before
    assert (run / "logs.json").read_bytes() == logs_before
    assert contract_path.read_bytes() == contract_before
    assert _read(run / "run.json")["state"] == "RUNNING"
    assert not (run / "summary.json").exists(), "Only Raven finalizes its coordinator lifecycle"


def _native_status(receipt):
    return {"run_id": _read(receipt)["diagnostic_run_id"], "terminal": True,
        "completed": True, "report_complete": True, "failed_units": [],
        "units": {"report": {"job_id": "12345", "scheduler_state": "COMPLETED",
            "worker_status": "COMPLETED", "artifact_validation": "SEALED_UNVERIFIED",
            "scientific_outcome": "NEGATIVE"}}}


def _publish_hook(project, context, receipt, *, status_file=None):
    _frozen, _run, env = context
    command = [sys.executable, str(project / "scripts" / HOOK), "publish",
        "--receipt", str(receipt)]
    if status_file is not None:
        command.extend(["--status-file", str(status_file)])
    return subprocess.run(command, cwd=project, env=env, capture_output=True, text=True)


def _publish_native_status(project, context, *, padding=""):
    receipt, _command = _initialize_hook_receipt(project, context)
    status = _native_status(receipt)
    if padding:
        status["fixture_placeholder"] = padding
    path = receipt.parent / "response-diagnostic-controlled-status.json"
    path.write_text(json.dumps(status))
    result = _publish_hook(project, context, receipt, status_file=path)
    assert result.returncode == 0, result.stderr
    return receipt, path


def test_status_source_larger_than_one_mib_is_observed_without_copying_dense_fields(
        project, frozen_context):
    _frozen, run, _env = frozen_context
    receipt, status = _publish_native_status(project, frozen_context, padding="x" * (1 << 20))
    assert status.stat().st_size > 1 << 20
    paths = _read(run / "outputs/research_paths.json")
    observed = paths["latest_native_status"]
    assert observed["run_id"] == _read(receipt)["diagnostic_run_id"]
    assert observed["source"]["path"] == status.relative_to(project).as_posix()
    assert observed["counts"]["registered_units"] == observed["counts"]["completed_units"] == 1
    assert "fixture_placeholder" not in observed
    assert (run / "outputs/research_paths.json").stat().st_size < 1 << 20


def test_final_receipt_update_rereads_and_retains_exact_native_status(
        project, frozen_context):
    _frozen, run, _env = frozen_context
    receipt, status = _publish_native_status(project, frozen_context)
    prior = _read(run / "outputs/research_paths.json")["latest_native_status"]
    value = _read(receipt)
    value.update(status="COMPLETED", phase="COMPLETED")
    value["events"].append({"phase": "COMPLETED", "status": "COMPLETED"})
    receipt.write_text(json.dumps(value))
    result = _publish_hook(project, frozen_context, receipt)
    assert result.returncode == 0, result.stderr
    latest = _read(run / "outputs/research_paths.json")["latest_native_status"]
    assert latest == prior
    assert latest["source"]["path"] == status.relative_to(project).as_posix()
    metric = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    assert metric["phase"] == "campaign/COMPLETED"
    assert metric["metrics"]["campaign_completed"] == 1
    assert metric["metrics"]["registered_units"] == metric["metrics"]["completed_units"] == 1
    assert metric["progress"] == {"completed": 1, "total": 1, "unit": "registered native allocations"}
    assert _read(run / "run.json")["state"] == "RUNNING"


def test_retained_native_status_hash_drift_rejects_without_new_metric(
        project, frozen_context):
    _frozen, run, _env = frozen_context
    receipt, status = _publish_native_status(project, frozen_context)
    paths_before = (run / "outputs/research_paths.json").read_bytes()
    metrics_before = (run / "metrics.jsonl").read_bytes()
    value = _read(status)
    value["fixture_placeholder"] = "bytes changed after the observed native status"
    status.write_text(json.dumps(value))
    result = _publish_hook(project, frozen_context, receipt)
    assert result.returncode != 0
    assert (run / "metrics.jsonl").read_bytes() == metrics_before
    assert (run / "outputs/research_paths.json").read_bytes() == paths_before
    assert _read(run / "run.json")["state"] == "RUNNING"


@pytest.mark.parametrize("invalid", ["foreign_run", "unknown_failed_unit", "nonboolean_terminal"])
def test_publish_rejects_foreign_or_inconsistent_status_before_metrics(
        project, frozen_context, invalid):
    _frozen, run, env = frozen_context
    receipt, _command = _initialize_hook_receipt(project, frozen_context)
    status = {"run_id": "response-diagnostic-controlled", "terminal": True,
        "completed": True, "report_complete": True, "failed_units": [],
        "units": {"report": {"job_id": "12345", "scheduler_state": "COMPLETED",
            "worker_status": "COMPLETED", "artifact_validation": "SEALED_UNVERIFIED",
            "scientific_outcome": "NEGATIVE"}}}
    if invalid == "foreign_run":
        status["run_id"] = "response-diagnostic-foreign"
    elif invalid == "unknown_failed_unit":
        status["failed_units"] = ["unregistered-unit"]
    else:
        status["terminal"] = 1
    snapshot = receipt.parent / "response-diagnostic-controlled-status.json"
    snapshot.write_text(json.dumps(status))
    before = (run / "metrics.jsonl").read_bytes()
    result = subprocess.run([sys.executable, str(project / "scripts" / HOOK), "publish",
        "--receipt", str(receipt), "--status-file", str(snapshot)], cwd=project,
        env=env, capture_output=True, text=True)
    assert result.returncode != 0, invalid
    assert (run / "metrics.jsonl").read_bytes() == before
    assert _read(run / "run.json")["state"] == "RUNNING"


def test_launch_hook_is_inactive_without_a_launch_plan(project):
    # Stale report variables alone must never attach this command to an attempt.
    env = _clean_environment(RAVEN_RUN_DIR=str(project / "runs/foreign"), RAVEN_RUN_ID="foreign")
    result = subprocess.run([sys.executable, str(project / "scripts" / HOOK), "check",
        "--snapshot", str(project / "scripts" / CAMPAIGN)], cwd=project,
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (project / "runs").exists()


def test_outer_slurm_campaign_fails_before_native_commands(project):
    result = subprocess.run(["bash", str(project / "scripts" / CAMPAIGN)], cwd=project,
        env=_clean_environment(SLURM_JOB_ID="12345"), capture_output=True, text=True)
    assert result.returncode == 2
    assert "--local" in result.stderr
    assert not (project / "native-calls.jsonl").exists()
