"""Regularization launch contracts; scheduler fixtures are not native execution evidence."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "scripts/response_preconditioner_regularization.sh"


@pytest.fixture
def native(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("regularization_native_under_test", ROOT / "scripts/response_preconditioner_regularization_native.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "input_identity", lambda root: {"archive_sha256": "b" * 64})
    return module


@pytest.fixture
def raven(monkeypatch):
    root = Path(os.environ.get("TDN_RAVEN_TEST_ROOT", "/workspace/Raven"))
    if not (root / "raven/launch.py").is_file():
        pytest.skip("Optional Raven compatibility requires a trusted Raven checkout")
    monkeypatch.syspath_prepend(str(root))
    return {name: importlib.import_module("raven." + name) for name in ("launch", "launch_runtime")}


def clean_env():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("RAVEN_", "SLURM_", "TDN_", "SBATCH_"))}


def root_at(native, monkeypatch, root):
    monkeypatch.setattr(native, "ROOT", root)
    monkeypatch.setattr(native.common, "ROOT", root)


def test_shell_contract_and_non_native_rejection():
    subprocess.run(["bash", "-n", str(SHELL)], check=True)
    result = subprocess.run(["bash", str(SHELL), "--help"], text=True, capture_output=True, env=clean_env())
    assert result.returncode == 0 and "--mode all|verify" in result.stdout
    assert "--recipe response-preconditioner-regularization" in result.stdout
    result = subprocess.run(["bash", str(SHELL)], text=True, capture_output=True, env=clean_env())
    assert result.returncode == 2 and "use raven launch" in result.stderr


def test_cli_study_rejects_standalone_and_gpu_before_science():
    for args in (["study", "--contract", "absent.json"], ["verify", "--device", "cuda"]):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/response_preconditioner_regularization.py"), *args],
                                text=True, capture_output=True, env=clean_env(), cwd=ROOT)
        assert result.returncode == 2 and "Traceback" not in result.stderr


def test_recipe_has_only_regularization_science_and_cpu_memory():
    settings = json.loads((ROOT / ".raven/automation.json").read_text())
    recipe = settings["recipes"]["response-preconditioner-regularization"]
    assert recipe == dict(features=["experiment", "evidence", "artifacts"], gpu=False,
                          trace=False, cpu_memory=True, sampling_seconds=1, auto_export=False)
    assert "response-preconditioner" in settings["recipes"]


@pytest.mark.parametrize("changed", [{"NumCPUs": "8"}, {"AllocTRES": "cpu=4,mem=16G"},
    {"TimeLimit": "00:21:00"}, {"AllocTRES": "cpu=4,mem=8G,gres/gpu=1"}])
def test_regularization_rejects_resource_expansion(native, changed):
    job = dict(NumCPUs="4", AllocTRES="cpu=4,mem=8G,node=1", TimeLimit="00:20:00")
    native.resource_check({"job": job})
    with pytest.raises(ValueError):
        native.resource_check({"job": {**job, **changed}})


def test_prepare_separate_frozen_protocol_and_modes(native, monkeypatch, tmp_path, capsys):
    root_at(native, monkeypatch, tmp_path)
    monkeypatch.setattr(native, "identity", lambda: {"source": {"sha256": "a" * 64}})
    for key in ("RAVEN_LAUNCH_PLAN", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT"):
        monkeypatch.delenv(key, raising=False)
    native.prepare(); first = capsys.readouterr().out.strip()
    native.prepare(); second = capsys.readouterr().out.strip()
    assert first != second and first.startswith(native.CONTRACT_FOLDER + "/")
    assert (tmp_path / "logs/response-preconditioner-regularization").is_dir()
    contract, digest = native.load_contract(first)
    assert contract["input_identity"] == {"archive_sha256": "b" * 64}
    assert contract["modes"] == ["all", "verify"] and contract["decision"] == "BOUNDED_RETAINED_DATA_DEVELOPMENT"
    assert set(contract["protocols"]) == {"verify", "study"}
    assert len(digest) == 64 and not (tmp_path / first).stat().st_mode & 0o222
    path = tmp_path / first; path.chmod(0o644)
    contract["protocols"]["study"]["unauthorized_change"] = True
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="protocol changed"):
        native.load_contract(first)


def test_prepared_input_identity_must_remain_exact(native, monkeypatch, tmp_path, capsys):
    root_at(native, monkeypatch, tmp_path)
    monkeypatch.setattr(native, "identity", lambda: {"source": {"sha256": "a" * 64}})
    for key in ("RAVEN_LAUNCH_PLAN", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT"):
        monkeypatch.delenv(key, raising=False)
    native.prepare()
    contract = capsys.readouterr().out.strip()
    monkeypatch.setattr(native, "input_identity", lambda root: {"archive_sha256": "c" * 64})
    with pytest.raises(ValueError, match="P1 evidence changed"):
        native.load_contract(contract)


@pytest.mark.parametrize("failure", ["packing", "source", "input"])
def test_post_study_identity_and_metrics_precede_optional_packing(native, monkeypatch, tmp_path, failure):
    """Synthetic stage files prove failure order, never scientific execution."""
    from tdn.runtime import preflight, desktop_slurm, storage
    from tdn.analysis.tangent_response import telemetry
    root_at(native, monkeypatch, tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(native, "context", lambda snapshot: {"run": tmp_path})
    profile = b'{"synthetic_profile": true}\n'
    (tmp_path / "profile.json").write_bytes(profile)
    bound = dict(profile_path="profile.json", profile_sha256=hashlib.sha256(profile).hexdigest())
    inputs = {"archive_sha256": "b" * 64}
    contract = dict(identity=bound, input_identity=inputs)
    raw = json.dumps(contract).encode()
    (tmp_path / "frozen.json").write_bytes(raw)
    monkeypatch.setattr(native, "load_contract", lambda path: (contract, hashlib.sha256(raw).hexdigest()))
    changed = {"source": False, "input": False}
    monkeypatch.setattr(native, "identity", lambda: "changed" if changed["source"] else bound)
    def checked_input(root):
        if changed["input"]:
            raise ValueError("retained archive hash mismatch")
        return inputs
    monkeypatch.setattr(native, "input_identity", checked_input)
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(desktop_slurm, "verify_allocation", lambda *args, **kwargs:
        {"job": dict(NumCPUs="4", AllocTRES="cpu=4,mem=8G,node=1", TimeLimit="00:20:00")})
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(telemetry, "start_recording", lambda *args: object())
    monkeypatch.setattr(telemetry, "stop_recording", lambda *args, **kwargs: {"status": "STOPPED"})
    monkeypatch.setattr(native, "report_batch", lambda *args: None)
    monkeypatch.setattr(native, "report", lambda *args: None)
    events = []
    def execute(ctx, command, argv):
        events.append(command)
        output = tmp_path / "outputs" / ("verification" if command == "verify" else "study")
        output.mkdir()
        for name in ("protocol.json", "source.json", "source_after.json", "software.json", "artifacts.json",
                     "checks.json", "verification.xml", "metrics.csv", "metrics.json", "selection.json",
                     "denominators.json", "input_identity.json"):
            (output / name).write_text("{}")
        (output / "summary.json").write_text(json.dumps(dict(command=command, status="COMPLETED",
            scientific_promotion=False, device="cpu", decision=native.DECISION)))
        (output / "regularization_metrics.json").write_text(json.dumps([
            dict(phase_suffix="fixture", step=0, values=dict(rms=.1))]))
        if command == "study" and failure in changed:
            changed[failure] = True
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(native, "execute", execute)
    real_publish = native.publish_metrics
    def publish(ctx, output):
        guard = json.loads((tmp_path / "outputs/native/post-study-identity.json").read_text())
        assert guard["source_equal"] and guard["input_equal"]
        events.append("publish")
        real_publish(ctx, output)
    monkeypatch.setattr(native, "publish_metrics", publish)
    def pack(ctx, output):
        events.append("pack")
        raise ValueError("synthetic packing failure")
    monkeypatch.setattr(native, "pack_study_evidence", pack)
    args = type("Args", (), dict(contract="frozen.json", mode="all"))()
    assert native.run(args) == 2
    status = json.loads((tmp_path / "outputs/native/status.json").read_text())
    assert status["status"] == "FAILED"
    assert (tmp_path / "outputs/native/verification-release.json").is_file()
    if failure == "packing":
        assert events == ["verify", "study", "publish", "pack"]
        assert status["source_unchanged"] and status["input_unchanged"]
        assert status["research_metrics_published"]
        assert status["study_execution_status"] == "COMPLETED"
    else:
        assert events == ["verify", "study"]
        assert not status["research_metrics_published"]
        assert status[failure + "_unchanged"] is False
        assert status["identity_checks"][-1]["point"] == "after"


def test_recursive_artifacts_preserve_binary_and_reject_symlinks(native, monkeypatch, tmp_path):
    root_at(native, monkeypatch, tmp_path)
    output = tmp_path / "outputs/study"; (output / "inputs/queries").mkdir(parents=True)
    (tmp_path / "outputs/native").mkdir()
    (output / "summary.json").write_text("{}")
    (output / "figures").mkdir()
    (output / "figures/error.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (output / "figures/error.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    with (output / "metrics.json").open("wb") as stream:
        stream.truncate(65 << 20)
    (output / "report-one.txt").write_text("x" * (600 << 10))
    (output / "report-two.txt").write_text("x" * (600 << 10))
    raw = b"binary\x00\xff"; (output / "inputs/queries/q001.npz").write_bytes(raw)
    row = dict(path="inputs/queries/q001.npz", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (output / "artifacts.json").write_text(json.dumps(dict(files=[row])))
    commands = native.artifact_commands({"run": tmp_path}, output)
    assert not any("queries/q001" in " ".join(command) for command in commands)
    assert ["artifact", "outputs/study/summary.json", "--format", "json"] in commands
    assert ["artifact", "outputs/study/figures/error.png"] in commands
    assert ["artifact", "outputs/study/figures/error.svg", "--format", "text"] in commands
    assert ["artifact", "outputs/study/metrics.json"] in commands
    assert ["artifact", "outputs/study/report-two.txt"] in commands
    assert sum((tmp_path / command[1]).stat().st_size for command in commands
               if "--format" in command) <= native.ARTIFACT_FORMAT_BYTES
    packed = native.pack_study_evidence({"run": tmp_path}, output)
    assert packed["files"] == [row] and packed["stage_artifact_seal_present"]
    import zipfile
    with zipfile.ZipFile(tmp_path / packed["archive"]) as bundle:
        assert bundle.read("inputs/queries/q001.npz") == raw
    fixture = output / "pytest-temp"; fixture.mkdir()
    (fixture / "alias").symlink_to(output / "summary.json")
    for name in (".matplotlib-cache", ".font-cache"):
        cache = output / name
        cache.mkdir()
        (cache / "alias").symlink_to(output / "summary.json")
    assert native.artifact_commands({"run": tmp_path}, output) == commands
    (output / "alias").symlink_to(output / "summary.json")
    with pytest.raises(ValueError, match="symlinks"):
        native.artifact_commands({"run": tmp_path}, output)


def test_stage_preserves_failed_outputs_and_rejects_promotion(native, monkeypatch, tmp_path):
    root_at(native, monkeypatch, tmp_path)
    output = tmp_path / "outputs/study"; output.mkdir(parents=True)
    (output / "failure.txt").write_text("reference failure remains visible")
    batches = []
    monkeypatch.setattr(native, "report_batch", lambda ctx, commands: batches.append(commands))
    native.stage_evidence({"run": tmp_path}, output, "study", 1)
    assert any("outputs/study/failure.txt" in command for command in batches[0])
    (output / "summary.json").write_text(json.dumps(dict(command="study", status="COMPLETED", device="cpu",
        scientific_promotion=True, decision="BOUNDED_RETAINED_DATA_DEVELOPMENT")))
    with pytest.raises(ValueError, match="nonpromoting"):
        native.stage_evidence({"run": tmp_path}, output, "study", 0)


def test_metrics_publish_named_finite_research_rows(native, monkeypatch, tmp_path):
    root_at(native, monkeypatch, tmp_path)
    rows = [dict(phase_suffix="diagnostic/parent-001/coefficient/selected", step=0,
                 values=dict(rms=.01, maximum=.02, mean_component_rms=.006,
                             spatial_fluctuation_rms=.008, complete=1)),
            dict(phase_suffix="fit/p032-s08/action_energy", step=1,
                 values=dict(validation_loss=1e-12, effective_degrees_of_freedom=32.1))]
    path = tmp_path / "regularization_metrics.json"
    path.write_text(json.dumps(rows))
    batches = []
    monkeypatch.setattr(native, "report_batch", lambda ctx, commands: batches.append(commands))
    native.publish_metrics({}, tmp_path)
    assert len(batches) == 1 and len(batches[0]) == 2
    assert "regularization/diagnostic/parent-001/coefficient/selected" in batches[0][0]
    assert "spatial_fluctuation_rms=0.008" in batches[0][0]
    assert "effective_degrees_of_freedom=32.1" in batches[0][1]
    for bad in (float("nan"), float("inf"), "0.1", True):
        rows[0]["values"]["rms"] = bad
        path.write_text(json.dumps(rows))
        with pytest.raises(ValueError, match="finite"):
            native.publish_metrics({}, tmp_path)
    for bad in (dict(phase_suffix="../../foreign", step=0, values={}),
                dict(phase_suffix="x" * 160, step=0, values=dict(rms=1)),
                dict(phase_suffix="fit", step=True, values=dict(rms=1))):
        path.write_text(json.dumps([bad]))
        with pytest.raises(ValueError):
            native.publish_metrics({}, tmp_path)

def test_preflight_failure_never_starts_study_and_is_preserved(native, monkeypatch, tmp_path):
    root_at(native, monkeypatch, tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(native, "context", lambda snapshot: {"run": tmp_path})
    monkeypatch.setattr(native, "load_contract", lambda path: ({"identity": "before", "input_identity": {"archive_sha256": "b" * 64}}, "a" * 64))
    monkeypatch.setattr(native, "identity", lambda: "after")
    monkeypatch.setattr(native, "execute", lambda *args: pytest.fail("Preflight must not start any science"))
    calls = []
    monkeypatch.setattr(native, "report_batch", lambda ctx, commands: calls.extend(commands))
    args = type("Args", (), dict(contract="frozen.json", mode="all"))()
    assert native.run(args) == 2
    state = json.loads((tmp_path / "outputs/native/status.json").read_text())
    assert state["status"] == "FAILED" and state["source_unchanged"] is False
    assert state["commands"] == [] and state["worker_telemetry"]["status"] == "NOT_STARTED"
    assert any("outputs/native/status.json" in command for command in calls)
    with pytest.raises(FileExistsError):
        native.run(args)

    # A nonzero child exit also gates retained fitting, and raw
    # preflight contract/profile bytes remain externally hash-verifiable.
    from tdn.runtime import preflight, desktop_slurm, storage
    from tdn.analysis.tangent_response import telemetry
    second = tmp_path / "second-attempt"; (second / "outputs").mkdir(parents=True)
    monkeypatch.setattr(native, "context", lambda snapshot: {"run": second})
    profile = b'{"fixture_profile": true}\n'
    (tmp_path / "profile.json").write_bytes(profile)
    bound = dict(profile_path="profile.json", profile_sha256=hashlib.sha256(profile).hexdigest())
    contract = dict(identity=bound, input_identity={"archive_sha256": "b" * 64})
    raw = json.dumps(contract).encode(); (tmp_path / "frozen.json").write_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(native, "load_contract", lambda path: (contract, checksum))
    monkeypatch.setattr(native, "identity", lambda: bound)
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(desktop_slurm, "verify_allocation", lambda *args, **kwargs:
        {"job": dict(NumCPUs="4", AllocTRES="cpu=4,mem=8G,node=1", TimeLimit="00:20:00")})
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(telemetry, "start_recording", lambda *args: object())
    monkeypatch.setattr(telemetry, "stop_recording", lambda *args, **kwargs: {"status": "STOPPED"})
    monkeypatch.setattr(native, "report", lambda *args: None)
    invoked = []
    def failed_verify(ctx, command, argv):
        invoked.append(command)
        return subprocess.CompletedProcess(argv, 1)
    monkeypatch.setattr(native, "execute", failed_verify)
    assert native.run(args) == 1 and invoked == ["verify"]
    status = json.loads((second / "outputs/native/status.json").read_text())
    assert status["source_unchanged"] is True and status["status"] == "FAILED"
    assert (second / "outputs/native/contract.json").read_bytes() == raw
    assert (second / "outputs/native/profile.json").read_bytes() == profile
    assert not (second / "outputs/native/verification-release.json").exists()



def test_export_terminal_snapshot_and_refusals(native, monkeypatch, tmp_path):
    root_at(native, monkeypatch, tmp_path)
    run = tmp_path / "runs" / ("launch-" + "a" * 32 + "-a1")
    (run / "outputs/native").mkdir(parents=True)
    (run / "outputs/study/inputs/queries").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "logs/study.stdout.log").write_text("completed research\n")
    cache = run / "outputs/study/.font-cache"
    cache.mkdir()
    (cache / "fontlist.json").write_text('{"cache":true}')
    (run / ".raven").mkdir()
    (run / ".raven/retained.json").write_text('{"evidence":true}')
    inventory = dict(schema="raven.run/v1", run_id=run.name, state="RUNNING")
    (run / "run.json").write_text(json.dumps(inventory))
    (run / "outputs/native/status.json").write_text(json.dumps(dict(schema=native.SCHEMA, status="COMPLETED")))
    with pytest.raises(ValueError, match="terminal"):
        native.export_review(str(run))
    inventory["state"] = "COMPLETED"
    (run / "run.json").write_text(json.dumps(inventory))
    raw = b"retained-array"
    (run / "outputs/study/inputs/queries/q1.npz").write_bytes(raw)
    native.pack_study_evidence({"run": run}, run / "outputs/study")
    folder = native.export_review(str(run))
    index = json.loads((folder / "index.json").read_text())
    assert index["file_count"] > 4 and len(index["parts"]) == 1
    assert index["omitted_duplicates"][0]["path"] == "outputs/native/portable-study-evidence.zip"
    import zipfile
    with zipfile.ZipFile(folder / index["archive"]) as bundle:
        assert bundle.read(run.name + "/outputs/study/inputs/queries/q1.npz") == raw
        assert run.name + "/outputs/native/portable-study-evidence.zip" not in bundle.namelist()
        assert run.name + "/logs/study.stdout.log" in bundle.namelist()
        assert run.name + "/outputs/study/.font-cache/fontlist.json" not in bundle.namelist()
        assert run.name + "/.raven/retained.json" in bundle.namelist()
    assert hashlib.sha256((folder / index["archive"]).read_bytes()).hexdigest() == index["archive_sha256"]
    # Exercise the multipart upload boundary with small synthetic parts.
    monkeypatch.setattr(native, "UPLOAD_PART_BYTES", 512)
    second = native.export_review(str(run))
    assert second != folder and folder.is_dir()
    split_index = json.loads((second / "index.json").read_text())
    assert len(split_index["parts"]) > 1 and not (second / split_index["archive"]).exists()
    joined = b"".join((second / part["path"]).read_bytes() for part in split_index["parts"])
    assert hashlib.sha256(joined).hexdigest() == split_index["archive_sha256"]
    assert all(part["bytes"] <= 512 for part in split_index["parts"])
    import io
    with zipfile.ZipFile(io.BytesIO(joined)) as bundle:
        assert bundle.read(run.name + "/outputs/study/inputs/queries/q1.npz") == raw
    (run / "outputs/study/inputs/queries/q1.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw numerical evidence changed"):
        native.export_review(str(run))
    (run / "outputs/study/inputs/queries/q1.npz").write_bytes(raw)
    (run / "bad-link").symlink_to(run / "run.json")
    with pytest.raises(ValueError, match="symlinks"):
        native.export_review(str(run))
    with pytest.raises(ValueError, match="exact native"):
        native.export_review(str(tmp_path / "other"))


def test_full_inventory_uses_explicit_metadata_budget(native, monkeypatch, tmp_path):
    """The historical packing failure must not recur for a full-size inventory."""
    root_at(native, monkeypatch, tmp_path)
    run = tmp_path / "runs" / ("launch-" + "d" * 32 + "-a1")
    native_folder = run / "outputs/native"
    native_folder.mkdir(parents=True)
    arrays = run / "outputs/study/inputs/queries"
    arrays.mkdir(parents=True)
    for index in range(5200):
        (arrays / (f"q{index:05d}-" + "x" * 80 + ".npz")).write_bytes(b"array")
    (run / "run.json").write_text(json.dumps(dict(schema="raven.run/v1", run_id=run.name, state="FAILED")))
    (native_folder / "status.json").write_text(json.dumps(dict(schema=native.SCHEMA, status="FAILED")))
    manifest = native.pack_study_evidence({"run": run}, run / "outputs/study")
    assert len(manifest["files"]) == 5200
    assert (native_folder / "packing.json").stat().st_size > 1 << 20
    export = native.export_review(str(run))
    assert (export / "index.json").stat().st_size > 1 << 20
    index = json.loads((export / "index.json").read_text())
    assert index["file_count"] == 5203
    assert index["omitted_duplicates"][0]["path"] == "outputs/native/portable-study-evidence.zip"

@pytest.fixture
def captured(native, raven, monkeypatch, tmp_path):
    root = tmp_path / "Regularization checkout with spaces"
    (root / "scripts").mkdir(parents=True); (root / ".raven").mkdir()
    shutil.copyfile(SHELL, root / "scripts" / SHELL.name)
    shutil.copyfile(ROOT / ".raven/automation.json", root / ".raven/automation.json")
    plan = raven["launch"].resolve_plan([str(root / "scripts" / SHELL.name), "--project", str(root),
        "--recipe", "response-preconditioner-regularization", "--sbatch=--partition=local", "--sbatch=--chdir=" + str(root),
        "--", "--contract", "frozen.json"], cwd=root, environ={})
    plan = raven["launch_runtime"].materialize(plan)
    folder = Path(plan["runtime_bundle"]).parent
    run = root / "runs" / ("launch-" + plan["token"] + "-a1")
    (run / "outputs").mkdir(parents=True); (run / "logs").mkdir()
    (run / "metrics.jsonl").touch()
    record = dict(schema="raven.run/v1", run_id=run.name, state="RUNNING", job_id="98765",
                  attempt=1, experiment_id="response-preconditioner-regularization", paths={"metrics": "metrics.jsonl"},
                  metadata={"automation": {"execution": "slurm", "plan": {"token": plan["token"],
                  "plan_sha256": hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest()}}})
    (run / "run.json").write_text(json.dumps(record))
    root_at(native, monkeypatch, root)
    for key, value in dict(RAVEN_LAUNCH_PLAN=folder / "plan.json", RAVEN_SOURCE_SCRIPT=root / native.SCRIPT,
        RAVEN_PROJECT_ROOT=root, RAVEN_RUN_DIR=run, RAVEN_RUN_ID=run.name, SLURM_JOB_ID="98765",
        RAVEN_REPORT_PYTHON=plan["report_python"], RAVEN_REPORT_HELPER=folder / "report.py",
        RAVEN_RUNTIME_BUNDLE=plan["runtime_bundle"], TDN_RESPONSE_PRECONDITIONER_REGULARIZATION_SNAPSHOT=folder / "payload.sh").items():
        monkeypatch.setenv(key, str(value))
    return dict(root=root, run=run, folder=folder, plan=plan, snapshot=folder / "payload.sh")


def test_raven_regularization_plan_and_exact_capture(native, captured):
    ctx = native.context(captured["snapshot"])
    plan = ctx["plan"]
    assert plan["script_relative"] == native.SCRIPT and plan["execution"] == "slurm"
    assert plan["settings"]["cpu_memory"] is True and plan["gpu"] is False
    assert plan["effective_options"]["cpus-per-task"] == "4"
    assert plan["effective_options"]["mem"] == "8G"
    assert plan["effective_options"]["time"] == "00:20:00"
    assert plan["effective_options"]["output"].startswith("logs/response-preconditioner-regularization/")
    # Shared verifier still binds its old entrypoint by default.
    with pytest.raises(ValueError, match="exact captured"):
        native.common.context(captured["snapshot"])


@pytest.mark.parametrize("field,value", [("RAVEN_RUN_ID", "other"), ("RAVEN_PROJECT_ROOT", "/foreign"),
    ("SLURM_JOB_ID", "456"), ("RAVEN_REPORT_PYTHON", "/foreign/python")])
def test_raven_changed_identity_rejected(native, captured, monkeypatch, field, value):
    monkeypatch.setenv(field, value)
    with pytest.raises(ValueError):
        native.context(captured["snapshot"])


def test_raven_batch_reporting_uses_captured_runtime(native, captured):
    ctx = native.context(captured["snapshot"])
    path = captured["run"] / "outputs/raw.npz"; path.write_bytes(b"portable-array")
    with (path.parent / "metrics.json").open("wb") as stream:
        stream.truncate(65 << 20)
    (path.parent / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    native.report_batch(ctx, native.artifact_commands(ctx, path.parent) + [
        ["metric", "--phase", "regularization/diagnostic/withheld/random/ordered_8", "--step", "0", "--value", "rms=0.1"]])
    metrics = [json.loads(line) for line in (captured["run"] / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["metrics"]["rms"] == .1
    assert metrics[-1]["phase"].endswith("/ordered_8")
    assert "outputs/raw.npz" in (captured["run"] / ".raven/contracts/outputs.v1.json").read_text()
    contract = json.loads((captured["run"] / ".raven/contracts/outputs.v1.json").read_text())
    checked = importlib.import_module("raven.artifacts").validate_contract(contract, captured["run"])
    assert checked["valid"] and checked["status"] == "valid"
    assert checked["budget"]["bytes_read"] == 0  # Presence checks, not a false full-table parse claim.


def test_raven_study_admission_requires_sealed_verification(native, captured, monkeypatch):
    from tdn.runtime import preflight, desktop_slurm
    ctx = native.context(captured["snapshot"])
    monkeypatch.setattr(native, "load_contract", lambda path: ({"identity": "frozen", "input_identity": {"archive_sha256": "b" * 64}}, "a" * 64))
    monkeypatch.setattr(native, "identity", lambda: "frozen")
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(desktop_slurm, "verify_allocation", lambda *args, **kwargs:
        {"job": dict(NumCPUs="4", AllocTRES="cpu=4,mem=8G,node=1", TimeLimit="00:20:00")})
    output = ctx["run"] / "outputs/study"
    with pytest.raises((ValueError, OSError)):
        native.require_study_context("frozen.json", output)
    verification = ctx["run"] / "outputs/verification"; verification.mkdir()
    for name in ("summary.json", "checks.json", "artifacts.json"):
        (verification / name).write_text("{}")
    (verification / "verification.xml").write_text("<testsuites/>")
    native_folder = ctx["run"] / "outputs/native"; native_folder.mkdir()
    native.release_verification(native_folder, verification, "a" * 64)
    admitted = native.require_study_context("frozen.json", output)
    assert admitted["run"] == ctx["run"]
    assert admitted["input_identity"] == {"archive_sha256": "b" * 64}
    (verification / "checks.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence changed"):
        native.require_study_context("frozen.json", output)


def test_raven_captured_shell_executes_original_checkout_srun(native, captured, monkeypatch):
    root = captured["root"]
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    (root / "scripts/response_preconditioner_regularization_native.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path('worker-call.json').write_text(json.dumps({'argv':sys.argv[1:],"
        "'root':os.environ['TDN_PROJECT_ROOT'],'snapshot':os.environ['TDN_RESPONSE_PRECONDITIONER_REGULARIZATION_SNAPSHOT'],"
        "'mode':os.environ['TDN_EXECUTION_MODE']}))\n")
    bin_dir = root / "fixture-bin"; bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text("#!" + sys.executable + "\nimport os,sys\nargs=sys.argv[1:]\n"
                    "while args and args[0].startswith('--'):args.pop(0)\nos.execv(args[0],args)\n")
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(captured["snapshot"]), "--contract", "frozen.json", "--mode", "all"],
                            text=True, capture_output=True, cwd=root)
    assert result.returncode == 0, result.stderr
    record = json.loads((root / "worker-call.json").read_text())
    assert record == dict(argv=["run", "--contract", "frozen.json", "--mode", "all"], root=str(root),
                          snapshot=str(captured["snapshot"]), mode="desktop-slurm")
