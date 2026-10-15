"""Native launch policy fixtures; these are never native Slurm/CUDA evidence."""
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
SHELL = ROOT / "scripts/response_preconditioner_verify.sh"


@pytest.fixture
def native(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("preconditioner_native_under_test", ROOT / "scripts/response_preconditioner_native.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def raven(monkeypatch):
    root = Path(os.environ.get("TDN_RAVEN_TEST_ROOT", "/workspace/Raven"))
    if not (root / "raven/launch.py").is_file():
        pytest.skip("Optional Raven compatibility needs a trusted Raven checkout")
    monkeypatch.syspath_prepend(str(root))
    return {name: importlib.import_module("raven." + name)
            for name in ("launch", "launch_runtime", "reporting", "metrics", "artifacts")}


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("RAVEN_", "SLURM_", "TDN_", "SBATCH_"))}


def test_shell_help_and_rejects_unallocated_execution():
    result = subprocess.run(["bash", str(SHELL), "--help"], text=True, capture_output=True, env=clean_env())
    assert result.returncode == 0 and "--contract PATH" in result.stdout
    result = subprocess.run(["bash", str(SHELL)], text=True, capture_output=True, env=clean_env())
    assert result.returncode == 2 and "use raven launch" in result.stderr


def test_native_script_syntax():
    subprocess.run(["bash", "-n", str(SHELL)], check=True)


def test_recipe_and_native_plan_preserve_bounded_cpu_request(raven, tmp_path):
    root = tmp_path / "TDN checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(SHELL, root / "scripts" / SHELL.name)
    (root / ".raven").mkdir()
    shutil.copyfile(ROOT / ".raven/automation.json", root / ".raven/automation.json")
    plan = raven["launch"].resolve_plan([str(root / "scripts" / SHELL.name), "--project", str(root),
        "--recipe", "response-preconditioner", "--sbatch=--partition=local", "--sbatch=--chdir=" + str(root),
        "--", "--contract", ".runtime/response-preconditioner-launches/frozen.json"], cwd=root, environ={})
    assert plan["execution"] == "slurm"
    assert plan["settings"]["cpu_memory"] is True and plan["gpu"] is False
    assert plan["features"] == ["experiment", "evidence", "artifacts"]
    assert plan["effective_options"]["cpus-per-task"] == "4"
    assert plan["effective_options"]["mem"] == "8G"
    assert plan["effective_options"]["time"] == "00:20:00"
    assert plan["workdir"] == str(root)
    assert plan["effective_options"]["output"].startswith("logs/response-preconditioner/")


@pytest.fixture
def captured(native, raven, monkeypatch, tmp_path):
    root = tmp_path / "captured checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(SHELL, root / "scripts" / SHELL.name)
    plan = raven["launch"].resolve_plan([str(root / "scripts" / SHELL.name), "--project", str(root),
        "--sbatch=--partition=local", "--sbatch=--chdir=" + str(root)], cwd=root, environ={})
    plan = raven["launch_runtime"].materialize(plan)
    folder = Path(plan["runtime_bundle"]).parent
    run = root / "runs" / ("launch-" + plan["token"] + "-a1")
    (run / "outputs").mkdir(parents=True)
    (run / "logs").mkdir()
    record = {"schema": "raven.run/v1", "run_id": run.name, "state": "RUNNING", "job_id": "98765",
              "metadata": {"automation": {"execution": "slurm", "plan": {"token": plan["token"],
                  "plan_sha256": hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest()}}}}
    (run / "run.json").write_text(json.dumps(record))
    monkeypatch.setattr(native, "ROOT", root)
    for key, value in dict(RAVEN_LAUNCH_PLAN=folder / "plan.json", RAVEN_SOURCE_SCRIPT=root / native.SCRIPT,
        RAVEN_PROJECT_ROOT=root, RAVEN_RUN_DIR=run, RAVEN_RUN_ID=run.name, SLURM_JOB_ID="98765",
        RAVEN_REPORT_PYTHON=plan["report_python"], RAVEN_REPORT_HELPER=folder / "report.py",
        RAVEN_RUNTIME_BUNDLE=plan["runtime_bundle"]).items():
        monkeypatch.setenv(key, str(value))
    return dict(root=root, run=run, folder=folder, plan=plan, snapshot=folder / "payload.sh")


def test_exact_captured_context_resolves_original_checkout(native, captured):
    ctx = native.context(captured["snapshot"])
    assert ctx["run"] == captured["run"] and ctx["plan"]["execution"] == "slurm"


@pytest.mark.parametrize("field,value", [("RAVEN_RUN_ID", "other"), ("RAVEN_PROJECT_ROOT", "/foreign"),
    ("RAVEN_SOURCE_SCRIPT", "/foreign/script.sh"), ("SLURM_JOB_ID", "123"),
    ("RAVEN_REPORT_PYTHON", "/foreign/python"), ("RAVEN_RUNTIME_BUNDLE", "/foreign/runtime.pyz")])
def test_changed_launch_identity_rejected(native, captured, monkeypatch, field, value):
    monkeypatch.setenv(field, value)
    with pytest.raises(ValueError):
        native.context(captured["snapshot"])


def test_changed_captured_or_checkout_script_rejected(native, captured):
    source = captured["root"] / native.SCRIPT
    source.write_text(source.read_text() + "\n# changed while queued\n")
    with pytest.raises(ValueError, match="launcher changed"):
        native.context(captured["snapshot"])


def test_capture_cannot_follow_symlink_snapshot(native, captured):
    link = captured["root"] / "alias.sh"
    link.symlink_to(captured["snapshot"])
    with pytest.raises(ValueError):
        native.context(link)


def test_prepare_exclusive_identity_and_changed_source_detection(native, tmp_path, monkeypatch, capsys):
    root = tmp_path / "project"; root.mkdir()
    monkeypatch.setattr(native, "ROOT", root)
    monkeypatch.setattr(native, "identity", lambda: {"source": {"sha256": "a" * 64}})
    for key in ("RAVEN_LAUNCH_PLAN", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT"):
        monkeypatch.delenv(key, raising=False)
    native.prepare(); first = capsys.readouterr().out.strip()
    native.prepare(); second = capsys.readouterr().out.strip()
    assert first != second and (root / "logs/response-preconditioner").is_dir()
    value, digest = native.load_contract(first)
    assert value["scientific_promotion"] is False and value["device"] == "cpu"
    assert len(digest) == 64
    assert not (root / first).stat().st_mode & 0o222


def test_source_identity_ignores_outputs_but_tracks_code_and_recipe(native, tmp_path, monkeypatch):
    root = tmp_path / "project"; root.mkdir()
    monkeypatch.setattr(native, "ROOT", root)
    for name in ("tdn", "reference", "scripts", "configs", "tests", ".raven", "runs"):
        (root / name).mkdir()
    for name in ("requirements.txt", "pyproject.toml", ".raven/automation.json", "scripts/worker.py"):
        (root / name).write_text("initial\n")
    first = native.source_identity()
    (root / "runs/report.json").write_text("observed output")
    assert native.source_identity() == first
    (root / "scripts/worker.py").write_text("new implementation")
    assert native.source_identity() != first
    second = native.source_identity()
    (root / ".raven/automation.json").write_text("new recipe")
    assert native.source_identity() != second


@pytest.mark.parametrize("changed", [{"NumCPUs": "16"}, {"AllocTRES": "cpu=4,mem=64G"},
    {"AllocTRES": "cpu=4,mem=8G,gres/gpu=1"}, {"TimeLimit": "00:45:00"},
    {"TimeLimit": "UNLIMITED"}, {"AllocTRES": "cpu=4"}])
def test_allocation_resource_expansion_rejected(native, changed):
    job = dict(NumCPUs="4", AllocTRES="cpu=4,mem=8G,node=1", TimeLimit="00:20:00")
    native.resource_check({"job": job})
    with pytest.raises(ValueError):
        native.resource_check({"job": {**job, **changed}})


def test_shared_venv_lock_rejects_setup_writer(native, tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(native, "ROOT", tmp_path)
    (tmp_path / ".cache").mkdir()
    with (tmp_path / ".cache/carc-phase.lock").open("a+b") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="venv is being changed"):
            with native.venv_lock():
                pass


def test_metrics_preserve_split_method_and_task_without_aggregate(native, tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    output = tmp_path / "smoke"; output.mkdir()
    rows = [dict(kind="endpoint", split="diagnostic", method="strang", rms=.1, maximum=.2,
                 mean_error=.01, centered_rms=.09, horizon=.1, direction=None),
            dict(kind="action", split="diagnostic", method="learned", rms=.03, maximum=.06,
                 mean_error=.001, centered_rms=.02, horizon=.1, direction=7)]
    (output / "metrics.json").write_text(json.dumps(rows))
    calls = []
    monkeypatch.setattr(native, "report", lambda ctx, *args: calls.append(args))
    native.publish_smoke_metrics({}, output)
    assert len(calls) == 2
    assert "smoke/endpoint/diagnostic/strang" in calls[0]
    assert "smoke/action/diagnostic/learned" in calls[1]
    assert "direction=7" in calls[1]


def test_stage_registers_exact_verification_xml_and_binary_artifacts(native, tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    output = tmp_path / "outputs/verification"; output.mkdir(parents=True)
    for name in ("protocol.json", "source.json", "software.json", "artifacts.json", "checks.json"):
        (output / name).write_text("{}")
    (output / "verification.xml").write_text("<testsuites/>")
    (output / "summary.json").write_text(json.dumps(dict(command="verify", status="COMPLETED",
        scientific_promotion=False, device="cpu", decision="IMPLEMENTATION_ONLY")))
    calls = []
    monkeypatch.setattr(native, "report", lambda ctx, *args: calls.append(args))
    native.stage_evidence({"run": tmp_path}, output, "verify", 0)
    assert any("outputs/verification/verification.xml" in call for call in calls)
    assert not any("outputs/verification/junit.xml" in call for call in calls)
    (output / "arrays.npz").write_bytes(b"\xffbinary\x00")
    native.artifact({"run": tmp_path}, output / "arrays.npz")
    assert calls[-1] == ("artifact", "outputs/verification/arrays.npz")
    (output / "verification.xml").unlink()
    with pytest.raises(ValueError, match="missing required"):
        native.stage_evidence({"run": tmp_path}, output, "verify", 0)


def test_stage_success_cannot_promote_science_or_hide_failed_summary(native, tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    monkeypatch.setattr(native, "artifact", lambda *args: None)
    (tmp_path / "summary.json").write_text(json.dumps(dict(command="verify", status="FAILED",
        scientific_promotion=False, device="cpu", decision="IMPLEMENTATION_ONLY")))
    with pytest.raises(ValueError, match="matching nonpromoting"):
        native.stage_evidence({}, tmp_path, "verify", 0)
    # A failed child with a partial inventory stays inspectable.
    native.stage_evidence({}, tmp_path, "verify", 1)


def test_failed_preflight_is_retained_without_child_work(native, tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(native, "context", lambda *args: {"run": tmp_path})
    monkeypatch.setattr(native, "load_contract", lambda *args: ({"identity": {"source": "before"}}, "hash"))
    monkeypatch.setattr(native, "identity", lambda: {"source": "after"})
    registered = []
    monkeypatch.setattr(native, "artifact", lambda ctx, path: registered.append(path.name))
    monkeypatch.setattr(native, "execute", lambda *args: pytest.fail("Preflight must not execute child work"))
    args = type("Args", (), {"contract": "frozen.json", "mode": "all"})()
    assert native.run(args) == 2
    status = json.loads((tmp_path / "outputs/native/status.json").read_text())
    assert status["status"] == "FAILED" and status["scientific_promotion"] is False
    assert status["source_unchanged"] is False and status["worker_telemetry"]["status"] == "NOT_STARTED"
    assert "status.json" in registered
    with pytest.raises(FileExistsError):
        native.run(args)


def test_execution_keeps_stdout_stderr_and_child_exit(native, tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(native, "report", lambda ctx, *args: calls.append(args))
    child = native.execute({"run": tmp_path}, "smoke", [sys.executable, "-c",
        "import sys; print('real stdout'); print('real stderr',file=sys.stderr); sys.exit(3)"])
    assert child.returncode == 3
    assert (tmp_path / "logs/smoke.stdout.log").read_text() == "real stdout\n"
    assert (tmp_path / "logs/smoke.stderr.log").read_text() == "real stderr\n"
    assert len(calls) == 2 and all(call[0] == "log" for call in calls)


def test_captured_shell_uses_original_checkout_and_srun(native, captured, monkeypatch):
    root = captured["root"]
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    (root / "scripts/response_preconditioner_native.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path('worker-call.json').write_text(json.dumps({'argv':sys.argv[1:],"
        "'root':os.environ['TDN_PROJECT_ROOT'],'snapshot':os.environ['TDN_RESPONSE_PRECONDITIONER_SNAPSHOT'],"
        "'mode':os.environ['TDN_EXECUTION_MODE']}))\n")
    bin_dir = root / "fixture-bin"; bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text("#!" + sys.executable + "\nimport os,sys\n"
                    "args=sys.argv[1:]\nwhile args and args[0].startswith('--'):args.pop(0)\n"
                    "os.execv(args[0],args)\n")
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(captured["snapshot"]), "--contract", "frozen.json", "--mode", "verify"],
                            text=True, capture_output=True, cwd=root)
    assert result.returncode == 0, result.stderr
    record = json.loads((root / "worker-call.json").read_text())
    assert record == {"argv": ["run", "--contract", "frozen.json", "--mode", "verify"],
                      "root": str(root), "snapshot": str(captured["snapshot"]), "mode": "desktop-slurm"}
