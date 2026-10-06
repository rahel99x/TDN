"""CPU mechanism workflow contracts; fake Slurm calls are not CARC validation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("protocol.json", "summary.json", "config.json", "mechanism-audit.json", "temporal.json",
             "coordinates.json", "structure.json")


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mechanism_controller", ROOT / "scripts" / "research_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(f"{name} source\n")
    (root / "tests" / "test_mechanism_example.py").write_text("def test_ok(): pass\n")
    (root / "tests" / "test_research_example.py").write_text("def test_ok(): pass\n")
    (root / "configs" / "mechanism-audit.yaml").write_text("benchmark_suite: mechanism-audit\n")
    (root / "requirements.txt").write_text("numpy==2.2.6\n")
    (root / "pyproject.toml").write_text("[project]\nname='tdn'\n")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.cw, "ROOT", root)
    monkeypatch.setattr(module.cw, "CARC_ROOT", root)
    monkeypatch.setattr(module.cw.getpass, "getuser", lambda: "aadaniel")
    for name in ("TDN_LOCAL_TEST_ROOT", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT", "SLURM_JOB_ID", "SLURM_STEP_ID",
                 "SLURM_JOB_ACCOUNT", "CONDA_PREFIX", "CONDA_SHLVL", "CARC_ACCOUNT", "TORCH_VERSION", "TDN_CPU_PARTITION"):
        monkeypatch.delenv(name, raising=False)
    return module


def plan(module, run_id="mechanisms-cpu", *extra):
    return module.prepare(module.parser().parse_args(["start", "--mechanism-audit", "--run-id", run_id, *extra]))


def finish_cpu(module, workflow):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "config.yaml").write_bytes(Path(workflow["original_config"]).read_bytes())
    module.cw.atomic_json(base / "research-workflow.json", workflow)
    module.cw.atomic_json(base / "jobs.json", [])
    experiment = base / "experiment"
    experiment.mkdir()
    protocol = {"version": 1, "benchmark_suite": "mechanism-audit", "config": {"fixed": True}}
    module.cw.atomic_json(experiment / "protocol.json", protocol)
    module.cw.atomic_json(experiment / "summary.json", {"status": "COMPLETED", "device": "cpu",
                         "benchmark_suite": "mechanism-audit", "training_performed": False})
    for name in ARTIFACTS[2:]:
        module.cw.atomic_json(experiment / name, {"fixture": "sealed controller fixture"})
    manifest = {"version": 1, "source_tree_sha256": workflow["source_tree_sha256"],
                "config_file_sha256": workflow["config_sha256"],
                "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                                            allow_nan=False).encode()).hexdigest(),
                "files": {name: module.cw.digest(experiment / name) for name in ARTIFACTS}}
    module.cw.atomic_json(experiment / "manifest.json", manifest)
    (experiment / "COMPLETED").write_text(module.cw.digest(experiment / "manifest.json") + "\n")
    module.cw.atomic_json(module.software_path(workflow), {"packages": {"torch": workflow["torch_version"]}})
    module.cw.atomic_json(module.state_path(workflow), {"status": "COMPLETED", "exit_code": 0, "stage": "experiment",
                         "source_sha256": workflow["source_sha256"], "config_sha256": workflow["config_sha256"],
                         "software_sha256": module.cw.digest(module.software_path(workflow))})
    return base


def reseal(module, base, *, protocol=None, summary=None, manifest_update=None):
    experiment = base / "experiment"
    manifest_path = experiment / "manifest.json"
    manifest = module.cw.read_json(manifest_path)
    if protocol is not None:
        module.cw.atomic_json(experiment / "protocol.json", protocol)
        manifest["files"]["protocol.json"] = module.cw.digest(experiment / "protocol.json")
        manifest["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                                               allow_nan=False).encode()).hexdigest()
    if summary is not None:
        module.cw.atomic_json(experiment / "summary.json", summary)
        manifest["files"]["summary.json"] = module.cw.digest(experiment / "summary.json")
    if manifest_update:
        manifest.update(manifest_update)
    module.cw.atomic_json(manifest_path, manifest)
    (experiment / "COMPLETED").write_text(module.cw.digest(manifest_path) + "\n")


def test_cpu_preview_no_writes_or_scheduler_calls(controller, monkeypatch, capsys):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("Preview contacted scheduler"))
    monkeypatch.setattr(controller.cw, "actual_policy", lambda **kw: pytest.fail("Preview required allocation"))
    assert controller.main(["start", "--mechanism-audit", "--submit", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "carc-mechanisms-" in output and "Suite: mechanism-audit" in output
    assert "no neural training or GPU successor" in output
    assert output.count("sbatch --parsable") == 1
    for item in ("--account=anakano_81", "--cpus-per-task=4", "--mem=16G", "--time=00:30:00"):
        assert item in output
    assert "--gpus-per-task" not in output and "--dependency" not in output and "--requeue" not in output
    assert not (controller.ROOT / "runs").exists() and not (controller.ROOT / ".cache").exists()


def test_login_controller_does_not_import_numerical_packages(controller):
    code = r'''
import importlib.abc, importlib.util, pathlib, sys
class RejectNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy','yaml','tdn'}:
            raise RuntimeError('Numerical import on login: '+fullname)
sys.meta_path.insert(0, RejectNumerics())
spec=importlib.util.spec_from_file_location('mechanism_preview',sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.ROOT=module.cw.ROOT=pathlib.Path(sys.argv[2])
raise SystemExit(module.main(['start','--mechanism-audit']))
'''
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "scripts" / "research_workflow.py"),
                             str(controller.ROOT)], cwd=controller.ROOT, env=env, text=True,
                            capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout and not (controller.ROOT / "runs").exists()


def test_submission_one_cpu_job_immutable_configuration(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller.cw, "live_check", lambda *a: {"account": "anakano_81"})
    def command(args, *, env=None, **kwargs):
        calls.append(args)
        assert args[0] == "sbatch" and env["TDN_RESEARCH_PHASE"] == "cpu"
        assert Path(env["TDN_RESEARCH_WORKFLOW"]).is_file()
        return SimpleNamespace(stdout="12699999;cluster\n")
    monkeypatch.setattr(controller.cw, "command", command)
    assert controller.main(["start", "--mechanism-audit", "--run-id", "submitted", "--submit"]) == 0
    assert len(calls) == 1
    base = controller.ROOT / "runs" / "submitted"
    workflow = controller.load(base / "research-workflow.json")
    assert workflow["benchmark_suite"] == "mechanism-audit"
    assert workflow["config_sha256"] == controller.cw.digest(base / "config.yaml")
    assert not (base / "config.yaml").stat().st_mode & 0o222
    assert not (base / "research-workflow.json").stat().st_mode & 0o222
    assert controller.cw.read_json(base / "jobs.json")[0]["job_id"] == "12699999"


def test_worker_runs_only_mechanism_tests_and_training_free_cli(controller):
    workflow = plan(controller, "commands", "--smoke")
    commands = controller.worker_commands(workflow)
    assert [name for name, _ in commands] == ["tests", "experiment"]
    tests, engine = [args for _, args in commands]
    assert "test_mechanism_example.py" in " ".join(tests)
    assert "test_research_example.py" not in " ".join(tests)
    assert tests[tests.index("-m", 3) + 1] == "not gpu"
    assert engine == [str(controller.ROOT / ".venv" / "bin" / "python"),
                      str(controller.ROOT / "scripts" / "mechanism_audit.py"),
                      "--config", workflow["config_path"], "--run-dir",
                      str(Path(workflow["run_dir"]) / "experiment"), "--smoke"]
    assert "--junitxml" in tests and "--basetemp" in tests


def test_tower_identifies_cpu_suite_resources(controller, monkeypatch):
    workflow = plan(controller)
    monkeypatch.setenv("SLURM_JOB_ID", "12699999")
    observed = {}
    def begin_report(base, **kwargs):
        observed.update(kwargs)
        return Path(base) / "tower" / "fake-attempt"
    monkeypatch.setattr(controller.cw, "reporting_api", lambda: SimpleNamespace(begin_report=begin_report))
    controller.cw.begin_workflow_report(workflow, "cpu", research=True)
    assert observed["name"] == "TDN/research/mechanism-audit/cpu"
    assert observed["parameters"]["benchmark_suite"] == "mechanism-audit"
    assert observed["resources"]["gpus"] == 0 and observed["resources"]["time_seconds"] == 1800
    assert observed["resources"]["account"] == "anakano_81"


def test_allocation_ownership_checked_before_attempt_evidence(controller, monkeypatch):
    workflow = plan(controller)
    monkeypatch.setattr(controller.cw, "actual_policy", lambda **kw: None)
    monkeypatch.setenv("SLURM_JOB_ID", "12699999")
    monkeypatch.setattr(controller.cw, "command", lambda *a: SimpleNamespace(stdout=
        "JobId=12699999 JobName=tdn-research-cpu Account=another_account UserId=aadaniel(123) "
        "Comment=tdn-research:mechanisms-cpu:cpu JobState=RUNNING"))
    with pytest.raises(ValueError, match="not this research workflow"):
        controller.worker(workflow, "cpu")
    assert not Path(workflow["run_dir"]).exists()


def test_artifacts_verify_without_dataset_or_checkpoint(controller):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    manifest = controller.verify_experiment(workflow)
    assert set(manifest["files"]) == set(ARTIFACTS)
    assert not (base / "experiment" / "dataset.pt").exists()
    assert not (base / "experiment" / "checkpoints").exists()


@pytest.mark.parametrize("damage", ["file", "missing", "marker", "source", "config", "version", "protocol", "suite", "device"])
def test_corrupt_or_mismatched_artifacts_rejected(controller, damage):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    if damage == "file":
        (base / "experiment" / "temporal.json").write_text("{}\n")
    elif damage == "missing":
        (base / "experiment" / "coordinates.json").unlink()
    elif damage == "marker":
        (base / "experiment" / "COMPLETED").write_text("incorrect\n")
    elif damage in ("source", "config", "version"):
        key = {"source": "source_tree_sha256", "config": "config_file_sha256", "version": "version"}[damage]
        reseal(controller, base, manifest_update={key: 2 if damage == "version" else "incorrect"})
    elif damage == "protocol":
        reseal(controller, base, protocol={"version": 2, "benchmark_suite": "mechanism-audit"})
    else:
        reseal(controller, base, summary={"status": "COMPLETED", "device": "cuda" if damage == "device" else "cpu",
                                         "benchmark_suite": "architecture" if damage == "suite" else "mechanism-audit"})
    with pytest.raises(ValueError):
        controller.verify_experiment(workflow)


@pytest.mark.parametrize("damage", ["source", "config"])
def test_changed_execution_fingerprint_rejected(controller, damage):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    target = controller.ROOT / "scripts" / "source.txt" if damage == "source" else base / "config.yaml"
    target.write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        controller.load(base / "research-workflow.json")


def test_cpu_suite_cannot_authorize_gpu(controller, monkeypatch, capsys):
    workflow = plan(controller)
    finish_cpu(controller, workflow)
    monkeypatch.setattr(controller.cw, "live_check", lambda *a: pytest.fail("CPU-only run checked GPU allocation"))
    assert controller.main(["benchmark", workflow["run_id"], "--submit"]) == 2
    assert "CPU-only" in capsys.readouterr().err
    assert len(list((controller.ROOT / "runs").iterdir())) == 1
    gpu = {**workflow, "phase": "gpu", "source_workflow": "anything",
           "resources": {"gpu": {**controller.RESOURCE, "partition": "gpu"}}}
    with pytest.raises(ValueError, match="CPU-only"):
        controller.validate(gpu)
    with pytest.raises(ValueError, match="CPU-only"):
        controller.worker_commands(gpu)


def test_status_logs_collect_preserve_completed_artifacts(controller, capsys):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    (base / "logs").mkdir()
    (base / "logs" / "cpu-123.out").write_text("mechanism audit complete\n")
    (base / "pytest-work").mkdir()
    (base / "pytest-work" / "scratch.txt").write_text("excluded\n")
    hashes = {name: controller.cw.digest(base / "experiment" / name) for name in ARTIFACTS}
    controller.status(workflow)
    controller.logs(workflow, 10)
    archive = controller.collect(workflow)
    output = capsys.readouterr().out
    assert "experiment: COMPLETED" in output and "mechanism audit complete" in output
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert any(name.endswith("/experiment/temporal.json") for name in names)
    assert not any("pytest-work" in name for name in names)
    assert hashes == {name: controller.cw.digest(base / "experiment" / name) for name in ARTIFACTS}


def test_mechanism_suite_cannot_be_combined_with_training_suites(controller):
    for flag in ("--neural-benchmarks", "--neural-replication"):
        with pytest.raises(SystemExit):
            controller.parser().parse_args(["start", "--mechanism-audit", flag])


@pytest.fixture
def wrapper(tmp_path):
    if os.name == "nt":
        pytest.skip("Bash wrapper requires a POSIX host")
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts" / "carc_mechanisms.sh", scripts / "carc_mechanisms.sh")
    (scripts / "carc_research.sh").write_text("#!/usr/bin/env bash\nexec " + str(sys.executable) +
        " -c 'import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(37)' \"$@\"\n")
    return root


@pytest.mark.parametrize("args", [(), ("--submit",), ("start", "--smoke", "--run-id", "quoted run")])
def test_wrapper_defaults_preserve_quoting(wrapper, args):
    result = subprocess.run(["bash", str(wrapper / "scripts" / "carc_mechanisms.sh"), *args], cwd=wrapper,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 37, result.stderr
    expected = list(args[1:] if args and args[0] == "start" else args)
    assert json.loads(result.stdout) == ["start", "--mechanism-audit", "--config",
        str(wrapper / "scripts" / ".." / "configs" / "mechanism-audit.yaml"), *expected]


@pytest.mark.parametrize("args", [("status", "latest"), ("logs", "run-123", "--lines", "200"),
                                  ("collect", "run-123"), ("--mechanism-audit", "--config=config with spaces.yaml", "--submit")])
def test_wrapper_passes_read_commands_and_custom_config(wrapper, args):
    result = subprocess.run(["bash", str(wrapper / "scripts" / "carc_mechanisms.sh"), *args], cwd=wrapper,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 37, result.stderr
    assert json.loads(result.stdout) == (["start", *args] if args[0].startswith("--") else list(args))


def test_wrapper_rejects_gpu_and_explains_help(wrapper):
    command = ["bash", str(wrapper / "scripts" / "carc_mechanisms.sh")]
    result = subprocess.run([*command, "benchmark", "run-123", "--submit"], cwd=wrapper,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 2 and "CPU-only" in result.stderr
    result = subprocess.run([*command, "--help"], cwd=wrapper,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0 and "1,200-second" in result.stdout
