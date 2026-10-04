"""Mocked CARC safety/workflow contracts; these tests do not exercise Slurm/GPU."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("research_controller_test", ROOT / "scripts" / "research_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(f"{name} source\n")
    (root / "tests" / "test_research_example.py").write_text("def test_ok(): pass\n")
    (root / "configs" / "research.yaml").write_text("purpose: research\n")
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


def plan(module, run_id="research-test", *extra):
    return module.prepare(module.parser().parse_args(["start", "--run-id", run_id, *extra]))


def persist(module, workflow):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "config.yaml").write_bytes(Path(workflow["original_config"]).read_bytes())
    module.cw.atomic_json(base / "research-workflow.json", workflow)
    module.cw.atomic_json(base / "jobs.json", [])
    return base


def finish_cpu(module, workflow, *, headroom=True):
    base = persist(module, workflow)
    experiment = base / "experiment"
    experiment.mkdir()
    (experiment / "dataset.pt").write_bytes(b"small test dataset")
    (experiment / "protocol.json").write_text('{"scope":"mock"}\n')
    module.cw.atomic_json(experiment / "summary.json", {"status": "COMPLETED", "device": "cpu", "headroom": {"passed": headroom}})
    manifest = {"source_tree_sha256": workflow["source_tree_sha256"],
                "config_file_sha256": workflow["config_sha256"],
                "protocol_sha256": hashlib.sha256(b'{"scope":"mock"}').hexdigest(),
                "files": {name: module.cw.digest(experiment / name) for name in ("dataset.pt", "protocol.json", "summary.json")}}
    module.cw.atomic_json(experiment / "manifest.json", manifest)
    (experiment / "COMPLETED").write_text(module.cw.digest(experiment / "manifest.json") + "\n")
    module.cw.atomic_json(module.software_path(workflow), {"packages": {"torch": workflow["torch_version"]}})
    module.cw.atomic_json(module.state_path(workflow), {"status": "COMPLETED", "exit_code": 0, "stage": "experiment",
                         "source_sha256": workflow["source_sha256"], "config_sha256": workflow["config_sha256"],
                         "software_sha256": module.cw.digest(module.software_path(workflow))})
    return base


def test_default_preview_is_one_bounded_cpu_job_without_writes(controller, monkeypatch, capsys):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("Preview called an external command"))
    assert controller.main(["start", "--run-id", "preview"]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 1
    assert "--account=anakano_81" in output
    assert "--cpus-per-task=4" in output and "--mem=16G" in output and "--time=00:30:00" in output
    assert "--gpus-per-task" not in output and "DRY RUN" in output
    assert not (controller.ROOT / "runs").exists()
    assert not (controller.ROOT / ".cache").exists()


def test_explicit_dry_run_wins_over_submit(controller, monkeypatch):
    monkeypatch.setattr(controller.cw, "actual_policy", lambda **kw: pytest.fail("Preview required CARC policy"))
    assert controller.main(["start", "--submit", "--dry-run"]) == 0
    assert not (controller.ROOT / "runs").exists()


def test_actual_identity_checked_before_controller_lock(controller, monkeypatch):
    monkeypatch.setattr(controller.cw.getpass, "getuser", lambda: "someone_else")
    assert controller.main(["start", "--submit"]) == 2
    assert not (controller.ROOT / ".cache").exists()


def test_submission_checks_policy_then_creates_and_records_one_job(controller, monkeypatch):
    calls = []
    def check(workflow, phase):
        calls.append("policy")
        assert not Path(workflow["run_dir"]).exists()
        return {"account": "anakano_81"}
    def command(args, *, env=None, **kwargs):
        calls.append("sbatch")
        assert args[0] == "sbatch"
        assert "--dependency" not in " ".join(args)
        assert env["TDN_RESEARCH_PHASE"] == "cpu"
        assert Path(env["TDN_RESEARCH_WORKFLOW"]).is_file()
        assert (Path(env["TDN_RESEARCH_WORKFLOW"]).parent / "logs").is_dir()
        return SimpleNamespace(stdout="9876;cluster\n")
    monkeypatch.setattr(controller.cw, "live_check", check)
    monkeypatch.setattr(controller.cw, "command", command)
    assert controller.main(["start", "--run-id", "submitted", "--submit"]) == 0
    assert calls == ["policy", "sbatch"]
    base = controller.ROOT / "runs" / "submitted"
    assert controller.cw.read_json(base / "jobs.json")[0]["job_id"] == "9876"
    assert controller.cw.read_json(controller.ROOT / "runs" / ".research-latest.json")["run_id"] == "submitted"
    assert not (controller.ROOT / "runs" / ".carc-latest.json").exists()


@pytest.mark.parametrize("returned", ["Submitted batch job 12", "12\n13", "0", "12;cluster;extra"])
def test_ambiguous_job_id_fails_preserving_manifest(controller, monkeypatch, returned):
    monkeypatch.setattr(controller.cw, "live_check", lambda *a: {})
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: SimpleNamespace(stdout=returned))
    assert controller.main(["start", "--run-id", "ambiguous", "--submit"]) == 2
    assert (controller.ROOT / "runs" / "ambiguous" / "research-workflow.json").is_file()
    assert controller.cw.read_json(controller.ROOT / "runs" / "ambiguous" / "jobs.json") == []


def test_failed_live_policy_creates_no_run(controller, monkeypatch):
    def check(*args):
        raise ValueError("No account authorization")
    monkeypatch.setattr(controller.cw, "live_check", check)
    assert controller.main(["start", "--run-id", "blocked", "--submit"]) == 2
    assert not (controller.ROOT / "runs").exists()


def test_gpu_preview_requires_real_completed_headroom_and_preserves_cpu(controller, monkeypatch, capsys):
    workflow = plan(controller, "cpu-source")
    finish_cpu(controller, workflow)
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("GPU preview contacted scheduler"))
    assert controller.main(["benchmark", "cpu-source", "--run-id", "gpu-preview"]) == 0
    output = capsys.readouterr().out
    assert "--gpus-per-task=a100:1" in output and "--constraint=a100-40gb" in output
    assert "--time=00:30:00" in output
    assert not (controller.ROOT / "runs" / "gpu-preview").exists()


def test_no_headroom_blocks_gpu_before_submission(controller, monkeypatch):
    finish_cpu(controller, plan(controller, "no-headroom"), headroom=False)
    monkeypatch.setattr(controller.cw, "live_check", lambda *a: pytest.fail("Ineligible run checked GPU allocation"))
    assert controller.main(["benchmark", "no-headroom", "--submit"]) == 2
    assert len(list((controller.ROOT / "runs").iterdir())) == 1


@pytest.mark.parametrize("damage", ["dataset", "config", "source", "marker", "software", "phase"])
def test_benchmark_rejects_stale_or_incomplete_prerequisites(controller, damage):
    workflow = plan(controller, "cpu-source")
    base = finish_cpu(controller, workflow)
    if damage == "dataset":
        (base / "experiment" / "dataset.pt").write_bytes(b"changed")
    elif damage == "config":
        (base / "config.yaml").write_text("changed\n")
    elif damage == "source":
        (controller.ROOT / "tdn" / "source.txt").write_text("changed\n")
    elif damage == "marker":
        (base / "experiment" / "COMPLETED").write_text("invalid\n")
    elif damage == "software":
        controller.cw.atomic_json(controller.software_path(workflow), {"changed": True})
    else:
        record = controller.cw.read_json(controller.state_path(workflow))
        record["status"] = "FAILED"
        controller.cw.atomic_json(controller.state_path(workflow), record)
    assert controller.main(["benchmark", "cpu-source"]) == 2


def test_manifest_paths_cannot_escape_experiment(controller):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    path = base / "experiment" / "manifest.json"
    manifest = controller.cw.read_json(path)
    manifest["files"]["../config.yaml"] = workflow["config_sha256"]
    controller.cw.atomic_json(path, manifest)
    (base / "experiment" / "COMPLETED").write_text(controller.cw.digest(path))
    with pytest.raises(ValueError, match="escaping path"):
        controller.verify_experiment(workflow)


def test_run_ids_existing_runs_and_resource_increases_rejected(controller):
    with pytest.raises(ValueError, match="simple research"):
        plan(controller, "../escape")
    workflow = plan(controller)
    persist(controller, workflow)
    with pytest.raises(ValueError, match="exists"):
        plan(controller)
    workflow["resources"]["cpu"]["walltime"] = "24:00:00"
    with pytest.raises(ValueError, match="capped"):
        controller.validate(workflow)


def test_worker_cpu_runs_focused_tests_before_engine_and_no_gpu(controller):
    workflow = plan(controller, "cpu", "--smoke")
    commands = controller.worker_commands(workflow)
    assert [name for name, args in commands] == ["tests", "experiment"]
    tests = commands[0][1]
    assert tests[tests.index("-m", 3) + 1] == "not gpu"
    assert "--basetemp" in tests and str(Path(workflow["run_dir"]) / "pytest-work") in tests
    assert commands[1][1][-3:] == ["--device", "cpu", "--smoke"]
    assert "--config" in commands[1][1]


def test_worker_gpu_preflight_and_benchmark_are_same_coordinator_children(controller):
    finish_cpu(controller, plan(controller, "cpu-source"))
    workflow = controller.prepare(controller.parser().parse_args(["benchmark", "cpu-source"]))
    commands = controller.worker_commands(workflow)
    assert [name for name, args in commands] == ["preflight", "gpu-tests", "benchmark"]
    assert commands[0][1][1].endswith("gpu_preflight.py")
    assert "--junitxml" in commands[1][1]
    assert "--source-run" in commands[2][1]
    assert "cuda" in commands[2][1]
    assert all("srun" not in args for _, args in commands)


def test_worker_rejects_mismatched_scheduler_ownership_before_software(controller, monkeypatch):
    workflow = plan(controller)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: SimpleNamespace(stdout="JobId=123 Account=wrong JobState=RUNNING"))
    monkeypatch.setattr(controller, "software_report", lambda *a: pytest.fail("Unowned worker inspected venv"))
    with pytest.raises(ValueError, match="not this research"):
        controller.verify_worker(workflow, "cpu")


def test_worker_failure_preserves_stage_and_never_starts_experiment(controller, monkeypatch):
    workflow = plan(controller)
    base = persist(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(controller, "verify_worker", lambda *a: {"verified": True})
    monkeypatch.setattr(controller, "software_report", lambda *a: {"verified": True})
    launches = []
    class Child:
        def __init__(self, args, **kwargs):
            launches.append(args)
        def wait(self, timeout=None):
            return 7
        def poll(self):
            return 7
    monkeypatch.setattr(controller.subprocess, "Popen", Child)
    with pytest.raises(ValueError, match="tests failed"):
        controller.worker(workflow, "cpu")
    record = controller.cw.read_json(controller.state_path(workflow))
    assert record["status"] == "FAILED" and record["exit_code"] == 7 and record["stage"] == "tests"
    assert len(launches) == 1 and not (base / "experiment").exists()


def test_collection_preserves_failures_and_includes_reproducibility_artifacts(controller):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    (base / "logs").mkdir()
    (base / "logs" / "cpu-1.err").write_text("retained failure\n")
    (base / "pytest-work").mkdir()
    (base / "pytest-work" / "large.bin").write_bytes(b"ignored")
    archive = controller.collect(workflow)
    with tarfile.open(archive) as stream:
        names = stream.getnames()
        assert f"{workflow['run_id']}/experiment/dataset.pt" in names
        assert f"{workflow['run_id']}/logs/cpu-1.err" in names
        assert not any("pytest-work" in name for name in names)
    assert (base / "logs" / "cpu-1.err").read_text() == "retained failure\n"


def test_collection_refuses_symlinks(controller):
    workflow = plan(controller)
    base = persist(controller, workflow)
    (base / "escape").symlink_to(controller.ROOT / "requirements.txt")
    with pytest.raises(ValueError, match="symlink"):
        controller.collect(workflow)


def test_new_bash_scripts_parse_without_executing():
    for name in ("carc_research.sh", "research_worker.sh", "research_local.sh", "research_cpu.sbatch", "research_gpu.sbatch"):
        result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
