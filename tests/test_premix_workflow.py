"""Orchestration contracts with real Tower I/O, without live CARC execution."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("premix_workflow_test", ROOT / "scripts" / "premix_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(name + " source\n")
    for name in ("test_premix_gpu.py", "test_premix_numerics.py", "test_premix_workflow.py"):
        (root / "tests" / name).write_text("def test_ok(): pass\n")
    (root / "requirements.txt").write_text("numpy==2.2.6\n")
    (root / "pyproject.toml").write_text("[project]\nname='tdn'\n")
    for owner in (module, module.rw, module.cw):
        monkeypatch.setattr(owner, "ROOT", root)
    monkeypatch.setattr(module.cw, "CARC_ROOT", root)
    monkeypatch.setattr(module.cw.getpass, "getuser", lambda: "aadaniel")
    for key in ("TDN_LOCAL_TEST_ROOT", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT", "SLURM_JOB_ID", "SLURM_STEP_ID",
                "SLURM_JOB_ACCOUNT", "CONDA_PREFIX", "CONDA_SHLVL", "CARC_ACCOUNT", "TORCH_VERSION", "TDN_CPU_PARTITION"):
        monkeypatch.delenv(key, raising=False)
    return module


def plan(module, name="test", *flags):
    return module.prepare(module.parser().parse_args(["plan", "--run-id", name, *flags]))


def freeze(module, workflow):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "logs").mkdir()
    module.cw.atomic_json(base / "premix-workflow.json", workflow)
    module.cw.atomic_json(base / "protocol.json", module.declaration(workflow["profile"]))
    module.cw.atomic_json(base / "jobs.json", [])
    return base


def seal(module, workflow, stage, *, device=None):
    base = Path(workflow["run_dir"]) / stage
    base.mkdir(parents=True)
    files = {"summary.json": {"status": "COMPLETED"}, "protocol.json": {"panel": stage, "profile": workflow["profile"]},
             "execution.json": {"stage": stage, "profile": workflow["profile"], "execution_mode": "carc",
                                "device": device or ("cuda" if stage == "neural" else "cpu")}}
    if stage == "prepare":
        files.update({"dataset_manifest.json": {"parents": "fixture"}})
        (base / "dataset.pt").write_bytes(b"mocked dataset; not a scientific result")
    for name, contents in files.items():
        module.cw.atomic_json(base / name, contents)
    manifest = {"version": 1, "benchmark_suite": "premix", "stage": stage, "profile": workflow["profile"],
                "source_tree_sha256": workflow["source_tree_sha256"],
                "protocol_sha256": module.canonical_hash(files["protocol.json"]),
                "files": {path.name: module.cw.digest(path) for path in base.iterdir()}}
    module.cw.atomic_json(base / "manifest.json", manifest)
    (base / "COMPLETED").write_text(module.cw.digest(base / "manifest.json") + "\n")
    return base


def completed_cpu(module, workflow, software):
    for stage in module.CPU_STAGES:
        seal(module, workflow, stage)
        software_path = Path(workflow["run_dir"]) / "state" / f"{stage}-software.json"
        module.cw.atomic_json(software_path, software)
        module.cw.atomic_json(module.state_path(workflow, stage), {"status": "COMPLETED", "exit_code": 0,
            "source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
            "software_sha256": module.cw.digest(software_path)})


def reseal(module, stage_root, name, update):
    value = module.cw.read_json(stage_root / name)
    value.update(update)
    module.cw.atomic_json(stage_root / name, value)
    manifest = module.cw.read_json(stage_root / "manifest.json")
    manifest["files"][name] = module.cw.digest(stage_root / name)
    if name == "protocol.json":
        manifest["protocol_sha256"] = module.canonical_hash(value)
    module.cw.atomic_json(stage_root / "manifest.json", manifest)
    (stage_root / "COMPLETED").write_text(module.cw.digest(stage_root / "manifest.json") + "\n")


@pytest.mark.parametrize("flags", [[], ["--smoke"]])
def test_plan_has_no_side_effects_and_fixed_resources(controller, monkeypatch, capsys, flags):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("plan called scheduler"))
    monkeypatch.setattr(controller.cw, "actual_policy", lambda *a, **kw: pytest.fail("plan checked live policy"))
    assert controller.main(["plan", *flags]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 4
    assert output.count("--gpus-per-task=a100:1") == 1
    assert output.count("--time=00:30:00") == 4
    assert output.count("--mem=16G") == 4
    assert output.count("--account=anakano_81") == 4
    assert "afterok:<accuracy job>:<scaling job>:<prepare job>" in output
    assert "PLAN ONLY" in output and "--max-pending" not in output
    assert not (controller.ROOT / "runs").exists()
    assert not (controller.ROOT / ".cache").exists()


def test_login_plan_imports_no_numerical_packages(controller):
    program = """
import importlib.abc, importlib.util, pathlib, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy','yaml','tdn'}:
            raise RuntimeError('Numerical package imported on login: '+fullname)
sys.meta_path.insert(0, Reject())
spec=importlib.util.spec_from_file_location('premix_preview',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=module.rw.ROOT=module.cw.ROOT=pathlib.Path(sys.argv[2])
raise SystemExit(module.main(['plan']))
"""
    result = subprocess.run([sys.executable, "-B", "-c", program, str(ROOT / "scripts" / "premix_workflow.py"),
                             str(controller.ROOT)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not (controller.ROOT / "runs").exists()


def mock_submit(controller, monkeypatch, *, fail_at=None, ambiguous=False):
    calls = []
    monkeypatch.setattr(controller.cw, "live_check", lambda *a: {"policy": "mock verified"})
    def command(args, *, env=None, **kwargs):
        assert args[0] == "sbatch"
        assert env["CARC_ACCOUNT"] == "anakano_81"
        assert Path(env["TDN_PREMIX_WORKFLOW"]).is_file()
        calls.append((args, env))
        if len(calls) == fail_at:
            if ambiguous:
                return SimpleNamespace(stdout="unexpected output\n")
            raise ValueError("mock submission rejected")
        return SimpleNamespace(stdout=f"{9000 + len(calls)};mock-cluster\n")
    monkeypatch.setattr(controller.cw, "command", command)
    return calls


def test_run_submits_exact_fixed_dag_and_freezes_protocol(controller, monkeypatch):
    calls = mock_submit(controller, monkeypatch)
    assert controller.main(["run", "--run-id", "submitted", "--smoke"]) == 0
    assert [env["TDN_PREMIX_STAGE"] for _, env in calls] == list(controller.STAGES)
    assert all(not any(arg.startswith("--dependency=") for arg in args) for args, _ in calls[:3])
    assert "--dependency=afterok:9001:9002:9003" in calls[3][0]
    assert "--kill-on-invalid-dep=yes" in calls[3][0]
    base = controller.ROOT / "runs" / "submitted"
    workflow = controller.load(base / "premix-workflow.json")
    assert workflow["profile"] == "smoke"
    assert not (base / "premix-workflow.json").stat().st_mode & 0o222
    assert not (base / "protocol.json").stat().st_mode & 0o222
    assert len(controller.cw.read_json(base / "jobs.json")) == 4
    assert controller.workflow_path("latest") == base / "premix-workflow.json"
    assert controller.main(["run", "--run-id", "submitted"]) == 2
    assert len(calls) == 4


@pytest.mark.parametrize("ambiguous", [False, True])
def test_partial_submission_preserves_evidence_and_never_submits_gpu(controller, monkeypatch, ambiguous):
    calls = mock_submit(controller, monkeypatch, fail_at=2, ambiguous=ambiguous)
    assert controller.main(["run", "--run-id", "partial"]) == 2
    assert len(calls) == 2
    base = controller.ROOT / "runs" / "partial"
    assert len(controller.cw.read_json(base / "jobs.json")) == 1
    failure = controller.cw.read_json(base / "state" / "submission.json")
    assert failure["status"] == "FAILED" and failure["stage"] == "scaling"
    assert failure["submitted_jobs"][0]["job_id"] == "9001"


def test_all_live_resource_checks_precede_any_submission(controller, monkeypatch):
    def live(workflow, phase):
        if phase == "gpu":
            raise ValueError("no authorized GPU partition")
        return {}
    monkeypatch.setattr(controller.cw, "live_check", live)
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("resource check failed before sbatch"))
    assert controller.main(["run"]) == 2
    assert not (controller.ROOT / "runs").exists()


@pytest.mark.parametrize("dependency", ["afterany:1:2:3", "afterok:1:2", "afterok:1:2:2", "afterok:1:2:3:4", "afterok:1:2:0"])
def test_scheduler_rejects_invalid_dependencies(controller, dependency):
    with pytest.raises(ValueError, match="dependency|distinct"):
        controller.scheduler_args(plan(controller), "neural", dependency)


@pytest.mark.parametrize("field,value", [("account", "other"), ("user", "other"), ("profile", "pilot"),
                                         ("protocol_sha256", "0" * 64), ("torch_version", "2.10.0+cpu")])
def test_manifest_rejects_policy_mutations(controller, field, value):
    workflow = plan(controller)
    workflow[field] = value
    with pytest.raises(ValueError):
        controller.validate(workflow)


def test_source_and_protocol_changes_block_worker_loading(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    controller.load(base / "premix-workflow.json")
    (controller.ROOT / "scripts" / "source.txt").write_text("changed source")
    with pytest.raises(ValueError, match="Execution source changed"):
        controller.load(base / "premix-workflow.json")
    controller.load(base / "premix-workflow.json", verify=False)


def test_frozen_protocol_mutation_blocks_worker_loading(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    controller.cw.atomic_json(base / "protocol.json", {"replaced": True})
    with pytest.raises(ValueError, match="protocol changed"):
        controller.load(base / "premix-workflow.json")


@pytest.mark.parametrize("stage", ["accuracy", "scaling", "prepare", "neural"])
def test_stage_seal_accepts_only_matching_artifacts(controller, stage):
    workflow = plan(controller)
    freeze(controller, workflow)
    base = seal(controller, workflow, stage)
    controller.verify_stage(workflow, stage)
    (base / "summary.json").write_text('{"status":"FAILED"}\n')
    with pytest.raises(ValueError, match="artifact changed"):
        controller.verify_stage(workflow, stage)


@pytest.mark.parametrize("update", [{"device": "cpu"}, {"execution_mode": "desktop"}, {"stage": "prepare"}, {"profile": "smoke"}])
def test_neural_execution_cannot_masquerade_as_allocated_gpu(controller, update):
    workflow = plan(controller)
    freeze(controller, workflow)
    base = seal(controller, workflow, "neural")
    reseal(controller, base, "execution.json", update)
    with pytest.raises(ValueError, match="execution"):
        controller.verify_stage(workflow, "neural")


def test_completed_scheduler_does_not_replace_cpu_seals(controller):
    workflow = plan(controller)
    freeze(controller, workflow)
    software = {"packages": {"torch": "mock"}}
    completed_cpu(controller, workflow, software)
    hashes = controller.verify_prerequisites(workflow, software)
    assert set(hashes) == set(controller.CPU_STAGES)
    (Path(workflow["run_dir"]) / "prepare" / "dataset.pt").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="artifact changed"):
        controller.verify_prerequisites(workflow, software)


def test_prerequisite_requires_matching_software_and_worker_success(controller):
    workflow = plan(controller)
    freeze(controller, workflow)
    software = {"packages": {"torch": "mock"}}
    completed_cpu(controller, workflow, software)
    with pytest.raises(ValueError, match="software differs"):
        controller.verify_prerequisites(workflow, {"packages": {"torch": "changed"}})
    path = controller.state_path(workflow, "scaling")
    record = controller.cw.read_json(path)
    record.update(status="INTERRUPTED", exit_code=75)
    controller.cw.atomic_json(path, record)
    with pytest.raises(ValueError, match="did not complete"):
        controller.verify_prerequisites(workflow, software)


def test_worker_commands_place_tests_and_preflight_before_training(controller):
    workflow = plan(controller)
    assert [label for label, _ in controller.worker_commands(workflow, "prepare")] == ["tests", "experiment"]
    commands = controller.worker_commands(workflow, "neural")
    assert [label for label, _ in commands] == ["preflight", "gpu-tests", "experiment"]
    engine = commands[-1][1]
    assert engine[engine.index("--device") + 1] == "cuda"
    assert engine[engine.index("--dataset-dir") + 1] == str(Path(workflow["run_dir"]) / "prepare")
    assert "test_premix_gpu.py" in " ".join(commands[1][1])
    assert all(args[0] == str(controller.ROOT / ".venv" / "bin" / "python") for _, args in commands)


@pytest.mark.parametrize("xml", ["<testsuites/>", '<testsuites><testsuite tests="3" skipped="1"/></testsuites>',
                                  '<testsuite tests="1" failures="1"/>', '<testsuite tests="2" errors="1"/>'])
def test_gpu_junit_rejects_skips_missing_tests_and_failures(controller, tmp_path, xml):
    path = tmp_path / "gpu.xml"
    path.write_text(xml)
    with pytest.raises(ValueError, match="must execute"):
        controller.validate_gpu_junit(path)


def test_gpu_junit_accepts_actual_pass_record(controller, tmp_path):
    path = tmp_path / "gpu.xml"
    path.write_text('<testsuites><testsuite tests="3" skipped="0" failures="0" errors="0"/></testsuites>')
    controller.validate_gpu_junit(path)


@pytest.fixture
def real_reports(controller, monkeypatch):
    """Use the reporting boundary that mocked worker tests cannot exercise."""
    from tdn import reporting

    monkeypatch.setattr(reporting, "ROOT", controller.ROOT)
    for key in ("TDN_TOWER_DIR", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(key, raising=False)
    # The reporter hashes this actual script. Create it before freezing source.
    (controller.ROOT / "scripts" / "premix_worker.sh").write_bytes(
        (ROOT / "scripts" / "premix_worker.sh").read_bytes())
    return reporting


@pytest.mark.parametrize("stage", ["accuracy", "scaling", "prepare", "neural"])
def test_real_report_stage_sidecar_inheritance_and_finalization(
        controller, real_reports, monkeypatch, capsys, stage):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    protocol_before = (base / "protocol.json").read_bytes()
    job = str(9001 + controller.STAGES.index(stage))
    monkeypatch.setenv("SLURM_JOB_ID", job)
    controller.cw.atomic_json(base / "jobs.json", [{"stage": stage, "job_id": job}])
    for suffix in ("out", "err"):
        (base / "logs" / f"{stage}-{job}.{suffix}").write_text("scheduler fixture\n")

    # Match allocation startup exactly: the coordinator protocol already exists,
    # but the child must still be able to create its own science directory.
    source = base / stage
    report = controller.begin_report(workflow, stage)
    assert not source.exists()
    assert report.parent == base / "tower"
    assert not report.is_relative_to(source)
    inventory = real_reports.read_json(report / "run.json")
    assert inventory["metadata"]["source_directory"] == source.relative_to(controller.ROOT).as_posix()
    assert inventory["job_id"] == job
    assert inventory["state"] == "RUNNING"
    assert inventory["resources"]["gpus"] == int(stage == "neural")

    # The child inherits this job's stream without making or owning another
    # report. Fixture artifacts are not evidence of a numerical experiment.
    seal(controller, workflow, stage)
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setenv("TDN_TOWER_DIR", str(report))
    assert real_reports.attach_report(source) == (report, False)
    monkeypatch.setenv("SLURM_JOB_ID", "9999")
    with pytest.raises(ValueError, match="different job"):
        real_reports.attach_report(source)
    monkeypatch.setenv("SLURM_JOB_ID", job)
    real_reports.emit({"stage_completed": 1}, phase=f"premix/{stage}")

    # Use actual analytics publication and terminal sealing, rather than a
    # mock that would hide another layout or ownership failure.
    controller.finish_report(workflow, stage, report, state="COMPLETED",
                             runtime_seconds=0.1, exit_code=0)
    summary = real_reports.read_json(report / "summary.json")
    assert summary["state"] == "COMPLETED" and summary["job_id"] == job
    assert summary["metadata"]["source_directory"] == inventory["metadata"]["source_directory"]
    assert summary["results"]["science_sources_mutated"] is False
    assert summary["results"]["reporting_omission_count"] == 0
    assert summary["results"]["artifact_count"] == len(source_before)
    assert real_reports.read_json(report / "run.json")["state"] == "COMPLETED"
    artifacts = real_reports.read_json(report / "outputs" / "artifacts.json")
    assert artifacts["source_dirs"] == [os.path.relpath(source, report)]
    assert source_before == {path.name: path.read_bytes() for path in source.iterdir()}
    assert (base / "protocol.json").read_bytes() == protocol_before
    assert controller.cw.tower_reports_for_job(workflow, job) == [report]
    controller.paths(workflow)
    assert f"TDN_TOWER_METRICS={report / 'metrics.jsonl'}" in capsys.readouterr().out
    with pytest.raises(ValueError, match="terminal"):
        real_reports.attach_report(source)


def test_real_report_worker_startup_failure_is_terminal(controller, real_reports, monkeypatch):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    protocol_before = (base / "protocol.json").read_bytes()
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    monkeypatch.setattr(controller, "verify_allocation", lambda *args: None)

    def mismatched_venv(*args):
        raise ValueError("Existing venv differs from pinned dependencies")

    monkeypatch.setattr(controller.rw, "software_report", mismatched_venv)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: pytest.fail("venv mismatch"))
    with pytest.raises(controller.rw.WorkerFailure, match="venv differs"):
        controller.worker(workflow, "accuracy")
    state = controller.cw.read_json(controller.state_path(workflow, "accuracy"))
    assert state["status"] == "FAILED" and state["stage"] == "startup"
    assert "venv differs" in state["error"]
    reports = controller.cw.tower_reports_for_job(workflow, "9001")
    assert len(reports) == 1
    summary = real_reports.read_json(reports[0] / "summary.json")
    assert summary["state"] == "FAILED" and summary["exit_code"] == 2
    assert "venv differs" in summary["metadata"]["error"]
    assert summary["results"]["artifact_count"] == 0
    assert real_reports.read_json(reports[0] / "run.json")["state"] == "FAILED"
    assert not (base / "accuracy").exists()
    assert (base / "protocol.json").read_bytes() == protocol_before
    assert "TDN_TOWER_DIR" not in os.environ


def worker_fixture(controller, monkeypatch, *, stage="accuracy"):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    monkeypatch.setattr(controller, "verify_allocation", lambda *args: None)
    monkeypatch.setattr(controller, "begin_report", lambda *args: base / "tower" / "mock-report")
    finished = []
    monkeypatch.setattr(controller, "finish_report", lambda *a, **kw: finished.append(kw))
    software = {"packages": {"torch": "mock"}}
    monkeypatch.setattr(controller.rw, "software_report", lambda *a: software)
    monkeypatch.setattr(controller, "worker_commands", lambda *a: [("experiment", ["mock-program"])])
    if stage == "neural":
        completed_cpu(controller, workflow, software)
    return workflow, base, finished


def test_worker_completes_only_after_seal_verification(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    class Child:
        pid = 123456789
        def poll(self): return 0
        def wait(self, timeout=None):
            seal(controller, workflow, "accuracy")
            return 0
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: Child())
    assert controller.worker(workflow, "accuracy") == 0
    assert controller.cw.read_json(controller.state_path(workflow, "accuracy"))["status"] == "COMPLETED"
    assert finished[-1]["state"] == "COMPLETED"
    with pytest.raises(ValueError, match="already started"):
        controller.worker(workflow, "accuracy")


def test_worker_zero_exit_without_seal_fails(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123456789,
        poll=lambda: 0, wait=lambda timeout=None: 0))
    with pytest.raises(controller.rw.WorkerFailure):
        controller.worker(workflow, "accuracy")
    assert controller.cw.read_json(controller.state_path(workflow, "accuracy"))["status"] == "FAILED"
    assert finished[-1]["state"] == "FAILED"


def test_worker_preserves_child_exit_code(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123456789,
        poll=lambda: 23, wait=lambda timeout=None: 23))
    with pytest.raises(controller.rw.WorkerFailure) as error:
        controller.worker(workflow, "accuracy")
    assert error.value.exit_code == 23
    assert controller.cw.read_json(controller.state_path(workflow, "accuracy"))["exit_code"] == 23


def test_usr1_forwards_graceful_stop_and_blocks_successor(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller, "worker_commands", lambda *a: [("one", ["mock"]), ("two", ["must-not-run"])])
    started, sent = [], []
    monkeypatch.setattr(controller.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    class Child:
        pid = 123456789
        done = False
        def poll(self): return 75 if self.done else None
        def wait(self, timeout=None):
            signal.raise_signal(signal.SIGUSR1)
            self.done = True
            return 75
    def popen(args, **kwargs):
        started.append(args)
        return Child()
    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    with pytest.raises(controller.rw.WorkerFailure) as error:
        controller.worker(workflow, "accuracy")
    assert error.value.exit_code == 75
    assert len(started) == 1 and sent == [(123456789, signal.SIGUSR1)]
    assert finished[-1]["state"] == "INTERRUPTED"
    assert controller.cw.read_json(controller.state_path(workflow, "accuracy"))["status"] == "INTERRUPTED"


def test_worker_source_change_is_reported_before_numerics(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    (controller.ROOT / "tdn" / "source.txt").write_text("mutated source")
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: pytest.fail("source changed"))
    with pytest.raises(controller.rw.WorkerFailure, match="Execution source changed"):
        controller.worker(workflow, "accuracy")
    assert finished[-1]["state"] == "FAILED"


def test_stop_allows_thirty_second_grace_then_kills_ignoring_child(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    current, sent = [0.0], []
    monkeypatch.setattr(controller.time, "monotonic", lambda: current[0])
    monkeypatch.setattr(controller.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    class Child:
        pid = 123456789
        waits = 0
        def poll(self): return -9 if self.waits > 2 else None
        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                signal.raise_signal(signal.SIGUSR1)
                current[0] = 29.0
                raise subprocess.TimeoutExpired("mock", timeout)
            if self.waits == 2:
                assert sent == [(self.pid, signal.SIGUSR1)]
                current[0] = 30.0
                raise subprocess.TimeoutExpired("mock", timeout)
            return -9
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: Child())
    with pytest.raises(controller.rw.WorkerFailure) as error:
        controller.worker(workflow, "accuracy")
    assert error.value.exit_code == 75
    assert sent == [(123456789, signal.SIGUSR1), (123456789, signal.SIGKILL)]
    assert finished[-1]["state"] == "INTERRUPTED"


def test_gpu_worker_refuses_missing_cpu_state_before_preflight(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: pytest.fail("missing prerequisites"))
    with pytest.raises(controller.rw.WorkerFailure):
        controller.worker(workflow, "neural")
    assert finished[-1]["state"] == "FAILED"


def test_worker_venv_failure_keeps_startup_report(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch)
    def broken(*args):
        raise ValueError("Existing venv differs from pinned dependencies")
    monkeypatch.setattr(controller.rw, "software_report", broken)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: pytest.fail("venv mismatch"))
    with pytest.raises(controller.rw.WorkerFailure, match="venv differs"):
        controller.worker(workflow, "accuracy")
    assert finished[-1]["state"] == "FAILED"
    assert "venv differs" in controller.cw.read_json(controller.state_path(workflow, "accuracy"))["error"]


def test_gpu_junit_failure_prevents_experiment(controller, monkeypatch):
    workflow, base, finished = worker_fixture(controller, monkeypatch, stage="neural")
    monkeypatch.setattr(controller, "worker_commands", lambda *a: [("gpu-tests", ["mock-tests"]),
                                                                   ("experiment", ["must-not-run"])])
    started = []
    def popen(args, **kwargs):
        started.append(args)
        (base / "neural-tests.xml").write_text('<testsuite tests="3" skipped="3"/>')
        return SimpleNamespace(pid=123456789, poll=lambda: 0, wait=lambda timeout=None: 0)
    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    with pytest.raises(controller.rw.WorkerFailure, match="must execute"):
        controller.worker(workflow, "neural")
    assert started == [["mock-tests"]]
    assert finished[-1]["state"] == "FAILED"


def test_allocation_ownership_checks_identity_account_partition(controller, monkeypatch):
    workflow = plan(controller)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    fields = {"JobId": "9001", "JobName": "tdn-premix-accuracy", "Account": "anakano_81",
              "Comment": "tdn-premix:test:accuracy", "JobState": "RUNNING", "Partition": "main", "UserId": "aadaniel(1234)"}
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: SimpleNamespace(stdout=" ".join(f"{k}={v}" for k, v in fields.items())))
    controller.verify_allocation(workflow, "accuracy")
    fields["Account"] = "other"
    with pytest.raises(ValueError, match="Allocation is not"):
        controller.verify_allocation(workflow, "accuracy")


def test_review_archive_retains_failure_but_excludes_test_temp(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (base / "logs" / "failure.err").write_text("preserved failure\n")
    (base / "prepare-pytest-work").mkdir()
    (base / "prepare-pytest-work" / "large.bin").write_bytes(b"excluded")
    archive = controller.collect(workflow)
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert "test/logs/failure.err" in names
    assert not any("pytest-work" in name for name in names)


def test_review_archive_rejects_symlink(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (base / "escape").symlink_to(controller.ROOT / "requirements.txt")
    with pytest.raises(ValueError, match="refuses symlink"):
        controller.collect(workflow)


@pytest.mark.parametrize("script", ["carc_premix.sh", "premix_worker.sh", "premix_local.sh"])
def test_shell_syntax(script):
    result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_local_wrapper_rejects_slurm_and_carc_run_rejects_cloud():
    env = {**os.environ, "SLURM_JOB_ID": "123456"}
    result = subprocess.run(["bash", str(ROOT / "scripts" / "premix_local.sh")], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "without Slurm" in result.stderr or "Use carc_premix.sh run on CARC" in result.stderr
    result = subprocess.run(["bash", str(ROOT / "scripts" / "carc_premix.sh"), "run"],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "CARC execution requires" in result.stderr or "Submit premix from the CARC login shell" in result.stderr


def test_shared_venv_lock_uses_readable_descriptor(controller):
    import fcntl
    with controller.rw.acquire_venv_lock() as handle:
        assert handle.readable() and handle.writable()
        contender = (controller.ROOT / ".cache" / "carc-phase.lock").open("a+b")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            contender.close()
