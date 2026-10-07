"""Bounded scheduler contracts and real reporting I/O; no live GPU claim."""
from __future__ import annotations

import csv
import getpass
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("consistency_workflow_test", ROOT / "scripts" / "consistency_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(name + "\n")
    for name in ("test_consistency_gpu.py", "test_consistency_models.py", "test_consistency_workflow.py",
                 "test_desktop_slurm_runtime.py", "test_fedora_workflow.py"):
        (root / "tests" / name).write_text("def test_ok(): pass\n")
    (root / "scripts" / "consistency_worker.sh").write_bytes((ROOT / "scripts" / "consistency_worker.sh").read_bytes())
    (root / "requirements.txt").write_text("numpy==2.2.6\n")
    (root / "pyproject.toml").write_text("[project]\nname='tdn'\n")
    for owner in (module, module.fw, module.pw, module.rw, module.cw):
        monkeypatch.setattr(owner, "ROOT", root)
    for key in ("TDN_LOCAL_TEST_ROOT", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT", "SLURM_JOB_ID", "SLURM_STEP_ID",
                "SLURM_JOB_ACCOUNT", "CONDA_PREFIX", "CONDA_SHLVL", "TDN_SLURM_CONFIG", "TDN_TOWER_DIR",
                "TDN_CONSISTENCY_WORKFLOW", "TDN_CONSISTENCY_PROTOCOL_SHA256", "TDN_CONSISTENCY_STAGE"):
        monkeypatch.delenv(key, raising=False)
    profile = {"schema_version": 1, "kind": "desktop-slurm", "root": str(root), "user": getpass.getuser(),
               "cpu_partition": "local", "gpu_partition": "local", "account": None,
               "gpu_gres": "gpu:1", "expected_gpu_name": "RTX 4090", "gpu_vram_gib": 24,
               "torch_version": "2.10.0+cu126", "torch_wheel_index": "https://download.pytorch.org/whl/cu126"}
    module.cw.atomic_json(root / ".tdn" / "fedora-slurm.json", profile)
    return module


def plan(module, name="test", *flags):
    return module.prepare(module.parser().parse_args(["plan", "--run-id", name, *flags]))


def freeze(module, workflow, software=None):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "logs").mkdir()
    for path, value in ((base / module.MANIFEST, workflow),
                        (base / "protocol.json", module.declaration(workflow["profile"])),
                        (base / "slurm-profile.json", workflow["slurm_profile"]), (base / "jobs.json", []),
                        (base / "state" / "submission-software.json", software or {"packages": {"torch": "fixture"}})):
        module.cw.atomic_json(path, value)
    return base


def seal_audit(module, workflow, **changes):
    """Synthetic identities test seals; these rows are not numerical evidence."""
    from tdn.analysis.consistency.artifacts import seal_stage
    base = Path(workflow["run_dir"]) / "audit"
    base.mkdir(parents=True)
    protocol = module.scientific_protocol(workflow["profile"])
    required = set(protocol["audit_required_case_ids"])
    rows = [{"check_id": identity, "required": identity in required,
             "status": "PASS" if identity in required else "OBSERVED"} for identity in protocol["audit_case_ids"]]
    execution = {"stage": "audit", "profile": workflow["profile"], "device": "cpu", "execution_mode": "desktop-slurm",
                 "slurm_profile_sha256": workflow["slurm_profile_sha256"],
                 "workflow_protocol_sha256": workflow["protocol_sha256"],
                 "protocol_sha256": module.pw.canonical_hash(protocol),
                 "software": {"source_tree_sha256": workflow["source_tree_sha256"]}, **changes}
    files = {"protocol.json": protocol, "execution.json": execution, "checks.json": {"rows": rows},
             "summary.json": {"status": "COMPLETED", "stage": "audit", "profile": workflow["profile"], "device": "cpu",
                "coverage": {"expected": len(rows), "reported": len(rows)}, "correctness_failures": 0}}
    for name, value in files.items():
        module.cw.atomic_json(base / name, value)
    seal_stage(base, stage="audit", profile=workflow["profile"], source_tree_sha256=workflow["source_tree_sha256"])
    return base


@pytest.mark.parametrize("flags", [[], ["--smoke"]])
def test_plan_read_only_three_bounded_allocations(controller, monkeypatch, capsys, flags):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("plan called scheduler"))
    assert controller.main(["plan", *flags]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 3 and output.count("--time=00:30:00") == 3
    assert output.count("--gres=none") == 2 and output.count("--gres=gpu:1") == 1
    assert "anakano_81" not in output and "a100" not in output and "--account=" not in output
    assert "audit -> prepare -> neural" in output
    assert not (controller.ROOT / "runs").exists() and not (controller.ROOT / ".cache").exists()


def mock_submission(module, monkeypatch, *, fail_at=None, outputs=None):
    calls = []
    monkeypatch.setattr(module.rw, "software_report", lambda *a: {"packages": {"torch": "fixture"}})
    def command(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "scontrol":
            return SimpleNamespace(stdout="PartitionName=local State=UP MaxTime=7-00:00:00")
        count = len([row for row in calls if row[0][0] == "sbatch"])
        if count == fail_at:
            raise ValueError("simulated submission failure")
        return SimpleNamespace(stdout=outputs[count - 1] if outputs else str(9000 + count))
    monkeypatch.setattr(module.cw, "command", command)
    return calls


def test_fixed_serial_dag_and_independent_pointer(controller, monkeypatch):
    calls = mock_submission(controller, monkeypatch)
    assert controller.main(["run", "--run-id", "serial"]) == 0
    workflow = controller.load(controller.workflow_path("latest"))
    batches = [(args, kwargs) for args, kwargs in calls if args[0] == "sbatch"]
    assert len(batches) == 3
    for (args, kwargs), dependency in zip(batches, [None, "afterok:9001", "afterok:9001:9002"]):
        assert [item for item in args if item.startswith("--dependency=")] == (["--dependency=" + dependency] if dependency else [])
        assert kwargs["env"]["TDN_EXECUTION_MODE"] == "desktop-slurm"
        assert kwargs["env"]["TDN_SLURM_CONFIG"] == workflow["slurm_profile_path"]
        assert kwargs["env"]["TDN_CONSISTENCY_WORKFLOW"].endswith("consistency-workflow.json")
    assert Path(workflow["slurm_profile_path"]).stat().st_mode & 0o222 == 0
    assert (controller.ROOT / "runs" / controller.POINTER).is_file()
    assert not (controller.ROOT / "runs" / ".fedora-premix-latest.json").exists()
    assert not (controller.ROOT / "runs" / "serial" / "premix-workflow.json").exists()


def test_account_requires_explicit_local_profile(controller):
    workflow = plan(controller)
    assert not any(value.startswith("--account=") for value in controller.scheduler_args(workflow, "neural"))
    workflow["slurm_profile"]["account"] = "local-research"
    assert "--account=local-research" in controller.scheduler_args(workflow, "neural")


@pytest.mark.parametrize("stage,dependency", [("audit", "afterok:1"), ("prepare", "afterok:1:2"),
    ("prepare", "afterany:1"), ("neural", "afterok:1"), ("neural", "afterok:1:1"), ("neural", "afterok:1:x")])
def test_dependency_contract(controller, stage, dependency):
    with pytest.raises(ValueError, match="dependency"):
        controller.scheduler_args(plan(controller), stage, dependency)


def test_submission_failure_preserves_jobs_and_never_submits_gpu(controller, monkeypatch):
    calls = mock_submission(controller, monkeypatch, fail_at=2)
    assert controller.main(["run", "--run-id", "partial"]) == 2
    workflow = controller.load(controller.workflow_path("latest"))
    state = controller.cw.read_json(Path(workflow["run_dir"]) / "state" / "submission.json")
    assert state["status"] == "FAILED" and len(state["submitted_jobs"]) == 1
    assert len([row for row in calls if row[0][0] == "sbatch"]) == 2


@pytest.mark.parametrize("outputs", [["9001", "9001"], ["9001", "ambiguous output"]])
def test_invalid_scheduler_id_cannot_submit_successor(controller, monkeypatch, outputs):
    calls = mock_submission(controller, monkeypatch, outputs=outputs)
    assert controller.main(["run", "--run-id", "ambiguous"]) == 2
    assert len([row for row in calls if row[0][0] == "sbatch"]) == 2


def test_submission_clears_other_project_defaults(controller, monkeypatch):
    stale = {"SBATCH_ACCOUNT": "anakano_81", "SBATCH_CONSTRAINT": "a100-40gb", "SBATCH_GRES": "gpu:a100:4",
             "SRUN_NTASKS": "30", "CARC_ACCOUNT": "anakano_81", "TORCH_CUDA_ARCH_LIST": "8.0",
             "TDN_PREMIX_WORKFLOW": "old", "TDN_PREMIX_STAGE": "neural", "TDN_PREMIX_PROTOCOL_SHA256": "old"}
    for key, value in stale.items():
        monkeypatch.setenv(key, value)
    calls = mock_submission(controller, monkeypatch)
    assert controller.main(["run", "--run-id", "clean-env"]) == 0
    for args, kwargs in calls:
        if args[0] == "sbatch":
            assert stale.keys().isdisjoint(kwargs["env"])


@pytest.mark.parametrize("env", [{"SLURM_JOB_ID": "123"}, {"TDN_EXECUTION_MODE": "desktop"},
                                  {"TDN_LOCAL_TEST_ROOT": "spoof"}, {"CONDA_PREFIX": "/conda"}])
def test_submission_rejects_runtime_bypass(controller, monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("unsafe scheduler request"))
    assert controller.main(["run"]) == 2


@pytest.mark.parametrize("target", ["source", "profile", "protocol"])
def test_frozen_mutation_blocks_work(controller, target):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    if target == "source":
        (controller.ROOT / "tdn" / "source.txt").write_text("changed\n")
    elif target == "profile":
        controller.cw.atomic_json(base / "slurm-profile.json", {**workflow["slurm_profile"], "account": "changed"})
    else:
        controller.cw.atomic_json(base / "protocol.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        controller.load(base / controller.MANIFEST)
    assert controller.load(base / controller.MANIFEST, verify=False)["run_id"] == "test"


def test_sealed_audit_has_exact_execution_binding(controller):
    workflow = plan(controller, "audit", "--smoke")
    freeze(controller, workflow)
    base = seal_audit(controller, workflow)
    assert controller.verify_stage(workflow, "audit")["benchmark_suite"] == "consistency"
    execution = controller.cw.read_json(base / "execution.json")
    execution["workflow_protocol_sha256"] = "0" * 64
    controller.cw.atomic_json(base / "execution.json", execution)
    with pytest.raises(ValueError, match="artifact changed"):
        controller.verify_stage(workflow, "audit")


def test_resealed_wrong_profile_never_becomes_a_prerequisite(controller):
    workflow = plan(controller, "wrong-profile", "--smoke")
    freeze(controller, workflow)
    seal_audit(controller, workflow, slurm_profile_sha256="0" * 64)
    with pytest.raises(ValueError, match="provenance differs"):
        controller.verify_stage(workflow, "audit")


def complete_audit(module, workflow, software):
    base = seal_audit(module, workflow)
    software_path = Path(workflow["run_dir"]) / "state" / "audit-software.json"
    module.cw.atomic_json(software_path, software)
    module.cw.atomic_json(module.pw.state_path(workflow, "audit"), {"status": "COMPLETED", "exit_code": 0,
        "source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
        "software_sha256": module.cw.digest(software_path)})
    return base


def test_prepare_requires_audit_state_seal_and_same_software(controller):
    workflow = plan(controller, "prior", "--smoke")
    freeze(controller, workflow)
    software = {"packages": {"torch": "fixture"}}
    with pytest.raises(FileNotFoundError):
        controller.verify_prerequisites(workflow, software, "prepare")
    base = complete_audit(controller, workflow, software)
    assert controller.verify_prerequisites(workflow, software, "prepare") == {"audit": controller.cw.digest(base / "manifest.json")}
    with pytest.raises(ValueError, match="software differs"):
        controller.verify_prerequisites(workflow, {"packages": {"torch": "changed"}}, "prepare")
    (base / "checks.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        controller.verify_prerequisites(workflow, software, "prepare")


def test_worker_commands_run_bounded_new_program_only(controller):
    workflow = plan(controller)
    audit = controller.worker_commands(workflow, "audit")
    assert [name for name, _ in audit] == ["tests", "experiment"]
    assert str(controller.ROOT / "tests" / "test_consistency_workflow.py") in audit[0][1]
    assert str(controller.ROOT / "tests" / "test_desktop_slurm_runtime.py") in audit[0][1]
    prepare = controller.worker_commands(workflow, "prepare")
    assert [name for name, _ in prepare] == ["experiment"]
    neural = controller.worker_commands(workflow, "neural")
    assert [name for name, _ in neural] == ["preflight", "gpu-tests", "experiment"]
    assert neural[0][1][1].endswith("fedora_gpu_preflight.py")
    assert str(controller.ROOT / "tests" / "test_consistency_gpu.py") in neural[1][1]
    assert str(controller.junit_path(workflow, "neural")) in neural[1][1]
    for name, command in prepare + neural[-1:]:
        assert str(controller.ROOT / "scripts" / "consistency.py") in command
        assert "--audit-dir" in command
        assert "premix.py" not in " ".join(command)
    assert "--dataset-dir" in neural[-1][1] and "cuda" in neural[-1][1]


def test_test_environment_clears_science_binding_only(controller):
    env = {"TDN_EXECUTION_MODE": "desktop-slurm", "TDN_SLURM_CONFIG": "frozen", "SLURM_JOB_ID": "9001",
           **controller.workflow_environment(plan(controller), "neural")}
    gpu = dict(env)
    controller.prepare_test_environment(env, "tests")
    controller.prepare_test_environment(gpu, "gpu-tests")
    assert env == {"SLURM_JOB_ID": "9001"}
    assert gpu == {"TDN_EXECUTION_MODE": "desktop-slurm", "TDN_SLURM_CONFIG": "frozen", "SLURM_JOB_ID": "9001"}


@pytest.mark.parametrize("stage", ["audit", "neural"])
def test_allocation_accepts_actual_slurm_fields_and_frozen_identity(controller, monkeypatch, stage):
    from tdn.runtime import desktop_slurm
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("TDN_SLURM_CONFIG", str(base / "slurm-profile.json"))
    job = (f"JobId=9001 JobName=tdn-consistency-{stage} UserId={getpass.getuser()}({os.getuid()}) "
           f"JobState=RUNNING Partition=local AllocNode:Sid=system:8749 "
           f"NumNodes=1 AllocTRES=cpu=4,mem=16G,node=1,gres/gpu=1 "
           f"WorkDir={controller.ROOT} Comment={controller.comment(workflow, stage)}")
    step = f"StepId=9001.0 UserId={os.getuid()} State=RUNNING Partition=local Nodes=1 Tasks=1 TRES=cpu=4,node=1,gres/gpu=1"
    queries = []
    def scheduler(command, **kwargs):
        queries.append(command)
        return SimpleNamespace(returncode=0, stdout=job if command[2] == "job" else step, stderr="")
    monkeypatch.setattr(desktop_slurm.subprocess, "run", scheduler)
    controller.verify_allocation(workflow, stage)
    assert queries == [["scontrol", "show", "job", "9001", "-o"], ["scontrol", "show", "step", "9001.0", "-o"]]
    controller.cw.atomic_json(base / "jobs.json", [{"stage": stage, "job_id": "9002"}])
    with pytest.raises(ValueError, match="submission record"):
        controller.verify_allocation(workflow, stage)


@pytest.fixture
def reporting(controller, monkeypatch):
    from tdn import reporting
    monkeypatch.setattr(reporting, "ROOT", controller.ROOT)
    for key in ("TDN_TOWER_DIR", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(key, raising=False)
    return reporting


@pytest.mark.parametrize("stage", ["audit", "prepare", "neural"])
def test_tower_sidecar_starts_outside_science_and_carries_desktop_profile(controller, reporting, monkeypatch, stage):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    report = controller.begin_report(workflow, stage)
    assert report.parent == base / "tower" and not (base / stage).exists()
    run = reporting.read_json(report / "run.json")
    assert run["parameters"]["execution_mode"] == "desktop-slurm"
    assert run["parameters"]["slurm_profile_sha256"] == workflow["slurm_profile_sha256"]
    assert run["metadata"]["source_directory"] == str((base / stage).relative_to(controller.ROOT))


def test_tower_failure_ingests_just_its_stage_junit(controller, reporting, monkeypatch):
    workflow = plan(controller)
    freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    report = controller.begin_report(workflow, "audit")
    path = controller.junit_path(workflow, "audit")
    path.parent.mkdir(parents=True)
    path.write_text('<testsuite tests="1" failures="1"><testcase name="structural"><failure message="bad"/></testcase></testsuite>')
    other = controller.junit_path(workflow, "neural")
    other.parent.mkdir(parents=True)
    other.write_text('<testsuite tests="99"/>')
    controller.finish_report(workflow, "audit", report, state="FAILED", runtime_seconds=.1, exit_code=2)
    summary = reporting.read_json(report / "summary.json")
    assert summary["state"] == "FAILED"
    assert summary["results"]["test_outcomes"] == {"FAILED": 1, "ERROR": 0, "PASSED": 0, "SKIPPED": 0}
    with (report / "outputs" / "tests.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["status"] == "FAILED" and rows[0]["test"] == "structural"
    assert rows[0]["source_path"].endswith("reporter-tests/audit/tests.xml")
    inventory = reporting.read_json(report / "outputs" / "artifacts.json")
    assert inventory["source_dirs"] == ["../../reporter-tests/audit"]


def worker_fixture(module, monkeypatch):
    software = {"packages": {"torch": "fixture"}}
    workflow = plan(module, "worker", "--smoke")
    base = freeze(module, workflow, software)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(module, "verify_allocation", lambda *a: None)
    monkeypatch.setattr(module, "begin_report", lambda *a: base / "tower" / "fixture-report")
    monkeypatch.setattr(module.rw, "software_report", lambda *a: software)
    finished = []
    monkeypatch.setattr(module, "finish_report", lambda *a, **kw: finished.append(kw))
    monkeypatch.setattr(module, "worker_commands", lambda *a: [("experiment", ["fixture-program"])])
    return workflow, base, software, finished


def test_shared_worker_uses_new_manifest_environment_and_requires_seal(controller, monkeypatch):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    environments = []
    def popen(args, **kwargs):
        environments.append(kwargs["env"])
        def wait(timeout=None):
            seal_audit(controller, workflow)
            return 0
        return SimpleNamespace(pid=123456789, poll=lambda: 0, wait=wait)
    monkeypatch.setattr(controller.pw.subprocess, "Popen", popen)
    assert controller.worker(workflow, "audit") == 0
    assert environments[0]["TDN_CONSISTENCY_WORKFLOW"] == str(base / controller.MANIFEST)
    assert environments[0]["TDN_CONSISTENCY_PROTOCOL_SHA256"] == workflow["protocol_sha256"]
    assert "TDN_PREMIX_WORKFLOW" not in environments[0]
    assert controller.cw.read_json(controller.pw.state_path(workflow, "audit"))["status"] == "COMPLETED"
    assert finished[-1]["state"] == "COMPLETED"
    with pytest.raises(ValueError, match="already started"):
        controller.worker(workflow, "audit")


def test_software_change_since_submission_stops_before_child(controller, monkeypatch):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller.rw, "software_report", lambda *a: {"packages": {"torch": "changed"}})
    monkeypatch.setattr(controller.pw.subprocess, "Popen", lambda *a, **kw: pytest.fail("software changed"))
    with pytest.raises(controller.rw.WorkerFailure, match="submission environment"):
        controller.worker(workflow, "audit")
    assert finished[-1]["state"] == "FAILED"


@pytest.mark.parametrize("stage", ["prepare", "neural"])
def test_missing_structural_prerequisite_stops_every_successor_before_child(controller, monkeypatch, stage):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller.pw.subprocess, "Popen", lambda *a, **kw: pytest.fail("missing audit"))
    with pytest.raises(controller.rw.WorkerFailure):
        controller.worker(workflow, stage)
    assert finished[-1]["state"] == "FAILED"


def test_skipped_gpu_tests_prevent_training_and_are_reported(controller, monkeypatch):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller, "verify_prerequisites", lambda *a: {"audit": "fixture", "prepare": "fixture"})
    monkeypatch.setattr(controller, "worker_commands", lambda *a: [("gpu-tests", ["fixture-tests"]), ("experiment", ["must-not-run"])])
    started = []
    def popen(args, **kwargs):
        started.append(args)
        path = controller.gpu_junit_path(workflow)
        path.parent.mkdir(parents=True)
        path.write_text('<testsuite tests="3" skipped="3"/>')
        return SimpleNamespace(pid=123456789, poll=lambda: 0, wait=lambda timeout=None: 0)
    monkeypatch.setattr(controller.pw.subprocess, "Popen", popen)
    with pytest.raises(controller.rw.WorkerFailure, match="must execute"):
        controller.worker(workflow, "neural")
    assert started == [["fixture-tests"]] and finished[-1]["state"] == "FAILED"


def test_usr1_preserves_partial_evidence_and_blocks_successor(controller, monkeypatch):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller, "worker_commands", lambda *a: [("first", ["fixture"]), ("second", ["must-not-run"])])
    sent, started = [], []
    monkeypatch.setattr(controller.pw.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
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
    monkeypatch.setattr(controller.pw.subprocess, "Popen", popen)
    with pytest.raises(controller.rw.WorkerFailure) as error:
        controller.worker(workflow, "audit")
    assert error.value.exit_code == 75 and sent == [(123456789, signal.SIGUSR1)]
    assert len(started) == 1 and finished[-1]["state"] == "INTERRUPTED"


@pytest.mark.parametrize("script", ["fedora_consistency.sh", "consistency_worker.sh", "consistency_local.sh"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)


def test_controller_import_does_not_import_numerical_dependencies():
    code = "import importlib.util,sys; s=importlib.util.spec_from_file_location('isolated_controller',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.declaration('smoke'); from tdn.analysis.consistency.artifacts import verify_stage; assert not {'torch','numpy','yaml'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts" / "consistency_workflow.py")], check=True)


@pytest.mark.parametrize("failure_stage,expected", [(None, ["audit", "prepare", "neural"]), ("prepare", ["audit", "prepare"])])
def test_local_wrapper_has_same_sequential_dependencies_and_preserves_failures(tmp_path, failure_stage, expected):
    root = tmp_path / "local project"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    for name in ("consistency_local.sh", "common.sh"):
        (root / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    (root / ".venv" / "pyvenv.cfg").write_text("fixture: shell sequencing only\n")
    interpreter = root / ".venv" / "bin" / "python"
    interpreter.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -c ]]; then exit 0; fi
stage=""; audit=""; dataset=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage="$2"; shift 2 ;;
        --audit-dir) audit="$2"; shift 2 ;;
        --dataset-dir) dataset="$2"; shift 2 ;;
        *) shift ;;
    esac
done
[[ "$stage" == audit || "$audit" == "$TDN_REPO_ROOT/runs/local-test/audit" ]]
[[ "$stage" != neural || "$dataset" == "$TDN_REPO_ROOT/runs/local-test/prepare" ]]
printf '%s\\n' "$stage" >> "$TDN_REPO_ROOT/sequence.txt"
[[ "$stage" != "${FIXTURE_FAILURE_STAGE:-}" ]]
''')
    interpreter.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in (
        "SLURM_JOB_ID", "SLURM_STEP_ID", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT", "CONDA_PREFIX", "CONDA_SHLVL")}
    env["FIXTURE_FAILURE_STAGE"] = failure_stage or ""
    result = subprocess.run(["bash", str(root / "scripts" / "consistency_local.sh"), "--run-dir", "runs/local-test"],
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (failure_stage is None), result.stderr
    assert (root / "sequence.txt").read_text().splitlines() == expected
    assert (root / "runs" / "local-test").is_dir()
    repeat = subprocess.run(["bash", str(root / "scripts" / "consistency_local.sh"), "--run-dir", "runs/local-test"],
                            env=env, capture_output=True, text=True)
    assert repeat.returncode == 2 and "Run path exists" in repeat.stderr
    assert (root / "sequence.txt").read_text().splitlines() == expected
