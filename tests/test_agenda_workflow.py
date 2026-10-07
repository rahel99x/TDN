"""Scheduler invariants, archival integrity and real Tower I/O; no native GPU claim."""
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
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {
    "structure": {"seconds": 600, "cpus": 8, "mem_gib": 48, "walltime": "00:20:00"},
    "prepare": {"seconds": 1800, "cpus": 8, "mem_gib": 48, "walltime": "00:45:00"},
    "controls": {"seconds": 1200, "cpus": 4, "mem_gib": 32, "walltime": "00:30:00"},
    "optimize": {"seconds": 1800, "cpus": 4, "mem_gib": 32, "walltime": "00:45:00"},
    "compression": {"seconds": 1200, "cpus": 4, "mem_gib": 32, "walltime": "00:30:00"},
    "kernel": {"seconds": 1200, "cpus": 4, "mem_gib": 32, "walltime": "00:30:00"},
    "confirm": {"seconds": 1800, "cpus": 4, "mem_gib": 48, "walltime": "00:45:00"},
    "policy": {"seconds": 1200, "cpus": 4, "mem_gib": 32, "walltime": "00:30:00"},
}


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("agenda_workflow_test", ROOT / "scripts" / "agenda_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(name + "\n")
    for name in ("test_agenda_gpu.py", "test_agenda_models.py", "test_agenda_workflow.py", "test_consistency_gpu.py",
                 "test_desktop_slurm_runtime.py", "test_fedora_workflow.py"):
        (root / "tests" / name).write_text("def test_ok(): pass\n")
    (root / "scripts" / "agenda_worker.sh").write_bytes((ROOT / "scripts" / "agenda_worker.sh").read_bytes())
    (root / "requirements.txt").write_text("numpy==2.2.6\n")
    (root / "pyproject.toml").write_text("[project]\nname='tdn'\n")
    for owner in (module, module.fw, module.pw, module.rw, module.cw):
        monkeypatch.setattr(owner, "ROOT", root)
    for key in tuple(os.environ):
        if key.startswith(("TDN_", "SLURM_")) or key in ("CONDA_PREFIX", "CONDA_SHLVL"):
            monkeypatch.delenv(key, raising=False)
    profile = {"schema_version": 1, "kind": "desktop-slurm", "root": str(root), "user": getpass.getuser(),
               "cpu_partition": "local", "gpu_partition": "local", "account": None,
               "gpu_gres": "gpu:1", "expected_gpu_name": "RTX 4090", "gpu_vram_gib": 24,
               "torch_version": "2.10.0+cu126", "torch_wheel_index": "https://download.pytorch.org/whl/cu126"}
    module.cw.atomic_json(root / ".tdn" / "fedora-slurm.json", profile)
    # Synthetic protocol isolates controller policy from numerical engines.
    monkeypatch.setattr(module, "scientific_protocol", lambda profile: {"fixture": "scheduler contracts only", "profile": profile,
                        "budgets": {stage: dict(row, seconds=120 if profile == "smoke" else row["seconds"]) for stage, row in BUDGETS.items()}})
    return module


def plan(module, name="test", *flags):
    return module.prepare(module.parser().parse_args(["plan", "--run-id", name, *flags]))


def freeze(module, workflow, software=None):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "logs").mkdir()
    for path, value in ((base / module.MANIFEST, workflow), (base / "protocol.json", module.declaration(workflow["profile"])),
                        (base / "slurm-profile.json", workflow["slurm_profile"]), (base / "jobs.json", []),
                        (base / "state" / "submission-software.json", software or {"packages": {"torch": "fixture"}})):
        module.cw.atomic_json(path, value)
    return base


def mock_submission(module, monkeypatch, *, fail_at=None, outputs=None, capacity="system|16|110000|gpu:1"):
    calls = []
    monkeypatch.setattr(module.rw, "software_report", lambda *a: {"packages": {"torch": "fixture"}})
    def command(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "scontrol":
            return SimpleNamespace(stdout="PartitionName=local State=UP MaxTime=7-00:00:00")
        if args[0] == "sinfo":
            return SimpleNamespace(stdout=capacity)
        count = len([row for row in calls if row[0][0] == "sbatch"])
        if count == fail_at:
            raise ValueError("simulated submission failure")
        return SimpleNamespace(stdout=outputs[count - 1] if outputs else str(9000 + count))
    monkeypatch.setattr(module.cw, "command", command)
    return calls


@pytest.mark.parametrize("flags", [[], ["--smoke"], ["--development"]])
def test_plan_is_read_only_and_respects_actual_hardware(controller, monkeypatch, capsys, flags):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("plan called scheduler"))
    assert controller.main(["plan", *flags]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 8
    assert output.count("--gres=none") == 2 and output.count("--gres=gpu:1") == 6
    assert output.count("--cpus-per-task=8") == 2 and output.count("--cpus-per-task=4") == 6
    assert output.count("--mem=48G") == 3 and output.count("--mem=32G") == 5
    assert output.count("--time=00:45:00") == 3
    assert "anakano_81" not in output and "a100" not in output and "--account=" not in output
    assert "structure -> prepare -> controls -> optimize -> compression -> kernel -> confirm -> policy" in output
    assert not (controller.ROOT / "runs").exists() and not (controller.ROOT / ".cache").exists()


def test_eight_stage_serial_dag_binds_all_prior_stages_without_other_run_pointer(controller, monkeypatch):
    calls = mock_submission(controller, monkeypatch)
    assert controller.main(["run", "--run-id", "serial"]) == 0
    workflow = controller.load(controller.workflow_path("latest"))
    batches = [(args, kwargs) for args, kwargs in calls if args[0] == "sbatch"]
    assert len(batches) == 8
    for index, (args, kwargs) in enumerate(batches):
        dependency = "afterok:" + ":".join(str(9001 + prior) for prior in range(index)) if index else None
        assert [item for item in args if item.startswith("--dependency=")] == (["--dependency=" + dependency] if dependency else [])
        stage = controller.STAGES[index]
        assert kwargs["env"]["TDN_AGENDA_STAGE"] == stage
        assert kwargs["env"]["TDN_AGENDA_CPUS"] == str(BUDGETS[stage]["cpus"])
        assert kwargs["env"]["TDN_SLURM_CONFIG"] == workflow["slurm_profile_path"]
        assert kwargs["env"]["TDN_AGENDA_WORKFLOW"].endswith("agenda-workflow.json")
    assert Path(workflow["slurm_profile_path"]).stat().st_mode & 0o222 == 0
    assert (controller.ROOT / "runs" / controller.POINTER).is_file()
    assert not (controller.ROOT / "runs" / ".fedora-consistency-latest.json").exists()
    assert not (controller.ROOT / "runs" / ".fedora-premix-latest.json").exists()


@pytest.mark.parametrize("capacity", ["system|4|110000|gpu:1", "system|16|32000|gpu:1", "system|16|110000|(null)", "ambiguous"])
def test_insufficient_scheduler_capacity_cannot_submit_any_job(controller, monkeypatch, capacity):
    calls = mock_submission(controller, monkeypatch, capacity=capacity)
    assert controller.main(["run"]) == 2
    assert not [args for args, kw in calls if args[0] == "sbatch"]
    assert not (controller.ROOT / "runs").exists()


@pytest.mark.parametrize("alter", [{"cpus": 16}, {"mem_gib": 108}, {"walltime": "20:00:00"}])
def test_resources_cannot_exceed_physical_cpu_memory_or_wall_cap(controller, alter):
    resources = controller.stage_resources("full")
    resources["confirm"].update(alter)
    with pytest.raises(ValueError):
        controller.validate_resources(resources)


def test_scheduler_limits_are_independent_of_other_projects_and_local_account_optional(controller, monkeypatch):
    stale = {"SBATCH_ACCOUNT": "anakano_81", "SBATCH_CONSTRAINT": "a100-40gb", "SBATCH_GRES": "gpu:a100:4",
             "SRUN_NTASKS": "30", "CARC_ACCOUNT": "anakano_81", "TORCH_CUDA_ARCH_LIST": "8.0",
             "TDN_PREMIX_WORKFLOW": "old", "TDN_CONSISTENCY_STAGE": "neural", "TDN_AGENDA_CPUS": "16"}
    for key, value in stale.items():
        monkeypatch.setenv(key, value)
    calls = mock_submission(controller, monkeypatch)
    assert controller.main(["run", "--run-id", "clean-env"]) == 0
    for args, kwargs in calls:
        if args[0] == "sbatch":
            assert (stale.keys() - {"TDN_AGENDA_CPUS"}).isdisjoint(kwargs["env"])
            assert kwargs["env"]["TDN_AGENDA_CPUS"] in ("4", "8")
            assert not any(item.startswith("--account=") for item in args)
    workflow = plan(controller, "account")
    workflow["slurm_profile"]["account"] = "local-research"
    assert "--account=local-research" in controller.scheduler_args(workflow, "controls")


@pytest.mark.parametrize("stage,dependency", [("structure", "afterok:1"), ("prepare", "afterok:1:2"),
    ("prepare", "afterany:1"), ("controls", "afterok:1"), ("controls", "afterok:1:1"), ("controls", "afterok:1:x"),
    ("policy", "afterok:1:2:3:4:5:6")])
def test_only_fixed_all_prior_stage_dependencies_are_allowed(controller, stage, dependency):
    with pytest.raises(ValueError, match="dependency"):
        controller.scheduler_args(plan(controller), stage, dependency)


@pytest.mark.parametrize("env", [{"SLURM_JOB_ID": "123"}, {"TDN_EXECUTION_MODE": "desktop"},
                                  {"TDN_LOCAL_TEST_ROOT": "spoof"}, {"CONDA_PREFIX": "/conda"}])
def test_submission_rejects_runtime_bypass(controller, monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("unsafe scheduler request"))
    assert controller.main(["run"]) == 2


@pytest.mark.parametrize("target", ["source", "profile", "protocol"])
def test_frozen_mutation_blocks_successors_and_preserves_inspectability(controller, target):
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


def test_partial_submission_and_ambiguous_id_preserve_evidence_without_retry(controller, monkeypatch):
    calls = mock_submission(controller, monkeypatch, fail_at=3)
    assert controller.main(["run", "--run-id", "partial"]) == 2
    workflow = controller.load(controller.workflow_path("latest"))
    state = controller.cw.read_json(Path(workflow["run_dir"]) / "state" / "submission.json")
    assert state["status"] == "FAILED" and len(state["submitted_jobs"]) == 2
    assert len([row for row in calls if row[0][0] == "sbatch"]) == 3
    calls = mock_submission(controller, monkeypatch, outputs=["9001", "9001"])
    assert controller.main(["run", "--run-id", "duplicate"]) == 2
    assert len([row for row in calls if row[0][0] == "sbatch"]) == 2


def test_stage_commands_retain_every_prior_seal_and_real_cuda_readiness(controller):
    workflow = plan(controller)
    structure = controller.worker_commands(workflow, "structure")
    assert [name for name, _ in structure] == ["tests", "experiment"]
    assert str(controller.ROOT / "tests" / "test_agenda_workflow.py") in structure[0][1]
    for stage in controller.STAGES:
        commands = controller.worker_commands(workflow, stage)
        if stage in controller.GPU_STAGES:
            assert [name for name, _ in commands] == ["preflight", "gpu-tests", "check-gpu-tests", "experiment"]
            assert str(controller.ROOT / "tests" / "test_consistency_gpu.py") in commands[1][1]
            assert str(controller.ROOT / "tests" / "test_agenda_gpu.py") in commands[1][1]
            assert str(controller.junit_path(workflow, stage)) in commands[1][1]
            assert commands[2][1][-3:] == ["check-gpu-tests", "--junit", str(controller.junit_path(workflow, stage))]
            assert "--dataset-dir" in commands[-1][1] and "cuda" in commands[-1][1]
        command = commands[-1][1]
        assert str(controller.ROOT / "scripts" / "agenda.py") in command
        bound = [command[i + 1] for i, value in enumerate(command) if value == "--prerequisite-dir"]
        assert bound == [f"{prior}={Path(workflow['run_dir']) / prior}" for prior in controller.STAGES[:controller.STAGES.index(stage)]]


@pytest.mark.parametrize("stage", ["structure", "prepare", "controls", "confirm"])
def test_allocation_reads_real_compound_slurm_fields_and_exact_stage_resources(controller, monkeypatch, stage):
    from tdn.runtime import desktop_slurm
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("TDN_SLURM_CONFIG", str(base / "slurm-profile.json"))
    resource = workflow["resources"][stage]
    job = (f"JobId=9001 JobName=tdn-agenda-{stage} UserId={getpass.getuser()}({os.getuid()}) "
           f"JobState=RUNNING Partition=local AllocNode:Sid=system:8749 NumNodes=1 "
           f"AllocTRES=cpu={resource['cpus']},mem={resource['mem_gib']}G,node=1,gres/gpu=1 "
           f"TimeLimit={resource['walltime']} WorkDir={controller.ROOT} Comment={controller.comment(workflow, stage)}")
    step = f"StepId=9001.0 UserId={os.getuid()} State=RUNNING Partition=local Nodes=1 Tasks=1 TRES=cpu={resource['cpus']},node=1,gres/gpu=1"
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


@pytest.mark.parametrize("mutation", ["cpu", "memory", "wall", "step_cpu"])
def test_mismatched_actual_allocations_cannot_start_compute(controller, monkeypatch, mutation):
    workflow = plan(controller)
    freeze(controller, workflow)
    resource = workflow["resources"]["controls"]
    fields = {"JobId": "9001", "JobName": "tdn-agenda-controls", "Comment": controller.comment(workflow, "controls"),
              "JobState": "RUNNING", "Partition": "local", "WorkDir": str(controller.ROOT), "UserId": getpass.getuser(),
              "AllocTRES": "cpu=4,mem=32G", "TimeLimit": resource["walltime"]}
    step = {"TRES": "cpu=4"}
    if mutation == "cpu": fields["AllocTRES"] = "cpu=8,mem=32G"
    if mutation == "memory": fields["AllocTRES"] = "cpu=4,mem=128G"
    if mutation == "wall": fields["TimeLimit"] = "20:00:00"
    if mutation == "step_cpu": step["TRES"] = "cpu=16"
    monkeypatch.setattr(controller, "runtime_allocation", lambda *a, **kw: {"job_id": "9001", "job": fields, "step": step})
    with pytest.raises(ValueError, match="budget"):
        controller.verify_allocation(workflow, "controls")


def test_test_environment_keeps_real_gpu_binding_but_removes_production_evidence(controller):
    env = {"TDN_EXECUTION_MODE": "desktop-slurm", "TDN_SLURM_CONFIG": "frozen", "SLURM_JOB_ID": "9001",
           "PYTEST_ADDOPTS": "-k one_test", "PYTEST_PLUGINS": "filter_plugin",
           **controller.workflow_environment(plan(controller), "controls")}
    gpu = dict(env)
    controller.prepare_test_environment(env, "tests")
    controller.prepare_test_environment(gpu, "gpu-tests")
    assert env == {"SLURM_JOB_ID": "9001"}
    assert gpu == {"TDN_EXECUTION_MODE": "desktop-slurm", "TDN_SLURM_CONFIG": "frozen", "SLURM_JOB_ID": "9001"}


def fake_verified_stage(module, workflow, stage):
    directory = Path(workflow["run_dir"]) / stage
    directory.mkdir()
    module.cw.atomic_json(directory / "manifest.json", {"fixture": "hash-only prerequisite test"})
    return directory


def test_prerequisites_cover_all_prior_stages_and_same_software(controller, monkeypatch):
    workflow = plan(controller)
    freeze(controller, workflow)
    software = {"packages": {"torch": "fixture"}}
    monkeypatch.setattr(controller, "verify_stage", lambda *a: None)
    with pytest.raises(FileNotFoundError):
        controller.verify_prerequisites(workflow, software, "controls")
    for stage in controller.STAGES[:-1]:
        directory = fake_verified_stage(controller, workflow, stage)
        software_path = Path(workflow["run_dir"]) / "state" / f"{stage}-software.json"
        controller.cw.atomic_json(software_path, software)
        controller.cw.atomic_json(controller.pw.state_path(workflow, stage), {"status": "COMPLETED", "exit_code": 0,
            "source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
            "software_sha256": controller.cw.digest(software_path)})
    hashes = controller.verify_prerequisites(workflow, software, "policy")
    assert set(hashes) == set(controller.STAGES[:-1])
    with pytest.raises(ValueError, match="software differs"):
        controller.verify_prerequisites(workflow, {"packages": {"torch": "changed"}}, "policy")
    state = controller.pw.state_path(workflow, "kernel")
    controller.cw.atomic_json(state, {"status": "FAILED"})
    with pytest.raises(ValueError, match="worker prerequisite"):
        controller.verify_prerequisites(workflow, software, "policy")


@pytest.fixture
def reporting(controller, monkeypatch):
    from tdn import reporting
    monkeypatch.setattr(reporting, "ROOT", controller.ROOT)
    return reporting


@pytest.mark.parametrize("stage", ["structure", "prepare", "controls", "confirm", "policy"])
def test_tower_sidecar_records_stage_resources_outside_science(controller, reporting, monkeypatch, stage):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    report = controller.begin_report(workflow, stage)
    assert report.parent == base / "tower" and not (base / stage).exists()
    run = reporting.read_json(report / "run.json")
    assert run["parameters"]["execution_mode"] == "desktop-slurm"
    assert run["parameters"]["slurm_profile_sha256"] == workflow["slurm_profile_sha256"]
    assert run["resources"]["mem_bytes"] == BUDGETS[stage]["mem_gib"] * 2**30
    assert run["resources"]["cpus"] == BUDGETS[stage]["cpus"]


def test_tower_failure_ingests_only_its_stage_junit(controller, reporting, monkeypatch):
    workflow = plan(controller)
    freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    report = controller.begin_report(workflow, "controls")
    path = controller.junit_path(workflow, "controls")
    path.parent.mkdir(parents=True)
    path.write_text('<testsuite tests="1" failures="1"><testcase name="physical"><failure message="bad"/></testcase></testsuite>')
    other = controller.junit_path(workflow, "confirm")
    other.parent.mkdir(parents=True)
    other.write_text('<testsuite tests="99"/>')
    controller.finish_report(workflow, "controls", report, state="FAILED", runtime_seconds=.1, exit_code=2)
    summary = reporting.read_json(report / "summary.json")
    assert summary["state"] == "FAILED" and summary["results"]["test_outcomes"]["FAILED"] == 1
    with (report / "outputs" / "tests.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["test"] == "physical"
    assert rows[0]["source_path"].endswith("reporter-tests/controls/tests.xml")


def worker_fixture(module, monkeypatch):
    software = {"packages": {"torch": "fixture"}}
    workflow = plan(module, "worker", "--smoke")
    base = freeze(module, workflow, software)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(module, "verify_allocation", lambda *a: None)
    monkeypatch.setattr(module, "verify_stage", lambda *a: None)
    monkeypatch.setattr(module, "begin_report", lambda *a: base / "tower" / "fixture-report")
    monkeypatch.setattr(module.rw, "software_report", lambda *a: software)
    finished = []
    monkeypatch.setattr(module, "finish_report", lambda *a, **kw: finished.append(kw))
    monkeypatch.setattr(module, "worker_commands", lambda *a: [("experiment", ["fixture-program"])])
    return workflow, base, software, finished


def test_shared_worker_binds_agenda_and_checks_submission_software_before_compute(controller, monkeypatch):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    envs = []
    def popen(args, **kwargs):
        envs.append(kwargs["env"])
        return SimpleNamespace(pid=123456789, poll=lambda: 0, wait=lambda timeout=None: 0)
    monkeypatch.setattr(controller.pw.subprocess, "Popen", popen)
    assert controller.worker(workflow, "structure") == 0
    assert envs[0]["TDN_AGENDA_WORKFLOW"] == str(base / controller.MANIFEST)
    assert envs[0]["TDN_AGENDA_PROTOCOL_SHA256"] == workflow["protocol_sha256"]
    assert "TDN_PREMIX_WORKFLOW" not in envs[0]
    assert finished[-1]["state"] == "COMPLETED"
    with pytest.raises(ValueError, match="already started"):
        controller.worker(workflow, "structure")


def test_software_change_and_missing_prerequisite_stop_before_child(controller, monkeypatch):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller.pw.subprocess, "Popen", lambda *a, **kw: pytest.fail("unsafe child"))
    monkeypatch.setattr(controller.rw, "software_report", lambda *a: {"packages": {"torch": "changed"}})
    with pytest.raises(controller.rw.WorkerFailure, match="submission environment"):
        controller.worker(workflow, "structure")
    assert finished[-1]["state"] == "FAILED"
    monkeypatch.setattr(controller.rw, "software_report", lambda *a: software)
    with pytest.raises(controller.rw.WorkerFailure):
        controller.worker(workflow, "controls")
    assert finished[-1]["state"] == "FAILED"


def test_skipped_cuda_tests_block_training_with_its_stage_junit(controller, monkeypatch):
    workflow, base, software, finished = worker_fixture(controller, monkeypatch)
    monkeypatch.setattr(controller, "verify_prerequisites", lambda *a: {"fixture": "mock seals only"})
    monkeypatch.setattr(controller, "worker_commands", lambda *a: [("gpu-tests", ["fixture-tests"]), ("experiment", ["must-not-run"])])
    started = []
    def popen(args, **kwargs):
        started.append(args)
        assert kwargs["env"]["TDN_REQUIRE_GPU_TESTS"] == "1"
        path = controller.junit_path(workflow, "compression")
        path.parent.mkdir(parents=True)
        path.write_text('<testsuite tests="3" skipped="3"/>')
        return SimpleNamespace(pid=123456789, poll=lambda: 0, wait=lambda timeout=None: 0)
    monkeypatch.setattr(controller.pw.subprocess, "Popen", popen)
    with pytest.raises(controller.rw.WorkerFailure, match="must execute"):
        controller.worker(workflow, "compression")
    assert started == [["fixture-tests"]] and finished[-1]["state"] == "FAILED"


def test_usr1_preserves_partial_evidence_without_a_successor(controller, monkeypatch):
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
        controller.worker(workflow, "structure")
    assert error.value.exit_code == 75 and sent == [(123456789, signal.SIGUSR1)]
    assert len(started) == 1 and finished[-1]["state"] == "INTERRUPTED"


def test_collect_preserves_checkpoints_failures_and_verified_upload_parts(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (base / "controls").mkdir()
    (base / "controls" / "checkpoint.pt").write_bytes(b"fixture evidence")
    controller.cw.atomic_json(base / "state" / "failure.json", {"status": "FAILED"})
    (base / "structure-pytest-work").mkdir()
    (base / "structure-pytest-work" / "discard.txt").write_text("temporary")
    archive = controller.collect(workflow, part_bytes=100)
    index = controller.cw.read_json(archive.with_name(archive.name + ".index.json"))
    assert index["sha256"] == controller.cw.digest(archive)
    assert len(index["parts"]) > 1
    rebuilt = b"".join((archive.parent / row["path"]).read_bytes() for row in index["parts"])
    assert rebuilt == archive.read_bytes()
    for row in index["parts"]:
        assert row["bytes"] <= 100 and controller.cw.digest(archive.parent / row["path"]) == row["sha256"]
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert f"{base.name}/controls/checkpoint.pt" in names
    assert f"{base.name}/state/failure.json" in names
    assert not any("pytest-work" in name for name in names)


def test_collect_rejects_symlink_without_exporting_external_files(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (base / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        controller.collect(workflow)


@pytest.mark.parametrize("script", ["fedora_agenda.sh", "agenda_worker.sh", "agenda_local.sh"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)


@pytest.mark.parametrize("failure_stage,expected_count", [(None, 8), ("compression", 5)])
def test_local_wrapper_passes_exact_sequential_prerequisites_and_preserves_failure(tmp_path, failure_stage, expected_count):
    root = tmp_path / "local project"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    for name in ("agenda_local.sh", "common.sh"):
        (root / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    (root / ".venv" / "pyvenv.cfg").write_text("fixture: shell sequencing only\n")
    interpreter = root / ".venv" / "bin" / "python"
    interpreter.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -c ]]; then exit 0; fi
stage=""; dataset=""; count=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage="$2"; shift 2 ;;
        --prerequisite-dir) [[ "$2" == *"=$TDN_REPO_ROOT/runs/local-test/"* ]]; count=$((count+1)); shift 2 ;;
        --dataset-dir) dataset="$2"; shift 2 ;;
        *) shift ;;
    esac
done
case "$stage" in structure|prepare) ;; *) [[ "$dataset" == "$TDN_REPO_ROOT/runs/local-test/prepare" ]] ;; esac
printf '%s:%s\\n' "$stage" "$count" >> "$TDN_REPO_ROOT/sequence.txt"
[[ "$stage" != "${FIXTURE_FAILURE_STAGE:-}" ]]
''')
    interpreter.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("TDN_", "SLURM_", "CONDA_"))}
    env["FIXTURE_FAILURE_STAGE"] = failure_stage or ""
    result = subprocess.run(["bash", str(root / "scripts" / "agenda_local.sh"), "--run-dir", "runs/local-test"],
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (failure_stage is None), result.stderr
    sequence = (root / "sequence.txt").read_text().splitlines()
    stages = list(BUDGETS)[:expected_count]
    assert sequence == [f"{stage}:{index}" for index, stage in enumerate(stages)]
    repeat = subprocess.run(["bash", str(root / "scripts" / "agenda_local.sh"), "--run-dir", "runs/local-test"],
                            env=env, capture_output=True, text=True)
    assert repeat.returncode == 2 and "Run path exists" in repeat.stderr


def test_local_wrapper_refuses_fresh_full_confirmation():
    env = {key: value for key, value in os.environ.items() if not key.startswith(("TDN_", "SLURM_", "CONDA_"))}
    result = subprocess.run(["bash", str(ROOT / "scripts" / "agenda_local.sh"), "--full"], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "allocated Fedora workflow" in result.stderr


def test_controller_declarations_import_without_numerical_dependencies():
    code = "import importlib.util,sys; s=importlib.util.spec_from_file_location('isolated_agenda',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.declaration('smoke'); assert not {'torch','numpy','yaml'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts" / "agenda_workflow.py")], check=True)


def test_all_real_protocol_profiles_share_safe_scheduler_limits():
    spec = importlib.util.spec_from_file_location("real_agenda_budget", ROOT / "scripts" / "agenda_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for profile in module.PROFILES:
        declaration = module.declaration(profile)
        resources = module.validate_resources(declaration["resources_per_stage"])
        assert resources == {stage: {key: row[key] for key in ("cpus", "mem_gib", "walltime")} for stage, row in BUDGETS.items()}
        assert max(resource["mem_gib"] * 1024 for resource in resources.values()) < 110000
        assert max(module.wall_seconds(resource["walltime"]) for resource in resources.values()) == 2700
        for stage in module.STAGES:
            budget = declaration["scientific_protocol"]["budgets"][stage]
            assert 0 < budget["seconds"] <= module.wall_seconds(budget["walltime"]) - 180


def mandatory_gpu_junit(path, *, mutation=None, multiple_suites=False):
    """Synthetic metadata tests completeness checks; no CUDA execution is claimed."""
    from xml.etree import ElementTree
    tree = ElementTree.Element("testsuites", tests="152")
    suite = ElementTree.SubElement(tree, "testsuite", tests="152", failures="0", errors="0", skipped="0")
    rows = [("tests.test_consistency_gpu", f"consistency-{index}") for index in range(38)]
    rows += [("tests.test_agenda_gpu", f"agenda-{index}") for index in range(114)]
    if mutation == "filtered": rows = rows[:1]
    elif mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1] = rows[-2]
    elif mutation == "wrong_module": rows[-1] = ("tests.test_other_gpu", rows[-1][1])
    if multiple_suites:
        suite.set("tests", "38")
        second = ElementTree.SubElement(tree, "testsuite", tests="114", failures="0", errors="0", skipped="0")
    for module, name in rows:
        target = second if multiple_suites and module == "tests.test_agenda_gpu" else suite
        case = ElementTree.SubElement(target, "testcase", classname=module, name=name)
        if mutation in ("skipped", "failure", "error") and len(target.findall("testcase")) == 1:
            ElementTree.SubElement(case, mutation)
    if not multiple_suites and mutation not in ("declared_count", "aggregate_count"):
        suite.set("tests", str(len(rows)))
        tree.set("tests", str(len(rows)))
    if mutation == "declared_count": suite.set("tests", "153")
    if mutation == "aggregate_count": tree.set("tests", "151")
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(tree).write(path, encoding="unicode")
    return path


@pytest.mark.parametrize("multiple_suites", [False, True])
def test_gpu_completeness_validator_accepts_only_full_named_modules(controller, multiple_suites, capsys):
    path = mandatory_gpu_junit(controller.ROOT / "metadata-tests.xml", multiple_suites=multiple_suites)
    result = controller.check_gpu_tests(path)
    assert result == {"validation": "mandatory_agenda_gpu_suite", "test_cases": 152,
                      "modules": {"tests.test_consistency_gpu": 38, "tests.test_agenda_gpu": 114}}
    assert controller.main(["check-gpu-tests", "--junit", str(path)]) == 0
    assert '"test_cases": 152' in capsys.readouterr().out


@pytest.mark.parametrize("mutation,message", [("filtered", "152"), ("missing", "152"), ("duplicate", "unique"),
    ("wrong_module", "declared modules"), ("skipped", "without failures"), ("failure", "without failures"),
    ("error", "without failures"), ("declared_count", "declared test counts"), ("aggregate_count", "aggregate test count")])
def test_gpu_readiness_rejects_filtered_duplicated_missing_and_misleading_junit(controller, mutation, message):
    path = mandatory_gpu_junit(controller.ROOT / "metadata-tests.xml", mutation=mutation)
    with pytest.raises(ValueError, match=message):
        controller.check_gpu_tests(path)
    assert controller.main(["check-gpu-tests", "--junit", str(path)]) == 2
