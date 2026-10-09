"""Native scheduler contracts and artifact transport; mocks are not GPU evidence."""
from __future__ import annotations

import getpass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("audit", "screen", "prepare", "train", "confirm_prepare", "confirm", "scaling", "policy", "report")
GPU = ("train", "confirm", "scaling", "policy")


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("frontier_workflow_test", ROOT / "scripts/frontier_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(name + "\n")
    for name in ("test_frontier_gpu.py", "test_frontier_models.py", "test_frontier_workflow.py",
                 "test_desktop_slurm_runtime.py", "test_fedora_workflow.py"):
        (root / "tests" / name).write_text("def test_ok(): pass\n")
    for name in ("frontier_worker.sh", "frontier.py"):
        (root / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    (root / "requirements.txt").write_text("numpy==2.2.6\n")
    (root / "pyproject.toml").write_text("[project]\nname='tdn'\n")
    for owner in (module, module.fw, module.pw, module.rw, module.cw):
        monkeypatch.setattr(owner, "ROOT", root)
    for key in tuple(os.environ):
        if key.startswith(("TDN_", "SLURM_")) or key in ("CONDA_PREFIX", "CONDA_SHLVL"):
            monkeypatch.delenv(key, raising=False)
    profile = {"schema_version": 1, "kind": "desktop-slurm", "root": str(root), "user": getpass.getuser(),
               "cpu_partition": "local", "gpu_partition": "local", "account": None, "gpu_gres": "gpu:1",
               "expected_gpu_name": "RTX 4090", "gpu_vram_gib": 24, "torch_version": "2.10.0+cu126",
               "torch_wheel_index": "https://download.pytorch.org/whl/cu126"}
    module.cw.atomic_json(root / ".tdn/fedora-slurm.json", profile)
    budgets = {stage: {"cpus": 4 if stage in GPU else 8, "mem_gib": 32 if stage in GPU else 48,
                      "walltime": "00:30:00", "seconds": 120} for stage in STAGES}
    from tdn.analysis.frontier.protocol import build_protocol
    monkeypatch.setattr(module, "scientific_protocol", lambda profile: {
        **build_protocol(profile), "budgets": budgets})
    return module


def plan(module, name="test", *flags):
    return module.prepare(module.parser().parse_args(["plan", "--run-id", name, *flags]))


def freeze(module, workflow):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "logs").mkdir()
    for path, value in ((base / module.MANIFEST, workflow), (base / "protocol.json", module.declaration(workflow["profile"])),
                        (base / "slurm-profile.json", workflow["slurm_profile"]), (base / "jobs.json", []),
                        (base / "state/submission-software.json", {"packages": {"torch": "fixture"}})):
        module.cw.atomic_json(path, value)
    return base


def submission(module, monkeypatch, *, fail_at=None, capacity="system|16|110000|gpu:1"):
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
            raise ValueError("submission failed")
        return SimpleNamespace(stdout=str(9000 + count))
    monkeypatch.setattr(module.cw, "command", command)
    return calls


@pytest.mark.parametrize("flags", [[], ["--smoke"], ["--development"], ["--full"]])
def test_plan_is_read_only_and_contains_all_allocations(controller, monkeypatch, capsys, flags):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("plan accessed scheduler"))
    assert controller.main(["plan", *flags]) == 0
    output = capsys.readouterr().out
    part_count = 2 if "--smoke" in flags else 4 if "--development" in flags else 6
    assert output.count("sbatch --parsable") == 9 + part_count
    assert output.count("--gres=none") == 5 and output.count("--gres=gpu:1") == 4 + part_count
    assert "anakano_81" not in output and "a100" not in output
    assert "pending-job cap" in output
    assert not (controller.ROOT / "runs").exists()


def test_entire_chain_is_queued_and_report_runs_after_failed_or_cancelled_stages(controller, monkeypatch):
    calls = submission(controller, monkeypatch)
    monkeypatch.setenv("SBATCH_ACCOUNT", "wrong")
    monkeypatch.setenv("TDN_AGENDA_STAGE", "wrong")
    assert controller.main(["run", "--run-id", "chain"]) == 0
    workflow = controller.load(controller.workflow_path("latest"))
    batches = [(args, kwargs) for args, kwargs in calls if args[0] == "sbatch"]
    tasks = controller.physical_stages(workflow)
    assert len(batches) == 15
    for index, (args, kwargs) in enumerate(batches):
        kind = "afterany" if tasks[index] == "report" else "afterok"
        dependency = kind + ":" + ":".join(str(9001 + prior) for prior in range(index)) if index else None
        assert [x for x in args if x.startswith("--dependency=")] == (["--dependency=" + dependency] if dependency else [])
        assert "--kill-on-invalid-dep=yes" in args
        assert kwargs["env"]["TDN_FRONTIER_STAGE"] == tasks[index]
        assert "SBATCH_ACCOUNT" not in kwargs["env"] and "TDN_AGENDA_STAGE" not in kwargs["env"]
    assert workflow["kind"] == "desktop-slurm-frontier"
    assert not (controller.ROOT / "runs/.fedora-agenda-latest.json").exists()


@pytest.mark.parametrize("stage,dependency", [("report", "afterok:" + ":".join(str(i) for i in range(1, 9))),
    ("train", "afterany:1:2:3"), ("audit", "afterok:1"), ("train", "afterok:1:1:2"), ("train", "afterok:1:2")])
def test_wrong_dependency_semantics_are_rejected(controller, stage, dependency):
    with pytest.raises(ValueError, match="dependency"):
        controller.scheduler_args(plan(controller), stage, dependency)


@pytest.mark.parametrize("capacity", ["system|2|110000|gpu:1", "system|16|30000|gpu:1", "system|16|110000|(null)"])
def test_capacity_failure_happens_before_any_submission(controller, monkeypatch, capacity):
    calls = submission(controller, monkeypatch, capacity=capacity)
    assert controller.main(["run", "--run-id", "capacity"]) == 2
    assert not any(args[0] == "sbatch" for args, _ in calls)
    assert not (controller.ROOT / "runs/capacity").exists()


def test_partial_submission_is_preserved_without_retry_or_cancellation(controller, monkeypatch):
    calls = submission(controller, monkeypatch, fail_at=4)
    assert controller.main(["run", "--run-id", "partial"]) == 2
    base = controller.ROOT / "runs/partial"
    state = json.loads((base / "state/submission.json").read_text())
    assert state["stage"] == "train" and len(state["submitted_jobs"]) == 4
    jobs = json.loads((base / "jobs.json").read_text())
    assert len(jobs) == 4 and jobs[-1]["stage"] == "report"
    assert jobs[-1]["dependency"] == "afterany:9001:9002:9003"
    assert state["aggregation_job_id"] == "9005"
    assert sum(args[0] == "sbatch" for args, _ in calls) == 5
    assert not any(args[0] == "scancel" for args, _ in calls)


@pytest.mark.parametrize("key,value", [("cpus", 9), ("cpus", 2), ("mem_gib", 64), ("walltime", "00:46:00"), ("walltime", "00:60:00")])
def test_fixed_hardware_bounds_are_enforced(controller, key, value):
    resources = controller.stage_resources("full")
    resources["screen"][key] = value
    with pytest.raises(ValueError):
        controller.validate_resources(resources)


def test_gpu_cpu_budget_and_environment_cleaning(controller):
    resources = controller.stage_resources("full")
    resources["train"]["cpus"] = 8
    with pytest.raises(ValueError, match="four"):
        controller.validate_resources(resources)
    env = {"TDN_FRONTIER_STAGE": "train", "TDN_ROADMAP_STAGE": "train", "TDN_AGENDA_STAGE": "controls", "PYTEST_ADDOPTS": "-k skip", "TDN_EXECUTION_MODE": "desktop-slurm"}
    controller.prepare_test_environment(env, "gpu-tests")
    assert env == {"TDN_EXECUTION_MODE": "desktop-slurm"}
    controller.prepare_test_environment(env, "tests")
    assert not env


def test_worker_commands_execute_full_cuda_suite_on_every_gpu_stage(controller):
    workflow = plan(controller)
    for stage in STAGES:
        commands = dict(controller.worker_commands(workflow, stage))
        experiment = commands["experiment"]
        assert experiment[experiment.index("--stage") + 1] == stage
        assert experiment.count("--prerequisite-dir") == STAGES.index(stage)
        assert "--dataset-dir" not in experiment
        if stage in GPU:
            assert {"preflight", "gpu-tests", "check-gpu-tests", "experiment"} == set(commands)
            assert str(controller.ROOT / "tests/test_frontier_gpu.py") in commands["gpu-tests"]
            assert "--device" in experiment and "cuda" in experiment
        elif stage == "audit":
            assert set(commands) == {"tests", "experiment"}
        elif stage == "report":
            assert list(commands) == ["accounting", "experiment"]
            assert "snapshot-accounting" in commands["accounting"]
        else:
            assert set(commands) == {"experiment"}


def test_frozen_protocol_and_source_are_checked_before_execution(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    assert controller.load(base / controller.MANIFEST) == workflow
    (controller.ROOT / "scripts/source.txt").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        controller.load(base / controller.MANIFEST)
    assert controller.load(base / controller.MANIFEST, verify=False) == workflow


@pytest.mark.parametrize("command", ["status", "validate", "run", "plan", "logs", "paths", "collect", "recover"])
def test_shell_uses_project_venv_without_activation(tmp_path, monkeypatch, command):
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "fedora_frontier.sh"
    wrapper.write_bytes((ROOT / "scripts/fedora_frontier.sh").read_bytes())
    # Real Bash execution distinguishes the project interpreter from PATH's
    # system Python and records argument boundaries, including spaces.
    interpreter = root / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('#!/bin/bash\nprintf "venv\\n"\nprintf "%s\\n" "$@"\nexit 7\n')
    interpreter.chmod(0o755)
    system = root / "system-bin/python3"
    system.parent.mkdir()
    system.write_text('#!/bin/bash\necho "wrong system Python" >&2\nexit 99\n')
    system.chmod(0o755)
    monkeypatch.delenv("TDN_PYTHON", raising=False)
    monkeypatch.setenv("PATH", str(system.parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(wrapper), command, "argument with spaces"],
                            text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode == 7
    assert result.stderr == ""
    assert result.stdout.splitlines() == ["venv", str(scripts / "frontier_workflow.py"), command, "argument with spaces"]


@pytest.mark.parametrize("override", [False, True])
def test_shell_keeps_explicit_python_override_and_pre_setup_planning(tmp_path, monkeypatch, override):
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "fedora_frontier.sh"
    wrapper.write_bytes((ROOT / "scripts/fedora_frontier.sh").read_bytes())
    interpreter = root / "custom interpreter/python3"
    interpreter.parent.mkdir()
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    interpreter.chmod(0o755)
    monkeypatch.delenv("TDN_PYTHON", raising=False)
    if override:
        monkeypatch.setenv("TDN_PYTHON", str(interpreter))
        project_python = root / ".venv/bin/python"
        project_python.parent.mkdir(parents=True)
        project_python.write_text('#!/bin/bash\nexit 99\n')
        project_python.chmod(0o755)
    else:
        monkeypatch.setenv("PATH", str(interpreter.parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(wrapper)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(scripts / "frontier_workflow.py"), "plan"]


@pytest.mark.parametrize("field,value", [("Partition", "wrong"), ("WorkDir", "/elsewhere"), ("JobName", "other"),
    ("TimeLimit", "00:45:00"), ("AllocTRES", "cpu=4,mem=32G"), ("UserId", "nobody(1)")])
def test_actual_allocation_must_match_frozen_resources_and_identity(controller, monkeypatch, field, value):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    resource = workflow["resources"]["audit"]
    fields = {"JobId": "101", "JobName": "tdn-frontier-audit", "Comment": controller.comment(workflow, "audit"),
        "JobState": "RUNNING", "Partition": "local", "WorkDir": str(controller.ROOT),
        "UserId": getpass.getuser() + "(1)", "AllocTRES": f"cpu={resource['cpus']},mem={resource['mem_gib']}G",
        "TimeLimit": resource["walltime"]}
    monkeypatch.setattr(controller, "runtime_allocation", lambda *a, **kw: {"job_id": "101", "job": fields,
                      "step": {"TRES": f"cpu={resource['cpus']}"}})
    controller.verify_allocation(workflow, "audit")
    fields[field] = value
    with pytest.raises(ValueError):
        controller.verify_allocation(workflow, "audit")


def test_report_backend_does_not_require_successful_predecessors(controller, monkeypatch):
    captured = {}
    def worker(workflow, stage, *, backend):
        captured["backend"] = backend
        return 0
    monkeypatch.setattr(controller.pw, "worker", worker)
    assert controller.worker(plan(controller), "report") == 0
    backend = captured["backend"]
    assert "report" not in backend.prerequisite_stages and "train" in backend.prerequisite_stages
    assert backend.workflow_label == "frontier"


def test_worker_startup_allocation_failure_is_visible_to_aggregation(controller, monkeypatch):
    workflow = plan(controller)
    freeze(controller, workflow)
    def fail(*args, **kwargs):
        raise ValueError("actual partition differs")
    monkeypatch.setattr(controller.pw, "worker", fail)
    with pytest.raises(ValueError, match="partition differs"):
        controller.worker(workflow, "audit")
    record = json.loads(controller.pw.state_path(workflow, "audit").read_text())
    assert record["status"] == "FAILED" and record["stage"] == "startup"
    assert "partition differs" in record["error"]


def test_collection_is_lossless_has_small_hashed_parts_and_excludes_test_scratch(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (base / "report").mkdir()
    (base / "report/review.csv").write_text("experiment_id,assessment,score\nx,NA,\n")
    (base / "train-pytest-work").mkdir()
    (base / "train-pytest-work/ignored.bin").write_bytes(b"irrelevant")
    path = controller.collect(workflow, part_bytes=100)
    index = json.loads(path.with_name(path.name + ".index.json").read_text())
    assert index["sha256"] == controller.cw.digest(path)
    parts = [path.parent / item["path"] for item in index["parts"]]
    assert len(parts) > 1 and b"".join(p.read_bytes() for p in parts) == path.read_bytes()
    for row, part in zip(index["parts"], parts):
        assert row["bytes"] <= 100 and row["sha256"] == controller.cw.digest(part)
    with tarfile.open(path) as archive:
        names = archive.getnames()
        assert any(name.endswith("report/review.csv") for name in names)
        assert not any("pytest-work" in name for name in names)


def test_collection_refuses_symlinks(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (base / "unsafe").symlink_to(controller.ROOT / "requirements.txt")
    with pytest.raises(ValueError, match="symlink"):
        controller.collect(workflow)


def accounting_workflow(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    controller.cw.atomic_json(base / "jobs.json", [{"stage": "audit", "job_id": "101"}, {"stage": "train", "job_id": "102"}])
    return workflow, base


def test_accounting_queries_only_frozen_jobs_and_keeps_allocation_and_steps_distinct(controller, capsys):
    workflow, base = accounting_workflow(controller)
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            "101|COMPLETED|0:0|60|00:02:00|cpu=8,mem=48G||480\n"
            "101.0|COMPLETED|0:0|58|00:01:59|cpu=8,mem=48G|120M|464\n"
            "102|COMPLETED|0:0|90|00:01:20|cpu=4,mem=32G,gres/gpu=1||360\n"))
    result = controller.scheduler_accounting(workflow, runner=runner)
    command, options = calls[0]
    assert command[command.index("--jobs") + 1] == "101,102" and "--allocations" not in command
    assert command[-1] == "--format=JobIDRaw,State,ExitCode,ElapsedRaw,TotalCPU,AllocTRES,MaxRSS,CPUTimeRAW"
    assert options["timeout"] == 30
    assert result["status"] == "RECORDED" and result["monetary_cost"] is None
    assert result["all_allocations_terminal"] is True and result["nonterminal_allocation_job_ids"] == []
    assert len(result["records"]) == 3 and result["records"][1]["record_kind"] == "step"
    assert result["records"][0]["allocated_cpu_seconds"] == 480
    assert result["records"][0]["MaxRSS"] == "" and "MaxRSS" in result["records"][0]["missing_fields"]
    assert "total_cpu_seconds" not in result
    assert json.loads((base / "state/scheduler-accounting.json").read_text()) == result
    assert len(list((base / "state").glob("scheduler-accounting-*.json"))) == 1
    assert not (base / "audit").exists()
    assert "Scheduler accounting: RECORDED" in capsys.readouterr().out


@pytest.mark.parametrize("stdout", ["999|COMPLETED|0:0|1|1|cpu=1||1\n", "101|too|few\n",
    "101|COMPLETED|0:0|1|1|cpu=1||1\n101|COMPLETED|0:0|1|1|cpu=1||1\n",
    "101|COMPLETED|0:0|wrong|1|cpu=1||1\n"])
def test_accounting_malformed_or_unowned_records_never_become_usage_totals(controller, stdout):
    workflow, _ = accounting_workflow(controller)
    result = controller.scheduler_accounting(workflow,
        runner=lambda *a, **kw: SimpleNamespace(returncode=0, stderr="", stdout=stdout))
    assert result["status"] == "UNAVAILABLE" and result["records"] == [] and result["reason"]


def test_unavailable_accounting_is_explicit_and_does_not_block_archive(controller, monkeypatch):
    workflow, base = accounting_workflow(controller)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("sacct unavailable")
    monkeypatch.setattr(controller.subprocess, "run", unavailable)
    archive = controller.collect(workflow)
    assert archive.is_file()
    result = json.loads((base / "state/scheduler-accounting.json").read_text())
    assert result["status"] == "UNAVAILABLE" and "sacct unavailable" in result["reason"]
    with tarfile.open(archive) as stream:
        assert any(name.endswith("state/scheduler-accounting.json") for name in stream.getnames())


def test_accounting_query_can_be_explicitly_disabled(controller):
    workflow, _ = accounting_workflow(controller)
    result = controller.scheduler_accounting(workflow, enabled=False,
        runner=lambda *a, **kw: pytest.fail("disabled accounting queried scheduler"))
    assert result["status"] == "DISABLED" and result["command"] is None
    assert controller.parser().parse_args(["collect", "latest", "--no-accounting"]).no_accounting


def test_report_time_accounting_labels_running_allocations_as_partial(controller):
    workflow, _ = accounting_workflow(controller)
    result = controller.scheduler_accounting(workflow, runner=lambda *a, **kw: SimpleNamespace(
        returncode=0, stderr="", stdout=(
            "101|COMPLETED|0:0|60|00:02:00|cpu=8,mem=48G||480\n"
            "102|RUNNING|0:0|3|00:00:01|cpu=4,mem=32G||12\n")))
    assert result["status"] == "RECORDED"
    assert result["all_allocations_terminal"] is False
    assert result["nonterminal_allocation_job_ids"] == ["102"]
    assert result["records"][1]["terminal_state"] is False


def test_validation_reports_all_missing_stages_without_claiming_success(controller, capsys):
    workflow = plan(controller)
    freeze(controller, workflow)
    with pytest.raises(ValueError, match="missing or invalid"):
        controller.validate_results(workflow)
    result = json.loads(capsys.readouterr().out)
    assert set(result["stages"]) == set(controller.physical_stages(workflow))
    assert all(row == {"validation": "MISSING", "scientific_outcome": "NA"} for row in result["stages"].values())


def test_paths_advertises_complete_analytical_outputs(controller, capsys):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (base / "report/figures").mkdir(parents=True)
    for filename in ("frontier-atlas.pdf", "frontier-overview.png", "index.html", "chart-data.json.gz", "manifest.json"):
        (base / "report/figures" / filename).write_bytes(b"fixture")
    (base / "report/analysis.json").write_text("{}")
    controller.paths(workflow)
    output = capsys.readouterr().out
    assert "TDN_FRONTIER_FIGURES_FRONTIER_ATLAS_PDF=" in output
    assert "TDN_FRONTIER_FIGURES_FRONTIER_OVERVIEW_PNG=" in output
    assert "TDN_FRONTIER_FIGURES_INDEX_HTML=" in output
    assert "TDN_FRONTIER_ANALYSIS_JSON=" in output
    assert all("/" not in line.split("=", 1)[0] for line in output.splitlines())


def gpu_junit(path, names, *, outcome=None, module="tests.test_frontier_gpu"):
    tree = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(tree, "testsuite", tests=str(len(names)), failures="0", errors="0", skipped="0")
    for index, name in enumerate(names):
        case = ElementTree.SubElement(suite, "testcase", classname=module, name=name)
        if index == 0 and outcome:
            ElementTree.SubElement(case, outcome, message="synthetic readiness rejection")
    ElementTree.ElementTree(tree).write(path)


def test_gpu_validator_requires_exact_declared_testcase_identities(controller):
    from tdn.analysis.frontier.protocol import GPU_TEST_CASES
    path = controller.ROOT / "junit.xml"
    gpu_junit(path, GPU_TEST_CASES)
    result = controller.check_gpu_tests(path)
    assert result["test_cases"] == len(GPU_TEST_CASES)
    for names, module in ((GPU_TEST_CASES[:-1], "tests.test_frontier_gpu"),
                          (GPU_TEST_CASES + (GPU_TEST_CASES[0],), "tests.test_frontier_gpu"),
                          (GPU_TEST_CASES, "tests.unrelated"),
                          (GPU_TEST_CASES[:-1] + ("invented_case",), "tests.test_frontier_gpu")):
        gpu_junit(path, names, module=module)
        with pytest.raises(ValueError):
            controller.check_gpu_tests(path)


@pytest.mark.parametrize("outcome", ["skipped", "error", "failure"])
def test_gpu_validator_never_accepts_skipped_failed_or_errored_evidence(controller, outcome):
    from tdn.analysis.frontier.protocol import GPU_TEST_CASES
    path = controller.ROOT / "junit.xml"
    gpu_junit(path, GPU_TEST_CASES, outcome=outcome)
    with pytest.raises(ValueError):
        controller.check_gpu_tests(path)


@pytest.mark.parametrize("filename", ["fedora_frontier.sh", "frontier_worker.sh", "frontier_local.sh"])
def test_shell_syntax(filename):
    response = subprocess.run(["bash", "-n", str(ROOT / "scripts" / filename)], capture_output=True, text=True)
    assert response.returncode == 0, response.stderr


@pytest.mark.parametrize("profile,count,parents_per_part", [("smoke", 2, 1), ("development", 4, 2), ("full", 6, 4)])
def test_confirmation_partition_plan_preserves_exact_cohort_and_gpu_budget(controller, profile, count, parents_per_part):
    workflow = plan(controller, "parts", "--" + profile)
    protocol = controller.scientific_protocol(profile)
    parts = controller.confirmation_shards(profile)
    assert len(parts) == count
    assert all(len(part["parent_ids"]) == parents_per_part for part in parts)
    actual = [parent for part in parts for parent in part["parent_ids"]]
    assert actual == [parent["parent_id"] for parent in protocol["parents"] if parent["split"] == "confirmation"]
    assert len(actual) == len(set(actual))
    for part in parts:
        resource = controller.resource_for(workflow, part["shard_id"])
        assert resource == workflow["resources"]["confirm"]
        assert controller.wall_seconds(resource["walltime"]) <= 2700
        commands = dict(controller.worker_commands(workflow, part["shard_id"]))
        assert set(commands) == {"preflight", "gpu-tests", "check-gpu-tests", "experiment"}
        command = commands["experiment"]
        assert command[command.index("--stage") + 1] == "confirm"
        assert command[command.index("--confirm-shard") + 1] == part["shard_id"]
        assert command.count("--prerequisite-dir") == 5
        assert "--confirm-shard-dir" not in command
        assert command[command.index("--device") + 1] == "cuda"
        assert "--gres=gpu:1" in controller.scheduler_args(workflow, part["shard_id"])
    command = dict(controller.worker_commands(workflow, "confirm"))["experiment"]
    assert command.count("--confirm-shard-dir") == count
    assert "--confirm-shard" not in command
    assert workflow["protocol_sha256"] == controller.pw.canonical_hash(controller.declaration(profile, 2))


def test_unknown_confirmation_part_cannot_be_submitted_or_executed(controller):
    workflow = plan(controller)
    for method in (controller.scheduler_args, controller.resource_for, controller.stage_path):
        with pytest.raises(ValueError, match="Unknown frontier"):
            method(workflow, "confirm-part-999")


def test_v1_manifest_remains_readable_without_rewriting_protocol(controller):
    workflow = plan(controller)
    workflow.pop("execution_version")
    workflow["protocol_sha256"] = controller.pw.canonical_hash(controller.declaration(workflow["profile"], 1))
    base = freeze(controller, workflow)
    controller.cw.atomic_json(base / "protocol.json", controller.declaration(workflow["profile"], 1))
    assert controller.load(base / controller.MANIFEST) == workflow
    assert controller.physical_stages(workflow) == STAGES
    command = dict(controller.worker_commands(workflow, "confirm"))["experiment"]
    assert "--confirm-shard-dir" not in command and "--confirm-shard" not in command


def test_part_submission_failure_still_queues_one_afterany_report(controller, monkeypatch):
    calls = submission(controller, monkeypatch, fail_at=7)
    assert controller.main(["run", "--run-id", "part-failure"]) == 2
    base = controller.ROOT / "runs/part-failure"
    jobs = json.loads((base / "jobs.json").read_text())
    assert [job["stage"] for job in jobs] == list(STAGES[:5]) + ["confirm-part-000", "report"]
    assert jobs[-1]["dependency"] == "afterany:9001:9002:9003:9004:9005:9006"
    assert sum(command[0] == "sbatch" for command, _ in calls) == 8
    assert not any(command[0] == "scancel" for command, _ in calls)


def add_recovery_bridge(controller, workflow, stages):
    base = Path(workflow["run_dir"])
    bridge = {"stage_paths": {stage: str(base.parent / "origin" / stage) for stage in stages},
        "target_software": {"fixture": True}, "origin_workflow_path": str(base.parent / "origin" / controller.MANIFEST)}
    controller.cw.atomic_json(base / "recovery.json", bridge)
    workflow["recovery"] = {"manifest_path": str(base / "recovery.json"),
        "sha256": controller.cw.digest(base / "recovery.json"), "stage_paths": bridge["stage_paths"]}
    return bridge


def test_recovery_schedules_only_missing_parts_and_uses_origin_stage_paths(controller):
    workflow = plan(controller)
    freeze(controller, workflow)
    inherited = list(STAGES[:5]) + ["confirm-part-000", "confirm-part-001"]
    bridge = add_recovery_bridge(controller, workflow, inherited)
    controller.validate(workflow)
    pending = controller.pending_stages(workflow)
    assert pending == ("confirm-part-002", "confirm-part-003", "confirm-part-004", "confirm-part-005", "confirm", "scaling", "policy", "report")
    assert controller.stage_path(workflow, "train") == Path(bridge["stage_paths"]["train"])
    command = dict(controller.worker_commands(workflow, "confirm-part-002"))["experiment"]
    assert command[command.index("--recovery-manifest") + 1] == workflow["recovery"]["manifest_path"]
    assert "train=" + bridge["stage_paths"]["train"] in command
    aggregate = dict(controller.worker_commands(workflow, "confirm"))["experiment"]
    assert "confirm-part-000=" + bridge["stage_paths"]["confirm-part-000"] in aggregate
    assert not any(flag.startswith("--dependency") for flag in controller.scheduler_args(workflow, pending[0]))
    assert "--dependency=afterok:123" in controller.scheduler_args(workflow, pending[1], "afterok:123")
    (Path(workflow["run_dir"]) / "recovery.json").write_text("{}")
    with pytest.raises(ValueError, match="bridge changed"):
        controller.stage_path(workflow, "train")


def test_recovery_mapping_cannot_inherit_unsealed_logical_confirmation(controller):
    workflow = plan(controller)
    freeze(controller, workflow)
    add_recovery_bridge(controller, workflow, list(STAGES[:5]) + ["confirm"])
    with pytest.raises(ValueError, match="five sealed prerequisites"):
        controller.validate(workflow)


def test_accounting_distinguishes_confirmation_parts_and_aggregate(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    add_recovery_bridge(controller, workflow, STAGES[:5])
    controller.cw.atomic_json(base / "jobs.json", [
        {"stage": "confirm-part-000", "job_id": "101"}, {"stage": "confirm", "job_id": "102"}])
    result = controller.scheduler_accounting(workflow, runner=lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stderr="", stdout="101|COMPLETED|0:0|60|00:00:40|cpu=4,mem=48G,gres/gpu=1||240\n"
        "102|COMPLETED|0:0|2|00:00:01|cpu=4,mem=48G,gres/gpu=1||8\n"))
    assert result["requested_job_ids"] == ["101", "102"]
    assert [row["allocation_role"] for row in result["records"]] == ["confirmation_part", "confirmation_aggregate"]
    assert [row["logical_stage"] for row in result["records"]] == ["confirm", "confirm"]
    assert set(result["inherited_stage_paths"]) == set(STAGES[:5])


def test_recover_command_submits_finite_remaining_dag_without_retraining(controller, monkeypatch):
    import tdn.analysis.frontier.recovery as recovery
    origin = plan(controller, "origin")
    origin.pop("execution_version")
    origin["protocol_sha256"] = controller.pw.canonical_hash(controller.declaration(origin["profile"], 1))
    base = freeze(controller, origin)
    controller.cw.atomic_json(base / "protocol.json", controller.declaration(origin["profile"], 1))
    original_bytes = (base / controller.MANIFEST).read_bytes()
    controller.cw.atomic_json(base / "jobs.json", [{"stage": "audit", "job_id": "100"}])
    monkeypatch.setattr(controller.cw, "scheduler_state", lambda job: "COMPLETED")
    bridge = {"stage_paths": {stage: str(base / stage) for stage in STAGES[:5]},
        "target_software": {"fixture": True}, "origin_workflow_path": str(base / controller.MANIFEST)}
    calls = submission(controller, monkeypatch)
    monkeypatch.setattr(controller, "execution_software", lambda: {"fixture": True})
    monkeypatch.setattr(recovery, "create_recovery", lambda *args, **kwargs: bridge)
    assert controller.main(["recover", "origin", "--run-id", "fresh"]) == 0
    fresh = controller.load(controller.workflow_path("fresh"))
    assert controller.recovery_bridge(fresh) == bridge
    jobs = controller.cw.read_json(Path(fresh["run_dir"]) / "jobs.json")
    assert [row["stage"] for row in jobs] == [f"confirm-part-{index:03d}" for index in range(6)] + list(STAGES[5:])
    assert len(jobs) == 10 and jobs[0]["dependency"] is None
    assert jobs[-1]["dependency"] == "afterany:" + ":".join(row["job_id"] for row in jobs[:-1])
    assert (base / controller.MANIFEST).read_bytes() == original_bytes
    assert len([args for args, _ in calls if args[0] == "sbatch"]) == 10
    assert not any((Path(fresh["run_dir"]) / stage).exists() for stage in STAGES[:5])


def test_incompatible_recovery_does_not_write_or_submit(controller, monkeypatch):
    import tdn.analysis.frontier.recovery as recovery
    origin = plan(controller, "origin")
    base = freeze(controller, origin)
    controller.cw.atomic_json(base / "jobs.json", [{"stage": "audit", "job_id": "100"}])
    monkeypatch.setattr(controller.cw, "scheduler_state", lambda job: "COMPLETED")
    calls = submission(controller, monkeypatch)
    monkeypatch.setattr(controller, "execution_software", lambda: {"fixture": True})
    def incompatible(*args, **kwargs):
        raise ValueError("checkpoint source differs")
    monkeypatch.setattr(recovery, "create_recovery", incompatible)
    assert controller.main(["recover", "origin", "--run-id", "blocked"]) == 2
    assert not (controller.ROOT / "runs/blocked").exists()
    assert not any(args[0] == "sbatch" for args, _ in calls)


def test_recovery_collection_keeps_origin_evidence_and_failed_attempt(controller, monkeypatch):
    workflow = plan(controller, "fresh")
    base = freeze(controller, workflow)
    bridge = add_recovery_bridge(controller, workflow, STAGES[:5])
    origin = controller.ROOT / "runs/origin"
    (origin / "confirm").mkdir(parents=True)
    (origin / "confirm/summary.json").write_text('{"status":"INTERRUPTED"}\n')
    (origin / controller.MANIFEST).write_text('{}\n')
    (origin / "train").mkdir()
    (origin / "train/checkpoint.pt").write_bytes(b"unchanged evidence")
    monkeypatch.setattr(controller, "verify_recovery_bridge", lambda *args, **kwargs: bridge["stage_paths"])
    archive = controller.collect(workflow, accounting=False)
    with tarfile.open(archive) as stream:
        names = stream.getnames()
        assert "origin/confirm/summary.json" in names
        assert "origin/train/checkpoint.pt" in names
        assert "fresh/recovery.json" in names
        assert not any(name.startswith("fresh/train/") for name in names)
    index = json.loads(archive.with_name(archive.name + ".index.json").read_text())
    assert index["included_workflow_directories"] == ["fresh", "origin"]


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "UNKNOWN"])
def test_recovery_refuses_live_or_unverified_origin_jobs(controller, monkeypatch, state):
    workflow = plan(controller, "origin")
    base = freeze(controller, workflow)
    controller.cw.atomic_json(base / "jobs.json", [{"stage": "confirm-part-000", "job_id": "100"}])
    monkeypatch.setattr(controller.cw, "scheduler_state", lambda job: state)
    calls = submission(controller, monkeypatch)
    assert controller.main(["recover", "origin", "--run-id", "unsafe"]) == 2
    assert not (controller.ROOT / "runs/unsafe").exists()
    assert not any(args[0] == "sbatch" for args, _ in calls)


def test_recovery_requires_every_origin_allocation_terminal(controller, monkeypatch):
    workflow = plan(controller, "origin")
    base = freeze(controller, workflow)
    jobs = [{"stage": "confirm-part-000", "job_id": "100"}, {"stage": "report", "job_id": "101"}]
    controller.cw.atomic_json(base / "jobs.json", jobs)
    seen = []
    def state(job):
        seen.append(job)
        return {"100": "FAILED", "101": "COMPLETED"}[job]
    monkeypatch.setattr(controller.cw, "scheduler_state", state)
    assert controller.require_terminal_origin(workflow) == {"100": "FAILED", "101": "COMPLETED"}
    assert seen == ["100", "101"]



def test_unknown_worker_identity_cannot_write_startup_state(controller, monkeypatch):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setattr(controller.pw, "worker", lambda *args, **kwargs: pytest.fail("unknown worker started"))
    with pytest.raises(ValueError, match="Unknown or inherited"):
        controller.worker(workflow, "../../elsewhere")
    assert not (base / "state/startup.json").exists()
    assert not (base.parent / "elsewhere.json").exists()
