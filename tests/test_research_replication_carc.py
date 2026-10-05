"""Replication launcher/gate contracts with fake scheduling and sealed fixtures."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tdn.research.protocol import digest
from tdn.research.replication_protocol import FAMILIES, replicates

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("replication_carc_controller", ROOT / "scripts" / "research_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(f"{name} source\n")
    (root / "tests" / "test_research_replication_example.py").write_text("def test_ok(): pass\n")
    shutil.copyfile(ROOT / "configs" / "neural-replication.yaml", root / "configs" / "neural-replication.yaml")
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


def plan(module, run_id="replication-cpu"):
    return module.prepare(module.parser().parse_args([
        "start", "--neural-replication", "--config", str(module.ROOT / "configs" / "neural-replication.yaml"),
        "--run-id", run_id]))


def finish_cpu(module, workflow, *, selected=None, omit_checkpoint=None, changed_context=None):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "config.yaml").write_bytes(Path(workflow["original_config"]).read_bytes())
    module.cw.atomic_json(base / "research-workflow.json", workflow)
    module.cw.atomic_json(base / "jobs.json", [])
    experiment = base / "experiment"
    experiment.mkdir()
    (experiment / "dataset.pt").write_bytes(b"sealed dataset fixture, not a tensor")
    protocol = {"version": 2, "benchmark_suite": "neural-replication",
                "config": {"protocol_version": 2, "families": list(FAMILIES)}, "replicates": replicates()}
    if changed_context:
        protocol.update(changed_context)
    module.cw.atomic_json(experiment / "protocol.json", protocol)
    for name in ("replication.json", "normalization.json", "references.json"):
        module.cw.atomic_json(experiment / name, {"fixture": "sealed controller fixture"})
    training = []
    for replicate in replicates():
        for family in FAMILIES:
            key = (replicate["replicate_id"], family)
            path = experiment / "replicates" / replicate["replicate_id"] / "checkpoints" / f"{family}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if key != omit_checkpoint:
                path.write_bytes(f"{key} frozen checkpoint fixture".encode())
            training.append({**replicate, "family": family, "status": "COMPLETED", "selection_split": "validation",
                             "selected_step": (8 if selected is None or key in selected else 0),
                             "selected_parameters_changed": selected is None or key in selected,
                             "checkpoint_sha256": module.cw.digest(path) if path.exists() else "missing-checkpoint"})
    module.cw.atomic_json(experiment / "summary.json", {"status": "COMPLETED", "device": "cpu",
                         "benchmark_suite": "neural-replication", "training": training, "headroom": {"passed": False}})
    manifest = {"version": 2, "source_tree_sha256": workflow["source_tree_sha256"],
                "config_file_sha256": workflow["config_sha256"], "protocol_sha256": digest(protocol),
                "files": {str(path.relative_to(experiment)): module.cw.digest(path)
                          for path in experiment.rglob("*") if path.is_file()}}
    module.cw.atomic_json(experiment / "manifest.json", manifest)
    (experiment / "COMPLETED").write_text(module.cw.digest(experiment / "manifest.json") + "\n")
    module.cw.atomic_json(module.software_path(workflow), {"packages": {"torch": workflow["torch_version"]}})
    module.cw.atomic_json(module.state_path(workflow), {"status": "COMPLETED", "exit_code": 0, "stage": "experiment",
                         "source_sha256": workflow["source_sha256"], "config_sha256": workflow["config_sha256"],
                         "software_sha256": module.cw.digest(module.software_path(workflow))})
    return base


def test_replication_preview_and_worker_preserve_bounded_suite(controller, monkeypatch, capsys):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("Preview contacted scheduler"))
    assert controller.main(["start", "--neural-replication", "--config", str(controller.ROOT / "configs" / "neural-replication.yaml"),
                            "--run-id", "preview"]) == 0
    output = capsys.readouterr().out
    assert "Suite: neural-replication" in output
    assert output.count("sbatch --parsable") == 1
    for item in ("--account=anakano_81", "--cpus-per-task=4", "--mem=16G", "--time=00:30:00"):
        assert item in output
    assert "--gpus-per-task" not in output and "--dependency" not in output and "--requeue" not in output
    assert not (controller.ROOT / "runs").exists()
    commands = controller.worker_commands(plan(controller))
    assert "test_research_replication_example.py" in " ".join(commands[0][1])
    args = commands[1][1]
    assert args[args.index("--expected-suite") + 1] == "neural-replication"
    assert args[args.index("--device") + 1] == "cpu"


def test_replication_submission_is_exactly_one_cpu_job(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller.cw, "live_check", lambda *a: {"account": "anakano_81"})
    def command(args, *, env=None, **kwargs):
        calls.append(args)
        assert args[0] == "sbatch" and env["TDN_RESEARCH_PHASE"] == "cpu"
        assert Path(env["TDN_RESEARCH_WORKFLOW"]).is_file()
        return SimpleNamespace(stdout="12699999;cluster\n")
    monkeypatch.setattr(controller.cw, "command", command)
    assert controller.main(["start", "--neural-replication", "--config", str(controller.ROOT / "configs" / "neural-replication.yaml"),
                            "--run-id", "submitted", "--submit"]) == 0
    assert len(calls) == 1
    base = controller.ROOT / "runs" / "submitted"
    assert controller.cw.read_json(base / "research-workflow.json")["benchmark_suite"] == "neural-replication"
    assert controller.cw.read_json(base / "jobs.json") == [{"job_id": "12699999", "phase": "cpu",
                                                         "submitted_at": controller.cw.read_json(base / "jobs.json")[0]["submitted_at"]}]


def test_allocated_tower_report_identifies_replication_suite(controller, monkeypatch):
    workflow = plan(controller)
    monkeypatch.setenv("SLURM_JOB_ID", "12699999")
    observed = {}
    def begin_report(base, **kwargs):
        observed.update(kwargs)
        return Path(base) / "tower" / "fake-attempt"
    monkeypatch.setattr(controller.cw, "reporting_api", lambda: SimpleNamespace(begin_report=begin_report))
    controller.cw.begin_workflow_report(workflow, "cpu", research=True)
    assert observed["name"] == "TDN/research/neural-replication/cpu"
    assert observed["parameters"]["benchmark_suite"] == "neural-replication"
    assert observed["resources"]["time_seconds"] == 1800 and observed["resources"]["gpus"] == 0
    assert observed["resources"]["account"] == "anakano_81"


def test_gpu_inherits_suite_and_frozen_source_without_classical_headroom(controller, monkeypatch):
    cpu = plan(controller)
    finish_cpu(controller, cpu, selected={("seed-74021", "reaction_clock"), ("seed-74021", "fno_split")})
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("GPU preview contacted scheduler"))
    gpu = controller.prepare(controller.parser().parse_args(["benchmark", cpu["run_id"], "--run-id", "replication-gpu"]))
    assert gpu["benchmark_suite"] == "neural-replication" and gpu["config_sha256"] == cpu["config_sha256"]
    request = controller.scheduler_args(gpu)
    assert "--time=00:30:00" in request and "--gpus-per-task=a100:1" in request and "--constraint=a100-40gb" in request
    assert not any(arg.startswith("--dependency") or arg == "--requeue" for arg in request)
    commands = controller.worker_commands(gpu)
    args = commands[-1][1]
    assert "benchmark" in args and "run" not in args
    assert args[args.index("--source-run") + 1] == str(Path(cpu["run_dir"]) / "experiment")
    assert args[args.index("--expected-suite") + 1] == "neural-replication"


@pytest.mark.parametrize("selected", [set(), {("seed-74011", "reaction_clock"), ("seed-74021", "fno_split")}])
def test_initialization_and_cross_seed_union_do_not_authorize_gpu(controller, monkeypatch, selected):
    workflow = plan(controller)
    finish_cpu(controller, workflow, selected=selected)
    monkeypatch.setattr(controller.cw, "live_check", lambda *a: pytest.fail("Ineligible run checked GPU allocation"))
    assert controller.main(["benchmark", workflow["run_id"], "--submit"]) == 2
    assert len(list((controller.ROOT / "runs").iterdir())) == 1


def test_eligible_checkpoint_must_be_in_sealed_manifest(controller):
    workflow = plan(controller)
    finish_cpu(controller, workflow, omit_checkpoint=("seed-74011", "reaction_clock"))
    with pytest.raises(ValueError, match="selected checkpoint.*sealed"):
        controller.completed_source(controller.workflow_path(workflow["run_id"]))


@pytest.mark.parametrize("damage", ["suite", "seed", "dataset", "config", "replication", "version"])
def test_replication_source_and_suite_integrity(controller, damage):
    workflow = plan(controller)
    changed_context = ({"benchmark_suite": "neural-benchmarks"} if damage == "suite" else
                       {"replicates": [{"replicate_id": "seed-74011", "training_seed": 999, "sample_schedule_seed": 74012}]}
                       if damage == "seed" else None)
    base = finish_cpu(controller, workflow, changed_context=changed_context)
    if damage == "dataset":
        (base / "experiment" / "dataset.pt").write_bytes(b"tampered")
    if damage == "config":
        (base / "config.yaml").write_text("tampered\n")
    if damage == "replication":
        (base / "experiment" / "replication.json").unlink()
    if damage == "version":
        manifest_path = base / "experiment" / "manifest.json"
        manifest = controller.cw.read_json(manifest_path)
        manifest["version"] = 1
        controller.cw.atomic_json(manifest_path, manifest)
        (base / "experiment" / "COMPLETED").write_text(controller.cw.digest(manifest_path))
    assert controller.main(["benchmark", workflow["run_id"]]) == 2
    assert len(list((controller.ROOT / "runs").iterdir())) == 1


def test_conflicting_suite_flags_fail_at_parse_time(controller):
    with pytest.raises(SystemExit):
        controller.parser().parse_args(["start", "--neural-benchmarks", "--neural-replication"])


@pytest.fixture
def wrappers(tmp_path):
    if os.name == "nt":
        pytest.skip("Bash wrappers require a POSIX host")
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("carc_neural_replication.sh", "neural_replication_local.sh"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    for name in ("carc_research.sh", "research_local.sh"):
        (scripts / name).write_text("#!/usr/bin/env bash\nexec " + str(sys.executable) +
                                  " -c 'import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(37)' \"$@\"\n")
    return root


def forwarded(root, name, *args):
    result = subprocess.run(["bash", str(root / "scripts" / name), *args], cwd=root,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 37, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("args", [(), ("--submit",), ("start", "--smoke", "--run-id", "quoted run")])
def test_carc_replication_wrapper_preserves_quoting_and_defaults(wrappers, args):
    expected = list(args[1:] if args and args[0] == "start" else args)
    assert forwarded(wrappers, "carc_neural_replication.sh", *args) == [
        "start", "--neural-replication", "--config", str(wrappers / "scripts" / ".." / "configs" / "neural-replication.yaml"),
        *expected]


@pytest.mark.parametrize("args", [("status", "latest"), ("logs", "run-123", "--lines", "200"),
                                  ("collect", "run-123"), ("benchmark", "cpu-123", "--submit")])
def test_replication_wrapper_readonly_and_benchmark_passthrough(wrappers, args):
    assert forwarded(wrappers, "carc_neural_replication.sh", *args) == list(args)


def test_replication_explicit_config_and_marker_are_preserved(wrappers):
    args = ("--neural-replication", "--config=configs/custom with spaces.yaml", "--submit")
    assert forwarded(wrappers, "carc_neural_replication.sh", *args) == ["start", *args]
    assert forwarded(wrappers, "neural_replication_local.sh", "--config", "configs/custom with spaces.yaml") == [
        "--config", "configs/custom with spaces.yaml"]


def test_local_replication_wrapper_uses_existing_runtime_checks(wrappers):
    assert forwarded(wrappers, "neural_replication_local.sh", "--run-dir", "runs/with spaces", "--smoke") == [
        "--config", str(wrappers / "scripts" / ".." / "configs" / "neural-replication.yaml"),
        "--run-dir", "runs/with spaces", "--smoke"]


@pytest.mark.parametrize("name", ["carc_neural_replication.sh", "neural_replication_local.sh"])
def test_replication_wrapper_help_and_syntax(wrappers, name):
    result = subprocess.run(["bash", str(wrappers / "scripts" / name), "--help"], cwd=wrappers,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0 and "neural-replication.yaml" in result.stdout
    result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
