"""CPU work-precision workflow contracts; fake Slurm is not CARC execution."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shlex
import shutil
import venv
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("protocol.json", "config.json", "summary.json", "work-precision.json", "summary.txt")


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("work_precision_controller", ROOT / "scripts" / "research_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(f"{name} source\n")
    for name in ("test_work_precision_numerics.py", "test_work_precision_workflow.py", "test_work_precision_cli.py",
                 "test_interaction_carc.py", "test_mechanism_example.py", "test_research_example.py"):
        (root / "tests" / name).write_text("def test_ok(): pass\n")
    (root / "configs" / "work-precision.yaml").write_text("benchmark_suite: work-precision\n")
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


def plan(module, run_id="work-precision-cpu", *extra):
    return module.prepare(module.parser().parse_args(["start", "--work-precision", "--run-id", run_id, *extra]))


def finish_cpu(module, workflow):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "config.yaml").write_bytes(Path(workflow["original_config"]).read_bytes())
    module.cw.atomic_json(base / "research-workflow.json", workflow)
    module.cw.atomic_json(base / "jobs.json", [])
    experiment = base / "experiment"
    experiment.mkdir()
    protocol = {"version": 1, "benchmark_suite": "work-precision", "config": {"fixed": True}}
    module.cw.atomic_json(experiment / "protocol.json", protocol)
    module.cw.atomic_json(experiment / "summary.json", {"status": "COMPLETED", "device": "cpu",
        "benchmark_suite": "work-precision", "training_attempted": False, "training_performed": False,
        "scientific_outcome": "INCONCLUSIVE"})
    for name in ("config.json", "work-precision.json"):
        module.cw.atomic_json(experiment / name, {"fixture": "sealed controller fixture"})
    (experiment / "summary.txt").write_text("Computational status: COMPLETED; scientific outcome: INCONCLUSIVE\n")
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


def reseal(module, base, *, protocol_update=None, summary_update=None, manifest_update=None):
    experiment = base / "experiment"
    manifest_path = experiment / "manifest.json"
    manifest = module.cw.read_json(manifest_path)
    for name, update in (("protocol.json", protocol_update), ("summary.json", summary_update)):
        if update is not None:
            value = module.cw.read_json(experiment / name)
            value.update(update)
            module.cw.atomic_json(experiment / name, value)
            manifest["files"][name] = module.cw.digest(experiment / name)
            if name == "protocol.json":
                manifest["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                                       allow_nan=False).encode()).hexdigest()
    if manifest_update:
        manifest.update(manifest_update)
    module.cw.atomic_json(manifest_path, manifest)
    (experiment / "COMPLETED").write_text(module.cw.digest(manifest_path) + "\n")


@pytest.mark.parametrize("options", [[], ["--dry-run"], ["--submit", "--dry-run"]])
def test_cpu_preview_no_writes_or_scheduler_calls(controller, monkeypatch, capsys, options):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("Preview contacted scheduler"))
    monkeypatch.setattr(controller.cw, "actual_policy", lambda **kw: pytest.fail("Preview required allocation"))
    assert controller.main(["start", "--work-precision", *options]) == 0
    output = capsys.readouterr().out
    assert "carc-work-precision-" in output and "Suite: work-precision" in output
    assert "no neural training or GPU successor" in output and "1,200-second numerical cap" in output
    assert output.count("sbatch --parsable") == 1
    for item in ("--account=anakano_81", "--cpus-per-task=4", "--mem=16G", "--time=00:30:00"):
        assert item in output
    for forbidden in ("--gpus-per-task", "--dependency", "--requeue", "--deadline", "--max-pending"):
        assert forbidden not in output
    assert not (controller.ROOT / "runs").exists() and not (controller.ROOT / ".cache").exists()


def test_login_controller_does_not_import_numerical_packages(controller):
    code = r'''
import importlib.abc, importlib.util, pathlib, sys
class RejectNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy','yaml','tdn'}:
            raise RuntimeError('Numerical import on login: '+fullname)
sys.meta_path.insert(0, RejectNumerics())
spec=importlib.util.spec_from_file_location('work_precision_preview',sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.ROOT=module.cw.ROOT=pathlib.Path(sys.argv[2])
raise SystemExit(module.main(['start','--work-precision']))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "scripts" / "research_workflow.py"),
                            str(controller.ROOT)], cwd=controller.ROOT,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, text=True,
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
    assert controller.main(["start", "--work-precision", "--run-id", "submitted", "--submit"]) == 0
    assert len(calls) == 1
    base = controller.ROOT / "runs" / "submitted"
    workflow = controller.load(base / "research-workflow.json")
    assert workflow["benchmark_suite"] == "work-precision"
    assert workflow["config_sha256"] == controller.cw.digest(base / "config.yaml")
    assert not (base / "config.yaml").stat().st_mode & 0o222
    assert not (base / "research-workflow.json").stat().st_mode & 0o222
    assert controller.cw.read_json(base / "jobs.json")[0]["job_id"] == "12699999"


def test_worker_runs_all_work_precision_tests_and_training_free_cli(controller):
    workflow = plan(controller, "commands", "--smoke")
    commands = controller.worker_commands(workflow)
    assert [name for name, _ in commands] == ["tests", "experiment"]
    tests, engine = [args for _, args in commands]
    assert [Path(arg).name for arg in tests if arg.endswith(".py")] == [
        "test_work_precision_cli.py", "test_work_precision_numerics.py", "test_work_precision_workflow.py"]
    assert tests[tests.index("-m", 3) + 1] == "not gpu"
    assert engine == [str(controller.ROOT / ".venv" / "bin" / "python"),
                      str(controller.ROOT / "scripts" / "work_precision.py"),
                      "--config", workflow["config_path"], "--run-dir",
                      str(Path(workflow["run_dir"]) / "experiment"), "--smoke"]
    assert "--junitxml" in tests and "--basetemp" in tests


def test_worker_refuses_empty_test_selection(controller):
    workflow = plan(controller)
    for path in (controller.ROOT / "tests").glob("test_work_precision_*.py"):
        path.unlink()
    with pytest.raises(ValueError, match="Focused work_precision tests are missing"):
        controller.worker_commands(workflow)


def test_tower_identifies_cpu_suite_resources(controller, monkeypatch):
    workflow = plan(controller)
    monkeypatch.setenv("SLURM_JOB_ID", "12699999")
    observed = {}
    def begin_report(base, **kwargs):
        observed.update(kwargs)
        return Path(base) / "tower" / "fake-attempt"
    monkeypatch.setattr(controller.cw, "reporting_api", lambda: SimpleNamespace(begin_report=begin_report))
    controller.cw.begin_workflow_report(workflow, "cpu", research=True)
    assert observed["name"] == "TDN/research/work-precision/cpu"
    assert observed["parameters"]["benchmark_suite"] == "work-precision"
    assert observed["resources"]["gpus"] == 0 and observed["resources"]["time_seconds"] == 1800
    assert observed["resources"]["account"] == "anakano_81"


def test_allocation_ownership_checked_before_attempt_evidence(controller, monkeypatch):
    workflow = plan(controller)
    monkeypatch.setattr(controller.cw, "actual_policy", lambda **kw: None)
    monkeypatch.setenv("SLURM_JOB_ID", "12699999")
    monkeypatch.setattr(controller.cw, "command", lambda *a: SimpleNamespace(stdout=
        "JobId=12699999 JobName=tdn-research-cpu Account=another_account UserId=aadaniel(123) "
        "Comment=tdn-research:work-precision-cpu:cpu JobState=RUNNING"))
    with pytest.raises(ValueError, match="not this research workflow"):
        controller.worker(workflow, "cpu")
    assert not Path(workflow["run_dir"]).exists()


def test_artifacts_verify_without_training_even_when_scientifically_inconclusive(controller):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    manifest = controller.verify_experiment(workflow)
    assert set(manifest["files"]) == set(ARTIFACTS)
    assert not (base / "experiment" / "dataset.pt").exists()
    assert not (base / "experiment" / "checkpoints").exists()


@pytest.mark.parametrize("damage", ["file", "missing", "marker", "source", "config", "manifest_version", "protocol_version",
                                   "protocol_suite", "summary_suite", "device", "training_attempted", "training_performed"])
def test_corrupt_or_mismatched_artifacts_rejected(controller, damage):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    if damage == "file":
        (base / "experiment" / "work-precision.json").write_text("{}\n")
    elif damage == "missing":
        (base / "experiment" / "summary.txt").unlink()
    elif damage == "marker":
        (base / "experiment" / "COMPLETED").write_text("incorrect\n")
    elif damage in ("source", "config", "manifest_version"):
        key = {"source": "source_tree_sha256", "config": "config_file_sha256", "manifest_version": "version"}[damage]
        reseal(controller, base, manifest_update={key: 2 if damage == "manifest_version" else "incorrect"})
    elif damage in ("protocol_version", "protocol_suite"):
        reseal(controller, base, protocol_update={"version": 2} if damage == "protocol_version" else
               {"benchmark_suite": "mechanism-audit"})
    else:
        update = {"summary_suite": {"benchmark_suite": "mechanism-audit"}, "device": {"device": "cuda"},
                  "training_attempted": {"training_attempted": True}, "training_performed": {"training_performed": True}}[damage]
        reseal(controller, base, summary_update=update)
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


def test_status_logs_collect_identify_suite_and_preserve_completed_artifacts(controller, capsys):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    (base / "logs").mkdir()
    (base / "logs" / "cpu-123.out").write_text("work-precision experiment complete\n")
    (base / "pytest-work").mkdir()
    (base / "pytest-work" / "scratch.txt").write_text("excluded\n")
    hashes = {name: controller.cw.digest(base / "experiment" / name) for name in ARTIFACTS}
    controller.status(workflow)
    output = capsys.readouterr().out
    assert "Suite: work-precision" in output and "experiment: COMPLETED" in output
    assert "scientific outcome: INCONCLUSIVE" in output and "sealed work-precision artifacts: VERIFIED" in output
    controller.logs(workflow, 10)
    output = capsys.readouterr().out
    assert "Suite: work-precision" in output and "work-precision experiment complete" in output
    archive = controller.collect(workflow)
    output = capsys.readouterr().out
    assert "Suite: work-precision" in output and "Work-precision artifact verification: VERIFIED" in output
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert any(name.endswith("/experiment/work-precision.json") for name in names)
    assert any(name.endswith("/experiment/summary.txt") for name in names)
    assert not any("pytest-work" in name for name in names)
    assert hashes == {name: controller.cw.digest(base / "experiment" / name) for name in ARTIFACTS}


def test_damaged_completion_rejected_by_status_but_collected_as_unverified(controller, capsys):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    (base / "experiment" / "work-precision.json").write_text("damaged evidence\n")
    assert controller.main(["status", workflow["run_id"]]) == 2
    assert "Research artifact changed" in capsys.readouterr().err
    archive = controller.collect(workflow)
    assert "Work-precision artifact verification: NOT VERIFIED" in capsys.readouterr().out
    with tarfile.open(archive) as stream:
        assert stream.extractfile(f"{workflow['run_id']}/experiment/work-precision.json").read() == b"damaged evidence\n"


def test_status_rejects_completed_worker_with_missing_summary(controller, capsys):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    (base / "experiment" / "summary.json").unlink()
    assert controller.main(["status", workflow["run_id"]]) == 2
    captured = capsys.readouterr()
    assert "Research artifact is missing" in captured.err
    assert "sealed work-precision artifacts: VERIFIED" not in captured.out


@pytest.mark.parametrize("flag", ["--interaction-screen", "--mechanism-audit", "--neural-benchmarks", "--neural-replication"])
def test_work_precision_selector_cannot_be_combined_with_other_suites(controller, flag):
    with pytest.raises(SystemExit):
        controller.parser().parse_args(["start", "--work-precision", flag])


def test_completed_run_cannot_be_reused(controller, monkeypatch):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    before = {path: path.read_bytes() for path in base.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="fresh --run-id"):
        plan(controller)
    with pytest.raises(ValueError, match="fresh workflow"):
        # Ownership must still be checked before the reused-run guard.
        monkeypatch.setattr(controller, "verify_allocation", lambda *args: None)
        controller.worker(workflow, "cpu")
    assert before == {path: path.read_bytes() for path in base.rglob("*") if path.is_file()}


@pytest.mark.parametrize("stage,exit_code,reporting_error", [
    ("tests", 7, False), ("tests", 75, True), ("experiment", 7, True), ("experiment", 75, False)])
def test_worker_failure_retains_code_artifacts_and_stops_successors(
        controller, monkeypatch, capsys, stage, exit_code, reporting_error):
    workflow = plan(controller)
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "config.yaml").write_bytes(Path(workflow["original_config"]).read_bytes())
    controller.cw.atomic_json(base / "research-workflow.json", workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "12699999")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("TDN_TOWER_DIR", "caller-report")
    monkeypatch.setattr(controller, "verify_allocation", lambda *args: None)
    monkeypatch.setattr(controller, "verify_worker", lambda *args: {"fixture": "software"})
    monkeypatch.setattr(controller, "software_report", lambda *args: {"fixture": "software"})
    monkeypatch.setattr(controller.cw, "begin_workflow_report", lambda *args, **kwargs: base / "tower" / "fixture")
    monkeypatch.setattr(controller.cw, "reporting_api", lambda: SimpleNamespace(emit=lambda *args, **kwargs: None))
    finishes = []

    def finish(*args, **kwargs):
        finishes.append(kwargs)
        if reporting_error:
            raise OSError("reporting fixture failure")

    monkeypatch.setattr(controller.cw, "finish_workflow_report", finish)
    commands = controller.worker_commands(workflow)
    launches = []

    class Child:
        def __init__(self, args, **kwargs):
            self.stage = "tests" if "pytest" in args else "experiment"
            launches.append((args, kwargs["env"]))
            if self.stage == "experiment":
                (base / "experiment").mkdir()
                (base / "experiment" / "partial.json").write_text('{"preserved": true}\n')

        def wait(self, timeout=None):
            return exit_code if self.stage == stage else 0

        def poll(self):
            return self.wait()

    monkeypatch.setattr(controller.subprocess, "Popen", Child)
    assert controller.main(["worker", "--workflow", str(base / "research-workflow.json"), "--phase", "cpu"]) == exit_code
    record = controller.cw.read_json(controller.state_path(workflow))
    assert record["stage"] == stage and record["exit_code"] == exit_code
    assert record["status"] == ("INTERRUPTED" if exit_code == 75 else "FAILED")
    assert [args for args, _ in launches] == [args for _, args in commands[:1 if stage == "tests" else 2]]
    assert "TDN_TOWER_DIR" not in launches[0][1]
    if stage == "experiment":
        assert launches[1][1]["TDN_TOWER_DIR"] == str(base / "tower" / "fixture")
        assert (base / "experiment" / "partial.json").read_text() == '{"preserved": true}\n'
        assert not (base / "experiment" / "COMPLETED").exists()
    assert os.environ["TDN_TOWER_DIR"] == "caller-report"
    assert finishes[0]["exit_code"] == exit_code
    archive = controller.collect(workflow)
    assert "Work-precision artifact verification: NOT VERIFIED" in capsys.readouterr().out
    with tarfile.open(archive) as stream:
        archived = json.load(stream.extractfile(f"{workflow['run_id']}/state/phase.json"))
    assert archived == record


def test_collect_refuses_symlinks_and_does_not_modify_evidence(controller):
    workflow = plan(controller)
    base = finish_cpu(controller, workflow)
    (base / "evidence-link").symlink_to(base / "experiment" / "summary.json")
    with pytest.raises(ValueError, match="refuses symlink"):
        controller.collect(workflow)
    assert not list((controller.ROOT / "runs").glob("*.tar.gz"))
    controller.verify_experiment(workflow)


@pytest.mark.parametrize("lines", ["0", "100001"])
def test_logs_reject_out_of_bounds_line_counts(controller, lines):
    workflow = plan(controller)
    finish_cpu(controller, workflow)
    assert controller.main(["logs", workflow["run_id"], "--lines", lines]) == 2


@pytest.fixture
def project(tmp_path):
    if os.name == "nt":
        pytest.skip("Bash launchers require a POSIX host")
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("carc_work_precision.sh", "work_precision_local.sh", "common.sh"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    (scripts / "carc_research.sh").write_text(
        "#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) +
        " -c 'import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(37)' \"$@\"\n")
    return root


@pytest.fixture
def clean_env():
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(("SLURM_", "TDN_", "CONDA_")):
            del env[key]
    return env


def invoke(project, name, *args, env=None):
    return subprocess.run(["bash", str(project / "scripts" / name), *args],
                          cwd=project, env=env, text=True, capture_output=True,
                          timeout=20, check=False)


def forwarded(project, *args):
    result = invoke(project, "carc_work_precision.sh", *args)
    assert result.returncode == 37, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("args", [(), ("--submit",), ("start", "--smoke", "--run-id", "quoted run")])
def test_carc_defaults_preserve_quoted_arguments(project, args):
    expected = list(args[1:] if args and args[0] == "start" else args)
    assert forwarded(project, *args) == [
        "start", "--work-precision", "--config",
        str(project / "scripts" / ".." / "configs" / "work-precision.yaml"), *expected]


@pytest.mark.parametrize("args", [("status", "latest"), ("logs", "quoted run", "--lines", "200"),
                                  ("collect", "quoted run")])
def test_carc_read_commands_delegate_unchanged(project, args):
    assert forwarded(project, *args) == list(args)


@pytest.mark.parametrize("option", [("--config", "configs/custom with spaces.yaml"),
                                    ("--config=configs/custom with spaces.yaml",)])
def test_carc_custom_config_and_explicit_marker_are_preserved(project, option):
    assert forwarded(project, *option, "--submit") == ["start", "--work-precision", *option, "--submit"]
    assert forwarded(project, "--work-precision", *option) == ["start", "--work-precision", *option]


def test_carc_benchmark_rejected_before_delegation(project):
    result = invoke(project, "carc_work_precision.sh", "benchmark", "run-123", "--submit")
    assert result.returncode == 2 and "no GPU benchmark successor" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("name", ["carc_work_precision.sh", "work_precision_local.sh"])
def test_help_needs_no_venv_or_environment_preparation(project, name, clean_env):
    clean_env["TDN_EXECUTION_MODE"] = "desktop"
    result = invoke(project, name, "--help", env=clean_env)
    assert result.returncode == 0, result.stderr
    assert "work-precision.yaml" in result.stdout and "CPU-only" in result.stdout
    if name == "carc_work_precision.sh":
        assert all(text in result.stdout for text in ("4 CPUs", "16 GiB", "30 minutes", "1,200-second"))
    assert not (project / ".cache").exists()
    syntax = subprocess.run(["bash", "-n", str(project / "scripts" / name)],
                            text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


@pytest.fixture
def local_project(project):
    # A real dependency-free venv exercises common.sh's prefix/Conda validation.
    venv.EnvBuilder(with_pip=False).create(project / ".venv")
    (project / "scripts" / "work_precision.py").write_text(
        "import json, os, sys\n"
        "keys = ['TDN_LOCAL_TEST_ROOT', 'TDN_ENV_PREFIX', 'TMPDIR', 'TMP', 'TEMP', "
        "'PIP_CACHE_DIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR', "
        "'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', "
        "'PYTHONPYCACHEPREFIX', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME']\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'prefix': sys.prefix, "
        "'env': {key: os.environ.get(key) for key in keys}}))\n"
        "raise SystemExit(37)\n")
    return project


def test_local_uses_project_venv_quoted_paths_and_project_storage(local_project, clean_env):
    result = invoke(local_project, "work_precision_local.sh", "--run-dir", "runs/quoted run", "--smoke",
                    "--config", "configs/custom config.yaml", env=clean_env)
    assert result.returncode == 37, result.stderr
    record = json.loads(result.stdout)
    assert record["args"] == ["--config", str(local_project / "configs/custom config.yaml"),
                              "--run-dir", str(local_project / "runs/quoted run"), "--smoke"]
    assert record["cwd"] == str(local_project)
    assert record["prefix"] == str(local_project / ".venv")
    assert record["env"]["TDN_LOCAL_TEST_ROOT"] == str(local_project)
    assert record["env"]["TDN_ENV_PREFIX"] == str(local_project / ".venv")
    for key, value in record["env"].items():
        assert Path(value).is_relative_to(local_project), key


def test_local_defaults_to_screen_config_and_fresh_run(local_project, clean_env):
    result = invoke(local_project, "work_precision_local.sh", env=clean_env)
    assert result.returncode == 37, result.stderr
    args = json.loads(result.stdout)["args"]
    assert args[:3] == ["--config", str(local_project / "configs/work-precision.yaml"), "--run-dir"]
    assert Path(args[3]).parent == local_project / "runs"
    assert Path(args[3]).name.startswith("work-precision-local-")
    assert len(args) == 4 and not Path(args[3]).exists()


@pytest.mark.parametrize("name", ["SLURM_JOB_ID", "SLURM_STEP_ID", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT"])
def test_local_rejects_slurm_and_desktop_overrides_before_work(project, clean_env, name):
    clean_env[name] = "active"
    result = invoke(project, "work_precision_local.sh", env=clean_env)
    assert result.returncode == 2 and "Slurm or desktop overrides" in result.stderr
    assert not (project / ".cache").exists()


def test_local_rejects_carc_root_before_work(project, clean_env):
    common = project / "scripts/common.sh"
    common.write_text(common.read_text().replace("TDN_CARC_ROOT=/home1/aadaniel/projects/TDN",
                                               "TDN_CARC_ROOT=" + shlex.quote(str(project))))
    result = invoke(project, "work_precision_local.sh", env=clean_env)
    assert result.returncode == 2 and "login node" in result.stderr
    assert "carc_work_precision.sh --submit" in result.stderr
    assert not (project / ".cache").exists()


@pytest.mark.parametrize("option", ["--run-dir", "--config"])
def test_local_rejects_paths_outside_project(local_project, clean_env, option):
    result = invoke(local_project, "work_precision_local.sh", option, str(local_project.parent / "outside"),
                    env=clean_env)
    assert result.returncode == 2 and "Path leaves the project" in result.stderr
    assert result.stdout == ""


def test_local_rejects_symlink_storage_escape(project, clean_env):
    outside = project.parent / "outside-cache"
    outside.mkdir()
    (project / ".cache").symlink_to(outside, target_is_directory=True)
    result = invoke(project, "work_precision_local.sh", env=clean_env)
    assert result.returncode == 2 and "Path leaves the project" in result.stderr
    assert list(outside.iterdir()) == []


def test_local_preserves_existing_run(local_project, clean_env):
    run = local_project / "runs" / "complete"
    run.mkdir(parents=True)
    artifact = run / "result.json"
    artifact.write_text('{"status":"COMPLETED"}')
    result = invoke(local_project, "work_precision_local.sh", "--run-dir", str(run), env=clean_env)
    assert result.returncode == 2 and "Run path already exists" in result.stderr
    assert artifact.read_text() == '{"status":"COMPLETED"}' and result.stdout == ""


def test_local_requires_existing_project_venv(project, clean_env):
    result = invoke(project, "work_precision_local.sh", env=clean_env)
    assert result.returncode == 2 and "Python venv missing" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("args", [("--device", "cuda"), ("--submit",), ("--config",), ("--run-dir",)])
def test_local_rejects_unsupported_and_incomplete_arguments_before_work(project, clean_env, args):
    result = invoke(project, "work_precision_local.sh", *args, env=clean_env)
    assert result.returncode == 2 and ("Unknown option" in result.stderr or "needs a path" in result.stderr)
    assert not (project / ".cache").exists()
