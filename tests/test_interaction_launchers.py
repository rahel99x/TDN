"""Interaction shell contracts exercise delegation and real local venv checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import venv

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Bash launchers require a POSIX host")


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("carc_interactions.sh", "interactions_local.sh", "common.sh"):
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
    result = invoke(project, "carc_interactions.sh", *args)
    assert result.returncode == 37, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("args", [(), ("--submit",), ("start", "--smoke", "--run-id", "quoted run")])
def test_carc_defaults_preserve_quoted_arguments(project, args):
    expected = list(args[1:] if args and args[0] == "start" else args)
    assert forwarded(project, *args) == [
        "start", "--interaction-screen", "--config",
        str(project / "scripts" / ".." / "configs" / "interaction-screen.yaml"), *expected]


@pytest.mark.parametrize("args", [("status", "latest"), ("logs", "quoted run", "--lines", "200"),
                                  ("collect", "quoted run")])
def test_carc_read_commands_delegate_unchanged(project, args):
    assert forwarded(project, *args) == list(args)


@pytest.mark.parametrize("option", [("--config", "configs/custom with spaces.yaml"),
                                    ("--config=configs/custom with spaces.yaml",)])
def test_carc_custom_config_and_explicit_marker_are_preserved(project, option):
    assert forwarded(project, *option, "--submit") == ["start", "--interaction-screen", *option, "--submit"]
    assert forwarded(project, "--interaction-screen", *option) == ["start", "--interaction-screen", *option]


def test_carc_benchmark_rejected_before_delegation(project):
    result = invoke(project, "carc_interactions.sh", "benchmark", "run-123", "--submit")
    assert result.returncode == 2 and "no GPU benchmark successor" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("name", ["carc_interactions.sh", "interactions_local.sh"])
def test_help_needs_no_venv_or_environment_preparation(project, name, clean_env):
    clean_env["TDN_EXECUTION_MODE"] = "desktop"
    result = invoke(project, name, "--help", env=clean_env)
    assert result.returncode == 0, result.stderr
    assert "interaction-screen.yaml" in result.stdout and "CPU-only" in result.stdout
    if name == "carc_interactions.sh":
        assert all(text in result.stdout for text in ("4 CPUs", "16 GiB", "30 minutes", "1,200-second"))
    assert not (project / ".cache").exists()
    syntax = subprocess.run(["bash", "-n", str(project / "scripts" / name)],
                            text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


@pytest.fixture
def local_project(project):
    # A real dependency-free venv exercises common.sh's prefix/Conda validation.
    venv.EnvBuilder(with_pip=False).create(project / ".venv")
    (project / "scripts" / "interaction_screen.py").write_text(
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
    result = invoke(local_project, "interactions_local.sh", "--run-dir", "runs/quoted run", "--smoke",
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
    result = invoke(local_project, "interactions_local.sh", env=clean_env)
    assert result.returncode == 37, result.stderr
    args = json.loads(result.stdout)["args"]
    assert args[:3] == ["--config", str(local_project / "configs/interaction-screen.yaml"), "--run-dir"]
    assert Path(args[3]).parent == local_project / "runs"
    assert Path(args[3]).name.startswith("interactions-local-")
    assert len(args) == 4 and not Path(args[3]).exists()


@pytest.mark.parametrize("name", ["SLURM_JOB_ID", "SLURM_STEP_ID", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT"])
def test_local_rejects_slurm_and_desktop_overrides_before_work(project, clean_env, name):
    clean_env[name] = "active"
    result = invoke(project, "interactions_local.sh", env=clean_env)
    assert result.returncode == 2 and "Slurm or desktop overrides" in result.stderr
    assert not (project / ".cache").exists()


def test_local_rejects_carc_root_before_work(project, clean_env):
    common = project / "scripts/common.sh"
    common.write_text(common.read_text().replace("TDN_CARC_ROOT=/home1/aadaniel/projects/TDN",
                                               "TDN_CARC_ROOT=" + shlex.quote(str(project))))
    result = invoke(project, "interactions_local.sh", env=clean_env)
    assert result.returncode == 2 and "login node" in result.stderr
    assert "carc_interactions.sh --submit" in result.stderr
    assert not (project / ".cache").exists()


@pytest.mark.parametrize("option", ["--run-dir", "--config"])
def test_local_rejects_paths_outside_project(local_project, clean_env, option):
    result = invoke(local_project, "interactions_local.sh", option, str(local_project.parent / "outside"),
                    env=clean_env)
    assert result.returncode == 2 and "Path leaves the project" in result.stderr
    assert result.stdout == ""


def test_local_rejects_symlink_storage_escape(project, clean_env):
    outside = project.parent / "outside-cache"
    outside.mkdir()
    (project / ".cache").symlink_to(outside, target_is_directory=True)
    result = invoke(project, "interactions_local.sh", env=clean_env)
    assert result.returncode == 2 and "Path leaves the project" in result.stderr
    assert list(outside.iterdir()) == []


def test_local_preserves_existing_run(local_project, clean_env):
    run = local_project / "runs" / "complete"
    run.mkdir(parents=True)
    artifact = run / "result.json"
    artifact.write_text('{"status":"COMPLETED"}')
    result = invoke(local_project, "interactions_local.sh", "--run-dir", str(run), env=clean_env)
    assert result.returncode == 2 and "Run path already exists" in result.stderr
    assert artifact.read_text() == '{"status":"COMPLETED"}' and result.stdout == ""


def test_local_requires_existing_project_venv(project, clean_env):
    result = invoke(project, "interactions_local.sh", env=clean_env)
    assert result.returncode == 2 and "Python venv missing" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("args", [("--device", "cuda"), ("--submit",), ("--config",), ("--run-dir",)])
def test_local_rejects_unsupported_and_incomplete_arguments_before_work(project, clean_env, args):
    result = invoke(project, "interactions_local.sh", *args, env=clean_env)
    assert result.returncode == 2 and ("Unknown option" in result.stderr or "needs a path" in result.stderr)
    assert not (project / ".cache").exists()
