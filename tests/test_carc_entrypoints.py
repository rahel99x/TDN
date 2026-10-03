"""Login Bash entrypoints with a substituted controller and module command."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="CARC Bash entrypoints require POSIX")


@pytest.fixture
def login_shell(tmp_path):
    root = tmp_path / "checkout with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("common.sh", "carc_defaults.sh", "carc.sh", "carc_start.sh",
                 "carc_status.sh", "carc_restart.sh"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    common = scripts / "common.sh"
    common.write_text(common.read_text().replace(
        "TDN_CARC_ROOT=/home1/aadaniel/projects/TDN", f"TDN_CARC_ROOT={str(root)!r}"))
    (scripts / "carc_workflow.py").write_text(
        "import json,os,sys\n"
        "keys=('CARC_ACCOUNT','TDN_CPU_PARTITION','TDN_PYTHON_MODULE','TORCH_VERSION',"
        "'TORCH_WHEEL_INDEX','TDN_RESUME','TDN_REQUIRE_GPU_TESTS','TDN_PROJECT_ROOT')\n"
        "print(json.dumps({'args':sys.argv[1:],'env':{k:os.environ.get(k) for k in keys}}))\n")
    mocks = root / "mocks"
    mocks.mkdir()
    (mocks / "id").write_text("#!/usr/bin/env bash\nprintf 'aadaniel\\n'\n")
    (mocks / "id").chmod(0o755)
    bootstrap = root / "mock-module.sh"
    bootstrap.write_text("module() { return 0; }\n")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("TDN_", "TORCH_", "SLURM_", "CONDA_")) or key == "CARC_ACCOUNT":
            env.pop(key)
    env.update(PATH=str(mocks) + os.pathsep + env["PATH"], BASH_ENV=str(bootstrap))
    return root, env


def invoke(login_shell, name, *args, additions=None):
    root, env = login_shell
    env = env | (additions or {})
    return subprocess.run(["bash", str(root / "scripts" / name), *args],
                          cwd=root, env=env, text=True, capture_output=True, check=False)


@pytest.mark.parametrize("name,args,expected", [
    ("carc.sh", [], ["start"]),
    ("carc_start.sh", [], ["start"]),
    ("carc_status.sh", [], ["status", "latest"]),
    ("carc_restart.sh", ["--dry-run"], ["restart", "latest", "--dry-run"]),
])
def test_shortcuts_supply_defaults_without_runtime_directories(login_shell, name, args, expected):
    result = invoke(login_shell, name, *args)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["args"] == expected
    assert record["env"]["CARC_ACCOUNT"] == "anakano_81"
    assert record["env"]["TDN_CPU_PARTITION"] == "main"
    assert record["env"]["TDN_PYTHON_MODULE"] == "python/3.11.9"
    assert record["env"]["TORCH_VERSION"] == "2.10.0+cu126"
    assert record["env"]["TORCH_WHEEL_INDEX"] == "https://download.pytorch.org/whl/cu126"
    root, _ = login_shell
    assert not (root / ".cache").exists()
    assert not (root / "runs").exists()


def test_submission_prepares_local_storage_and_clears_stale_gpu_test_flag(login_shell):
    result = invoke(login_shell, "carc_start.sh", "--submit", additions={
        "TDN_REQUIRE_GPU_TESTS": "1", "TDN_PROJECT_ROOT": "old-desktop-path"})
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["args"] == ["start", "--submit"]
    assert record["env"]["TDN_REQUIRE_GPU_TESTS"] is None
    assert record["env"]["TDN_PROJECT_ROOT"] is None
    root, _ = login_shell
    assert (root / ".cache" / "temp").is_dir()


@pytest.mark.parametrize("args", [["--submit", "--dry-run"], ["--dry-run", "--submit"]])
def test_dry_run_overrides_submit_before_any_storage_preparation(login_shell, args):
    result = invoke(login_shell, "carc_start.sh", *args)
    assert result.returncode == 0, result.stderr
    root, _ = login_shell
    assert not (root / ".cache").exists()
    assert not (root / "runs").exists()


@pytest.mark.parametrize("additions", [
    {"CARC_ACCOUNT": "another-lab"}, {"SLURM_JOB_ID": "123"},
    {"TDN_EXECUTION_MODE": "desktop"}, {"CONDA_PREFIX": "conda", "CONDA_SHLVL": "1"},
])
def test_invalid_submission_context_stops_before_controller_and_storage(login_shell, additions):
    result = invoke(login_shell, "carc_start.sh", "--submit", additions=additions)
    assert result.returncode != 0
    assert not result.stdout
    root, _ = login_shell
    assert not (root / ".cache").exists()
    assert not (root / "runs").exists()
