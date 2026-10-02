"""Scheduler tests use isolated command mocks; no real allocation is requested."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="CARC Bash/Slurm mocks require a POSIX host")

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_cluster(tmp_path):
    root = tmp_path / "project with spaces"
    shutil.copytree(ROOT / "scripts", root / "scripts")
    (root / "configs").mkdir()
    (root / "configs" / "smoke.yaml").write_text("purpose: development\nproblem:\n  grid: [8, 8]\ntraining:\n  max_steps: 12\nruntime:\n  confirmatory_authorized: false\n")
    common = root / "scripts" / "common.sh"
    common.write_text(common.read_text().replace("TDN_CARC_ROOT=/home1/aadaniel/projects/TDN", f"TDN_CARC_ROOT={str(root)!r}"))
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "pyvenv.cfg").write_text("Mock venv identity; no installation occurs.\n")
    python = root / ".venv" / "bin" / "python"
    python.write_text("#!/usr/bin/env bash\nif [[ \"${1:-}\" == -c ]]; then exit 0; fi\nexec " + str(sys.executable) + " \"$@\"\n")
    python.chmod(0o755)
    bin_dir = root / "mocks"
    bin_dir.mkdir()
    mocks = {
        "id": "printf 'aadaniel\\n'",
        "sacctmgr": "printf 'anakano_81|aadaniel||normal|1-00:00:00|\\n'",
        "myaccount": "printf 'Account | Partition\\nanakano_81 | ALL\\n'",
        "scontrol": "printf 'PartitionName=%s State=UP AllowAccounts=anakano_81 MaxTime=1-00:00:00 MaxCPUsPerNode=64 MaxMemPerNode=131072\\n' \"${@: -1}\"",
        "sinfo": "if [[ \"$*\" == *'%P|'* ]]; then printf 'cpu*|up|(null)|1-00:00:00|64|131072\\n'; elif [[ \"$*\" == *'-p gpu'* ]]; then printf 'gpu001|gpu:a100:4(S:0-1)|a100-40gb|131072|64\\n'; else printf 'cpu001|(null)|cpu|131072|64\\n'; fi",
    }
    for name, body in mocks.items():
        file = bin_dir / name
        file.write_text("#!/usr/bin/env bash\nset -eu\n" + body + "\n")
        file.chmod(0o755)
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/usr/bin/env python3\nimport json, os, pathlib, sys\n"
                      "root=pathlib.Path(os.environ['MOCK_ROOT'])\n"
                      "args=sys.argv[1:]\n"
                      "assert '--account=anakano_81' in args\n"
                      "output=next(x.split('=',1)[1] for x in args if x.startswith('--output='))\n"
                      "assert pathlib.Path(output).parent.is_dir()\n"
                      "record=root/'sbatch_calls.jsonl'\n"
                      "prior=record.read_text().splitlines() if record.exists() else []\n"
                      "with record.open('a') as f: f.write(json.dumps({'args':args,'stage':os.environ['TDN_STAGE'],'run':os.environ['TDN_RUN_DIR'],'device':os.environ['TDN_DEVICE']})+'\\n')\n"
                      "print(str(841+len(prior))+';discovery')\n")
    sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update({"PATH": str(bin_dir) + os.pathsep + env["PATH"], "USER": "aadaniel",
                "LOGNAME": "aadaniel", "MOCK_ROOT": str(root)})
    for name in ("TDN_LOCAL_TEST_ROOT", "CARC_ACCOUNT", "TDN_CPU_PARTITION", "TDN_GATE_REPORT",
                 "TDN_CALIBRATION_REPORT", "TDN_PYTHON_MODULE", "TDN_LOADED_PYTHON_MODULE", "TDN_RESUME",
                 "CONDA_PREFIX", "CONDA_SHLVL"):
        env.pop(name, None)
    return root, env


def run(root, env, script, *args):
    return subprocess.run(["bash", str(root / "scripts" / script), *args],
                          cwd=root, env=env, text=True, capture_output=True, check=False)


def run_shell(root, env, commands, *, interactive=False):
    """Run a clean Bash, including real prompt hooks for interactive regressions."""
    shell_env = env.copy()
    for name in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"):
        shell_env.pop(name, None)
    shell_env.update({"PS1": "", "PS2": "", "PROMPT_COMMAND": "false"})
    args = ["bash", "--noprofile", "--norc"]
    if interactive:
        args += ["-i"]
        return subprocess.run(args, input=commands + "\nexit 0\n", cwd=root,
                              env=shell_env, text=True, capture_output=True, check=False)
    return subprocess.run(args + ["-c", commands], cwd=root, env=shell_env,
                          text=True, capture_output=True, check=False)


@pytest.mark.parametrize("strict", [False, True])
def test_common_preserves_callers_shell_options(mock_cluster, strict):
    root, env = mock_cluster
    mode = "set -euo pipefail" if strict else "set +e +u; set +o pipefail"
    result = run_shell(root, env, mode + '\n'
                       'flags_before=$-\noptions_before="$(set +o)"\n'
                       'source "$MOCK_ROOT/scripts/common.sh"\n'
                       '[[ $- == "$flags_before" ]] || exit 11\n'
                       '[[ "$(set +o)" == "$options_before" ]] || exit 12\n'
                       'printf "SOURCE_OPTIONS_PRESERVED\\n"\n')
    assert result.returncode == 0, result.stderr
    assert "SOURCE_OPTIONS_PRESERVED" in result.stdout


def test_common_source_survives_failing_interactive_prompt_hook(mock_cluster):
    root, env = mock_cluster
    result = run_shell(root, env,
                       'flags_before=$-\noptions_before="$(set +o)"\n'
                       'source "$MOCK_ROOT/scripts/common.sh"\n'
                       '[[ $- == "$flags_before" ]] || exit 11\n'
                       '[[ "$(set +o)" == "$options_before" ]] || exit 12\n'
                       'printf "LOGIN_SHELL_SURVIVED\\n"\n', interactive=True)
    assert result.returncode == 0, result.stderr
    assert "LOGIN_SHELL_SURVIVED" in result.stdout
    assert not (root / ".cache").exists()


@pytest.mark.parametrize("command", [
    "tdn_inside relative-path",
    'tdn_inside "$MOCK_ROOT/../outside"',
    "tdn_inside",
    'CARC_ACCOUNT=wrong_account; tdn_runtime_policy',
    'unset SLURM_JOB_ID SLURM_STEP_ID; tdn_allocation',
    'rm -- "$MOCK_ROOT/.venv/pyvenv.cfg"; tdn_python',
])
def test_helper_validation_returns_error_without_ending_interactive_shell(mock_cluster, command):
    root, env = mock_cluster
    result = run_shell(root, env, 'source "$MOCK_ROOT/scripts/common.sh"\n' + command + '\n'
                       'printf "HELPER_STATUS=%s\\n" "$?"\n'
                       'printf "LOGIN_SHELL_SURVIVED\\n"\n', interactive=True)
    assert result.returncode == 0, result.stderr
    assert "HELPER_STATUS=2" in result.stdout
    assert "LOGIN_SHELL_SURVIVED" in result.stdout
    assert "TDN:" in result.stderr
    assert not (root / "sbatch_calls.jsonl").exists()


@pytest.mark.parametrize("failure", ["active-conda", "cache-symlink"])
def test_interactive_environment_failure_creates_no_runtime_directories(mock_cluster, tmp_path, failure):
    root, env = mock_cluster
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    if failure == "active-conda":
        env.update({"CONDA_PREFIX": str(root / "conda"), "CONDA_SHLVL": "1"})
    else:
        (root / ".cache").symlink_to(outside, target_is_directory=True)
    result = run_shell(root, env, 'source "$MOCK_ROOT/scripts/common.sh"\n'
                       'tdn_prepare_env\n'
                       'printf "HELPER_STATUS=%s\\n" "$?"\n'
                       'printf "LOGIN_SHELL_SURVIVED\\n"\n', interactive=True)
    assert result.returncode == 0, result.stderr
    assert "HELPER_STATUS=2" in result.stdout
    assert "LOGIN_SHELL_SURVIVED" in result.stdout
    assert not list(outside.iterdir())
    assert not (root / "logs").exists()
    assert not (root / "runs").exists()
    if failure == "active-conda":
        assert not (root / ".cache").exists()


def test_failed_module_load_returns_error_and_does_not_mark_or_validate_module(mock_cluster):
    root, env = mock_cluster
    result = run_shell(root, env, 'source "$MOCK_ROOT/scripts/common.sh"\n'
                       'module() { return 37; }\n'
                       'python3() { touch "$MOCK_ROOT/python-called"; }\n'
                       'export TDN_PYTHON_MODULE=python/mock\n'
                       'tdn_load_python_module\n'
                       'printf "HELPER_STATUS=%s\\n" "$?"\n'
                       'printf "MODULE_FLAG=%s\\n" "${TDN_LOADED_PYTHON_MODULE-unset}"\n'
                       'printf "LOGIN_SHELL_SURVIVED\\n"\n', interactive=True)
    assert result.returncode == 0, result.stderr
    assert "HELPER_STATUS=2" in result.stdout
    assert "MODULE_FLAG=unset" in result.stdout
    assert "LOGIN_SHELL_SURVIVED" in result.stdout
    assert not (root / "python-called").exists()


@pytest.mark.parametrize("command", [
    'tdn_inside "$MOCK_ROOT/../outside"',
    'CARC_ACCOUNT=wrong_account; tdn_runtime_policy',
    'CONDA_PREFIX="$MOCK_ROOT/conda"; CONDA_SHLVL=1; tdn_prepare_env',
    'module() { return 37; }; TDN_PYTHON_MODULE=python/mock; tdn_load_python_module',
])
def test_noninteractive_helpers_fail_closed_even_without_errexit(mock_cluster, command):
    root, env = mock_cluster
    result = run_shell(root, env, 'set +e +u; set +o pipefail\n'
                       'source "$MOCK_ROOT/scripts/common.sh"\n' + command + '\n'
                       'printf "UNSAFE_CONTINUATION\\n"\n')
    assert result.returncode != 0
    assert "UNSAFE_CONTINUATION" not in result.stdout
    assert not (root / "sbatch_calls.jsonl").exists()


@pytest.mark.parametrize("failure", ["active-conda", "failed-module"])
def test_submission_stays_fail_closed_for_environment_errors(mock_cluster, failure):
    root, env = mock_cluster
    if failure == "active-conda":
        env.update({"CONDA_PREFIX": str(root / "conda"), "CONDA_SHLVL": "1"})
    else:
        env["TDN_PYTHON_MODULE"] = "python/mock"
        module = root / "mocks/module"
        module.write_text("#!/usr/bin/env bash\nexit 37\n")
        module.chmod(0o755)
    result = run(root, env, "submit.sh", "setup", str(root / "configs/smoke.yaml"), "--submit")
    assert result.returncode != 0
    assert not (root / "sbatch_calls.jsonl").exists()
    if failure == "active-conda":
        assert not (root / ".cache").exists()


def test_all_shell_scripts_parse():
    for file in (ROOT / "scripts").iterdir():
        if file.suffix in (".sh", ".sbatch"):
            result = subprocess.run(["bash", "-n", str(file)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr


def test_dry_run_allocates_nothing_and_cpu_requests_no_gpu(mock_cluster):
    root, env = mock_cluster
    result = run(root, env, "submit.sh", "cpu-tests", str(root / "configs/smoke.yaml"))
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "--account=anakano_81" in result.stdout
    assert "discover-authorized-cpu-at-submit" in result.stdout
    assert "--gpus" not in result.stdout
    assert not (root / "sbatch_calls.jsonl").exists()
    assert not (root / "logs").exists()
    assert not (root / ".cache").exists()


def test_mock_submission_quotes_paths_and_verifies_resources(mock_cluster):
    root, env = mock_cluster
    config = str(root / "configs/smoke.yaml")
    result = run(root, env, "submit.sh", "train", config, "--submit", "--pilot-budget", "12",
                 "--dependency", "afterok:123:456", "--run-dir", str(root / "runs/run with spaces"))
    assert result.returncode == 0, result.stderr
    assert "TDN_JOB_ID=841" in result.stdout
    record = json.loads((root / "sbatch_calls.jsonl").read_text())
    args = record["args"]
    assert "--gpus-per-task=a100:1" in args and "--constraint=a100-40gb" in args
    assert "--cpus-per-task=8" in args and "--mem=64G" in args and "--time=04:00:00" in args
    assert "--dependency=afterok:123:456" in args
    assert args[-1] == str(root / "scripts/gpu.sbatch")
    assert record["run"] == str(root / "runs/run with spaces")
    assert (root / "runs/run with spaces/slurm_policy.json").is_file()


def test_cpu_partition_is_discovered(mock_cluster):
    root, env = mock_cluster
    result = run(root, env, "submit.sh", "audit", str(root / "configs/smoke.yaml"), "--submit")
    assert result.returncode == 0, result.stderr
    record = json.loads((root / "sbatch_calls.jsonl").read_text())
    assert "--partition=cpu" in record["args"]
    assert not any(a.startswith("--gpus") for a in record["args"])
    assert record["device"] == "cpu"


@pytest.mark.parametrize("option,value", [("--dependency", "afterok:1;echo hacked"),
                                         ("--pilot-budget", "1001"), ("--resume", "/outside/checkpoint")])
def test_invalid_submission_options_fail_closed(mock_cluster, option, value):
    root, env = mock_cluster
    result = run(root, env, "submit.sh", "train", str(root / "configs/smoke.yaml"), option, value)
    assert result.returncode != 0
    assert not (root / "sbatch_calls.jsonl").exists()


def test_submission_rejects_wrong_account_and_local_override(mock_cluster):
    root, env = mock_cluster
    env["CARC_ACCOUNT"] = "anakano_81_extra"
    result = run(root, env, "submit.sh", "audit", str(root / "configs/smoke.yaml"), "--submit")
    assert result.returncode != 0
    env["CARC_ACCOUNT"] = "anakano_81"
    env["TDN_LOCAL_TEST_ROOT"] = str(root)
    result = run(root, env, "submit.sh", "audit", str(root / "configs/smoke.yaml"), "--submit")
    assert result.returncode != 0
    assert not (root / "sbatch_calls.jsonl").exists()


def test_cache_symlink_escape_is_rejected(mock_cluster, tmp_path):
    root, env = mock_cluster
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".cache").symlink_to(outside, target_is_directory=True)
    result = run(root, env, "submit.sh", "audit", str(root / "configs/smoke.yaml"), "--submit")
    assert result.returncode != 0
    assert not list(outside.iterdir())
    assert not (root / "sbatch_calls.jsonl").exists()


def test_every_runtime_cache_remains_in_project(mock_cluster):
    root, env = mock_cluster
    result = subprocess.run(["bash", "-c", 'source "$1/scripts/common.sh"; tdn_prepare_env; '
                             'printf "%s\\n" "$TMPDIR" "$TMP" "$TEMP" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" '
                             '"$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" '
                             '"$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$PYTHONPYCACHEPREFIX"',
                             "bash", str(root)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    for path in result.stdout.splitlines():
        assert Path(path).resolve().is_relative_to(root)


def test_pipeline_chains_only_actual_job_ids_and_defaults_to_readiness(mock_cluster):
    root, env = mock_cluster
    result = run(root, env, "pipeline.sh", str(root / "configs/smoke.yaml"), "--submit", "--pipeline-id", "checked")
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in (root / "sbatch_calls.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in records] == ["cpu-tests", "audit", "generate", "gpu-tests", "calibrate"]
    assert not any(a.startswith("--dependency=") for a in records[0]["args"])
    for index, record in enumerate(records[1:], 1):
        assert f"--dependency=afterok:{840+index}" in record["args"]


def test_failed_account_audit_never_reaches_sbatch(mock_cluster):
    root, env = mock_cluster
    (root / "mocks/sacctmgr").write_text("#!/usr/bin/env bash\nprintf 'anakano_81_extra|aadaniel||normal|1-00:00:00|\\n'\n")
    (root / "mocks/myaccount").write_text("#!/usr/bin/env bash\nprintf 'anakano_81_extra | ALL\\n'\n")
    result = run(root, env, "submit.sh", "gpu-tests", str(root / "configs/smoke.yaml"), "--submit")
    assert result.returncode != 0
    assert not (root / "sbatch_calls.jsonl").exists()


def test_preflight_and_work_share_one_srun_task():
    batch = (ROOT / "scripts/gpu.sbatch").read_text()
    stage = (ROOT / "scripts/run_stage.sh").read_text()
    assert "gpu_preflight.py" not in batch
    assert 'exec srun' in batch and 'scripts/run_stage.sh' in batch
    assert stage.index('gpu_preflight.py') < stage.index('-u -m tdn.cli')
    assert "CUDA_VISIBLE_DEVICES=" not in batch + stage
    for file in (ROOT / "scripts").iterdir():
        if file.suffix in (".sh", ".sbatch"):
            text = file.read_text()
            assert "/tmp/" not in text and "/scratch1/" not in text


def test_exact_account_parser_and_limits():
    module = load_script("carc_check")
    assert module.association_verified("anakano_81|aadaniel|gpu|normal|", "gpu")
    assert not module.association_verified("anakano_81_extra|aadaniel|gpu|normal|", "gpu")
    assert not module.association_verified("anakano_81|other|gpu|normal|", "gpu")
    assert not module.association_verified("anakano_81|aadaniel|private|normal|", "gpu")
    assert module.myaccount_verified("Account | Partition\nanakano_81 | gpu")
    assert not module.myaccount_verified("not authorized for anakano_81")
    assert not module.myaccount_verified("anakano_81_extra | gpu")
    assert module.seconds("1-02:03:04") == 93784
    assert module.seconds("04:00:00") == 14400


def test_gpu_preflight_rejects_wrong_hardware_without_gpu_execution():
    module = load_script("gpu_preflight")
    props = SimpleNamespace(name="NVIDIA A100-SXM4-40GB", major=8, minor=0, total_memory=40 * 1024**3)
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_properties=lambda _: props,
                           mem_get_info=lambda: (38 * 1024**3, 40 * 1024**3))
    fake_torch = SimpleNamespace(cuda=cuda)
    assert module.validate_gpu(fake_torch)["soft_reserved_budget_bytes"] == 30 * 1024**3
    props.name = "NVIDIA A100 80GB"
    props.total_memory = 80 * 1024**3
    with pytest.raises(RuntimeError, match="40 GB"):
        module.validate_gpu(fake_torch)
    props.name = "NVIDIA A100 MIG"
    props.total_memory = 20 * 1024**3
    with pytest.raises(RuntimeError, match="full A100"):
        module.validate_gpu(fake_torch)
