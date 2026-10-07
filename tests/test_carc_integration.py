"""Actual controller/worker coupling with isolated scheduler and stage substitutes.

These tests execute no numerical stages, installation, Slurm allocation or GPU
readiness checks. Only copied policy roots and mocked software facts are used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="CARC worker integration requires Bash")


@pytest.fixture
def carc_integration(tmp_path):
    root = tmp_path / "mock CARC project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "carc_workflow.py", "carc_check.py", "common.sh",
        "carc_cpu.sbatch", "carc_gpu.sbatch", "carc_worker.sh",
    ):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    common = scripts / "common.sh"
    common.write_text(common.read_text().replace(
        "TDN_CARC_ROOT=/home1/aadaniel/projects/TDN",
        "TDN_CARC_ROOT=" + shlex.quote(str(root)),
    ))
    controller = scripts / "carc_workflow.py"
    controller.write_text(controller.read_text().replace(
        'CARC_ROOT = Path("/home1/aadaniel/projects/TDN")',
        "CARC_ROOT = Path(" + repr(str(root)) + ")",
    ))
    for name in ("requirements.txt", "pyproject.toml"):
        shutil.copy2(ROOT / name, root / name)
    for name in ("configs", "tdn", "reference", "tests"):
        (root / name).mkdir()
    shutil.copy2(ROOT / "configs/carc-smoke.yaml", root / "configs/carc-smoke.yaml")
    # Copy the reporting dependency closure used by real worker finalization.
    # These declarations/parsers use only the standard library; numerical
    # engines and model implementations remain absent from this substitute.
    for name in (
        "__init__.py", "config.py", "reporting.py", "tower_analytics.py",
        "premix_reporting.py", "consistency_reporting.py", "analysis/__init__.py",
        "analysis/consistency/__init__.py", "analysis/consistency/protocol.py",
        "analysis/consistency/comparison.py", "analysis/premix/__init__.py",
        "analysis/premix/comparison.py", "research/__init__.py", "research/protocol.py",
        "runtime/__init__.py", "runtime/metadata.py",
    ):
        destination = root / "tdn" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tdn" / name, destination)
    mocks = root / "mocks"
    mocks.mkdir()
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("Explicit unit-test interpreter substitute, no installation.\n")
    python = venv / "bin/python"
    python.write_text(
        "#!" + sys.executable + "\n" +
        """import json, os, pathlib, sys
REAL_PYTHON=""" + repr(sys.executable) + """
if len(sys.argv)>2 and sys.argv[1]=='-c':
    if 'versions={name:m.version(name)' in sys.argv[2]:
        from tdn.config import config_hash, load_config
        root=pathlib.Path(sys.argv[3])
        packages={'torch':sys.argv[5], 'setuptools':'80.9.0', 'wheel':'0.45.1'}
        for line in (root/'requirements.txt').read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                name, version=line.strip().split('==');packages[name]=version
        print(json.dumps({'python':sys.argv[6], 'prefix':str(root/'.venv'),
             'base_prefix':'unit-test-standalone-python', 'packages':packages,
             'config_hash':config_hash(load_config(sys.argv[4]))}))
    raise SystemExit(0)
os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])
"""
    )
    python.chmod(0o755)
    (scripts / "run_stage.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'exec "$TDN_REPO_ROOT/.venv/bin/python" "$TDN_REPO_ROOT/mocks/stage.py"\n'
    )
    (mocks / "stage.py").write_text(
        """import json, os, pathlib, sys
base=pathlib.Path(os.environ['TDN_WORKFLOW_ROOT'])
stage=os.environ['TDN_STAGE']
run=pathlib.Path(os.environ['TDN_RUN_DIR'])
run.mkdir(parents=True,exist_ok=True)
with (base/'mock-executed.jsonl').open('a') as stream:
    stream.write(json.dumps({'stage':stage,'device':os.environ['TDN_DEVICE'],
                             'resume':os.environ['TDN_RESUME']})+'\\n')
if stage==os.environ.get('MOCK_FAIL_STAGE'):
    code=int(os.environ.get('MOCK_FAIL_CODE','17'))
    status='PAUSED_NEEDS_RESUME' if code==75 else 'FAILED'
    (run/'stage.json').write_text(json.dumps({'stage':stage,'status':status,
        'category':'isolated_mock_execution','note':'No numerical stage ran.'}))
    raise SystemExit(code)
if stage not in ('setup','cpu-tests','gpu-tests'):
    workflow=json.loads((base/'workflow.json').read_text())
    software=json.loads((base/'state/software.json').read_text())
    status={'stage':stage,'status':'COMPLETED','actually_ran':True,
        'category':'isolated_mock_execution',
        'note':'Mock completion proof only; no numerical work or GPU validation.',
        'config_hash':software['config_hash'],'device':os.environ['TDN_DEVICE'],
        'software':{'source_tree_sha256':workflow['source_tree_sha256'],
                    'python':software['python'],
                    **{key:software['packages'][key] for key in ('torch','numpy','scipy')}}}
    (run/'stage.json').write_text(json.dumps(status))
    (run/'COMPLETED').write_text(software['config_hash']+'\\n')
"""
    )
    commands = {
        "id": "printf 'aadaniel\\n'",
        "sacctmgr": "printf 'anakano_81|aadaniel||normal|2-00:00:00|\\n'",
        "scontrol": (
            'if [[ "${2:-}" == job ]]; then '
            'printf "JobId=%s UserId=aadaniel(123) Account=anakano_81 '
            'JobName=tdn-carc-%s Comment=tdn:%s:%s JobState=RUNNING\\n" '
            '"$SLURM_JOB_ID" "$TDN_WORKFLOW_PHASE" '
            '"${MOCK_OWNER_RUN:-integrated-unit}" "$TDN_WORKFLOW_PHASE"; else '
            'printf "PartitionName=%s State=UP AllowAccounts=anakano_81 '
            'MaxTime=2-00:00:00 MaxCPUsPerNode=64 MaxMemPerNode=131072\\n" "${@: -1}"; fi'
        ),
        "sinfo": (
            'if [[ "$*" == *"-p gpu"* ]]; then '
            "printf 'unit-gpu|gpu:a100:1|a100-40gb|131072|64\\n'; else "
            "printf 'unit-cpu|(null)|cpu|131072|64\\n'; fi"
        ),
        "squeue": "exit 0",
        "sacct": (
            "printf '%s|COMPLETED|0:0\\n' "
            '"${MOCK_STATUS_JOB:-7101}"'
        ),
    }
    for name, body in commands.items():
        command = mocks / name
        command.write_text("#!/usr/bin/env bash\nset -eu\n" + body + "\n")
        command.chmod(0o755)
    sbatch = mocks / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env python3\n" +
        """import json, os, pathlib, sys
root=pathlib.Path(os.environ['MOCK_ROOT'])
path=root/'mock-submissions.jsonl'
prior=path.read_text().splitlines() if path.exists() else []
job=str(7101+len(prior))
env={key:value for key,value in os.environ.items()
     if key.startswith(('TDN_','TORCH_')) or key=='CARC_ACCOUNT'}
record={'job':job,'args':sys.argv[1:],'env':env}
with path.open('a') as stream:stream.write(json.dumps(record)+'\\n')
print(job+';unit-scheduler')
"""
    )
    sbatch.chmod(0o755)
    srun = mocks / "srun"
    srun.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'printf "%s|%s\\n" "$TDN_WORKFLOW_PHASE" "$*" >> "$MOCK_ROOT/mock-srun.txt"\n'
        'export SLURM_STEP_ID=0\n'
        'while [[ "${1:-}" == --* ]]; do shift; done\n'
        'exec "$@"\n'
    )
    srun.chmod(0o755)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("SLURM_", "TDN_", "MOCK_")) or name in (
            "CONDA_PREFIX", "CONDA_SHLVL", "TORCH_VERSION", "TORCH_WHEEL_INDEX", "CARC_ACCOUNT",
        ):
            env.pop(name, None)
    env.update(
        PATH=str(mocks) + os.pathsep + env["PATH"], USER="aadaniel",
        LOGNAME="aadaniel", MOCK_ROOT=str(root), PYTHONNOUSERSITE="1",
        PYTHONPYCACHEPREFIX=str(root / ".cache/pycache"),
    )
    return root, env


def controller(project, *args, **updates):
    root, environment = project
    env = {**environment, **updates}
    return subprocess.run(
        ["python3", str(root / "scripts/carc_workflow.py"), *args],
        cwd=root, env=env, text=True, capture_output=True, timeout=30,
    )


def submit(project):
    root, _env = project
    result = controller(project, "start", "--run-id", "integrated-unit", "--submit")
    assert result.returncode == 0, result.stdout + result.stderr
    base = root / "runs/integrated-unit"
    submitted = [
        json.loads(line) for line in (root / "mock-submissions.jsonl").read_text().splitlines()
    ]
    assert len(submitted) == 2
    assert "--dependency=afterok:7101" in submitted[1]["args"]
    assert all("--account=anakano_81" in item["args"] for item in submitted)
    return base, submitted


def run_phase(project, submitted, phase, **updates):
    root, environment = project
    entry = next(item for item in submitted if item["env"]["TDN_WORKFLOW_PHASE"] == phase)
    env = {**environment, **entry["env"], **updates}
    env.update(SLURM_JOB_ID=entry["job"], SLURM_JOB_ACCOUNT="anakano_81")
    return subprocess.run(
        ["bash", str(root / "scripts" / f"carc_{phase}.sbatch")],
        cwd=root, env=env, text=True, capture_output=True, timeout=45,
    )


def stage_records(base, phase):
    directory = base / "state" / phase
    return {path.stem: json.loads(path.read_text()) for path in directory.glob("*.json")}


def test_actual_controller_and_workers_complete_two_mocked_allocations(carc_integration):
    root, _env = carc_integration
    base, submitted = submit(carc_integration)
    cpu = run_phase(carc_integration, submitted, "cpu")
    assert cpu.returncode == 0, cpu.stdout + cpu.stderr
    software = json.loads((base / "state/software.json").read_text())
    assert software["packages"]["torch"] == "2.10.0+cu126"
    assert all(item["status"] == "COMPLETED" for item in stage_records(base, "cpu").values())
    gpu = run_phase(carc_integration, submitted, "gpu")
    assert gpu.returncode == 0, gpu.stdout + gpu.stderr
    assert set(stage_records(base, "gpu")) == {"gpu-tests", "calibrate", "train", "evaluate", "benchmark"}
    assert all(item["status"] == "COMPLETED" for item in stage_records(base, "gpu").values())
    entries = [json.loads(line) for line in (base / "mock-executed.jsonl").read_text().splitlines()]
    assert [entry["stage"] for entry in entries] == [
        "cpu-tests", "audit", "generate", "gpu-tests", "calibrate", "train", "evaluate", "benchmark",
    ]
    assert all(entry["resume"] == str(base / "train/checkpoints/best.pt") for entry in entries[-2:])
    assert [line.split("|", 1)[0] for line in (root / "mock-srun.txt").read_text().splitlines()] == ["cpu", "gpu"]
    status = controller(carc_integration, "status", "integrated-unit", MOCK_STATUS_JOB="7101")
    assert status.returncode == 0, status.stderr
    assert "setup: COMPLETED" in status.stdout and "benchmark: COMPLETED" in status.stdout
    assert "NOT_STARTED" not in status.stdout
    tower_summaries = [json.loads(path.read_text()) for path in (base / "tower").glob("tdn-*/summary.json")]
    assert {item["job_id"] for item in tower_summaries} == {"7101", "7102"}
    assert all(item["state"] == "COMPLETED" for item in tower_summaries)
    assert status.stdout.count("TDN_TOWER_DIR=") == 2


@pytest.mark.parametrize("code,status", [("17", "FAILED"), ("75", "PAUSED_NEEDS_RESUME")])
def test_actual_helper_marks_train_failure_and_blocks_successors(carc_integration, code, status):
    base, submitted = submit(carc_integration)
    cpu = run_phase(carc_integration, submitted, "cpu")
    assert cpu.returncode == 0, cpu.stdout + cpu.stderr
    gpu = run_phase(
        carc_integration, submitted, "gpu", MOCK_FAIL_STAGE="train", MOCK_FAIL_CODE=code,
    )
    assert gpu.returncode == int(code), gpu.stdout + gpu.stderr
    records = stage_records(base, "gpu")
    assert records["train"]["status"] == status
    assert records["train"]["exit_code"] == int(code)
    assert "evaluate" not in records and "benchmark" not in records
    status_result = controller(carc_integration, "status", "integrated-unit")
    assert status_result.returncode == 0, status_result.stderr
    assert f"First stopping stage: train, exit {code}" in status_result.stdout


def test_actual_gpu_helper_rejects_missing_cpu_phase(carc_integration):
    base, submitted = submit(carc_integration)
    gpu = run_phase(carc_integration, submitted, "gpu")
    assert gpu.returncode != 0
    assert "CPU prerequisites" in gpu.stderr
    assert not (base / "mock-executed.jsonl").exists()


def test_actual_worker_rejects_another_workflows_scheduler_ownership(carc_integration):
    base, submitted = submit(carc_integration)
    cpu = run_phase(carc_integration, submitted, "cpu", MOCK_OWNER_RUN="different-workflow")
    assert cpu.returncode != 0
    assert "ownership" in cpu.stderr
    assert not (base / "mock-executed.jsonl").exists()
    assert not (base / "tower").exists()
