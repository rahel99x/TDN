"""Shell dispatch tests use explicit stubs; they do not certify native Slurm/CUDA."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("fedora_response_diagnostic.sh", "response_diagnostic_local.sh", "response_diagnostic_campaign.sh")


@pytest.fixture
def shell_project(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    (root / "stub-bin").mkdir()
    for name in SCRIPTS:
        shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
    python = root / ".venv/bin/python"
    python.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
if sys.argv[1] == '-':
    os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
root=Path(__file__).resolve().parents[2]
args=sys.argv[1:];script=Path(args[0]).name
with (root/'calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'script':script,'args':args[1:],'pointer':os.environ.get('TDN_RESPONSE_POINTER'),'slurm':os.environ.get('SLURM_JOB_ID')})+'\\n')
command=args[1] if len(args)>1 else ''
if os.environ.get('STUB_FAIL')==script+':'+command:
    print('explicit stub failure',file=sys.stderr);raise SystemExit(37)
if script=='response_workflow.py':
    if command in ('plan','run'):
        run=args[args.index('--run-id')+1] if '--run-id' in args else 'default'
        if command=='run':(root/'runs'/run).mkdir(parents=True)
        print('stub '+command+' '+run)
    elif command=='status' and '--json' in args:
        run=args[2];path=root/('poll-'+run+'.json');count=int(path.read_text()) if path.exists() else 0;path.write_text(str(count+1))
        if os.environ.get('STUB_BAD_JSON'):print('invalid json');raise SystemExit(0)
        failed=os.environ.get('STUB_FAILED_UNITS')=='1';terminal=count>0
        print(json.dumps({'run_id':run,'profile':'diagnostic','terminal':terminal,'completed':terminal and not failed,'report_complete':terminal,'failed_units':['fit'] if failed else [],'units':{'report':{'worker_status':'COMPLETED' if terminal else 'RUNNING'}}}))
    elif command=='validate':
        if os.environ.get('STUB_FAILED_UNITS')=='1':print('{"units":{"fit":{"validation":"INVALID"}}}');raise SystemExit(2)
        print('{"units":{"report":{"validation":"VERIFIED"}}}')
    elif command=='decision':print('{"decision":"STOP_OR_UNRESOLVED","scientific_success":false}')
else:print('stub '+script)
""")
    python.chmod(0o755)
    (root / "scripts/common.sh").write_text("""TDN_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TDN_CARC_ROOT=/nonexistent-carc
tdn_die() { printf '%s\\n' "$*" >&2; exit 2; }
tdn_prepare_env() { mkdir -p "$TDN_REPO_ROOT/.cache"; }
tdn_python() { printf '%s\\n' "$TDN_REPO_ROOT/.venv/bin/python"; }
tdn_inside() { local resolved; resolved="$(realpath -m -- "$1")"; [[ "$resolved" == "$TDN_REPO_ROOT/"* ]] || tdn_die 'outside project'; printf '%s\\n' "$resolved"; }
""")
    sleeper = root / "stub-bin/sleep"
    sleeper.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${STUB_SLEEP_LOG:?}"\n')
    sleeper.chmod(0o755)
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith(("TDN_", "SLURM_", "STUB_")):
            env.pop(name)
    env.update(PATH=str(root / "stub-bin") + os.pathsep + env["PATH"], STUB_SLEEP_LOG=str(root / "sleep.log"))
    return root, env


def launch(project, script, *args, **extra_env):
    root, env = project
    return subprocess.run(["bash", str(root / "scripts" / script), *args], cwd=root,
                          env={**env, **extra_env}, capture_output=True, text=True, timeout=20)


def calls(project):
    path = project[0] / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("script", SCRIPTS)
def test_shell_syntax(script):
    value = subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], capture_output=True, text=True)
    assert value.returncode == 0, value.stderr


def test_native_default_and_exact_argument_forwarding(shell_project):
    assert launch(shell_project, SCRIPTS[0]).returncode == 0
    assert calls(shell_project)[-1]["args"] == ["plan", "--diagnostic-smoke"]
    result = launch(shell_project, SCRIPTS[0], "run", "--diagnostic", "--run-id", "new-run", "--calibration", "cal-run",
                    "--targets", "fd_same_grid", "--trial-seconds", "temporal=7200")
    assert result.returncode == 0, result.stderr
    row = calls(shell_project)[-1]
    assert row["args"] == ["run", "--diagnostic", "--run-id", "new-run", "--calibration", "cal-run",
                            "--targets", "fd_same_grid", "--trial-seconds", "temporal=7200"]
    assert row["pointer"] == ".fedora-response-diagnostic-latest.json"


def test_native_rejects_legacy_profile_and_preserves_failure(shell_project):
    assert launch(shell_project, SCRIPTS[0], "run", "--pilot").returncode == 2
    assert not calls(shell_project)
    result = launch(shell_project, SCRIPTS[0], "plan", STUB_FAIL="response_workflow.py:plan")
    assert result.returncode == 37


def test_compact_latest_uses_only_diagnostic_pointer(shell_project):
    root = shell_project[0]
    (root / "runs").mkdir()
    (root / "runs/.fedora-response-latest.json").write_text('{"run_id":"historical"}')
    (root / "runs/.fedora-response-diagnostic-latest.json").write_text('{"run_id":"diagnostic-new"}')
    result = launch(shell_project, SCRIPTS[0], "compact", "latest", "--mode", "review")
    assert result.returncode == 0, result.stderr
    assert calls(shell_project)[-1]["args"] == ["pack", "diagnostic-new", "--mode", "review"]


def test_local_smoke_retains_project_path_and_targets(shell_project):
    result = launch(shell_project, SCRIPTS[1], "--run-dir", "runs/cpu with spaces", "--targets", "fd_same_grid", "galerkin_same_grid")
    assert result.returncode == 0, result.stderr
    row = calls(shell_project)[-1]
    assert row["script"] == "response_diagnostic_cpu.py"
    assert row["args"] == ["--run-dir", str(shell_project[0] / "runs/cpu with spaces"), "--targets", "fd_same_grid", "galerkin_same_grid"]
    assert row["slurm"] is None and row["pointer"] is None


def test_local_refuses_native_profiles_scheduler_identity_and_existing_paths(shell_project):
    assert launch(shell_project, SCRIPTS[1], "--diagnostic").returncode == 2
    assert launch(shell_project, SCRIPTS[1], SLURM_JOB_ID="invented").returncode == 2
    assert launch(shell_project, SCRIPTS[1], "--run-dir", "../outside").returncode == 2
    (shell_project[0] / "existing").mkdir()
    assert launch(shell_project, SCRIPTS[1], "--run-dir", "existing").returncode == 2
    assert not calls(shell_project)


def test_campaign_fresh_ids_calibration_binding_and_negative_science_are_retained(shell_project):
    result = launch(shell_project, SCRIPTS[2], "--targets", "fd_same_grid", "--updates", "temporal=1200", "--chunk-seconds", "900")
    assert result.returncode == 0, result.stderr
    rows = calls(shell_project)
    workflow = [r for r in rows if r["script"] == "response_workflow.py"]
    submissions = [r for r in workflow if r["args"][0] == "run"]
    assert len(submissions) == 2
    smoke = submissions[0]["args"]; diagnostic = submissions[1]["args"]
    smoke_id = smoke[smoke.index("--run-id") + 1]
    diagnostic_id = diagnostic[diagnostic.index("--run-id") + 1]
    assert smoke_id != diagnostic_id
    assert "--diagnostic-smoke" in smoke and "--diagnostic" in diagnostic
    assert diagnostic[diagnostic.index("--calibration") + 1] == smoke_id
    assert "--updates" not in smoke and "temporal=1200" in diagnostic
    receipts = list((shell_project[0] / "runs/.response-diagnostic-campaigns").glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "COMPLETED" and receipt["smoke_run_id"] == smoke_id
    assert (receipts[0].parent / "diagnostic-decision.json").is_file()
    assert json.loads((receipts[0].parent / "diagnostic-decision.json").read_text())["scientific_success"] is False
    assert (shell_project[0] / "sleep.log").read_text().splitlines() == ["30", "30"]
    assert len([r for r in rows if r["script"] == "compact_review.py"]) == 2


def test_campaign_failed_native_work_exports_then_stops_without_diagnostic(shell_project):
    result = launch(shell_project, SCRIPTS[2], STUB_FAILED_UNITS="1")
    assert result.returncode == 2
    rows = calls(shell_project)
    assert len([r for r in rows if r["script"] == "response_workflow.py" and r["args"][0] == "run"]) == 1
    assert len([r for r in rows if r["script"] == "compact_review.py"]) == 1
    receipt = next((shell_project[0] / "runs/.response-diagnostic-campaigns").glob("*/receipt.json"))
    assert json.loads(receipt.read_text())["status"] == "FAILED"


def test_campaign_ambiguous_submission_is_never_retried(shell_project):
    result = launch(shell_project, SCRIPTS[2], STUB_FAIL="response_workflow.py:run")
    assert result.returncode == 2
    rows = calls(shell_project)
    submissions = [r for r in rows if r["script"] == "response_workflow.py" and r["args"][0] == "run"]
    assert len(submissions) == 1
    assert len([r for r in rows if r["script"] == "compact_review.py"]) == 1
    receipt = next((shell_project[0] / "runs/.response-diagnostic-campaigns").glob("*/receipt.json"))
    value = json.loads(receipt.read_text())
    assert value["status"] == "FAILED"
    assert any("Submission exit 37" in event["detail"] for event in value["events"])


def test_campaign_invalid_status_is_not_a_completion_or_retry(shell_project):
    result = launch(shell_project, SCRIPTS[2], STUB_BAD_JSON="1")
    assert result.returncode != 0
    rows = calls(shell_project)
    assert len([r for r in rows if r["script"] == "response_workflow.py" and r["args"][0] == "run"]) == 1
    assert not (shell_project[0] / "sleep.log").exists()


def test_campaign_failed_plan_stops_before_submission_with_receipt(shell_project):
    result = launch(shell_project, SCRIPTS[2], STUB_FAIL="response_workflow.py:plan")
    assert result.returncode == 37
    assert not [r for r in calls(shell_project) if r["script"] == "response_workflow.py" and r["args"][0] == "run"]
    receipt = next((shell_project[0] / "runs/.response-diagnostic-campaigns").glob("*/receipt.json"))
    assert json.loads(receipt.read_text())["status"] == "FAILED"
