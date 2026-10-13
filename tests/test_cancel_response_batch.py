"""Exact-batch cancellation guards, using private recorded Slurm responses."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess

import pytest

from scripts import cancel_response_batch as cancellation


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+"\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inventory(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class RecordedScheduler:
    """Only declared private records are queried; scancel mutates no run files."""
    def __init__(self, project, run, site_hash):
        self.calls = []
        self.cancelled = []
        self.stuck = False
        self.fail_command = None
        self.on_cancel = None
        self.on_accounting = None
        self.accounting_overrides = {}
        self.queue = {"100", "101", "900", "901"}
        user = pwd.getpwuid(os.geteuid()).pw_name
        self.records = {}
        for job, stage in (("100", "audit"), ("101", "trial-old-000"), ("102", "report")):
            self.records[job] = dict(JobId=job, JobName="tdn-response-"+stage,
                UserId=f"{user}({os.geteuid()})", JobState="PENDING" if job == "100" else "RUNNING",
                WorkDir=str(project), Comment=f"tdn-response:{run}:{stage}:{site_hash}", Account="(null)")
        self.records["900"] = dict(JobId="900", JobName="tdn-tangent-response-audit", Comment="tdn-tangent-response:other:audit:hash")
        self.records["901"] = dict(JobId="901", JobName="samso", Comment="unrelated")
        self.queue_comments = {job: row["Comment"] for job, row in self.records.items()}
        self.accounting = {"102": "COMPLETED"}

    def __call__(self, command, **kwargs):
        command = list(map(str, command))
        self.calls.append(command)
        name = Path(command[0]).name
        code, output, error = 0, "", ""
        if self.fail_command == name:
            code, error = 1, "recorded scheduler failure"
        elif name == "squeue":
            output = "".join(f"{job}|{self.queue_comments[job]}\n" for job in sorted(self.queue))
        elif name == "scontrol":
            assert command[1:3] == ["show", "job"]
            job = command[3]
            if job not in self.queue:
                code, error = 1, "slurm_load_jobs error: Invalid job id specified"
            else:
                output = " ".join(f"{key}={value}" for key, value in self.records[job].items())+"\n"
        elif name == "scancel":
            # Broad flags, shell interpolation, names and users are forbidden.
            assert command[1:] and all(value.isdecimal() and int(value) > 0 for value in command[1:])
            self.cancelled.extend(command[1:])
            if self.on_cancel is not None:
                self.on_cancel()
            if not self.stuck:
                for job in command[1:]:
                    self.queue.discard(job)
                    self.accounting[job] = self.accounting_overrides.get(job, "CANCELLED")
        elif name == "sacct":
            if self.on_accounting is not None:
                self.on_accounting()
            jobs = command[command.index("-j")+1].split(",")
            output = "".join(f"{job}|{self.accounting[job]}\n" for job in jobs if job in self.accounting)
        else:
            pytest.fail(f"Unexpected command: {command}")
        result = subprocess.CompletedProcess(command, code, output, error)
        if code and kwargs.get("check"):
            raise subprocess.CalledProcessError(code, command, output=output, stderr=error)
        return result


@pytest.fixture
def batch(tmp_path):
    project = tmp_path/"checkout with spaces"
    run = "response-old-owned"
    base = project/"runs"/run
    user = pwd.getpwuid(os.geteuid()).pw_name
    site = dict(schema_version=1, kind="desktop-slurm", root=str(project), user=user,
                account=None, cpu_partition="local", gpu_partition="local", gpu_gres="gpu:1")
    site_hash = digest(site)
    protocol = dict(profile="diagnostic-smoke", units={name: dict(kind="train") for name in ("audit", "trial-old-000", "report")})
    workflow = dict(schema_version=1, kind="desktop-slurm-response", execution_mode="desktop-slurm",
                    profile="diagnostic-smoke", root=str(project), run_id=run, run_dir=str(base),
                    slurm_profile=site, slurm_profile_sha256=site_hash, slurm_profile_path=str(base/"slurm-profile.json"),
                    scientific_protocol=protocol, protocol_path=str(base/"protocol.json"), protocol_sha256=digest(protocol),
                    source_sha256="a"*64, source_tree_sha256="b"*64)
    put(base/"response-workflow.json", workflow)
    put(base/"slurm-profile.json", site)
    put(base/"protocol.json", protocol)
    put(base/"jobs.json", [dict(stage=stage, job_id=job) for job, stage in (("100", "audit"), ("101", "trial-old-000"), ("102", "report"))])
    put(base/"state"/"submission.json", dict(status="COMPLETED"))
    (project/".cache").mkdir()
    (project/".cache"/"carc-controller.lock").touch()
    (base/"trial-old-000").mkdir()
    (base/"trial-old-000"/"selected.pt").write_bytes(b"private preservation sentinel; never loaded")
    (project/"scripts").mkdir()
    (project/"scripts"/"source.txt").write_text("current source need not match the old campaign\n")
    return project, run, base, workflow, RecordedScheduler(project, run, site_hash)


def test_plan_reads_only_exact_batch_and_ignores_tangent_and_samso(batch):
    project, run, base, _, scheduler = batch
    before = inventory(project)
    plan = cancellation.BatchCancellation(project, run, runner=scheduler).inspect()
    assert {row["job_id"] for row in plan["active_jobs"]} == {"100", "101"}
    assert set(plan["skipped_job_ids"]) == {"102"}
    assert not scheduler.cancelled
    assert all(command[0] != "scancel" for command in scheduler.calls)
    assert all(command[3] not in {"900", "901"} for command in scheduler.calls if command[0] == "scontrol")
    assert inventory(project) == before


def test_cli_defaults_to_plan_and_requires_explicit_execute(batch, monkeypatch, capsys):
    project, run, _, _, scheduler = batch
    before = inventory(project)
    monkeypatch.setattr(cancellation.subprocess, "run", scheduler)
    assert cancellation.main(["--project", str(project), "--run", run]) == 0
    assert "100" in capsys.readouterr().out
    assert not scheduler.cancelled and inventory(project) == before
    assert cancellation.main(["--project", str(project), "--run", run, "--execute", "--wait-seconds", "0"]) == 0
    assert set(scheduler.cancelled) == {"100", "101"}


def test_execute_cancels_only_verified_owned_ids_preserves_evidence_and_is_idempotent(batch):
    project, run, base, _, scheduler = batch
    before = inventory(base)
    api = cancellation.BatchCancellation(project, run, runner=scheduler)
    receipt = api.execute(wait_seconds=0, poll_seconds=.01)
    assert receipt and set(scheduler.cancelled) == {"100", "101"}
    assert {"900", "901"} <= scheduler.queue
    assert inventory(base) == before
    first_receipt = api.receipt_path.read_bytes()
    repeated = cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0, poll_seconds=.01)
    assert len(scheduler.cancelled) == 2
    assert repeated["status"] == "NO_ACTIVE_JOBS"
    assert api.receipt_path.read_bytes() == first_receipt
    assert inventory(base) == before
    assert (project/".runtime"/"response-cancellations").is_dir()


@pytest.mark.parametrize("field,value", [
    ("UserId", "somebody_else(424242)"), ("UserId", f"{pwd.getpwuid(os.geteuid()).pw_name}(424242)"),
    ("JobId", "999"), ("JobName", "tdn-tangent-response-audit"),
    ("Comment", "tdn-response:other:trial-old-000:hash"), ("WorkDir", "/wrong/checkout"),
    ("JobState", "UNKNOWN"),
])
def test_all_live_identities_must_validate_before_any_cancel(batch, field, value):
    project, run, base, _, scheduler = batch
    before = inventory(base)
    # The first job is valid; a later mismatch must prevent all action.
    scheduler.records["101"][field] = value
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert not scheduler.cancelled and inventory(base) == before


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_stage", "malformed_id", "empty_stage", "jobs_not_list"])
def test_malformed_or_duplicate_submission_records_abort_before_action(batch, change):
    project, run, base, _, scheduler = batch
    jobs = json.loads((base/"jobs.json").read_text())
    if change == "duplicate_id": jobs[1]["job_id"] = jobs[0]["job_id"]
    elif change == "duplicate_stage": jobs[1]["stage"] = jobs[0]["stage"]
    elif change == "malformed_id": jobs[1]["job_id"] = "101;scancel 900"
    elif change == "empty_stage": jobs[1]["stage"] = ""
    else: jobs = {"100": "audit"}
    put(base/"jobs.json", jobs)
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert not scheduler.cancelled


@pytest.mark.parametrize("change", ["tangent_kind", "wrong_run", "wrong_root", "wrong_owner", "profile_hash", "profile_file", "malformed_protocol"])
def test_frozen_workflow_and_profile_identity_are_required(batch, change):
    project, run, base, workflow, scheduler = batch
    workflow = copy.deepcopy(workflow)
    if change == "tangent_kind": workflow["kind"] = "desktop-slurm-tangent-response"
    elif change == "wrong_run": workflow["run_id"] = "some-other-run"
    elif change == "wrong_root": workflow["root"] = str(project.parent)
    elif change == "wrong_owner":
        workflow["slurm_profile"]["user"] = "someone_else"
        workflow["slurm_profile_sha256"] = digest(workflow["slurm_profile"])
        put(base/"slurm-profile.json", workflow["slurm_profile"])
    elif change == "profile_hash": workflow["slurm_profile_sha256"] = "f"*64
    elif change == "malformed_protocol": workflow["scientific_protocol"] = None
    else:
        changed = copy.deepcopy(workflow["slurm_profile"]);changed["account"] = "another"
        put(base/"slurm-profile.json", changed)
    put(base/"response-workflow.json", workflow)
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert not scheduler.cancelled


def test_source_changes_do_not_block_stopping_the_named_batch(batch):
    project, run, base, _, scheduler = batch
    (project/"scripts"/"source.txt").write_text("updated after the original jobs began\n")
    cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert set(scheduler.cancelled) == {"100", "101"}


def test_held_requeued_special_exit_is_active_and_is_cancelled(batch):
    project, run, _, _, scheduler = batch
    scheduler.records["100"]["JobState"] = "SPECIAL_EXIT"
    api = cancellation.BatchCancellation(project, run, runner=scheduler)
    plan = api.inspect()
    assert any(row["job_id"] == "100" and row["state"] == "SPECIAL_EXIT" for row in plan["active_jobs"])
    receipt = api.execute(wait_seconds=0)
    assert receipt["status"] == "TERMINAL" and set(scheduler.cancelled) == {"100", "101"}


def test_matching_batch_comment_on_unrecorded_job_aborts_without_cancelling(batch):
    project, run, _, workflow, scheduler = batch
    scheduler.queue.add("777")
    scheduler.queue_comments["777"] = f"tdn-response:{run}:unrecorded:{workflow['slurm_profile_sha256']}"
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert not scheduler.cancelled


def test_frozen_metadata_change_during_preflight_blocks_all_cancellation(batch):
    project, run, base, _, scheduler = batch
    def changed(command, **kwargs):
        completed = scheduler(command, **kwargs)
        if command[0] == "scontrol" and command[3] == "101":
            jobs = json.loads((base/"jobs.json").read_text())
            jobs[0]["job_id"] = "777"
            put(base/"jobs.json", jobs)
        return completed
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=changed).execute(wait_seconds=0)
    assert not scheduler.cancelled


@pytest.mark.parametrize("status", ["RUNNING", "UNKNOWN"])
def test_unfinished_submission_cannot_be_cancelled_from_incomplete_job_registry(batch, status):
    project, run, base, _, scheduler = batch
    put(base/"state"/"submission.json", dict(status=status))
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert not scheduler.cancelled


def test_unknown_post_cancel_accounting_state_remains_unverified(batch):
    project, run, base, _, scheduler = batch
    before = inventory(base)
    scheduler.accounting_overrides["100"] = "UNKNOWN"
    api = cancellation.BatchCancellation(project, run, runner=scheduler)
    with pytest.raises(ValueError):
        api.execute(wait_seconds=0)
    assert set(scheduler.cancelled) == {"100", "101"}
    assert json.loads(api.receipt_path.read_text())["status"] == "UNVERIFIED"
    assert inventory(base) == before


def test_declared_account_must_match_live_jobs(batch):
    project, run, base, workflow, scheduler = batch
    workflow = copy.deepcopy(workflow)
    workflow["slurm_profile"]["account"] = "expected_account"
    workflow["slurm_profile_sha256"] = digest(workflow["slurm_profile"])
    put(base/"response-workflow.json", workflow)
    put(base/"slurm-profile.json", workflow["slurm_profile"])
    for job in ("100", "101", "102"):
        stage = scheduler.records[job]["JobName"].removeprefix("tdn-response-")
        scheduler.records[job]["Comment"] = f"tdn-response:{run}:{stage}:{workflow['slurm_profile_sha256']}"
        scheduler.queue_comments[job] = scheduler.records[job]["Comment"]
        scheduler.records[job]["Account"] = "expected_account"
    scheduler.records["101"]["Account"] = "another_account"
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert not scheduler.cancelled


@pytest.mark.parametrize("command", ["squeue", "scontrol"])
def test_scheduler_read_failures_abort_without_any_cancellation(batch, command):
    project, run, _, _, scheduler = batch
    scheduler.fail_command = command
    with pytest.raises(ValueError):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert not scheduler.cancelled


@pytest.mark.parametrize("command", ["sacct", "scancel"])
def test_cancel_or_confirmation_failure_preserves_unverified_receipt(batch, command):
    project, run, base, _, scheduler = batch
    before = inventory(base)
    scheduler.fail_command = command
    api = cancellation.BatchCancellation(project, run, runner=scheduler)
    with pytest.raises(ValueError):
        api.execute(wait_seconds=0)
    receipt = json.loads(api.receipt_path.read_text())
    assert receipt["status"] == "UNVERIFIED" and "scheduler failure" in receipt["error"]
    assert set(receipt["requested_job_ids"]) == {"100", "101"}
    assert inventory(base) == before
    assert set(scheduler.cancelled) == ({"100", "101"} if command == "sacct" else set())


def test_submission_lock_covers_cancellation_but_is_released_before_waiting(batch):
    project, run, _, _, scheduler = batch
    path = project/".cache"/"carc-controller.lock"
    observations = []
    def inspect_lock(expect_locked):
        with path.open("r+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locked = True
            else:
                locked = False
            observations.append(locked)
            assert locked == expect_locked
    scheduler.on_cancel = lambda: inspect_lock(True)
    scheduler.on_accounting = lambda: inspect_lock(False)
    cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert observations == [True, False]


def test_nonterminal_or_missing_accounting_after_cancel_is_not_reported_as_success(batch):
    project, run, base, _, scheduler = batch
    scheduler.stuck = True
    before = inventory(base)
    with pytest.raises(ValueError, match="(?i)(timeout|timed out|terminal|waiting)"):
        cancellation.BatchCancellation(project, run, runner=scheduler).execute(wait_seconds=0)
    assert set(scheduler.cancelled) == {"100", "101"}
    assert inventory(base) == before and {"900", "901"} <= scheduler.queue
