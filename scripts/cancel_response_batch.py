#!/usr/bin/env python3
"""Cancel one frozen Response batch without importing or updating its source.

This standalone standard-library helper can be copied into .runtime while a
different TDN campaign remains active. Default mode only prints a plan.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys
import time
import uuid

TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"})
ACTIVE = frozenset({"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "STAGE_OUT", "SIGNALING", "SPECIAL_EXIT"})
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
JOB = re.compile(r"[1-9][0-9]*\Z")
MAX_BYTES = 16 * 2**20
MAX_JOBS = 20000


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata JSON key")
        result[key] = value
    return result


def _fields(line):
    # Slurm does not consistently quote WorkDir paths containing spaces.
    markers = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*(?::[A-Za-z][A-Za-z0-9_]*)*)=", line))
    result = {}
    for index, marker in enumerate(markers):
        key = marker.group(1)
        if key in result:
            raise ValueError("Repeated scontrol identity field")
        stop = markers[index+1].start() if index+1 < len(markers) else len(line)
        result[key] = line[marker.end():stop].strip()
    return result


def _state(value):
    parts = value.split()
    state = parts[0].rstrip("+") if parts else ""
    if state not in ACTIVE | TERMINAL:
        raise ValueError(f"Unknown scheduler state: {value!r}")
    return state


class BatchCancellation:
    def __init__(self, project, run, *, runner=None):
        if not isinstance(run, str) or not IDENTIFIER.fullmatch(run):
            raise ValueError("Select one exact safe run ID; latest, ranges and name filters are unsupported")
        if run == "latest":
            raise ValueError("Select the literal old run ID, not latest")
        self.root = Path(project).resolve(strict=True)
        self.run = run
        self.runner = runner or subprocess.run
        self.user = pwd.getpwuid(os.geteuid()).pw_name
        self.uid = os.geteuid()
        self.receipt_path = None

    def confined(self, relative, *, exists=True):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Cancellation paths must remain inside the selected project")
        current = self.root
        for part in path.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Cancellation metadata/output paths cannot be symlinks")
        if exists and not current.exists():
            raise ValueError(f"Required batch metadata is missing: {path}")
        return current

    def read(self, relative):
        path = self.confined(relative)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
                raise ValueError("Batch metadata must be a bounded regular file")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(MAX_BYTES+1)
            signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
            if len(raw) > MAX_BYTES or signature(before) != signature(os.fstat(fd)) or signature(before) != signature(path.stat(follow_symlinks=False)):
                raise ValueError("Batch metadata changed while being read")
        finally:
            os.close(fd)
        value = json.loads(raw, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite batch metadata")))
        return value, hashlib.sha256(raw).hexdigest()

    @contextmanager
    def controller_lock(self):
        path = self.confined(".cache/carc-controller.lock")
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Controller lock must be a regular file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("A controller is submitting jobs; retry after submission finishes") from None
            yield
        finally:
            os.close(fd)

    def load(self):
        base = Path("runs")/self.run
        workflow, whash = self.read(base/"response-workflow.json")
        if (not isinstance(workflow, dict) or workflow.get("kind") != "desktop-slurm-response"
                or workflow.get("execution_mode") != "desktop-slurm"
                or workflow.get("run_id") != self.run or workflow.get("root") != str(self.root)
                or workflow.get("run_dir") != str(self.root/base)):
            raise ValueError("Selected metadata is not this project's exact native Response batch")
        site = workflow.get("slurm_profile")
        if (not isinstance(site, dict) or site.get("kind") != "desktop-slurm"
                or site.get("root") != str(self.root) or site.get("user") != self.user):
            raise ValueError("Frozen profile owner/project differs from the invoking user/project")
        profile_hash = hashlib.sha256(json.dumps(site, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if workflow.get("slurm_profile_sha256") != profile_hash:
            raise ValueError("Frozen profile fingerprint differs")
        frozen, phash = self.read(base/"slurm-profile.json")
        if frozen != site:
            raise ValueError("Frozen profile file differs from the workflow")
        submission, shash = self.read(base/"state/submission.json")
        if not isinstance(submission, dict) or submission.get("status") not in {"COMPLETED", "FAILED"}:
            raise ValueError("Submission has not finished; stop or wait for its coordinator before cancelling")
        protocol = workflow.get("scientific_protocol")
        declared = protocol.get("units") if isinstance(protocol, dict) else None
        recorded, jhash = self.read(base/"jobs.json")
        if (not isinstance(declared, dict) or not declared or len(declared) > MAX_JOBS
                or not isinstance(recorded, list) or not recorded or len(recorded) > MAX_JOBS):
            raise ValueError("Missing or excessive frozen batch/job inventory")
        jobs = {}
        stages = set()
        for row in recorded:
            if (not isinstance(row, dict) or not isinstance(row.get("stage"), str)
                    or not IDENTIFIER.fullmatch(row["stage"]) or row["stage"] not in declared
                    or row["stage"] in stages or not isinstance(row.get("job_id"), str)
                    or not JOB.fullmatch(row["job_id"]) or row["job_id"] in jobs):
                raise ValueError("Invalid, duplicate or undeclared batch allocation")
            stages.add(row["stage"])
            jobs[row["job_id"]] = dict(job_id=row["job_id"], stage=row["stage"],
                expected_name="tdn-response-"+row["stage"],
                expected_comment=f"tdn-response:{self.run}:{row['stage']}:{profile_hash}")
        snapshots = {str(base/name): value for name, value in (
            ("response-workflow.json", whash), ("slurm-profile.json", phash),
            ("state/submission.json", shash), ("jobs.json", jhash))}
        return workflow, jobs, snapshots

    def unchanged(self, snapshots):
        for path, expected in snapshots.items():
            if self.read(path)[1] != expected:
                raise ValueError("Batch metadata changed; no further cancellation is allowed")

    def command(self, argv):
        result = self.runner(argv, capture_output=True, text=True, timeout=30)
        if result.returncode or len(result.stdout.encode()) > 32 * 2**20:
            raise ValueError(f"Native {argv[0]} failed: {result.stderr[-2000:]}")
        return result.stdout

    def queue(self, jobs):
        output = self.command(["squeue", "--noheader", "--format=%.40i|%.512k"])
        queued = set()
        for line in output.splitlines():
            cells = line.split("|", 1)
            if len(cells) != 2 or not cells[0].strip():
                raise ValueError("Cannot parse the live scheduler inventory")
            job, comment = (cell.strip() for cell in cells)
            if job in queued:
                raise ValueError("Duplicate live scheduler allocation")
            queued.add(job)
            if comment.startswith(f"tdn-response:{self.run}:") and job not in jobs:
                raise ValueError(f"Unrecorded live allocation {job} belongs to this run; inspect the stopped submission first")
        return queued

    def live(self, expected, site):
        output = self.command(["scontrol", "show", "job", expected["job_id"], "-o"])
        fields = _fields(output.strip())
        required = dict(JobId=expected["job_id"], UserId=f"{self.user}({self.uid})",
            WorkDir=str(self.root), JobName=expected["expected_name"], Comment=expected["expected_comment"])
        if any(fields.get(key) != value for key, value in required.items()):
            raise ValueError(f"Job {expected['job_id']} live owner/root/name/comment differs from the frozen batch")
        if site.get("account") and fields.get("Account") != site["account"]:
            raise ValueError("Live scheduler charging account differs")
        return dict(job_id=expected["job_id"], stage=expected["stage"], state=_state(fields.get("JobState", "")))

    def _inspect(self):
        workflow, jobs, snapshots = self.load()
        queued = self.queue(jobs)
        active = []
        for job in jobs.values():
            if job["job_id"] not in queued:
                continue
            try:
                row = self.live(job, workflow["slurm_profile"])
            except ValueError:
                # A job may finish between squeue and scontrol. Disappearance
                # permits skipping it, never falling back to a guessed identity.
                if job["job_id"] in self.queue(jobs):
                    raise
                continue
            if row["state"] not in TERMINAL:
                active.append(row)
        self.unchanged(snapshots)
        plan = dict(schema="tdn.response-batch-cancellation/v1", run_id=self.run,
            project_root=str(self.root), mode="PLAN", user=self.user,
            recorded_job_count=len(jobs), active_jobs=active,
            skipped_job_ids=sorted(set(jobs)-{r["job_id"] for r in active}, key=int),
            skipped_scope="Absent from the queue or already terminal; no scientific status is inferred",
            metadata_sha256=snapshots, historical_source_tree_sha256=workflow.get("source_tree_sha256"),
            source_comparison_required=False, run_artifacts="PRESERVED", coordinator_signaled=False)
        return plan, workflow, jobs, snapshots

    def inspect(self):
        with self.controller_lock():
            return self._inspect()[0]

    def save(self, value):
        if self.receipt_path is None:
            directory = self.confined(".runtime/response-cancellations", exists=False)
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self.receipt_path = directory/f"{self.run}-{stamp}-{uuid.uuid4().hex[:12]}.json"
        payload = {**value, "receipt_path": str(self.receipt_path),
            "updated_at": datetime.now(timezone.utc).isoformat()}
        temporary = self.receipt_path.with_suffix(".partial")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, self.receipt_path)
        return payload

    def terminal_states(self, targets):
        if not targets:
            return {}
        output = self.command(["sacct", "-nP", "-j", ",".join(targets), "--format=JobIDRaw,State%40"])
        states = {}
        for line in output.splitlines():
            cells = line.strip().split("|")
            if cells[-1:] == [""]:
                cells.pop()
            if not cells or cells[0] not in targets:
                continue  # Ignore .batch/.extern steps and unrelated allocations.
            if len(cells) != 2 or cells[0] in states:
                raise ValueError("Ambiguous exact allocation accounting")
            states[cells[0]] = _state(cells[1])
        return states

    def execute(self, *, wait_seconds=120, poll_seconds=2):
        if not 0 <= wait_seconds <= 300 or not 0 < poll_seconds <= 10:
            raise ValueError("Use wait seconds in [0,300] and poll seconds in (0,10]")
        value = None
        try:
            with self.controller_lock():
                value, workflow, jobs, snapshots = self._inspect()
                # Revalidate every active identity before the single exact-ID
                # cancellation, and validate the entire batch before acting.
                queued = self.queue(jobs)
                targets = []
                for row in value["active_jobs"]:
                    job = row["job_id"]
                    if job in queued and self.live(jobs[job], workflow["slurm_profile"])["state"] not in TERMINAL:
                        targets.append(job)
                self.unchanged(snapshots)
                value.update(mode="EXECUTE", status="CANCELLING", requested_job_ids=targets)
                value = self.save(value)
                if targets:
                    self.command(["scancel", *targets])
            # Release the shared submission lock before polling; other batches
            # remain free to submit work while these allocations terminate.
            deadline = time.monotonic()+wait_seconds
            while True:
                self.unchanged(snapshots)
                queued = self.queue(jobs)
                states = self.terminal_states(targets)
                pending = sorted((set(jobs)&queued) | {job for job in targets if states.get(job) not in TERMINAL}, key=int)
                value.update(terminal_states=states, outstanding_job_ids=pending)
                if not pending:
                    value.update(status="TERMINAL" if targets else "NO_ACTIVE_JOBS",
                        cancelled_batch_scope="Selected recorded allocations only; original science is not reclassified",
                        accounting_scope="Terminal accounting checked for newly requested cancellation IDs; queue absence alone does not classify historical jobs")
                    return self.save(value)
                if time.monotonic() >= deadline:
                    raise ValueError("Cancellation requested but terminal accounting is not yet verified; retain the receipt and inspect the outstanding jobs")
                time.sleep(min(poll_seconds, max(0, deadline-time.monotonic())))
        except (ValueError, OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if value is not None and self.receipt_path is not None:
                self.save({**value, "status": "UNVERIFIED", "error": str(error)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Actual checkout root; its source is not modified")
    parser.add_argument("--run", required=True, help="Literal old Response run ID; never latest or a job range")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Read-only preview (default)")
    mode.add_argument("--execute", action="store_true", help="Cancel only independently verified active allocations")
    parser.add_argument("--wait-seconds", type=float, default=120)
    parser.add_argument("--poll-seconds", type=float, default=2)
    args = parser.parse_args(argv)
    try:
        helper = BatchCancellation(args.project, args.run)
        result = helper.execute(wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds) if args.execute else helper.inspect()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"TDN Response cancellation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
