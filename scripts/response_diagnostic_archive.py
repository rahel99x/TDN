#!/usr/bin/env python3
"""Preserve one terminal native diagnostic campaign in a project-local archive.

This metadata-only command never validates historical science against today's
source, executes uploaded code, deletes files, cancels jobs, or resumes trials.
"""
from __future__ import annotations

import argparse
import ctypes
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdn.runtime.desktop_slurm import load_profile, validate_profile, _fields

SCHEMA = "tdn.response-diagnostic-archive/v1"
CAMPAIGNS = "runs/.response-diagnostic-campaigns"
POINTER = "runs/.fedora-response-diagnostic-latest.json"
TERMINAL = frozenset(("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                     "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"))
RECEIPT_TERMINAL = TERMINAL | {"INTERRUPTED"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
JOB = re.compile(r"[1-9][0-9]*\Z")
MAX_METADATA = 16 * 2**20
MAX_JOBS = 20000
MAX_CAMPAIGNS = 1024


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Expected one safe campaign/run identifier")
    return value


def confined(root, path, *, exists=False):
    """Exact lexical path under root, rejecting symlink ancestors/leaves."""
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if ".." in path.parts or not path.is_relative_to(root):
        raise ValueError("Archive path leaves the project")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Archive metadata/root may not be a symlink: {current}")
        if current.exists() and current != path and not current.is_dir():
            raise ValueError("Archive path has a non-directory ancestor")
    if exists and not path.exists():
        raise ValueError(f"Required archive evidence is missing: {path}")
    return path


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate metadata JSON key")
        value[key] = item
    return value


def read_json(root, path, limit=MAX_METADATA):
    path = confined(root, path, exists=True)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Archive metadata must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        named = path.stat(follow_symlinks=False)
        signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if len(raw) > limit or signature(info) != signature(after) or signature(after) != signature(named):
            raise ValueError("Archive metadata changed during inspection")
    finally:
        os.close(fd)
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite metadata")))
    return value, dict(path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                       bytes=len(raw))


def native_policy(root):
    # Keep the established local/CARC, allocation, Conda and project-root checks.
    from scripts import fedora_workflow
    if Path(root).resolve() != fedora_workflow.ROOT.resolve():
        raise ValueError("Archive policy must run in the actual checkout")
    fedora_workflow.controller_policy()
    return load_profile(root=root)


@contextmanager
def controller_lock(root):
    path = confined(root, ".cache/carc-controller.lock")
    path.parent.mkdir(exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Controller lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A controller is submitting/managing jobs; archive after it finishes") from None
        yield
    finally:
        os.close(fd)


class NativeScheduler:
    """Read-only real Slurm queries; no status or ownership is inferred from files."""
    def command(self, args):
        value = subprocess.run(args, capture_output=True, text=True, timeout=30,
                               env={**os.environ, "SLURM_TIME_FORMAT": "standard"})
        if value.returncode or len(value.stdout.encode()) > 32 * 2**20:
            raise ValueError(f"Cannot obtain bounded native scheduler evidence from {args[0]}")
        return value.stdout

    def observe(self, jobs, *, user, root, run_ids):
        # Also catch a job accepted by sbatch before its ID reached jobs.json.
        live = self.command(["squeue", "-h", "-u", user, "-o", "%.40i|%.512k|%.256j|%.4096Z"])
        prefixes = tuple("tdn-response:" + name + ":" for name in run_ids)
        ids = {job["job_id"] for job in jobs}
        for line in live.splitlines():
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) != 4:
                raise ValueError("Ambiguous live scheduler inventory")
            if (cells[0] in ids or cells[1].startswith(prefixes)
                    or not cells[1] and cells[2].startswith("tdn-response-") and cells[3] == str(root)):
                raise ValueError(f"Campaign still has a live submitted job {cells[0]}; preserve all directories")
        records = {}
        for start in range(0, len(jobs), 128):
            selected = jobs[start:start+128]
            output = self.command(["sacct", "-nP", "-j", ",".join(j["job_id"] for j in selected),
                "--format=JobIDRaw,State%40,User,Account,JobName%256,Comment%512,WorkDir%4096"])
            expected = {j["job_id"] for j in selected}
            for line in output.splitlines():
                cells = line.split("|")
                if cells and cells[-1] == "":
                    cells.pop()
                if not cells or cells[0] not in expected:
                    continue  # Batch/extern steps are not allocation identities.
                if len(cells) != 7 or cells[0] in records:
                    raise ValueError("Ambiguous exact scheduler allocation record")
                records[cells[0]] = dict(zip(("job_id", "state", "user", "account", "job_name", "comment", "workdir"), cells))
        if set(records) != {j["job_id"] for j in jobs}:
            raise ValueError("Scheduler accounting is missing an exact submitted job; no archive moves allowed")
        for row in records.values():
            if row["comment"] == "(null)":
                row["comment"] = ""
            if not all(row.get(k) for k in ("user", "job_name", "workdir", "comment")):
                # Terminal accounting remains authoritative for lifecycle. Use
                # retained controller records only for absent ownership fields.
                try:
                    retained = subprocess.run(["scontrol", "show", "job", row["job_id"], "-o"],
                        capture_output=True, text=True, timeout=10)
                except (OSError,subprocess.TimeoutExpired):
                    retained = None
                if retained is not None and retained.returncode == 0:
                    fields = _fields(retained.stdout)
                    if fields.get("JobId") != row["job_id"]:
                        raise ValueError("Retained scheduler job identity differs")
                    retained_state = fields.get("JobState", "UNKNOWN").split()[0].rstrip("+")
                    accounting_state = row["state"].split()[0].rstrip("+")
                    if retained_state not in TERMINAL or retained_state != accounting_state:
                        raise ValueError("Retained scheduler lifecycle is nonterminal or contradicts accounting")
                    extra = dict(user=fields.get("UserId", "").split("(")[0], account=fields.get("Account", ""),
                        job_name=fields.get("JobName", ""), comment=fields.get("Comment", ""), workdir=fields.get("WorkDir", ""))
                    for key, value in extra.items():
                        if value == "(null)":
                            value = ""
                        if value and row.get(key) and row[key] != value:
                            raise ValueError("Retained and accounting scheduler ownership differ")
                        if value and not row.get(key):
                            row[key] = value
            row["comment_coverage"] = "VERIFIED" if row["comment"] else "NOT_RETAINED_BY_SCHEDULER"
        return records


def discover(root, *, campaign=None, run_id=None):
    base = confined(root, CAMPAIGNS, exists=True)
    if not base.is_dir():
        raise ValueError("Campaign receipt inventory is not a directory")
    if campaign:
        path = confined(root, base/identifier(campaign)/"receipt.json", exists=True)
        return path, *read_json(root, path)
    entries = sorted(base.iterdir())
    if len(entries) > MAX_CAMPAIGNS:
        raise ValueError("Campaign inventory exceeds its bound; select --campaign explicitly")
    candidates = []
    for folder in entries:
        confined(root, folder, exists=True)
        if not folder.is_dir() or not (folder/"receipt.json").exists():
            continue
        value, fingerprint = read_json(root, folder/"receipt.json")
        if value.get("schema") != "tdn.response-diagnostic-campaign/v1":
            raise ValueError("Unexpected campaign receipt; select an explicit valid campaign")
        if run_id and run_id not in (value.get("smoke_run_id"), value.get("diagnostic_run_id")):
            continue
        try:
            created = datetime.fromisoformat(value["created_at"])
            if created.tzinfo is None:
                raise ValueError("Campaign timestamp lacks timezone")
        except (KeyError, TypeError, ValueError):
            raise ValueError("Campaign creation time invalid; select --campaign explicitly") from None
        candidates.append((created, folder.name, folder/"receipt.json", value, fingerprint))
    if not candidates or (run_id and len(candidates) != 1):
        raise ValueError("No unique campaign receipt for selection; use --campaign TOKEN")
    _, _, path, value, fingerprint = max(candidates, key=lambda x: (x[0], x[1]))
    return path, value, fingerprint


def raven_association(root, receipt, metadata):
    binding = receipt.get("raven_launch")
    if binding is None:
        return dict(association="NOT_RECORDED", action="PRESERVE_ALL_OUTER_RAVEN_RECORDS",
                    reason="No exact campaign association; no Raven run guessed or moved")
    if not isinstance(binding, dict) or binding.get("project_root") != str(root):
        raise ValueError("Raven coordinator association has a different project")
    name = identifier(binding.get("coordinator_run_id"))
    token = identifier(binding.get("launch_token"))
    if name != "launch-"+token+"-a1" or binding.get("coordinator_run_dir") != "runs/"+name:
        raise ValueError("Raven coordinator association is not the exact launch attempt")
    base = confined(root, binding["coordinator_run_dir"], exists=True)
    run, one = read_json(root, base/"run.json")
    summary, two = read_json(root, base/"summary.json")
    metadata.extend((one, two))
    automation = run.get("metadata", {}).get("automation", {})
    plan = automation.get("plan", {})
    if (run.get("schema") != "raven.run/v1" or run.get("run_id") != name or run.get("job_id") is not None
            or automation.get("execution") != "local" or automation.get("project_root") != str(root)
            or plan.get("token") != token or plan.get("project_root") != str(root)
            or plan.get("script_relative") != "scripts/response_diagnostic_campaign.sh"
            or run.get("provenance", {}).get("script") != "scripts/response_diagnostic_campaign.sh"):
        raise ValueError("Raven association does not identify this exact local diagnostic coordinator")
    if (run.get("state") not in RECEIPT_TERMINAL or summary.get("schema") != "raven.summary/v1"
            or summary.get("id") != name or summary.get("state") not in RECEIPT_TERMINAL
            or run["state"] != summary["state"]):
        raise ValueError("Associated Raven coordinator must have matching terminal run and summary records")
    return dict(association="EXACT", action="PRESERVED", project_root=str(root),
                coordinator_run_id=name, coordinator_run_dir=binding["coordinator_run_dir"],
                launch_token=token, state=run["state"],
                reason="Outer Raven launch records and source snapshots retain original absolute paths")


def inspect(root, *, campaign=None, run_id=None, scheduler=None, site=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Project root missing")
    site = validate_profile(site, root=root) if site is not None else native_policy(root)
    receipt_path, receipt, fingerprint = discover(root, campaign=campaign, run_id=run_id)
    if (receipt.get("schema") != "tdn.response-diagnostic-campaign/v1" or receipt.get("project_root") != str(root)
            or receipt.get("status") not in RECEIPT_TERMINAL):
        raise ValueError("Campaign coordinator receipt must be terminal and belong to this checkout")
    token = identifier(receipt_path.parent.name)
    names = (identifier(receipt.get("smoke_run_id")), identifier(receipt.get("diagnostic_run_id")))
    if names != ("response-diagnostic-smoke-"+token, "response-diagnostic-"+token):
        raise ValueError("Campaign receipt run IDs do not match its exact token")
    metadata, moves, jobs, absent = [fingerprint], [], [], []
    for name, profile in zip(names, ("diagnostic-smoke", "diagnostic")):
        base = confined(root, "runs/"+name)
        if not base.exists():
            absent.append(name)
            continue
        if not base.is_dir():
            raise ValueError("Diagnostic run root must be a real directory")
        workflow, mark = read_json(root, base/"response-workflow.json")
        metadata.append(mark)
        if (workflow.get("kind") != "desktop-slurm-response" or workflow.get("run_id") != name
                or workflow.get("profile") != profile or workflow.get("run_dir") != str(base)):
            raise ValueError("Run root is not this receipt's declared native diagnostic")
        frozen_site = validate_profile(workflow.get("slurm_profile"), root=root)
        if frozen_site["user"] != site["user"]:
            raise ValueError("Native diagnostic owner differs from the current profile")
        profile_hash = workflow.get("slurm_profile_sha256")
        if not isinstance(profile_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", profile_hash):
            raise ValueError("Diagnostic frozen profile identity missing")
        declared = workflow.get("scientific_protocol", {}).get("units")
        if not isinstance(declared, dict) or len(declared) > MAX_JOBS:
            raise ValueError("Diagnostic unit inventory missing or exceeds archive bound")
        recorded, mark = read_json(root, base/"jobs.json")
        metadata.append(mark)
        if not isinstance(recorded, list) or len(recorded) > MAX_JOBS:
            raise ValueError("Diagnostic job inventory invalid or exceeds archive bound")
        stages = set()
        for row in recorded:
            if (not isinstance(row, dict) or row.get("stage") not in declared or row["stage"] in stages
                    or not isinstance(row.get("job_id"), str) or not JOB.fullmatch(row["job_id"])):
                raise ValueError("Invalid or repeated diagnostic allocation identity")
            stages.add(row["stage"])
            jobs.append(dict(job_id=row["job_id"], stage=row["stage"], run_id=name,
                             expected_comment=f"tdn-response:{name}:{row['stage']}:{profile_hash}",
                             expected_name="tdn-response-"+row["stage"], expected_account=frozen_site.get("account")))
        moves.append(dict(kind="native_run", original="runs/"+name, archived="native/"+name))
    if len(jobs) > MAX_JOBS or len({r["job_id"] for r in jobs}) != len(jobs):
        raise ValueError("Selected campaign allocations duplicate or exceed the archive bound")
    raven = raven_association(root, receipt, metadata)
    observed = (scheduler or NativeScheduler()).observe(jobs, user=site["user"], root=root, run_ids=names)
    proof = []
    for job in jobs:
        row = observed.get(job["job_id"], {})
        state = row.get("state", "UNKNOWN").split()[0].rstrip("+")
        if state not in TERMINAL:
            raise ValueError(f"Job {job['job_id']} is {state}; only actual terminal submitted jobs can be archived")
        if (row.get("user") != site["user"] or row.get("workdir") != str(root)
                or row.get("job_name") != job["expected_name"]
                or row.get("comment") and row["comment"] != job["expected_comment"]
                or job["expected_account"] is not None and row.get("account") != job["expected_account"]):
            raise ValueError(f"Job {job['job_id']} scheduler ownership/workdir/comment differs from its frozen diagnostic")
        proof.append({**row, "run_id":job["run_id"], "stage":job["stage"], "state":state,
                      "comment_coverage":"VERIFIED" if row.get("comment") else "NOT_RETAINED_BY_SCHEDULER"})
    moves.append(dict(kind="campaign_receipt_directory", original=receipt_path.parent.relative_to(root).as_posix(),
                      archived="campaign/"+token))
    pointer = confined(root, POINTER)
    if pointer.exists():
        value, mark = read_json(root, pointer, limit=65536)
        metadata.append(mark)
        if value.get("run_id") in names:
            moves.append(dict(kind="diagnostic_latest_pointer", original=POINTER,
                              archived="pointers/.fedora-response-diagnostic-latest.json"))
    for move in moves:
        source = confined(root, move["original"], exists=True)
        move["source_identity"] = [source.stat().st_dev, source.stat().st_ino]
    return dict(schema=SCHEMA, campaign_token=token, coordinator_status=receipt["status"], run_ids=list(names),
                absent_run_ids=absent, moves=moves, inspected_metadata=metadata, submitted_jobs=len(jobs),
                scheduler_state_counts=dict(Counter(row["state"] for row in proof)), scheduler_proof=proof,
                scheduler_comment_coverage=dict(Counter(row["comment_coverage"] for row in proof)),
                raven_coordinator=raven, native_source_fingerprints="PRESERVED_WITHOUT_CURRENT_SOURCE_COMPARISON",
                preserved_outside_scope=["runs/review-exports", ".raven/launches", ".raven/wrappers", "all unrelated runs and pointers"],
                historical_path_contract="Frozen original absolute paths unchanged; archived runs are evidence, not resumable coordinators")


def _write(path, value, limit=32*2**20):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    if len(raw) > limit:
        raise ValueError("Archive manifest/proof exceeds its bound")
    temporary = path.with_name(path.name+".partial-"+uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(raw).hexdigest()


def _directory_fd(root, path):
    path = confined(root, path, exists=True)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.relative_to(root).parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def rename_noreplace(root, source, target, expected_identity=None):
    """Linux/Fedora atomic no-overwrite rename, through confined parent fds."""
    source, target = confined(root, source, exists=True), confined(root, target)
    left = _directory_fd(root, source.parent)
    right = None
    try:
        right = _directory_fd(root, target.parent)
        item = os.stat(source.name, dir_fd=left, follow_symlinks=False)
        if stat.S_ISLNK(item.st_mode) or expected_identity is not None and [item.st_dev,item.st_ino] != expected_identity:
            raise ValueError("Archive source root identity changed before rename")
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc,"renameat2",None)
        if rename is None:
            raise ValueError("Atomic no-overwrite rename is unavailable; no archive moves allowed")
        rename.argtypes = (ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(left,os.fsencode(source.name),right,os.fsencode(target.name),1) != 0:
            error = ctypes.get_errno()
            raise OSError(error,os.strerror(error),str(target))
    finally:
        os.close(left)
        if right is not None:
            os.close(right)


def archive(root, *, campaign=None, run_id=None, plan=False, scheduler=None, site=None):
    root = Path(root).resolve()
    with controller_lock(root):
        selection = inspect(root, campaign=campaign, run_id=run_id, scheduler=scheduler, site=site)
        preview = {k:v for k,v in selection.items() if k != "scheduler_proof"}
        if plan:
            return dict(preview, action="PLAN_ONLY", archive_destination="runs/archived-response-diagnostics/<fresh-id>")
        # Re-read metadata and real scheduler evidence immediately before mutation.
        repeated = inspect(root, campaign=selection["campaign_token"], scheduler=scheduler, site=site)
        if repeated != selection:
            raise ValueError("Campaign metadata/scheduler evidence changed; no archive moves allowed")
        # A scheduler call can take time; check bytes again after the final call.
        for expected in selection["inspected_metadata"]:
            _, current = read_json(root,expected["path"])
            if current != expected:
                raise ValueError("Campaign metadata changed during the final scheduler inspection")
        parent = confined(root, "runs/archived-response-diagnostics")
        parent.mkdir(exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:12]
        destination = confined(root, parent/name)
        destination.mkdir(mode=0o700)  # exclusive, never reuse or overwrite archives
        if any(confined(root, row["original"]).stat().st_dev != destination.stat().st_dev for row in selection["moves"]):
            raise ValueError("Archive moves must stay on the same project filesystem")
        proof_hash = _write(destination/"scheduler-proof.json", selection.pop("scheduler_proof"))
        manifest = dict(selection, action="ARCHIVE", status="MOVING", created_at=datetime.now(timezone.utc).isoformat(),
                        archive_directory=destination.relative_to(root).as_posix(), completed_moves=[],
                        scheduler_proof=dict(path="scheduler-proof.json", sha256=proof_hash))
        manifest_path = destination/"archive-manifest.json"
        _write(manifest_path, manifest, limit=128*2**10)
        moved = []
        try:
            for move in selection["moves"]:
                source = confined(root, move["original"], exists=True)
                target = confined(root, destination/move["archived"])
                if target.exists() or [source.stat().st_dev, source.stat().st_ino] != move["source_identity"]:
                    raise ValueError("Archive destination exists or source root identity changed")
                target.parent.mkdir(parents=True, exist_ok=True)
                rename_noreplace(root, source, target, move["source_identity"])
                moved.append((source, target, move["source_identity"]))
                manifest["completed_moves"].append(move["original"])
                _write(manifest_path, manifest, limit=128*2**10)
        except BaseException:
            restored, retained = [], []
            for source, target, expected_identity in reversed(moved):
                if not source.exists() and not source.is_symlink() and target.exists():
                    try:
                        rename_noreplace(root, target, source, expected_identity)
                        restored.append(source.relative_to(root).as_posix())
                    except (OSError,ValueError):
                        retained.append(target.relative_to(root).as_posix())
                else:
                    retained.append(target.relative_to(root).as_posix())
            manifest.update(status="INTERRUPTED_PRESERVED", restored_paths=restored, retained_archive_paths=retained)
            _write(manifest_path, manifest, limit=128*2**10)
            raise
        manifest.update(status="ARCHIVED", finished_at=datetime.now(timezone.utc).isoformat())
        _write(manifest_path, manifest, limit=128*2**10)
        return dict(manifest, manifest_path=manifest_path.relative_to(root).as_posix())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--campaign", help="Exact campaign receipt token")
    choice.add_argument("--run-id", help="Find the unique receipt associated with this diagnostic run")
    parser.add_argument("--plan", action="store_true", help="Check native terminal evidence and list intended moves")
    args = parser.parse_args(argv)
    try:
        result = archive(ROOT, campaign=identifier(args.campaign) if args.campaign else None,
                         run_id=identifier(args.run_id) if args.run_id else None, plan=args.plan)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"TDN diagnostic archive: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
