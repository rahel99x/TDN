#!/usr/bin/env python3
"""Retain a diagnostic campaign inside Raven's exact local launch attempt.

This stdlib-only bridge records the submitting coordinator. Native science,
GPU verification and terminal launch state remain owned by their existing
workers and Raven supervisor. It never imports Raven or loads a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid

ROOT = Path(__file__).absolute().parent.parent
SCRIPT = "scripts/response_diagnostic_campaign.sh"
SMALL_BYTES = 1 << 20
DECISION_BYTES = 64 << 20
SCRIPT_BYTES = 8 << 20
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[0-9a-f]{32}\Z")
JOB_ID = re.compile(r"[1-9][0-9]*(?:_[0-9]+)?(?:\+[0-9]+)?\Z")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}


def relative(value):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or not value.isprintable() or value.startswith("/")
            or any(c in value for c in "\\*?[]")
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError("Campaign pointers must be exact confined relative POSIX paths")
    return value


def confined(value, root=ROOT):
    path, root = Path(value).absolute(), Path(root).absolute()
    if ".." in path.parts or ".." in root.parts or not path.is_relative_to(root):
        raise ValueError("Campaign reporting path is outside its declared root")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("Campaign reporting paths cannot traverse symlinks")
    return path


def read_bytes(value, *, root=ROOT, limit=SMALL_BYTES):
    path = confined(value, root)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"Campaign reporting requires a bounded regular file: {path} "
                             f"({before.st_size} bytes; limit {limit}; regular={stat.S_ISREG(before.st_mode)})")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    signature = lambda value:(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if (len(raw) != before.st_size or signature(before) != signature(after)
            or signature(after) != signature(path.stat(follow_symlinks=False))):
        raise ValueError("Campaign reporting source changed during inspection")
    return raw


def read_json(path, *, root=ROOT, limit=SMALL_BYTES):
    raw = read_bytes(path, root=root, limit=limit)
    def unique(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("Duplicate campaign reporting JSON key")
            value[key] = child
        return value
    def nonfinite(value):
        raise ValueError("Nonfinite campaign reporting JSON value")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("Campaign reporting metadata must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def write_json(path, value, *, root, limit=SMALL_BYTES):
    """Atomically write bounded metadata; larger inventories need an explicit cap."""
    path = confined(path, root)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(raw) > limit:
        raise ValueError(f"Retained coordinator metadata exceeds its small-file budget: {path} "
                         f"({len(raw)} bytes; limit {limit})")
    path.parent.mkdir(parents=True, exist_ok=True)
    confined(path, root)
    temporary = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        confined(path, root)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def identity(value, label):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid "+label+" identity")
    return value


def context(snapshot=None, *, script=SCRIPT):
    """Validate Raven's frozen launch, original checkout and current attempt."""
    if not os.environ.get("RAVEN_LAUNCH_PLAN"):
        return None
    if any(os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT")):
        raise ValueError("Run the campaign coordinator outside Slurm with raven launch --local")
    root = confined(ROOT)
    expected_root = str(root)
    if os.environ.get("RAVEN_PROJECT_ROOT") != expected_root:
        raise ValueError("Raven project root must be the exact TDN checkout")
    relative(script)
    source = confined(root/script, root)
    if os.environ.get("RAVEN_SOURCE_SCRIPT") != str(source):
        raise ValueError("Raven original source script does not belong to this campaign")
    plan_path = confined(os.environ["RAVEN_LAUNCH_PLAN"], root)
    plan, plan_hash = read_json(plan_path, root=root)
    token = plan.get("token")
    if not isinstance(token, str) or not TOKEN.fullmatch(token):
        raise ValueError("Raven launch token is invalid")
    folder = root/".raven"/"wrappers"/token
    if plan_path != folder/"plan.json":
        raise ValueError("Raven launch plan has a different project membership")
    if (plan.get("schema") != "raven.launch-plan/v1" or plan.get("execution") != "local"
            or plan.get("project_root") != expected_root or plan.get("script") != str(source)
            or plan.get("script_relative") != script):
        raise ValueError("The diagnostic coordinator requires its exact local Raven launch plan")
    checksum = plan.get("script_sha256")
    if not isinstance(checksum, str) or not SHA256.fullmatch(checksum):
        raise ValueError("Raven captured script identity is invalid")
    payload = folder/"payload.sh"
    if plan.get("source_snapshot") != payload.relative_to(root).as_posix():
        raise ValueError("Raven captured payload path differs from the launch plan")
    if snapshot is not None and confined(snapshot, root) != payload:
        raise ValueError("Campaign must execute the exact captured Raven payload")
    for script in (source, payload):
        if hashlib.sha256(read_bytes(script, root=root, limit=SCRIPT_BYTES)).hexdigest() != checksum:
            raise ValueError("Captured or current campaign source differs from its Raven source SHA256")
    python = plan.get("report_python")
    if (not isinstance(python, str) or not Path(python).is_absolute()
            or os.environ.get("RAVEN_REPORT_PYTHON") != python or not os.access(python, os.X_OK)):
        raise ValueError("Raven reporting Python must match its frozen executable")
    helper = confined(folder/"report.py", root)
    bundle = confined(folder/"runtime.pyz", root)
    if (os.environ.get("RAVEN_REPORT_HELPER") != str(helper)
            or os.environ.get("RAVEN_RUNTIME_BUNDLE") != str(bundle)
            or plan.get("runtime_bundle") != str(bundle)):
        raise ValueError("Raven reporting helper must match the frozen private runtime")
    if hashlib.sha256(read_bytes(bundle, root=root, limit=DECISION_BYTES)).hexdigest() != plan.get("runtime_sha256"):
        raise ValueError("Raven private runtime differs from its frozen SHA256")
    expected_helper = ("import sys\nsys.path.insert(0, "+repr(str(bundle))+ ")\n"
                       "from raven.research_reporting import main\nraise SystemExit(main())\n").encode()
    if read_bytes(helper, root=root) != expected_helper:
        raise ValueError("Raven reporting helper differs from its captured runtime entry point")
    run_id = identity(os.environ.get("RAVEN_RUN_ID"), "coordinator run")
    if run_id != "launch-"+token+"-a1" or plan.get("run_id_base") != "launch-"+token:
        raise ValueError("Raven coordinator run differs from its launch token")
    run_dir = confined(root/"runs"/run_id, root)
    if os.environ.get("RAVEN_RUN_DIR") != str(run_dir):
        raise ValueError("Raven coordinator run directory has changed")
    run, _ = read_json(run_dir/"run.json", root=root)
    automation = run.get("metadata", {}).get("automation", {})
    binding = automation.get("plan", {})
    if (run.get("schema") != "raven.run/v1" or run.get("run_id") != run_id
            or run.get("state") != "RUNNING" or run.get("job_id") is not None
            or automation.get("execution") != "local" or automation.get("project_root") != expected_root
            or binding.get("token") != token or binding.get("plan_sha256") != plan_hash
            or binding.get("plan_path") != plan_path.relative_to(root).as_posix()
            or run.get("provenance", {}).get("script_sha256") != checksum):
        raise ValueError("Raven exact coordinator inventory does not match the frozen local launch")
    return dict(root=root, run_dir=run_dir, run_id=run_id, token=token,
                report_python=python, report_helper=helper)


def association(ctx):
    return dict(project_root=str(ctx["root"]), coordinator_run_id=ctx["run_id"],
                coordinator_run_dir=ctx["run_dir"].relative_to(ctx["root"]).as_posix(), launch_token=ctx["token"])


def receipt_data(ctx, path):
    path = confined(path, ctx["root"])
    receipt, checksum = read_json(path, root=ctx["root"])
    if (receipt.get("schema") != "tdn.response-diagnostic-campaign/v1"
            or receipt.get("project_root") != str(ctx["root"])
            or receipt.get("raven_launch") != association(ctx)):
        raise ValueError("Campaign receipt does not describe this exact Raven coordinator")
    smoke = identity(receipt.get("smoke_run_id"), "native calibration")
    diagnostic = identity(receipt.get("diagnostic_run_id"), "native diagnostic")
    if (not smoke.startswith("response-diagnostic-smoke-")
            or not diagnostic.startswith("response-diagnostic-")
            or diagnostic.startswith("response-diagnostic-smoke-") or smoke == diagnostic):
        raise ValueError("Campaign receipt requires distinct native diagnostic run identities")
    phase = receipt.get("phase")
    if not isinstance(phase, str) or not phase or not phase.isprintable() or len(phase) > 128:
        raise ValueError("Invalid campaign phase")
    if receipt.get("status") not in ("RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"):
        raise ValueError("Invalid campaign status")
    if not isinstance(receipt.get("events"), list):
        raise ValueError("Campaign events must be an observed list")
    return path, receipt, checksum


def helper(ctx, *arguments):
    subprocess.run([ctx["report_python"], "-I", str(ctx["report_helper"]), *arguments], check=True)


def register_artifact(ctx, name, *, optional=False):
    run, _ = read_json(ctx["run_dir"]/"run.json", root=ctx["root"])
    pointer = run.get("paths", {}).get("contract")
    if pointer:
        contract_path = confined(ctx["run_dir"]/relative(pointer), ctx["run_dir"])
        if contract_path.exists():
            contract, _ = read_json(contract_path, root=ctx["run_dir"])
            for output in contract.get("outputs", []):
                if output.get("path") == name:
                    if output.get("format") != "json" or output.get("required") is not (not optional):
                        raise ValueError("Campaign artifact declaration has changed")
                    return
    helper(ctx, "artifact", relative(name), "--format", "json", *( ["--optional"] if optional else []))


def register_log(ctx, suffix):
    name = "campaign."+suffix
    pointer = "logs/"+name+".log"
    path = confined(ctx["run_dir"]/pointer, ctx["run_dir"])
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
    else:
        read_bytes(path, root=ctx["run_dir"], limit=DECISION_BYTES)
    index, _ = read_json(ctx["run_dir"]/"logs.json", root=ctx["root"])
    for row in index.get("logs", []):
        if row.get("id") == name:
            if row.get("path") != pointer:
                raise ValueError("Campaign log identity has a different path")
            return
    helper(ctx, "log", name, pointer, "--label", "Campaign "+suffix)


def research_paths(ctx, receipt, path, checksum):
    result = dict(schema="tdn.response-diagnostic-research-paths/v1",
        path_base="TDN project root; child projects retain their own exact native attempt identities",
        coordinator_scope="Submission and evidence collection only; numerical metrics live in the named native research projects",
        campaign_receipt=dict(path=relative(path.relative_to(ctx["root"]).as_posix()), sha256=checksum), runs={})
    for phase, key in (("calibration", "smoke_run_id"), ("diagnostic", "diagnostic_run_id")):
        base = "runs/"+receipt[key]
        pointers = dict(run_id=receipt[key], raven_project=base+"/raven",
            project_catalog=base+"/raven/project_catalog.json",
            portable_research_bundle=base+"/report/response-diagnostic-raven.zip",
            portable_research_manifest=base+"/report/response-diagnostic-raven.manifest.json")
        for field, pointer in pointers.items():
            if field != "run_id":
                confined(ctx["root"]/relative(pointer), ctx["root"])
        result["runs"][phase] = pointers
    return result


def decision_excerpt(ctx, receipt_path):
    path = confined(receipt_path.parent/"diagnostic-decision.json", ctx["root"])
    if not path.exists():
        return None
    source, checksum = read_json(path, root=ctx["root"], limit=DECISION_BYTES)
    value = {key:source[key] for key in ("decision", "eligible_for_development", "historical_pilot_gate",
             "historical_gate_policy", "no_development_authorization", "summary") if key in source}
    assessments = source.get("assessments", {})
    if not isinstance(assessments, dict):
        raise ValueError("Diagnostic decision assessments must be a mapping")
    value["assessments"] = {key:assessments[key] for key in ("historical_gate", "variant_outcomes") if key in assessments}
    value.update(schema="tdn.response-diagnostic-campaign-decision/v1",
        source=dict(path=relative(path.relative_to(ctx["root"]).as_posix()), sha256=checksum),
        export_scope="Decision, summary and variant outcomes; complete fit cells remain in the exact native source")
    return value


def status_counts(ctx, receipt, source, *, expected=None):
    status_path = confined(source, ctx["root"])
    status, checksum = read_json(status_path, root=ctx["root"], limit=DECISION_BYTES)
    if expected is not None and checksum != expected:
        raise ValueError("Retained native status source SHA256 changed; publish its new exact snapshot")
    if status.get("run_id") not in (receipt["smoke_run_id"], receipt["diagnostic_run_id"]):
        raise ValueError("Native status belongs to a different campaign")
    units, failed = status.get("units"), status.get("failed_units")
    if (not isinstance(units, dict) or not isinstance(failed, list)
            or any(not isinstance(x, str) or x not in units for x in failed) or len(failed) != len(set(failed))
            or type(status.get("terminal")) is not bool or type(status.get("completed")) is not bool):
        raise ValueError("Native status requires exact unit and infrastructure completion records")
    counts = dict(registered_units=len(units), submitted_units=0, completed_units=0,
                  failed_units=len(failed), terminal_jobs=0, worker_completed_units=0, worker_failed_units=0)
    for name, unit in units.items():
        identity(name, "native unit")
        if not isinstance(unit, dict):
            raise ValueError("Native unit status must be a mapping")
        job = unit.get("job_id")
        if job is not None and (not isinstance(job, str) or not JOB_ID.fullmatch(job)):
            raise ValueError("Native status job identity must be an actual scheduler ID")
        scheduler, worker = unit.get("scheduler_state"), unit.get("worker_status")
        if not isinstance(scheduler, str) or not isinstance(worker, str):
            raise ValueError("Native unit requires scheduler and worker states")
        counts["submitted_units"] += int(job is not None)
        counts["terminal_jobs"] += int(job is not None and scheduler in TERMINAL)
        counts["worker_completed_units"] += int(worker in ("COMPLETED", "INHERITED"))
        counts["worker_failed_units"] += int(worker in ("FAILED", "INTERRUPTED"))
        counts["completed_units"] += int(scheduler == "COMPLETED" and worker in ("COMPLETED", "INHERITED")
                                           and unit.get("artifact_validation") == "SEALED_UNVERIFIED")
    return counts, dict(run_id=status["run_id"], terminal=status["terminal"], completed=status["completed"],
        source=dict(path=relative(status_path.relative_to(ctx["root"]).as_posix()), sha256=checksum),
        counts=counts, scope="Observed infrastructure states; sealed status is not a scientific acceptance decision")


def publish(ctx, receipt, *, status_file=None):
    path, value, checksum = receipt_data(ctx, receipt)
    output = ctx["run_dir"]/"outputs"
    paths = research_paths(ctx, value, path, checksum)
    observed = dict(campaign_events_recorded=len(value["events"]),
                    campaign_completed=int(value["status"] == "COMPLETED"),
                    campaign_failed=int(value["status"] == "FAILED"))
    progress = []
    expected_status = None
    expected_run = None
    previous = output/"research_paths.json"
    if status_file is None and previous.exists():
        retained, _ = read_json(previous, root=ctx["run_dir"])
        latest = retained.get("latest_native_status")
        if latest is not None:
            if not isinstance(latest, dict) or not isinstance(latest.get("source"), dict):
                raise ValueError("Retained native status reference must be an exact source mapping")
            source = latest["source"]
            expected_status = source.get("sha256")
            expected_run = latest.get("run_id")
            if not isinstance(expected_status, str) or not SHA256.fullmatch(expected_status):
                raise ValueError("Retained native status reference requires its source SHA256")
            status_file = confined(ctx["root"]/relative(source.get("path")), ctx["root"])
    if status_file is not None:
        counts, status = status_counts(ctx, value, status_file, expected=expected_status)
        if expected_run is not None and status["run_id"] != expected_run:
            raise ValueError("Retained native status run identity has changed")
        paths["latest_native_status"] = status
        observed.update(counts)
        if counts["registered_units"]:
            progress = ["--completed", str(counts["terminal_jobs"]), "--total", str(counts["registered_units"]),
                        "--unit", "registered native allocations"]
    decision = decision_excerpt(ctx, path)
    if decision is not None:
        summary = decision.get("summary", {})
        if not isinstance(summary, dict):
            raise ValueError("Diagnostic decision summary must be a mapping")
        for key in ("registered_trials", "completed_trials", "missing_trials", "failed_trials", "time_censored_trials",
                    "no_headroom_trials", "achieved_updates", "requested_updates"):
            number = summary.get(key)
            if number is not None:
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                    raise ValueError("Recorded diagnostic research counts must be nonnegative and finite")
                observed["research_"+key] = number
    # Validate the entire observation before replacing any retained view.
    write_json(output/"campaign.json", value, root=ctx["run_dir"])
    write_json(output/"research_paths.json", paths, root=ctx["run_dir"])
    if decision is not None:
        write_json(output/"diagnostic_decision.json", decision, root=ctx["run_dir"])
    helper(ctx, "metric", "--phase", "campaign/"+value["phase"], "--step", str(len(value["events"])),
           *[argument for key, number in observed.items() for argument in ("--value", key+"="+str(number))], *progress)


def initialize(ctx, receipt):
    receipt_data(ctx, receipt)
    for suffix in ("stdout", "stderr"):
        register_log(ctx, suffix)
    publish(ctx, receipt)
    register_artifact(ctx, "outputs/campaign.json")
    register_artifact(ctx, "outputs/research_paths.json")
    register_artifact(ctx, "outputs/diagnostic_decision.json", optional=True)


def main(argv=None):
    if not os.environ.get("RAVEN_LAUNCH_PLAN"):
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check");check.add_argument("--snapshot", type=Path, required=True)
    init = commands.add_parser("initialize");init.add_argument("--receipt", type=Path, required=True)
    init.add_argument("--snapshot", type=Path, required=True)
    update = commands.add_parser("publish");update.add_argument("--receipt", type=Path, required=True)
    update.add_argument("--status-file", type=Path)
    args = parser.parse_args(argv)
    try:
        ctx = context(getattr(args, "snapshot", None))
        if args.command == "initialize":initialize(ctx, args.receipt)
        elif args.command == "publish":publish(ctx, args.receipt, status_file=args.status_file)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError, subprocess.CalledProcessError) as error:
        print("TDN diagnostic Raven: "+str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
