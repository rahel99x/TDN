#!/usr/bin/env python3
"""Source-bound R7 development study in Raven's genuine desktop Slurm worker.

Preparation is read-only with respect to numerical evidence. The native worker
verifies implementation before opening the bounded scientific bank. Completion
does not promote historical gates, scientific utility, novelty, or GPU readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import response_preconditioner_native as common
from response_diagnostic_raven import confined, read_bytes, read_json, write_json

SCRIPT = "scripts/response_preconditioner_r7.sh"
SCHEMA = "tdn.response-preconditioner-r7-native/v1"
DECISION = "BOUNDED_DEVELOPMENT"
CONTRACT_FOLDER = ".runtime/response-preconditioner-r7-launches"
identity = common.identity
venv_lock = common.venv_lock
now = common.now
report = common.report
execute = common.execute


def context(snapshot):
    return common.context(snapshot, script=SCRIPT)


def prepare():
    if any(os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT", "RAVEN_LAUNCH_PLAN")):
        raise ValueError("Prepare outside active Slurm/Raven attempts; submission is a separate action")
    with venv_lock():
        from tdn.analysis.response_preconditioner.r7_protocol import protocol
        bound = identity()
        token = uuid.uuid4().hex
        path = confined(ROOT / CONTRACT_FOLDER / (token + ".json"), ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        confined(ROOT / "logs/response-preconditioner-r7", ROOT).mkdir(parents=True, exist_ok=True)
        payload = dict(schema=SCHEMA, token=token, created_at=now(), project_root=str(ROOT),
                       identity=bound, device="cpu", decision=DECISION, scientific_promotion=False,
                       modes=["all", "verify"], protocols={name: protocol(name) for name in ("verify", "study")},
                       allocation=dict(cpus=4, memory_mib=8192, wall_seconds=1200))
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False); stream.write("\n")
        path.chmod(0o444)
    print(path.relative_to(ROOT).as_posix())


def load_contract(path):
    path = confined(ROOT / path if not Path(path).is_absolute() else path, ROOT)
    value, checksum = read_json(path, root=ROOT)
    token = value.get("token", "")
    if (value.get("schema") != SCHEMA or value.get("project_root") != str(ROOT)
            or not re.fullmatch(r"[0-9a-f]{32}", token)
            or path != ROOT / CONTRACT_FOLDER / (token + ".json")
            or value.get("device") != "cpu" or value.get("scientific_promotion") is not False
            or value.get("decision") != DECISION or value.get("modes") != ["all", "verify"]
            or value.get("allocation") != dict(cpus=4, memory_mib=8192, wall_seconds=1200)):
        raise ValueError("Not this checkout's frozen bounded R7 native contract")
    from tdn.analysis.response_preconditioner.r7_protocol import protocol
    if value.get("protocols") != {name: protocol(name) for name in ("verify", "study")}:
        raise ValueError("R7 protocol changed after preparation; create a fresh contract")
    return value, checksum


def resource_check(allocation):
    common.resource_check(allocation)
    job = allocation["job"]
    tres = dict(x.split("=", 1) for x in job.get("AllocTRES", "").split(",") if "=" in x)
    memory = re.fullmatch(r"([0-9]+)([KMGT]?)", tres.get("mem", ""))
    memory_mib = int(memory[1]) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[memory[2]]
    wall = sum(int(part) * 60 ** i for i, part in enumerate(reversed(job["TimeLimit"].split(":"))))
    if int(job["NumCPUs"]) != 4 or memory_mib > 8192 or wall > 1200:
        raise ValueError("R7 is frozen to four CPUs, at most 8 GiB and 20 minutes")


def report_batch(ctx, commands):
    """Amortize startup while calling the exact captured Raven reporting API.

    Data travels as JSON stdin, never interpolated Python or shell source. The
    private runtime and interpreter have already passed context() verification.
    """
    if not commands:
        return
    if len(commands) > 8192:
        raise ValueError("R7 reporting command count exceeds its frozen transport cap")
    payload = json.dumps(commands, allow_nan=False)
    if len(payload.encode()) > 8 << 20:
        raise ValueError("R7 reporting payload exceeds 8 MiB")
    code = ("import json,sys\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "from raven.research_reporting import main\n"
            "for args in json.load(sys.stdin):\n"
            "    result=main(args)\n"
            "    if result not in (None,0): raise SystemExit(result)\n")
    subprocess.run([ctx["report_python"], "-I", "-c", code, ctx["plan"]["runtime_bundle"]],
                   input=payload, text=True, check=True, timeout=120, stdout=subprocess.DEVNULL)


def artifact_commands(ctx, folder):
    folder = confined(folder, ctx["run"])
    commands = []
    if folder.exists():
        for path in sorted(folder.rglob("*")):
            if "pytest-temp" in path.relative_to(folder).parts:
                # Test-only copied checkouts intentionally exercise symlinks;
                # verification.xml/log retain their outcomes, not their fixtures.
                continue
            if path.is_symlink():
                raise ValueError("R7 portable evidence may not contain symlinks")
            if path.is_file():
                confined(path, ctx["run"])
                fmt = None if path.suffix == ".npz" else "json" if path.suffix == ".json" else "csv" if path.suffix == ".csv" else "text"
                commands.append(["artifact", path.relative_to(ctx["run"]).as_posix(),
                                 *(["--format", fmt] if fmt else [])])
                if len(commands) > 2048:
                    raise ValueError("R7 stage exceeds 2048 portable artifacts")
    return commands


def stage_evidence(ctx, output, command, exit_code):
    # Failed and partial queries are equally part of the retained evidence.
    report_batch(ctx, artifact_commands(ctx, output))
    if exit_code:
        return
    summary, _ = read_json(output / "summary.json", root=ROOT)
    if (summary.get("command") != command or summary.get("status") != "COMPLETED"
            or summary.get("scientific_promotion") is not False or summary.get("device") != "cpu"
            or summary.get("decision") != DECISION):
        raise ValueError("Completed R7 child lacks matching nonpromoting CPU evidence")
    required = ("protocol.json", "source.json", "software.json", "artifacts.json")
    required += (("verification.xml", "checks.json") if command == "verify" else
                 ("metrics.csv", "metrics.json", "selection.json", "references.json", "timing.json"))
    if any(not (output / name).is_file() for name in required):
        raise ValueError("Completed R7 child is missing required declared output evidence")


def publish_metrics(ctx, output):
    path = output / "metrics.json"
    if not path.is_file():
        return
    rows = json.loads(read_bytes(path, root=ROOT, limit=8 << 20))
    if not isinstance(rows, list) or len(rows) > 4096:
        raise ValueError("Expected bounded R7 metric rows")
    commands = []
    keys = ("rms", "maximum", "mean_error", "centered_rms", "start", "stop",
            "reference_uncertainty_rms", "reference_uncertainty_maximum", "correction_rms",
            "correction_relative_rms", "elapsed_seconds", "U_minus_P_rms", "U_minus_G_rms",
            "U_minus_P_uncertainty_rms", "U_minus_G_uncertainty_rms",
            "U_minus_P_relative_error", "U_minus_G_relative_error", "K_minus_P_rms")
    for step, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("R7 metric rows must be objects")
        phase = "r7/" + "/".join(str(row.get(key, "unknown")) for key in ("split", "role", "family", "method"))
        if len(phase) > 160:
            raise ValueError("R7 method identity exceeds Raven phase contract")
        args = ["metric", "--phase", phase, "--step", str(step)]
        count = 0
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("R7 measurements must be finite numerical values")
            args += ["--value", key + "=" + str(value)]; count += 1
        if "reference_accepted" in row:
            if not isinstance(row["reference_accepted"], bool):
                raise ValueError("Reference acceptance must be Boolean")
            args += ["--value", "reference_accepted=" + str(int(row["reference_accepted"]))]; count += 1
        if count:
            commands.append(args)
    report_batch(ctx, commands)


def publish_timing(ctx, output):
    """Retain every timing attempt, including unsuccessful paired rounds."""
    incremental = output / "timing_records.json"
    completed = output / "timing.json"
    if incremental.is_file():
        rows = json.loads(read_bytes(incremental, root=ROOT, limit=8 << 20))
        if not isinstance(rows, list) or len(rows) > 24 * 256:
            raise ValueError("Expected bounded incremental R7 timing records")
        grouped = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("context_id"), str):
                raise ValueError("R7 timing records must retain their context identity")
            grouped.setdefault(row["context_id"], []).append(row)
        contexts = [dict(context_id=key, records=value) for key, value in grouped.items()]
    elif completed.is_file():
        contexts = json.loads(read_bytes(completed, root=ROOT, limit=8 << 20))
    else:
        return
    if not isinstance(contexts, list) or len(contexts) > 24:
        raise ValueError("Expected at most 24 declared R7 timing contexts")
    commands = []
    for context_index, context_record in enumerate(contexts):
        records = context_record.get("records", [])
        if not isinstance(records, list) or len(records) > 256:
            raise ValueError("Expected bounded R7 timing rounds")
        for row in records:
            phase = "r7/timing/" + str(row["phase"]) + "/" + str(row["method"])
            args = ["metric", "--phase", phase, "--step", str(len(commands)),
                    "--value", "context_index=" + str(context_index),
                    "--value", "success=" + str(int(row.get("status") == "PASS"))]
            for key, name in (("seconds", "bundle_seconds"), ("round", "round"), ("position", "position")):
                value = row.get(key)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("R7 timings must be finite numerical values")
                    args += ["--value", name + "=" + str(value)]
            commands.append(args)
    report_batch(ctx, commands)


def release_verification(native, output, checksum):
    """Record exact accepted verification bytes before the study is allowed."""
    hashes = {name: hashlib.sha256(read_bytes(output / name, root=ROOT, limit=8 << 20)).hexdigest()
              for name in ("summary.json", "checks.json", "verification.xml", "artifacts.json")}
    write_json(native / "verification-release.json", dict(schema=SCHEMA, contract_sha256=checksum,
               verification=output.relative_to(native.parent.parent).as_posix(), hashes=hashes), root=ROOT)


def require_study_context(contract_path, output):
    """CLI admission uses real scheduler checks, never environment flags alone."""
    ctx = context(os.environ.get("TDN_RESPONSE_PRECONDITIONER_R7_SNAPSHOT", ""))
    if output is None or confined(output, ROOT) != ctx["run"] / "outputs/study":
        raise ValueError("R7 study output must be this Raven attempt's fresh outputs/study")
    contract, checksum = load_contract(contract_path)
    if identity() != contract["identity"]:
        raise ValueError("Source/software/profile changed after preparation")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.desktop_slurm import verify_allocation
    verify_runtime("cpu", "response-preconditioner-r7")
    resource_check(verify_allocation("cpu", root=ROOT))
    release, _ = read_json(ctx["run"] / "outputs/native/verification-release.json", root=ROOT)
    if (release.get("schema") != SCHEMA or release.get("contract_sha256") != checksum
            or release.get("verification") != "outputs/verification"):
        raise ValueError("R7 study requires successful verification in this exact source-bound attempt")
    verification = ctx["run"] / "outputs/verification"
    expected = {name: hashlib.sha256(read_bytes(verification / name, root=ROOT, limit=8 << 20)).hexdigest()
                for name in ("summary.json", "checks.json", "verification.xml", "artifacts.json")}
    if release.get("hashes") != expected:
        raise ValueError("R7 verification evidence changed before scientific acquisition")
    return ctx


def run(args):
    ctx = context(os.environ.get("TDN_RESPONSE_PRECONDITIONER_R7_SNAPSHOT", ""))
    native = confined(ctx["run"] / "outputs/native", ctx["run"])
    native.mkdir(exist_ok=False)
    state = dict(schema=SCHEMA, status="RUNNING", created_at=now(), device="cpu", decision=DECISION,
                 scientific_promotion=False, commands=[], historical_gate="B STOP is unchanged",
                 source_unchanged=None, telemetry_scope="Actual worker process tree; outer Raven supervisor is separate")
    started = time.monotonic(); code = 2; observer = None
    try:
        with venv_lock():
            contract, checksum = load_contract(args.contract)
            state["contract_sha256"] = checksum
            if identity() != contract["identity"]:
                state["source_unchanged"] = False
                raise ValueError("Source, software or profile changed after preparation; prepare a fresh launch")
            from tdn.runtime.preflight import verify_runtime
            from tdn.runtime.desktop_slurm import verify_allocation
            from tdn.runtime.storage import configure_storage
            verify_runtime("cpu", "response-preconditioner-r7")
            allocation = verify_allocation("cpu", root=ROOT)
            resource_check(allocation); configure_storage()
            state["allocation"] = allocation
            state["cpu_affinity"] = sorted(os.sched_getaffinity(0))
            # Preserve original bytes, so uploaded receipts can verify both hashes.
            contract_file = ROOT / args.contract if not Path(args.contract).is_absolute() else Path(args.contract)
            for name, source, expected_sha in (("contract.json", contract_file, checksum),
                    ("profile.json", ROOT / contract["identity"]["profile_path"], contract["identity"]["profile_sha256"])):
                raw = read_bytes(source, root=ROOT)
                if hashlib.sha256(raw).hexdigest() != expected_sha:
                    raise ValueError("Frozen contract/profile bytes changed during native preflight")
                with (native / name).open("xb") as stream:
                    stream.write(raw)
            from tdn.analysis.tangent_response.telemetry import start_recording
            observer = start_recording(native, dict(families=["cpu", "memory"], interval_seconds=1.))
            commands = ("verify", "study") if args.mode == "all" else ("verify",)
            code = 0
            for step, command in enumerate(commands, 1):
                if identity() != contract["identity"]:
                    state["source_unchanged"] = False
                    raise ValueError("Source/software/profile changed before stage; preserve this attempt")
                output = ctx["run"] / "outputs" / ("verification" if command == "verify" else "study")
                argv = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/response_preconditioner_r7.py"),
                        command, "--device", "cpu", "--output", str(output)]
                if command == "study":
                    argv += ["--contract", args.contract]
                begin = time.monotonic()
                child = execute(ctx, command, argv)
                elapsed = time.monotonic() - begin
                state["commands"].append(dict(command=command, argv=argv, exit_code=child.returncode,
                                              elapsed_seconds=elapsed, output=output.relative_to(ctx["run"]).as_posix()))
                write_json(native / "status.json", state, root=ROOT)
                stage_evidence(ctx, output, command, child.returncode)
                if command == "study":
                    publish_metrics(ctx, output)
                    publish_timing(ctx, output)
                report(ctx, "metric", "--phase", "r7/" + command, "--step", str(step),
                       "--value", "stage_seconds=" + str(elapsed),
                       "--value", "stage_exit_code=" + str(child.returncode),
                       "--completed", str(step), "--total", str(len(commands)), "--unit", "bounded R7 stages")
                if identity() != contract["identity"]:
                    state["source_unchanged"] = False
                    raise ValueError("Source/software/profile changed during stage; evidence cannot promote")
                state["source_unchanged"] = True
                if child.returncode:
                    code = child.returncode if child.returncode > 0 else 128 - child.returncode
                    break
                if command == "verify":
                    release_verification(native, output, checksum)
            state["status"] = "COMPLETED" if code == 0 else "FAILED"
    except Exception as error:
        state.update(status="FAILED", error=type(error).__name__ + ": " + str(error))
        print("TDN R7: " + state["error"], file=sys.stderr); code = 2
    finally:
        if observer is not None:
            from tdn.analysis.tangent_response.telemetry import stop_recording
            state["worker_telemetry"] = stop_recording(observer, state=state["status"])
        else:
            state["worker_telemetry"] = dict(status="NOT_STARTED", reason="Preflight did not complete")
        state.update(finished_at=now(), elapsed_seconds=time.monotonic() - started, exit_code=code)
        write_json(native / "status.json", state, root=ROOT)
        report_batch(ctx, artifact_commands(ctx, native))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Freeze source/software/profile and R7 protocols")
    worker = commands.add_parser("run", help="Only the captured native Raven worker calls this")
    worker.add_argument("--contract", required=True)
    worker.add_argument("--mode", choices=("all", "verify"), default="all")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(); return 0
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print("TDN R7: " + str(error), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
