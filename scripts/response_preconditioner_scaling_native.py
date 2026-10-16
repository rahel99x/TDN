#!/usr/bin/env python3
"""Source-bound Phase 1 scaling development study in Raven's genuine desktop Slurm worker.

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
import stat
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import response_preconditioner_native as common
from response_preconditioner_r7_native import report_batch as captured_report_batch
from response_diagnostic_raven import confined, read_bytes, read_json, write_json

SCRIPT = "scripts/response_preconditioner_scaling.sh"
SCHEMA = "tdn.response-preconditioner-scaling-native/v1"
DECISION = "BOUNDED_DEVELOPMENT"
CONTRACT_FOLDER = ".runtime/response-preconditioner-scaling-launches"
UPLOAD_PART_BYTES = 30 << 20
IO_CHUNK_BYTES = 1 << 20
INVENTORY_BYTES = 16 << 20
PACK_TOTAL_BYTES = 512 << 20
REVIEW_TOTAL_BYTES = 1024 << 20
identity = common.identity
venv_lock = common.venv_lock
now = common.now
report = common.report
execute = common.execute


def _stream_evidence(path, *, root, limit, sink=None):
    """Hash/copy a stable regular file without loading a full artifact into RAM."""
    path = confined(path, root)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                value.st_mtime_ns, value.st_ctime_ns)
    digest = hashlib.sha256(); size = 0
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Evidence must be a regular file: {path}")
        if before.st_size > limit:
            raise ValueError(f"Evidence exceeds remaining byte budget: {path} "
                             f"({before.st_size} bytes; remaining limit {limit})")
        while raw := stream.read(min(IO_CHUNK_BYTES, limit-size+1)):
            size += len(raw)
            if size > limit:
                raise ValueError(f"Evidence grew beyond remaining byte budget: {path} (limit {limit})")
            digest.update(raw)
            if sink is not None:
                sink.write(raw)
        after = os.fstat(stream.fileno())
    confined(path, root)
    named = path.stat(follow_symlinks=False)
    if size != before.st_size or signature(before) != signature(after) or signature(after) != signature(named):
        raise ValueError(f"Evidence changed during inspection: {path}")
    return dict(bytes=size, sha256=digest.hexdigest())


def context(snapshot):
    return common.context(snapshot, script=SCRIPT)


def prepare():
    if any(os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT", "RAVEN_LAUNCH_PLAN")):
        raise ValueError("Prepare outside active Slurm/Raven attempts; submission is a separate action")
    with venv_lock():
        from tdn.analysis.response_preconditioner.scaling_protocol import protocol
        bound = identity()
        token = uuid.uuid4().hex
        path = confined(ROOT / CONTRACT_FOLDER / (token + ".json"), ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        confined(ROOT / "logs/response-preconditioner-scaling", ROOT).mkdir(parents=True, exist_ok=True)
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
        raise ValueError("Not this checkout's frozen bounded Phase 1 scaling native contract")
    from tdn.analysis.response_preconditioner.scaling_protocol import protocol
    if value.get("protocols") != {name: protocol(name) for name in ("verify", "study")}:
        raise ValueError("Phase 1 scaling protocol changed after preparation; create a fresh contract")
    return value, checksum


def resource_check(allocation):
    common.resource_check(allocation)
    job = allocation["job"]
    tres = dict(x.split("=", 1) for x in job.get("AllocTRES", "").split(",") if "=" in x)
    memory = re.fullmatch(r"([0-9]+)([KMGT]?)", tres.get("mem", ""))
    memory_mib = int(memory[1]) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[memory[2]]
    wall = sum(int(part) * 60 ** i for i, part in enumerate(reversed(job["TimeLimit"].split(":"))))
    if int(job["NumCPUs"]) != 4 or memory_mib > 8192 or wall > 1200:
        raise ValueError("Phase 1 scaling is frozen to four CPUs, at most 8 GiB and 20 minutes")


def report_batch(ctx, commands):
    """Bound transport while reusing Raven's exact captured reporting runtime."""
    if len(commands) > 32768:
        raise ValueError("Scaling reporting exceeds its bounded row transport")
    for first in range(0, len(commands), 512):
        captured_report_batch(ctx, commands[first:first + 512])


def artifact_commands(ctx, folder):
    """Register compact stage metadata; bulk query arrays have a packed artifact.

    The complete stage artifacts.json remains authoritative. Registering every
    query individually would exceed Raven's 4,096-entry/256-KiB contract bounds.
    """
    folder = confined(folder, ctx["run"])
    commands = []
    if folder.exists():
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if "pytest-temp" in relative.parts:
                continue
            if path.is_symlink():
                raise ValueError("Scaling portable evidence may not contain symlinks")
            if path.is_file():
                confined(path, ctx["run"])
                if relative.parts[0] in ("queries", "predictions"):
                    continue
                fmt = (None if path.suffix in (".npz", ".zip") else
                       "json" if path.suffix == ".json" else
                       "csv" if path.suffix == ".csv" else "text")
                commands.append(["artifact", path.relative_to(ctx["run"]).as_posix(),
                                 *(["--format", fmt] if fmt else [])])
                if len(commands) > 512:
                    raise ValueError("Scaling compact stage exceeds 512 portable artifacts")
    return commands


def pack_study_evidence(ctx, output):
    """Pack immutable bulk arrays outside the sealed study directory."""
    output = confined(output, ctx["run"])
    if not output.exists():
        return None
    sources = []
    for folder in ("queries", "predictions"):
        location = output / folder
        confined(location, ctx["run"])
        for path in sorted(location.rglob("*")) if location.exists() else ():
            confined(path, ctx["run"])
            if path.is_file():
                sources.append(path)
            elif not path.is_dir():
                raise ValueError("Only regular numerical evidence can be packed")
    if not sources:
        return None
    if len(sources) > 16384:
        raise ValueError("Scaling packed evidence exceeds 16,384 files")
    sealed = None
    seal = output / "artifacts.json"
    if seal.exists():
        record, _ = read_json(seal, root=ROOT, limit=INVENTORY_BYTES)
        rows = record.get("files", [])
        sealed = {}
        for row in rows:
            name = row.get("path")
            if not isinstance(name, str) or name in sealed:
                raise ValueError("Invalid stage artifact inventory")
            confined(output / name, output)
            sealed[name] = row
    native = ctx["run"] / "outputs/native"
    archive = native / "portable-study-evidence.zip"
    manifest = dict(schema=SCHEMA, scope="queries and predictions; originals preserved in outputs/study",
                    stage_artifact_seal_present=sealed is not None, files=[])
    began = time.monotonic(); total = 0
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sources:
            name = path.relative_to(output).as_posix()
            with bundle.open(name, "w") as destination:
                row = dict(path=name, **_stream_evidence(path, root=ROOT,
                           limit=PACK_TOTAL_BYTES-total, sink=destination))
            total += row["bytes"]
            if sealed is not None and sealed.get(name) != row:
                raise ValueError("Numerical evidence differs from its sealed stage artifact inventory: " + name)
            manifest["files"].append(row)
        bundle.writestr("packing-manifest.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    manifest.update(archive=archive.relative_to(ctx["run"]).as_posix(),
                    archive_bytes=archive.stat().st_size, elapsed_seconds=time.monotonic() - began,
                    packing_cost_scope="native application wall time; excluded from scientific action timing")
    write_json(native / "packing.json", manifest, root=ROOT, limit=INVENTORY_BYTES)
    return manifest


def stage_evidence(ctx, output, command, exit_code):
    # Failed and partial queries are equally part of the retained evidence.
    if command == "study":
        pack_study_evidence(ctx, output)
    report_batch(ctx, artifact_commands(ctx, output))
    if exit_code:
        return
    summary, _ = read_json(output / "summary.json", root=ROOT)
    if (summary.get("command") != command or summary.get("status") != "COMPLETED"
            or summary.get("scientific_promotion") is not False or summary.get("device") != "cpu"
            or summary.get("decision") != DECISION):
        raise ValueError("Completed Phase 1 scaling child lacks matching nonpromoting CPU evidence")
    required = ("protocol.json", "source.json", "software.json", "artifacts.json")
    required += (("verification.xml", "checks.json") if command == "verify" else
                 ("metrics.csv", "metrics.json", "selection.json", "learning_curves.json",
                  "parent_metrics.json", "paired_comparisons.json", "denominators.json", "timing.json"))
    if any(not (output / name).is_file() for name in required):
        raise ValueError("Completed Phase 1 scaling child is missing required declared output evidence")


def publish_metrics(ctx, output):
    """Publish parent/cohort summaries; complete per-query rows remain artifacts."""
    commands = []
    for source in ("learning_curves", "parent_metrics"):
        path = output / (source + ".json")
        if not path.is_file():
            continue
        rows = json.loads(read_bytes(path, root=ROOT, limit=16 << 20))
        if source == "learning_curves" and isinstance(rows, dict):
            rows = rows.get("groups")
        if not isinstance(rows, list) or len(rows) > 4096:
            raise ValueError("Expected bounded scaling aggregate rows")
        expanded = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Scaling aggregate rows must be objects")
            expanded.append(row)
            if source == "learning_curves":
                strata = row.get("strata", {})
                if not isinstance(strata, dict):
                    raise ValueError("Learning-curve strata must be named objects")
                identity = {key: row[key] for key in ("split", "role", "training_membership", "method",
                            "training_parent_count", "calibration_source_count") if key in row}
                for axis, field in (("parent_family", "parent_family"), ("family", "source_family")):
                    groups = strata.get(axis, {})
                    if not isinstance(groups, dict):
                        raise ValueError("Learning-curve family strata must be objects")
                    for name, values in sorted(groups.items()):
                        if not isinstance(values, dict):
                            raise ValueError("Learning-curve stratum measurements must be objects")
                        expanded.append({**identity, **values, "parent_family": "all", "source_family": "all", field: name})
            if len(expanded) > 4096:
                raise ValueError("Expanded scaling aggregate rows exceed 4096")
        for row in expanded:
            identity_keys = ("split", "role", "training_membership",
                             *(["parent_id"] if source == "parent_metrics" else []),
                             "parent_family", "source_family", "method")
            phase = "scaling/" + source + "/" + "/".join(str(row.get(key, "all")) for key in identity_keys)
            if len(phase) > 160:
                raise ValueError("Scaling identity exceeds Raven phase contract")
            args = ["metric", "--phase", phase, "--step", str(len(commands))]
            for key in ("parent_count", "source_count", "rms", "maximum", "pooled_rms", "worst_maximum",
                        "mean_error", "centered_rms", "expected", "observed", "measured", "missing",
                        "failed", "unresolved", "coverage", "parent_index", "complete_parent_count",
                        "training_parent_count", "calibration_source_count", "independent_parent_count",
                        "median_parent_rms", "worst_parent_rms", "reference_rms", "remainder_rms",
                        "feedback_rms", "reference_uncertainty_rms", "error_to_remainder_rms",
                        "error_to_feedback_rms", "planned_queries", "measured_queries", "missing_queries",
                        "failed_queries", "unresolved_queries", "invalid_measured_queries",
                        "training_reference_seconds", "validation_reference_seconds", "fitting_seconds"):
                value = row.get(key)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Scaling measurements must be finite numerical values")
                args += ["--value", key + "=" + str(value)]
            if "complete" in row:
                if not isinstance(row["complete"], bool):
                    raise ValueError("Aggregate completeness must be Boolean")
                args += ["--value", "complete=" + str(int(row["complete"]))]
            if len(args) > 5:
                commands.append(args)
    selection_path = output / "selection.json"
    if selection_path.is_file():
        selection, _ = read_json(selection_path, root=ROOT, limit=8 << 20)
        for cell in selection.get("cells", []):
            for candidate in cell.get("candidates", []):
                args = ["metric", "--phase", "scaling/fit/" + cell["model_id"] + "/" + candidate["name"],
                        "--step", str(len(commands)), "--value", "eligible=" + str(int(candidate["eligible"])),
                        "--value", "selected=" + str(int(candidate["name"] == (cell.get("selected") or {}).get("name")))]
                values = {**candidate.get("diagnostics", {}), **{key: candidate.get(key) for key in
                          ("ridge", "validation_loss", "fitting_seconds")},
                          **{key: cell.get(key) for key in ("training_parent_count", "calibration_source_count")}}
                for key in ("ridge", "validation_loss", "fitting_seconds", "training_parent_count", "calibration_source_count",
                            "original_design_numerical_rank", "augmented_condition_number", "data_loss", "ridge_loss",
                            "objective", "optimality_gap_upper_bound"):
                    value = values.get(key)
                    if value is None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Fit measurements must be finite numerical values")
                    args += ["--value", key + "=" + str(value)]
                commands.append(args)
    report_batch(ctx, commands)


def publish_timing(ctx, output):
    """Retain every timing attempt, including unsuccessful paired rounds."""
    incremental = output / "timing_records.json"
    completed = output / "timing.json"
    if incremental.is_file():
        rows = json.loads(read_bytes(incremental, root=ROOT, limit=8 << 20))
        if not isinstance(rows, list) or len(rows) > 24 * 256:
            raise ValueError("Expected bounded incremental Phase 1 scaling timing records")
        grouped = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("context_id"), str):
                raise ValueError("Phase 1 scaling timing records must retain their context identity")
            grouped.setdefault(row["context_id"], []).append(row)
        contexts = [dict(context_id=key, records=value) for key, value in grouped.items()]
    elif completed.is_file():
        contexts = json.loads(read_bytes(completed, root=ROOT, limit=8 << 20))
    else:
        return
    if not isinstance(contexts, list) or len(contexts) > 24:
        raise ValueError("Expected at most 24 declared Phase 1 scaling timing contexts")
    commands = []
    for context_index, context_record in enumerate(contexts):
        records = context_record.get("records", [])
        if not isinstance(records, list) or len(records) > 256:
            raise ValueError("Expected bounded Phase 1 scaling timing rounds")
        for row in records:
            phase = "scaling/timing/" + str(row["phase"]) + "/" + str(row["method"])
            args = ["metric", "--phase", phase, "--step", str(len(commands)),
                    "--value", "context_index=" + str(context_index),
                    "--value", "success=" + str(int(row.get("status") == "PASS"))]
            for key, name in (("seconds", "bundle_seconds"), ("round", "round"), ("position", "position")):
                value = row.get(key)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Phase 1 scaling timings must be finite numerical values")
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
    ctx = context(os.environ.get("TDN_RESPONSE_PRECONDITIONER_SCALING_SNAPSHOT", ""))
    if output is None or confined(output, ROOT) != ctx["run"] / "outputs/study":
        raise ValueError("Phase 1 scaling study output must be this Raven attempt's fresh outputs/study")
    contract, checksum = load_contract(contract_path)
    if identity() != contract["identity"]:
        raise ValueError("Source/software/profile changed after preparation")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.desktop_slurm import verify_allocation
    verify_runtime("cpu", "response-preconditioner-scaling")
    resource_check(verify_allocation("cpu", root=ROOT))
    release, _ = read_json(ctx["run"] / "outputs/native/verification-release.json", root=ROOT)
    if (release.get("schema") != SCHEMA or release.get("contract_sha256") != checksum
            or release.get("verification") != "outputs/verification"):
        raise ValueError("Phase 1 scaling study requires successful verification in this exact source-bound attempt")
    verification = ctx["run"] / "outputs/verification"
    expected = {name: hashlib.sha256(read_bytes(verification / name, root=ROOT, limit=8 << 20)).hexdigest()
                for name in ("summary.json", "checks.json", "verification.xml", "artifacts.json")}
    if release.get("hashes") != expected:
        raise ValueError("Phase 1 scaling verification evidence changed before scientific acquisition")
    return ctx


def run(args):
    ctx = context(os.environ.get("TDN_RESPONSE_PRECONDITIONER_SCALING_SNAPSHOT", ""))
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
            verify_runtime("cpu", "response-preconditioner-scaling")
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
                argv = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/response_preconditioner_scaling.py"),
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
                report(ctx, "metric", "--phase", "scaling/" + command, "--step", str(step),
                       "--value", "stage_seconds=" + str(elapsed),
                       "--value", "stage_exit_code=" + str(child.returncode),
                       "--completed", str(step), "--total", str(len(commands)), "--unit", "bounded Phase 1 scaling stages")
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
        print("TDN Phase 1 scaling: " + state["error"], file=sys.stderr); code = 2
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



def export_review(run_path):
    """Create a fresh portable terminal snapshot, split for 32-MiB upload limits.

    This is a read-only export of historical evidence, so it never requires the
    current checkout identity to match the completed run or executes its code.
    """
    run = confined(ROOT / run_path if not Path(run_path).is_absolute() else run_path, ROOT)
    if run.parent != ROOT / "runs" or not re.fullmatch(r"launch-[0-9a-f]{32}-a1", run.name):
        raise ValueError("Export requires an exact native Raven attempt under this checkout's runs/")
    inventory, _ = read_json(run / "run.json", root=ROOT)
    status, _ = read_json(run / "outputs/native/status.json", root=ROOT)
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT", "INTERRUPTED"}
    if (inventory.get("schema") != "raven.run/v1" or inventory.get("run_id") != run.name
            or inventory.get("state") not in terminal or status.get("schema") != SCHEMA
            or status.get("status") not in ("COMPLETED", "FAILED")):
        raise ValueError("Both Raven and native application must have terminal records before export")
    duplicate = "outputs/native/portable-study-evidence.zip"
    omitted = []
    packing = run / "outputs/native/packing.json"
    if (run / duplicate).is_file() and packing.is_file():
        packed, _ = read_json(packing, root=ROOT, limit=INVENTORY_BYTES)
        if packed.get("archive") != duplicate or not isinstance(packed.get("files"), list):
            raise ValueError("Invalid packed evidence manifest")
        for row in packed["files"]:
            path = confined(run / "outputs/study" / row["path"], run / "outputs/study")
            observed = _stream_evidence(path, root=ROOT, limit=PACK_TOTAL_BYTES)
            if observed != {key: row[key] for key in ("bytes", "sha256")}:
                raise ValueError("Cannot omit duplicate bundle: raw numerical evidence changed")
        omitted.append(dict(path=duplicate, reason="Duplicate of hash-verified original study files included in this export"))
    sources = []
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run).as_posix()
        if "pytest-temp" in path.relative_to(run).parts:
            continue
        confined(path, run)
        if relative == duplicate and omitted:
            continue
        if path.is_file():
            sources.append(path)
        elif not path.is_dir():
            raise ValueError("Export accepts only regular files and directories")
    if len(sources) > 16384:
        raise ValueError("Review export exceeds 16,384 files")
    export = confined(ROOT / ".runtime/response-preconditioner-scaling-exports" / (run.name + "-" + uuid.uuid4().hex[:12]), ROOT)
    export.mkdir(parents=True, exist_ok=False)
    archive = export / (run.name + "-scaling-review.zip")
    files = []; total = 0
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sources:
            relative = path.relative_to(run).as_posix()
            with bundle.open(run.name + "/" + relative, "w") as destination:
                record = _stream_evidence(path, root=ROOT, limit=REVIEW_TOTAL_BYTES-total, sink=destination)
            total += record["bytes"]
            files.append(dict(path=relative, **record))
    # A terminal record is necessary but not proof of global filesystem quiescence.
    # Check every snapshotted file again; changed inputs leave a failed export.
    for path, record in zip(sources, files):
        observed = _stream_evidence(path, root=ROOT, limit=REVIEW_TOTAL_BYTES)
        if observed != {key: record[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"Evidence changed during export: {path}; wait for the attempt to settle")
    after_paths = []
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run).as_posix()
        if "pytest-temp" in path.relative_to(run).parts or (relative == duplicate and omitted):
            continue
        confined(path, run)
        if path.is_file():
            after_paths.append(path)
    if after_paths != sources:
        raise ValueError("Evidence inventory changed during export")
    digest = hashlib.sha256(); parts = []; part_bytes = UPLOAD_PART_BYTES
    size = archive.stat().st_size
    with archive.open("rb") as stream:
        for number in range(1, 10000):
            raw = stream.read(part_bytes)
            if not raw:
                break
            digest.update(raw)
            name = archive.name if size <= part_bytes else archive.name + f".part{number:03d}"
            if size > part_bytes:
                with (export / name).open("xb") as target:
                    target.write(raw)
            parts.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    index = dict(schema=SCHEMA, run_id=run.name, created_at=now(), archive=archive.name,
                 archive_bytes=size, archive_sha256=digest.hexdigest(), parts=parts,
                 file_count=len(files), uncompressed_bytes=total, files=files, omitted_duplicates=omitted,
                 scope="Terminal application snapshot; not an independent scheduler-outcome certificate",
                 upload="Upload index.json, README.txt, and every listed archive part")
    write_json(export / "index.json", index, root=ROOT, limit=INVENTORY_BYTES)
    instructions = ("TDN Phase 1 independent-parent scaling review\n\n"
                    "Upload index.json, README.txt, and every file listed in index.json parts.\n"
                    "The ZIP contains the exact Raven run directory, raw research arrays, logs, and telemetry.\n"
                    "Original run files are preserved. The duplicate native numerical ZIP is omitted only\n"
                    "when every original numerical file is present and its hash matches packing.json.\n"
                    "Test fixture directories named pytest-temp are excluded.\n"
                    "Verify each part's SHA-256 against index.json before reassembly.\n")
    if len(parts) > 1:
        instructions += ("\nReassemble from this export directory:\n  cat -- " + archive.name + ".part* > " + archive.name + "\n"
                         "Then verify the complete archive SHA-256 in index.json before extracting.\n")
        archive.unlink()  # Owned duplicate only; never an original run artifact.
    with (export / "README.txt").open("x", encoding="utf-8") as stream:
        stream.write(instructions)
    print(export.relative_to(ROOT).as_posix())
    return export

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Freeze source/software/profile and Phase 1 scaling protocols")
    worker = commands.add_parser("run", help="Only the captured native Raven worker calls this")
    worker.add_argument("--contract", required=True)
    worker.add_argument("--mode", choices=("all", "verify"), default="all")
    export = commands.add_parser("export", help="Package a terminal native attempt for portable review")
    export.add_argument("--run", required=True, help="Exact runs/launch-...-a1 directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(); return 0
        if args.command == "export":
            export_review(args.run); return 0
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print("TDN Phase 1 scaling: " + str(error), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
