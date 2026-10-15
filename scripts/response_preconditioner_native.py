#!/usr/bin/env python3
"""Frozen, bounded CPU verification through Raven and actual desktop Slurm.

Preparation reads source and package metadata only. The allocated worker owns
its CPU/RSS observations; Raven owns the outer attempt and terminal lifecycle.
No scheduler submission, CUDA readiness or scientific promotion occurs here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from response_diagnostic_raven import confined, read_bytes, read_json, write_json

SCRIPT = "scripts/response_preconditioner_verify.sh"
SCHEMA = "tdn.response-preconditioner-native/v1"
SMALL = 1 << 20
STAGE_ARTIFACTS = (
    "summary.json", "source.json", "source_after.json", "software.json", "protocol.json", "timing.json",
    "metrics.csv", "metrics.json", "journal.jsonl", "README.md", "verification.xml", "verification.log",
    "command.json", "checks.json", "selection.json", "fitting.json", "references.json", "work.json",
    "derivatives.json", "audit.json", "architecture.json", "array_index.json", "arrays.npz", "artifacts.json",
    "failure.txt",
)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def venv_lock():
    """The same shared, readable lock used by existing allocated workers."""
    import fcntl
    path = confined(ROOT / ".cache/carc-phase.lock", ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Project venv is being changed; preserve this attempt") from error
        yield


def installed_packages():
    """Deduplicate discovery aliases, never distinct installed metadata.

    Fedora venv lib/lib64 aliases and repeated sys.path entries can yield the
    same distribution more than once. The supported stdlib PathDistribution
    stores its metadata directory in _path; locate_file('') gives only the
    shared site-packages root, and files may be absent without a RECORD.
    Unknown/non-filesystem metadata identities cannot justify deduplication.
    """
    observed = {}
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        version = dist.version
        try:
            location = str(Path(dist._path).resolve(strict=True))
        except (AttributeError, TypeError, OSError, RuntimeError):
            location = None
        current = (version, location)
        if name in observed:
            previous = observed[name]
            if location is not None and current == previous:
                continue
            def describe(value):
                return str(value[0]) + " at " + (value[1] or "an unverified metadata location")
            raise ValueError("Duplicate installed distribution: " + name + "; "
                             + describe(previous) + "; " + describe(current)
                             + ". Distinct or unverified metadata cannot be merged; inspect these locations.")
        observed[name] = current
    return {name: value[0] for name, value in sorted(observed.items())}


def software():
    if (Path(sys.prefix).resolve() != ROOT / ".venv" or sys.prefix == sys.base_prefix
            or os.environ.get("CONDA_PREFIX") or (Path(sys.base_prefix) / "conda-meta").exists()
            or any(x in str(sys.base_prefix).lower() for x in ("conda", "miniforge", "mambaforge"))):
        raise ValueError("Use this checkout's standalone .venv")
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise ValueError("Use the supported standalone Python 3.11–3.13")
    return dict(python=platform.python_version(), prefix=sys.prefix,
                base_prefix=sys.base_prefix, packages=installed_packages())


def source_identity():
    digest = hashlib.sha256()
    files = []
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        files.extend(path for path in sorted((ROOT / name).rglob("*"))
                     if path.is_file() and "__pycache__" not in path.parts)
    files.extend(ROOT / name for name in ("requirements.txt", "pyproject.toml", ".raven/automation.json"))
    for path in files:
        raw = read_bytes(path, root=ROOT, limit=64 << 20)
        digest.update(path.relative_to(ROOT).as_posix().encode()); digest.update(b"\0")
        digest.update(hashlib.sha256(raw).digest())
    return dict(sha256=digest.hexdigest(), files=len(files),
                scope="tdn/reference/scripts/configs/tests/dependencies/Raven settings")


def identity():
    from tdn.runtime.desktop_slurm import load_profile, profile_path
    profile = load_profile(root=ROOT)
    profile_file = profile_path(root=ROOT)
    profile_raw = read_bytes(profile_file, root=ROOT)
    return dict(source=source_identity(), software=software(), profile=profile,
                profile_path=profile_file.relative_to(ROOT).as_posix(),
                profile_sha256=hashlib.sha256(profile_raw).hexdigest())


def prepare():
    if any(os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT", "RAVEN_LAUNCH_PLAN")):
        raise ValueError("Prepare outside active Slurm/Raven attempts; submission is a separate action")
    with venv_lock():
        bound = identity()
        token = uuid.uuid4().hex
        path = confined(ROOT / ".runtime/response-preconditioner-launches" / (token + ".json"), ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        confined(ROOT / "logs/response-preconditioner", ROOT).mkdir(parents=True, exist_ok=True)
        payload = dict(schema=SCHEMA, token=token, created_at=now(), project_root=str(ROOT),
                       identity=bound, device="cpu", decision="IMPLEMENTATION_ONLY",
                       scientific_promotion=False, modes=["all", "verify", "smoke"])
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
            or path != ROOT / ".runtime/response-preconditioner-launches" / (token + ".json")
            or value.get("device") != "cpu" or value.get("scientific_promotion") is not False):
        raise ValueError("Not this checkout's frozen CPU verification contract")
    return value, checksum


def context(snapshot, *, script=SCRIPT):
    """Bind the original checkout, captured shell and exact actual Raven job."""
    plan_path = confined(os.environ.get("RAVEN_LAUNCH_PLAN", ""), ROOT)
    plan, plan_hash = read_json(plan_path, root=ROOT)
    token = plan.get("token", "")
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("Invalid Raven launch token")
    folder = ROOT / ".raven/wrappers" / token
    source, payload = ROOT / script, folder / "payload.sh"
    if (plan.get("schema") != "raven.launch-plan/v1" or plan.get("execution") != "slurm"
            or plan.get("project_root") != str(ROOT) or plan.get("script") != str(source)
            or plan.get("script_relative") != script or plan_path != folder / "plan.json"
            or plan.get("source_snapshot") != payload.relative_to(ROOT).as_posix()
            or confined(snapshot, ROOT) != payload
            or os.environ.get("RAVEN_PROJECT_ROOT") != str(ROOT)
            or os.environ.get("RAVEN_SOURCE_SCRIPT") != str(source)):
        raise ValueError("Expected the exact captured native Raven launch")
    for path in (source, payload):
        if hashlib.sha256(read_bytes(path, root=ROOT, limit=8 << 20)).hexdigest() != plan.get("script_sha256"):
            raise ValueError("Captured or current launcher changed; create a fresh launch")
    run_id = "launch-" + token + "-a1"
    run = confined(ROOT / "runs" / run_id, ROOT)
    if os.environ.get("RAVEN_RUN_ID") != run_id or os.environ.get("RAVEN_RUN_DIR") != str(run):
        raise ValueError("Raven attempt identity differs from the frozen launch")
    record, _ = read_json(run / "run.json", root=ROOT)
    automation = record.get("metadata", {}).get("automation", {})
    binding = automation.get("plan", {})
    if (record.get("schema") != "raven.run/v1" or record.get("run_id") != run_id
            or record.get("state") != "RUNNING" or record.get("job_id") != os.environ.get("SLURM_JOB_ID")
            or automation.get("execution") != "slurm"
            or binding.get("token") != token or binding.get("plan_sha256") != plan_hash):
        raise ValueError("Raven inventory does not identify this actual running Slurm job")
    interpreter = plan.get("report_python")
    bundle, helper = folder / "runtime.pyz", folder / "report.py"
    if (not isinstance(interpreter, str) or not Path(interpreter).is_absolute()
            or os.environ.get("RAVEN_REPORT_PYTHON") != interpreter
            or os.environ.get("RAVEN_REPORT_HELPER") != str(helper)
            or os.environ.get("RAVEN_RUNTIME_BUNDLE") != str(bundle)
            or plan.get("runtime_bundle") != str(bundle)
            or hashlib.sha256(read_bytes(bundle, root=ROOT, limit=64 << 20)).hexdigest() != plan.get("runtime_sha256")):
        raise ValueError("Raven private reporting runtime differs from the frozen launch")
    expected_helper = ("import sys\nsys.path.insert(0, " + repr(str(bundle)) + ")\n"
                       "from raven.research_reporting import main\nraise SystemExit(main())\n").encode()
    if read_bytes(helper, root=ROOT) != expected_helper:
        raise ValueError("Raven reporting helper is not its captured entry point")
    return dict(run=run, report_python=interpreter, report_helper=helper, plan=plan)


def report(ctx, *args):
    subprocess.run([ctx["report_python"], "-I", str(ctx["report_helper"]), *args], check=True,
                   timeout=30, stdout=subprocess.DEVNULL)


def artifact(ctx, path):
    path = confined(path, ctx["run"])
    if path.is_file():
        fmt = None if path.suffix == ".npz" else "json" if path.suffix == ".json" else "csv" if path.suffix == ".csv" else "text"
        report(ctx, "artifact", path.relative_to(ctx["run"]).as_posix(), *(["--format", fmt] if fmt else []))


def stage_evidence(ctx, output, command, exit_code):
    """Keep failed evidence too; a zero exit alone is not a complete result."""
    for name in STAGE_ARTIFACTS:
        artifact(ctx, output / name)
    if exit_code == 0:
        summary, _ = read_json(output / "summary.json", root=ROOT)
        if (summary.get("command") != command or summary.get("status") != "COMPLETED"
                or summary.get("scientific_promotion") is not False or summary.get("device") != "cpu"
                or summary.get("decision") != "IMPLEMENTATION_ONLY"):
            raise ValueError("Completed child does not have matching nonpromoting CPU evidence")
        required = ("protocol.json", "source.json", "software.json", "artifacts.json")
        required += (("verification.xml", "checks.json") if command == "verify" else
                     ("metrics.csv", "metrics.json", "selection.json", "work.json", "references.json",
                      "derivatives.json", "audit.json", "array_index.json", "arrays.npz"))
        if any(not (output / name).is_file() for name in required):
            raise ValueError("Completed child is missing required declared output evidence")


def resource_check(allocation):
    job = allocation["job"]
    cpus = int(job.get("NumCPUs", "0"))
    if not 4 <= cpus <= 8:
        raise ValueError("Native verification requires four to eight allocated CPUs")
    tres = dict(x.split("=", 1) for x in job.get("AllocTRES", "").split(",") if "=" in x)
    memory = re.fullmatch(r"([0-9]+)([KMGT]?)", tres.get("mem", ""))
    if not memory or not 0 < int(memory[1]) * {"":1, "K":1/1024, "M":1, "G":1024, "T":1024**2}[memory[2]] <= 48 * 1024:
        raise ValueError("Cannot verify allocated memory within 48 GiB")
    if any(key.startswith("gres/gpu") and value != "0" for key, value in tres.items()):
        raise ValueError("This CPU verification must not reserve a GPU")
    parts = job.get("TimeLimit", "").split(":")
    if len(parts) not in (2, 3) or any(not part.isdecimal() for part in parts):
        raise ValueError("Cannot verify a bounded allocation walltime")
    seconds = sum(int(part) * 60 ** i for i, part in enumerate(reversed(parts)))
    if not 0 < seconds <= 35 * 60:
        raise ValueError("Native verification must remain within 35 minutes")


def execute(ctx, command, argv):
    """Preserve exact per-stage streams inside the portable attempt."""
    streams = []
    paths = []
    try:
        for suffix in ("stdout", "stderr"):
            path = confined(ctx["run"] / "logs" / (command + "." + suffix + ".log"), ctx["run"])
            path.parent.mkdir(parents=True, exist_ok=True)
            streams.append(path.open("x", encoding="utf-8")); paths.append(path)
            report(ctx, "log", "preconditioner-" + command + "-" + suffix,
                   path.relative_to(ctx["run"]).as_posix(), "--label", command + " " + suffix)
        print("TDN preconditioner: starting " + command + "; logs " + str(paths[0]), flush=True)
        child = subprocess.run(argv, cwd=ROOT, stdout=streams[0], stderr=streams[1], check=False)
    finally:
        for stream in streams:
            stream.close()
    if child.returncode:
        for path in paths:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 8192))
                print(stream.read().decode("utf-8", "replace"), file=sys.stderr)
    return child


def publish_smoke_metrics(ctx, output):
    """Retain measured rows with method/split/kind; never merge different tasks."""
    path = output / "metrics.json"
    if not path.is_file():
        return
    rows = json.loads(read_bytes(path, root=ROOT))
    if not isinstance(rows, list) or len(rows) > 1024:
        raise ValueError("Expected bounded smoke metric rows")
    for step, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Smoke metric rows must be objects")
        phase = "smoke/" + "/".join(str(row.get(key, "unknown")) for key in ("kind", "split", "method"))
        args = ["metric", "--phase", phase, "--step", str(step)]
        for key in ("rms", "maximum", "mean_error", "centered_rms", "horizon", "direction",
                    "action_correction_rms", "correction_uncertainty_rms", "action_correction_relative_rms"):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Smoke measurements must be finite numerical values")
            args += ["--value", key + "=" + str(value)]
        report(ctx, *args)


def run(args):
    ctx = context(os.environ.get("TDN_RESPONSE_PRECONDITIONER_SNAPSHOT", ""))
    native = confined(ctx["run"] / "outputs/native", ctx["run"])
    native.mkdir(exist_ok=False)
    state = dict(schema=SCHEMA, status="RUNNING", created_at=now(), device="cpu",
                 decision="IMPLEMENTATION_ONLY", scientific_promotion=False, commands=[],
                 historical_gate="B STOP is unchanged", source_unchanged=None,
                 telemetry_scope="Actual allocated worker process tree; outer Raven supervisor telemetry has a separate scope")
    started = time.monotonic(); code = 2; observer = None
    try:
        with venv_lock():
            contract, checksum = load_contract(args.contract)
            state["contract_sha256"] = checksum
            if identity() != contract["identity"]:
                state["source_unchanged"] = False
                raise ValueError("Source, software or profile changed after preparation; create a fresh contract and launch")
            from tdn.runtime.preflight import verify_runtime
            from tdn.runtime.desktop_slurm import verify_allocation
            from tdn.runtime.storage import configure_storage
            verify_runtime("cpu", "response-preconditioner-verify")
            allocation = verify_allocation("cpu", root=ROOT)
            resource_check(allocation)
            configure_storage()
            state["allocation"] = allocation
            state["cpu_hardware"] = dict(node=platform.node(), platform=platform.platform(),
                                         visible_affinity=sorted(os.sched_getaffinity(0)))
            write_json(native / "contract.json", contract, root=ROOT)
            from tdn.analysis.tangent_response.telemetry import start_recording
            observer = start_recording(native, dict(families=["cpu", "memory"], interval_seconds=1.))
            commands = ("verify", "smoke") if args.mode == "all" else (args.mode,)
            code = 0
            for step, command in enumerate(commands, 1):
                if identity() != contract["identity"]:
                    state["source_unchanged"] = False
                    raise ValueError("Source/software/profile changed before stage; preserve this attempt")
                output = ctx["run"] / "outputs" / ("verification" if command == "verify" else "smoke")
                argv = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/response_preconditioner.py"),
                        command, "--device", "cpu", "--output", str(output)]
                begin = time.monotonic()
                child = execute(ctx, command, argv)
                elapsed = time.monotonic() - begin
                state["commands"].append(dict(command=command, argv=argv, exit_code=child.returncode,
                                              elapsed_seconds=elapsed, output=output.relative_to(ctx["run"]).as_posix()))
                write_json(native / "status.json", state, root=ROOT)
                stage_evidence(ctx, output, command, child.returncode)
                if command == "smoke":
                    publish_smoke_metrics(ctx, output)
                report(ctx, "metric", "--phase", command, "--step", str(step),
                       "--value", "implementation_stage_seconds=" + str(elapsed),
                       "--value", "implementation_stage_exit_code=" + str(child.returncode),
                       "--completed", str(step), "--total", str(len(commands)), "--unit", "implementation stages")
                if identity() != contract["identity"]:
                    state["source_unchanged"] = False
                    raise ValueError("Source/software/profile changed during stage; evidence is invalid for promotion")
                state["source_unchanged"] = True
                if child.returncode:
                    code = child.returncode if child.returncode > 0 else 128 - child.returncode
                    break
            state["status"] = "COMPLETED" if code == 0 else "FAILED"
    except Exception as error:
        state.update(status="FAILED", error=type(error).__name__ + ": " + str(error))
        print("TDN preconditioner: " + state["error"], file=sys.stderr)
        code = 2
    finally:
        if observer is not None:
            from tdn.analysis.tangent_response.telemetry import stop_recording
            state["worker_telemetry"] = stop_recording(observer, state=state["status"])
        else:
            state["worker_telemetry"] = dict(status="NOT_STARTED", reason="Preflight did not complete")
        state.update(finished_at=now(), elapsed_seconds=time.monotonic() - started, exit_code=code)
        write_json(native / "status.json", state, root=ROOT)
        for name in ("status.json", "contract.json", "telemetry/index.json", "telemetry/cpu-memory.jsonl",
                     "telemetry/cpu-memory.csv", "telemetry/observer.log"):
            artifact(ctx, native / name)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Freeze source/software/profile; print fresh contract path")
    worker = commands.add_parser("run", help="Called only by the captured Raven script's actual srun task")
    worker.add_argument("--contract", required=True)
    worker.add_argument("--mode", choices=("all", "verify", "smoke"), default="all")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(); return 0
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print("TDN preconditioner: " + str(error), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
