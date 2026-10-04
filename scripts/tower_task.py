#!/usr/bin/env python3
"""Report a manual allocated TDN stage, including bootstrap and test commands.

This coordinator uses only the standard library. Numerical children continue
to use the existing project venv and real srun allocation checks.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_fingerprint():
    digest = hashlib.sha256()
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                if path.is_symlink():
                    raise ValueError("Source identity cannot follow symlinks")
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def scheduler_context(stage):
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job) or not os.environ.get("SLURM_STEP_ID"):
        raise ValueError("Manual Tower reporting requires an actual allocated srun task")
    observed = subprocess.run(["scontrol", "show", "job", job, "-o"],
                              text=True, capture_output=True, timeout=30, check=True)
    fields = dict(re.findall(r"(\w+)=([^\s]+)", observed.stdout))
    if (fields.get("JobId") != job or fields.get("Account") != "anakano_81"
            or fields.get("UserId", "").split("(")[0] != "aadaniel"
            or fields.get("JobName") != f"tdn-{stage}"):
        raise ValueError("Scheduler identity does not match this manual TDN stage")
    resources = {"account": fields["Account"]}
    for key, field in (("partition", "Partition"), ("qos", "QOS")):
        if fields.get(field) not in (None, "(null)", "N/A", "Unknown"):
            resources[key] = fields[field]
    for key, field in (("cpus", "NumCPUs"), ("nodes", "NumNodes")):
        if fields.get(field, "").isdigit() and int(fields[field]) > 0:
            resources[key] = int(fields[field])
    # Do not infer zero GPUs, memory scope or time limits from missing fields.
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    if tres.get("gres/gpu", "").isdigit():
        resources["gpus"] = int(tres["gres/gpu"])
    elif os.environ.get("TDN_DEVICE") == "cpu" and stage != "driver-audit":
        # The already validated TDN CPU-only request explicitly requests no GPU.
        resources["gpus"] = 0
    logs = []
    for suffix, field in (("stdout", "StdOut"), ("stderr", "StdErr")):
        path = fields.get(field)
        if path and path not in ("(null)", "/dev/null"):
            logs.append({"id": f"scheduler.{suffix}", "path": path,
                         "label": f"Slurm {suffix}", "group": "Scheduler"})
    return job, resources, logs


def execute(command, *, source, config, stage):
    from tdn.reporting import begin_report, emit, finish_report
    from tdn.tower_analytics import publish_outputs
    job, resources, logs = scheduler_context(stage)
    for path in (source, config):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Manual report paths must remain in the project")
    report = begin_report(
        source, name=f"TDN/manual/{stage}", script="scripts/run_stage.sh",
        parameters={"config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "source_sha256": source_fingerprint(),
                    "stage": stage, "device": os.environ.get("TDN_DEVICE", "unknown")},
        resources=resources, job_id=job, logs=logs,
        metadata={"resource_source": "verified scontrol allocation; omitted fields remain unknown",
                  "execution_mode": "carc", "stage": stage},
    )
    print(f"TDN_TOWER_DIR={report}", flush=True)
    previous = os.environ.get("TDN_TOWER_DIR")
    os.environ["TDN_TOWER_DIR"] = str(report)
    started = time.monotonic()
    child, stop_signal, outcome, error = None, None, 1, None
    old_handlers = {}

    def forward(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            old_handlers[signum] = signal.signal(signum, forward)
        emit({}, phase=f"manual.{stage}", completed=0, total=1, unit="stages")
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy())
        if stop_signal is not None:
            forward(stop_signal, None)
        result = child.wait()
        outcome = result if result >= 0 else 128 - result
        if stop_signal is not None and outcome == 0:
            outcome = 75
        emit({"stage_runtime_seconds": time.monotonic() - started, "stage_exit_code": outcome},
             phase=f"manual.{stage}", completed=int(outcome == 0), total=1, unit="stages")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"TDN manual stage: {error}", file=sys.stderr)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        try:
            results = publish_outputs(report, [source])
            finish_report(report, state="COMPLETED" if outcome == 0 else
                          "INTERRUPTED" if stop_signal is not None or outcome in (75, 130, 143) else "FAILED",
                          runtime_seconds=time.monotonic() - started, exit_code=outcome,
                          results=results, metadata={"error": error, "received_signal": stop_signal,
                          "measurement_scope": "manual allocated stage coordinator; no queue time"})
        except Exception as exc:
            print(f"TDN Tower reporting failed: {exc}", file=sys.stderr)
            if outcome == 0:
                outcome = 1
        finally:
            if previous is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a stage command is required")
    try:
        return execute(command, source=args.source, config=args.config, stage=args.stage)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"TDN Tower manual reporting: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
