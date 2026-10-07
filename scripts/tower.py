#!/usr/bin/env python3
"""Inspect, collect, and attach project-owned Tower reports without compute."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdn import reporting

MAX_ENTRIES = 4096
MAX_REPORTS = 256
MAX_METRICS_BYTES = 64 << 20
JOB_ID = re.compile(r"[0-9]+(?:_[0-9]+)?\Z")


def project_path(root, value, *, directory=True):
    """Require an exact project path and refuse symlinks, including ancestors."""
    root = Path(root).absolute()
    raw = str(value)
    if (not raw or not raw.isprintable() or any(c in raw for c in "\\*?[]")
            or any(p in {".", ".."} for p in raw.split("/")) or "//" in raw):
        raise ValueError("Use an exact project path without traversal, globs, or symlinks")
    path = Path(raw)
    path = path if path.is_absolute() else root / path
    if not path.is_relative_to(root):
        raise ValueError("Report paths must stay inside this project")
    for parent in reversed((path, *path.parents)):
        if not parent.exists() and not parent.is_symlink():
            if directory:
                raise ValueError(f"Missing report directory: {parent}")
            continue
        mode = parent.lstat().st_mode
        if stat.S_ISLNK(mode) or parent != path and not stat.S_ISDIR(mode):
            raise ValueError(f"Refusing a symlink or non-directory ancestor: {parent}")
    if directory and not path.is_dir():
        raise ValueError(f"Directory required: {path}")
    return path


def read_json(path):
    return reporting._read_json(path)


def child_directories(path, budget):
    if not path.exists():
        return []
    children = []
    with os.scandir(path) as entries:
        for entry in entries:
            budget[0] += 1
            if budget[0] > MAX_ENTRIES:
                raise ValueError("Report discovery limit reached; select a specific source run")
            if entry.is_dir(follow_symlinks=False):
                children.append(Path(entry.path))
    return sorted(children)


def find_reports(root, source=None):
    """Inspect only known report layouts; never recurse through data/caches."""
    root = Path(root).absolute()
    budget = [0]
    selected = project_path(root, source) if source is not None else root / "runs"
    if not selected.exists():
        return []
    project_path(root, selected)
    sources = [selected] if source is not None else child_directories(selected, budget)
    candidates = set()
    for source_dir in sources:
        if (source_dir / "run.json").is_file() and not (source_dir / "run.json").is_symlink():
            candidates.add(source_dir)
        parents = [source_dir / "tower", source_dir.parent / (source_dir.name + "-tower")]
        if source_dir.name == "tower" or source_dir.name.endswith("-tower"):
            parents.append(source_dir)
        # Workflow stage reports use stage-name-tower beside their source stage.
        for child in child_directories(source_dir, budget):
            if child.name == "tower" or child.name.endswith("-tower"):
                parents.append(child)
        for parent in set(parents):
            if not parent.exists() or parent.is_symlink():
                continue
            project_path(root, parent)
            for report in child_directories(parent, budget):
                if (report / "run.json").is_file() and not (report / "run.json").is_symlink():
                    candidates.add(report)
                    if len(candidates) > MAX_REPORTS:
                        raise ValueError("More than 256 reports; select a specific source run")
    found = []
    for path in sorted(candidates):
        manifest = read_json(path / "run.json")
        if isinstance(manifest, dict) and manifest.get("schema") == "tower.run/v1":
            found.append((path, manifest))
    return found


def tower_profile(root, report):
    manifest = read_json(Path(report) / "run.json")
    mode = manifest.get("parameters", {}).get("execution_mode", manifest.get("metadata", {}).get("execution_mode"))
    if mode == "desktop-slurm":
        config = project_path(root, Path(root) / ".tower/fedora-slurm.json", directory=False)
        if not config.is_file():
            raise ValueError("Fedora Tower configuration is missing; run fedora_slurm.sh configure first")
        return "desktop-slurm", config
    return "carc", Path(root) / ".tower/config.json"


def tower_command(root, report_dir, *, view="experiment"):
    root = Path(root).absolute()
    report = project_path(root, report_dir)
    profile, config = tower_profile(root, report)
    return ["tower", "--profile", profile, "--config", str(config),
            "--workdir", str(report), "--tab", "research", "--research-view", view]


def selected_report(root, value):
    """Resolve the explicit convenience token without consulting live job order."""
    if str(value) != "latest":
        return project_path(root, value)
    rows = find_reports(root)
    if not rows:
        raise ValueError("No Tower reports exist yet; import a historical stage or start a new TDN workflow")
    def created(row):
        path, manifest = row
        metadata = manifest.get("metadata", {})
        value = metadata.get("exported_at", manifest.get("start")) if isinstance(metadata, dict) else manifest.get("start")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            value = path.stat().st_mtime
        return value, path.as_posix()
    path, manifest = max(rows, key=created)
    metadata = manifest.get("metadata", {})
    source = metadata.get("source_directory", "unknown") if isinstance(metadata, dict) else "unknown"
    print(f"Selected report: {path}\nSource: {source}; job: {manifest.get('job_id', 'unknown')}; "
          f"name: {manifest.get('name', 'unknown')}; state: {manifest.get('state', 'unknown')}", file=sys.stderr)
    return path


def native_validation_command(root, report):
    """Use Tower's existing API for the explicitly bounded larger suite only."""
    manifest = read_json(report / "run.json")
    parameters = manifest.get("parameters", {})
    consistency = (isinstance(parameters, dict) and parameters.get("benchmark_suite") == "consistency"
                   or str(manifest.get("name", "")).startswith("TDN/consistency/"))
    _, config = tower_profile(root, report)
    ordinary = ["tower", "--no-state", "--no-plugins", "--config", str(config),
                "run", "validate", str(root / ".tower/contracts/outputs.v1.json"), str(report)]
    if not consistency:
        return ordinary, {}
    executable = shutil.which("tower")
    if executable is None:
        raise ValueError("Native validation requires your existing Tower executable on PATH")
    interpreter, source_root = tower_interpreter(executable)
    command = [interpreter, str(root / "scripts/tower_native_validate.py"),
               str(root / ".tower/contracts/outputs.v1.json"), str(report), "--max-bytes", str(32 << 20)]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if source_root is not None:
        environment["PYTHONPATH"] = str(source_root) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    return command, {"env": environment}


def tower_interpreter(executable):
    """Resolve installed Python/source launchers without evaluating shell text."""
    supplied = os.environ.get("TDN_TOWER_PYTHON")
    if supplied:
        interpreter = Path(supplied)
        if not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError("TDN_TOWER_PYTHON must name an existing absolute executable Python path")
        return str(interpreter), None
    path = Path(executable).resolve()
    if not path.is_file():
        raise ValueError("Existing Tower launcher is not a regular file")
    with path.open("rb") as stream:
        first = stream.readline(4097)
    if not first.startswith(b"#!") or len(first) > 4096:
        raise ValueError("Cannot resolve Tower's Python; set TDN_TOWER_PYTHON to its absolute interpreter path")
    try:
        words = shlex.split(first[2:].decode().strip())
    except (UnicodeError, ValueError) as error:
        raise ValueError("Cannot read Tower launcher shebang; set TDN_TOWER_PYTHON") from error
    if words and Path(words[0]).name == "env":
        if len(words) == 2 and re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", words[1]):
            resolved = shutil.which(words[1])
            if resolved:
                return resolved, None
    elif len(words) == 1 and re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(words[0]).name):
        if Path(words[0]).is_absolute() and Path(words[0]).is_file() and os.access(words[0], os.X_OK):
            return words[0], None
    # The unmodified upstream source launcher has this exact conventional
    # scripts/tower -> ../tower package layout. Import its package read-only.
    source = path.parent.parent
    if path.name == "tower" and path.parent.name == "scripts" and (source / "tower/artifacts.py").is_file():
        own_python = source / ".venv/bin/python"
        interpreter = str(own_python) if own_python.is_file() and os.access(own_python, os.X_OK) else shutil.which("python3")
        if interpreter:
            return interpreter, source
    raise ValueError("Unsupported Tower launcher; set TDN_TOWER_PYTHON to the Python interpreter that imports your installed Tower")


def inspect_report(root, report_dir):
    """Bounded structural checks, deliberately not a full JSON Schema validator."""
    report = project_path(root, report_dir)
    manifest = read_json(report / "run.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != "tower.run/v1":
        raise ValueError("Expected a tower.run/v1 inventory")
    if manifest.get("run_id") != report.name:
        raise ValueError("Run identity differs from its report directory")
    for key in ("experiment_id", "attempt", "state"):
        if key not in manifest:
            raise ValueError(f"Missing run field: {key}")
    logs = read_json(report / "logs.json")
    reporting._log_index(report, logs, manifest)
    summary = None
    if (report / "summary.json").exists():
        summary = reporting._summary(read_json(report / "summary.json"))
        if summary["id"] != manifest["run_id"]:
            raise ValueError("Summary identity differs from the inventory")
    metrics_path = report / "metrics.jsonl"
    fd = os.open(metrics_path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    rows = 0
    metric_names = set()
    def unique_pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError("Duplicate JSON keys in metrics")
            value[key] = child
        return value

    def numeric(value):
        try:
            return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            return False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METRICS_BYTES:
            raise ValueError("Metrics must be a regular file of at most 64 MiB")
        with os.fdopen(fd, "rb") as stream:
            fd = None
            while True:
                raw = stream.readline(65537)
                if not raw:
                    break
                if len(raw) > 65536 or not raw.endswith(b"\n"):
                    raise ValueError("Metrics contains an oversized or incomplete row; retry a live report")
                row = json.loads(raw, object_pairs_hook=unique_pairs,
                                 parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Nonfinite metric")))
                if (not isinstance(row, dict) or not isinstance(row.get("metrics"), dict)
                        or len(row["metrics"]) > 64):
                    raise ValueError("Invalid metrics row")
                for key, value in row["metrics"].items():
                    if not isinstance(key, str) or not key or len(key) > 96 or not key.isprintable():
                        raise ValueError("Invalid metric name")
                    if not numeric(value):
                        raise ValueError("Metrics must contain finite numbers")
                metric_names.update(row["metrics"])
                if len(metric_names) > 64:
                    raise ValueError("Metrics stream exceeds 64 unique names")
                timestamp = row.get("t")
                if not numeric(timestamp) or timestamp < 0:
                    raise ValueError("Metrics require a finite nonnegative epoch timestamp")
                if "step" in row and (isinstance(row["step"], bool) or not isinstance(row["step"], int)
                                      or not 0 <= row["step"] <= (1 << 63) - 1):
                    raise ValueError("Metrics step must be a bounded nonnegative integer")
                if "phase" in row and (not isinstance(row["phase"], str) or not row["phase"]
                                       or len(row["phase"]) > 160 or not row["phase"].isprintable()):
                    raise ValueError("Invalid metrics phase")
                if "progress" in row:
                    progress = row["progress"]
                    if (not isinstance(progress, dict) or not numeric(progress.get("completed"))
                            or not numeric(progress.get("total")) or progress["total"] <= 0
                            or not 0 <= progress["completed"] <= progress["total"]):
                        raise ValueError("Metrics progress must be finite and within a positive total")
                    if "unit" in progress and (not isinstance(progress["unit"], str) or not progress["unit"]
                                              or len(progress["unit"]) > 64 or not progress["unit"].isprintable()):
                        raise ValueError("Invalid metrics progress unit")
                rows += 1
    finally:
        if fd is not None:
            os.close(fd)
    return {"validation": "tdn_structural_checks_only", "report_dir": str(report),
            "state": manifest["state"], "terminal_summary_present": summary is not None,
            "metric_rows": rows, "log_entries": len(logs["logs"])}


def accounting_record(job_id, *, runner=subprocess.run):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError("Reconciliation requires a recorded exact numeric Slurm job or array-task ID")
    command = ["sacct", "--noheader", "--parsable2", "--allocations", "--jobs", job_id,
               "--format=JobIDRaw,State,ExitCode,ElapsedRaw"]
    result = runner(command, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(f"sacct failed for job {job_id}: {result.stderr[:2000]}")
    if len(result.stdout.encode("utf-8")) > 1 << 20:
        raise ValueError("Accounting output exceeded the bounded read budget")
    records = []
    for line in result.stdout.splitlines():
        values = line.strip().split("|")
        if values and values[-1] == "":
            values.pop()
        if len(values) == 4 and values[0] == job_id:
            state = values[1].split(" ", 1)[0]
            if state.endswith("+"):
                raise ValueError("Accounting state is truncated; refusing to infer it")
            if state not in reporting.TERMINAL - {"UNKNOWN", "INTERRUPTED"}:
                raise ValueError(f"Job {job_id} has no verified terminal scheduler state: {state}")
            if not re.fullmatch(r"\d+:\d+", values[2]) or not values[3].isdigit():
                raise ValueError("Accounting exit status or elapsed duration is malformed")
            records.append({"job_id": job_id, "state": state, "exit_code": values[2],
                            "elapsed_seconds": int(values[3]), "command": command})
    if len(records) != 1:
        raise ValueError("Accounting did not return exactly one matching allocation record")
    return records[0]


def reconcile_report(root, report_dir, *, runner=subprocess.run):
    report = project_path(root, report_dir)
    manifest = read_json(report / "run.json")
    evidence = accounting_record(manifest.get("job_id"), runner=runner)
    return reporting.reconcile_report(report, evidence)


def import_report(root, source_dir):
    """Snapshot a terminal historical stage without inventing execution events."""
    from tdn.tower_analytics import publish_outputs

    root = Path(root).absolute()
    source = project_path(root, source_dir)
    stage = read_json(source / "stage.json")
    if not isinstance(stage, dict):
        raise ValueError("Import requires an explicit science stage with stage.json; select a workflow's stage directory")
    original_status = stage.get("status")
    terminal_state = "INTERRUPTED" if original_status == "PAUSED_NEEDS_RESUME" else original_status
    if not isinstance(terminal_state, str) or terminal_state not in reporting.TERMINAL:
        raise ValueError("Historical import requires recorded terminal stage evidence; live stages cannot be imported")
    if terminal_state == "COMPLETED" and stage.get("actually_ran") is not True:
        raise ValueError("Historical import requires recorded actually_ran=true execution evidence")
    runtime = stage.get("elapsed_seconds")
    if runtime is not None and (isinstance(runtime, bool) or not isinstance(runtime, (int, float))
                                or not math.isfinite(runtime) or runtime < 0):
        raise ValueError("Recorded historical elapsed_seconds must be finite and nonnegative")
    source_identity = hashlib.sha256(json.dumps({"source": source.relative_to(root).as_posix(), "stage": stage},
                                    sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    software = stage.get("software") if isinstance(stage.get("software"), dict) else {}
    parameters = {"historical_import": True}
    for source_key, target_key, owner in (("source_tree_sha256", "source_tree_sha256", software),
                                          ("config_hash", "config_sha256", stage),
                                          ("protocol_sha256", "protocol_sha256", stage)):
        value = owner.get(source_key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            parameters[target_key] = value
    metadata = {"source_directory": source.relative_to(root).as_posix(),
                "source_observation_id": source_identity, "imported_observation": True,
                "independent_repeat": False, "source_verifiability": "unknown; historical source was not rerun or revalidated",
                "stale_source_possible": True, "reporting_scope": "historical science stage snapshot",
                "original_application_state": original_status,
                "exported_at": time.time(), "execution_mode": stage.get("execution_mode", software.get("execution_mode", "unknown"))}
    stage_name = stage.get("stage", "unknown")
    if not isinstance(stage_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", stage_name):
        raise ValueError("Historical stage has no safe stable workload name")
    parent = source.parent / (source.name + "-tower")
    project_path(root, parent, directory=False)
    parent.mkdir(mode=0o700, exist_ok=True)
    report_id = "tdn-import-" + uuid.uuid4().hex
    report = parent / report_id
    report.mkdir(mode=0o700, exist_ok=False)
    (report / "outputs").mkdir(mode=0o700)
    (report / "passports").mkdir(mode=0o700)
    with (report / "metrics.jsonl").open("x", encoding="utf-8") as stream:
        # An export event is not a historical training observation or ETA sample.
        stream.write(json.dumps({"t": metadata["exported_at"], "phase": "historical-report-export", "metrics": {}}) + "\n")
    job = software.get("slurm_job_id")
    if not isinstance(job, str) or not JOB_ID.fullmatch(job):
        job = None
    manifest = {"schema": "tower.run/v1", "run_id": report_id, "project_id": "TDN",
                "experiment_id": "tdn-import-" + source_identity[:24], "attempt": 1,
                "name": "tdn/" + stage_name, "state": "RUNNING", "parameters": parameters,
                "metadata": metadata,
                "paths": {"metrics": "metrics.jsonl", "summary": "summary.json", "log_index": "logs.json", "outputs": "outputs", "passports": "passports"}}
    if job:
        manifest["job_id"] = job
    index = {"schema": "tower.logs/v1", "run_id": report_id,
             "logs": [{"id": "historical.stage", "path": os.path.relpath(source / "stage.json", report),
                       "label": "Original stage evidence", "group": "Historical source"}]}
    if job:
        index["job_id"] = job
    reporting._atomic_json(report / "run.json", manifest)
    reporting._atomic_json(report / "logs.json", index)
    results = publish_outputs(report, [source])
    summary = {"schema": "tower.summary/v1", "project_id": "TDN", "id": report_id,
               "name": manifest["name"], "experiment_id": manifest["experiment_id"], "attempt": 1,
               "state": terminal_state, "parameters": parameters, "metadata": metadata,
               "results": results, "fingerprint": "tdn-source-" + source_identity}
    if job:
        summary["job_id"] = job
    if runtime is not None:
        summary["runtime_seconds"] = runtime
        summary["metadata"]["runtime_source"] = "original stage.json elapsed_seconds; excludes this export"
    reporting._summary(summary)
    reporting._atomic_json(report / "summary.json", summary)
    manifest["state"] = terminal_state
    reporting._atomic_json(report / "run.json", manifest, replace=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    listing = commands.add_parser("list", help="List bounded, known report locations")
    listing.add_argument("source", nargs="?", type=Path)
    for action in ("show", "open"):
        sub = commands.add_parser(action, help="Print a Tower command" if action == "show" else "Launch installed Tower explicitly")
        sub.add_argument("report", type=Path)
        sub.add_argument("--view", default="experiment", choices=("experiment", "artifacts", "predict", "tradeoffs"))
    validate = commands.add_parser("validate", help="Bounded TDN structure checks, optionally Tower's actual validator")
    validate.add_argument("report", type=Path)
    validate.add_argument("--native", action="store_true", help="Also run the installed Tower artifact validator")
    export = commands.add_parser("export", help="Export explicitly selected independent observations")
    export.add_argument("reports", nargs="+", type=Path)
    export.add_argument("--reference", type=Path)
    export.add_argument("--output", type=Path, default=Path("reports/planning.json"))
    reconcile = commands.add_parser("reconcile", help="Record actual terminal sacct evidence for the report's exact job")
    reconcile.add_argument("report", type=Path)
    importing = commands.add_parser("import", help="Create a derivative report for an explicitly terminal historical stage")
    importing.add_argument("source", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            rows = find_reports(ROOT, args.source)
            for path, manifest in rows:
                print(f"{path}\t{manifest['state']}\tjob={manifest.get('job_id', 'unknown')}")
            if not rows:
                print("No Tower reports found in the selected source. Completed historical runs require an explicit import.")
        elif args.action in {"show", "open"}:
            report = selected_report(ROOT, args.report)
            inspect_report(ROOT, report)
            command = tower_command(ROOT, report, view=args.view)
            print(shlex.join(command), flush=True)
            if args.action == "open":
                if shutil.which("tower") is None:
                    raise ValueError("Tower is not on PATH; use your existing Tower installation")
                return subprocess.run(command, cwd=ROOT, check=False).returncode
        elif args.action == "validate":
            report = selected_report(ROOT, args.report)
            print(json.dumps(inspect_report(ROOT, report), indent=2))
            if args.native:
                if shutil.which("tower") is None:
                    raise ValueError("Native validation requires your existing Tower executable on PATH")
                command, options = native_validation_command(ROOT, report)
                return subprocess.run(command, cwd=ROOT, check=False, **options).returncode
        elif args.action == "export":
            reports = [project_path(ROOT, path) for path in args.reports]
            reference = project_path(ROOT, args.reference) if args.reference else None
            output = project_path(ROOT, args.output, directory=False)
            result = reporting.export_planning(reports, reference_run=reference, output=output)
            print(result if isinstance(result, (str, Path)) else json.dumps(result, indent=2))
        elif args.action == "reconcile":
            print(json.dumps(reconcile_report(ROOT, args.report), indent=2))
        elif args.action == "import":
            print(import_report(ROOT, args.source))
    except (ValueError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"TDN Tower: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
