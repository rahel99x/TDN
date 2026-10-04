#!/usr/bin/env python3
"""TDN producers for Tower v1, without a Tower or numerical dependency.

Adapted from Slurm-Tower contributors (2026), MIT; see
docs/TOWER_THIRD_PARTY.md. Science files are never rewritten by this module.
Unknown measurements stay unknown. Cooperating writers serialize complete
records under a project-local lock; each execution attempt owns a fresh report.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from numbers import Real

ROOT = Path(__file__).resolve().parents[1]

MAX_JSON = 1 << 20
MAX_RUNS = 256
MAX_AGGREGATE = 1 << 20
MAX_SCRIPT = 8 << 20
MAX_LOGS = 256
MAX_LOG_JSON = 256 << 10
RESOURCE_KEYS = {"partition", "cpus", "nodes", "gpus", "gpu_type", "account", "qos",
                 "mem_bytes", "time_seconds"}
QUERY_KEYS = RESOURCE_KEYS | {"name", "script_sha256", "parameters", "input_size", "memory_scope"}
SUMMARY_KEYS = RESOURCE_KEYS | {"schema", "id", "name", "state", "memory_scope", "runtime_seconds",
    "memory_bytes", "cpu_seconds", "script_sha256", "input_size", "start", "end", "submit", "job_id",
    "experiment_id", "attempt", "project_id", "exit_code", "workers", "problem_size", "repeat",
    "fingerprint", "work_units", "parameters", "results", "metadata"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "UNKNOWN", "INTERRUPTED"}


def _text(value, name, limit=128):
    if not isinstance(value, str) or not value or len(value) > limit or not value.isprintable():
        raise ValueError(f"{name} must be a nonempty printable string of at most {limit} characters")
    return value


def _number(value, name, *, minimum=0, maximum=1e100, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid or integer and (not isinstance(value, int) or value > (1 << 63) - 1):
        raise ValueError(f"{name} must be a finite {'integer ' if integer else ''}number >= {minimum}")
    return value


def _directory(path):
    path = Path(path).absolute()
    for component in reversed((path, *path.parents)):
        if not stat.S_ISDIR(component.lstat().st_mode):
            raise ValueError(f"directory required; symlinks refused: {component}")
    return path


def _under(root, relative, *, create_parent=False):
    root = _directory(root)
    raw_input = str(relative)
    if (len(raw_input) > 4096 or "\\" in raw_input or any(c in raw_input for c in "*?[]") or not raw_input.isprintable()
            or any(p in {".", ".."} for p in raw_input.split("/"))
            or "//" in raw_input or raw_input.endswith("/")):
        raise ValueError("path must be exact, printable, and without traversal, globs, or empty components")
    path = Path(relative)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError as exc:
            raise ValueError("path must be inside the selected project") from exc
    raw = str(path)
    if raw in {"", "."} or "\\" in raw or any(p in {"", ".", ".."} for p in raw.split("/")):
        raise ValueError("path must be an exact project-relative path without traversal")
    parent = root
    for component in path.parts[:-1]:
        parent /= component
        if create_parent:
            parent.mkdir(mode=0o700, exist_ok=True)
        _directory(parent)
    return root / path


def _encoded(value, limit=MAX_JSON):
    stack, count, text_chars = [(value, 0)], 0, 0
    while stack:
        current, depth = stack.pop()
        count += 1
        if depth > 32 or count > 100000:
            raise ValueError("JSON exceeds the depth or value budget")
        if isinstance(current, dict):
            if any(not isinstance(key, str) for key in current):
                raise ValueError("JSON object keys must be strings")
            if len(current) + len(stack) + count > 100000:
                raise ValueError("JSON exceeds the value budget")
            text_chars += sum(len(key) for key in current)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            if len(current) + len(stack) + count > 100000:
                raise ValueError("JSON exceeds the value budget")
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            text_chars += len(current)
        elif isinstance(current, int) and not isinstance(current, bool) and current.bit_length() > 256:
            raise ValueError("JSON integer exceeds the bounded reader profile")
        if text_chars > limit:
            raise ValueError(f"JSON text exceeds the {limit}-byte producer budget")
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("utf-8")
    if len(raw) > limit:
        raise ValueError(f"JSON exceeds the {limit}-byte producer budget")
    return raw


def _read_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON:
            raise ValueError("JSON must be a regular file of at most 1 MiB")
        chunks, remaining = [], MAX_JSON + 1
        while remaining:
            part = os.read(fd, min(65536, remaining))
            if not part:
                break
            chunks.append(part)
            remaining -= len(part)
        raw = b"".join(chunks)
        after, named = os.fstat(fd), os.stat(path, follow_symlinks=False)
        signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                   value.st_mtime_ns, value.st_ctime_ns)
        if len(raw) > MAX_JSON or len(raw) != before.st_size or signature(before) != signature(after) or signature(after) != signature(named):
            raise ValueError("JSON changed while being read or exceeds the byte budget")
    finally:
        os.close(fd)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON keys are refused")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid bounded UTF-8 JSON") from exc
    _encoded(value)  # rejects exponent overflow and out-of-budget values
    return value


def _atomic_json(path, value, *, replace=False):
    path = Path(path)
    _directory(path.parent)
    raw = _encoded(value)
    if replace and os.path.lexists(path) and not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("refusing to replace a nonregular file")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # exclusive publication: existing files survive
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resources(value):
    if not isinstance(value, dict) or set(value) - RESOURCE_KEYS:
        raise ValueError("resources must contain only documented request/allocation fields")
    out = {}
    for key, item in value.items():
        if item is None:
            continue
        if key in {"cpus", "nodes", "gpus", "mem_bytes"}:
            out[key] = _number(item, key, minimum=0 if key == "gpus" else 1,
                               maximum=(1 << 63) - 1 if key == "mem_bytes" else 1_000_000_000, integer=True)
        elif key == "time_seconds":
            out[key] = _number(item, key, minimum=1e-9, maximum=3_162_240_000)
        else:
            out[key] = "" if key == "gpu_type" and item == "" else _text(item, key)
    return out


def _parameters(value):
    if not isinstance(value, dict) or len(value) > 64:
        raise ValueError("parameters must be an object with at most 64 entries")
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > 4 or count > 128:
            raise ValueError("parameters exceed four nesting levels or 128 total values")
        if isinstance(item, dict):
            if len(item) > 64:
                raise ValueError("parameters object exceeds 64 entries")
            for key, child in item.items():
                _text(key, "parameter name", 64)
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            if len(item) > 64:
                raise ValueError("parameter list exceeds 64 entries")
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            if item != "":
                _text(item, "parameter value", 256)
        elif item is None or isinstance(item, bool):
            pass
        else:
            _number(item, "parameter value", minimum=-1e100)
    _encoded(value, 16384)
    return value


def _log_path(run_dir, value):
    """Validate an exact location without opening, scanning, or creating logs."""
    raw = _text(value, "log path", 4096)
    if "\\" in raw or any(c in raw for c in "*?[]") or raw.endswith("/") or raw.startswith("~"):
        raise ValueError("log path must name an exact file without globs or backslashes")
    supplied = Path(raw)
    path = Path(os.path.normpath(supplied if supplied.is_absolute() else run_dir / supplied))
    if len(str(path)) > 4096:
        raise ValueError("resolved log path exceeds 4096 characters")
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue  # a registered log may be created later by its actual writer
        if component == path:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("registered log must be a regular file when present")
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError("log path ancestors must be directories; symlinks refused")
    return str(path)


def _portable_log_path(run_dir, value):
    """Keep project-owned attachments usable after the bundle is relocated."""
    raw = Path(value).as_posix()
    path = Path(raw)
    if path.is_absolute() and path.is_relative_to(ROOT):
        return Path(os.path.relpath(path, run_dir)).as_posix()
    return raw


def _log_index(run_dir, value, manifest=None):
    if (not isinstance(value, dict) or value.get("schema") != "tower.logs/v1"
            or set(value) - {"schema", "logs", "run_id", "job_id"}
            or not isinstance(value.get("logs"), list) or len(value["logs"]) > MAX_LOGS):
        raise ValueError("log index must be a bounded tower.logs/v1 object")
    if "run_id" in value and value["run_id"] != run_dir.name:
        raise ValueError("log index run_id must describe the selected run")
    if "job_id" in value:
        _text(value["job_id"], "log index job_id")
        if manifest is not None and value["job_id"] != manifest.get("job_id"):
            raise ValueError("log index job_id must match the actual run job_id")
    ids, paths = set(), set()
    for entry in value["logs"]:
        if not isinstance(entry, dict) or set(entry) - {"id", "path", "label", "group", "description"}:
            raise ValueError("log entries use id, path, label, group, and description only")
        ident = entry.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", ident):
            raise ValueError("log id must be a safe stable ID of at most 128 characters")
        path = _log_path(run_dir, entry.get("path"))
        if ident in ids or path in paths:
            raise ValueError("log IDs and resolved paths must be unique")
        ids.add(ident)
        paths.add(path)
        for key, limit in (("label", 160), ("group", 160), ("description", 512)):
            if key in entry and (entry[key] != "" or key != "description"):
                _text(entry[key], f"log {key}", limit)
    _encoded(value, MAX_LOG_JSON)
    return value


def register_log(run_dir, id, path, *, label="", group="", description=""):
    """Atomically register another exact log location for one run coordinator.

    Relative paths, including explicit sibling locations, resolve to this run
    directory. Absolute paths can name
    explicit external logs; they must be rebound when a project is relocated.
    Logs may not exist yet. This helper never opens their contents or scans a
    directory, and it refuses duplicate IDs/paths and symlinks when present.
    """
    run_dir = _project_directory(run_dir)
    with _lock(run_dir):
        manifest = _manifest(run_dir)
        index_path = run_dir / "logs.json"
        index = _log_index(run_dir, _read_json(index_path), manifest)
        entry = {"id": id, "path": _portable_log_path(run_dir, path)}
        for key, value in (("label", label), ("group", group), ("description", description)):
            if value != "":
                entry[key] = value
        index["logs"].append(entry)
        _log_index(run_dir, index, manifest)
        _atomic_json(index_path, index, replace=True)
        return entry


def _summary(row):
    """Check the shared producer profile without adding a schema dependency."""
    if (not isinstance(row, dict) or row.get("schema") != "tower.summary/v1"
            or not isinstance(row.get("state"), str) or row["state"] not in TERMINAL):
        raise ValueError("each selected run must have a terminal tower.summary/v1 object")
    if set(row) - SUMMARY_KEYS:
        raise ValueError("unknown summary fields; place project extensions in results or metadata")
    ident = row.get("id")
    if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", ident):
        raise ValueError("summary.id must be a safe unique attempt ID of at most 128 characters")
    _text(row.get("name"), "summary.name")
    _resources({key: value for key, value in row.items() if key in RESOURCE_KEYS})
    if "parameters" in row:
        _parameters(row["parameters"])
    if row.get("script_sha256") is not None and (not isinstance(row["script_sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", row["script_sha256"])):
        raise ValueError("script_sha256 must be a real 64-character hexadecimal digest")
    for key in ("job_id", "experiment_id", "project_id"):
        if row.get(key) is not None:
            _text(row[key], key)
    if row.get("fingerprint") is not None:
        _text(row["fingerprint"], "fingerprint", 256)
    if row.get("attempt") is not None:
        _number(row["attempt"], "attempt", minimum=1, maximum=2_147_483_647, integer=True)
    for key in ("start", "end", "submit"):
        if row.get(key) is not None:
            _number(row[key], key, maximum=253402300799)
    if row.get("start") is not None and row.get("end") is not None and row["end"] < row["start"]:
        raise ValueError("end must not precede start")
    for key in ("runtime_seconds", "cpu_seconds", "memory_bytes", "input_size"):
        if row.get(key) is not None:
            _number(row[key], key)
    if row.get("memory_scope") is not None and (not isinstance(row["memory_scope"], str) or row["memory_scope"] not in {"job_peak", "per_node_peak", "max_task_rss"}):
        raise ValueError("invalid memory_scope")
    if row.get("memory_bytes") is not None and row.get("memory_scope") is None:
        raise ValueError("observed memory requires memory_scope")
    for key, maximum in (("workers", 1_000_000), ("repeat", 10_000)):
        if row.get(key) is not None:
            _number(row[key], key, minimum=1, maximum=maximum, integer=True)
    for key in ("problem_size", "work_units"):
        if row.get(key) is not None:
            _number(row[key], key, minimum=1e-300, maximum=1e300)
    if row.get("exit_code") is not None:
        _number(row["exit_code"], "exit_code", minimum=-2_147_483_648, maximum=2_147_483_647, integer=True)
        if row["state"] == "COMPLETED" and row["exit_code"] != 0:
            raise ValueError("COMPLETED requires exit_code zero when recorded")
    for key in ("results", "metadata"):
        if key in row and not isinstance(row[key], dict):
            raise ValueError(f"{key} must be an object")
    _encoded(row)
    return row


def _project_path(path, *, create_parent=False):
    # Native Windows filesystem inputs use backslashes; interchange paths do
    # not. Path.as_posix normalizes native separators before strict checking.
    return _under(ROOT, Path(path).as_posix(), create_parent=create_parent)


def _project_directory(path):
    return _directory(_project_path(path))


def read_json(path):
    """Read bounded, finite, unique-key JSON from a regular project file."""
    return _read_json(_project_path(path))


def atomic_json(path, value, *, replace=True):
    """Publish bounded JSON atomically; all temporary files stay in the project."""
    path = _project_path(path, create_parent=True)
    _atomic_json(path, value, replace=replace)


@contextmanager
def _lock(directory):
    """Serialize cooperating processes without lock files outside the project."""
    path = directory / ".report.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0), 0o600)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("report lock must be a regular file")
        if os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _manifest(run_dir):
    manifest = _read_json(run_dir / "run.json")
    if (not isinstance(manifest, dict) or manifest.get("schema") != "tower.run/v1"
            or manifest.get("run_id") != run_dir.name):
        raise ValueError("run inventory must describe the selected report directory")
    return manifest


def scheduler_job_id():
    """Return only an actual declared Slurm scheduling identity, never a step."""
    parent, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if parent and task:
        value = f"{parent}_{task}"
    else:
        value = os.environ.get("SLURM_JOB_ID")
    if value is not None and not re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", value):
        raise ValueError("invalid actual Slurm job identity")
    return value


def begin_report(source_dir, *, name, script, parameters=None, resources=None,
                 job_id=None, report_parent=None, logs=None, metadata=None):
    """Create one exclusive execution report beside, never inside, science data.

    ``resources`` contains only known allocation/request facts supplied by the
    launcher. Local reports do not infer allocation counts from the machine.
    The caller controls ownership and propagates TDN_TOWER_DIR to child work.
    """
    source = _project_path(source_dir)
    if source.exists():
        _directory(source)
    name = _text(name, "name")
    parameters = _parameters(parameters or {})
    resource_data = _resources(resources or {})
    metadata = {} if metadata is None else dict(metadata)
    metadata.setdefault("source_directory", source.relative_to(ROOT).as_posix())
    metadata.setdefault("reporting_scope", "application coordinator")
    _encoded(metadata, 16384)
    script_path = _project_path(script)
    if Path(script).is_absolute():
        raise ValueError("script must be relative to the project root")
    # A bounded regular-file read also detects symlink/device replacement.
    fd = os.open(script_path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SCRIPT:
            raise ValueError("script must be a regular file of at most 8 MiB")
        digest = hashlib.sha256()
        consumed = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > MAX_SCRIPT:
                raise ValueError("script exceeds its byte budget")
            digest.update(chunk)
        after, named = os.fstat(fd), script_path.stat(follow_symlinks=False)
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if signature(before) != signature(after) or signature(after) != signature(named):
            raise ValueError("script changed while hashing")
    finally:
        os.close(fd)
    actual_job = scheduler_job_id()
    if job_id is None:
        job_id = actual_job
    if job_id is not None:
        if not isinstance(job_id, str) or not re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", job_id):
            raise ValueError("job_id must be an actual Slurm scheduling identity")
        if actual_job is not None and job_id != actual_job:
            raise ValueError("report job_id differs from the allocated job")
    parent = _project_path(report_parent or source.parent / (source.name + "-tower"), create_parent=True)
    if parent == source or parent.is_relative_to(source):
        coordinator_child = (report_parent is not None and parent == source / "tower"
                             and not any((source / key).exists() for key in ("protocol.json", "manifest.json")))
        if not coordinator_child:
            raise ValueError("Tower reports must be outside science directories; only an explicit coordinator tower/ is allowed")
    parent.mkdir(mode=0o700, exist_ok=True)
    _directory(parent)
    experiment = "tdn-" + hashlib.sha256(_encoded({"name": name, "parameters": parameters})).hexdigest()[:24]
    with _lock(parent):
        counter_path = parent / ".attempts.json"
        counters = _read_json(counter_path) if counter_path.exists() else {}
        if not isinstance(counters, dict):
            raise ValueError("invalid report attempt counter")
        attempt = counters.get(experiment, 0) + 1
        _number(attempt, "attempt", minimum=1, maximum=2_147_483_647, integer=True)
        run_id = "tdn-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12]
        run_dir = parent / run_id
        run_dir.mkdir(mode=0o700)
        counters[experiment] = attempt
        _atomic_json(counter_path, counters, replace=True)
    for child in ("outputs", "passports"):
        (run_dir / child).mkdir(mode=0o700)
    (run_dir / "metrics.jsonl").touch(mode=0o600, exist_ok=False)
    manifest = {"schema": "tower.run/v1", "project_id": "TDN", "run_id": run_id,
                "experiment_id": experiment, "attempt": attempt, "name": name,
                "state": "RUNNING", "start": time.time(), "parameters": parameters,
                "resources": resource_data, "metadata": metadata,
                "provenance": {"script": script_path.relative_to(ROOT).as_posix(), "script_sha256": digest.hexdigest()},
                "paths": {"metrics": "metrics.jsonl", "summary": "summary.json", "outputs": "outputs",
                          "log_index": "logs.json", "passports": "passports"}}
    if job_id is not None:
        manifest["job_id"] = job_id
    entries = [{**entry, "path": _portable_log_path(run_dir, entry["path"])} for entry in (logs or [])]
    index = {"schema": "tower.logs/v1", "run_id": run_id, "logs": entries}
    if job_id is not None:
        index["job_id"] = job_id
    _log_index(run_dir, index, manifest)
    _atomic_json(run_dir / "run.json", manifest)
    _atomic_json(run_dir / "logs.json", index)
    _write_metric(run_dir, {"attempt_started": 1}, phase="lifecycle", step=0)
    return run_dir


def attach_report(source_dir, **kwargs):
    """Return ``(report_directory, owned)``; children inherit a single report."""
    inherited = os.environ.get("TDN_TOWER_DIR")
    if inherited:
        run_dir = _project_directory(inherited)
        manifest = _manifest(run_dir)
        if manifest.get("state") != "RUNNING":
            raise ValueError("inherited Tower report is already terminal")
        actual_job = scheduler_job_id()
        if actual_job is not None and manifest.get("job_id") != actual_job:
            raise ValueError("inherited Tower report belongs to a different job")
        return run_dir, False
    return begin_report(source_dir, **kwargs), True


def _write_metric(run_dir, metrics, *, phase, step=None, completed=None, total=None, unit=None):
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    clean = {}
    for key, value in metrics.items():
        key = _text(key, "metric name", 96)
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        try:
            numeric = float(value)
        except (ValueError, OverflowError):
            continue
        if math.isfinite(numeric):
            clean[key] = int(value) if isinstance(value, int) and abs(value) < (1 << 63) else numeric
    if len(clean) > 64:
        raise ValueError("metrics exceeds the 64-series limit")
    phase = _text(phase, "phase", 160)
    row = {"metrics": clean, "phase": phase}
    if step is not None:
        row["step"] = _number(step, "step", maximum=(1 << 63) - 1, integer=True)
    if completed is not None or total is not None:
        done = _number(completed, "completed")
        count = _number(total, "total", minimum=1e-300)
        if done > count:
            raise ValueError("progress completed exceeds total")
        row["progress"] = {"completed": done, "total": count}
        if unit is not None:
            row["progress"]["unit"] = _text(unit, "unit", 64)
    elif unit is not None:
        raise ValueError("progress unit requires completed and total")
    with _lock(run_dir):
        if _manifest(run_dir).get("state") != "RUNNING":
            raise ValueError("cannot append metrics to a terminal report")
        state_path = run_dir / ".metrics-state.json"
        saved = _read_json(state_path) if state_path.exists() else {"names": [], "phases": {}}
        names = sorted(set(saved["names"]) | set(clean))
        if len(names) > 64:
            raise ValueError("metric stream exceeds 64 distinct series")
        previous = saved["phases"].get(phase, {})
        if phase not in saved["phases"] and len(saved["phases"]) >= 1024:
            raise ValueError("metric stream exceeds bounded phase count")
        if step is not None and step < previous.get("step", 0):
            raise ValueError("metric steps must not decrease within a phase")
        timestamp = max(time.time(), saved.get("t", 0))
        row["t"] = timestamp
        raw = _encoded(row, 65536)
        fd = os.open(run_dir / "metrics.jsonl", os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("metrics must be a regular file")
            if info.st_size + len(raw) > 4 << 20:
                raise ValueError("metric stream exceeds the 4 MiB contract limit")
            if os.write(fd, raw) != len(raw):
                raise OSError("incomplete metric append")
        finally:
            os.close(fd)
        saved.update(names=names, t=timestamp)
        if step is not None:
            previous["step"] = step
        saved["phases"][phase] = previous
        _atomic_json(state_path, saved, replace=True)


def emit(metrics, *, phase, step=None, completed=None, total=None, unit=None):
    """Emit actual observations when attached; omit unknown/nonfinite values."""
    selected = os.environ.get("TDN_TOWER_DIR")
    if not selected:
        return
    return _write_metric(_project_directory(selected), metrics, phase=phase, step=step,
                         completed=completed, total=total, unit=unit)


def finish_report(report_dir, *, state, runtime_seconds=None, exit_code=None, results=None, metadata=None):
    """Seal a report once; execution completion does not imply scientific merit."""
    run_dir = _project_directory(report_dir)
    if state not in TERMINAL:
        raise ValueError("finish_report requires an explicit terminal state")
    with _lock(run_dir):
        manifest = _manifest(run_dir)
        if manifest.get("state") != "RUNNING":
            raise ValueError("report is already terminal")
        summary = {"schema": "tower.summary/v1", "project_id": "TDN", "id": manifest["run_id"],
                   "name": manifest["name"], "experiment_id": manifest["experiment_id"],
                   "attempt": manifest["attempt"], "state": state, "start": manifest["start"],
                   "end": max(time.time(), manifest["start"]), "parameters": manifest["parameters"],
                   "script_sha256": manifest["provenance"]["script_sha256"],
                   "metadata": {**manifest.get("metadata", {}), **(metadata or {})},
                   **_resources(manifest.get("resources", {}))}
        if "job_id" in manifest:
            summary["job_id"] = manifest["job_id"]
        if runtime_seconds is not None:
            summary["runtime_seconds"] = _number(runtime_seconds, "runtime_seconds")
        if exit_code is not None:
            summary["exit_code"] = exit_code
        if results is not None:
            summary["results"] = results
        _summary(summary)
        _encoded(summary, 262144)
        _atomic_json(run_dir / "summary.json", summary)
        manifest.update(state=state, end=summary["end"])
        _atomic_json(run_dir / "run.json", manifest, replace=True)
        return summary


def reconcile_report(report_dir, evidence):
    """Apply an explicit, exact-job terminal accounting observation.

    The caller obtains ``evidence`` from the scheduler. Reconciliation retains
    application measurements, results and the original application outcome;
    scheduler elapsed time is kept in its own scope, never substituted for an
    application duration. A newly observed terminal state supplies no invented
    application end timestamp or missing scientific completion evidence.
    """
    run_dir = _project_directory(report_dir)
    if not isinstance(evidence, dict):
        raise ValueError("scheduler evidence must be an object")
    state = evidence.get("state")
    if state not in TERMINAL - {"UNKNOWN", "INTERRUPTED"}:
        raise ValueError("reconciliation requires a verified terminal scheduler state")
    command = evidence.get("command")
    if (not isinstance(command, list) or not command or command[0] != "sacct"
            or any(not isinstance(part, str) or not part.isprintable() for part in command)):
        raise ValueError("record the actual sacct evidence command")
    raw_exit = evidence.get("exit_code")
    if not isinstance(raw_exit, str) or not re.fullmatch(r"[0-9]+:[0-9]+", raw_exit):
        raise ValueError("scheduler exit_code must use Slurm status:signal syntax")
    status, signal_number = map(int, raw_exit.split(":"))
    _number(status, "scheduler exit status", maximum=2_147_483_647, integer=True)
    _number(signal_number, "scheduler signal", maximum=255, integer=True)
    if state == "COMPLETED" and (status != 0 or signal_number != 0):
        raise ValueError("scheduler COMPLETED conflicts with a nonzero exit code")
    if evidence.get("elapsed_seconds") is not None:
        _number(evidence["elapsed_seconds"], "scheduler elapsed seconds")
    _encoded(evidence, 16384)
    with _lock(run_dir):
        manifest = _manifest(run_dir)
        if not manifest.get("job_id") or evidence.get("job_id") != manifest["job_id"]:
            raise ValueError("scheduler evidence must match the report's exact job_id")
        existing = (run_dir / "summary.json").exists()
        if existing:
            summary = _summary(_read_json(run_dir / "summary.json"))
            if summary["id"] != manifest["run_id"] or summary.get("job_id") != manifest["job_id"]:
                raise ValueError("summary identity differs from the run inventory")
        else:
            summary = {"schema": "tower.summary/v1", "project_id": "TDN", "id": manifest["run_id"],
                       "name": manifest["name"], "experiment_id": manifest["experiment_id"],
                       "attempt": manifest["attempt"], "job_id": manifest["job_id"],
                       "parameters": manifest["parameters"],
                       "script_sha256": manifest["provenance"]["script_sha256"],
                       "metadata": dict(manifest.get("metadata", {})),
                       **_resources(manifest.get("resources", {}))}
            if "start" in manifest:
                summary["start"] = manifest["start"]
        meta = summary.setdefault("metadata", {})
        meta.setdefault("application_state", summary.get("state", manifest["state"]))
        if "exit_code" in summary:
            meta.setdefault("application_exit_code", summary["exit_code"])
        prior = meta.get("scheduler_reconciliation")
        if isinstance(prior, dict) and prior.get("evidence") == evidence:
            return summary
        meta["scheduler_reconciliation"] = {"observed_at": time.time(), "evidence": evidence,
                                             "runtime_scope": "whole Slurm job; application runtime unchanged"}
        application_state = meta["application_state"]
        if state == "COMPLETED" and application_state != "COMPLETED":
            # A successful batch exit cannot repair a failed or unfinished
            # scientific application. Keep it out of completed prediction fits.
            final_state = application_state if application_state in TERMINAL else "UNKNOWN"
            summary["state"] = final_state
        else:
            final_state = state
            summary["state"] = state
            summary["exit_code"] = status if signal_number == 0 else 128 + signal_number
        _summary(summary)
        _encoded(summary, 262144)
        _atomic_json(run_dir / "summary.json", summary, replace=existing)
        manifest["state"] = final_state
        manifest["metadata"] = {**manifest.get("metadata", {}),
                                "scheduler_reconciliation": meta["scheduler_reconciliation"]}
        _atomic_json(run_dir / "run.json", manifest, replace=True)
        return summary


def export_planning(run_dirs, *, reference_run=None, output="reports/planning.json"):
    """Export at most 256 explicit reports; preserve failures and unknown fields."""
    if isinstance(run_dirs, (str, bytes, Path)):
        raise ValueError("run_dirs must be explicit paths, not a single string")
    rows, seen, observations, paths, consumed = [], set(), set(), {}, 0
    for index, value in enumerate(run_dirs):
        if index >= MAX_RUNS:
            raise ValueError("at most 256 report directories may be aggregated")
        run_dir = _project_directory(value)
        row = _summary(_read_json(run_dir / "summary.json"))
        manifest = _manifest(run_dir)
        if row["id"] != run_dir.name or manifest["state"] != row["state"]:
            raise ValueError("report inventory and terminal summary disagree")
        if row["id"] in seen:
            raise ValueError("duplicate report identity in aggregate")
        seen.add(row["id"])
        if row.get("job_id") is not None:
            observation = (row["job_id"], row["name"])
            if observation in observations:
                raise ValueError("duplicate scheduler observation for the same workload")
            observations.add(observation)
        source_observation = row.get("metadata", {}).get("source_observation_id")
        if source_observation is not None:
            _text(source_observation, "source observation identity", 256)
            observation = ("source_observation", source_observation)
            if observation in observations:
                raise ValueError("duplicate imported observation in aggregate")
            observations.add(observation)
        consumed += len(_encoded(row))
        if consumed > MAX_AGGREGATE:
            raise ValueError("selected summaries exceed the 1 MiB aggregate budget")
        rows.append(row)
        paths[run_dir] = row
    if not rows:
        raise ValueError("select at least one report directory")
    bundle = {"version": 1, "kind": "tower.planning", "history": rows, "scaling": []}
    if reference_run is not None:
        reference = _project_directory(reference_run)
        if reference not in paths:
            raise ValueError("reference must be one of the selected reports")
        bundle["query"] = {key: value for key, value in paths[reference].items() if key in QUERY_KEYS and value is not None}
    bundle["scaling"] = [row for row in rows if all(row.get(key) is not None for key in ("workers", "problem_size", "repeat"))]
    _encoded(bundle, MAX_AGGREGATE)
    target = _project_path(output, create_parent=True)
    if os.path.lexists(target):
        existing = _read_json(target)
        if not isinstance(existing, dict) or existing.get("version") != 1 or existing.get("kind") != "tower.planning":
            raise ValueError("refusing to replace a non-planning file")
    _atomic_json(target, bundle, replace=True)
    return bundle
