"""Bounded worker resource observations in Raven's retained-reader format.

CPU/RSS snapshots cover the local worker and its descendants, not a scheduler
allocation total. GPU samples cover explicitly assigned devices and can include
display/co-resident activity. Collection is optional evidence, never CUDA proof.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid

INDEX_SCHEMA = "raven.telemetry-index/v1"
CPU_SCHEMA = "raven.cpu-memory-sample/v1"
GPU_SCHEMA = "raven.gpu-sample/v1"
MAX_INDEX_BYTES = 256 << 10
CPU_COLUMNS = ("timestamp", "elapsed_seconds", "cpu_seconds", "cpu_utilization_percent", "rss_bytes",
               "process_count", "observer_wall_seconds", "status", "reason")
GPU_COLUMNS = ("timestamp", "elapsed_seconds", "device_index", "device_uuid", "device_bdf",
               "utilization_percent", "memory_used_mib", "memory_total_mib", "power_watts", "temperature_celsius",
               "observer_wall_seconds", "status", "reason")


def _safe(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Worker telemetry paths must not traverse parents or symlinks")
    return path


def _error(error):
    return "".join(c if c.isprintable() else " " for c in str(error))[:512]


def _atomic(path, value):
    path = _safe(path)
    raw = (json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(raw) > MAX_INDEX_BYTES:
        raise ValueError("Worker telemetry index exceeds its bounded metadata size")
    temporary = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:stream.write(raw)
        _safe(path);os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def options(value=None):
    """Resolve already-declared selectors; empty/disabled is explicit."""
    value = {"families":["cpu", "memory", "gpu"]} if value is None else value
    if value is False or value == "none":value = {"families":[]}
    if isinstance(value, str):value = {"families":value.split(",")}
    if not isinstance(value, dict):raise ValueError("Worker telemetry requires declared family options")
    families = value.get("families", value.get("metrics", value.get("enabled", ["cpu", "memory", "gpu"])))
    if families is True:families = ["cpu", "memory", "gpu"]
    if isinstance(families, str):families = families.split(",")
    if value.get("enabled") is False:families = []
    if (not isinstance(families, (list, tuple)) or len(families) != len(set(families))
            or set(families)-{"cpu", "memory", "gpu"}):
        raise ValueError("Worker telemetry selectors are cpu,memory,gpu or none")
    interval = value.get("interval_seconds", value.get("sampling_seconds", 5.))
    timeout = value.get("timeout_seconds", 2.)
    budget = value.get("max_bytes_per_stream", 32 << 20)
    for number, low, high in ((interval, .02, 300.), (timeout, .2, 10.), (budget, 4096, 256 << 20)):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError("Worker telemetry period/timeout/storage is outside its declared bound")
    return dict(families=list(families), interval_seconds=float(interval), timeout_seconds=float(timeout),
                max_bytes_per_stream=int(budget), allocated_gpu_ids=value.get("allocated_gpu_ids"))


def process_observation(pid, *, expected_start=None, proc_root="/proc"):
    """Bounded procfs tree snapshot with PID-reuse and descendant checks."""
    ticks, pages = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    pending = collections.deque([(pid, None)])
    seen, counters, issues = set(), {}, []
    cpu, rss, root_start = 0., 0, None
    deadline = time.monotonic()+.1
    while pending and len(seen) < 4096 and time.monotonic() < deadline:
        current, parent = pending.popleft()
        if current in seen:continue
        seen.add(current)
        base = Path(proc_root)/str(current)
        try:
            with (base/"stat").open("rb") as stream:raw = stream.read(8193)
            if len(raw) > 8192:raise ValueError("Process stat exceeds its inspection bound")
            fields = raw.decode("utf-8", "replace").rsplit(")", 1)[1].split()
            start = int(fields[19])
            if parent is not None and int(fields[1]) != parent:
                issues.append("A descendant changed parent during inspection");continue
            if current == pid:
                root_start = start
                if expected_start is not None and start != expected_start:
                    raise ValueError("Worker PID was reused; refuse another process's measurements")
            seconds = (int(fields[11])+int(fields[12]))/ticks
            counters[(current, start)] = seconds
            cpu += seconds;rss += max(0, int(fields[21]))*pages
            for index, entry in enumerate((base/"task").iterdir()):
                if index >= 4096 or time.monotonic() >= deadline:
                    issues.append("Thread inspection budget reached");break
                try:
                    with (entry/"children").open("rb") as stream:children = stream.read(32769)
                    if len(children) > 32768:
                        issues.append("Child list inspection budget reached");continue
                    for child in children.split():
                        if len(pending) >= 4096:
                            issues.append("Process inspection budget reached");break
                        pending.append((int(child), current))
                except FileNotFoundError:pass
        except FileNotFoundError:
            if current == pid:raise ProcessLookupError("Worker process ended before observation")
        except PermissionError:
            if current == pid:raise
            issues.append("A descendant was inaccessible")
    if pending:issues.append("Descendant snapshot is partial")
    return dict(cpu_seconds=cpu, rss_bytes=rss, process_count=len(counters),
                status="partial" if issues else "recorded", reason="; ".join(dict.fromkeys(issues))), root_start, counters


def assigned_gpu_ids(environ, declared=None):
    if not environ.get("SLURM_JOB_ID"):
        raise RuntimeError("No allocated native GPU identity; CPU smoke does not sample node-visible GPUs")
    text = declared or environ.get("SLURM_STEP_GPUS") or environ.get("SLURM_JOB_GPUS")
    ids = list(text) if isinstance(text, (list, tuple)) else str(text or "").split(",")
    if (not ids or len(ids) > 16 or any(not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", x) for x in ids)
            or len(ids) != len(set(ids))):
        raise RuntimeError("Assigned GPU identifiers are absent or unsupported; scope must not widen to the whole node")
    return ids


def gpu_observation(environ, ids, *, timeout=2., runner=subprocess.run):
    command = ["nvidia-smi", "--id="+",".join(ids),
        "--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"]
    result = runner(command, capture_output=True, text=True, timeout=timeout, check=True, env=dict(environ))
    if len(result.stdout.encode()) > 65536:raise ValueError("GPU observation exceeds its bounded output")
    rows = []
    for fields in csv.reader(io.StringIO(result.stdout)):
        if len(fields) != 8:raise ValueError("GPU provider did not supply the declared eight counters")
        fields = [x.strip() for x in fields]
        if not fields[0].isdecimal() or not fields[1].startswith("GPU-"):
            raise ValueError("GPU provider did not supply a stable device identity")
        if not any(x == fields[0] or x == fields[1] for x in ids):
            raise ValueError("GPU observation is outside the assigned device set")
        row = dict(provider="nvidia", device_index=int(fields[0]), device_uuid=fields[1], device_bdf=fields[2], partition="")
        missing = []
        for name, text in zip(("utilization_percent", "memory_used_mib", "memory_total_mib", "power_watts", "temperature_celsius"), fields[3:]):
            try:value = float(text)
            except ValueError:value = None
            if value is not None and (not math.isfinite(value) or value < 0):value = None
            if name == "utilization_percent" and value is not None and value > 100:value = None
            row[name] = value
            if value is None:missing.append(name)
        row.update(status="partial" if missing else "recorded", reason="Unavailable counters: "+",".join(missing) if missing else "")
        rows.append(row)
    if not rows:raise RuntimeError("No assigned GPU counters were observed")
    if len({r["device_uuid"] for r in rows}) != len(rows):raise ValueError("GPU provider duplicated device identities")
    return rows


class Recording:
    def __init__(self, attempt, declared=None, *, pid=None, environ=None):
        self.attempt, self.settings = _safe(attempt), options(declared)
        self.pid = os.getpid() if pid is None else pid
        if type(self.pid) is not int or self.pid <= 0:raise ValueError("Actual worker PID is required")
        self.env = dict(os.environ if environ is None else environ)
        self.node = self.env.get("SLURMD_NODENAME") or socket.gethostname()
        self.start = time.time();self.monotonic_start = time.monotonic()
        self.stop = threading.Event();self.sources = [];self.streams = {};self.csv_streams = {}
        self.previous = None;self.expected_start = None;self.error = "";self.worker_state = "RUNNING"
        self.observer_wall = 0.;self.observer_cpu = 0.
        directory = _safe(self.attempt/"telemetry");directory.mkdir(exist_ok=False)
        families = self.settings["families"]
        for family in (["cpu-memory"] if {"cpu", "memory"}&set(families) else [])+(["gpu"] if "gpu" in families else []):
            reason = ""
            if family == "cpu-memory" and sys.platform != "linux":reason = "Linux procfs unavailable"
            if family == "gpu":
                try:self.gpu_ids = assigned_gpu_ids(self.env, self.settings["allocated_gpu_ids"])
                except (ValueError, RuntimeError) as error:reason = _error(error)
            columns = CPU_COLUMNS if family == "cpu-memory" else GPU_COLUMNS
            source = dict(family=family, path="telemetry/"+family+".jsonl", schema=CPU_SCHEMA if family == "cpu-memory" else GPU_SCHEMA,
                status="unavailable" if reason else "recording", reason=reason, devices=[], metrics=list(columns[2:-3]),
                coverage=dict(scope="local-workload-process-tree" if family == "cpu-memory" else "allocated-device",
                    pid=self.pid, node=self.node, started_at=self.start, ended_at=None, samples=0, missed_samples=0,
                    interval_seconds=self.settings["interval_seconds"], bytes=0, errors=[],
                    limitation="CPU percent is per logical core and can exceed 100%; local descendants exclude departed/remote processes; RSS can repeat shared pages" if family == "cpu-memory"
                    else "Allocated-device counters include any display/co-resident activity; not per-process GPU accounting",
                    native_gpu_verification="NOT_INFERRED_FROM_TELEMETRY"))
            self.sources.append(source)
            if not reason:
                self.streams[family] = (directory/(family+".jsonl")).open("x", encoding="utf-8")
                stream = (directory/(family+".csv")).open("x", newline="", encoding="utf-8")
                writer = csv.DictWriter(stream, fieldnames=columns);writer.writeheader();stream.flush()
                self.csv_streams[family] = (stream, writer)
        self._publish()
        self.thread = None
        if self.streams:
            self.thread = threading.Thread(target=self._loop, name="tdn-worker-telemetry", daemon=True)
            self.thread.start()

    def _publish(self):
        _atomic(self.attempt/"telemetry/index.json", dict(schema=INDEX_SCHEMA, run_id=self.attempt.name,
            node=self.node, created_at=self.start, updated_at=time.time(), sources=self.sources,
            requested_families=self.settings["families"], worker_state=self.worker_state,
            observer_wall_seconds=self.observer_wall, observer_cpu_seconds=self.observer_cpu,
            overhead_scope="Periodic observer loop sampling/file publication; final cleanup and subprocess CPU are excluded",
            missing_convention="Null/empty cells are unavailable; zero is an observed zero"))

    def _append(self, source, values, observed_seconds):
        now = time.time();family = source["family"];coverage = source["coverage"]
        row = dict(schema=source["schema"], run_id=self.attempt.name, sequence=coverage["samples"], timestamp=now,
            timestamp_utc=dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-self.monotonic_start,
            interval_seconds=self.settings["interval_seconds"], node=self.node, scope=coverage["scope"],
            source="linux-procfs" if family == "cpu-memory" else "nvidia-smi", quality="sampled",
            observer_wall_seconds=observed_seconds, **values)
        raw = json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n"
        if coverage["bytes"]+len(raw.encode()) > self.settings["max_bytes_per_stream"]:
            source.update(status="partial", reason="Declared stream storage limit reached; preserved prefix")
            return
        self.streams[family].write(raw);self.streams[family].flush()
        stream, writer = self.csv_streams[family]
        writer.writerow({key:row.get(key) for key in writer.fieldnames});stream.flush()
        coverage["samples"]+=1;coverage["bytes"]+=len(raw.encode())
        coverage.setdefault("first_timestamp", now);coverage["last_timestamp"] = now
        if row.get("status") == "partial":coverage["partial_samples"] = coverage.get("partial_samples", 0)+1
        if family == "gpu":
            device = {key:row[key] for key in ("node", "provider", "device_index", "device_uuid", "device_bdf", "partition")}
            if device not in source["devices"]:source["devices"].append(device)

    def _sample(self, source):
        started = time.monotonic()
        if source["family"] == "cpu-memory":
            values, self.expected_start, counters = process_observation(self.pid, expected_start=self.expected_start)
            now = time.monotonic();values["cpu_utilization_percent"] = None
            if self.previous is not None:
                previous_time, previous = self.previous
                if set(previous) == set(counters):
                    values["cpu_utilization_percent"] = 100*sum(counters[k]-previous[k] for k in counters)/(now-previous_time)
                else:values.update(status="partial", reason="Process tree changed; interval CPU utilization unavailable")
            self.previous = (now, counters)
            if "cpu" not in self.settings["families"]:values.update(cpu_seconds=None, cpu_utilization_percent=None)
            if "memory" not in self.settings["families"]:values["rss_bytes"] = None
            self._append(source, values, time.monotonic()-started)
        else:
            for row in gpu_observation(self.env, self.gpu_ids, timeout=self.settings["timeout_seconds"]):
                self._append(source, row, time.monotonic()-started)

    def _loop(self):
        due = time.monotonic()
        try:
            while not self.stop.is_set():
                wall, cpu = time.monotonic(), time.thread_time()
                late = max(0., wall-due)
                for source in self.sources:
                    if source["status"] != "recording":continue
                    if late > self.settings["interval_seconds"]:
                        source["coverage"]["missed_samples"] += int(late/self.settings["interval_seconds"])
                    try:self._sample(source)
                    except Exception as error:
                        coverage = source["coverage"];coverage["missed_samples"]+=1
                        coverage["errors"] = (coverage["errors"]+[_error(error)])[-16:]
                        source["reason"] = _error(error)
                self._publish()
                self.observer_wall += time.monotonic()-wall;self.observer_cpu += time.thread_time()-cpu
                due = wall+self.settings["interval_seconds"]
                self.stop.wait(max(.01, due-time.monotonic()))
        except Exception as error:self.error = _error(error)
        finally:
            for source in self.sources:
                coverage = source["coverage"];coverage["ended_at"] = time.time()
                if source["status"] == "recording":
                    source["status"] = ("partial" if coverage["errors"] or coverage["missed_samples"] or coverage.get("partial_samples") else "complete") if coverage["samples"] else "unavailable"
                if self.error:source.update(status="partial" if coverage["samples"] else "failed", reason=self.error)
            for stream in self.streams.values():stream.close()
            for stream, _ in self.csv_streams.values():stream.close()
            try:self._publish()
            except Exception as error:self.error = _error(error)

    def close(self, state="UNKNOWN"):
        self.worker_state = state;self.stop.set()
        if self.thread:
            self.thread.join(self.settings["timeout_seconds"]+2)
            if self.thread.is_alive():self.error = "Observer cleanup exceeded its bounded wait; partial evidence retained"
        else:
            for source in self.sources:source["coverage"]["ended_at"] = time.time()
            self._publish()
        result = dict(status="DISABLED" if not self.sources else "PARTIAL" if self.error or any(s["status"] != "complete" for s in self.sources) else "COMPLETE",
            index="telemetry/index.json", requested_families=self.settings["families"], error=self.error,
            observer_wall_seconds=self.observer_wall, observer_cpu_seconds=self.observer_cpu,
            sources=self.sources, scope="Worker resource observations; collection outcome is separate from scientific acceptance")
        _atomic(self.attempt/"telemetry/observer.log", dict(timestamp=time.time(), worker_state=state,
            event="Optional worker telemetry finalized", **result))
        return result


class UnavailableRecording:
    def __init__(self, error):self.error = _error(error)
    def close(self, state="UNKNOWN"):
        return dict(status="UNAVAILABLE", error=self.error, scope="Optional observer failure; science state is unchanged")


def start_recording(attempt, options=None, *, pid=None, environ=None):
    try:return Recording(attempt, options, pid=pid, environ=environ)
    except Exception as error:
        print("TDN worker telemetry warning: "+_error(error), file=sys.stderr)
        return UnavailableRecording(error)


def stop_recording(handle, *, state="UNKNOWN"):
    if handle is None:return dict(status="DISABLED", scope="No worker observer requested")
    try:return handle.close(state)
    except Exception as error:
        print("TDN worker telemetry warning: "+_error(error), file=sys.stderr)
        return dict(status="PARTIAL" if getattr(handle, "sources", None) else "UNAVAILABLE", error=_error(error),
            sources=getattr(handle, "sources", []), scope="Optional observer cleanup failure; science state is unchanged")
