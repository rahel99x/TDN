"""Real CPU observations and controlled GPU reader contracts, not CUDA proof."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from tdn.analysis.tangent_response import telemetry


def native(monkeypatch):
    root = Path(os.environ.get("TDN_RAVEN_TEST_ROOT", "/workspace/Raven"))
    if not (root/"raven/retained_telemetry_io.py").is_file():pytest.skip("Trusted optional Raven checkout unavailable")
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("raven.retained_telemetry_io")


def test_actual_cpu_memory_stream_is_readable_after_relocation(tmp_path, monkeypatch):
    reader = native(monkeypatch)
    attempt = tmp_path/"project"/"runs"/"actual-cpu";attempt.mkdir(parents=True)
    recorder = telemetry.start_recording(attempt, {"enabled":["cpu", "memory"], "sampling_seconds":.05},
        pid=os.getpid(), environ={})
    deadline = time.monotonic()+.25
    allocation = bytearray(2 << 20)
    while time.monotonic() < deadline:sum(range(20000))
    coverage = telemetry.stop_recording(recorder, state="FAILED")
    assert coverage["status"] in ("COMPLETE", "PARTIAL")
    moved = tmp_path/"other-machine"/"actual-cpu";shutil.copytree(attempt, moved)
    index = reader.validate_index(json.loads((moved/"telemetry/index.json").read_text()), moved.name)
    assert index["worker_state"] == "FAILED"
    assert index["requested_families"] == ["cpu", "memory"]
    source = index["sources"][0]
    rows = reader.parse_rows((moved/source["path"]).read_bytes(), source, moved.name)
    assert rows and all(r["rss_bytes"] > 0 and r["cpu_seconds"] >= 0 for r in rows)
    assert any(r.get("cpu_utilization_percent") is not None for r in rows)
    assert index["observer_wall_seconds"] > 0 and index["observer_cpu_seconds"] >= 0
    assert (moved/"telemetry/cpu-memory.csv").is_file()
    assert allocation[0] == 0


def test_disabled_observer_has_no_background_thread_or_fake_counters(tmp_path):
    attempt = tmp_path/"disabled";attempt.mkdir()
    recorder = telemetry.start_recording(attempt, {"enabled":[], "sampling_seconds":5}, environ={})
    result = telemetry.stop_recording(recorder, state="COMPLETED")
    assert result["status"] == "DISABLED" and recorder.thread is None
    index = json.loads((attempt/"telemetry/index.json").read_text())
    assert not index["sources"] and not list((attempt/"telemetry").glob("*.jsonl"))


def test_gpu_requires_assignment_even_if_node_visible(monkeypatch):
    with pytest.raises(RuntimeError, match="Assigned GPU"):
        telemetry.assigned_gpu_ids({"SLURM_JOB_ID":"controlled", "CUDA_VISIBLE_DEVICES":"0"})
    with pytest.raises(RuntimeError, match="No allocated"):
        telemetry.assigned_gpu_ids({"CUDA_VISIBLE_DEVICES":"0"})


def test_controlled_gpu_rows_keep_missing_power_and_actual_reader_contract(tmp_path, monkeypatch):
    reader = native(monkeypatch)
    calls = []
    def controlled(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0,
            stdout="0, GPU-controlled-fixture, 00000000:01:00.0, 42, 1024, 24564, N/A, 49\n")
    observed = telemetry.gpu_observation({"SLURM_JOB_ID":"controlled"}, ["0"], runner=controlled)
    assert calls[0][1] == "--id=0"
    assert observed[0]["power_watts"] is None and observed[0]["temperature_celsius"] == 49
    assert observed[0]["status"] == "partial"
    source = dict(family="gpu", path="telemetry/gpu.jsonl", schema=telemetry.GPU_SCHEMA,
        status="partial", reason="Controlled reader fixture; no native GPU readiness", coverage={}, metrics=[], devices=[])
    row = dict(observed[0], schema=telemetry.GPU_SCHEMA, timestamp=time.time(), run_id="controlled-fixture")
    index = reader.validate_index(dict(schema=telemetry.INDEX_SCHEMA, run_id="controlled-fixture", sources=[source]), "controlled-fixture")
    original=tmp_path/"fixture-project/runs/controlled-fixture/telemetry";original.mkdir(parents=True)
    (original/"index.json").write_text(json.dumps(index))
    (original/"gpu.jsonl").write_text(json.dumps(row)+"\n")
    moved=tmp_path/"relocated-host/runs/controlled-fixture";shutil.copytree(original.parent,moved)
    relocated=reader.validate_index(json.loads((moved/"telemetry/index.json").read_text()),moved.name)
    assert reader.parse_rows((moved/relocated["sources"][0]["path"]).read_bytes(), relocated["sources"][0], moved.name)[0]["power_watts"] is None
    def foreign(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="1, GPU-foreign, 0000:02:00.0, 1, 1, 2, 1, 1\n")
    with pytest.raises(ValueError, match="outside the assigned"):
        telemetry.gpu_observation({}, ["0"], runner=foreign)


def test_observer_failure_retains_partial_records_without_science_state_change(tmp_path, monkeypatch):
    attempt = tmp_path/"partial";attempt.mkdir()
    real = telemetry.process_observation;calls = []
    def fails_later(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:raise PermissionError("Controlled observer loss")
        return real(*args, **kwargs)
    monkeypatch.setattr(telemetry, "process_observation", fails_later)
    recorder = telemetry.start_recording(attempt, {"families":["cpu", "memory"], "interval_seconds":.05}, environ={})
    time.sleep(.16)
    result = telemetry.stop_recording(recorder, state="COMPLETED")
    assert result["status"] == "PARTIAL"
    assert result["sources"][0]["coverage"]["samples"] >= 1
    assert result["sources"][0]["coverage"]["errors"]
    assert json.loads((attempt/"telemetry/index.json").read_text())["worker_state"] == "COMPLETED"


def test_symlink_telemetry_directory_is_a_separate_unavailable_warning(tmp_path):
    attempt = tmp_path/"safe";attempt.mkdir();outside = tmp_path/"outside";outside.mkdir()
    (attempt/"telemetry").symlink_to(outside, target_is_directory=True)
    result = telemetry.stop_recording(telemetry.start_recording(attempt, "cpu", environ={}), state="COMPLETED")
    assert result["status"] == "UNAVAILABLE" and not list(outside.iterdir())


def test_optional_cleanup_publication_failure_cannot_raise_into_science():
    class BrokenObserver:
        sources = [{"status":"partial"}]
        def close(self, state):raise PermissionError("Controlled optional cleanup failure")
    result = telemetry.stop_recording(BrokenObserver(), state="COMPLETED")
    assert result["status"] == "PARTIAL" and "Controlled" in result["error"]
