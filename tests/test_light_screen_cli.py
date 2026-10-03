"""CLI allocation, preservation and completion-proof checks without numerical work."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import signal

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("light_screen_cli", ROOT / "scripts/light_screen.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def isolated_cli(tmp_path, monkeypatch):
    from tdn.analysis import light_screen
    from tdn.runtime import metadata, precision, preflight, storage
    import torch

    monkeypatch.setattr(storage, "configure_storage", lambda: ROOT)
    monkeypatch.setattr(preflight, "verify_runtime", lambda device, stage: None)
    monkeypatch.setattr(preflight, "execution_mode", lambda: "unit-test-substitute")
    monkeypatch.setattr(metadata, "software_metadata", lambda: {"source_tree_sha256": "same-source"})
    monkeypatch.setattr(precision, "reference_precision", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    return tmp_path / "screen", light_screen


def invoke(run_dir):
    return cli.main(["--config", str(ROOT / "configs/carc-light.yaml"), "--run-dir", str(run_dir)])


def test_runtime_rejection_precedes_config_reads_and_run_outputs(tmp_path, monkeypatch):
    from tdn import config
    from tdn.runtime import preflight, storage

    monkeypatch.setattr(storage, "configure_storage", lambda: ROOT)
    def reject(device, stage):
        raise ValueError("CARC compute requires a real allocated task")
    monkeypatch.setattr(preflight, "verify_runtime", reject)
    monkeypatch.setattr(config, "load_config", lambda path: pytest.fail("Read config before runtime policy"))
    destination = tmp_path / "screen"
    assert invoke(destination) == 1
    assert not destination.exists()


def test_existing_outputs_are_preserved(isolated_cli, monkeypatch):
    destination, module = isolated_cli
    destination.mkdir()
    evidence = destination / "keep.txt"
    evidence.write_text("preserved previous evidence")
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: pytest.fail("Existing evidence was reused"))
    assert invoke(destination) == 1
    assert evidence.read_text() == "preserved previous evidence"
    assert not (destination / "stage.json").exists()


def test_completed_screen_can_retain_scientific_failures(isolated_cli, monkeypatch):
    destination, module = isolated_cli
    def collect(config, run_dir, device):
        assert device == "cpu" and config["validation"]["require_headroom"] is True
        assert json.loads((run_dir / "stage.json").read_text())["status"] == "RUNNING"
        result = {"status": "COMPLETED", "pilot_authorized": False, "headroom_passed": False}
        (run_dir / "summary.json").write_text(json.dumps(result))
        (run_dir / "summary.txt").write_text("No case cleared scientific screening.\n")
        return result
    monkeypatch.setattr(module, "run", collect)
    assert invoke(destination) == 0
    status = json.loads((destination / "stage.json").read_text())
    assert status["status"] == "COMPLETED" and status["device"] == "cpu"
    assert (destination / "COMPLETED").read_text().strip() == status["config_hash"]
    assert json.loads((destination / "summary.json").read_text())["pilot_authorized"] is False


@pytest.mark.parametrize("result", [{"status": "INCOMPLETE"}, {"status": "FAILED"}])
def test_incomplete_scientific_run_has_no_completion_marker(isolated_cli, monkeypatch, result):
    destination, module = isolated_cli
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: result)
    assert invoke(destination) == 1
    assert json.loads((destination / "stage.json").read_text())["status"] == "FAILED"
    assert not (destination / "COMPLETED").exists()


def test_changed_execution_source_invalidates_completed_measurement(isolated_cli, monkeypatch):
    destination, module = isolated_cli
    from tdn.runtime import metadata
    reports = iter([{"source_tree_sha256": "before"}, {"source_tree_sha256": "after"}])
    monkeypatch.setattr(metadata, "software_metadata", lambda: next(reports))
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: {"status": "COMPLETED"})
    assert invoke(destination) == 1
    assert "source changed" in json.loads((destination / "stage.json").read_text())["error"]
    assert not (destination / "COMPLETED").exists()


def test_interrupt_retains_failure_and_does_not_certify_completion(isolated_cli, monkeypatch):
    destination, module = isolated_cli
    previous_handlers = {getattr(signal, name): signal.getsignal(getattr(signal, name))
                         for name in ("SIGUSR1", "SIGTERM") if hasattr(signal, name)}
    def stop(*args, **kwargs):
        raise InterruptedError("Screen interrupted; start a fresh run")
    monkeypatch.setattr(module, "run", stop)
    assert invoke(destination) == 1
    status = json.loads((destination / "stage.json").read_text())
    assert status["status"] == "FAILED" and "InterruptedError" in status["error"]
    assert not (destination / "COMPLETED").exists()
    assert all(signal.getsignal(signum) == handler for signum, handler in previous_handlers.items())
