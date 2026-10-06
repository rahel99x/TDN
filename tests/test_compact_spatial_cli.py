"""CLI ownership, interruption and sealing; no PDE solves or Slurm calls."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from tdn import cli, reporting
from tdn.analysis.compact_spatial import experiment
from tdn.analysis.compact_spatial.protocol import ARTIFACTS, DEFAULTS, digest, file_digest
from tdn.runtime import metadata, precision, preflight, storage


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("compact_spatial_cli_lifecycle",
        ROOT / "scripts/compact_spatial.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(DEFAULTS))
    state = SimpleNamespace(module=module, config=config, output=tmp_path / "run",
        report=tmp_path / "report", source="source-original", attached=[], finalized=[],
        owned=True, finalize_success=True)
    state.argv = ["--config", str(config), "--run-dir", str(state.output), "--smoke"]
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(storage, "contained_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(preflight, "execution_mode", lambda: "local")
    monkeypatch.setattr(metadata, "software_metadata", lambda: {"source_tree_sha256": state.source})
    monkeypatch.setattr(precision, "reference_precision", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setenv("TDN_TOWER_DIR", str(tmp_path / "caller-report"))
    monkeypatch.setattr(reporting, "emit", lambda *args, **kwargs: None)

    def attach(*args, **kwargs):
        state.attached.append((args, kwargs))
        return state.report, state.owned

    def finalize(report, **kwargs):
        state.finalized.append((report, kwargs))
        return state.finalize_success

    monkeypatch.setattr(reporting, "attach_report", attach)
    monkeypatch.setattr(cli, "finalize_tower_report", finalize)
    return state


def completed_evidence(directory):
    result = {"status": "COMPLETED", "computational_status": "COMPLETED",
        "scientific_outcome": "OBSERVED_MIXED", "benchmark_suite": "compact-spatial",
        "device": "cpu", "training_attempted": False, "training_performed": False}
    repeats = [{"repeat_index": index, "seconds": seconds, "status": "VALID",
        "trajectory_completed": True, "completed_steps": 4, "failure_reason": None,
        "work": {"fft_forward": 4, "fft_inverse": 4}}
        for index, seconds in enumerate((.0000123456789, .0000111, .0000108, .0000112, .0000115))]
    candidate = {"candidate_id": "fixture/strang/n4", "timing_repeats": repeats,
        "error_rms": 1.2e-7, "error_max": 3.4e-7}
    metadata.write_json(directory / "summary.json", result)
    experiment.write_canonical(directory / "compact-spatial.json", {
        **result, "candidate_rows": [candidate], "reference_rows": [],
        "parity_rows": [], "frontier_rows": [], "errors": []})
    (directory / "summary.txt").write_text("Completed lifecycle fixture; no PDE work\n")
    return result, candidate


def test_help_does_not_require_scientific_dependencies():
    child = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/compact_spatial.py"), "--help"],
        cwd=ROOT, text=True, capture_output=True, timeout=10, check=False)
    assert child.returncode == 0, child.stderr
    assert "--smoke" in child.stdout and "--run-dir" in child.stdout


@pytest.mark.parametrize("populated", [False, True])
def test_existing_run_is_preserved_before_claiming_report(launcher, monkeypatch, populated):
    launcher.output.mkdir()
    if populated:
        (launcher.output / "COMPLETED").write_text("existing seal\n")
        (launcher.output / "summary.json").write_text('{"status":"COMPLETED"}\n')
    before = {p.name: p.read_bytes() for p in launcher.output.iterdir()}
    monkeypatch.setattr(experiment, "run", lambda *a, **k: pytest.fail("Existing run was executed"))
    assert launcher.module.main(launcher.argv) == 1
    assert before == {p.name: p.read_bytes() for p in launcher.output.iterdir()}
    assert not launcher.attached and not launcher.finalized
    assert os.environ["TDN_TOWER_DIR"] == str(launcher.output.parent / "caller-report")


def test_actual_engine_stop_retains_unsealed_evidence_and_exit_75(launcher, monkeypatch):
    # Exercise the real StopRequest and engine stop path, before any PDE work.
    original_run = experiment.run
    prior_handler = signal.getsignal(signal.SIGTERM)
    def stopped(protocol, directory, *, stop, **kwargs):
        installed = signal.getsignal(signal.SIGTERM)
        assert installed == stop._handler
        installed(signal.SIGTERM, None)
        assert stop.requested and stop.signal_number == signal.SIGTERM
        return original_run(protocol, directory, stop=stop, **kwargs)
    monkeypatch.setattr(experiment, "run", stopped)
    monkeypatch.setattr(experiment, "_run_case", lambda *a, **k: pytest.fail("Stop ran a PDE case"))
    assert launcher.module.main(launcher.argv) == 75
    assert signal.getsignal(signal.SIGTERM) == prior_handler
    report, terminal = launcher.finalized[0]
    assert report == launcher.report
    assert terminal["state"] == "INTERRUPTED" and terminal["exit_code"] == 75
    assert terminal["metadata"]["scientific_status"] == "INCOMPLETE"
    for name in ("stage.json", "summary.json", "compact-spatial.json"):
        assert json.loads((launcher.output / name).read_text())["status"] == "INCOMPLETE"
    payload = json.loads((launcher.output / "compact-spatial.json").read_text())
    assert payload["candidate_rows"] == [] and payload["reference_rows"] == []
    assert any("Stop requested" in error for error in payload["errors"])
    assert not (launcher.output / "manifest.json").exists()
    assert not (launcher.output / "COMPLETED").exists()
    assert os.environ["TDN_TOWER_DIR"] == str(launcher.output.parent / "caller-report")


@pytest.mark.parametrize("changed", ["source", "config"])
def test_execution_changes_preserve_samples_but_prevent_sealing(launcher, monkeypatch, changed):
    retained = {}
    def changed_run(protocol, directory, **kwargs):
        result, retained["candidate"] = completed_evidence(directory)
        if changed == "source":
            launcher.source = "source-changed"
        else:
            launcher.config.write_text(launcher.config.read_text() + "\n# changed during run\n")
        return result
    monkeypatch.setattr(experiment, "run", changed_run)
    assert launcher.module.main(launcher.argv) == 1
    terminal = launcher.finalized[0][1]
    assert terminal["state"] == "FAILED" and terminal["exit_code"] == 1
    expected = "Execution source changed" if changed == "source" else "Configuration changed"
    stage = json.loads((launcher.output / "stage.json").read_text())
    assert stage["status"] == "FAILED" and expected in stage["error"]
    raw = (launcher.output / "compact-spatial.json").read_text()
    assert raw.endswith("\n") and raw.count("\n") == 1
    canonical = json.loads(raw)
    assert canonical["candidate_rows"] == [retained["candidate"]]
    for payload in (canonical, json.loads((launcher.output / "summary.json").read_text())):
        assert payload["status"] == payload["computational_status"] == "INCOMPLETE"
        assert payload["scientific_outcome"] == "INCONCLUSIVE"
        assert expected in payload["sealing_error"]
    assert "UNSEALED" in (launcher.output / "summary.txt").read_text()
    assert not (launcher.output / "manifest.json").exists()
    assert not (launcher.output / "COMPLETED").exists()


@pytest.mark.parametrize("owned,success,expected_exit", [(True, True, 0), (True, False, 1), (False, False, 0)])
def test_sealed_science_and_owned_reporting_have_separate_outcomes(
        launcher, monkeypatch, owned, success, expected_exit):
    launcher.owned, launcher.finalize_success = owned, success
    monkeypatch.delenv("TDN_TOWER_DIR")
    monkeypatch.setattr(experiment, "run", lambda protocol, directory, **kwargs: completed_evidence(directory)[0])
    assert launcher.module.main(launcher.argv) == expected_exit
    # A reporting failure must be nonzero without destroying valid sealed science.
    manifest = json.loads((launcher.output / "manifest.json").read_text())
    assert (launcher.output / "COMPLETED").read_text().strip() == file_digest(launcher.output / "manifest.json")
    assert manifest["files"] == {name: file_digest(launcher.output / name) for name in ARTIFACTS}
    assert manifest["protocol_sha256"] == digest(json.loads((launcher.output / "protocol.json").read_text()))
    assert manifest["config_file_sha256"] == file_digest(launcher.config)
    assert manifest["source_tree_sha256"] == launcher.source
    assert json.loads((launcher.output / "stage.json").read_text())["status"] == "COMPLETED"
    assert len(launcher.finalized) == int(owned)
    if owned:
        assert launcher.finalized[0][1]["state"] == "COMPLETED"
        assert launcher.finalized[0][1]["exit_code"] == 0
    assert "TDN_TOWER_DIR" not in os.environ
