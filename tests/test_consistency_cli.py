"""New-stage ownership, audit prerequisites and interruption/seal boundaries."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tdn import cli, reporting
from tdn.analysis.consistency import engine
from tdn.runtime import metadata, precision, preflight, storage
from test_consistency_artifacts import audit_evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("consistency_cli_fixture", ROOT / "scripts/consistency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = SimpleNamespace(module=module, path=tmp_path / "audit", source="a" * 64, finalize=[], attached=[])
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(storage, "contained_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(preflight, "execution_mode", lambda: "local-cpu")
    monkeypatch.setattr(metadata, "software_metadata", lambda: {"source_tree_sha256": state.source})
    monkeypatch.setattr(precision, "reference_precision", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(reporting, "emit", lambda *a, **k: None)

    def audit(protocol, directory, **kwargs):
        # Preserve the CLI-owned execution/protocol documents while supplying
        # explicit seal fixtures for the audit engine boundary.
        execution = (directory / "execution.json").read_bytes()
        audit_evidence(directory)
        (directory / "execution.json").write_bytes(execution)
        return {"status": "COMPLETED"}

    def attach(*args, **kwargs):
        state.attached.append(args)
        return tmp_path / "report", True

    def finalize(*args, **kwargs):
        state.finalize.append(kwargs)
        return True

    monkeypatch.setattr(engine, "audit", audit)
    monkeypatch.setattr(reporting, "attach_report", attach)
    monkeypatch.setattr(cli, "finalize_tower_report", finalize)
    state.argv = ["--stage", "audit", "--profile", "smoke", "--run-dir", str(state.path)]
    return state


def test_audit_cli_seals_fresh_run_and_preserves_completed_directory(launcher):
    assert launcher.module.main(launcher.argv) == 0
    assert (launcher.path / "COMPLETED").exists()
    before = {path.name: path.read_bytes() for path in launcher.path.iterdir() if path.is_file()}
    assert launcher.module.main(launcher.argv) == 1
    assert before == {path.name: path.read_bytes() for path in launcher.path.iterdir() if path.is_file()}


@pytest.mark.parametrize("change", ["source", "protocol", "interrupt"])
def test_midrun_change_or_stop_cannot_complete_audit(launcher, monkeypatch, change):
    original = engine.audit

    def fail(protocol, directory, **kwargs):
        result = original(protocol, directory, **kwargs)
        if change == "source":
            launcher.source = "b" * 64
        elif change == "protocol":
            protocol = {**protocol, "updates": 100}
            metadata.write_json(directory / "protocol.json", protocol)
        else:
            raise InterruptedError("Stop requested")
        return result

    monkeypatch.setattr(engine, "audit", fail)
    assert launcher.module.main(launcher.argv) == (75 if change == "interrupt" else 1)
    assert not (launcher.path / "COMPLETED").exists()
    record = json.loads((launcher.path / "stage.json").read_text())
    assert record["status"] == ("INTERRUPTED" if change == "interrupt" else "FAILED")


def test_prepare_requires_matching_audit_before_opening_stage(launcher, monkeypatch):
    monkeypatch.setattr(engine, "prepare", lambda *a, **k: pytest.fail("Preparation started without verified audit"))
    args = ["--stage", "prepare", "--profile", "smoke", "--run-dir", str(launcher.path.parent / "prepare")]
    assert launcher.module.main(args) == 1
    assert not (launcher.path.parent / "prepare").exists()
    assert launcher.module.main(launcher.argv) == 0
    with (launcher.path / "checks.json").open("a") as handle:
        handle.write("changed")
    assert launcher.module.main(args + ["--audit-dir", str(launcher.path)]) == 1
    assert not (launcher.path.parent / "prepare").exists()
