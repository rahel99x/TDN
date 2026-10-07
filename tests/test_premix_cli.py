"""Fresh-run ownership and immutable predecessor tests for the premix CLI."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.premix import artifacts
from tdn.runtime import metadata, precision, preflight, storage
from tdn import cli, reporting

ROOT = Path(__file__).resolve().parents[1]


def evidence(path, stage="accuracy"):
    path.mkdir(parents=True, exist_ok=True)
    protocol = {"fixture": True}
    execution = {"stage": stage, "profile": "smoke", "device": "cpu", "execution_mode": "local-cpu",
                 "protocol_sha256": artifacts.digest(protocol), "software": {"source_tree_sha256": "a" * 64}}
    for name, value in {"protocol.json": protocol, "execution.json": execution,
                        "config.json": {}, "summary.json": {"status": "COMPLETED", "profile": "smoke", "panel": stage},
                        "metrics.json": {"status": "COMPLETED", "profile": "smoke", "panel": stage,
                                         "candidate_rows": [], "frontier_rows": [], "reference_rows": [], "parity_rows": []}}.items():
        metadata.write_json(path / name, value)
    (path / "summary.txt").write_text("Fixture without scientific measurements\n")
    if stage == "prepare":
        (path / "dataset.pt").write_bytes(b"fixture, not a torch dataset")
    return protocol


def test_seal_rejects_changed_missing_and_escaping_artifacts(tmp_path):
    root = tmp_path / "stage"
    evidence(root)
    artifacts.seal_stage(root, stage="accuracy", profile="smoke", source_tree_sha256="a" * 64)
    artifacts.verify_stage(root, stage="accuracy", profile="smoke", source_tree_sha256="a" * 64)
    with pytest.raises(ValueError, match="profile"):
        artifacts.verify_stage(root, profile="full")
    with pytest.raises(ValueError, match="source_tree"):
        artifacts.verify_stage(root, source_tree_sha256="b" * 64)
    (root / "metrics.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        artifacts.verify_stage(root)
    (root / "metrics.json").unlink()
    with pytest.raises(ValueError, match="Missing"):
        artifacts.verify_stage(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"] = {"../outside": "a" * 64, **manifest["files"]}
    metadata.write_json(root / "manifest.json", manifest)
    (root / "COMPLETED").write_text(artifacts.file_digest(root / "manifest.json"))
    with pytest.raises(ValueError, match="Unsafe"):
        artifacts.verify_stage(root)


def test_seal_preserves_results_and_excludes_mutable_reporting(tmp_path):
    evidence(tmp_path)
    metadata.write_json(tmp_path / "stage.json", {"status": "RUNNING"})
    metadata.write_json(tmp_path / "tower" / "report.json", {"rows": []})
    manifest = artifacts.seal_stage(tmp_path, stage="accuracy", profile="smoke", source_tree_sha256="a" * 64)
    assert "stage.json" not in manifest["files"]
    assert "tower/report.json" not in manifest["files"]
    with pytest.raises(ValueError, match="Preserve"):
        artifacts.seal_stage(tmp_path, stage="accuracy", profile="smoke", source_tree_sha256="a" * 64)


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("premix_cli_fixture", ROOT / "scripts/premix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tdn.analysis.premix import mechanisms
    state = SimpleNamespace(module=module, root=tmp_path / "run", source="a" * 64, finalized=[], attached=[],
                            engine=mechanisms, report=tmp_path / "report", owned=True, finalize_success=True)
    state.argv = ["--stage", "accuracy", "--profile", "smoke", "--run-dir", str(state.root)]
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(storage, "contained_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(preflight, "execution_mode", lambda: "local-cpu")
    monkeypatch.setattr(metadata, "software_metadata", lambda: {"source_tree_sha256": state.source})
    monkeypatch.setattr(precision, "reference_precision", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(mechanisms, "build_protocol", lambda **kwargs: {"fixture": True})
    monkeypatch.setattr(reporting, "emit", lambda *a, **k: None)

    def run(protocol, directory, **kwargs):
        evidence(directory)
        return {"status": "COMPLETED"}

    def attach(*args, **kwargs):
        state.attached.append(args)
        return state.report, state.owned

    def finalize(*args, **kwargs):
        state.finalized.append(kwargs)
        return state.finalize_success

    monkeypatch.setattr(mechanisms, "run", run)
    monkeypatch.setattr(reporting, "attach_report", attach)
    monkeypatch.setattr(cli, "finalize_tower_report", finalize)
    return state


def test_existing_run_is_not_modified(launcher):
    launcher.root.mkdir()
    (launcher.root / "keep").write_text("original")
    assert launcher.module.main(launcher.argv) == 1
    assert list(p.name for p in launcher.root.iterdir()) == ["keep"]
    assert not launcher.attached


@pytest.mark.parametrize("changed", ["source", "protocol"])
def test_midrun_changes_prevent_completion(launcher, monkeypatch, changed):
    def run(protocol, directory, **kwargs):
        evidence(directory)
        if changed == "source":
            launcher.source = "modified"
        else:
            metadata.write_json(directory / "protocol.json", {"changed": True})
        return {"status": "COMPLETED"}
    monkeypatch.setattr(launcher.engine, "run", run)
    assert launcher.module.main(launcher.argv) == 1
    assert not (launcher.root / "COMPLETED").exists()
    assert json.loads((launcher.root / "summary.json").read_text())["status"] == "FAILED"
    assert launcher.finalized[0]["state"] == "FAILED"


def test_stop_before_seal_keeps_partial_evidence(launcher, monkeypatch):
    def run(protocol, directory, *, stop, **kwargs):
        evidence(directory)
        stop.requested = True
        return {"status": "COMPLETED"}
    monkeypatch.setattr(launcher.engine, "run", run)
    assert launcher.module.main(launcher.argv) == 75
    assert not (launcher.root / "COMPLETED").exists()
    assert launcher.finalized[0]["state"] == "INTERRUPTED"
    assert (launcher.root / "metrics.json").is_file()


def test_success_reporting_failure_does_not_destroy_seal(launcher):
    launcher.finalize_success = False
    assert launcher.module.main(launcher.argv) == 1
    assert artifacts.verify_stage(launcher.root)["stage"] == "accuracy"
    assert launcher.finalized[0]["state"] == "COMPLETED"


@pytest.mark.parametrize("change_profile", [False, True])
def test_desktop_slurm_cli_freezes_profile_in_execution(launcher, monkeypatch, change_profile):
    from tdn.runtime import desktop_slurm
    profile = {"root": str(ROOT), "user": "fixture"}
    monkeypatch.setattr(preflight, "execution_mode", lambda: "desktop-slurm")
    monkeypatch.setattr(desktop_slurm, "load_profile", lambda: dict(profile))

    def run(protocol, directory, **kwargs):
        execution = json.loads((directory / "execution.json").read_text())
        assert execution["execution_mode"] == "desktop-slurm"
        assert execution["slurm_profile_sha256"] == artifacts.digest(profile)
        evidence(directory)
        metadata.write_json(directory / "execution.json", execution)
        if change_profile:
            profile["user"] = "changed"
        return {"status": "COMPLETED"}

    monkeypatch.setattr(launcher.engine, "run", run)
    assert launcher.module.main(launcher.argv) == int(change_profile)
    assert (launcher.root / "COMPLETED").exists() is not change_profile
    if change_profile:
        assert "profile changed" in json.loads((launcher.root / "stage.json").read_text())["error"]
    else:
        assert artifacts.verify_stage(launcher.root)["stage"] == "accuracy"
