"""Integration boundaries for immutable adjacent inputs and interrupted work.

Tiny real CPU fits test file lifecycle. Mock scheduler assertions test dependency
contracts only; they are never CUDA readiness or scientific result claims.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.adjacent import engine, pilots
from tdn.analysis.adjacent.protocol import build_protocol, digest
from tdn.runtime.metadata import write_json

ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow_tests = _module("adjacent_recovery_workflow_fixtures", ROOT / "tests/test_adjacent_workflow.py")
controller = workflow_tests.controller
cli = _module("adjacent_recovery_cli", ROOT / "scripts/adjacent.py")


@pytest.mark.parametrize("profile", ["smoke", "development", "full"])
def test_dispatch_fitting_and_evaluation_bind_identical_scientific_settings(tmp_path, monkeypatch, profile):
    """Different allocation deadlines must not alter the frozen model identity."""
    protocol = build_protocol(profile)
    captured = {}

    def fit(p, path, **kwargs):
        captured[str(path)] = copy.deepcopy(p)
        write_json(path / "pilot-summary.json", {"status": "UNIT_TEST_ONLY"})
        return {"status": "UNIT_TEST_ONLY"}

    def evaluate(p, path, prior, **kwargs):
        assert digest(p) == digest(captured[str(prior)])
        assert pilots._identity(p) == pilots._identity(captured[str(prior)])
        write_json(path / "evaluation-summary.json", {"status": "UNIT_TEST_ONLY"})
        return {"status": "UNIT_TEST_ONLY"}

    monkeypatch.setattr(pilots, "run_pilot", fit)
    monkeypatch.setattr(pilots, "evaluate_pilot", evaluate)
    recorded = []
    for stage, unit in protocol["units"].items():
        if unit["kind"] not in ("train", "confirm"):
            continue
        path = tmp_path / stage
        path.mkdir()
        ctx = SimpleNamespace(stage=stage, unit=unit, protocol=protocol, path=path, device="cpu",
            budget=SimpleNamespace(check=lambda: None),
            prerequisites={unit["pilot_unit"]: tmp_path / unit["pilot_unit"]} if unit["kind"] == "confirm" else {},
            record=lambda *args, **kwargs: recorded.append(kwargs))
        engine._dispatch(ctx)
    assert recorded
    assert all(row["evidence"] and row["evidence"][0]["sha256"] for row in recorded)


def test_pilot_freeze_survives_outer_sealing_and_recovery_copy_but_not_checkpoint_mutation(tmp_path):
    """A real tiny fit owns scientific inputs, not its enclosing worker state."""
    torch.set_num_threads(1)
    protocol = build_protocol("smoke")
    # Explicitly a unit-test-sized direct pilot, not the immutable native plan.
    protocol["pilot"].update(models=["quad2_fixed", "channel_neural"], tracks=["discrete"],
        train_fields=1, val_fields=1, evaluation_fields=1, updates=1,
        endpoint_steps=[1], max_seconds=90, evaluation_seconds=90)
    origin = tmp_path / "origin"
    origin.mkdir()
    outer = {"execution.json": {"status": "RUNNING"}, "stage.json": {"status": "RUNNING"},
             "protocol.json": {"outer_controller_scope": "unit_test"}, "rows.jsonl": {"initial": True}}
    for name, payload in outer.items():
        write_json(origin / name, payload)
    pilots.run_pilot(protocol, origin)
    freeze = pilots.verify_freeze(protocol, origin)
    assert not set(outer) & set(freeze["artifacts"])
    checkpoint_bytes = {name: (origin / name).read_bytes() for name in freeze["artifacts"]
                        if name.startswith("checkpoints/")}
    assert checkpoint_bytes
    # This is the lifecycle performed by engine/CLI *after* fitting freezes.
    for name in outer:
        write_json(origin / name, {"status": "COMPLETED", "outer_controller_scope": "unit_test"})
    write_json(origin / "summary.json", {"outer_engine_finished": True})
    write_json(origin / "prior_attempts/attempt-0/summary.json", {"status": "INTERRUPTED"})
    assert pilots.verify_freeze(protocol, origin) == freeze
    copied = tmp_path / "fresh-recovery-attempt"
    shutil.copytree(origin, copied)
    assert pilots.verify_freeze(protocol, copied) == freeze
    assert all((copied / name).read_bytes() == payload for name, payload in checkpoint_bytes.items())
    first = next(iter(checkpoint_bytes))
    (copied / first).write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        pilots.verify_freeze(protocol, copied)
    assert (origin / first).read_bytes() == checkpoint_bytes[first]
    assert pilots.verify_freeze(protocol, origin) == freeze


def test_recovery_with_inherited_audit_still_defers_to_live_main_campaign(controller, monkeypatch):
    origin = workflow_tests.plan(controller, "origin")
    base = workflow_tests.freeze(controller, origin)
    (base / "audit").mkdir()
    (base / "audit/workflow-seal.json").write_text("synthetic sealed audit")
    controller.cw.atomic_json(base / "jobs.json", [{"stage": "audit", "job_id": "50"}])
    before = {str(p.relative_to(base)): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    monkeypatch.setattr(controller.cw, "scheduler_state", lambda _: "COMPLETED")
    monkeypatch.setattr(controller, "verify_stage", lambda *a, **k: {"execution": {"stage": "audit"}})
    monkeypatch.setattr(controller, "stage_cli", lambda: SimpleNamespace(
        verify_execution=lambda *a, **k: {"execution": {"stage": "audit"}}))
    calls = workflow_tests.runner(controller, monkeypatch)
    original_command = controller.cw.command

    def command(args, **kwargs):
        if args[0] == "squeue":
            return SimpleNamespace(stdout="301|tdn-portfolio-confirm-000\n302|tdn-portfolio-report\n400|unrelated-project\n")
        return original_command(args, **kwargs)

    monkeypatch.setattr(controller.cw, "command", command)
    assert controller.main(["recover", "origin", "--run-id", "recovered-after-main"]) == 0
    recovered = controller.load(controller.workflow_path("recovered-after-main"))
    jobs = controller.cw.read_json(Path(recovered["run_dir"]) / "jobs.json")
    assert "audit" not in {row["stage"] for row in jobs}
    assert jobs[0]["stage"] == "D01"
    assert jobs[0]["dependency"] == "afterany:301:302"
    assert jobs[-1]["stage"] == "report"
    assert jobs[-1]["dependency"] == "afterany:" + ":".join(row["job_id"] for row in jobs[:-1])
    assert not any("400" in str(row["dependency"]) for row in jobs)
    assert not any(args[0] == "scancel" for args, _ in calls)
    after = {str(p.relative_to(base)): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("changed", ["executable", "source_tree_sha256", "git_commit", "torch_cuda_runtime"])
def test_resume_does_not_relax_source_or_venv_identity_to_import_another_checkout(tmp_path, changed):
    protocol = build_protocol("smoke")
    software = dict(executable="/isolated/.venv/bin/python", source_tree_sha256="a" * 64,
                    git_commit="b" * 40, torch_cuda_runtime=None, python="3.13.13")
    lineage = {"audit": {"run_dir": "retained/audit", "workflow_seal_sha256": "c" * 64}}
    execution = dict(stage="pilot", profile="smoke", protocol_sha256=cli.digest(protocol),
        execution_mode="local-cpu", device="cpu", prerequisites=lineage, software=software)
    write_json(tmp_path / "execution.json", execution)
    write_json(tmp_path / "stage.json", {"status": "INTERRUPTED"})
    (tmp_path / "retained.pt").write_bytes(b"unit-test checkpoint byte identity")
    original = cli.resume_fingerprint(tmp_path)
    proposed = dict(software)
    proposed[changed] = "different"
    with pytest.raises(ValueError, match="scope, software, or prerequisite lineage"):
        cli.check_resume(tmp_path, stage="pilot", protocol=protocol, software=proposed,
                         mode="local-cpu", device="cpu", lineage=lineage)
    assert cli.resume_fingerprint(tmp_path) == original


def test_changed_recovery_origin_is_rejected_before_new_stage_runs(controller, monkeypatch):
    workflow = workflow_tests.plan(controller, "recovery")
    base = workflow_tests.freeze(controller, workflow)
    interrupted = controller.ROOT / "runs/interrupted/pilot"
    interrupted.mkdir(parents=True)
    (interrupted / "checkpoint.pt").write_bytes(b"original")
    original_sha = cli.resume_fingerprint(interrupted)
    bridge = dict(source_sha256=workflow["source_sha256"], protocol_sha256=workflow["protocol_sha256"],
                  stage_paths={}, resume_paths={"pilot": str(interrupted)},
                  resume_fingerprints={"pilot": original_sha})
    monkeypatch.setattr(controller, "recovery_bridge", lambda _: bridge)
    monkeypatch.setattr(controller, "stage_cli", lambda: cli)
    assert controller.verify_recovery_bridge(workflow) == {}
    (interrupted / "checkpoint.pt").write_bytes(b"modified after recovery was planned")
    with pytest.raises(ValueError, match="Interrupted origin artifacts changed"):
        controller.verify_recovery_bridge(workflow)
    assert not (base / "pilot").exists()
