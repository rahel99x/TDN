import json
from pathlib import Path

import pytest

from tdn.analysis.agenda import engine
from tdn.analysis.agenda.protocol import build_protocol


def _catalog(protocol):
    return [{**trial, "source_stage": stage, "status": "COMPLETED", "best_validation_loss": trial["seed"] / 1e8,
             "selection": "TRAINED_CHECKPOINT"} for stage in ("controls", "compression", "kernel") for trial in protocol["stage_trials"][stage]]


def test_confirmation_preserves_all_three_paired_seeds_and_a_kernel_slot():
    p = build_protocol("full")
    selected = engine.select_confirmation(p, _catalog(p))
    assert len(selected) == 12
    for family in ("source", "precompress_source", "fno_source_matched"):
        assert {r["seed"] for r in selected if r["family"] == family and r["source_stage"] == "compression"} == set(p["seeds"])
    assert sum(r["source_stage"] == "kernel" for r in selected) == 1


def test_missing_confirmation_slot_remains_visible():
    p = build_protocol("smoke")
    catalog = [r for r in _catalog(p) if r["family"] != "precompress_source"]
    selected = engine.select_confirmation(p, catalog)
    ledger = engine.confirmation_ledger(p, catalog, selected)
    missing = [r for r in ledger if r["status"] == "NO_ELIGIBLE_CHECKPOINT"]
    assert len(missing) == 1 and missing[0]["slot"]["family"] == "precompress_source"


def test_single_norm_pass_cannot_become_joint_success():
    p = build_protocol("smoke")
    record = {"trial_id": "a", "family": "source", "seed": 1, "selection": "TRAINED_CHECKPOINT"}
    rows = [{"trial_id": "a", "track": "discrete", "norm": norm, "classical": classical,
             "status": "PASS" if norm == "rms" else "FAIL"} for norm in p["norms"] for classical in ("etdrk4", "gl3_fused")]
    assert engine.joint_criteria(p, rows, [record])[0]["status"] == "FAIL"
    for row in rows:
        row["status"] = "PASS"
    record["selection"] = "SELECTED_INITIALIZATION"
    assert engine.joint_criteria(p, rows, [record])[0]["status"] == "INCONCLUSIVE"


def test_resolved_hyperparameters_and_base_orientation_are_identical_for_matched_arms(tmp_path):
    p = build_protocol("smoke")
    optimize, controls = tmp_path / "optimize", tmp_path / "controls"
    optimize.mkdir(); controls.mkdir()
    (optimize / "summary.json").write_text(json.dumps({"selected_hyperparameters": {"source": {"learning_rate": .0003, "batch_size": 2}}}))
    (controls / "catalog.json").write_text(json.dumps({"records": [{"family": "source", "status": "COMPLETED", "best_validation_loss": 1.,
        "trial_id": "diffusion-control", "base_orientation": "diffusion-first"}]}))
    trials = engine._resolved_trials(p, "compression", {"optimize": optimize, "controls": controls})
    assert all(t["learning_rate"] == .0003 and t["batch_size"] == 2 and t["base_orientation"] == t["orientation"] == "diffusion-first" for t in trials)


def test_raw_float_schedule_sums_share_one_frontier_group():
    p = build_protocol("smoke")
    rows = [{"parent_id": "p", "continuous_field_id": "f", "grid_size": 8, "track": "discrete", "horizon": h,
             "trial_id": "a", "family": "source", "seed": 1, "status": "COMPLETED", "upper_rms": .00001,
             "upper_max": .00001, "timing_seconds": seconds, "schedule": [h], "uncertainty_rms": 0., "uncertainty_max": 0.}
            for h, seconds in ((.27, 1.), (sum([.045] * 6), .5))]
    frontiers = engine._frontiers(p, rows)
    assert len(frontiers) == 8
    assert all(r["timing_seconds"] == .5 for r in frontiers if r["status"] == "FEASIBLE")


def test_inner_seal_survives_only_mutable_outer_journal_updates(tmp_path):
    p = build_protocol("smoke")
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    (tmp_path / "summary.json").write_text(json.dumps({"status": "COMPLETED"}))
    (tmp_path / "stage.json").write_text('{"status":"RUNNING"}')
    engine._seal(p, tmp_path)
    (tmp_path / "stage.json").write_text('{"status":"COMPLETED"}')
    (tmp_path / "manifest.json").write_text('{}')
    engine.verify_science(p, tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"status": "FAILED"}))
    with pytest.raises(ValueError, match="changed"):
        engine.verify_science(p, tmp_path)
