"""R7 access gates and portable evidence on a reduced implementation fixture.

The only numeric workflow here has 12 source queries on three existing parents
and one interval. It is not the frozen 192-query development study.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdn.analysis.response_preconditioner import r7_references as refs
from tdn.analysis.response_preconditioner import r7_actions as actions
from tdn.analysis.response_preconditioner import r7_workflow as w
from tdn.analysis.response_preconditioner import workflow as legacy


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_config():
    config = w.protocol("study")
    config.update(parent_seeds={key: values[:1] for key, values in config["parent_seeds"].items()},
                  intervals=[[.02, .04]], calibration_sources=2, withheld_sources=2,
                  calibration_families=["dense_random", "dense_random"],
                  withheld_families=["dense_random", "constant"],
                  reference_count=12, max_reference_queries=12, local_context_weight=1.)
    config["budget"]["study_seconds"] = 45.
    config["timing"].update(rounds=1, warmup=0)
    config["cohort"] = "reduced unit fixture only, not the frozen development study"
    return config


def read(path):
    return json.loads(path.read_text())


def events(output):
    return [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]


def test_frozen_sources_normalized_reproducible_role_separated_and_family_declared():
    config = w.protocol("study")
    for role in ("calibration", "withheld"):
        first = w.source_bank(2026101501, 3, role, config)
        repeated = w.source_bank(2026101501, 3, role, config)
        assert [row["family"] for row in first] == config[role+"_families"]
        for left, right in zip(first, repeated):
            assert left["z"].shape == (1, 1, 5, 5)
            assert left["z"].dtype == torch.float64
            torch.testing.assert_close(left["z"], right["z"], rtol=0, atol=0)
            assert float(left["z"].square().mean()) == pytest.approx(1., abs=5e-16)
    train = w.source_bank(2026101501, 3, "calibration", config)
    held = w.source_bank(2026101501, 3, "withheld", config)
    other_context = w.source_bank(2026101501, 2, "calibration", config)
    assert not torch.equal(train[0]["z"], held[0]["z"])
    assert not torch.equal(train[0]["z"], other_context[0]["z"])
    assert torch.equal(held[-1]["z"], torch.ones_like(held[-1]["z"]))
    for row in held[1:3]:
        assert abs(float(row["z"].mean())) < 5e-16
    with pytest.raises(ValueError, match="role"):
        w.source_bank(2026101501, 0, "confirmation", config)


def test_r7_reuses_exact_old_parent_fields_and_physics_rng_draws():
    old, new = legacy.protocol("smoke"), w.protocol("study")
    assert new["source_directions_per_parent"] == old["source_directions_per_parent"] == 2
    for split in ("train", "validation", "diagnostic"):
        historical = list(legacy.parents(old, split))
        current = list(legacy.parents(new, split))
        assert len(historical) == len(current)
        for a, b in zip(historical, current):
            assert a["parent_id"] == b["parent_id"]
            assert a["eq"] == b["eq"]
            assert a["geom"] == b["geom"]
            torch.testing.assert_close(a["u"], b["u"], rtol=0, atol=0)
    assert new["reference_count"] == 4*6*(4+4) == 192
    assert new["max_reference_queries"] == 192
    assert new["budget"]["native_only"]
    assert not new["scientific_promotion"]
    assert new["historical_gate"] == "B STOP unchanged"


def test_planned_reference_cap_blocks_before_any_acquisition(tmp_path):
    config = fixture_config()
    config["max_reference_queries"] = 11
    output = tmp_path / "cap"
    output.mkdir()
    with pytest.raises(ValueError, match="Planned queries exceed"):
        w.Study(output, config)
    assert not list((output / "queries").iterdir())


def test_initialization_is_an_eligible_candidate_and_wins_exact_selection_ties(tmp_path):
    output = tmp_path / "selection"
    output.mkdir()
    study = w.Study(output, fixture_config())
    for indexed in study.index:
        if (indexed["split"], indexed["role"]) not in (("train", "calibration"), ("validation", "withheld")):
            continue
        row = dict(indexed, status="ACCEPTED", teacher=np.zeros((1, 1, 5, 5)),
                   base=np.zeros((1, 1, 5, 5)), design=np.ones((40, 1, 1, 5, 5)))
        study.records[row["query_id"]] = row
    selected = w._global_fit(study)
    assert selected["candidates"][0]["name"] == "zero_readout"
    assert selected["selected"]["name"] == "zero_readout"
    assert selected["selected"]["eligible"]
    assert all(row["validation_loss"] == 0. for row in selected["candidates"])
    assert not selected["diagnostic_truth_accessed"]
    assert not selected["training_withheld_truth_accessed"]
    assert all("train-" in q and "calibration" in q for q in selected["train_query_ids"])
    assert all("validation-" in q and "withheld" in q for q in selected["validation_query_ids"])


def test_unresolved_training_blocks_fit_and_preserves_all_planned_denominators(tmp_path, monkeypatch):
    def unresolved(u, z, *args, **kwargs):
        return dict(teacher=None, refined=z.clone(), matched_mean=None, matched_mean_refined=None,
                    reference=dict(accepted=False, status="UNRESOLVED", uncertainty_rms=None,
                                   uncertainty_maximum=None, matched_mean=dict(accepted=False)))
    monkeypatch.setattr(refs, "acquire_reference", unresolved)
    output = tmp_path / "unresolved"
    output.mkdir()
    with pytest.raises(RuntimeError, match="Unresolved training references"):
        w._study(output, fixture_config())
    counts = read(output / "denominators.json")
    assert counts["planned"] == 12
    assert counts["attempted"] == counts["unresolved"] == 6
    assert counts["missing"] == 6
    assert counts["accepted"] == counts["failed"] == 0
    assert not (output / "selection.json").exists()
    assert not any(row["event"] == "selection_frozen" for row in events(output))
    index = read(output / "query_index.json")
    assert len(index) == 12
    assert all(row["status"] == "NOT_ACQUIRED" for row in index if row["split"] == "diagnostic")
    assert not list((output / "fits").iterdir())
    for row in index:
        path = output / row["array_path"]
        if row["status"] != "NOT_ACQUIRED":
            with np.load(path, allow_pickle=False) as arrays:
                assert "teacher" not in arrays
                assert "refined" in arrays


def test_failed_query_retains_initial_source_error_and_remaining_denominators(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate acquisition failure")
    monkeypatch.setattr(refs, "acquire_reference", fail)
    output = tmp_path / "failed"
    output.mkdir()
    with pytest.raises(FloatingPointError, match="deliberate acquisition"):
        w._study(output, fixture_config())
    counts = read(output / "denominators.json")
    assert counts["planned"] == 12 and counts["attempted"] == 1
    assert counts["failed"] == 1 and counts["missing"] == 11
    index = read(output / "query_index.json")
    first = index[0]
    assert "deliberate acquisition failure" in first["error"]
    assert first["array_sha256"]
    with np.load(output / first["array_path"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"initial", "source"}
    assert (output / "metrics.csv").is_file()
    assert read(output / "metrics.json") == []


def test_source_change_refuses_verification_seal_and_preserves_failure(tmp_path, monkeypatch):
    count = iter(("before", "after"))
    monkeypatch.setattr(legacy, "source_snapshot", lambda: dict(fingerprint=next(count), files=[]))
    monkeypatch.setattr(legacy, "_software", lambda: {"test_fixture": True})
    monkeypatch.setattr(w, "_verify", lambda *_: dict(checks={"total": 1}))
    result = w.execute("verify", tmp_path / "changed")
    assert result["status"] == "FAILED"
    assert "Source changed" in result["error"]
    output = Path(result["output"])
    assert read(output / "source_after.json")["fingerprint"] == "after"
    assert "Source changed" in (output / "failure.txt").read_text()
    assert not result["scientific_promotion"]
    assert not result["native_gpu_readiness"]
    assert read(output / "artifacts.json")["status"] == "FAILED"
    with pytest.raises(ValueError, match="explicit CPU"):
        w.execute("verify", tmp_path / "gpu", device="cuda")


def prepared_score_fixture(output):
    """Two real calibration references, then score one source in failure tests."""
    output.mkdir()
    study = w.Study(output, fixture_config())
    cid = next(iter(study.contexts))
    calibration = study.acquire(cid, "calibration")
    assert all(row["status"] == "ACCEPTED" for row in calibration)
    local = w._local_fit(study, cid, calibration)
    assert all(fit["status"] == "SOLVED" for fit in local["fits"])
    return study, cid, calibration[0], local


def test_partial_method_deadline_preserves_completed_prediction_and_missing_counts(tmp_path, monkeypatch):
    output = tmp_path / "partial-score"
    study, cid, row, local = prepared_score_fixture(output)
    checked = []
    def stop_after_one():
        checked.append(True)
        if len(checked) > 1:
            raise TimeoutError("test deadline after completed method")
    monkeypatch.setattr(study, "check", stop_after_one)
    with pytest.raises(TimeoutError, match="after completed method"):
        w._score(study, cid, [row], local)
    metrics = read(output / "metrics.json")
    assert len(metrics) == 1
    assert metrics[0]["method"] == "ordered_1" and metrics[0]["status"] == "MEASURED"
    with np.load(output / "queries" / (row["query_id"]+"-predictions.npz"), allow_pickle=False) as archive:
        assert archive.files == ["ordered_1"]
        assert np.isfinite(archive["ordered_1"]).all()
    counts = read(output / "evaluation_denominators.json")
    assert counts["expected_metrics"] == 132
    assert counts["attempted_metrics"] == counts["measured_metrics"] == 1
    assert counts["missing_metrics"] == 131
    assert counts["failed_metrics"] == counts["unresolved_metrics"] == 0
    assert counts["expected_timing_bundles"] == counts["missing_timing_bundles"] == 16
    assert counts["attempted_timing_bundles"] == 0
    work = read(output / "work.json")
    assert work[-1]["method"] == "ordered_1" and work[-1]["error"] is None


def test_failed_fresh_method_is_retained_while_other_methods_continue(tmp_path, monkeypatch):
    output = tmp_path / "failed-method"
    study, cid, row, local = prepared_score_fixture(output)
    original = actions.fresh_action
    def fail_one(method, *args, **kwargs):
        if method == "ordered_2":
            raise ArithmeticError("deliberate ordered control failure")
        return original(method, *args, **kwargs)
    monkeypatch.setattr(actions, "fresh_action", fail_one)
    w._score(study, cid, [row], local)
    metrics = read(output / "metrics.json")
    assert len(metrics) == 11
    failed = [metric for metric in metrics if metric["status"] == "FAILED"]
    assert len(failed) == 1 and failed[0]["method"] == "ordered_2"
    assert "deliberate ordered control failure" in failed[0]["error"]
    assert failed[0]["rms"] is None
    assert sum(metric["status"] == "MEASURED" for metric in metrics) == 10
    assert metrics[-1]["method"] == "local_ridge_1e-06"
    assert metrics[-1]["status"] == "MEASURED"
    with np.load(output / "queries" / (row["query_id"]+"-predictions.npz"), allow_pickle=False) as archive:
        assert len(archive.files) == 10 and "ordered_2" not in archive
        assert "lawson_8" in archive and "fixed_feature_fitted" in archive
    counts = read(output / "evaluation_denominators.json")
    assert counts["expected_metrics"] == 132 and counts["attempted_metrics"] == 11
    assert counts["failed_metrics"] == 1 and counts["measured_metrics"] == 10
    assert counts["missing_metrics"] == 121
    work = read(output / "work.json")
    assert any(item["method"] == "ordered_2" and item["error"] for item in work)


def test_partial_timing_round_retains_observation_before_deadline(tmp_path):
    output = tmp_path / "partial-timing"
    output.mkdir()
    study = w.Study(output, fixture_config())
    cid = next(cid for cid, ctx in study.contexts.items() if ctx["parent"]["split"] == "diagnostic")
    checked = []
    def stop_after_one():
        checked.append(True)
        if len(checked) > 1:
            raise TimeoutError("test timing deadline")
    with pytest.raises(TimeoutError, match="timing deadline"):
        actions.measure_paired_actions(
            {"first": lambda: torch.ones(2), "second": lambda: torch.zeros(2)},
            rounds=1, warmup=0, seed=7, budget=stop_after_one,
            on_record=lambda row: study.timing_record(cid, row))
    records = read(output / "timing_records.json")
    assert len(records) == 1 and records[0]["status"] == "PASS"
    assert records[0]["context_id"] == cid
    counts = read(output / "evaluation_denominators.json")
    assert counts["expected_timing_bundles"] == 16
    assert counts["attempted_timing_bundles"] == 1
    assert counts["missing_timing_bundles"] == 15
    assert counts["failed_timing_bundles"] == 0
    assert events(output)[-1]["event"] == "timing_observation"


def test_reduced_end_to_end_fixture_seals_choices_before_withheld_truth(tmp_path):
    output = tmp_path / "tiny-r7"
    output.mkdir()
    result = w._study(output, fixture_config())
    assert result["reference_count"] == 12
    assert result["unresolved_references"] == 0
    assert result["unresolved_matched_mean_references"] == 0
    assert not result["native_gpu_readiness"]
    assert not result["matched_accuracy_speedup_established"]
    counts = read(output / "denominators.json")
    assert counts["planned"] == counts["attempted"] == counts["accepted"] == 12
    assert counts["missing"] == counts["unresolved"] == counts["failed"] == 0
    assert counts["independent_parent_count"] == 3
    assert not counts["descendants_are_independent"]
    evaluations = read(output / "evaluation_denominators.json")
    assert evaluations["expected_metrics"] == evaluations["attempted_metrics"] == evaluations["measured_metrics"] == 132
    assert evaluations["missing_metrics"] == evaluations["failed_metrics"] == evaluations["unresolved_metrics"] == 0
    assert evaluations["expected_timing_bundles"] == evaluations["attempted_timing_bundles"] == 16
    assert evaluations["missing_timing_bundles"] == evaluations["failed_timing_bundles"] == 0
    journal = events(output)
    assert journal[0]["event"] == "protocol_frozen"
    selected_at = next(i for i, row in enumerate(journal) if row["event"] == "selection_frozen")
    local_seals = {row["context_id"]: i for i, row in enumerate(journal)
                   if row["event"] == "local_coefficients_sealed"}
    for i, row in enumerate(journal):
        if row["event"] != "query_started":
            continue
        if row["split"] == "diagnostic" or (row["split"] == "train" and row["role"] == "withheld"):
            assert i > selected_at
        if row["split"] == "diagnostic" and row["role"] == "withheld":
            cid = row["query_id"].rsplit("-withheld-", 1)[0]
            assert i > local_seals[cid] > selected_at
    selection = read(output / "selection.json")
    assert selection["candidates"][0]["name"] == "zero_readout"
    assert selection["selected"]["validation_loss"] <= selection["candidates"][0]["validation_loss"]
    assert all("diagnostic" not in q for q in selection["train_query_ids"]+selection["validation_query_ids"])
    metrics = read(output / "metrics.json")
    assert len(metrics) == result["metrics_count"] == 132
    assert all(row["status"] == "MEASURED" and not row["speedup_claim"] for row in metrics)
    assert {row["method"] for row in metrics} >= {"projection", "local_ridge_1e-10", "local_ridge_1e-06",
                                                    "fixed_feature_fitted", "ordered_8", "lawson_8"}
    assert all(not row["deployable"] for row in metrics if row["method"].startswith(("projection", "local_ridge")))
    timing = read(output / "timing.json")
    assert len(timing) == 1 and timing[0]["status"] == "PASS"
    assert timing[0]["expected_calls"] == 16
    assert timing[0]["failed_calls"] == 0
    assert len(read(output / "timing_records.json")) == 16
    frontier = read(output / "work_precision.json")
    assert len(frontier) == 8
    for row in frontier:
        assert row["all_references_and_predictions_resolved"] and not row["speedup_claim"]
        paired = [r for r in metrics if r["split"] == "diagnostic" and r["role"] == "withheld"
                  and r["method"] == row["method"]]
        assert row["pooled_source_rms"] == pytest.approx(np.sqrt(np.mean([r["rms"]**2 for r in paired])))
        assert row["maximum"] == max(r["maximum"] for r in paired)
        assert row["median_seconds"] == timing[0]["summary"][row["method"]]["median_seconds"]
    for name in timing[0]["methods"]:
        assert not name.startswith(("projection", "local_ridge"))
    for filename in (output / "queries").glob("*.npz"):
        with np.load(filename, allow_pickle=False) as archive:
            assert all(archive[name].dtype != object for name in archive.files)
    decisions = read(output / "decisions.json")
    assert not decisions["encoder_promotion"]
    assert not decisions["superiority"]
    assert not decisions["matched_tolerance_speedup"]
