"""Frozen work–precision declarations and honest canonical row validation."""
from copy import deepcopy

import pytest
import torch

from tdn.analysis.work_precision.protocol import (
    DEFAULTS, METHODS, NORMS, NSTEPS, TABLE_IDS, TARGETS,
    digest, make_protocol, validate_config, validate_rows,
)


@pytest.mark.parametrize("key,value", [
    ("protocol_version", True), ("suite", "interaction-screen"),
    ("max_seconds", 1201), ("reference_tolerance", 1e-7),
    ("reference_attempts", 4), ("intraop_threads", 2),
    ("interop_threads", 2), ("warmup_rollouts", 0), ("timing_repeats", 4),
    ("nsteps", [1, 2, 4, 8, 16]),
    ("nsteps", [True, 2, 4, 8, 16, 32]),
    ("nsteps", [1.0, 2, 4, 8, 16, 32]),
    ("nsteps", (1, 2, 4, 8, 16, 32)),
    ("error_targets", [True, 2e-4, 2e-5, 2e-6]),
    ("error_targets", [2e-3, 2e-4, 2e-5]),
    ("methods", list(reversed(METHODS))),
    ("methods", list(METHODS[:-1])),
])
def test_configuration_is_frozen_including_nested_json_types(key, value):
    raw = deepcopy(DEFAULTS)
    raw[key] = value
    with pytest.raises(ValueError):
        validate_config(raw)


@pytest.mark.parametrize("damage", ["extra", "missing", "not_mapping"])
def test_configuration_requires_exact_protocol_fields(damage):
    raw = deepcopy(DEFAULTS)
    if damage == "extra":
        raw["adaptive"] = True
    elif damage == "missing":
        del raw["nsteps"]
    else:
        raw = list(raw)
    with pytest.raises(ValueError):
        validate_config(raw)


def test_smoke_only_changes_the_explicit_budget_and_preserves_the_input():
    raw = deepcopy(DEFAULTS)
    full = validate_config(raw)
    smoke = validate_config(raw, smoke=True)
    assert raw == DEFAULTS and "smoke" not in raw
    assert full["max_seconds"] == 1200 and not full["smoke"]
    assert smoke["max_seconds"] == 180 and smoke["smoke"]
    for key in DEFAULTS.keys() - {"max_seconds"}:
        assert full[key] == smoke[key] == DEFAULTS[key]
    full["nsteps"].append(64)
    assert raw["nsteps"] == smoke["nsteps"] == list(NSTEPS)


def toy_plan():
    return {
        "cases": [{"case_id": "fixed"}],
        "expected_ids": {name: [name + "/one"] for name in TABLE_IDS},
        "timing_orders": {},
        "parity": {},
    }


def toy_rows():
    result = {name: [{key: name + "/one"}] for name, key in TABLE_IDS.items()}
    result["candidate_rows"][0].update(
        case_id="fixed", variant="strang", nsteps=1, reference_accepted=True,
        status="VALID", trajectory_completed=True,
        error_rms=1e-4, error_max=1e-4, error_mean=0.,
        error_upper_rms=1e-4, error_upper_max_estimate=1e-4,
        preparation_seconds=2., prepared_median_seconds=3.,
        setup_inclusive_median_seconds=5.,
        warmup={"status": "VALID", "trajectory_completed": True,
                "completed_steps": 1, "seconds": 7., "repeat_index": -1},
        timing_repeats=[{"status": "VALID", "trajectory_completed": True,
                         "completed_steps": 1, "seconds": float(index + 1),
                         "repeat_index": index} for index in range(5)],
    )
    result["reference_rows"][0].update(case_id="fixed", reference_accepted=True,
                                      uncertainty_rms=0., uncertainty_max_estimate=0.)
    result["frontier_rows"][0].update(
        case_id="fixed", variant="strang", norm="rms", selected_nsteps=1,
        status="FEASIBLE", selected_candidate_id="candidate_rows/one",
        selected_error=1e-4, adjusted_error=1e-4, tolerance=2e-4, prepared_seconds=3.,
        setup_selected_candidate_id="candidate_rows/one", setup_selected_nsteps=1,
        setup_selected_error=1e-4, setup_selected_adjusted_error=1e-4,
        setup_selected_seconds=5.,
    )
    return result


def test_protocol_records_cpu_training_free_scope_and_hashes_declared_cases():
    plans = toy_plan()
    protocol = make_protocol(validate_config(deepcopy(DEFAULTS)), plans, {"torch": "test"}, ["test"])
    assert protocol["case_plan_sha256"] == digest(plans["cases"])
    assert protocol["device"] == "cpu" and protocol["dtype"] == "float64"
    assert not protocol["training_attempted"] and not protocol["training_performed"]
    assert protocol["software"] == {"torch": "test"}
    assert protocol["command"] == ["test"]
    assert any("post hoc" in item for item in protocol["interpretation"])
    assert any("not a rigorous certificate" in item for item in protocol["interpretation"])


@pytest.mark.parametrize("damage", ["missing_timing", "duplicate_case", "empty_cases",
                                    "missing_table", "duplicate_id", "empty_id", "nonstring_id"])
def test_missing_or_ambiguous_predeclarations_are_rejected(damage):
    plans = toy_plan()
    if damage == "missing_timing":
        del plans["timing_orders"]
    elif damage == "duplicate_case":
        plans["cases"] *= 2
    elif damage == "empty_cases":
        plans["cases"] = []
    elif damage == "missing_table":
        del plans["expected_ids"]["reference_rows"]
    elif damage == "duplicate_id":
        plans["expected_ids"]["candidate_rows"] *= 2
    else:
        plans["expected_ids"]["candidate_rows"] = ["" if damage == "empty_id" else 1]
    with pytest.raises(ValueError):
        make_protocol(validate_config(deepcopy(DEFAULTS)), plans, {}, [])


def test_partial_row_coverage_is_explicit_and_complete_coverage_is_enforced():
    plans, tables = toy_plan(), toy_rows()
    assert validate_rows(tables, plans, complete=True) is tables
    tables["parity_rows"].clear()
    validate_rows(tables, plans, complete=False)
    with pytest.raises(ValueError, match="Missing declared parity_rows"):
        validate_rows(tables, plans, complete=True)


@pytest.mark.parametrize("field", ["error_rms", "error_max", "error_mean",
                                   "error_upper_rms", "error_upper_max_estimate"])
def test_invalid_partial_trajectory_cannot_publish_final_horizon_errors(field):
    tables = toy_rows()
    tables["candidate_rows"][0].update(status="INVALID", trajectory_completed=False,
        error_rms=None, error_max=None, error_mean=None,
        error_upper_rms=None, error_upper_max_estimate=None)
    tables["candidate_rows"][0][field] = 0.0
    with pytest.raises(ValueError, match="Partial trajectories"):
        validate_rows(tables, toy_plan(), complete=False)


@pytest.mark.parametrize("damage", ["duplicate", "undeclared", "nonfinite", "unknown_status",
                                    "missing_table", "unselected_frontier", "over_tolerance"])
def test_malformed_canonical_rows_are_rejected(damage):
    tables = toy_rows()
    if damage == "duplicate":
        tables["candidate_rows"] *= 2
    elif damage == "undeclared":
        tables["candidate_rows"][0]["candidate_id"] = "unplanned"
    elif damage == "nonfinite":
        tables["reference_rows"][0]["uncertainty_rms"] = float("nan")
    elif damage == "unknown_status":
        tables["candidate_rows"][0]["status"] = "PASS"
    elif damage == "missing_table":
        del tables["parity_rows"]
    elif damage == "unselected_frontier":
        tables["frontier_rows"][0]["selected_candidate_id"] = None
    else:
        tables["frontier_rows"][0]["adjusted_error"] = 1e-2
    with pytest.raises(ValueError):
        validate_rows(tables, toy_plan(), complete=False)


@pytest.mark.parametrize("damage", ["missing_warmup", "missing_repeat", "repeated_index",
                                    "partial_sample", "nonpositive_time", "incorrect_median",
                                    "incorrect_setup", "rejected_reference", "invalid_selected",
                                    "unaccepted_selected", "wrong_prepared_cost", "wrong_setup_cost",
                                    "wrong_setup_error", "missing_setup_candidate"])
def test_incomplete_or_tampered_measurement_cannot_validate_a_frontier(damage):
    tables = toy_rows()
    candidate = tables["candidate_rows"][0]
    frontier = tables["frontier_rows"][0]
    if damage == "missing_warmup":
        candidate["warmup"] = None
    elif damage == "missing_repeat":
        candidate["timing_repeats"].pop()
    elif damage == "repeated_index":
        candidate["timing_repeats"][-1]["repeat_index"] = 0
    elif damage == "partial_sample":
        candidate["timing_repeats"][0]["completed_steps"] = 0
    elif damage == "nonpositive_time":
        candidate["timing_repeats"][0]["seconds"] = 0.
    elif damage == "incorrect_median":
        candidate["prepared_median_seconds"] = .001
    elif damage == "incorrect_setup":
        candidate["setup_inclusive_median_seconds"] = .001
    elif damage == "rejected_reference":
        tables["reference_rows"][0]["reference_accepted"] = False
    elif damage == "invalid_selected":
        candidate["status"] = "INVALID"
    elif damage == "unaccepted_selected":
        candidate["reference_accepted"] = False
    elif damage == "wrong_prepared_cost":
        frontier["prepared_seconds"] = .001
    elif damage == "wrong_setup_cost":
        frontier["setup_selected_seconds"] = .001
    elif damage == "wrong_setup_error":
        frontier["setup_selected_adjusted_error"] = 0.
    else:
        frontier["setup_selected_candidate_id"] = "unplanned"
    with pytest.raises(ValueError):
        validate_rows(tables, toy_plan(), complete=True)


def test_step_methods_and_error_targets_are_separate_fixed_tracks():
    assert NSTEPS == (1, 2, 4, 8, 16, 32)
    assert METHODS == ("strang", "etdrk2", "etdrk4", "gl3", "gl5", "gl3_mean_full", "gl3_mean_spectral")
    assert NORMS == ("rms", "max")
    assert TARGETS == (2e-3, 2e-4, 2e-5, 2e-6)


def test_full_bank_has_twenty_fresh_and_four_exact_historical_controls():
    from tdn.analysis.interactions import experiment as historical
    from tdn.analysis.work_precision import experiment

    bank = experiment.cases()
    assert len(bank) == len({case["case_id"] for case in bank}) == 24
    fresh = [case for case in bank if case["case_role"] == "fresh"]
    controls = [case for case in bank if case["case_role"] == "historical_control"]
    assert len(fresh) == 20 and len(controls) == 4
    assert sum(len(case["grid"]) == 1 for case in fresh) == 12
    assert sum(len(case["grid"]) == 2 for case in fresh) == 8
    old_steps, old_rollouts = historical.cases()
    old_bank = {case["case_id"]: case for case in old_steps + old_rollouts}
    assert {case["historical_case_id"] for case in controls} == {
        "two_d/0", "two_d/1", "rollout/two_d", "historical/phi_counterexample"}
    for case in controls:
        old = old_bank[case["historical_case_id"]]
        assert all(case[key] == value for key, value in old.items() if key != "case_id")
        assert torch.equal(experiment.state(case), historical.state(old))
    for case in bank:
        field = experiment.state(case)
        assert field.shape == (1, 1, *case["grid"])
        assert field.device.type == "cpu" and field.dtype == torch.float64
        assert bool(torch.isfinite(field).all()) and bool(((field >= 0) & (field <= 1)).all())


def test_smoke_is_four_predeclared_cases_with_the_full_scientific_tracks():
    from tdn.analysis.work_precision import experiment

    full = experiment.plan(validate_config(deepcopy(DEFAULTS)))
    smoke = experiment.plan(validate_config(deepcopy(DEFAULTS), smoke=True))
    assert len(full["cases"]) == 24
    assert {case["case_id"] for case in smoke["cases"]} == {
        "fresh/1d/00", "fresh/2d/07", "historical/two_d_rollout", "historical/phi_counterexample"}
    for name in TABLE_IDS:
        assert set(smoke["expected_ids"][name]) <= set(full["expected_ids"][name])
    for plans, count in ((full, 24), (smoke, 4)):
        assert len(plans["expected_ids"]["candidate_rows"]) == count * 6 * 7
        assert len(plans["expected_ids"]["frontier_rows"]) == count * 7 * 2 * 4
        assert len(plans["expected_ids"]["reference_rows"]) == count
        assert len(plans["expected_ids"]["parity_rows"]) == count * (4 + 6)


def test_every_step_count_predeclares_all_method_warmup_then_five_rotations():
    from tdn.analysis.work_precision import experiment

    config = validate_config(deepcopy(DEFAULTS))
    plans = experiment.plan(config)
    assert plans == experiment.plan(config)
    expected = {(case["case_id"], n) for case in plans["cases"] for n in NSTEPS}
    orders = plans["timing_orders"]
    assert len(orders) == len(expected) == 24 * 6
    assert {(item["case_id"], item["nsteps"]) for item in orders} == expected
    for item in orders:
        assert item["preparation_order"] == item["warmup_order"]
        assert set(item["warmup_order"]) == set(METHODS)
        assert len(item["warmup_order"]) == len(METHODS)
        measured = item["measured_orders"]
        assert len(measured) == 5 and len({tuple(order) for order in measured}) == 5
        previous = item["warmup_order"]
        for order in measured:
            assert order == previous[1:] + previous[:1]
            previous = order
