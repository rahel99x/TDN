"""Native-batch evidence, independently accepted teachers and safe partials."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.compact_spatial import experiment
from tdn.analysis.compact_spatial.protocol import (
    DEFAULTS, METHODS, TABLE_IDS, make_protocol, validate_config, validate_rows,
)
from tdn.numerics.reference import reference_step, refined_reference
from tdn.numerics.types import Equation, Geometry


def declaration():
    config = validate_config(deepcopy(DEFAULTS), smoke=True)
    plans = experiment.plan(config)
    spec = next(item for item in plans["cases"] if item["panel"] == "scaling")
    plans["cases"] = [spec]
    plans["timing_orders"] = [item for item in plans["timing_orders"] if item["case_id"] == spec["case_id"]]
    plans["expected_ids"] = {name: [key for key in ids if key.startswith(spec["case_id"] + "/")]
                             for name, ids in plans["expected_ids"].items()}
    return make_protocol(config, plans, {}, [])


@pytest.fixture(scope="module")
def measured_batch():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        protocol = declaration()
        tables = {name: [] for name in TABLE_IDS}
        with torch.no_grad():
            experiment._run_case(protocol, protocol["plans"]["cases"][0], tables, lambda: None)
        tables["frontier_rows"] = experiment.select_frontiers(protocol, tables["candidate_rows"], tables["reference_rows"])
        yield protocol, tables
    finally:
        torch.set_num_threads(original_threads)


def test_measured_batch_has_independent_references_and_complete_raw_repeats(measured_batch):
    protocol, tables = measured_batch
    validate_rows(tables, protocol["plans"], complete=True)
    spec = protocol["plans"]["cases"][0]
    assert spec["batch_size"] == 4
    assert len(tables["candidate_rows"]) == len(METHODS) * len(spec["nsteps"])
    reference = tables["reference_rows"][0]
    assert reference["reference_accepted"] and reference["completed_members"] == 4
    assert [r["member_index"] for r in reference["member_references"]] == list(range(4))
    assert all(r["status"] == "ACCEPTED" and r["rhs_evaluations_complete"] for r in reference["member_references"])
    for row in tables["candidate_rows"]:
        assert row["status"] == "VALID" and row["trajectory_completed"]
        assert len(row["member_errors"]) == 4
        assert row["error_rms"] == max(member["error_rms"] for member in row["member_errors"])
        assert row["error_upper_rms"] == max(member["error_upper_rms"] for member in row["member_errors"])
        assert row["throughput_members_per_second"] == 4 / row["prepared_median_seconds"]
        assert len(row["timing_repeats"]) == 5
        assert row["warmup"]["repeat_index"] == -1
        order = next(item for item in protocol["plans"]["timing_orders"] if item["nsteps"] == row["nsteps"])
        for index, sample in enumerate(row["timing_repeats"]):
            assert sample["repeat_index"] == index
            assert sample["method_order_index"] == order["measured_orders"][index].index(row["variant"])
            assert sample["completed_steps"] == row["nsteps"] and sample["seconds"] > 0
            assert sample["work"] == row["work_per_rollout"]
        assert row["setup_inclusive_median_seconds"] == row["preparation_seconds"] + row["prepared_median_seconds"]


def test_raw_defect_approximations_are_not_identity_failures(measured_batch):
    _, tables = measured_batch
    approximate = [row for row in tables["parity_rows"] if row["check"] == "approximation"]
    identities = [row for row in tables["parity_rows"] if row["check"] != "approximation"]
    assert len(approximate) == 4 and len(identities) == 4
    assert all(row["status"] == "OBSERVED" and row["passed"] is None for row in approximate)
    assert all(row["reference_defect_rms"] > 0 and row["relative_rms_when_resolved"] is not None for row in approximate)
    assert all(row["status"] == "PASS" and row["passed"] for row in identities)


def test_work_counts_include_all_batch_members_and_actual_grid_work(measured_batch):
    _, tables = measured_batch
    strang = next(row for row in tables["candidate_rows"] if row["variant"] == "strang" and row["nsteps"] == 2)
    work = strang["work_per_rollout"]
    assert work["fft_forward"] == work["fft_inverse"] == 2
    assert work["fft_total_fields"] == 4 * 4
    assert work["fft_transformed_cells"] == 4 * 4 * 256
    fused = next(row for row in tables["candidate_rows"] if row["variant"] == "gl3_fused" and row["nsteps"] == 2)
    assert fused["work_per_rollout"]["fft_total_fields"] > fused["work_per_rollout"]["fft_forward"] + fused["work_per_rollout"]["fft_inverse"]


def _synthetic_endpoint_row():
    spec = deepcopy(experiment.cases()[0])
    spec.update(grid=[4], cells=4, batch_size=2)
    reference_row = experiment._new_reference(spec)
    for index, member in enumerate(reference_row["member_references"]):
        member.update(reference_accepted=True, uncertainty_rms=.001 if index == 0 else .00001,
                      uncertainty_max_estimate=.002 if index == 0 else .00002)
    reference_row.update(reference_accepted=True, uncertainty_rms=.001, uncertainty_max_estimate=.002)
    row = experiment._new_candidate(spec, "strang", 1, reference_row)
    row["preparation_seconds"] = .01
    sample = dict(status="VALID", completed_steps=1, trajectory_completed=True,
                  finite=True, in_bounds=True, seconds=.1, failure_reason=None, work={})
    row["warmup"] = {**sample, "repeat_index": -1}
    row["timing_repeats"] = [{**sample, "repeat_index": index} for index in range(5)]
    return row, reference_row


def test_worst_member_error_cannot_be_diluted_and_uncertainty_is_paired():
    row, reference_row = _synthetic_endpoint_row()
    # Member zero is exact but has the larger uncertainty. Member one has
    # nonuniform error and the smaller uncertainty. Independent maxima summed
    # together would be unnecessarily pessimistic, batch RMS would be lenient.
    value = torch.tensor([[[0., 0., 0., 0.]], [[.001, .002, .003, .004]]], dtype=torch.float64)
    reference = SimpleNamespace(state=torch.zeros_like(value))
    experiment._refresh_candidate(row, value, reference, reference_row, 5)
    worst_rms = float(value[1].square().mean().sqrt())
    assert row["error_rms"] == worst_rms > float(value.square().mean().sqrt())
    assert row["error_upper_rms"] == worst_rms + .00001
    assert row["error_upper_rms"] < row["error_rms"] + reference_row["uncertainty_rms"]
    assert row["error_rms_worst_member_index"] == 1
    assert row["error_mean_abs"] == pytest.approx(.0025)
    for member in row["member_errors"]:
        assert member["error_rms"] ** 2 == pytest.approx(member["error_mean"] ** 2 + member["error_spatial_rms"] ** 2)


def test_partial_invalid_trajectory_has_no_final_horizon_errors():
    row, reference_row = _synthetic_endpoint_row()
    row["warmup"].update(status="INVALID", completed_steps=0, trajectory_completed=False)
    value = torch.zeros((2, 1, 4), dtype=torch.float64)
    experiment._refresh_candidate(row, value, SimpleNamespace(state=value), reference_row, 5)
    assert row["status"] == "INVALID" and not row["trajectory_completed"]
    assert row["member_errors"] == []
    assert all(row[key] is None for key in ("error_rms", "error_max", "error_mean_abs", "error_spatial_rms", "error_upper_rms"))
    assert row["prepared_median_seconds"] is None


@pytest.mark.parametrize("damage", ["one_rejected_member", "incomplete_schedule", "failed_candidate"])
def test_frontier_cannot_use_unresolved_batch_or_schedule(measured_batch, damage):
    protocol, original = measured_batch
    tables = deepcopy(original)
    if damage == "one_rejected_member":
        tables["reference_rows"][0]["member_references"][2]["reference_accepted"] = False
        tables["reference_rows"][0]["reference_accepted"] = False
        expected = {"INCONCLUSIVE"}
    else:
        affected = tables["candidate_rows"][0]["variant"]
        if damage == "incomplete_schedule":
            tables["candidate_rows"].pop(0)
        else:
            tables["candidate_rows"][0]["status"] = "FAILED"
        expected = {"INCONCLUSIVE"}
    rows = experiment.select_frontiers(protocol, tables["candidate_rows"], tables["reference_rows"])
    checked = rows if damage == "one_rejected_member" else [row for row in rows if row["variant"] == affected]
    assert {row["status"] for row in checked} == expected
    assert all(row["selected_candidate_id"] is None for row in checked)


def test_teacher_uses_individual_members_and_retains_completed_attempts(monkeypatch):
    protocol = declaration()
    spec = protocol["plans"]["cases"][0]
    u = experiment.state(spec)
    seen = []
    def reference(field, horizon, equation, geometry, count, **kwargs):
        assert "check" in kwargs and field.shape[0] == 1
        seen.append(field.clone())
        accepted = len(seen) != 2
        return SimpleNamespace(state=field, uncertainty=1e-12, accepted=accepted,
            refinement_substeps=(count, 2 * count, 4 * count), refinement_differences=(1e-11, 1e-12),
            observed_order=3., reason="accepted" if accepted else "unresolved_refinement")
    monkeypatch.setattr(experiment, "refined_reference", reference)
    result, row = experiment._teacher(u, spec, Equation(spec["kappa"], spec["reaction_rate"]),
        Geometry(tuple(spec["grid"]), tuple(spec["lengths"])), protocol["config"], lambda: None)
    assert torch.equal(result.state, u)
    assert len(seen) == 5 and torch.equal(seen[1], seen[2])
    assert row["reference_accepted"] and row["completed_members"] == 4
    assert row["member_references"][1]["attempts"] == 2


def test_stop_during_reference_preserves_finished_members_and_pending_members(tmp_path, monkeypatch):
    protocol = declaration()
    stop = SimpleNamespace(requested=False)
    calls = []
    def reference(field, horizon, equation, geometry, count, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            stop.requested = True
            kwargs["check"]()
        return SimpleNamespace(state=field, uncertainty=1e-12, accepted=True,
            refinement_substeps=(count, 2 * count, 4 * count), refinement_differences=(1e-11, 1e-12),
            observed_order=3., reason="accepted")
    monkeypatch.setattr(experiment, "refined_reference", reference)
    summary = experiment.run(protocol, tmp_path, stop=stop)
    assert summary["status"] == "INCOMPLETE" and summary["scientific_outcome"] == "INCONCLUSIVE"
    payload = json.loads((tmp_path / "compact-spatial.json").read_text())
    ref = payload["reference_rows"][0]
    assert ref["completed_members"] == ref["accepted_members"] == 1
    assert [m["status"] for m in ref["member_references"]] == ["ACCEPTED", "INCOMPLETE", "NOT_STARTED", "NOT_STARTED"]
    assert not ref["reference_accepted"]
    assert all(row["status"] == "INCONCLUSIVE" for row in payload["frontier_rows"])
    validate_rows({name: payload[name] for name in TABLE_IDS}, protocol["plans"], complete=False)
    assert not (tmp_path / "COMPLETED").exists()


def test_stop_after_final_reference_keeps_acceptance_and_member_counts_consistent(tmp_path, monkeypatch):
    protocol = declaration()
    stop = SimpleNamespace(requested=False)
    calls = []
    def reference(field, horizon, equation, geometry, count, **kwargs):
        calls.append(None)
        if len(calls) == 4:
            stop.requested = True
        return SimpleNamespace(state=field, uncertainty=1e-12, accepted=True,
            refinement_substeps=(count, 2 * count, 4 * count), refinement_differences=(1e-11, 1e-12),
            observed_order=3., reason="accepted")
    monkeypatch.setattr(experiment, "refined_reference", reference)
    summary = experiment.run(protocol, tmp_path, stop=stop)
    assert summary["status"] == "INCOMPLETE"
    payload = json.loads((tmp_path / "compact-spatial.json").read_text())
    ref = payload["reference_rows"][0]
    assert ref["reference_accepted"] and ref["completed_members"] == ref["accepted_members"] == 4
    assert all(member["status"] == "ACCEPTED" for member in ref["member_references"])
    assert not payload["candidate_rows"]
    validate_rows({name: payload[name] for name in TABLE_IDS}, protocol["plans"], complete=False)


@pytest.mark.parametrize("reason", ["stop", "budget"])
def test_stop_before_compute_publishes_complete_frontier_inventory(tmp_path, reason):
    protocol = declaration()
    times = iter([0., 2000., 2000.])
    summary = experiment.run(protocol, tmp_path, stop=SimpleNamespace(requested=reason == "stop"),
        clock=(lambda: next(times)) if reason == "budget" else (lambda: 0.))
    assert summary["status"] == "INCOMPLETE"
    assert summary["reported_candidates"] == 0 and summary["reported_references"] == 0
    assert summary["reported_frontiers"] == len(METHODS) * 2 * 4
    assert summary["frontier_status_counts"] == {"INCONCLUSIVE": len(METHODS) * 2 * 4}


def test_failed_identity_prevents_completed_outcome(tmp_path, measured_batch):
    protocol, original = measured_batch
    tables = deepcopy(original)
    tables["parity_rows"][0].update(status="FAIL", passed=False)
    summary = experiment._publish(protocol, tmp_path, tables, "COMPLETED", [], 1.)
    assert summary["status"] == "FAILED" and summary["scientific_outcome"] == "INCONCLUSIVE"
    assert summary["correctness_failures"] == [tables["parity_rows"][0]["parity_id"]]


def test_reference_callbacks_preserve_numerics_and_can_interrupt_between_steps():
    u = torch.tensor([[[.4, .5, .6, .5]]], dtype=torch.float64)
    equation, geometry = Equation(.001, 1.), Geometry((4,), (1.,))
    calls = []
    plain = reference_step(u, .1, equation, geometry, 65)
    observed = reference_step(u, .1, equation, geometry, 65, check=lambda: calls.append(None))
    assert torch.equal(plain, observed) and len(calls) == 4
    calls.clear()
    def stop():
        calls.append(None)
        if len(calls) == 2:
            raise InterruptedError("bounded teacher stop")
    with pytest.raises(InterruptedError, match="bounded teacher stop"):
        refined_reference(u, .1, equation, geometry, 65, check=stop)
    assert len(calls) == 2
