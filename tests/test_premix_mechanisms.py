"""Mechanism evidence includes difficult spectra, per-member errors and stops."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.premix import mechanisms as m
from tdn.analysis.work_precision.protocol import TABLE_IDS
from tdn.numerics.types import Equation, Geometry


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("panel,cases,candidates", [("accuracy", 24, 1308), ("scaling", 36, 648)])
def test_full_plan_is_frozen_complete_and_exercises_larger_domains(panel, cases, candidates):
    protocol = m.build_protocol("full", panel)
    plans = protocol["plans"]
    assert len(plans["cases"]) == cases
    assert len(plans["expected_ids"]["candidate_rows"]) == candidates
    assert protocol["config"]["max_seconds"] == 1200
    assert protocol["config"]["reference_attempts"] == 7
    assert min(protocol["config"]["error_targets"]) == 2e-8
    for ids in plans["expected_ids"].values():
        assert len(ids) == len(set(ids))
    grids = {tuple(s["grid"]) for s in plans["cases"]}
    assert ({(128,), (512,), (32, 32), (64, 64)} <= grids)
    if panel == "accuracy":
        assert {s["regime"] for s in plans["cases"]} == {"favorable", "typical", "adverse"}
        assert {s["pattern"] for s in plans["cases"]} == {"smooth", "broadband", "cutoff_pair", "near_nyquist", "localized", "front"}
    else:
        assert {(2048,), (128, 128)} <= grids
        for grid in grids:
            group = [s for s in plans["cases"] if tuple(s["grid"]) == grid]
            assert {(s["pattern"], s["batch_size"]) for s in group} == {(p, b) for p in ("smooth", "cutoff_pair") for b in (1, 4, 16)}
            assert all("selected_output_gl3_m4" not in s["methods"] for s in group)


@pytest.mark.parametrize("profile,panel", [("bad", "accuracy"), ("full", "bad")])
def test_invalid_protocol_mode_rejected(profile, panel):
    with pytest.raises(ValueError):
        m.build_protocol(profile, panel)


def test_all_states_are_admissible_and_prefix_members_are_identical():
    for panel in ("accuracy", "scaling"):
        protocol = m.build_protocol("full", panel)
        for spec in protocol["plans"]["cases"]:
            u = m.state(spec)
            assert u.shape == (spec["batch_size"], 1, *spec["grid"])
            assert bool(((u >= 0) & (u <= 1)).all())
            assert float(u.std()) > 0
            if spec["batch_size"] > 1:
                single = deepcopy(spec)
                single["batch_size"] = 1
                assert torch.equal(m.state(single), u[:1])


def test_timing_orders_rotate_all_declared_methods_without_dropping_slow_controls():
    for panel in ("accuracy", "scaling"):
        p = m.build_protocol("full", panel)
        for order in p["plans"]["timing_orders"]:
            spec = next(s for s in p["plans"]["cases"] if s["case_id"] == order["case_id"])
            for values in [order["preparation_order"], order["warmup_order"], *order["measured_orders"]]:
                assert sorted(values) == sorted(spec["methods"])
            assert len(order["measured_orders"]) == 5
            assert order["measured_orders"][0] != order["warmup_order"]


@pytest.fixture(scope="module")
def measured_cutoff():
    p = m.build_protocol("smoke", "accuracy")
    spec = next(s for s in p["plans"]["cases"] if s["pattern"] == "cutoff_pair")
    tables = {name: [] for name in TABLE_IDS}
    with torch.no_grad():
        m._run_case(p, spec, tables, lambda: None, {})
    tables["frontier_rows"] = m.select_frontiers(p, tables["candidate_rows"], tables["reference_rows"])
    return p, spec, tables


def test_actual_cutoff_candidates_have_accepted_reference_and_retained_raw_costs(measured_cutoff):
    p, spec, tables = measured_cutoff
    m.validate_tables(p, tables, complete=False)
    assert tables["reference_rows"][0]["reference_accepted"]
    assert len(tables["candidate_rows"]) == len(spec["methods"]) * len(spec["nsteps"])
    assert all(r["status"] == "VALID" for r in tables["candidate_rows"])
    for row in tables["candidate_rows"]:
        assert len(row["timing_repeats"]) == 5
        assert row["warmup"]["repeat_index"] == -1
        assert row["error_upper_rms"] == max(r["error_upper_rms"] for r in row["member_errors"])
        assert row["error_upper_max_estimate"] == max(r["error_upper_max_estimate"] for r in row["member_errors"])
        assert row["setup_inclusive_median_seconds"] == row["preparation_seconds"] + row["prepared_median_seconds"]
        assert row["work_per_rollout"]["fft_transformed_cells"] > 0
        if row["variant"] == "selected_output_gl3_m4":
            assert row["work_per_rollout"]["pair_product_cells"] == 128 * 9 * row["nsteps"]


def test_output_projection_restores_retained_high_high_modes_without_claiming_full_field_identity(measured_cutoff):
    _, _, tables = measured_cutoff
    rows = tables["parity_rows"]
    assert all(r["status"] == "PASS" for r in rows if r["check"] != "approximation")
    approximate = {r["variant"]: r for r in rows if r["check"] == "approximation"}
    assert approximate["compact_gl3_m4"]["retained_low_output_rms_difference"] > 1e-8
    assert approximate["output_gl3_m4"]["retained_low_output_rms_difference"] < 1e-12
    assert approximate["output_gl3_m4"]["rms_difference"] > 0
    assert all(r["status"] == "OBSERVED" and r["passed"] is None for r in approximate.values())


def test_reference_cache_reuses_only_identical_accepted_prefix_members():
    p = m.build_protocol("smoke", "scaling")
    cache, records = {}, []
    for spec in p["plans"]["cases"][:2]:
        u = m.state(spec)
        ref = {**m._new_reference(spec), **m._context(spec)}
        _, ref = m._teacher(u, spec, Equation(spec["kappa"], spec["reaction_rate"]),
                            Geometry(tuple(spec["grid"]), tuple(spec["lengths"])),
                            p["config"], lambda: None, ref, cache)
        records.append(ref)
    assert records[0]["reference_accepted"] and records[1]["reference_accepted"]
    assert records[0]["cached_members"] == 0 and records[1]["cached_members"] == 1
    reused = records[1]["member_references"][0]
    assert reused["rhs_evaluations_this_case"] == 0
    assert reused["reference_source"] == records[0]["member_references"][0]["reference_source"]
    assert reused["refinement_attempts"] == records[0]["member_references"][0]["refinement_attempts"]
    spec = p["plans"]["cases"][0]
    changed = {**spec, "reaction_rate": spec["reaction_rate"] * 2}
    assert m._reference_key(m.state(spec), spec) != m._reference_key(m.state(spec), changed)


def test_frontier_is_inconclusive_when_any_declared_schedule_is_missing(measured_cutoff):
    p, spec, tables = measured_cutoff
    candidates = [r for r in tables["candidate_rows"] if not (r["variant"] == "strang" and r["nsteps"] == 4)]
    rows = m.select_frontiers(p, candidates, tables["reference_rows"])
    assert all(r["status"] == "INCONCLUSIVE" for r in rows if r["case_id"] == spec["case_id"] and r["variant"] == "strang")
    assert all(r["selected_candidate_id"] is None for r in rows if r["status"] == "INCONCLUSIVE")


def test_warmed_and_setup_frontiers_are_selected_independently(measured_cutoff):
    p, spec, tables = measured_cutoff
    candidates = deepcopy(tables["candidate_rows"])
    for row in candidates:
        if row["variant"] == "strang":
            row.update(error_upper_rms=1e-10, error_upper_max_estimate=1e-10,
                       prepared_median_seconds=.1 if row["nsteps"] == 1 else .2,
                       setup_inclusive_median_seconds=10. if row["nsteps"] == 1 else .3)
    rows = m.select_frontiers(p, candidates, tables["reference_rows"])
    selected = [r for r in rows if r["case_id"] == spec["case_id"] and r["variant"] == "strang"]
    assert all(r["selected_nsteps"] == 1 and r["setup_selected_nsteps"] == 4 for r in selected)


def test_corrupted_context_or_member_aggregation_fails_validation(measured_cutoff):
    p, _, tables = measured_cutoff
    corrupted = deepcopy(tables)
    corrupted["candidate_rows"][0]["regime"] = "favorable"
    with pytest.raises(ValueError, match="context"):
        m.validate_tables(p, corrupted, complete=False)
    corrupted = deepcopy(tables)
    corrupted["candidate_rows"][0]["member_errors"][0]["error_rms"] *= 2
    with pytest.raises(ValueError, match="member"):
        m.validate_tables(p, corrupted, complete=False)


@pytest.mark.parametrize("reason", ["stop", "budget"])
def test_stop_and_budget_write_partial_evidence_without_completion_seal(tmp_path, reason):
    p = m.build_protocol("smoke", "accuracy")
    ticks = iter([0., 300., 301.])
    kwargs = {"stop": SimpleNamespace(requested=True)} if reason == "stop" else {"clock": lambda: next(ticks)}
    result = m.run(p, tmp_path / reason, **kwargs)
    assert result["status"] == "INCOMPLETE" and result["scientific_outcome"] == "INCONCLUSIVE"
    assert result["errors"]
    canonical = json.loads((tmp_path / reason / "metrics.json").read_text())
    assert canonical["candidate_rows"] == []
    assert all(r["status"] == "INCONCLUSIVE" for r in canonical["frontier_rows"])
    assert not (tmp_path / reason / "COMPLETED").exists()
    with pytest.raises(ValueError, match="Preserve"):
        m.run(p, tmp_path / reason)


def test_changed_frozen_protocol_is_rejected_before_execution(tmp_path):
    p = m.build_protocol("smoke", "accuracy")
    p["config"]["max_seconds"] = 10_000
    with pytest.raises(ValueError, match="frozen"):
        m.run(p, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_existing_conflicting_protocol_is_not_overwritten(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text('{"preexisting": true}\n')
    with pytest.raises(ValueError, match="contradictory"):
        m.run(m.build_protocol("smoke", "accuracy"), tmp_path)
    assert json.loads(path.read_text()) == {"preexisting": True}
    assert not (tmp_path / "metrics.json").exists()
