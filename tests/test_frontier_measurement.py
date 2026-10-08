"""Regression coverage for the measurement failures found in the native review."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tdn.analysis.frontier.measurement import (endpoint_eligibility, field_cluster_interval,
    final_time, matched_key, measure_paired, paired_frontiers, run_audit, supported_amortization)


def endpoint(**updates):
    return {"model_id": "rank1-seed-1", "family": "rank1", "parent_id": "parent-0",
        "field_cluster": "field-0", "grid": 8, "track": "discrete", "final_time": .1,
        "upper_rms": 1e-6, "upper_max": 2e-6, "reference_accepted": True,
        "cost_seconds": 2., "seed": 1, "schedule_id": "short", **updates}


class Budget:
    def __init__(self):
        self.calls = 0
    def check(self):
        self.calls += 1


class Context:
    def __init__(self, path):
        self.path = path
        path.mkdir(exist_ok=True)
        self.rows = []
        self.protocol = {"profile": "smoke"}
        self.budget = Budget()
    def record(self, experiment_id, mechanisms, **payload):
        self.rows.append({"experiment_id": experiment_id, "mechanisms": mechanisms, **payload})


def test_final_time_uses_explicit_schedule_and_decimal_reference_convention():
    assert final_time({"schedule": [.1, .2]}) == final_time({"final_time": .3})
    with pytest.raises(ValueError, match="differs"):
        final_time({"final_time": .1, "schedule": [.1, .2]})
    with pytest.raises(ValueError, match="required"):
        final_time({"schedule_id": "long"})


@pytest.mark.parametrize("field,value", [("parent_id", "different"), ("grid", 16),
    ("track", "continuum"), ("final_time", .3)])
def test_matched_key_preserves_every_target_coordinate(field, value):
    assert matched_key(endpoint(), 1e-4, 2e-4) != matched_key(endpoint(**{field: value}), 1e-4, 2e-4)


def test_matched_key_preserves_separate_rms_and_maximum_targets():
    assert matched_key(endpoint(), 1e-4, 2e-4) != matched_key(endpoint(), 2e-4, 1e-4)


@pytest.mark.parametrize("updates,reason", [
    ({"numerical_failure": "NONFINITE"}, "NUMERICAL_FAILURE"),
    ({"reference_accepted": False}, "REFERENCE_UNACCEPTED"),
    ({"reference_accepted": None}, "REFERENCE_UNACCEPTED"),
    ({"upper_rms": None}, "MISSING_METRIC"),
    ({"upper_max": math.nan}, "NONFINITE_OR_NEGATIVE_METRIC"),
    ({"cost_seconds": math.inf}, "NONFINITE_OR_NEGATIVE_METRIC"),
    ({"cost_seconds": -1}, "NONFINITE_OR_NEGATIVE_METRIC"),
    ({"cost_seconds": 0}, "INVALID_COST"),
    ({"upper_max": 1.}, "ACCURACY_INFEASIBLE"),
])
def test_eligibility_preserves_invalid_and_unaccepted_values(updates, reason):
    assert endpoint_eligibility(endpoint(**updates), 1e-4, 1e-4) == reason


def test_frontier_cannot_replace_earlier_target_by_cheap_later_output():
    rows = [endpoint(), endpoint(final_time=.3, cost_seconds=.01, schedule_id="long"),
            endpoint(model_id="df_base", family="df_base", seed=None, cost_seconds=1.)]
    result = paired_frontiers(rows, target_pairs=[(1e-4, 1e-4)], comparators={"classical": "df_base"})
    candidate = [r for r in result if r["model_id"] == "rank1-seed-1"]
    short = next(r for r in candidate if r["final_time"] == .1)
    long = next(r for r in candidate if r["final_time"] == .3)
    assert short["control_over_candidate_speed_ratio"] == .5
    assert long["eligibility"] == "CONTROL_INFEASIBLE"
    assert long["control_statuses"] == ["MISSING_ARM"]


def test_legacy_production_frontier_and_inference_separate_final_times(tmp_path):
    from tdn.analysis.roadmap.neural import _matched_comparisons
    model = SimpleNamespace(family="rank1", selection_metadata={"seed": 1},
                            architecture_metadata=lambda: {"combination_ids": []})
    rows = [endpoint(), endpoint(final_time=.3, cost_seconds=.01, schedule_id="long"),
        endpoint(model_id="df_base", family="df_base", seed=None, cost_seconds=1.),
        endpoint(model_id="df_base", family="df_base", seed=None, cost_seconds=1., final_time=.3, schedule_id="long")]
    ctx = Context(tmp_path)
    assert _matched_comparisons(ctx, rows, {"rank1-seed-1": model}, [(1e-4, 1e-4)]) == 2
    payload = json.loads((tmp_path / "matched_comparisons.json").read_text())
    short = next(r for r in payload["comparisons"] if r["final_time"] == .1)
    assert short["control_over_candidate_speed_ratio"]["best_classical"] == .5
    assert {r["final_time"] for r in payload["paired_inference"]} == {.1, .3}
    assert len({r["experiment_id"] for r in ctx.rows}) == len(ctx.rows)


def test_paired_timer_randomizes_interleaved_rounds_and_preserves_every_call():
    order = []
    budget = Budget()
    calls = {name: (lambda name=name: order.append(name) or len(order)) for name in ("a", "b", "c")}
    outputs, timing = measure_paired(calls, repeats=5, warmup=1, seed=71, budget=budget)
    assert set(outputs) == set(calls)
    assert len(order) == 3*(1+1+5)
    assert len(timing["cold_samples"]) == 3
    reported = timing["cold_order"] + sum((r["order"] for r in timing["warmup_rounds"] + timing["rounds"]), [])
    assert reported == order
    assert len({tuple(r["order"]) for r in timing["rounds"]}) > 1
    for name in calls:
        entry = timing["methods"][name]
        assert entry["samples_seconds"] == [r["samples_seconds"][name] for r in timing["rounds"]]
        assert entry["total_seconds"] == sum(entry["samples_seconds"])
        assert all(v > 0 for v in entry["samples_seconds"])
    assert budget.calls == 2*len(order)
    assert "no cold-process" in timing["cold_scope"]


@pytest.mark.parametrize("repeats,warmup", [(1, 1), (3, 0), (3, -1)])
def test_repeated_paired_measurement_cannot_silently_degrade(repeats, warmup):
    with pytest.raises(ValueError):
        measure_paired({"one": lambda: 1}, repeats=repeats, warmup=warmup)


def test_paired_timer_propagates_budget_exhaustion():
    class Exhausted:
        def check(self):
            raise TimeoutError("bounded")
    with pytest.raises(TimeoutError, match="bounded"):
        measure_paired({"one": lambda: 1}, budget=Exhausted())


def test_teacher_refines_actual_time_cap_and_crosschecks_another_method():
    from tdn.analysis.roadmap.portability import independent_reference
    initial = np.array([.2, .7])
    value, info = independent_reference(initial, .3, lambda x: 2*x*(1-x))
    truth = initial*np.exp(.6)/(1+initial*np.expm1(.6))
    assert np.max(np.abs(value-truth)) < 1e-11
    assert info["max_steps"] == [.3/8, .3/16, .3/32]
    assert info["methods"] == ["DOP853", "DOP853", "RK45"]
    assert info["accepted_time_steps"][1] > info["accepted_time_steps"][0]
    assert info["rhs_evaluations"] == sum(info["rhs_evaluations_per_solve"])
    assert info["uncertainty_max"] >= info["roundoff_floor"] > 0


def test_zero_rhs_reference_never_claims_zero_uncertainty():
    from tdn.analysis.roadmap.portability import independent_reference
    value, info = independent_reference(np.array([.3]), .1, lambda x: 0*x)
    assert value[0] == .3
    assert info["uncertainty_max"] > 0


def test_cluster_bootstrap_does_not_count_repeated_queries_as_new_fields():
    rows = [{"field_cluster": f, "seed": s, "difference": f+s} for f in range(4) for s in range(3)]
    original = field_cluster_interval(rows, repeats=100)
    copies = field_cluster_interval(rows*10, repeats=100)
    assert copies["independent_fields"] == original["independent_fields"] == 4
    assert copies["lower"] == original["lower"]
    assert copies["upper"] == original["upper"]
    assert field_cluster_interval(rows[:-1], repeats=100)["status"] == "INCOMPLETE_CROSSED_DESIGN"


def test_all_classical_policy_cannot_claim_neural_amortization():
    result = supported_amortization({"train": 30.}, 1., 2., matched_accuracy=True,
                                   accepted_neural_queries=0, saving_lower=.5)
    assert result["status"] == "NO_ACCEPTED_NEURAL_QUERIES"
    assert result["break_even_queries"] is None
    uncertain = supported_amortization({"train": 30.}, 1., 2., matched_accuracy=True,
                                      accepted_neural_queries=3, saving_lower=-.1)
    assert uncertain["status"] == "SAVING_UNCERTAINTY_INCLUDES_ZERO"
    assert uncertain["break_even_queries"] is None


def test_paired_deployment_uses_components_from_actual_median_sample():
    from tdn.analysis.roadmap.policy import _paired_deployment_measure
    count = []
    def policy():
        count.append(len(count))
        return count[-1], {"standalone_seconds": 0., "accounted_component_seconds": 0.}
    (value, outcome), control, timing = _paired_deployment_measure(policy, lambda: (7, {}), device="cpu", repeats=3)
    assert len(count) == 5
    assert value in (2, 3, 4)
    assert control == (7, {})
    assert outcome["standalone_seconds"] == timing["methods"]["policy"]["median_seconds"]
    assert outcome["accounted_component_seconds"] <= outcome["standalone_seconds"]


def test_gate_one_executes_all_real_measurement_probes(tmp_path):
    ctx = Context(tmp_path)
    result = run_audit(ctx)
    assert result["audit_probes"] == len(ctx.rows) == 4
    assert all(c["verdict"] == "GOOD" for row in ctx.rows for c in row["checks"])


def reaggregate_module():
    spec = importlib.util.spec_from_file_location("frontier_reaggregate", Path(__file__).resolve().parents[1]/"scripts/frontier_reaggregate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_reaggregate_recovers_horizon_from_frozen_schedule():
    module = reaggregate_module()
    rows = [endpoint(schedule_id="schedule-0"), endpoint(schedule_id="schedule-1", cost_seconds=.01),
        endpoint(model_id="df_base", family="df_base", seed=None, cost_seconds=1., schedule_id="schedule-0")]
    for row in rows:
        row.pop("final_time")
    result = module.recompute(rows, {"confirm_schedules": [[.1], [.3]], "targets": [1e-4]})
    assert result["final_times"] == [.1, .3]
    short = next(r for r in result["comparisons"] if r["final_time"] == .1)
    assert short["control_over_candidate_speed_ratio"]["best_classical"] == .5
    assert not next(r for r in result["comparisons"] if r["final_time"] == .3)["control_feasible"]["best_classical"]


def test_offline_reaggregate_never_overwrites_source_or_existing_destination(tmp_path):
    module = reaggregate_module()
    module.ROOT = tmp_path
    source = tmp_path/"source"; source.mkdir()
    existing = tmp_path/"existing"; existing.mkdir()
    for output in (source/"derived", source, existing):
        with pytest.raises(ValueError, match="fresh output"):
            module.run(source, output)


def test_offline_json_parser_rejects_ambiguous_duplicate_and_nonfinite_values(tmp_path):
    module = reaggregate_module()
    path = tmp_path/"sample.json"
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            module.read_json(path)
