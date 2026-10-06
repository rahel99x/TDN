"""Frozen case coverage and truthful scientific versus computational status."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tdn.analysis.interactions import experiment
from tdn.analysis.interactions.protocol import DEFAULTS, make_protocol, validate_config, validate_rows


def declaration(*, smoke=True):
    config = validate_config(deepcopy(DEFAULTS), smoke=smoke)
    return make_protocol(config, experiment.plan(config), {}, [])


@pytest.mark.parametrize("key,value", [("max_seconds", 1201), ("max_seconds", True),
    ("panels", ["one_step"]), ("reference_tolerance", .01), ("reference_attempts", 9),
    ("intraop_threads", 4), ("suite", "learned")])
def test_frozen_protocol_cannot_silently_change(key, value):
    raw = deepcopy(DEFAULTS)
    raw[key] = value
    with pytest.raises(ValueError):
        validate_config(raw)


def test_predeclared_crossing_and_smoke_subset():
    full, smoke = declaration(smoke=False), declaration()
    cross = [case for case in full["plans"]["one_step"]["cases"] if case["case_id"].startswith("cross/")]
    assert len(cross) == 16
    assert len({(tuple(c["grid"]), c["kappa"], c["reaction_rate"], c["h"]) for c in cross}) == 16
    assert {(c["mean"], c["pattern"], c["amplitude"]) for c in cross} == {(.5, "mixed", .14)}
    assert full["config"]["max_seconds"] == 1200 and smoke["config"]["max_seconds"] == 180
    for name in full["plans"]:
        assert set(smoke["plans"][name]["expected_case_ids"]) <= set(full["plans"][name]["expected_case_ids"])
    assert sum(len(plan["expected_case_ids"]) for plan in smoke["plans"].values()) == 124


def test_missing_duplicate_and_nonfinite_rows_rejected():
    plans = {"controls": {"expected_case_ids": ["one"]}}
    row = experiment._row("one", "controls", "fixed", "correctness", "PASS", {"error": 0.}, {}, "control")
    validate_rows([row], plans, complete=True)
    for rows in ([], [row, row], [{**row, "metrics": {"bad": float("nan")}}], [{**row, "panel": "rollout"}]):
        with pytest.raises(ValueError):
            validate_rows(rows, plans, complete=True)


def test_stop_retains_partial_rows_and_unreported_ids(tmp_path, monkeypatch):
    protocol = declaration()
    stop = SimpleNamespace(requested=False)

    def partial(rows, check):
        case_id = protocol["plans"]["controls"]["expected_case_ids"][0]
        rows.append(experiment._row(case_id, "controls", "fixed", "correctness", "PASS", {"error": 0.}, {}, "control"))
        stop.requested = True
        check()

    monkeypatch.setattr(experiment, "_controls", partial)
    summary = experiment.run(protocol, tmp_path, stop=stop)
    assert summary["status"] == "INCOMPLETE" and summary["scientific_outcome"] == "INCONCLUSIVE"
    assert summary["reported_cases"] == 1
    canonical = json.loads((tmp_path / "interaction-screen.json").read_text())
    assert len(canonical["unreported_case_ids"]) == summary["expected_cases"] - 1
    assert not (tmp_path / "COMPLETED").exists()


def test_budget_retains_finished_rows(tmp_path, monkeypatch):
    protocol = declaration()
    tick = [0.]

    def controls(rows, check):
        case_id = protocol["plans"]["controls"]["expected_case_ids"][0]
        rows.append(experiment._row(case_id, "controls", "fixed", "correctness", "PASS", {"error": 0.}, {}, "control"))
        tick[0] = 181.
        check()

    monkeypatch.setattr(experiment, "_controls", controls)
    summary = experiment.run(protocol, tmp_path, clock=lambda: tick[0])
    assert summary["status"] == "INCOMPLETE" and summary["reported_cases"] == 1
    assert "budget" in summary["errors"][0]


def test_unaccepted_reference_is_explicitly_scientifically_inconclusive(tmp_path):
    protocol = declaration()
    case = protocol["plans"]["one_step"]["expected_case_ids"][0]
    row = experiment._row(case, "one_step", "reference", "scientific", "INCONCLUSIVE",
                          {"reference_accepted": False}, {}, "reference rejected")
    result = experiment._publish(protocol, tmp_path, [row], "COMPLETED", [], 0.)
    assert result["scientific_outcome"] == "INCONCLUSIVE"


@pytest.mark.parametrize("outcome,computational,scientific", [
    ("OBSERVED", "COMPLETED", "OBSERVED_MIXED"),
    ("INCONCLUSIVE", "COMPLETED", "INCONCLUSIVE"),
    ("FAIL", "FAILED", "INCONCLUSIVE")])
def test_completed_coverage_does_not_hide_scientific_outcome(tmp_path, monkeypatch, outcome, computational, scientific):
    config = validate_config(deepcopy(DEFAULTS), smoke=True)
    plans = {"controls": {"expected_case_ids": ["control"]},
             "one_step": {"expected_case_ids": ["one/reference"], "cases": [{"case_id": "one"}]},
             "rollout": {"expected_case_ids": ["roll/reference"], "cases": [{"case_id": "roll"}]}}
    protocol = make_protocol(config, plans, {}, [])

    def controls(rows, check):
        rows.append(experiment._row("control", "controls", "fixed", "correctness", "PASS", {}, {}, "control"))

    def case(rows, spec, panel, config, check):
        rows.append(experiment._row(f"{spec['case_id']}/reference", panel, "reference", "scientific", outcome, {}, {}, "fixed case"))

    monkeypatch.setattr(experiment, "_controls", controls)
    monkeypatch.setattr(experiment, "_case", case)
    result = experiment.run(protocol, tmp_path)
    assert result["status"] == result["computational_status"] == computational
    assert result["scientific_outcome"] == scientific
    assert result["reported_cases"] == result["expected_cases"] == 3
