"""Coverage, failure semantics and bounded execution are scientific safeguards."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tdn.analysis.mechanisms.protocol import DEFAULTS, PANELS, make_protocol, validate_config, validate_panel
from tdn.analysis.mechanisms.run import run


def declaration():
    config = validate_config(deepcopy(DEFAULTS), smoke=True)
    plans = {name: {"expected_case_ids": ["one"]} for name in PANELS}
    return make_protocol(config, plans, {}, [])


def panel(name, *, outcome="PASS", kind="correctness"):
    return {"panel": name, "status": "COMPLETED", "rows": [
        {"case_id": "one", "mechanism": "control", "test": "control", "variant": "fixed",
         "kind": kind, "outcome": outcome, "inputs": {}, "metrics": {"error": .01}, "note": "control"}]}


@pytest.mark.parametrize("key,value", [("max_seconds", 1201), ("max_seconds", True),
    ("panels", ["temporal"]), ("tolerance", .02), ("intraop_threads", 4), ("suite", "architecture")])
def test_protocol_cannot_silently_relax_or_expand(key, value):
    raw = deepcopy(DEFAULTS)
    raw[key] = value
    with pytest.raises(ValueError):
        validate_config(raw)


def test_coverage_rejects_missing_duplicate_and_nonfinite_rows():
    plan = {"expected_case_ids": ["one"]}
    for change in (lambda r: r.update(rows=[]), lambda r: r["rows"].extend(deepcopy(r["rows"])),
                   lambda r: r["rows"][0]["metrics"].update(error=float("nan")),
                   lambda r: r["rows"][0].pop("note"),
                   lambda r: r["rows"][0]["metrics"].update({"": 1})):
        result = panel("temporal")
        change(result)
        with pytest.raises(ValueError):
            validate_panel(result, "temporal", plan)


def test_negative_findings_are_completed_but_correctness_failure_is_not(tmp_path):
    modules = {name: SimpleNamespace(run=lambda c, check, name=name:
                panel(name, outcome="EXPECTED_LIMITATION", kind="negative_control")) for name in PANELS}
    summary = run(declaration(), tmp_path, modules)
    assert summary["status"] == "COMPLETED"
    assert summary["outcome_counts"] == {"EXPECTED_LIMITATION": 3}
    modules["coordinates"] = SimpleNamespace(run=lambda c, check: panel("coordinates", outcome="FAIL"))
    summary = run(declaration(), tmp_path, modules)
    assert summary["status"] == "FAILED"
    assert summary["correctness_failures"] == ["coordinates/one"]
    modules["coordinates"] = SimpleNamespace(run=lambda c, check:
        panel("coordinates", outcome="FAIL", kind="negative_control"))
    summary = run(declaration(), tmp_path, modules)
    assert summary["status"] == "FAILED"
    assert summary["audit_failures"] == ["coordinates/one"]


def test_exhaustion_retains_completed_panels_and_declares_unreported_cases(tmp_path):
    tick = [0.]

    def first(config, check):
        tick[0] = 1.
        return panel("coordinates")

    def second(config, check):
        tick[0] = 181.
        check()

    modules = {"coordinates": SimpleNamespace(run=first), "temporal": SimpleNamespace(run=second),
               "structure": SimpleNamespace(run=lambda *args: pytest.fail("must not run"))}
    summary = run(declaration(), tmp_path, modules, clock=lambda: tick[0])
    assert summary["status"] == "INCOMPLETE"
    assert summary["reported_cases"] == 1
    assert summary["expected_cases"] == 3
    import json
    assert json.loads((tmp_path / "temporal.json").read_text())["unreported_case_ids"] == ["one"]
    assert not (tmp_path / "COMPLETED").exists()


def test_unaccepted_reference_cannot_be_reported_as_completed(tmp_path):
    modules = {name: SimpleNamespace(run=lambda c, check, name=name:
                panel(name, outcome="INCONCLUSIVE", kind="scientific")) for name in PANELS}
    assert run(declaration(), tmp_path, modules)["status"] == "INCOMPLETE"


def test_budget_after_panel_return_retains_its_finished_rows(tmp_path):
    tick = [0.]

    def finish_late(config, check):
        tick[0] = 181.
        return panel("coordinates")

    modules = {name: SimpleNamespace(run=finish_late) for name in PANELS}
    summary = run(declaration(), tmp_path, modules, clock=lambda: tick[0])
    assert summary["status"] == "INCOMPLETE" and summary["reported_cases"] == 1
    import json
    assert json.loads((tmp_path / "coordinates.json").read_text())["unreported_case_ids"] == []
