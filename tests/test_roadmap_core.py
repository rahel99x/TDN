"""Evidence grading, missing checks and independently reconstructible logs."""
import copy
import csv
import json

import numpy as np
import pytest
import torch

from tdn.analysis.roadmap.core import Context, check, score_checks, validate_row, write_reviews
from tdn.analysis.roadmap.protocol import build_protocol
from tdn.research.experiment import Budget
from tdn.research.protocol import digest


def context(path):
    return Context(build_protocol("smoke"), "audit", path, {}, "cpu", Budget(30))


def evidence():
    return [check("maximum_error", 1e-5, 2e-4, category="gap"),
            check("same_step_no_harm", 1e-5, 2e-5, category="gap"),
            check("physical_null", 0.0, 1e-12, category="math")]


@pytest.mark.parametrize("measured,relation,target,verdict", [
    (1.0, "le", 2.0, "GOOD"), (1.0, "ge", 2.0, "BAD"),
    (True, "eq", True, "GOOD"), (2, "lt", 2, "BAD"),
    (3, "gt", 2, "GOOD"), (None, "le", 2, "NA"),
    (float("nan"), "le", 2, "BAD"), (float("inf"), "ge", 2, "BAD"),
    (np.float64("nan"), "le", 2, "BAD"), (torch.tensor(float("nan")), "le", 2, "BAD"),
])
def test_scalar_checks_keep_nonfinite_failure_distinct_from_missing(measured, relation, target, verdict):
    result = check("scientific_requirement", measured, target, relation)
    assert result["verdict"] == verdict
    assert result["score_1_100"] == (100 if verdict == "GOOD" else 1)
    assert result["proof_status"] == "NOT_A_PROOF"
    json.dumps(result, allow_nan=False)


def test_missing_required_evidence_retains_weight_and_cannot_score_good():
    checks = [check("math", 0, 0, "eq", category="math"),
              check("utility_gap", None, 1, category="gap")]
    score = score_checks(checks)
    assert score["verdict"] == "NA"
    assert score["score_1_100"] == 41
    assert score["evidence_coverage"] == pytest.approx(0.4)
    assert score["na_checks"] == 1
    assert score_checks([])["score_1_100"] == 1
    assert score_checks([])["evidence_coverage"] == 0


def test_optional_failed_diagnostic_does_not_replace_required_scientific_gate():
    values = evidence() + [check("exploratory", 2, 1, required=False)]
    assert score_checks(values)["verdict"] == "GOOD"
    values[0] = check("maximum_error", 1, 0.1, category="gap")
    score = score_checks(values)
    assert score["verdict"] == "BAD"
    assert 1 <= score["score_1_100"] < 100


def test_record_supplies_explicit_na_for_missing_gap_or_math(tmp_path):
    ctx = context(tmp_path)
    row = ctx.record("only-correctness", ["M22"], checks=[check("finite", True, True, "eq", category="correctness")])
    assert row["gap_assessment"]["verdict"] == "NA"
    assert row["math_assessment"]["verdict"] == "NA"
    assert row["assessment"]["verdict"] == "NA"
    assert {"unmeasured-gap", "unmeasured-math"} <= set(row["required_check_ids"])
    assert validate_row(row) == row


def test_live_tower_emission_uses_count_without_inventing_total(tmp_path, monkeypatch):
    from pathlib import Path
    from tdn.reporting import begin_report
    source = tmp_path / "audit"
    source.mkdir()
    report = Path(begin_report(source, name="TDN/roadmap/audit", script="scripts/roadmap.py",
                              parameters={"benchmark_suite": "roadmap"}, report_parent=tmp_path / "tower"))
    monkeypatch.setenv("TDN_TOWER_DIR", str(report))
    ctx = context(source)
    ctx.record("live-case", ["M02"], checks=evidence())
    row = json.loads((report / "metrics.jsonl").read_text().splitlines()[-1])
    assert row["step"] == 1
    assert row["metrics"]["experiment_rows"] == 1
    assert "progress" not in row  # The eventual row count is not yet known.


def test_rows_reconstruct_operand_grades_and_preserve_cost_configuration(tmp_path):
    ctx = context(tmp_path)
    row = ctx.record("solver/case", ["M02"], combination_ids=["C1"], checks=evidence(),
        metrics={"parameters": 120, "median_seconds": 0.25, "cost": {"rejected_seconds": 0.1}},
        config={"grid": 32, "rank": 2, "steps": [0.03, 0.07]})
    encoded = json.loads((tmp_path / "rows.jsonl").read_text())
    assert validate_row(encoded) == row
    assert encoded["effective_config"]["steps"] == [0.03, 0.07]
    assert encoded["metrics"]["cost"]["rejected_seconds"] == 0.1
    assert encoded["assessment"]["score_1_100"] == 100
    with pytest.raises(ValueError, match="Repeated experiment"):
        ctx.record("solver/case", ["M02"], checks=evidence())
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("change", ["grade", "operand", "metric", "duplicate", "omitted", "math_category"])
def test_tampering_or_omitting_a_declared_check_is_rejected(tmp_path, change):
    row = context(tmp_path).record("sealed", ["M02"], checks=evidence())
    bad = copy.deepcopy(row)
    if change == "grade":
        bad["checks"][0]["verdict"] = "BAD"
    elif change == "operand":
        bad["checks"][0]["measured"] = 1
    elif change == "metric":
        bad["metrics"]["parameters"] = 100
    elif change == "duplicate":
        bad["checks"].append(copy.deepcopy(bad["checks"][0]))
    else:
        removed = bad["checks"].pop(-1 if change == "math_category" else 1)
        bad["assessment"] = score_checks(bad["checks"])
        bad["gap_assessment"] = score_checks([v for v in bad["checks"] if v["category"] == "gap"])
        bad["math_assessment"] = score_checks([v for v in bad["checks"] if v["category"] == "math"])
        if change == "math_category":
            bad["required_check_ids"].remove(removed["check_id"])
    if change != "metric":
        bad["row_sha256"] = digest({k: v for k, v in bad.items() if k != "row_sha256"})
    with pytest.raises(ValueError):
        validate_row(bad)


def test_nonfinite_value_roundtrip_stays_bad_and_inapplicable_stays_na(tmp_path):
    ctx = context(tmp_path)
    row = ctx.record("nonfinite", ["M02"], checks=[
        check("gap_nan", float("nan"), 1, category="gap"),
        check("math_inapplicable", float("nan"), 1, category="math", applicable=False)])
    roundtrip = json.loads(json.dumps(row, allow_nan=False))
    assert validate_row(roundtrip)["checks"][0]["verdict"] == "BAD"
    assert roundtrip["checks"][1]["verdict"] == "NA"


def test_measure_separates_warmup_and_repeated_complete_calls(tmp_path):
    ctx = context(tmp_path)
    calls = []
    result, cost = ctx.measure(lambda: calls.append(len(calls)) or len(calls), repeats=3, warmup=1)
    assert result == 4 and len(calls) == 4
    assert len(cost["samples_seconds"]) == 3
    assert cost["total_seconds"] == sum(cost["samples_seconds"])
    assert cost["min_seconds"] <= cost["median_seconds"] <= cost["max_seconds"]
    assert cost["peak_allocated_bytes"] is None
    with pytest.raises(ValueError):
        ctx.measure(lambda: 1, repeats=0)


def test_review_csv_has_every_experiment_and_all_missing_failed_checks(tmp_path):
    ctx = context(tmp_path)
    ctx.record("good", ["M02"], checks=evidence(), metrics={"parameters": 0})
    ctx.record("bad", ["M02"], checks=[check("error", 1, 0.1, category="gap")])
    counts = write_reviews(tmp_path, ctx.rows)
    assert counts == {"GOOD": 1, "BAD": 1}
    rows = list(csv.DictReader((tmp_path / "review.csv").open()))
    assert [r["experiment_id"] for r in rows] == ["good", "bad"]
    assert rows[1]["failed_checks"] == "error"
    assert "unmeasured-math" in rows[1]["na_checks"]
    assert json.loads((tmp_path / "rows.json").read_text())["rows"] == ctx.rows


def test_nested_measured_cost_remains_visible_in_condensed_csv(tmp_path):
    ctx = context(tmp_path)
    row = ctx.record("neural-cost", ["M03"], checks=evidence(), metrics={"parameters": 120,
        "cost": {"median_seconds": 0.123, "peak_allocated_bytes": 4096}})
    write_reviews(tmp_path, ctx.rows)
    condensed = next(csv.DictReader((tmp_path / "review.csv").open()))
    assert float(condensed["median_seconds"]) == 0.123
    assert int(condensed["peak_allocated_bytes"]) == 4096
    assert row["metrics"]["cost"]["median_seconds"] == 0.123


def test_unknown_mechanism_or_combination_cannot_be_logged(tmp_path):
    ctx = context(tmp_path)
    with pytest.raises(ValueError, match="declared mechanisms"):
        ctx.record("wrong-id", ["M24"], checks=evidence())
    with pytest.raises(ValueError, match="Undeclared combination"):
        ctx.record("wrong-combo", ["M02"], combination_ids=["C9"], checks=evidence())
