"""Interaction-screen analytics retain scientific provenance in Tower tables."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tdn.reporting import begin_report
from tdn.tower_analytics import publish_outputs


def make_report(tmp_path, *, status="COMPLETED", outcome="OBSERVED", **changes):
    source = tmp_path / "experiment"
    source.mkdir()
    payload = {
        "schema": "tdn.interaction-screen/v1", "benchmark_suite": "interaction-screen",
        "status": status, "device": "cpu", "training_attempted": False,
        "elapsed_seconds": 1.5, "numerical_budget_seconds": 1200,
        "rows": [{"case_id": "mixed-0/gl3", "panel": "finite_amplitude",
                  "mechanism": "runtime_interaction", "test": "one_step", "variant": "gl3",
                  "kind": "scientific", "outcome": outcome,
                  "metrics": {"rms/error": .002, "admissible": True, "speedup": None},
                  "inputs": {"h": .1, "grid": [16]},
                  "note": "Same-grid reference; no training or GPU work."}],
        **changes,
    }
    canonical = source / "interaction-screen.json"
    canonical.write_text(json.dumps(payload, allow_nan=False))
    # Summary copies must not project the same cases a second time.
    (source / "summary.json").write_text(json.dumps(payload, allow_nan=False))
    report = Path(begin_report(source, name="tdn/interaction-screen",
                               script="scripts/interaction_screen.py",
                               report_parent=tmp_path / "reports"))
    return source, report, canonical


def test_interaction_metrics_preserve_types_pointers_and_originals(tmp_path):
    source, report, canonical = make_report(tmp_path)
    before = hashlib.sha256(canonical.read_bytes()).hexdigest()
    result = publish_outputs(report, [source])
    with (report / "outputs/mechanisms.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["table_rows"]["mechanisms"] == 3
    assert result["reporting_omission_count"] == 0
    assert result["table_rows"]["training"] == 0
    assert result["scientific_outcomes"] == []
    summary, = result["reports"]
    assert summary["kind"] == "interaction_screen"
    assert summary["benchmark_suite"] == "interaction-screen"
    assert summary["scientific_gate_authorized"] is False
    assert summary["actual_neural_training"] is False
    assert rows[0]["source_record"] == "/rows/0/metrics/rms~1error"
    assert rows[1]["value_type"] == "boolean" and rows[1]["value"] == "true"
    assert rows[2]["value_type"] == "null" and rows[2]["value"] == "null"
    assert rows[0]["note"] == "Same-grid reference; no training or GPU work."
    assert rows[1]["note"] == rows[2]["note"] == ""
    for row in rows:
        original = json.loads((report / row["source_path"]).read_text())
        assert original["rows"][0]["case_id"] == row["case_id"]
        assert row["source_path"].endswith("/interaction-screen.json")
        assert row["case_record"] == "/rows/0"
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == before
    logs = json.loads((report / "logs.json").read_text())["logs"]
    assert sum(row["id"] == "analytics.mechanisms" for row in logs) == 1


def test_inconclusive_interaction_is_not_a_completed_scientific_result(tmp_path):
    source, report, _ = make_report(tmp_path, status="INCOMPLETE", outcome="INCONCLUSIVE")
    result = publish_outputs(report, [source])
    assert result["scientific_outcomes"] == ["INTERACTION_SCREEN_INCOMPLETE"]
    assert result["reports"][0]["outcome_counts"]["INCONCLUSIVE"] == 1
    assert result["numerical_failure_records"] == 0


@pytest.mark.parametrize("changes", [
    {"schema": "tdn.mechanism-audit/v1"}, {"benchmark_suite": "mechanism-audit"},
    {"device": "cuda"}, {"training_attempted": True},
])
def test_mismatched_interaction_scope_is_not_silently_projected(tmp_path, changes):
    source, report, _ = make_report(tmp_path, **changes)
    result = publish_outputs(report, [source])
    assert result["table_rows"]["mechanisms"] == 0
    assert result["reports"] == []
    assert result["reporting_omission_count"] == 1
