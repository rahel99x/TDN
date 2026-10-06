"""Training-free mechanism evidence stays typed, bounded and source-addressable."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tdn import tower_analytics as analytics
from tdn.reporting import begin_report


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def case(identifier="c0", **changes):
    return {"case_id": identifier, "panel": "temporal", "mechanism": "modal_response",
            "test": "manufactured_forcing", "variant": "phi3", "kind": "correctness",
            "outcome": "PASS", "metrics": {"absolute_error": 1e-12},
            "inputs": {"h": .03, "dtype": "float64"}, "note": "Independent analytic reference.", **changes}


def document(cases=None, **changes):
    return {"schema": "tdn.mechanism-audit/v1", "benchmark_suite": "mechanism-audit",
            "status": "COMPLETED", "device": "cpu", "training_attempted": False,
            "rows": [case()] if cases is None else cases,
            "resources": {"host_process_peak_rss_bytes": 4096, "memory_scope": "process lifetime"},
            "elapsed_seconds": .5, "numerical_budget_seconds": 1200, **changes}


def fixture(tmp_path, value=None):
    source = tmp_path / "science"
    source.mkdir()
    report = Path(begin_report(source, name="tdn/mechanism-audit", script="scripts/research.py",
                               report_parent=tmp_path / "reports"))
    value = document() if value is None else value
    put(source / "mechanism-audit.json", value)
    # These may contain identical rows; only the canonical document projects.
    for name in ("summary.json", "temporal.json", "coordinates.json", "structure.json"):
        put(source / name, value)
    return source, report


def rows(report):
    with (report / "outputs/mechanisms.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def pointer(value, path):
    for field in path.split("/")[1:]:
        field = field.replace("~1", "/").replace("~0", "~")
        value = value[int(field)] if isinstance(value, list) else value[field]
    return value


def test_metrics_keep_exact_identity_types_empty_cases_and_no_duplicates(tmp_path):
    value = document([
        case(metrics={"a/b~c": .0004, "zero": 0, "condition": True, "unresolved": None, "label": ""},
             source_path="forged.json", source_record="/forged", case_record="/wrong"),
        case("empty", kind="representation", outcome="INCONCLUSIVE", metrics={}, note="Budget exhausted."),
        case("observed", kind="scientific", outcome="OBSERVED", metrics={"ratio": 1.2}),
        case("negative", kind="negative_control", outcome="EXPECTED_LIMITATION", metrics={"reachable": False}),
    ])
    source, report = fixture(tmp_path, value)
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob("*.json")}
    result = analytics.publish_outputs(report, [source])
    projected = rows(report)
    assert len(projected) == result["table_rows"]["mechanisms"] == 8
    assert result["reporting_omission_count"] == 0
    assert result["numerical_failure_records"] == 0
    assert result["scientific_outcomes"] == []
    assert result["table_rows"]["training"] == result["table_rows"]["tests"] == 0
    assert result["application_completion_is_scientific_success"] is False
    for record in projected:
        assert record["source_path"].endswith("/mechanism-audit.json")
        original = json.loads((report / record["source_path"]).read_text())
        source_case = pointer(original, record["case_record"])
        assert source_case["case_id"] == record["case_id"]
        assert json.loads(record["inputs"]) == source_case["inputs"]
        assert record["training_attempted"] == "false" and record["device"] == "cpu"
        if record["value_type"] == "absent":
            assert record["metric"] == record["value"] == ""
            assert pointer(original, record["source_record"]) == source_case
        else:
            metric_value = pointer(original, record["source_record"])
            actual = record["value"] if record["value_type"] == "string" else json.loads(record["value"])
            assert actual == metric_value == source_case["metrics"][record["metric"]]
    assert projected[0]["source_record"] == "/rows/0/metrics/a~1b~0c"
    assert projected[3]["value_type"] == "null" and projected[3]["value"] == "null"
    summary, = result["reports"]
    assert summary["kind"] == "mechanism_audit"
    assert summary["valid_case_count"] == summary["source_case_count"] == 4
    assert summary["metric_count"] == 7 and summary["no_metric_case_count"] == 1
    assert summary["outcome_counts"]["EXPECTED_LIMITATION"] == 1
    assert summary["kind_outcome_counts"]["negative_control"]["EXPECTED_LIMITATION"] == 1
    assert summary["training_attempted"] is summary["actual_neural_training"] is False
    assert summary["scientific_gate_authorized"] is False
    assert summary["elapsed_seconds"] == .5 and summary["numerical_budget_seconds"] == 1200
    assert summary["resources"]["host_process_peak_rss_bytes"] == 4096
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes}
    logs = json.loads((report / "logs.json").read_text())["logs"]
    entry, = [entry for entry in logs if entry["path"] == "outputs/mechanisms.csv"]
    assert entry["id"] == "analytics.mechanisms"
    assert all((report / entry["path"]).is_file() for entry in logs)
    # Reconciliation does not create duplicate catalog entries.
    analytics.publish_outputs(report, [source])
    assert json.loads((report / "logs.json").read_text())["logs"] == logs


def test_case_outcomes_are_separate_from_unit_tests_and_training(tmp_path):
    source, report = fixture(tmp_path, document([
        case("failed", outcome="FAIL"),
        case("expected", kind="negative_control", outcome="EXPECTED_LIMITATION"),
        case("probe", kind="representation", outcome="FAIL"),
        case("unknown", kind="scientific", outcome="INCONCLUSIVE", metrics={}),
    ], status="INCOMPLETE"))
    result = analytics.publish_outputs(report, [source])
    assert result["scientific_outcomes"] == ["MECHANISM_AUDIT_INCOMPLETE", "MECHANISM_CORRECTNESS_FAILURE"]
    assert result["numerical_failure_records"] == 0 and "test_outcomes" not in result
    assert not any(key.startswith("neural_") for key in result)
    summary, = result["reports"]
    assert summary["outcome_counts"]["FAIL"] == 2
    assert summary["kind_outcome_counts"]["correctness"]["FAIL"] == 1


def test_case_identity_is_panel_scoped_and_duplicate_within_panel_is_explicit(tmp_path):
    source, report = fixture(tmp_path, document([
        case("shared", panel="temporal", metrics={"error": .001}),
        case("shared", panel="coordinates", metrics={"error": .002}),
        case("shared", panel="coordinates", metrics={"error": .003}),
    ]))
    result = analytics.publish_outputs(report, [source])
    records = rows(report)
    assert [(row["panel"], row["case_id"], row["value"]) for row in records] == [
        ("temporal", "shared", "0.001"), ("coordinates", "shared", "0.002")]
    assert [row["case_record"] for row in records] == ["/rows/0", "/rows/1"]
    assert [row["source_record"] for row in records] == ["/rows/0/metrics/error", "/rows/1/metrics/error"]
    summary, = result["reports"]
    assert summary["source_case_count"] == 3 and summary["valid_case_count"] == 2
    assert summary["unsupported_case_count"] == result["reporting_omission_count"] == 1


def test_failed_negative_control_is_explicit_without_relabeling_expected_limits(tmp_path):
    source, report = fixture(tmp_path, document([
        case("control", kind="negative_control", outcome="FAIL"),
        case("limit", kind="scientific", outcome="EXPECTED_LIMITATION"),
    ], status="FAILED"))
    result = analytics.publish_outputs(report, [source])
    assert result["scientific_outcomes"] == ["MECHANISM_AUDIT_INCOMPLETE", "MECHANISM_NEGATIVE_CONTROL_FAILURE"]
    assert result["numerical_failure_records"] == 0 and "test_outcomes" not in result
    summary, = result["reports"]
    assert summary["kind_outcome_counts"]["negative_control"]["FAIL"] == 1
    assert summary["kind_outcome_counts"]["scientific"]["EXPECTED_LIMITATION"] == 1
    assert summary["outcome_counts"]["FAIL"] == summary["outcome_counts"]["EXPECTED_LIMITATION"] == 1


@pytest.mark.parametrize("changes", [
    {"schema": "tdn.mechanism-audit/v2"}, {"device": "cuda"}, {"training_attempted": True},
    {"training_attempted": 0}, {"status": "SUCCEEDED"}, {"rows": None}, {"benchmark_suite": "neural"},
])
def test_unsupported_canonical_scope_is_explicit(tmp_path, changes):
    source, report = fixture(tmp_path, document(**changes))
    result = analytics.publish_outputs(report, [source])
    assert rows(report) == [] and result["reports"] == []
    assert result["reporting_omission_count"] == 1


def test_missing_invalid_duplicate_case_and_nested_metric_are_explicit(tmp_path):
    missing = case("missing")
    missing.pop("metrics")
    value = document([case(), case(), missing, case("nested", metrics={"error": [1, 2]}),
                      case("badkind", kind="trained_success"), case("valid")])
    source, report = fixture(tmp_path, value)
    result = analytics.publish_outputs(report, [source])
    assert [row["case_id"] for row in rows(report)] == ["c0", "valid"]
    summary, = result["reports"]
    assert summary["valid_case_count"] == 2 and summary["unsupported_case_count"] == 4
    assert result["reporting_omission_count"] == 4
    assert result["scientific_outcomes"] == ["MECHANISM_AUDIT_INCOMPLETE"]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_metrics_cannot_appear_as_valid_science(tmp_path, literal):
    source, report = fixture(tmp_path)
    canonical = source / "mechanism-audit.json"
    canonical.write_text(json.dumps(document()).replace("1e-12", literal))
    result = analytics.publish_outputs(report, [source])
    assert rows(report) == [] and result["reports"] == []
    assert result["reporting_omission_count"] == 1
    omitted = json.loads((report / "outputs/artifacts.json").read_text())["omissions"]
    assert "nonfinite" in omitted[0]["reason"]


def test_metric_rows_use_dedicated_row_budget_and_summary_counts_cases(tmp_path, monkeypatch):
    source, report = fixture(tmp_path, document([case(str(i), metrics={"a": i, "b": i}) for i in range(4)]))
    monkeypatch.setattr(analytics, "MAX_MECHANISM_ROWS", 3)
    result = analytics.publish_outputs(report, [source])
    assert len(rows(report)) == 3 and result["reporting_omission_count"] == 1
    assert result["reports"][0]["valid_case_count"] == 4
    assert result["reports"][0]["metric_count"] == 8
    assert "before table" in result["reports"][0]["counts_scope"]


def test_full_diagnostic_metrics_exceed_ordinary_limits_without_omissions(tmp_path):
    # A full audit emits >10,000 scalar metrics from hundreds of independent
    # cases. Repeated case context makes its CSV larger than the source JSON.
    cases = [case(str(index), metrics={f"metric_{j}": index + j / 100 for j in range(20)},
                  note="Bounded representation diagnostic; source coefficients and conditions are retained. " * 2)
             for index in range(600)]
    source, report = fixture(tmp_path, document(cases))
    assert (source / "mechanism-audit.json").stat().st_size < analytics.MAX_FILE_BYTES
    result = analytics.publish_outputs(report, [source])
    projected = rows(report)
    assert len(projected) == result["table_rows"]["mechanisms"] == 12000
    assert result["reporting_omission_count"] == 0
    size = (report / "outputs/mechanisms.csv").stat().st_size
    assert analytics.MAX_TABLE_BYTES < size < analytics.MAX_MECHANISM_TABLE_BYTES
    for index, record in enumerate(projected):
        case_index, metric_index = divmod(index, 20)
        assert record["case_record"] == f"/rows/{case_index}"
        assert record["source_record"] == f"/rows/{case_index}/metrics/metric_{metric_index}"
        assert float(record["value"]) == cases[case_index]["metrics"][f"metric_{metric_index}"]
    metadata = json.loads((report / "outputs/tables.json").read_text())
    assert metadata["limits"] == {
        "ordinary_tables_shared_rows": 10000, "ordinary_table_bytes": 2 << 20,
        "table_overrides": {"mechanisms": {"rows": 12000, "bytes": 8 << 20}}, "combined_rows": 22000}
    inventory = json.loads((report / "outputs/artifacts.json").read_text())
    assert inventory["limits"]["combined_canonical_rows"] == 22000
    assert inventory["limits"]["table_bytes_overrides"] == {"mechanisms.csv": 8 << 20}


def test_mechanism_reserve_and_ordinary_rows_have_independent_hard_caps(monkeypatch):
    monkeypatch.setattr(analytics, "MAX_MECHANISM_ROWS", 3)
    monkeypatch.setattr(analytics, "MAX_ROWS", 2)
    budget = analytics._Budget()
    projection = analytics._Projection(budget, set())
    projection.consume(Path("mechanism-audit.json"), "../mechanism-audit.json",
                       document([case(str(i)) for i in range(5)]))
    for index in range(4):
        projection.add("frontier", "../frontier.json", f"/rows/{index}", {"family": "split"})
    assert len(projection.tables["mechanisms"]) == 3
    assert len(projection.tables["frontier"]) == 2
    assert sum(map(len, projection.tables.values())) == 5
    assert budget.omission_count == 2
    assert {item["reason"] for item in budget.omissions} == {
        "mechanisms row budget exhausted; originals retained", "canonical row budget exhausted; originals retained"}


def test_mechanism_csv_hard_byte_cap_is_explicit(tmp_path, monkeypatch):
    source, report = fixture(tmp_path, document([case(str(i)) for i in range(8)]))
    monkeypatch.setattr(analytics, "MAX_MECHANISM_TABLE_BYTES", 1024)
    result = analytics.publish_outputs(report, [source])
    assert 0 < len(rows(report)) < 8
    assert (report / "outputs/mechanisms.csv").stat().st_size <= 1024
    assert result["reporting_omission_count"] == 1
    inventory = json.loads((report / "outputs/artifacts.json").read_text())
    assert inventory["omissions"][0]["path"] == "mechanisms.csv"
    assert "table byte budget exhausted" in inventory["omissions"][0]["reason"]
    assert result["reports"][0]["valid_case_count"] == 8


def test_oversize_canonical_artifact_is_unread_and_retained(tmp_path, monkeypatch):
    source, report = fixture(tmp_path)
    canonical = source / "mechanism-audit.json"
    raw = canonical.read_bytes()
    raw += b" " * (analytics.MAX_FILE_BYTES + 1 - len(raw))
    canonical.write_bytes(raw)
    original_read = analytics._Budget.read
    def read(self, path, expected):
        assert path != canonical
        return original_read(self, path, expected)
    monkeypatch.setattr(analytics._Budget, "read", read)
    result = analytics.publish_outputs(report, [source])
    assert rows(report) == []
    inventory = json.loads((report / "outputs/artifacts.json").read_text())
    item, = [item for item in inventory["artifacts"] if item["path"].endswith("/mechanism-audit.json")]
    assert item["hash_status"] == "not_read_budget" and "sha256" not in item
    assert canonical.read_bytes() == raw
    assert result["reports"] == []


def test_mechanisms_contract_is_optional_and_existing_required_outputs_unchanged():
    path = Path(__file__).parents[1] / ".tower/contracts/outputs.v1.json"
    contract = json.loads(path.read_text())
    assert {item["path"] for item in contract["outputs"] if item["required"]} == {
        "run.json", "logs.json", "summary.json", "metrics.jsonl"}
    item, = [item for item in contract["outputs"] if item["path"] == "outputs/mechanisms.csv"]
    assert item["required"] is False and item["columns"] == analytics.COLUMNS["mechanisms"]
    assert item["max_rows"] == analytics.MAX_MECHANISM_ROWS
    assert item["max_bytes"] == analytics.MAX_MECHANISM_TABLE_BYTES
