"""Consistency evidence remains source-linked, bounded and Tower-compatible."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tdn import consistency_reporting as reporting
from tdn.premix_reporting import GROUP_COUNTS
from tdn.reporting import begin_report
from tdn.tower_analytics import publish_outputs


def put(path, data):
    path.write_text(json.dumps(data, allow_nan=False))


def fixture(tmp_path):
    source = tmp_path / "neural"
    source.mkdir()
    report = Path(begin_report(source, name="TDN/consistency-test", script="scripts/fedora_consistency.sh",
                              report_parent=tmp_path / "reports"))
    return source, report


def rows(report, name):
    with (report / "outputs" / (name + ".csv")).open(newline="") as handle:
        return list(csv.DictReader(handle))


def seal(source):
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in source.iterdir() if path.is_file() and path.name != "stage.json"}
    put(source / "manifest.json", {"version": 1, "files": files})
    (source / "COMPLETED").write_text(hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() + "\n")


def resolve(report, source, pointer):
    value = json.loads((report / source).read_text())
    for item in pointer.split("/")[1:]:
        item = item.replace("~1", "/").replace("~0", "~")
        value = value[int(item)] if isinstance(value, list) else value[item]
    return value


def test_all_consistency_tables_keep_seals_diagnostics_and_exact_source_pointers(tmp_path):
    source, report = fixture(tmp_path)
    candidate = {"parent_id": "fresh/adverse", "family": "premix_moment", "seed": 11, "grid": [16, 16],
        "category": "adverse", "regime": "near_zero", "status": "COMPLETED", "rms": .001,
        "signed_mean_error": -.0009, "mean_error": .0009, "spatial_rms": .0004,
        "base_rms": .002, "base_max_error": .004, "base_mean_error": .0018, "base_spatial_rms": .0008,
        "rms_vs_base": .5, "max_vs_base": .5, "mean_vs_base": .5, "spatial_vs_base": .5,
        "checkpoint_selection": "SELECTED_INITIALIZATION", "timing": {"wall_seconds_median": .01}}
    put(source / "candidates.json", {"schema": "tdn.consistency-neural/v1", "rows": [candidate]})
    put(source / "frontiers.json", {"schema": "tdn.consistency-neural/v1", "rows": [{"family": "premix_moment", "status": "FEASIBLE", "norm": "rms", "target": .002}]})
    put(source / "training.json", {"schema": "tdn.consistency-neural/v1", "rows": [
        {"family": "premix_moment", "seed": 11, "status": "COMPLETED", "selection": "SELECTED_INITIALIZATION",
         "selected_step": 0, "selected_parameters_changed": False, "history": [{"step": 0, "validation_loss": .01}]}]})
    check = {"family": "premix_moment", "seed": 11, "case": "constant", "status": "FAIL", "required": True, "max_base_difference": .1}
    put(source / "consistency.json", {"schema": "tdn.consistency-neural/v1", "rows": [check]})
    count = {key: 0 for key in GROUP_COUNTS}; count["total"] = 1
    comparison = {"comparison_id": "fresh:11:premix_moment:fno_moment:rms:0", "comparison_kind": "matched_moment",
        "ours": "premix_moment", "baseline": "fno_moment", "seed": 11, "ours_selection": "SELECTED_INITIALIZATION",
        "trained_pair_eligible": False, "ours_base_rms": .002, "ours_signed_mean_error": -.0009,
        "ours_rms_regresses_base": False, "ours_diagnostic_issues": ["INVALID_BASE_MEAN_ERROR"]}
    put(source / "comparisons.json", {"schema": "tdn.consistency-comparisons/v1", "coverage": {"comparisons": {"expected": 1, "reported": 1}},
        "counts": {"total": 1}, "rows": [comparison], "groups": [{"group_scope": "category", "group": "adverse", "comparison_kind": "matched_moment", "counts": count}]})
    put(source / "summary.json", {"schema": "tdn.consistency-neural/v1", "stage": "neural", "status": "COMPLETED", "scientific_outcome": "INCONCLUSIVE"})
    seal(source)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    result = reporting.publish_consistency_outputs(report, [source, source])
    assert result["reporting_complete"]
    assert not result["application_completion_is_scientific_success"]
    assert result["sources"][0]["seal_status"] == "PROJECTED_CANONICAL_FILES_VERIFIED"
    assert result["table_rows"]["consistency_training"] == 2
    row = rows(report, "consistency_candidates")[0]
    assert float(row["signed_mean_error"]) == candidate["signed_mean_error"]
    assert float(row["base_rms"]) == candidate["base_rms"]
    assert row["checkpoint_selection"] == "SELECTED_INITIALIZATION"
    assert resolve(report, row["source_path"], row["source_record"]) == candidate
    check_row = rows(report, "consistency_checks")[0]
    assert check_row["status"] == "FAIL" and check_row["required"] == "true"
    assert resolve(report, check_row["source_path"], check_row["source_record"]) == check
    pair = rows(report, "consistency_comparisons")[0]
    assert pair["comparison_kind"] == "matched_moment" and pair["trained_pair_eligible"] == "false"
    assert resolve(report, pair["source_path"], pair["ours_diagnostic_issues_record"]) == ["INVALID_BASE_MEAN_ERROR"]
    metadata = json.loads((report / "outputs/consistency-tables.json").read_text())
    assert metadata["source_sha256"][row["source_path"]] == hashlib.sha256((source / "candidates.json").read_bytes()).hexdigest()
    assert original == {path.name: path.read_bytes() for path in source.iterdir()}
    assert not (report / "outputs/premix-results.json").exists()


def test_structural_audit_keeps_array_checks_as_exact_pointers(tmp_path):
    source, report = fixture(tmp_path)
    check = {"check_id": "moment-final", "status": "PASS", "required": True, "error": 1e-8, "tolerance": 1e-6,
             "target_mean": [.1, .2], "final_mean": [.1, .2], "observed_orders": [2.9, 3.0],
             "mean_law_error": 1e-16, "physical_mean_derivative": [.1, .2], "mean_state": [.1, .2], "variance": [.01, .02],
             "missing_parameter_gradients": [], "nonfinite_parameter_gradients": []}
    put(source / "checks.json", {"schema": "tdn.consistency-neural/v1", "rows": [check]})
    put(source / "summary.json", {"schema": "tdn.consistency-neural/v1", "stage": "audit", "status": "INCOMPLETE"})
    result = reporting.publish_consistency_outputs(report, [source])
    assert result["reporting_complete"]
    row = rows(report, "consistency_checks")[0]
    assert row["check_id"] == "moment-final"
    for field in ("target_mean", "final_mean", "observed_orders", "missing_parameter_gradients",
                  "physical_mean_derivative", "mean_state", "variance"):
        assert resolve(report, row["source_path"], row[field + "_record"]) == check[field]
    assert row["target_mean"] == ""


def test_general_analytics_delegates_hashes_and_imports_explicit_junit_directory(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "summary.json", {"schema": "tdn.consistency-neural/v1", "stage": "neural", "status": "INCOMPLETE"})
    put(source / "candidates.json", {"schema": "tdn.consistency-neural/v1", "rows": []})
    test_source = tmp_path / "reporter-tests" / "neural"
    test_source.mkdir(parents=True)
    test_xml = test_source / "neural-tests.xml"
    test_xml.write_text('<testsuites><testsuite name="gpu"><testcase name="actual"><failure message="bad"/></testcase><testcase name="passed"/></testsuite></testsuites>')
    (tmp_path / "unrelated-secret.json").write_text('{"private":true}')
    result = publish_outputs(report, [source, test_source])
    assert result["consistency"]["reporting_complete"]
    assert result["test_outcomes"] == {"FAILED": 1, "PASSED": 1, "ERROR": 0, "SKIPPED": 0}
    assert len(rows(report, "tests")) == 2
    inventory = json.loads((report / "outputs/artifacts.json").read_text())["artifacts"]
    entry = next(row for row in inventory if row["path"].endswith("candidates.json"))
    assert entry["hash_status"] == "computed_consistency_projection"
    assert entry["projection"] == "outputs/consistency-tables.json"
    assert not any("unrelated-secret" in row["path"] for row in inventory)
    xml = next(row for row in inventory if row["path"].endswith("neural-tests.xml"))
    assert xml["sha256"] == hashlib.sha256(test_xml.read_bytes()).hexdigest()


@pytest.mark.parametrize("bad", ['{"schema":"tdn.consistency-neural/v1","rows":[{"rms":NaN}]}',
                                 '{"schema":"tdn.consistency-neural/v1","rows":[],"rows":[]}'])
def test_corrupt_canonical_rows_fail_reporting_explicitly(tmp_path, bad):
    source, report = fixture(tmp_path)
    put(source / "summary.json", {"schema": "tdn.consistency-neural/v1", "stage": "neural", "status": "INCOMPLETE"})
    (source / "candidates.json").write_text(bad)
    result = reporting.publish_consistency_outputs(report, [source])
    assert not result["reporting_complete"]
    assert result["table_rows"]["consistency_candidates"] == 0
    omissions = json.loads((report / "outputs/consistency-artifacts.json").read_text())["omissions"]
    assert any("candidates.json" in row["path"] for row in omissions)


def test_complete_source_requires_canonical_tables_and_intact_seal(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "summary.json", {"schema": "tdn.consistency-neural/v1", "stage": "neural", "status": "COMPLETED"})
    result = reporting.publish_consistency_outputs(report, [source])
    assert not result["reporting_complete"]
    for name in ("training", "candidates", "frontiers"):
        put(source / (name + ".json"), {"schema": "tdn.consistency-neural/v1", "rows": []})
    put(source / "consistency.json", {"schema": "tdn.consistency-neural/v1", "rows": []})
    put(source / "comparisons.json", {"schema": "tdn.consistency-comparisons/v1", "rows": [], "groups": [],
        "counts": {"total": 0}, "coverage": {"comparisons": {"expected": 0, "reported": 0}}})
    seal(source)
    assert reporting.publish_consistency_outputs(report, [source])["reporting_complete"]
    put(source / "candidates.json", {"schema": "tdn.consistency-neural/v1", "rows": [{"rms": 1.0}]})
    assert not reporting.publish_consistency_outputs(report, [source])["reporting_complete"]


def test_report_files_match_contract_and_repeated_export_deduplicates_logs(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "summary.json", {"schema": "tdn.consistency-neural/v1", "stage": "neural", "status": "INCOMPLETE"})
    first = reporting.publish_consistency_outputs(report, [source])
    assert reporting.publish_consistency_outputs(report, [source]) == first
    contract = json.loads((Path(__file__).resolve().parents[1] / ".tower/contracts/outputs.v1.json").read_text())
    entries = {entry["path"]: entry for entry in contract["outputs"]}
    for name, expected in reporting.CONSISTENCY_COLUMNS.items():
        with (report / "outputs" / (name + ".csv")).open(newline="") as handle:
            actual = next(csv.reader(handle))
        assert entries[f"outputs/{name}.csv"]["columns"] == actual == expected
    logs = json.loads((report / "logs.json").read_text())["logs"]
    assert len([row for row in logs if row["id"].startswith("analytics.consistency_")]) == 6


def test_unrelated_source_does_not_create_consistency_outputs(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "summary.json", {"schema": "other/v1", "status": "COMPLETED"})
    assert reporting.publish_consistency_outputs(report, [source]) is None
    assert not (report / "outputs/consistency-results.json").exists()


def test_structural_audit_summary_cannot_hide_check_failures(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "checks.json", {"schema": "tdn.consistency-neural/v1", "rows": [
        {"check_id": "gate-constant", "status": "FAILED", "required": True}]})
    put(source / "summary.json", {"schema": "tdn.consistency-neural/v1", "stage": "audit", "status": "FAILED",
        "case_count": 1, "coverage": {"expected": 1, "reported": 1}, "outcomes": {"PASS": 1}, "correctness_failures": 0})
    result = reporting.publish_consistency_outputs(report, [source])
    assert not result["reporting_complete"]
    assert rows(report, "consistency_checks")[0]["status"] == "FAILED"
    omission = json.loads((report / "outputs/consistency-artifacts.json").read_text())["omissions"]
    assert any("outcomes" in row["reason"] for row in omission)
    assert any("correctness" in row["reason"] for row in omission)


def test_native_helper_imports_existing_package_instead_of_sibling_tower_script(tmp_path):
    import os
    import subprocess
    import sys
    installed = tmp_path / "installed/tower"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    (installed / "artifacts.py").write_text(
        "def load_contract(path):\n    return {'version':1,'outputs':[]}\n"
        "def validate_contract(contract,root,*,max_bytes):\n"
        "    return {'valid':True,'status':'valid','budget':{'max_bytes':max_bytes}}\n")
    helper = Path(__file__).resolve().parents[1] / "scripts/tower_native_validate.py"
    environment = {**os.environ, "PYTHONPATH": str(installed.parent), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, str(helper), "contract.json", str(tmp_path),
        "--max-bytes", str(32 << 20)], env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    decoded = json.loads(result.stdout)
    assert decoded["valid"] and decoded["budget"]["max_bytes"] == 32 << 20
    assert not (installed / "__pycache__").exists()
    rejected = subprocess.run([sys.executable, str(helper), "contract.json", str(tmp_path),
        "--max-bytes", str(64 << 20)], env=environment, text=True, capture_output=True)
    assert rejected.returncode == 2
