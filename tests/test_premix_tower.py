"""Canonical premix provenance, failure visibility and Tower output bounds."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tdn import premix_reporting as reporting
from tdn.reporting import begin_report


def put(path, data):
    path.write_text(json.dumps(data, allow_nan=False))


def fixture(tmp_path):
    source = tmp_path / "accuracy"
    source.mkdir()
    report = Path(begin_report(source, name="TDN/premix-test", script="scripts/carc.sh",
                              report_parent=tmp_path / "reports"))
    return source, report


def metrics(**changes):
    return {"schema": "tdn.premix-mechanisms/v1", "panel": "accuracy", "status": "INCOMPLETE",
            "candidate_rows": [], "frontier_rows": [], "reference_rows": [], "parity_rows": [], **changes}


def read_rows(report, name):
    with (report / "outputs" / (name + ".csv")).open(newline="") as handle:
        return list(csv.DictReader(handle))


def seal(source):
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in source.iterdir() if path.is_file() and path.name != "stage.json"}
    put(source / "manifest.json", {"version": 1, "files": files})
    (source / "COMPLETED").write_text(hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() + "\n")


def resolve(root, source_path, pointer):
    value = json.loads((root / source_path).read_text())
    for item in pointer.split("/")[1:]:
        item = item.replace("~1", "/").replace("~0", "~")
        value = value[int(item)] if isinstance(value, list) else value[item]
    return value


def test_numerical_projection_preserves_pairing_uncertainty_and_seal(tmp_path):
    source, report = fixture(tmp_path)
    candidate = {"candidate_id": "c0", "case_id": "adverse/pair", "variant": "selected_output_gl3",
                 "regime": "worst", "pattern": "cutoff_pair", "grid": [64, 64], "batch_size": 2,
                 "status": "VALID", "nsteps": 4, "error_rms": 1e-6, "error_max": 2e-6,
                 "error_upper_rms": 1.1e-6, "error_upper_max_estimate": 2.2e-6,
                 "reference_uncertainty_rms": 1e-7, "reference_uncertainty_max_estimate": 2e-7,
                 "prepared_median_seconds": 0.013, "member_errors": [{"error_rms": 1e-6}],
                 "work_per_rollout": {"pair_product_cells": 32}, "timing_repeats": [0.013, 0.014]}
    put(source / "metrics.json", metrics(status="COMPLETED", candidate_rows=[candidate],
        frontier_rows=[{"frontier_id": "f0", "case_id": "adverse/pair", "norm": "rms", "tolerance": 2e-6,
                        "status": "FEASIBLE", "selected_candidate_id": "c0", "adjusted_error": 1.1e-6,
                        "prepared_seconds": 0.013, "setup_selected_seconds": 0.015}],
        reference_rows=[{"reference_id": "r0", "reference_accepted": True,
                         "member_references": [{"uncertainty_rms": 1e-7}]}],
        parity_rows=[{"parity_id": "p0", "status": "OBSERVED", "passed": None}]))
    put(source / "summary.json", {"status": "COMPLETED", "scientific_outcome": "INCONCLUSIVE"})
    seal(source)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    result = reporting.publish_premix_outputs(report, [source, source])
    assert result["reporting_complete"]
    assert result["application_completion_is_scientific_success"] is False
    assert result["sources"][0]["seal_status"] == "PROJECTED_CANONICAL_FILES_VERIFIED"
    assert result["sources"][0]["scientific_outcome"] == "INCONCLUSIVE"
    row = read_rows(report, "premix_candidates")[0]
    for key in ("error_rms", "error_upper_rms", "prepared_median_seconds"):
        assert float(row[key]) == candidate[key]
    assert json.loads(row["grid"]) == [64, 64]
    assert row["regime"] == "worst"
    assert resolve(report, row["source_path"], row["source_record"]) == candidate
    assert resolve(report, row["source_path"], row["work_record"]) == {"pair_product_cells": 32}
    metadata = json.loads((report / "outputs/premix-tables.json").read_text())
    assert metadata["source_sha256"][row["source_path"]] == hashlib.sha256((source / "metrics.json").read_bytes()).hexdigest()
    assert read_rows(report, "premix_frontiers")[0]["setup_selected_seconds"] == "0.015"
    assert read_rows(report, "premix_checks")[1]["passed"] == "null"
    assert original == {path.name: path.read_bytes() for path in source.iterdir()}


def test_neural_history_has_exact_observation_pointers_and_separate_seed_rows(tmp_path):
    source, report = fixture(tmp_path)
    rows = [{"family": "premix", "seed": seed, "status": "SELECTED_INITIALIZATION", "selected_step": 0,
             "selected_parameters_changed": False, "history": [{"step": 0, "validation_loss": 0.1},
             {"step": 10, "validation_loss": 0.2, "training_loss": 0.15}], "architecture": {"modes": 4}}
            for seed in (101, 102)]
    put(source / "training.json", {"schema": "tdn.premix-neural/v1", "rows": rows})
    put(source / "candidates.json", {"schema": "tdn.premix-neural/v1", "rows": [
        {"parent_id": "shared-parent", "seed": seed, "family": "premix", "regime": "rough",
         "category": "worst", "distribution": "OOD", "rms": 0.1, "max_error": 0.2, "mean_error": -0.01,
         "spatial_rms": 0.09, "upper_rms": 0.11, "upper_max": 0.22, "uncertainty_rms": 0.01,
         "uncertainty_max_bound": 0.02, "timing": {"wall_seconds_median": 0.004, "wall_seconds_raw": [0.004]}}
        for seed in (101, 102)]})
    result = reporting.publish_premix_outputs(report, [source])
    assert result["reporting_complete"]
    assert result["table_rows"]["premix_training"] == 6
    final, history = read_rows(report, "premix_training")[:2]
    assert final["status"] == "SELECTED_INITIALIZATION"
    assert final["selected_parameters_changed"] == "false"
    assert history["seed"] == "101"
    assert resolve(report, history["source_path"], history["source_record"]) == rows[0]["history"][0]
    actual = read_rows(report, "premix_candidates")
    assert [row["seed"] for row in actual] == ["101", "102"]
    assert actual[0]["error_mean"] == "-0.01"  # Numerical negatives are not formula strings.
    assert actual[0]["error_upper_rms"] == "0.11"
    assert actual[0]["wall_seconds_median"] == "0.004"
    metadata = json.loads((report / "outputs/premix-tables.json").read_text())
    assert "not independent evidence" in metadata["interpretation"]


def test_control_characters_and_formula_strings_cannot_add_csv_rows(tmp_path):
    source, report = fixture(tmp_path)
    reason = '=HYPERLINK("example")\nsecond\rthird\tfourth'
    put(source / "metrics.json", metrics(candidate_rows=[{"status": "INVALID", "failure_reason": reason,
                                                          "error_rms": None}]))
    result = reporting.publish_premix_outputs(report, [source])
    assert result["reporting_complete"]
    raw = (report / "outputs/premix_candidates.csv").read_text()
    assert len(raw.splitlines()) == 2
    row = read_rows(report, "premix_candidates")[0]
    assert row["failure_reason"].startswith("'=HYPERLINK")
    assert "\\n" in row["failure_reason"] and "\\t" in row["failure_reason"]
    assert resolve(report, row["source_path"], row["source_record"])["failure_reason"] == reason
    assert row["error_rms"] == "null" and row["error_max"] == ""


@pytest.mark.parametrize("corrupt", ["marker", "canonical", "missing_manifest"])
def test_corrupt_completion_seal_is_visible_and_never_reported_verified(tmp_path, corrupt):
    source, report = fixture(tmp_path)
    put(source / "metrics.json", metrics(status="COMPLETED"))
    seal(source)
    if corrupt == "marker":
        (source / "COMPLETED").write_text("0" * 64)
    elif corrupt == "canonical":
        (source / "metrics.json").write_text((source / "metrics.json").read_text() + " ")
    else:
        (source / "manifest.json").unlink()
    result = reporting.publish_premix_outputs(report, [source])
    assert not result["reporting_complete"]
    assert result["sources"][0]["seal_status"] == "INVALID"


@pytest.mark.parametrize("bad", ['{"schema":"tdn.premix-neural/v1","rows":[{"rms":NaN}]}',
                                 '{"schema":"tdn.premix-neural/v1","rows":[{"rms":1e999}]}',
                                 '{"schema":"tdn.premix-neural/v1","rows":[],"rows":[]}'])
def test_invalid_json_remains_an_explicit_omission(tmp_path, bad):
    source, report = fixture(tmp_path)
    put(source / "metrics.json", metrics())
    (source / "candidates.json").write_text(bad)
    result = reporting.publish_premix_outputs(report, [source])
    assert not result["reporting_complete"]
    assert result["table_rows"]["premix_candidates"] == 0
    inventory = json.loads((report / "outputs/premix-artifacts.json").read_text())
    assert any("candidates.json" in item["path"] for item in inventory["omissions"])


def test_row_and_byte_budgets_explicitly_retain_source_and_mark_incomplete(tmp_path, monkeypatch):
    source, report = fixture(tmp_path)
    put(source / "metrics.json", metrics(candidate_rows=[{"case_id": str(index)} for index in range(5)]))
    monkeypatch.setattr(reporting, "MAX_ROWS", 2)
    result = reporting.publish_premix_outputs(report, [source])
    assert not result["reporting_complete"]
    assert result["table_rows"]["premix_candidates"] == 2
    assert len(json.loads((source / "metrics.json").read_text())["candidate_rows"]) == 5
    header_size = max(len((",".join(columns) + "\n").encode()) for columns in reporting.COLUMNS.values())
    monkeypatch.setattr(reporting, "MAX_TABLE_BYTES", header_size + 8)
    result = reporting.publish_premix_outputs(report, [source])
    assert not result["reporting_complete"] and result["table_rows"]["premix_candidates"] == 0


def test_symlink_sources_are_not_read_and_output_symlinks_fail_closed(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "metrics.json", metrics())
    target = tmp_path / "external.json"
    put(target, {"schema": "tdn.premix-neural/v1", "rows": []})
    (source / "candidates.json").symlink_to(target)
    result = reporting.publish_premix_outputs(report, [source])
    assert not result["reporting_complete"]
    output = report / "outputs/premix_candidates.csv"
    output.unlink()
    output.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        reporting.publish_premix_outputs(report, [source])
    assert json.loads(target.read_text())["schema"] == "tdn.premix-neural/v1"


def test_generic_publisher_hook_reimport_is_idempotent_and_preserves_terminal_state(tmp_path):
    from tdn.tower_analytics import publish_outputs
    from tdn.reporting import finish_report
    source, report = fixture(tmp_path)
    put(source / "metrics.json", metrics(candidate_rows=[{"status": "INVALID", "error_rms": None}]))
    first = publish_outputs(report, [source])
    finish_report(report, state="INTERRUPTED", exit_code=75, results=first)
    second = publish_outputs(report, [source])
    assert first["premix"] == second["premix"]
    assert json.loads((report / "run.json").read_text())["state"] == "INTERRUPTED"
    assert json.loads((report / "summary.json").read_text())["exit_code"] == 75
    logs = json.loads((report / "logs.json").read_text())["logs"]
    assert len([row for row in logs if row["id"] == "analytics.premix_candidates"]) == 1


def test_non_premix_does_not_create_extra_outputs_and_contract_matches(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "summary.json", {"status": "COMPLETED"})
    assert reporting.publish_premix_outputs(report, [source]) is None
    assert not (report / "outputs/premix-results.json").exists()
    contract = json.loads((Path(__file__).resolve().parents[1] / ".tower/contracts/outputs.v1.json").read_text())
    by_path = {item["path"]: item for item in contract["outputs"]}
    for name, columns in reporting.COLUMNS.items():
        item = by_path["outputs/" + name + ".csv"]
        assert item["columns"] == columns and item["required"] is False
        assert item["max_rows"] == reporting.MAX_ROWS and item["max_bytes"] == reporting.MAX_TABLE_BYTES


def test_stale_declared_counts_and_duplicate_candidate_ids_are_explicit(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "metrics.json", metrics(status="COMPLETED", candidate_rows=[{"candidate_id": "same"}] * 2))
    put(source / "summary.json", {"status": "COMPLETED", "coverage": {
        "candidate_rows": {"expected": 1, "reported": 1}}})
    seal(source)
    result = reporting.publish_premix_outputs(report, [source])
    assert not result["reporting_complete"]
    assert result["sources"][0]["seal_status"] == "PROJECTED_CANONICAL_FILES_VERIFIED"
    assert result["sources"][0]["declared_counts_consistent"] is False
    omissions = json.loads((report / "outputs/premix-artifacts.json").read_text())["omissions"]
    assert any("coverage" in item["reason"] for item in omissions)
    assert any("duplicate" in item["reason"] for item in omissions)


def test_neural_summary_cannot_hide_missing_canonical_table(tmp_path):
    source, report = fixture(tmp_path)
    put(source / "summary.json", {"schema": "tdn.premix-neural/v1", "stage": "neural", "status": "COMPLETED"})
    seal(source)
    result = reporting.publish_premix_outputs(report, [source])
    assert not result["reporting_complete"]
    omissions = json.loads((report / "outputs/premix-artifacts.json").read_text())["omissions"]
    assert len([item for item in omissions if "required canonical table" in item["reason"]]) == 3


def test_dedicated_canonical_parser_does_not_consume_ordinary_json_budget(tmp_path):
    from tdn.tower_analytics import publish_outputs
    source, report = fixture(tmp_path)
    # This real canonical document exceeds the ordinary 100,000-value limit,
    # while staying within the separately declared premix allowance.
    put(source / "metrics.json", metrics(raw_numerical_diagnostics=list(range(100_010)),
                                        candidate_rows=[{"case_id": "full-source"}]))
    result = publish_outputs(report, [source])
    assert result["reporting_omission_count"] == 0
    assert result["premix"]["reporting_complete"] is True
    inventory = json.loads((report / "outputs/artifacts.json").read_text())
    item = next(row for row in inventory["artifacts"] if row["path"].endswith("/metrics.json"))
    assert item["hash_status"] == "computed_premix_projection"
    assert item["sha256"] == hashlib.sha256((source / "metrics.json").read_bytes()).hexdigest()
    assert inventory["observed_read_bytes"] == 0


def test_comparison_and_regime_groups_preserve_untrained_selection_and_parent_links(tmp_path):
    source, report = fixture(tmp_path)
    group_counts = {key: 0 for key in reporting.GROUP_COUNTS}
    group_counts.update(total=1, ours_feasible=1, baseline_feasible=1, both_feasible=1,
                        timing_eligible=1, initialization_in_eligible_pair=1, ours_wins=1)
    comparison = {"comparison_id": "p:1:premix:fno:rms:0", "parent_id": "p", "seed": 1,
        "ours": "premix", "baseline": "fno", "ours_selection": "SELECTED_INITIALIZATION",
        "baseline_selection": "TRAINED_CHECKPOINT", "timing_eligible": True, "trained_pair_eligible": False,
        "outcome": "WIN", "speedup_baseline_over_ours": 2., "ours_training_record": "/rows/0",
        "issues": []}
    put(source / "comparisons.json", {"schema": "tdn.premix-comparisons/v1", "rows": [comparison],
        "coverage": {"comparisons": {"expected": 1, "reported": 1}}, "counts": group_counts,
        "groups": [{"group_scope": "category", "group": "adverse", "seed": 1, "ours": "premix", "baseline": "fno",
                    "norm": "rms", "target": 0.002, "counts": group_counts, "parent_ids": ["p"]}]})
    result = reporting.publish_premix_outputs(report, [source])
    assert result["reporting_complete"]
    row = read_rows(report, "premix_comparisons")[0]
    assert row["trained_pair_eligible"] == "false"
    assert row["ours_training_record"] == "/rows/0"
    assert row["speedup_baseline_over_ours"] == "2.0"
    group = read_rows(report, "premix_groups")[0]
    assert group["group"] == "adverse"
    assert group["count_initialization_in_eligible_pair"] == "1"
    assert group["count_trained_pair_wins"] == "0"
    assert resolve(report, group["source_path"], group["parent_ids_record"]) == ["p"]
