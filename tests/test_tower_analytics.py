"""Actual source projection, provenance preservation, failures and bounded I/O."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tdn import tower_analytics as analytics
from tdn.reporting import begin_report
from tdn.research.neural_comparison import summarize_neural_comparisons


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def report(tmp_path):
    source = tmp_path / "science"
    source.mkdir()
    result = begin_report(source, name="TDN/analytics-test", script="scripts/carc.sh",
                          report_parent=tmp_path / "reports")
    return source, Path(result)


def rows(report_dir, name):
    with (report_dir / "outputs" / (name + ".csv")).open(newline="") as stream:
        return list(csv.DictReader(stream))


def inventory(report_dir):
    return json.loads((report_dir / "outputs/artifacts.json").read_text())


def test_research_failures_nulls_and_hashes_are_preserved(tmp_path):
    source, target = report(tmp_path)
    failed = {"family": "transport", "status": "NUMERICAL_FAILURE", "steps": 42, "selected_step": None,
              "error": "Nonfinite neural gradient", "history": [{"step": 0, "validation_loss": None,
              "admissible_validation": False}]}
    put(source / "training/transport.json", failed)
    put(source / "summary.json", {"status": "COMPLETED", "device": "cpu", "headroom_passed": False,
        "headroom": {"passed": False}, "training": [failed], "comparisons": [
            {"eligible": True, "speedup": 0.5, "twenty_percent_faster": False}]})
    put(source / "frontier.json", {"rows": [{"family": "transport", "status": "NUMERICAL_FAILURE",
        "error": None, "feasible": False, "reason": "invalid state", "timing": None},
        {"family": "split", "status": "COMPLETED", "error": 0.0, "feasible": True,
         "timing": {"wall_seconds_median": 0.001, "peak_reserved_bytes": None}}]})
    put(source / "heldout.json", {"rows": [{"family": "transport", "status": "NUMERICAL_FAILURE",
        "reason": "invalid intermediate", "one_step": {"rms": None, "upper": None}}]})
    put(source / "references.json", {"records": [{"parent_id": "p", "accepted": False,
        "uncertainty": None, "reason": "reference budget exhausted", "refinement_substeps": [8, 16, 32]}]})
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*") if path.is_file()}
    result = analytics.publish_outputs(target, [source])
    assert result["scientific_outcomes"] == ["NO_NUMERICAL_HEADROOM", "NO_SPEED_ADVANTAGE_OBSERVED", "NUMERICAL_FAILURE"]
    assert result["application_completion_is_scientific_success"] is False
    assert result["numerical_failure_records"] == 1
    assert rows(target, "training")[0]["selected_step"] == "null"
    assert rows(target, "training")[1]["admissible_validation"] == "false"
    assert rows(target, "frontier")[0]["error"] == "null"
    assert rows(target, "frontier")[1]["error"] == "0.0"
    assert rows(target, "frontier")[0]["wall_seconds_median"] == ""
    assert rows(target, "references")[0]["refinement_substeps"] == "[8,16,32]"
    assert rows(target, "heldout")[0]["one_step_rms"] == "null"
    assert source_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes}
    assert len(rows(target, "training")) == 2  # summary's embedded history is not duplicated


def test_inventory_covers_nested_binary_and_text_without_loading_large_checkpoint(tmp_path, monkeypatch):
    source, target = report(tmp_path)
    contents = {"tables/raw.csv": b"error,status\n,FAILED\n", "metrics.jsonl": b'{"loss":null}\n',
                "plots/a.png": b"PNG", "plots/a.pdf": b"PDF", "dataset/arrays.npz": b"NPZ",
                "stderr.log": b"failure retained\n"}
    for name, content in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    binary = source / "checkpoints/model.pt"
    binary.parent.mkdir()
    with binary.open("wb") as handle:
        handle.truncate(analytics.MAX_FILE_BYTES + 1)
    declared = "a" * 64
    put(source / "manifest.json", {"files": {"checkpoints/model.pt": declared}})
    real_read = analytics._Budget.read
    def read(self, path, expected):
        assert path != binary, "large checkpoint must never be read for analytics"
        return real_read(self, path, expected)
    monkeypatch.setattr(analytics._Budget, "read", read)
    analytics.publish_outputs(target, [source, source / "plots", source])
    entries = inventory(target)["artifacts"]
    checkpoint = next(row for row in entries if row["path"].endswith("model.pt"))
    assert checkpoint["kind"] == "checkpoint"
    assert checkpoint["sha256"] == declared
    assert checkpoint["hash_status"] == "declared_unverified"
    assert len(entries) == len(contents) + 2
    assert {row["kind"] for row in entries} >= {"csv", "jsonl", "image", "pdf", "array_archive", "log", "checkpoint"}
    assert not list((target / "outputs").glob("*.pt"))
    index = json.loads((target / "logs.json").read_text())
    assert any(row["path"].endswith("stderr.log") for row in index["logs"])
    assert not any(row["path"].endswith((".pt", ".png", ".npz", ".pdf")) for row in index["logs"])


def test_light_screen_and_legacy_artifacts_are_projected(tmp_path):
    source, target = report(tmp_path)
    put(source / "light/summary.json", {"stage": "light-screen", "status": "COMPLETED", "headroom_passed_cases": 0,
        "decision": "no_joint_candidate", "pilot_authorized": False, "cases": [{"case_id": "case-a",
        "local_references": [{"accepted": True, "h": 0.02}], "global_reference": {"accepted": False},
        "classical_methods": [{"method": "split", "failed": False, "error": 0.001}],
        "numerical_headroom_screen": {"passed": False}, "temporal_representation_screen": {"passed": True},
        "temporal_oracle": {"models": {"fixed_rate": {"heldout_absolute_rms": [0.1, None]}}}}]})
    put(source / "train/training_result.json", {"status": "PAUSED_NEEDS_RESUME", "global_step": 8,
        "history": [{"step": 8, "loss": 0.0}]})
    put(source / "audit/gates.json", {"G2": {"passed": False, "reason": "no headroom"}})
    put(source / "train/stage.json", {"stage": "train", "status": "PAUSED_NEEDS_RESUME", "elapsed_seconds": 10})
    put(source / "evaluate/evaluation.json", {"scope": "development", "diagnostic_outcomes": {
        "learned": [{"failed": True, "error": None, "failure_reason": "invalid state", "candidate_h": 0.1}]}})
    result = analytics.publish_outputs(target, [source])
    assert "NO_NUMERICAL_HEADROOM" in result["scientific_outcomes"]
    assert any(row["status"] == "PAUSED_NEEDS_RESUME" for row in rows(target, "stages"))
    assert any(row["gate"] == "G2" and row["passed"] == "false" for row in rows(target, "gates"))
    assert len(rows(target, "frontier")) == 2
    assert rows(target, "heldout")[0]["heldout_absolute_rms"] == "[0.1,null]"
    assert any(row["error"] == "null" and row["failed"] == "true" for row in rows(target, "frontier"))


def test_symlinks_caches_and_tower_subtrees_are_not_ingested(tmp_path):
    source, target = report(tmp_path)
    for child in ["tower/previous", "tower-new/nested", ".cache", "pytest-work"]:
        put(source / child / "secret.json", {"do_not_read": True})
    outside = tmp_path / "outside"
    outside.mkdir()
    put(outside / "outside.json", {"not_a_source": True})
    (source / "symlink-dir").symlink_to(outside, target_is_directory=True)
    (source / "symlink.json").symlink_to(outside / "outside.json")
    put(source / "safe.json", {"safe": True})
    analytics.publish_outputs(target, [source])
    observed = inventory(target)
    assert len(observed["artifacts"]) == 1
    assert observed["artifacts"][0]["path"].endswith("safe.json")
    assert observed["omission_count"] == 6
    with pytest.raises(ValueError, match="source cannot be"):
        analytics.publish_outputs(target, [target])


def test_malformed_json_and_manifest_mismatch_do_not_become_success(tmp_path):
    source, target = report(tmp_path)
    (source / "summary.json").write_text('{"status":"COMPLETED","status":"FAILED"}')
    (source / "invalid.json").write_text('{"loss":1e999}')
    (source / "results.json").write_text('{"status":"FAILED"}')
    put(source / "manifest.json", {"files": {"results.json": "f" * 64}})
    result = analytics.publish_outputs(target, [source])
    artifacts = inventory(target)
    assert result["reports"] == []
    assert artifacts["omission_count"] == 3
    entry = next(row for row in artifacts["artifacts"] if row["path"].endswith("results.json"))
    assert entry["hash_status"] == "computed_manifest_mismatch"


def test_budgets_are_explicit_and_zero_science_is_unknown(tmp_path, monkeypatch):
    source, target = report(tmp_path)
    for i in range(5):
        put(source / f"data-{i}.json", {"i": i})
    monkeypatch.setattr(analytics, "MAX_FILES", 2)
    result = analytics.publish_outputs(target, [source])
    assert result["artifact_count"] == 2
    assert result["reporting_omission_count"] >= 1
    assert result["scientific_outcomes"] == []
    assert result["table_rows"]["training"] == 0


def test_junit_retains_skips_errors_failures_and_rejects_entities(tmp_path):
    source, target = report(tmp_path)
    (source / "pytest-results.xml").write_text('''<testsuites><testsuite name="pytest">
      <testcase name="okay" time=".12"/><testcase name="bad"><failure message="bad gradient"/></testcase>
      <testcase name="gpu"><skipped message="no allocation"/></testcase>
      <testcase name="setup"><error message="fixture"/></testcase></testsuite></testsuites>''')
    (source / "unsafe.xml").write_text('<!DOCTYPE foo [<!ENTITY a "payload">]><testsuite/>')
    result = analytics.publish_outputs(target, [source])
    assert result["test_outcomes"] == {"PASSED": 1, "FAILED": 1, "ERROR": 1, "SKIPPED": 1}
    assert len(rows(target, "tests")) == 4
    assert result["reporting_omission_count"] == 1


def test_repeated_publication_deduplicates_logs_and_does_not_change_sources(tmp_path):
    source, target = report(tmp_path)
    put(source / "summary.json", {"stage": "light-screen", "status": "COMPLETED", "cases": []})
    first = analytics.publish_outputs(target, [source])
    second = analytics.publish_outputs(target, [source])
    assert first == second
    index = json.loads((target / "logs.json").read_text())
    assert len(index["logs"]) == 1


def neural_fixture():
    """Use the scientific comparator with missing, initialized and failed models."""
    frontier, heldout = [], []
    families = ["reaction_clock", "residual_cnn", "unet", "fno"]
    training = [
        {"family": "reaction_clock", "status": "COMPLETED", "steps": 8, "selected_step": 5},
        {"family": "residual_cnn", "status": "COMPLETED", "steps": 8, "selected_step": 3},
        {"family": "unet", "status": "COMPLETED", "steps": 8, "selected_step": 0,
         "selected_parameters_changed": False},
        {"family": "fno", "status": "NUMERICAL_FAILURE", "steps": 2, "selected_step": None,
         "error": "Nonfinite neural gradient"},
    ]
    for parent in ["p1", "p2"]:
        for h in [.1, .2]:
            for family, wall, error in [("reaction_clock", .5, .1), ("residual_cnn", 1., .2), ("unet", .25, .05)]:
                if parent == "p2" and family == "residual_cnn":
                    continue  # absence must not be treated as a victory
                frontier.append({"parent_id": parent, "family": family, "h": h, "status": "COMPLETED",
                    "error": error, "error_upper": error + .001, "feasible": True, "device": "cpu",
                    "timing": {"wall_seconds_median": wall}})
                heldout.append({"parent_id": parent, "family": family, "h": h, "status": "COMPLETED",
                    "one_step": {"rms": error, "upper": error + .001},
                    "two_step": {"rms": 2 * error, "upper": 2 * error + .001}})
            frontier.append({"parent_id": parent, "family": "fno", "h": h, "status": "NUMERICAL_FAILURE",
                "error": None, "feasible": False, "device": "cpu", "timing": None,
                "error_message": "Nonfinite neural gradient"})
            heldout.append({"parent_id": parent, "family": "fno", "h": h, "status": "NUMERICAL_FAILURE",
                            "one_step": {"rms": None, "upper": None}})
    return summarize_neural_comparisons(frontier, heldout, families, training)


def test_neural_tables_preserve_missing_initialization_and_failure_selections(tmp_path):
    source, target = report(tmp_path)
    document = neural_fixture()
    artifact = source / "experiment/neural-comparisons.json"
    put(artifact, document)
    before = artifact.read_bytes()
    put(source / "experiment/summary.json", {"status": "COMPLETED", "comparisons": [],
                                           "headroom_passed": False, "headroom": {"passed": False}})
    result = analytics.publish_outputs(target, [source])
    comparisons = rows(target, "neural_comparisons")
    assert len(comparisons) == 6  # declared pairs include absent/failed observations
    trained = next(row for row in comparisons if row["baseline_family"] == "residual_cnn" and row["parent_id"] == "p1")
    assert trained["eligible"] == "true"
    assert trained["speedup"] == "2.0"
    assert trained["speed_outcome"] == "WIN"
    assert trained["tdn_training_selection"] == trained["baseline_training_selection"] == "TRAINED_CHECKPOINT"
    assert trained["tdn_selected_step"] == "5" and trained["baseline_selected_step"] == "3"
    assert trained["baseline_kind"] == "direct_neural"
    absent = next(row for row in comparisons if row["baseline_family"] == "residual_cnn" and row["parent_id"] == "p2")
    assert absent["eligible"] == "false"
    assert absent["baseline_h"] == absent["speedup"] == "null"
    assert absent["baseline_missing_h_count"] == "2"
    assert json.loads(absent["ineligible_reasons"]) == {"baseline": "MISSING_RESULTS"}
    initialized = next(row for row in comparisons if row["baseline_family"] == "unet")
    assert initialized["baseline_training_selection"] == "SELECTED_INITIALIZATION"
    assert initialized["baseline_selected_step"] == "0"
    assert initialized["speed_outcome"] == "LOSS"
    failed = next(row for row in comparisons if row["baseline_family"] == "fno")
    assert failed["eligible"] == "false"
    assert failed["baseline_training_selection"] == "TRAINING_FAILURE"
    assert failed["baseline_selected_step"] == "null"
    assert failed["baseline_failure_reason"] == "Nonfinite neural gradient"
    assert failed["baseline_invalid_h_count"] == "2"
    assert failed["baseline_wall_seconds"] == "null"
    assert json.loads(failed["ineligible_reasons"]) == {"baseline": "TRAINING_FAILURE"}
    assert result["neural_comparison_count"] == 6
    assert result["neural_eligible_comparison_count"] == 3
    assert result["neural_speed_advantage_count"] == 1
    assert result["neural_trained_baseline_eligible_comparison_count"] == 1
    assert result["neural_trained_baseline_speed_advantage_count"] == 1
    assert result["neural_baseline_initialization_comparison_count"] == 2
    assert result["neural_baseline_training_failure_comparison_count"] == 2
    assert "NO_NUMERICAL_HEADROOM" in result["scientific_outcomes"]
    assert result["neural_speed_advantage_observed"] is True  # independent of the classical screen
    assert artifact.read_bytes() == before
    assert any(row["path"].endswith("neural-comparisons.json") for row in inventory(target)["artifacts"])


def test_neural_accuracy_counts_retain_failures_and_explicit_units(tmp_path):
    source, target = report(tmp_path)
    put(source / "neural-comparisons.json", neural_fixture())
    analytics.publish_outputs(target, [source])
    accuracy = rows(target, "neural_accuracy")
    assert len(accuracy) == 3 * len(analytics.NEURAL_METRICS)
    measured = next(row for row in accuracy if row["baseline_family"] == "residual_cnn"
                    and row["metric"] == "heldout_two_step_rms")
    assert measured["cases"] == "4" and measured["eligible"] == measured["wins"] == "2"
    assert measured["ineligible"] == "2"
    assert json.loads(measured["failure_reasons"]) == {"baseline:MISSING_RESULT": 2}
    failed = next(row for row in accuracy if row["baseline_family"] == "fno"
                  and row["metric"] == "same_h_rollout_upper_error")
    assert failed["eligible"] == failed["wins"] == "0"
    assert json.loads(failed["failure_reasons"]) == {"baseline:NUMERICAL_FAILURE": 4}
    metadata = json.loads((target / "outputs/tables.json").read_text())
    assert metadata["columns"]["neural_accuracy"] == analytics.COLUMNS["neural_accuracy"]
    assert metadata["units"]["speedup"].startswith("baseline complete-rollout")
    assert "independently of classical headroom" in metadata["neural_comparison_interpretation"]
    contract = json.loads((Path(__file__).parents[1] / ".tower/contracts/outputs.v1.json").read_text())
    for table in ["neural_comparisons", "neural_accuracy"]:
        entry = next(row for row in contract["outputs"] if row["path"] == f"outputs/{table}.csv")
        assert entry["required"] is False
        assert entry["columns"] == analytics.COLUMNS[table]
        assert entry["min_rows"] == 0 and entry["max_rows"] == analytics.MAX_ROWS


def test_neural_wins_against_initialization_do_not_become_trained_baseline_wins(tmp_path):
    source, target = report(tmp_path)
    document = neural_fixture()
    document["training"] = [row for row in document["training"] if row["family"] in {"reaction_clock", "unet"}]
    document["comparisons"] = [row for row in document["comparisons"] if row["baseline_family"] == "unet"]
    for row in document["comparisons"]:
        row.update(speedup=2., speed_outcome="WIN")
        row["baseline_best"]["wall_seconds"] = 1.
    document["aggregates"] = []
    put(source / "neural-comparisons.json", document)
    result = analytics.publish_outputs(target, [source])
    assert result["neural_speed_advantage_count"] == 2
    assert result["neural_speed_advantage_observed"] is True
    assert result["neural_trained_baseline_eligible_comparison_count"] == 0
    assert result["neural_trained_baseline_speed_advantage_count"] == 0
    assert result["neural_trained_baseline_speed_advantage_observed"] is None


def test_unsupported_neural_records_are_omitted_without_fabricating_success(tmp_path):
    source, target = report(tmp_path)
    document = neural_fixture()
    document["comparisons"].insert(0, {"parent_id": {}, "tdn_family": [], "baseline_family": []})
    put(source / "neural-comparisons.json", document)
    result = analytics.publish_outputs(target, [source])
    assert result["reporting_omission_count"] == 1
    assert result["neural_comparison_count"] == 6
    assert rows(target, "neural_comparisons")[0]["source_record"] == "/comparisons/1"


def test_training_table_exposes_actual_baseline_cost_and_declared_budget(tmp_path):
    source, target = report(tmp_path)
    put(source / "training/residual_cnn.json", {"family": "residual_cnn", "status": "COMPLETED",
        "steps": 8, "selected_step": 3, "history": [], "parameter_count": 2049,
        "training_seconds": 1.75, "benchmark_role": "neural_baseline", "learning_rate": .001,
        "optimizer": "Adam", "maximum_optimizer_updates": 8,
        "architecture": {"track": "direct", "backbone": "residual_cnn"}})
    analytics.publish_outputs(target, [source])
    trained = rows(target, "training")[0]
    assert trained["parameter_count"] == "2049"
    assert trained["training_seconds"] == "1.75"
    assert trained["learning_rate"] == "0.001"
    assert trained["optimizer"] == "Adam" and trained["maximum_optimizer_updates"] == "8"
    assert trained["benchmark_role"] == "neural_baseline"
    assert trained["architecture_track"] == "direct" and trained["architecture_backbone"] == "residual_cnn"
    assert trained["step"] == "8" and trained["selected_step"] == "3"
