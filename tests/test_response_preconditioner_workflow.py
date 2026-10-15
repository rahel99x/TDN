"""Evidence isolation and end-to-end bounded fitting, not scientific promotion."""
import json
from pathlib import Path

import pytest
import torch

from tdn.analysis.response_preconditioner import workflow as w


def test_output_is_fresh_project_contained_and_symlink_safe(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    output = w.create_output("smoke", "runs/new", root=root)
    (output / "evidence.json").write_text('{"keep": true}')
    with pytest.raises(ValueError, match="exists"):
        w.create_output("smoke", output, root=root)
    assert json.loads((output / "evidence.json").read_text())["keep"]
    with pytest.raises(ValueError, match="inside"):
        w.create_output("verify", tmp_path / "outside", root=root)
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        w.create_output("verify", "escape/unwanted", root=root)


def test_parent_protocol_is_disjoint_reproducible_and_explicitly_nonpromoting():
    protocol = w.protocol("smoke")
    seeds = [x for group in protocol["parent_seeds"].values() for x in group]
    assert len(seeds) == len(set(seeds)) == 4
    assert not protocol["scientific_promotion"]
    assert protocol["future_stages"]["promotion"] == "BLOCKED"
    assert protocol["schema"] == "tdn.response_preconditioner.implementation.v2"
    assert "not fresh confirmation" in protocol["cohort_reuse"]
    first = list(w.parents(protocol, "train"))
    repeated = list(w.parents(protocol, "train"))
    assert first[0]["u"].dtype == torch.float32
    torch.testing.assert_close(first[0]["u"], repeated[0]["u"], rtol=0, atol=0)
    assert not torch.equal(first[0]["u"], first[1]["u"])


def test_source_snapshot_includes_uncommitted_new_implementation_and_changes(tmp_path):
    source = tmp_path / "tdn" / "analysis" / "response_preconditioner" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("first = 1\n")
    before = w.source_snapshot(tmp_path)
    source.write_text("first = 2\n")
    after = w.source_snapshot(tmp_path)
    assert before["fingerprint"] != after["fingerprint"]
    assert before["files"][0]["path"] == "tdn/analysis/response_preconditioner/model.py"


def test_correction_metric_retains_null_and_unresolved_sources():
    zero = torch.zeros((1, 1, 5, 5), dtype=torch.float64)
    row = dict(kind="action", teacher=zero, matched_mean=zero,
               reference=dict(accepted=True, uncertainty_rms=1e-12,
                              matched_mean=dict(accepted=True, uncertainty_rms=1e-12)))
    metrics = w._correction_metrics(row, zero+1, w.protocol("smoke"))
    assert metrics["correction_status"] == "UNRESOLVED_NEAR_NULL"
    assert metrics["action_correction_relative_rms"] is None
    row["reference"]["uncertainty_rms"] = None
    metrics = w._correction_metrics(row, zero, w.protocol("smoke"))
    assert metrics["correction_status"] == "UNRESOLVED_REFERENCE"
    assert metrics["correction_uncertainty_rms"] is None


def test_failed_work_preserves_protocol_software_and_traceback(tmp_path, monkeypatch):
    def fail(*_):
        raise RuntimeError("deliberate workflow interruption")
    monkeypatch.setattr(w, "_smoke", fail)
    result = w.execute("smoke", tmp_path / "failed-evidence")
    assert result["status"] == "FAILED"
    assert not result["scientific_promotion"]
    output = Path(result["output"])
    assert "deliberate workflow interruption" in (output / "failure.txt").read_text()
    assert all((output / filename).is_file() for filename in
               ("protocol.json", "source.json", "software.json", "summary.json", "journal.jsonl"))
    with pytest.raises(ValueError, match="exists"):
        w.execute("smoke", output)


def test_cpu_only_and_source_change_prevent_unsupported_sealing(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="explicit CPU"):
        w.execute("smoke", tmp_path / "gpu", device="cuda")
    count = iter(("before", "after"))
    monkeypatch.setattr(w, "source_snapshot", lambda:dict(fingerprint=next(count), files=[]))
    monkeypatch.setattr(w, "_smoke", lambda *_:dict(metrics_count=0))
    result = w.execute("smoke", tmp_path / "changed")
    assert result["status"] == "FAILED"
    assert "Source changed" in result["error"]


def test_actual_bounded_smoke_freezes_readout_before_diagnostics(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path / "actual-smoke"
        output.mkdir()
        summary = w._smoke(output, w.protocol("smoke"))
    finally:
        torch.set_num_threads(old)
    selection = json.loads((output / "selection.json").read_text())
    assert len(selection["coefficients"]) == 40
    assert len(selection["training_parent_ids"]) == 2
    assert not selection["diagnostic_access_before_selection"]
    assert selection["candidates"][0]["candidate"] == "zero_readout"
    assert all(row["diagnostics"]["optimality_verified"] for row in selection["candidates"][1:])
    assert all(row["diagnostics"]["baseline_nonregression_verified"] for row in selection["candidates"][1:])
    chosen = next(row for row in selection["candidates"] if row["candidate"] == selection["selected_candidate"])
    assert chosen["eligible"] and chosen["validation_loss"] <= selection["candidates"][0]["validation_loss"]
    events = [json.loads(line)["event"] for line in (output / "journal.jsonl").read_text().splitlines()]
    assert events.index("selection_frozen") < events.index("diagnostic_parent_horizon_complete")
    assert summary["reference_count"] == 24 and summary["unresolved_references"] == 0
    assert summary["metrics_count"] == 82
    assert summary["whole_model_derivative_cases"] == 2
    rows = json.loads((output / "metrics.json").read_text())
    assert {row["method"] for row in rows if row["split"] == "diagnostic"} >= {
        "strang_df_1", "strang_rf_8", "ordered_response_8", "fixed_feature_fitted",
        "ordered_response_corrected_2"}
    baseline_rows = [row for row in rows if row["method"] == "ordered_response_corrected_2"]
    assert len(baseline_rows) == 2
    assert all(row["timing_scope"] == "complete_uncached_call" for row in baseline_rows)
    counts = json.loads((output / "work.json").read_text())
    classical_counts = [row for row in counts if row["method"] == "ordered_response_corrected_2"]
    assert len(classical_counts) == 2
    assert all(row["work"].get("fixed_feature_encodings", 0) == 0 for row in classical_counts)
    assert all(row["work"].get("laplacian_actions", 0) == 0 for row in classical_counts)
    assert all(not row["speedup_claim"] for row in rows)
    actions = [row for row in rows if row["kind"] == "action"]
    assert all(row["correction_status"] == "RESOLVED_NONTRIVIAL" for row in actions)
    assert all(row["action_correction_relative_rms"] is not None for row in actions)
    derivatives = json.loads((output / "derivatives.json").read_text())
    assert all("reference_accepted" not in case and case["endpoint_reference_accepted"] for case in derivatives)
    assert all(case["derivative_reference_status"] == "INDEPENDENT_DIAGNOSTIC_UNCERTAINTY_UNQUALIFIED"
               for case in derivatives)
    assert all(min(row["maximum"] for row in case["finite_difference_sweep"]) < 1e-7 for case in derivatives)
    audit = json.loads((output / "audit.json").read_text())
    assert len(audit) == 2
    assert all(row["audit"]["status"] == "CERTIFIED_FINITE_GRID_PANEL" for row in audit)
