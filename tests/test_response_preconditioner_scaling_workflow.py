"""P1 workflow gates and one real 48-query private implementation fixture.

The production 2,688-query bank is never executed by these tests. Mocked
references below test failure accounting only; the shared completed fixture
uses both actual independent/refined reference implementations.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdn.analysis.response_preconditioner import scaling_workflow as w
from tdn.analysis.response_preconditioner.model import FixedFeatureResponse
from tdn.analysis.response_preconditioner.scaling_protocol import _fixture_config, protocol


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in (path / "journal.jsonl").read_text().splitlines()]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def completed_fixture(tmp_path_factory):
    output = tmp_path_factory.mktemp("scaling-complete-private")
    config = _fixture_config()
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        summary = w._study(output, config, allow_fixture=True)
    finally:
        torch.set_num_threads(before)
    return output, config, summary


def test_real_reduced_fixture_completes_with_seals_before_protected_truth(completed_fixture):
    output, config, summary = completed_fixture
    assert summary["reference_count"] == 48
    assert summary["metric_rows"] == summary["measured_metrics"] == 288
    assert summary["failed_metrics"] == summary["unresolved_references"] == 0
    assert summary["selected_models"] == 4
    assert summary["timing_bundles"] == 12 and summary["failed_timing_bundles"] == 0
    assert not summary["matched_accuracy_speedup_established"]
    journal = events(output)
    frozen = [i for i, row in enumerate(journal) if row["event"] == "all_selections_frozen"]
    assert len(frozen) == 1
    protected = [i for i, row in enumerate(journal) if row["event"] == "query_started" and (
        row["split"] == "diagnostic" or (row["split"] == "train" and row["role"] == "withheld"))]
    assert protected and min(protected) > frozen[0]
    assert sum(row["event"] == "query_started" for row in journal) == 48
    denominator = read(output / "denominators.json")
    assert denominator["planned"] == denominator["attempted"] == denominator["accepted"] == 48
    assert denominator["failed"] == denominator["unresolved"] == denominator["missing"] == 0
    counts = read(output / "evaluation_denominators.json")
    assert counts["expected_metrics"] == counts["attempted_metrics"] == counts["measured_metrics"] == 288
    assert counts["missing_metrics"] == counts["failed_metrics"] == counts["unresolved_metrics"] == 0
    assert counts["expected_timing_bundles"] == counts["attempted_timing_bundles"] == 12
    assert counts["missing_timing_bundles"] == counts["failed_timing_bundles"] == 0
    assert read(output / "decisions.json")["status"] == "DESCRIPTIVE_DEVELOPMENT_ONLY"


def test_each_sealed_cell_uses_exact_nested_parent_and_source_prefixes(completed_fixture):
    output, config, _ = completed_fixture
    selection = read(output / "selection.json")
    rows = {row["query_id"]: row for row in read(output / "query_index.json")}
    train = [p["parent_id"] for p in read(output / "parents.json") if p["split"] == "train"]
    assert selection["all_selections_sealed"] and len(selection["cells"]) == 4
    for cell in selection["cells"]:
        n, k = cell["training_parent_count"], cell["calibration_source_count"]
        assert cell["training_parent_ids"] == train[:n]
        assert len(cell["train_query_ids"]) == n*k
        fitted = [rows[qid] for qid in cell["train_query_ids"]]
        assert all(row["role"] == "calibration" and row["split"] == "train" for row in fitted)
        assert Counter(row["parent_id"] for row in fitted) == {p: k for p in train[:n]}
        for parent in train[:n]:
            assert sorted(row["source_index"] for row in fitted if row["parent_id"] == parent) == list(range(k))
        assert len(cell["validation_query_ids"]) == 8
        assert all(rows[q]["split"] == "validation" and rows[q]["role"] == "withheld"
                   for q in cell["validation_query_ids"])
        assert cell["candidates"][0]["name"] == "zero_readout"
        assert [candidate["ridge"] for candidate in cell["candidates"]] == [None, 1e-10, 1e-6]
        for candidate in cell["candidates"][1:]:
            assert candidate["diagnostics"]["parent_count"] == n
            assert candidate["diagnostics"]["parent_total_weight"] == 1/n
            assert set(candidate["diagnostics"]["parent_row_counts"].values()) == {25*k}
        assert cell["selection_seal"]
        assert not cell["diagnostic_truth_accessed"] and not cell["training_withheld_truth_accessed"]


def test_all_metrics_replay_from_retained_fresh_predictions_and_parity(completed_fixture):
    output, _, _ = completed_fixture
    rows = read(output / "metrics.json")
    selections = {c["model_id"]: c for c in read(output / "selection.json")["cells"]}
    query_index = {row["query_id"]: row for row in read(output / "query_index.json")}
    for row in rows:
        query = query_index[row["query_id"]]
        with np.load(output / query["array_path"], allow_pickle=False) as saved:
            teacher, base, design = saved["teacher"], saved["base"], saved["design"]
            matched = saved["matched_mean"]
        with np.load(output / "predictions" / (row["query_id"]+".npz"), allow_pickle=False) as saved:
            prediction = saved[row["method"]]
        error = prediction-teacher
        mse = float(np.mean(error*error))
        rms = float(np.sqrt(mse))
        assert row["status"] == "MEASURED"
        assert row["mse"] == mse
        assert row["rms"] == rms
        assert row["maximum"] == float(np.max(np.abs(error)))
        assert row["mean_error"] == float(np.mean(error))
        assert row["centered_rms"] == float(np.sqrt(np.mean((error-error.mean())**2)))
        assert row["relative_reference_rms"] == rms/float(np.sqrt(np.mean(teacher*teacher)))
        for name, signal, status in (("remainder", teacher-base, row["remainder_status"]),
                                     ("feedback", teacher-matched, row["feedback_status"])):
            norm = float(np.sqrt(np.mean(signal*signal)))
            assert row[name+"_rms"] == norm
            assert row["relative_"+name+"_rms"] == (rms/norm if status == "RESOLVED" else None)
        assert row["prediction_mode"] == "fresh_deployable_action"
        if row["method"] in selections:
            cell = selections[row["method"]]
            coefficients = np.asarray(cell["selected"]["coefficients"])
            reconstructed = base + np.einsum("j,j...->...", coefficients, design)
            difference = float(np.max(np.abs(prediction-reconstructed)))
            assert row["inference_parity_maximum"] == difference
            assert difference <= row["inference_parity_tolerance"]
            if row["split"] == "train":
                expected = "in_cohort" if row["parent_id"] in cell["training_parent_ids"] else "outside_cohort"
                assert row["training_membership"] == expected


def test_costs_charge_shared_labels_once_and_expose_nested_demand(completed_fixture):
    output, _, _ = completed_fixture
    costs = read(output / "costs.json")
    index = read(output / "query_index.json")
    assert costs["global_acquisition"]["planned"] == costs["global_acquisition"]["attempted"] == 48
    assert costs["global_acquisition"]["reference_seconds"] == sum(r["reference_seconds"] for r in index)
    assert costs["global_acquisition"]["design_seconds"] == sum(r["design_seconds"] for r in index)
    selection = {c["model_id"]: c for c in read(output / "selection.json")["cells"]}
    query = {r["query_id"]: r for r in index}
    for cell in costs["cells"]:
        fit = selection[cell["model_id"]]
        assert cell["training"]["planned"] == cell["training"]["accepted"] == len(fit["train_query_ids"])
        assert cell["training_reference_seconds"] == sum(query[q]["reference_seconds"] for q in fit["train_query_ids"])
        assert cell["validation_reference_seconds"] == sum(query[q]["reference_seconds"] for q in fit["validation_query_ids"])
        assert cell["fitting_seconds"] == fit["fitting_seconds"] > 0
    assert costs["fitting_seconds"] > 0 and costs["accuracy_evaluation_seconds"] > 0
    assert costs["timing_seconds"] == sum(r["seconds"] for r in read(output / "timing_records.json"))
    assert len({c["validation_reference_seconds"] for c in costs["cells"]}) == 1


def test_native_only_and_modified_protocol_refuse_before_acquisition(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No acquisition may occur before protocol/native validation")

    monkeypatch.setattr(w.references, "acquire_reference", forbidden)
    with pytest.raises(ValueError, match="forbidden in production"):
        w.ScalingStudy(tmp_path, _fixture_config())
    config = protocol()
    config["budget"]["study_seconds"] = 901.
    with pytest.raises(ValueError, match="frozen P1"):
        w.ScalingStudy(tmp_path, config)
    with pytest.raises(ValueError, match="explicit CPU"):
        w.execute("study", tmp_path / "gpu", device="cuda")
    with pytest.raises(ValueError, match="source-bound native Raven contract"):
        w.execute("study", tmp_path / "no-contract")

    def native_rejected(*args, **kwargs):
        raise RuntimeError("fixture: real native allocation required")

    monkeypatch.setitem(sys.modules, "response_preconditioner_scaling_native",
                        SimpleNamespace(require_study_context=native_rejected))
    with pytest.raises(RuntimeError, match="real native allocation"):
        w.execute("study", tmp_path / "no-native", contract=tmp_path / "untrusted.json")
    assert not (tmp_path / "queries").exists()
    assert not (tmp_path / "no-contract").exists()
    assert not (tmp_path / "no-native").exists()


def test_diagnostic_and_training_withheld_access_refused_before_all_selections(tmp_path, monkeypatch):
    study = w.ScalingStudy(tmp_path, _fixture_config(), allow_fixture=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Protected reference acquired before selection")

    monkeypatch.setattr(w.references, "acquire_reference", forbidden)
    for split in ("train", "diagnostic"):
        cid = next(cid for cid, ctx in study.contexts.items() if ctx["parent"]["split"] == split)
        with pytest.raises(RuntimeError, match="must seal"):
            study.acquire(cid, "withheld")
    assert study.attempted == 0
    assert read(tmp_path / "denominators.json")["missing"] == 48


def _synthetic_reference(u, source, *args, **kwargs):
    """Failure-accounting stub only; never called by the numerical fixture."""
    mean = .9*source
    reference = dict(accepted=True, status="ACCEPTED", uncertainty_rms=1e-12,
        uncertainty_maximum=1e-12, matched_mean=dict(accepted=True, status="ACCEPTED",
        uncertainty_rms=1e-12, uncertainty_maximum=1e-12))
    return dict(teacher=source.clone(), refined=source.clone(), matched_mean=mean,
                matched_mean_refined=mean.clone(), reference=reference)


@pytest.mark.parametrize("error_type", (TimeoutError, FloatingPointError))
def test_partial_reference_failure_preserves_arrays_and_all_denominators(tmp_path, monkeypatch, error_type):
    calls = 0

    def reference(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("deliberate reference interruption")
        return _synthetic_reference(*args, **kwargs)

    monkeypatch.setattr(w.references, "acquire_reference", reference)
    with pytest.raises(error_type, match="deliberate"):
        w._study(tmp_path, _fixture_config(), allow_fixture=True)
    denominator = read(tmp_path / "denominators.json")
    assert denominator["planned"] == 48
    assert denominator["attempted"] == 2 and denominator["accepted"] == denominator["failed"] == 1
    assert denominator["missing"] == 46
    records = read(tmp_path / "query_index.json")
    assert [r["status"] for r in records[:2]] == ["ACCEPTED", "FAILED"]
    with np.load(tmp_path / records[0]["array_path"], allow_pickle=False) as arrays:
        assert {"teacher", "design", "initial", "source"} <= set(arrays.files)
    with np.load(tmp_path / records[1]["array_path"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"initial", "source"}
    counts = read(tmp_path / "evaluation_denominators.json")
    assert counts["expected_metrics"] == counts["missing_metrics"] == 288
    assert counts["expected_fit_cells"] == 4 and counts["selected_fit_cells"] == 0
    assert counts["expected_timing_bundles"] == counts["missing_timing_bundles"] == 12
    assert len(read(tmp_path / "metrics.json")) == 288
    assert not (tmp_path / "selection.json").exists()
    costs = read(tmp_path / "costs.json")
    assert costs["global_acquisition"]["attempted"] == 2
    assert costs["global_acquisition"]["reference_seconds"] == sum(r["reference_seconds"] for r in records[:2])
    assert costs["global_acquisition"]["reference_seconds"] > 0
    assert costs["accuracy_evaluation_seconds"] == 0


def test_unresolved_training_blocks_every_selection_without_dropping_queries(tmp_path, monkeypatch):
    def unresolved(*args, **kwargs):
        result = _synthetic_reference(*args, **kwargs)
        result["reference"].update(accepted=False, status="UNRESOLVED")
        return result

    monkeypatch.setattr(w.references, "acquire_reference", unresolved)
    with pytest.raises(RuntimeError, match="Unresolved training/validation"):
        w._study(tmp_path, _fixture_config(), allow_fixture=True)
    denominator = read(tmp_path / "denominators.json")
    assert denominator["planned"] == 48
    assert denominator["attempted"] == denominator["unresolved"] == 24
    assert denominator["missing"] == 24 and denominator["failed"] == 0
    selection = read(tmp_path / "selection.json")
    assert not selection["all_selections_sealed"]
    assert len(selection["cells"]) == 4
    assert all(c["status"] == "BLOCKED" and c["selected"] is None for c in selection["cells"])
    assert not any(row["event"] == "all_selections_frozen" for row in events(tmp_path))
    assert not any(row["event"] == "query_started" and row["split"] == "diagnostic"
                   for row in events(tmp_path))


def _loaded_study(output, completed_fixture):
    source, config, _ = completed_fixture
    study = w.ScalingStudy(output, config, allow_fixture=True)
    # Reuse only a real, private fixture's sealed coefficients, avoiding another
    # numerical bank in tests concerned with post-selection failure persistence.
    study.selection = read(source / "selection.json")
    study.models = {cell["model_id"]: FixedFeatureResponse().set_readout(cell["selected"]["coefficients"])
                    for cell in study.selection["cells"]}
    index = next(row for row in read(source / "query_index.json") if row["split"] == "diagnostic")
    with np.load(source / index["array_path"], allow_pickle=False) as arrays:
        row = dict(index, **{key: arrays[key] for key in arrays.files})
    return study, row


def test_failed_method_remains_visible_and_other_methods_continue(tmp_path, monkeypatch, completed_fixture):
    study, row = _loaded_study(tmp_path, completed_fixture)
    original = w.actions.fresh_action

    def failure(method, *args, **kwargs):
        if method == "lawson_4":
            raise ArithmeticError("deliberate action failure")
        return original(method, *args, **kwargs)

    monkeypatch.setattr(w.actions, "fresh_action", failure)
    study.score([row])
    study.finalize()
    selected = [r for r in read(tmp_path / "metrics.json") if r["query_id"] == row["query_id"]]
    assert len(selected) == 9
    assert sum(r["status"] == "FAILED" for r in selected) == 1
    assert sum(r["status"] == "MEASURED" for r in selected) == 8
    assert next(r for r in selected if r["method"] == "lawson_4")["rms"] is None
    with np.load(tmp_path / "predictions" / (row["query_id"]+".npz"), allow_pickle=False) as arrays:
        assert "lawson_4" not in arrays.files and len(arrays.files) == 8
    denominator = read(tmp_path / "evaluation_denominators.json")
    assert denominator["attempted_metrics"] == 9 and denominator["failed_metrics"] == 1
    assert denominator["missing_metrics"] == 279


def test_deadline_during_scoring_preserves_completed_methods_and_pending_rows(tmp_path, monkeypatch, completed_fixture):
    study, row = _loaded_study(tmp_path, completed_fixture)
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("deliberate scoring deadline")

    monkeypatch.setattr(study, "check", check)
    with pytest.raises(TimeoutError, match="scoring deadline"):
        study.score([row])
    study.finalize()
    path = tmp_path / "predictions" / (row["query_id"]+".npz")
    with np.load(path, allow_pickle=False) as arrays:
        assert len(arrays.files) == 2
    assert len(read(path.with_suffix(".json"))) == 9
    denominator = read(tmp_path / "evaluation_denominators.json")
    assert denominator["attempted_metrics"] == denominator["measured_metrics"] == 2
    assert denominator["missing_metrics"] == 286


def test_partial_timing_preserves_raw_records_when_deadline_interrupts(tmp_path, monkeypatch, completed_fixture):
    study, _ = _loaded_study(tmp_path, completed_fixture)
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("deliberate timing deadline")

    monkeypatch.setattr(study, "check", check)
    with pytest.raises(TimeoutError, match="timing deadline"):
        study.time_context(study.timing_contexts()[0])
    study.finalize()
    records = read(tmp_path / "timing_records.json")
    assert len(records) == 2 and all(row["status"] == "PASS" for row in records)
    assert read(tmp_path / "timing.json") == []
    denominator = read(tmp_path / "evaluation_denominators.json")
    assert denominator["expected_timing_bundles"] == 12
    assert denominator["attempted_timing_bundles"] == 2
    assert denominator["missing_timing_bundles"] == 10


def test_source_change_rejects_verification_seal_and_keeps_failure(tmp_path, monkeypatch):
    calls = iter(("before", "after"))
    monkeypatch.setattr(w.legacy, "source_snapshot", lambda: dict(fingerprint=next(calls), files=[]))
    monkeypatch.setattr(w.legacy, "_software", lambda: dict(implementation_fixture=True))
    monkeypatch.setattr(w.r7, "_verify", lambda *args: dict(checks=dict(total=1)))
    summary = w.execute("verify", tmp_path / "changed")
    assert summary["status"] == "FAILED" and "Source changed" in summary["error"]
    output = Path(summary["output"])
    assert read(output / "source_after.json")["fingerprint"] == "after"
    assert (output / "failure.txt").is_file()
    assert read(output / "artifacts.json")["status"] == "FAILED"
    assert not summary["scientific_promotion"]
