import json
from types import SimpleNamespace

import pytest

from tdn.analysis.agenda.structural import run
from tdn.analysis.agenda.structural_spec import QUESTION_IDS, specification


def test_specification_is_stdlib_only_and_declares_unique_cases():
    for profile in ("smoke", "development", "full"):
        spec = specification(profile)
        assert len(spec["case_ids"]) == len(set(spec["case_ids"]))
        assert set(spec["required_case_ids"]) <= set(spec["case_ids"])
        assert set(spec["question_ids"]) == set(QUESTION_IDS)
    assert specification("smoke")["max_seconds"] == 120
    assert specification("full")["max_seconds"] == 600
    assert len(specification("smoke")["patterns"]) == 8


def test_smoke_structural_audit_resolves_quadratic_span_and_every_literature_panel(tmp_path):
    protocol = dict(profile="smoke", structural=specification("smoke"))
    summary = run(protocol, tmp_path)
    assert summary["status"] == "COMPLETED", summary
    cases = json.loads((tmp_path / "cases.json").read_text())["rows"]
    assert summary["reported_cases"] == len(protocol["structural"]["case_ids"])
    assert {x["case_id"] for x in cases} == set(protocol["structural"]["case_ids"])
    assert {x["question_id"] for x in cases} == set(QUESTION_IDS)
    assert set(q for row in cases for q in row["literature_question_ids"]) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
    assert all(x["outcome"] == "PASS" for x in cases if x["required"])
    assert (tmp_path / "fields.npz").stat().st_size > 0
    span = next(x for x in cases if x["case_id"] == "response/0/mixed_phase0/span")
    assert span["oracle_spans"]["local_gate_and_constant"]["rms"] > 0
    assert span["oracle_spans"]["signed_source_dictionary"]["rms"] <= span["oracle_spans"]["local_gate"]["rms"] + 1e-14
    receiver = next(x for x in cases if x["case_id"] == "response/0/remote_bump/span")
    assert receiver["receiver_cells"] > 0
    assert receiver["receiver_source_rms"] == 0.
    assert receiver["receiver_defect_rms"] > 5 * receiver["uncertainty_rms"]
    assert set(summary["rank_summary"]) == {"2", "4", "8"}


def test_interrupted_audit_preserves_unreported_ids_and_does_not_complete(tmp_path):
    result = run(dict(profile="smoke"), tmp_path, SimpleNamespace(requested=True))
    assert result["status"] == "INCOMPLETE"
    assert result["reported_cases"] == 0
    assert result["rank_promising"] is False
    assert result["selected_rank"] is None
    assert result["unreported_case_ids"] == specification("smoke")["case_ids"] or set(result["unreported_case_ids"]) == set(specification("smoke")["case_ids"])


def test_specification_tamper_rejected_before_outputs(tmp_path):
    spec = specification("smoke")
    spec["rank_relative_tolerance"] = 1.
    with pytest.raises(ValueError, match="immutable"):
        run(dict(structural=spec), tmp_path)
    assert not (tmp_path / "summary.json").exists()
