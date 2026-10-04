"""Accuracy-matched neural comparisons retain failures and checkpoint evidence."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tdn.research.neural_comparison import (
    BASELINE_FAMILIES, TDN_FAMILIES, readable_neural_summary, summarize_neural_comparisons,
)


def frontier(family, *, parent="diagnostic-a", h=.08, seconds=2., error=.001,
             feasible=True, status="COMPLETED", device="cpu"):
    return {"parent_id": parent, "family": family, "h": h, "status": status,
            "error": error, "error_upper": error + .00001,
            "reference_uncertainty": .00001, "feasible": feasible,
            "timing": {"wall_seconds_median": seconds}, "device": device}


def heldout(family, *, parent="diagnostic-a", h=.03, one=.001, two=.002, status="COMPLETED"):
    return {"parent_id": parent, "family": family, "h": h, "status": status,
            "one_step": {"rms": one, "upper": one + .00001},
            "two_step": {"rms": two, "upper": two + .00001}}


def comparison(result, *, family="reaction_clock", baseline="fno", parent="diagnostic-a"):
    return next(row for row in result["comparisons"]
                if (row["tdn_family"], row["baseline_family"], row["parent_id"]) == (family, baseline, parent))


def test_neural_speed_does_not_require_classical_headroom_or_classical_speed():
    # The classical control is both accurate and far faster; neither property
    # changes the TDN-vs-neural question at the same declared error tolerance.
    rows = [frontier("split", seconds=.001), frontier("richardson_split", seconds=.002),
            frontier("reaction_clock", seconds=2.), frontier("fno", seconds=3.)]
    result = summarize_neural_comparisons(rows, [], ["reaction_clock", "fno"])
    pair = comparison(result)
    assert pair["eligible"] and pair["speedup"] == 1.5
    assert pair["speed_outcome"] == "WIN"
    assert result["aggregates"][0]["matched_tolerance_speed"] == {
        "cases": 1, "eligible": 1, "wins": 1, "losses": 0, "ties": 0,
        "ineligible": 0, "failure_reasons": {},
    }
    assert "headroom" not in result
    assert "overall_rank" not in result


def test_every_declared_family_retained_on_every_parent_even_if_missing():
    rows = [frontier("reaction_clock"), frontier("fno"), frontier("split", parent="diagnostic-b")]
    training = [{"family": "transport", "status": "NUMERICAL_FAILURE", "steps": 42,
                 "selected_step": None, "error": "Nonfinite neural gradient"}]
    result = summarize_neural_comparisons(rows, [], ["reaction_clock", "transport", "fno", "unet"], training)
    assert len(result["parents"]) == 2
    assert all(len(parent["families"]) == 4 for parent in result["parents"])
    assert len(result["comparisons"]) == 8
    missing = comparison(result, baseline="unet")
    assert not missing["eligible"] and missing["speedup"] is None
    assert missing["ineligible_reasons"] == {"baseline": "MISSING_RESULTS"}
    failed = comparison(result, family="transport")
    assert failed["ineligible_reasons"] == {"tdn": "TRAINING_FAILURE"}
    failure_record = next(row for row in result["training"] if row["family"] == "transport")
    assert failure_record["failure_reason"] == "Nonfinite neural gradient"


def test_fast_invalid_and_inaccurate_rollouts_cannot_beat_feasible_results():
    rows = [frontier("reaction_clock", h=.04, seconds=4.),
            frontier("reaction_clock", h=.08, seconds=.0001, status="INVALID_TRAJECTORY"),
            frontier("fno", h=.04, seconds=6.),
            frontier("fno", h=.08, seconds=.0001, error=.01, feasible=False)]
    result = summarize_neural_comparisons(rows, [], ["reaction_clock", "fno"])
    pair = comparison(result)
    assert pair["tdn_best"]["h"] == pair["baseline_best"]["h"] == .04
    assert pair["speedup"] == 1.5
    family = result["parents"][0]["families"][0]
    assert family["feasible_h_count"] == 1 and family["invalid_h_count"] == 1
    assert family["infeasible_or_invalid_h"][0]["reason"] == "INVALID_TRAJECTORY"
    assert pair["same_h_rollout_rms"]["failed"] == 1


def test_best_feasible_step_can_differ_and_same_h_errors_are_separate():
    rows = [frontier("reaction_clock", h=.04, seconds=3., error=.0001),
            frontier("reaction_clock", h=.08, seconds=2., error=.001),
            frontier("fno", h=.04, seconds=4., error=.0002),
            frontier("fno", h=.08, seconds=1., error=.01, feasible=False)]
    result = summarize_neural_comparisons(rows, [], ["reaction_clock", "fno"])
    pair = comparison(result)
    assert pair["tdn_best"]["h"] == .08 and pair["baseline_best"]["h"] == .04
    assert pair["speedup"] == 2.
    assert pair["tdn_best"]["error"] > pair["baseline_best"]["error"]
    assert pair["same_h_rollout_rms"]["wins"] == 2
    assert pair["same_h_rollout_rms"]["eligible"] == 2
    assert "regardless of tolerance" in result["accuracy_scope"]


def test_ties_losses_and_heldout_failures_have_explicit_counts():
    rows = [frontier("reaction_clock"), frontier("fno"),
            frontier("reaction_clock", parent="diagnostic-b", seconds=4.),
            frontier("fno", parent="diagnostic-b", seconds=2.)]
    diagnostics = [heldout("reaction_clock", one=.001, two=.002),
                   heldout("fno", one=.001, two=.001),
                   heldout("reaction_clock", parent="diagnostic-b", status="INVALID_TRAJECTORY")]
    result = summarize_neural_comparisons(rows, diagnostics, ["reaction_clock", "fno"])
    counts = result["aggregates"][0]
    assert counts["matched_tolerance_speed"]["ties"] == 1
    assert counts["matched_tolerance_speed"]["losses"] == 1
    assert counts["heldout_one_step_rms"]["ties"] == 1
    assert counts["heldout_two_step_rms"]["losses"] == 1
    assert counts["heldout_two_step_rms"]["ineligible"] == 1
    assert counts["heldout_one_step_rms"]["failure_reasons"] == {
        "baseline:MISSING_RESULT": 1, "tdn:INVALID_TRAJECTORY": 1,
    }


def test_selected_initialization_is_not_reported_as_trained_baseline():
    training = [{"family": "fno", "status": "COMPLETED", "steps": 96,
                 "selected_step": 0, "selected_parameters_changed": False},
                {"family": "reaction_clock", "status": "COMPLETED", "steps": 96,
                 "selected_step": 24, "selected_parameters_changed": True}]
    result = summarize_neural_comparisons([frontier("reaction_clock"), frontier("fno")], [],
                                         ["reaction_clock", "fno"], training)
    assert comparison(result)["baseline_training_selection"] == "SELECTED_INITIALIZATION"
    assert result["training"][0]["selection"] == "TRAINED_CHECKPOINT"
    assert "SELECTED_INITIALIZATION" in readable_neural_summary(result)
    # Frozen GPU timing retains CPU training evidence rather than implying a
    # separately trained model or silently removing failed training branches.
    gpu_result = summarize_neural_comparisons([frontier("reaction_clock", device="cuda"),
                                              frontier("fno", device="cuda")], [],
                                             ["reaction_clock", "fno"], training)
    assert gpu_result["training"] == result["training"]


@pytest.mark.parametrize("baseline", BASELINE_FAMILIES)
def test_each_declared_baseline_has_pairwise_comparison(baseline):
    result = summarize_neural_comparisons([frontier("temporal_mlp"), frontier(baseline)], [],
                                         ["temporal_mlp", baseline])
    pair = comparison(result, family="temporal_mlp", baseline=baseline)
    assert pair["eligible"]
    if baseline == "generic_mlp":
        assert pair["baseline_kind"] == "time_conditioned_feature_additive_hybrid"


@pytest.mark.parametrize("change,reason", [
    ({"error": float("nan")}, "INVALID_OR_MISSING_ERROR"),
    ({"error_upper": None, "reference_uncertainty": None}, "INVALID_OR_MISSING_ERROR"),
    ({"timing": {"wall_seconds_median": 0.}}, "INVALID_OR_MISSING_WALL_TIME"),
    ({"timing": {"wall_seconds_median": float("inf")}}, "INVALID_OR_MISSING_WALL_TIME"),
    ({"feasible": False, "timing": {"wall_seconds_median": 1., "soft_budget_passed": False}}, "DEVICE_MEMORY_BUDGET_FAILED"),
    ({"feasible": True, "timing": {"wall_seconds_median": 1., "soft_budget_passed": False}}, "DEVICE_MEMORY_BUDGET_FAILED"),
])
def test_bad_feasible_evidence_is_not_an_eligible_timing(change, reason):
    rows = [frontier("reaction_clock"), {**frontier("fno"), **change}]
    result = summarize_neural_comparisons(rows, [], ["reaction_clock", "fno"])
    assert not comparison(result)["eligible"]
    assert result["parents"][0]["families"][1]["infeasible_or_invalid_h"][0]["reason"] == reason
    json.dumps(result, allow_nan=False)


def test_cross_device_timings_cannot_be_compared():
    result = summarize_neural_comparisons([frontier("reaction_clock", device="cuda"), frontier("fno")], [],
                                         ["reaction_clock", "fno"])
    assert comparison(result)["ineligible_reasons"] == {"comparison": "DEVICE_MISMATCH"}


def test_empty_measurements_do_not_invent_diagnostic_parents_or_results():
    result = summarize_neural_comparisons([], [], ["reaction_clock", "fno"])
    assert result["parents"] == result["comparisons"] == []
    assert result["aggregates"][0]["matched_tolerance_speed"]["cases"] == 0
    assert "entirely unobserved parents cannot be inferred" in result["coverage_scope"]


def test_heldout_only_parent_preserves_missing_rollout_coverage():
    result = summarize_neural_comparisons([], [heldout("reaction_clock"), heldout("fno")],
                                         ["reaction_clock", "fno"])
    assert not comparison(result)["eligible"]
    assert comparison(result)["heldout_one_step_rms"]["eligible"] == 1
    assert comparison(result)["heldout_two_step_rms"]["eligible"] == 1


def test_duplicate_or_unbounded_inputs_are_rejected_without_selective_omission():
    row = frontier("fno")
    with pytest.raises(ValueError, match="Duplicate frontier"):
        summarize_neural_comparisons([row, row], [], ["fno"])
    with pytest.raises(ValueError, match="unique"):
        summarize_neural_comparisons([], [], ["fno", "fno"])
    with pytest.raises(ValueError, match="bounded input"):
        summarize_neural_comparisons([row] * 8193, [], ["fno"])
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_neural_comparisons([{**row, "h": float("nan")}], [], ["fno"])


def test_full_declared_grid_fits_tower_one_mib_json_read_limit():
    families = [*TDN_FAMILIES[:8], *BASELINE_FAMILIES]
    rows, diagnostics = [], []
    for index in range(9):
        parent = f"diagnostic-intermediate-mixed_frequency-{33001 + index}"
        for family in families:
            for h in (.04, .08, .16, .32):
                rows.append(frontier(family, parent=parent, h=h, seconds=.012345678901234))
            for h in (.03, .11):
                diagnostics.append(heldout(family, parent=parent, h=h))
    result = summarize_neural_comparisons(rows, diagnostics, families)
    assert len(result["comparisons"]) == 504
    assert len(result["aggregates"]) == 56
    assert len(json.dumps(result, indent=2, allow_nan=False).encode()) < 1024 * 1024


def test_module_imports_without_site_packages_or_scientific_dependencies():
    root = Path(__file__).resolve().parents[1]
    check = subprocess.run([sys.executable, "-S", "-c",
                            "from tdn.research.neural_comparison import summarize_neural_comparisons; "
                            "import sys; assert 'torch' not in sys.modules; "
                            "assert summarize_neural_comparisons([], [], ['fno'])['comparisons'] == []"],
                           cwd=root, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
