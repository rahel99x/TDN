"""Scientific denominators, pairing and failure-preserving derived reports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from tdn.analysis.premix.comparison import compare, summarize, SOURCE_SCHEMA
from tdn.analysis.premix.neural import build_protocol


def _frontiers(protocol, candidates):
    result = []
    groups = {}
    for row in candidates:
        groups.setdefault((row["parent_id"], row["seed"], row["family"]), []).append(row)
    for rows in groups.values():
        for norm in ("rms", "max"):
            for target in protocol["targets"]:
                eligible = [row for row in rows if row["status"] == "COMPLETED" and row.get("memory_ok", False)
                            and row["upper_" + norm] <= target]
                selected = min(eligible, key=lambda row: row["timing"]["wall_seconds_median"]) if eligible else None
                result.append({**{key: rows[0][key] for key in ("parent_id", "seed", "family", "regime", "category", "distribution", "grid")},
                    "norm": norm, "target": target, "status": "FEASIBLE" if selected else "NO_FEASIBLE_CANDIDATE",
                    "selected_steps": selected["steps"] if selected else None,
                    "seconds": selected["timing"]["wall_seconds_median"] if selected else None,
                    "error_upper": selected["upper_" + norm] if selected else None})
    return result


@pytest.fixture
def evidence():
    protocol = build_protocol("smoke")
    protocol["parents"] = [row for row in protocol["parents"] if row["split"] != "diagnostic"] + [
        row for row in protocol["parents"] if row["regime"] in ("smooth", "mixed") and row["split"] == "diagnostic"]
    protocol.update(seeds=[11, 22], targets=[2e-3, 2e-5], timing_repeats=3)
    training = [{"seed": seed, "family": family, "status": "COMPLETED", "selected_step": 2,
                 "selected_parameters_changed": True, "selection": "TRAINED_CHECKPOINT"}
                for seed in protocol["seeds"] for family in protocol["families"]]
    methods = [(None, family) for family in protocol["classical"]] + [(seed, family) for seed in protocol["seeds"] for family in protocol["families"]]
    seconds = dict(premix=1., premix_local=.8, precompress=1.5, fno=2., cnn=3., strang=.1, etdrk4=.2, gl3_fused=.3)
    candidates = []
    for parent in protocol["parents"]:
        if parent["split"] != "diagnostic":
            continue
        for seed, family in methods:
            for steps in protocol["step_counts"]:
                time = seconds[family] * steps
                error = 3e-4 if steps == 1 else 1e-7
                candidates.append({**{key: parent[key] for key in ("parent_id", "regime", "category", "distribution", "grid")},
                    "seed": seed, "family": family, "steps": steps, "horizon": parent["horizons"][0],
                    "status": "COMPLETED", "admissible": True, "memory_ok": True, "device": "cpu",
                    "checkpoint_selection": "TRAINED_CHECKPOINT" if seed is not None else None,
                    "rms": error, "max_error": error, "uncertainty_rms": 1e-10, "uncertainty_max_bound": 1e-9,
                    "upper_rms": error + 1e-10, "upper_max": error + 1e-9,
                    "timing": {"wall_seconds_median": time, "wall_seconds_raw": [.9 * time, time, 1.1 * time], "device": "cpu"}})
    return protocol, candidates, _frontiers(protocol, candidates), training


def test_exact_plan_independent_pairs_and_source_pointers(evidence):
    result = compare(*evidence)
    assert result["counts"]["total"] == 96
    assert result["counts"]["timing_eligible"] == 96
    assert result["counts"]["ours_wins"] == 96
    assert result["counts"]["trained_pair_wins"] == 96
    assert result["coverage"]["complete_source_plan"] is True
    assert result["coverage"]["comparisons"] == {"expected": 96, "reported": 96}
    assert len({row["comparison_id"] for row in result["rows"]}) == 96
    for row in result["rows"]:
        source = evidence[1][int(row["ours_candidate_record"].split("/")[-1])]
        baseline = evidence[1][int(row["baseline_candidate_record"].split("/")[-1])]
        assert source["seed"] == baseline["seed"] == row["seed"]
        assert source["parent_id"] == baseline["parent_id"] == row["parent_id"]
        assert row["speedup_baseline_over_ours"] == baseline["timing"]["wall_seconds_median"] / source["timing"]["wall_seconds_median"]
    for group in result["groups"]:
        assert len(set(group["parent_ids"])) == group["counts"]["total"]
    assert {row["group"] for row in result["groups"] if row["group_scope"] == "category"} == {"favorable", "typical"}


def test_full_protocol_empty_run_retains_all_6912_comparison_cases():
    result = compare(build_protocol("full"), [], [], [])
    assert result["coverage"]["comparisons"] == {"expected": 6912, "reported": 6912}
    assert result["coverage"]["candidates"]["expected"] == 2592
    assert result["counts"]["incomplete_or_invalid"] == 6912
    assert result["counts"]["timing_eligible"] == 0
    assert result["counts"]["ours_only_feasible"] == 0
    assert all(row["speedup_baseline_over_ours"] is None for row in result["rows"])


def test_initialization_is_not_reported_as_a_trained_victory(evidence):
    protocol, candidates, frontiers, training = evidence
    selected = next(row for row in training if row["seed"] == 11 and row["family"] == "premix")
    selected.update(selected_step=0, selected_parameters_changed=False, selection="SELECTED_INITIALIZATION")
    for row in candidates:
        if row["seed"] == 11 and row["family"] == "premix":
            row["checkpoint_selection"] = "SELECTED_INITIALIZATION"
    result = compare(protocol, candidates, frontiers, training)
    assert result["counts"]["ours_wins"] == 96
    assert result["counts"]["trained_pair_wins"] == 72
    assert result["counts"]["initialization_in_eligible_pair"] == 24
    assert all(not row["trained_pair_eligible"] for row in result["rows"] if row["ours_selection"] == "SELECTED_INITIALIZATION")


def test_coverage_improvement_is_never_an_eligible_speed_win(evidence):
    protocol, candidates, _, training = evidence
    for row in candidates:
        if row["family"] == "fno":
            row.update(rms=.1, max_error=.1, upper_rms=.1 + 1e-10, upper_max=.1 + 1e-9)
    result = compare(protocol, candidates, _frontiers(protocol, candidates), training)
    assert result["counts"]["ours_only_feasible"] == 32
    assert result["counts"]["ours_wins"] == 64
    for row in result["rows"]:
        if row["baseline"] == "fno":
            assert not row["timing_eligible"] and row["speedup_baseline_over_ours"] is None
            assert row["outcome"] is None and not row["robust_speed_win"]


def test_missing_scheduled_candidate_is_unknown_not_a_coverage_loss(evidence):
    protocol, candidates, frontiers, training = evidence
    missing = next(index for index, row in enumerate(candidates) if row["family"] == "fno" and row["seed"] == 11 and row["steps"] == 2)
    candidates.pop(missing)
    result = compare(protocol, candidates, frontiers, training)
    assert result["coverage"]["complete_source_plan"] is False
    assert result["counts"]["ours_feasible_baseline_unresolved"] == 8
    assert result["counts"]["ours_only_feasible"] == 0
    assert result["counts"]["timing_eligible"] == 88


def test_failed_step_does_not_hide_a_valid_feasible_alternative(evidence):
    protocol, candidates, _, training = evidence
    row = next(row for row in candidates if row["family"] == "premix" and row["seed"] == 11 and row["steps"] == 1)
    row.update(status="INVALID_TRAJECTORY", admissible=False)
    result = compare(protocol, candidates, _frontiers(protocol, candidates), training)
    assert result["counts"]["timing_eligible"] == 96
    selected = [row for row in result["rows"] if row["parent_id"] == candidates[0]["parent_id"] and row["ours"] == "premix" and row["seed"] == 11]
    assert all(row["ours_selected_steps"] == 2 for row in selected)


def test_overlapping_repeat_ranges_do_not_become_robust_wins(evidence):
    protocol, candidates, frontiers, training = evidence
    for row in candidates:
        if row["family"] == "fno":
            time = row["timing"]["wall_seconds_median"]
            row["timing"]["wall_seconds_raw"] = [.1 * time, time, 10 * time]
    result = compare(protocol, candidates, frontiers, training)
    rows = [row for row in result["rows"] if row["baseline"] == "fno"]
    assert all(row["outcome"] == "WIN" and not row["robust_speed_win"] and row["range_status"] == "OVERLAPPING" for row in rows)


def test_single_repeat_smoke_cannot_claim_robust_timing(evidence):
    protocol, candidates, frontiers, training = evidence
    protocol["timing_repeats"] = 1
    for row in candidates:
        row["timing"]["wall_seconds_raw"] = [row["timing"]["wall_seconds_median"]]
    result = compare(protocol, candidates, frontiers, training)
    assert result["counts"]["ours_wins"] == 96
    assert result["counts"]["range_separated_wins"] == 0
    assert all(row["range_status"] == "INSUFFICIENT_REPEATS" for row in result["rows"])


def test_device_mismatch_excludes_speed_but_retains_observed_feasibility(evidence):
    protocol, candidates, frontiers, training = evidence
    for row in candidates:
        if row["family"] == "fno":
            row["device"] = row["timing"]["device"] = "cuda"
    result = compare(protocol, candidates, frontiers, training)
    assert result["counts"]["both_feasible"] == 96
    assert result["counts"]["timing_eligible"] == 64
    assert all(row["speedup_baseline_over_ours"] is None and "DEVICE_MISMATCH" in row["issues"]
               for row in result["rows"] if row["baseline"] == "fno")


@pytest.mark.parametrize("table_index", (1, 2, 3))
def test_duplicate_source_identity_is_rejected(evidence, table_index):
    arguments = list(deepcopy(evidence))
    arguments[table_index].append(deepcopy(arguments[table_index][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        compare(*arguments)


@pytest.mark.parametrize("change", ("parent", "seed", "steps", "bool_steps"))
def test_undeclared_or_malformed_candidate_identity_is_rejected(evidence, change):
    protocol, candidates, frontiers, training = evidence
    row = candidates[0]
    key, value = {"parent": ("parent_id", "test-leak"), "seed": ("seed", 999),
                  "steps": ("steps", 1024), "bool_steps": ("steps", True)}[change]
    row[key] = value
    with pytest.raises(ValueError):
        compare(protocol, candidates, frontiers, training)


@pytest.mark.parametrize("change", ("error", "median", "category", "selection", "nonminimum"))
def test_corrupt_scientific_provenance_or_selection_remains_ineligible(evidence, change):
    protocol, candidates, frontiers, training = evidence
    candidate = next(row for row in candidates if row["family"] == "premix" and row["seed"] == 11 and row["steps"] == 1)
    if change == "error":
        candidate["upper_rms"] = 0
    elif change == "median":
        candidate["timing"]["wall_seconds_median"] = .0001
    elif change == "category":
        candidate["category"] = "favorable" if candidate["category"] != "favorable" else "adverse"
    elif change == "selection":
        candidate["checkpoint_selection"] = "SELECTED_INITIALIZATION"
    else:
        frontier = next(row for row in frontiers if row["family"] == "premix" and row["seed"] == 11 and row["target"] == 2e-3)
        frontier["selected_steps"] = 2
    result = compare(protocol, candidates, frontiers, training)
    assert result["counts"]["incomplete_or_invalid"] > 0
    assert result["counts"]["timing_eligible"] < 96


def test_summary_writes_only_derived_files_and_preserves_source_bytes(evidence, tmp_path):
    protocol, candidates, frontiers, training = evidence
    documents = {"protocol.json": protocol, "candidates.json": {"schema": SOURCE_SCHEMA, "rows": candidates},
        "frontiers.json": {"schema": SOURCE_SCHEMA, "rows": frontiers}, "training.json": {"schema": SOURCE_SCHEMA, "rows": training},
        "summary.json": {"status": "COMPLETED", "untouched": True}, "neural_manifest.json": {"untouched": True}}
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value))
    before = {name: (tmp_path / name).read_bytes() for name in documents}
    counts = summarize(tmp_path)
    result = json.loads((tmp_path / "comparisons.json").read_text())
    assert counts == result["counts"]
    assert (tmp_path / "summary.txt").read_text().startswith("TDN premix paired neural review")
    assert not (tmp_path / "summary.txt.partial").exists()
    assert before == {name: (tmp_path / name).read_bytes() for name in documents}
    for source in result["sources"]:
        assert source["sha256"] == hashlib.sha256(before[source["path"]]).hexdigest()
    (tmp_path / "COMPLETED").write_text("sealed")
    with pytest.raises(ValueError, match="before sealing"):
        summarize(tmp_path)


def test_summary_missing_tables_stays_explicitly_incomplete(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps(build_protocol("smoke")))
    counts = summarize(tmp_path)
    result = json.loads((tmp_path / "comparisons.json").read_text())
    assert counts["timing_eligible"] == 0
    assert sum(source["status"] == "MISSING" for source in result["sources"]) == 3
    assert result["coverage"]["comparisons"] == {"expected": 576, "reported": 576}
