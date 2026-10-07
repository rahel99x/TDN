"""Declared architecture/constraint pairs, diagnostics and missing evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from tdn.analysis.consistency.comparison import compare, summarize, PAIR_PLAN, SOURCE_SCHEMA, FAMILIES
from tdn.analysis.premix.neural import build_protocol


def frontiers(protocol, candidates):
    grouped = {}
    for row in candidates:
        grouped.setdefault((row["parent_id"], row["seed"], row["family"]), []).append(row)
    result = []
    for rows in grouped.values():
        for norm in ("rms", "max"):
            for target in protocol["targets"]:
                eligible = [row for row in rows if row["status"] == "COMPLETED" and row.get("memory_ok")
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
    protocol.update(schema=SOURCE_SCHEMA, families=list(FAMILIES), seeds=[11, 22], targets=[2e-3, 2e-5], timing_repeats=3)
    protocol["parents"] = [row for row in protocol["parents"] if row["split"] != "diagnostic"] + [
        row for row in protocol["parents"] if row["regime"] in ("smooth", "mixed") and row["split"] == "diagnostic"]
    training = [{"seed": seed, "family": family, "status": "COMPLETED", "selected_step": 2,
                 "selected_parameters_changed": True, "selection": "TRAINED_CHECKPOINT"}
                for seed in protocol["seeds"] for family in protocol["families"]]
    methods = [(None, family) for family in protocol["classical"]] + [(seed, family) for seed in protocol["seeds"] for family in protocol["families"]]
    candidates = []
    for parent in protocol["parents"]:
        if parent["split"] != "diagnostic":
            continue
        for seed, family in methods:
            for steps in protocol["step_counts"]:
                time = (1 + methods.index((seed, family))) * .01 * steps
                error = 3e-4 if steps == 1 else 1e-7
                candidates.append({**{key: parent[key] for key in ("parent_id", "regime", "category", "distribution", "grid")},
                    "seed": seed, "family": family, "steps": steps, "horizon": parent["horizons"][0],
                    "status": "COMPLETED", "admissible": True, "memory_ok": True, "device": "cpu",
                    "checkpoint_selection": "TRAINED_CHECKPOINT" if seed is not None else None,
                    "rms": error, "max_error": error, "uncertainty_rms": 1e-10, "uncertainty_max_bound": 1e-9,
                    "upper_rms": error + 1e-10, "upper_max": error + 1e-9,
                    "signed_mean_error": -error / 2, "mean_error": error / 2, "spatial_rms": error * .8,
                    "base_rms": error * 2, "base_max_error": error * 2, "base_mean_error": error,
                    "base_spatial_rms": error * 1.6, "rms_vs_base": .5, "max_vs_base": .5,
                    "mean_vs_base": .5, "spatial_vs_base": .5,
                    "timing": {"wall_seconds_median": time, "wall_seconds_raw": [.9 * time, time, 1.1 * time], "device": "cpu"}})
    return protocol, candidates, frontiers(protocol, candidates), training


def test_all_declared_matched_constraints_and_within_backbone_pairs(evidence):
    result = compare(*evidence)
    assert result["schema"] == "tdn.consistency-comparisons/v1"
    assert result["coverage"]["comparisons"] == {"expected": 160, "reported": 160}
    assert result["coverage"]["complete_source_plan"]
    assert result["counts"]["timing_eligible"] == 160
    assert {(row["ours"], row["baseline"], row["comparison_kind"]) for row in result["rows"]} == set(PAIR_PLAN)
    for row in result["rows"]:
        selected = evidence[1][int(row["ours_candidate_record"].rsplit("/", 1)[1])]
        assert row["ours_signed_mean_error"] == selected["signed_mean_error"] < 0
        assert row["ours_spatial_rms"] == selected["spatial_rms"]
        assert row["ours_base_rms"] == selected["base_rms"]
        assert row["ours_rms_vs_base"] == .5
        assert row["ours_rms_regresses_base"] is False
        assert row["ours_diagnostic_issues"] == []
    assert {row["group_scope"] for row in result["groups"]} == {"all", "category", "regime"}
    assert all(len(set(row["parent_ids"])) == row["counts"]["total"] for row in result["groups"])


def test_untrained_selection_visible_across_all_declared_pairs(evidence):
    protocol, candidates, derived, training = evidence
    record = next(row for row in training if row["seed"] == 11 and row["family"] == "premix_moment")
    record.update(selected_step=0, selected_parameters_changed=False, selection="SELECTED_INITIALIZATION")
    for row in candidates:
        if row["seed"] == 11 and row["family"] == "premix_moment":
            row["checkpoint_selection"] = "SELECTED_INITIALIZATION"
    result = compare(protocol, candidates, derived, training)
    initialized = [row for row in result["rows"] if "SELECTED_INITIALIZATION" in (row["ours_selection"], row["baseline_selection"])]
    assert len(initialized) == 24
    assert all(row["timing_eligible"] and not row["trained_pair_eligible"] for row in initialized)
    assert result["counts"]["initialization_in_eligible_pair"] == 24


def test_missing_and_infeasible_are_not_speed_victories(evidence):
    protocol, candidates, _, training = evidence
    for row in candidates:
        if row["family"] == "fno_moment":
            row.update(rms=.1, max_error=.1, upper_rms=.1 + 1e-10, upper_max=.1 + 1e-9)
    derived = frontiers(protocol, candidates)
    candidates.pop(next(i for i, row in enumerate(candidates) if row["family"] == "premix_gated" and row["seed"] == 11))
    result = compare(protocol, candidates, derived, training)
    assert not result["coverage"]["complete_source_plan"]
    unmatched = [row for row in result["rows"] if row["comparison_kind"] == "matched_moment"]
    assert all(row["ours_feasible"] and not row["baseline_feasible"] and row["outcome"] is None for row in unmatched)
    assert all(row["speedup_baseline_over_ours"] is None for row in result["rows"] if not row["timing_eligible"])


def test_invalid_optional_diagnostic_does_not_hide_timing_or_claim_regression(evidence):
    protocol, candidates, derived, training = evidence
    for row in candidates:
        if row["family"] == "premix_gated":
            row["base_rms"] = -1
    result = compare(protocol, candidates, derived, training)
    affected = [row for row in result["rows"] if row["ours"] == "premix_gated"]
    assert all(row["ours_base_rms"] is None and row["ours_rms_regresses_base"] is None for row in affected)
    assert all("INVALID_BASE_RMS" in row["ours_diagnostic_issues"] for row in affected)
    assert all(row["timing_eligible"] for row in affected)


def test_empty_observations_retain_all_comparisons(evidence):
    result = compare(evidence[0], [], [], [])
    assert result["coverage"]["comparisons"] == {"expected": 160, "reported": 160}
    assert result["counts"]["incomplete_or_invalid"] == 160
    assert all(row["ours_base_rms"] is None for row in result["rows"])


def test_summarize_hashes_canonical_sources_and_preserves_documents(tmp_path, evidence):
    names = ("protocol.json", "candidates.json", "frontiers.json", "training.json")
    original = {}
    for name, rows in zip(names, evidence):
        document = rows if name == "protocol.json" else {"schema": SOURCE_SCHEMA, "rows": rows}
        original[name] = json.dumps(document).encode()
        (tmp_path / name).write_bytes(original[name])
    assert summarize(tmp_path)["total"] == 160
    assert original == {name: (tmp_path / name).read_bytes() for name in names}
    result = json.loads((tmp_path / "comparisons.json").read_text())
    for source in result["sources"]:
        assert source["sha256"] == hashlib.sha256(original[source["path"]]).hexdigest()
    assert "consistency" in (tmp_path / "summary.txt").read_text()
    (tmp_path / "COMPLETED").write_text("frozen")
    with pytest.raises(ValueError, match="Preserve"):
        summarize(tmp_path)


@pytest.mark.parametrize("change", ["families", "schema", "duplicate"])
def test_rejects_wrong_plan_and_duplicate_identities(evidence, change):
    protocol, candidates, derived, training = deepcopy(evidence)
    if change == "families":
        protocol["families"] = list(reversed(protocol["families"]))
    elif change == "schema":
        protocol["schema"] = "tdn.premix-neural/v1"
    else:
        candidates.append(candidates[0])
    with pytest.raises(ValueError):
        compare(protocol, candidates, derived, training)


@pytest.mark.parametrize("bad", [
    '{"schema":"tdn.consistency-neural/v1","rows":[{"rms":NaN}]}',
    '{"schema":"tdn.consistency-neural/v1","rows":[{"rms":1e999}]}',
    '{"schema":"tdn.consistency-neural/v1","rows":[],"rows":[]}'])
def test_summarize_rejects_nonfinite_or_duplicate_canonical_sources(tmp_path, evidence, bad):
    (tmp_path / "protocol.json").write_text(json.dumps(evidence[0]))
    (tmp_path / "candidates.json").write_text(bad)
    with pytest.raises(ValueError):
        summarize(tmp_path)
    assert not (tmp_path / "comparisons.json").exists()
