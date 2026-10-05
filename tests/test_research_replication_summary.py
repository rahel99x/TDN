"""Declared parents and seeds, rather than available successes, fix denominators."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tdn.research.replication_protocol import DEFAULT, FAMILIES, make_protocol, replicates, validate_config
from tdn.research.replication_summary import readable_replication_summary, summarize_replication


def protocol(*, smoke=False):
    return make_protocol(validate_config(DEFAULT, smoke=smoke), smoke=smoke, software={}, command=[])


def observations(declared, *, seconds=1., error=.001):
    records, blocks = [], []
    for index, plan in enumerate(declared["replicates"]):
        records.extend({**plan, "family": family, "status": "COMPLETED", "steps": 96,
                        "selected_step": 8, "selected_parameters_changed": True,
                        "parameter_count": 242, "training_seconds": 2.} for family in FAMILIES)
        for block in range(3):
            frontier, heldout = [], []
            parents = [row for row in declared["parents"] if row.get("diagnostic_block") == block]
            for parent in parents:
                for family in FAMILIES:
                    for h in declared["config"]["benchmark_steps"]:
                        frontier.append({"parent_id": parent["parent_id"], "family": family, "h": h,
                                         "status": "COMPLETED", "error": error, "error_upper": error + .00001,
                                         "feasible": True, "device": "cpu", "timing": {
                                             "wall_seconds_median": seconds * (index + 1) * (1. if family == "reaction_clock" else 2.)}})
                    for h in declared["config"]["heldout_horizons"]:
                        heldout.append({"parent_id": parent["parent_id"], "family": family, "h": h,
                                        "status": "COMPLETED", "one_step": {"rms": error, "upper": error + .00001},
                                        "two_step": {"rms": error, "upper": error + .00001}})
            blocks.append({**plan, "diagnostic_block": block, "frontier": frontier, "heldout": heldout,
                           "neural": {"parents": []}})
    return records, blocks


def endpoint(summary, family="reaction_clock", seed=74011):
    return next(row for row in summary["endpoint_rows"] if row["family"] == family and row["training_seed"] == seed)


def comparison(summary, family="generic_mlp", seed=74011):
    return next(row for row in summary["comparison_rows"] if row["baseline_family"] == family and row["training_seed"] == seed)


def test_full_grid_denominators_joint_endpoint_and_per_seed_pairing():
    declared = protocol()
    records, blocks = observations(declared)
    result = summarize_replication(declared, records, blocks)
    assert len(result["endpoint_rows"]) == 15 and len(result["comparison_rows"]) == 12
    assert result["diagnostic_parent_count"] == 27
    assert result["observed_block_result_count"] == result["expected_block_result_count"] == 9
    for row in result["endpoint_rows"]:
        assert row["parent_count"] == row["feasible_parent_count"] == row["robust_joint_parent_pass_count"] == 27
        assert row["rollout_expected"] == row["rollout_completed"] == row["rollout_pass"] == 108
        assert row["heldout_one_expected"] == row["heldout_one_completed"] == row["heldout_one_pass"] == 54
        assert row["heldout_two_expected"] == row["heldout_two_completed"] == row["heldout_two_pass"] == 54
        assert row["robust_joint_trained_parent_pass_count"] == 27
        assert all(group["parent_count"] == 9 for group in row["regime_counts"] + row["state_counts"])
    for row in result["comparison_rows"]:
        assert row["expected_parent_count"] == row["eligible"] == row["wins"] == 27
        assert row["speedup_min"] == row["speedup_median"] == row["speedup_max"] == 2.
        assert row["trained_pair"]
        assert row["heldout_one"]["ties"] == row["heldout_two"]["ties"] == 54
        assert row["same_h_rollout"]["ties"] == 108
    assert "not 81 independent" in result["pairing_scope"]
    assert len(json.dumps(result, indent=2, allow_nan=False).encode()) < 1024 * 1024


def test_wholly_missing_block_and_stratum_remain_in_expected_denominators():
    declared = protocol()
    records, blocks = observations(declared)
    blocks.pop(0)
    # Remove one additional parent's learned rows without leaving a classical row.
    missing_parent = blocks[0]["frontier"][0]["parent_id"]
    for field in ("frontier", "heldout"):
        blocks[0][field] = [row for row in blocks[0][field] if row["parent_id"] != missing_parent]
    result = summarize_replication(declared, records, blocks)
    row = endpoint(result)
    assert row["parent_count"] == 27 and row["rollout_expected"] == 108
    assert row["rollout_missing"] == 40 and row["rollout_completed"] == 68
    assert row["heldout_one_missing"] == row["heldout_two_missing"] == 20
    assert row["feasible_parent_count"] == row["robust_joint_parent_pass_count"] == 17
    assert comparison(result)["eligible"] == 17 and comparison(result)["ineligible"] == 10
    assert endpoint(result, seed=74021)["rollout_missing"] == 0
    assert result["observed_block_result_count"] == 8


def test_missing_all_work_retains_all_expected_family_seed_rows():
    result = summarize_replication(protocol(), [], [])
    assert len(result["endpoint_rows"]) == 15 and len(result["comparison_rows"]) == 12
    assert all(row["rollout_missing"] == 108 and row["heldout_one_missing"] == 54
               and row["heldout_two_missing"] == 54 and row["robust_joint_parent_pass_count"] == 0
               and row["selection"] == "UNKNOWN" for row in result["endpoint_rows"])
    assert all(row["eligible"] == 0 and row["ineligible"] == 27 and row["speedup_median"] is None
               for row in result["comparison_rows"])


def test_invalid_partial_nonfinite_and_above_tolerance_endpoint_coverage():
    declared = protocol()
    records, blocks = observations(declared)
    initial = blocks[0]
    chosen = [row for row in initial["heldout"] if row["family"] == "reaction_clock"]
    chosen[0].update(status="INVALID_TRAJECTORY", error_message="bounds")
    chosen[1]["one_step"]["upper"] = float("nan")
    chosen[2]["one_step"]["upper"] = .0487
    chosen[3]["two_step"]["upper"] = .00200001
    row = next(row for row in initial["frontier"] if row["family"] == "reaction_clock")
    row["error_upper"] = float("inf")
    result = summarize_replication(declared, records, blocks)
    first = endpoint(result)
    assert first["heldout_one_invalid"] == 2 and first["heldout_two_invalid"] == 1
    assert first["heldout_one_pass"] == 51 and first["heldout_two_pass"] == 52
    assert first["robust_joint_parent_pass_count"] == 25
    assert first["rollout_invalid"] == 1
    assert first["worst_one_step_upper_error"] == .0487
    assert first["worst_heldout_one"]["parent_id"] == chosen[2]["parent_id"]
    assert first["heldout_one_invalid_reasons"] == {"INVALID_OR_MISSING_ERROR": 1, "INVALID_TRAJECTORY": 1}
    json.dumps(result, allow_nan=False)


def test_joint_requires_both_horizons_and_both_steps_not_any_pass():
    declared = protocol(smoke=True)
    records, blocks = observations(declared)
    target = next(row for row in blocks[0]["heldout"] if row["family"] == "reaction_clock" and row["h"] == .11)
    target["two_step"]["upper"] = .003
    result = summarize_replication(declared, records, blocks)
    row = endpoint(result)
    assert row["heldout_one_pass"] == 18 and row["heldout_two_pass"] == 17
    assert row["robust_joint_parent_pass_count"] == 8
    assert row["rollout_expected"] == 36 and row["parent_count"] == 9


def test_initializations_failures_and_unverified_training_are_visible():
    declared = protocol()
    records, blocks = observations(declared)
    records[0].update(status="NUMERICAL_FAILURE", error="gradient failed", selected_step=None)
    records[1].update(selected_step=0, selected_parameters_changed=False)
    records[2].pop("selected_parameters_changed")
    records[3].update(status="NOT_STARTED", selected_step=None)
    records[4].update(training_seconds=float("inf"), parameter_count=True)
    result = summarize_replication(declared, records, blocks)
    assert endpoint(result)["selection"] == "TRAINING_FAILURE"
    assert endpoint(result)["failure_reason"] == "gradient failed"
    assert endpoint(result, "generic_mlp")["selection"] == "SELECTED_INITIALIZATION"
    assert endpoint(result, "residual_cnn_split")["selection"] == "UNKNOWN"
    assert endpoint(result, "unet_split")["status"] == "NOT_STARTED"
    assert endpoint(result, "fno_split")["training_seconds"] is None
    assert endpoint(result, "fno_split")["parameter_count"] is None
    # Actual finite trajectories remain visible; they do not turn initialization
    # or failed/unverified training evidence into a trained-model result.
    assert endpoint(result)["robust_joint_parent_pass_count"] == 27
    assert endpoint(result)["robust_joint_trained_parent_pass_count"] == 0
    assert not comparison(result)["trained_pair"]
    assert "TRAINING_FAILURE" in readable_replication_summary(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"feasible": False}, {"timing": {"wall_seconds_median": 0.}},
    {"timing": {"wall_seconds_median": float("nan")}},
    {"timing": {"wall_seconds_median": .00001, "soft_budget_passed": False}},
    {"error": .003, "error_upper": .00301},
])
def test_all_ineligible_timing_points_cannot_create_speed_wins(change):
    declared = protocol(smoke=True)
    records, blocks = observations(declared)
    for block in blocks:
        if block["training_seed"] == 74011:
            for row in block["frontier"]:
                if row["family"] == "generic_mlp":
                    row.update(deepcopy(change))
    result = summarize_replication(declared, records, blocks)
    assert comparison(result)["eligible"] == comparison(result)["wins"] == 0
    assert comparison(result)["ineligible"] == 9
    assert comparison(result, seed=74021)["wins"] == 9


def test_device_mismatch_and_extreme_ratio_not_eligible():
    declared = protocol(smoke=True)
    records, blocks = observations(declared)
    for block in blocks:
        if block["training_seed"] == 74011:
            for row in block["frontier"]:
                if row["family"] == "generic_mlp":
                    row["device"] = "cuda"
    result = summarize_replication(declared, records, blocks)
    assert comparison(result)["failure_reasons"] == {"comparison:DEVICE_MISMATCH_OR_MISSING": 9}
    for block in blocks:
        if block["training_seed"] == 74011:
            for row in block["frontier"]:
                row["device"] = "cpu"
                row["timing"]["wall_seconds_median"] = 1e-300 if row["family"] == "reaction_clock" else 1e300
    result = summarize_replication(declared, records, blocks)
    assert comparison(result)["eligible"] == 0
    assert comparison(result)["failure_reasons"] == {"comparison:NONFINITE_SPEEDUP": 9}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["block_duplicate", "record_duplicate", "wrong_seed", "wrong_block",
                                  "wrong_parent", "wrong_horizon", "row_duplicate", "extra_parent_plan"])
def test_bad_context_duplicates_and_undeclared_observations_are_rejected(case):
    declared = protocol(smoke=True)
    records, blocks = observations(declared)
    if case == "block_duplicate":
        blocks[-1] = deepcopy(blocks[0])
    elif case == "record_duplicate":
        records[-1] = deepcopy(records[0])
    elif case == "wrong_seed":
        blocks[0]["training_seed"] = 74021
    elif case == "wrong_block":
        blocks[0]["diagnostic_block"] = 3
    elif case == "wrong_parent":
        blocks[0]["frontier"][0]["parent_id"] = blocks[1]["frontier"][0]["parent_id"]
    elif case == "wrong_horizon":
        blocks[0]["frontier"][0]["h"] = .02
    elif case == "row_duplicate":
        blocks[0]["frontier"].append(deepcopy(blocks[0]["frontier"][0]))
    elif case == "extra_parent_plan":
        declared["parents"].append(deepcopy(declared["parents"][0]))
    with pytest.raises(ValueError):
        summarize_replication(declared, records, blocks)


def test_summary_readable_skip_and_import_without_scientific_dependencies():
    assert "No GPU numerical work" in readable_replication_summary({"status": "SKIPPED", "reason": "No pair"})
    root = Path(__file__).resolve().parents[1]
    check = subprocess.run([sys.executable, "-S", "-c",
                            "from tdn.research.replication_protocol import DEFAULT, make_protocol; "
                            "from tdn.research.replication_summary import summarize_replication; "
                            "import sys; assert 'torch' not in sys.modules; "
                            "p=make_protocol(DEFAULT,smoke=False,software={},command=[]); "
                            "s=summarize_replication(p,[],[]); assert len(s['endpoint_rows'])==15"],
                           cwd=root, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
