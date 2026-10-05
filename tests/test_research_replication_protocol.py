"""Fixed seed replication keeps teachers, horizons and parent splits intact."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from tdn.research.protocol import parent_plan as original_parent_plan
from tdn.research.replication_protocol import (
    DEFAULT, FAMILIES, SAMPLE_SCHEDULE_SEEDS, SUITE, TRAINING_SEEDS,
    make_protocol, parent_plan, replicates, trained_checkpoint_pairs, validate_config,
)


def checkpoint(family, index=0, **updates):
    return {**replicates()[index], "family": family, "status": "COMPLETED", "selected_step": 8,
            "selected_parameters_changed": True, **updates}


def test_fixed_families_budget_precision_and_seed_pairs():
    checked = validate_config(deepcopy(DEFAULT))
    protocol = make_protocol(checked, smoke=False, software={"python": "test"}, command=["test"])
    assert protocol["version"] == checked["protocol_version"] == 2
    assert protocol["benchmark_suite"] == SUITE == "neural-replication"
    assert checked["families"] == list(FAMILIES)
    assert checked["max_steps"] == 96 and checked["max_seconds"] == 1200
    assert checked["learning_rate"] == .001 and checked["width"] == 16
    assert protocol["state_precision"] == "float32" and protocol["teacher_precision"] == "float64"
    assert protocol["tf32"] is False
    assert protocol["replicates"] == [
        {"replicate_id": f"seed-{seed}", "training_seed": seed, "sample_schedule_seed": schedule}
        for seed, schedule in zip(TRAINING_SEEDS, SAMPLE_SCHEDULE_SEEDS)]
    assert protocol["reference"] == {"method": "coupled RK4 n,2n,4n", "error_fraction": .05,
                                      "tolerance_fraction": .1, "noise_floor": 1e-10}
    assert "not 81 independent observations" in protocol["diagnostic_design"]["pairing"]


def test_parent_disjoint_development_data_unchanged_and_diagnostics_fresh():
    parents = parent_plan()
    original = original_parent_plan()
    development = [row for row in parents if row["split"] != "diagnostic"]
    assert development == [row for row in original if row["split"] != "diagnostic"]
    assert len(development) == 18 and len(parents) == 45
    assert len({row["seed"] for row in parents}) == len(parents)
    assert len({row["parent_id"] for row in parents}) == len(parents)
    diagnostic = [row for row in parents if row["split"] == "diagnostic"]
    assert len(diagnostic) == 27
    assert not {row["seed"] for row in diagnostic} & {row["seed"] for row in original}
    for block in range(3):
        group = [row for row in diagnostic if row["diagnostic_block"] == block]
        assert {row["seed"] for row in group} == set(range(34001 + 100 * block, 34010 + 100 * block))
        assert len({(row["regime"], row["state_class"]) for row in group}) == 9
    assert all("diagnostic_block" not in row for row in development)


def test_smoke_retains_three_seeds_blocks_tolerances_and_independence():
    checked = validate_config(deepcopy(DEFAULT), smoke=True)
    assert checked["grid"] == [8, 8] and checked["max_steps"] == 2
    assert checked["validation_every"] == checked["timing_repeats"] == 1
    assert checked["max_seconds"] == 180
    for key in ("training_seeds", "sample_schedule_seeds", "diagnostic_replicates", "tolerance",
                "train_horizons", "heldout_horizons", "rollout_time", "benchmark_steps", "max_finest_substeps"):
        assert checked[key] == DEFAULT[key]
    parents = parent_plan(smoke=True)
    assert [sum(row["split"] == split for row in parents)
            for split in ("train", "validation", "diagnostic")] == [3, 3, 9]
    for block in range(3):
        group = [row for row in parents if row.get("diagnostic_block") == block]
        assert [row["seed"] for row in group] == [34001 + 100 * block, 34005 + 100 * block, 34009 + 100 * block]
    protocol = make_protocol(checked, smoke=True, software={}, command=[])
    assert protocol["diagnostic_design"]["independent_parent_count"] == 9


@pytest.mark.parametrize("key,value", [
    ("protocol_version", 1), ("grid", [16, 16]), ("max_steps", 97), ("max_steps", 95),
    ("max_seconds", 1201), ("max_seconds", 180), ("width", 32), ("learning_rate", .003),
    ("families", ["reaction_clock"]), ("tolerance", .003), ("train_horizons", [.02, .08]),
    ("heldout_horizons", [.02, .08]), ("rollout_time", .16), ("benchmark_steps", [.16]),
    ("training_seeds", [74011, 74011, 74031]), ("training_seeds", [74011, 74021, 74041]),
    ("sample_schedule_seeds", [74012, 74022, 74022]), ("diagnostic_replicates", 2),
    ("protocol_version", 2.), ("timing_repeats", 3.), ("training_seeds", [74011., 74021, 74031]),
])
def test_configuration_changes_rejected_even_for_smoke(key, value):
    config = deepcopy(DEFAULT)
    config[key] = value
    with pytest.raises(ValueError, match="fixes"):
        validate_config(config, smoke=True)


def test_extra_missing_and_nonfinite_fields_rejected_and_validation_does_not_mutate():
    config = deepcopy(DEFAULT)
    checked = validate_config(config, smoke=True)
    checked["training_seeds"].append(999)
    assert config == DEFAULT
    with pytest.raises(ValueError, match="exactly"):
        validate_config({**config, "queue_cap": 1})
    config.pop("training_seeds")
    with pytest.raises(ValueError, match="exactly"):
        validate_config(config)
    with pytest.raises(ValueError, match="finite JSON"):
        validate_config({**DEFAULT, "learning_rate": float("nan")})


def test_checkpoint_gate_pairs_clock_and_baseline_within_same_declared_seed():
    records = [checkpoint("reaction_clock", 0), checkpoint("generic_mlp", 1)]
    assert trained_checkpoint_pairs(records) == []
    records.extend([checkpoint("reaction_clock", 1), checkpoint("fno_split", 0),
                    checkpoint("reaction_clock", 2), checkpoint("unet_split", 2)])
    assert trained_checkpoint_pairs(records) == [row["replicate_id"] for row in replicates()]


@pytest.mark.parametrize("updates", [
    {"selected_step": 0}, {"selected_step": True}, {"selected_step": 8.},
    {"selected_parameters_changed": False}, {"selected_parameters_changed": 1},
    {"status": "NUMERICAL_FAILURE"}, {"status": "NOT_STARTED"},
])
def test_checkpoint_gate_does_not_promote_initialization_or_ambiguous_evidence(updates):
    assert trained_checkpoint_pairs([checkpoint("reaction_clock"), checkpoint("generic_mlp", **updates)]) == []


@pytest.mark.parametrize("updates", [
    {"training_seed": 74021}, {"sample_schedule_seed": 74022},
    {"training_seed": 74011.}, {"replicate_id": "seed-unknown"}, {"family": "transport"},
])
def test_checkpoint_gate_rejects_mismatched_context(updates):
    with pytest.raises(ValueError, match="undeclared|match"):
        trained_checkpoint_pairs([{**checkpoint("reaction_clock"), **updates}])


def test_checkpoint_gate_rejects_duplicates_and_is_stdlib_only():
    row = checkpoint("reaction_clock")
    with pytest.raises(ValueError, match="Duplicate"):
        trained_checkpoint_pairs([row, row])
    root = Path(__file__).resolve().parents[1]
    check = subprocess.run([sys.executable, "-S", "-c",
                            "from tdn.research.replication_protocol import DEFAULT, validate_config, trained_checkpoint_pairs; "
                            "import sys; assert 'torch' not in sys.modules; "
                            "assert validate_config(DEFAULT)['protocol_version']==2; "
                            "assert trained_checkpoint_pairs([])==[]"],
                           cwd=root, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
