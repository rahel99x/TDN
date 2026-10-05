"""Prespecified seed replication of the compact neural screen; stdlib only."""
from __future__ import annotations

import json

from .protocol import (
    DEFAULT as ARCHITECTURE_DEFAULT, REGIMES, STATE_CLASSES,
    assert_parent_disjoint,
)

VERSION = 2
SUITE = "neural-replication"
FAMILIES = ("reaction_clock", "generic_mlp", "residual_cnn_split", "unet_split", "fno_split")
TRAINING_SEEDS = (74011, 74021, 74031)
SAMPLE_SCHEDULE_SEEDS = (74012, 74022, 74032)
DIAGNOSTIC_REPLICATES = 3
DEFAULT = {
    **ARCHITECTURE_DEFAULT,
    "protocol_version": VERSION, "families": list(FAMILIES), "learning_rate": .001,
    "training_seeds": list(TRAINING_SEEDS),
    "sample_schedule_seeds": list(SAMPLE_SCHEDULE_SEEDS),
    "diagnostic_replicates": DIAGNOSTIC_REPLICATES,
}


def validate_config(config: dict, *, smoke: bool = False) -> dict:
    """Accept the fixed bounded experiment, never a seed or horizon sweep."""
    if not isinstance(config, dict) or set(config) != set(DEFAULT):
        raise ValueError("Replication configuration must contain exactly the documented protocol fields")
    try:
        checked = json.loads(json.dumps(config, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Replication configuration must contain finite JSON values") from error
    for key, planned in DEFAULT.items():
        if checked[key] != planned or type(checked[key]) is not type(planned):
            raise ValueError(f"Replication protocol fixes {key}; do not weaken or expand the comparison")
        if isinstance(planned, list) and any(type(actual) is not type(expected)
                                            for actual, expected in zip(checked[key], planned)):
            raise ValueError(f"Replication protocol fixes {key} element types")
    if smoke:
        checked.update(grid=[8, 8], max_steps=2, validation_every=1,
                       timing_repeats=1, max_seconds=180)
    return checked


def replicates() -> list[dict]:
    return [{"replicate_id": f"seed-{seed}", "training_seed": seed,
             "sample_schedule_seed": schedule}
            for seed, schedule in zip(TRAINING_SEEDS, SAMPLE_SCHEDULE_SEEDS)]


def trained_checkpoint_pairs(training_records) -> list[str]:
    """Return same-seed eligible clock/baseline pairs without scientific imports.

    An incomplete list is permitted for interrupted work. Declared seed contexts
    and family uniqueness are still checked; a clock from one seed cannot be
    combined with a baseline from another seed to authorize frozen GPU timing.
    """
    if not isinstance(training_records, list) or len(training_records) > 15:
        raise ValueError("Replication checkpoint records exceed the bounded family/seed plan")
    plans = {row["replicate_id"]: row for row in replicates()}
    records = {}
    for row in training_records:
        if not isinstance(row, dict) or not isinstance(row.get("replicate_id"), str) or row["replicate_id"] not in plans:
            raise ValueError("Replication checkpoint record has an undeclared replicate_id")
        identity = row["replicate_id"]
        for key in ("training_seed", "sample_schedule_seed"):
            if type(row.get(key)) is not int or row[key] != plans[identity][key]:
                raise ValueError(f"Replication checkpoint record does not match its declared {key}")
        family = row.get("family")
        if not isinstance(family, str) or family not in FAMILIES:
            raise ValueError("Replication checkpoint record has an undeclared family")
        if (identity, family) in records:
            raise ValueError("Duplicate replication checkpoint record")
        records[identity, family] = row

    def trained(identity, family):
        row = records.get((identity, family), {})
        return (row.get("status") == "COMPLETED" and type(row.get("selected_step")) is int
                and row["selected_step"] > 0 and row.get("selected_parameters_changed") is True)

    return [identity for identity in plans if trained(identity, "reaction_clock")
            and any(trained(identity, family) for family in FAMILIES if family != "reaction_clock")]


def parent_plan(*, smoke: bool = False) -> list[dict]:
    """Keep development data fixed and add three fresh diagnostic blocks.

    A block contains all nine regime/state strata, or the three diagonal smoke
    strata. Every trained seed is measured on the same diagnostic parents; those
    repeated measurements are paired and do not create new independent parents.
    """
    result = []
    for split, offset, block in [("train", 31000, None), ("validation", 32000, None),
                                 *(("diagnostic", 34000 + 100 * index, index)
                                   for index in range(DIAGNOSTIC_REPLICATES))]:
        for i, (regime, kappa, rate) in enumerate(REGIMES):
            for j, state_class in enumerate(STATE_CLASSES):
                if smoke and i != j:
                    continue
                seed = offset + 3 * i + j + 1
                parent = {"parent_id": f"{split}-{regime}-{state_class}-{seed}",
                          "split": split, "seed": seed, "state_class": state_class,
                          "regime": regime, "kappa": kappa, "reaction_rate": rate}
                if block is not None:
                    parent["diagnostic_block"] = block
                result.append(parent)
    assert_parent_disjoint(result)
    return result


def make_protocol(config: dict, *, smoke: bool, software: dict, command: list[str]) -> dict:
    """Seal the existing physical/reference rules and the new replication plan."""
    expected = validate_config(DEFAULT, smoke=smoke)
    if config != expected:
        raise ValueError("Replication protocol requires its validated fixed configuration")
    result = {
        "version": VERSION, "config": config, "smoke": smoke,
        "benchmark_suite": SUITE, "parents": parent_plan(smoke=smoke),
        "replicates": replicates(), "diagnostic_replicates": DIAGNOSTIC_REPLICATES,
        "software": software, "command": command,
        "state_precision": "float32", "teacher_precision": "float64", "tf32": False,
        "normalization": "training initial states only; shared across all learned families and seeds",
        "rate_sharing": "one learned shared rate across all training regimes; no per-parent rate oracle",
        "training_loss": "physical one-step MSE + 0.5*two-step MSE, divided by tolerance squared",
        "checkpoint_selection": "minimum validation loss only, including initialization; diagnostics after freeze",
        "training_budget": {
            "optimizer": "Adam", "learning_rate": config["learning_rate"],
            "maximum_updates_per_family": config["max_steps"],
            "maximum_updates_per_family_per_seed": config["max_steps"],
            "batch": "one identical scheduled training parent/horizon per update within each seed",
            "initialization_seeds": list(TRAINING_SEEDS),
            "sample_schedule_seeds": list(SAMPLE_SCHEDULE_SEEDS),
            "matched": "data, updates, objective, validation schedule, horizons and device",
            "unmatched": "parameter count, FLOPs and training walltime; recorded per family",
            "scope": "three prespecified training seeds; five fixed families; bounded replication, no tuning sweep",
        },
        "reference": {"method": "coupled RK4 n,2n,4n", "error_fraction": .05,
                      "tolerance_fraction": .1, "noise_floor": 1e-10},
        "headroom_rule": "both split and Richardson error minus reference uncertainty >0.002 at h=.32,T=.32",
        "cost_rule": "whole FP32 rollout, state checks included, warmup1, all declared step sizes, no fallback",
    }
    result["diagnostic_design"] = {
        "independent_parent_count": 9 if smoke else 27,
        "blocks": DIAGNOSTIC_REPLICATES,
        "pairing": (f"all three training seeds use the same diagnostic parents; {9 if smoke else 27} parents, "
                    f"not {27 if smoke else 81} independent observations"),
        "robust_joint_endpoint": "all heldout horizons pass upper-error tolerance for both one-step and two-step trajectories",
        "missing_or_invalid": "never counted as a pass; all declared families, seeds, parents and horizons retained",
        "classical_repetitions": "same parent/method repeated across training seeds; descriptive timing repeats, not independent samples",
        "inference": "descriptive per-seed endpoints and ranges only; no significance or SOTA claim",
    }
    result["headroom_rule"] += "; descriptive classical reference only, not a neural replication eligibility gate"
    result["scope"] = "bounded diagnostic replication; no G0-G6 authorization or confirmatory claim"
    return result
