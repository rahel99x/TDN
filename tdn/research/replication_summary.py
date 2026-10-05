"""Failure-preserving descriptive replication endpoints; standard library only."""
from __future__ import annotations

from collections import Counter
from statistics import median

from .neural_comparison import _error, _finite, _frontier_issue, _relation
from .replication_protocol import (
    DEFAULT, FAMILIES, SUITE, VERSION, parent_plan, replicates, validate_config,
)

_MAX_BLOCK_ROWS = 1024
_FAILURES = {"FAILED", "NUMERICAL_FAILURE", "INTERRUPTED"}


def _replicate_identity(row, plans):
    if not isinstance(row, dict):
        raise ValueError("Replication records must be dictionaries")
    identity = row.get("replicate_id")
    if not isinstance(identity, str) or identity not in plans:
        raise ValueError("Replication record has an undeclared replicate_id")
    plan = plans[identity]
    for key in ("training_seed", "sample_schedule_seed"):
        if type(row.get(key)) is not int or row[key] != plan[key]:
            raise ValueError(f"Replication record does not match its declared {key}")
    return identity


def _selection(row):
    status = row.get("status", "NOT_RECORDED")
    if not isinstance(status, str):
        status = "INVALID_TRAINING_STATUS"
    selected = row.get("selected_step")
    selected = selected if type(selected) is int and selected >= 0 else None
    changed = row.get("selected_parameters_changed")
    if status in _FAILURES:
        selection = "TRAINING_FAILURE"
    elif status == "COMPLETED" and (selected == 0 or changed is False):
        selection = "SELECTED_INITIALIZATION"
    elif status == "COMPLETED" and selected is not None and selected > 0 and changed is True:
        selection = "TRAINED_CHECKPOINT"
    else:
        selection = "UNKNOWN"
    return {
        "status": str(status)[:512], "selection": selection, "selected_step": selected,
        "optimizer_steps": row.get("steps") if type(row.get("steps")) is int and row["steps"] >= 0 else None,
        "parameter_count": row.get("parameter_count") if type(row.get("parameter_count")) is int and row["parameter_count"] >= 0 else None,
        "training_seconds": float(row["training_seconds"]) if _finite(row.get("training_seconds")) else None,
        "failure_reason": str(row["error"])[:512] if row.get("error") else None,
    }


def _index_rows(rows, kind, parents, families, horizons):
    if not isinstance(rows, list) or len(rows) > _MAX_BLOCK_ROWS:
        raise ValueError(f"Replication {kind} exceeds its bounded row input")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Replication {kind} rows must be dictionaries")
        parent, family, h = row.get("parent_id"), row.get("family"), row.get("h")
        if not isinstance(parent, str) or parent not in parents:
            raise ValueError(f"Replication {kind} has an undeclared or wrong-block parent")
        # Existing evaluation also records these parameter-free classical controls.
        if not isinstance(family, str) or family not in (*families, "split", "richardson_split", "adaptive_split", "coupled_rk4", "e3_anchor"):
            raise ValueError(f"Replication {kind} has an undeclared family")
        if not _finite(h, positive=True) or h not in horizons:
            raise ValueError(f"Replication {kind} has an undeclared or nonfinite horizon")
        key = (parent, family, float(h))
        if key in result:
            raise ValueError(f"Duplicate replication {kind} observation")
        result[key] = row
    return result


def _coverage(observations, step, tolerance):
    counts = {"expected": len(observations), "completed": 0, "invalid": 0, "missing": 0, "pass": 0}
    reasons = Counter()
    worst = None
    for parent, h, row in observations:
        if row is None:
            counts["missing"] += 1
            continue
        error = _error(row, step)
        if error is None:
            counts["invalid"] += 1
            reason = (str(row.get("status", "MISSING_STATUS"))[:512]
                      if row.get("status") != "COMPLETED" else "INVALID_OR_MISSING_ERROR")
            reasons[reason] += 1
            continue
        counts["completed"] += 1
        counts["pass"] += error["upper"] <= tolerance
        if worst is None or error["upper"] > worst["upper_error"]:
            worst = {"upper_error": error["upper"], "rms_error": error["rms"],
                     "parent_id": parent["parent_id"], "h": h, "regime": parent["regime"],
                     "state_class": parent["state_class"], "diagnostic_block": parent["diagnostic_block"]}
    return counts, dict(sorted(reasons.items())), worst


def _feasible(row, tolerance):
    return (_frontier_issue(row) is None and _error(row)["upper"] <= tolerance)


def _best(rows, parent, family, horizons, tolerance):
    feasible = [row for h in horizons if (row := rows.get((parent, family, h)))
                and _feasible(row, tolerance)]
    return min(feasible, key=lambda row: (row["timing"]["wall_seconds_median"], row["h"])) if feasible else None


def _joint_pass(rows, parent, family, horizons, tolerance):
    return all((error := _error(rows.get((parent, family, h)), step)) is not None
               and error["upper"] <= tolerance for h in horizons for step in ("one_step", "two_step"))


def _endpoint(plan, family, record, parents, frontier, heldout, config):
    identity = {**plan, "family": family}
    row = {**identity, **_selection(record), "parent_count": len(parents)}
    for prefix, source, horizons, step in (
        ("rollout", frontier, config["benchmark_steps"], None),
        ("heldout_one", heldout, config["heldout_horizons"], "one_step"),
        ("heldout_two", heldout, config["heldout_horizons"], "two_step"),
    ):
        observations = [(parent, h, source.get((parent["parent_id"], family, h))) for parent in parents for h in horizons]
        counts, reasons, worst = _coverage(observations, step, config["tolerance"])
        row.update({f"{prefix}_{key}": value for key, value in counts.items()})
        row[f"{prefix}_invalid_reasons"] = reasons
        row[f"worst_{prefix}"] = worst
        label = {"rollout": "rollout", "heldout_one": "one_step", "heldout_two": "two_step"}[prefix]
        row[f"worst_{label}_upper_error"] = worst["upper_error"] if worst else None
    feasible = {parent["parent_id"] for parent in parents if _best(frontier, parent["parent_id"], family,
                                                                   config["benchmark_steps"], config["tolerance"])}
    joint = {parent["parent_id"] for parent in parents if _joint_pass(heldout, parent["parent_id"], family,
                                                                    config["heldout_horizons"], config["tolerance"])}
    row.update(feasible_parent_count=len(feasible), robust_joint_parent_pass_count=len(joint),
               trained_checkpoint=row["selection"] == "TRAINED_CHECKPOINT",
               robust_joint_trained_parent_pass_count=len(joint) if row["selection"] == "TRAINED_CHECKPOINT" else 0)
    for field, output in (("regime", "regime_counts"), ("state_class", "state_counts")):
        groups = []
        for value in sorted({parent[field] for parent in parents}):
            subset = [parent for parent in parents if parent[field] == value]
            group = {field: value, "parent_count": len(subset),
                     "feasible_parent_count": sum(parent["parent_id"] in feasible for parent in subset),
                     "robust_joint_parent_pass_count": sum(parent["parent_id"] in joint for parent in subset)}
            for prefix, source, horizons, step in (
                ("rollout", frontier, config["benchmark_steps"], None),
                ("heldout_one", heldout, config["heldout_horizons"], "one_step"),
                ("heldout_two", heldout, config["heldout_horizons"], "two_step"),
            ):
                counts, _, _ = _coverage([(parent, h, source.get((parent["parent_id"], family, h)))
                                         for parent in subset for h in horizons], step, config["tolerance"])
                group.update({f"{prefix}_{key}": number for key, number in counts.items()})
            groups.append(group)
        row[output] = groups
    return row


def _accuracy_counts(frontier, heldout, parents, family, config):
    result = {}
    for name, source, horizons, step in (
        ("same_h_rollout", frontier, config["benchmark_steps"], None),
        ("heldout_one", heldout, config["heldout_horizons"], "one_step"),
        ("heldout_two", heldout, config["heldout_horizons"], "two_step"),
    ):
        counts = {"expected": len(parents) * len(horizons), "eligible": 0,
                  "wins": 0, "losses": 0, "ties": 0, "ineligible": 0}
        for parent in parents:
            for h in horizons:
                first, second = [_error(source.get((parent["parent_id"], current, h)), step)
                                 for current in ("reaction_clock", family)]
                if first is None or second is None:
                    counts["ineligible"] += 1
                else:
                    counts["eligible"] += 1
                    relation = _relation(first["upper"], second["upper"], smaller_better=True)
                    counts[{"WIN": "wins", "LOSS": "losses", "TIE": "ties"}[relation]] += 1
        result[name] = counts
    return result


def _comparison(plan, family, selections, parents, frontier, heldout, config):
    ratios, outcomes, failures = [], Counter(), Counter()
    blocks = []
    for block in range(3):
        block_ratios, block_outcomes = [], Counter()
        subset = [parent for parent in parents if parent["diagnostic_block"] == block]
        for parent in subset:
            best = [_best(frontier, parent["parent_id"], current, config["benchmark_steps"], config["tolerance"])
                    for current in ("reaction_clock", family)]
            if any(row is None for row in best):
                for label, row in zip(("tdn", "baseline"), best):
                    if row is None:
                        failures[f"{label}:NO_FEASIBLE_OR_MISSING_ROLLOUT"] += 1
                continue
            if best[0].get("device") != best[1].get("device") or not best[0].get("device"):
                failures["comparison:DEVICE_MISMATCH_OR_MISSING"] += 1
                continue
            ratio = best[1]["timing"]["wall_seconds_median"] / best[0]["timing"]["wall_seconds_median"]
            if not _finite(ratio, positive=True):
                failures["comparison:NONFINITE_SPEEDUP"] += 1
                continue
            relation = _relation(ratio, 1., smaller_better=False)
            block_outcomes[relation] += 1
            block_ratios.append(ratio)
        ratios.extend(block_ratios)
        outcomes.update(block_outcomes)
        blocks.append({"diagnostic_block": block, "expected_parent_count": len(subset),
                       "eligible": len(block_ratios), "wins": block_outcomes["WIN"],
                       "losses": block_outcomes["LOSS"], "ties": block_outcomes["TIE"],
                       "speedup_median": median(block_ratios) if block_ratios else None})
    return {
        **plan, "tdn_family": "reaction_clock", "baseline_family": family,
        "expected_parent_count": len(parents), "eligible": len(ratios),
        "wins": outcomes["WIN"], "losses": outcomes["LOSS"], "ties": outcomes["TIE"],
        "ineligible": len(parents) - len(ratios), "failure_reasons": dict(sorted(failures.items())),
        "speedup_median": median(ratios) if ratios else None,
        "speedup_min": min(ratios) if ratios else None, "speedup_max": max(ratios) if ratios else None,
        "trained_pair": all(selections[current]["selection"] == "TRAINED_CHECKPOINT" for current in ("reaction_clock", family)),
        "tdn_training_selection": selections["reaction_clock"]["selection"],
        "baseline_training_selection": selections[family]["selection"],
        "block_rows": blocks, **_accuracy_counts(frontier, heldout, parents, family, config),
    }


def summarize_replication(protocol, training_records, block_results):
    """Retain every declared endpoint, including wholly absent parents/blocks.

    Raw frontier/heldout rows are authoritative. Existing neural summaries inside
    blocks are accepted as accompanying detail, never used to infer coverage.
    """
    if not isinstance(protocol, dict) or protocol.get("version") != VERSION or protocol.get("benchmark_suite") != SUITE:
        raise ValueError("Replication summary requires the version2 replication protocol")
    smoke = protocol.get("smoke")
    if type(smoke) is not bool:
        raise ValueError("Replication summary requires an explicit smoke flag")
    config = protocol.get("config")
    if config != validate_config(DEFAULT, smoke=smoke) or protocol.get("parents") != parent_plan(smoke=smoke):
        raise ValueError("Replication summary requires the fixed config and declared parent plan")
    if protocol.get("replicates") != replicates():
        raise ValueError("Replication summary requires the three declared seed pairs")
    plans = {row["replicate_id"]: row for row in protocol["replicates"]}
    if not isinstance(training_records, list) or len(training_records) > 15:
        raise ValueError("Replication training records exceed the bounded family/seed plan")
    training = {}
    for row in training_records:
        identity = _replicate_identity(row, plans)
        family = row.get("family")
        if not isinstance(family, str) or family not in FAMILIES:
            raise ValueError("Replication training record has an undeclared family")
        key = identity, family
        if key in training:
            raise ValueError("Duplicate replication training record")
        training[key] = row
    if not isinstance(block_results, list) or len(block_results) > 9:
        raise ValueError("Replication block results exceed the bounded seed/block plan")
    parents = [row for row in protocol["parents"] if row["split"] == "diagnostic"]
    combined = {identity: ({}, {}) for identity in plans}
    seen_blocks = set()
    for block in block_results:
        identity = _replicate_identity(block, plans)
        index = block.get("diagnostic_block")
        if type(index) is not int or not 0 <= index < 3:
            raise ValueError("Replication result has an undeclared diagnostic block")
        if (identity, index) in seen_blocks:
            raise ValueError("Duplicate replication seed/block result")
        seen_blocks.add((identity, index))
        expected = {parent["parent_id"] for parent in parents if parent["diagnostic_block"] == index}
        for target, name, horizons in zip(combined[identity], ("frontier", "heldout"),
                                         (config["benchmark_steps"], config["heldout_horizons"])):
            target.update(_index_rows(block.get(name, []), name, expected, FAMILIES, horizons))
    endpoints, comparisons = [], []
    for identity, plan in plans.items():
        frontier, heldout = combined[identity]
        selections = {family: _selection(training.get((identity, family), {})) for family in FAMILIES}
        endpoints.extend(_endpoint(plan, family, training.get((identity, family), {}), parents,
                                   frontier, heldout, config) for family in FAMILIES)
        comparisons.extend(_comparison(plan, family, selections, parents, frontier, heldout, config)
                           for family in FAMILIES if family != "reaction_clock")
    return {
        "version": 1, "protocol_version": VERSION, "benchmark_suite": SUITE, "smoke": smoke,
        "families": list(FAMILIES), "replicates": replicates(),
        "diagnostic_parent_count": len(parents), "diagnostic_blocks": 3,
        "expected_block_result_count": 9, "observed_block_result_count": len(seen_blocks),
        "training_record_count": len(training), "endpoint_rows": endpoints, "comparison_rows": comparisons,
        "scope": "Prespecified bounded descriptive replication; no significance, SOTA, ranking or G0-G6 authorization.",
        "pairing_scope": f"{len(parents)} distinct diagnostic parents reused across three training seeds; paired observations, not {3 * len(parents)} independent parents.",
        "classical_scope": "Repeated classical timings across training seeds reuse the same parents; not independent samples.",
        "endpoint_scope": "Pass requires finite completed upper error <= fixed tolerance; invalid/missing is never a pass. Robust joint parent pass requires both heldout horizons for both one-step and two-step.",
        "timing_scope": "Per-seed speedup is baseline/clock wall time using each parent's fastest feasible declared step size; post-hoc finite frontier, no deployable step selector.",
        "selection_scope": "Initialization and failed/unrecorded training selections remain visible; trained_pair and robust_joint_trained_parent_pass_count require positive selected step and verified parameter changes.",
        "missing_scope": "Expected families, seeds, parents and horizons come from the sealed protocol, including wholly absent blocks or strata.",
    }


def readable_replication_summary(summary) -> str:
    """Short progress/report lines, with full failure coverage in the JSON."""
    if summary.get("status") == "SKIPPED":
        return f"TDN neural replication: SKIPPED\n{summary.get('reason', 'No eligible frozen checkpoint pair.')}\nNo GPU numerical work was performed.\n"
    lines = [f"TDN neural replication: {summary['diagnostic_parent_count']} distinct diagnostic parents; three paired training seeds."]
    if summary.get("smoke"):
        lines.append("Smoke integration fixture; not the full replication comparison.")
    for row in summary["endpoint_rows"]:
        lines.append(f"  {row['replicate_id']} {row['family']}: {row['selection']}; "
                     f"feasible parents {row['feasible_parent_count']}/{row['parent_count']}; "
                     f"joint off-grid one/two-step pass {row['robust_joint_parent_pass_count']}/{row['parent_count']}.")
    for row in summary["comparison_rows"]:
        speedup = f"{row['speedup_median']:.3g}x" if row["speedup_median"] is not None else "unavailable"
        lines.append(f"  {row['replicate_id']} clock vs {row['baseline_family']}: "
                     f"speed wins/losses/ties {row['wins']}/{row['losses']}/{row['ties']}; "
                     f"eligible {row['eligible']}/{row['expected_parent_count']}; median {speedup}.")
    lines.append("Descriptive per-seed endpoints only. Repeated parents and classical timings are paired, not independent evidence.")
    return "\n".join(lines) + "\n"
