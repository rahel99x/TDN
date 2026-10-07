"""Post-hoc, failure-preserving paired neural comparisons; standard library only.

Every declared diagnostic parent, seed, pair, norm and tolerance remains in the
output. Coverage gains and eligible timing comparisons are separate endpoints.
Initialization-selected models are physical-base controls, not trained wins.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from tdn.runtime.metadata import write_json


SCHEMA = "tdn.premix-comparisons/v1"
SOURCE_SCHEMA = "tdn.premix-neural/v1"
OURS = ("premix", "premix_local")
BASELINES = ("precompress", "fno", "cnn")
FAMILIES = (*OURS, *BASELINES)
SOURCE_FILES = ("protocol.json", "training.json", "candidates.json", "frontiers.json")


def _finite(value, *, positive=False):
    return (type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _close(first, second):
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-15)


def _plan(protocol, *, source_schema=SOURCE_SCHEMA, families=FAMILIES):
    if protocol.get("schema") != source_schema or protocol.get("version") != 1:
        raise ValueError("Unsupported premix neural protocol")
    if protocol.get("families") != list(families):
        raise ValueError("Comparison requires all declared neural families in protocol order")
    seeds, targets = protocol.get("seeds"), protocol.get("targets")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(type(seed) is not int or seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Comparison requires unique declared training seeds")
    if (not isinstance(targets, list) or not 1 <= len(targets) <= 8
            or any(not _finite(target, positive=True) for target in targets) or len(set(targets)) != len(targets)):
        raise ValueError("Comparison requires unique positive declared targets")
    all_parents = protocol.get("parents")
    if not isinstance(all_parents, list) or len(all_parents) > 256:
        raise ValueError("Invalid bounded parent plan")
    parents = [parent for parent in all_parents if parent.get("split") == "diagnostic"]
    if not parents or len({parent["parent_id"] for parent in all_parents}) != len(all_parents):
        raise ValueError("Diagnostic parents must be declared and globally unique")
    for parent in parents:
        if (not isinstance(parent["parent_id"], str) or not parent["parent_id"]
                or parent.get("category") not in ("favorable", "typical", "adverse")
                or not isinstance(parent.get("regime"), str)
                or not isinstance(parent.get("distribution"), str)):
            raise ValueError("Malformed a priori diagnostic grouping")
    for key in ("step_counts", "long_step_counts"):
        values = protocol.get(key)
        if (not isinstance(values, list) or not 1 <= len(values) <= 8
                or any(type(value) is not int or value < 1 for value in values)
                or len(set(values)) != len(values)):
            raise ValueError("Invalid declared step-count plan")
    if type(protocol.get("timing_repeats")) is not int or protocol["timing_repeats"] < 1:
        raise ValueError("Invalid declared timing-repeat count")
    return parents


def _steps(protocol, parent):
    return protocol["long_step_counts"] if parent["regime"] == "long_rollout" else protocol["step_counts"]


def _index(rows, fields, expected, label):
    if not isinstance(rows, list) or len(rows) > 32768:
        raise ValueError(f"Invalid bounded {label} table")
    result = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Non-object {label} row")
        key = tuple(row.get(field) for field in fields)
        if ("seed" in fields and row.get("seed") is not None and type(row.get("seed")) is not int
                or "steps" in fields and type(row.get("steps")) is not int
                or "target" in fields and not _finite(row.get("target"), positive=True)):
            raise ValueError(f"Malformed {label} numeric identity")
        try:
            if key not in expected:
                raise ValueError(f"Undeclared {label} identity: {key}")
            if key in result:
                raise ValueError(f"Duplicate {label} identity: {key}")
        except TypeError as error:
            raise ValueError(f"Malformed {label} identity") from error
        result[key] = (row, f"/rows/{index}")
    return result


def _training(record):
    if record is None:
        return "MISSING", ["MISSING_TRAINING_RECORD"]
    if record.get("status") != "COMPLETED":
        return record.get("selection", "NO_ELIGIBLE_CHECKPOINT"), ["TRAINING_NOT_COMPLETED"]
    step, changed = record.get("selected_step"), record.get("selected_parameters_changed")
    if type(step) is not int or step < 0 or type(changed) is not bool or (step == 0 and changed):
        return "INCONSISTENT_SELECTION", ["INCONSISTENT_TRAINING_SELECTION"]
    selection = "TRAINED_CHECKPOINT" if step > 0 and changed else "SELECTED_INITIALIZATION"
    if record.get("selection") != selection:
        return selection, ["INCONSISTENT_TRAINING_SELECTION"]
    return selection, []


def _candidate_issues(row, parent, protocol, selection):
    issues = []
    if any(row.get(field) != parent.get(field) for field in ("regime", "category", "distribution", "grid")):
        issues.append("CANDIDATE_PARENT_METADATA_MISMATCH")
    if row.get("horizon") != parent["horizons"][0]:
        issues.append("CANDIDATE_HORIZON_MISMATCH")
    if row.get("checkpoint_selection") != selection:
        issues.append("CANDIDATE_CHECKPOINT_SELECTION_MISMATCH")
    if row.get("status") not in ("COMPLETED", "TRAINING_FAILED", "INVALID_TRAJECTORY"):
        issues.append("SOURCE_CANDIDATE_NOT_TERMINAL")
    if row.get("status") != "COMPLETED":
        return issues + [str(row.get("status", "MISSING_STATUS"))]
    if row.get("admissible") is not True or row.get("memory_ok") is not True:
        issues.append("ADMISSIBILITY_OR_MEMORY_FAILED")
    for norm, error_name, uncertainty_name in (("rms", "rms", "uncertainty_rms"),
                                              ("max", "max_error", "uncertainty_max_bound")):
        error, uncertainty, upper = row.get(error_name), row.get(uncertainty_name), row.get("upper_" + norm)
        if not all(_finite(value) for value in (error, uncertainty, upper)) or not _close(upper, error + uncertainty):
            issues.append("INVALID_ERROR_BOUND_" + norm.upper())
    timing = row.get("timing", {})
    median, raw = timing.get("wall_seconds_median"), timing.get("wall_seconds_raw")
    if (not _finite(median, positive=True) or not isinstance(raw, list)
            or len(raw) != protocol["timing_repeats"]
            or not all(_finite(value, positive=True) for value in raw)
            or not _close(median, statistics.median(raw))):
        issues.append("INVALID_TIMING_REPEATS")
    if row.get("device") not in ("cpu", "cuda") or timing.get("device") != row.get("device"):
        issues.append("INVALID_DEVICE_RECORD")
    return issues


def _side(parent, seed, family, norm, target, protocol, candidate_map, frontier_map, training_map):
    training, training_pointer = training_map.get((seed, family), (None, None))
    selection, issues = _training(training)
    frontier, pointer = frontier_map.get((parent["parent_id"], seed, family, norm, target), (None, None))
    result = {"selection": selection, "training_record": training_pointer,
              "frontier_record": pointer, "candidate_record": None, "selected_steps": None,
              "seconds": None, "raw_min_seconds": None, "raw_max_seconds": None,
              "repeat_count": 0, "device": None, "status": "INCOMPLETE_OR_INVALID_RESULT", "issues": issues,
              "feasible": False, "selected_step": training.get("selected_step") if training else None}
    if frontier is None:
        issues.append("MISSING_FRONTIER")
    elif any(frontier.get(key) != parent.get(key) for key in ("regime", "category", "distribution", "grid")):
        issues.append("FRONTIER_PARENT_METADATA_MISMATCH")
    eligible = []
    for steps in _steps(protocol, parent):
        row, candidate_pointer = candidate_map.get((parent["parent_id"], seed, family, steps), (None, None))
        if row is None:
            issues.append(f"MISSING_CANDIDATE_STEPS_{steps}")
            continue
        candidate_issues = _candidate_issues(row, parent, protocol, selection)
        if not candidate_issues and row["upper_" + norm] <= target:
            eligible.append((row, candidate_pointer))
        # Numerical and memory failures remain infeasible candidates. Corrupt
        # provenance/timing/errors cannot silently become scientific evidence.
        issues.extend(reason for reason in candidate_issues
                      if reason.startswith(("INVALID_", "CANDIDATE_", "SOURCE_")) and reason != "INVALID_TRAJECTORY")
    if issues:
        result["issues"] = sorted(set(issues))
        return result
    if not eligible:
        if (frontier.get("status") != "NO_FEASIBLE_CANDIDATE"
                or any(frontier.get(key) is not None for key in ("selected_steps", "seconds", "error_upper"))):
            issues.append("FRONTIER_FEASIBILITY_MISMATCH")
            return result
        result.update(status="NO_FEASIBLE_CANDIDATE", issues=[])
        return result
    # Tied medians may validly select any exact minimum. Do not use a tolerance
    # to silently relabel a slower measured candidate as the minimum.
    minimum = min(row["timing"]["wall_seconds_median"] for row, _ in eligible)
    selected = [(row, record) for row, record in eligible if row["steps"] == frontier.get("selected_steps")]
    if (frontier.get("status") != "FEASIBLE" or len(selected) != 1
            or selected[0][0]["timing"]["wall_seconds_median"] != minimum):
        issues.append("FRONTIER_NOT_MINIMUM_FEASIBLE_CANDIDATE")
        return result
    row, candidate_pointer = selected[0]
    if frontier.get("seconds") != minimum or frontier.get("error_upper") != row["upper_" + norm]:
        issues.append("FRONTIER_SELECTED_CANDIDATE_MISMATCH")
        return result
    raw = row["timing"]["wall_seconds_raw"]
    result.update(feasible=True, status="FEASIBLE", candidate_record=candidate_pointer,
                  selected_steps=row["steps"], seconds=minimum, raw_min_seconds=min(raw),
                  raw_max_seconds=max(raw), repeat_count=len(raw), device=row["device"], issues=[])
    return result


def _counts(rows):
    result = {
        "total": len(rows),
        "ours_feasible": sum(row["ours_feasible"] for row in rows),
        "baseline_feasible": sum(row["baseline_feasible"] for row in rows),
        "both_feasible": sum(row["ours_feasible"] and row["baseline_feasible"] for row in rows),
        "timing_eligible": sum(row["timing_eligible"] for row in rows),
        "ours_only_feasible": sum(row["ours_feasible"] and row["baseline_status"] == "NO_FEASIBLE_CANDIDATE" for row in rows),
        "baseline_only_feasible": sum(row["baseline_feasible"] and row["ours_status"] == "NO_FEASIBLE_CANDIDATE" for row in rows),
        "ours_feasible_baseline_unresolved": sum(row["ours_feasible"] and row["baseline_status"] == "INCOMPLETE_OR_INVALID_RESULT" for row in rows),
        "baseline_feasible_ours_unresolved": sum(row["baseline_feasible"] and row["ours_status"] == "INCOMPLETE_OR_INVALID_RESULT" for row in rows),
        "trained_pair_eligible": sum(row["trained_pair_eligible"] for row in rows),
        "initialization_in_eligible_pair": sum(row["timing_eligible"] and
            "SELECTED_INITIALIZATION" in (row["ours_selection"], row["baseline_selection"]) for row in rows),
        "incomplete_or_invalid": sum(bool(row["issues"]) for row in rows),
    }
    for outcome, name in (("WIN", "wins"), ("TIE", "ties"), ("LOSS", "losses")):
        result["ours_" + name] = sum(row["outcome"] == outcome for row in rows)
        result["trained_pair_" + name] = sum(row["trained_pair_eligible"] and row["outcome"] == outcome for row in rows)
    result["range_separated_wins"] = sum(row["robust_speed_win"] for row in rows)
    result["range_separated_losses"] = sum(row["robust_speed_loss"] for row in rows)
    return result


def compare(protocol, candidates, frontiers, training, *, source_schema=SOURCE_SCHEMA,
            families=FAMILIES, pairs=None, schema=SCHEMA):
    """Derive paired rows from immutable observations without running numerics."""
    parents = _plan(protocol, source_schema=source_schema, families=families)
    pairs = tuple((ours, baseline) for ours in OURS for baseline in BASELINES) if pairs is None else tuple(pairs)
    if (not pairs or len(set(pairs)) != len(pairs)
            or any(len(pair) != 2 or pair[0] == pair[1] or any(family not in families for family in pair) for pair in pairs)):
        raise ValueError("Invalid declared comparison pairs")
    methods = [(None, family) for family in protocol["classical"]]
    methods += [(seed, family) for seed in protocol["seeds"] for family in protocol["families"]]
    expected_training = {(seed, family) for seed in protocol["seeds"] for family in protocol["families"]}
    expected_candidates = {(parent["parent_id"], seed, family, steps) for parent in parents
                           for seed, family in methods for steps in _steps(protocol, parent)}
    expected_frontiers = {(parent["parent_id"], seed, family, norm, target) for parent in parents
                          for seed, family in methods for norm in ("rms", "max") for target in protocol["targets"]}
    training_map = _index(training, ("seed", "family"), expected_training, "training")
    candidate_map = _index(candidates, ("parent_id", "seed", "family", "steps"), expected_candidates, "candidate")
    frontier_map = _index(frontiers, ("parent_id", "seed", "family", "norm", "target"), expected_frontiers, "frontier")
    side_cache, rows = {}, []
    for parent in parents:
        for seed in protocol["seeds"]:
            for ours, baseline in pairs:
                for norm in ("rms", "max"):
                    for target_index, target in enumerate(protocol["targets"]):
                        sides = []
                        for family in (ours, baseline):
                            key = (parent["parent_id"], seed, family, norm, target)
                            if key not in side_cache:
                                side_cache[key] = _side(parent, seed, family, norm, target, protocol,
                                                       candidate_map, frontier_map, training_map)
                            sides.append(side_cache[key])
                        first, second = sides
                        issues = [f"{label}:{issue}" for label, side in zip(("ours", "baseline"), sides)
                                  for issue in side["issues"]]
                        eligible = first["feasible"] and second["feasible"]
                        if eligible and first["device"] != second["device"]:
                            issues.append("DEVICE_MISMATCH")
                            eligible = False
                        row = {key: parent[key] for key in ("parent_id", "regime", "category", "distribution", "grid")}
                        row.update(comparison_id=f"{parent['parent_id']}:{seed}:{ours}:{baseline}:{norm}:{target_index}",
                            seed=seed, ours=ours, baseline=baseline, norm=norm, target=target,
                            timing_eligible=eligible, outcome=None, speedup_baseline_over_ours=None,
                            trained_pair_eligible=eligible and all(side["selection"] == "TRAINED_CHECKPOINT" for side in sides),
                            robust_speed_win=False, robust_speed_loss=False, range_status="NOT_ELIGIBLE", issues=issues)
                        for label, side in zip(("ours", "baseline"), sides):
                            for key in ("feasible", "status", "selection", "selected_step", "selected_steps", "seconds",
                                        "frontier_record", "candidate_record", "training_record", "raw_min_seconds",
                                        "raw_max_seconds", "repeat_count", "device"):
                                row[label + "_" + key] = side[key]
                        row["status"] = ("PAIRED_FEASIBLE" if eligible else "INCOMPLETE_OR_INVALID_RESULT" if issues
                                         else "UNPAIRED_FEASIBILITY" if any(side["feasible"] for side in sides)
                                         else "NO_PAIRED_FEASIBLE_RESULT")
                        if eligible:
                            row["speedup_baseline_over_ours"] = second["seconds"] / first["seconds"]
                            row["outcome"] = ("TIE" if _close(first["seconds"], second["seconds"])
                                              else "WIN" if first["seconds"] < second["seconds"] else "LOSS")
                            if min(first["repeat_count"], second["repeat_count"]) >= 2:
                                row["robust_speed_win"] = first["raw_max_seconds"] < second["raw_min_seconds"]
                                row["robust_speed_loss"] = second["raw_max_seconds"] < first["raw_min_seconds"]
                                row["range_status"] = ("DISJOINT_OURS_FASTER" if row["robust_speed_win"] else
                                                       "DISJOINT_BASELINE_FASTER" if row["robust_speed_loss"] else "OVERLAPPING")
                            else:
                                row["range_status"] = "INSUFFICIENT_REPEATS"
                        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        for scope, value in (("all", "all"), ("category", row["category"]), ("regime", row["regime"])):
            grouped[(scope, value, row["seed"], row["ours"], row["baseline"], row["norm"], row["target"])].append(row)
    group_fields = ("group_scope", "group", "seed", "ours", "baseline", "norm", "target")
    groups = [{**dict(zip(group_fields, key)), "counts": _counts(value),
               "parent_ids": [row["parent_id"] for row in value]} for key, value in sorted(grouped.items())]
    expected_pairs = len(parents) * len(protocol["seeds"]) * len(pairs) * 2 * len(protocol["targets"])
    coverage = {
        "diagnostic_parents": len(parents), "training_seeds": len(protocol["seeds"]),
        "training": {"expected": len(expected_training), "reported": len(training_map)},
        "candidates": {"expected": len(expected_candidates), "reported": len(candidate_map)},
        "frontiers": {"expected": len(expected_frontiers), "reported": len(frontier_map)},
        "comparisons": {"expected": expected_pairs, "reported": len(rows)},
        "complete_source_plan": (len(training_map) == len(expected_training) and len(candidate_map) == len(expected_candidates)
                                 and len(frontier_map) == len(expected_frontiers)),
    }
    return {"schema": schema, "status": "DERIVED", "coverage": coverage, "counts": _counts(rows),
        "rows": rows, "groups": groups,
        "training_selections": [{"seed": seed, "family": family,
            "selection": _training(training_map.get((seed, family), (None, None))[0])[0],
            "source_record": training_map.get((seed, family), (None, None))[1]}
            for seed in protocol["seeds"] for family in protocol["families"]],
        "source_pointer_files": {"*_frontier_record": "frontiers.json", "*_candidate_record": "candidates.json",
                                 "*_training_record": "training.json"},
        "scope": "Post-hoc reference-informed fixed-step frontiers; no deployable adaptive policy or FNO-paper reproduction claim",
        "interpretation": {
            "speedup": "baseline median seconds divided by ours; only both feasible on the same device",
            "wins": "descriptive timing outcomes, including initialization-selected implementations; not evidence of successful learning",
            "trained_pair": "both selections must have a positive selected step and changed selected parameters",
            "coverage": "one feasible side against an infeasible side is a coverage difference, never a speed win",
            "missing": "missing candidates, frontiers or training records remain visible and ineligible",
            "timing_ranges": "robust_speed_* require disjoint observed raw ranges and at least two repeats each; heuristic, not confidence intervals",
            "grouping": "favorable/typical/adverse and named regimes were assigned before measurements",
            "dependence": "training seeds, norms and targets reuse the same diagnostic parents; rows and groups are not independent trials",
            "classical": "classical rows use seed=null and are counted once in source coverage; no repeated classical seed evidence is created",
            "cost": "steady-state full-rollout latency; training and setup excluded and reported separately in source artifacts",
            "statistics": "no p-values, pooled significance tests or all-purpose winning score",
        }}


def _readable(result):
    coverage = result["coverage"]
    lines = ["TDN premix paired neural review", result["scope"],
        f"Independent diagnostic parents: {coverage['diagnostic_parents']}; paired training seeds: {coverage['training_seeds']}.",
        "Rows reuse parents across seeds, norms and targets; these are descriptive counts, not independent trials.",
        "Initialization-selected models are physical-base controls; a timing win is not a trained-model victory.", ""]
    for name in ("training", "candidates", "frontiers", "comparisons"):
        item = coverage[name]
        lines.append(f"{name}: {item['reported']}/{item['expected']} declared records")
    lines.append("")
    for row in result["training_selections"]:
        lines.append(f"seed {row['seed']} {row['family']}: {row['selection']}")
    lines.extend(["", "Matched-accuracy warmed timing, each seed and pair (all declared norms/targets):"])
    pairs = defaultdict(list)
    for row in result["rows"]:
        pairs[row["seed"], row["ours"], row["baseline"]].append(row)
    for (seed, ours, baseline), rows in sorted(pairs.items()):
        count = _counts(rows)
        lines.append(f"  {seed} {ours} vs {baseline}: eligible {count['timing_eligible']}/{count['total']}; "
            f"W/L/T {count['ours_wins']}/{count['ours_losses']}/{count['ours_ties']}; "
            f"both-trained eligible {count['trained_pair_eligible']}, wins {count['trained_pair_wins']}; "
            f"only-ours/only-baseline feasible {count['ours_only_feasible']}/{count['baseline_only_feasible']}.")
    lines.extend(["", "Inspect comparisons.json groups for separate a priori category/regime, seed, norm and tolerance results.",
                  "Raw-range separation is a descriptive repeat check, not a confidence interval or significance claim."])
    return "\n".join(lines) + "\n"


def summarize(run_dir: Path) -> dict:
    """Write derived comparisons before the coordinator seals the stage.

    Canonical source JSON and the engine summary are never modified. Missing
    tables on interrupted runs become empty observations, not successful cases.
    """
    root = Path(run_dir)
    if root.is_symlink() or (root / "COMPLETED").exists() or (root / "manifest.json").exists():
        raise ValueError("Preserve completed/sealed premix stages; derive comparisons before sealing")
    documents, sources = {}, []
    for name in SOURCE_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError("Comparison sources cannot be symlinks")
        if not path.exists():
            if name == "protocol.json":
                raise ValueError("A declared protocol is required")
            documents[name] = {"schema": SOURCE_SCHEMA, "rows": []}
            sources.append({"path": name, "status": "MISSING", "sha256": None})
            continue
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("Comparison source exceeds bounded JSON size")
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or document.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"Unsupported comparison source schema: {name}")
        documents[name] = document
        sources.append({"path": name, "status": "READ", "sha256": hashlib.sha256(raw).hexdigest()})
    result = compare(documents["protocol.json"], documents["candidates.json"].get("rows"),
                     documents["frontiers.json"].get("rows"), documents["training.json"].get("rows"))
    result["sources"] = sources
    write_json(root / "comparisons.json", result)
    temporary = root / "summary.txt.partial"
    temporary.write_text(_readable(result))
    os.replace(temporary, root / "summary.txt")
    return result["counts"]


__all__ = ["compare", "summarize", "SCHEMA"]
