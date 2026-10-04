"""Bounded, failure-preserving neural comparisons; standard library only.

These are diagnostic comparisons of declared implementations on a finite grid.
They neither rank incomparable families nor claim published benchmark parity.
"""
from __future__ import annotations

from collections import Counter
import math


TDN_FAMILIES = ("confluent_decay", "fixed_decay", "fixed_decay_r", "fixed_undamped",
                "reaction_clock", "reaction_additive", "transport", "temporal_mlp",
                "reaction_hybrid", "reaction_polynomial")
DIRECT_BASELINES = ("residual_cnn", "unet", "fno")
HYBRID_BASELINES = ("residual_cnn_split", "unet_split", "fno_split")
TEMPORAL_BASELINES = ("generic_mlp",)
BASELINE_FAMILIES = DIRECT_BASELINES + HYBRID_BASELINES + TEMPORAL_BASELINES
_FAILURES = {"FAILED", "NUMERICAL_FAILURE", "INTERRUPTED"}
_MAX_ROWS = 8192
_MAX_PARENTS = 64
_MAX_HORIZONS = 16
_MAX_PAIRED_ROWS = 16384


def _finite(value, *, positive=False):
    try:
        return (not isinstance(value, bool) and isinstance(value, (float, int))
                and math.isfinite(value) and (value > 0 if positive else value >= 0))
    except OverflowError:
        return False


def _label(value, field):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"Neural comparison {field} must be a bounded nonempty string")
    return value


def _index(rows, kind):
    if not isinstance(rows, (list, tuple)) or len(rows) > _MAX_ROWS:
        raise ValueError(f"Neural comparison {kind} rows exceed the bounded input size")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Neural comparison {kind} rows must be dictionaries")
        parent = _label(row.get("parent_id"), "parent_id")
        family = _label(row.get("family"), "family")
        h = row.get("h")
        if not _finite(h, positive=True):
            raise ValueError("Neural comparison horizons must be finite and positive")
        key = (parent, family, float(h))
        if key in result:
            raise ValueError(f"Duplicate {kind} observation for {parent}/{family}/h={h}")
        result[key] = row
    return result


def _training_records(records, families):
    if records is None:
        records = []
    if isinstance(records, dict):
        records = list(records.values())
    if not isinstance(records, (list, tuple)) or len(records) > 32:
        raise ValueError("Neural training records must be a bounded sequence")
    by_family = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Neural training records must be dictionaries")
        family = _label(row.get("family"), "training family")
        if family in by_family:
            raise ValueError(f"Duplicate neural training record for {family}")
        by_family[family] = row
    result = {}
    for family in families:
        row = by_family.get(family, {})
        status = row.get("status", "NOT_RECORDED")
        selected = row.get("selected_step")
        selected = selected if type(selected) is int and selected >= 0 else None
        if status in _FAILURES:
            selection = "TRAINING_FAILURE"
        elif status == "COMPLETED" and (selected == 0 or row.get("selected_parameters_changed") is False):
            selection = "SELECTED_INITIALIZATION"
        elif status == "COMPLETED" and selected is not None and selected > 0:
            selection = "TRAINED_CHECKPOINT"
        else:
            selection = "UNKNOWN"
        result[family] = {
            "family": family, "status": str(status)[:512], "selection": selection,
            "selected_step": selected,
            "optimizer_steps": row.get("steps") if type(row.get("steps")) is int and row["steps"] >= 0 else None,
            "failure_reason": str(row["error"])[:512] if row.get("error") else None,
        }
    return result


def _error(row, step=None):
    if not row or row.get("status") != "COMPLETED":
        return None
    if step is None:
        rms, upper = row.get("error"), row.get("error_upper")
        if upper is None and _finite(rms) and _finite(row.get("reference_uncertainty")):
            upper = rms + row["reference_uncertainty"]
    else:
        error = row.get(step)
        if not isinstance(error, dict):
            return None
        rms, upper = error.get("rms"), error.get("upper")
        if upper is None and _finite(rms) and _finite(error.get("reference_uncertainty")):
            upper = rms + error["reference_uncertainty"]
    if not _finite(rms) or not _finite(upper) or upper < rms:
        return None
    return {"rms": float(rms), "upper": float(upper)}


def _missing_reason(row, *, step=None):
    if row is None:
        return "MISSING_RESULT"
    if row.get("status") != "COMPLETED":
        return str(row.get("status", "MISSING_STATUS"))[:512]
    if _error(row, step) is None:
        return "INVALID_OR_MISSING_ERROR"
    return None


def _frontier_issue(row):
    reason = _missing_reason(row)
    if reason:
        return reason
    timing = row.get("timing")
    if not isinstance(timing, dict) or not _finite(timing.get("wall_seconds_median"), positive=True):
        return "INVALID_OR_MISSING_WALL_TIME"
    if timing.get("soft_budget_passed") is False or timing.get("hard_device_memory_warning") is True:
        return "DEVICE_MEMORY_BUDGET_FAILED"
    if row.get("feasible") is not True:
        return "NOT_FEASIBLE_AT_DECLARED_TOLERANCE"
    return None


def _best_record(row):
    error = _error(row)
    return {"h": float(row["h"]), "wall_seconds": float(row["timing"]["wall_seconds_median"]),
            "error": error["rms"], "error_upper": error["upper"],
            "device": str(row["device"])[:512] if row.get("device") is not None else None,
            "macrosteps": row.get("macrosteps") if type(row.get("macrosteps")) is int else None}


def _relation(first, second, *, smaller_better):
    if math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-15):
        return "TIE"
    return "WIN" if (first < second if smaller_better else first > second) else "LOSS"


def _paired_errors(first, second, *, h, step=None):
    reasons = {name: reason for name, row in (("tdn", first), ("baseline", second))
               if (reason := _missing_reason(row, step=step))}
    first_error, second_error = _error(first, step), _error(second, step)
    # Hardware does not affect accuracy eligibility, but status and finite error do.
    result = {"h": h, "eligible": not reasons, "ineligible_reasons": reasons,
              "tdn_error": first_error, "baseline_error": second_error,
              "rms_outcome": None, "upper_error_outcome": None}
    if not reasons:
        result.update(rms_outcome=_relation(first_error["rms"], second_error["rms"], smaller_better=True),
                      upper_error_outcome=_relation(first_error["upper"], second_error["upper"], smaller_better=True))
    if step is None:
        result.update(tdn_feasible=bool(first and first.get("feasible") is True),
                      baseline_feasible=bool(second and second.get("feasible") is True))
    else:
        result["steps"] = 1 if step == "one_step" else 2
    return result


def _counts(rows, field):
    outcomes = Counter(row.get(field) for row in rows if row["eligible"])
    reasons = Counter(f"{side}:{reason}" for row in rows for side, reason in row["ineligible_reasons"].items())
    return {"cases": len(rows), "eligible": sum(row["eligible"] for row in rows),
            "wins": outcomes["WIN"], "losses": outcomes["LOSS"], "ties": outcomes["TIE"],
            "ineligible": sum(not row["eligible"] for row in rows), "failure_reasons": dict(sorted(reasons.items()))}


def _pair_counts(rows, field):
    """Do not duplicate global horizons and verbose failure causes per pair."""
    counts = _counts(rows, field)
    return {key: counts[key] for key in ("eligible", "wins", "losses", "ties")} | {
        # A row with one missing side and one failed side contributes to both
        # counts. Neither is represented as an error win for the available side.
        "missing": sum("MISSING_RESULT" in row["ineligible_reasons"].values() for row in rows),
        "failed": sum(any(reason != "MISSING_RESULT" for reason in row["ineligible_reasons"].values()) for row in rows),
    }


def summarize_neural_comparisons(frontier_rows, heldout_rows, declared_families, training_records=None):
    """Compare each declared TDN family with each declared neural baseline.

    ``frontier_rows[*].feasible`` carries the experiment's fixed tolerance and
    accepted-reference/memory checks. Classical headroom is never an input to
    eligibility. The fastest feasible point is selected only from the observed
    declared grid, independently for each family and diagnostic parent.
    """
    if not isinstance(declared_families, (list, tuple)) or not 1 <= len(declared_families) <= 32:
        raise ValueError("Declare between 1 and 32 neural families")
    families = [_label(family, "declared family") for family in declared_families]
    if len(set(families)) != len(families):
        raise ValueError("Declared neural families must be unique")
    frontier, heldout = _index(frontier_rows, "frontier"), _index(heldout_rows, "heldout")
    parent_ids = sorted({key[0] for key in frontier} | {key[0] for key in heldout})
    frontier_h = sorted({key[2] for key in frontier})
    heldout_h = sorted({key[2] for key in heldout})
    tdn = [family for family in families if family in TDN_FAMILIES]
    baselines = [family for family in families if family in BASELINE_FAMILIES]
    if (len(parent_ids) > _MAX_PARENTS or max(len(frontier_h), len(heldout_h)) > _MAX_HORIZONS
            or len(parent_ids) * len(tdn) * len(baselines) * (1 + len(frontier_h) + 2 * len(heldout_h)) > _MAX_PAIRED_ROWS):
        raise ValueError("Neural comparison output exceeds the bounded development size")
    training = _training_records(training_records, families)
    parents, family_results = [], {}
    for parent in parent_ids:
        observations = []
        for family in families:
            results = [frontier.get((parent, family, h)) for h in frontier_h]
            feasible = [row for row in results if row is not None and _frontier_issue(row) is None]
            best = min(feasible, key=lambda row: (row["timing"]["wall_seconds_median"], row["h"])) if feasible else None
            issues = [{"h": h, "reason": _frontier_issue(row),
                       "detail": str(row["error_message"])[:512] if row and row.get("error_message") else None}
                      for h, row in zip(frontier_h, results) if _frontier_issue(row)]
            status = ("FEASIBLE" if best else "TRAINING_FAILURE" if training[family]["selection"] == "TRAINING_FAILURE"
                      else "MISSING_RESULTS" if not any(row is not None for row in results) else "NO_FEASIBLE_ROLLOUT")
            observation = {
                "family": family, "status": status, "training_selection": training[family]["selection"],
                "expected_h_count": len(frontier_h), "observed_h_count": sum(row is not None for row in results),
                "feasible_h_count": len(feasible), "missing_h_count": sum(row is None for row in results),
                "invalid_h_count": sum(row is not None and row.get("status") != "COMPLETED" for row in results),
                "heldout_missing_h_count": sum((parent, family, h) not in heldout for h in heldout_h),
                "heldout_invalid_h_count": sum((row := heldout.get((parent, family, h))) is not None
                                               and row.get("status") != "COMPLETED" for h in heldout_h),
                "infeasible_or_invalid_h": issues, "best_feasible_rollout": _best_record(best) if best else None,
            }
            observations.append(observation)
            family_results[parent, family] = observation
        parents.append({"parent_id": parent, "families": observations})
    comparisons, aggregates = [], []
    for family in tdn:
        for baseline in baselines:
            pair_rows = []
            same_h = []
            heldout_pairs = []
            for parent in parent_ids:
                first, second = family_results[parent, family], family_results[parent, baseline]
                first_best, second_best = first["best_feasible_rollout"], second["best_feasible_rollout"]
                reasons = {side: row["status"] for side, row in (("tdn", first), ("baseline", second))
                           if row["best_feasible_rollout"] is None}
                if first_best and second_best and first_best["device"] != second_best["device"]:
                    reasons["comparison"] = "DEVICE_MISMATCH"
                speedup = second_best["wall_seconds"] / first_best["wall_seconds"] if not reasons else None
                if speedup is not None and not math.isfinite(speedup):
                    speedup = None
                    reasons["comparison"] = "NONFINITE_SPEEDUP"
                rollout_pairs = [_paired_errors(frontier.get((parent, family, h)), frontier.get((parent, baseline, h)), h=h)
                                 for h in frontier_h]
                diagnostic_pairs = [_paired_errors(heldout.get((parent, family, h)), heldout.get((parent, baseline, h)), h=h, step=step)
                                    for h in heldout_h for step in ("one_step", "two_step")]
                same_h.extend(rollout_pairs)
                heldout_pairs.extend(diagnostic_pairs)
                row = {
                    "parent_id": parent, "tdn_family": family, "baseline_family": baseline,
                    "baseline_kind": ("direct_neural" if baseline in DIRECT_BASELINES else
                                      "physics_split_hybrid" if baseline in HYBRID_BASELINES else
                                      "time_conditioned_feature_additive_hybrid"),
                    "baseline_training_selection": training[baseline]["selection"],
                    "eligible": not reasons, "ineligible_reasons": reasons,
                    "tdn_best": first_best, "baseline_best": second_best, "speedup": speedup,
                    "speed_outcome": _relation(speedup, 1., smaller_better=False) if speedup is not None else None,
                    "same_h_rollout_rms": _pair_counts(rollout_pairs, "rms_outcome"),
                    "heldout_one_step_rms": _pair_counts([item for item in diagnostic_pairs if item["steps"] == 1], "rms_outcome"),
                    "heldout_two_step_rms": _pair_counts([item for item in diagnostic_pairs if item["steps"] == 2], "rms_outcome"),
                }
                pair_rows.append(row)
                comparisons.append(row)
            aggregates.append({
                "tdn_family": family, "baseline_family": baseline,
                "baseline_training_selection": training[baseline]["selection"],
                "matched_tolerance_speed": _counts(pair_rows, "speed_outcome"),
                "same_h_rollout_rms": _counts(same_h, "rms_outcome"),
                "same_h_rollout_upper_error": _counts(same_h, "upper_error_outcome"),
                "heldout_one_step_rms": _counts([row for row in heldout_pairs if row["steps"] == 1], "rms_outcome"),
                "heldout_two_step_rms": _counts([row for row in heldout_pairs if row["steps"] == 2], "rms_outcome"),
                "heldout_one_step_upper_error": _counts([row for row in heldout_pairs if row["steps"] == 1], "upper_error_outcome"),
                "heldout_two_step_upper_error": _counts([row for row in heldout_pairs if row["steps"] == 2], "upper_error_outcome"),
            })
    return {
        "version": 1, "diagnostic_parent_count": len(parent_ids), "declared_families": families,
        "tdn_families": tdn, "baseline_families": baselines,
        "frontier_horizons": frontier_h, "heldout_horizons": heldout_h,
        "training": list(training.values()), "parents": parents,
        "comparisons": comparisons, "aggregates": aggregates,
        "scope": "Finite diagnostic development comparison of local neural implementations; no published SOTA claim or overall ranking.",
        "cost_scope": "Complete rollout wall time on the same device, including features, network, physics and state checks; training cost is separate.",
        "accuracy_scope": "Speed uses the declared fixed-tolerance feasible frontier. Same-h and heldout errors include finite completed trajectories regardless of tolerance; failures and absence are explicit.",
        "selection_scope": "Post-hoc fastest feasible declared-grid point per diagnostic parent, not a deployable step selector; checkpoints selected on validation only.",
        "coverage_scope": "Parents and expected horizon grids are inferred from all supplied rows; entirely unobserved parents cannot be inferred.",
        "tie_rule": "Error and speed ties use relative tolerance 1e-12 and absolute tolerance 1e-15; not a statistical significance test.",
        "error_details": "Exact paired horizon errors, invalid statuses and messages remain in frontier.json and heldout.json; summary counts never omit failed or absent observations.",
        "error_count_scope": "Per-parent pairs report RMS counts; per-family aggregates additionally report uncertainty-adjusted upper-error counts. Missing and failed side counts can overlap on one pair.",
        "budget_scope": "Finite declared grids, parent-disjoint data and equal optimizer-step budgets; parameter counts and architectures are reported rather than matched. No extra baseline tuning or classical-headroom requirement.",
    }


def readable_neural_summary(result):
    """Produce compact log lines while leaving detailed outcomes in JSON."""
    lines = [f"Neural diagnostic comparison: {result['diagnostic_parent_count']} parents; {len(result['baseline_families'])} declared baselines."]
    for row in result["training"]:
        if row["family"] in result["baseline_families"]:
            lines.append(f"  Baseline {row['family']}: {row['selection']}; selected step {row['selected_step']}.")
    for row in result["aggregates"]:
        counts = row["matched_tolerance_speed"]
        lines.append(f"  {row['tdn_family']} vs {row['baseline_family']}: {counts['eligible']}/{counts['cases']} feasible pairs; "
                     f"speed wins/losses/ties {counts['wins']}/{counts['losses']}/{counts['ties']}; "
                     f"ineligible {counts['ineligible']}.")
    lines.append("Speed eligibility is independent of classical headroom. Failure/initialization selections remain visible; no overall ranking or SOTA claim.")
    return "\n".join(lines) + "\n"
