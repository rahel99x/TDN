"""Physical metrics and portable reports for the bounded regularization audit.

There is no fitting, teacher acquisition or timing in this module. Callers must
retain one row for every planned method/query, including NOT_EVALUATED rows.
Summaries cannot reconstruct a planned query that was omitted from the input.
Every query has uniform physical cell mass on the declared common grid; every
independent parent has equal weight after averaging its own queries.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import re

import numpy as np


IDENTITY_KEYS = ("query_id", "parent_id", "parent_family", "split", "role", "family",
                 "context_id", "source_index", "start", "stop")
META_KEYS = ("training_parent_count", "calibration_source_count", "training_membership",
             "kind", "cell_id", "regularization_family", "selected_family")
ERROR_KEYS = ("mse", "rms", "maximum", "mean_error", "mean_component_mse",
              "mean_component_rms", "spatial_centered_mse", "spatial_centered_rms",
              "centered_mse", "centered_rms")
DENOMINATORS = ("reference", "remainder", "feedback")
MISSING = {None, "NOT_EVALUATED", "NOT_ACQUIRED", "MISSING"}
RESOLUTION_MULTIPLE = 10.


def _finite_number(value, *, nonnegative=False):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
            and (not nonnegative or value >= 0))


def _uncertainty(value, name):
    if value is None:
        return None
    if not _finite_number(value, nonnegative=True):
        raise ValueError(f"{name} must be finite and nonnegative or None")
    return float(value)


def _array(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or result.ndim == 0 or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must have the same nonempty physical array shape")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _rms(value):
    return math.sqrt(float(np.mean(value * value)))


def _ratio_fields(item, *, resolved=None):
    for name in DENOMINATORS:
        denominator = item.get(name + "_rms")
        uncertainty = item.get(name + "_uncertainty_rms")
        floor = RESOLUTION_MULTIPLE * uncertainty if uncertainty is not None else None
        status = ("UNRESOLVED_REFERENCE" if denominator is None or floor is None
                  else "RESOLVED" if denominator > floor else "UNRESOLVED_NEAR_NULL")
        if resolved is not None and not resolved.get(name, False):
            status = "UNRESOLVED_MEMBER"
        item[name + "_floor_rms"] = floor
        item[name + "_status"] = status
        item[name + "_resolved"] = status == "RESOLVED"
        item["relative_" + name + "_rms"] = (
            item["rms"] / denominator if status == "RESOLVED" and item.get("rms") is not None else None)
    uncertainty = item.get("reference_uncertainty_rms")
    item["error_to_uncertainty_rms"] = (item["rms"] / uncertainty
                                         if uncertainty and item.get("rms") is not None else None)


def metric(row, method, prediction, teacher, base, matched_mean,
           reference_uncertainty_rms, reference_uncertainty_maximum, **metadata):
    """Measure one fresh prediction, retaining numerical failures as rows.

    ``matched_mean_uncertainty_rms`` may be supplied in metadata. Without it,
    the feedback denominator remains visible but its relative error is not
    declared resolved. Reference uncertainty is an estimate, never a bound.
    Invalid identity data raise; numerical failures return status FAILED.
    """
    for key in ("query_id", "parent_id", "split", "role"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a nonempty string")
    item = {key: row[key] for key in IDENTITY_KEYS if key in row}
    item.update({key: metadata.get(key, row.get(key)) for key in META_KEYS})
    item["training_membership"] = item["training_membership"] or "not_applicable"
    item.update(method=method, status="FAILED", **{key: None for key in ERROR_KEYS},
                reference_uncertainty_is_bound=False, signal_resolution_multiple=RESOLUTION_MULTIPLE,
                decomposition="MSE = mean_component_mse + spatial_centered_mse, evaluated within each physical query")
    try:
        u = _uncertainty(reference_uncertainty_rms, "reference_uncertainty_rms")
        umax = _uncertainty(reference_uncertainty_maximum, "reference_uncertainty_maximum")
        gu = _uncertainty(metadata.get("matched_mean_uncertainty_rms"), "matched_mean_uncertainty_rms")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            truth = _array(teacher, "teacher")
            predicted = _array(prediction, "prediction", truth.shape)
            classical = _array(base, "base", truth.shape)
            matched = None if matched_mean is None else _array(matched_mean, "matched_mean", truth.shape)
            error = predicted - truth
            mean = float(np.mean(error))
            mse = float(np.mean(error * error))
            mean_mse = mean * mean
            spatial_mse = float(np.mean((error - mean) ** 2))
            values = dict(mse=mse, rms=math.sqrt(mse), maximum=float(np.max(np.abs(error))),
                          mean_error=mean, mean_component_mse=mean_mse, mean_component_rms=abs(mean),
                          spatial_centered_mse=spatial_mse, spatial_centered_rms=math.sqrt(spatial_mse),
                          centered_mse=spatial_mse, centered_rms=math.sqrt(spatial_mse),
                          reference_rms=_rms(truth), remainder_rms=_rms(truth - classical),
                          feedback_rms=_rms(truth - matched) if matched is not None else None,
                          reference_uncertainty_rms=u, reference_uncertainty_maximum=umax,
                          remainder_uncertainty_rms=u,
                          feedback_uncertainty_rms=u + gu if u is not None and gu is not None else None,
                          matched_mean_uncertainty_rms=gu)
            _ratio_fields(values)
            # Python float division and addition can overflow outside NumPy's
            # error state. Never serialize a partially nonfinite measurement.
            json.dumps(values, allow_nan=False)
        item.update(values, status="MEASURED")
    except (ValueError, TypeError, ArithmeticError) as error:
        item["error"] = f"{type(error).__name__}: {error}"
    return item


def _valid(row):
    if row.get("status") != "MEASURED":
        return False
    nonnegative = ("mse", "rms", "maximum", "mean_component_mse", "spatial_centered_mse")
    if not all(_finite_number(row.get(key), nonnegative=True) for key in nonnegative):
        return False
    if not _finite_number(row.get("mean_error")):
        return False
    scale = max(float(row["mse"]), float(row["mean_component_mse"]),
                float(row["spatial_centered_mse"]), np.finfo(float).tiny)
    tolerance = 64 * np.finfo(float).eps * scale
    try:
        return bool(abs(row["mse"] - row["rms"] ** 2) <= tolerance
                and abs(row["mean_component_mse"] - row["mean_error"] ** 2) <= tolerance
                and abs(row["mse"] - row["mean_component_mse"] - row["spatial_centered_mse"]) <= tolerance)
    except (ArithmeticError, ValueError):
        return False


def _account(rows):
    valid = [_valid(row) for row in rows]
    missing = sum(row.get("status") in MISSING for row in rows)
    unresolved = sum(row.get("status") == "UNRESOLVED" for row in rows)
    measured = sum(valid)
    failed = len(rows) - measured - missing - unresolved
    return dict(expected=len(rows), measured=measured, failed=failed, missing=missing,
                unresolved=unresolved, complete=bool(rows) and measured == len(rows),
                planned_queries=len(rows), measured_queries=measured, failed_queries=failed,
                missing_queries=missing, unresolved_queries=unresolved,
                invalid_measured_queries=sum(row.get("status") == "MEASURED" and not ok
                                             for row, ok in zip(rows, valid)))


def _mean(values):
    # Dividing first avoids overflow from summing individually finite MSEs.
    return math.fsum(float(value) / len(values) for value in values)


def _pooled_rms(values):
    scale = max(values)
    return scale * math.sqrt(_mean([(value / scale) ** 2 for value in values])) if scale else 0.


def _summed_metrics(rows):
    """Equal-row reduction; rows can be queries or equally weighted parents."""
    result = {key: None for key in ERROR_KEYS}
    if not rows:
        return result
    mse = _mean([row["mse"] for row in rows])
    mean_mse = _mean([row["mean_component_mse"] for row in rows])
    spatial_mse = _mean([row["spatial_centered_mse"] for row in rows])
    result.update(mse=mse, rms=math.sqrt(mse), maximum=max(row["maximum"] for row in rows),
                  mean_error=_mean([row["mean_error"] for row in rows]),
                  mean_component_mse=mean_mse, mean_component_rms=math.sqrt(mean_mse),
                  spatial_centered_mse=spatial_mse, spatial_centered_rms=math.sqrt(spatial_mse),
                  centered_mse=spatial_mse, centered_rms=math.sqrt(spatial_mse))
    for key in ("reference_rms", "remainder_rms", "feedback_rms", "reference_uncertainty_rms",
                "remainder_uncertainty_rms", "feedback_uncertainty_rms"):
        values = [row.get(key) for row in rows]
        result[key] = (_pooled_rms(values)
                       if all(_finite_number(value, nonnegative=True) for value in values) else None)
    _ratio_fields(result, resolved={name: all(row.get(name + "_resolved", False) for row in rows)
                                   for name in DENOMINATORS})
    result["reference_uncertainty_is_bound"] = False
    return result


def _parent_summary(rows, parent):
    result = dict(parent_id=parent, **_account(rows), **{key: None for key in ERROR_KEYS})
    families = {row.get("parent_family") for row in rows if row.get("parent_family") is not None}
    if len(families) > 1:
        raise ValueError("One parent cannot have multiple morphology families")
    result["parent_family"] = next(iter(families), None)
    good = [row for row in rows if _valid(row)]
    if good:
        available = _summed_metrics(good)
        result.update(available_mse=available["mse"], available_rms=available["rms"],
                      available_maximum=available["maximum"])
        if result["complete"]:
            result.update(available)
    return result


def summarize(metrics):
    """Summarize all planned rows with equal independent-parent weights.

    The mean component is the average *squared within-query mean error*, not
    the square of a pooled signed mean. Opposing query means must not cancel
    and then be mislabeled as spatial error. Complete aggregate values remain
    null when any planned row is missing, failed or unresolved.
    """
    rows = list(metrics)
    buckets, seen, parent_splits = defaultdict(list), set(), {}
    for row in rows:
        for key in ("method", "query_id", "parent_id", "split", "role"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{key} must be a nonempty string")
        identity = row["method"], row["query_id"]
        if identity in seen:
            raise ValueError("Duplicate query/method row")
        seen.add(identity)
        prior = parent_splits.setdefault(row["parent_id"], row["split"])
        if prior != row["split"]:
            raise ValueError("A physical parent cannot cross splits")
        key = row["method"], row["split"], row["role"], row.get("training_membership") or "not_applicable"
        buckets[key].append(row)
    groups, parent_metrics = [], []
    for (method, split, role, membership), subset in sorted(buckets.items()):
        metadata = dict(method=method, split=split, role=role, training_membership=membership)
        for key in META_KEYS:
            if key == "training_membership":
                continue
            values = {row.get(key) for row in subset}
            if len(values) > 1:
                raise ValueError(f"Inconsistent method metadata: {key}")
            metadata[key] = next(iter(values))
        by_parent = defaultdict(list)
        for row in subset:
            by_parent[row["parent_id"]].append(row)
        parents = [_parent_summary(values, parent) for parent, values in sorted(by_parent.items())]
        complete_parents = [row for row in parents if row["complete"]]
        group = dict(**metadata, **_account(subset), **{key: None for key in ERROR_KEYS},
                     expected_parent_count=len(parents), independent_parent_count=len(parents),
                     complete_parent_count=len(complete_parents),
                     missing_parent_ids=[p["parent_id"] for p in parents if not p["complete"]],
                     descendant_queries_are_independent=False, parents=parents,
                     aggregation="equal parent mass; equal query mass within each parent",
                     decomposition="average within-query mean-error squares plus average within-query centered MSE")
        if complete_parents:
            available = _summed_metrics(complete_parents)
            group.update(available_parent_balanced_rms=available["rms"], available_mse=available["mse"])
            if group["complete"]:
                group.update(available, median_parent_rms=float(np.median([p["rms"] for p in parents])),
                             worst_parent_rms=max(p["rms"] for p in parents))
        parent_metrics.extend(dict(**metadata, **parent) for parent in parents)
        groups.append(group)
    result = dict(learning_curves=groups, parent_metrics=parent_metrics)
    json.dumps(result, allow_nan=False)
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path, rows):
    rows = [{key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
            for row in rows]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _phase(value):
    value = re.sub(r"[^A-Za-z0-9_./-]", "_", value)
    if len(value) > 128:
        value = value[:110] + "-" + hashlib.sha256(value.encode()).hexdigest()[:16]
    return value


def _research_series(groups, selection, costs):
    series = []
    summary_keys = (*ERROR_KEYS, "expected", "measured", "failed", "missing", "unresolved",
                    "expected_parent_count", "complete_parent_count", "training_parent_count",
                    "calibration_source_count", "median_parent_rms", "worst_parent_rms",
                    "reference_rms", "remainder_rms", "feedback_rms", "reference_uncertainty_rms",
                    "relative_reference_rms", "relative_remainder_rms", "relative_feedback_rms")
    for group in groups:
        phase = _phase("accuracy/" + "/".join(group[key] for key in
                                              ("split", "method", "role", "training_membership")))
        values = {key: group[key] for key in summary_keys if _finite_number(group.get(key))}
        values["complete"] = int(group["complete"])
        series.append(dict(phase_suffix=phase, step=0, values=values))
    cells = selection.get("cells", []) if isinstance(selection, dict) else list(selection or [])
    fit_keys = ("data_loss", "ridge_loss", "penalty_loss", "objective", "zero_objective",
                "optimality_gap_upper_bound", "gradient_norm", "augmented_condition_number",
                "original_design_numerical_rank", "original_design_condition_number",
                "action_energy", "action_energy_penalty", "effective_degrees_of_freedom",
                "generalized_effective_degrees_of_freedom", "penalty_rank", "penalty_nullity",
                "original_augmented_condition_number", "original_augmented_numerical_rank")
    for cell in cells:
        for step, candidate in enumerate(cell.get("candidates", [])):
            values = {key: candidate[key] for key in ("ridge", "validation_mse")
                      if _finite_number(candidate.get(key))}
            diagnostics = candidate.get("diagnostics") or {}
            values.update({key: diagnostics.get(key) for key in fit_keys
                           if _finite_number(diagnostics.get(key))})
            values["eligible"] = int(bool(candidate.get("eligible")))
            phase = _phase("fit/" + str(cell.get("cell_id", "unknown")) + "/" + str(candidate.get("family", "unknown")))
            series.append(dict(phase_suffix=phase, step=step, values=values))
    # This is offline audit work, not inference timing. Do not surface arbitrary
    # telemetry, run IDs or booleans supplied in the cost record as research.
    allowed_costs = ("reference_seconds", "design_seconds", "fitting_seconds", "accuracy_evaluation_seconds",
                     "sensitivity_seconds", "evaluation_seconds", "input_read_seconds",
                     "original_reference_seconds", "original_design_seconds", "reused_reference_pairs",
                     "solver_seconds", "validation_seconds", "penalty_setup_seconds")
    if isinstance(costs, dict):
        values = {key: costs[key] for key in allowed_costs if _finite_number(costs.get(key), nonnegative=True)}
        if values:
            series.append(dict(phase_suffix="offline_cost", step=0, values=values))
    if len(series) > 500:
        raise ValueError("Bounded regularization research-series budget exceeded; no rows silently dropped")
    return series


def _plots(output, groups, selection):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".font-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "tdn-regularization-audit"}):
        chosen = [g for g in groups if g["split"] == "diagnostic" and g["role"] == "withheld"]
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4.2, .32 * len(chosen) + 1.5)))
        fig.subplots_adjust(left=.31, right=.97, top=.85, bottom=.21, wspace=.33)
        fig.suptitle("Bounded regularization audit: held-out physical error", x=.04, ha="left", fontweight="bold")
        labels = [g["method"] for g in chosen]
        for index, group in enumerate(chosen):
            if group["complete"]:
                axes[0].scatter(group["rms"], index, color="#0072B2", s=35)
                axes[1].barh(index, group["mean_component_mse"], color="#0072B2")
                axes[1].barh(index, group["spatial_centered_mse"], left=group["mean_component_mse"], color="#D55E00")
            else:
                axes[0].text(.02, index, "incomplete", transform=axes[0].get_yaxis_transform(), color="#777777")
        axes[0].set_yticks(range(len(labels)), labels=labels)
        axes[1].set_yticks(range(len(labels)), labels=[])
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.25)
        positive = [g["rms"] for g in chosen if g["complete"] and g["rms"] > 0]
        if positive:
            has_zero = any(g["complete"] and g["rms"] == 0 for g in chosen)
            if has_zero:
                axes[0].set_xscale("symlog", linthresh=min(positive) / 10)
            else:
                axes[0].set_xscale("log")
        axes[0].set_xlabel("Physical RMS error")
        axes[1].set_xlabel("Physical MSE: mean component + spatial component")
        axes[1].plot([], [], color="#0072B2", lw=7, label="Within-query mean-error square")
        axes[1].plot([], [], color="#D55E00", lw=7, label="Within-query centered MSE")
        axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=7)
        if not chosen:
            axes[0].text(.5, .5, "No diagnostic rows available", transform=axes[0].transAxes, ha="center")
        fig.text(.04, .035, "Equal independent-parent weights; no confidence interval or speedup claim.\n"
                 "The mean component averages query-wise squared means; opposite signed means do not cancel.", fontsize=8)
        fig.savefig(figures / "diagnostic_errors.png", dpi=180)
        fig.savefig(figures / "diagnostic_errors.svg", metadata={"Date": None})
        plt.close(fig)
        cells = selection.get("cells", []) if isinstance(selection, dict) else list(selection or [])
        fig, axes = plt.subplots(1, max(1, len(cells)), figsize=(max(10, 4.5 * len(cells)), 4.8), squeeze=False)
        fig.subplots_adjust(left=.09, right=.97, top=.77, bottom=.25, wspace=.30)
        fig.suptitle("Validation-only regularization paths", x=.07, ha="left", fontweight="bold")
        families = {"coefficient_ridge": ("#0072B2", "Coefficient penalty"),
                    "action_energy": ("#D55E00", "Action-energy penalty")}
        for ax, cell in zip(axes.flat, cells):
            for family, (color, label) in families.items():
                candidates = [candidate for candidate in cell.get("candidates", [])
                              if candidate.get("family") == family
                              and _finite_number(candidate.get("ridge"), nonnegative=True)
                              and candidate["ridge"] > 0
                              and _finite_number(candidate.get("validation_mse"), nonnegative=True)]
                candidates.sort(key=lambda candidate: candidate["ridge"])
                if candidates:
                    ax.plot([c["ridge"] for c in candidates], [math.sqrt(c["validation_mse"]) for c in candidates],
                            color=color, lw=1.2, label=label)
                    for candidate in candidates:
                        ax.scatter(candidate["ridge"], math.sqrt(candidate["validation_mse"]), s=28,
                                   facecolor=color if candidate.get("eligible") else "none", edgecolor=color)
                selected = (cell.get("selected") or {}).get(family)
                if (isinstance(selected, dict) and _finite_number(selected.get("ridge"), nonnegative=True)
                        and selected["ridge"] > 0 and _finite_number(selected.get("validation_mse"), nonnegative=True)):
                    ax.scatter(selected["ridge"], math.sqrt(selected["validation_mse"]), marker="*", s=115,
                               color=color, edgecolor="black", linewidth=.6, zorder=4)
            for candidate in cell.get("candidates", []):
                if (candidate.get("name", "").startswith("zero")
                        and _finite_number(candidate.get("validation_mse"), nonnegative=True)):
                    selected_zero = any(isinstance(c, dict) and c.get("name") == candidate.get("name")
                                        for c in (cell.get("selected") or {}).values())
                    ax.axhline(math.sqrt(candidate["validation_mse"]), color="#777777", ls="--", lw=1,
                               label="Zero correction" + (" (selected)" if selected_zero else ""))
                    break
            ax.set_xscale("log")
            observed_errors = [math.sqrt(c["validation_mse"]) for c in cell.get("candidates", [])
                               if _finite_number(c.get("validation_mse"), nonnegative=True)]
            if observed_errors and min(observed_errors) > 0:
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=max(observed_errors, default=1e-13) * 1e-3 or 1e-16)
            ax.set_xlabel("Penalty coefficient λ")
            ax.set_ylabel("Equal-parent validation RMS")
            ax.set_title(str(cell.get("cell_id", "cell")))
            ax.grid(alpha=.25)
        if not cells:
            axes.flat[0].text(.5, .5, "No fitted candidates available", ha="center", transform=axes.flat[0].transAxes)
        handles, labels = [], []
        for ax in axes.flat:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle); labels.append(label)
        if handles:
            fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.06, .89), ncol=3, frameon=False)
        fig.text(.07, .06, "Stars mark sealed selections; hollow points are ineligible. Diagnostic data do not select penalties.\n"
                 "Penalty definitions and units differ: compare validation errors, not λ across families.\n"
                 "No inference timings are measured in this audit.",
                 fontsize=8)
        fig.savefig(figures / "validation_paths.png", dpi=180)
        fig.savefig(figures / "validation_paths.svg", metadata={"Date": None})
        plt.close(fig)


def render(output, metrics, selection, costs, *, plots=True):
    """Write analysis tables, bounded Raven research rows, and optional figures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = list(metrics)
    result = summarize(metrics)
    series = _research_series(result["learning_curves"], selection, costs)
    _write_csv(output / "metrics.csv", metrics)
    for key in ("learning_curves", "parent_metrics"):
        _write_json(output / f"{key}.json", result[key])
        _write_csv(output / f"{key}.csv", result[key])
    _write_json(output / "regularization_metrics.json", series)
    if plots:
        _plots(output, result["learning_curves"], selection)
    return result
