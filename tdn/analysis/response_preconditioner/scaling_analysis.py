"""Phase-one parent/source scaling on the unchanged 40-column R7 family.

These helpers acquire no references and change no architecture or objective.
Planned rows, including unavailable rows, are inputs: dropping an unavailable
training/validation query never makes its cell eligible. Validation selects
only the original positive ridges and initialization. Diagnostic metrics are
aggregated over independent parents, without source-level confidence claims.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np

from .fitting import fit_parent_balanced_ridge


COEFFICIENTS = 40
RIDGES = (1e-10, 1e-6)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _selection_seal(cell):
    sealed = {k: v for k, v in cell.items() if k not in ("selection_seal", "fitting_seconds")}
    sealed["candidates"] = [{k: v for k, v in c.items() if k != "fitting_seconds"}
                            for c in cell["candidates"]]
    sealed["selected"] = {k: v for k, v in cell["selected"].items() if k != "fitting_seconds"}
    return _digest(sealed)


def _array(value, name, *, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _physical_vectors(row):
    teacher = _array(row["teacher"], "teacher")
    if teacher.ndim != 1 or not teacher.size:
        raise ValueError("teacher must be a nonempty vector")
    base = _array(row["base"], "base", shape=teacher.shape)
    return _array(teacher - base, "target", shape=teacher.shape), base, teacher


def _vectors(row):
    target, base, teacher = _physical_vectors(row)
    design = _array(row["design"], "design", shape=(teacher.size, COEFFICIENTS))
    return design, target, base, teacher


def _counts(values, name):
    values = tuple(values)
    if (not values or any(type(x) is not int or x <= 0 for x in values)
            or sorted(set(values)) != list(values)):
        raise ValueError(f"{name} must be strictly increasing positive integers")
    return values


def _identities(rows):
    ids, slots, parents = set(), set(), {}
    for row in rows:
        for key in ("query_id", "parent_id", "split", "context_id", "role", "family"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{key} must be a nonempty string")
        if row["query_id"] in ids:
            raise ValueError("Duplicate query_id")
        ids.add(row["query_id"])
        if type(row.get("source_index")) is not int or row["source_index"] < 0:
            raise ValueError("source_index must be a nonnegative integer")
        if row["parent_id"] in parents and parents[row["parent_id"]] != row["split"]:
            raise ValueError("A parent cannot cross splits")
        parents[row["parent_id"]] = row["split"]
        slot = (row["parent_id"], row["context_id"], row["role"], row["source_index"])
        if slot in slots:
            raise ValueError("Duplicate parent/context/role/source slot")
        slots.add(slot)


def _account(rows):
    statuses = Counter(row.get("status", "NOT_ACQUIRED") for row in rows)
    invalid = []
    for row in rows:
        if row.get("status") == "ACCEPTED":
            try:
                _vectors(row)
            except (ValueError, KeyError, TypeError) as error:
                invalid.append(dict(query_id=row["query_id"], error=str(error)))
    return dict(planned=len(rows), status_counts=dict(sorted(statuses.items())),
                accepted=statuses["ACCEPTED"], missing=statuses["NOT_ACQUIRED"],
                failed=statuses["FAILED"], unresolved=statuses["UNRESOLVED"],
                invalid_numeric=len(invalid), invalid_rows=invalid,
                complete=bool(rows) and statuses["ACCEPTED"] == len(rows) and not invalid)


def _parent_loss(rows, coefficients):
    values = defaultdict(list)
    for row in rows:
        design, target, _, _ = _vectors(row)
        residual = design @ coefficients - target
        loss = float(np.mean(residual * residual))
        if not math.isfinite(loss):
            raise ArithmeticError("Nonfinite validation loss")
        values[row["parent_id"]].append(loss)
    return float(np.mean([np.mean(v) for v in values.values()]))


def fit_nested_models(training_rows, validation_rows, *, training_parent_ids,
                      parent_counts=(8, 16, 32), source_counts=(2, 4, 8),
                      ridge_menu=RIDGES, expected_contexts_per_parent=None,
                      expected_withheld_sources=4, expected_validation_parent_ids=None,
                      budget_check=None, on_cell=None):
    """Fit each frozen nested cell; input contains every planned query row.

    Designs are [cells,40], base/teacher are [cells]. Calibration source indices
    begin at zero and cohorts are prefixes of ``training_parent_ids``. All
    validation withheld rows select each cell; no diagnostic row is accepted.
    Equal physical-cell mass within a query and equal query mass within a
    parent give equal total mass to each independent parent. With the frozen
    common grid this is the original R7 parent-balanced objective.
    """
    training_rows, validation_rows = list(training_rows), list(validation_rows)
    parent_counts = _counts(parent_counts, "parent_counts")
    source_counts = _counts(source_counts, "source_counts")
    if tuple(ridge_menu) != RIDGES:
        raise ValueError("Phase one preserves the original ridge menu")
    parents = list(training_parent_ids)
    if (len(parents) != len(set(parents)) or len(parents) < max(parent_counts)
            or any(not isinstance(p, str) or not p for p in parents)):
        raise ValueError("Ordered unique training parent IDs must cover every cohort")
    if expected_contexts_per_parent is not None and (
            type(expected_contexts_per_parent) is not int or expected_contexts_per_parent < 1):
        raise ValueError("expected_contexts_per_parent must be positive")
    if type(expected_withheld_sources) is not int or expected_withheld_sources < 1:
        raise ValueError("expected_withheld_sources must be positive")
    _identities(training_rows + validation_rows)
    if any((r["split"], r["role"]) != ("train", "calibration") for r in training_rows):
        raise ValueError("Only train calibration rows may fit a cell")
    if any((r["split"], r["role"]) != ("validation", "withheld") for r in validation_rows):
        raise ValueError("Only validation withheld rows may select a cell")
    if any(r["parent_id"] not in parents for r in training_rows):
        raise ValueError("Training row outside declared parent order")
    validation_account = _account(validation_rows)
    validation_problems = []
    observed_validation_parents = {r["parent_id"] for r in validation_rows}
    validation_parents = (sorted(observed_validation_parents) if expected_validation_parent_ids is None
                          else list(expected_validation_parent_ids))
    if (len(validation_parents) != len(set(validation_parents))
            or any(not isinstance(p, str) or not p for p in validation_parents)
            or set(validation_parents) & set(parents)):
        raise ValueError("Expected validation parents must be unique and disjoint from training")
    if observed_validation_parents - set(validation_parents):
        raise ValueError("Validation row outside expected parent identities")
    if not validation_parents or observed_validation_parents != set(validation_parents):
        validation_problems.append("Incomplete expected validation parent coverage")
    for parent in validation_parents:
        contexts = defaultdict(list)
        for row in validation_rows:
            if row["parent_id"] == parent:
                contexts[row["context_id"]].append(row["source_index"])
        if expected_contexts_per_parent is not None and len(contexts) != expected_contexts_per_parent:
            validation_problems.append(f"Incomplete validation contexts for {parent}")
        if any(sorted(indices) != list(range(expected_withheld_sources)) for indices in contexts.values()):
            validation_problems.append(f"Incomplete validation sources for {parent}")
    cells = []
    for parent_count in parent_counts:
        cohort = parents[:parent_count]
        for source_count in source_counts:
            if budget_check is not None:
                budget_check()
            started = time.perf_counter()
            rows = [r for r in training_rows if r["parent_id"] in cohort
                    and r["source_index"] < source_count]
            account = _account(rows)
            problems = list(validation_problems)
            for parent in cohort:
                by_context = defaultdict(list)
                for row in rows:
                    if row["parent_id"] == parent:
                        by_context[row["context_id"]].append(row["source_index"])
                if not by_context or (expected_contexts_per_parent is not None
                                      and len(by_context) != expected_contexts_per_parent):
                    problems.append(f"Incomplete context coverage for {parent}")
                for context, indices in by_context.items():
                    if sorted(indices) != list(range(source_count)):
                        problems.append(f"Incomplete or duplicate calibration sources for {context}")
            if not account["complete"]:
                problems.append("Unavailable or nonfinite training rows; no rows dropped")
            if not validation_account["complete"]:
                problems.append("Unavailable or nonfinite validation rows; no rows dropped")
            cell = dict(model_id=f"fit-p{parent_count:03d}-s{source_count:02d}",
                        training_parent_count=parent_count, calibration_source_count=source_count,
                        training_parent_ids=cohort, status="BLOCKED", candidates=[], selected=None,
                        train_query_ids=[r["query_id"] for r in rows],
                        validation_query_ids=[r["query_id"] for r in validation_rows],
                        training_accounting=account, validation_accounting=validation_account,
                        reasons=problems, selection_seal=None,
                        diagnostic_truth_accessed=False, training_withheld_truth_accessed=False,
                        criterion="equal-parent validation withheld physical MSE; first candidate wins exact ties",
                        objective="equal-parent physical action MSE plus original coefficient L2 ridge",
                        automatic_expansion=False)
            if not problems:
                zero = np.zeros(COEFFICIENTS)
                candidates = [dict(name="zero_readout", ridge=None, coefficients=zero.tolist(),
                                   eligible=True, validation_loss=_parent_loss(validation_rows, zero),
                                   fitting_seconds=0., diagnostics=dict(status="INITIALIZATION"))]
                matrices = [_vectors(row) for row in rows]
                A = np.concatenate([m[0] for m in matrices])
                b = np.concatenate([m[1] for m in matrices])
                ids = [r["parent_id"] for r, m in zip(rows, matrices) for _ in m[1]]
                masses = np.concatenate([np.full(m[1].size, 1 / m[1].size) for m in matrices])
                for ridge in ridge_menu:
                    if budget_check is not None:
                        budget_check()
                    began = time.perf_counter()
                    try:
                        fitted = fit_parent_balanced_ridge(A, b, ids, ridge=ridge, row_weights=masses)
                        coeff = fitted["coefficients"].numpy()
                        diag = fitted["diagnostics"]
                        candidate = dict(name=f"ridge_{ridge:g}", ridge=ridge,
                                         coefficients=coeff.tolist(), diagnostics=diag,
                                         eligible=bool(diag["optimality_verified"] and
                                                       diag["baseline_nonregression_verified"]),
                                         validation_loss=_parent_loss(validation_rows, coeff))
                    except (ValueError, ArithmeticError, RuntimeError) as error:
                        candidate = dict(name=f"ridge_{ridge:g}", ridge=ridge, coefficients=None,
                                         eligible=False, validation_loss=None,
                                         error=f"{type(error).__name__}: {error}")
                    candidate["fitting_seconds"] = time.perf_counter() - began
                    candidates.append(candidate)
                cell.update(status="SELECTED", candidates=candidates,
                            selected=min((c for c in candidates if c["eligible"]),
                                         key=lambda c: c["validation_loss"]))
                # Seal binds data identity, candidates and selection, independent
                # of nondeterministic timing. The workflow binds NPZ/source hashes.
                cell["selection_seal"] = _selection_seal(cell)
            cell["fitting_seconds"] = time.perf_counter() - started
            cells.append(cell)
            if on_cell is not None:
                on_cell(cell)
    return dict(schema="tdn.response_preconditioner.scaling.fits.v1", cells=cells,
                all_selections_sealed=all(c["status"] == "SELECTED" for c in cells),
                automatic_expansion=False, architecture_unchanged=True, ridge_menu=list(RIDGES))


def evaluate_prediction(row, prediction, *, method, training_parent_count=None,
                        calibration_source_count=None, training_membership="not_applicable"):
    """Score one authoritative physical prediction, including classical controls.

    The same prediction supplies every error metric. Unavailable references and
    nonfinite predictions remain explicit failed/unresolved/missing rows.
    """
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a nonempty string")
    _identities([row])
    item = {key: row[key] for key in ("query_id", "parent_id", "parent_family", "split",
            "context_id", "source_index", "role", "family", "start", "stop") if key in row}
    item.update(method=method, training_parent_count=training_parent_count,
                calibration_source_count=calibration_source_count,
                training_membership=training_membership, status="MISSING")
    if row.get("status") != "ACCEPTED":
        item.update(status={"FAILED": "FAILED", "UNRESOLVED": "UNRESOLVED"}.get(
            row.get("status"), "MISSING"), reference_status=row.get("status", "NOT_ACQUIRED"))
    else:
        try:
            target, base, teacher = _physical_vectors(row)
            prediction = _array(prediction, "prediction", shape=teacher.shape)
            error = prediction - teacher
            mse = float(np.mean(error * error))
            rms = math.sqrt(mse)
            mean = float(np.mean(error))
            response_rms = float(np.sqrt(np.mean(teacher * teacher)))
            remainder_rms = float(np.sqrt(np.mean(target * target)))
            reference = row.get("reference", {})
            uncertainty = reference.get("uncertainty_rms")
            uncertainty_max = reference.get("uncertainty_maximum")
            for value in (uncertainty, uncertainty_max):
                if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                    raise ValueError("Reference uncertainty must be finite and nonnegative")
            feedback_rms = None
            signal_uncertainty = uncertainty
            matched_reference = reference.get("matched_mean", {})
            matched_uncertainty = matched_reference.get("uncertainty_rms")
            if matched_uncertainty is not None and (not math.isfinite(float(matched_uncertainty))
                                                     or float(matched_uncertainty) < 0):
                raise ValueError("Matched-mean uncertainty must be finite and nonnegative")
            if row.get("matched_mean") is not None:
                matched = _array(row["matched_mean"], "matched_mean", shape=teacher.shape)
                feedback_rms = float(np.sqrt(np.mean((teacher - matched) ** 2)))
            remainder_status = ("UNRESOLVED_REFERENCE" if signal_uncertainty is None or not reference.get("accepted", False)
                                else "RESOLVED" if remainder_rms > 10 * signal_uncertainty
                                else "UNRESOLVED_NEAR_NULL")
            feedback_uncertainty = (signal_uncertainty + matched_uncertainty
                                    if signal_uncertainty is not None and matched_uncertainty is not None else None)
            feedback_status = ("UNRESOLVED_REFERENCE" if feedback_uncertainty is None or feedback_rms is None
                               or not reference.get("accepted", False) or not matched_reference.get("accepted", False)
                               else "RESOLVED" if feedback_rms > 10 * feedback_uncertainty
                               else "UNRESOLVED_NEAR_NULL")
            item.update(status="MEASURED", mse=mse, rms=rms,
                        maximum=float(np.max(np.abs(error))), mean_error=mean,
                        centered_rms=float(np.sqrt(np.mean((error - mean) ** 2))),
                        reference_rms=response_rms, remainder_rms=remainder_rms,
                        feedback_rms=feedback_rms, reference_uncertainty_rms=uncertainty,
                        reference_uncertainty_maximum=uncertainty_max,
                        relative_reference_rms=rms / response_rms if response_rms > 0 else None,
                        relative_remainder_rms=rms / remainder_rms if remainder_status == "RESOLVED" else None,
                        relative_feedback_rms=rms / feedback_rms if feedback_status == "RESOLVED" else None,
                        remainder_status=remainder_status, feedback_status=feedback_status,
                        signal_resolution_multiple=10.,
                        signal_resolution_uncertainty="sum of applicable estimated reference RMS errors",
                        error_to_uncertainty_rms=rms / uncertainty if uncertainty else None,
                        reference_uncertainty_is_bound=False)
            json.dumps(item, allow_nan=False)
        except (ValueError, ArithmeticError, KeyError, TypeError) as error:
            item.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            # Do not retain partially computed nonfinite quantities.
            item = {k: v for k, v in item.items()
                    if not isinstance(v, (float, np.floating)) or math.isfinite(v)}
    return item


def evaluate_model(cell, rows, *, predictions=None):
    """Score a sealed affine model, optionally from fresh authoritative calls.

    When ``predictions`` is supplied it maps every accepted query ID to its
    complete fresh action. A missing entry fails; it never falls back to a
    design reconstruction. Without this argument, retained designs support
    pure analysis and replay. No fit or selection occurs here.
    """
    if cell.get("status") != "SELECTED" or not cell.get("selection_seal"):
        raise ValueError("Evaluation requires a sealed selected cell")
    if _selection_seal(cell) != cell["selection_seal"]:
        raise ValueError("Selection seal does not match retained candidates and coefficients")
    coefficients = _array(cell["selected"]["coefficients"], "coefficients", shape=(COEFFICIENTS,))
    rows = list(rows)
    _identities(rows)
    output = []
    for row in rows:
        prediction = None
        calculation_error = None
        if row.get("status") == "ACCEPTED":
            try:
                if predictions is not None:
                    if row["query_id"] not in predictions:
                        raise ValueError("Missing authoritative prediction")
                    prediction = predictions[row["query_id"]]
                else:
                    design, _, base, _ = _vectors(row)
                    prediction = base + design @ coefficients
            except (ValueError, ArithmeticError, KeyError, TypeError) as error:
                calculation_error = f"{type(error).__name__}: {error}"
        membership = (("in_cohort" if row["parent_id"] in cell["training_parent_ids"] else "outside_cohort")
                      if row["split"] == "train" else "not_training_split")
        item = evaluate_prediction(row, prediction, method=cell["model_id"],
                                   training_parent_count=cell["training_parent_count"],
                                   calibration_source_count=cell["calibration_source_count"],
                                   training_membership=membership)
        item.update(model_id=cell["model_id"], selection_seal=cell["selection_seal"],
                    prediction_scope="fresh_action" if predictions is not None else "retained_design_reconstruction")
        if calculation_error is not None:
            item.update(status="FAILED", error=calculation_error)
        output.append(item)
    return output


def _metric_valid(row):
    return row.get("status") == "MEASURED" and all(
        isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)
        and math.isfinite(row[k]) and row[k] >= 0 for k in ("rms", "maximum")) and math.isfinite(row["rms"] * row["rms"])


def _parent_summary(rows, parent_id):
    valid = [r for r in rows if _metric_valid(r)]
    complete = bool(rows) and len(valid) == len(rows)
    values = dict(parent_id=parent_id, planned_queries=len(rows), measured_queries=len(valid),
                  missing_queries=sum(r.get("status") in ("MISSING", "NOT_ACQUIRED", "NOT_EVALUATED") for r in rows),
                  failed_queries=sum(r.get("status") == "FAILED" for r in rows),
                  unresolved_queries=sum(r.get("status") == "UNRESOLVED" for r in rows),
                  invalid_measured_queries=sum(r.get("status") == "MEASURED" and not _metric_valid(r)
                                               for r in rows),
                  complete=complete, rms=None, mse=None, maximum=None)
    values["remainder_resolved"] = complete and all(r.get("remainder_status") == "RESOLVED" for r in rows)
    values["feedback_resolved"] = complete and all(r.get("feedback_status") == "RESOLVED" for r in rows)
    values["mean_error"], values["centered_rms"] = None, None
    if valid:
        mse = float(np.mean([r["rms"] ** 2 for r in valid]))
        values.update(available_rms=math.sqrt(mse), available_mse=mse,
                      available_maximum=max(r["maximum"] for r in valid))
        if complete:
            values.update(rms=math.sqrt(mse), mse=mse, maximum=values["available_maximum"])
            means = [r.get("mean_error") for r in valid]
            if all(isinstance(x, (int, float)) and math.isfinite(x) for x in means):
                mean = float(np.mean(means))
                values.update(mean_error=mean, centered_rms=math.sqrt(max(0., mse - mean*mean)))
        for source, destination in (("reference_rms", "reference_rms"),
                                    ("remainder_rms", "remainder_rms"),
                                    ("feedback_rms", "feedback_rms"),
                                    ("reference_uncertainty_rms", "reference_uncertainty_rms")):
            present = [r.get(source) for r in valid]
            usable = all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in present)
            values[destination] = math.sqrt(float(np.mean(np.square(present)))) if usable and complete else None
    return values


def _aggregate(rows, expected):
    by_parent = defaultdict(list)
    for row in rows:
        by_parent[row["parent_id"]].append(row)
    unexpected = sorted(set(by_parent) - set(expected))
    if unexpected:
        raise ValueError(f"Metric parents outside declared split: {unexpected}")
    summaries = [_parent_summary(by_parent[parent], parent) for parent in expected]
    complete = bool(summaries) and all(p["complete"] for p in summaries)
    measured = [p for p in summaries if p["complete"]]
    result = dict(independent_parent_count=len(expected), complete_parent_count=len(measured),
                  complete=complete, parents=summaries, rms=None, maximum=None,
                  missing_parent_ids=[p["parent_id"] for p in summaries if not p["complete"]],
                  descendant_queries_are_independent=False)
    if measured:
        mse = float(np.mean([p["mse"] for p in measured]))
        result.update(available_parent_balanced_rms=math.sqrt(mse),
                      available_worst_parent_rms=max(p["rms"] for p in measured))
        if complete:
            result.update(mse=mse, rms=math.sqrt(mse), maximum=max(p["maximum"] for p in measured),
                          median_parent_rms=float(np.median([p["rms"] for p in measured])),
                          worst_parent_rms=max(p["rms"] for p in measured))
            means = [p.get("mean_error") for p in measured]
            if all(x is not None for x in means):
                mean = float(np.mean(means))
                result.update(mean_error=mean, centered_rms=math.sqrt(max(0., mse - mean*mean)))
        for key in ("reference_rms", "remainder_rms", "feedback_rms", "reference_uncertainty_rms"):
            values = [p.get(key) for p in measured]
            denominator = (math.sqrt(float(np.mean(np.square(values))))
                           if complete and all(v is not None for v in values) else None)
            result[key] = denominator
            resolved = (all(p.get(key.replace("_rms", "_resolved"), False) for p in measured)
                        if key in ("remainder_rms", "feedback_rms") else True)
            result["error_to_" + key] = math.sqrt(mse) / denominator if denominator and resolved else None
            if key in ("remainder_rms", "feedback_rms"):
                result[key.replace("_rms", "_resolved")] = complete and resolved
    return result


def summarize_learning_curves(metric_rows, *, expected_parent_ids):
    """Aggregate complete parent results and pair both scaling axes descriptively.

    Inputs may include classical controls. A missing entire parent remains in
    the denominator. The caller must retain one row per planned query/method;
    this helper rejects duplicates and cannot infer omitted individual queries.
    No p-values, confidence intervals, automatic expansion or utility gate.
    """
    rows = list(metric_rows)
    expected = {split: list(ids) for split, ids in expected_parent_ids.items()}
    seen, parent_split = set(), {}
    for split, ids in expected.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Expected parent IDs must be nonempty and unique per split")
        for parent in ids:
            if parent in parent_split:
                raise ValueError("Expected parent crosses splits")
            parent_split[parent] = split
    for row in rows:
        key = row["method"], row["query_id"]
        if key in seen:
            raise ValueError("Duplicate metric query/method")
        seen.add(key)
        if row["split"] not in expected:
            raise ValueError("Undeclared metric split")
    buckets = defaultdict(list)
    for row in rows:
        membership = row.get("training_membership", "not_applicable")
        buckets[(row["method"], row["split"], row.get("role", "withheld"), membership)].append(row)
    groups = []
    parent_rows = []
    for (method, split, role, membership), subset in sorted(buckets.items()):
        # Training membership changes with cohort size. Keep those descriptive
        # views separate and do not treat them as a common paired test bank.
        members = (expected[split] if split != "train" or membership == "not_applicable"
                   else [p for p in expected[split] if any(r["parent_id"] == p for r in subset)])
        group = dict(method=method, split=split, role=role, training_membership=membership,
                     training_parent_count=subset[0].get("training_parent_count"),
                     calibration_source_count=subset[0].get("calibration_source_count"),
                     **_aggregate(subset, members))
        group["strata"] = {}
        for key in ("family", "parent_family"):
            stratum = {}
            for value in sorted({r[key] for r in subset if key in r}):
                selected = [r for r in subset if r.get(key) == value]
                parent_ids = (members if key == "family" else
                              [p for p in members if any(r["parent_id"] == p for r in selected)])
                stratum[value] = _aggregate(selected, parent_ids)
            group["strata"][key] = stratum
        intervals = sorted({(r["start"], r["stop"]) for r in subset if "start" in r and "stop" in r})
        group["strata"]["interval"] = {
            f"{start:g}:{stop:g}": _aggregate([r for r in subset if r.get("start") == start and r.get("stop") == stop], members)
            for start, stop in intervals}
        for parent in group["parents"]:
            parent_rows.append(dict(method=method, split=split, role=role, training_membership=membership,
                                    training_parent_count=group["training_parent_count"],
                                    calibration_source_count=group["calibration_source_count"], **parent))
        groups.append(group)
    result = dict(schema="tdn.response_preconditioner.scaling.learning_curves.v1", groups=groups, parent_rows=parent_rows,
                  aggregation="equal mean squared error weight per independent parent; equal query weight within parent",
                  uncertainty_scope="descriptive parent values only; teacher uncertainty is not a certificate",
                  automatic_expansion=False, scientific_promotion=False,
                  metric_accounting=dict(planned=len(rows), measured=sum(_metric_valid(r) for r in rows),
                                         status_counts=dict(Counter(r.get("status", "MISSING") for r in rows)),
                                         invalid_measured=sum(r.get("status") == "MEASURED" and not _metric_valid(r) for r in rows)),
                  paired_parent_count=[], paired_source_count=[])
    for changing, fixed, output in (("training_parent_count", "calibration_source_count", "paired_parent_count"),
                                    ("calibration_source_count", "training_parent_count", "paired_source_count")):
        settings = defaultdict(list)
        for group in groups:
            if group["split"] != "train" and group.get(changing) is not None and group.get(fixed) is not None:
                settings[(group["split"], group["role"], group[fixed])].append(group)
        for (split, role, fixed_count), values in sorted(settings.items()):
            values.sort(key=lambda g: g[changing])
            for before, after in zip(values, values[1:]):
                pairs = []
                for a, b in zip(before["parents"], after["parents"]):
                    if a["complete"] and b["complete"]:
                        pairs.append(dict(parent_id=a["parent_id"], before_rms=a["rms"], after_rms=b["rms"],
                                          rms_difference=b["rms"] - a["rms"],
                                          mse_difference=b["mse"] - a["mse"],
                                          improvement_ratio=a["rms"] / b["rms"] if b["rms"] > 0 else None,
                                          improved=b["rms"] < a["rms"]))
                deltas = [p["rms_difference"] for p in pairs]
                result[output].append(dict(split=split, role=role, fixed_axis=fixed, fixed_count=fixed_count,
                    before_method=before["method"], after_method=after["method"],
                    before_count=before[changing], after_count=after[changing],
                    expected_parent_count=len(expected[split]), paired_parent_count=len(pairs), parents=pairs,
                    complete=len(pairs) == len(expected[split]),
                    median_rms_difference=float(np.median(deltas)) if pairs else None,
                    worst_rms_difference=max(deltas) if pairs else None,
                    improved_parent_fraction=sum(p["improved"] for p in pairs) / len(pairs) if pairs else None,
                    descriptive_only=True))
    json.dumps(result, allow_nan=False)
    return result
