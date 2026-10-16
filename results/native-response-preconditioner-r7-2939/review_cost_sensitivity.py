#!/usr/bin/env python3
"""Read-only descriptive audit of retained R7 timing and perturbation arrays.

No TDN imports, uploaded executable code, teachers, model actions, refitting,
threshold selection, or new scientific labels. JSON goes to stdout. Input NPZ
files are loaded with allow_pickle=False. Invocation is documented in README.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np


def describe(values):
    x = np.asarray(list(values), dtype=float)
    return {"count": len(x), "minimum": float(x.min()),
            "median": float(np.median(x)), "maximum": float(x.max())}


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(list(values), dtype=float) ** 2)))


def close(actual, expected, label, rtol=3e-10, atol=2e-25):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"Retained evidence replay differs: {label}")


def main(study):
    study = study.resolve()
    hashes = {}

    def read_bytes(relative):
        path = study / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular evidence file: {relative}")
        data = path.read_bytes()
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        return data

    def read_json(relative):
        return json.loads(read_bytes(relative))

    def read_npz(relative):
        read_bytes(relative)
        with np.load(study / relative, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"Nonfinite retained numerical array: {relative}")
        return arrays

    protocol = read_json("protocol.json")
    assert protocol["schema"] == "tdn.response_preconditioner.r7.v1"
    timing = read_json("timing_records.json")
    work_precision = read_json("work_precision.json")
    metrics = read_json("metrics.json")
    query_index = read_json("query_index.json")
    selection = read_json("selection.json")
    summary = read_json("summary.json")
    work = read_json("work.json")
    journal = [json.loads(line) for line in read_bytes("journal.jsonl").splitlines()]
    by_query = {row["query_id"]: row for row in query_index}
    acquired_ids = [row["query_id"] for row in journal if row["event"] == "query_finished"]
    assert len(set(acquired_ids)) == len(acquired_ids) == 192
    queries = [by_query[key] for key in acquired_ids]
    arrays = {row["query_id"]: read_npz(row["array_path"]) for row in queries}
    assert len(timing) == 336 and len(work_precision) == 48
    assert Counter(row["phase"] for row in timing) == {"cold": 48, "warmup": 48, "paired": 240}
    assert all(row["status"] == "PASS" for row in timing)
    methods = sorted({row["method"] for row in work_precision})
    contexts = sorted({row["context_id"] for row in work_precision})
    assert len(methods) == 8 and len(contexts) == 6
    timing_rows = []
    per_context = []
    ratios_by_method = {method: [] for method in methods if method != "fixed_feature_fitted"}

    for context in contexts:
        group = {row["method"]: row for row in work_precision if row["context_id"] == context}
        records = [row for row in timing if row["context_id"] == context]
        paired = {(row["method"], row["round"]): row["seconds"] for row in records if row["phase"] == "paired"}
        assert len(paired) == 40
        fitted = group["fixed_feature_fitted"]
        comparison = []
        for method, row in group.items():
            observations = [paired[method, index] for index in range(5)]
            close(np.median(observations), row["median_seconds"], "timing median")
            close(min(observations), row["minimum_seconds"], "timing minimum")
            close(max(observations), row["maximum_seconds"], "timing maximum")
            observations_metrics = [x for x in metrics if x["context_id"] == context and x["method"] == method and x["role"] == "withheld"]
            assert len(observations_metrics) == 4
            close(rms(x["rms"] for x in observations_metrics), row["pooled_source_rms"], "pooled RMS")
            close(max(x["maximum"] for x in observations_metrics), row["maximum"], "maximum")
            assert len({x["output_sha256"] for x in records if x["method"] == method}) == 1
            timing_rows.append(dict(context_id=context, method=method,
                pooled_source_rms=row["pooled_source_rms"], maximum=row["maximum"],
                paired_seconds=observations, paired_seconds_summary=describe(observations),
                paired_maximum_to_minimum=max(observations)/min(observations),
                cold_seconds=next(x["seconds"] for x in records if x["method"] == method and x["phase"] == "cold"),
                warmup_seconds=next(x["seconds"] for x in records if x["method"] == method and x["phase"] == "warmup")))
            if method == "fixed_feature_fitted":
                continue
            ratios = [paired["fixed_feature_fitted", index]/paired[method, index] for index in range(5)]
            ratios_by_method[method].extend(ratios)
            comparison.append(dict(method=method, fitted_to_competitor_rms=fitted["pooled_source_rms"]/row["pooled_source_rms"],
                fitted_to_competitor_maximum=fitted["maximum"]/row["maximum"],
                fitted_to_competitor_median_seconds=fitted["median_seconds"]/row["median_seconds"],
                fitted_to_competitor_paired_seconds=ratios,
                paired_ratio_summary=describe(ratios),
                fitted_faster_rounds=sum(value < 1 for value in ratios)))

        def dominates(first, second, fields):
            return all(first[field] <= second[field] for field in fields) and any(first[field] < second[field] for field in fields)

        axes = ("median_seconds", "pooled_source_rms", "maximum")
        frontier = [name for name, row in group.items() if not any(dominates(other, row, axes) for other in group.values())]
        dominated = {name: [other_name for other_name, other in group.items() if dominates(other, row, axes)]
                     for name, row in group.items()}
        per_context.append(dict(context_id=context,
            start=next(x["start"] for x in queries if x["context_id"] == context),
            stop=next(x["stop"] for x in queries if x["context_id"] == context),
            comparisons=comparison, empirical_frontier=frontier, dominated_by=dominated))

    timing_review = dict(
        records=len(timing), phase_counts=dict(Counter(row["phase"] for row in timing)),
        complete_source_calls=len(timing)*4, deployable_methods=methods,
        scope="Four-source sequential bundle with a fresh original path/setup per source; no cross-source cache. One numerical thread, CPU, gradients enabled.",
        cold_scope="First benchmark invocation after accuracy acquisition; already warm process, not process-startup timing.",
        axes_for_descriptive_frontier=["median_seconds", "pooled_source_rms", "maximum"],
        uncertainty="Five paired randomized rounds per context only; no independent repetition/parent confidence claim, no frozen practical margin or utility threshold.",
        superiority=False, matched_tolerance_speedup=False,
        fitted_to_competitor_pooled_paired_ratios={name: describe(values) for name, values in ratios_by_method.items()},
        per_context=per_context, observations=timing_rows)

    def matching(query_rows, method):
        keys = {row["query_id"] for row in query_rows}
        rows = [row for row in metrics if row["query_id"] in keys and row["method"] == method]
        assert len(rows) == len(keys)
        return rows

    def matrix(query_rows, key):
        return np.concatenate([arrays[row["query_id"]][key].reshape(arrays[row["query_id"]][key].shape[0], -1).T for row in query_rows])

    sensitivity_rows = []
    maximum_replay_errors = Counter()
    all_fits = {path.stem: read_json(path.relative_to(study)) for path in sorted((study / "fits").glob("*.json"))}
    paths = sorted((study / "sensitivity").glob("*.json"))
    assert len(paths) == 49
    for path in paths:
        diagnostics = read_json(path.relative_to(study))
        retained = read_npz(path.relative_to(study).with_suffix(".npz"))
        if path.stem == "global":
            cal = [row for row in queries if row["role"] == "calibration" and row["split"] == "train"]
            test = [row for row in queries if row["role"] == "withheld"]
            key, method, context = "design", "fixed_feature_fitted", None
            expected_ridge = selection["selected"]["ridge"]
            coefficient_norm = float(np.linalg.norm(selection["selected"]["coefficients"]))
            fit_diagnostics = selection["selected"]["diagnostics"]
        else:
            context, method = path.stem.split("-local_ridge_")
            method = "local_ridge_" + method
            cal = [row for row in queries if row["role"] == "calibration" and row["context_id"] == context]
            test = [row for row in queries if row["role"] == "withheld" and row["context_id"] == context]
            key = "branches"
            fit = next(row for row in all_fits[context]["fits"] if row["name"] == method)
            eta = arrays[cal[0]["query_id"]]["features"]
            expected_ridge = fit["ridge"]/(protocol["local_context_weight"]*float(eta@eta))
            coefficient_norm = float(np.linalg.norm(fit["coefficients"]))
            fit_diagnostics = fit["diagnostics"]
        A, At = matrix(cal, key), matrix(test, key)
        weights, test_weights = retained["row_weights"], retained["test_row_weights"]
        close(weights, np.full(len(A), 1/len(A)), "uniform equal-mass calibration rows")
        close(test_weights, np.full(len(At), 1/len(At)), "uniform equal-mass test rows")
        close(diagnostics["ridge"], expected_ridge, "induced or global ridge")
        uncertainty = np.concatenate([np.full(25, row["reference"]["uncertainty_maximum"]) for row in cal])
        radius = float(np.linalg.norm(np.sqrt(weights)*uncertainty))
        close(radius, diagnostics["uncertainty_radius"], "reference uncertainty radius")
        B = np.sqrt(weights)[:, None]*A
        T = np.sqrt(test_weights)[:, None]*At
        S = retained["weighted_label_to_coefficients"]
        P = retained["weighted_label_to_normalized_prediction"]
        # Algebraic replay of already retained linear maps, not a new fit.
        close(T@S, P, "normalized prediction map", atol=3e-16)
        normal_residual = (B.T@B + expected_ridge*np.eye(B.shape[1]))@S-B.T
        normal_relative = float(np.linalg.norm(normal_residual)/np.linalg.norm(B.T))
        assert normal_relative < 1e-9
        U, singular, _ = np.linalg.svd(S, full_matrices=False)
        coefficient_gain = float(singular[0])
        # S has <=40 rows: compute the spectral norm using its thin left factor.
        prediction_gain = float(np.linalg.svd((T@U)*singular, compute_uv=False)[0])
        close(coefficient_gain, diagnostics["weighted_label_to_coefficient_gain"], "coefficient gain")
        close(prediction_gain, diagnostics["weighted_label_to_withheld_rms_gain"], "prediction gain")
        exact_arithmetic_bound = 1/(2*math.sqrt(expected_ridge))
        assert coefficient_gain <= exact_arithmetic_bound*(1+1e-12)
        dd = retained["weighted_label_deltas"]
        dc = dd@S.T
        dp = dc@At.T
        dn = dp*np.sqrt(test_weights)
        for label, replay, saved in (("coefficient_delta", dc, retained["coefficient_deltas"]),
                                     ("prediction_delta", dp, retained["prediction_deltas"]),
                                     ("normalized_prediction_delta", dn, retained["normalized_prediction_deltas"]),
                                     ("prediction_map", T@S, P)):
            close(replay, saved, label, atol=3e-24 if label != "prediction_map" else 3e-16)
            maximum_replay_errors[label] = max(maximum_replay_errors[label], float(np.max(np.abs(replay-saved))))
        independent = np.concatenate([(arrays[row["query_id"]]["teacher"]-arrays[row["query_id"]]["refined"]).reshape(-1) for row in cal])
        close(dd[-1], np.sqrt(weights)*independent, "actual independent reference difference")
        error_rows = matching(test, method)
        error_rms = rms([row["rms"] for row in error_rows])
        test_uncertainty_rms = rms([row["reference"]["uncertainty_rms"] for row in test])
        scenarios = []
        for index, scenario in enumerate(diagnostics["scenarios"]):
            label_norm = float(np.linalg.norm(dd[index]))
            coefficient_change = float(np.linalg.norm(dc[index]))
            prediction_rms = float(np.linalg.norm(dn[index]))
            prediction_maximum = float(np.max(np.abs(dp[index])))
            for name, value in (("weighted_label_l2", label_norm), ("coefficient_l2", coefficient_change),
                                ("withheld_prediction_rms", prediction_rms), ("withheld_prediction_maximum", prediction_maximum)):
                close(value, scenario[name], "scenario "+name)
            if index == 0:
                close(prediction_rms, prediction_gain*radius, "leading direction attains measured gain")
            source_deltas = dp[index].reshape(len(test), 25)
            source_details = []
            for source, delta in zip(test, source_deltas):
                source_error = next(row for row in error_rows if row["query_id"] == source["query_id"])
                source_details.append(dict(query_id=source["query_id"], split=source["split"], family=source["family"],
                    prediction_rms=rms(delta), prediction_maximum=float(np.max(np.abs(delta))),
                    fraction_of_existing_error=rms(delta)/source_error["rms"],
                    multiple_of_reference_uncertainty=rms(delta)/source["reference"]["uncertainty_rms"]))
            scenarios.append(dict(name=scenario["name"], weighted_label_l2=label_norm,
                coefficient_l2=coefficient_change, coefficient_relative_l2=coefficient_change/coefficient_norm,
                prediction_rms=prediction_rms, prediction_maximum=prediction_maximum,
                fraction_of_existing_pooled_error=prediction_rms/error_rms,
                multiple_of_pooled_test_reference_uncertainty=prediction_rms/test_uncertainty_rms,
                family_prediction_rms={family: rms([x["prediction_rms"] for x in source_details if x["family"] == family])
                    for family in sorted({x["family"] for x in source_details})},
                worst_source_by_fraction_of_existing_error=max(source_details, key=lambda x: x["fraction_of_existing_error"]),
                diagnostic_prediction_rms=(rms([x["prediction_rms"] for x in source_details if x["split"] == "diagnostic"])
                                           if any(x["split"] == "diagnostic" for x in source_details) else None)))
        spectrum = np.asarray(diagnostics["singular_values"])
        shrinkage = spectrum**2/(spectrum**2+expected_ridge)
        sensitivity_rows.append(dict(name=path.stem, context_id=context, method=method,
            split="all_withheld" if context is None else test[0]["split"],
            start=None if context is None else test[0]["start"], stop=None if context is None else test[0]["stop"],
            calibration_source_count=len(cal), withheld_source_count=len(test), ridge=expected_ridge,
            coefficient_norm=coefficient_norm, original_design_rank=diagnostics["numerical_rank"],
            coefficient_count=diagnostics["coefficient_count"],
            numerical_ridge_map_normal_relative_residual=normal_relative,
            augmented_condition_number=diagnostics["augmented_condition_number"],
            coefficient_gain=coefficient_gain, coefficient_gain_upper_bound=exact_arithmetic_bound,
            normalized_withheld_prediction_gain=prediction_gain, uncertainty_radius=radius,
            worst_ball_coefficient_l2=coefficient_gain*radius,
            worst_ball_prediction_rms=prediction_gain*radius,
            pooled_withheld_error_rms=error_rms, pooled_withheld_reference_uncertainty_rms=test_uncertainty_rms,
            worst_ball_fraction_of_pooled_error=prediction_gain*radius/error_rms,
            ridge_data_singular_direction_shrinkage=shrinkage.tolist(),
            minimum_data_direction_shrinkage=float(shrinkage.min()), maximum_data_direction_shrinkage=float(shrinkage.max()),
            calibration_data_loss=fit_diagnostics["data_loss"], ridge_loss=fit_diagnostics["ridge_loss"],
            scenarios=scenarios))

    group_ranges = {}
    for name, predicate in (
        ("all_local_ridge_1e-10", lambda row: row["method"] == "local_ridge_1e-10"),
        ("all_local_ridge_1e-06", lambda row: row["method"] == "local_ridge_1e-06"),
        ("diagnostic_local_ridge_1e-10", lambda row: row["method"] == "local_ridge_1e-10" and row["split"] == "diagnostic"),
        ("diagnostic_local_ridge_1e-06", lambda row: row["method"] == "local_ridge_1e-06" and row["split"] == "diagnostic")):
        selected = [row for row in sensitivity_rows if predicate(row)]
        group_ranges[name] = {key: describe(row[key] for row in selected) for key in
            ("ridge", "coefficient_gain", "normalized_withheld_prediction_gain", "uncertainty_radius",
             "worst_ball_prediction_rms", "worst_ball_fraction_of_pooled_error",
             "minimum_data_direction_shrinkage", "maximum_data_direction_shrinkage")}
    projection = [fit["diagnostics"] for context in all_fits.values() for fit in context["fits"] if fit["name"] == "projection"]
    references = [row["reference"] for row in queries]
    stages_seconds = [row["elapsed_seconds"] for row in references]
    work_by_method = {}
    for method in sorted({row["method"] for row in work}):
        rows = [row["work"] for row in work if row["method"] == method]
        work_by_method[method] = dict(rows=len(rows), example=rows[0], all_records_identical=all(row == rows[0] for row in rows))
    result = dict(
        schema="tdn.native_r7.cost_sensitivity_review.v1", status="DESCRIPTIVE_REPLAY_COMPLETE",
        source_fingerprint=summary["source_fingerprint"], protocol_sha256=summary["protocol_sha256"],
        scope="Retained native R7 CPU development evidence; no new teachers/actions/fits and no scientific promotion.",
        timing=timing_review,
        sensitivity=dict(map_count=len(sensitivity_rows), scenarios_replayed=sum(len(row["scenarios"]) for row in sensitivity_rows),
            replay_maximum_absolute_differences=dict(maximum_replay_errors),
            uncertainty_scope="Estimated per-source maximum uncertainties form an aggregate weighted calibration-label L2 ball; not certified teacher error, not per-source bounds.",
            selection_scope="Selected ridge/model held fixed. Sensitivity excludes model-selection discontinuity and changing-design/physics errors.",
            global_row_order="Actual query_finished journal order; global withheld rows begin with validation, then training, then diagnostic sources.",
            local_objective_scope="Induced single-context ridge; independent local fits relax cross-context coupling and do not share the global joint objective.",
            projection_coefficient_gain_range=describe(row["weighted_label_to_coefficient_gain"] for row in projection),
            projection_sensitivity_scope="Only recorded coefficient gains; no retained unregularized prediction maps/scenarios. Do not transfer positive-ridge robustness to projection.",
            ranges=group_ranges, maps=sensitivity_rows),
        costs=dict(study_elapsed_seconds=summary["elapsed_seconds"],
            reference_acquisition_seconds=sum(stages_seconds), per_reference_query_seconds=describe(stages_seconds),
            global_fit_and_validation_selection_seconds=selection["fitting_seconds"],
            all_context_local_fit_seconds=sum(row["offline_fitting_seconds"] for row in all_fits.values()),
            timed_bundles_seconds=sum(row["seconds"] for row in timing),
            offline_scope="Reference acquisition, design construction and local oracle fitting are not deployable speedup; reference and fit sums are subsets of study elapsed, not additive independent wall costs.",
            counter_scope="Explicit action counters omit internal anchor/autograd FFT operations; they cannot establish total operation savings. End-to-end timing includes execution. Native worker RSS is not per-method peak memory.",
            action_work_records=work_by_method),
        gates=dict(implementation_receipt_review="Separate integrity review required", practical_utility="NOT_ESTABLISHED",
            capacity="Source-sampled development only; full-operator and fresh-parent confirmation remain pending",
            encoder_promotion=False, endpoint_claim=False, gpu_claim=False, novelty_claim=False),
        input_sha256=dict(sorted(hashes.items())))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(main(arguments.study), indent=2, sort_keys=True, allow_nan=False))
