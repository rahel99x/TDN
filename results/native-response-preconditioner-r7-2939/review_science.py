#!/usr/bin/env python3
"""Review retained R7 arrays; no teacher solve, model execution, or new fit.

Only Python's standard library and NumPy are used. The uploaded archive is
data, never imported/executed. Numeric NPZ files are opened without pickle.
All pooled quantities are descriptive paired descendants of four inspected
parents, not estimates from 192 independent parents. Timing and sensitivity
have separate review receipts.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def span(values):
    values = np.asarray(values, dtype=np.float64)
    return {"minimum": float(values.min()), "median": float(np.median(values)),
            "maximum": float(values.max())}


def spectrum(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    cutoff = float(np.finfo(np.float64).eps * max(matrix.shape) * singular[0])
    return {"shape": list(matrix.shape), "singular_values": singular.tolist(),
            "numerical_rank": int(np.sum(singular > cutoff)), "rank_threshold": cutoff,
            "rank_convention": "eps64 * max(matrix.shape) * largest singular value"}


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for identity, selected in sorted(groups.items()):
        mse = np.mean([r["rms"] ** 2 for r in selected])
        out = dict(zip(keys, identity))
        out.update(queries=len(selected), parent_count=len({r["parent_id"] for r in selected}),
                   pooled_rms=float(np.sqrt(mse)), worst_nodal_error=max(r["maximum"] for r in selected),
                   mean_signed_error=float(np.mean([r["mean_error"] for r in selected])),
                   pooled_centered_rms=float(np.sqrt(np.mean([r["centered_rms"] ** 2 for r in selected]))),
                   mean_error_fraction_of_mse=float(np.mean([r["mean_error"] ** 2 for r in selected]) / mse),
                   error_over_reference_uncertainty=span([r["rms"] / r["reference_uncertainty_rms"] for r in selected]),
                   errors_above_ten_times_estimated_reference_uncertainty=sum(
                       r["rms"] > 10 * r["reference_uncertainty_rms"] for r in selected))
        for correction in ("U_minus_P", "U_minus_G"):
            amplitudes = [r[correction + "_rms"] for r in selected]
            out[correction + "_pooled_signal_rms"] = rms(np.asarray(amplitudes))
            out[correction + "_pooled_relative_error"] = float(np.sqrt(mse) / rms(np.asarray(amplitudes)))
        result.append(out)
    return result


def review(stage):
    hashes = {}

    def read(name):
        path = stage / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def arrays(name):
        path = stage / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            result = {k: archive[k] for k in archive.files}
        assert all(a.dtype.kind in "biufc" and np.isfinite(a).all() for a in result.values())
        return result

    protocol, summary = read("protocol.json"), read("summary.json")
    index, metrics, selection = read("query_index.json"), read("metrics.json"), read("selection.json")
    references = read("references.json")
    assert len(index) == 192 and len({r["query_id"] for r in index}) == 192
    assert len(metrics) == 2112 and len({(r["query_id"], r["method"]) for r in metrics}) == 2112
    assert all(r["status"] == "ACCEPTED" for r in index)
    assert all(r["status"] == "MEASURED" for r in metrics)
    assert references == [{"query_id": q["query_id"], "reference": q["reference"]} for q in index]
    queries = {q["query_id"]: q for q in index}
    retained, predictions, local = {}, {}, {}
    for q in index:
        retained[q["query_id"]] = arrays(q["array_path"])
        predictions[q["query_id"]] = arrays("queries/" + q["query_id"] + "-predictions.npz")
        if q["context_id"] not in local:
            local[q["context_id"]] = read("fits/" + q["context_id"] + ".json")
    weights = np.asarray(selection["selected"]["coefficients"])
    discrepancies = defaultdict(float)
    numeric_fields = ("rms", "maximum", "mean_error", "centered_rms", "K_minus_P_rms",
                      "U_minus_P_rms", "U_minus_G_rms", "U_minus_P_relative_error", "U_minus_G_relative_error")
    resolution = Counter()
    for metric in metrics:
        qid, method = metric["query_id"], metric["method"]
        q, a, pred = queries[qid], retained[qid], predictions[qid][method]
        error = pred - a["teacher"]
        computed = dict(rms=rms(error), maximum=float(np.max(np.abs(error))),
                        mean_error=float(np.mean(error)), centered_rms=rms(error - error.mean()),
                        K_minus_P_rms=rms(pred - a["base"]),
                        U_minus_P_rms=rms(a["teacher"] - a["base"]),
                        U_minus_G_rms=rms(a["teacher"] - a["matched_mean"]))
        for name in ("U_minus_P", "U_minus_G"):
            computed[name + "_relative_error"] = computed["rms"] / computed[name + "_rms"]
            uncertainty = q["reference"]["uncertainty_rms"]
            if name == "U_minus_G":
                uncertainty += q["reference"]["matched_mean"]["uncertainty_rms"]
            status = "RESOLVED" if computed[name + "_rms"] > protocol["reference_resolution_multiple"] * uncertainty else "UNRESOLVED_NEAR_NULL"
            assert metric[name + "_status"] == status
            resolution[name + ":" + status] += 1
        for field in numeric_fields:
            delta = abs(computed[field] - metric[field])
            discrepancies["metric_" + field] = max(discrepancies["metric_" + field], delta)
            assert np.isclose(computed[field], metric[field], rtol=2e-12, atol=2e-26), (qid, method, field, delta)
    features = {}
    for qid, a in retained.items():
        q = queries[qid]
        features.setdefault(q["context_id"], a["features"])
        assert np.array_equal(features[q["context_id"]], a["features"])
        discrepancies["source_unit_rms"] = max(discrepancies["source_unit_rms"], abs(rms(a["source"]) - 1))
        reconstructed = (a["branches"][:, None, ...] * a["features"].reshape(1, 8, 1, 1, 1, 1)).reshape(a["design"].shape)
        discrepancies["design_factorization_maximum"] = max(discrepancies["design_factorization_maximum"], float(np.max(np.abs(reconstructed - a["design"]))))
        actual = predictions[qid]
        discrepancies["ordered2_equals_retained_base"] = max(discrepancies["ordered2_equals_retained_base"], float(np.max(np.abs(actual["ordered_2"] - a["base"]))))
        reconstructed = a["base"] + np.einsum("j,j...->...", weights, a["design"])
        discrepancies["global_design_prediction_maximum"] = max(discrepancies["global_design_prediction_maximum"], float(np.max(np.abs(reconstructed - actual["fixed_feature_fitted"]))))
        coeff = weights.reshape(5, 8) @ a["features"]
        reconstructed = a["base"] + np.einsum("j,j...->...", coeff, a["branches"])
        discrepancies["global_five_branch_prediction_maximum"] = max(discrepancies["global_five_branch_prediction_maximum"], float(np.max(np.abs(reconstructed - actual["fixed_feature_fitted"]))))
        for fit in local[q["context_id"]]["fits"]:
            reconstructed = a["base"] + np.einsum("j,j...->...", fit["coefficients"], a["branches"])
            discrepancies["local_prediction_maximum"] = max(discrepancies["local_prediction_maximum"], float(np.max(np.abs(reconstructed - actual[fit["name"]]))))
        ref = q["reference"]
        discrepancies["independent_refined_rms"] = max(discrepancies["independent_refined_rms"], abs(rms(a["teacher"] - a["refined"]) - ref["cross_difference"]["rms"]))
        discrepancies["matched_mean_independent_refined_rms"] = max(discrepancies["matched_mean_independent_refined_rms"], abs(rms(a["matched_mean"] - a["matched_mean_refined"]) - ref["matched_mean"]["cross_difference"]["rms"]))
    assert discrepancies["source_unit_rms"] < 1e-14
    assert all(value < 2e-14 for key, value in discrepancies.items() if "prediction_maximum" in key)
    assert discrepancies["design_factorization_maximum"] < 1e-16

    def matrix(ids, key="design"):
        return (np.concatenate([retained[q][key].reshape(retained[q][key].shape[0], -1).T for q in ids]),
                np.concatenate([(retained[q]["teacher"] - retained[q]["base"]).reshape(-1) for q in ids]))

    train_ids = [q["query_id"] for q in index if q["split"] == "train" and q["role"] == "calibration"]
    val_ids = [q["query_id"] for q in index if q["split"] == "validation" and q["role"] == "withheld"]
    assert train_ids == selection["train_query_ids"] and val_ids == selection["validation_query_ids"]
    A, b = matrix(train_ids)
    Av, bv = matrix(val_ids)
    # The two training parents have exactly equal row counts: uniform 1/1200
    # weights exactly implement this frozen parent-balanced physical loss.
    parent_counts = Counter(queries[q]["parent_id"] for q in train_ids)
    assert len(parent_counts) == 2 and len(set(parent_counts.values())) == 1
    candidates = []
    for candidate in selection["candidates"]:
        c = np.asarray(candidate["coefficients"])
        lam = candidate["ridge"] or 0.0
        residual = A @ c - b
        validation_mse = float(np.mean((Av @ c - bv) ** 2))
        data_loss = float(np.mean(residual ** 2))
        penalty = float(lam * np.dot(c, c))
        gradient = 2 * (A.T @ residual / b.size + lam * c)
        assert np.isclose(validation_mse, candidate["validation_loss"], rtol=1e-12)
        item = dict(name=candidate["name"], eligible=candidate["eligible"], ridge=candidate["ridge"],
                    validation_mse=validation_mse, validation_rms=float(np.sqrt(validation_mse)),
                    training_data_loss=data_loss, ridge_loss=penalty, training_objective=data_loss + penalty,
                    coefficient_l2=float(np.linalg.norm(c)), gradient_l2=float(np.linalg.norm(gradient)))
        item["ridge_fraction_of_training_objective"] = penalty / (data_loss + penalty)
        if lam:
            d = candidate["diagnostics"]
            item.update(ordinary_fp64_gap_diagnostic=float(np.dot(gradient, gradient) / (4 * lam)),
                        augmented_condition_number=d["augmented_condition_number"])
            assert np.isclose(data_loss + penalty, d["objective"], rtol=1e-12)
        candidates.append(item)
    assert min(candidates, key=lambda c: c["validation_mse"])["name"] == selection["selected"]["name"]

    local_checks, decomposition = [], []
    for cid, fit_payload in sorted(local.items()):
        calibration = [q["query_id"] for q in index if q["context_id"] == cid and q["role"] == "calibration"]
        B, d = matrix(calibration, "branches")
        singular = np.linalg.svd(B / np.sqrt(d.size), compute_uv=False)
        eta = features[cid]
        for fit in fit_payload["fits"]:
            c = np.asarray(fit["coefficients"])
            error = B @ c - d
            lam = 0.0 if fit["ridge"] is None else fit["ridge"] / (protocol["local_context_weight"] * np.dot(eta, eta))
            gradient = 2 * (B.T @ error / d.size + lam * c)
            loss = float(np.mean(error ** 2))
            objective = loss + lam * float(np.dot(c, c))
            diagnostic = fit["diagnostics"]
            assert np.isclose(objective, diagnostic["objective"], rtol=2e-11)
            assert np.allclose(singular, diagnostic["singular_values"], rtol=2e-12, atol=0)
            local_checks.append(dict(context_id=cid, method=fit["name"], ridge_induced=lam,
                rank=int(np.sum(singular > diagnostic["rank_threshold"])), condition_number=float(singular[0] / singular[-1]),
                calibration_rms=float(np.sqrt(loss)), objective=objective, gradient_l2=float(np.linalg.norm(gradient)),
                coefficients=c.tolist(), coefficient_l2=float(np.linalg.norm(c)),
                effective_global_coefficients=(weights.reshape(5, 8) @ eta).tolist(),
                ridge_svd_retention_factors=(singular ** 2 / (singular ** 2 + lam)).tolist()))
        for role in ("calibration", "withheld"):
            qids = [q["query_id"] for q in index if q["context_id"] == cid and q["role"] == role]
            projection_error = np.concatenate([(predictions[q]["projection"] - retained[q]["teacher"]).ravel() for q in qids])
            map_difference = np.concatenate([(predictions[q]["fixed_feature_fitted"] - predictions[q]["projection"]).ravel() for q in qids])
            total = projection_error + map_difference
            decomposition.append(dict(context_id=cid, role=role,
                global_mse=float(np.mean(total ** 2)), projection_mse=float(np.mean(projection_error ** 2)),
                coefficient_map_difference_mse=float(np.mean(map_difference ** 2)),
                signed_cross_term=float(2 * np.mean(projection_error * map_difference)),
                identity_residual=float(np.mean(total ** 2) - np.mean(projection_error ** 2) - np.mean(map_difference ** 2) - 2 * np.mean(projection_error * map_difference))))

    E = np.asarray([features[c] for c in sorted(features) if c.startswith("train-")])
    svd = spectrum(E)
    _, _, vt = np.linalg.svd(E, full_matrices=False)
    row_basis = vt[:svd["numerical_rank"]]
    feature_rows = []
    for cid, eta in sorted(features.items()):
        distance = float(np.linalg.norm(eta - eta @ row_basis.T @ row_basis))
        feature_rows.append(dict(context_id=cid, feature_l2=float(np.linalg.norm(eta)),
                                 distance_outside_training_feature_span=distance,
                                 relative_distance=distance / float(np.linalg.norm(eta))))

    diagnostic = [r for r in metrics if r["split"] == "diagnostic" and r["role"] == "withheld"]
    by_query = {(r["query_id"], r["method"]): r for r in metrics}
    pairs = []
    for candidate in ("fixed_feature_fitted", "projection", "local_ridge_1e-10", "local_ridge_1e-06"):
        for baseline in ("ordered_2", "ordered_4", "ordered_8", "lawson_2", "lawson_4", "lawson_8"):
            a = [r for r in diagnostic if r["method"] == candidate]
            base = [by_query[(r["query_id"], baseline)] for r in a]
            pairs.append(dict(candidate=candidate, baseline=baseline, queries=len(a),
                rms_ratio_pooled=float(np.sqrt(sum(r["rms"] ** 2 for r in a) / sum(r["rms"] ** 2 for r in base))),
                query_ratio=span([x["rms"] / y["rms"] for x, y in zip(a, base)]),
                candidate_lower_rms_queries=sum(x["rms"] < y["rms"] for x, y in zip(a, base)),
                candidate_lower_maximum_queries=sum(x["maximum"] < y["maximum"] for x, y in zip(a, base)),
                significance_claim=False))
    longest = []
    for split in ("train", "validation", "diagnostic"):
        for role in ("calibration", "withheld"):
            for method in sorted({r["method"] for r in diagnostic}):
                selected = [r for r in metrics if r["split"] == split and r["role"] == role and r["method"] == method]
                numerator = sum(r["rms"] ** 2 for r in selected if r["start"] == 0 and r["stop"] == .08)
                longest.append(dict(split=split, role=role, method=method,
                    longest_context_fraction_of_mse=numerator / sum(r["rms"] ** 2 for r in selected)))
    ref = [q["reference"] for q in index]
    labeled_metrics = [dict(r, source_time_category="zero_start" if r["start"] == 0 else "nonzero_start") for r in metrics]
    return dict(schema="tdn.native_response_preconditioner_r7_science_review.v1",
        study_status=summary["status"], protocol_sha256=summary["protocol_sha256"],
        source_fingerprint=summary["source_fingerprint"], input_sha256=dict(sorted(hashes.items())),
        review_scope="Read-only descriptive replay of retained arrays; no new reference acquisition, candidate fitting, tuning or model execution",
        aggregation="sqrt(mean(per-query physical RMS squared)); equal cells and equal queries within groups; descendants remain paired, four inspected parents only",
        counts=dict(queries=len(index), metrics=len(metrics), contexts=len(local), local_fits=len(local_checks),
                    global_candidates=len(candidates), independent_parents=len({q["parent_id"] for q in index}),
                    metric_status=dict(Counter(r["status"] for r in metrics)), correction_status=dict(resolution)),
        array_replay_maximum_discrepancies=dict(sorted(discrepancies.items())),
        references=dict(uncertainty_rms=span([r["uncertainty_rms"] for r in ref]),
            uncertainty_maximum=span([r["uncertainty_maximum"] for r in ref]),
            matched_mean_uncertainty_rms=span([r["matched_mean"]["uncertainty_rms"] for r in ref]),
            reference_acceptance_is_certificate=False),
        selection=dict(name=selection["selected"]["name"], candidates=candidates,
            training_queries=len(train_ids), validation_queries=len(val_ids),
            scope="Action-only equal-parent ridge; validation withheld sources select candidate; diagnostic truth and training withheld truth do not fit/select global W",
            zero_over_selected_validation_mse=candidates[0]["validation_mse"] / next(c["validation_mse"] for c in candidates if c["name"] == selection["selected"]["name"])),
        identifiability=dict(training_feature_matrix=svd,
            training_static_feature_matrix=spectrum(E[:, [0, 1, 2, 3, 6]]),
            training_dynamic_feature_matrix=spectrum(E[:, [4, 5, 7]]),
            all_four_parent_features=spectrum(np.asarray([features[c] for c in sorted(features)])),
            original_weighted_global_design=spectrum(A / np.sqrt(len(b))),
            feature_span_by_context=feature_rows, global_weight_dimensions=40,
            mathematical_rank_upper_bound_for_this_bank=25,
            explanation="Five features are constant within each of two training parents (rank at most 2); three time-dependent features add at most 3. Feature rank is at most 5 and each of five branch design blocks has rank at most 5. Actual ranks are 5 and 25. More source directions alone cannot identify the 15 missing global weight combinations in this bank. Positive ridge uniquely minimizes the conditional regularized objective but does not identify those combinations from data.",
            branch_rank_scope="Every local calibration design has rank 5; this is coefficient identifiability for sampled branches, not physical operator rank or a full operator test"),
        local_fit_replay=local_checks,
        global_projection_error_decomposition=decomposition,
        by_split_role_method=grouped(metrics, ("split", "role", "method")),
        by_parent_role_method=grouped(metrics, ("parent_id", "role", "method")),
        by_split_role_family_method=grouped(metrics, ("split", "role", "family", "method")),
        by_context_role_method=grouped(metrics, ("context_id", "role", "method")),
        by_source_time_category=grouped(labeled_metrics, ("split", "role", "source_time_category", "method")),
        diagnostic_withheld_pairwise_accuracy=pairs, longest_interval_loss_contribution=longest,
        interpretation=[
            "All U-P and U-G action signals are resolved against the protocol's empirical reference floor; endpoint benefit is a separate unresolved native2936 question.",
            "Global fixed-feature action fitting improves the ordered2 baseline but loses accuracy to ordered8 on every diagnostic withheld query; Lawson2 is much more accurate on this bank.",
            "Projection is a locally label-informed sampled representation diagnostic. Its calibration residual is nonzero and resolved; its withheld result is neither a full-operator lower bound nor a deployable method.",
            "Projection errors differ sharply by source family and context. At diagnostic (0,.08) the projection is worse than ordered8 and dominates aggregate error; at the five shorter diagnostic contexts pooled projection error is smaller than ordered8. This supports a resolved sampled span residual on the longest interval alongside a gap between global and local coefficients elsewhere. Source draws and absolute interval change across contexts: this is not a paired convergence-order experiment, and the gap does not identify one isolated universal bottleneck.",
            "Induced local ridge shrinks short-interval coefficients strongly because branch amplitudes scale cubically. Local ridge and coupled global ridge minimize different objectives; local ridge being worse does not show that projection or coupling is faulty.",
            "Global versus unregularized local projection also changes penalty and access to context-local labels. Its gap cannot be attributed solely to feature-map expressiveness; regularization bias and parent/source coverage are confounded. No new ridge value was fitted or selected in this review.",
            "Equal physical row weights do not give equal relative-error importance to every lag. The longest (0,.08) context supplies most squared error in the selected training, validation and diagnostic losses. Reweighting would define a new objective and needs a new frozen protocol.",
            "The comparison global minus projection requires its signed cross term on withheld sources; squared errors cannot be subtracted as independent causal contributions.",
            "The original global design remains rank25/40 for an exact feature-coverage reason. Additional independent parent feature coverage, reparameterization and regularization choices need a new frozen development protocol; a larger neural encoder is not justified by coefficient nonidentifiability alone.",
            "No timing, Pareto, endpoint, rollout, GPU, novelty or fresh-confirmation claim follows from these descriptive accuracy aggregates."
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Extracted outputs/study directory")
    parser.add_argument("--output", type=Path, help="Review JSON destination (otherwise stdout)")
    args = parser.parse_args()
    result = review(args.study)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
