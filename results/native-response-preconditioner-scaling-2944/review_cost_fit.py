#!/usr/bin/env python3
"""Replay P1 saved fits, cost accounting and paired timings; acquire/refit nothing.

Usage: .venv/bin/python review_cost_fit.py --study PATH --output PATH
Only NumPy and the standard library are used. Every objective is evaluated
directly from retained design/target arrays and saved coefficients. Numerical
optimality remains an ordinary-FP64 diagnostic, not an interval certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(count=int(a.size), minimum=float(a.min()), median=float(np.median(a)),
                maximum=float(a.max()), mean=float(a.mean()))


def review(study):
    failures, checks, input_hashes = [], Counter(), {}

    def check(ok, label):
        checks["checked"] += 1
        if not ok:
            failures.append(label)

    def close(actual, expected, label, *, rtol=2e-10, atol=1e-28):
        check(bool(np.allclose(actual, expected, rtol=rtol, atol=atol)), label)

    def read(name):
        path = study / name
        raw = path.read_bytes()
        input_hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    queries = read("query_index.json")
    byid = {q["query_id"]: q for q in queries}
    check(len(byid) == len(queries) == 2688, "unique complete query bank")
    selection = read("selection.json")
    cells = selection["cells"]
    check(len(cells) == 9 and selection["all_selections_sealed"], "nine selected cells")
    protocol = read("protocol.json")
    arrays = {}
    for q in queries:
        if q["split"] == "validation" or (q["split"] == "train" and q["role"] == "calibration"):
            raw = (study / q["array_path"]).read_bytes()
            check(hashlib.sha256(raw).hexdigest() == q["array_sha256"], "array hash " + q["query_id"])
            with np.load(study / q["array_path"], allow_pickle=False) as a:
                arrays[q["query_id"]] = (a["design"].reshape(40, -1).T,
                    (a["teacher"] - a["base"]).ravel())
    validation_ids = [q["query_id"] for q in queries if q["split"] == "validation"]
    train_parents = sorted({q["parent_id"] for q in queries if q["split"] == "train"})
    validation_parents = sorted({byid[q]["parent_id"] for q in validation_ids})

    def stack(ids):
        counts = Counter(byid[q]["parent_id"] for q in ids)
        A = np.concatenate([arrays[q][0] for q in ids])
        b = np.concatenate([arrays[q][1] for q in ids])
        # Unit physical mass per query, then equal total mass per parent.
        omega = np.concatenate([np.full(arrays[q][1].size,
            1.0 / (len(counts) * counts[byid[q]["parent_id"]] * arrays[q][1].size)) for q in ids])
        return A, b, omega

    VA, vb, vw = stack(validation_ids)
    fits = []
    for cell in cells:
        mid = cell["model_id"]
        check(read("fits/" + mid + ".json") == cell, mid + " separate fit record agrees")
        n, k = cell["training_parent_count"], cell["calibration_source_count"]
        expected = [q["query_id"] for q in queries if q["split"] == "train" and
                    q["role"] == "calibration" and q["parent_id"] in train_parents[:n]
                    and q["source_index"] < k]
        check(cell["train_query_ids"] == expected and len(expected) == n*k*6,
              mid + " exact nested training bank")
        check(cell["validation_query_ids"] == validation_ids and len(validation_parents) == 8,
              mid + " fixed validation-only bank")
        check(cell["training_parent_ids"] == train_parents[:n], mid + " parent prefix")
        check(not cell["diagnostic_truth_accessed"] and not cell["training_withheld_truth_accessed"],
              mid + " declared access restriction")
        sealed = {a: b for a, b in cell.items() if a not in ("selection_seal", "fitting_seconds")}
        sealed["candidates"] = [{a: b for a, b in c.items() if a != "fitting_seconds"}
                                 for c in cell["candidates"]]
        sealed["selected"] = {a: b for a, b in cell["selected"].items() if a != "fitting_seconds"}
        check(digest(sealed) == cell["selection_seal"], mid + " selection seal")
        A, b, omega = stack(expected)
        Aw = A * np.sqrt(omega[:, None])
        bw = b * np.sqrt(omega)
        singular = np.linalg.svd(Aw, compute_uv=False)
        rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(A.shape) * singular[0]))
        zero_loss = float(bw @ bw)
        candidate_reviews = []
        check([(c["name"], c["ridge"]) for c in cell["candidates"]] ==
              [("zero_readout", None), ("ridge_1e-10", 1e-10), ("ridge_1e-06", 1e-6)],
              mid + " original three-candidate menu")
        for c in cell["candidates"]:
            w = np.asarray(c["coefficients"], dtype=float)
            validation_loss = float(np.sum(vw * (VA @ w - vb)**2))
            close(validation_loss, c["validation_loss"], mid + " " + c["name"] + " validation MSE")
            lam = c["ridge"] or 0.0
            residual = Aw @ w - bw
            data_loss = float(residual @ residual)
            penalty = float(lam * (w @ w))
            objective = data_loss + penalty
            cr = dict(name=c["name"], ridge=c["ridge"], validation_mse=validation_loss,
                      training_mse=data_loss, penalty=penalty, objective=objective,
                      coefficient_l2=float(np.linalg.norm(w)))
            if lam:
                d = c["diagnostics"]
                gradient = 2 * (Aw.T @ residual + lam*w)
                gap = float((gradient @ gradient) / (4*lam))
                tolerance = 1e-8 * zero_loss + 1e-28
                for key, value in (("data_loss", data_loss), ("ridge_loss", penalty),
                                   ("objective", objective), ("zero_objective", zero_loss),
                                   ("objective_improvement", zero_loss-objective),
                                   ("optimality_tolerance", tolerance)):
                    close(value, d[key], mid + " " + c["name"] + " " + key)
                close(d["gradient_norm"]**2 / (4*lam), d["optimality_gap_upper_bound"],
                      mid + " saved gap arithmetic", atol=1e-100)
                check(gap <= tolerance and objective <= zero_loss + tolerance,
                      mid + " independently recomputed numerical optimality")
                check(d["optimality_verified"] and d["baseline_nonregression_verified"] and c["eligible"],
                      mid + " eligibility flags")
                check(d["rows"] == len(b) and d["parent_count"] == n and
                      d["parent_row_counts"] == {p: k*6*25 for p in train_parents[:n]},
                      mid + " equal-parent row bookkeeping")
                close(d["parent_total_weight"], 1/n, mid + " parent weight")
                close(singular, d["original_design_singular_values"], mid + " data singular values",
                      rtol=2e-8, atol=1e-19)
                check(rank == d["original_design_numerical_rank"], mid + " numerical data rank")
                close(singular[0]/singular[-1], d["original_design_condition_number"],
                      mid + " original design condition", rtol=2e-8)
                norms = np.linalg.norm(Aw, axis=0)
                scales = np.hypot(norms, math.sqrt(lam))
                augmented = np.vstack([Aw / scales, np.diag(math.sqrt(lam) / scales)])
                aug_singular = np.linalg.svd(augmented, compute_uv=False)
                close(norms, d["training_column_norms"], mid + " training-only column norms")
                close(scales, d["column_scales"], mid + " original augmented scales")
                close(aug_singular, d["augmented_singular_values"], mid + " augmented singular values")
                close(aug_singular[0]/aug_singular[-1], d["augmented_condition_number"],
                      mid + " augmented condition")
                check(d["original_penalty_preserved"] and not d["singular_value_truncation"] and
                      not d["certificate"], mid + " unchanged objective and certificate scope")
                shrinkage = singular**2 / (singular**2 + lam)
                cr.update(recomputed_gradient_norm=float(np.linalg.norm(gradient)),
                    recomputed_optimality_gap=gap, saved_optimality_gap=d["optimality_gap_upper_bound"],
                    optimality_tolerance=tolerance, gap_to_training_data_loss=gap/data_loss,
                    penalty_fraction_of_objective=penalty/objective,
                    augmented_condition=float(aug_singular[0]/aug_singular[-1]),
                    linear_smoother_effective_degrees_of_freedom=float(shrinkage.sum()),
                    data_singular_directions_above_half_shrinkage=int(np.count_nonzero(shrinkage > .5)),
                    minimum_data_shrinkage=float(shrinkage[-1]), maximum_data_shrinkage=float(shrinkage[0]))
            candidate_reviews.append(cr)
        expected_selected = min((c for c in cell["candidates"] if c["eligible"]),
                                key=lambda c: c["validation_loss"])
        check(expected_selected == cell["selected"], mid + " minimum validation eligible candidate")
        fits.append(dict(model_id=mid, selected=cell["selected"]["name"], train_queries=len(expected),
                         validation_queries=len(validation_ids), rank=rank,
                         design_condition=float(singular[0]/singular[-1]), candidates=candidate_reviews))

    costs = read("costs.json")
    work = read("work.json")
    timing = read("timing.json")
    records = read("timing_records.json")
    metrics = read("metrics.json")
    precision = read("work_precision.json")
    curves = read("learning_curves.json")

    def demand(ids):
        rows = [byid[q] for q in ids]
        return dict(planned=len(rows), attempted=sum("reference_seconds" in q for q in rows),
                    accepted=sum(q["status"] == "ACCEPTED" for q in rows),
                    reference_seconds=sum(q.get("reference_seconds", 0.0) for q in rows),
                    design_seconds=sum(q.get("design_seconds", 0.0) for q in rows))

    def compare_dict(actual, expected, label):
        check(actual.keys() == expected.keys(), label + " keys")
        for key in expected:
            close(actual[key], expected[key], label + " " + key)

    compare_dict(demand(byid), costs["global_acquisition"], "globally unique acquisitions")
    cost_byid = {c["model_id"]: c for c in costs["cells"]}
    fit_byid = {c["model_id"]: c for c in cells}
    for cell in cells:
        mid, c = cell["model_id"], cost_byid[cell["model_id"]]
        compare_dict(demand(cell["train_query_ids"]), c["training"], mid + " training demand")
        compare_dict(demand(validation_ids), c["validation"], mid + " shared validation demand")
        close(c["training_reference_seconds"], c["training"]["reference_seconds"], mid + " demand alias")
        close(c["validation_reference_seconds"], c["validation"]["reference_seconds"], mid + " validation alias")
        close(c["fitting_seconds"], cell["fitting_seconds"], mid + " fit duration")
    # The outer fitting timer also covers input conversion, validation and
    # callback persistence; it is not defined as the sum of inner cell timers.
    cell_fit_sum = sum(c["fitting_seconds"] for c in cells)
    check(costs["fitting_seconds"] >= cell_fit_sum >= 0, "fitting outer timer covers cell timers")
    close(costs["accuracy_evaluation_seconds"], sum(w.get("accuracy_evaluation_seconds", 0.) for w in work),
          "accuracy evaluation total")
    close(costs["timing_seconds"], sum(r["seconds"] for r in records), "timing observation total")
    for c in curves:
        if c["method"] in cost_byid:
            for key in ("training_reference_seconds", "validation_reference_seconds", "fitting_seconds"):
                close(c[key], cost_byid[c["method"]][key], "curve cost " + c["method"] + " " + key)

    methods = {"fit-p032-s08", "lawson_2", "lawson_4", "lawson_8", "ordered_2", "ordered_8"}
    recordmap = {}
    timing_byid = {t["context_id"]: t for t in timing}
    expected_contexts = {q["context_id"] for q in queries if q["split"] == "diagnostic" and
                         q["parent_id"].split("-")[1] in {"000", "001", "002", "003"}}
    check(len(timing) == len(timing_byid) == 24 and set(timing_byid) == expected_contexts,
          "exact predeclared 24 timing contexts")
    for r in records:
        key = (r["context_id"], r["method"], r["phase"], r["round"])
        check(key not in recordmap, "unique timing record " + str(key))
        recordmap[key] = r
        check(r["status"] == "PASS" and math.isfinite(r["seconds"]) and r["seconds"] > 0,
              "valid timing " + str(key))
    expectedkeys = {(cid, m, phase, rnd) for cid in expected_contexts for m in methods
                    for phase, rounds in (("cold", 1), ("warmup", 1), ("paired", 5)) for rnd in range(rounds)}
    check(set(recordmap) == expectedkeys and len(records) == 1008, "all 1008 timing slots")
    precision_bykey = {(p["context_id"], p["method"]): p for p in precision}
    check(len(precision_bykey) == len(precision) == 144, "144 work/precision rows")
    metrics_bykey = defaultdict(list)
    for m in metrics:
        metrics_bykey[(m["context_id"], m["method"])].append(m)
    for cid, t in timing_byid.items():
        for r in t["records"]:
            actual = recordmap[(cid, r["method"], r["phase"], r["round"])]
            check(all(actual[k] == v for k, v in r.items()), "timing nested/flat record identity")
        for phase, rounds in (("cold", 1), ("warmup", 1), ("paired", 5)):
            for rnd in range(rounds):
                check(sorted(recordmap[(cid, m, phase, rnd)]["position"] for m in methods) == list(range(6)),
                      "complete randomized method order")
        for method in methods:
            observation = [recordmap[(cid, method, "paired", rnd)]["seconds"] for rnd in range(5)]
            summary = dict(expected_rounds=5, passed_rounds=5, failed_rounds=0,
                           minimum_seconds=min(observation), maximum_seconds=max(observation),
                           median_seconds=float(np.median(observation)))
            compare_dict(summary, t["summary"][method], "paired timing summary")
            p = precision_bykey[(cid, method)]
            mm = metrics_bykey[(cid, method)]
            check(len(mm) == 4 and all(m["status"] == "MEASURED" for m in mm) and p["all_sources_resolved"],
                  "same-context accuracy denominator")
            close(p["pooled_source_rms"], np.sqrt(np.mean([m["mse"] for m in mm])), "work precision RMS")
            close(p["maximum"], max(m["maximum"] for m in mm), "work precision maximum")
            for key in summary:
                close(p[key], summary[key], "work precision timing " + key)
            check(not p["speedup_claim"], "no unmatched speedup promotion")
            hashes = {recordmap[(cid, method, phase, rnd)]["output_sha256"]
                      for phase, rr in (("cold", 1), ("warmup", 1), ("paired", 5)) for rnd in range(rr)}
            check(len(hashes) == 1, "repeat timing output digest stable")

    comparisons = []
    for competitor in sorted(methods - {"fit-p032-s08"}):
        context_rows, paired_ratios = [], []
        for cid in sorted(expected_contexts):
            ratios = [recordmap[(cid, competitor, "paired", rnd)]["seconds"] /
                      recordmap[(cid, "fit-p032-s08", "paired", rnd)]["seconds"] for rnd in range(5)]
            paired_ratios.extend(ratios)
            our = precision_bykey[(cid, "fit-p032-s08")]
            other = precision_bykey[(cid, competitor)]
            uncertainty = float(np.sqrt(np.mean([m["reference_uncertainty_rms"]**2
                                                for m in metrics_bykey[(cid, competitor)]])))
            context_rows.append(dict(context_id=cid, parent_id=our["parent_id"], parent_family=our["parent_family"],
                competitor_seconds_over_fit=float(np.median(ratios)),
                fit_rms_over_competitor=our["pooled_source_rms"]/other["pooled_source_rms"],
                fit_rms=our["pooled_source_rms"], competitor_rms=other["pooled_source_rms"],
                reference_uncertainty_rms=uncertainty,
                competitor_rms_over_estimated_reference_uncertainty=other["pooled_source_rms"]/uncertainty,
                competitor_error_exceeds_ten_reference_uncertainties=other["pooled_source_rms"] > 10*uncertainty,
                fit_median_seconds=our["median_seconds"], competitor_median_seconds=other["median_seconds"]))
        comparisons.append(dict(competitor=competitor, paired_round_time_ratios=distribution(paired_ratios),
            context_median_time_ratios=distribution([r["competitor_seconds_over_fit"] for r in context_rows]),
            context_rms_ratios=distribution([r["fit_rms_over_competitor"] for r in context_rows]),
            fit_faster_contexts=sum(r["competitor_seconds_over_fit"] > 1 for r in context_rows),
            fit_more_accurate_contexts=sum(r["fit_rms_over_competitor"] < 1 for r in context_rows),
            competitor_error_exceeds_ten_reference_uncertainties_contexts=sum(
                r["competitor_error_exceeds_ten_reference_uncertainties"] for r in context_rows),
            fit_faster_and_more_accurate_contexts=sum(r["competitor_seconds_over_fit"] > 1 and
                r["fit_rms_over_competitor"] < 1 for r in context_rows), contexts=context_rows))

    learning_costs = []
    for c in curves:
        if c["method"] in cost_byid and c["split"] in ("validation", "diagnostic"):
            cc = cost_byid[c["method"]]
            learning_costs.append(dict(model_id=c["method"], split=c["split"], rms=c["rms"],
                unique_training_labels=cc["training"]["planned"], training_reference_seconds=cc["training_reference_seconds"],
                shared_validation_reference_seconds=cc["validation_reference_seconds"], fitting_seconds=cc["fitting_seconds"]))
    return dict(schema="tdn.native_p1.cost_fit_review.v1", status="PASS" if not failures else "FAIL",
        scope="Read-only replay of saved coefficients/designs/metrics/cost/timing records; no refitting, teacher acquisition or new scientific cohort. FP64 checks are not rigorous interval certificates.",
        checks=int(checks["checked"]), failures=failures, input_sha256=input_hashes,
        query_arrays_hash_checked=len(arrays), fits=fits, costs=costs,
        sum_cell_fitting_seconds=cell_fit_sum,
        fitting_outer_minus_cell_seconds=costs["fitting_seconds"]-cell_fit_sum,
        summed_overlapping_training_reference_demands_seconds=sum(c["training_reference_seconds"] for c in costs["cells"]),
        learning_curve_costs=learning_costs, timing_record_count=len(records), timing_contexts=len(timing),
        paired_comparisons=comparisons,
        interpretation_limits=[
            "Rank and singular shrinkage describe this finite weighted feature-design bank, not full response-operator rank or model approximation capacity.",
            "Effective degrees of freedom are trace of the ordinary-ridge linear smoother in the original coefficient coordinates, not an independently validated physical dimension.",
            "A small positive-ridge optimality gap proves neither low unregularized bias nor best possible validation error; no smaller ridge or local oracle is fitted in this review.",
            "Timing ratios are descriptive paired observations on four diagnostic parents and six contexts each; repeated rounds are not independent physical experiments.",
            "Ratios to near-reference-floor errors are numerical observations, not resolved precision gains; pooled uncertainty estimates are reported and are not certified bounds.",
            "Whole four-source sequential bundles include per-source setup but exclude offline teacher acquisition and fitting; no cached-source or full-solver matched-tolerance speedup is established.",
            "Durations are retained observations, not independently remeasured wall time. Per-cell label demands overlap and cannot be summed as global costs.",
        ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = review(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "failures", "timing_record_count")}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
