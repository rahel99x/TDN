"""Frozen, parent-clustered comparisons with explicit reference sensitivity.

Intervals are conditional on the trained models and the frozen query menu.  A
training seed or another horizon of the same field is never another parent.
This module does not select models or infer missing experimental outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np
from scipy.stats import t as student_t


def reference_interval(error, uncertainty, tolerance):
    """Deterministic triangle-inequality bounds; not a confidence interval."""
    if any(not math.isfinite(float(x)) or x < 0 for x in (error, uncertainty, tolerance)):
        raise ValueError("Error, uncertainty, and tolerance must be finite and nonnegative")
    lower, upper = max(0., float(error)-float(uncertainty)), float(error)+float(uncertainty)
    outcome = "QUALIFIED" if upper <= tolerance else "FAILED" if lower > tolerance else "UNRESOLVED"
    return dict(lower=lower, upper=upper, qualification=outcome,
                interpretation="deterministic reference sensitivity; not statistical uncertainty")


def holm_adjust(pvalues):
    """Holm familywise adjustment. Unresolved tests retain their family slot."""
    values = [1. if p is None else float(p) for p in pvalues]
    if any(not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid p-value")
    order = np.argsort(values, kind="stable")
    result, running = [1.]*len(values), 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[index]))
        result[int(index)] = running
    return result


def _number(row, key):
    value = row.get(key)
    return value is not None and math.isfinite(float(value)) and float(value) >= 0


def _bootstrap(values, statistic, *, draws, seed, alpha):
    values = np.asarray(values, float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = np.array([statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)])
    if not np.isfinite(samples).all():
        return None
    return [float(np.quantile(samples, alpha/2)), float(np.quantile(samples, 1-alpha/2))]


def _registry(registry):
    required = ("parents", "query_ids", "seeds", "target_kind")
    if any(not registry.get(k) for k in required):
        raise ValueError("A complete frozen parent/query/seed/target registry is required")
    for key in ("parents", "query_ids", "seeds"):
        values = [str(v) for v in registry[key]]
        if len(values) != len(set(values)):
            raise ValueError("Registry repeats an identity")
    if registry.get("analysis_kind") == "confirmation" and not registry.get("frozen_before_evaluation"):
        raise ValueError("Confirmation requires a protocol frozen before evaluation")
    alpha = float(registry.get("alpha", .05))
    if not 0 < alpha < 1 or registry.get("draws", 2000) < 100:
        raise ValueError("Invalid uncertainty configuration")
    weights = registry.get("query_weights", {str(q): 1. for q in registry["query_ids"]})
    if set(weights) != {str(q) for q in registry["query_ids"]} or any(not math.isfinite(float(v)) or v <= 0 for v in weights.values()):
        raise ValueError("Positive frozen weights must cover exactly the query menu")
    return {str(k): float(v)/sum(weights.values()) for k, v in weights.items()}


def compare_endpoints(rows, candidate, comparator, registry):
    """Compare one predeclared pair without dropping failed or absent outputs.

    Rows use parent_id/query_id/seed/method, RMS and maximum in the *same physical
    norm*, accepted reference flags and deterministic uncertainty estimates.
    The registry, rather than rows that happened to finish, sets denominators.
    """
    weights = _registry(registry)
    parents, queries, seeds = ([str(v) for v in registry[k]] for k in ("parents", "query_ids", "seeds"))
    lookup = {}
    for row in rows:
        if row.get("method") not in (candidate, comparator):
            continue
        key = (str(row["parent_id"]), str(row["query_id"]), str(row["seed"]), row["method"])
        if key[:3][0] not in parents or key[1] not in queries or key[2] not in seeds:
            raise ValueError("Endpoint is outside the frozen parent/query/seed menu")
        if row.get("target_kind") != registry["target_kind"]:
            raise ValueError("Different target equations cannot share a comparison")
        if registry.get("norm") and row.get("norm") != registry["norm"]:
            raise ValueError("Endpoint physical norm differs from the registered estimand")
        if key in lookup:
            raise ValueError("Duplicate endpoint would change descendant weighting")
        lookup[key] = row
    counts = {m: Counter(attempted=len(parents)*len(queries)*len(seeds)) for m in (candidate, comparator)}
    parent_rows = []
    rms_tol = float(registry.get("primary_tolerance", 2e-5))
    max_tol = float(registry.get("maximum_tolerance", rms_tol))
    bound_delta = 0
    for parent in parents:
        data = {m: dict(rms=0., rms_lower=0., rms_upper=0., peaks=[], peak_upper=[], peak_lower=[], coverage=0.,
                       complete=True, bound=False, accepted_weight=0.) for m in (candidate, comparator)}
        for query in queries:
            for seed in seeds:
                weight = weights[query]/len(seeds)
                pair = [lookup.get((parent, query, seed, m)) for m in (candidate, comparator)]
                for j, method in enumerate((candidate, comparator)):
                    row, other = pair[j], pair[1-j]
                    current, counter = data[method], counts[method]
                    # A paired valid teacher remains valid even when a method did not write its prediction.
                    teacher = row if row is not None else other
                    accepted = teacher is not None and bool(teacher.get("reference_accepted"))
                    if row is None:
                        counter["missing_prediction"] += 1
                    if not accepted:
                        counter["unresolved_reference"] += 1
                        current["complete"] = False
                        continue
                    counter["accepted_reference"] += 1
                    current["accepted_weight"] += weight
                    if row is None or not row.get("prediction_finite", True) or not _number(row, "rms") or not _number(row, "maximum"):
                        counter["failed_prediction_on_valid_reference"] += 1
                        current["complete"] = False
                        continue
                    if not _number(row, "uncertainty_rms") or not _number(row, "uncertainty_maximum"):
                        counter["missing_uncertainty"] += 1
                        current["complete"] = False
                        continue
                    if type(row.get("reference_bounded")) is not bool or (row.get("reference_bounded") and type(row.get("bound_violation")) is not bool):
                        counter["missing_bound_measurement"] += 1
                    counter["finite_prediction_on_valid_reference"] += 1
                    rms = reference_interval(row["rms"], row["uncertainty_rms"], rms_tol)
                    peak = reference_interval(row["maximum"], row["uncertainty_maximum"], max_tol)
                    counter["rms_"+rms["qualification"].lower()] += 1
                    counter["maximum_"+peak["qualification"].lower()] += 1
                    current["rms"] += weight*row["rms"]
                    current["rms_lower"] += weight*rms["lower"]
                    current["rms_upper"] += weight*rms["upper"]
                    current["peaks"].append(row["maximum"])
                    current["peak_lower"].append(peak["lower"])
                    current["peak_upper"].append(peak["upper"])
                    if rms["qualification"] == peak["qualification"] == "QUALIFIED":
                        current["coverage"] += weight
                        counter["joint_qualified"] += 1
                    if row.get("reference_bounded", False) and row.get("bound_violation", False):
                        current["bound"] = True
                        counter["bound_violation"] += 1
                if pair[0] is not None and pair[1] is not None and pair[0].get("reference_accepted") != pair[1].get("reference_accepted"):
                    raise ValueError("Paired methods disagree about reference acceptance")
                if all(r is not None and r.get("reference_accepted") for r in pair):
                    for key in ("uncertainty_rms", "uncertainty_maximum"):
                        if pair[0].get(key) != pair[1].get(key):
                            raise ValueError("Paired methods must use the same deterministic reference uncertainty")
        c, b = data[candidate], data[comparator]
        complete = c["complete"] and b["complete"]
        ratio = c["rms"]/b["rms"] if complete and b["rms"] > 0 and c["rms"] > 0 else None
        bound_delta += int(c["bound"] and not b["bound"])
        parent_rows.append(dict(parent_id=parent, complete=complete, candidate_rms=c["rms"] if complete else None,
            comparator_rms=b["rms"] if complete else None, rms_ratio=ratio,
            reference_ratio_lower=c["rms_lower"]/b["rms_upper"] if complete and b["rms_upper"] > 0 else None,
            reference_ratio_upper=c["rms_upper"]/b["rms_lower"] if complete and b["rms_lower"] > 0 else None,
            candidate_peak=max(c["peaks"], default=None) if complete else None,
            comparator_peak=max(b["peaks"], default=None) if complete else None,
            candidate_peak_upper=max(c["peak_upper"], default=None) if complete else None,
            comparator_peak_lower=max(b["peak_lower"], default=None) if complete else None,
            candidate_coverage=c["coverage"], comparator_coverage=b["coverage"],
            coverage_difference=c["coverage"]-b["coverage"], candidate_new_bound_violation=c["bound"] and not b["bound"]))
    usable = [r for r in parent_rows if r["rms_ratio"] is not None]
    alpha, draws, seed = float(registry.get("alpha", .05)), int(registry.get("draws", 2000)), int(registry.get("bootstrap_seed", 92417))
    logs = np.log([r["rms_ratio"] for r in usable])
    mean_log = float(logs.mean()) if len(logs) else None
    ratio = math.exp(mean_log) if mean_log is not None else None
    minimum = int(registry.get("minimum_parents", 16))
    sufficient = len(usable) >= max(2, minimum) and len(usable) == len(parents)
    pvalue = None
    if sufficient:
        stderr = float(logs.std(ddof=1)/math.sqrt(len(logs)))
        pvalue = float(student_t.cdf(mean_log/stderr, len(logs)-1)) if stderr else (0. if mean_log < 0 else 1.)
    ratio_ci = _bootstrap(logs, lambda x: np.exp(x.mean()), draws=draws, seed=seed, alpha=alpha)
    peak_data = np.array([[r["candidate_peak"], r["comparator_peak"]] for r in usable])
    peak_ratio = float(peak_data[:, 0].mean()/peak_data[:, 1].mean()) if len(usable) and peak_data[:, 1].mean() > 0 else None
    peak_ci = _bootstrap(peak_data, lambda x: x[:, 0].mean()/x[:, 1].mean() if x[:, 1].mean() else np.inf,
                         draws=draws, seed=seed+1, alpha=alpha) if len(usable) else None
    coverage = np.array([r["coverage_difference"] for r in parent_rows])
    coverage_ci = _bootstrap(coverage, np.mean, draws=draws, seed=seed+2, alpha=alpha)
    robust_ratios = [r["reference_ratio_upper"] for r in usable]
    robust_ratio = float(np.exp(np.log(robust_ratios).mean())) if robust_ratios and all(v is not None and v > 0 for v in robust_ratios) else None
    robust_pvalue = None
    if sufficient and robust_ratio is not None:
        robust_logs = np.log(robust_ratios)
        robust_stderr = float(robust_logs.std(ddof=1)/math.sqrt(len(robust_logs)))
        robust_pvalue = float(student_t.cdf(robust_logs.mean()/robust_stderr, len(robust_logs)-1)) if robust_stderr else (0. if robust_logs.mean() < 0 else 1.)
    peak_robust = (sum(r["candidate_peak_upper"] for r in usable)/sum(r["comparator_peak_lower"] for r in usable)
                   if usable and sum(r["comparator_peak_lower"] for r in usable) > 0 else None)
    robust_peak_data = np.array([[r["candidate_peak_upper"], r["comparator_peak_lower"]] for r in usable])
    robust_peak_ci = _bootstrap(robust_peak_data, lambda x: x[:, 0].mean()/x[:, 1].mean() if x[:, 1].mean() else np.inf,
                         draws=draws, seed=seed+1, alpha=alpha) if len(usable) else None
    registry_hash = hashlib.sha256(json.dumps(registry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return dict(schema="tdn.response-claim/v1", candidate=candidate, comparator=comparator,
        target_kind=registry["target_kind"], analysis_kind=registry.get("analysis_kind", "development"), registry_sha256=registry_hash,
        estimand="equal independent parents; frozen weighted mean RMS across queries and training seeds within parent; geometric paired ratio across parents",
        uncertainty_scope="conditional on fitted models and fixed query/seed menu; percentile parent bootstrap is descriptive, paired log-t evidence is model-assumption dependent",
        reference_uncertainty_scope="shared deterministic sensitivity; never sampled as independent noise",
        attempted_parents=len(parents), complete_paired_parents=len(usable), minimum_parents=minimum,
        denominators={m: dict(v) for m, v in counts.items()}, parent_rows=parent_rows,
        sufficient_complete_parents=sufficient, rms_geometric_ratio=ratio, rms_ratio_interval=ratio_ci,
        primary_one_sided_pvalue=pvalue, reference_worst_case_one_sided_pvalue=robust_pvalue, reference_worst_case_rms_ratio=robust_ratio,
        parent_peak_mean_ratio=peak_ratio, parent_peak_ratio_interval=peak_ci, reference_worst_case_peak_ratio=peak_robust,
        reference_worst_case_peak_ratio_interval=robust_peak_ci,
        worst_parent_peak={m: max((r["candidate_peak" if m == candidate else "comparator_peak"] for r in usable), default=None) for m in (candidate, comparator)},
        joint_coverage_difference=float(coverage.mean()), joint_coverage_difference_interval=coverage_ci,
        candidate_new_bound_violation_parents=bound_delta,
        bound_measurement_complete=all(v.get("missing_bound_measurement", 0) == 0 for v in counts.values()),
        primary_point_threshold=.90, peak_noninferiority_margin=1.05, coverage_noninferiority_margin=-.02,
        tail_noninferiority_supported=bool(sufficient and peak_ci is not None and peak_ci[1] <= 1.05 and robust_peak_ci is not None and robust_peak_ci[1] <= 1.05),
        coverage_noninferiority_supported=bool(sufficient and coverage_ci is not None and coverage_ci[0] >= -.02),
        observation="Exploratory development comparison" if registry.get("analysis_kind") != "confirmation" else "Frozen confirmation comparison")


def claims_payload(comparisons, alpha=.05):
    """Apply a declared comparison-family correction; preserve unresolved slots."""
    if not comparisons:
        return dict(schema="tdn.response-claims/v1", comparisons=[], status="NO_EVIDENCE")
    adjusted = holm_adjust([max(c["primary_one_sided_pvalue"], c["reference_worst_case_one_sided_pvalue"])
                           if c["primary_one_sided_pvalue"] is not None and c.get("reference_worst_case_one_sided_pvalue") is not None else None for c in comparisons])
    result = []
    for c, p in zip(comparisons, adjusted):
        c = dict(c, holm_adjusted_pvalue=p)
        primary = bool(c["sufficient_complete_parents"] and p <= alpha and c["rms_geometric_ratio"] <= .90
                       and c["reference_worst_case_rms_ratio"] is not None and c["reference_worst_case_rms_ratio"] < 1.)
        c["primary_rms_supported"] = primary
        c["composite_advantage_supported"] = bool(primary and c["tail_noninferiority_supported"] and
            c["coverage_noninferiority_supported"] and c["bound_measurement_complete"] and c["candidate_new_bound_violation_parents"] == 0)
        c["verdict"] = ("DEVELOPMENT_SIGNAL" if c["analysis_kind"] != "confirmation" else "SUPPORTED_WITHIN_FROZEN_SCOPE") if c["composite_advantage_supported"] else "RMS_ONLY_UNRESOLVED_TRADEOFF" if primary else "UNRESOLVED_OR_NO_ADVANTAGE"
        result.append(c)
    return dict(schema="tdn.response-claims/v1", alpha=alpha, multiplicity="Holm on the complete declared primary family",
                comparisons=result, status="REPORTED", universal_leaderboard=False)
