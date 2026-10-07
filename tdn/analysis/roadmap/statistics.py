"""Function-cluster inference for the frozen roadmap experiments.

Repeated grids, schedules, coefficient variants and training seeds are not
new independent functions. These helpers preserve that distinction, return
JSON-safe missing values, and never turn finite samples into formal PDE proofs.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

import numpy as np
from scipy.stats import beta


def _finite_nonnegative(value, label="score"):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def conformal_quantile(scores, alpha=.01):
    """Split-conformal order statistic for one score per independent function.

    k>n means the formal quantile is infinite. It is represented as null plus
    an explicit unavailable status, rather than nonstandard JSON Infinity.
    Exchangeability and a frozen scoring rule remain external assumptions.
    """
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    values = sorted(_finite_nonnegative(v) for v in scores)
    n = len(values)
    rank = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    finite = n > 0 and rank <= n
    return {"quantile": values[rank - 1] if finite else None, "rank": rank,
            "independent_functions": n, "alpha": alpha,
            "status": "FINITE_QUANTILE" if finite else "INSUFFICIENT_INDEPENDENT_FUNCTIONS",
            "minimum_functions_for_finite_quantile": math.ceil(1 / alpha - 1 - 1e-12),
            "guarantee_scope": "marginal function-level coverage under exchangeability and a fixed score",
            "conditional_on_acceptance_guarantee": False,
            "deterministic_certificate": False}


def independent_function_scores(rows, *, score_key="score", cluster_key="field_cluster"):
    """Collapse every declared query/variant/seed in a function to its maximum."""
    grouped = defaultdict(list)
    for row in rows:
        if cluster_key not in row or row[cluster_key] is None:
            raise ValueError("A real independent function cluster identifier is required")
        grouped[str(row[cluster_key])].append(_finite_nonnegative(row[score_key]))
    return [{"field_cluster": key, "score": max(values), "query_rows": len(values)}
            for key, values in sorted(grouped.items())]


def _cohort_tokens(parent):
    tokens = set()
    for key in ("parent_id", "field_cluster", "continuous_field_id"):
        if parent.get(key) is not None:
            tokens.add((key, str(parent[key])))
    # Catch renaming an already used continuous function. Coefficient terms
    # include phase, so deliberately shared variant families are also caught.
    if parent.get("field_terms"):
        payload = json.dumps(parent["field_terms"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        tokens.add(("field_terms_sha256", hashlib.sha256(payload.encode()).hexdigest()))
    if not tokens:
        raise ValueError("Cohort member has no traceable independent identity")
    return tokens


def assert_disjoint_cohorts(**cohorts):
    seen = {}
    counts = {}
    for name, parents in cohorts.items():
        own = set()
        for parent in parents:
            own.update(_cohort_tokens(parent))
        for token in own:
            if token in seen:
                raise ValueError(f"Cohort contamination between {seen[token]} and {name}: {token[0]}")
            seen[token] = name
        counts[name] = len({str(p.get("field_cluster", p.get("continuous_field_id", p.get("parent_id")))) for p in parents})
    return {"disjoint": True, "independent_field_clusters": counts,
            "checks": ["parent_id", "field_cluster", "continuous_field_id", "immutable_field_terms"]}


def paired_cluster_bootstrap(rows, *, value_key="difference", cluster_key="field_cluster",
                             seed_key="seed", repeats=1000, random_seed=740991, confidence=.95):
    """Crossed field/seed bootstrap of paired differences.

    First average repeated queries within each field x training-seed cell;
    then give every field and seed equal weight. Resampling keeps the two
    methods paired. Missing cells make the confirmatory crossed design NA.
    Intervals are descriptive bootstrap inference, not distribution-free bounds.
    """
    if type(repeats) is not int or repeats < 1 or not 0 < confidence < 1:
        raise ValueError("Positive repeat count and an interior confidence level are required")
    groups = defaultdict(list)
    for row in rows:
        if cluster_key not in row or seed_key not in row:
            raise ValueError("Paired bootstrap requires both field and training-seed identities")
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("Bootstrap observations must be finite")
        groups[(str(row[cluster_key]), str(row[seed_key]))].append(value)
    fields = sorted({key[0] for key in groups})
    seeds = sorted({key[1] for key in groups})
    base = {"query_rows": len(rows), "independent_fields": len(fields), "training_seeds": len(seeds),
            "paired_cells": len(groups), "repeats": repeats, "confidence": confidence,
            "unit": "independent field; training seeds crossed; repeated queries averaged within cells",
            "interval_scope": "descriptive paired crossed-cluster bootstrap; no population guarantee"}
    if not groups:
        return {**base, "status": "NO_OBSERVATIONS", "mean": None, "lower": None, "upper": None}
    if any((field, seed) not in groups for field in fields for seed in seeds):
        return {**base, "status": "INCOMPLETE_CROSSED_DESIGN", "mean": None, "lower": None, "upper": None}
    matrix = np.array([[np.mean(groups[(field, seed)]) for seed in seeds] for field in fields])
    mean = float(matrix.mean())
    if len(fields) < 2:
        return {**base, "status": "INSUFFICIENT_INDEPENDENT_FIELDS", "mean": mean, "lower": None, "upper": None}
    rng = np.random.default_rng(random_seed)
    draws = []
    for _ in range(repeats):
        sampled_fields = rng.integers(len(fields), size=len(fields))
        sampled_seeds = rng.integers(len(seeds), size=len(seeds))
        draws.append(float(matrix[np.ix_(sampled_fields, sampled_seeds)].mean()))
    lo, hi = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {**base, "status": "DESCRIPTIVE_INTERVAL", "mean": mean, "lower": float(lo), "upper": float(hi),
            "seed_generalization_tested": len(seeds) >= 3}


def selective_risk_summary(rows, *, confidence=.95, target_risk=.01):
    """Audit false acceptance with independent function clusters as the unit.

    A unit is bad if any accepted query in that function is bad. This is a
    joint-query function-level endpoint, not a pooled per-query Bernoulli test.
    The binomial bound is conditional on iid comparable field units and a
    policy frozen before these fields. It is separate from conformal coverage.
    """
    if not 0 < confidence < 1 or not 0 < target_risk < 1:
        raise ValueError("Confidence and risk targets must be strictly interior")
    accepted = defaultdict(list)
    all_clusters = set()
    for row in rows:
        if row.get("field_cluster") is None:
            raise ValueError("Selective risk requires independent field clusters")
        cluster = str(row["field_cluster"])
        all_clusters.add(cluster)
        if row.get("accepted", not row.get("fallback", True)):
            if row.get("false_accept") is None:
                raise ValueError("Accepted observations require an audited false_accept label")
            accepted[cluster].append(bool(row["false_accept"]))
    n, failures = len(accepted), sum(any(values) for values in accepted.values())
    upper = (1. if failures == n else float(beta.ppf(confidence, failures + 1, n - failures))) if n else None
    required_zero = math.ceil(math.log(1 - confidence) / math.log(1 - target_risk))
    return {"accepted_independent_functions": n, "bad_accepted_functions": failures,
            "all_independent_functions": len(all_clusters),
            "accepted_query_rows": sum(len(v) for v in accepted.values()),
            "conditional_function_risk": failures / n if n else None,
            "one_sided_binomial_upper": upper, "confidence": confidence, "target_risk": target_risk,
            "zero_failure_functions_needed": required_zero,
            "target_supported_under_iid_assumptions": upper <= target_risk if upper is not None else None,
            "status": "NO_ACCEPTED_FUNCTIONS" if n == 0 else "FINITE_SAMPLE_AUDIT",
            "assumptions": "frozen policy and iid comparable independent fields; not guaranteed under distribution shift",
            "endpoint": "at least one bad accepted declared query within a function",
            "conformal_marginal_coverage_is_conditional_risk": False}


def cost_distribution(samples):
    values = np.asarray([_finite_nonnegative(v, "cost") for v in samples], dtype=float)
    if not len(values):
        return {"count": 0, "total_seconds": 0., "mean_seconds": None, "median_seconds": None,
                "p90_seconds": None, "p95_seconds": None, "p99_seconds": None, "max_seconds": None}
    return {"count": len(values), "total_seconds": float(values.sum()), "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)), "p90_seconds": float(np.quantile(values, .9)),
            "p95_seconds": float(np.quantile(values, .95)), "p99_seconds": float(np.quantile(values, .99)),
            "max_seconds": float(values.max())}


def amortization_summary(offline_components, deployed_mean_seconds, classical_mean_seconds, *, matched_accuracy):
    """Break-even queries exist only with matched accuracy and positive savings.

    Components must be actually recorded wall times. Missing setup/training/
    teacher costs remain null; they are never imputed as free work.
    """
    components = {str(key): None if value is None else _finite_nonnegative(value, "offline cost")
                  for key, value in offline_components.items()}
    complete = bool(components) and all(value is not None for value in components.values())
    total = sum(components.values()) if complete else None
    deployed = None if deployed_mean_seconds is None else _finite_nonnegative(deployed_mean_seconds, "deployed mean cost")
    baseline = None if classical_mean_seconds is None else _finite_nonnegative(classical_mean_seconds, "classical mean cost")
    saving = baseline - deployed if baseline is not None and deployed is not None else None
    status = ("ACCURACY_NOT_MATCHED" if not matched_accuracy else
              "MISSING_OFFLINE_COST" if not complete else
              "MISSING_QUERY_COST" if saving is None else
              "NO_POSITIVE_PER_QUERY_SAVING" if saving <= 0 else "FINITE_BREAK_EVEN")
    return {"status": status, "offline_components_seconds": components, "offline_total_seconds": total,
            "per_query_saving_seconds": saving, "break_even_queries": math.ceil(total / saving) if status == "FINITE_BREAK_EVEN" else None,
            "fractional_break_even_queries": total / saving if status == "FINITE_BREAK_EVEN" else None,
            "matched_accuracy": bool(matched_accuracy),
            "scope": "complete recorded shared-bank prepare/train/calibration/loading cost; conservative whole-experiment cost, not per-arm attribution",
            "future_timings_are_forecasts": True}
