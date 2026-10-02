"""Uncertainty across independent physical parents, never across cells."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import beta


def failure_upper_bound(failures: int, parents: int, *, alpha: float = 0.05) -> float:
    """One-sided exact Clopper--Pearson upper bound (confidence 1-alpha)."""
    if parents < 1 or not 0 <= failures <= parents or not 0 < alpha < 1:
        raise ValueError("Need independent parents >= 1, valid failures and alpha")
    if failures == parents:
        return 1.0
    if failures == 0:
        return float(1.0 - alpha ** (1.0 / parents))
    return float(beta.ppf(1.0 - alpha, failures + 1, parents - failures))


def parent_summary(errors: Sequence[float | None], *, bootstrap_samples: int = 2000,
                   seed: int = 0, alpha: float = 0.05) -> dict:
    """Report failure counts independently of successful-parent error summaries.

    Null/nonfinite entries represent failed parents. They remain in the denominator
    and prevent an all-parent mean claim; successful-only metrics are labeled.
    """
    values = np.asarray([np.nan if x is None else x for x in errors], dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot summarize no parents")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    finite = values[np.isfinite(values)]
    failures = int(len(values) - len(finite))
    result = {
        "independent_parents": len(values), "failed_parents": failures,
        "confidence_level": 1.0 - alpha,
        "failure_probability_upper_bound": failure_upper_bound(failures, len(values), alpha=alpha),
        "all_parent_mean": None if failures else float(finite.mean()),
        "all_parent_mean_unavailable_reason": "Failures remain in denominator" if failures else None,
        "successful_parent_mean": float(finite.mean()) if len(finite) else None,
        "successful_parent_max": float(finite.max()) if len(finite) else None,
        "successful_parent_quantiles": {str(q): float(np.quantile(finite, q)) for q in (0.5, 0.9, 0.95)} if len(finite) else {},
        "bootstrap_unit": "independent physical parent",
    }
    if len(finite):
        rng = np.random.default_rng(seed)
        # Bounded storage even when a campaign has many parents.
        means = np.empty(bootstrap_samples)
        for index in range(bootstrap_samples):
            means[index] = rng.choice(finite, size=len(finite), replace=True).mean()
        result["successful_parent_mean_bootstrap_interval"] = np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist()
    else:
        result["successful_parent_mean_bootstrap_interval"] = None
    return result


def paired_parent_comparison(baseline: Sequence[float], candidate: Sequence[float], *,
                             bootstrap_samples: int = 2000, seed: int = 0,
                             alpha: float = 0.05) -> dict:
    """Paired bootstrap; refuses to silently drop a failed parent."""
    base, learned = np.asarray(baseline, float), np.asarray(candidate, float)
    if base.shape != learned.shape or base.ndim != 1 or not len(base):
        raise ValueError("Need aligned nonempty parent vectors")
    if not np.isfinite(base).all() or not np.isfinite(learned).all():
        return {"available": False, "reason": "Failed/nonfinite parent; do not drop it from a paired comparison"}
    deltas = learned - base
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(deltas, len(deltas), replace=True).mean()
                        for _ in range(bootstrap_samples)])
    return {"available": True, "parents": len(base), "candidate_minus_baseline_mean": float(deltas.mean()),
            "paired_parent_bootstrap_interval": np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist(),
            "confidence_level": 1 - alpha, "bootstrap_unit": "independent paired parent"}


def break_even_uses(data_seconds: float, train_seconds: float, setup_seconds: float,
                    baseline_seconds: float, learned_seconds: float) -> int | None:
    if min(data_seconds, train_seconds, setup_seconds, baseline_seconds, learned_seconds) < 0:
        raise ValueError("Costs must be nonnegative and in consistent seconds")
    gain = baseline_seconds - learned_seconds
    return None if gain <= 0 else math.ceil((data_seconds + train_seconds + setup_seconds) / gain)
