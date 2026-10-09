"""Bounded crossed-field/seed bootstrap preserving the historical RNG draws.

Vectorization changes execution cost, not the pairing or inferential unit.
Repeated queries are averaged inside each field/seed cell. A missing cell
makes the crossed design unavailable; it is never silently dropped.
"""
from collections import defaultdict
import math
import numpy as np


def paired_cluster_interval_fast(
    rows, *, value_key="difference", cluster_key="field_cluster", seed_key="seed",
    repeats=1000, random_seed=740991, confidence=.95, minimum_fields=2,
):
    if type(repeats) is not int or repeats < 1 or not 0 < confidence < 1:
        raise ValueError("Positive repeat count and an interior confidence level are required")
    if type(minimum_fields) is not int or minimum_fields < 2:
        raise ValueError("Minimum independent field count must be at least two")
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
    base = dict(
        query_rows=len(rows), independent_fields=len(fields), training_seeds=len(seeds),
        paired_cells=len(groups), repeats=repeats, confidence=confidence,
        unit="independent field; training seeds crossed; repeated queries averaged within cells",
        interval_scope="descriptive paired crossed-cluster bootstrap; no population guarantee",
    )
    if not groups:
        return {**base, "status": "NO_OBSERVATIONS", "mean": None, "lower": None, "upper": None}
    if any((field, seed) not in groups for field in fields for seed in seeds):
        return {**base, "status": "INCOMPLETE_CROSSED_DESIGN", "mean": None, "lower": None, "upper": None}
    matrix = np.array([[np.mean(groups[(field, seed)]) for seed in seeds] for field in fields])
    mean = float(matrix.mean())
    if len(fields) < minimum_fields:
        return {**base, "status": "INSUFFICIENT_INDEPENDENT_FIELDS", "mean": mean,
                "lower": None, "upper": None}

    # Broadcasting preserves the old interleaved RNG order: all field indices
    # followed by all seed indices for each replicate. It does not draw one
    # giant field block followed by a seed block, which would change the draw.
    rng = np.random.default_rng(random_seed)
    bounds = np.array([len(fields)] * len(fields) + [len(seeds)] * len(seeds))
    # Bound the resampled matrix to one million entries (about 8 MiB FP64).
    # Chunks consume the same RNG stream and retain the exact repeated draws.
    chunk = max(1, min(repeats, 1_000_000 // max(1, len(fields) * len(seeds))))
    draws = []
    for start in range(0, repeats, chunk):
        count = min(chunk, repeats - start)
        indices = rng.integers(0, bounds, size=(count, len(bounds)))
        field_indices = indices[:, :len(fields)]
        seed_indices = indices[:, len(fields):]
        resampled = matrix[field_indices[:, :, None], seed_indices[:, None, :]]
        draws.extend(resampled.mean(axis=(1, 2)).tolist())
    lo, hi = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {**base, "status": "DESCRIPTIVE_INTERVAL", "mean": mean,
            "lower": float(lo), "upper": float(hi), "seed_generalization_tested": len(seeds) >= 3}


def field_cluster_interval(rows, *, value_key="difference", repeats=1000, seed=73421, minimum_fields=2):
    """Legacy descriptive interval; primary gates separately require five fields."""
    return paired_cluster_interval_fast(
        rows, value_key=value_key, repeats=repeats, random_seed=seed,
        minimum_fields=minimum_fields,
    )
