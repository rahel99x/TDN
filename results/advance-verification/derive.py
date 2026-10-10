"""Recompute the bounded diagnostic findings from the published exact-byte data.

Run with any Python 3.10+ interpreter. No model fitting or confirmation selection.
The 32 cases share four constructed fields across two tracks and four horizons;
they are not 32 independent random fields, so no sampling CI is calculated.
"""
from __future__ import annotations

import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads(gzip.decompress((ROOT / name).read_bytes()))


def main():
    data = read("development/diagnose/diagnostic_rows.json.gz")["rows"]
    matched = defaultdict(dict)
    for row in data:
        matched[(row["track"], row["regime"], row["grid"], row["horizon"])][row["family"]] = row
    comparisons = {}
    for family in ("commutator_raw", "commutator_cubic", "analytic_quad_cubic", "quad4_full"):
        ratios, worsened, qualified = [], [], []
        by_regime = defaultdict(list)
        for key, methods in sorted(matched.items()):
            base, candidate = methods["quad2_full"], methods[family]
            assert base["reference_accepted"] and candidate["reference_accepted"]
            ratio = base["error_rms"] / candidate["error_rms"]
            ratios.append(ratio)
            by_regime["/".join(key[:2])].append(ratio)
            if candidate["error_rms"] > base["error_rms"]:
                worsened.append(list(key))
            if all(r["error_rms"] > r["reference_uncertainty_rms"] for r in (base, candidate)):
                qualified.append(list(key))
        comparisons[family] = {
            "paired_cases": len(ratios), "rms_worse_cases": worsened,
            "both_errors_above_estimated_reference_uncertainty": len(qualified),
            "gl2_error_over_candidate_error_min": min(ratios),
            "gl2_error_over_candidate_error_max": max(ratios),
            "median_by_track_and_regime": {k: statistics.median(v) for k, v in by_regime.items()},
        }
    absolute = {}
    for family in sorted({r["family"] for r in data}):
        rows = [r for r in data if r["family"] == family]
        absolute[family] = {
            "cases": len(rows), "rms_min": min(r["error_rms"] for r in rows),
            "rms_max": max(r["error_rms"] for r in rows),
            "joint_rms_max_2e_minus_5_passes": sum(r["upper_rms"] <= 2e-5 and r["upper_max"] <= 2e-5 for r in rows),
        }
    oracle = read("development/diagnose/residual_projections.json.gz")["rows"]
    result = {
        "classification": "post-hoc development diagnostics; CPU FP64; no independent-field inference or speed claim",
        "grid": [16, 16], "endpoint_rows": len(data), "paired_cases": len(matched),
        "comparisons": comparisons, "absolute_error_and_feasibility": absolute,
        "teacher_informed_oracle": {
            "scope": "unbounded per-case projection; not a learned deployable model",
            "median_one_basis_error_over_two_basis_error": statistics.median(r["one_basis_oracle_error_rms"] / r["oracle_error_rms"] for r in oracle),
            "coefficient_ranges": [[min(r["oracle_coefficients"][i] for r in oracle), max(r["oracle_coefficients"][i] for r in oracle)] for i in (0, 1)],
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
