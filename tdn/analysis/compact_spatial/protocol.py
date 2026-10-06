"""Versioned declarations for compact spatial accuracy and cost diagnostics."""
from __future__ import annotations

import json

from tdn.analysis.work_precision.protocol import (
    TABLE_IDS, candidate_id, digest, file_digest, frontier_id,
    validate_rows as _validate_work_rows,
)

METHODS = ("strang", "etdrk4", "gl3", "gl5", "gl3_mean_spectral", "gl3_fused",
           "compact_gl3_m2", "compact_gl3_m4", "compact_gl3_m8", "compact_gl3_m4_no_mean")
COMPACT_METHODS = METHODS[6:]
NSTEPS = (1, 2, 4, 8, 16, 32)
SCALING_NSTEPS = (2, 8)
TARGETS = (2e-3, 2e-4, 2e-5, 2e-6)
NORMS = ("rms", "max")
ARTIFACTS = ("protocol.json", "config.json", "summary.json", "compact-spatial.json", "summary.txt")
DEFAULTS = {
    "protocol_version": 1, "suite": "compact-spatial", "max_seconds": 1200,
    "reference_tolerance": 1e-8, "reference_attempts": 5,
    "intraop_threads": 1, "interop_threads": 1, "warmup_rollouts": 1,
    "timing_repeats": 5, "nsteps": list(NSTEPS), "scaling_nsteps": list(SCALING_NSTEPS),
    "error_targets": list(TARGETS), "methods": list(METHODS),
}


def validate_config(raw, *, smoke=False):
    if not isinstance(raw, dict) or set(raw) != set(DEFAULTS):
        raise ValueError("Compact spatial requires exactly the frozen protocol fields")
    for key, expected in DEFAULTS.items():
        if type(raw[key]) is not type(expected) or digest(raw[key]) != digest(expected):
            raise ValueError(f"Compact spatial fixes {key}={expected!r}; declare a new protocol to change it")
    return {**json.loads(json.dumps(raw)), "smoke": bool(smoke), "max_seconds": 240 if smoke else 1200}


def make_protocol(config, plans, software, command):
    if set(plans) != {"cases", "expected_ids", "timing_orders", "parity"}:
        raise ValueError("Freeze cases, row identities, timing orders and diagnostics before execution")
    cases = plans["cases"]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be nonempty and unique")
    if set(plans["expected_ids"]) != set(TABLE_IDS):
        raise ValueError("Every canonical table needs predeclared identities")
    for name, ids in plans["expected_ids"].items():
        if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError(f"Invalid row identities for {name}")
    result = {
        "version": 1, "benchmark_suite": "compact-spatial", "config": config,
        "smoke": config["smoke"], "plans": plans, "case_plan_sha256": digest(cases),
        "software": software, "command": command, "device": "cpu", "dtype": "float64",
        "training_attempted": False, "training_performed": False,
        "scope": "Frozen CPU compact spatial accuracy and grid/batch cost diagnostic; no training or adaptive controller",
        "hypotheses": [
            {"id": "H1", "question": "Does stacking GL3 quadrature operations preserve its outputs while reducing complete-rollout runtime?"},
            {"id": "H2", "question": "Do fixed input-mode cutoffs preserve enough spatial correction to reduce cost at matched error?"},
            {"id": "H3", "question": "Does the exact GL3 raw mean help the compact K=4 spatial correction compared with the zero-mean-only ablation?"},
            {"id": "H4", "question": "When do high-frequency pairs, amplitude and retained historical controls defeat compression?"},
            {"id": "H5", "question": "How do full-rollout cost, transformed cells and matched accuracy vary with grid and batch size?"},
        ],
        "interpretation": [
            "The 24 review-bank cases are reused evidence, not fresh discoveries; the 16 new stress cases are deterministic diagnostics, not independent statistical replicates.",
            "Accuracy and scaling panels have separate denominators; batch members and resolutions are paired and must not be counted as independent evidence.",
            "All methods use the same FP64 states, physical parameters, input data, reference and complete rollout schedule within a comparison.",
            "Batch feasibility requires every member: RMS and maximum frontiers use the maximum of each member's error plus its own accepted reference uncertainty.",
            "References solve the same spatially discrete PDE with refined FP64 RK4; sqrt(N)-scaled maximum uncertainty is a refinement estimate, not a certificate or continuum error.",
            "Five bounded reference attempts preserve the previous precision criterion; old rejected records are never rewritten.",
            "GL3 mean is the raw quadratic correction mean, not exact nonlinear mean evolution or mass conservation.",
            "Compact modes are chosen before execution and never selected using the reference; missing high-frequency interactions may feed low output modes.",
            "Approximation-to-GL3 diagnostics are distinct from PDE endpoint accuracy; OBSERVED approximation differences are not failed identities.",
            "One complete warmup precedes five rotated timing repetitions; preparation is measured once separately, and its variance is unknown.",
            "Setup-inclusive cost is preparation plus warmed median, not cold-cache timing; neither cost includes teacher construction, parity or post-hoc selection.",
            "FFT dispatch counts differ from batch-inclusive transformed fields and cells; coefficient storage is not peak process memory.",
            "Frontiers are reference-informed post-hoc fixed-step-grid selections, not a deployable adaptive policy.",
            "All invalid and infeasible candidates remain visible; no clipping, reference-informed corrections, neural training, FNO advantage or GPU performance is claimed.",
        ],
    }
    digest(result)
    return result


def validate_rows(tables, plans, *, complete):
    _validate_work_rows(tables, plans, complete=complete)
    specs = {spec["case_id"]: spec for spec in plans["cases"]}
    for name, rows in tables.items():
        for row in rows:
            spec = specs.get(row["case_id"])
            if spec is None:
                raise ValueError("Unknown case")
            for field in ("panel", "batch_size", "cells"):
                if row.get(field) != spec[field]:
                    raise ValueError(f"Incorrect {field} in {name}")
            if name == "candidate_rows":
                if row["variant"] not in METHODS or row["nsteps"] not in spec["nsteps"]:
                    raise ValueError("Undeclared method or step count")
                members = row.get("member_errors", [])
                if row["trajectory_completed"] and row["finite"]:
                    if [m["member_index"] for m in members] != list(range(spec["batch_size"])):
                        raise ValueError("Complete endpoint needs every member's error")
                    for metric in ("error_rms", "error_max", "error_spatial_rms", "error_spatial_max",
                                   "error_upper_rms", "error_upper_max_estimate"):
                        values = [m.get(metric) for m in members]
                        expected = max(values) if all(v is not None for v in values) else None
                        if row.get(metric) != expected:
                            raise ValueError(f"Aggregate {metric} must preserve the worst member")
                elif members or any(row.get(k) is not None for k in ("error_spatial_rms", "error_spatial_max", "error_mean_abs")):
                    raise ValueError("Partial trajectories cannot have spatial endpoint errors")
                if row["status"] == "VALID":
                    if row.get("throughput_members_per_second") != spec["batch_size"] / row["prepared_median_seconds"]:
                        raise ValueError("Throughput must use complete measured batch runtime")
            elif name == "reference_rows":
                members = row.get("member_references", [])
                if len(members) > spec["batch_size"] or ((complete or row["reference_accepted"]) and len(members) != spec["batch_size"]):
                    raise ValueError("Reference evidence must include each batch member")
                accepted = len(members) == spec["batch_size"] and all(m["reference_accepted"] for m in members)
                if row["reference_accepted"] != accepted:
                    raise ValueError("Reference acceptance requires every member")
            elif name == "parity_rows":
                if row["check"] == "approximation":
                    if row["status"] != "OBSERVED" or row.get("passed") is not None:
                        raise ValueError("Approximation diagnostics are observations, not identities")
                elif row["status"] not in {"PASS", "FAIL", "UNAVAILABLE"}:
                    raise ValueError("Unknown identity outcome")
    return tables
