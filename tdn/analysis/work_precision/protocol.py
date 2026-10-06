"""JSON-only declarations for the bounded, post-hoc work–precision screen."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

METHODS = ("strang", "etdrk2", "etdrk4", "gl3", "gl5", "gl3_mean_full", "gl3_mean_spectral")
NSTEPS = (1, 2, 4, 8, 16, 32)
TARGETS = (2e-3, 2e-4, 2e-5, 2e-6)
NORMS = ("rms", "max")
ARTIFACTS = ("protocol.json", "config.json", "summary.json", "work-precision.json", "summary.txt")
DEFAULTS = {"protocol_version": 1, "suite": "work-precision", "max_seconds": 1200,
            "reference_tolerance": 1e-8, "reference_attempts": 3,
            "intraop_threads": 1, "interop_threads": 1, "warmup_rollouts": 1,
            "timing_repeats": 5, "nsteps": list(NSTEPS), "error_targets": list(TARGETS),
            "methods": list(METHODS)}
TABLE_IDS = {"candidate_rows": "candidate_id", "frontier_rows": "frontier_id",
             "reference_rows": "reference_id", "parity_rows": "parity_id"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(raw, *, smoke=False):
    if not isinstance(raw, dict) or set(raw) != set(DEFAULTS):
        raise ValueError("Work–precision requires exactly the frozen protocol fields")
    for key, expected in DEFAULTS.items():
        # JSON digests distinguish integer, float and boolean list elements too.
        if type(raw[key]) is not type(expected) or digest(raw[key]) != digest(expected):
            raise ValueError(f"Work–precision fixes {key}={expected!r}; declare a new protocol to change it")
    return {**json.loads(json.dumps(raw)), "smoke": bool(smoke), "max_seconds": 180 if smoke else 1200}


def candidate_id(case_id, variant, nsteps):
    return f"{case_id}/{variant}/n{nsteps}"


def frontier_id(case_id, variant, norm, tolerance):
    return f"{case_id}/{variant}/{norm}/t{tolerance:.0e}"


def make_protocol(config, plans, software, command):
    if set(plans) != {"cases", "expected_ids", "timing_orders", "parity"}:
        raise ValueError("Cases, timing orders and parity must be frozen before execution")
    cases = plans["cases"]
    case_ids = [case["case_id"] for case in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("Work–precision case identifiers must be nonempty and unique")
    expected = plans["expected_ids"]
    if set(expected) != set(TABLE_IDS):
        raise ValueError("Every canonical table needs predeclared row identifiers")
    for name, identifiers in expected.items():
        if (not identifiers or len(identifiers) != len(set(identifiers)) or
                any(not isinstance(value, str) or not value for value in identifiers)):
            raise ValueError(f"Invalid predeclared identifiers in {name}")
    result = {"version": 1, "benchmark_suite": "work-precision", "config": config,
              "smoke": config["smoke"], "plans": plans, "case_plan_sha256": digest(cases),
              "software": software, "command": command, "device": "cpu",
              "dtype": "float64", "training_attempted": False, "training_performed": False,
              "scope": "Fixed-horizon, prepared CPU work–precision diagnostic; no training or adaptive controller",
              "interpretation": [
                  "Candidate selection is post hoc over the declared step grid using the reference; it is not a deployable adaptive policy.",
                  "RMS and maximum error define separate frontiers; feasibility uses error plus accepted refinement uncertainty.",
                  "sqrt(N) times the RMS reference uncertainty is a conservative maximum-norm estimate, not a rigorous certificate.",
                  "The coupled FP64 reference has the same spatial grid; no continuum accuracy claim follows.",
                  "All methods have prepared coefficients, one complete warmup, and five interleaved complete rollout measurements.",
                  "Setup-inclusive time is preparation plus median warmed rollout; it excludes reference, warmup, parity and post-hoc selection.",
                  "Preparation is one timed sample in a predeclared rotating order; its sampling noise is not estimated.",
                  "The bank is a partial stress design with coupled factors, not a full factorial or independent statistical sample.",
                  "Historical controls retain the earlier 2D and high-rate weaknesses; every invalid candidate and unattained target is retained.",
                  "This diagnostic establishes neither neural/FNO benefit nor GPU performance nor a population-wide ranking."]}
    digest(result)
    return result


def validate_rows(tables, plans, *, complete):
    """Reject nonfinite JSON, duplicates, undeclared rows and partial final errors."""
    if set(tables) != set(TABLE_IDS):
        raise ValueError("All four canonical row tables are required")
    for name, identity in TABLE_IDS.items():
        expected = set(plans["expected_ids"][name])
        seen = set()
        for row in tables[name]:
            key = row.get(identity)
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or repeated {identity}: {key}")
            seen.add(key)
            if name == "candidate_rows":
                if row["status"] not in {"VALID", "INVALID", "INCOMPLETE", "FAILED"}:
                    raise ValueError("Unknown candidate status")
                if not row["trajectory_completed"] and any(row.get(field) is not None for field in
                        ("error_rms", "error_max", "error_mean", "error_upper_rms", "error_upper_max_estimate")):
                    raise ValueError("Partial trajectories cannot have final-horizon errors")
                if row["status"] == "VALID":
                    warmup, samples = row.get("warmup"), row.get("timing_repeats", [])
                    if (not isinstance(warmup, dict) or len(samples) != DEFAULTS["timing_repeats"] or
                            [sample["repeat_index"] for sample in samples] != list(range(DEFAULTS["timing_repeats"]))):
                        raise ValueError("Valid candidates require one warmup and five ordered measurements")
                    if any(sample["status"] != "VALID" or not sample["trajectory_completed"] or
                           sample["completed_steps"] != row["nsteps"] or sample["seconds"] <= 0
                           for sample in [warmup, *samples]):
                        raise ValueError("Valid candidates require complete valid full-rollout timings")
                    if (row.get("preparation_seconds", -1) < 0 or
                            row.get("prepared_median_seconds") != statistics.median(sample["seconds"] for sample in samples) or
                            row.get("setup_inclusive_median_seconds") != row["preparation_seconds"] + row["prepared_median_seconds"]):
                        raise ValueError("Candidate setup and rollout timing summaries disagree")
            elif name == "frontier_rows":
                if row["status"] not in {"FEASIBLE", "NO_FEASIBLE_CANDIDATE", "INCONCLUSIVE"}:
                    raise ValueError("Unknown frontier status")
                if row["status"] == "FEASIBLE" and (row["selected_candidate_id"] is None or not row["adjusted_error"] <= row["tolerance"]):
                    raise ValueError("Feasible frontiers need a selected candidate within tolerance")
        if complete and seen != expected:
            raise ValueError(f"Missing declared {name}")
    digest(tables)
    candidates = {row["candidate_id"]: row for row in tables["candidate_rows"]}
    references = {row["case_id"]: row for row in tables["reference_rows"]}
    for frontier in tables["frontier_rows"]:
        if frontier["status"] != "FEASIBLE":
            continue
        reference = references.get(frontier["case_id"])
        if not reference or not reference["reference_accepted"]:
            raise ValueError("Feasible frontiers require an accepted same-case reference")
        error_key = "error_rms" if frontier["norm"] == "rms" else "error_max"
        upper_key = "error_upper_rms" if frontier["norm"] == "rms" else "error_upper_max_estimate"
        for prefix in ("", "setup_"):
            candidate = candidates.get(frontier[f"{prefix}selected_candidate_id"])
            if (not candidate or candidate["status"] != "VALID" or not candidate["reference_accepted"] or
                    candidate["case_id"] != frontier["case_id"] or candidate["variant"] != frontier["variant"] or
                    candidate["nsteps"] != frontier[f"{prefix}selected_nsteps"]):
                raise ValueError("Frontier selection must identify a same-case/method valid candidate")
            adjusted = frontier["adjusted_error"] if not prefix else frontier["setup_selected_adjusted_error"]
            cost = frontier["prepared_seconds"] if not prefix else frontier["setup_selected_seconds"]
            candidate_cost = candidate["prepared_median_seconds"] if not prefix else candidate["setup_inclusive_median_seconds"]
            if (frontier[f"{prefix}selected_error"] != candidate[error_key] or adjusted != candidate[upper_key] or
                    adjusted > frontier["tolerance"] or cost != candidate_cost):
                raise ValueError("Frontier error and timing must match its selected candidate")
    return tables
