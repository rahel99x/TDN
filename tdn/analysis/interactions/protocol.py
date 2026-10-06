"""Frozen declarations and validation independent of scientific imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PANELS = ("controls", "one_step", "rollout")
ARTIFACTS = ("protocol.json", "config.json", "summary.json", "interaction-screen.json", "summary.txt")
DEFAULTS = {"protocol_version": 1, "suite": "interaction-screen", "max_seconds": 1200,
            "reference_tolerance": 1e-8, "reference_attempts": 3,
            "intraop_threads": 1, "interop_threads": 1, "panels": list(PANELS)}
KINDS = {"correctness", "representation", "scientific", "negative_control"}
OUTCOMES = {"PASS", "FAIL", "OBSERVED", "INCONCLUSIVE", "EXPECTED_LIMITATION"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(raw, *, smoke=False):
    if not isinstance(raw, dict) or set(raw) != set(DEFAULTS):
        raise ValueError("Interaction screen requires exactly the frozen protocol fields")
    for key, expected in DEFAULTS.items():
        if type(raw[key]) is not type(expected) or raw[key] != expected:
            raise ValueError(f"Interaction screen fixes {key}={expected!r}; declare a new protocol to change it")
    return {**raw, "panels": list(PANELS), "smoke": bool(smoke),
            "max_seconds": 180 if smoke else 1200}


def make_protocol(config, plans, software, command):
    if set(plans) != set(PANELS):
        raise ValueError("All interaction panels must be declared before execution")
    all_ids = []
    for name, plan in plans.items():
        ids = plan.get("expected_case_ids")
        if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                or any(not isinstance(case, str) or not case for case in ids)):
            raise ValueError(f"Invalid predeclared identifiers: {name}")
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Interaction case identifiers must be globally unique")
    result = {"version": 1, "benchmark_suite": "interaction-screen", "config": config,
              "smoke": config["smoke"], "plans": plans, "software": software,
              "command": command, "device": "cpu", "training_attempted": False,
              "training_performed": False,
              "scope": "Fixed finite-amplitude same-grid CPU screen; no fitted or learned models",
              "interpretation": [
                  "The input field alone determines every runtime correction; generating modes are diagnostics only.",
                  "Quadratic response is truncated at finite amplitude; improvements and regressions are retained.",
                  "References are refined FP64 coupled solves on the same grid, not continuum convergence evidence.",
                  "Unaccepted or uncertainty-unresolved references are inconclusive, never scientific passes.",
                  "Full step and rollout timing includes corrections; work counts describe each implementation.",
                  "Single CPU timings do not certify matched cost to tolerance, learned superiority, or out-of-distribution generalization.",
                  "Computational completion and scientific outcome are separate; no automatic training or GPU follow-up."]}
    digest(result)
    return result


def validate_rows(rows, plans, *, complete):
    expected = {case: name for name, plan in plans.items() for case in plan["expected_case_ids"]}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Interaction rows must be objects")
        for key in ("case_id", "panel", "mechanism", "test", "variant"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Interaction rows require nonempty {key}")
        case = row["case_id"]
        if case in seen or case not in expected or row["panel"] != expected[case]:
            raise ValueError("Interaction rows differ from predeclared case coverage")
        seen.add(case)
        if row.get("kind") not in KINDS or row.get("outcome") not in OUTCOMES:
            raise ValueError("Unknown interaction kind or outcome")
        if not isinstance(row.get("metrics"), dict) or not isinstance(row.get("inputs"), dict):
            raise ValueError("Interaction metrics and inputs must be objects")
        if not isinstance(row.get("note"), str):
            raise ValueError("Interaction notes must be strings")
        if any(not isinstance(key, str) or not key or type(value) not in (str, float, int, bool, type(None))
               for key, value in row["metrics"].items()):
            raise ValueError("Interaction metrics must be named JSON scalars")
    if complete and seen != set(expected):
        raise ValueError("Interaction screen is missing predeclared cases")
    digest(rows)
    return rows
