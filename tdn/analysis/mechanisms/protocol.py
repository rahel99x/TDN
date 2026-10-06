"""Standard-library protocol validation and artifact fingerprints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PANELS = ("coordinates", "temporal", "structure")
ARTIFACTS = ("protocol.json", "config.json", "summary.json", "mechanism-audit.json",
             "coordinates.json", "temporal.json", "structure.json")
DEFAULTS = {"protocol_version": 1, "suite": "mechanism-audit", "max_seconds": 1200,
            "tolerance": .002, "intraop_threads": 1, "interop_threads": 1,
            "panels": list(PANELS)}
KINDS = {"correctness", "representation", "scientific", "negative_control"}
OUTCOMES = {"PASS", "FAIL", "OBSERVED", "INCONCLUSIVE", "EXPECTED_LIMITATION"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(raw, *, smoke=False):
    if not isinstance(raw, dict) or set(raw) != set(DEFAULTS):
        raise ValueError("Mechanism audit requires exactly the frozen protocol fields")
    for key, expected in DEFAULTS.items():
        value = raw[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Mechanism audit fixes {key}={expected!r}; declare a new protocol to change it")
    return {**raw, "panels": list(PANELS), "smoke": bool(smoke),
            "max_seconds": 180 if smoke else 1200}


def make_protocol(config, plans, software, command):
    if set(plans) != set(PANELS):
        raise ValueError("All three panels must be declared before execution")
    for name in PANELS:
        ids = plans[name].get("expected_case_ids")
        if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                or any(not isinstance(case, str) or not case for case in ids)):
            raise ValueError(f"Invalid predeclared case identifiers: {name}")
    result = {"version": 1, "benchmark_suite": "mechanism-audit", "config": config,
              "smoke": config["smoke"], "plans": plans, "software": software,
              "command": command, "device": "cpu", "training_attempted": False,
              "scope": "Deterministic mechanism diagnostics, not trained neural solver comparisons",
              "interpretation": [
                  "Oracle fits and controlled coefficients are representation probes, not trained accuracy.",
                  "Negative scientific results are retained; failed declared controls prevent completion.",
                  "Budget exhaustion or unaccepted references are inconclusive, never a pass.",
                  "No result authorizes GPU work or establishes superiority to the FNO paper."]}
    digest(result)  # Reject nonfinite or non-JSON declarations before work.
    return result


def validate_panel(result, name, plan):
    if not isinstance(result, dict) or result.get("panel") != name or result.get("status") != "COMPLETED":
        raise ValueError(f"Panel {name} did not return a completed report")
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Panel {name} has no rows")
    identifiers = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Audit rows must be objects")
        for key in ("case_id", "mechanism", "test", "variant"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Audit rows require nonempty {key}")
        if row.get("kind") not in KINDS or row.get("outcome") not in OUTCOMES:
            raise ValueError("Unknown audit kind or outcome")
        if not isinstance(row.get("metrics"), dict) or not isinstance(row.get("inputs"), dict):
            raise ValueError("Audit metrics and inputs must be objects")
        if not isinstance(row.get("note"), str):
            raise ValueError("Audit notes must be strings")
        if any(not isinstance(key, str) or not key or type(value) not in (str, float, int, bool, type(None))
               for key, value in row["metrics"].items()):
            raise ValueError("Audit metrics must be named JSON scalars")
        identifiers.append(row["case_id"])
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(plan["expected_case_ids"]):
        raise ValueError(f"Panel {name} differs from its predeclared case coverage")
    digest(result)
    return result
