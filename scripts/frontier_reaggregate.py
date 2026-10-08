#!/usr/bin/env python3
"""Recompute historical roadmap frontiers at identical final simulation times.

Reads sealed JSON and hashes files only: no checkpoint deserialization, archived
Python imports or numerical reruns. Writes a fresh derived directory and never
updates an original run, its scores, its manifest, or its Tower reports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdn.analysis.frontier.measurement import endpoint_eligibility, final_time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)


def verify_inputs(stage, expected_source=None):
    manifest = read_json(stage / "science_manifest.json")
    if manifest.get("schema") != "tdn.roadmap-science-manifest/v1" or manifest.get("stage") != "confirm":
        raise ValueError("Expected a sealed roadmap confirmation stage")
    if (stage / "COMPLETED").read_text().strip() != digest(manifest):
        raise ValueError("Confirmation completion marker differs from science manifest")
    actual = {}
    for name, expected in manifest["artifacts"].items():
        relative = Path(name)
        path = stage / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to(stage.resolve())
            or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(stage))):
            raise ValueError("Unsafe archived scientific artifact path")
        actual[name] = file_digest(path)
        if actual[name] != expected:
            raise ValueError(f"Archived artifact hash mismatch: {name}")
    for required in ("protocol.json", "confirmation_rows.json", "execution.json", "summary.json"):
        if required not in actual:
            raise ValueError(f"Missing sealed input: {required}")
    protocol = read_json(stage / "protocol.json")
    execution = read_json(stage / "execution.json")
    summary = read_json(stage / "summary.json")
    source = manifest.get("source_tree_sha256")
    if (not isinstance(source, str) or len(source) != 64
        or execution.get("software", {}).get("source_tree_sha256") != source
        or summary.get("source_tree_sha256") != source
        or summary.get("status") != "COMPLETED"
        or summary.get("stage") != "confirm"
        or execution.get("stage") != "confirm"
        or digest(protocol) != manifest["protocol_sha256"]
        or summary.get("protocol_sha256") != manifest["protocol_sha256"]
        or execution.get("protocol_sha256") != manifest["protocol_sha256"]):
        raise ValueError("Archived source/protocol lineage does not agree")
    if expected_source is not None and source != expected_source:
        raise ValueError("Archived source differs from the requested source fingerprint")
    return protocol, {"source_tree_sha256": source, "protocol_sha256": digest(protocol),
        "science_manifest_sha256": file_digest(stage / "science_manifest.json"), "verified_files": actual,
        "execution_mode": execution.get("execution_mode"), "device": execution.get("device"),
        "verification_scope": "sealed archive self-consistency and optional expected source; not a fresh source-tree execution"}


def recompute(rows, protocol):
    schedules = {f"schedule-{i}": v for i, v in enumerate(protocol["confirm_schedules"])}
    cells, models = defaultdict(list), {}
    for original in rows:
        row = dict(original)
        if row["schedule_id"] not in schedules:
            raise ValueError("Endpoint refers to an undeclared schedule")
        row["schedule"] = schedules[row["schedule_id"]]
        row["final_time"] = final_time(row)
        cells[(row["parent_id"], row["grid"], row["track"], row["final_time"])].append(row)
        info = (row["family"], row.get("seed"))
        if row["model_id"] in models and models[row["model_id"]] != info:
            raise ValueError("Model identity changes family or seed")
        models[row["model_id"]] = info
    targets = protocol["targets"]
    pairs = [(v[0], v[1]) if isinstance(v, (list, tuple)) else (v, v) for v in targets]
    classical = {"df_base", "rf_base", "etdrk4", "gl3_fused"}
    output = []
    for (parent, grid, track, horizon), members in cells.items():
        for rt, mt in pairs:
            def best(items):
                eligible = [r for r in items if endpoint_eligibility(r, rt, mt) == "ELIGIBLE"]
                return min(eligible, key=lambda r: r["cost_seconds"]) if eligible else None
            base = best([r for r in members if r["family"] in classical])
            for mid, (family, seed) in models.items():
                if family in classical or family == "df_quad2":
                    continue
                own = [r for r in members if r["model_id"] == mid]
                candidate = best(own)
                controls = {"best_classical": base}
                for control_family in ("cheap_fno", "deep_fno", "direct_fno"):
                    controls[control_family] = best([r for r in members if r["family"] == control_family and r.get("seed") == seed])
                output.append({"model_id": mid, "family": family, "seed": seed,
                    "parent_id": parent, "grid": grid, "track": track, "final_time": horizon,
                    "rms_target": rt, "max_target": mt, "candidate_feasible": candidate is not None,
                    "candidate_schedule": candidate["schedule_id"] if candidate else None,
                    "candidate_cost_seconds": candidate["cost_seconds"] if candidate else None,
                    "candidate_statuses": dict(Counter(endpoint_eligibility(r, rt, mt) for r in own)) or {"MISSING_ARM": 1},
                    "control_feasible": {k: v is not None for k, v in controls.items()},
                    "control_schedule": {k: v["schedule_id"] if v else None for k, v in controls.items()},
                    "control_model": {k: v["model_id"] if v else None for k, v in controls.items()},
                    "control_over_candidate_speed_ratio": {k: v["cost_seconds"] / candidate["cost_seconds"] if v and candidate else None for k, v in controls.items()},
                    "selection_scope": "fixed-final-time reference-informed post-hoc schedule frontier; not deployment",
                    "timing_uncertainty": "historical timing samples unchanged; ratios alone do not establish a robust speed advantage"})
    return {"comparisons": output, "source_endpoint_rows": len(rows), "comparison_rows": len(output),
        "final_times": sorted({key[-1] for key in cells}),
        "failed_endpoint_rows": sum(bool(r.get("numerical_failure")) for r in rows),
        "unaccepted_reference_rows": sum(r.get("reference_accepted") is not True for r in rows),
        "paper_reproduction": False, "frontier_is_deployment_policy": False,
        "scientific_scope": "derived historical development evidence; unchanged errors/timings; matching repaired, not new confirmation"}


def run(run_dir, output_dir, expected_source=None):
    source = Path(run_dir).resolve()
    stage = source if source.name == "confirm" else source / "confirm"
    output = Path(output_dir).resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("All project inputs and outputs must remain under this checkout")
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a fresh output directory outside the immutable source run")
    protocol, provenance = verify_inputs(stage, expected_source)
    records = read_json(stage / "confirmation_rows.json")["rows"]
    result = recompute(records, protocol)
    # Detect source mutation between the first integrity check and publication.
    _, checked = verify_inputs(stage, expected_source)
    if checked != provenance:
        raise ValueError("Source changed during reaggregation")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "fixed-horizon-comparisons.json"
    with target.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    ledger = {"schema": "tdn.frontier-reaggregation/v1", "status": "COMPLETED", "source": str(source),
        "input_provenance": provenance, "output_sha256": {target.name: file_digest(target)},
        "script_sha256": file_digest(Path(__file__)), "comparison_implementation_sha256": file_digest(ROOT / "tdn/analysis/frontier/measurement.py"),
        "original_artifacts_modified": False, "checkpoint_deserialization": False,
        "archived_code_executed": False, "numerical_reruns": False}
    with (output / "reaggregation-manifest.json").open("x") as handle:
        json.dump(ledger, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {"status": "COMPLETED", "report": str(target), "comparison_rows": result["comparison_rows"],
            "source_endpoint_rows": result["source_endpoint_rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    try:
        result = run(args.run_dir, args.output_dir, args.expected_source_sha256)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"TDN reaggregate: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
