"""Declared consistency ablations with failure-preserving paired endpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from tdn.analysis.premix.comparison import compare as _compare, _readable
from tdn.runtime.metadata import write_json

SCHEMA = "tdn.consistency-comparisons/v1"
SOURCE_SCHEMA = "tdn.consistency-neural/v1"
FAMILIES = ("premix", "fno", "premix_gated", "fno_gated", "premix_moment", "fno_moment", "precompress")
# Positive endpoints always favor the first member. Each pair has a concrete
# interpretation; architecture and constraint effects are not pooled together.
PAIR_PLAN = (
    ("premix", "fno", "matched_current"),
    ("premix_gated", "fno_gated", "matched_gate"),
    ("premix_moment", "fno_moment", "matched_moment"),
    ("premix_gated", "premix", "gate_effect_premix"),
    ("fno_gated", "fno", "gate_effect_fno"),
    ("premix_moment", "premix", "moment_total_effect_premix"),
    ("fno_moment", "fno", "moment_total_effect_fno"),
    ("premix_moment", "premix_gated", "moment_increment_premix"),
    ("fno_moment", "fno_gated", "moment_increment_fno"),
    ("premix", "precompress", "compression_order"),
)
DIAGNOSTIC_FIELDS = ("rms", "max_error", "signed_mean_error", "mean_error", "spatial_rms",
    "base_rms", "base_max_error", "base_mean_error", "base_spatial_rms",
    "rms_vs_base", "max_vs_base", "mean_vs_base", "spatial_vs_base")
SOURCE_FILES = ("protocol.json", "training.json", "candidates.json", "frontiers.json")


def compare(protocol, candidates, frontiers, training):
    """Retain every declared parent/seed/pair/norm/target, including failures."""
    result = _compare(protocol, candidates, frontiers, training, source_schema=SOURCE_SCHEMA,
        families=FAMILIES, pairs=[pair[:2] for pair in PAIR_PLAN], schema=SCHEMA)
    pair_kinds = {(first, second): kind for first, second, kind in PAIR_PLAN}
    for row in result["rows"]:
        row["comparison_kind"] = pair_kinds[row["ours"], row["baseline"]]
        for side in ("ours", "baseline"):
            pointer = row[side + "_candidate_record"]
            source = candidates[int(pointer.rsplit("/", 1)[1])] if pointer else {}
            issues = []
            for field in DIAGNOSTIC_FIELDS:
                value = source.get(field)
                valid = value is None or (type(value) in (int, float) and math.isfinite(value)
                                          and (field == "signed_mean_error" or value >= 0))
                row[side + "_" + field] = value if valid else None
                if not valid:
                    issues.append("INVALID_" + field.upper())
            row[side + "_diagnostic_issues"] = issues
            row[side + "_rms_regresses_base"] = (
                row[side + "_rms"] > row[side + "_base_rms"]
                if row[side + "_rms"] is not None and row[side + "_base_rms"] is not None else None)
    for group in result["groups"]:
        group["comparison_kind"] = pair_kinds[group["ours"], group["baseline"]]
    result["pair_plan"] = [{"ours": first, "baseline": second, "comparison_kind": kind}
                            for first, second, kind in PAIR_PLAN]
    result["scope"] = ("Bounded consistency ablations on fresh diagnostic parents; post-hoc reference-informed "
                       "fixed-step frontiers, not a deployable adaptive policy or FNO-paper reproduction")
    result["interpretation"].update(
        pair_plan="Matched-constraint backbone comparisons, within-backbone ablations and compression order are separate endpoints",
        moment="Moment arms include the gate plus learned target-mean redistribution; moment versus gated isolates its increment",
        diagnostics="Mean/spatial/base errors refer to the selected feasible candidate; null means no selected candidate or absent diagnostic",
        regression="rms_regresses_base compares measured RMS against the same-parent same-step Strang physical base; uncertainty remains in source candidates",
        ratios="*_vs_base are descriptive engine-recorded ratios; a null ratio may reflect an unresolved or zero base error, never an automatic improvement",
    )
    return result


def _decode(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate comparison source key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite comparison source value: " + value)
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite comparison source number")
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def summarize(run_dir: Path):
    """Write only derived comparisons before the coordinator seals the stage."""
    root = Path(run_dir)
    if root.is_symlink() or (root / "COMPLETED").exists() or (root / "manifest.json").exists():
        raise ValueError("Preserve completed/sealed consistency stages")
    documents, sources = {}, []
    for name in SOURCE_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError("Comparison sources cannot be symlinks")
        if not path.exists():
            if name == "protocol.json":
                raise ValueError("A declared protocol is required")
            documents[name] = {"schema": SOURCE_SCHEMA, "rows": []}
            sources.append({"path": name, "status": "MISSING", "sha256": None})
            continue
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("Comparison source exceeds bounded JSON size")
        raw = path.read_bytes()
        document = _decode(raw)
        if not isinstance(document, dict) or document.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"Unsupported consistency comparison source: {name}")
        documents[name] = document
        sources.append({"path": name, "status": "READ", "sha256": hashlib.sha256(raw).hexdigest()})
    result = compare(documents["protocol.json"], documents["candidates.json"].get("rows"),
                     documents["frontiers.json"].get("rows"), documents["training.json"].get("rows"))
    result["sources"] = sources
    write_json(root / "comparisons.json", result)
    temporary = root / "summary.txt.partial"
    temporary.write_text(_readable(result).replace("TDN premix paired neural review", "TDN consistency paired neural review"))
    os.replace(temporary, root / "summary.txt")
    return result["counts"]


__all__ = ["compare", "summarize", "SCHEMA", "PAIR_PLAN"]
