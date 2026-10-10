"""Advance-specific, exact-source reference journals and split access controls."""
from __future__ import annotations
import json
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tdn.analysis.frontier.data import (_prepared_parents, _write_parent, _safe_file,
    validate_cohorts, field_state, reference_horizons, _parent_grids)
from tdn.research.experiment import state_digest, horizon_key
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json, software_metadata

SCHEMA = "tdn.advance-data/v1"


def teacher_protocol(protocol):
    """Prepare requested endpoints only; a schedule menu is not an output list.

    This explicit projection is local to advance. The historical generator and
    historical protocols remain unchanged. Rollout path diagnostics require a
    separate declared teacher request, not unused intermediate truth.
    """
    result = copy.deepcopy(protocol)
    result["confirm_schedules"] = [[float(h)] for h in protocol["evaluation_horizons"]]
    return result


def binding(ctx):
    metadata = software_metadata()
    stable = {k: metadata[k] for k in ("python", "torch", "numpy", "scipy", "torch_cuda_runtime")}
    return dict(protocol_sha256=digest(ctx.protocol), source_tree_sha256=metadata["source_tree_sha256"],
                software_sha256=digest(stable))


def unit(ctx, stage=None):
    stage = stage or ctx.stage
    units = ctx.protocol["units"]
    return units[stage] if isinstance(units, dict) else next(x for x in units if x["id"] == stage)


def declarations(ctx, stage=None):
    scope = unit(ctx, stage)
    ids = scope.get("parent_ids")
    if ids is None:
        splits = scope.get("splits", ["scaling"] if scope.get("kind") == "scaling_prepare" else ["train", "validation"])
        return [p for p in ctx.protocol["parents"] if p["split"] in splits]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate parent identifiers in preparation scope")
    result = [p for p in ctx.protocol["parents"] if p["parent_id"] in set(ids)]
    if len(result) != len(ids):
        raise ValueError("Unknown prepared parent")
    return result


def prepare(ctx):
    from .learning import verify_frozen, _record
    validate_cohorts(ctx.protocol)
    parents = declarations(ctx)
    fresh = any(p["split"] in ("confirmation", "scaling") for p in parents)
    if fresh and unit(ctx).get("kind") not in ("confirm_prepare", "scaling_prepare"):
        raise ValueError("Fresh references require a frozen preparation stage")
    frozen = verify_frozen(ctx)[1] if fresh else None
    root = Path(ctx.path); root.mkdir(parents=True, exist_ok=True)
    identity = {**binding(ctx), "stage": ctx.stage, "parent_ids": [p["parent_id"] for p in parents], "frozen_training": frozen}
    journal_path = root / "prepare-journal.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else dict(identity=identity, records=[])
    if journal["identity"] != identity:
        raise ValueError("Reference recovery identity changed")
    records = journal["records"]
    if [r["parent_id"] for r in records] != identity["parent_ids"][:len(records)]:
        raise ValueError("Reference recovery parent order changed")
    for record in records:
        for name, sha in record["hashes"].items():
            if file_digest(_safe_file(root, name)) != sha:
                raise ValueError("Previously completed reference bytes changed")
    teacher_ctx = SimpleNamespace(protocol=teacher_protocol(ctx.protocol), budget=ctx.budget)
    for index, parent in enumerate(_prepared_parents(teacher_ctx, parents[len(records):]), len(records)):
        ctx.budget.check()
        parent_root = root / f"parent-{index:05d}"; attempt = 0
        while (parent_root / f"attempt-{attempt:03d}").exists():
            attempt += 1
        target = parent_root / f"attempt-{attempt:03d}"; target.mkdir(parents=True)
        record = _write_parent(target, index, parent)
        for key in ("metadata", "arrays"):
            record[key] = str((target / record[key]).relative_to(root))
        record["hashes"] = {record[k]: file_digest(root / record[k]) for k in ("metadata", "arrays")}
        records.append(record); write_json(journal_path, journal)
    references = [ref for record in records for ref in json.loads((root / record["metadata"]).read_text())["references"].values()]
    accepted = sum(bool(r["accepted"]) for r in references)
    write_json(root / "data_manifest.json", dict(schema=SCHEMA, **identity, records=records,
        teacher_request_sha256=digest(teacher_ctx.protocol),
        accepted_references=accepted, unresolved_references=len(references)-accepted, uncertainty_is_certificate=False))
    _record(ctx, "reference/preparation", dict(parents=len(records), references=len(references),
        accepted_references=accepted, unresolved_references=len(references)-accepted,
        uncertainty_is_certificate=False, frozen_before_reference_generation=fresh), good=accepted == len(references))
    return dict(parents=len(records), accepted_references=accepted, unresolved_references=len(references)-accepted,
                training_labels_ready=accepted == len(references), frozen_before_reference_generation=fresh)


def load_bank(ctx, stage):
    root = Path(ctx.path if ctx.stage == stage else ctx.prerequisites[stage])
    manifest = json.loads(_safe_file(root, "data_manifest.json").read_text())
    projected = teacher_protocol(ctx.protocol)
    if manifest.get("schema") != SCHEMA or manifest.get("stage") != stage or manifest.get("teacher_request_sha256") != digest(projected) or any(manifest.get(k) != v for k, v in binding(ctx).items()):
        raise ValueError("Advance reference bank identity changed")
    parents = declarations(ctx, stage)
    expected_ids = [p["parent_id"] for p in parents]
    if manifest["parent_ids"] != expected_ids or [r["parent_id"] for r in manifest["records"]] != expected_ids:
        raise ValueError("Reference bank parent coverage changed")
    fresh = any(p["split"] in ("confirmation", "scaling") for p in parents)
    if fresh:
        if unit(ctx).get("kind") not in ("confirm_prepare", "confirm", "aggregate", "scaling_prepare", "scaling", "report"):
            raise ValueError("Fresh reference access forbidden during development/selection")
        from .learning import verify_frozen
        if manifest["frozen_training"] != verify_frozen(ctx)[1]:
            raise ValueError("Reference bank was not generated for the frozen models")
    result = []
    for declared, record in zip(parents, manifest["records"]):
        if set(record["hashes"]) != {record["metadata"], record["arrays"]}:
            raise ValueError("Reference artifact hash coverage changed")
        for name, sha in record["hashes"].items():
            if file_digest(_safe_file(root, name)) != sha:
                raise ValueError("Reference artifact changed")
        parent = json.loads((root / record["metadata"]).read_text())
        if any(parent.get(k) != v for k, v in declared.items()):
            raise ValueError("Reference field declaration changed")
        with np.load(root / record["arrays"], allow_pickle=False) as arrays:
            parent["states"] = {n: torch.from_numpy(arrays[key].copy()) for n, key in parent["states"].items()}
            for ref in parent["references"].values():
                ref["state"] = torch.from_numpy(arrays[ref.pop("array")].copy())
        grids = _parent_grids(ctx.protocol, declared["split"])
        keys = {f"{track}:{n}:{horizon_key(h)}" for track in ctx.protocol["tracks"] for n in grids
                for h in reference_horizons(projected, declared["split"])}
        if set(parent["states"]) != set(map(str, grids)) or set(parent["references"]) != keys:
            raise ValueError("Reference grid/horizon coverage changed")
        for n, state in parent["states"].items():
            if state.dtype != torch.float64 or state_digest(state) != state_digest(field_state(declared, int(n))):
                raise ValueError("Reference initial state changed")
        for ref in parent["references"].values():
            if ref["state"].dtype != torch.float64 or state_digest(ref["state"]) != ref["state_sha256"]:
                raise ValueError("Reference endpoint changed")
        result.append(parent)
    return result
