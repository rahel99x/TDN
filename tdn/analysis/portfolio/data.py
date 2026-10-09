"""Fresh split-disjoint numeric teachers with recoverable, hash-bound parents."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from tdn.analysis.frontier.data import (generate_parent, _write_parent, _safe_file,
    validate_cohorts, field_state, reference_horizons, _parent_grids, _prepared_parents)
from tdn.research.experiment import state_digest, horizon_key
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json, software_metadata


def binding(ctx):
    meta=software_metadata()
    stable={k:meta[k] for k in ("python","torch","numpy","scipy","torch_cuda_runtime")}
    return {"protocol_sha256": digest(ctx.protocol), "source_tree_sha256": meta["source_tree_sha256"],
            "software_sha256":digest(stable)}


def unit(ctx):
    units = ctx.protocol["units"]
    return units[ctx.stage] if isinstance(units, dict) else next(x for x in units if x["id"] == ctx.stage)


def declarations(ctx):
    scope = unit(ctx)
    if "parent_ids" in scope:
        wanted = set(scope["parent_ids"])
        result = [p for p in ctx.protocol["parents"] if p["parent_id"] in wanted]
        if len(result) != len(wanted):
            raise ValueError("Unknown or duplicate prepared parent declaration")
        return result
    return [p for p in ctx.protocol["parents"] if p["split"] in ("train", "validation")]


def prepare(ctx):
    from .learning import verify_frozen
    validate_cohorts(ctx.protocol)
    parents = declarations(ctx)
    fresh = any(p["split"] in ("confirmation", "scaling") for p in parents)
    if fresh and unit(ctx).get("kind") not in ("confirm_prepare", "scaling"):
        raise ValueError("Fresh reference generation is forbidden before the frozen confirmation stage")
    frozen = verify_frozen(ctx)[1] if fresh else None
    root = Path(ctx.path); root.mkdir(parents=True, exist_ok=True)
    identity = {**binding(ctx), "stage": ctx.stage, "parent_ids": [p["parent_id"] for p in parents],
                "frozen_training": frozen}
    journal_path = root / "prepare-journal.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"identity": identity, "records": []}
    if journal["identity"] != identity:
        raise ValueError("Prepared reference recovery identity differs")
    records = journal["records"]
    if [r["parent_id"] for r in records] != identity["parent_ids"][:len(records)]:
        raise ValueError("Prepared reference journal scope changed")
    for record in records:
        for name, sha in record["hashes"].items():
            if file_digest(_safe_file(root, name)) != sha:
                raise ValueError("Prepared reference journal bytes changed")
    pending = parents[len(records):]
    for index, parent in enumerate(_prepared_parents(ctx, pending), len(records)):
        ctx.budget.check()
        # A crash before the parent journal commit can leave two orphan files.
        # They are never trusted; use a new deterministic attempt suffix directory.
        parent_dir = root / f"parent-{index:05d}"
        attempt = 0
        while (parent_dir / f"attempt-{attempt:03d}").exists(): attempt += 1
        target = parent_dir / f"attempt-{attempt:03d}"; target.mkdir(parents=True)
        record = _write_parent(target, index, parent)
        for key in ("metadata", "arrays"):
            record[key] = str((target / record[key]).relative_to(root))
        record["hashes"] = {record[k]: file_digest(root / record[k]) for k in ("metadata", "arrays")}
        records.append(record); write_json(journal_path, journal)
    accepted = unresolved = 0
    for record in records:
        meta = json.loads((root / record["metadata"]).read_text())
        accepted += sum(bool(r["accepted"]) for r in meta["references"].values())
        unresolved += sum(not r["accepted"] for r in meta["references"].values())
    manifest = {"schema": "tdn.portfolio-data/v1", **identity, "records": records,
                "accepted_references": accepted, "unresolved_references": unresolved,
                "uncertainty_is_certificate": False}
    write_json(root / "data_manifest.json", manifest)
    return {"parents": len(records), "accepted_references": accepted, "unresolved_references": unresolved,
            "training_labels_ready": unresolved == 0, "frozen_before_reference_generation": fresh}


def load_bank(ctx, stage):
    root = Path(ctx.path if ctx.stage == stage else ctx.prerequisites[stage])
    manifest = json.loads(_safe_file(root, "data_manifest.json").read_text())
    if manifest.get("schema") != "tdn.portfolio-data/v1" or any(manifest.get(k) != v for k,v in binding(ctx).items()) or manifest["stage"] != stage:
        raise ValueError("Data bank identity changed")
    units = ctx.protocol["units"]
    scope = units[stage] if isinstance(units, dict) else next(x for x in units if x["id"] == stage)
    wanted = scope.get("parent_ids", [p["parent_id"] for p in ctx.protocol["parents"] if p["split"] in ("train", "validation")])
    parents = [p for p in ctx.protocol["parents"] if p["parent_id"] in wanted]
    if manifest["parent_ids"] != [p["parent_id"] for p in parents] or [r["parent_id"] for r in manifest["records"]] != manifest["parent_ids"]:
        raise ValueError("Data bank missing or duplicate parent")
    fresh = any(p["split"] in ("confirmation", "scaling") for p in parents)
    if fresh:
        if unit(ctx).get("kind") not in ("confirm_prepare", "confirm", "aggregate", "scaling", "report"):
            raise ValueError("Fresh confirmation access forbidden during model selection")
        from .learning import verify_frozen
        if manifest["frozen_training"] != verify_frozen(ctx)[1]:
            raise ValueError("Reference bank was not generated for these frozen models")
    loaded = []
    for declaration, record in zip(parents, manifest["records"]):
        if set(record["hashes"]) != {record["metadata"], record["arrays"]}:
            raise ValueError("Reference artifact coverage changed")
        for name, fingerprint in record["hashes"].items():
            if file_digest(_safe_file(root, name)) != fingerprint: raise ValueError("Reference bytes changed")
        parent = json.loads((root / record["metadata"]).read_text())
        if any(parent.get(k) != v for k,v in declaration.items()): raise ValueError("Parent declaration changed")
        with np.load(root / record["arrays"], allow_pickle=False) as arrays:
            parent["states"] = {n: torch.from_numpy(arrays[key].copy()) for n,key in parent["states"].items()}
            for ref in parent["references"].values(): ref["state"] = torch.from_numpy(arrays[ref.pop("array")].copy())
        grids = _parent_grids(ctx.protocol, declaration["split"])
        expected = {f"{t}:{n}:{horizon_key(h)}" for t in ctx.protocol["tracks"] for n in grids
                    for h in reference_horizons(ctx.protocol, declaration["split"])}
        if set(parent["states"]) != {str(n) for n in grids} or set(parent["references"]) != expected:
            raise ValueError("Reference bank grid/horizon coverage differs")
        for n,state in parent["states"].items():
            if state.dtype != torch.float64 or state_digest(state) != state_digest(field_state(declaration, int(n))):
                raise ValueError("Initial field changed")
        for ref in parent["references"].values():
            if ref["state"].dtype != torch.float64 or state_digest(ref["state"]) != ref["state_sha256"]:
                raise ValueError("Reference state changed")
        loaded.append(parent)
    return loaded
