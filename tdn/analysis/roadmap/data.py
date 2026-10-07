"""Independent frozen cohorts and accepted teachers for the roadmap program.

Confirmation does not exist until every selected checkpoint is frozen.  The
teachers reuse the independently validated FP64 reference integrators, while
the parent declaration, derived horizon list and manifests belong to this new
program.  Worker IPC carries bytes and creates no shared tensor storage files.
"""
from __future__ import annotations

import io
import json
import math
import os
import time
from pathlib import Path

import torch

from tdn.analysis.agenda.data import continuous_field, _generate_parent
from tdn.numerics import Equation, Geometry, choose_substeps, refined_reference
from tdn.research.experiment import Budget, atomic_torch_save, horizon_key, state_digest
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json

SPLITS = ("train", "validation", "calibration", "confirmation")
CONFIRMATION_READ_STAGES = frozenset(("confirm", "policy", "scaling", "report"))


def _safe_file(base, name):
    base = Path(base)
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe sealed artifact path")
    target = base / relative
    if not target.is_file() or base.is_symlink() or any(
            (base / Path(*relative.parts[:index])).is_symlink()
            for index in range(1, len(relative.parts) + 1)):
        raise ValueError(f"Missing or unsafe sealed artifact: {name}")
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("Sealed artifact escaped its stage")
    return target


def verify_frozen_training(ctx):
    """Verify selected models without opening confirmation or allocating models."""
    base = Path(ctx.prerequisites["train"])
    catalog_path = _safe_file(base, "catalog.json")
    freeze_path = _safe_file(base, "freeze.json")
    catalog = json.loads(catalog_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    fingerprint = digest(ctx.protocol)
    if (catalog.get("protocol_sha256") != fingerprint or freeze.get("protocol_sha256") != fingerprint
            or freeze.get("catalog_sha256") != file_digest(catalog_path)
            or catalog.get("selection_split") != "validation" or catalog.get("frozen_before_confirmation") is not True):
        raise ValueError("Training catalog must freeze on validation under the same protocol")
    training_plan = _safe_file(base, "training_plan.json")
    if catalog.get("training_plan_sha256") != file_digest(training_plan):
        raise ValueError("Frozen training plan changed")
    expected = {(family, int(seed)) for family in ctx.protocol["models"] for seed in ctx.protocol["seeds"]}
    records = catalog.get("records", [])
    actual = {(row["family"], int(row["seed"])) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("Every declared family and paired seed must freeze before confirmation")
    selected_ids = [row["model_id"] for row in records]
    if len(set(selected_ids)) != len(selected_ids) or set(freeze.get("checkpoint_hashes", {})) != set(selected_ids):
        raise ValueError("Frozen checkpoint identities differ")
    for row in records:
        checkpoint_path = _safe_file(base, row["checkpoint"])
        actual_hash = file_digest(checkpoint_path)
        if actual_hash != row["checkpoint_sha256"] or actual_hash != freeze["checkpoint_hashes"][row["model_id"]]:
            raise ValueError("Frozen checkpoint bytes changed")
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (value.get("protocol_sha256") != fingerprint or value.get("selection_split") != "validation"
                or value.get("family") != row["family"] or value.get("seed") != row["seed"]):
            raise ValueError("Checkpoint protocol, selection split or family/seed changed")
    return {"catalog_sha256": file_digest(catalog_path), "freeze_sha256": file_digest(freeze_path),
            "checkpoint_hashes": freeze["checkpoint_hashes"], "selection_split": "validation"}


def _validate_cohorts(protocol):
    identities, split_clusters, split_fields = set(), {}, {}
    for parent in protocol["parents"]:
        if parent["split"] not in SPLITS or parent["parent_id"] in identities:
            raise ValueError("Parent IDs must be unique and splits supported")
        identities.add(parent["parent_id"])
        for key, seen in (("field_cluster", split_clusters), ("continuous_field_id", split_fields)):
            value = parent[key]
            if value in seen and seen[value] != parent["split"]:
                raise ValueError("Continuous fields and independent clusters must be split-disjoint")
            seen[value] = parent["split"]
    return {split: len({p["field_cluster"] for p in protocol["parents"] if p["split"] == split}) for split in SPLITS}


def reference_horizons(protocol, split, *, include_intermediate=True):
    if split in ("train", "validation"):
        times = protocol[f"{split}_horizons"]
        return sorted({round(float(h), 12) for h in [*times, *(2 * v for v in times)]})
    times = set()
    for sequence in protocol["confirm_schedules"]:
        times.add(round(sum(sequence), 12))
        if include_intermediate and len(sequence) > 1:
            # One interior checkpoint per prescribed schedule, not every step.
            times.add(round(sum(sequence[:max(1, len(sequence) // 2)]), 12))
    return sorted(times)


def _parent(declared, protocol, budget):
    start = time.monotonic()
    parent, rows = _generate_parent(declared, protocol, budget)
    split = declared["split"]
    if split in ("calibration", "confirmation"):
        additional = set(reference_horizons(protocol, split)) - set(reference_horizons(protocol, split, include_intermediate=False))
        for n in protocol["grids"]:
            initial = parent["states"][str(n)]
            eq, geometry = Equation(declared["kappa"], declared["reaction_rate"]), Geometry((n, n), (1., 1.))
            for horizon in sorted(additional):
                budget.check()
                count = max(8, choose_substeps(horizon, eq, geometry))
                reference = None
                for _ in range(protocol.get("reference_attempts", 5)):
                    if 4 * count > protocol["max_finest_substeps"]:
                        break
                    reference = refined_reference(initial, horizon, eq, geometry, count,
                        tolerance=20 * protocol["reference_tolerance"] / n, check=budget.check)
                    if reference.accepted:
                        break
                    count *= 2
                if reference is None or not reference.accepted:
                    raise FloatingPointError(f"Intermediate teacher unresolved: {parent['parent_id']}, N={n}, h={horizon}")
                row = dict(parent_id=parent["parent_id"], split=split, grid=[n, n], horizon=horizon,
                    track="discrete", accepted=True, uncertainty_rms=reference.uncertainty,
                    uncertainty_max_bound=reference.uncertainty * n, refinement_substeps=list(reference.refinement_substeps),
                    reason=reference.reason, state_sha256=state_digest(reference.state), reference_role="intermediate_trajectory")
                parent["references"][f"discrete:{n}:{horizon_key(horizon)}"] = {**row, "state": reference.state}
                rows.append(row)
    return parent, rows, time.monotonic() - start


def _worker_parent(declared, protocol, deadline):
    torch.set_num_threads(1)
    parent, rows, elapsed = _parent(declared, protocol, Budget(max(0.001, deadline - time.monotonic())))
    buffer = io.BytesIO()
    torch.save(parent, buffer)
    return buffer.getvalue(), rows, elapsed


def _worker_count(ctx, parent_count):
    requested = int(ctx.protocol.get("data", {}).get("teacher_workers", 1))
    cap = int(ctx.protocol["budgets"][ctx.stage]["cpus"])
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is not None:
        cap = min(cap, int(allocated))
    if requested < 1 or cap < 1:
        raise ValueError("Teacher workers and allocated CPU cap must be positive")
    return min(8, requested, cap, parent_count)


def prepare(ctx, confirmation=False):
    from .core import check
    expected_stage = "confirm_prepare" if confirmation else "prepare"
    if ctx.stage != expected_stage:
        raise ValueError("Teacher generation invoked from the wrong declared stage")
    clusters = _validate_cohorts(ctx.protocol)
    splits = ("confirmation",) if confirmation else ("train", "validation", "calibration")
    frozen = verify_frozen_training(ctx) if confirmation else None
    declared = [p for p in ctx.protocol["parents"] if p["split"] in splits]
    if not declared or any(not any(p["split"] == split for p in declared) for split in splits):
        raise ValueError("Every prepared split must have declared parents")
    path = Path(ctx.path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "data_manifest.json").exists() or any((path / f"{split}.pt").exists() for split in splits):
        raise FileExistsError("Preserve completed datasets; use a fresh workflow")
    if frozen is not None:
        write_json(path / "frozen_training.json", frozen)
    workers = _worker_count(ctx, len(declared))
    completed, references = {}, []
    fingerprint = digest(ctx.protocol)

    def retain(index, parent, rows, elapsed):
        completed[index] = parent
        references.extend(rows)
        for split in splits:
            values = [completed[i] for i in sorted(completed) if completed[i]["split"] == split]
            if values:
                atomic_torch_save({"protocol_sha256": fingerprint, "split": split, "parents": values}, path / f"{split}.partial.pt")
        write_json(path / "references.json", {"rows": references, "completed_parents": len(completed), "worker_processes": workers})
        declared_parent = declared[index]
        initial_difference = max(float((state - continuous_field(declared_parent, int(n))).abs().max())
                                 for n, state in parent["states"].items())
        max_uncertainty = max(row["uncertainty_max_bound"] for row in rows)
        ctx.record(f"data/{parent['split']}/{parent['parent_id']}", ["M17", "M18"],
            metrics={"parameters": 0, "generation_seconds": elapsed, "references": len(rows),
                     "max_reference_uncertainty": max_uncertainty, "initial_field_pairing_error": initial_difference,
                     "independent_split_clusters": clusters[parent["split"]]},
            checks=[check("declared_continuous_field", initial_difference, 1e-13, category="math"),
                    check("accepted_teacher_count", sum(bool(r["accepted"]) for r in rows), len(rows), "eq", category="correctness"),
                    check("split_disjoint_fields", True, True, "eq", category="gap", evidence_kind="structural_inventory"),
                    check("reference_uncertainty_vs_primary_target", max_uncertainty, float(ctx.protocol["targets"][0]), category="gap"),
                    check("population_risk_certificate", None, None, applicable=False, required=False, category="math",
                          reason="Independent parents and refinement estimates do not establish a population risk bound")],
            config={**declared_parent, "reference_horizons": reference_horizons(ctx.protocol, parent["split"]),
                    "worker_processes": workers, "independent_unit": "field_cluster",
                    "continuum_scope": "endpoint teachers only; discrete includes one interior prefix per schedule"},
            evidence={"state_digests": parent["state_digests"], "reference_rows": rows,
                      "frozen_training_catalog_sha256": None if frozen is None else frozen["catalog_sha256"]})

    if workers == 1:
        for index, specification in enumerate(declared):
            ctx.budget.check()
            retain(index, *_parent(specification, ctx.protocol, ctx.budget))
    else:
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        import multiprocessing
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        futures = {executor.submit(_worker_parent, value, ctx.protocol, ctx.budget.deadline): index
                   for index, value in enumerate(declared)}
        try:
            while futures:
                ctx.budget.check()
                done, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    payload, rows, elapsed = future.result()
                    parent = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                    retain(index, parent, rows, elapsed)
        except BaseException:
            for process in tuple((executor._processes or {}).values()):
                process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    artifacts, counts = {}, {}
    for split in splits:
        values = [completed[i] for i in sorted(completed) if completed[i]["split"] == split]
        atomic_torch_save({"schema": "tdn.roadmap-data/v1", "protocol_sha256": fingerprint,
                           "split": split, "parents": values}, path / f"{split}.pt")
        artifacts[f"{split}.pt"] = file_digest(path / f"{split}.pt")
        counts[split] = len(values)
        (path / f"{split}.partial.pt").unlink(missing_ok=True)
    artifacts["references.json"] = file_digest(path / "references.json")
    if frozen is not None:
        artifacts["frozen_training.json"] = file_digest(path / "frozen_training.json")
    manifest = {"schema": "tdn.roadmap-data-manifest/v1", "protocol_sha256": fingerprint,
        "stage": ctx.stage, "artifacts": artifacts, "splits": list(splits), "parent_counts": counts,
        "reference_count": len(references), "independent_clusters": {s: clusters[s] for s in splits},
        "teacher_workers": workers, "confirmation_requires_frozen_training": True,
        "uncertainty_is_certificate": False,
        "intermediate_reference_scope": "one nonterminal prefix per schedule; discrete target only"}
    write_json(path / "data_manifest.json", manifest)
    return {"parents": counts, "accepted_references": sum(r["accepted"] for r in references),
            "independent_clusters": manifest["independent_clusters"], "teacher_workers": workers,
            "data_manifest_sha256": file_digest(path / "data_manifest.json")}


def _expected_keys(protocol, parent):
    split = parent["split"]
    grids = [protocol["train_grid"]] if split in ("train", "validation") else protocol["grids"]
    keys = {f"discrete:{n}:{horizon_key(h)}" for n in grids for h in reference_horizons(protocol, split)}
    if parent["parent_id"] in protocol["continuum_parent_ids"]:
        keys.update(f"continuum:{n}:{horizon_key(h)}" for n in grids
                    for h in reference_horizons(protocol, split, include_intermediate=False))
    return keys


def load_bank(ctx, split):
    if split not in SPLITS:
        raise ValueError("Unsupported cohort")
    if split == "confirmation" and ctx.stage not in CONFIRMATION_READ_STAGES:
        raise ValueError("Confirmation access forbidden before frozen confirmation/policy/scaling/report")
    _validate_cohorts(ctx.protocol)
    stage = "confirm_prepare" if split == "confirmation" else "prepare"
    if stage not in ctx.prerequisites:
        raise ValueError(f"Missing sealed {stage} data prerequisite")
    base = Path(ctx.prerequisites[stage])
    manifest = json.loads(_safe_file(base, "data_manifest.json").read_text())
    expected_splits = ["confirmation"] if split == "confirmation" else ["train", "validation", "calibration"]
    if (manifest.get("schema") != "tdn.roadmap-data-manifest/v1" or manifest.get("protocol_sha256") != digest(ctx.protocol)
            or manifest.get("stage") != stage or manifest.get("splits") != expected_splits):
        raise ValueError("Data manifest protocol, stage or split differs")
    required = {f"{value}.pt" for value in expected_splits} | {"references.json"}
    if split == "confirmation":
        required.add("frozen_training.json")
    if set(manifest.get("artifacts", {})) != required:
        raise ValueError("Dataset manifest has missing or unexpected artifacts")
    for name, fingerprint in manifest["artifacts"].items():
        if file_digest(_safe_file(base, name)) != fingerprint:
            raise ValueError(f"Dataset artifact digest differs: {name}")
    if split == "confirmation":
        frozen = verify_frozen_training(ctx)
        if json.loads((base / "frozen_training.json").read_text()) != frozen:
            raise ValueError("Confirmation teachers were prepared under different frozen checkpoints")
    bank = torch.load(base / f"{split}.pt", map_location="cpu", weights_only=True)
    expected = [p for p in ctx.protocol["parents"] if p["split"] == split]
    if (bank.get("schema") != "tdn.roadmap-data/v1" or bank.get("protocol_sha256") != digest(ctx.protocol)
            or bank.get("split") != split or [p["parent_id"] for p in bank["parents"]] != [p["parent_id"] for p in expected]
            or manifest["parent_counts"][split] != len(expected)):
        raise ValueError("Sealed bank identities or protocol differ")
    for actual, declared in zip(bank["parents"], expected):
        if any(actual.get(key) != value for key, value in declared.items()):
            raise ValueError("Parent parameters differ from the frozen declaration")
        grids = [ctx.protocol["train_grid"]] if split in ("train", "validation") else ctx.protocol["grids"]
        if set(actual["states"]) != {str(n) for n in grids} or set(actual["references"]) != _expected_keys(ctx.protocol, declared):
            raise ValueError("State grids or teacher horizons are incomplete")
        for n, state in actual["states"].items():
            if state.dtype != torch.float64 or state.device.type != "cpu" or not torch.isfinite(state).all():
                raise ValueError("Data states must be finite CPU FP64")
            if state_digest(state) != actual["state_digests"][n] or state_digest(state) != state_digest(continuous_field(declared, int(n))):
                raise ValueError("Continuous-field state or digest changed")
        for row in actual["references"].values():
            state = row["state"]
            if (not row["accepted"] or state.dtype != torch.float64 or not torch.isfinite(state).all()
                    or state_digest(state) != row["state_sha256"]):
                raise ValueError("Reference state not accepted or digest changed")
            if any(not math.isfinite(float(row[key])) or row[key] < 0 for key in ("uncertainty_rms", "uncertainty_max_bound")):
                raise ValueError("Reference uncertainty must be finite and nonnegative")
    return bank["parents"]
