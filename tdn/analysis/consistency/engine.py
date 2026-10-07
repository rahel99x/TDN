"""Frozen, bounded learned comparison of interaction-before-compression models.

The independent teachers solve the *coupled PDE*, not a GL3 correction. Every
validation-selected checkpoint is frozen before diagnostics are evaluated. The
FNO is a representative repository implementation with the same physical split
and output constraints; this experiment does not reproduce the FNO paper.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time

import torch

from tdn.numerics import Equation, Geometry, choose_substeps, refined_reference, split_step
from tdn.numerics.invariants import validate_state
from tdn.research.common import fit_feature_normalization
from tdn.research.experiment import Budget, atomic_torch_save, horizon_key, state_digest
from tdn.research.protocol import assert_parent_disjoint, digest, file_digest
from tdn.runtime.metadata import write_json
from tdn.runtime.precision import reference_precision
from tdn.reporting import emit


from .protocol import SCHEMA, FAMILIES, CLASSICAL, REGIMES, build_protocol, validate_protocol, initial_state as _initial
from .audits import audit
from tdn.analysis.premix.neural import _RunBudget, _errors as _reference_errors, _frontiers


def _table(path: Path, rows: list[dict]) -> None:
    write_json(path, {"schema": SCHEMA, "rows": rows})


def _begin(path: Path, protocol: dict) -> None:
    validate_protocol(protocol)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("summary.json", "dataset.pt", "training.json", "checks.json", "COMPLETED")):
        raise FileExistsError("Preserve existing consistency artifacts; use a fresh run directory")
    existing = path / "protocol.json"
    if existing.exists() and digest(json.loads(existing.read_text())) != digest(protocol):
        raise ValueError("Existing protocol differs from the requested immutable experiment")
    write_json(existing, protocol)


def _manifest(path: Path, names: list[str], protocol: dict, filename: str) -> None:
    write_json(path / filename, {"schema": SCHEMA, "protocol_sha256": digest(protocol),
        "artifacts": {name: file_digest(path / name) for name in names}})


def _verify_manifest(path: Path, protocol: dict, filename: str, expected: set[str] | None = None) -> dict:
    manifest = json.loads((path / filename).read_text())
    if manifest.get("protocol_sha256") != digest(protocol):
        raise ValueError("Artifact manifest belongs to a different protocol")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or (expected is not None and set(artifacts) != expected):
        raise ValueError("Artifact manifest file plan differs from the expected files")
    for name, fingerprint in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (path / relative).resolve().is_relative_to(path.resolve()):
            raise ValueError("Artifact manifest escapes the run directory")
        if file_digest(path / relative) != fingerprint:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    return manifest



def _summary(path, protocol, *, stage, status, device, start, error=None, **values):
    result = {"schema": SCHEMA, "status": status, "stage": stage, "profile": protocol["profile"],
        "device": device, "elapsed_seconds": time.monotonic() - start, "scope": protocol["scope"],
        "scientific_outcome": "DESCRIPTIVE_CONSISTENCY_ABLATION" if status == "COMPLETED" and stage == "neural" else "NOT_ESTABLISHED",
        **values}
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(path / "summary.json", result)
    return result


def _status(error):
    return "INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "FAILED"


def prepare(protocol: dict, run_dir: Path, *, stop=None) -> dict:
    """CPU-only teacher generation, with all accepted references sealed for reuse."""
    path, start = Path(run_dir), time.monotonic()
    _begin(path, protocol)
    budget = Budget(protocol["prepare_seconds"], stop)
    dataset = {"schema": SCHEMA, "protocol_sha256": digest(protocol), "parents": []}
    records, seen = [], set()
    try:
        for declared in protocol["parents"]:
            budget.check()
            state = _initial(declared)
            fingerprint = state_digest(state)
            if fingerprint in seen:
                raise ValueError("Duplicate actual initial fields across supposedly independent parents")
            seen.add(fingerprint)
            parent = {**declared, "initial": state, "initial_state_sha256": fingerprint, "references": {}}
            equation, geometry = Equation(declared["kappa"], declared["reaction_rate"]), Geometry(tuple(declared["grid"]), (1., 1.))
            for horizon in declared["horizons"]:
                count = max(8, choose_substeps(horizon, equation, geometry))
                attempts = []
                reference = None
                for _ in range(protocol["reference_attempts"]):
                    budget.check()
                    if count * 4 > protocol["max_finest_substeps"]:
                        break
                    reference = refined_reference(state, horizon, equation, geometry, count,
                        tolerance=protocol["reference_tolerance"] / math.sqrt(math.prod(declared["grid"])), check=budget.check)
                    attempts.append({"substeps": list(reference.refinement_substeps),
                        "differences": [x if math.isfinite(x) else None for x in reference.refinement_differences],
                        "reason": reference.reason, "accepted": reference.accepted})
                    if reference.accepted:
                        break
                    count *= 2
                accepted = reference is not None and reference.accepted
                record = {key: declared[key] for key in ("parent_id", "split", "regime", "category", "grid")}
                record.update(horizon=horizon, accepted=accepted, attempts=attempts,
                    uncertainty_rms=reference.uncertainty if reference is not None and math.isfinite(reference.uncertainty) else None,
                    uncertainty_max_bound=reference.uncertainty * math.sqrt(state.numel()) if reference is not None and math.isfinite(reference.uncertainty) else None,
                    reference_state_sha256=state_digest(reference.state) if accepted else None)
                records.append(record)
                _table(path / "references.json", records)
                if not accepted:
                    raise FloatingPointError(f"Independent reference acceptance failed for {declared['parent_id']} at {horizon}; training withheld")
                parent["references"][horizon_key(horizon)] = {**record, "state": reference.state.cpu()}
            dataset["parents"].append(parent)
            atomic_torch_save(dataset, path / "dataset.partial.pt")
            emit({"accepted_references": len(records), "completed_parents": len(dataset["parents"])},
                 phase="consistency-references", step=len(dataset["parents"]), completed=len(dataset["parents"]),
                 total=len(protocol["parents"]), unit="parents")
        train = [parent for parent in dataset["parents"] if parent["split"] == "train"]
        mean, std = fit_feature_normalization(((parent["initial"], Equation(parent["kappa"], parent["reaction_rate"]),
            Geometry(tuple(parent["grid"]), (1., 1.))) for parent in train), t_ref=protocol["t_ref"], U_ref=protocol["U_ref"])
        write_json(path / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "split": "train",
            "parent_ids": [parent["parent_id"] for parent in train], "protocol_sha256": digest(protocol),
            "scope": protocol["normalization"]})
        atomic_torch_save(dataset, path / "dataset.pt")
        (path / "dataset.partial.pt").unlink()
        summary = _summary(path, protocol, stage="dataset", status="COMPLETED", device="cpu", start=start,
            counts={"parents": len(dataset["parents"]), "references": len(records), "accepted_references": len(records)})
        _manifest(path, ["protocol.json", "dataset.pt", "normalization.json", "references.json", "summary.json"], protocol, "dataset_manifest.json")
        return summary
    except BaseException as error:
        _summary(path, protocol, stage="dataset", status=_status(error), device="cpu", start=start, error=error,
            counts={"completed_parents": len(dataset["parents"]), "references": len(records)})
        raise


def _load_dataset(protocol: dict, path: Path) -> tuple[dict, tuple]:
    _verify_manifest(path, protocol, "dataset_manifest.json", {"protocol.json", "dataset.pt", "normalization.json", "references.json", "summary.json"})
    if digest(json.loads((path / "protocol.json").read_text())) != digest(protocol):
        raise ValueError("Dataset protocol mismatch")
    source_summary = json.loads((path / "summary.json").read_text())
    if source_summary.get("status") != "COMPLETED" or source_summary.get("stage") != "dataset":
        raise ValueError("Neural training requires a completed, accepted dataset")
    dataset = torch.load(path / "dataset.pt", map_location="cpu", weights_only=True)
    if dataset.get("schema") != SCHEMA or dataset.get("protocol_sha256") != digest(protocol) or len(dataset.get("parents", [])) != len(protocol["parents"]):
        raise ValueError("Dataset identity or parent count mismatch")
    assert_parent_disjoint(dataset["parents"])
    seen = set()
    for parent, declared in zip(dataset["parents"], protocol["parents"]):
        if {key: parent.get(key) for key in declared} != declared:
            raise ValueError("Dataset parent plan mismatch")
        state = parent["initial"]
        if state.dtype != torch.float64 or list(state.shape) != [1, 1, *declared["grid"]]:
            raise ValueError("Dataset field shape or precision mismatch")
        fingerprint = state_digest(state)
        if fingerprint != parent["initial_state_sha256"] or fingerprint in seen:
            raise ValueError("Dataset initial field identity mismatch or duplicate")
        seen.add(fingerprint)
        validate_state(state)
        if set(parent["references"]) != {horizon_key(h) for h in declared["horizons"]}:
            raise ValueError("Dataset reference horizons mismatch")
        for reference in parent["references"].values():
            if reference["accepted"] is not True or reference["state"].dtype != torch.float64 or reference["state"].shape != state.shape:
                raise ValueError("Dataset reference acceptance/shape/precision mismatch")
            if state_digest(reference["state"]) != reference["reference_state_sha256"]:
                raise ValueError("Dataset reference state digest mismatch")
    normalization = json.loads((path / "normalization.json").read_text())
    expected_ids = [parent["parent_id"] for parent in protocol["parents"] if parent["split"] == "train"]
    if normalization.get("parent_ids") != expected_ids or normalization.get("split") != "train" or normalization.get("protocol_sha256") != digest(protocol):
        raise ValueError("Normalization is not the shared training-only fit")
    return dataset, (torch.tensor(normalization["mean"], dtype=torch.float32), torch.tensor(normalization["std"], dtype=torch.float32))


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _safe(model, state, h, equation, geometry):
    value = model(state, h, equation, geometry)
    try:
        validate_state(value)
    except ValueError as error:
        raise FloatingPointError(f"Neural trajectory inadmissible: {error}") from error
    return value


def _loss(model, parent, h, protocol, device):
    equation, geometry = Equation(parent["kappa"], parent["reaction_rate"]), Geometry(tuple(parent["grid"]), (1., 1.))
    initial = parent["initial"].to(device=device, dtype=torch.float32)
    first = _safe(model, initial, h, equation, geometry)
    second = _safe(model, first, h, equation, geometry)
    target1 = parent["references"][horizon_key(h)]["state"].to(device=device, dtype=torch.float32)
    target2 = parent["references"][horizon_key(2 * h)]["state"].to(device=device, dtype=torch.float32)
    loss = ((first - target1).square().mean() + protocol["two_step_loss_weight"] * (second - target2).square().mean()) / protocol["loss_scale"]**2
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite physical PDE endpoint loss")
    return loss


def _validation(model, validation, protocol, device, budget):
    model.eval()
    values = []
    with torch.no_grad():
        for parent in validation:
            for horizon in protocol["validation_horizons"]:
                budget.check()
                values.append(float(_loss(model, parent, horizon, protocol, device)))
    return statistics.fmean(values)


def _state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train(family, seed, dataset, normalization, protocol, path, device, budget):
    from tdn.research.consistency_neural import build_model
    _sync(device)
    start = time.monotonic()
    torch.manual_seed(seed)
    model = build_model(family, width=protocol["width"], modes=protocol["modes"], normalization=normalization,
                        t_ref=protocol["t_ref"], U_ref=protocol["U_ref"]).to(device=device, dtype=torch.float32)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    train = [parent for parent in dataset["parents"] if parent["split"] == "train"]
    validation = [parent for parent in dataset["parents"] if parent["split"] == "validation"]
    pairs = [(parent, h) for parent in train for h in protocol["train_horizons"]]
    generator = torch.Generator().manual_seed(seed + 1)
    schedule, queue = [], []
    for _ in range(protocol["updates"]):
        if not queue:
            queue = torch.randperm(len(pairs), generator=generator).tolist()
        schedule.append(queue.pop())
    record = {"family": family, "seed": seed, "status": "RUNNING", "steps": 0, "selected_step": None,
        "selected_parameters_changed": False, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_parent_ids": [parent["parent_id"] for parent in train],
        "validation_parent_ids": [parent["parent_id"] for parent in validation],
        "diagnostics_seen_during_training": False, "sample_schedule_sha256": digest(schedule),
        "history": [], "architecture": model.architecture_metadata() if hasattr(model, "architecture_metadata") else {"family": family}}
    checkpoint = path / "checkpoints" / f"seed-{seed}-{family}.pt"
    record_path = path / "training" / f"seed-{seed}-{family}.json"
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    best, last_loss = math.inf, None
    _sync(device)
    record["initialization_seconds"] = time.monotonic() - start
    try:
        for step in range(protocol["updates"] + 1):
            budget.check()
            if step:
                parent, horizon = pairs[schedule[step - 1]]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, parent, horizon, protocol, device)
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                    raise FloatingPointError("Nonfinite parameter gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                budget.check()
                record["steps"], last_loss = step, float(loss.detach())
            if step % protocol["validation_every"] == 0 or step == protocol["updates"]:
                try:
                    score = _validation(model, validation, protocol, device, budget)
                except FloatingPointError:
                    score = math.inf
                record["history"].append({"step": step, "validation_loss": score if math.isfinite(score) else None, "training_loss": last_loss})
                if math.isfinite(score) and score < best:
                    best = score
                    record["selected_step"] = step
                    atomic_torch_save({"family": family, "seed": seed, "protocol_sha256": digest(protocol),
                        "selected_step": step, "validation_loss": best, "state_dict": _state(model)}, checkpoint)
                write_json(record_path, record)
                emit({"optimizer_steps": step, "finite_validation": int(math.isfinite(score)),
                      **({"training_loss": last_loss} if last_loss is not None else {}),
                      **({"validation_loss": score} if math.isfinite(score) else {})},
                     phase=f"consistency-train-{seed}-{family}", step=step,
                     completed=step, total=protocol["updates"], unit="updates")
        if record["selected_step"] is None:
            raise FloatingPointError("No finite admissible validation checkpoint")
        if not any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.named_parameters()):
            raise FloatingPointError("Optimizer completed without changing any trainable parameter")
        selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["state_dict"], strict=True)
        model.eval()
        record.update(status="COMPLETED", checkpoint_sha256=file_digest(checkpoint),
            selected_parameters_changed=any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.named_parameters()),
            best_validation_loss=best)
        record["selection"] = "TRAINED_CHECKPOINT" if record["selected_step"] > 0 and record["selected_parameters_changed"] else "SELECTED_INITIALIZATION"
    except FloatingPointError as error:
        record.update(status="NUMERICAL_FAILURE", error=str(error), selection="NO_ELIGIBLE_CHECKPOINT")
        model = None
    except BaseException as error:
        _sync(device)
        record.update(status=_status(error), error=f"{type(error).__name__}: {error}", training_seconds=time.monotonic() - start)
        write_json(record_path, record)
        raise
    _sync(device)
    record["training_seconds"] = time.monotonic() - start
    write_json(record_path, record)
    return model, record


def _errors(value, reference):
    result = _reference_errors(value, reference)
    result["signed_mean_error"] = float((value.detach().double().cpu() - reference["state"]).mean())
    return result


def _base_regression(candidates):
    """Pair every endpoint with the same-parent, same-step physical Strang base."""
    base = {(row["parent_id"], row["steps"]): row for row in candidates
            if row["family"] == "strang" and row["status"] == "COMPLETED"}
    fields = (("rms", "rms_vs_base"), ("max_error", "max_vs_base"),
              ("mean_error", "mean_vs_base"), ("spatial_rms", "spatial_vs_base"))
    for row in candidates:
        paired = base.get((row["parent_id"], row["steps"]))
        row["base_comparison_status"] = "AVAILABLE" if paired is not None else "BASE_UNAVAILABLE"
        if paired is None:
            continue
        for field, ratio in fields:
            row[f"base_{field}"] = paired[field]
            row[ratio] = row.get(field) / paired[field] if field in row and paired[field] > 0 else None


def _trained_consistency(protocol, models, records, path, device, budget):
    """Post-freeze exact-limit checks cannot influence checkpoint selection."""
    rows = []
    generator = torch.Generator().manual_seed(950001)
    varied = (.1 + .8 * torch.rand(1, 1, 8, 8, generator=generator)).to(device)
    geometry = Geometry((8, 8), (1., 1.))
    cases = (("constant", torch.full_like(varied, .35), .12, Equation(.003, 2.)),
             ("zero_reaction", varied, .12, Equation(.003, 0.)),
             ("zero_diffusion", varied, .12, Equation(0., 2.)),
             ("zero_horizon", varied, 0., Equation(.003, 2.)))
    with torch.no_grad():
        for record in records:
            seed, family = record["seed"], record["family"]
            model = models.get((seed, family))
            for case, state, horizon, equation in cases:
                budget.check()
                required = family.endswith(("_gated", "_moment")) or case == "zero_horizon"
                row = {"family": family, "seed": seed, "case": case, "required": required,
                       "checkpoint_selection": record["selection"], "after_checkpoint_freeze": True,
                       "diagnostic_parents_used": False, "absolute_tolerance": 2e-6}
                if model is None:
                    row.update(status="TRAINING_FAILED", max_base_difference=None)
                else:
                    value = _safe(model, state, horizon, equation, geometry)
                    difference = float((value - split_step(state, horizon, equation, geometry)).abs().max())
                    row.update(max_base_difference=difference,
                               status="PASS" if difference <= 2e-6 else "FAIL" if required else "OBSERVED")
                rows.append(row)
                _table(path / "consistency.json", rows)
                if row["status"] == "FAIL":
                    raise RuntimeError(f"Frozen trained exact-limit check failed: {seed}/{family}/{case}")
    return rows


def _target_coverage(frontiers):
    groups = {}
    for row in frontiers:
        key = (row["family"], row["seed"], row["norm"], row["target"])
        if key not in groups:
            groups[key] = {"family": row["family"], "seed": row["seed"], "norm": row["norm"],
                           "target": row["target"], "parents": 0, "feasible_parents": 0}
        groups[key]["parents"] += 1
        groups[key]["feasible_parents"] += row["status"] == "FEASIBLE"
    return list(groups.values())


def _evaluate(protocol, dataset, models, records, path, device, budget, candidates):
    from tdn.analysis.profiling import measure
    from tdn.research.work_precision import prepare_step
    from tdn.research.compact_spatial import prepare_compact_step
    diagnostics = [parent for parent in dataset["parents"] if parent["split"] == "diagnostic"]
    methods = [(None, family, None) for family in CLASSICAL]
    methods.extend((seed, family, models.get((seed, family))) for seed in protocol["seeds"] for family in FAMILIES)
    record_map = {(record["seed"], record["family"]): record for record in records}
    with torch.no_grad():
        for index, parent in enumerate(diagnostics, 1):
            equation, geometry = Equation(parent["kappa"], parent["reaction_rate"]), Geometry(tuple(parent["grid"]), (1., 1.))
            initial = parent["initial"].to(device=device, dtype=torch.float32)
            horizon = parent["horizons"][0]
            reference = parent["references"][horizon_key(horizon)]
            counts = protocol["long_step_counts"] if parent["regime"] == "long_rollout" else protocol["step_counts"]
            order = torch.randperm(len(methods), generator=torch.Generator().manual_seed(parent["seed"] + 99)).tolist()
            for timing_order, method_index in enumerate(order):
                seed, family, model = methods[method_index]
                for steps in counts:
                    budget.check()
                    row = {key: parent[key] for key in ("parent_id", "regime", "category", "distribution", "grid")}
                    row.update(seed=seed, family=family, steps=steps, horizon=horizon, device=device,
                        state_precision="float32", status="RUNNING", admissible=False,
                        timing_order=timing_order,
                        uncertainty_rms=reference["uncertainty_rms"], uncertainty_max_bound=reference["uncertainty_max_bound"])
                    if seed is not None:
                        row["checkpoint_selection"] = record_map[seed, family]["selection"]
                        if model is None:
                            row.update(status="TRAINING_FAILED", error=record_map[seed, family].get("error"))
                            candidates.append(row)
                            continue
                    try:
                        _sync(device)
                        setup_start = time.perf_counter()
                        if family in CLASSICAL:
                            prepared = (prepare_compact_step(initial, horizon / steps, equation, geometry, "gl3_fused") if family == "gl3_fused"
                                        else prepare_step(initial, horizon / steps, equation, geometry, family))
                        _sync(device)
                        row["setup_seconds"] = time.perf_counter() - setup_start
                        def operation():
                            state = initial.clone()
                            for _ in range(steps):
                                budget.check()
                                if seed is None:
                                    state = prepared(state)
                                    try:
                                        validate_state(state)
                                    except ValueError as error:
                                        raise FloatingPointError(f"Classical output inadmissible: {error}") from error
                                else:
                                    state = _safe(model, state, horizon / steps, equation, geometry)
                            return state
                        value, timing = measure(operation, device=device, warmup=protocol["timing_warmup"],
                                                repeats=protocol["timing_repeats"], memory_policy=budget.timing_memory_policy)
                        row.update(status="COMPLETED", admissible=True, timing=timing, **_errors(value, reference),
                                   memory_ok=timing.get("soft_budget_passed", True) and not timing.get("hard_device_memory_warning", False))
                    except FloatingPointError as error:
                        row.update(status="INVALID_TRAJECTORY", error=str(error))
                    except BaseException as error:
                        row.update(status=_status(error), error=f"{type(error).__name__}: {error}",
                                   timing_complete=False)
                        candidates.append(row)
                        raise
                    candidates.append(row)
            _table(path / "candidates.json", candidates)
            _table(path / "frontiers.json", _frontiers(candidates, protocol))
            emit({"candidate_rows": len(candidates), "diagnostic_parents": index}, phase="consistency-diagnostics",
                 step=index, completed=index, total=len(diagnostics), unit="parents")
    _base_regression(candidates)
    _table(path / "candidates.json", candidates)


def run(protocol: dict, dataset_dir: Path, run_dir: Path, *, device: str = "cpu", stop=None) -> dict:
    """Actual bounded training and independent diagnosis on the requested device."""
    validate_protocol(protocol)
    if device not in ("cpu", "cuda"):
        raise ValueError("Neural device must be explicitly cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden")
    if device == "cuda":
        from tdn.runtime.preflight import verify_runtime
        verify_runtime("cuda", "consistency-neural")
    path, source, start = Path(run_dir), Path(dataset_dir), time.monotonic()
    if path.resolve() == source.resolve():
        raise ValueError("Dataset and neural artifacts require separate directories")
    _begin(path, protocol)
    budget = _RunBudget(protocol["run_seconds"], stop, device)
    records, candidates, models = [], [], {}
    reference_precision()
    try:
        dataset, normalization = _load_dataset(protocol, source)
        dataset_hash = file_digest(source / "dataset.pt")
        write_json(path / "dataset_source.json", {"path": str(source.resolve()), "dataset_sha256": dataset_hash,
                   "manifest_sha256": file_digest(source / "dataset_manifest.json"), "protocol_sha256": digest(protocol)})
        for seed in protocol["seeds"]:
            for family in FAMILIES:
                budget.check()
                model, record = _train(family, seed, dataset, normalization, protocol, path, device, budget)
                records.append(record)
                models[seed, family] = model
                budget.observe(force=True)
                write_json(path / "memory.json", budget.memory)
                _table(path / "training.json", records)
        # Freeze and verify *every* candidate before exposing any diagnostic.
        frozen = {f"checkpoints/seed-{record['seed']}-{record['family']}.pt": record["checkpoint_sha256"]
                  for record in records if record["status"] == "COMPLETED"}
        for name, fingerprint in frozen.items():
            if file_digest(path / name) != fingerprint:
                raise ValueError("Checkpoint changed before diagnostic freeze")
        write_json(path / "checkpoint_freeze.json", {"protocol_sha256": digest(protocol), "dataset_sha256": dataset_hash,
            "selection_split": "validation", "all_training_completed_before_diagnostics": True, "diagnostic_parent_ids": [p["parent_id"] for p in protocol["parents"] if p["split"] == "diagnostic"], "artifacts": frozen})
        consistency = _trained_consistency(protocol, models, records, path, device, budget)
        _evaluate(protocol, dataset, models, records, path, device, budget, candidates)
        budget.observe(force=True)
        write_json(path / "memory.json", budget.memory)
        for name, fingerprint in frozen.items():
            if file_digest(path / name) != fingerprint:
                raise ValueError("Frozen checkpoint changed during diagnostics")
        target_coverage = _target_coverage(_frontiers(candidates, protocol))
        _table(path / "target_coverage.json", target_coverage)
        from .comparison import summarize
        comparison = summarize(path)
        summary = _summary(path, protocol, stage="neural", status="COMPLETED", device=device, start=start,
            counts={"training_records": len(records), "trained_checkpoints": sum(row.get("selection") == "TRAINED_CHECKPOINT" for row in records),
                    "selected_initializations": sum(row.get("selection") == "SELECTED_INITIALIZATION" for row in records),
                    "numerical_training_failures": sum(row["status"] == "NUMERICAL_FAILURE" for row in records),
                    "candidates": len(candidates), "completed_candidates": sum(row["status"] == "COMPLETED" for row in candidates),
                    "diagnostic_parents": sum(parent["split"] == "diagnostic" for parent in protocol["parents"]),
                    "trained_consistency_checks": len(consistency)},
            actual_training_device=device, actual_neural_training=any(row["steps"] > 0 for row in records),
            pairing="training seeds reuse the same independent diagnostic parents; repeated parents are not independent evidence",
            resolution_design=protocol["resolution_design"],
            preparation_seconds=json.loads((source / "summary.json").read_text())["elapsed_seconds"],
            setup_timing_scope="classical coefficient preparation only; learned model construction/training belongs to training_seconds; steady-state timing excludes preparation for every family",
            frontier_scope=protocol["frontier"], fnopaper_reproduction=False,
            checkpoint_selection_scope="validation-only; neither consistency checks nor fresh diagnostic endpoints can select or retune",
            mean_dynamics=protocol["mean_law"], target_coverage=target_coverage,
            comparisons=comparison)
        _manifest(path, ["protocol.json", "dataset_source.json", "training.json", "candidates.json", "frontiers.json",
            "checkpoint_freeze.json", "summary.json", "memory.json", "consistency.json", "target_coverage.json", "comparisons.json", *frozen], protocol, "neural_manifest.json")
        return summary
    except BaseException as error:
        # A failed/interrupted family writes its per-family training record;
        # preserve it in the aggregate even when its call did not return.
        known = {(row["seed"], row["family"]) for row in records}
        for record_path in sorted((path / "training").glob("*.json")):
            record = json.loads(record_path.read_text())
            if (record["seed"], record["family"]) not in known:
                records.append(record)
        _table(path / "training.json", records)
        write_json(path / "memory.json", budget.memory)
        _table(path / "candidates.json", candidates)
        _table(path / "frontiers.json", _frontiers(candidates, protocol))
        _summary(path, protocol, stage="neural", status=_status(error), device=device, start=start, error=error,
                 counts={"training_records": len(records), "candidate_rows": len(candidates)})
        raise
