"""Actual small neural training and whole-solver development comparisons.

No per-parent teacher-fitted coefficient oracle is used by this experiment.
Diagnostics are opened only after validation-selected checkpoints are frozen.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import time

import torch

from tdn.analysis.profiling import measure
from tdn.analysis.workflow import _rollout as classical_rollout, step_schedule
from tdn.numerics import Equation, Geometry, choose_substeps, refined_reference
from tdn.numerics.invariants import validate_state
from tdn.runtime.metadata import write_json
from tdn.reporting import emit
from .protocol import (CLASSICAL, NEURAL_BASELINES, TOLERANCE, assert_parent_disjoint,
                       benchmark_suite, digest, file_digest)


class InvalidTrajectory(FloatingPointError):
    """A numerical admissibility failure, distinct from a programming error."""


class Budget:
    def __init__(self, seconds: float, stop=None):
        self.deadline = time.monotonic() + seconds
        self.stop = stop

    def check(self):
        if self.stop is not None and self.stop.requested:
            raise InterruptedError(f"Stopped by signal {self.stop.signal_number}; completed artifacts retained")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Bounded research walltime exceeded; completed artifacts retained")


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def initial_state(parent: dict, geometry: Geometry) -> torch.Tensor:
    generator = torch.Generator().manual_seed(parent["seed"])
    x, y = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) / n for n in geometry.grid], indexing="ij")
    if parent["state_class"] == "mixed_frequency":
        phase = 2 * math.pi * torch.rand(3, generator=generator, dtype=torch.float64)
        value = (.5 + .15 * torch.sin(2 * math.pi * x + phase[0])
                 + .1 * torch.cos(2 * math.pi * y + phase[1])
                 + .1 * torch.sin(6 * math.pi * (x + y) + phase[2]))
    elif parent["state_class"] == "bounded_random":
        value = .15 + .7 * torch.rand(geometry.grid, generator=generator, dtype=torch.float64)
    elif parent["state_class"] == "boundary":
        shift = torch.rand(2, generator=generator, dtype=torch.float64)
        value = ((torch.remainder(x + shift[0], 1) < .5)
                 ^ (torch.remainder(y + shift[1], 1) < .5)).to(torch.float64)
    else:
        raise ValueError("Unknown research state class")
    value = value[None, None]
    validate_state(value)
    return value


def horizon_key(h: float) -> str:
    return format(h, ".12g")


def state_digest(state: torch.Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().double().contiguous().numpy().tobytes()).hexdigest()


def generate_data(protocol: dict, run_dir: Path, budget: Budget) -> dict:
    config = protocol["config"]
    geometry = Geometry(tuple(config["grid"]), (1., 1.))
    dataset = {"protocol_sha256": digest(protocol), "parents": [], "geometry": config["grid"]}
    records = []
    initial_hashes = set()
    emit({"reference_records": 0, "accepted_references": 0}, phase="research-references",
         completed=0, total=len(protocol["parents"]), unit="parents")
    for declared in protocol["parents"]:
        budget.check()
        equation = Equation(declared["kappa"], declared["reaction_rate"])
        state = initial_state(declared, geometry)
        initial_hash = state_digest(state)
        if initial_hash in initial_hashes:
            raise ValueError("Duplicate actual initial state despite distinct parent seeds")
        initial_hashes.add(initial_hash)
        hs = config["train_horizons"] if declared["split"] == "train" else config["heldout_horizons"]
        horizons = sorted({*hs, *(2 * h for h in hs), config["rollout_time"]})
        parent = {**declared, "initial": state, "initial_state_sha256": initial_hash, "references": {}}
        for h in horizons:
            budget.check()
            count = max(8, choose_substeps(h, equation, geometry))
            if 4 * count > config["max_finest_substeps"]:
                raise ValueError("Reference count exceeds the predeclared finest-step bound")
            ref = refined_reference(state, h, equation, geometry, count,
                                    error_fraction=.05, tolerance=TOLERANCE * .1, noise_floor=1e-10)
            record = {"parent_id": declared["parent_id"], "h": h, "accepted": ref.accepted,
                      "reason": ref.reason, "uncertainty": ref.uncertainty if math.isfinite(ref.uncertainty) else None,
                      "refinement_substeps": list(ref.refinement_substeps),
                      "refinement_differences": [x if math.isfinite(x) else None for x in ref.refinement_differences]}
            records.append(record)
            parent["references"][horizon_key(h)] = {"state": ref.state.cpu(), **record}
        dataset["parents"].append(parent)
        write_json(run_dir / "references.json", {"records": records, "completed_parents": len(dataset["parents"]),
                                                  "declared_parents": len(protocol["parents"])})
        atomic_torch_save(dataset, run_dir / "dataset.partial.pt")
        emit({"reference_records": len(records),
              "accepted_references": sum(row["accepted"] for row in records)},
             phase="research-references", step=len(dataset["parents"]),
             completed=len(dataset["parents"]), total=len(protocol["parents"]), unit="parents")
    if not all(row["accepted"] for row in records):
        raise FloatingPointError("Teacher acceptance failed; all cases retained and training withheld")
    atomic_torch_save(dataset, run_dir / "dataset.pt")
    (run_dir / "dataset.partial.pt").unlink()
    return dataset


def split_parents(dataset: dict) -> dict[str, list[dict]]:
    assert_parent_disjoint(dataset["parents"])
    return {name: [parent for parent in dataset["parents"] if parent["split"] == name]
            for name in ("train", "validation", "diagnostic")}


def training_normalization(dataset: dict, geometry: Geometry):
    from .common import fit_feature_normalization
    train = split_parents(dataset)["train"]
    values = [(parent["initial"], Equation(parent["kappa"], parent["reaction_rate"]), geometry) for parent in train]
    return fit_feature_normalization(values), [parent["parent_id"] for parent in train]


def physical_loss(model, parent: dict, h: float, geometry: Geometry, device: str, *, audit=False) -> torch.Tensor:
    equation = Equation(parent["kappa"], parent["reaction_rate"])
    state = parent["initial"].to(device=device, dtype=torch.float32)
    first = _safe_step(model, state, h, equation, geometry) if audit else model(state, h, equation, geometry)
    second = _safe_step(model, first, h, equation, geometry) if audit else model(first, h, equation, geometry)
    one = parent["references"][horizon_key(h)]["state"].to(device=device, dtype=torch.float32)
    two = parent["references"][horizon_key(2 * h)]["state"].to(device=device, dtype=torch.float32)
    loss = ((first - one).square().mean() + .5 * (second - two).square().mean()) / TOLERANCE**2
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite common physical-state training loss")
    return loss


def validation_loss(model, parents: list[dict], horizons: list[float], geometry: Geometry, device: str,
                    budget: Budget, failure_details: list | None = None) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for parent in parents:
            for h in horizons:
                budget.check()
                try:
                    values.append(float(physical_loss(model, parent, h, geometry, device, audit=True)))
                except FloatingPointError as error:
                    if failure_details is not None:
                        failure_details.append({"parent_id": parent["parent_id"], "h": h,
                                                "error": f"{type(error).__name__}: {error}"})
                    return math.inf
    return sum(values) / len(values)


def checkpoint_is_better(candidate: float, current: float) -> bool:
    return math.isfinite(candidate) and candidate < current


def _cpu_state(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_family(family: str, dataset: dict, protocol: dict, run_dir: Path,
                 normalization: tuple, budget: Budget, device="cpu", *, initialization_seed=74001,
                 sample_schedule_seed=74002, replicate_id: str | None = None) -> tuple[object | None, dict]:
    started = time.monotonic()
    from . import build_research_model
    config = protocol["config"]
    geometry = Geometry(tuple(config["grid"]), (1., 1.))
    parents = split_parents(dataset)
    torch.manual_seed(initialization_seed)
    model = build_research_model(family, ndim=2, width=config["width"]).to(device=device, dtype=torch.float32)
    model.set_normalization(*normalization)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    record = {"family": family, "status": "RUNNING", "steps": 0, "selected_step": None,
              "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
              "selection_split": "validation", "normalization_split": "train", "history": [],
              "training_parent_ids": [parent["parent_id"] for parent in parents["train"]],
              "validation_parent_ids": [parent["parent_id"] for parent in parents["validation"]],
              "diagnostics_seen_during_training": False}
    record.update(benchmark_role="neural_baseline" if family in NEURAL_BASELINES else "tdn",
                  optimizer="Adam", learning_rate=config["learning_rate"],
                  maximum_optimizer_updates=config["max_steps"], initialization_seed=initialization_seed,
                  sample_schedule_seed=sample_schedule_seed,
                  architecture=(model.architecture_metadata() if hasattr(model, "architecture_metadata")
                                else {"backbone": family, "track": "hybrid", "time_input": "structured temporal decoder"}))
    replication = ({"replicate_id": replicate_id, "training_seed": initialization_seed,
                    "sample_schedule_seed": sample_schedule_seed} if replicate_id is not None else {})
    record.update(replication)
    checkpoint_path = run_dir / "checkpoints" / f"{family}.pt"
    record_path = run_dir / "training" / f"{family}.json"
    best = math.inf
    candidates = [(parent, h) for parent in parents["train"] for h in config["train_horizons"]]
    generator = torch.Generator().manual_seed(sample_schedule_seed)
    order = []
    phase_suffix = f"{replicate_id}-{family}" if replicate_id is not None else family
    failure_context = "initialization"
    emit({"optimizer_steps": 0, "parameter_count": record["parameter_count"]},
         phase=f"research-train-{phase_suffix}", step=0, completed=0, total=config["max_steps"], unit="optimizer steps")
    try:
        for step in range(config["max_steps"] + 1):
            budget.check()
            if step:
                failure_context = "training"
                if not order:
                    order = torch.randperm(len(candidates), generator=generator).tolist()
                parent, h = candidates[order.pop()]
                record.update(optimizer_attempt=step, training_parent_id=parent["parent_id"], training_h=h)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = physical_loss(model, parent, h, geometry, device)
                loss.backward()
                bad_gradients = [name for name, parameter in model.named_parameters()
                                 if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                if bad_gradients:
                    record["nonfinite_gradient_parameters"] = bad_gradients
                    raise FloatingPointError("Nonfinite neural gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                record["steps"] = step
                record["last_training_loss"] = float(loss.detach())
                emit({"loss": record["last_training_loss"], "optimizer_steps": step},
                     phase=f"research-train-{phase_suffix}", step=step,
                     completed=step, total=config["max_steps"], unit="optimizer steps")
            if step % config["validation_every"] == 0 or step == config["max_steps"]:
                failure_context = "validation"
                validation_failures = []
                score = validation_loss(model, parents["validation"], config["heldout_horizons"], geometry,
                                        device, budget, validation_failures)
                history = {"step": step, "validation_loss": score if math.isfinite(score) else None,
                           "admissible_validation": math.isfinite(score)}
                if validation_failures:
                    failures = [{**row, "step": step} for row in validation_failures]
                    history["failures"] = failures
                    record.setdefault("validation_failures", []).extend(failures)
                record["history"].append(history)
                if checkpoint_is_better(score, best):
                    best = score
                    record["selected_step"] = step
                    atomic_torch_save({"family": family, "state_dict": _cpu_state(model),
                                       "protocol_sha256": digest(protocol), "selected_step": step,
                                       "validation_loss": score, **replication}, checkpoint_path)
                record["best_validation_loss"] = best if math.isfinite(best) else None
                write_json(record_path, record)
                metrics = {"validation_admissible": int(math.isfinite(score))}
                if math.isfinite(score):
                    metrics["validation_loss"] = score
                if math.isfinite(best):
                    metrics["best_validation_loss"] = best
                emit(metrics, phase=f"research-validation-{phase_suffix}", step=step)
        failure_context = "checkpoint_selection"
        record["parameters_changed"] = any(not torch.equal(initial[name], parameter.detach().cpu())
                                           for name, parameter in model.named_parameters())
        if not record["parameters_changed"]:
            raise FloatingPointError("No trainable parameters changed; neural training is not substantiated")
        if record["selected_step"] is None:
            raise FloatingPointError("No checkpoint had admissible finite validation trajectories")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        record.update(status="COMPLETED", checkpoint_sha256=file_digest(checkpoint_path),
                      training_seconds=time.monotonic() - started,
                      selected_parameters_changed=any(not torch.equal(initial[name], parameter.detach().cpu())
                                                      for name, parameter in model.named_parameters()))
        write_json(record_path, record)
        emit({"training_seconds": record["training_seconds"],
              "selected_step": record["selected_step"],
              "selected_parameters_changed": int(record["selected_parameters_changed"])},
             phase=f"research-trained-{phase_suffix}")
        return model, record
    except FloatingPointError as error:
        # Numerical architecture failure is a recorded hypothesis outcome, not a silent omission.
        record.update(status="NUMERICAL_FAILURE", error=str(error), training_seconds=time.monotonic() - started,
                      failure_context=failure_context)
        if failure_context == "training" and record.get("optimizer_attempt") and hasattr(model, "audit_step"):
            try:
                with torch.no_grad():
                    _, stages = model.audit_step(parent["initial"].to(device=device, dtype=torch.float32), h,
                                                 Equation(parent["kappa"], parent["reaction_rate"]), geometry)
                    record["failure_stages"] = {name: {"finite": bool(torch.isfinite(value).all()),
                        "minimum": float(value.min()) if bool(torch.isfinite(value).all()) else None,
                        "maximum": float(value.max()) if bool(torch.isfinite(value).all()) else None}
                        for name, value in stages.items()}
            except Exception as audit_error:
                record["failure_stage_audit_error"] = str(audit_error)
        write_json(record_path, record)
        emit({"numerical_failures": 1, "optimizer_steps": record["steps"]},
             phase=f"research-failure-{phase_suffix}", step=record["steps"])
        return None, record
    except BaseException as error:
        record.update(status="INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "FAILED",
                      error=f"{type(error).__name__}: {error}", training_seconds=time.monotonic() - started)
        write_json(record_path, record)
        raise


def _safe_step(model, state, h, equation, geometry):
    if hasattr(model, "audit_step"):
        state, stages = model.audit_step(state, h, equation, geometry)
        for name, intermediate in stages.items():
            try:
                validate_state(intermediate)
            except ValueError as error:
                raise InvalidTrajectory(f"Invalid {name} stage: {error}") from error
    else:
        state = model(state, h, equation, geometry)
    try:
        validate_state(state)
    except ValueError as error:
        raise InvalidTrajectory(f"Invalid output state: {error}") from error
    return state


def whole_rollout(model, state, steps, equation, geometry, *, method: str):
    if method in ("adaptive_split", "coupled_rk4"):
        try:
            return classical_rollout(state, steps, equation, geometry, method, rtol=0., atol=TOLERANCE)[0]
        except ValueError as error:
            raise InvalidTrajectory(f"Classical {method}: {error}") from error
    for index, h in enumerate(steps):
        try:
            state = _safe_step(model, state, h, equation, geometry)
        except InvalidTrajectory as error:
            raise InvalidTrajectory(f"Macrostep {index}, h={h}: {error}") from error
    return state


def evaluate(models: dict, dataset: dict, protocol: dict, run_dir: Path, budget: Budget, device: str,
             training_records=None, *, replication: dict | None = None) -> dict:
    from . import build_research_model
    config = protocol["config"]
    geometry = Geometry(tuple(config["grid"]), (1., 1.))
    methods = {name: build_research_model(name, ndim=2).to(device=device, dtype=torch.float32)
               for name in CLASSICAL if name not in ("adaptive_split", "coupled_rk4")}
    methods["adaptive_split"] = None
    methods["coupled_rk4"] = None
    methods.update(models)
    parents = split_parents(dataset)["diagnostic"]
    rows, heldout = [], []
    phase = (f"research-evaluation-{replication['replicate_id']}-block-{replication['diagnostic_block']}"
             if replication is not None else "research-evaluation")
    emit({"timed_trajectories": 0}, phase=phase, completed=0,
         total=len(parents), unit="diagnostic parents")
    with torch.no_grad():
        for parent_index, parent in enumerate(parents, 1):
            equation = Equation(parent["kappa"], parent["reaction_rate"])
            initial = parent["initial"].to(device=device, dtype=torch.float32)
            reference = parent["references"][horizon_key(config["rollout_time"]) ]
            target = reference["state"].to(device=device, dtype=torch.float64)
            for name, model in methods.items():
                for h in config["heldout_horizons"]:
                    budget.check()
                    diagnostic = {"parent_id": parent["parent_id"], "family": name, "h": h, "device": device}
                    try:
                        first = whole_rollout(model, initial.clone(), [h], equation, geometry, method=name)
                        second = whole_rollout(model, first, [h], equation, geometry, method=name)
                        errors = []
                        for state, horizon in ((first, h), (second, 2 * h)):
                            ref = parent["references"][horizon_key(horizon)]
                            rms = float((state.double() - ref["state"].to(device)).square().mean().sqrt())
                            errors.append({"rms": rms, "upper": rms + ref["uncertainty"], "reference_uncertainty": ref["uncertainty"]})
                        diagnostic.update(status="COMPLETED", one_step=errors[0], two_step=errors[1])
                    except FloatingPointError as error:
                        diagnostic.update(status="INVALID_TRAJECTORY", error_message=str(error))
                    heldout.append(diagnostic)
                for h in config["benchmark_steps"]:
                    budget.check()
                    row = {"parent_id": parent["parent_id"], "family": name, "h": h, "device": device,
                           "state_precision": "float32", "reference_uncertainty": reference["uncertainty"],
                           "status": "RUNNING", "feasible": False}
                    steps = step_schedule(config["rollout_time"], h)
                    try:
                        def operation():
                            budget.check()
                            return whole_rollout(model, initial.clone(), steps, equation, geometry, method=name)
                        result, timing = measure(operation, device=device, warmup=1, repeats=config["timing_repeats"])
                        error = float((result.double() - target).square().mean().sqrt())
                        memory_ok = timing.get("soft_budget_passed", True) and not timing.get("hard_device_memory_warning", False)
                        row.update(status="COMPLETED", error=error, error_upper=error + reference["uncertainty"],
                                   feasible=bool(reference["accepted"] and error + reference["uncertainty"] <= TOLERANCE and memory_ok),
                                   timing=timing, macrosteps=len(steps))
                    except FloatingPointError as error:
                        # _safe_step's finite/bounds checks intentionally reject invalid trajectories.
                        row.update(status="INVALID_TRAJECTORY", error_message=f"{type(error).__name__}: {error}")
                    rows.append(row)
            context = {"replication": replication} if replication is not None else {}
            write_json(run_dir / "frontier.json", {"rows": rows, "scope": "diagnostic development parents; no confirmation", **context})
            write_json(run_dir / "heldout.json", {"rows": heldout, "checkpoint_selection": False,
                                                   "scope": "fresh diagnostic parents, off-training-grid horizons", **context})
            # Reporting stays outside measure(operation): solver timing includes
            # the scientific checks but excludes dashboard serialization.
            emit({"timed_trajectories": len(rows), "feasible_trajectories": sum(row["feasible"] for row in rows),
                  "invalid_trajectories": sum(row["status"] != "COMPLETED" for row in rows)},
                 phase=phase, step=parent_index,
                 completed=parent_index, total=len(parents), unit="diagnostic parents")
    summary = summarize_frontier(rows, config)
    if benchmark_suite(config) in ("neural-benchmarks", "neural-replication"):
        from .neural_comparison import summarize_neural_comparisons
        comparison = summarize_neural_comparisons(rows, heldout, config["families"], training_records)
        if replication is not None:
            comparison["replication"] = replication
        write_json(run_dir / "neural-comparisons.json", comparison)
        eligible = [row for row in comparison["comparisons"] if row["eligible"]]
        summary["neural_benchmark_results"] = {
            "comparison_file": "neural-comparisons.json", "baseline_families": comparison["baseline_families"],
            "comparison_count": len(comparison["comparisons"]), "eligible_comparison_count": len(eligible),
            "faster_than_neural_baseline_count": sum(row["speedup"] > 1 for row in eligible),
            "twenty_percent_faster_than_neural_baseline_count": sum(row["speedup"] >= 1.25 for row in eligible),
            "scope": "matched tolerance per-parent development frontier; training selection status retained"}
    if replication is not None:
        summary["replication"] = replication
    return summary


def summarize_frontier(rows: list[dict], config: dict) -> dict:
    parents = sorted({row["parent_id"] for row in rows})
    headroom_cases, comparisons = [], []
    for parent in parents:
        subset = [row for row in rows if row["parent_id"] == parent]
        coarse = {row["family"]: row for row in subset if row["h"] == max(config["benchmark_steps"])}
        if all(name in coarse and coarse[name]["status"] == "COMPLETED"
               and coarse[name]["error"] - coarse[name]["reference_uncertainty"] > TOLERANCE
               for name in ("split", "richardson_split")):
            headroom_cases.append(parent)
        classical = [row for row in subset if row["family"] in CLASSICAL and row["feasible"]]
        best = min(classical, key=lambda row: row["timing"]["wall_seconds_median"]) if classical else None
        for name in config["families"]:
            candidates = [row for row in subset if row["family"] == name and row["feasible"]]
            learned = min(candidates, key=lambda row: row["timing"]["wall_seconds_median"]) if candidates else None
            speedup = best["timing"]["wall_seconds_median"] / learned["timing"]["wall_seconds_median"] if best and learned else None
            comparisons.append({"parent_id": parent, "family": name,
                                "best_classical_family": best["family"] if best else None,
                                "best_classical_h": best["h"] if best else None,
                                "best_learned_h": learned["h"] if learned else None,
                                "speedup": speedup, "twenty_percent_faster": bool(speedup is not None and speedup >= 1.25),
                                "eligible": bool(best and learned)})
    return {"headroom": {"passed": bool(headroom_cases), "case_ids": headroom_cases,
                         "scope": "fixed coarse-step classical failure; no G0-G6 or pilot authorization"},
            "headroom_passed": bool(headroom_cases), "comparisons": comparisons,
            "diagnostic_parent_count": len(parents), "failed_trajectories": sum(row["status"] != "COMPLETED" for row in rows),
            "accuracy_tolerance": TOLERANCE, "cost_includes": "features, network, physics, state checks; FP32 all methods",
            "selection_scope": "post-hoc diagnostic accuracy/cost frontier; not deployable per-case step selection or confirmation"}


def run(protocol: dict, run_dir: Path, *, stop=None) -> dict:
    start = time.monotonic()
    budget = Budget(protocol["config"]["max_seconds"], stop)
    dataset = generate_data(protocol, run_dir, budget)
    geometry = Geometry(tuple(protocol["config"]["grid"]), (1., 1.))
    normalization, parent_ids = training_normalization(dataset, geometry)
    write_json(run_dir / "normalization.json", {"mean": normalization[0].tolist(), "std": normalization[1].tolist(),
                                                "parent_ids": parent_ids, "split": "train"})
    models, training = {}, []
    for family in protocol["config"]["families"]:
        model, record = train_family(family, dataset, protocol, run_dir, normalization, budget)
        training.append(record)
        if model is not None:
            models[family] = model
    summary = evaluate(models, dataset, protocol, run_dir, budget, "cpu", training_records=training)
    summary.update(status="COMPLETED", device="cpu", smoke=protocol["smoke"], training=training,
                   trained_family_count=len(models), elapsed_seconds=time.monotonic() - start,
                   scope=protocol["scope"], actual_neural_training=any(row["steps"] > 0 for row in training),
                   training_attempted=True)
    summary["benchmark_suite"] = benchmark_suite(protocol["config"])
    write_json(run_dir / "summary.json", summary)
    return summary


def benchmark(protocol: dict, source_run: Path, run_dir: Path, *, stop=None) -> dict:
    from . import build_research_model
    dataset = torch.load(source_run / "dataset.pt", map_location="cpu", weights_only=True)
    if dataset["protocol_sha256"] != digest(protocol):
        raise ValueError("Dataset does not belong to the immutable protocol")
    assert_parent_disjoint(dataset["parents"])
    declared = [{key: parent[key] for key in protocol["parents"][0]} for parent in dataset["parents"]]
    if declared != protocol["parents"]:
        raise ValueError("Dataset parent identities differ from the protocol")
    hashes = [state_digest(parent["initial"]) for parent in dataset["parents"]]
    if len(hashes) != len(set(hashes)) or hashes != [parent["initial_state_sha256"] for parent in dataset["parents"]]:
        raise ValueError("Dataset initial-state hashes are duplicated or corrupted")
    # Retain the exact verified input bytes with the GPU report for review.
    shutil.copyfile(source_run / "dataset.pt", run_dir / "dataset.pt")
    shutil.copyfile(source_run / "manifest.json", run_dir / "source_manifest.json")
    models = {}
    source_summary = json.loads((source_run / "summary.json").read_text())
    for record in source_summary["training"]:
        if record["status"] != "COMPLETED":
            continue
        family = record["family"]
        checkpoint = torch.load(source_run / "checkpoints" / f"{family}.pt", map_location="cpu", weights_only=True)
        if checkpoint["protocol_sha256"] != digest(protocol) or checkpoint["family"] != family:
            raise ValueError("Checkpoint family/protocol mismatch")
        model = build_research_model(family, ndim=2, width=protocol["config"]["width"]).to(device="cuda", dtype=torch.float32)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models[family] = model
    summary = evaluate(models, dataset, protocol, run_dir, Budget(min(600, protocol["config"]["max_seconds"]), stop),
                       "cuda", training_records=source_summary["training"])
    summary.update(status="COMPLETED", device="cuda", smoke=protocol["smoke"], source_run=str(source_run),
                   scope="Allocated GPU timing of frozen CPU checkpoints; no additional training or confirmation")
    summary["benchmark_suite"] = benchmark_suite(protocol["config"])
    write_json(run_dir / "summary.json", summary)
    return summary
