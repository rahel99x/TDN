"""Bounded development, frozen confirmation and audited policy stages.

No confirmation or acceptance labels choose an architecture, a checkpoint or
an optimization setting. Failed/initialization-selected training attempts are
retained. Timings include complete rollouts rather than correction-only calls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time

import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.invariants import validate_state
from tdn.research.experiment import atomic_torch_save, horizon_key
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json
from tdn.analysis.premix.neural import _RunBudget

from .data import load_bank, load_normalization, prepare_banks


def _table(path, stage, rows):
    write_json(Path(path) / "rows.json", {"schema": "tdn.agenda-rows/v1", "stage": stage, "rows": rows})


def _begin(protocol, path):
    from .protocol import validate_protocol
    validate_protocol(protocol)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("summary.json", "catalog.json", "data_manifest.json")):
        raise FileExistsError("Preserve previous agenda artifacts; use a fresh run directory")
    existing = path / "protocol.json"
    if existing.exists() and digest(json.loads(existing.read_text())) != digest(protocol):
        raise ValueError("Run protocol mismatch")
    write_json(existing, protocol)


def _seconds(protocol, stage):
    value = protocol.get("stage_seconds", protocol.get("budgets", {})).get(stage, protocol.get(f"{stage}_seconds", 1200))
    return value["seconds"] if isinstance(value, dict) else value


def _summary(protocol, path, stage, status, start, device, **values):
    result = {"schema": protocol["schema"], "stage": stage, "status": status, "profile": protocol["profile"],
              "device": device, "elapsed_seconds": time.monotonic() - start,
              "scientific_outcome": "DESCRIPTIVE_BOUNDED_AUDIT" if status == "COMPLETED" else "NOT_ESTABLISHED", **values}
    write_json(Path(path) / "summary.json", result)
    return result


def _seal(protocol, path):
    path = Path(path)
    files = sorted(p.relative_to(path).as_posix() for p in path.rglob("*")
                   if p.is_file() and p.name not in ("science_manifest.json", "stage.json", "manifest.json", "artifact_manifest.json", "COMPLETED")
                   and not p.name.endswith(".partial"))
    write_json(path / "science_manifest.json", {"schema": "tdn.agenda-science-manifest/v1", "protocol_sha256": digest(protocol),
                   "artifacts": {name: file_digest(path / name) for name in files}})


def verify_science(protocol, path):
    path = Path(path)
    value = json.loads((path / "science_manifest.json").read_text())
    if value["protocol_sha256"] != digest(protocol):
        raise ValueError("Prerequisite protocol differs")
    for name, fingerprint in value["artifacts"].items():
        relative = Path(name)
        target = path / relative
        if relative.is_absolute() or ".." in relative.parts or target.is_symlink() or not target.resolve().is_relative_to(path.resolve()):
            raise ValueError("Prerequisite artifact escapes its stage")
        if file_digest(target) != fingerprint:
            raise ValueError(f"Prerequisite artifact changed: {name}")
    if json.loads((path / "summary.json").read_text())["status"] != "COMPLETED":
        raise ValueError("An incomplete science stage cannot be a prerequisite")
    return value


def prepare(protocol, path, *, stop=None):
    path, start = Path(path), time.monotonic()
    _begin(protocol, path)
    budget = _RunBudget(_seconds(protocol, "prepare"), stop, "cpu")
    try:
        counts = prepare_banks(protocol, path, budget)
        rows = [{"record_type": "reference", "question_ids": ["Q4", "Q6"], **row}
                for row in json.loads((path / "references.json").read_text())["rows"]]
        _table(path, "prepare", rows)
        counts["rows"] = len(rows)
        summary = _summary(protocol, path, "prepare", "COMPLETED", start, "cpu", counts=counts,
            cohorts_sealed_separately=True, normalization_split="train", reference_uncertainty_is_certificate=False)
        _seal(protocol, path)
        return summary
    except BaseException as error:
        _summary(protocol, path, "prepare", "INTERRUPTED" if isinstance(error, (InterruptedError, TimeoutError)) else "FAILED", start, "cpu", error=str(error))
        raise


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def errors(value, reference):
    difference = value.detach().cpu().double() - reference["state"]
    mean = difference.mean()
    rms = float(difference.square().mean().sqrt())
    maximum = float(difference.abs().max())
    return {"error_rms": rms, "error_max": maximum, "mean_error": abs(float(mean)), "signed_mean_error": float(mean),
        "spatial_rms": float((difference - mean).square().mean().sqrt()),
        "uncertainty_rms": reference["uncertainty_rms"], "uncertainty_max": reference["uncertainty_max_bound"],
        "upper_rms": rms + reference["uncertainty_rms"], "upper_max": maximum + reference["uncertainty_max_bound"]}


def rollout(model, initial, schedule, equation, geometry, *, family=None, budget=None):
    """Complete solver operation. Classical diffusion halves are fused safely."""
    from tdn.numerics.subflows import diffusion_step, reaction_step
    from tdn.research.work_precision import prepare_step
    from tdn.research.compact_spatial import prepare_compact_step
    state = initial.clone()
    if family == "strang_diffusion_first":
        state = diffusion_step(state, schedule[0] / 2, equation, geometry)
        for index, h in enumerate(schedule):
            if budget:
                budget.check()
            state = reaction_step(state, h, equation)
            following = schedule[index + 1] if index + 1 < len(schedule) else 0
            state = diffusion_step(state, (h + following) / 2, equation, geometry)
            validate_state(state)
        return state
    prepared = {}
    for h in schedule:
        if budget:
            budget.check()
        if model is not None:
            state = model(state, h, equation, geometry)
        else:
            if h not in prepared:
                variant = "strang" if family == "strang_reaction_first" else family
                prepared[h] = prepare_compact_step(initial, h, equation, geometry, "gl3_fused") if variant == "gl3_fused" else prepare_step(initial, h, equation, geometry, "gl3_mean_spectral" if variant == "spectral_mean" else variant)
            state = prepared[h](state)
        try:
            validate_state(state)
        except ValueError as error:
            raise FloatingPointError(f"Inadmissible complete trajectory: {error}") from error
    return state


def timed_rollout(model, initial, schedule, equation, geometry, *, family=None, device="cpu", budget=None, repeats=1, warmup=0):
    for _ in range(warmup):
        rollout(model, initial, schedule, equation, geometry, family=family, budget=budget)
    values, result = [], None
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        result = rollout(model, initial, schedule, equation, geometry, family=family, budget=budget)
        _sync(device)
        values.append(time.perf_counter() - start)
    return result, {"timing_seconds": statistics.median(values), "timing_samples": values,
                    "complete_rollout": True, "preparation_included": True, "steps": len(schedule)}


def _model(trial, normalization, protocol, device):
    from tdn.research.agenda_neural import build_model
    return build_model(trial["family"], width=trial.get("width", protocol["width"]), modes=trial.get("modes", protocol["modes"]),
        normalization=normalization, t_ref=protocol["t_ref"], U_ref=protocol["U_ref"],
        base_orientation=trial.get("orientation", trial.get("base_orientation", "reaction-first"))).to(device=device, dtype=torch.float32)


def _state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _parent_loss(model, parents, h, protocol, device):
    n = protocol["train_grid"]
    first_parent = parents[0]
    equation = Equation(first_parent["kappa"], first_parent["reaction_rate"])
    if any((p["kappa"], p["reaction_rate"]) != (first_parent["kappa"], first_parent["reaction_rate"]) for p in parents):
        raise ValueError("A physical minibatch must share diffusion and reaction coefficients")
    geometry = Geometry((n, n), (1., 1.))
    initial = torch.cat([p["states"][str(n)] for p in parents]).to(device=device, dtype=torch.float32)
    first = model(initial, h, equation, geometry)
    second = model(first, h, equation, geometry)
    validate_state(first)
    validate_state(second)
    target1 = torch.cat([p["references"][f"discrete:{n}:{horizon_key(h)}"]["state"] for p in parents]).to(device=device, dtype=torch.float32)
    target2 = torch.cat([p["references"][f"discrete:{n}:{horizon_key(2 * h)}"]["state"] for p in parents]).to(device=device, dtype=torch.float32)
    loss = ((first - target1).square().mean() + protocol.get("two_step_loss_weight", .5) * (second - target2).square().mean()) / protocol["loss_scale"]**2
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite endpoint loss")
    component = model.correction_components(initial, h, equation, geometry)
    defect = target1 - component["base"]
    defect_rms = float(defect.detach().square().mean().sqrt())
    diagnostics = {"correction_rms": float(component["spatial_increment"].detach().square().mean().sqrt()),
                   "correction_max": float(component["spatial_increment"].detach().abs().max()),
                   "endpoint_correction_rms": float((first - component["base"]).detach().square().mean().sqrt()),
                   "physical_defect_rms": defect_rms, "physical_defect_max": float(defect.detach().abs().max())}
    diagnostics["correction_to_defect_ratio"] = diagnostics["endpoint_correction_rms"] / max(defect_rms, 1e-12)
    return loss, diagnostics


def _validate(model, validation, protocol, device, budget):
    by_regime = {}
    model.eval()
    with torch.no_grad():
        for parent in validation:
            for h in protocol["validation_horizons"]:
                budget.check()
                loss, diagnostics = _parent_loss(model, [parent], h, protocol, device)
                by_regime.setdefault(parent.get("spectrum", "mixed"), []).append({"loss": float(loss), **diagnostics})
    detail = {key: {field: statistics.fmean(row[field] for row in rows) for field in rows[0]} for key, rows in by_regime.items()}
    all_losses = [row["loss"] for rows in by_regime.values() for row in rows]
    return statistics.fmean(all_losses), detail


def _train(trial, protocol, train, validation, normalization, path, stage, device, budget):
    identifier = trial["trial_id"]
    torch.manual_seed(trial["seed"])
    model = _model(trial, normalization, protocol, device)
    initial_state = _state(model)
    checkpoint_base = Path(path) / "checkpoints" / identifier
    record_path = Path(path) / "training" / f"{identifier}.json"
    record = {"record_type": "optimization" if stage == "optimize" else "training", "question_ids": ["Q1", "Q2", "Q3", "Q5", "Q7"],
        **trial, "stage": stage, "status": "RUNNING", "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_record": f"training/{identifier}.json", "training_parent_ids": [p["parent_id"] for p in train],
        "validation_parent_ids": [p["parent_id"] for p in validation], "fresh_cohort_opened": False,
        "history": [], "gradient_history": [], "architecture": model.architecture_metadata(), "steps": 0}
    for suffix in ("initial", "selected"):
        atomic_torch_save({"protocol_sha256": digest(protocol), "trial": trial, "state_dict": initial_state, "selected_step": 0}, checkpoint_base.with_name(identifier + f"-{suffix}.pt"))
    buckets = {}
    for p in train:
        for h in protocol["train_horizons"]:
            buckets.setdefault((p["kappa"], p["reaction_rate"], h), []).append(p)
    groups = list(buckets.items())
    generator = torch.Generator().manual_seed(trial["seed"] + 1)
    schedule = []
    for _ in range(trial["updates"]):
        bucket = int(torch.randint(len(groups), (1,), generator=generator))
        indices = torch.randint(len(groups[bucket][1]), (trial["batch_size"],), generator=generator).tolist()
        schedule.append((bucket, indices))
    record["sample_schedule_sha256"] = digest(schedule)
    record["batch_design"] = "actual shared-physics/state-grid/horizon minibatches; sampled with replacement inside a frozen bucket"
    record["training_examples"] = trial["updates"] * trial["batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=trial["learning_rate"])
    best, start = math.inf, time.monotonic()
    try:
        for step in range(trial["updates"] + 1):
            budget.check()
            if step:
                bucket, indices = schedule[step - 1]
                (_, _, h), values = groups[bucket]
                parents = [values[index] for index in indices]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, diagnostics = _parent_loss(model, parents, h, protocol, device)
                loss.backward()
                gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                if len(gradients) != sum(p.requires_grad for p in model.parameters()):
                    raise FloatingPointError("A trainable parameter has no gradient in the attempted batch")
                if not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError("Nonfinite parameter gradient")
                norm = math.sqrt(sum(float(g.square().sum()) for g in gradients))
                maxgrad = max((float(g.abs().max()) for g in gradients), default=0.)
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.get("gradient_clip", 10.))
                clipped_norm = math.sqrt(sum(float(g.square().sum()) for g in gradients))
                optimizer.step()
                record["steps"] = step
                record["gradient_history"].append({"step": step, "training_loss": float(loss.detach()), "gradient_l2_before_clip": norm,
                    "gradient_max": maxgrad, "gradient_l2_after_clip": clipped_norm,
                    "parent_ids": [p["parent_id"] for p in parents], "distinct_parents": len(set(p["parent_id"] for p in parents)),
                    "horizon": h, "spectrum": [p.get("spectrum") for p in parents], **diagnostics})
            if step % protocol["validation_every"] == 0 or step == trial["updates"]:
                score, regimes = _validate(model, validation, protocol, device, budget)
                record["history"].append({"step": step, "validation_loss": score, "by_regime": regimes})
                if math.isfinite(score) and score < best:
                    best = score
                    record["selected_step"] = step
                    atomic_torch_save({"protocol_sha256": digest(protocol), "trial": trial, "state_dict": _state(model),
                        "selected_step": step, "validation_loss": score}, checkpoint_base.with_name(identifier + "-selected.pt"))
                write_json(record_path, record)
                from tdn.reporting import emit
                measured = {"validation_loss": score, "learning_rate": trial["learning_rate"],
                            "batch_size": trial["batch_size"]}
                measured.update({f"validation_loss.{regime}": values["loss"] for regime, values in regimes.items()})
                if record["gradient_history"]:
                    latest = record["gradient_history"][-1]
                    measured.update({key: latest[key] for key in ("training_loss", "gradient_l2_before_clip",
                        "gradient_l2_after_clip", "gradient_max", "correction_rms", "correction_max", "endpoint_correction_rms")})
                emit(measured, phase=f"agenda/{stage}/{identifier}", step=step,
                     completed=step, total=trial["updates"], unit="updates")
        final_state = _state(model)
        atomic_torch_save({"protocol_sha256": digest(protocol), "trial": trial, "state_dict": final_state,
                          "selected_step": trial["updates"]}, checkpoint_base.with_name(identifier + "-trained.pt"))
        selected_path = checkpoint_base.with_name(identifier + "-selected.pt")
        selected = torch.load(selected_path, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["state_dict"], strict=True)
        record.update(status="COMPLETED", best_validation_loss=best,
            selection="TRAINED_CHECKPOINT" if record["selected_step"] > 0 else "SELECTED_INITIALIZATION",
            parameters_changed=any(not torch.equal(initial_state[name], value) for name, value in final_state.items()),
            checkpoint=selected_path.relative_to(path).as_posix(), checkpoint_sha256=file_digest(selected_path),
            trained_checkpoint_sha256=file_digest(checkpoint_base.with_name(identifier + "-trained.pt")))
    except (FloatingPointError, ValueError) as error:
        record.update(status="NUMERICAL_FAILURE", error=str(error), selection="NO_ELIGIBLE_CHECKPOINT")
        model = None
    except BaseException as error:
        record.update(status="INTERRUPTED" if isinstance(error, (TimeoutError, InterruptedError)) else "FAILED", error=str(error))
        write_json(record_path, record)
        raise
    _sync(device)
    record["training_seconds"] = time.monotonic() - start
    write_json(record_path, record)
    return model, record


def _resolved_trials(protocol, stage, prerequisites):
    trials = [dict(trial) for trial in protocol["stage_trials"][stage]]
    if stage in ("compression", "kernel") and "optimize" in prerequisites:
        values = json.loads((Path(prerequisites["optimize"]) / "summary.json").read_text())
        selected = values.get("selected_hyperparameters", {}).get("source")
        if selected:
            for trial in trials:
                trial["learning_rate"], trial["batch_size"] = selected["learning_rate"], selected["batch_size"]
                trial["hyperparameter_source"] = "source validation-only optimization"
    if stage in ("compression", "kernel") and "controls" in prerequisites:
        records = json.loads((Path(prerequisites["controls"]) / "catalog.json").read_text())["records"]
        eligible = [record for record in records if record["family"] == "source" and record["status"] == "COMPLETED"]
        if eligible:
            chosen = min(eligible, key=lambda r: (r["best_validation_loss"], r["trial_id"]))
            for trial in trials:
                trial["base_orientation"] = chosen.get("base_orientation", "reaction-first")
                trial["orientation"] = trial["base_orientation"]
                trial["base_selection_trial"] = chosen["trial_id"]
    return trials


def _catalog(protocol, prerequisites):
    result = []
    for stage in ("controls", "compression", "kernel"):
        if stage not in prerequisites:
            continue
        base = Path(prerequisites[stage])
        verify_science(protocol, base)
        value = json.loads((base / "catalog.json").read_text())
        for record in value["records"]:
            if record["status"] == "COMPLETED":
                result.append({**record, "source_stage": stage, "source_dir": str(base)})
    return result


def select_confirmation(protocol, catalog):
    """Predeclared family coverage, with validation-only choices within a slot."""
    plan = protocol.get("confirmation_selection", protocol.get("confirm_selection", []))
    if isinstance(plan, dict):
        selected = []
        for family in plan["mandatory_compression_families"]:
            for seed in protocol["seeds"]:
                candidates = [r for r in catalog if r["family"] == family and r["source_stage"] == "compression" and r["seed"] == seed]
                if candidates:
                    selected.append(min(candidates, key=lambda r: (r["best_validation_loss"], r["trial_id"])))
        for family in plan["additional_control_families"]:
            candidates = [r for r in catalog if r["family"] == family and r["source_stage"] == "controls"]
            if candidates:
                selected.append(min(candidates, key=lambda r: (r["best_validation_loss"], r["trial_id"])))
        candidates = [r for r in catalog if r["source_stage"] == "kernel"]
        if candidates:
            selected.append(min(candidates, key=lambda r: (r["best_validation_loss"], r["trial_id"])))
        return selected[:plan["max_variants"]]
    if not plan:
        names = ("local_gate", "local_gate_time", "source", "source_time", "source_closure", "source_time_closure",
                 "precompress_source", "fno_source", "fno_source_matched", "pair_rank2", "pair_rank4", "pair_rank8")
        plan = [{"family": family} for family in names]
    selected = []
    for slot in plan:
        if isinstance(slot, str):
            slot = {"family": slot}
        candidates = [record for record in catalog if record["family"] == slot["family"]
                      and ("stage" not in slot or record["source_stage"] == slot["stage"])
                      and ("orientation" not in slot or record.get("orientation", record.get("base_orientation", "reaction-first")) == slot["orientation"])
                      and ("seed" not in slot or record["seed"] == slot["seed"])]
        if slot["family"] == "best_eligible_pair":
            candidates = [record for record in catalog if record["source_stage"] == "kernel"]
        if candidates:
            chosen = min(candidates, key=lambda r: (r["best_validation_loss"], r["trial_id"]))
            if chosen["trial_id"] not in {record["trial_id"] for record in selected}:
                selected.append(chosen)
    return selected[:protocol.get("max_confirmation_models", 12)]


def confirmation_ledger(protocol, catalog, selected):
    plan = protocol["confirmation_selection"]
    if not isinstance(plan, list):
        return []
    rows = []
    for index, slot in enumerate(plan):
        family = slot["family"]
        candidates = [record for record in selected if
            (record["family"] == family or family == "best_eligible_pair" and record["source_stage"] == "kernel")
            and ("stage" not in slot or record["source_stage"] == slot["stage"])
            and ("seed" not in slot or record["seed"] == slot["seed"])
            and ("orientation" not in slot or record.get("orientation", record.get("base_orientation")) == slot["orientation"])]
        chosen = min(candidates, key=lambda r: (r["best_validation_loss"], r["trial_id"])) if candidates else None
        rows.append({"record_type": "confirmation", "question_ids": ["Q4", "Q6"], "row_kind": "planned_slot",
            "slot_index": index, "slot": slot, "status": chosen["selection"] if chosen else "NO_ELIGIBLE_CHECKPOINT",
            "selected_trial_id": chosen["trial_id"] if chosen else None,
            "checkpoint_selection": chosen["selection"] if chosen else "NO_ELIGIBLE_CHECKPOINT",
            "efficacy_inconclusive": chosen is None or chosen["selection"] != "TRAINED_CHECKPOINT",
            "validation_only_selection": True})
    return rows


def load_selected_models(protocol, records, normalization, device):
    models = {}
    for record in records:
        path = Path(record["source_dir"]) / record["checkpoint"]
        if file_digest(path) != record["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["protocol_sha256"] != digest(protocol):
            raise ValueError("Checkpoint protocol mismatch")
        model = _model(record, normalization, protocol, device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        models[record["trial_id"]] = model
    return models


def _capacity_rows(protocol, records, models, validation, device, budget):
    n = protocol["train_grid"]
    parent = validation[0]
    u = parent["states"][str(n)].to(device=device, dtype=torch.float32)
    equation, geometry = Equation(parent["kappa"], parent["reaction_rate"]), Geometry((n, n), (1., 1.))
    measured = {}
    with torch.no_grad():
        for record in records:
            if record["status"] != "COMPLETED":
                continue
            _, timing = timed_rollout(models[record["trial_id"]], u, [protocol["validation_horizons"][0]], equation, geometry,
                device=device, budget=budget, warmup=protocol.get("timing_warmup", 1), repeats=protocol.get("timing_repeats", 3))
            measured[record["trial_id"]] = timing["timing_seconds"]
    rows = []
    for record in records:
        if record["status"] != "COMPLETED":
            continue
        base = next((r for r in records if r["family"] == "source" and r["seed"] == record["seed"] and r["status"] == "COMPLETED"), None)
        if base:
            rows.append({"record_type": "capacity", "question_ids": ["Q4"], "trial_id": record["trial_id"],
                "family": record["family"], "seed": record["seed"], "status": "OBSERVED", "selection": record["selection"],
                "paired_selection_inconclusive": record["selection"] == "SELECTED_INITIALIZATION" or base["selection"] == "SELECTED_INITIALIZATION",
                "parameter_count": record["parameter_count"], "source_parameter_count": base["parameter_count"],
                "parameter_relative_difference": abs(record["parameter_count"] - base["parameter_count"]) / base["parameter_count"],
                "timing_seconds": measured[record["trial_id"]], "source_timing_seconds": measured[base["trial_id"]],
                "cost_ratio": measured[record["trial_id"]] / measured[base["trial_id"]],
                "actual_support_matches": record["architecture"]["fourier_support"] == base["architecture"]["fourier_support"],
                "architecture": record["architecture"], "matching_data": "validation parent only; not fresh diagnostics"})
    return rows


def _trained_limits(protocol, records, models, device, budget):
    from tdn.research.agenda_neural import physical_base
    generator = torch.Generator().manual_seed(970019)
    state = (.25 + .5 * torch.rand((1, 1, 8, 8), generator=generator)).to(device)
    geometry = Geometry((8, 8), (1., 1.))
    cases = [("constant", torch.full_like(state, .4), .12, Equation(.003, 2.)),
             ("zero_reaction", state, .12, Equation(.003, 0.)),
             ("zero_diffusion", state, .12, Equation(0., 2.)),
             ("zero_time", state, 0., Equation(.003, 2.))]
    rows = []
    with torch.no_grad():
        for record in records:
            if record["status"] != "COMPLETED":
                continue
            model = models[record["trial_id"]]
            for case, value, h, equation in cases:
                budget.check()
                predicted = model(value, h, equation, geometry)
                base = physical_base(value, h, equation, geometry, record.get("base_orientation", record.get("orientation", "reaction-first")))
                error = float((predicted - base).abs().max())
                rows.append({"record_type": "case", "question_ids": ["Q1"], "trial_id": record["trial_id"],
                    "family": record["family"], "case": case, "required": True, "status": "PASS" if error <= 2e-6 else "FAIL",
                    "absolute_error": error, "tolerance": 2e-6, "after_checkpoint_freeze": True,
                    "checkpoint_sha256": record["checkpoint_sha256"], "fresh_cohort_opened": False})
                if error > 2e-6:
                    raise FloatingPointError(f"Trained physical null limit failed: {record['trial_id']}/{case}")
    return rows


def _frontiers(protocol, candidates):
    groups = {}
    for row in candidates:
        key = (row["parent_id"], row["grid_size"], row["track"], horizon_key(row["horizon"]), row["trial_id"])
        groups.setdefault(key, []).append(row)
    result = []
    for rows in groups.values():
        for norm in ("rms", "max"):
            for target in protocol["targets"]:
                eligible = [r for r in rows if r["status"] == "COMPLETED" and r[f"upper_{norm}"] <= target]
                best = min(eligible, key=lambda r: r["timing_seconds"]) if eligible else None
                row = {key: rows[0][key] for key in ("parent_id", "continuous_field_id", "grid_size", "track", "horizon", "trial_id", "family", "seed")}
                row.update(record_type="frontier", question_ids=["Q4", "Q6", "Q7"], norm=norm, target=target,
                    status="FEASIBLE" if best else "INCONCLUSIVE" if any(r.get(f"uncertainty_{norm}", 0) > target / 2 for r in rows) else "NO_FEASIBLE_CANDIDATE",
                    timing_seconds=best["timing_seconds"] if best else None, selected_schedule=best["schedule"] if best else None,
                    selected_error_upper=best[f"upper_{norm}"] if best else None, selection_rule="reference-informed posthoc fixed schedules; not deployment policy")
                result.append(row)
    return result


def evaluate_criteria(protocol, candidates, frontiers):
    """Evaluate frozen operational gates without changing any selected model."""
    criteria, output = protocol["success_criteria"], []
    target = criteria["primary_target"]
    learned = sorted({row["trial_id"] for row in frontiers if row["seed"] is not None})
    identity = lambda row: (row["parent_id"], row["grid_size"], row["track"], horizon_key(row["horizon"]))
    for identifier in learned:
        for track in ("discrete", "continuum"):
            for norm in ("rms", "max"):
                own = [row for row in frontiers if row["trial_id"] == identifier and row["track"] == track and row["norm"] == norm and row["target"] == target]
                if not own:
                    continue
                feasible = [row for row in own if row["status"] == "FEASIBLE"]
                cases = [row for row in candidates if row["trial_id"] == identifier and row["track"] == track]
                spectra = {row["parent_id"]: row["factors"]["spectrum"] for row in cases}
                regimes = {name: [row for row in own if spectra[row["parent_id"]] == name] for name in set(spectra.values())}
                coverage = len(feasible) / len(own)
                regime_coverage = {name: sum(row["status"] == "FEASIBLE" for row in values) / len(values) for name, values in regimes.items()}
                selected_candidates = [next(row for row in cases if identity(row) == identity(frontier) and row["schedule"] == frontier["selected_schedule"])
                                       for frontier in feasible]
                regressions = [row for row in selected_candidates if row["spatial_rms"] > max(criteria["spatial_absolute_floor"],
                    criteria["maximum_spatial_regression_ratio"] * row.get("base_spatial_rms", math.inf))]
                for classical in ("etdrk4", "gl3_fused"):
                    base = {identity(row): row for row in frontiers if row["family"] == classical and row["track"] == track and row["norm"] == norm and row["target"] == target}
                    joint = [(row, base[identity(row)]) for row in feasible if identity(row) in base and base[identity(row)]["status"] == "FEASIBLE"]
                    speedups = [b["timing_seconds"] / a["timing_seconds"] for a, b in joint]
                    missing = sum(b["status"] == "FEASIBLE" and next((r for r in own if identity(r) == key), {}).get("status") != "FEASIBLE" for key, b in base.items())
                    gates = {"overall_coverage": coverage >= criteria["minimum_overall_coverage"],
                        "each_regime_coverage": all(value >= criteria["minimum_each_regime_coverage"] for value in regime_coverage.values()),
                        "spatial_regressions": not regressions,
                        "classical_coverage_retained": missing == 0,
                        "median_speedup": bool(speedups) and statistics.median(speedups) >= criteria["minimum_median_speedup_over_classical"]}
                    output.append({"record_type": "comparison", "question_ids": ["Q4", "Q6", "Q7"], "trial_id": identifier,
                        "family": own[0]["family"], "seed": own[0]["seed"], "track": track, "norm": norm, "target": target,
                        "classical": classical, "status": "PASS" if all(gates.values()) else "INCONCLUSIVE" if any(row["status"] == "INCONCLUSIVE" for row in own) else "FAIL",
                        "gates": gates, "coverage": coverage, "regime_coverage": regime_coverage,
                        "feasible_cases": len(feasible), "declared_cases": len(own), "spatial_regressions": len(regressions),
                        "lost_classical_feasible_cases": missing, "jointly_feasible_cases": len(joint),
                        "median_speedup": statistics.median(speedups) if speedups else None,
                        "timing_wins": sum(value > 1 for value in speedups), "timing_losses": sum(value < 1 for value in speedups),
                        "independence": "paired physical cases; field-cluster repetitions and grids are not independent samples",
                        "criteria_frozen_before_confirmation": True})
    return output


def joint_criteria(protocol, criteria, records):
    result = []
    for record in records:
        for track in ("discrete", "continuum"):
            rows = [r for r in criteria if r["trial_id"] == record["trial_id"] and r["track"] == track]
            if not rows:
                continue
            required = {(norm, classical) for norm in protocol["norms"] for classical in ("etdrk4", "gl3_fused")}
            complete = {(r["norm"], r["classical"]) for r in rows} == required
            trained = record["selection"] == "TRAINED_CHECKPOINT"
            passed = complete and trained and all(r["status"] == "PASS" for r in rows)
            result.append({"record_type": "comparison", "question_ids": ["Q4", "Q6", "Q7"],
                "row_kind": "joint_criterion", "trial_id": record["trial_id"], "family": record["family"], "seed": record["seed"],
                "track": track, "status": "PASS" if passed else "INCONCLUSIVE" if not trained or not complete or any(r["status"] == "INCONCLUSIVE" for r in rows) else "FAIL",
                "checkpoint_selection": record["selection"], "both_norms_and_classical_controls_required": True,
                "criteria_complete": complete, "trained_checkpoint_required": True,
                "scope": "separate matched-norm frontiers; no simultaneous RMS/max schedule or statistical significance claim"})
    return result


def _confirm(protocol, dataset, prerequisites, path, device, budget):
    catalog = _catalog(protocol, prerequisites)
    selected = select_confirmation(protocol, catalog)
    ledger = confirmation_ledger(protocol, catalog, selected)
    normalization = load_normalization(protocol, dataset)
    models = load_selected_models(protocol, selected, normalization, device)
    frozen = {"protocol_sha256": digest(protocol), "selection_rule": "predeclared mechanism slots; minimum validation loss only",
              "confirmation_seen_during_selection": False, "records": selected, "planned_slots": ledger}
    write_json(path / "frozen_checkpoints.json", frozen)
    fingerprint = file_digest(path / "frozen_checkpoints.json")
    # The first read of confirmation tensors is strictly after the catalog is frozen.
    parents = load_bank(protocol, dataset, "confirmation")
    candidates = []
    schedules = protocol.get("confirm_schedules", protocol.get("confirmation_schedules", []))
    classical = protocol.get("classical", ["strang_reaction_first", "strang_diffusion_first", "etdrk4", "gl3_fused", "spectral_mean"])
    methods = [(family, family, None, None) for family in classical] + [(r["trial_id"], r["family"], models[r["trial_id"]], r) for r in selected]
    with torch.no_grad():
        for parent in parents:
            equation = Equation(parent["kappa"], parent["reaction_rate"])
            for grid in protocol["grids"]:
                n = grid if isinstance(grid, int) else grid[0]
                geometry = Geometry((n, n), (1., 1.))
                initial = parent["states"][str(n)].to(device=device, dtype=torch.float32)
                order = torch.randperm(len(methods), generator=torch.Generator().manual_seed(parent["seed"] + n)).tolist()
                for method_index in order:
                    identifier, family, model, record = methods[method_index]
                    for schedule_index, schedule in enumerate(schedules):
                        budget.check()
                        h = round(sum(schedule), 12)
                        row = {"record_type": "confirmation", "question_ids": ["Q1", "Q2", "Q4", "Q6", "Q7"],
                            "parent_id": parent["parent_id"], "continuous_field_id": parent.get("continuous_field_id", parent["parent_id"]),
                            "trial_id": identifier, "family": family, "seed": record["seed"] if record else None,
                            "grid_size": n, "schedule": schedule, "schedule_index": schedule_index, "horizon": h,
                            "factors": {key: parent.get(key) for key in ("mean", "variance", "amplitude", "spectrum", "phase", "kappa", "reaction_rate")},
                            "checkpoint_selection": record["selection"] if record else "CLASSICAL", "device": device,
                            "base_orientation": record.get("base_orientation", record.get("orientation", "reaction-first")) if record else
                                "diffusion-first" if family == "strang_diffusion_first" else "reaction-first",
                            "checkpoint_freeze_sha256": fingerprint, "status": "COMPLETED"}
                        try:
                            value, timing = timed_rollout(model, initial, schedule, equation, geometry, family=family if record is None else None,
                                device=device, budget=budget, repeats=protocol.get("timing_repeats", 3), warmup=protocol.get("timing_warmup", 1))
                            row.update(timing)
                        except (FloatingPointError, ValueError) as error:
                            row.update(status="NUMERICAL_FAILURE", error=str(error))
                            value = None
                        for track in ("discrete", "continuum"):
                            key = f"{track}:{n}:{horizon_key(h)}"
                            if key not in parent["references"]:
                                continue
                            targetrow = {**row, "track": track}
                            if value is not None:
                                targetrow.update(errors(value, parent["references"][key]))
                            candidates.append(targetrow)
            _table(path, "confirm", candidates)
    base = {(r["parent_id"], r["grid_size"], r["track"], r["schedule_index"], r["base_orientation"]): r for r in candidates
            if r["family"] in ("strang", "strang_reaction_first", "strang_diffusion_first") and r["status"] == "COMPLETED"}
    for row in candidates:
        paired = base.get((row["parent_id"], row["grid_size"], row["track"], row["schedule_index"], row["base_orientation"]))
        if paired and row["status"] == "COMPLETED":
            for key in ("error_rms", "error_max", "mean_error", "spatial_rms"):
                row[f"base_{key}"] = paired[key]
                row[f"{key}_vs_base"] = row[key] / paired[key] if paired[key] > 0 else None
            row["spatial_regression"] = row["spatial_rms"] > paired["spatial_rms"]
    frontiers = _frontiers(protocol, candidates)
    criteria = evaluate_criteria(protocol, candidates, frontiers)
    joint = joint_criteria(protocol, criteria, selected)
    write_json(path / "candidates.json", {"rows": candidates})
    write_json(path / "frontiers.json", {"rows": frontiers})
    write_json(path / "criteria.json", {"rows": criteria + joint, "frozen_success_criteria": protocol["success_criteria"]})
    _table(path, "confirm", candidates + frontiers + criteria + joint + ledger)
    if file_digest(path / "frozen_checkpoints.json") != fingerprint:
        raise ValueError("Confirmation altered its frozen checkpoint catalog")
    return {"models": len(selected), "candidates": len(candidates), "frontiers": len(frontiers),
            "continuous_field_clusters": len({p.get("continuous_field_id", p["parent_id"]) for p in parents}),
            "phase_seed_clusters": len({p["seed"] for p in parents}),
            "physical_parents": len(parents), "confirmed_criteria": protocol.get("success_criteria", {}),
            "planned_slots": len(ledger), "missing_slots": sum(row["status"] == "NO_ELIGIBLE_CHECKPOINT" for row in ledger),
            "criteria_rows": len(criteria), "descriptive_per_norm_passes": sum(row["status"] == "PASS" for row in criteria),
            "joint_criteria_rows": len(joint), "joint_criteria_passes": sum(row["status"] == "PASS" for row in joint),
            "scientific_outcome": "DESCRIPTIVE_JOINT_OPERATIONAL_GATE_PASS" if any(row["status"] == "PASS" for row in joint) else "NO_JOINT_PREDECLARED_GATE_PASS",
            "fresh_confirmation_used_for_tuning": False}


def run_stage(protocol, stage, dataset, prerequisites, path, *, device="cpu", stop=None):
    path, start = Path(path), time.monotonic()
    _begin(protocol, path)
    budget = _RunBudget(_seconds(protocol, stage), stop, device)
    prerequisites = {name: Path(value) for name, value in prerequisites.items()}
    for source in prerequisites.values():
        if (source / "science_manifest.json").exists():
            verify_science(protocol, source)
    try:
        if stage == "confirm":
            counts = _confirm(protocol, Path(dataset), prerequisites, path, device, budget)
        elif stage == "policy":
            from .policy import run_policy
            counts = run_policy(protocol, Path(dataset), prerequisites, path, device, budget)
        elif stage in ("controls", "optimize", "compression", "kernel"):
            records, rows, models = [], [], {}
            trials = _resolved_trials(protocol, stage, prerequisites)
            if stage == "kernel":
                structure = json.loads((prerequisites["structure"] / "summary.json").read_text())
                promising = structure.get("rank_promising", structure.get("rank_summary", {}).get("promising", False))
                if isinstance(promising, list):
                    promising = bool(promising)
                if not promising:
                    trials = []
                    for trial in protocol["stage_trials"][stage]:
                        record = {**trial, "record_type": "kernel", "question_ids": ["Q2", "Q3"], "status": "NOT_PROMOTED",
                                  "selection": "NO_ELIGIBLE_CHECKPOINT", "reason": "Offline low-rank response audit did not meet frozen promotion criterion"}
                        records.append(record)
                        rows.append(record)
            train, validation = load_bank(protocol, dataset, "train"), load_bank(protocol, dataset, "validation")
            normalization = load_normalization(protocol, dataset)
            for trial in trials:
                model, record = _train(trial, protocol, train, validation, normalization, path, stage, device, budget)
                records.append(record)
                if model is not None:
                    models[record["trial_id"]] = model
                rows.append({key: value for key, value in record.items() if key not in ("history", "gradient_history")})
                _table(path, stage, rows)
            if stage == "compression":
                rows.extend(_capacity_rows(protocol, records, models, validation, device, budget))
            limits = _trained_limits(protocol, records, models, device, budget)
            rows.extend(limits)
            write_json(path / "trained_physical_limits.json", {"rows": limits})
            write_json(path / "catalog.json", {"protocol_sha256": digest(protocol), "checkpoint_selection": "validation only including initialization",
                "fresh_cohort_opened": False, "records": [{key: value for key, value in r.items() if key not in ("history", "gradient_history")} for r in records]})
            counts = {"declared_trials": len(protocol["stage_trials"][stage]), "attempted_trials": sum(r["status"] != "NOT_PROMOTED" for r in records),
                      "completed_trials": sum(r["status"] == "COMPLETED" for r in records),
                      "trained_selected": sum(r.get("selection") == "TRAINED_CHECKPOINT" for r in records),
                      "initializations_selected": sum(r.get("selection") == "SELECTED_INITIALIZATION" for r in records)}
            if stage == "optimize":
                choices = {}
                for family in {r["family"] for r in records}:
                    eligible = [r for r in records if r["family"] == family and r["status"] == "COMPLETED"]
                    if eligible:
                        choice = min(eligible, key=lambda r: (r["best_validation_loss"], r["trial_id"]))
                        choices[family] = {key: choice[key] for key in ("trial_id", "learning_rate", "batch_size", "best_validation_loss", "selection")}
                counts["selected_hyperparameters"] = choices
            _table(path, stage, rows)
        else:
            raise ValueError(f"Unsupported agenda training stage: {stage}")
        summary = _summary(protocol, path, stage, "COMPLETED", start, device, **counts,
            counts={"rows": len(json.loads((path / "rows.json").read_text())["rows"])},
            memory=budget.memory, scope="bounded mechanism tests; no FNO-paper reproduction or universal superiority claim")
        _seal(protocol, path)
        return summary
    except BaseException as error:
        _summary(protocol, path, stage, "INTERRUPTED" if isinstance(error, (TimeoutError, InterruptedError)) else "FAILED", start, device, error=f"{type(error).__name__}: {error}")
        raise
