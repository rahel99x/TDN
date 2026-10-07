"""Bounded DF tuning, frozen checkpoints and independent all-schedule tests.

Selection consumes only train/validation. Every tuning attempt and selected
initialization is retained. Confirmation cannot silently select a good seed,
grid, time order or frontier in place of reporting the prescribed schedules.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import rhs
from tdn.research.experiment import atomic_torch_save, horizon_key
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json

from .models import FAMILIES, MECHANISMS, build_model


class TrialNumericalFailure(FloatingPointError):
    """Only explicit nonfinite numerical observations are recoverable.

    ValueError, RuntimeError, allocation failures, deadlines and interrupts
    are deliberately not caught by the training-trial recovery path.
    """


class RolloutNumericalFailure(FloatingPointError):
    """A measured nonfinite/inadmissible state, not an arbitrary runtime error."""
    def __init__(self, code, completed_steps=0):
        super().__init__(code)
        self.code, self.completed_steps = code, completed_steps


def _check_prediction(state, completed_steps=0):
    if not bool(torch.isfinite(state).all()):
        raise RolloutNumericalFailure("NONFINITE_PREDICTED_STATE", completed_steps)
    tolerance = 64 * torch.finfo(state.dtype).eps
    if bool((state < -tolerance).any()) or bool((state > 1 + tolerance).any()):
        raise RolloutNumericalFailure("PREDICTED_STATE_OUTSIDE_ADMISSIBLE_INTERVAL", completed_steps)

for _classical_family in ("rf_base", "etdrk4", "gl3_fused"):
    MECHANISMS[_classical_family] = ["M00", "M16"]


class ClassicalControl(torch.nn.Module):
    """Prepared state-independent classical constants with explicit setup cost."""
    def __init__(self, family):
        super().__init__()
        self.family, self.is_c1, self.direct = family, False, False
        self._prepared = {}
        self.selection_metadata = {"family": family, "selection": "UNTRAINED_PHYSICAL_CONTROL",
                                   "updates_selected": 0, "parameters": 0, "seed": None}

    def _key(self, u, h, eq, geometry):
        return (float(h), eq, geometry, str(u.device), u.dtype)

    def prepare(self, u, schedule, eq, geometry):
        from tdn.research.work_precision import prepare_step
        from tdn.research.compact_spatial import prepare_compact_step
        for h in schedule:
            key = self._key(u, h, eq, geometry)
            if key not in self._prepared:
                self._prepared[key] = (prepare_compact_step(u, h, eq, geometry, "gl3_fused")
                    if self.family == "gl3_fused" else prepare_step(u, h, eq, geometry,
                        "strang" if self.family == "rf_base" else self.family))

    def forward(self, u, h, eq, geometry):
        self.prepare(u, [h], eq, geometry)
        return self._prepared[self._key(u, h, eq, geometry)](u)

    def architecture_metadata(self):
        return {"family": self.family, "combination_ids": [], "parameters": 0,
                "track": "classical", "preparation": "state-independent coefficients only; setup timed separately"}


def _setting(ctx, name, default):
    return ctx.protocol.get("training", {}).get(name, default)


def _families(protocol):
    values = protocol.get("models", [f for f in FAMILIES if not f.startswith("df_")])
    if isinstance(values, dict):
        values = values.get("families", values.get("arms", list(values)))
    return [v["family"] if isinstance(v, dict) else v for v in values]


def _configuration(protocol):
    values = protocol.get("model_config", {})
    if isinstance(protocol.get("models"), dict):
        values = {**protocol["models"].get("config", {}), **values}
    return {"width": protocol.get("width", 8), "modes": protocol.get("modes", 3), "t_ref": protocol.get("t_ref", 1.),
            "trust_horizon": max(protocol.get("train_horizons", [.2])), **values}


def _eq_geom(parent, n):
    return Equation(parent["kappa"], parent["reaction_rate"]), Geometry((n, n), (1., 1.))


def _reference(parent, n, h, track="discrete"):
    key = f"{track}:{n}:{horizon_key(h)}"
    if key not in parent["references"]:
        raise ValueError(f"Missing independently prepared reference {key} for {parent['parent_id']}")
    reference = parent["references"][key]
    if not reference.get("accepted", False):
        raise ValueError(f"Unaccepted teacher {key}; training must not use it")
    return reference


def endpoint_errors(value, reference):
    error = value.detach().cpu().double() - reference["state"].cpu().double()
    mean = error.mean()
    rms = float(error.square().mean().sqrt())
    maximum = float(error.abs().max())
    return {"error_rms": rms, "error_max": maximum,
            "centered_rms": float((error - mean).square().mean().sqrt()),
            "mean_error": abs(float(mean)),
            "upper_rms": rms + float(reference.get("uncertainty_rms", 0.)),
            "upper_max": maximum + float(reference.get("uncertainty_max_bound", 0.)),
            "reference_accepted": bool(reference.get("accepted", False))}


def rollout(model, u, schedule, eq, geometry, budget=None, *, capture_intermediates=False):
    state = u
    intermediates = []
    if model.family == "df_base" and not capture_intermediates:
        # Adjacent exact heat half-steps are algebraically identical and may
        # be fused only for the uncorrected DF control. Endpoint corrections
        # prevent this fusion for every learned hybrid.
        from tdn.numerics.subflows import diffusion_step, reaction_step
        state = diffusion_step(state, schedule[0] / 2, eq, geometry)
        for index, h in enumerate(schedule):
            if budget is not None:
                budget.check()
            state = reaction_step(state, h, eq)
            following = schedule[index + 1] if index + 1 < len(schedule) else 0.
            state = diffusion_step(state, (h + following) / 2, eq, geometry)
            _check_prediction(state, index)
        return state, []
    for index, h in enumerate(schedule):
        if budget is not None:
            budget.check()
        state = model(state, float(h), eq, geometry)
        _check_prediction(state, index)
        intermediates.append(state)
    return state, intermediates


def _normalization(base, target, reference, target_floor):
    # Do not divide by a nearly solved/noisy defect. The frozen absolute
    # floor is shared across arms and has the same state units as targets.
    noise = max(float(reference.get("uncertainty_rms", 0.)), float(target_floor))
    return (base.detach() - target).square().mean().clamp_min(noise**2)


def _loss(model, parent, h, n, device, settings, *, enhanced=False, temporal=False):
    eq, geom = _eq_geom(parent, n)
    u = parent["states"][str(n)].to(device=device, dtype=torch.float32)
    reference = _reference(parent, n, h)
    target = reference["state"].to(device=device, dtype=u.dtype)
    components = model.correction_components(u, h, eq, geom)
    prediction = model.apply_increment(components)
    error, base_error = prediction - target, components["base"].detach() - target
    scale = _normalization(components["base"], target, reference, settings.get("loss_floor", 1.e-6))
    loss = error.square().mean() / scale
    if enhanced:
        centered = error - error.mean((-2, -1), keepdim=True)
        # The peak term is a differentiable exact L-infinity-square subgradient.
        base_centered = base_error - base_error.mean((-2, -1), keepdim=True)
        harm = (torch.relu(error.square().mean() - base_error.square().mean()) +
                torch.relu(centered.square().mean() - base_centered.square().mean())) / scale
        loss = loss + settings.get("centered_weight", .25) * centered.square().mean() / scale
        loss = loss + settings.get("peak_weight", .1) * error.square().amax() / scale
        loss = loss + settings.get("base_harm_weight", 1.) * harm
    if temporal:
        # Find distinct prepared constituent horizons whose sum is h. Both
        # intermediate and final coupled teachers are independent of models.
        keys = parent["references"]
        possible = []
        for first in settings.get("horizons", [.02, .04, .08, .12]):
            second = h - first
            if second > 1.e-10 and abs(first - second) > 1.e-10 and f"discrete:{n}:{horizon_key(first)}" in keys:
                possible.append((first, second))
        if possible:
            first, second = possible[len(possible) // 2]
            middle = model(u, first, eq, geom)
            target_middle = _reference(parent, n, first)["state"].to(device=device, dtype=u.dtype)
            composed = model(middle, second, eq, geom)
            loss = loss + settings.get("intermediate_weight", .5) * (middle - target_middle).square().mean() / scale
            loss = loss + settings.get("consistency_weight", .25) * (
                (composed - target).square().mean() + (composed - prediction).square().mean()) / scale
        epsilon = min(float(h) / 8, 1.e-3)
        generator = (model(u, epsilon, eq, geom) - u) / epsilon
        physical = rhs(u, eq, geom)
        generator_scale = physical.detach().square().mean().clamp_min(1.e-5)
        loss = loss + settings.get("generator_weight", .01) * (generator - physical).square().mean() / generator_scale
    return loss


@torch.no_grad()
def validation_metrics(model, parents, horizons, n, device, settings, budget):
    rows = []
    for parent in parents:
        for h in horizons:
            budget.check()
            eq, geom = _eq_geom(parent, n)
            u = parent["states"][str(n)].to(device=device, dtype=torch.float32)
            values = model.correction_components(u, h, eq, geom)
            pred = model.apply_increment(values)
            if not bool(torch.isfinite(pred).all()):
                raise TrialNumericalFailure("NONFINITE_VALIDATION_PREDICTION")
            ref = _reference(parent, n, h)
            target = ref["state"].to(device=device, dtype=u.dtype)
            err = endpoint_errors(pred, ref)
            base = endpoint_errors(values["base"], ref)
            scale = max(base["error_rms"]**2, settings.get("loss_floor", 1.e-6)**2,
                        float(ref.get("uncertainty_rms", 0.))**2)
            rows.append({"parent_id": parent["parent_id"], "regime": parent.get("regime", "unspecified"),
                         "horizon": h, **err, "normalized_mse": err["error_rms"]**2 / scale,
                         "normalized_peak": err["error_max"]**2 / scale,
                         "normalized_centered": err["centered_rms"]**2 / scale,
                         "base_harm": (max(0., err["error_rms"]**2 - base["error_rms"]**2) +
                                       max(0., err["centered_rms"]**2 - base["centered_rms"]**2)) / scale})
    # Every regime contributes equally despite unequal parent counts.
    regimes = sorted({r["regime"] for r in rows})
    enhanced = model.family in ("source_df_loss", "source_df_loss_shared", "c1_rank0", "c1_rank1", "c1_rank2", "c2_rank2")
    regime_losses = []
    for regime in regimes:
        members = [r for r in rows if r["regime"] == regime]
        regime_losses.append(statistics.mean(r["normalized_mse"] + (
            settings.get("peak_weight", .1) * r["normalized_peak"] +
            settings.get("centered_weight", .25) * r["normalized_centered"] +
            settings.get("base_harm_weight", 1.) * r["base_harm"] if enhanced else 0.) for r in members))
    objective = statistics.mean(regime_losses)
    if not math.isfinite(objective):
        raise TrialNumericalFailure("NONFINITE_VALIDATION_OBJECTIVE")
    return {"objective": objective, "worst_max": max(r["error_max"] for r in rows),
            "mean_rms": statistics.mean(r["error_rms"] for r in rows),
            "base_harm_cases": sum(r["base_harm"] > 1.e-6 for r in rows),
            "cases": len(rows), "regime_losses": dict(zip(regimes, regime_losses)), "rows": rows}


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _trial(ctx, family, seed, learning_rate, batch_size, updates, train_bank, validation_bank,
           name, *, phase, config, inherited_failure=None):
    from .core import check
    torch.manual_seed(seed)
    model = build_model(family, config).to(ctx.device)
    settings = {**ctx.protocol.get("training", {}), "horizons": ctx.protocol["train_horizons"]}
    n = int(ctx.protocol["train_grid"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=settings.get("weight_decay", 0.))
    start = time.perf_counter()
    initial = _snapshot(model)
    numerical_failure = None
    checkpoint_validated = True
    try:
        metric = validation_metrics(model, validation_bank, ctx.protocol["validation_horizons"], n,
                                    ctx.device, settings, ctx.budget)
    except TrialNumericalFailure as error:
        checkpoint_validated = False
        numerical_failure = {"phase": "initial_validation", "update": 0, "code": str(error)}
        metric = {"objective": None, "worst_max": None, "mean_rms": None,
                  "base_harm_cases": None, "cases": 0, "regime_losses": {}, "rows": []}
    selected_state, selected_metrics, selected_step = initial, metric, 0
    trajectory = [{"update": 0, "validation_objective": metric["objective"], "worst_max": metric["worst_max"]}]
    gradients, losses = [], []
    attempted_updates, completed_updates, attempted_examples = 0, 0, 0
    if inherited_failure is not None and numerical_failure is None:
        numerical_failure = {"phase": "optimizer_selection", "update": 0,
                             "code": "ALL_TUNING_TRIALS_NUMERICALLY_INELIGIBLE", "source": inherited_failure}
    generator = torch.Generator(device="cpu").manual_seed(seed + 1729)
    order = torch.randperm(len(train_bank), generator=generator).tolist()
    interval = max(1, int(settings.get("validation_every", max(1, updates // 3))))
    if ctx.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(ctx.device)
    for step in range(1, updates + 1):
        if numerical_failure is not None:
            break
        ctx.budget.check()
        attempted_updates = step
        failure_phase = "forward_loss"
        try:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for offset in range(batch_size):
                parent = train_bank[order[((step - 1) * batch_size + offset) % len(order)]]
                h = ctx.protocol["train_horizons"][(step + offset - 1) % len(ctx.protocol["train_horizons"])]
                attempted_examples += 1
                batch_loss = batch_loss + _loss(model, parent, h, n, ctx.device, settings,
                    enhanced=family in ("source_df_loss", "source_df_loss_shared", "c1_rank0", "c1_rank1", "c1_rank2", "c2_rank2"),
                    temporal=family in ("source_consistency", "c2_rank2")) / batch_size
            if not bool(torch.isfinite(batch_loss)):
                raise TrialNumericalFailure("NONFINITE_TRAINING_LOSS")
            failure_phase = "backward_gradient"
            batch_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("clip_grad_norm", 1.))
            if not bool(torch.isfinite(norm)):
                raise TrialNumericalFailure("NONFINITE_GRADIENT_NORM")
            failure_phase = "optimizer_update"
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise TrialNumericalFailure("NONFINITE_PARAMETER_AFTER_UPDATE")
            completed_updates = step
            gradients.append(float(norm))
            losses.append(float(batch_loss.detach()))
            if step % interval == 0 or step == updates:
                failure_phase = "validation"
                metric = validation_metrics(model, validation_bank, ctx.protocol["validation_horizons"], n,
                                             ctx.device, settings, ctx.budget)
                trajectory.append({"update": step, "train_loss": losses[-1], "validation_objective": metric["objective"],
                                   "worst_max": metric["worst_max"]})
                if metric["objective"] < selected_metrics["objective"]:
                    selected_state, selected_metrics, selected_step = _snapshot(model), metric, step
        except TrialNumericalFailure as error:
            numerical_failure = {"phase": failure_phase, "update": step, "code": str(error)}
            trajectory.append({"update": step, "numerical_failure": numerical_failure,
                               "checkpoint_retained_update": selected_step})
            break
    model.load_state_dict(selected_state)
    trust_calibration = []
    if family in ("source_trust", "c2_rank2") and numerical_failure is None:
        # Support is fitted only on validation, before any confirmation is
        # loaded. Record every candidate, including the unmodified envelope.
        for trust_scale in settings.get("trust_scales", [.5, 1., 2.]):
            model.trust_scale.fill_(float(trust_scale))
            try:
                calibrated = validation_metrics(model, validation_bank, ctx.protocol["validation_horizons"], n,
                                                 ctx.device, settings, ctx.budget)
                trust_calibration.append({"scale": float(trust_scale), "objective": calibrated["objective"]})
            except TrialNumericalFailure as error:
                numerical_failure = {"phase": "trust_calibration", "update": completed_updates, "code": str(error)}
                trust_calibration.append({"scale": float(trust_scale), "objective": None, "numerical_failure": str(error)})
                break
        if numerical_failure is None:
            chosen = min(trust_calibration, key=lambda r: (r["objective"], abs(r["scale"] - 1.)))
            model.trust_scale.fill_(chosen["scale"])
            selected_metrics = validation_metrics(model, validation_bank, ctx.protocol["validation_horizons"], n,
                                                 ctx.device, settings, ctx.budget)
            selected_state = _snapshot(model)
        else:
            model.load_state_dict(selected_state)
    probe_parent = validation_bank[0]
    probe_eq, probe_geometry = _eq_geom(probe_parent, n)
    probe_u = probe_parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
    zero = torch.zeros((), dtype=probe_u.dtype, device=probe_u.device)
    generator_error, null_error, zero_error = None, None, None
    if checkpoint_validated:
        _, derivative = torch.func.jvp(lambda h: model(probe_u, h, probe_eq, probe_geometry), (zero,), (torch.ones_like(zero),))
        expected = rhs(probe_u, probe_eq, probe_geometry)
        generator_error = float(((derivative - expected).abs().max() / expected.abs().max().clamp_min(1.e-6)).detach())
        null_values = model.correction_components(probe_u, .1, Equation(0., probe_eq.reaction_rate), probe_geometry)
        null_error = float(null_values["increment"].detach().abs().max())
        zero_error = float((model(probe_u, zero, probe_eq, probe_geometry) - probe_u).detach().abs().max())
    if ctx.device.startswith("cuda"):
        torch.cuda.synchronize(ctx.device)
    elapsed = time.perf_counter() - start
    checkpoint = Path(ctx.path) / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    selection = "TRAINED_CHECKPOINT" if selected_step else "SELECTED_INITIALIZATION"
    if numerical_failure is not None:
        selection = ("FALLBACK_PRE_FAILURE_CHECKPOINT" if selected_step else "FALLBACK_INITIALIZATION") if checkpoint_validated else "NO_VALIDATED_CHECKPOINT"
    parameter_names = set(dict(model.named_parameters()))
    changed = sum(float((selected_state[k] - initial[k]).abs().sum()) for k in parameter_names)
    checkpoint_value = {"schema": "tdn.roadmap-checkpoint/v1", "family": family, "config": config,
        "seed": seed, "state_dict": selected_state, "protocol_sha256": digest(ctx.protocol),
        "selected_update": selected_step, "selection": selection,
        "numerical_failure": numerical_failure, "checkpoint_validated": checkpoint_validated,
        "selection_split": "validation", "base_orientation": model.architecture_metadata()["base_orientation"]}
    atomic_torch_save(checkpoint_value, checkpoint)
    trial = {"model_id": name, "phase": phase, "family": family, "seed": seed,
        "checkpoint": str(checkpoint.relative_to(ctx.path)), "checkpoint_sha256": file_digest(checkpoint),
        "config": config, "learning_rate": learning_rate, "batch_size": batch_size,
        "updates_planned": updates, "updates_attempted": attempted_updates,
        "updates_completed": completed_updates, "updates_selected": selected_step, "selection": selection,
        "training_outcome": "NUMERICAL_FAILURE" if numerical_failure is not None else "COMPLETED",
        "numerical_failure": numerical_failure, "checkpoint_validated": checkpoint_validated,
        "eligible_for_optimizer_selection": numerical_failure is None and checkpoint_validated,
        "parameter_change_l1": changed, "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": elapsed, "elapsed_seconds": elapsed, "total_seconds": elapsed,
        "generator_relative_error": generator_error, "zero_diffusion_correction_error": null_error,
        "gradient_norm_max": max(gradients, default=0.),
        "trust_validation_calibration": trust_calibration,
        "gradient_norm_mean": statistics.mean(gradients) if gradients else 0.,
        "gradient_clip_fraction": sum(g > settings.get("clip_grad_norm", 1.) for g in gradients) / max(len(gradients), 1),
        "examples_seen": attempted_examples, "examples_in_completed_updates": completed_updates * batch_size,
        "validation": {k: v for k, v in selected_metrics.items() if k != "rows"},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(ctx.device) if ctx.device.startswith("cuda") else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(ctx.device) if ctx.device.startswith("cuda") else None,
        "architecture": model.architecture_metadata()}
    write_json(Path(ctx.path) / "trials" / f"{name}.json", {**trial, "learning_curve": trajectory,
        "training_losses": losses, "gradient_norms": gradients, "selected_validation_rows": selected_metrics["rows"]})
    ctx.record(f"train/{name}", MECHANISMS[family], combination_ids=model.architecture_metadata()["combination_ids"],
        metrics={k: v for k, v in trial.items() if k not in ("architecture", "validation", "config")},
        config={**config, "orientation": model.architecture_metadata()["base_orientation"], "learning_rate": learning_rate, "batch_size": batch_size,
                "phase": phase, "seed": seed, "loss_settings": settings,
                "base_harm_definition": "sum of positive same-step total-MSE and centered-MSE regressions, normalized by declared defect/noise floor",
                "trust_scales": settings.get("trust_scales", [.5, 1., 2.]),
                "selected_trust_scale": float(model.trust_scale)},
        checks=[check("numerical_training_completed", numerical_failure is None, True, "eq", category="correctness",
                      reason=str(numerical_failure) if numerical_failure is not None else "All planned updates and validation completed"),
                check("planned_updates_completed", completed_updates, updates, "eq", category="gap"),
                check("finite_gradients", all(math.isfinite(x) for x in gradients), True, "eq", category="correctness", applicable=bool(gradients)),
                check("validation_selection_only", True, True, "eq", category="correctness"),
                check("generator_consistency", generator_error, 1.e-4, "le", category="math", applicable=not model.direct,
                      reason="Direct FNO learns its generator and has no exact physical-generator constraint" if model.direct else ""),
                check("exact_zero_diffusion_correction", null_error, 1.e-7, "le", category="math", applicable=not model.direct,
                      reason="Neural-only FNO declares no commuting-physics constraint" if model.direct else ""),
                check("zero_time_identity", zero_error, 1.e-6, "le", category="math"),
                check("trained_parameter_change", changed > 0., True, "eq", category="gap", applicable=selected_step > 0,
                      reason="Initialization selections cannot establish learned efficacy"),
                check("validation_improvement", selected_metrics["objective"], trajectory[0]["validation_objective"], "lt", category="gap")],
        evidence={"checkpoint_sha256": trial["checkpoint_sha256"], "selection": selection,
                  "learning_curve": f"trials/{name}.json", "numerical_failure": numerical_failure},
        status="NUMERICAL_FAILURE" if numerical_failure is not None else "COMPLETED")
    return trial


def train(ctx):
    from .data import load_bank
    train_bank, validation_bank = load_bank(ctx, "train"), load_bank(ctx, "validation")
    if {p["field_cluster"] for p in train_bank} & {p["field_cluster"] for p in validation_bank}:
        raise ValueError("Train and validation independent field clusters overlap")
    families, config = _families(ctx.protocol), _configuration(ctx.protocol)
    if "source" in families and "source_df_loss" in families and "source_df_loss_shared" not in families:
        families = [*families, "source_df_loss_shared"]
    invalid = set(families) - set(FAMILIES)
    if invalid:
        raise ValueError(f"Unknown neural families {sorted(invalid)}")
    seeds = ctx.protocol.get("seeds", [73001])
    settings = ctx.protocol.get("training", {})
    combinations = list(itertools.product(settings.get("learning_rates", [.001]), settings.get("batches", [1])))
    plan = {"schema": "tdn.roadmap-training-plan/v1", "protocol_sha256": digest(ctx.protocol),
            "families": families, "seeds": seeds, "config": config,
            "tuning_grid": [{"learning_rate": lr, "batch_size": batch} for lr, batch in combinations],
            "tuning_updates": settings.get("tuning_updates", 1), "updates": settings.get("updates", 2),
            "selection": "per-family validation-only optimizer selection; all declared final seeds retained",
            "timing_model_order": "deterministic shuffle from SHA256(parent_id, grid, schedule_id); fixed before timing",
            "confirmation_access": False}
    write_json(Path(ctx.path) / "training_plan.json", plan)
    records, tuning, optimizer_selections = [], [], {}
    for family in families:
        if family.startswith("df_"):
            continue
        if family == "source_df_loss_shared":
            selected = optimizer_selections["source"]
            for seed in seeds:
                name = f"{family}-seed-{seed}"
                record = _trial(ctx, family, seed, selected["learning_rate"], selected["batch_size"],
                    int(settings.get("updates", 2)), train_bank, validation_bank, name,
                    phase="paired_final_shared_source_optimizer", config=config,
                    inherited_failure=selected.get("all_tuning_failed"))
                record["tuning_selection"] = selected["model_id"]
                records.append(record)
            continue
        candidates = []
        for index, (lr, batch) in enumerate(combinations):
            name = f"{family}-tune-{index:02d}"
            candidates.append(_trial(ctx, family, seeds[0], lr, batch, int(settings.get("tuning_updates", 1)),
                train_bank, validation_bank, name, phase="tuning", config=config))
        tuning.extend(candidates)
        eligible = [candidate for candidate in candidates if candidate["eligible_for_optimizer_selection"]]
        if eligible:
            selected = min(eligible, key=lambda r: (r["validation"]["objective"], r["model_id"]))
        else:
            # This is a deterministic initialized fallback configuration,
            # explicitly not a winning hyperparameter or successful trial.
            selected = dict(min(candidates, key=lambda r: r["model_id"]))
            selected["all_tuning_failed"] = [candidate["model_id"] for candidate in candidates]
        optimizer_selections[family] = selected
        for seed in seeds:
            name = f"{family}-seed-{seed}"
            record = _trial(ctx, family, seed, selected["learning_rate"], selected["batch_size"],
                int(settings.get("updates", 2)), train_bank, validation_bank, name, phase="paired_final", config=config,
                inherited_failure=selected.get("all_tuning_failed"))
            record["tuning_selection"] = selected["model_id"]
            records.append(record)
    catalog = {"schema": "tdn.roadmap-catalog/v1", "protocol_sha256": digest(ctx.protocol),
        "records": records, "tuning_records": tuning, "frozen_before_confirmation": True,
        "numerically_failed_trials": sum(r["training_outcome"] == "NUMERICAL_FAILURE" for r in records + tuning),
        "numerically_failed_final_arms": sum(r["training_outcome"] == "NUMERICAL_FAILURE" for r in records),
        "selection_split": "validation", "family_selection": "all declared arms and seeds retained",
        "training_plan_sha256": file_digest(Path(ctx.path) / "training_plan.json")}
    write_json(Path(ctx.path) / "catalog.json", catalog)
    write_json(Path(ctx.path) / "freeze.json", {"catalog_sha256": file_digest(Path(ctx.path) / "catalog.json"),
        "protocol_sha256": digest(ctx.protocol), "checkpoint_hashes": {r["model_id"]: r["checkpoint_sha256"] for r in records}})
    return {"selected_models": len(records), "tuning_trials": len(tuning), "trained_selections": sum(r["updates_selected"] > 0 for r in records),
            "initialization_selections": sum(r["updates_selected"] == 0 for r in records),
            "numerically_failed_trials": catalog["numerically_failed_trials"],
            "numerically_failed_final_arms": catalog["numerically_failed_final_arms"],
            "catalog_sha256": file_digest(Path(ctx.path) / "catalog.json")}


def load_frozen_models(ctx):
    base = Path(ctx.prerequisites["train"])
    catalog = json.loads((base / "catalog.json").read_text())
    freeze = json.loads((base / "freeze.json").read_text())
    if freeze["catalog_sha256"] != file_digest(base / "catalog.json") or catalog["protocol_sha256"] != digest(ctx.protocol):
        raise ValueError("Frozen neural catalog or protocol changed")
    if not catalog.get("frozen_before_confirmation") or catalog.get("selection_split") != "validation":
        raise ValueError("Neural selection must freeze on validation before confirmation")
    models = {}
    for record in catalog["records"]:
        path = (base / record["checkpoint"]).resolve()
        if not path.is_relative_to(base.resolve()) or path.is_symlink():
            raise ValueError("Checkpoint path escapes its sealed training stage")
        if file_digest(path) != freeze["checkpoint_hashes"][record["model_id"]] or file_digest(path) != record["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint digest changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["protocol_sha256"] != digest(ctx.protocol) or checkpoint["selection_split"] != "validation":
            raise ValueError("Checkpoint provenance changed")
        if checkpoint.get("checkpoint_validated", True) != record.get("checkpoint_validated", True) or checkpoint.get("numerical_failure") != record.get("numerical_failure"):
            raise ValueError("Checkpoint numerical eligibility differs from the frozen catalog")
        if not record.get("checkpoint_validated", True):
            from .core import check
            ctx.record(f"unavailable-frozen-model/{record['model_id']}", MECHANISMS[record["family"]],
                metrics={"model_id": record["model_id"], "training_outcome": record["training_outcome"],
                         "parameters": record["parameters"], "numerical_failure": record["numerical_failure"]},
                checks=[check("validated_checkpoint_available", False, True, "eq", category="correctness"),
                        check("learned_efficacy", None, None, category="gap"),
                        check("mathematical_response", None, None, category="math")],
                config={"family": record["family"], "seed": record["seed"]},
                evidence={"checkpoint_sha256": record["checkpoint_sha256"], "scope": "No validated checkpoint exists; scientific evidence remains missing"},
                status="NUMERICAL_FAILURE")
            continue
        model = build_model(record["family"], record["config"]).to(ctx.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().requires_grad_(False)
        model.selection_metadata = record
        models[record["model_id"]] = model
    for family in ("df_base", "df_quad2"):
        model = build_model(family, _configuration(ctx.protocol)).to(ctx.device).eval().requires_grad_(False)
        model.selection_metadata = {"selection": "UNTRAINED_PHYSICAL_CONTROL", "updates_selected": 0,
                                    "parameters": 0, "family": family, "seed": None}
        models[family] = model
    for family in ("rf_base", "etdrk4", "gl3_fused"):
        models[family] = ClassicalControl(family).to(ctx.device).eval()
    return models


def _schedules(protocol):
    result = []
    values = protocol.get("confirm_schedules", [[.03, .07, .11], [.11, .07, .03]])
    if isinstance(values, dict):
        values = [{"id": k, "steps": v} for k, v in values.items()]
    for index, item in enumerate(values):
        if isinstance(item, dict):
            steps = item.get("steps", item.get("schedule"))
            name = item.get("id", item.get("name", f"schedule-{index}"))
        else:
            steps, name = item, f"schedule-{index}"
        result.append((name, list(steps)))
    return result


@torch.no_grad()
def _frozen_structure(ctx, models, parent, h):
    """Finite learned-map probes against the scalar FD comparison principle.

    This is a counterexample search on a declared field and positive bump;
    absence of a violation cannot establish global monotonicity or stability.
    """
    from .core import check
    n = int(ctx.protocol["train_grid"])
    eq, geometry = _eq_geom(parent, n)
    u = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
    x = torch.arange(n, device=ctx.device, dtype=u.dtype) / n
    bump = (torch.cos(2 * torch.pi * x) + 1)[:, None] * (torch.cos(2 * torch.pi * x) + 1)[None, :] / 4
    bump = bump[None, None]
    epsilon = min(.001, float(u.min()) / 4, float(1 - u.max()) / 4)
    for model_id, model in models.items():
        ctx.budget.check()
        probe_start = time.perf_counter()
        try:
            center = model(u, h, eq, geometry)
            _check_prediction(center)
            plus = model(u + epsilon * bump, h, eq, geometry)
            _check_prediction(plus)
            minus = model(u - epsilon * bump, h, eq, geometry)
            _check_prediction(minus)
        except RolloutNumericalFailure as error:
            ctx.record(f"learned-structure/{model_id}", ["M21", *MECHANISMS[model.family]],
                combination_ids=model.architecture_metadata()["combination_ids"],
                metrics={"model_id": model_id, "numerical_failure": error.code,
                         "failed_attempt_seconds": time.perf_counter() - probe_start,
                         "parameters": sum(p.numel() for p in model.parameters())},
                checks=[check("finite_admissible_probe", False, True, "eq", category="math"),
                        check("usable_stress_prediction", False, True, "eq", category="gap")],
                config={"parent_id": parent["parent_id"], "model_id": model_id, "horizon": h, "grid": n},
                evidence={"scope": "Observed numerical failure; no error or stability guarantee inferred"},
                status="NUMERICAL_FAILURE")
            continue
        growth = float((plus - center).abs().max()) / max(epsilon, 1.e-12)
        order_violation = max(0., float((center - plus).max()))
        concavity_violation = max(0., float(((plus + minus) / 2 - center).max()))
        ctx.record(f"learned-structure/{model_id}", ["M21", *MECHANISMS[model.family]],
            combination_ids=model.architecture_metadata()["combination_ids"],
            metrics={"local_linf_growth": growth, "physical_growth_bound": math.exp(eq.reaction_rate * h),
                     "monotonicity_violation": order_violation, "concavity_violation": concavity_violation,
                     "perturbation_epsilon": epsilon, "parameters": sum(p.numel() for p in model.parameters())},
            checks=[check("physical_growth_envelope", growth, math.exp(eq.reaction_rate * h) + .005, "le", category="math"),
                    check("positive_input_order", order_violation, 5.e-7, "le", category="math"),
                    check("logistic_flow_concavity_probe", concavity_violation, 5.e-7, "le", category="math")],
            config={"parent_id": parent["parent_id"], "field_cluster": parent["field_cluster"],
                    "model_id": model_id, "horizon": h, "grid": n, "probe": "one fixed nonnegative cosine bump"},
            evidence={"scope": "finite perturbation probe, not a global theorem or a uniform Lipschitz bound"})


def _matched_comparisons(ctx, rows, models, target_pairs):
    """Post-hoc fixed-schedule utility, explicitly separate from deployment.

    Compare the minimum accurately feasible measured whole-rollout cost,
    retaining no-feasible cases rather than dropping them from coverage.
    """
    from .core import check
    from .statistics import paired_cluster_bootstrap
    cells = {}
    for row in rows:
        cells.setdefault((row["parent_id"], row["grid"], row["track"]), []).append(row)
    summaries, paired = [], {}
    classical = {"df_base", "rf_base", "etdrk4", "gl3_fused"}
    for (parent_id, n, track), members in cells.items():
        for rms_target, max_target in target_pairs:
            def feasible(item):
                return (not item.get("numerical_failure") and item["reference_accepted"] and
                        item["upper_rms"] is not None and item["upper_max"] is not None and
                        item["cost_seconds"] is not None and
                        item["upper_rms"] <= rms_target and item["upper_max"] <= max_target)
            physical = [r for r in members if r["family"] in classical and feasible(r)]
            best_physical = min(physical, key=lambda r: r["cost_seconds"]) if physical else None
            for model_id, model in models.items():
                if model_id in classical or model_id == "df_quad2":
                    continue
                selected = [r for r in members if r["model_id"] == model_id and feasible(r)]
                best = min(selected, key=lambda r: r["cost_seconds"]) if selected else None
                seed = model.selection_metadata.get("seed")
                controls = {"best_classical": best_physical}
                for family in ("cheap_fno", "deep_fno", "direct_fno"):
                    candidates = [r for r in members if r["family"] == family and r["seed"] == seed and feasible(r)]
                    controls[family] = min(candidates, key=lambda r: r["cost_seconds"]) if candidates else None
                ratios = {key: control["cost_seconds"] / best["cost_seconds"] if control is not None and best is not None else None
                          for key, control in controls.items()}
                entry = {"model_id": model_id, "family": model.family, "parent_id": parent_id, "grid": n,
                    "track": track, "seed": seed, "rms_target": rms_target, "max_target": max_target,
                    "candidate_feasible": best is not None, "candidate_schedule": best["schedule_id"] if best else None,
                    "candidate_cost_seconds": best["cost_seconds"] if best else None,
                    "control_feasible": {key: value is not None for key, value in controls.items()},
                    "control_schedule": {key: value["schedule_id"] if value else None for key, value in controls.items()},
                    "control_model": {key: value["model_id"] if value else None for key, value in controls.items()},
                    "control_over_candidate_speed_ratio": ratios,
                    "selection_scope": "reference-informed post-hoc minimum over frozen schedules, not a deployable policy"}
                summaries.append(entry)
                checks = [check("accuracy_feasible_on_fixed_grid", best is not None, True, "eq", category="gap"),
                          check("same_reference_target", True, True, "eq", category="math")]
                for control_id, ratio in ratios.items():
                    checks.append(check(f"matched_cost_vs_{control_id}", ratio, 1., "ge", category="utility",
                        applicable=model.family != control_id and best is not None and controls[control_id] is not None,
                        reason="Unavailable if either method has no feasible schedule; self-comparisons are NA"))
                    if ratio is not None and model.family != control_id:
                        key = (model.family, control_id, track, rms_target, max_target)
                        paired.setdefault(key, []).append({"field_cluster": members[0]["field_cluster"],
                                                           "seed": seed, "difference": math.log(ratio)})
                ctx.record(f"matched/{model_id}/{parent_id}/N{n}/{track}/{rms_target:g}-{max_target:g}",
                    ["M16", *[m for m in MECHANISMS[model.family] if m != "M16"]],
                    combination_ids=model.architecture_metadata()["combination_ids"], metrics=entry,
                    checks=checks, config={"model_id": model_id, "track": track,
                        "target_rms": rms_target, "target_max": max_target, "paired_seed": seed},
                    evidence={"published_paper_reproduction": False, "candidate_track": model.architecture_metadata().get("track"),
                              "classical_controls": sorted(classical)})
    inference = []
    for (family, control, track, rms_target, max_target), values in paired.items():
        result = paired_cluster_bootstrap(values, repeats=200 if ctx.protocol["profile"] == "smoke" else 1000)
        record = {"family": family, "comparator": control, "track": track, "rms_target": rms_target,
                  "max_target": max_target, "log_speed_ratio": result,
                  "eligibility": "both methods accurately feasible; coverage reported independently in every matched row"}
        inference.append(record)
        ctx.record(f"paired-inference/{family}/{control}/{track}/{rms_target:g}-{max_target:g}", ["M16", "M18"],
            metrics=record, config={"family": family, "comparator": control, "track": track},
            checks=[check("three_paired_training_seeds", result["training_seeds"], 3, "ge", category="gap"),
                    check("independent_field_count", result["independent_fields"], 2, "ge", category="math"),
                    check("positive_paired_log_speed_lower_bound", result.get("lower"), 0., "gt", category="utility",
                          applicable=result.get("lower") is not None,
                          reason="Descriptive crossed field/seed bootstrap, conditional on both methods being feasible")])
    write_json(Path(ctx.path) / "matched_comparisons.json", {"comparisons": summaries, "paired_inference": inference,
        "paper_reproduction": False, "frontier_is_deployment_policy": False})
    return len(summaries)


def _failed_confirmation(ctx, rows, model_id, model, parent, n, schedule_id, schedule,
                         refs, target_pairs, failure, attempt_seconds, order_seed, order_index):
    """Keep every failed endpoint in accuracy/coverage denominators."""
    from .core import check
    selection = model.selection_metadata
    for track, reference in refs.items():
        if reference is None:
            continue
        errors = {key: None for key in ("error_rms", "error_max", "centered_rms", "mean_error", "upper_rms", "upper_max")}
        targets = {f"rms-{rt:g}_max-{mt:g}": False for rt, mt in target_pairs}
        entry = {"model_id": model_id, "family": model.family, "parent_id": parent["parent_id"],
            "field_cluster": parent["field_cluster"], "grid": n, "track": track, "schedule_id": schedule_id,
            "passed_both": False, "target_results": targets, **errors,
            "reference_accepted": bool(reference.get("accepted")), "seed": selection.get("seed"),
            "cost_seconds": None, "numerical_failure": failure.code,
            "completed_steps_before_failure": failure.completed_steps}
        rows.append(entry)
        ctx.record(f"confirm/{model_id}/{parent['parent_id']}/N{n}/{schedule_id}/{track}",
            MECHANISMS[model.family], combination_ids=model.architecture_metadata()["combination_ids"],
            metrics={**entry, "median_seconds": None, "failed_attempt_seconds": attempt_seconds,
                "measured_attempt_id": f"{model_id}/{parent['parent_id']}/N{n}/{schedule_id}",
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "selection": selection["selection"], "updates_selected": selection["updates_selected"],
                "training_outcome": selection.get("training_outcome", "UNTRAINED_PHYSICAL_CONTROL"),
                "training_numerical_failure": selection.get("numerical_failure"),
                "cost_scope": "elapsed failed attempt retained; no valid completed-rollout latency"},
            config={"model_id": model_id, "family": model.family, "parent_id": parent["parent_id"],
                "field_cluster": parent["field_cluster"], "grid": n, "schedule": schedule,
                "schedule_id": schedule_id, "track": track, "seed": selection.get("seed"),
                "timing_order_seed": order_seed, "timing_order_index": order_index},
            checks=[check("finite_admissible_trajectory", False, True, "eq", category="math"),
                    check("usable_accuracy_result", False, True, "eq", category="gap"),
                    check("accepted_independent_teacher", reference.get("accepted", False), True, "eq", category="correctness"),
                    *[check(f"target_{norm}_{target:g}", None, target, "le", category="gap",
                            reason="Prediction failed numerically; no finite error estimate exists")
                      for rt, mt in target_pairs for norm, target in (("rms", rt), ("max", mt))],
                    check("completed_rollout_cost", None, None, category="utility")],
            evidence={"checkpoint_sha256": selection.get("checkpoint_sha256"),
                      "numerical_failure": failure.code, "scope": "failed case retained in every schedule/target denominator"},
            status="NUMERICAL_FAILURE")


def confirm(ctx):
    from .core import check
    from .data import load_bank
    # Loading/validating every checkpoint and writing its hash precedes the
    # first independent confirmation tensor access.
    models = load_frozen_models(ctx)
    write_json(Path(ctx.path) / "frozen_model_ledger.json", {"models": {k: getattr(v, "selection_metadata", {}) for k, v in models.items()},
        "catalog_sha256": file_digest(Path(ctx.prerequisites["train"]) / "catalog.json"),
        "protocol_sha256": digest(ctx.protocol)})
    parents = load_bank(ctx, "confirmation")
    schedules = _schedules(ctx.protocol)
    _frozen_structure(ctx, models, parents[0], max(schedules[0][1]))
    for model in models.values():
        if isinstance(model, ClassicalControl):
            model._prepared.clear()
    targets = ctx.protocol.get("targets", [2.e-4])
    if isinstance(targets, dict):
        rms_target = float(targets.get("rms", targets.get("rms_target", 2.e-4)))
        max_target = float(targets.get("max", targets.get("max_target", 2.e-4)))
        target_pairs = [(rms_target, max_target)]
    else:
        target_pairs = [(float(value), float(value)) for value in targets]
        rms_target, max_target = target_pairs[0]
    rows = []
    with torch.no_grad():
        for parent in parents:
            for n in ctx.protocol.get("grids", [ctx.protocol["train_grid"]]):
                n = int(n)
                if str(n) not in parent["states"]:
                    continue
                eq, geometry = _eq_geom(parent, n)
                u = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
                for schedule_id, schedule in schedules:
                    horizon = float(sum(schedule))
                    refs = {track: parent["references"].get(f"{track}:{n}:{horizon_key(horizon)}") for track in ("discrete", "continuum")}
                    # Classical timings here deliberately measure standalone
                    # full rollouts; root classical panel also benchmarks fusion.
                    base_failure, base_result, base_cost = None, None, None
                    base_attempt_start = time.perf_counter()
                    try:
                        base_result, base_cost = ctx.measure(lambda: rollout(models["df_base"], u, schedule, eq, geometry, ctx.budget)[0],
                                                            warmup=ctx.protocol.get("timing_warmup", 0))
                    except RolloutNumericalFailure as error:
                        base_failure = error
                    base_attempt_seconds = time.perf_counter() - base_attempt_start
                    ordered_models = list(models.items())
                    order_seed = int(digest([parent["parent_id"], n, schedule_id])[:16], 16)
                    random.Random(order_seed).shuffle(ordered_models)
                    for timing_order_index, (model_id, model) in enumerate(ordered_models):
                        ctx.budget.check()
                        setup_cost = None
                        attempt_start = time.perf_counter()
                        try:
                            if isinstance(model, ClassicalControl):
                                _, setup_cost = ctx.measure(lambda: model.prepare(u, schedule, eq, geometry), repeats=1)
                            if model_id == "df_base":
                                if base_failure is not None:
                                    raise base_failure
                                prediction, cost = base_result, base_cost
                            else:
                                prediction, cost = ctx.measure(lambda: rollout(model, u, schedule, eq, geometry, ctx.budget)[0],
                                                              warmup=ctx.protocol.get("timing_warmup", 0))
                            _, middle = rollout(model, u, schedule, eq, geometry, ctx.budget, capture_intermediates=True)
                        except RolloutNumericalFailure as error:
                            attempt_seconds = base_attempt_seconds if model_id == "df_base" else time.perf_counter() - attempt_start
                            _failed_confirmation(ctx, rows, model_id, model, parent, n, schedule_id, schedule,
                                refs, target_pairs, error, attempt_seconds, order_seed, timing_order_index)
                            continue
                        transient_errors = []
                        elapsed = 0.
                        for h, value in zip(schedule, middle):
                            elapsed += h
                            reference = parent["references"].get(f"discrete:{n}:{horizon_key(elapsed)}")
                            if reference is not None and reference.get("accepted"):
                                transient_errors.append(endpoint_errors(value, reference))
                        for track, reference in refs.items():
                            if reference is None:
                                continue
                            errors = endpoint_errors(prediction, reference)
                            base_errors = endpoint_errors(base_result, reference) if base_result is not None else {"error_rms": None, "error_max": None, "centered_rms": None}
                            selection = model.selection_metadata
                            passed = errors["upper_rms"] <= rms_target and errors["upper_max"] <= max_target and bool(reference.get("accepted"))
                            metrics = {**errors, **cost, "passed_both": passed, "base_error_rms": base_errors["error_rms"],
                                "base_error_max": base_errors["error_max"], "base_centered_rms": base_errors["centered_rms"],
                                "cost": cost, "base_cost": base_cost,
                                "classical_setup_cost": setup_cost,
                                "relative_to_base_cost": cost["median_seconds"] / max(base_cost["median_seconds"], 1.e-12) if base_cost is not None else None,
                                "base_numerical_failure": base_failure.code if base_failure is not None else None,
                                "parameters": sum(p.numel() for p in model.parameters()),
                                "model_id": model_id, "selection": selection["selection"],
                                "training_outcome": selection.get("training_outcome", "UNTRAINED_PHYSICAL_CONTROL"),
                                "training_numerical_failure": selection.get("numerical_failure"),
                                "updates_selected": selection["updates_selected"],
                                "intermediate_reference_count": len(transient_errors),
                                "intermediate_reference_track": "discrete",
                                "target_results": {f"rms-{rt:g}_max-{mt:g}": errors["upper_rms"] <= rt and errors["upper_max"] <= mt and bool(reference.get("accepted")) for rt, mt in target_pairs},
                                "intermediate_error_max": max((e["error_max"] for e in transient_errors), default=None)}
                            name = f"confirm/{model_id}/{parent['parent_id']}/N{n}/{schedule_id}/{track}"
                            target_checks = [check(f"target_{norm}_{target:g}", errors[f"upper_{norm}"], target, "le", category="gap", applicable=bool(reference.get("accepted")), reason="Accuracy requires an accepted independent teacher")
                                for rt, mt in target_pairs for norm, target in (("rms", rt), ("max", mt))]
                            training_checks = ([check("numerical_training_completed", selection["training_outcome"] == "COMPLETED", True, "eq", category="correctness",
                                reason="A pre-failure checkpoint may be measured, but does not erase the failed training trial")]
                                if "training_outcome" in selection else [])
                            ctx.record(name, MECHANISMS[model.family], combination_ids=model.architecture_metadata()["combination_ids"],
                                metrics=metrics, config={"model_id": model_id, "family": model.family, "seed": selection.get("seed"),
                                    "parent_id": parent["parent_id"], "field_cluster": parent["field_cluster"],
                                    "regime": parent.get("regime"), "grid": n, "schedule": schedule,
                                    "schedule_id": schedule_id, "track": track, "kappa": eq.kappa, "reaction_rate": eq.reaction_rate,
                                    "timing_order_seed": order_seed, "timing_order_index": timing_order_index,
                                    "timing_order_scope": "parent/grid/schedule deterministic shuffled model order; bare DF reference timed first"},
                                checks=[*training_checks, *target_checks, check("accepted_independent_teacher", reference.get("accepted", False), True, "eq", category="correctness"),
                                    check("same_step_spatial_no_harm", errors["centered_rms"], base_errors["centered_rms"] + max(1.e-6, float(reference.get("uncertainty_rms", 0.))) if base_result is not None else None, "le", category="gap", applicable=bool(reference.get("accepted")) and base_result is not None),
                                    check("same_schedule_cheaper_than_df", cost["median_seconds"], base_cost["median_seconds"] if base_cost is not None else None, "le", category="utility", applicable=model.family != "df_base" and base_cost is not None),
                                    check("discrete_intermediate_accuracy", metrics["intermediate_error_max"], max_target, "le", category="math", applicable=bool(transient_errors),
                                          reason="Measured trajectory errors support this tested schedule only; no general theorem")],
                                evidence={"teacher_track": track, "checkpoint_sha256": selection.get("checkpoint_sha256"),
                                          "claim_scope": "project-adapted controls; prescribed schedules; no paper reproduction"})
                            rows.append({"model_id": model_id, "family": model.family, "parent_id": parent["parent_id"],
                                "field_cluster": parent["field_cluster"], "grid": n, "track": track, "schedule_id": schedule_id,
                                "passed_both": passed, "target_results": metrics["target_results"], **errors,
                                "seed": selection.get("seed"), "cost_seconds": cost["median_seconds"]})
    groups = {}
    for row in rows:
        key = (row["model_id"], row["parent_id"], row["grid"], row["track"])
        groups.setdefault(key, []).append(row)
    aggregates = []
    for (model_id, parent_id, n, track), members in groups.items():
        complete = len(members) == len(schedules)
        all_pass = complete and all(r["passed_both"] for r in members)
        any_pass = any(r["passed_both"] for r in members)
        aggregate = {"model_id": model_id, "parent_id": parent_id, "grid": n, "track": track,
                     "schedules_reported": len(members), "schedules_completed": sum(not r.get("numerical_failure") for r in members),
                     "numerically_failed_schedules": sum(bool(r.get("numerical_failure")) for r in members), "all_schedules_pass": all_pass,
                     "any_schedule_pass": any_pass, "worst_error_max": max((r["error_max"] for r in members if r["error_max"] is not None), default=None),
                     "target_all_schedule_pass": {target: complete and all(r["target_results"][target] for r in members) for target in members[0]["target_results"]}}
        aggregates.append(aggregate)
        model = models[model_id]
        ctx.record(f"all-schedules/{model_id}/{parent_id}/N{n}/{track}", MECHANISMS[model.family],
            combination_ids=model.architecture_metadata()["combination_ids"], metrics=aggregate,
            checks=[check("all_prescribed_schedules_accurate", all_pass, True, "eq", category="gap"),
                    check("complete_schedule_reporting", complete, True, "eq", category="correctness")],
            config={"track": track, "model_id": model_id})
    write_json(Path(ctx.path) / "confirmation_rows.json", {"rows": rows, "all_schedule_groups": aggregates,
        "independence_unit": "field_cluster; model seeds/grids/schedules are paired", "paper_reproduction": False})
    comparisons = _matched_comparisons(ctx, rows, models, target_pairs)
    return {"models": len(models), "endpoint_rows": len(rows), "all_schedule_groups": len(aggregates),
            "numerically_failed_endpoints": sum(bool(r.get("numerical_failure")) for r in rows),
            "matched_comparison_rows": comparisons,
            "all_schedule_passes": sum(r["all_schedules_pass"] for r in aggregates),
            "scope": "frozen project neural controls; no published-paper superiority claim"}
