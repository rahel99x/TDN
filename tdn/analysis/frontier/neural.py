"""Frozen, bounded architecture comparison for the five-gate program.

The training unit is an independent continuous field, never an endpoint row.
All architectures receive the same data, physical inputs, loss, number of
optimizer candidates and update budget. Equal updates are *not* equal compute:
both are logged. Initialization and unsuccessful trials remain in the ledger.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from tdn.numerics import Equation, Geometry
from tdn.research.experiment import atomic_torch_save, horizon_key
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json

TRAINABLE = frozenset(("rank1", "rank1_postcompression", "fno_small", "fno_standard", "direct_fno"))
PAIRED = TRAINABLE | {"rank1_frozen"}
CLASSICAL = frozenset(("df", "etdrk4", "analytic_quad", "analytic_quad_cubic"))


class TrialNumericalFailure(FloatingPointError):
    """Only explicit nonfinite numerics are recoverable scientific failures."""


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _budget(ctx):
    ctx.budget.check()


def _config(protocol, family):
    return {**protocol.get("model_config", {}),
            **protocol.get("model_configs", {}).get("rank1" if family == "rank1_frozen" else family, {})}


def expected_model_specs(protocol):
    """The complete final inventory; no seed or data fraction is selected away."""
    specs = []
    for track in protocol["tracks"]:
        for family in protocol["models"]:
            pairs = ((seed, count) for count in protocol["training"]["subset_sizes"]
                     for seed in protocol["seeds"]) if family in PAIRED else [(None, 0)]
            for seed, count in pairs:
                specs.append({"model_id": f"{track}/{family}/n{count}/seed{seed}",
                              "family": family, "track": track, "seed": seed,
                              "train_count": count})
    return specs


def nested_subset(parents, count, seed=41701):
    """Balanced deterministic field prefixes, retaining all physics variants."""
    groups = {}
    for parent in parents:
        cluster = parent["field_cluster"]
        groups.setdefault(cluster, []).append(parent)
    if not 1 <= count <= len(groups):
        raise ValueError("Data efficiency counts are independent fields, not repeated parent rows")
    regimes = {}
    for cluster, members in groups.items():
        regimes.setdefault(members[0].get("regime", "unspecified"), []).append(cluster)
    for regime in regimes:
        regimes[regime].sort(key=lambda key: digest([seed, key]))
    ordered = []
    while any(regimes.values()):
        for regime in sorted(regimes):
            if regimes[regime]:
                ordered.append(regimes[regime].pop(0))
    return [parent for cluster in ordered[:count] for parent in groups[cluster]]


def eq_geom(parent, n):
    return (Equation(parent["kappa"], parent["reaction_rate"]),
            Geometry((n, n), tuple(parent.get("lengths", (1., 1.)))))


def reference(parent, n, horizon, track):
    key = f"{track}:{n}:{horizon_key(horizon)}"
    if key not in parent["references"]:
        raise ValueError(f"Missing required teacher {key} for {parent['parent_id']}")
    return parent["references"][key]


def endpoint_errors(value, ref):
    """Absolute and spectral errors in the declared target, without clipping."""
    predicted, truth = value.detach().cpu().double(), ref["state"].cpu().double()
    if predicted.shape != truth.shape:
        raise ValueError("Prediction and reference shapes differ")
    if not bool(torch.isfinite(predicted).all()):
        return {name: None for name in ("error_rms", "error_max", "centered_rms", "mean_error",
            "upper_rms", "upper_max", "spectral_low_rms", "spectral_high_rms", "minimum", "maximum",
            "physical_interval_violations")} | {"finite": False, "reference_accepted": bool(ref["accepted"])}
    error = predicted - truth
    mean = error.mean((-2, -1), keepdim=True)
    n, m = error.shape[-2:]
    fx = torch.fft.fftfreq(n) * n
    fy = torch.fft.fftfreq(m) * m
    low_mask = (fx[:, None].square() + fy[None, :].square()).sqrt() <= min(n, m) / 4
    transformed = torch.fft.fft2(error, norm="ortho")
    low = torch.fft.ifft2(transformed * low_mask, norm="ortho").real
    high = error - low
    rms, maximum = float(error.square().mean().sqrt()), float(error.abs().max())
    uncertainty_rms = ref.get("uncertainty_rms")
    uncertainty_max = ref.get("uncertainty_max_bound")
    return {"error_rms": rms, "error_max": maximum,
            "centered_rms": float((error - mean).square().mean().sqrt()),
            "mean_error": float(mean.abs().max()),
            "upper_rms": rms + float(uncertainty_rms) if uncertainty_rms is not None else None,
            "upper_max": maximum + float(uncertainty_max) if uncertainty_max is not None else None,
            "spectral_low_rms": float(low.square().mean().sqrt()),
            "spectral_high_rms": float(high.square().mean().sqrt()),
            "minimum": float(predicted.min()), "maximum": float(predicted.max()),
            "physical_interval_violations": int(((predicted < 0) | (predicted > 1)).sum()),
            "finite": True, "reference_accepted": bool(ref["accepted"])}


def rollout(model, initial, schedule, equation, geometry, budget=None, *, capture_intermediates=False):
    if not schedule or any(not math.isfinite(float(h)) or float(h) <= 0 for h in schedule):
        raise ValueError("Rollout requires a nonempty positive finite schedule")
    state, middle = initial, []
    if getattr(model, "family", None) == "df" and not capture_intermediates:
        # Exact heat semigroup fusion belongs to the uncorrected splitting
        # control only. A learned/analytic endpoint increment breaks adjacency.
        from .numerics import heat_step, galerkin_reaction_step
        from tdn.numerics.subflows import reaction_step
        state = heat_step(initial, schedule[0] / 2, equation, geometry, model.track, cache=model.cache)
        for index, h in enumerate(schedule):
            if budget is not None:
                budget.check()
            state = (reaction_step(state, h, equation) if model.track == "discrete" else
                     galerkin_reaction_step(state, h, equation, substeps=model.reaction_substeps))
            next_h = schedule[index + 1] if index + 1 < len(schedule) else 0.
            state = heat_step(state, (h + next_h) / 2, equation, geometry, model.track, cache=model.cache)
        return state, []
    for h in schedule:
        if budget is not None:
            budget.check()
        state = model(state, float(h), equation, geometry)
        if capture_intermediates:
            middle.append(state)
    return state, middle


def _samples(parents, horizons, n, track, device, base, floor):
    """Prepare train/validation tensors once; no confirmation access."""
    samples = []
    with torch.no_grad():
        for parent in parents:
            eq, geom = eq_geom(parent, n)
            initial = parent["states"][str(n)].to(device=device, dtype=torch.float32)
            for horizon in horizons:
                ref = reference(parent, n, horizon, track)
                if not ref["accepted"]:
                    raise ValueError("Training and validation require accepted teachers")
                truth = ref["state"].to(device=device, dtype=torch.float32)
                baseline = base(initial, horizon, eq, geom)
                scale = max(float((baseline - truth).square().mean()),
                            float(ref.get("uncertainty_rms", 0.)) ** 2, floor ** 2)
                samples.append({"parent_id": parent["parent_id"], "field_cluster": parent["field_cluster"],
                    "regime": parent.get("regime", "unspecified"), "u": initial, "h": horizon,
                    "eq": eq, "geom": geom, "target": truth, "scale": scale})
    return samples


@torch.no_grad()
def validation_metrics(model, samples, ctx):
    values = []
    model.eval()
    for sample in samples:
        _budget(ctx)
        prediction = model(sample["u"], sample["h"], sample["eq"], sample["geom"])
        if not bool(torch.isfinite(prediction).all()):
            raise TrialNumericalFailure("NONFINITE_VALIDATION_PREDICTION")
        error = prediction.double() - sample["target"].double()
        values.append({"field_cluster": sample["field_cluster"],
                       "loss": float(error.square().mean() + float(ctx.protocol["training"].get("peak_weight", .1))
                                     * error.square().amax()) / sample["scale"],
                       "rms": float(error.square().mean().sqrt()), "maximum": float(error.abs().max())})
    clusters = sorted({item["field_cluster"] for item in values})
    if not clusters:
        raise ValueError("Validation cannot be empty")
    # Each independent field gets one weight despite physics/horizon repeats.
    return {"objective": statistics.mean(statistics.mean(item["loss"] for item in values
                if item["field_cluster"] == cluster) for cluster in clusters),
            "rms": statistics.mean(statistics.mean(item["rms"] for item in values
                if item["field_cluster"] == cluster) for cluster in clusters),
            "maximum": max(item["maximum"] for item in values), "field_count": len(clusters)}


def _trial(ctx, spec, config, samples, validation, learning_rate, updates, phase):
    from .models import make_model
    start = time.perf_counter()
    seed = spec["seed"] if spec["seed"] is not None else 0
    torch.manual_seed(seed)
    model = make_model(spec["family"], spec["track"], config).to(ctx.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    settings = ctx.protocol["training"]
    batch_size = int(settings.get("batch_size", 1))
    rng = random.Random(seed)
    filename = spec["model_id"].replace("/", "__")
    initial_state = copy.deepcopy(model.state_dict())
    initial_path = Path(ctx.path) / "checkpoints" / f"{filename}.initial.pt"
    payload = {"protocol_sha256": digest(ctx.protocol), "selection_split": "validation",
               **spec, "effective_config": config}
    atomic_torch_save({**payload, "state_dict": initial_state, "updates_selected": 0}, initial_path)
    initial = best = None
    selected_state, selected_update, completed, failure = initial_state, 0, 0, None
    curves = Path(ctx.path) / "learning_curves.jsonl"
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate,
        weight_decay=float(settings.get("weight_decay", 0.))) if parameters else None
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in parameters)
    examples_seen, last_validation, accumulated_update_seconds = 0, None, 0.

    def log_curve(update, loss, gradient, update_seconds, measured_validation, train_rms=None, train_max=None):
        row = {**spec, "phase": phase, "update": update, "train_loss": loss,
               "validation_loss": measured_validation["objective"] if measured_validation else None,
               "validation_rms": measured_validation["rms"] if measured_validation else None,
               "validation_max": measured_validation["maximum"] if measured_validation else None,
               "validation_measured": measured_validation is not None,
               "train_rms": train_rms, "train_max": train_max,
               "learning_rate": learning_rate if parameters else None, "gradient_norm": gradient,
               "elapsed_seconds": time.perf_counter() - start, "update_seconds": update_seconds,
               "examples_seen": examples_seen, "parameters": parameter_count,
               "trainable_parameters": trainable_count, "device": str(ctx.device)}
        with curves.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    try:
        initial = best = validation_metrics(model, validation, ctx)
        last_validation = initial
        log_curve(0, None, None, 0., initial)
        for update in range(1, updates + 1 if parameters else 1):
            _budget(ctx)
            _sync(ctx.device)
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, raw_mses, raw_maxima = [], [], []
            for _ in range(batch_size):
                sample = samples[rng.randrange(len(samples))]
                prediction = model(sample["u"], sample["h"], sample["eq"], sample["geom"])
                squared = (prediction - sample["target"]).square()
                loss = (squared.mean() + float(settings.get("peak_weight", .1)) * squared.amax()) / sample["scale"]
                if not bool(torch.isfinite(loss)):
                    raise TrialNumericalFailure("NONFINITE_TRAINING_LOSS")
                (loss / batch_size).backward()
                losses.append(float(loss.detach()))
                raw_mses.append(float(squared.detach().mean()))
                raw_maxima.append(float(squared.detach().amax().sqrt()))
            gradient = torch.nn.utils.clip_grad_norm_(parameters, float(settings.get("clip_grad_norm", 1.)))
            if not bool(torch.isfinite(gradient)):
                raise TrialNumericalFailure("NONFINITE_GRADIENT")
            optimizer.step()
            _sync(ctx.device)
            update_seconds = time.perf_counter() - tick
            accumulated_update_seconds += update_seconds
            examples_seen += batch_size
            completed = update
            measured = None
            if update % int(settings.get("validation_every", 25)) == 0 or update == updates:
                measured = validation_metrics(model, validation, ctx)
                last_validation = measured
                if measured["objective"] < best["objective"]:
                    best, selected_update = measured, update
                    selected_state = copy.deepcopy(model.state_dict())
            log_curve(update, statistics.mean(losses), float(gradient), update_seconds, measured,
                      math.sqrt(statistics.mean(raw_mses)), max(raw_maxima))
    except TrialNumericalFailure as error:
        failure = str(error)
        with curves.open("a") as handle:
            handle.write(json.dumps({**spec, "phase": phase, "update": completed,
                "status": "NUMERICAL_FAILURE", "error": failure,
                "elapsed_seconds": time.perf_counter() - start}, allow_nan=False) + "\n")
    # Programming errors, OOM and walltime are not disguised as bad science.
    selected = ("TRAINED_CHECKPOINT" if selected_update else "SELECTED_INITIALIZATION") if parameters else (
        "FROZEN_INITIALIZATION" if spec["family"] == "rank1_frozen" else "ANALYTIC_CONTROL")
    path = Path(ctx.path) / "checkpoints" / f"{filename}.selected.pt"
    atomic_torch_save({**payload, "state_dict": selected_state, "updates_selected": selected_update,
                      "checkpoint_validated": best is not None, "numerical_failure": failure}, path)
    record = {**spec, "phase": phase, "effective_config": config,
        "architecture": model.architecture_metadata(), "parameter_count": parameter_count,
        "parameters": parameter_count, "trainable_parameters": trainable_count,
        "learning_rate": learning_rate if parameters else None, "batch_size": batch_size,
        "updates_requested": updates if parameters else 0, "updates_completed": completed,
        "updates_selected": selected_update, "examples_seen": examples_seen,
        "selected_state": selected, "selection": selected, "validation": best,
        "initial_validation": initial, "last_validation": last_validation,
        "training_outcome": "NUMERICAL_FAILURE" if failure else "COMPLETED",
        "numerical_failure": failure, "checkpoint_validated": best is not None,
        "checkpoint": path.relative_to(ctx.path).as_posix(), "checkpoint_sha256": file_digest(path),
        "initial_checkpoint": initial_path.relative_to(ctx.path).as_posix(),
        "initial_checkpoint_sha256": file_digest(initial_path),
        "elapsed_seconds": time.perf_counter() - start, "optimizer_seconds": accumulated_update_seconds,
        "eligible_for_tuning": best is not None and failure is None,
        "baseline_adequacy": "PROJECT_ADAPTED_CONTROL_NOT_PAPER_REPRODUCTION",
        "training_gain_over_initialization": (initial["objective"] / max(best["objective"], 1.e-30)
                                               if initial and best else None)}
    write_json(Path(ctx.path) / "trials" / f"{filename}.json", record)
    from .core import check
    ctx.record(f"training/{spec['model_id']}", ["G3"], metrics=record,
        checks=[check("finite_validated_checkpoint", best is not None, True, "eq", category="correctness"),
                check("requested_updates_complete", completed, updates if parameters else 0, "eq", category="correctness"),
                check("validation_improved_initialization", best["objective"] if best else None,
                      initial["objective"] if initial else None, "lt", category="gap", applicable=bool(parameters)),
                check("learned_architecture_benefit", None, True, "eq", category="math",
                      reason="Structural tests and validation selection do not establish fresh-field superiority")],
        config={**spec, "effective_config": config}, evidence={"selection_split": "validation"},
        status=record["training_outcome"])
    return record


def train(ctx):
    from .data import load_split
    from .models import make_model
    train_bank, validation_bank = load_split(ctx, "train"), load_split(ctx, "validation")
    if {p["field_cluster"] for p in train_bank} & {p["field_cluster"] for p in validation_bank}:
        raise ValueError("Independent fields overlap between training and validation")
    protocol, settings = ctx.protocol, ctx.protocol["training"]
    specs = expected_model_specs(protocol)
    plan = {"schema": "tdn.frontier-training-plan/v1", "protocol_sha256": digest(protocol),
        "expected_models": specs, "training": settings, "models": protocol["models"],
        "objective": "field-balanced validation mean of DF-defect-normalized endpoint MSE plus weighted maximum squared error",
        "normalization": "same track-matched DF control and absolute noise floor for every family",
        "confirmation_access": False, "equal_updates_are_not_equal_compute": True,
        "confirmation_schedule_plan": confirmation_plan(protocol),
        "nested_field_subsets": {str(count): sorted({p["field_cluster"] for p in nested_subset(train_bank, count)})
                                  for count in settings["subset_sizes"]}}
    write_json(Path(ctx.path) / "training_plan.json", plan)
    records, tuning = [], []
    for track in protocol["tracks"]:
        base = make_model("df", track, _config(protocol, "df")).to(ctx.device).eval()
        validation = _samples(validation_bank, protocol["validation_horizons"], protocol["train_grid"],
                              track, ctx.device, base, settings.get("loss_floor", 1.e-6))
        sample_banks = {count: _samples(nested_subset(train_bank, count), protocol["train_horizons"],
                protocol["train_grid"], track, ctx.device, base, settings.get("loss_floor", 1.e-6))
                       for count in settings["subset_sizes"]}
        largest = max(settings["subset_sizes"])
        for family in protocol["models"]:
            config = _config(protocol, family)
            selected_tuning = None
            if family in TRAINABLE:
                candidates = []
                for index, rate in enumerate(settings["learning_rates"]):
                    spec = {"model_id": f"{track}/{family}/tuning{index}", "family": family,
                            "track": track, "seed": protocol["seeds"][0], "train_count": largest}
                    candidate = _trial(ctx, spec, config, sample_banks[largest], validation, float(rate),
                                       int(settings["tuning_updates"]), "tuning")
                    candidates.append(candidate)
                tuning.extend(candidates)
                eligible = [r for r in candidates if r["eligible_for_tuning"]]
                selected_tuning = min(eligible, key=lambda r: (r["validation"]["objective"], r["model_id"])) if eligible else None
            for spec in (s for s in specs if s["family"] == family and s["track"] == track):
                rate = selected_tuning["learning_rate"] if selected_tuning else float(settings["learning_rates"][0])
                record = _trial(ctx, spec, config, sample_banks[spec["train_count"] or largest], validation,
                    rate, int(settings["updates"]) if family in TRAINABLE else 0, "final")
                record["tuning_selection"] = selected_tuning["model_id"] if selected_tuning else None
                record["all_tuning_failed"] = family in TRAINABLE and selected_tuning is None
                records.append(record)
    catalog = {"schema": "tdn.frontier-catalog/v1", "protocol_sha256": digest(protocol),
        "selection_split": "validation", "frozen_before_confirmation": True,
        "records": records, "tuning_records": tuning, "training_plan_sha256": file_digest(Path(ctx.path) / "training_plan.json"),
        "scope": "All declared data fractions and seeds retained; project adapted controls, not a paper reproduction",
        "selected_policy_models": {track: next((r["model_id"] for r in records
            if r["track"] == track and r["family"] == "rank1" and r["seed"] == protocol["seeds"][0]
            and r["train_count"] == max(settings["subset_sizes"])), None) for track in protocol["tracks"]},
        "policy_selection": "predeclared rank1, first seed, largest data fraction; no confirmation-based selection"}
    write_json(Path(ctx.path) / "catalog.json", catalog)
    write_json(Path(ctx.path) / "freeze.json", {"protocol_sha256": digest(protocol),
        "catalog_sha256": file_digest(Path(ctx.path) / "catalog.json"),
        "checkpoint_hashes": {r["model_id"]: r["checkpoint_sha256"] for r in records}})
    return {"selected_models": len(records), "tuning_trials": len(tuning),
            "initialization_selections": sum(r["selected_state"] == "SELECTED_INITIALIZATION" for r in records),
            "numerically_failed_trials": sum(r["numerical_failure"] is not None for r in records + tuning),
            "updates_completed": sum(r["updates_completed"] for r in records + tuning)}


def load_models(ctx):
    """Verify checkpoint bytes and declared inventory before any fresh data read."""
    from .data import verify_frozen_training
    from .models import make_model
    verify_frozen_training(ctx)
    base = Path(ctx.prerequisites["train"])
    catalog = json.loads((base / "catalog.json").read_text())
    records = catalog["records"]
    if {r["model_id"] for r in records} != {r["model_id"] for r in expected_model_specs(ctx.protocol)}:
        raise ValueError("Frozen model inventory differs from the complete declared comparison")
    models = {}
    for record in records:
        if not record["checkpoint_validated"]:
            continue
        checkpoint = torch.load(base / record["checkpoint"], map_location="cpu", weights_only=True)
        if (checkpoint.get("protocol_sha256") != digest(ctx.protocol) or
                checkpoint.get("selection_split") != "validation" or
                any(checkpoint.get(key) != record.get(key) for key in
                    ("model_id", "family", "track", "seed", "train_count", "effective_config")) or
                checkpoint.get("checkpoint_validated") is not True):
            raise ValueError("Checkpoint identity differs from its frozen selection record")
        model = make_model(record["family"], record["track"], record["effective_config"]).to(ctx.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().requires_grad_(False)
        model.selection_metadata = record
        models[record["model_id"]] = model
    return models


def schedules(protocol):
    result = []
    for index, value in enumerate(protocol["confirm_schedules"]):
        if isinstance(value, dict):
            identity, steps = value.get("id", f"schedule-{index}"), value["steps"]
        else:
            identity, steps = f"schedule-{index}", value
        if not steps or any(not math.isfinite(float(h)) or h <= 0 for h in steps):
            raise ValueError("Confirmation schedules must be positive and finite")
        result.append((identity, [float(h) for h in steps]))
    if len({name for name, _ in result}) != len(result):
        raise ValueError("Duplicate schedule identity")
    return result


def confirmation_plan(protocol):
    """Endpoint-only classical frontiers plus separately audited primary paths.

    If a uniform classical workload already equals a primary schedule, use its
    existing measurement. Repeated copies must not create an extra minimum of
    timing noise. Every requested horizon/count maps to one exact schedule.
    """
    planned = [{"schedule_id": name, "schedule": steps, "families": list(protocol["models"]),
                "kind": "primary", "requested_outputs": "endpoint_only", "diagnostic_intermediates": True}
               for name, steps in schedules(protocol)]
    count_grid = protocol.get("classical_frontier_steps", [1, 2, 4, 8])
    if not count_grid or any(type(count) is not int or count < 1 for count in count_grid) or len(set(count_grid)) != len(count_grid):
        raise ValueError("Classical frontier step counts must be distinct positive integers")
    controls = [family for family in protocol["models"] if family in CLASSICAL]
    horizons = sorted({round(math.fsum(steps), 12) for _, steps in schedules(protocol)})
    if "evaluation_horizons" in protocol and set(horizons) != {round(float(h), 12) for h in protocol["evaluation_horizons"]}:
        raise ValueError("Evaluation horizons and primary endpoint horizons differ")
    coverage = []
    for horizon in horizons:
        for count in count_grid:
            steps = [horizon / count] * count
            existing = next((item for item in planned if len(item["schedule"]) == count and
                all(math.isclose(a, b, rel_tol=1.e-11, abs_tol=1.e-14)
                    for a, b in zip(item["schedule"], steps))), None)
            if existing is None:
                existing = {"schedule_id": f"classical-uniform-T{horizon_key(horizon)}-steps{count}",
                    "schedule": steps, "families": controls, "kind": "classical_frontier",
                    "requested_outputs": "endpoint_only", "diagnostic_intermediates": False}
                planned.append(existing)
            coverage.append({"final_time": horizon, "steps": count,
                "schedule_id": existing["schedule_id"], "reused_primary_measurement": existing["kind"] == "primary",
                "families": controls, "requested_outputs": "endpoint_only"})
    if len({item["schedule_id"] for item in planned}) != len(planned):
        raise ValueError("Classical and primary schedule identities collide")
    return {"schedules": planned, "classical_frontier_coverage": coverage,
            "scope": "fixed requested endpoints; primary intermediate diagnostics are audited separately and not charged as inference outputs"}


def paired_field_summary(rows, *, bootstrap=1000, seed=819):
    """Same-schedule rank-one vs FNO ratios; resample independent fields only."""
    anchors, physical = {}, {}
    for row in rows:
        if row.get("status") != "COMPLETED" or not row.get("reference_accepted"):
            continue
        key = (row["track"], row["parent_id"], row["grid"], row["schedule_id"],
               row["final_time"], row["seed"], row["train_count"])
        anchors.setdefault(key, {})[row["family"]] = row
        if row["family"] in ("analytic_quad", "analytic_quad_cubic", "df", "etdrk4"):
            physical.setdefault(key[:5], {})[row["family"]] = row
    grouped = {}
    for anchor_key, members in anchors.items():
        ours = members.get("rank1")
        if ours is None:
            continue
        for competitor in ("fno_small", "fno_standard", "direct_fno", "rank1_postcompression", "rank1_frozen",
                           "analytic_quad", "analytic_quad_cubic", "df", "etdrk4"):
            other = members.get(competitor, physical.get(anchor_key[:5], {}).get(competitor))
            if other is None or min(ours["error_rms"], other["error_rms"], ours["cost_seconds"], other["cost_seconds"]) <= 0:
                continue
            group_key = (ours["track"], ours["final_time"], ours["grid"], ours["train_count"], competitor)
            grouped.setdefault(group_key, {}).setdefault(ours["field_cluster"], []).append({
                "log_error_ratio": math.log(other["error_rms"] / ours["error_rms"]),
                "log_speed_ratio": math.log(other["cost_seconds"] / ours["cost_seconds"])})
    summaries = []
    rng = random.Random(seed)
    for (track, horizon, grid, count, competitor), fields in sorted(grouped.items()):
        row = {"track": track, "final_time": horizon, "grid": grid,
               "train_count": count, "family": "rank1", "competitor": competitor,
               "independent_fields": len(fields), "paired_rows": sum(map(len, fields.values())),
               "independence_unit": "field_cluster; physics/grids/schedules/seeds averaged within field",
               "training_seed_scope": "registered seeds are fixed; field interval is conditional on their ensemble",
               "inference": "descriptive paired cluster bootstrap; no multiple-comparison superiority declaration"}
        for metric in ("log_error_ratio", "log_speed_ratio"):
            cluster_means = [statistics.mean(item[metric] for item in items) for items in fields.values()]
            estimate = math.exp(statistics.mean(cluster_means))
            interval = None
            if len(cluster_means) >= 2:
                sampled = sorted(math.exp(statistics.mean(rng.choices(cluster_means, k=len(cluster_means))))
                                 for _ in range(bootstrap))
                interval = [sampled[int(.025 * (bootstrap - 1))], sampled[int(.975 * (bootstrap - 1))]]
            row[metric.removeprefix("log_")] = estimate
            row[metric.removeprefix("log_") + "_ci95"] = interval
        summaries.append(row)
    return summaries


def matched_comparisons(rows, protocol):
    """Compare paired seeds/fractions; never select the best competitor seed."""
    from .measurement import paired_frontiers
    output = []
    for track in protocol["tracks"]:
        for count in protocol["training"]["subset_sizes"]:
            for seed in protocol["seeds"]:
                subset = [r for r in rows if r["track"] == track and (
                    (r["seed"] == seed and r["train_count"] == count) or
                    r["family"] in ("df", "etdrk4", "analytic_quad", "analytic_quad_cubic"))]
                ids = {r["family"]: r["model_id"] for r in subset}
                if "rank1" not in ids:
                    continue
                comparators = {family: [ids[family]] for family in ("fno_small", "fno_standard", "direct_fno",
                    "analytic_quad", "analytic_quad_cubic") if family in ids}
                classical = [ids[family] for family in ("df", "etdrk4", "analytic_quad", "analytic_quad_cubic") if family in ids]
                if classical:
                    comparators["strongest_classical"] = classical
                if not comparators:
                    continue
                comparisons = paired_frontiers(subset, target_pairs=[(v, v) for v in protocol["targets"]],
                                                comparators=comparators)
                output.extend({**item, "train_count": count} for item in comparisons if item["model_id"] == ids["rank1"])
    return output


def confirmation_gate(catalog, rows, comparisons, protocol):
    """A continuation gate, never a post-confirmation model selection rule."""
    selected = catalog["selected_policy_models"]
    threshold = float(protocol.get("practical_speedup", 1.2))
    primary = float(protocol.get("primary_target", protocol["targets"][0]))
    results = {}
    for track in protocol["tracks"]:
        model_id = selected[track]
        members = [r for r in comparisons if r["track"] == track and r["model_id"] == model_id
                   and r["comparator"] == "strongest_classical" and r["rms_target"] == primary
                   and r["grid"] == protocol["policies"]["grid"]
                   and math.isclose(r["final_time"], protocol["policies"]["horizon"], rel_tol=1.e-10)]
        eligible = [r for r in members if r["eligibility"] == "ELIGIBLE"]
        clustered = {}
        for item in eligible:
            clustered.setdefault(item["field_cluster"], []).append(math.log(item["control_over_candidate_speed_ratio"]))
        values = [statistics.mean(v) for v in clustered.values()]
        ci = None
        if len(values) >= 2:
            rng = random.Random(8701)
            boot = sorted(math.exp(statistics.mean(rng.choices(values, k=len(values))))
                          for _ in range(int(protocol.get("bootstrap_replicates", 1000))))
            ci = [boot[int(.025 * (len(boot) - 1))], boot[int(.975 * (len(boot) - 1))]]
        gain = float(protocol["training"].get("baseline_min_relative_improvement", .05))
        adequate = {}
        for family in ("fno_small", "fno_standard", "direct_fno"):
            records = [r for r in catalog["records"] if r["track"] == track and r["family"] == family
                       and r["train_count"] == max(protocol["training"]["subset_sizes"])]
            trained = bool(records) and all(r["training_outcome"] == "COMPLETED" and
                r["selected_state"] == "TRAINED_CHECKPOINT" and r["validation"]["objective"] <=
                (1 - gain) * r["initial_validation"]["objective"] for r in records)
            measured = [r for r in rows if r["track"] == track and r["family"] == family and
                        r["train_count"] == max(protocol["training"]["subset_sizes"])]
            accurate = any(r["target_results"].get(f"{primary:g}", False) for r in measured)
            adequate[family] = {"trained_all_seeds": trained, "any_primary_accuracy": accurate,
                                "adequate_for_local_claim": trained and accurate}
        results[track] = {"selected_policy_model_id": model_id, "primary_target": primary,
            "matched_cells": len(members), "eligible_cells": len(eligible), "independent_fields": len(values),
            "classical_over_candidate_geometric_ratio": math.exp(statistics.mean(values)) if values else None,
            "field_cluster_ci95": ci, "speed_threshold": threshold,
            "solver_utility_eligible": bool(members and len(eligible) == len(members) and ci and ci[0] >= threshold),
            "neural_baseline_adequate": all(adequate[f]["adequate_for_local_claim"] for f in ("fno_small", "fno_standard")),
            "baseline_details": adequate}
    return {"schema": "tdn.frontier-confirmation-gate/v1", "tracks": results,
        "solver_utility_eligible": any(r["solver_utility_eligible"] for r in results.values()),
        "neural_baseline_adequate": all(r["neural_baseline_adequate"] for r in results.values()),
        "selected_policy_models": selected,
        "selection": "fixed before confirmation; gate can block deployment but cannot choose a new model",
        "scope": "Local baseline adequacy diagnostic, not proof of literature competitiveness; posthoc schedule margins are necessary, not sufficient, for deployment"}


def confirm(ctx):
    from .core import check
    from .data import load_split
    from .measurement import measure_paired
    models = load_models(ctx)
    catalog = json.loads((Path(ctx.prerequisites["train"]) / "catalog.json").read_text())
    write_json(Path(ctx.path) / "frozen_model_ledger.json", {"models": catalog["records"],
        "catalog_sha256": file_digest(Path(ctx.prerequisites["train"]) / "catalog.json"),
        "protocol_sha256": digest(ctx.protocol)})
    plan = confirmation_plan(ctx.protocol)
    write_json(Path(ctx.path) / "confirmation_plan.json", plan)
    parents = load_split(ctx, "confirmation")
    targets = [float(value) for value in ctx.protocol["targets"]]
    rows, timings = [], []
    timing = ctx.protocol.get("timing", {})
    with torch.no_grad():
        for parent in parents:
            for n in ctx.protocol["grids"]:
                equation, geometry = eq_geom(parent, n)
                initial = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
                for track in ctx.protocol["tracks"]:
                    for specification in plan["schedules"]:
                        active_families = specification["families"]
                        if not active_families:
                            continue
                        selected = {name: model for name, model in models.items()
                                    if model.track == track and model.family in active_families}
                        schedule_id, schedule = specification["schedule_id"], specification["schedule"]
                        diagnostic_intermediates = specification["diagnostic_intermediates"]
                        _budget(ctx)
                        group_start = len(rows)
                        final_time = round(sum(schedule), 12)
                        ref = reference(parent, n, final_time, track)
                        order_seed = int(digest([parent["parent_id"], n, track, schedule_id])[:12], 16)
                        calls = {name: (lambda model=model: rollout(model, initial, schedule, equation, geometry)[0])
                                 for name, model in selected.items()}
                        # All arms, including every classical reference, enter the
                        # same randomized interleaved timing rounds.
                        if calls:
                            answers, timing_report = measure_paired(calls, device=ctx.device,
                                repeats=int(timing.get("repeats", ctx.protocol.get("timing_repeats", 3))),
                                warmup=int(timing.get("warmup", ctx.protocol.get("timing_warmup", 1))),
                                seed=order_seed, budget=ctx.budget)
                        else:
                            answers, timing_report = {}, {"methods": {}, "rounds": [], "status": "NO_VALIDATED_CHECKPOINTS"}
                        timing_id = f"{parent['parent_id']}/N{n}/{track}/{schedule_id}"
                        timings.append({"timing_id": timing_id, "parent_id": parent["parent_id"], "grid": n,
                                        "track": track, "schedule": schedule, "final_time": final_time,
                                        "requested_outputs": "endpoint_only", "schedule_kind": specification["kind"], **timing_report})
                        for model_id, model in selected.items():
                            metadata = model.selection_metadata
                            errors = endpoint_errors(answers[model_id], ref)
                            cost = timing_report["methods"][model_id]
                            intermediate = []
                            if errors["finite"] and diagnostic_intermediates:
                                _, middle = rollout(model, initial, schedule, equation, geometry,
                                                    ctx.budget, capture_intermediates=True)
                                elapsed = 0.
                                for h, value in zip(schedule[:-1], middle[:-1]):
                                    elapsed = round(elapsed + h, 12)
                                    intermediate.append({"time": elapsed,
                                        **endpoint_errors(value, reference(parent, n, elapsed, track))})
                            middle_valid = bool(intermediate) and all(item["finite"] and item["reference_accepted"]
                                and item["upper_rms"] is not None and item["upper_max"] is not None for item in intermediate)
                            middle_upper = max((item["upper_max"] for item in intermediate), default=None) if middle_valid else None
                            row = {"model_id": model_id, "family": model.family, "seed": metadata["seed"],
                                "train_count": metadata["train_count"], "parameters": metadata["parameter_count"],
                                "selected_state": metadata["selected_state"], "updates_selected": metadata["updates_selected"],
                                "parent_id": parent["parent_id"], "field_cluster": parent["field_cluster"],
                                "regime": parent.get("regime", "unspecified"), "grid": n, "track": track,
                                "kappa": equation.kappa, "reaction_rate": equation.reaction_rate,
                                "schedule_id": schedule_id, "schedule": schedule, "horizon": final_time,
                                "schedule_kind": specification["kind"], "requested_outputs": "endpoint_only",
                                "diagnostic_intermediates": diagnostic_intermediates,
                                "final_time": final_time, **errors, "cost_seconds": cost["median_seconds"],
                                "median_seconds": cost["median_seconds"], "cold_seconds": cost["cold_seconds"],
                                "timing_id": timing_id, "timing": cost, "intermediates": intermediate,
                                "classical_heat_fusion": model.family == "df",
                                "intermediate_reference_count": len(intermediate),
                                "intermediate_reference_track": track,
                                "intermediate_upper_max": middle_upper,
                                "intermediate_references_accepted": middle_valid,
                                "diagnostic_intermediates_unfused_df": model.family == "df",
                                "intermediate_target_results": {f"{value:g}": bool(middle_valid and
                                    all(item["upper_rms"] <= value and item["upper_max"] <= value for item in intermediate))
                                    for value in targets},
                                "target_results": {f"{value:g}": bool(errors["finite"] and ref["accepted"] and
                                    errors["upper_rms"] <= value and errors["upper_max"] <= value) for value in targets},
                                "status": "COMPLETED" if errors["finite"] else "NUMERICAL_FAILURE"}
                            rows.append(row)
                            ctx.record(f"confirmation/{model_id}/{timing_id}", ["G3"], metrics=row,
                                config={"family": model.family, "track": track, "final_time": final_time},
                                checks=[check("finite_prediction", errors["finite"], True, "eq", category="correctness"),
                                    check("accepted_reference", ref["accepted"], True, "eq", category="correctness"),
                                    check("joint_rms_accuracy", errors["upper_rms"], targets[0], "le", category="gap", applicable=bool(ref["accepted"])),
                                    check("joint_max_accuracy", errors["upper_max"], targets[0], "le", category="gap", applicable=bool(ref["accepted"])),
                                    check("physical_interval", errors["physical_interval_violations"], 0, "eq", category="math",
                                          reason="Measured sampled states only; no invariant theorem"),
                                    check("all_intermediate_accuracy", middle_upper, targets[0], "le", category="math",
                                          applicable=middle_valid, required=diagnostic_intermediates,
                                          reason="Every primary interior state uses an independent same-track teacher; classical extras request endpoints only"),
                                    check("deployment_speedup", None, True, "eq", category="utility",
                                          reason="Prescribed endpoint tests are not an adaptive deployment policy")],
                                evidence={"checkpoint_sha256": metadata["checkpoint_sha256"],
                                          "paper_reproduction": False}, status=row["status"])
                        # Preserve unvalidated training arms as explicit missing
                        # measurements rather than removing them from coverage.
                        for metadata in catalog["records"]:
                            if (metadata["track"] != track or metadata["model_id"] in selected or
                                    metadata["family"] not in active_families):
                                continue
                            row = {**{key: metadata[key] for key in ("model_id", "family", "seed", "train_count")},
                                "parent_id": parent["parent_id"], "field_cluster": parent["field_cluster"],
                                "regime": parent.get("regime", "unspecified"), "grid": n, "track": track,
                                "schedule_id": schedule_id, "schedule": schedule, "final_time": final_time,
                                "schedule_kind": specification["kind"], "requested_outputs": "endpoint_only",
                                "diagnostic_intermediates": diagnostic_intermediates,
                                "horizon": final_time, "status": "UNAVAILABLE_CHECKPOINT", "error_rms": None,
                                "error_max": None, "upper_rms": None, "upper_max": None, "cost_seconds": None,
                                "reference_accepted": bool(ref["accepted"]), "target_results": {f"{v:g}": False for v in targets}}
                            rows.append(row)
                            ctx.record(f"confirmation/{metadata['model_id']}/{timing_id}", ["G3"], metrics=row,
                                config={"family": metadata["family"], "track": track, "final_time": final_time},
                                checks=[check("validated_checkpoint_available", False, True, "eq", category="correctness"),
                                        check("accuracy", None, targets[0], "le", category="gap"),
                                        check("mathematical_response", None, True, "eq", category="math")],
                                evidence={"numerical_failure": metadata["numerical_failure"]},
                                status="UNAVAILABLE_CHECKPOINT")
                        with (Path(ctx.path) / "confirmation_rows.partial.jsonl").open("a") as handle:
                            for row in rows[group_start:]:
                                handle.write(json.dumps(row, allow_nan=False) + "\n")
    comparison = matched_comparisons(rows, ctx.protocol)
    comparisons = {"scope": "reference-informed posthoc frontiers, not deployable adaptive decisions",
        "horizon_matching": "parent/grid/track/final_time/RMS target/maximum target",
        "frontiers": comparison, "paired_field_summaries": paired_field_summary(rows,
            bootstrap=int(ctx.protocol.get("bootstrap_replicates", 1000))),
        "paper_reproduction": False, "independence_unit": "field_cluster"}
    write_json(Path(ctx.path) / "confirmation_rows.json", {"rows": rows,
        "independence_unit": "field_cluster; seeds, grids, schedules and physics are paired"})
    write_json(Path(ctx.path) / "timing_rounds.json", {"groups": timings,
        "cold_definition": "first observed invocation, not guaranteed process-cold startup"})
    write_json(Path(ctx.path) / "comparisons.json", comparisons)
    gate = confirmation_gate(catalog, rows, comparison, ctx.protocol)
    write_json(Path(ctx.path) / "gate.json", gate)
    partial = Path(ctx.path) / "confirmation_rows.partial.jsonl"
    if partial.exists():
        partial.unlink()
    return {"models": len(models), "declared_models": len(catalog["records"]), "endpoint_rows": len(rows),
            "independent_fields": len({p["field_cluster"] for p in parents}),
            "failed_or_missing_endpoints": sum(r["status"] != "COMPLETED" for r in rows),
            "scope": "frozen project controls; descriptive cluster uncertainty; no FNO paper superiority claim"}
