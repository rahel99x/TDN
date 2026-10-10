"""Bounded advance training, exact recovery, frozen selection and paired tests.

An independent field is the statistical unit. Timings and failed feasibility
remain visible. No confirmation truth chooses a checkpoint or schedule.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time
import torch
from tdn.analysis.frontier.core import clean, check
from tdn.analysis.frontier.data import _safe_file
from tdn.analysis.frontier.neural import (nested_subset, eq_geom, reference, rollout,
    _samples, validation_metrics, TrialNumericalFailure, _sync)
from tdn.analysis.frontier.measurement import endpoint_eligibility, measure_paired
from tdn.analysis.portfolio.statistics import field_cluster_interval
from tdn.research.experiment import atomic_torch_save
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json
from .data import binding, unit, load_bank, declarations


def model_specs(protocol, scope=None):
    from .models import MODEL_SPECS
    scope = scope or {}
    rows = []
    for family in scope.get("families", protocol["models"]):
        fitted = bool(MODEL_SPECS[family]["trainable"])
        seeds = scope.get("seeds", protocol["seeds"])
        if not fitted and protocol["seeds"][0] not in seeds:
            continue
        for track in scope.get("tracks", protocol["tracks"]):
            sizes = protocol["training"].get("subset_sizes_by_family", {}).get(family, protocol["training"]["subset_sizes"])
            for seed, size in itertools.product(seeds, sizes) if fitted else [(None, 0)]:
                rows.append(dict(model_id=f"{track}/{family}/n{size}/seed{seed}", family=family,
                                 track=track, seed=seed, train_count=size))
    return rows


def shared_schedules(protocol):
    result = {}
    for index, schedule in enumerate(protocol["primary_schedules"]):
        schedule = list(map(float, schedule))
        key = tuple(round(h, 12) for h in schedule)
        result[key] = dict(schedule_id=f"primary-{index:02d}", schedule=schedule,
                           final_time=round(math.fsum(schedule), 12), primary=True)
    for final in protocol["evaluation_horizons"]:
        for steps in protocol.get("endpoint_steps", [1, 2, 4, 8]):
            schedule = [final / steps] * steps
            key = tuple(round(h, 12) for h in schedule)
            result.setdefault(key, dict(schedule_id=f"endpoint-{final:.12g}-{steps}", schedule=schedule,
                                       final_time=final, primary=False))
    return list(result.values())


def _targets(protocol):
    return [(float(t.get("rms", t.get("rms_target"))), float(t.get("max", t.get("max_target"))))
            if isinstance(t, dict) else (float(t), float(t)) for t in protocol["targets"]]


def _config(protocol, family):
    return {**protocol.get("model_config", {}), **protocol.get("model_configs", {}).get(family, {})}


def _record(ctx, identity, metrics, *, good=True, gap=None):
    if identity in getattr(ctx, "identities", set()):
        return
    hypotheses = unit(ctx).get("mechanism_ids", list(ctx.protocol["mechanisms"])[:1])
    ctx.record(identity, hypotheses, metrics=clean(metrics), checks=[
        check("finite-implementation", good, True, "eq", category="math"),
        check("scientific-benefit", gap, True, "eq", category="gap",
              reason="Computational completion is distinct from comparative benefit")])


def _trial(ctx, spec, config, samples, validation, rate, control, phase, *, updates, seconds):
    """Recover optimizer, selected checkpoint and exact deterministic sampler."""
    from .models import make_model
    identity = dict(spec=spec, config=config, rate=rate, control=control, phase=phase,
                    updates=updates, seconds=seconds, binding=binding(ctx))
    stem = digest(identity); root = Path(ctx.path)
    complete = root / "trials" / (stem + ".json")
    if complete.exists():
        row = json.loads(complete.read_text())
        if row["identity"] != identity or file_digest(_safe_file(root, row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("Completed advance trial changed")
        return row
    torch.manual_seed(spec["seed"] or 0)
    model = make_model(spec["family"], spec["track"], config).to(device=ctx.device, dtype=torch.float32)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=0.) if parameters else None
    settings = ctx.protocol["training"]
    progress = root / "progress" / (stem + ".pt")
    initial_path = root / "checkpoints" / (stem + ".initial.pt")
    initial_state = copy.deepcopy(model.state_dict())
    selected_state = initial_state; initial = best = None
    completed = selected_update = examples = 0
    curves = []; prior_seconds = optimizer_seconds = 0.; failure = None; pending_validation = False
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Interrupted trial source/software/protocol differs")
        model.load_state_dict(saved["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(saved["optimizer"])
        selected_state = saved["selected_state"]
        initial, best = saved["initial"], saved["best"]
        completed, selected_update, examples = saved["completed"], saved["selected_update"], saved["examples"]
        curves = saved["curves"]; prior_seconds = saved["elapsed_seconds"]
        optimizer_seconds = saved["optimizer_seconds"]
        pending_validation = saved.get("pending_validation", False)
    elif not initial_path.exists():
        atomic_torch_save(dict(state_dict=initial_state, identity=identity), initial_path)
    start = time.perf_counter()
    elapsed = lambda: prior_seconds + time.perf_counter() - start
    def save_progress():
        atomic_torch_save(dict(identity=identity, state_dict=model.state_dict(),
            optimizer=optimizer.state_dict() if optimizer else None, selected_state=selected_state,
            initial=initial, best=best, completed=completed, selected_update=selected_update,
            examples=examples, curves=curves, elapsed_seconds=elapsed(), optimizer_seconds=optimizer_seconds,
            initial_df_max_difference=initial_parity, pending_validation=pending_validation), progress)
    def curve(update, loss=None, gradient=None, validation_value=None):
        curves.append({**spec, "trial_id": stem, "phase": phase, "update": update,
            "train_loss": loss, "gradient_norm": gradient, "gradient_norm_scope": "pre-clipping norm of scaled loss",
            "validation_loss": validation_value["objective"] if validation_value else None,
            "validation_rms": validation_value["rms"] if validation_value else None,
            "validation_max": validation_value["maximum"] if validation_value else None,
            "examples_seen": examples, "elapsed_seconds": elapsed(), "optimizer_seconds": optimizer_seconds,
            "optimizer_control": control, "learning_rate": rate, "device": str(ctx.device)})
    initial_parity = saved.get("initial_df_max_difference") if progress.exists() else None
    try:
        if initial is None:
            initial = best = validation_metrics(model, validation, ctx)
            curve(0, validation_value=best)
            with torch.no_grad():
                base = make_model("df", spec["track"], _config(ctx.protocol, "df")).to(device=ctx.device, dtype=torch.float32)
                initial_parity = max(float((model(s["u"], s["h"], s["eq"], s["geom"]) -
                    base(s["u"], s["h"], s["eq"], s["geom"])).abs().max()) for s in samples[:3])
        elif pending_validation:
            # An interruption may occur after a completed optimizer update but
            # during its validation. Finish that declared selection opportunity.
            value = validation_metrics(model, validation, ctx)
            curves[-1].update(validation_loss=value["objective"], validation_rms=value["rms"], validation_max=value["maximum"])
            if value["objective"] < best["objective"]:
                best = value; selected_state = copy.deepcopy(model.state_dict()); selected_update = completed
            pending_validation = False
        for update in range(completed + 1, int(updates) + 1 if parameters else 1):
            ctx.budget.check()
            if elapsed() >= seconds:
                break
            rng = random.Random((spec["seed"] or 0) + 104729 * update)
            _sync(ctx.device); tick = time.perf_counter()
            model.train(); optimizer.zero_grad(set_to_none=True); losses = []
            batch = int(settings.get("batch_size", 1))
            for _ in range(batch):
                sample = samples[rng.randrange(len(samples))]
                value = model(sample["u"], sample["h"], sample["eq"], sample["geom"])
                square = (value - sample["target"]).square()
                loss = (square.mean() + float(settings.get("peak_weight", .1)) * square.amax()) / sample["scale"]
                if not bool(torch.isfinite(loss)):
                    raise TrialNumericalFailure("NONFINITE_TRAINING_LOSS")
                (loss * float(control.get("loss_scale", 1.)) / batch).backward()
                losses.append(float(loss.detach())); examples += 1
            clipping = control.get("clip_grad_norm", 1.)
            gradient = torch.nn.utils.clip_grad_norm_(parameters, float(clipping)) if clipping is not None else torch.sqrt(
                sum(p.grad.detach().square().sum() for p in parameters if p.grad is not None))
            if not bool(torch.isfinite(gradient)):
                raise TrialNumericalFailure("NONFINITE_GRADIENT")
            optimizer.step(); _sync(ctx.device)
            optimizer_seconds += time.perf_counter() - tick; completed = update
            value = None
            if update % int(settings.get("validation_every", 25)) == 0 or update == updates or elapsed() >= seconds:
                pending_validation = True
                try:
                    value = validation_metrics(model, validation, ctx)
                except BaseException:
                    curve(update, statistics.mean(losses), float(gradient), None)
                    raise
                pending_validation = False
                if value["objective"] < best["objective"]:
                    best = value; selected_state = copy.deepcopy(model.state_dict()); selected_update = update
            curve(update, statistics.mean(losses), float(gradient), value)
            if value is not None:
                save_progress()
    except (TrialNumericalFailure, FloatingPointError) as error:
        failure = str(error)
    except BaseException:
        save_progress()
        write_json(root / "interrupted-trials" / f"{stem}-{time.time_ns()}.json",
                   dict(identity=identity, completed_updates=completed, total_elapsed_seconds=elapsed(),
                        attempt_seconds=time.perf_counter()-start, status="INTERRUPTED_RESUMABLE"))
        raise
    model.load_state_dict(selected_state); model.eval()
    checkpoint = root / "checkpoints" / (stem + ".selected.pt")
    atomic_torch_save(dict(state_dict=selected_state, **spec, config=config, binding=binding(ctx)), checkpoint)
    elapsed_seconds = elapsed()
    report = model.parameter_report() if hasattr(model, "parameter_report") else {}
    stop = "FROZEN" if not parameters else "NUMERICAL_FAILURE" if failure else "TIME_CAP" if elapsed_seconds >= seconds else "UPDATE_CAP"
    record = {**spec, "identity": identity, "binding": binding(ctx), "trial_id": stem, "phase": phase,
        "effective_config": config, "learning_rate": rate, "optimizer_control": control,
        "failure": failure, "checkpoint_validated": best is not None,
        "initial_validation_objective": initial["objective"] if initial else None,
        "validation_objective": best["objective"] if best else None,
        "selection_status": "FROZEN_CONTROL" if not parameters else "FITTED_CHECKPOINT" if selected_update else "SELECTED_INITIALIZATION",
        "updates_requested": updates if parameters else 0, "updates_completed": completed, "updates_selected": selected_update,
        "examples_seen": examples, "training_seconds": elapsed_seconds, "optimizer_seconds": optimizer_seconds,
        "time_cap_seconds": seconds, "stopping_reason": stop, "time_cap_overshoot_seconds": max(0., elapsed_seconds-seconds),
        "equal_time_status": ("ATTAINED" if stop == "TIME_CAP" else "UPDATE_CAP_CENSORED" if stop == "UPDATE_CAP" else stop) if phase == "equal_time" else "NOT_AN_EQUAL_TIME_TRIAL",
        "initial_df_max_difference": initial_parity, "initial_df_probe_scope": "first three training examples; absent after interrupted initial audit",
        "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in parameters),
        "parameter_report": clean(report), "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": file_digest(checkpoint), "initial_checkpoint": str(initial_path.relative_to(root)),
        "initial_checkpoint_sha256": file_digest(initial_path), "curves": curves}
    write_json(complete, clean(record)); return record


def _load_model(ctx, row, base):
    from .models import make_model
    path = _safe_file(base, row["checkpoint"])
    if file_digest(path) != row["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint digest changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("binding") != binding(ctx) or payload.get("model_id") != row["model_id"]:
        raise ValueError("Selected checkpoint identity changed")
    model = make_model(row["family"], row["track"], row["effective_config"]).to(device=ctx.device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    return model


def _validation_rows(ctx, row, parents):
    from .metrics import endpoint_metrics
    model = _load_model(ctx, row, ctx.path); n = ctx.protocol["train_grid"]; rows = []
    for parent in parents:
        eq, geom = eq_geom(parent, n); initial = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
        for item in shared_schedules(ctx.protocol):
            with torch.no_grad():
                value, costs = measure_paired({row["model_id"]: lambda: rollout(model, initial, item["schedule"], eq, geom, ctx.budget)[0]},
                    device=ctx.device, repeats=2, warmup=1, budget=ctx.budget)
            ref = reference(parent, n, item["final_time"], row["track"])
            metric = endpoint_metrics(value[row["model_id"]], ref["state"], initial, eq, geom, row["track"], reference=ref)
            rows.append({**{k: row[k] for k in ("model_id", "family", "track", "seed", "train_count")},
                "parent_id": parent["parent_id"], "field_cluster": parent["field_cluster"], "grid": n, **item, **metric,
                "grid": n, "cost_seconds": costs["methods"][row["model_id"]]["median_seconds"], "raw_timing": costs,
                "timing_batch_size": 1, "selection_split": "validation"})
    return rows


def train(ctx):
    from .models import make_model, MODEL_SPECS
    bank = load_bank(ctx, unit(ctx).get("prepared_unit", "prepare"))
    training = [p for p in bank if p["split"] == "train"]; validation = [p for p in bank if p["split"] == "validation"]
    if {p["field_cluster"] for p in training} & {p["field_cluster"] for p in validation}:
        raise ValueError("Training/validation fields overlap")
    settings = ctx.protocol["training"]; root = Path(ctx.path); records = []; curves = []; validations = []; audits = []
    specs = model_specs(ctx.protocol, unit(ctx))
    plan = dict(binding=binding(ctx), models=specs, training=settings, validation_batch_size=1,
                shared_schedules=shared_schedules(ctx.protocol))
    planpath = root / "training_plan.json"
    if planpath.exists() and json.loads(planpath.read_text()) != plan:
        raise ValueError("Training recovery declaration changed")
    write_json(planpath, plan)
    for spec in specs:
        ctx.budget.check(); saved_path = root / "completed-models" / (digest(spec) + ".json")
        if saved_path.exists():
            saved = json.loads(saved_path.read_text())
            if saved["binding"] != binding(ctx):
                raise ValueError("Completed model binding changed")
            _load_model(ctx, saved["record"], root)
        else:
            config = _config(ctx.protocol, spec["family"])
            base = make_model("df", spec["track"], _config(ctx.protocol, "df")).to(device=ctx.device, dtype=torch.float32)
            subset = nested_subset(training, spec["train_count"]) if spec["train_count"] else training
            samples = _samples(subset, ctx.protocol["train_horizons"], ctx.protocol["train_grid"], spec["track"], ctx.device,
                               base, float(settings.get("normalization_floor", 1e-6)))
            validation_samples = _samples(validation, ctx.protocol["validation_horizons"], ctx.protocol["train_grid"], spec["track"],
                                          ctx.device, base, float(settings.get("normalization_floor", 1e-6)))
            trainable = MODEL_SPECS[spec["family"]]["trainable"]
            default = dict(id="default", clip_grad_norm=1., loss_scale=1.)
            controls = settings.get("optimizer_controls", [default]) if spec["family"].startswith("fno") else [default]
            trials = []
            for rate, control in itertools.product(settings.get("learning_rates", [.001]) if trainable else [0.], controls if trainable else [default]):
                trials.append(_trial(ctx, spec, config, samples, validation_samples, rate, control, "tuning" if trainable else "frozen",
                    updates=int(settings.get("tuning_updates", 40)), seconds=float(settings.get("tuning_seconds", 15))))
            valid = [t for t in trials if t["checkpoint_validated"] and not t["failure"]]
            winner = min(valid, key=lambda t: t["validation_objective"]) if valid else trials[0]
            if trainable and valid:
                trials.append(_trial(ctx, spec, config, samples, validation_samples, winner["learning_rate"], winner["optimizer_control"],
                    "final", updates=int(settings["updates"]), seconds=float(settings.get("trial_seconds", 90))))
                if settings.get("equal_time_seconds", 0) > 0:
                    trials.append(_trial(ctx, spec, config, samples, validation_samples, winner["learning_rate"], winner["optimizer_control"],
                        "equal_time", updates=int(settings.get("equal_time_max_updates", 4000)), seconds=float(settings["equal_time_seconds"])))
            valid = [t for t in trials if t["checkpoint_validated"] and not t["failure"]]
            selected = min(valid, key=lambda t: t["validation_objective"]) if valid else trials[0]
            audit = None
            if spec["family"].startswith("fno") and trainable:
                # Deliberately single-example training/selection is diagnostic,
                # never part of validation-selected model candidates.
                probe = max(samples, key=lambda s: s["scale"])
                audit = _trial(ctx, spec, config, [probe], [probe], winner["learning_rate"], winner["optimizer_control"], "tiny_overfit",
                    updates=int(settings.get("overfit_updates", 20)), seconds=float(settings.get("overfit_seconds", 5)))
            record = {k: v for k, v in selected.items() if k != "curves"}
            record["trials"] = [{k: v for k, v in t.items() if k != "curves"} for t in trials]
            record["total_training_seconds"] = sum(t["training_seconds"] for t in trials) + (audit["training_seconds"] if audit else 0.)
            record["training_parent_ids"] = [p["parent_id"] for p in subset]
            record["training_samples"] = len(samples); record["independent_training_fields"] = len(subset)
            saved = dict(binding=binding(ctx), record=record, curves=[c for t in trials + ([audit] if audit else []) for c in t["curves"]],
                         validation_rows=_validation_rows(ctx, record, validation), fairness_audit={k: v for k, v in audit.items() if k != "curves"} if audit else None)
            write_json(saved_path, clean(saved))
        records.append(saved["record"]); curves.extend(saved["curves"]); validations.extend(saved["validation_rows"])
        if saved["fairness_audit"]: audits.append(saved["fairness_audit"])
        _record(ctx, "training/" + spec["model_id"], saved["record"], good=saved["record"]["checkpoint_validated"])
    write_json(root / "catalog.json", dict(schema="tdn.advance-catalog/v1", **binding(ctx), stage=ctx.stage, records=records,
        selection_split="validation", training_plan_sha256=file_digest(planpath)))
    write_json(root / "validation_rows.json", dict(rows=validations))
    write_json(root / "fairness_diagnostics.json", dict(rows=audits, selection_scope="single-example diagnostic, never eligible for checkpoint selection"))
    with (root / "learning_curves.jsonl").open("w") as out:
        for row in curves: out.write(json.dumps(clean(row), allow_nan=False) + "\n")
    return dict(models=len(records), trials=sum(len(r["trials"]) for r in records),
                training_seconds=sum(r["total_training_seconds"] for r in records),
                initialization_selections=sum(r["selection_status"] == "SELECTED_INITIALIZATION" for r in records))


def validation_selection(rows, targets):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["track"], row["final_time"])].append(row)
    result = []
    for (model_id, track, final), members in sorted(grouped.items()):
        fields = {r["field_cluster"] for r in members}
        for rms, maximum in targets:
            feasible = []
            for schedule in sorted({r["schedule_id"] for r in members}):
                cells = [r for r in members if r["schedule_id"] == schedule]
                if len(cells) != len(fields): raise ValueError("Duplicate or missing validation field")
                if all(endpoint_eligibility(r, rms, maximum) == "ELIGIBLE" for r in cells):
                    feasible.append((statistics.mean(r["cost_seconds"] for r in cells), schedule))
            winner = min(feasible) if feasible else None
            result.append(dict(model_id=model_id, track=track, final_time=final, rms_target=rms, max_target=maximum,
                schedule_id=winner[1] if winner else None, validation_cost_seconds=winner[0] if winner else None,
                status="VALIDATION_SELECTED" if winner else "NO_FEASIBLE_VALIDATION_SCHEDULE",
                independent_validation_fields=len(fields), selection_split="validation", timing_batch_size=1))
    return result


def freeze(ctx):
    root = Path(ctx.path); records = []; validations = []; sources = {}
    for stage, scope in ctx.protocol["units"].items():
        if scope["kind"] != "train": continue
        base = Path(ctx.prerequisites[stage]); catalog = json.loads(_safe_file(base, "catalog.json").read_text())
        if any(catalog.get(k) != v for k, v in binding(ctx).items()) or catalog.get("selection_split") != "validation":
            raise ValueError("Training selection identity changed")
        expected = {r["model_id"] for r in model_specs(ctx.protocol, scope)}
        ids = [r["model_id"] for r in catalog["records"]]
        if set(ids) != expected or len(ids) != len(expected): raise ValueError("Training catalog coverage changed")
        if file_digest(_safe_file(base, "training_plan.json")) != catalog["training_plan_sha256"]:
            raise ValueError("Training plan changed")
        sources[stage] = dict(catalog_sha256=file_digest(base / "catalog.json"))
        for row in catalog["records"]:
            source = _safe_file(base, row["checkpoint"])
            if file_digest(source) != row["checkpoint_sha256"]: raise ValueError("Selected checkpoint changed")
            destination = root / "checkpoints" / (digest(row["model_id"]) + ".pt"); destination.parent.mkdir(exist_ok=True)
            if destination.exists() and file_digest(destination) != row["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint differs")
            if not destination.exists(): shutil.copyfile(source, destination)
            records.append({**row, "checkpoint": str(destination.relative_to(root)), "training_unit": stage})
        validations.extend(json.loads(_safe_file(base, "validation_rows.json").read_text())["rows"])
    expected = {r["model_id"] for r in model_specs(ctx.protocol)}
    ids = [r["model_id"] for r in records]
    if set(ids) != expected or len(ids) != len(expected): raise ValueError("Frozen model roster differs")
    write_json(root / "catalog.json", dict(schema="tdn.advance-catalog/v1", **binding(ctx), records=records,
        selection_split="validation", frozen_before_confirmation=True, sources=sources))
    frontiers = validation_selection(validations, _targets(ctx.protocol))
    write_json(root / "validation_frontier.json", dict(rows=frontiers, raw_validation_rows=validations,
        selection_split="validation", timing_batch_size=1))
    frozen = dict(**binding(ctx), catalog_sha256=file_digest(root / "catalog.json"),
        validation_frontier_sha256=file_digest(root / "validation_frontier.json"),
        checkpoint_hashes={r["model_id"]: r["checkpoint_sha256"] for r in records})
    write_json(root / "freeze.json", frozen)
    _record(ctx, "freeze/selection", dict(frozen_models=len(records), validation_frontier_cells=len(frontiers),
        fresh_confirmation_accessed=False, selection_split="validation"))
    return dict(frozen_models=len(records), validation_frontier_cells=len(frontiers), fresh_confirmation_accessed=False)


def verify_frozen(ctx):
    root = Path(ctx.path if ctx.stage == "freeze" else ctx.prerequisites["freeze"])
    frozen = json.loads(_safe_file(root, "freeze.json").read_text())
    catalog = json.loads(_safe_file(root, "catalog.json").read_text())
    if any(frozen.get(k) != v or catalog.get(k) != v for k, v in binding(ctx).items()):
        raise ValueError("Frozen source, protocol or software changed")
    if frozen["catalog_sha256"] != file_digest(root / "catalog.json") or frozen["validation_frontier_sha256"] != file_digest(_safe_file(root, "validation_frontier.json")):
        raise ValueError("Frozen selection bytes changed")
    expected = {r["model_id"] for r in model_specs(ctx.protocol)}; ids = [r["model_id"] for r in catalog["records"]]
    if len(ids) != len(expected) or set(ids) != expected or set(frozen["checkpoint_hashes"]) != expected:
        raise ValueError("Frozen model roster changed")
    if catalog.get("selection_split") != "validation" or catalog.get("frozen_before_confirmation") is not True:
        raise ValueError("Selection was not frozen before confirmation")
    for row in catalog["records"]:
        if frozen["checkpoint_hashes"][row["model_id"]] != row["checkpoint_sha256"] or file_digest(_safe_file(root, row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint bytes changed")
    return catalog, frozen


def evaluation_plan(protocol, catalog, frontiers, parents):
    """Only primary schedules plus validation-locked schedules are evaluated."""
    schedules = shared_schedules(protocol); result = {}
    locked = defaultdict(set)
    for row in frontiers:
        if row["schedule_id"] is not None:
            locked[row["model_id"]].add(row["schedule_id"])
    for parent in parents:
        for n, track, schedule in itertools.product(protocol["grids"], protocol["tracks"], schedules):
            models = [r["model_id"] for r in catalog if r["track"] == track and
                      (schedule["primary"] or schedule["schedule_id"] in locked[r["model_id"]])]
            if not models: continue
            key = f"{parent['parent_id']}/N{n}/{track}/{schedule['schedule_id']}"
            result[key] = dict(parent=parent, grid=n, track=track, schedule=schedule, model_ids=models)
    return result


def _frontiers(ctx):
    root = Path(ctx.path if ctx.stage == "freeze" else ctx.prerequisites["freeze"])
    return json.loads(_safe_file(root, "validation_frontier.json").read_text())["rows"]


def _validate_completed_group(payload, identity, expected):
    if payload.get("identity") != identity or payload.get("expected") != expected:
        raise ValueError("Completed evaluation belongs to different frozen inputs")
    if payload.get("payload_sha256") != digest({k: v for k, v in payload.items() if k != "payload_sha256"}):
        raise ValueError("Completed evaluation payload changed")
    ids = [r["model_id"] for r in payload["rows"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected["model_ids"]):
        raise ValueError("Completed evaluation model coverage changed")
    if set(payload["timing"]["methods"]) != set(ids):
        raise ValueError("Completed timing model coverage changed")
    for round_ in payload["timing"]["rounds"]:
        if len(round_["order"]) != len(ids) or set(round_["order"]) != set(ids) or set(round_["samples_seconds"]) != set(ids):
            raise ValueError("Completed paired timing round coverage changed")
        if any(not math.isfinite(v) or v <= 0 for v in round_["samples_seconds"].values()):
            raise ValueError("Invalid paired timing observation")


def confirm(ctx):
    from .metrics import endpoint_metrics, measure_paired_metrics
    catalog, frozen = verify_frozen(ctx); rows = catalog["records"]
    prepared = unit(ctx).get("prepared_unit")
    if prepared is None:
        prepared = next(k for k in ctx.prerequisites if unit(ctx, k)["kind"] == "confirm_prepare")
    bank = load_bank(ctx, prepared); expected_parents = declarations(ctx)
    if {p["parent_id"] for p in bank} != {p["parent_id"] for p in expected_parents}:
        raise ValueError("Confirmation parent coverage changed")
    plan = evaluation_plan(ctx.protocol, rows, _frontiers(ctx), expected_parents)
    models = {r["model_id"]: _load_model(ctx, r, ctx.prerequisites["freeze"]) for r in rows}
    metadata = {r["model_id"]: r for r in rows}; parents = {p["parent_id"]: p for p in bank}
    root = Path(ctx.path); identity = dict(**binding(ctx), freeze_sha256=digest(frozen))
    endpoint_rows = []; timing_rows = []; hashes = {}
    for group_id, scope in plan.items():
        ctx.budget.check(); path = root / "completed-groups" / (digest(group_id) + ".json")
        if path.exists():
            payload = json.loads(path.read_text()); _validate_completed_group(payload, identity, scope)
        else:
            parent = parents[scope["parent"]["parent_id"]]; n = scope["grid"]; item = scope["schedule"]
            eq, geom = eq_geom(parent, n); initial = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
            ref = reference(parent, n, item["final_time"], scope["track"])
            calls = {key: (lambda model=models[key]: rollout(model, initial, item["schedule"], eq, geom, ctx.budget)[0]) for key in scope["model_ids"]}
            with torch.no_grad():
                values, timing = measure_paired_metrics(calls, device=ctx.device, repeats=int(ctx.protocol.get("timing_repeats", 5)),
                    warmup=int(ctx.protocol.get("timing_warmup", 1)), seed=int(ctx.protocol.get("timing_seed", 0)) + int(digest(group_id)[:8], 16), budget=ctx.budget)
            measured = []
            for key in scope["model_ids"]:
                row = metadata[key]
                metrics = endpoint_metrics(values[key], ref["state"], initial, eq, geom, scope["track"], reference=ref)
                measured.append({**{k: row[k] for k in ("model_id", "family", "track", "seed", "train_count", "parameters", "selection_status")},
                    "parent_id": parent["parent_id"], "field_cluster": parent["field_cluster"], "regime": parent["regime"],
                    "distribution": parent.get("distribution"), "grid": n, **item, **metrics,
                    "grid": n, "status": "MEASURED" if metrics["finite"] else "NUMERICAL_FAILURE",
                    "cost_seconds": timing["methods"][key]["median_seconds"], "raw_timing": timing["methods"][key],
                    "group_id": group_id, "checkpoint_sha256": row["checkpoint_sha256"],
                    "timing_batch_size": 1, "frozen_before_confirmation": True})
            payload = dict(identity=identity, expected=scope, rows=measured, timing=timing)
            payload["payload_sha256"] = digest(clean(payload)); write_json(path, clean(payload))
        hashes[group_id] = file_digest(path); endpoint_rows.extend(payload["rows"])
        timing_rows.append(dict(group_id=group_id, **payload["timing"]))
        _record(ctx, "confirmation/" + group_id, dict(models=len(payload["rows"]), parent_id=scope["parent"]["parent_id"],
            finite_models=sum(bool(r["finite"]) for r in payload["rows"]), reference_accepted=all(r["reference_accepted"] for r in payload["rows"])),
            good=all(r["finite"] for r in payload["rows"]))
    write_json(root / "endpoint_rows.json", dict(rows=endpoint_rows))
    write_json(root / "timing_rounds.json", dict(rows=timing_rows))
    write_json(root / "confirmation_manifest.json", dict(**identity, parent_ids=list(parents), groups=hashes,
        endpoint_sha256=file_digest(root / "endpoint_rows.json"), timing_sha256=file_digest(root / "timing_rounds.json"),
        row_count=len(endpoint_rows), scope="two primary schedules plus frozen validation schedules; no test-truth selection"))
    return dict(parents=len(parents), groups=len(hashes), endpoint_rows=len(endpoint_rows),
                numerical_failures=sum(not r["finite"] for r in endpoint_rows))


def locked_rows(rows, frontiers, protocol, parent_ids):
    lookup = {(r["model_id"], r["parent_id"], r["grid"], r["schedule_id"]): r for r in rows}
    specs = {r["model_id"]: r for r in model_specs(protocol)}
    result = []
    for selected in frontiers:
        for parent_id, n in itertools.product(parent_ids, protocol["grids"]):
            row = lookup.get((selected["model_id"], parent_id, n, selected["schedule_id"]))
            if row is None and selected["schedule_id"] is not None:
                raise ValueError("Missing frozen-schedule confirmation cell")
            parent = next(p for p in protocol["parents"] if p["parent_id"] == parent_id)
            result.append({**specs[selected["model_id"]], **(row or {}), **selected, "parent_id": parent_id, "field_cluster": parent["field_cluster"], "grid": n,
                "status": endpoint_eligibility(row, selected["rms_target"], selected["max_target"]) if row else "NO_FEASIBLE_VALIDATION_SCHEDULE",
                "cost_seconds": row["cost_seconds"] if row else None,
                "family": row["family"] if row else selected["model_id"].split("/")[1],
                "seed": specs[selected["model_id"]]["seed"],
                "selection_scope": "unchanged validation choice; absent choices retain denominator"})
    return result


def _claims(rows, locked, protocol):
    """Predeclared paired endpoint RMS and complete-field cost conditions."""
    primary = protocol.get("selection", {}).get("primary_family", "two_basis")
    controls = protocol.get("confirmation", {}).get("primary_controls", protocol.get("selection", {}).get("comparators",
        ["quad2_full", "analytic_quad_cubic", "df", "etdrk4", "fno_zero", "fno_scaled"]))
    lookup = {(r["family"], r["track"], r["train_count"], r["seed"], r["parent_id"], r["grid"], r["schedule_id"]): r for r in rows}
    thresholds = protocol.get("hypothesis_thresholds", {}); result = []
    primary_count = protocol.get("selection", {}).get("primary_train_count", max(protocol["training"]["subset_sizes"]))
    scopes = sorted({(r["track"], r["train_count"], r["final_time"]) for r in rows if r["family"] == primary and r["train_count"] == primary_count})
    for track, count, final in scopes:
        candidates = [r for r in rows if r["family"] == primary and r["track"] == track and r["train_count"] == count and r["final_time"] == final and r["primary"]]
        for control in controls:
            observed = []; failures = unknown = left_pass = right_pass = 0
            for left in candidates:
                suffix = (left["parent_id"], left["grid"], left["schedule_id"])
                right = lookup.get((control, track, count, left["seed"], *suffix)) or lookup.get((control, track, 0, None, *suffix))
                target = float(protocol.get("primary_target", 2e-5))
                left_pass += endpoint_eligibility(left, target, target) == "ELIGIBLE"
                right_pass += bool(right and endpoint_eligibility(right, target, target) == "ELIGIBLE")
                if not right or not left["reference_accepted"] or not right["reference_accepted"]:
                    unknown += 1; continue
                if not left["finite"] or not right["finite"]:
                    failures += 1; continue
                if any(r["error_rms"] <= float(r.get("reference_uncertainty_rms") or 0) for r in (left, right)):
                    unknown += 1; continue
                observed.append(dict(field_cluster=left["field_cluster"], seed=left["seed"], difference=math.log(right["error_rms"] / left["error_rms"])))
            interval = field_cluster_interval(observed, repeats=int(protocol.get("bootstrap_replicates", 1000)),
                                               minimum_fields=int(thresholds.get("minimum_fields", 5)))
            ratio = math.exp(interval["mean"]) if interval.get("mean") is not None else None
            lower = math.exp(interval["lower"]) if interval.get("lower") is not None else None
            coverage_change = (left_pass-right_pass)/len(candidates) if candidates else None
            verdict = "BAD" if failures else "NA" if unknown or lower is None else "GOOD" if ratio >= thresholds.get("learned_error_ratio", 1.1) and lower > 1 and coverage_change >= -thresholds.get("maximum_coverage_regression", .05) else "BAD"
            result.append(dict(claim="primary_same_schedule_accuracy", candidate_family=primary, control_family=control, track=track,
                train_count=count, final_time=final, ratio=ratio, interval=interval, verdict=verdict, denominator=len(candidates), numerical_failures=failures,
                unresolved_pairs=unknown, independent_fields=len({r["field_cluster"] for r in candidates}),
                candidate_passes=left_pass, control_passes=right_pass, coverage_difference=coverage_change,
                scope="predeclared primary schedules; field/seed clustered, unadjusted descriptive intervals"))
    # Complete cost claims retain failures and missing validation choices; no
    # interval computed only on successful cases can erase poor coverage.
    cost = []; target = float(protocol.get("primary_target", 2e-5))
    for track, count, final in scopes:
        candidates = [r for r in locked if r["family"] == primary and r["track"] == track and r["train_count"] == count and r["final_time"] == final and r["rms_target"] == target and r["max_target"] == target]
        for control in controls:
            rhs = [r for r in locked if r["family"] == control and r["track"] == track and r["train_count"] in (0, count) and r["final_time"] == final and r["rms_target"] == target and r["max_target"] == target]
            lookup_cost = {(r["seed"], r["parent_id"], r["grid"]): r for r in rhs}
            observed = []; fields = defaultdict(list); control_fields = defaultdict(list); unresolved = 0
            for left in candidates:
                right = lookup_cost.get((left["seed"], left["parent_id"], left["grid"])) or lookup_cost.get((None, left["parent_id"], left["grid"]))
                unknown_states = {"REFERENCE_UNACCEPTED", "MISSING_METRIC", "UNAVAILABLE_CHECKPOINT"}
                unresolved += left["status"] in unknown_states or bool(right and right["status"] in unknown_states)
                eligible = left["status"] == "ELIGIBLE"; control_eligible = bool(right and right["status"] == "ELIGIBLE")
                fields[left["field_cluster"]].append(eligible); control_fields[left["field_cluster"]].append(control_eligible)
                if eligible and control_eligible:
                    observed.append(dict(field_cluster=left["field_cluster"], seed=left["seed"], difference=math.log(right["cost_seconds"] / left["cost_seconds"])))
            coverage = statistics.mean(all(v) for v in fields.values()) if fields else 0.
            control_coverage = statistics.mean(all(v) for v in control_fields.values()) if fields else 0.
            interval = field_cluster_interval(observed, repeats=int(protocol.get("bootstrap_replicates", 1000)), minimum_fields=int(thresholds.get("minimum_fields", 5)))
            ratio = math.exp(interval["mean"]) if interval.get("mean") is not None else None
            lower = math.exp(interval["lower"]) if interval.get("lower") is not None else None
            verdict = "NA" if unresolved else "BAD" if coverage < thresholds.get("required_feasible_fraction", .95) or coverage < control_coverage - thresholds.get("maximum_coverage_regression", .05) else "NA" if lower is None else "GOOD" if ratio >= thresholds.get("practical_speedup", 1.2) and lower > 1 else "BAD"
            cost.append(dict(claim="validation_locked_solver_cost", candidate_family=primary, control_family=control, track=track,
                train_count=count, final_time=final, ratio=ratio, interval=interval, verdict=verdict, denominator=len(candidates), eligible_paired_rows=len(observed),
                candidate_joint_field_coverage=coverage, control_joint_field_coverage=control_coverage, independent_fields=len(fields),
                unresolved_pairs=unresolved,
                monetary_cost=None, energy=None, scope="batch-one synchronized warm complete-call cost; failed coverage retained; no policy deployed"))
    return dict(accuracy=result, cost=cost, published_fno_superiority=False, job_completion_is_not_scientific_success=True)


def aggregate(ctx):
    catalog, frozen = verify_frozen(ctx); rows = []; timing = []; parents = []; source_hashes = {}
    identity = dict(**binding(ctx), freeze_sha256=digest(frozen))
    for stage, scope in ctx.protocol["units"].items():
        if scope["kind"] != "confirm": continue
        base = Path(ctx.prerequisites[stage]); manifest = json.loads(_safe_file(base, "confirmation_manifest.json").read_text())
        if any(manifest.get(k) != v for k, v in identity.items()): raise ValueError("Confirmation identity changed")
        if set(manifest["parent_ids"]) != {p["parent_id"] for p in declarations(ctx, stage)}:
            raise ValueError("Confirmation parent part differs")
        for name, key in (("endpoint_rows.json", "endpoint_sha256"), ("timing_rounds.json", "timing_sha256")):
            if file_digest(_safe_file(base, name)) != manifest[key]: raise ValueError("Confirmation aggregate input changed")
        part = json.loads((base / "endpoint_rows.json").read_text())["rows"]
        plan = evaluation_plan(ctx.protocol, catalog["records"], _frontiers(ctx), declarations(ctx, stage))
        expected = {(group, m) for group, s in plan.items() for m in s["model_ids"]}
        actual = [(r["group_id"], r["model_id"]) for r in part]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Confirmation cell coverage incomplete or duplicate")
        rows.extend(part); timing.extend(json.loads((base / "timing_rounds.json").read_text())["rows"])
        parents.extend(manifest["parent_ids"]); source_hashes[stage] = file_digest(base / "confirmation_manifest.json")
    expected = {p["parent_id"] for p in ctx.protocol["parents"] if p["split"] == "confirmation"}
    if len(parents) != len(expected) or set(parents) != expected: raise ValueError("Full independent confirmation cohort missing or repeated")
    locked = locked_rows(rows, _frontiers(ctx), ctx.protocol, parents)
    claims = _claims(rows, locked, ctx.protocol); root = Path(ctx.path)
    write_json(root / "endpoint_rows.json", dict(rows=rows))
    write_json(root / "timing_rounds.json", dict(rows=timing))
    write_json(root / "locked_frontiers.json", dict(rows=locked))
    write_json(root / "claims.json", claims)
    write_json(root / "aggregate_manifest.json", dict(**identity, sources=source_hashes, parents=parents, endpoint_rows=len(rows)))
    for i, row in enumerate(claims["accuracy"] + claims["cost"]):
        _record(ctx, f"claim/{i}", row, good=True, gap=True if row["verdict"] == "GOOD" else False if row["verdict"] == "BAD" else None)
    return dict(parents=len(parents), endpoint_rows=len(rows), numerical_failures=sum(not r["finite"] for r in rows),
                primary_accuracy=claims["accuracy"], primary_cost=claims["cost"])


def scaling_groups(bank, count):
    """All batch-one fields; first distinct homogeneous batch for throughput."""
    batches = defaultdict(list)
    for parent in bank:
        batches[(parent["kappa"], parent["reaction_rate"], tuple(parent["lengths"]))].append(parent)
    for index, parents in enumerate(batches.values()):
        if count == 1:
            for parent in parents:
                yield index, parents, [parent]
        else:
            yield index, parents, parents[:count] if len(parents) >= count else []


def scaling(ctx):
    from .metrics import endpoint_metrics, measure_paired_metrics
    catalog, frozen = verify_frozen(ctx); options = ctx.protocol["scaling"]
    prepared = unit(ctx).get("prepared_unit", "scaling_prepare")
    bank = load_bank(ctx, prepared); root = Path(ctx.path)
    families = options.get("families", ctx.protocol["models"])
    seed = ctx.protocol["seeds"][0]; size = max(ctx.protocol["training"]["subset_sizes"])
    selected = [r for r in catalog["records"] if r["family"] in families and (r["seed"] is None or (r["seed"] == seed and r["train_count"] == size))]
    models = {r["model_id"]: _load_model(ctx, r, ctx.prerequisites["freeze"]) for r in selected}
    rows = []; timing_rows = []; group_hashes = {}
    used_by_batch = defaultdict(set); planned_cells = 0
    identity = dict(**binding(ctx), freeze_sha256=digest(frozen))
    for n, track, count in itertools.product(options["grids"], ctx.protocol["tracks"], options.get("batches", [1, 4])):
        for physics_index, parents, group in scaling_groups(bank, count):
            planned_cells += 1
            # Distinct fields; never repeat one field to fill a throughput batch.
            if not group:
                rows.append(dict(grid=n, track=track, batch_size=count, physics_group=physics_index,
                    available_parent_ids=[p["parent_id"] for p in parents], available_distinct_fields=len(parents),
                    status="INSUFFICIENT_DISTINCT_FIELDS", cost_seconds=None, accuracy_qualified=None)); continue
            used_by_batch[count].update(p["parent_id"] for p in group)
            eq, geom = eq_geom(group[0], n)
            initial = torch.cat([p["states"][str(n)] for p in group]).to(device=ctx.device, dtype=torch.float32)
            horizon = float(options["horizon"]); schedule = [horizon / int(options.get("steps", 2))] * int(options.get("steps", 2))
            metadata = {r["model_id"]: r for r in selected if r["track"] == track}
            scope = dict(grid=n, track=track, batch_size=count, physics_group=physics_index,
                parent_ids=[p["parent_id"] for p in group], model_ids=list(metadata), schedule=schedule)
            key = digest(scope); path = root / "completed-groups" / (key + ".json")
            if path.exists():
                payload = json.loads(path.read_text()); _validate_completed_group(payload, identity, scope)
            else:
                calls = {m: (lambda model=models[m]: rollout(model, initial, schedule, eq, geom, ctx.budget)[0]) for m in metadata}
                with torch.no_grad():
                    values, timing = measure_paired_metrics(calls, device=ctx.device, repeats=int(options.get("repeats", 5)), warmup=1,
                        seed=int(ctx.protocol.get("timing_seed", 0)) + int(key[:8], 16), budget=ctx.budget)
                measurements = []
                for model_id, spec in metadata.items():
                    per_field = [endpoint_metrics(values[model_id][j:j+1], reference(p, n, horizon, track)["state"],
                        initial[j:j+1], eq, geom, track, reference=reference(p, n, horizon, track)) for j, p in enumerate(group)]
                    cost = timing["methods"][model_id]
                    target = float(ctx.protocol["primary_target"])
                    measurements.append({**scope, **{k: spec[k] for k in ("model_id", "family", "seed", "train_count", "parameters")},
                        "per_field": per_field, "final_time": horizon, "status": "MEASURED",
                        "cost_seconds": cost["median_seconds"], "amortized_seconds_per_field": cost["median_seconds"] / count,
                        "fields_per_second": count / cost["median_seconds"], "raw_timing": cost,
                        "accuracy_qualified": all(endpoint_eligibility({**r, "cost_seconds": cost["median_seconds"]}, target, target) == "ELIGIBLE" for r in per_field),
                        "teacher_generated_on_gpu": False, "reference_generation_cost_included_in_inference": False,
                        "selection_scope": "first declared training seed, largest training subset, fixed schedule; no scaling-truth selection"})
                payload = dict(identity=identity, expected=scope, rows=measurements, timing=timing)
                payload["payload_sha256"] = digest(clean(payload)); write_json(path, clean(payload))
            rows.extend(payload["rows"]); timing_rows.append(dict(group_id=key, **payload["timing"])); group_hashes[key] = file_digest(path)
            _record(ctx, "scaling/" + key, dict(grid=n, track=track, batch_size=count, methods=len(payload["rows"]),
                accuracy_qualified=sum(r["accuracy_qualified"] for r in payload["rows"])))
    write_json(root / "scaling_rows.json", dict(rows=rows))
    write_json(root / "timing_rounds.json", dict(rows=timing_rows))
    all_ids = {p["parent_id"] for p in bank}
    coverage = {str(count): dict(used_parent_ids=sorted(used_by_batch[count]), unused_parent_ids=sorted(all_ids-used_by_batch[count]),
        scope="all declared fields" if count == 1 else "first declared homogeneous distinct-field batch per physics group")
        for count in options.get("batches", [1, 4])}
    write_json(root / "scaling_manifest.json", dict(**identity, prepared_unit=prepared, groups=group_hashes,
        planned_workload_cells=planned_cells, measured_workload_cells=len(group_hashes), parent_coverage_by_batch=coverage))
    return dict(measured_rows=sum(r["status"] == "MEASURED" for r in rows), missing_cells=sum(r["status"] != "MEASURED" for r in rows),
                accuracy_qualified_rows=sum(bool(r.get("accuracy_qualified")) for r in rows))
