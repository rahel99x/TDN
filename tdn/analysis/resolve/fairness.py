"""Independent, bounded optimizer opportunities for the adapted FNO controls.

This is a development trainability experiment, not a published FNO reproduction
or a fresh confirmation test. Every declared optimizer gets the long opportunity;
an initialization tie cannot prune alternatives. Tiny-set fits are diagnostics
and are never eligible for model selection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import torch

from tdn.analysis.advance.data import binding, load_bank, unit
from tdn.analysis.advance.learning import _config, _load_model, _trial
from tdn.analysis.advance.metrics import endpoint_metrics
from tdn.analysis.advance.models import make_model, MODEL_SPECS
from tdn.analysis.frontier.core import check, clean
from tdn.analysis.frontier.neural import _samples, nested_subset, reference
from tdn.research.protocol import file_digest
from tdn.runtime.metadata import write_json


FAMILIES = ("fno_scaled", "fno_mean", "two_basis")
CONTROLS = ("df", "quad2_full", "quad4_full")
SELECTION_PHASES = ("long_update", "equal_time")


def optimizer_plan(settings):
    """Validate the predeclared opportunities; IDs cannot hide duplicates."""
    rows = settings["optimizer_settings"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Fairness needs distinct optimizer setting IDs")
    result = []
    for row in rows:
        rate = float(row["learning_rate"])
        scale = float(row.get("loss_scale", 1.))
        clip = row.get("clip_grad_norm")
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Fairness optimizer rate and loss scale must be positive finite")
        if clip is not None and (not math.isfinite(float(clip)) or float(clip) <= 0):
            raise ValueError("Gradient clipping must be positive finite or null")
        result.append(dict(id=str(row["id"]), learning_rate=rate,
                           loss_scale=scale, clip_grad_norm=clip))
    signatures = {(r["learning_rate"], r["loss_scale"], r["clip_grad_norm"]) for r in result}
    if len(signatures) != len(result):
        raise ValueError("Duplicate optimizer opportunities are not independent controls")
    return result


def select_validation_trial(trials):
    """Never inspect tiny-fit/train errors when choosing a deployable checkpoint."""
    eligible = [r for r in trials if r["phase"] in SELECTION_PHASES
                and r.get("checkpoint_validated") and not r.get("failure")
                and r.get("validation_objective") is not None
                and math.isfinite(float(r["validation_objective"]))]
    if not eligible:
        return None
    # Tie breaking is deterministic only AFTER every declared trial ran.
    return min(eligible, key=lambda r: (r["validation_objective"], r["trial_id"]))


def run_family_trials(ctx, spec, config, samples, validation, settings, *, trial_fn=None):
    """Equal opportunities, with separately labelled update/time censoring."""
    trial_fn = trial_fn or _trial
    trials, tiny = [], []
    probe = max(samples, key=lambda sample: sample["scale"])
    for setting in optimizer_plan(settings):
        control = {k: setting[k] for k in ("id", "clip_grad_norm", "loss_scale")}
        rate = setting["learning_rate"]
        trials.append(trial_fn(ctx, spec, config, samples, validation, rate, control,
            "long_update", updates=int(settings["updates"]), seconds=float(settings["trial_seconds"])))
        if float(settings["equal_time_seconds"]) > 0:
            trials.append(trial_fn(ctx, spec, config, samples, validation, rate, control,
                "equal_time", updates=int(settings["equal_time_max_updates"]),
                seconds=float(settings["equal_time_seconds"])))
        tiny.append(trial_fn(ctx, spec, config, [probe], [probe], rate, control,
            "tiny_memorization", updates=int(settings["tiny_updates"]), seconds=float(settings["tiny_seconds"])))
    return dict(trials=trials, tiny=tiny, selected=select_validation_trial(trials),
                tiny_parent_id=probe["parent_id"], tiny_horizon=probe["h"])


def tiny_gate(row, settings):
    """An explicit low residual objective, independent of validation success."""
    initial, selected = row.get("initial_validation_objective"), row.get("validation_objective")
    ratio = selected / initial if initial is not None and initial > 1e-12 and selected is not None else None
    relative_target = float(settings["tiny_relative_objective_target"])
    absolute_target = float(settings["tiny_absolute_normalized_target"])
    reason = "Tiny training example only; never validation or confirmation evidence"
    if row.get("failure"):
        passed, verdict = False, "BAD"
        reason += "; numerical failure"
    elif ratio is None:
        passed, verdict = None, "NA"
        reason += "; no resolved initial residual"
    elif ratio <= relative_target and selected <= absolute_target:
        passed, verdict = True, "GOOD"
    elif row.get("stopping_reason") == "TIME_CAP":
        passed, verdict = None, "NA"
        reason += "; walltime censored before target, not evidence of incapacity"
    else:
        passed, verdict = False, "BAD"
        reason += "; target unmet within the declared update opportunity"
    return dict(passed=passed, verdict=verdict, relative_objective=ratio,
                relative_objective_target=relative_target,
                absolute_normalized_objective=selected, absolute_normalized_target=absolute_target,
                selected_update=row.get("updates_selected"), reason=reason,
                selection_eligible=False, proves_capacity_on_general_fields=False)


def trial_diagnostics(row):
    curves = row.get("curves", [])
    gradients = [float(r["gradient_norm"]) for r in curves if r.get("gradient_norm") is not None]
    threshold = row["optimizer_control"].get("clip_grad_norm")
    clipped = sum(g > threshold for g in gradients) if threshold is not None else 0
    return dict(gradient_observations=len(gradients), gradients_finite=all(math.isfinite(g) for g in gradients),
                gradient_norm_max=max(gradients, default=None), gradient_clip_threshold=threshold,
                gradient_clipped_fraction=clipped/len(gradients) if gradients else None,
                gradient_scope="Pre-clipping norm of the scaled objective; clipping is diagnostic, not an identified failure cause",
                equal_data=True, equal_data_scope="Identical examples across settings/families within this phase; tiny phase uses one train example",
                equal_update_attained=row["updates_completed"] == row["updates_requested"],
                equal_update_censored=row["stopping_reason"] == "TIME_CAP",
                equal_time_status=row.get("equal_time_status", "NOT_AN_EQUAL_TIME_TRIAL"),
                selected_initialization=row["selection_status"] == "SELECTED_INITIALIZATION")


def parameter_movement(root, row):
    states = []
    for key in ("initial_checkpoint", "checkpoint"):
        path = Path(root) / row[key]
        if file_digest(path) != row[key + "_sha256"]:
            raise ValueError("Fairness checkpoint bytes changed")
        states.append(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    initial, selected = states
    groups = {}
    for name in initial:
        if name not in selected or initial[name].shape != selected[name].shape:
            raise ValueError("Fairness state dictionary structure changed")
        group = "mean_head" if name.startswith("mean_head") else "spatial_head" if name.startswith("head") else "trunk_or_basis"
        before, after = initial[name], selected[name]
        difference = (after - before).abs().double()
        group_row = groups.setdefault(group, dict(delta_squared=0., initial_squared=0., values=0))
        group_row["delta_squared"] += float(difference.square().sum())
        group_row["initial_squared"] += float(before.abs().double().square().sum())
        group_row["values"] += before.numel()
    return {name: dict(delta_l2=math.sqrt(row["delta_squared"]),
                       initial_l2=math.sqrt(row["initial_squared"]), values=row["values"])
            for name, row in groups.items()}


@torch.no_grad()
def sample_metrics(model, samples, track, split, ctx):
    """Raw physical, mean, maximum and spectral errors outside optimizer time."""
    rows = []
    model.eval()
    for sample in samples:
        ctx.budget.check()
        prediction = model(sample["u"], sample["h"], sample["eq"], sample["geom"])
        metric = endpoint_metrics(prediction, sample.get("evaluation_target", sample["target"]), sample["u"],
                                  sample["eq"], sample["geom"], track, reference=sample.get("reference_metadata"))
        error = prediction.double() - sample["target"].double()
        objective = float(error.square().mean() + float(ctx.protocol["training"].get("peak_weight", .1)) * error.square().amax()) / sample["scale"]
        rows.append(dict(parent_id=sample["parent_id"], field_cluster=sample["field_cluster"], split=split,
            horizon=sample["h"], normalization_scale=sample["scale"], objective=objective,
            target_scope="Physical errors against sealed FP64 teacher; selection objective against FP32 training target. Teacher uncertainty is not a certificate",
            **metric))
    return clean(rows)


def _summarize(rows):
    fields = sorted({r["field_cluster"] for r in rows})
    def mean(key):
        return statistics.mean(statistics.mean(r[key] for r in rows if r["field_cluster"] == field) for field in fields)
    return dict(independent_fields=len(fields), queries=len(rows), objective=mean("objective"),
                field_mean_rms=mean("error_rms"), field_mean_absolute_mean_error=mean("mean_error"),
                worst_spatial_maximum=max(r["error_max"] for r in rows),
                physical_interval_violations=sum(r["physical_interval_violations"] for r in rows))


def run(ctx):
    settings = ctx.protocol["resolve"]["fairness"]
    scope = getattr(ctx, "unit", None) or unit(ctx)
    track = scope.get("track", scope.get("tracks", [None])[0])
    seed = int(scope.get("seed", scope.get("seeds", [0])[0]))
    families = settings.get("families", list(FAMILIES))
    if not families or any(f not in FAMILIES for f in families) or len(set(families)) != len(families):
        raise ValueError("Unexpected Resolve fairness family roster")
    optimizer_plan(settings)
    bank = load_bank(ctx, "prepare")
    if any(p["split"] not in ("train", "validation") for p in bank):
        raise ValueError("Fairness stage must never read confirmation/scaling truth")
    train = [p for p in bank if p["split"] == "train"]
    validation = [p for p in bank if p["split"] == "validation"]
    if not train or not validation or ({p["field_cluster"] for p in train} & {p["field_cluster"] for p in validation}):
        raise ValueError("Fairness needs nonempty parent-disjoint train and validation data")
    train_count = int(settings.get("train_count", len(train)))
    if train_count <= 0 or train_count > len(train):
        raise ValueError("Requested fairness training cohort is unavailable")
    train = nested_subset(train, train_count)
    root = Path(ctx.path); root.mkdir(parents=True, exist_ok=True)
    base = make_model("df", track, _config(ctx.protocol, "df")).to(device=ctx.device, dtype=torch.float32)
    arguments = (ctx.protocol["train_grid"], track, ctx.device, base,
                 float(ctx.protocol["training"].get("normalization_floor", 1e-6)))
    samples = _samples(train, ctx.protocol["train_horizons"], *arguments)
    val_samples = _samples(validation, ctx.protocol["validation_horizons"], *arguments)
    parent_by_id = {p["parent_id"]: p for p in train + validation}
    for sample in samples + val_samples:
        ref = reference(parent_by_id[sample["parent_id"]], ctx.protocol["train_grid"], sample["h"], track)
        sample["evaluation_target"] = ref["state"]
        sample["reference_metadata"] = {k: v for k, v in ref.items() if k != "state"}
    plan = dict(binding=binding(ctx), stage=ctx.stage, track=track, seed=seed, settings=settings,
        training_parent_ids=[p["parent_id"] for p in train], validation_parent_ids=[p["parent_id"] for p in validation],
        selection_split="validation", each_setting_receives_long_opportunity=True,
        tiny_selection_eligible=False, published_FNO_reproduction=False)
    plan_path = root / "fairness_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resolve fairness recovery declaration changed")
    write_json(plan_path, plan)
    records, curves, validations = [], [], []
    for family in families:
        ctx.budget.check()
        spec = dict(model_id=f"{track}/{family}/n{train_count}/seed{seed}", family=family,
                    track=track, seed=seed, train_count=train_count)
        complete = root / "completed-families" / f"{family}.json"
        result = run_family_trials(ctx, spec, _config(ctx.protocol, family), samples, val_samples, settings)
        if complete.exists():
            saved = json.loads(complete.read_text())
            if saved["plan_sha256"] != file_digest(plan_path):
                raise ValueError("Completed fairness family binding changed")
        else:
            trial_rows = []
            for row in result["trials"] + result["tiny"]:
                trial_rows.append({k: v for k, v in row.items() if k != "curves"} | dict(
                    diagnostics=trial_diagnostics(row), parameter_movement=parameter_movement(root, row),
                    tiny_gate=tiny_gate(row, settings) if row["phase"] == "tiny_memorization" else None))
            chosen = result["selected"]
            metric_rows, summaries = [], {}
            if chosen is not None:
                model = _load_model(ctx, chosen, root)
                for split, inputs in (("train", samples), ("validation", val_samples)):
                    values = sample_metrics(model, inputs, track, split, ctx)
                    metric_rows.extend(dict(model_id=spec["model_id"], family=family, **v) for v in values)
                    summaries[split] = _summarize(values)
            saved = dict(plan_sha256=file_digest(plan_path), family=family, model_id=spec["model_id"],
                selected_trial_id=chosen["trial_id"] if chosen else None,
                selected_optimizer=chosen["optimizer_control"] if chosen else None,
                selected_learning_rate=chosen["learning_rate"] if chosen else None,
                selected_phase=chosen["phase"] if chosen else None,
                selection_status=chosen["selection_status"] if chosen else "NO_VALIDATED_CHECKPOINT",
                trials=trial_rows, selected_metrics=summaries, metric_rows=metric_rows,
                tiny_parent_id=result["tiny_parent_id"], tiny_horizon=result["tiny_horizon"],
                ownership=MODEL_SPECS[family]["ownership"])
            write_json(complete, clean(saved))
        records.append(saved)
        curves.extend(c for r in result["trials"] + result["tiny"] for c in r["curves"])
        validations.extend(saved["metric_rows"])
        if spec["model_id"] not in getattr(ctx, "identities", set()):
            gates = [r["tiny_gate"]["passed"] for r in saved["trials"] if r["tiny_gate"] is not None]
            gate = True if any(g is True for g in gates) else None if any(g is None for g in gates) else False
            phases = 1 + int(float(settings["equal_time_seconds"]) > 0)
            ctx.record(spec["model_id"], ["R3"], metrics=clean({k:v for k,v in saved.items() if k not in ("trials", "metric_rows")}), checks=[
                check("all-optimizer-opportunities", len(result["trials"]), phases*len(optimizer_plan(settings)), "eq", category="correctness"),
                check("finite-optimizer-trials", all(not r["failure"] and r["diagnostics"]["gradients_finite"] for r in saved["trials"]),
                      True, "eq", category="math", reason="Finite sampled losses/gradients; not a convergence or general stability theorem"),
                check("tiny-residual-target", gate, True, "eq", category="gap", reason="Training-only diagnostic; not a validation accuracy claim"),
                check("validation-fit-selected", saved["selection_status"] == "FITTED_CHECKPOINT", True, "eq", category="gap"),
                check("published-fno-superiority", None, True, "eq", category="utility", reason="Local adaptations and development fields cannot establish published FNO superiority")])
    for family in CONTROLS:
        model = make_model(family, track, _config(ctx.protocol, family)).to(device=ctx.device, dtype=torch.float32)
        rows = sample_metrics(model, val_samples, track, "validation", ctx)
        validations.extend(dict(model_id=f"{track}/{family}/frozen", family=family, **r) for r in rows)
        records.append(dict(family=family, ownership=MODEL_SPECS[family]["ownership"], selection_status="FROZEN_CONTROL",
            selected_metrics={"validation": _summarize(rows)}, trials=[]))
    all_trials = [trial for row in records for trial in row["trials"]]
    result = dict(schema="tdn.resolve-fairness/v1", **plan, records=records,
        completed_trials=len(all_trials), training_seconds=sum(t["training_seconds"] for t in all_trials),
        optimizer_seconds=sum(t["optimizer_seconds"] for t in all_trials),
        accounting_scope="All optimizer opportunities including tiny diagnostics; teacher generation, evaluation and scheduler overhead are separate",
        equal_data=True, equal_updates="See every trial's attained/censored flags; no assertion from requested caps alone",
        equal_time="See attained/time and update censoring; timing is device-specific and overshoot is retained",
        scientific_scope="Development optimization audit; no untouched confirmation or paper-level FNO comparison")
    write_json(root / "fairness_results.json", clean(result))
    write_json(root / "fairness_metrics.json", dict(rows=validations, statistical_unit="independent field"))
    write_json(root / "learning_curves.json", dict(rows=curves, observed_range_is_confidence_interval=False))
    return dict(families=len(families), completed_trials=len(all_trials), training_seconds=result["training_seconds"],
                selected_initializations=sum(r["selection_status"] == "SELECTED_INITIALIZATION" for r in records),
                published_FNO_reproduction=False)
