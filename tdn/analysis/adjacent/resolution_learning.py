"""Prepared-reference 64²/128² development training and frozen evaluation.

GPU work only reads independently prepared CPU banks. Every seed, failed trial,
selected initialization and actual cost remains visible. This larger program
is still development evidence, not an automatic confirmatory comparison.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time

import numpy as np
import torch

from tdn.analysis.frontier.core import clean
from tdn.analysis.frontier.measurement import measure_paired, endpoint_eligibility
from tdn.analysis.frontier.neural import endpoint_errors, rollout, _sync
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import atomic_torch_save
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json
from .models import make_model, MODEL_SPECS
from .pilots import _safe, _validation

SCHEMA = "tdn.adjacent-resolution-learning/v1"
DEFAULT_FROZEN = ("quad2_fixed", "quad4_fixed", "quad2_full", "quad4_full",
                  "analytic_quad_cubic", "df", "etdrk4")


def _options(protocol):
    cfg = protocol["resolution"]
    smoke = str(protocol.get("profile", "")).endswith("smoke")
    training = dict(updates=2 if smoke else 128, rates=[.001] if smoke else [.001, .0003],
        trial_seconds=10 if smoke else 20, validation_every=1 if smoke else 16,
        checkpoint_every=1 if smoke else 8, peak_weight=.1, fit_iterations=4 if smoke else 80,
        optimizer_controls=[dict(id="clipped", clip_grad_norm=1., loss_scale=1.)] if smoke else
            [dict(id="clipped", clip_grad_norm=1., loss_scale=1.),
             dict(id="unclipped_scaled", clip_grad_norm=None, loss_scale=.01)])
    training.update(cfg.get("training", {}))
    if "learning_rates" in training:
        training["rates"] = training["learning_rates"]
    if any(float(control["loss_scale"]) <= 0 for control in training["optimizer_controls"]):
        raise ValueError("Optimizer loss scales must be positive")
    return cfg, training


def _json(value):
    if torch.is_tensor(value):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return clean(value)


def _record(ctx, identity, metrics, finite=None):
    if not hasattr(ctx, "record") or identity in getattr(ctx, "identities", set()):
        return
    from tdn.analysis.frontier.core import check
    ctx.record(identity, ["C01"], metrics=_json(metrics), checks=[
        check("finite-prepared-data-implementation", finite, True, "eq", category="math",
              reason="Finite numerical execution on declared inputs; no theorem claim"),
        check("independent-superiority", None, True, "eq", category="gap",
              reason="All resolution profiles are bounded development evidence")])


def _binding(ctx):
    return dict(schema=SCHEMA, protocol_sha256=digest(ctx.protocol), unit_sha256=digest(ctx.unit))


def _check_device(ctx):
    device = torch.device(ctx.device)
    if not str(ctx.protocol.get("profile", "")).endswith("smoke") and device.type != "cuda":
        raise ValueError("Larger resolution learning requires the real native CUDA workflow")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Resolution CUDA requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def _config(cfg, family):
    result = dict(modes=8, split_modes=4, quad_nodes=2, cubic_nodes=4, hidden=8,
                  reaction_substeps=4)
    result.update(cfg.get("model_config", {}))
    if family == "analytic_quad_cubic":
        result.update(quad_nodes=4, cubic_nodes=4)
    result.update(cfg.get("model_configs", {}).get(family, {}))
    return result


def _bank_dirs(ctx):
    names = ctx.unit.get("bank_units", [])
    if not names:
        raise ValueError("Prepared CPU reference bank prerequisites are required")
    if any(name not in ctx.prerequisites for name in names):
        raise ValueError("Missing prepared reference bank prerequisite")
    return [Path(ctx.prerequisites[name]) for name in names]


def _metadata(ctx, splits):
    from .resolution_diagnostics import iterate_bank_entries
    grid, track = int(ctx.unit["grid"]), ctx.unit["track"]
    rows, identities = [], set()
    for directory in _bank_dirs(ctx):
        for raw in iterate_bank_entries(directory):
            row = dict(raw)
            if row.get("split") not in splits or int(row.get("grid", row.get("n", -1))) != grid or row.get("track") != track:
                continue
            key = (row["parent_id"], track, grid, round(float(row["horizon"]), 12))
            if key in identities:
                raise ValueError("Duplicate prepared learning reference entry")
            identities.add(key); rows.append(row)
    if not rows:
        raise ValueError("No prepared references match this learning shard")
    return rows


def _load(ctx, metadata):
    from .resolution_diagnostics import load_bank_entry
    return load_bank_entry(_bank_dirs(ctx), metadata["parent_id"], metadata["track"],
                           int(ctx.unit["grid"]), float(metadata["horizon"]))


def _geometry(meta, grid):
    parent = meta.get("parent_json", meta.get("parent", {}))
    return Equation(float(meta.get("kappa", parent.get("kappa", .004))),
                    float(meta.get("reaction_rate", parent.get("reaction_rate", 3.)))), \
        Geometry((grid, grid), tuple(meta.get("domain", parent.get("domain", (1., 1.)))))


def _prepared_samples(ctx, device):
    """Only load CPU-generated references; there is no teacher call in this module."""
    cfg, _ = _options(ctx.protocol)
    rows = _metadata(ctx, {"train", "validation"})
    grid, track = int(ctx.unit["grid"]), ctx.unit["track"]
    base = make_model("df", track, _config(cfg, "df")).to(device=device, dtype=torch.float32)
    samples = {"train": [], "validation": []}
    skipped, identities = [], {"train": set(), "validation": set()}
    for row in rows:
        ctx.budget.check()
        initial, reference, meta = _load(ctx, row)
        split = row["split"]
        cluster = meta.get("field_cluster", row.get("field_cluster", row["parent_id"]))
        identities[split].add(cluster)
        if meta.get("accepted", meta.get("reference_accepted")) is not True:
            skipped.append(dict(parent_id=row["parent_id"], horizon=row["horizon"], split=split,
                                reason="REFERENCE_UNACCEPTED")); continue
        if isinstance(reference, dict):
            reference = reference["state"]
        u = initial.to(device=device, dtype=torch.float32)
        target = reference.to(device=device, dtype=torch.float32)
        eq, geom = _geometry(meta, grid)
        with torch.no_grad():
            scale = max(float((base(u, row["horizon"], eq, geom) - target).square().mean()), 1e-12)
        samples[split].append(dict(u=u, target=target, h=float(row["horizon"]), eq=eq, geom=geom,
            scale=scale, parent_id=row["parent_id"], field_cluster=cluster,
            parent=meta.get("parent_json", meta.get("parent", {"parent_id": row["parent_id"]})),
            reference=meta))
    expected_horizons = {round(float(h), 12) for h in cfg.get("training_horizons", cfg["horizons"])}
    for split in ("train", "validation"):
        inventory = [r for r in rows if r["split"] == split]
        if len(identities[split]) != cfg["splits"][split] or len(inventory) != cfg["splits"][split] * len(expected_horizons):
            raise ValueError("Prepared reference bank omits a required field or horizon")
        if any({round(float(r["horizon"]), 12) for r in inventory if r.get("field_cluster", r["parent_id"]) == parent} != expected_horizons
               for parent in identities[split]):
            raise ValueError("Prepared reference parent omits a training horizon")
    if identities["train"] & identities["validation"]:
        raise ValueError("Resolution training/validation parent leakage")
    return samples, dict(reference_entries=len(rows), skipped=skipped,
        train_parents=sorted(identities["train"]), validation_parents=sorted(identities["validation"]),
        accepted_train_samples=len(samples["train"]), accepted_validation_samples=len(samples["validation"]),
        reference_scope="prepared same-grid FD/nodal or dealiased Galerkin teacher; no continuum-truth substitution")


def _trial_id(family, seed, rate, control):
    return digest(dict(family=family, seed=seed, rate=rate, control=control))


class _TrialBudget:
    def __init__(self, parent, start, prior, limit):
        self.parent, self.start, self.prior, self.limit = parent, start, prior, limit
    def check(self):
        self.parent.check()
        if self.prior + time.perf_counter() - self.start >= self.limit:
            raise _TrialCap("TRIAL_WALLTIME_CAP")


class _TrialCap(TimeoutError):
    pass


def _fit(ctx, family, seed, rate, control, samples, device):
    cfg, settings = _options(ctx.protocol)
    root, grid, track = Path(ctx.path), int(ctx.unit["grid"]), ctx.unit["track"]
    trial_id = _trial_id(family, seed, rate, control)
    record_file = root / "trials" / (trial_id + ".json")
    if record_file.exists():
        record = json.loads(record_file.read_text())
        if record["binding"] != _binding(ctx):
            raise ValueError("Resolution trial recovery binding changed")
        for name in ("checkpoint", "initial_checkpoint"):
            if file_digest(_safe(root, record[name])) != record[name + "_sha256"]:
                raise ValueError("Resolution trial checkpoint bytes changed")
        return record
    torch.manual_seed(int(seed))
    model = make_model(family, track, _config(cfg, family)).to(device=device, dtype=torch.float32)
    model.eval(); parameters = [p for p in model.parameters() if p.requires_grad]
    fit = MODEL_SPECS[family].get("fit", "gradient" if parameters else "frozen")
    optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=0.) if fit == "gradient" else None
    initial_state = copy.deepcopy(model.state_dict())
    initial_path = root / "checkpoints" / (trial_id + ".initial.pt")
    progress_path = root / "progress" / (trial_id + ".pt")
    rng = random.Random(int(seed))
    curves, selected, completed, selected_update, prior_seconds = [], initial_state, 0, 0, 0.
    best = initial = None
    if progress_path.exists():
        saved = torch.load(_safe(root, str(progress_path.relative_to(root))), map_location="cpu", weights_only=True)
        if saved["binding"] != _binding(ctx) or saved["trial_id"] != trial_id:
            raise ValueError("Resolution interrupted trial identity changed")
        model.load_state_dict(saved["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["random_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        selected, completed, selected_update = saved["selected_state"], saved["completed"], saved["selected_update"]
        initial, best, curves, prior_seconds = saved["initial"], saved["best"], saved["curves"], saved["active_seconds"]
    else:
        atomic_torch_save(dict(state_dict=initial_state), initial_path)
    start = time.perf_counter()
    budget = _TrialBudget(ctx.budget, start, prior_seconds, float(settings["trial_seconds"]))
    failure, stop_reason, fit_info = None, None, None
    def journal():
        _sync(device)
        atomic_torch_save(dict(binding=_binding(ctx), trial_id=trial_id,
            state_dict=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
            random_state=rng.getstate(), torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            selected_state=selected, completed=completed,
            selected_update=selected_update, initial=initial, best=best, curves=curves,
            active_seconds=prior_seconds + time.perf_counter() - start), progress_path)
    try:
        if initial is None:
            initial = best = _validation(model, samples["validation"], settings["peak_weight"])
            curves.append(dict(update=0, train_loss=None, validation_loss=initial, gradient_norm=None,
                               elapsed_seconds=prior_seconds + time.perf_counter() - start))
        if not samples["train"] or best is None:
            failure = "NO_ACCEPTED_TRAIN_OR_VALIDATION_REFERENCES"
        elif fit not in ("gradient", "frozen"):
            budget.check()
            fit_info = model.fit_least_squares(
                [(s["u"], s["h"], s["eq"], s["geom"], s["target"], 1 / s["scale"]) for s in samples["train"]],
                ridge=1e-8, peak_weight=settings["peak_weight"], budget=budget,
                max_iterations=int(settings["fit_iterations"]))
            candidate = _validation(model, samples["validation"], settings["peak_weight"])
            completed = 1
            if candidate is not None and candidate < best:
                best, selected, selected_update = candidate, copy.deepcopy(model.state_dict()), 1
            curves.append(dict(update=1, train_loss=None, validation_loss=candidate, gradient_norm=None,
                               elapsed_seconds=prior_seconds + time.perf_counter() - start))
        elif fit == "gradient":
            for update in range(completed + 1, int(settings["updates"]) + 1):
                budget.check(); model.train(); optimizer.zero_grad(set_to_none=True)
                sample = samples["train"][rng.randrange(len(samples["train"]))]
                prediction = model(sample["u"], sample["h"], sample["eq"], sample["geom"])
                squared = (prediction - sample["target"]).square()
                loss = (squared.mean() + settings["peak_weight"] * squared.amax()) / sample["scale"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("NONFINITE_TRAINING_LOSS")
                (float(control["loss_scale"]) * loss).backward()
                if control["clip_grad_norm"] is None:
                    norm = torch.sqrt(sum(p.grad.detach().square().sum() for p in parameters if p.grad is not None))
                else:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, float(control["clip_grad_norm"]))
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("NONFINITE_GRADIENT")
                optimizer.step(); completed = update; model.eval()
                candidate = (_validation(model, samples["validation"], settings["peak_weight"])
                    if update % int(settings["validation_every"]) == 0 or update == int(settings["updates"]) else None)
                if candidate is not None and candidate < best:
                    best, selected, selected_update = candidate, copy.deepcopy(model.state_dict()), update
                curves.append(dict(update=update, train_loss=float(loss.detach()), validation_loss=candidate,
                    gradient_norm=float(norm), gradient_clipped=(control["clip_grad_norm"] is not None and float(norm) > control["clip_grad_norm"]),
                    elapsed_seconds=prior_seconds + time.perf_counter() - start))
                if update % int(settings["checkpoint_every"]) == 0:
                    journal()
    except _TrialCap as error:
        stop_reason = str(error)
    except FloatingPointError as error:
        failure = str(error)
    except BaseException as error:
        journal()
        write_json(root / "interrupted-trials" / f"{trial_id}-{time.time_ns()}.json",
            dict(binding=_binding(ctx), trial_id=trial_id, status="INTERRUPTED_NOT_SELECTED",
                 error=f"{type(error).__name__}: {error}", active_seconds=prior_seconds + time.perf_counter() - start,
                 completed_updates=completed))
        raise
    _sync(device)
    elapsed = prior_seconds + time.perf_counter() - start
    model.load_state_dict(selected); model.eval()
    selected_path = root / "checkpoints" / (trial_id + ".selected.pt")
    atomic_torch_save(dict(state_dict=model.state_dict()), selected_path)
    report = clean(model.parameter_report()) if hasattr(model, "parameter_report") else {}
    probes = []
    with torch.no_grad():
        for split in ("train", "validation"):
            for sample in samples[split][:2]:
                if hasattr(model, "parameter_report"):
                    detail = model.parameter_report(sample["u"], sample["h"], sample["eq"], sample["geom"])
                    probes.append(dict(split=split, parent_id=sample["parent_id"], horizon=sample["h"],
                        response=clean(detail.get("response")), input_shape=list(sample["u"].shape),
                        reference_truth_used_at_inference=False))
    fit_active_seconds = elapsed
    _sync(device)
    elapsed = prior_seconds + time.perf_counter() - start
    selection = "FROZEN_CONTROL" if fit == "frozen" else "FITTED_CHECKPOINT" if selected_update else "SELECTED_INITIALIZATION"
    row = dict(binding=_binding(ctx), model_id=f"n{grid}/{track}/{family}/seed{seed}",
        family=family, role=MODEL_SPECS[family]["role"], track=track, seed=int(seed), grid=grid, train_grid=grid,
        config=_config(cfg, family), fit=fit, fit_info=clean(fit_info), trial_id=trial_id, learning_rate=rate, optimizer_control=control,
        selection_status=selection, failure=failure, stop_reason=stop_reason,
        initial_validation_objective=initial, validation_objective=best, checkpoint_validated=best is not None,
        updates_completed=completed, updates_selected=selected_update,
        updates_requested=int(settings["updates"]) if fit == "gradient" else 0,
        train_seconds=elapsed, fit_active_seconds=fit_active_seconds, training_examples_seen=completed if fit == "gradient" else len(samples["train"]),
        parameter_count=sum(p.numel() for p in model.parameters()) + int(report.get("fitted_buffer_coefficients", 0)),
        trainable_parameter_count=sum(p.numel() for p in parameters), parameter_report=report,
        architecture=model.architecture_metadata() if hasattr(model, "architecture_metadata") else {}, response_probes=probes,
        checkpoint=str(selected_path.relative_to(root)), checkpoint_sha256=file_digest(selected_path),
        initial_checkpoint=str(initial_path.relative_to(root)), initial_checkpoint_sha256=file_digest(initial_path),
        curves=curves, per_trial_walltime_cap_seconds=settings["trial_seconds"],
        optimization_adequacy="FAILED" if failure else "INITIALIZATION_SELECTED" if selected_update == 0 else
            "TIME_BUDGET_LIMITED" if stop_reason else "BOUNDED_RECIPE_NOT_CONVERGENCE_CERTIFICATE",
        scientific_verdict="NA", scientific_reason="All resolution profiles are development evidence",
        energy_joules=None, monetary_cost=None)
    write_json(record_file, clean(row)); return clean(row)


def run_train(ctx):
    device = _check_device(ctx)
    root, start = Path(ctx.path), time.perf_counter(); root.mkdir(parents=True, exist_ok=True)
    samples, references = _prepared_samples(ctx, device)
    preprocess_seconds = time.perf_counter() - start
    cfg, settings = _options(ctx.protocol)
    records, catalog, curves = [], [], []
    for family in ctx.unit["families"]:
        for seed in ctx.unit.get("seeds", cfg.get("seeds", [731001])):
            fit = MODEL_SPECS[family].get("fit", "gradient")
            candidates = []
            controls = settings["optimizer_controls"] if fit == "gradient" else [dict(id="bounded_non_neural_fit", clip_grad_norm=None, loss_scale=1.)]
            rates = settings["rates"] if fit == "gradient" else [0.]
            for control in controls:
                for rate in rates:
                    ctx.budget.check()
                    trial = _fit(ctx, family, seed, rate, control, samples, device)
                    records.append(trial); candidates.append(trial)
                    curves.extend({k: trial[k] for k in ("model_id", "family", "role", "grid", "track", "seed", "trial_id")} | c
                                  for c in trial["curves"])
            valid = [r for r in candidates if r["checkpoint_validated"]]
            chosen = min(valid, key=lambda r: r["validation_objective"]) if valid else candidates[0]
            catalog.append({k: v for k, v in chosen.items() if k != "curves"} |
                dict(all_trial_ids=[r["trial_id"] for r in candidates],
                     total_tuning_and_training_seconds=sum(r["train_seconds"] for r in candidates)))
    for row in catalog:
        _record(ctx, row["model_id"], row, row["checkpoint_validated"])
    write_json(root / "catalog.json", catalog)
    write_json(root / "trials.json", records)
    write_json(root / "loss-curves.json", curves)
    write_json(root / "training-references.json", references)
    summary = dict(schema=SCHEMA, status="COMPLETED", rows=catalog, model_count=len(catalog), trial_count=len(records),
        grid=ctx.unit["grid"], track=ctx.unit["track"], reference_inventory=references,
        data_loading_and_scaling_seconds=preprocess_seconds, train_and_tuning_seconds=sum(r["train_seconds"] for r in records),
        elapsed_seconds=time.perf_counter() - start, device=str(device),
        fairness=dict(equal_data=True, equal_update_cap=True, shared_optimizer_menu=True, equal_measured_compute=False,
            all_actual_costs_retained=True, fno_paper_reproduction=False),
        scientific_outcome="DEVELOPMENT_ONLY", energy_joules=None, monetary_cost=None)
    write_json(root / "training-summary.json", clean(summary)); return clean(summary)


def run_freeze(ctx):
    root = Path(ctx.path); root.mkdir(parents=True, exist_ok=True)
    expected = {key for key, unit in ctx.protocol["units"].items() if unit["kind"] == "resolution_train"}
    if not expected <= set(ctx.prerequisites):
        raise ValueError("Freeze requires every declared training shard")
    catalog, source_units, seen = [], {}, set()
    parent_sets = {"train": set(), "validation": set()}
    for unit in sorted(expected):
        ctx.budget.check()
        directory = Path(ctx.prerequisites[unit])
        rows = json.loads(_safe(directory, "catalog.json").read_text())
        source_units[unit] = dict(catalog_sha256=file_digest(directory / "catalog.json"))
        reference = json.loads(_safe(directory, "training-references.json").read_text())
        parent_sets["train"].update(reference["train_parents"])
        parent_sets["validation"].update(reference["validation_parents"])
        declaration = ctx.protocol["units"][unit]
        expected_ids = {f"n{declaration['grid']}/{declaration['track']}/{family}/seed{seed}"
            for family in declaration["families"] for seed in declaration.get("seeds", ctx.protocol["resolution"]["seeds"])}
        if {r["model_id"] for r in rows} != expected_ids or len(rows) != len(expected_ids):
            raise ValueError("Training shard has an incomplete or duplicated model inventory")
        for row in rows:
            if row["model_id"] in seen:
                raise ValueError("Duplicate model while freezing resolution selections")
            seen.add(row["model_id"])
            if row["binding"] != dict(schema=SCHEMA, protocol_sha256=digest(ctx.protocol), unit_sha256=digest(declaration)):
                raise ValueError("Training catalog belongs to another resolution scope")
            source = _safe(directory, row["checkpoint"])
            if file_digest(source) != row["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed before freezing")
            name = "checkpoints/" + digest(row["model_id"]) + ".pt"
            destination = root / name; destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            catalog.append({**row, "checkpoint": name, "source_unit": unit,
                            "source_checkpoint": row["checkpoint"]})
    if parent_sets["train"] & parent_sets["validation"]:
        raise ValueError("Frozen resolution fields leak between train and validation")
    write_json(root / "catalog.json", catalog)
    artifacts = {"catalog.json": file_digest(root / "catalog.json")}
    artifacts.update({row["checkpoint"]: row["checkpoint_sha256"] for row in catalog})
    freeze = dict(schema=SCHEMA, protocol_sha256=digest(ctx.protocol), artifacts=artifacts,
        source_units=source_units, training_parents=sorted(parent_sets["train"]),
        validation_parents=sorted(parent_sets["validation"]), model_count=len(catalog),
        selection="independent validation only; all declared seeds retained; frozen before evaluation references",
        scientific_scope="development; no automatic confirmatory superiority")
    write_json(root / "freeze.json", dict(freeze, freeze_sha256=digest(freeze)))
    _record(ctx, "frozen-selection", dict(model_count=len(catalog), selection_sha256=digest(freeze)), True)
    return dict(status="COMPLETED", rows=catalog, model_count=len(catalog), scientific_outcome="DEVELOPMENT_ONLY")


def verify_freeze(protocol, root):
    root = Path(root)
    value = json.loads(_safe(root, "freeze.json").read_text())
    if value.get("schema") != SCHEMA or value.get("protocol_sha256") != digest(protocol):
        raise ValueError("Resolution freeze protocol differs")
    if value.get("freeze_sha256") != digest({k: v for k, v in value.items() if k != "freeze_sha256"}):
        raise ValueError("Resolution freeze digest differs")
    for relative, expected in value["artifacts"].items():
        if file_digest(_safe(root, relative)) != expected:
            raise ValueError("Frozen resolution artifact changed")
    return value


def run_evaluate(ctx):
    device = _check_device(ctx)
    cfg, _ = _options(ctx.protocol)
    root = Path(ctx.path); root.mkdir(parents=True, exist_ok=True)
    freeze_dir = Path(ctx.prerequisites[ctx.unit.get("freeze_unit", "freeze")])
    frozen = verify_freeze(ctx.protocol, freeze_dir)
    grid, track = int(ctx.unit["grid"]), ctx.unit["track"]
    all_models = json.loads(_safe(freeze_dir, "catalog.json").read_text())
    transfer_enabled = bool(cfg.get("cross_grid_transfer", True)) and grid == 128
    selected = [r for r in all_models if r["track"] == track and
                (r["train_grid"] == grid or transfer_enabled and r["train_grid"] == 64)]
    if not selected:
        raise ValueError("No frozen trained models for this evaluation grid/track")
    roster = []
    for row in selected:
        model = make_model(row["family"], track, row["config"]).to(device=device, dtype=torch.float32)
        payload = torch.load(_safe(freeze_dir, row["checkpoint"]), map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"], strict=True); model.eval()
        roster.append((row, model))
    for family in cfg.get("frozen_families", cfg.get("frozen_controls", DEFAULT_FROZEN)):
        model = make_model(family, track, _config(cfg, family)).to(device=device, dtype=torch.float32); model.eval()
        roster.append((dict(model_id=f"analytic/n{grid}/{track}/{family}", family=family, role=MODEL_SPECS[family]["role"],
            track=track, seed=None, grid=grid, train_grid=None, config=_config(cfg, family),
            selection_status="FROZEN_CONTROL", checkpoint_sha256=None, parameter_count=0,
            optimization_adequacy="NOT_APPLICABLE"), model))
    raw_rows = _metadata(ctx, {"evaluation", "confirmation"})
    old = set(frozen["training_parents"]) | set(frozen["validation_parents"])
    if any(row.get("field_cluster", row["parent_id"]) in old for row in raw_rows):
        raise ValueError("Resolution evaluation field overlaps frozen fitting fields")
    rows, timings, responses = [], [], []
    start = time.perf_counter()
    for meta_row in raw_rows:
        ctx.budget.check()
        initial, truth, meta = _load(ctx, meta_row)
        if isinstance(truth, dict):
            truth = truth["state"]
        reference = dict(meta, state=truth, accepted=meta.get("accepted", meta.get("reference_accepted", False)))
        eq, geom = _geometry(meta, grid)
        u = initial.to(device=device, dtype=torch.float32)
        shared_steps = cfg.get("endpoint_steps", [1, 2, 4, 8])
        for steps in [*shared_steps, *cfg.get("additional_classical_steps", [])]:
            active_roster = roster if steps in shared_steps else [(s, m) for s, m in roster if s["family"] in ("df", "etdrk4", "rf")]
            if not active_roster:
                continue
            horizon = float(meta_row["horizon"]); schedule = [horizon / steps] * steps
            group_id = f"{meta_row['parent_id']}/{track}/n{grid}/T{horizon:.12g}/steps{steps}"
            binding = dict(**_binding(ctx), freeze_sha256=frozen["freeze_sha256"], group_id=group_id,
                           reference_meta_sha256=digest(clean({k: v for k, v in meta.items() if k != "bank_dir"})))
            journal = root / "completed-groups" / (digest(binding) + ".json")
            if journal.exists():
                saved = json.loads(journal.read_text())
                if saved["binding"] != binding or saved["payload_sha256"] != digest(saved["payload"]):
                    raise ValueError("Resolution paired group recovery identity changed")
                rows.extend(saved["payload"]["rows"]); timings.append(saved["payload"]["timing"])
                responses.extend(saved["payload"].get("responses", [])); continue
            calls = {row["model_id"]: (lambda model=model: rollout(model, u, schedule, eq, geom, ctx.budget)[0])
                     for row, model in active_roster}
            with torch.no_grad():
                outputs, timing = measure_paired(calls, device=device,
                    repeats=int(cfg.get("timing", {}).get("repeats", 3)), warmup=int(cfg.get("timing", {}).get("warmup", 1)),
                    seed=int(digest(group_id)[:8], 16), budget=ctx.budget)
            group_rows, group_responses = [], []
            timing = dict(group_id=group_id, grid=grid, parent_id=meta_row["parent_id"], **timing)
            for spec, model in active_roster:
                measured = timing["methods"][spec["model_id"]]
                row = {k: spec.get(k) for k in ("model_id", "family", "role", "track", "seed", "train_grid",
                    "selection_status", "checkpoint_sha256", "parameter_count", "optimization_adequacy")}
                row.update(dict(grid=grid, evaluation_grid=grid,
                    transfer=spec.get("train_grid") is not None and spec["train_grid"] != grid,
                    parent_id=meta_row["parent_id"], field_cluster=meta.get("field_cluster", meta_row["parent_id"]),
                    regime=meta.get("regime"), alpha=meta.get("alpha"), generator=meta.get("generator"),
                    schedule_id=f"endpoint-{steps}", schedule=schedule, final_time=horizon,
                    cost_seconds=measured["median_seconds"], cold_seconds=measured["cold_seconds"],
                    raw_timing_seconds=measured["samples_seconds"], timing_id=group_id,
                    reference_uncertainty_rms=meta.get("uncertainty_rms"),
                    reference_uncertainty_max=meta.get("uncertainty_max_bound"),
                    batch_size=1, workload="single_field_latency",
                    input_shape=list(u.shape), output_shape=list(outputs[spec["model_id"]].shape),
                    **endpoint_errors(outputs[spec["model_id"]], reference)))
                target = float(cfg.get("rms_target", 2e-5)); max_target = float(cfg.get("max_target", target))
                row.update(eligibility=endpoint_eligibility(row, target, max_target), rms_target=target, max_target=max_target,
                    scientific_verdict="NA", scientific_reason="Resolution development; no locked superiority claim",
                    reference_scope="same-grid FD/nodal" if track == "discrete" else "same-grid dealiased Galerkin",
                    cutoff_policy="fixed absolute physical integer modes across grids",
                    output_modes=spec["config"].get("modes"), split_modes=spec["config"].get("split_modes"),
                    energy_joules=None, monetary_cost=None)
                group_rows.append(clean(row))
                if steps == 1 and hasattr(model, "response"):
                    with torch.no_grad():
                        response = model.response(u, horizon, eq, geom)
                    group_responses.append(dict(model_id=spec["model_id"], parent_id=meta_row["parent_id"],
                        train_grid=spec.get("train_grid"), grid=grid, horizon=horizon, response=_json(response),
                        reference_truth_used_at_inference=False, timing_scope="response probe outside deployment timing"))
            if steps == 1:
                _spatial_snapshot(ctx, u, reference, outputs, active_roster, meta, horizon)
            payload = dict(rows=group_rows, timing=timing, responses=group_responses)
            write_json(journal, dict(binding=binding, payload=clean(payload), payload_sha256=digest(clean(payload))))
            rows.extend(group_rows); timings.append(timing); responses.extend(group_responses)
        for row in rows:
            _record(ctx, f"{row['model_id']}/{row['parent_id']}/T{row['final_time']}/steps{len(row['schedule'])}/batch{row.get('batch_size', 1)}", row, row.get("finite"))
        write_json(root / "evaluation-rows.json", rows)
        write_json(root / "paired-timings.json", timings)
        write_json(root / "parameter-responses.json", responses)
    batch_rows, batch_timings = _throughput(ctx, roster, raw_rows, frozen, device)
    rows.extend(batch_rows); timings.extend(batch_timings)
    for row in batch_rows:
        _record(ctx, f"{row['model_id']}/{row['parent_id']}/T{row['final_time']}/steps{len(row['schedule'])}/batch{row['batch_size']}", row, row.get("finite"))
    write_json(root / "evaluation-rows.json", rows)
    write_json(root / "paired-timings.json", timings)
    summary = dict(schema=SCHEMA, status="COMPLETED", rows=rows, endpoint_count=len(rows),
        grid=grid, track=track, model_count=len(roster),
        independent_parents=len({r["field_cluster"] for r in rows}), freeze_sha256=frozen["freeze_sha256"],
        elapsed_seconds=time.perf_counter() - start, scientific_outcome="DEVELOPMENT_ONLY",
        teacher_work_in_gpu_job=False, timing_scope="complete model rollout, synchronized randomized paired timings",
        costs_not_measured=["energy", "money"], cross_grid_transfer_exploratory=transfer_enabled)
    write_json(root / "evaluation-summary.json", clean(summary)); return clean(summary)


def _throughput(ctx, roster, raw_rows, frozen, device):
    """Distinct-field batches, separately labeled from single-field latency."""
    from collections import defaultdict
    cfg, _ = _options(ctx.protocol)
    sizes = [int(b) for b in cfg.get("timing", {}).get("batches", [1]) if int(b) > 1]
    if not sizes:
        return [], []
    root, grid = Path(ctx.path), int(ctx.unit["grid"])
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(float(row["horizon"]), float(row.get("kappa", cfg.get("kappa", .004))),
                 float(row.get("reaction_rate", cfg.get("reaction_rate", 3.))), tuple(row.get("domain", (1., 1.))))].append(row)
    rows, timings = [], []
    for size in sizes:
        for key, members in sorted(grouped.items()):
            members = sorted(members, key=lambda r: r["parent_id"])
            for offset in range(0, len(members) - size + 1, size):
                chosen = members[offset:offset + size]
                if len({r.get("field_cluster", r["parent_id"]) for r in chosen}) != size:
                    raise ValueError("Throughput requires distinct independent parent fields")
                loaded = [_load(ctx, row) for row in chosen]
                initial = torch.cat([item[0] for item in loaded]).to(device=device, dtype=torch.float32)
                eq, geom = _geometry(loaded[0][2], grid)
                for count in cfg.get("endpoint_steps", [1, 2, 4, 8]):
                    ctx.budget.check()
                    schedule = [key[0] / count] * count
                    group_id = f"batch{size}/" + digest([key, [r["parent_id"] for r in chosen], count])
                    binding = dict(**_binding(ctx), freeze_sha256=frozen["freeze_sha256"], group_id=group_id)
                    journal = root / "completed-batches" / (digest(binding) + ".json")
                    if journal.exists():
                        saved = json.loads(journal.read_text())
                        if saved["binding"] != binding or digest(saved["payload"]) != saved["payload_sha256"]:
                            raise ValueError("Resolution throughput recovery identity changed")
                        rows.extend(saved["payload"]["rows"]); timings.append(saved["payload"]["timing"]); continue
                    calls = {row["model_id"]: (lambda model=model: rollout(model, initial, schedule, eq, geom, ctx.budget)[0])
                             for row, model in roster}
                    with torch.no_grad():
                        answers, timing = measure_paired(calls, device=device,
                            repeats=int(cfg["timing"]["repeats"]), warmup=int(cfg["timing"].get("warmup", 1)),
                            seed=int(digest(group_id)[:8], 16), budget=ctx.budget)
                    timing = dict(timing_id=group_id, grid=grid, batch_size=size,
                                  parent_ids=[r["parent_id"] for r in chosen], **timing)
                    group_rows = []
                    for spec, model in roster:
                        cost = timing["methods"][spec["model_id"]]
                        for index, (declaration, loaded_item) in enumerate(zip(chosen, loaded)):
                            _, truth, meta = loaded_item
                            if isinstance(truth, dict):
                                truth = truth["state"]
                            reference = dict(meta, state=truth, accepted=meta.get("accepted", meta.get("reference_accepted", False)))
                            row = {k: spec.get(k) for k in ("model_id", "family", "role", "track", "seed", "train_grid",
                                "selection_status", "checkpoint_sha256", "parameter_count", "optimization_adequacy")}
                            row.update(grid=grid, evaluation_grid=grid, parent_id=declaration["parent_id"],
                                field_cluster=meta.get("field_cluster", declaration["parent_id"]),
                                regime=meta.get("regime"), alpha=meta.get("alpha"), generator=meta.get("generator"),
                                transfer=spec.get("train_grid") is not None and spec["train_grid"] != grid,
                                schedule=schedule, schedule_id=f"endpoint-{count}", final_time=key[0], batch_size=size,
                                workload="distinct_field_throughput", timing_id=group_id,
                                cost_seconds=cost["median_seconds"] / size, batch_cost_seconds=cost["median_seconds"],
                                cold_seconds=cost["cold_seconds"] / size, raw_timing_seconds=cost["samples_seconds"],
                                timing_scope="amortized per-field batch cost; raw timing samples are complete batch seconds",
                                throughput_fields_per_second=size / cost["median_seconds"],
                                reference_uncertainty_rms=meta.get("uncertainty_rms"),
                                reference_uncertainty_max=meta.get("uncertainty_max_bound"),
                                **endpoint_errors(answers[spec["model_id"]][index:index + 1], reference))
                            target = float(cfg.get("rms_target", 2e-5)); maximum = float(cfg.get("max_target", target))
                            row.update(rms_target=target, max_target=maximum, eligibility=endpoint_eligibility(row, target, maximum),
                                scientific_verdict="NA", scientific_reason="Paired development batch; no confirmatory superiority",
                                energy_joules=None, monetary_cost=None)
                            group_rows.append(clean(row))
                    payload = clean(dict(rows=group_rows, timing=timing))
                    write_json(journal, dict(binding=binding, payload=payload, payload_sha256=digest(payload)))
                    rows.extend(group_rows); timings.append(timing)
    return rows, timings


def run_aggregate(ctx):
    """Verify every planned shard/model/parent/schedule before descriptive summaries."""
    from collections import Counter, defaultdict
    cfg, _ = _options(ctx.protocol)
    root = Path(ctx.path); root.mkdir(parents=True, exist_ok=True)
    freeze_dir = Path(ctx.prerequisites[ctx.unit.get("freeze_unit", "freeze")])
    frozen = verify_freeze(ctx.protocol, freeze_dir)
    catalog = json.loads(_safe(freeze_dir, "catalog.json").read_text())
    expected_units = set(ctx.unit.get("evaluation_units", [k for k, u in ctx.protocol["units"].items()
                                                           if u["kind"] == "resolution_evaluate"]))
    if not expected_units <= set(ctx.prerequisites):
        raise ValueError("Resolution aggregate is missing a required evaluation shard")
    rows, seen, sources = [], set(), {}
    for unit in sorted(expected_units):
        ctx.budget.check()
        directory = Path(ctx.prerequisites[unit])
        part = json.loads(_safe(directory, "evaluation-rows.json").read_text())
        summary = json.loads(_safe(directory, "evaluation-summary.json").read_text())
        if summary.get("freeze_sha256") != frozen["freeze_sha256"] or summary.get("status") != "COMPLETED":
            raise ValueError("Resolution evaluation shard has different or incomplete frozen selection")
        declaration = ctx.protocol["units"][unit]
        grid, track = declaration["grid"], declaration["track"]
        target_models = [r["model_id"] for r in catalog if r["track"] == track and
            (r["train_grid"] == grid or cfg.get("cross_grid_transfer", True) and grid == 128 and r["train_grid"] == 64)]
        target_models += [f"analytic/n{grid}/{track}/{family}" for family in cfg.get("frozen_families", DEFAULT_FROZEN)]
        single = [r for r in part if r.get("batch_size", 1) == 1]
        parents = sorted({r["parent_id"] for r in single})
        from .resolution_diagnostics import resolution_parent
        declared_parents = {resolution_parent(ctx.protocol, "evaluation", int(index), int(grid))["parent_id"]
                            for index in declaration["field_indices"]}
        if set(parents) != declared_parents or len(parents) != len(declaration["field_indices"]):
            raise ValueError("Resolution evaluation shard omitted or added a parent")
        expected = {(parent, model, round(float(h), 12), int(count), 1)
            for parent in parents for model in target_models for h in cfg["evaluation_horizons"]
            for count in cfg["endpoint_steps"]}
        classical = [model for model in target_models if model.rsplit("/", 1)[-1] in ("df", "etdrk4", "rf")]
        expected |= {(parent, model, round(float(h), 12), int(count), 1)
            for parent in parents for model in classical for h in cfg["evaluation_horizons"]
            for count in cfg.get("additional_classical_steps", [])}
        for batch in cfg.get("timing", {}).get("batches", [1]):
            if batch > 1 and len(parents) >= batch:
                # Frozen full shards intentionally group four distinct fields
                # with identical physics so every declared batch is realizable.
                if len(parents) % batch:
                    raise ValueError("Batch inventory requires complete declared parent groups")
                expected |= {(parent, model, round(float(h), 12), int(count), batch)
                    for parent in parents for model in target_models for h in cfg["evaluation_horizons"]
                    for count in cfg["endpoint_steps"]}
        actual = {(r["parent_id"], r["model_id"], round(float(r["final_time"]), 12), len(r["schedule"]), r.get("batch_size", 1))
                  for r in part}
        if actual != expected or len(actual) != len(part):
            raise ValueError("Resolution aggregate rejects missing, extra or duplicate endpoint cells")
        for row in part:
            if row["grid"] != grid or row["track"] != track or row.get("scientific_verdict") != "NA":
                raise ValueError("Resolution rows cross declared scope or promote development to confirmation")
            identity = (grid, track, row["parent_id"], row["model_id"], row["final_time"],
                        tuple(row["schedule"]), row.get("batch_size", 1))
            if identity in seen:
                raise ValueError("Resolution evaluation repeats an endpoint across shards")
            seen.add(identity); rows.append(row)
        sources[unit] = dict(rows_sha256=file_digest(directory / "evaluation-rows.json"),
                             summary_sha256=file_digest(directory / "evaluation-summary.json"), rows=len(part))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["grid"], row["track"], row["model_id"], row.get("batch_size", 1),
                row["final_time"], len(row["schedule"]))].append(row)
    descriptive = []
    for key, members in sorted(groups.items()):
        finite = [r for r in members if r.get("finite") and r.get("reference_accepted")]
        descriptive.append(dict(grid=key[0], track=key[1], model_id=key[2], batch_size=key[3],
            final_time=key[4], steps=key[5], independent_parent_denominator=len({r["field_cluster"] for r in members}),
            valid_parents=len({r["field_cluster"] for r in finite}),
            median_error_rms=statistics.median(r["error_rms"] for r in finite) if finite else None,
            median_cost_seconds=statistics.median(r["cost_seconds"] for r in finite) if finite else None,
            maximum_error=max((r["error_max"] for r in finite), default=None),
            failure_count=len(members) - len(finite), eligibility_counts=dict(Counter(r["eligibility"] for r in members)),
            scientific_verdict="NA", scope="descriptive fixed grid/track/schedule/batch; fields remain the independent unit"))
    write_json(root / "evaluation-rows.json", rows)
    write_json(root / "descriptive-comparisons.json", descriptive)
    write_json(root / "aggregate-sources.json", sources)
    summary = dict(schema=SCHEMA, status="COMPLETED", rows=descriptive,
        expected_shards=len(expected_units), verified_shards=len(sources), endpoint_count=len(rows),
        independent_parent_count=len({r["field_cluster"] for r in rows}),
        freeze_sha256=frozen["freeze_sha256"], scientific_outcome="DEVELOPMENT_ONLY",
        primary_claims="NA: bounded development, not an independently powered confirmatory campaign",
        no_universal_leaderboard=True)
    write_json(root / "aggregate-summary.json", summary)
    _record(ctx, "verified-resolution-aggregate", {k: v for k, v in summary.items() if k != "rows"}, True)
    return summary


def _spatial_snapshot(ctx, initial, reference, outputs, roster, meta, horizon):
    """A predeclared first-parent/horizon view, never selected by observed error."""
    cfg, _ = _options(ctx.protocol)
    if int(ctx.unit.get("part", 0)) != 0 or int(meta.get("index", -1)) != 0 or not math.isclose(
            horizon, float(cfg["evaluation_horizons"][0]), rel_tol=0, abs_tol=1e-12):
        return
    families = {"channel_global", "channel_neural", "quad2_conditioned", "fno_small", "analytic_quad_cubic"}
    root = Path(ctx.path)
    arrays = dict(initial=initial.detach().cpu().numpy(), reference=reference["state"].detach().cpu().numpy())
    rows = []
    for spec, model in roster:
        if spec["family"] not in families or spec.get("seed") not in (None, cfg["seeds"][0]) or spec.get("train_grid") not in (None, ctx.unit["grid"]):
            continue
        key = "prediction_" + digest(spec["model_id"])[:16]
        arrays[key] = outputs[spec["model_id"]].detach().cpu().numpy()
        arrays[key + "_error"] = arrays[key] - arrays["reference"]
        row = dict(model_id=spec["model_id"], family=spec["family"], role=spec["role"],
            grid=ctx.unit["grid"], track=ctx.unit["track"], parent_id=meta["parent_id"], horizon=horizon,
            array_file="spatial-observations.npz",
            array_keys=dict(initial="initial", reference="reference", prediction=key, error=key + "_error"),
            selection="predeclared parent index zero, first evaluation horizon, first training seed; not best-error selection",
            reference_scope="same-grid target", reference_accepted=reference["accepted"])
        if spec["family"] == "channel_neural":
            eq, geom = _geometry(meta, int(ctx.unit["grid"]))
            with torch.no_grad():
                channels = model.channels(initial, horizon, eq, geom).detach().cpu().numpy()
            for index, label in enumerate(("LL", "LH", "HH")):
                arrays[key + "_" + label] = channels[:, index:index + 1]
                row["array_keys"][label] = key + "_" + label
        rows.append(row)
    with (root / "spatial-observations.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(root / "spatial-observations.json", dict(rows=rows,
        array_sha256=file_digest(root / "spatial-observations.npz"), source_parent=meta,
        rendering_interpolation=None, observations_are_actual_grid_values=True))
