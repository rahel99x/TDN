"""Bounded, split-disjoint interaction-channel pilots.

Selection uses validation only. Fresh evaluation fields are created only after
an exact checkpoint/data freeze. Short pilots explicitly leave FNO adequacy and
scientific superiority unresolved; raw failed and initialization trials survive.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from tdn.analysis.frontier.core import clean
from tdn.analysis.frontier.data import _temporal_teacher
from tdn.analysis.frontier.measurement import measure_paired, endpoint_eligibility
from tdn.analysis.frontier.neural import endpoint_errors, rollout, _sync
from tdn.analysis.roadmap.numerics import fourier_resample
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import atomic_torch_save
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json

SCHEMA = "tdn.adjacent-pilot/v1"
DEFAULT_MODELS = ("quad2_fixed", "quad4_fixed", "channel_fixed", "channel_global",
                  "channel_affine", "channel_neural", "quad2_conditioned", "band_gain",
                  "feature_capacity", "analytic_quad_cubic", "df", "etdrk4", "fno_small")


class _Budget:
    def __init__(self, seconds, stop=None):
        self.start, self.seconds, self.stop = time.perf_counter(), float(seconds), stop

    def check(self):
        if self.stop is not None:
            if callable(self.stop):
                if self.stop():
                    raise InterruptedError("Adjacent pilot stop requested; partial trials retained")
            elif hasattr(self.stop, "is_set") and self.stop.is_set():
                raise InterruptedError("Adjacent pilot stop requested; partial trials retained")
            elif hasattr(self.stop, "check"):
                self.stop.check()
        if time.perf_counter() - self.start > self.seconds:
            raise TimeoutError("Adjacent pilot bounded walltime exceeded; artifacts retained")


class _TrialLimit(TimeoutError):
    pass


class _TrialBudget:
    def __init__(self, parent, start, seconds):
        self.parent, self.start, self.seconds = parent, start, seconds

    def check(self):
        self.parent.check()
        if time.perf_counter() - self.start >= self.seconds:
            raise _TrialLimit("TRIAL_TIME_CAP")


def settings(protocol):
    profile = protocol.get("profile", "smoke")
    small = profile == "smoke"
    values = dict(grid=8 if small else 16, tracks=["discrete", "continuum"],
        train_fields=2 if small else 8, validation_fields=2 if small else 4,
        evaluation_fields=2 if small else 4, updates=2 if small else 24,
        seeds=[731001], horizons=[.06] if small else [.04, .12],
        endpoint_steps=[1, 2, 4, 8], alpha_cycle=[.3, .7, 1.0],
        models=list(DEFAULT_MODELS), max_seconds=600 if small else 1200,
        evaluation_seconds=600 if small else 1200, trial_seconds=15 if small else 45,
        reference_tolerance=1e-7, reference_substeps=8, reference_max_substeps=128,
        peak_weight=.1, learning_rates=[.001], clip_grad_norm=1., fit_iterations=40,
        timing_repeats=2 if small else 5, timing_warmup=1, tolerance=2e-5,
        kappa=.004, reaction_rate=3., rms=.05, mean=.43,
        model_config=dict(modes=2 if small else 4, split_modes=1 if small else 2,
                          quad_nodes=2, cubic_nodes=4, hidden=8, reaction_substeps=4),
        seed_base=81000000 if small else 82000000,
        field_generators=["multiscale_2d", "ridge"])
    declared = protocol.get("pilot", {})
    values.update(declared)
    for old, new in (("val_fields", "validation_fields"), ("max_substeps", "reference_max_substeps"),
                     ("seed_offset", "seed_base")):
        if old in declared and new not in declared:
            values[new] = declared[old]
    values.setdefault("validation_every", 1 if small else 4)
    values.setdefault("primary_contrasts", [["channel_neural", "channel_global"],
                        ["channel_neural", "quad2_conditioned"], ["channel_neural", "band_gain"]])
    for key in ("grid", "train_fields", "validation_fields", "evaluation_fields", "updates",
                "reference_substeps", "reference_max_substeps", "timing_repeats", "fit_iterations"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"Positive integer pilot {key} required")
    if values["grid"] < 8 or values["grid"] & (values["grid"] - 1):
        raise ValueError("Pilot grids must be powers of two at least eight")
    if set(values["tracks"]) - {"discrete", "continuum"} or not values["tracks"]:
        raise ValueError("Explicit discrete and/or continuum target tracks required")
    if not values["horizons"] or any(not math.isfinite(x) or x <= 0 for x in values["horizons"]):
        raise ValueError("Positive finite pilot horizons required")
    if not values["endpoint_steps"] or any(type(x) is not int or x < 1 for x in values["endpoint_steps"]):
        raise ValueError("Positive integer shared endpoint steps required")
    return values


def _model_config(options, family):
    config = dict(options["model_config"])
    # Keep the strongest analytic GL4 quadratic+cubic control separately from
    # the two-node normalized/channel attribution comparison.
    if family == "analytic_quad_cubic":
        config.update(quad_nodes=4, cubic_nodes=4)
    config.update(options.get("model_configs", {}).get(family, {}))
    return config


def _device(protocol, device):
    device = torch.device(device)
    if protocol.get("profile") == "full" and device.type != "cuda":
        raise ValueError("Full adjacent confirmation requires native allocated CUDA execution")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def _identity(protocol):
    return dict(schema=SCHEMA, protocol_sha256=digest(protocol), settings_sha256=digest(settings(protocol)))


def _safe(root, relative):
    root, rel = Path(root), Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("Unsafe pilot artifact path")
    path = root / rel
    if not path.is_file() or root.is_symlink() or any((root / Path(*rel.parts[:i])).is_symlink()
                                                   for i in range(1, len(rel.parts) + 1)):
        raise ValueError("Missing or unsafe pilot artifact")
    return path


def _parents(options, split):
    from .fields import make_parent
    offsets = {"train": 0, "validation": 100000, "confirmation": 200000}
    counts = {"train": options["train_fields"], "validation": options["validation_fields"],
              "confirmation": options["evaluation_fields"]}
    levels = min(3, int(math.log2(options["grid"])) - 1)
    parents = []
    for i in range(counts[split]):
        seed = int(options["seed_base"]) + offsets[split] + i
        parent = make_parent(seed, options["alpha_cycle"][i % len(options["alpha_cycle"])],
            generator=options["field_generators"][(i % len(options["alpha_cycle"]) + i // len(options["alpha_cycle"])) % len(options["field_generators"])],
            levels=levels, mean=options["mean"], rms=options["rms"],
            phase_mode="random", split=split, parent_id=f"adjacent/{split}/{seed}")
        parents.append(parent)
    return parents


def _teacher(parent, n, horizon, track, options, budget):
    """Refined independent Lawson teacher; projected continuum is a distinct target."""
    from .fields import sample_field
    eq = Equation(options["kappa"], options["reaction_rate"])
    reference_options = dict(tolerance=options["reference_tolerance"] / (4 if track == "continuum" else 1),
        substeps=options["reference_substeps"], max_substeps=options["reference_max_substeps"],
        roundoff_floor=1e-12)
    factors = [1] if track == "discrete" else [2, 4]
    values, metadata, projected_differences = [], [], []
    start = time.perf_counter()
    for factor in factors:
        size = n * factor
        initial = sample_field(parent, size).double().cpu()
        value, meta = _temporal_teacher(initial, horizon, eq, Geometry((size, size), (1., 1.)),
                                        track, reference_options, budget)
        difference = meta.pop("_difference")
        projected_differences.append(difference if factor == 1 else fourier_resample(difference, (n, n)))
        values.append(value if factor == 1 else fourier_resample(value, (n, n)))
        metadata.append(meta)
    spatial = values[-1] - values[0] if len(values) > 1 else torch.zeros_like(values[0])
    rms_spatial, max_spatial = float(spatial.square().mean().sqrt()), float(spatial.abs().max())
    # Project the actual temporal-refinement differences: even-grid Nyquist
    # folding need not contract a fine-grid norm. These remain estimates.
    temporal_rms = sum(max(float(d.square().mean().sqrt()), 1e-12) for d in projected_differences)
    temporal_max = sum(max(float(d.abs().max()), 1e-12) for d in projected_differences)
    uncertainty_rms, uncertainty_max = rms_spatial + temporal_rms, max_spatial + temporal_max
    accepted = all(m["accepted"] for m in metadata) and max(uncertainty_rms, uncertainty_max) <= options["reference_tolerance"]
    result = dict(accepted=accepted, uncertainty_rms=uncertainty_rms,
        uncertainty_max_bound=uncertainty_max, track=track, grid=n, horizon=horizon,
        temporal_refinement=metadata, spatial_factors=factors, spatial_difference_rms=rms_spatial,
        spatial_difference_max=max_spatial, reference_seconds=time.perf_counter() - start,
        uncertainty_semantics="actual refinement difference plus temporal estimates; not an error certificate",
        target="FD nodal semidiscrete" if track == "discrete" else "projected spatially refined dealiased Galerkin")
    return values[-1], result


def _bank(protocol, root, splits, budget, *, evaluation_count=None):
    from .fields import sample_field
    options = settings(protocol)
    if evaluation_count is not None:
        options["evaluation_fields"] = evaluation_count
    bankpath, datafile = root / "reference-bank.json", root / "reference-bank.npz"
    expected = [p for split in splits for p in _parents(options, split)]
    declaration = [_parent_dict(p) for p in expected]
    binding = dict(**_identity(protocol), splits=list(splits), parents=declaration)
    if bankpath.exists():
        record = json.loads(bankpath.read_text())
        if record["binding"] != binding or file_digest(datafile) != record["data_sha256"]:
            raise ValueError("Pilot reference bank identity or data changed")
        with np.load(datafile, allow_pickle=False) as stored:
            arrays = {k: torch.from_numpy(v.copy()) for k, v in stored.items()}
        return record, arrays
    arrays, refs = {}, []
    for pi, parent in enumerate(expected):
        budget.check()
        parent_binding = dict(binding=_identity(protocol), parent=declaration[pi], parent_index=pi)
        stem = digest(parent_binding)
        journal = root / "reference-parents" / (stem + ".json")
        parent_data = root / "reference-parents" / (stem + ".npz")
        if journal.exists():
            saved = json.loads(journal.read_text())
            if saved["binding"] != parent_binding or file_digest(parent_data) != saved["data_sha256"]:
                raise ValueError("Recovered reference parent identity or bytes changed")
            with np.load(parent_data, allow_pickle=False) as stored:
                arrays.update({key: torch.from_numpy(value.copy()) for key, value in stored.items()})
            refs.extend(saved["references"])
            continue
        parent_arrays, parent_refs = {}, []
        parent_arrays[f"state-{pi}"] = sample_field(parent, options["grid"]).double().cpu()
        for track in options["tracks"]:
            for hi, horizon in enumerate(options["horizons"]):
                value, metadata = _teacher(parent, options["grid"], horizon, track, options, budget)
                key = f"reference-{pi}-{track}-{hi}"
                parent_arrays[key] = value
                parent_refs.append(dict(parent_index=pi, state_key=f"state-{pi}", key=key,
                                 split=splits[0] if len(splits) == 1 else declaration[pi]["split"],
                                 parent=declaration[pi], **metadata))
        journal.parent.mkdir(parents=True, exist_ok=True)
        with parent_data.open("wb") as stream:
            np.savez_compressed(stream, **{key: value.numpy() for key, value in parent_arrays.items()})
        write_json(journal, dict(binding=parent_binding, references=clean(parent_refs), data_sha256=file_digest(parent_data)))
        arrays.update(parent_arrays); refs.extend(parent_refs)
    root.mkdir(parents=True, exist_ok=True)
    with datafile.open("wb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
    record = dict(binding=binding, references=clean(refs), data_sha256=file_digest(datafile),
                  reference_seconds=sum(r["reference_seconds"] for r in refs),
                  status="ACCEPTED" if all(r["accepted"] for r in refs) else "UNRESOLVED_REFERENCES_RETAINED")
    write_json(bankpath, record)
    return record, arrays


def _parent_dict(parent):
    result = parent.to_dict()
    if "split" not in result:
        result["split"] = parent.split
    return result


def _samples(record, arrays, track, split, options, device):
    eq = Equation(options["kappa"], options["reaction_rate"])
    geom = Geometry((options["grid"],) * 2, (1., 1.))
    from .models import make_model
    base = make_model("df", track, _model_config(options, "df")).to(device=device, dtype=torch.float32)
    samples = []
    for row in record["references"]:
        if row["split"] != split or row["track"] != track or not row["accepted"]:
            continue
        initial = arrays[row["state_key"]].to(device=device, dtype=torch.float32)
        target = arrays[row["key"]].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            scale = max(float((base(initial, row["horizon"], eq, geom) - target).square().mean()), 1e-12)
        samples.append(dict(u=initial, target=target, h=row["horizon"], eq=eq, geom=geom, scale=scale,
                            parent=row["parent"], reference=row))
    return samples


def _validation(model, samples, peak_weight):
    if not samples:
        return None
    values = []
    with torch.no_grad():
        for s in samples:
            error = model(s["u"], s["h"], s["eq"], s["geom"]) - s["target"]
            value = (error.square().mean() + peak_weight * error.square().amax()) / s["scale"]
            if not bool(torch.isfinite(value)):
                return None
            values.append(float(value))
    return statistics.mean(values)


def _fit_trial(protocol, root, family, track, seed, rate, training, validation, budget, device):
    from .models import make_model, MODEL_SPECS
    options = settings(protocol)
    trial_id = digest([family, track, seed, rate])
    trialpath = root / "trials" / f"{trial_id}.json"
    if trialpath.exists():
        record = json.loads(trialpath.read_text())
        if record["binding"] != _identity(protocol):
            raise ValueError("Recovered pilot trial protocol differs")
        for kind in ("checkpoint", "initial_checkpoint"):
            if file_digest(_safe(root, record[kind])) != record[kind + "_sha256"]:
                raise ValueError("Recovered pilot checkpoint changed")
        return record
    torch.manual_seed(seed)
    model = make_model(family, track, _model_config(options, family)).to(device=device, dtype=torch.float32)
    model.eval()
    fit = MODEL_SPECS[family].get("fit", "gradient")
    initial_state = copy.deepcopy(model.state_dict())
    selected = initial_state
    start = time.perf_counter()
    trial_budget = _TrialBudget(budget, start, options["trial_seconds"])
    initial = best = _validation(model, validation, options["peak_weight"])
    selected_update = updates = examples = 0
    failure, fit_info = None, None
    curves = [dict(update=0, train_loss=None, validation_loss=initial, elapsed_seconds=time.perf_counter() - start)]
    initial_path = root / "checkpoints" / f"{trial_id}.initial.pt"
    atomic_torch_save({"state_dict": initial_state}, initial_path)
    try:
        if not training or best is None:
            failure = "NO_ACCEPTED_TRAIN_OR_VALIDATION_REFERENCES"
        elif fit not in ("frozen", "gradient"):
            fit_info = model.fit_least_squares(
                [(s["u"], s["h"], s["eq"], s["geom"], s["target"], 1 / s["scale"]) for s in training],
                ridge=1e-8, peak_weight=options["peak_weight"], budget=trial_budget,
                max_iterations=options["fit_iterations"])
            candidate = _validation(model, validation, options["peak_weight"])
            updates, examples = 1, len(training)
            if candidate is not None and candidate < best:
                best, selected, selected_update = candidate, copy.deepcopy(model.state_dict()), 1
            curves.append(dict(update=1, train_loss=None, validation_loss=candidate, elapsed_seconds=time.perf_counter() - start))
        elif fit == "gradient":
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=0.)
            rng = random.Random(seed)
            for update in range(1, options["updates"] + 1):
                budget.check()
                if time.perf_counter() - start >= options["trial_seconds"]:
                    break
                model.train(); optimizer.zero_grad(set_to_none=True)
                s = training[rng.randrange(len(training))]
                prediction = model(s["u"], s["h"], s["eq"], s["geom"])
                squared = (prediction - s["target"]).square()
                loss = (squared.mean() + options["peak_weight"] * squared.amax()) / s["scale"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("NONFINITE_TRAINING_LOSS")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, options["clip_grad_norm"])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("NONFINITE_GRADIENT")
                optimizer.step(); updates = update; examples += 1
                model.eval(); candidate = (_validation(model, validation, options["peak_weight"])
                             if update % options["validation_every"] == 0 or update == options["updates"] else None)
                if candidate is not None and candidate < best:
                    best, selected, selected_update = candidate, copy.deepcopy(model.state_dict()), update
                curves.append(dict(update=update, train_loss=float(loss.detach()), validation_loss=candidate,
                    gradient_norm=float(norm), examples_seen=examples, elapsed_seconds=time.perf_counter() - start))
    except (FloatingPointError, _TrialLimit) as error:
        failure = str(error)
    except BaseException:
        write_json(root / "interrupted-trials" / f"{trial_id}-{time.time_ns()}.json",
            clean(dict(family=family, track=track, seed=seed, trial_id=trial_id, status="INTERRUPTED_NOT_SELECTED",
                       elapsed_seconds=time.perf_counter() - start, updates=updates, curves=curves)))
        raise
    _sync(device)
    elapsed = time.perf_counter() - start
    model.load_state_dict(selected); model.eval()
    path = root / "checkpoints" / f"{trial_id}.selected.pt"
    atomic_torch_save({"state_dict": selected}, path)
    response = []
    if hasattr(model, "parameter_report"):
        with torch.no_grad():
            for split, samples in (("train", training), ("validation", validation)):
                for s in samples[:2]:
                    response.append(dict(split=split, parent=s["parent"], horizon=s["h"],
                        report=clean(model.parameter_report(s["u"], s["h"], s["eq"], s["geom"]))))
    record = dict(binding=_identity(protocol), family=family, role=MODEL_SPECS[family].get("role"),
        track=track, seed=seed, model_id=f"{track}/{family}/seed{seed}", trial_id=trial_id,
        config=_model_config(options, family), fit=fit, learning_rate=rate,
        selection_status="FROZEN_CONTROL" if fit == "frozen" else "FITTED_CHECKPOINT" if selected_update else "SELECTED_INITIALIZATION",
        failure=failure, initial_validation_objective=initial, validation_objective=best,
        checkpoint_validated=best is not None, selected_update=selected_update, updates=updates,
        updates_requested=options["updates"] if fit == "gradient" else 0,
        train_seconds=elapsed, train_examples_seen=examples,
        parameter_count=sum(p.numel() for p in model.parameters()) + (3 if family == "channel_global" else 12 if family == "channel_affine" else 0),
        fitted_buffer_coefficients=3 if family == "channel_global" else 12 if family == "channel_affine" else 0,
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        fit_info=clean(fit_info), response_probes=response,
        checkpoint=str(path.relative_to(root)), checkpoint_sha256=file_digest(path),
        initial_checkpoint=str(initial_path.relative_to(root)), initial_checkpoint_sha256=file_digest(initial_path),
        curves=curves, per_trial_walltime_cap_seconds=options["trial_seconds"],
        compute_scope="same data/update and walltime ceilings, measured actual costs; neither equal updates nor parameters means equal compute")
    write_json(trialpath, clean(record))
    return clean(record)


def verify_freeze(protocol, root):
    root = Path(root)
    freeze = json.loads(_safe(root, "freeze.json").read_text())
    body = {k: v for k, v in freeze.items() if k != "freeze_sha256"}
    if freeze.get("freeze_sha256") != digest(body) or freeze.get("binding") != _identity(protocol):
        raise ValueError("Pilot freeze identity changed")
    for relative, expected in freeze["artifacts"].items():
        if file_digest(_safe(root, relative)) != expected:
            raise ValueError("Pilot frozen artifact changed")
    return freeze


def run_pilot(protocol, path, device="cpu", stop=None):
    """Fit all declared controls on paired data and freeze validation selections."""
    device, root, options = _device(protocol, device), Path(path), settings(protocol)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "freeze.json").exists():
        verify_freeze(protocol, root)
        return json.loads((root / "pilot-summary.json").read_text())
    budget = _Budget(options["max_seconds"], stop)
    bank, arrays = _bank(protocol, root, ("train", "validation"), budget)
    train_ids = {r["parent"]["parent_id"] for r in bank["references"] if r["split"] == "train"}
    val_ids = {r["parent"]["parent_id"] for r in bank["references"] if r["split"] == "validation"}
    if train_ids & val_ids:
        raise ValueError("Pilot training/validation fields overlap")
    from .models import MODEL_SPECS
    catalog, trials, curves = [], [], []
    for track in options["tracks"]:
        train = _samples(bank, arrays, track, "train", options, device)
        validation = _samples(bank, arrays, track, "validation", options, device)
        for family in options["models"]:
            fit = MODEL_SPECS[family].get("fit", "gradient")
            for seed in options["seeds"] if fit != "frozen" else options["seeds"][:1]:
                candidates = []
                rates = options["learning_rates"] if fit == "gradient" else [0.]
                for rate in rates:
                    budget.check()
                    trial = _fit_trial(protocol, root, family, track, seed, rate, train, validation, budget, device)
                    trials.append(trial); candidates.append(trial)
                    curves.extend({k: trial[k] for k in ("family", "role", "track", "seed", "trial_id")} | c for c in trial["curves"])
                usable = [c for c in candidates if c["checkpoint_validated"]]
                chosen = min(usable, key=lambda c: c["validation_objective"]) if usable else candidates[0]
                catalog.append({k: v for k, v in chosen.items() if k != "curves"} |
                               {"all_trial_ids": [c["trial_id"] for c in candidates],
                                "total_tuning_and_training_seconds": sum(c["train_seconds"] for c in candidates)})
    validation_rows = _validation_rows(protocol, catalog, root, bank, arrays, device, budget)
    write_json(root / "validation-rows.json", validation_rows)
    confirmation_plan = _plan_confirmation(protocol, validation_rows)
    write_json(root / "confirmation-plan.json", confirmation_plan)
    write_json(root / "catalog.json", catalog)
    write_json(root / "trials.json", trials)
    write_json(root / "loss-curves.json", curves)
    summary = dict(schema=SCHEMA, stage="pilot", status="COMPLETED", rows=catalog,
        model_count=len(catalog), trial_count=len(trials), loss_rows=len(curves),
        train_fields=len(train_ids), validation_fields=len(val_ids),
        reference_status=bank["status"], reference_seconds=bank["reference_seconds"],
        train_and_tuning_seconds=sum(t["train_seconds"] for t in trials),
        elapsed_seconds=time.perf_counter() - budget.start, device=str(device),
        scientific_outcome="DEVELOPMENT_ONLY_NO_COMPARATIVE_CLAIM",
        fno_adequacy="UNRESOLVED: local adapted implementation, bounded recipe; not a published FNO reproduction",
        fairness_views=dict(equal_data=True, equal_updates_cap=True, equal_training_compute=False,
            equal_inference_cost=False, actual_costs_retained=True, validation_selection=True),
        energy_joules=None, monetary_cost=None)
    write_json(root / "pilot-summary.json", clean(summary))
    owned_files = {"catalog.json", "trials.json", "loss-curves.json", "pilot-summary.json",
        "validation-rows.json", "confirmation-plan.json", "reference-bank.json", "reference-bank.npz"}
    owned_dirs = {"checkpoints", "trials", "interrupted-trials", "reference-parents"}
    artifacts = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and (str(p.relative_to(root)) in owned_files or
                                    p.relative_to(root).parts[0] in owned_dirs)}
    freeze = dict(binding=_identity(protocol), artifacts=artifacts, model_count=len(catalog),
        training_parents=sorted(train_ids), validation_parents=sorted(val_ids),
        selection="validation only; all seeds retained; all schedules shared; no fresh truth inspected",
        evaluation_design=dict(fields=options["evaluation_fields"], horizons=options["horizons"],
                               endpoint_steps=options["endpoint_steps"], tracks=options["tracks"]))
    write_json(root / "freeze.json", dict(freeze, freeze_sha256=digest(freeze)))
    return clean(summary)


def evaluate_pilot(protocol, path, pilot_dir, device="cpu", stop=None):
    """Evaluate frozen models on fresh paired parents and common endpoint menus.

    These are independent held-out pilot fields, not a powered confirmatory
    sample. No schedule is chosen using their truth; all raw schedules remain.
    """
    from .models import make_model
    device, root, options = _device(protocol, device), Path(path), settings(protocol)
    frozen = verify_freeze(protocol, pilot_dir)
    root.mkdir(parents=True, exist_ok=True)
    budget = _Budget(options["evaluation_seconds"], stop)
    confirmation_plan = json.loads(_safe(pilot_dir, "confirmation-plan.json").read_text())
    evaluation_count = confirmation_plan["chosen_parents"] if protocol.get("profile") == "full" else options["evaluation_fields"]
    bank, arrays = _bank(protocol, root, ("confirmation",), budget, evaluation_count=evaluation_count)
    old = set(frozen["training_parents"]) | set(frozen["validation_parents"])
    if any(r["parent"]["parent_id"] in old for r in bank["references"]):
        raise ValueError("Fresh pilot evaluation parent overlaps fitting data")
    catalog = json.loads((Path(pilot_dir) / "catalog.json").read_text())
    models = {}
    for spec in catalog:
        budget.check()
        model = make_model(spec["family"], spec["track"], spec["config"]).to(device=device, dtype=torch.float32)
        payload = torch.load(_safe(pilot_dir, spec["checkpoint"]), map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"], strict=True); model.eval()
        models[spec["model_id"]] = model
    eq, geom = Equation(options["kappa"], options["reaction_rate"]), Geometry((options["grid"],) * 2, (1., 1.))
    rows, timings = [], []
    for index, reference in enumerate(bank["references"]):
        initial = arrays[reference["state_key"]].to(device=device, dtype=torch.float32)
        truth = dict(reference, state=arrays[reference["key"]])
        applicable = [s for s in catalog if s["track"] == reference["track"]]
        for count in options["endpoint_steps"]:
            budget.check()
            schedule = [reference["horizon"] / count] * count
            timing_id = f"parent-{reference['parent_index']}/{reference['track']}/{reference['horizon']}/steps-{count}"
            group_binding = dict(binding=_identity(protocol), freeze_sha256=frozen["freeze_sha256"],
                                 timing_id=timing_id, bank_sha256=bank["data_sha256"])
            group_path = root / "completed-groups" / (digest(group_binding) + ".json")
            if group_path.exists():
                group = json.loads(group_path.read_text())
                if group["binding"] != group_binding or group.get("payload_sha256") != digest(group["payload"]):
                    raise ValueError("Recovered paired timing group identity changed")
                rows.extend(group["payload"]["rows"])
                timings.append(group["payload"]["timing"])
                continue
            group_rows = []
            calls = {s["model_id"]: (lambda model=models[s["model_id"]]: rollout(model, initial, schedule, eq, geom, budget)[0])
                     for s in applicable}
            with torch.no_grad():
                answers, measured = measure_paired(calls, device=device, repeats=options["timing_repeats"],
                    warmup=options["timing_warmup"], seed=910003 + index * 17 + count, budget=budget)
            timing_id = f"parent-{reference['parent_index']}/{reference['track']}/{reference['horizon']}/steps-{count}"
            timings.append(dict(timing_id=timing_id, **measured))
            for spec in applicable:
                cost = measured["methods"][spec["model_id"]]
                parent = reference["parent"]
                row = {k: spec[k] for k in ("model_id", "family", "role", "track", "seed", "selection_status", "checkpoint_sha256", "parameter_count")}
                row.update(dict(parent_id=parent["parent_id"], field_cluster=parent.get("cluster_id", parent["parent_id"]),
                    alpha=parent.get("alpha"), generator=parent.get("generator"), grid=options["grid"],
                    schedule_id=f"endpoint-{count}", schedule=schedule, final_time=reference["horizon"],
                    timing_id=timing_id, cost_seconds=cost["median_seconds"], cold_seconds=cost["cold_seconds"],
                    raw_timing_seconds=cost["samples_seconds"], reference_uncertainty_rms=reference["uncertainty_rms"],
                    reference_uncertainty_max=reference["uncertainty_max_bound"],
                    **endpoint_errors(answers[spec["model_id"]], truth)))
                row["eligibility"] = endpoint_eligibility(row, options["tolerance"], options["tolerance"])
                row["scientific_verdict"] = "NA"
                row["scientific_reason"] = "Bounded development pilot, no powered familywise confirmatory claim"
                rows.append(clean(row)); group_rows.append(clean(row))
            payload = dict(rows=group_rows, timing=timings[-1])
            write_json(group_path, dict(binding=group_binding, payload=payload, payload_sha256=digest(payload)))
        write_json(root / "evaluation-rows.json", rows)
        write_json(root / "paired-timings.json", timings)
    comparisons = _primary_comparisons(protocol, rows, confirmation_plan)
    write_json(root / "primary-comparisons.json", comparisons)
    summary = dict(schema=SCHEMA, stage="pilot-evaluation", status="COMPLETED", rows=rows,
        primary_comparisons=comparisons,
        model_count=len(catalog), endpoint_count=len(rows), independent_fields=evaluation_count, confirmation_plan=confirmation_plan,
        freeze_sha256=frozen["freeze_sha256"], reference_status=bank["status"],
        reference_seconds=bank["reference_seconds"], elapsed_seconds=time.perf_counter() - budget.start,
        device=str(device), scientific_outcome="HELD_OUT_PILOT_DESCRIPTIVE_ONLY",
        selection="frozen on independent validation; every shared endpoint schedule retained",
        timing_scope="complete model rollout including features, transforms, transport, products, allocations; randomized paired warm rounds",
        cold_scope="first invocation per timer, not cold process", energy_joules=None, monetary_cost=None)
    write_json(root / "evaluation-summary.json", clean(summary))
    return clean(summary)


def _validation_rows(protocol, catalog, root, bank, arrays, device, budget):
    """Paired per-parent pilot variability before any fresh fields exist."""
    from .models import make_model
    options = settings(protocol)
    eq, geom = Equation(options["kappa"], options["reaction_rate"]), Geometry((options["grid"],) * 2, (1., 1.))
    rows = []
    for spec in catalog:
        budget.check()
        model = make_model(spec["family"], spec["track"], spec["config"]).to(device=device, dtype=torch.float32)
        state = torch.load(_safe(root, spec["checkpoint"]), map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"]); model.eval()
        with torch.no_grad():
            for ref in bank["references"]:
                if ref["split"] != "validation" or ref["track"] != spec["track"]:
                    continue
                budget.check()
                initial = arrays[ref["state_key"]].to(device=device, dtype=torch.float32)
                answer = model(initial, ref["horizon"], eq, geom)
                rows.append(dict(family=spec["family"], track=spec["track"], seed=spec["seed"],
                    field_cluster=ref["parent"].get("cluster_id", ref["parent"]["parent_id"]),
                    parent_id=ref["parent"]["parent_id"], horizon=ref["horizon"], steps=1,
                    reference_uncertainty_rms=ref["uncertainty_rms"],
                    **endpoint_errors(answer, dict(ref, state=arrays[ref["key"]]))))
    return rows


def _paired_parent_evidence(rows, candidate, control, *, track, horizons, seeds):
    """Require every preregistered within-parent cell, retaining missing denominators."""
    from collections import defaultdict
    cells, parents = defaultdict(dict), set()
    requested_horizons = {round(float(h), 12) for h in horizons}
    for row in rows:
        if row["family"] not in (candidate, control) or row["track"] != track:
            continue
        if row.get("steps", len(row.get("schedule", [0]))) != 1:
            continue
        h = round(float(row.get("horizon", row.get("final_time"))), 12)
        if h not in requested_horizons or row["seed"] not in seeds:
            continue
        parent = row["field_cluster"]; parents.add(parent)
        key = (parent, h, row["seed"])
        if row["family"] in cells[key]:
            raise ValueError("Duplicate within-parent primary endpoint")
        cells[key][row["family"]] = row
    effects, conservative, missing, reference_unresolved = {}, {}, {}, []
    for parent in sorted(parents):
        raw, robust, incomplete, resolved = [], [], [], True
        for h in sorted(requested_horizons):
            for seed in seeds:
                pair = cells.get((parent, h, seed), {})
                if any(f not in pair or pair[f].get("reference_accepted") is not True or
                       pair[f].get("finite") is not True or pair[f].get("error_rms") is None
                       for f in (candidate, control)):
                    incomplete.append(dict(horizon=h, seed=seed)); continue
                ce, re = float(pair[candidate]["error_rms"]), float(pair[control]["error_rms"])
                cu = pair[candidate].get("reference_uncertainty_rms")
                ru = pair[control].get("reference_uncertainty_rms")
                if cu is None or ru is None or not math.isfinite(cu) or not math.isfinite(ru):
                    incomplete.append(dict(horizon=h, seed=seed)); continue
                raw.append(math.log(max(re, 1e-12) / max(ce, 1e-12)))
                # Reverse triangle inequality supplies a conservative empirical
                # reference-uncertainty sensitivity interval, not a certificate.
                resolved = resolved and re > ru
                robust.append(math.log(max(re - ru, 1e-300) / max(ce + cu, 1e-12)))
        if incomplete:
            missing[parent] = incomplete
        elif raw:
            effects[parent] = statistics.mean(raw)
            conservative[parent] = statistics.mean(robust)
            if not resolved:
                reference_unresolved.append(parent)
    return dict(raw=effects, conservative=conservative, missing_parents=missing,
                reference_unresolved_parents=reference_unresolved, declared_parent_denominator=len(parents))


def _plan_confirmation(protocol, validation_rows):
    """Conservative pilot-variability sample-size planning, not guaranteed power.

    Bonferroni accounts for the three declared primary contrasts. The upper
    one-sided chi-square SD estimate protects against a spuriously tiny pilot
    variance, but remains a distributional planning approximation. Final
    uncertainty is estimated independently from fresh parent-cluster resampling.
    """
    from scipy.stats import norm, chi2
    options = settings(protocol)
    config = protocol.get("confirmation", {})
    cap = int(config.get("maximum_parents", options["evaluation_fields"]))
    halfwidth = float(config.get("desired_log_halfwidth", math.log(1.15)))
    level = float(config.get("confidence_level", .95))
    comparisons = int(config.get("primary_comparisons", len(options["primary_contrasts"]) * len(protocol.get("tracks", options["tracks"]))))
    if not 0 < level < 1 or halfwidth <= 0 or cap < 1:
        raise ValueError("Invalid confirmation precision declaration")
    z = float(norm.ppf(1 - (1 - level) / (2 * comparisons)))
    contrasts = []
    for track in options["tracks"]:
        for candidate, control in options["primary_contrasts"]:
            evidence = _paired_parent_evidence(validation_rows, candidate, control, track=track,
                                                horizons=options["horizons"], seeds=options["seeds"])
            paired = evidence["raw"]
            n = len(paired)
            sd = statistics.stdev(paired.values()) if n > 1 else None
            upper_sd = (sd * math.sqrt((n - 1) / float(chi2.ppf(.05, n - 1)))) if n > 2 else None
            requested = max(16, math.ceil((z * upper_sd / halfwidth) ** 2)) if upper_sd is not None else None
            contrasts.append(dict(track=track, candidate=candidate, control=control, validation_parent_count=n,
                paired_parent_log_error_ratios=paired, pilot_standard_deviation=sd,
                missing_parents=evidence["missing_parents"],
                reference_unresolved_parents=evidence["reference_unresolved_parents"],
                planning_upper_sd=upper_sd, required_parents=requested,
                status="PLANNED" if requested is not None and requested <= cap else "INSUFFICIENT_PRECISION_BUDGET" if requested else "INSUFFICIENT_PILOT_FIELDS"))
    requirements = [c["required_parents"] for c in contrasts]
    required = max(requirements) if all(n is not None for n in requirements) else None
    chosen = min(cap, required) if required is not None else cap
    return dict(schema=SCHEMA, contrasts=contrasts, chosen_parents=chosen, maximum_parents=cap,
        required_parents=required, desired_log_halfwidth=halfwidth, confidence_level=level,
        primary_comparisons=comparisons, multiple_comparisons=f"Bonferroni across all {comparisons} track-specific declared contrasts",
        primary_effect="track-specific one-step paired log RMS ratios; all preregistered horizons/seeds required then averaged within parent",
        planning_scope="validation-selected models on validation variability; conservative SD planning heuristic, not a power guarantee",
        frozen_before_fresh_references=True,
        status="PRECISION_PLAN_WITHIN_CAP" if required is not None and required <= cap else "INSUFFICIENT_PRECISION_BUDGET" if required else "INSUFFICIENT_PILOT_FIELDS")


def design_confirmation(protocol, pilot_dir):
    """Return the immutable pre-fresh-data sample-size decision."""
    verify_freeze(protocol, pilot_dir)
    return json.loads(_safe(pilot_dir, "confirmation-plan.json").read_text())


def _primary_comparisons(protocol, rows, plan):
    """Complete-cell fresh parent intervals with familywise/reference precision gates."""
    options = settings(protocol)
    config = protocol.get("confirmation", {})
    confidence = float(config.get("confidence_level", .95))
    multiplicity = int(config.get("primary_comparisons", len(options["primary_contrasts"]) * len(protocol.get("tracks", options["tracks"]))))
    tail = (1 - confidence) / (2 * multiplicity)
    rng = np.random.default_rng(930031)
    result = []
    for track in options["tracks"]:
        for candidate, control in options["primary_contrasts"]:
            evidence = _paired_parent_evidence(rows, candidate, control, track=track,
                horizons=options["horizons"], seeds=options["seeds"])
            effects = evidence["raw"]
            values = np.asarray(list(effects.values()), dtype=float)
            robust = np.asarray([evidence["conservative"][p] for p in effects], dtype=float)
            low = high = mean = robust_low = robust_high = None
            if len(values) >= 2:
                indices = rng.integers(0, len(values), size=(4000, len(values)))
                boot = values[indices].mean(axis=1)
                robust_boot = robust[indices].mean(axis=1)
                low, high = (float(x) for x in np.quantile(boot, [tail, 1-tail]))
                robust_low, robust_high = (float(x) for x in np.quantile(robust_boot, [tail, 1-tail]))
                mean = float(values.mean())
            precision = low is not None and (high - low) / 2 <= plan["desired_log_halfwidth"]
            native_full = protocol.get("profile") == "full"
            enough = (len(values) >= max(16, plan["chosen_parents"]) and not evidence["missing_parents"]
                      and plan["status"] == "PRECISION_PLAN_WITHIN_CAP")
            reference_resolved = not evidence["reference_unresolved_parents"]
            verdict = "NA"
            if native_full and enough and precision and reference_resolved:
                verdict = "GOOD" if robust_low > math.log(float(config.get("primary_effect_ratio", 1.1))) else "BAD"
            result.append(dict(track=track, candidate=candidate, control=control, independent_parents=len(values),
                paired_parent_effects=effects, mean_log_error_ratio=mean, lower_log_error_ratio=low,
                upper_log_error_ratio=high, conservative_reference_lower_log_ratio=robust_low,
                conservative_reference_upper_log_ratio=robust_high,
                error_ratio=math.exp(mean) if mean is not None else None,
                missing_parents=evidence["missing_parents"], declared_parent_denominator=evidence["declared_parent_denominator"],
                reference_unresolved_parents=evidence["reference_unresolved_parents"],
                interval=f"Bonferroni over {multiplicity} track-specific contrasts; percentile parent-cluster bootstrap, 4000 replicates; finite-sample approximation",
                precision_met=precision, required_inventory_met=enough, verdict=verdict,
                scope="held-out one-step complete-solution RMS; cost, regime subgroups and FNO adequacy are separate",
                cost_claim="NA: bounded shared 1/2/4/8 menu does not establish the optimal classical solver frontier",
                mathematical_validity="NA: empirical gain is not a theorem proof"))
    return result
