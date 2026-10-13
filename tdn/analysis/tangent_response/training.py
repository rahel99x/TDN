"""Bounded staged fitting with parent weights, honest targets and exact resume."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response import training as shared
from tdn.analysis.response.training_control import control_config, learning_rate, is_head, whole_history_summary, sampled_curve_rows, reporting_control
from tdn.analysis.frontier.core import clean
from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry
from tdn.research.experiment import atomic_torch_save
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json, software_metadata
from tdn.runtime.precision import reference_precision
from .models import create_model
from .tangent import validate_target
from .contracts import physical_metrics

SCHEMA = "tdn.tangent-response-training/v1"


def equation_geometry(sample):
    record = sample["record"]
    return sample.get("eq", Physics(record["kappa"], record["reaction_rate"])), sample.get("geom", Geometry(tuple(record["grid"]), tuple(record["lengths"])))


def _parent(sample):
    return str(sample["record"]["parent_id"])


def _teacher(sample):
    return sample.get("teacher64", sample.get("target"))


def _anchor(sample):
    return sample.get("smooth_anchor64", sample.get("anchor_fp64"))


def _target(sample, config):
    name = config.get("endpoint_target", "oracle_tangent")
    if name == "oracle_tangent":
        return sample.get("oracle_endpoint64")
    if name == "teacher":
        return _teacher(sample)
    raise ValueError("Declare oracle_tangent or a separately named teacher endpoint target")


def _accepted(sample, config):
    record = sample["record"]
    reference = bool(sample.get("reference_accepted", record.get("reference", {}).get("accepted", False)))
    oracle = sample.get("oracle_metadata", {}).get("accepted", False)
    return reference and bool(oracle) and _target(sample, config) is not None and _teacher(sample) is not None and _anchor(sample) is not None


def _mse(value, sample):
    _, geom = equation_geometry(sample)
    return n.inner(value.double(), value.double(), geom, sample["record"]["target_kind"]).mean()


def _scope(samples):
    grouped = defaultdict(set)
    shapes = defaultdict(set)
    keys = set()
    for sample in samples:
        record = sample["record"]
        if record["split"] not in ("train", "validation"):
            raise ValueError("Fit and checkpoint selection must not access diagnostic/test teachers")
        validate_target(record["target_kind"])
        eq, geom = equation_geometry(sample)
        if tuple(geom.grid) != tuple(record["grid"]) or tuple(geom.lengths) != tuple(record["lengths"]):
            raise ValueError("Prepared sample geometry disagrees with its sealed record")
        for value,key in ((eq.kappa, "kappa"), (eq.reaction_rate, "reaction_rate")):
            if torch.is_tensor(value): value = float(value.detach())
            if float(value) != float(record[key]): raise ValueError("Prepared sample physics disagrees with its sealed record")
        if sample["u0"].shape != (1, 1, *tuple(geom.grid)):
            raise ValueError("Prepared fit rows identify one parent and one full-field geometry")
        grouped[_parent(sample)].add(record["split"])
        shapes[record.get("shape_sha256", _parent(sample))].add(record["split"])
        key = (record["split"], record.get("query_id", digest(record)))
        if key in keys:
            raise ValueError("Duplicate prepared fit query")
        keys.add(key)
        for field in (sample["u0"], _teacher(sample), _anchor(sample), sample.get("oracle_endpoint64")):
            if field is not None and not bool(torch.isfinite(field).all()):
                raise ValueError("Nonfinite prepared fitting field")
            if field is not None and field.shape != sample["u0"].shape:
                raise ValueError("Prepared fitting fields have inconsistent geometry")
        if sample["u0"].dtype != torch.float32 or any(value is not None and value.dtype != torch.float64
                for value in (_teacher(sample), _anchor(sample), sample.get("oracle_endpoint64"), sample.get("oracle_correction64"))):
            raise ValueError("Actual FP32 inputs and FP64 teacher/smooth anchor labels are required")
        for label in sample.get("action_labels", []):
            for name in ("source", "target_action"):
                value = label.get(name)
                if value is not None and (value.dtype != torch.float64 or not bool(torch.isfinite(value).all())):
                    raise ValueError("Action sources and targets must remain finite FP64")
    if any(len(value) > 1 for value in (*grouped.values(), *shapes.values())):
        raise ValueError("Parent/shape leakage between fitting and validation")


def _plan(samples):
    parents = defaultdict(list)
    for index, sample in enumerate(samples):
        parents[_parent(sample)].append(index)
    return [(index, 1./(len(parents)*len(parents[parent]))) for parent in sorted(parents)
            for index in sorted(parents[parent], key=lambda i:str(samples[i]["record"].get("query_id", i)))]


def _data_hash(samples):
    tensors = {}
    records = []
    for index, sample in enumerate(samples):
        records.append(sample["record"])
        records.append(dict(oracle_metadata=sample.get("oracle_metadata"), reference_accepted=sample.get("reference_accepted")))
        for key in ("u0", "teacher64", "target", "smooth_anchor64", "anchor_fp64", "deployed_anchor32", "anchor_fp32", "oracle_endpoint64", "oracle_correction64"):
            if torch.is_tensor(sample.get(key)):
                tensors[f"{index}/{key}"] = sample[key]
        for number, label in enumerate(sample.get("action_labels", [])):
            for field in ("source", "target_action"):
                if torch.is_tensor(label.get(field)): tensors[f"{index}/action/{number}/{field}"] = label[field]
            records.append({key:value for key,value in label.items() if not torch.is_tensor(value)})
    return digest(dict(records=clean(records), tensor_sha256=shared._state_hash(tensors)))


def normalization(samples, config):
    """Frozen input/correction/action scales from accepted training parents only."""
    _scope(samples)
    train = [sample for sample in samples if sample["record"]["split"] == "train" and _accepted(sample, config)]
    if not train:
        raise ValueError("No accepted training labels")
    means = [(_parent(sample), float(sample["u0"].double().mean())) for sample in train]
    full_mean = shared._parent_mean(means)
    full, mean, centered, actions = [], [], [], []
    for sample in train:
        parent = _parent(sample)
        full.append((parent, float(_mse(sample["u0"].double()-full_mean, sample).sqrt())))
        defect = _target(sample, config)-_anchor(sample)
        mean.append((parent, abs(float(defect.mean()))))
        centered.append((parent, float(_mse(defect-defect.mean(), sample).sqrt())))
        for label in sample.get("action_labels", []):
            if label.get("accepted", False):
                actions.append((parent, float(_mse(label["target_action"], sample).sqrt())))
    floor = float(config.get("normalization_floor", 1e-8))
    return dict(full_mean=full_mean, full_scale=max(floor, shared._weighted_quantile(full, .75)),
        mean_scale=max(floor, shared._weighted_quantile(mean, .75)), centered_scale=max(floor, shared._weighted_quantile(centered, .75)),
        action_scale=max(floor, shared._weighted_quantile(actions, .75)) if actions else 1., floor=floor,
        estimator="training-only parent-weighted 75th percentiles; input full scale and declared target physical-correction channels",
        train_parent_ids=sorted({_parent(sample) for sample in train}), endpoint_target=config.get("endpoint_target", "oracle_tangent"))


def _gain(model, train, norm, config, budget):
    """One robust frozen gain for the complete declared TRAIN action workload."""
    train = [sample for sample in train if sample["record"]["split"] == "train" and _accepted(sample, config)]
    if model.family == "residual_fno":
        adapted = [{**sample, "target":_target(sample, config), "anchor_fp32":sample.get("deployed_anchor32", sample.get("anchor_fp32")),
            "anchor_fp64":_anchor(sample)} for sample in train]
        return shared.fit_correction_gains(adapted, norm, {"family":"residual_fno", "model":model.config})
    if not train:
        raise ValueError("Response gain fitting requires accepted training queries")
    floor = max(1e-8, float(norm.get("floor", 1e-8)))
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("Complete-response gain floor must be positive and finite")
    rows = []; work = dict(gain_encoder_calls=0, gain_action_basis_calls=0, gain_physical_norm_calls=0)
    with torch.no_grad():
        for index, query_weight in _plan(train):
            sample = train[index]
            budget.check()
            labels = [label for label in sample.get("action_labels", [])
                if label.get("accepted", False) and label.get("role", "fit") == "fit"]
            if not labels:
                raise ValueError("Response gain fitting requires declared accepted fit source probes in every training query")
            eq, geom = equation_geometry(sample)
            encoded = model.encode(sample["u0"], eq, geom, sample["record"]["target_kind"], budget=budget)
            work["gain_encoder_calls"] += 1
            for label in labels:
                budget.check()
                # Undo only an existing *global* frozen gain to measure the
                # complete causal basis before normalization. Source, gate,
                # time and every temporal channel remain in the measured field.
                design = model.action_basis(encoded, label["source"], label["s"], label["h"])*model.response_gain
                work["gain_action_basis_calls"] += 1
                fields = design.reshape(-1, 1, *geom.grid)
                gain = float(n.inner(fields, fields, geom, sample["record"]["target_kind"]).mean().sqrt())
                work["gain_physical_norm_calls"] += 1
                if not math.isfinite(gain):
                    raise FloatingPointError("Nonfinite complete causal response basis gain")
                rows.append(dict(parent_id=_parent(sample), query_id=sample["record"].get("query_id"),
                    probe_id=label.get("probe_id"), source_time=float(label["s"]), horizon=float(label["h"]),
                    physical_basis_rms=gain, weight=query_weight/len(labels)))
    # The objective gives every parent equal total weight, every query within
    # that parent equal weight, and every declared probe within a query equal
    # weight. Use the identical hierarchy for the robust gain estimator.
    ordered = sorted(rows, key=lambda row:row["physical_basis_rms"])
    threshold = .75*math.fsum(row["weight"] for row in ordered)
    cumulative = 0.; quantile = ordered[-1]["physical_basis_rms"]
    for row in ordered:
        cumulative += row["weight"]
        if cumulative >= threshold:
            quantile = row["physical_basis_rms"]
            break
    fitted = max(floor, quantile)
    return {**norm, "response_gain":fitted,
        "gain_contract":"Frozen parent/query/probe-weighted 75th percentile of complete causal decoded TRAIN fit-action basis physical RMS; no per-example time/gate/source division",
        "response_gain_fit":dict(schema="tdn.tangent-response-gain-fit/v1", estimator="weighted_physical_RMS_quantile",
            quantile=.75, minimum_gain=floor, unclamped_quantile=quantile, frozen_gain=fitted,
            weighting="equal parent; equal query within parent; equal accepted declared fit probe within query",
            scope="accepted TRAIN current input/physics/geometry and predeclared fit source/time pairs only; no validation or target_action values",
            training_parent_ids=sorted({_parent(sample) for sample in train}), queries=len(train), probes=len(rows),
            queried_probe_ids=[row["probe_id"] for row in rows], rows=rows, work=work)}


class Deadline:
    def __init__(self, started, previous, total, chunk, external=None):
        self.started, self.previous, self.total, self.chunk, self.external = started, previous, total, chunk, external
    def check(self):
        if self.external is not None:
            self.external.check()
        elapsed = time.perf_counter()-self.started
        if self.previous+elapsed >= self.total or elapsed >= self.chunk:
            raise TimeoutError("Declared tangent fitting work/time allowance exhausted")


def _components(model, sample, config, norm, phase, deadline):
    eq, geom = equation_geometry(sample)
    encoded = model.encode(sample["u0"], eq, geom, sample["record"]["target_kind"], budget=deadline)
    completed_actions = 0
    try:
        if phase == "action":
            labels = sample.get("action_labels", [])
            if not labels or not all(label.get("accepted", False) for label in labels):
                raise ValueError("Missing or unresolved action labels cannot disappear from the fit denominator")
            errors = []
            for label in labels:
                deadline.check()
                errors.append(model.response_action(encoded, label["source"], label["s"], label["h"])-label["target_action"])
                completed_actions += 1
            mean = torch.stack([.5*n.mean(error).square().mean()/norm["action_scale"]**2 for error in errors]).mean()
            centered = torch.stack([.5*n.inner(n.center(error), n.center(error), geom, sample["record"]["target_kind"]).mean()/norm["action_scale"]**2 for error in errors]).mean()
            return mean, centered, encoded.path.work(), len(labels)
        parts = model.prediction_parts(encoded, sample["record"]["horizon"])
        error = parts["output"].double()-_target(sample, config)
        mean = .5*n.mean(error).square().mean()
        centered = .5*n.inner(n.center(error), n.center(error), geom, sample["record"]["target_kind"]).mean()
        return mean, centered, parts["work"], 0
    except (TimeoutError, InterruptedError) as exc:
        exc.partial_query_work = encoded.path.work()
        exc.partial_action_calls = completed_actions
        raise


def evaluate(model, samples, config, budget=None):
    was_training = model.training; model.eval()
    rows = []
    try:
        with torch.no_grad():
            for sample in samples:
                if budget is not None: budget.check()
                eq, geom = equation_geometry(sample)
                parts = model.prediction_parts(model.encode(sample["u0"], eq, geom, sample["record"]["target_kind"], budget=budget), sample["record"]["horizon"])
                error = parts["output"].double()-_target(sample, config)
                teacher_error = parts["output"].double()-_teacher(sample)
                correction = sample.get("oracle_correction64", sample["oracle_endpoint64"]-_anchor(sample))
                reconstruction = parts["physical_correction"]-correction
                denominator = float(_mse(correction, sample).sqrt())
                relative = float(_mse(reconstruction, sample).sqrt())/denominator if denominator > float(config.get("relative_floor", 1e-8)) else None
                physical = physical_metrics(error, geom, sample["record"]["target_kind"], maximum_oversampling=int(config.get("maximum_oversampling", 4)))
                teacher_physical = physical_metrics(teacher_error, geom, sample["record"]["target_kind"], maximum_oversampling=int(config.get("maximum_oversampling", 4)))
                row = dict(parent_id=_parent(sample), query_id=sample["record"].get("query_id"), split=sample["record"]["split"],
                    target_kind=sample["record"]["target_kind"], horizon=sample["record"]["horizon"], grid=sample["record"]["grid"],
                    rms=physical["rms"], mse=physical["mse"], maximum=physical["maximum"],
                    maximum_kind=physical["maximum_kind"], maximum_grid=physical["maximum_grid"],
                    mean_error=float(error.mean()), centered_rms=float(_mse(n.center(error), sample).sqrt()),
                    teacher_rms=teacher_physical["rms"], teacher_maximum=teacher_physical["maximum"],
                    relative_oracle_correction_error=relative, oracle_correction_rms=denominator,
                    physical_reconstruction_rms=float(_mse(reconstruction, sample).sqrt()),
                    arithmetic_bridge_rms=float(_mse(parts["arithmetic_bridge"], sample).sqrt()),
                    assembly_difference_rms=float(_mse(parts["assembly_difference"], sample).sqrt()),
                    final_rounding_rms=float(_mse(parts["final_rounding"], sample).sqrt()),
                    finite=bool(torch.isfinite(parts["output"]).all()), reference_accepted=_accepted(sample, config),
                    oracle_accepted=bool(sample.get("oracle_metadata", {}).get("accepted", False)),
                    oracle_uncertainty_rms=sample.get("oracle_metadata", {}).get("uncertainty_rms"),
                    reference_uncertainty_rms=sample["record"].get("reference", {}).get("uncertainty_rms"),
                    endpoint_target=config.get("endpoint_target", "oracle_tangent"), work=parts["work"])
                row["physical_correction_rms"] = float(_mse(parts["physical_correction"], sample).sqrt())
                row["correction_diagnostics"] = {key:float(value.detach()) if torch.is_tensor(value) else value
                    for key,value in parts.get("diagnostics", {}).items()}
                rows.append(row)
    finally:
        model.train(was_training)
    return dict(rows=rows, queries=len(rows), parents=len({_parent(sample) for sample in samples}),
        mse=shared._parent_mean([(row["parent_id"], row["mse"]) for row in rows]),
        rms=shared._parent_mean([(row["parent_id"], row["rms"]) for row in rows]),
        maximum=max((row["maximum"] for row in rows), default=None),
        reconstruction_passed_queries=sum(row["relative_oracle_correction_error"] is not None and row["relative_oracle_correction_error"] <= .05 and row["finite"] for row in rows),
        reconstruction_requested_queries=len(samples))


def _unavailable(reason):
    return dict(rows=[], queries=0, parents=0, mse=None, rms=None, maximum=None, unavailable_reason=reason,
                reconstruction_passed_queries=0, reconstruction_requested_queries=0)


def _criterion(value):
    return (-value["reconstruction_passed_queries"], max((row["relative_oracle_correction_error"] if row["relative_oracle_correction_error"] is not None else float("inf") for row in value["rows"]), default=float("inf")), value["mse"])


def _history_summary(history):
    summary = whole_history_summary(history)
    rows = [row for row in history if row.get("update", 0) > 0]
    for name in ("mean_gradient_norm", "centered_gradient_norm", "mean_loss", "centered_loss", "action_loss"):
        values = [(row["update"], float(row[name])) for row in rows
            if isinstance(row.get(name), (int, float)) and math.isfinite(row[name])]
        summary["metrics"][name] = dict(count=len(values), missing_count=len(rows)-len(values),
            minimum=min((value for _,value in values), default=None), maximum=max((value for _,value in values), default=None),
            maximum_update=max(values, key=lambda pair:pair[1])[0] if values else None,
            mean=math.fsum(value/len(values) for _,value in values) if values else None)
    return summary


def _emit_live(row, phase):
    from .report import emit_live
    names = ("train_loss", "mean_loss", "centered_loss", "action_loss", "endpoint_loss",
        "validation_loss", "validation_rms", "validation_maximum", "full_train_mse", "full_train_rms", "full_train_maximum",
        "mean_gradient_norm", "centered_gradient_norm", "gradient_norm", "optimizer_gradient_norm", "gradient_norm_after_clip",
        "head_gradient_norm", "trunk_gradient_norm", "learning_rate", "query_visits", "examples_seen", "parameter_movement", "trunk_movement")
    metrics = {name:float(row[name]) for name in names
        if isinstance(row.get(name), (int, float)) and math.isfinite(row[name])}
    if "clipped" in row: metrics["clipped"] = int(row["clipped"])
    emit_live(metrics, phase=phase, step=row.get("update", 0))


def load_model(path, device="cpu"):
    reference_precision()
    path = Path(path)
    saved = torch.load(path/"selected.pt" if path.is_dir() else path, weights_only=True, map_location="cpu")
    model = create_model({**saved["model_config"], "normalization":saved["normalization"]}).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved["state_dict"]); model.eval()
    return model, {key:value for key,value in saved.items() if key != "state_dict"}


def conditional_readout_fit(samples, model_config, *, initial_state=None, budget=None, maximum_queries=64):
    """One bounded train-only affine solve with frozen initial spatial factors."""
    from .compression import conditional_fit, field_metrics
    train = sorted([sample for sample in samples if sample["record"]["split"] == "train"],
                   key=lambda sample:(_parent(sample), str(sample["record"].get("query_id"))))
    selected = train[:int(maximum_queries)]
    if not selected or int(maximum_queries) <= 0:
        return dict(status="NA", reason="No declared training-query opportunity")
    geometries = {(tuple(sample["record"]["grid"]), tuple(sample["record"]["lengths"]), sample["record"]["target_kind"]) for sample in selected}
    if len(geometries) != 1:
        return dict(status="NA", reason="This bounded conditional solve requires a single geometry; mixed-grid rows remain unfit",
                    requested_queries=len(train), fitted_queries=0)
    device = selected[0]["u0"].device
    config = {**model_config, "frozen_encoder":True}
    model = create_model(config).to(device=device, dtype=torch.float32)
    if initial_state is not None: model.load_state_dict(initial_state)
    model.eval(); designs, residuals, bases = [], [], []
    try:
        with torch.no_grad():
            for sample in selected:
                if budget is not None: budget.check()
                eq, geom = equation_geometry(sample)
                encoded = model.encode(sample["u0"], eq, geom, sample["record"]["target_kind"], budget=budget)
                base, design = model.affine_endpoint_design(encoded, sample["record"]["horizon"])
                bases.append(base); designs.append(design); residuals.append(sample["oracle_endpoint64"]-base)
            weights = [weight for _,weight in _plan(selected)]
            solve = conditional_fit(torch.stack(designs, dim=1), torch.stack(residuals), geom,
                selected[0]["record"]["target_kind"], weights=weights,
                ridge=float(config.get("readout_ridge", 0.)), rcond=float(config.get("readout_rcond", 1e-10)))
            model.set_affine_coefficients(solve["coefficients"])
            deployed = []
            for sample in selected:
                if budget is not None: budget.check()
                eq, geom = equation_geometry(sample)
                prediction = model(sample["u0"], sample["record"]["horizon"], eq, geom, sample["record"]["target_kind"], budget=budget)
                deployed.append(dict(query_id=sample["record"].get("query_id"), parent_id=_parent(sample),
                    **field_metrics(prediction.double()-sample["oracle_endpoint64"], geom, sample["record"]["target_kind"])))
    except (TimeoutError, InterruptedError) as exc:
        return dict(status="UNRESOLVED", reason="Bounded conditional solve incomplete: "+str(exc),
                    requested_queries=len(train), fitted_queries=len(designs))
    scalar_solve = {key:value for key,value in solve.items() if not torch.is_tensor(value)}
    scalar_solve["coefficients"] = solve["coefficients"].detach().cpu().tolist()
    return clean(dict(status="COMPLETED", control="frozen initial encoder conditional affine readout", affine=True,
        global_nonlinear_optimum=False, variable_projection="exact linear subproblem conditional on fixed factors/path; not installed in trained or selected model",
        requested_queries=len(train), fitted_queries=len(selected), training_parent_ids=sorted({_parent(sample) for sample in selected}),
        query_selection="first declared canonical training queries, bounded before accuracy inspection",
        solve=scalar_solve, deployed_rows=deployed, target_access="training-only oracle endpoint; no validation fit"))


def fit(samples, config, *, device="cpu", output_path, resume_from=None, chunk_seconds=None, initial_state=None, budget=None, max_chunk_updates=None):
    """Execute bounded logical work; continuation preserves every committed state."""
    started = time.perf_counter(); reference_precision()
    config = json.loads(json.dumps(clean(config), allow_nan=False)); _scope(samples)
    samples = [{**sample, "action_labels":[dict(label) for label in sample.get("action_labels", [])]} for sample in samples]
    family = config.get("family", config.get("arm", "learned_response"))
    seed, updates = int(config.get("seed", 0)), int(config.get("updates", 300))
    root = Path(output_path); root.mkdir(parents=True, exist_ok=True)
    requested = [sample for sample in samples if sample["record"]["split"] == "train"]
    validation_requested = [sample for sample in samples if sample["record"]["split"] == "validation"]
    missing = [sample["record"].get("query_id") for sample in samples if not _accepted(sample, config)]
    action_missing = [sample["record"].get("query_id") for sample in samples if family == "learned_response" and
        (not sample.get("action_labels") or any(not label.get("accepted", False) or label.get("source") is None or label.get("target_action") is None for label in sample.get("action_labels", [])))]
    if not requested or not validation_requested or missing or action_missing:
        result = dict(schema=SCHEMA, trial_id=config.get("id", family), family=family, status="UNRESOLVED_REFERENCE" if missing else "UNRESOLVED_ACTION_LABELS",
            requested_updates=updates, updates=0, requested_training_queries=len(requested), requested_validation_queries=len(validation_requested),
            missing_queries=missing, missing_action_queries=action_missing, selected_train=_unavailable("Required reference/oracle/action coverage missing"), selected_validation=_unavailable("Required coverage missing"))
        write_json(root/"trial.json", result); write_json(root/"training_results.json", dict(schema=SCHEMA, trials=[result])); return result
    train, validation = requested, validation_requested
    train = sorted(train, key=lambda sample:(_parent(sample), str(sample["record"].get("query_id"))))
    validation = sorted(validation, key=lambda sample:(_parent(sample), str(sample["record"].get("query_id"))))
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA fitting cannot silently fall back to CPU")
    for sample in samples:
        for key,value in sample.items():
            if torch.is_tensor(value): sample[key] = value.to(device)
        for label in sample.get("action_labels", []):
            for key,value in label.items():
                if torch.is_tensor(value): label[key] = value.to(device)
    norm = normalization(samples, config)
    model_config = {**config.get("model_config", config.get("model", {})), "family":family, "normalization":norm}
    if family == "residual_fno":
        model_config["normalization"].update(mean_gain=1., centered_gain=1.)
        model_config["gain_normalization"] = "frozen"
    torch.manual_seed(seed); np.random.seed(seed%(2**32)); random.seed(seed)
    model = create_model(model_config).to(device=device, dtype=torch.float32)
    if initial_state is not None: model.load_state_dict(initial_state)
    setting = dict(lr=float(config.get("lr", .001)), eps=float(config.get("eps", 1e-12)), weight_decay=0.,
                   betas=[.9, .999], clip=config.get("clip"), loss_multiplier=float(config.get("loss_multiplier", 1.)))
    recipe = {**config, "updates":updates, "setting":setting, "batch_mode":"full_parent",
        "lr_schedule":config.get("lr_schedule", dict(kind="cosine", minimum_ratio=.05, warmup_updates=min(5, max(0, updates-1)))),
        "head_warmup_updates":int(config.get("head_warmup_updates", min(5, max(0, updates-1))))}
    controls = control_config(recipe); reporting = reporting_control(recipe)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=setting["lr"], eps=setting["eps"], betas=tuple(setting["betas"]), weight_decay=0.)
    runtime = software_metadata()
    identity = dict(config=config, data=_data_hash(samples), source=config.get("source_tree_sha256", runtime.get("source_tree_sha256")),
        software={key:runtime.get(key) for key in ("python", "torch", "numpy", "scipy", "torch_cuda_runtime")}, device=str(device))
    identity = json.loads(json.dumps(clean(identity), allow_nan=False))
    resume = Path(resume_from) if resume_from else None
    if resume is not None and resume.is_dir(): resume = resume/"journal.pt"
    saved = torch.load(resume, weights_only=True, map_location="cpu") if resume else None
    if saved and saved["identity"] != identity:
        raise ValueError("Exact continuation data/config/source/software/device identity changed")
    sampler = random.Random(seed)
    if saved:
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        norm, model_config = saved["normalization"], saved["model_config"]
        completed, history, initial = saved["update"], saved["history"], saved["initial_state"]
        selected_state, criterion_state = saved["selected_state"], saved["criterion_state"]
        initial_train, initial_validation, best, criterion = saved["initial_train"], saved["initial_validation"], saved["best"], saved["criterion"]
        selected_update, criterion_update = saved["selected_update"], saved["criterion_update"]
        cumulative, work, pending = saved["cost"], defaultdict(int, saved["work"]), saved["pending_validation"]
        action_state = saved.get("action_state")
        attempts = saved["attempts"]
        shared._restore_rng(saved["rng"], sampler)
    else:
        completed, history = 0, []
        initial = copy.deepcopy(model.state_dict()); selected_state = copy.deepcopy(initial); criterion_state = copy.deepcopy(initial)
        initial_train = initial_validation = best = criterion = None
        selected_update = criterion_update = 0; pending = False; attempts = []
        cumulative = dict(total_seconds=0., optimization_seconds=0., validation_seconds=0., diagnostic_seconds=0., checkpoint_seconds=0.)
        work = defaultdict(int)
        action_state = None
    previous_seconds = cumulative["total_seconds"]
    deadline = Deadline(started, previous_seconds, float(config.get("total_seconds", float("inf"))), float(chunk_seconds or float("inf")), budget)
    if not saved:
        try:
            norm = _gain(model, train, norm, config, deadline)
        except (TimeoutError, InterruptedError) as exc:
            result = dict(schema=SCHEMA, trial_id=config.get("id", family), family=family, seed=seed,
                status="TIME_CENSORED" if time.perf_counter()-started >= float(config.get("total_seconds", float("inf"))) else "UNRESOLVED_INITIALIZATION",
                updates=0, requested_updates=updates, failure=dict(type=type(exc).__name__, message=str(exc), scientific_failure=False),
                selected_train=_unavailable("Train-only gain preparation incomplete"), selected_validation=_unavailable("Gain preparation incomplete"),
                cost=dict(total_seconds=time.perf_counter()-started), requested_training_queries=len(train), requested_validation_queries=len(validation),
                normalization=norm, initialization_restart_required=True)
            write_json(root/"trial.json", clean(result)); write_json(root/"training_results.json", clean(dict(schema=SCHEMA, trials=[result])))
            return result
        for key,value in norm.get("response_gain_fit", {}).get("work", {}).items():
            work[key] += value
        model_config["normalization"] = norm
        # Recreate the same initial weights, with only frozen normalization changed.
        torch.manual_seed(seed); model = create_model(model_config).to(device=device, dtype=torch.float32)
        if initial_state is not None: model.load_state_dict(initial_state)
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=setting["lr"], eps=setting["eps"], betas=tuple(setting["betas"]), weight_decay=0.)
        initial = copy.deepcopy(model.state_dict()); selected_state = copy.deepcopy(initial); criterion_state = copy.deepcopy(initial)
    # Exact immutable plan is stored once, not repeated in every commit.
    plan = _plan(train)
    plan_payload = [{"parent_id":_parent(train[index]), "query_id":train[index]["record"].get("query_id"), "weight":weight,
                     "action_probe_ids":[label.get("probe_id") for label in train[index].get("action_labels", [])]} for index,weight in plan]
    plan_hash = digest(plan_payload); write_json(root/"query_plan.json", dict(schema="tdn.tangent-response-query-plan/v1", plan=plan_payload, sha256=plan_hash))
    action_updates = int(config.get("action_updates", round(updates*float(config.get("action_fraction", .5))))) if family == "learned_response" else 0
    if not 0 <= action_updates <= updates: raise ValueError("Action opportunity must lie within total committed work")
    validation_every = int(config.get("validation_every", 40)); validation_indices = set(range(validation_every, updates+1, validation_every)) | {updates, action_updates}
    attempt = dict(index=len(attempts), start_update=completed, started_unix=time.time(), status="RUNNING", start_work=dict(work))
    attempts = attempts+[attempt]; start_update = completed

    def checkpoint():
        tick = time.perf_counter(); cumulative["total_seconds"] = previous_seconds+time.perf_counter()-started
        atomic_torch_save(dict(schema=SCHEMA, identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
            initial_state=initial, selected_state=selected_state, criterion_state=criterion_state,
            action_state=action_state,
            update=completed, history=history, initial_train=initial_train, initial_validation=initial_validation,
            best=best, criterion=criterion, selected_update=selected_update, criterion_update=criterion_update,
            normalization=norm, model_config=model_config, cost=cumulative, work=dict(work), attempts=attempts,
            pending_validation=pending, rng=shared._rng_state(sampler),
            scheduler_state=dict(completed_updates=completed, next_learning_rate=learning_rate(controls, completed))), root/"journal.pt")
        cumulative["checkpoint_seconds"] += time.perf_counter()-tick

    def validate():
        nonlocal best, criterion, selected_state, criterion_state, selected_update, criterion_update, pending
        tick = time.perf_counter(); value = evaluate(model, validation, config, deadline)
        cumulative["validation_seconds"] += time.perf_counter()-tick
        if not math.isfinite(value["mse"]): raise FloatingPointError("Nonfinite validation loss")
        if best is None or (value["mse"], value["maximum"]) < (best["mse"], best["maximum"]):
            best, selected_state, selected_update = value, copy.deepcopy(model.state_dict()), completed
        tick = time.perf_counter(); fit_value = evaluate(model, train, config, deadline)
        cumulative["diagnostic_seconds"] += time.perf_counter()-tick
        if criterion is None or _criterion(fit_value) < _criterion(criterion):
            criterion, criterion_state, criterion_update = fit_value, copy.deepcopy(model.state_dict()), completed
        history[-1].update(validation_loss=value["mse"], validation_rms=value["rms"], validation_maximum=value["maximum"],
            full_train_mse=fit_value["mse"], full_train_rms=fit_value["rms"], full_train_maximum=fit_value["maximum"])
        pending = False; checkpoint()
        _emit_live(history[-1], "validation")

    status, failure, active_rng, active_tick = "CONTINUATION_REQUIRED", None, None, None
    try:
        if initial_train is None:
            initial_train = evaluate(model, train, config, deadline)
            history.append(dict(update=0, initialization=True, train_loss=initial_train["mse"], train_rms=initial_train["rms"],
                full_train_mse=initial_train["mse"], full_train_rms=initial_train["rms"], full_train_maximum=initial_train["maximum"], query_plan_sha256=plan_hash))
            initial_validation = evaluate(model, validation, config, deadline)
            best, criterion = initial_validation, initial_train
            history[-1].update(validation_loss=best["mse"], validation_rms=best["rms"], validation_maximum=best["maximum"])
            atomic_torch_save(dict(schema=SCHEMA, state_dict=initial, model_config=model_config, normalization=norm, identity=identity), root/"initial.pt")
            checkpoint()
            _emit_live(history[-1], "initialization")
        if pending: validate()
        while completed < updates:
            if max_chunk_updates is not None and completed-start_update >= int(max_chunk_updates): break
            deadline.check(); active_rng = shared._rng_state(sampler)
            phase = "action" if completed < action_updates else "endpoint"
            rate = learning_rate(controls, completed)
            for group in optimizer.param_groups: group["lr"] = rate
            optimizer.zero_grad(set_to_none=True); model.train()
            shared._sync(device); tick = time.perf_counter(); active_tick = tick; losses = [0., 0.]
            component_gradients = [[torch.zeros_like(value) for value in parameters] for _ in range(2)]
            for index, weight in plan:
                deadline.check(); sample = train[index]
                work["query_attempts"] += 1
                mean_loss, centered_loss, query_work, action_calls = _components(model, sample, config, norm, phase, deadline)
                for component_index, loss in enumerate((mean_loss, centered_loss)):
                    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                    work["component_gradient_calls"] += 1
                    for total, gradient in zip(component_gradients[component_index], gradients):
                        if gradient is not None: total.add_(gradient.detach(), alpha=weight)
                    losses[component_index] += weight*float(loss.detach())
                loss = (mean_loss+centered_loss)*(weight*setting["loss_multiplier"])
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite staged fitting loss")
                loss.backward(); work["optimizer_backward_calls"] += 1
                work["response_action_calls"] += action_calls
                for key,value in query_work.items():
                    if isinstance(value, int) and not isinstance(value, bool) and key not in ("reaction_substeps", "source_nodes", "mean_nodes"):
                        work[key] += value
            pre = shared._gradient_norm(model)
            if not math.isfinite(pre): raise FloatingPointError("Nonfinite staged fitting gradient")
            head, trunk = shared._gradient_norm(model, head=True), shared._gradient_norm(model, head=False)
            head_only = completed < controls["head_warmup_updates"]
            if head_only:
                for name,value in model.named_parameters():
                    if not is_head(name): value.grad = None
            active = shared._gradient_norm(model)
            if setting["clip"] is not None: torch.nn.utils.clip_grad_norm_(parameters, float(setting["clip"]), error_if_nonfinite=True)
            pre_step_model, pre_step_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
            optimizer.step(); work["optimizer_step_attempts"] += 1
            if not all(bool(torch.isfinite(value).all()) for value in parameters):
                model.load_state_dict(pre_step_model); optimizer.load_state_dict(pre_step_optimizer)
                raise FloatingPointError("Nonfinite optimizer iterate; retained last committed state")
            completed += 1; active_rng = None
            if completed == action_updates: action_state = copy.deepcopy(model.state_dict())
            work["committed_optimizer_updates"] += 1; work["committed_query_visits"] += len(plan)
            shared._sync(device); cumulative["optimization_seconds"] += time.perf_counter()-tick; active_tick = None
            gradients = [float(torch.stack([value.double().square().sum() for value in group]).sum().sqrt()) for group in component_gradients]
            history.append(dict(update=completed, training_phase=phase, train_loss=sum(losses), mean_loss=losses[0], centered_loss=losses[1],
                mean_gradient_norm=gradients[0], centered_gradient_norm=gradients[1], gradient_norm=pre, optimizer_gradient_norm=active,
                head_gradient_norm=head, trunk_gradient_norm=trunk, gradient_norm_after_clip=shared._gradient_norm(model),
                action_loss=sum(losses) if phase == "action" else None, endpoint_loss=sum(losses) if phase == "endpoint" else None,
                clipped=setting["clip"] is not None and active > float(setting["clip"]), head_only_warmup=head_only,
                learning_rate=rate, query_visits=len(plan), examples_seen=work["committed_query_visits"], query_plan_sha256=plan_hash,
                parameter_movement=shared._movement(model, initial), trunk_movement=shared._movement(model, initial, head=False)))
            pending = completed in validation_indices; checkpoint()
            if pending: validate()
            elif completed % reporting["live_every"] == 0: _emit_live(history[-1], phase)
            if completed == action_updates:
                atomic_torch_save(dict(schema=SCHEMA, state_dict=model.state_dict(), update=completed, identity=identity,
                    model_config=model_config, normalization=norm, checkpoint_role="completed action-fitting stage"), root/"action-stage.pt")
        status = "COMPLETED_UPDATES" if completed >= updates else "CONTINUATION_REQUIRED"
    except (TimeoutError, InterruptedError) as exc:
        status = "TIME_CENSORED" if previous_seconds+time.perf_counter()-started >= float(config.get("total_seconds", float("inf"))) else "CONTINUATION_REQUIRED"
        failure = dict(type=type(exc).__name__, message=str(exc), scientific_failure=False)
        if active_rng is not None:
            shared._restore_rng(active_rng, sampler); optimizer.zero_grad(set_to_none=True)
        for key,value in getattr(exc, "partial_query_work", {}).items():
            if isinstance(value, int) and not isinstance(value, bool) and key not in ("reaction_substeps", "source_nodes", "mean_nodes"):
                work[key] += value
        work["response_action_calls"] += getattr(exc, "partial_action_calls", 0)
    except FloatingPointError as exc:
        status = "FAILED_NONFINITE"; failure = dict(type=type(exc).__name__, message=str(exc), scientific_failure=True)
        if active_rng is not None:
            shared._restore_rng(active_rng, sampler); optimizer.zero_grad(set_to_none=True)
    finally:
        if active_tick is not None: cumulative["optimization_seconds"] += time.perf_counter()-active_tick
        attempt.update(status=status, end_update=completed, finished_unix=time.time(), elapsed_seconds=time.perf_counter()-started,
            end_work=dict(work), work={key:value-attempt["start_work"].get(key, 0) for key,value in work.items()})
        checkpoint()
        if history: _emit_live(history[-1], "pending_validation" if pending else "attempt_terminal")
    final_state = copy.deepcopy(model.state_dict())
    try: final_train = evaluate(model, train, config, deadline)
    except (TimeoutError, InterruptedError): final_train = _unavailable("Bounded final training observation incomplete")
    model.load_state_dict(selected_state)
    try: selected_train = evaluate(model, train, config, deadline)
    except (TimeoutError, InterruptedError): selected_train = _unavailable("Bounded selected training observation incomplete")
    checkpoint_base = dict(schema=SCHEMA, identity=identity, model_config=model_config, normalization=norm, family=family,
                           architecture_metadata=model.architecture_metadata())
    atomic_torch_save({**checkpoint_base, "state_dict":initial, "update":0, "checkpoint_role":"initialization"}, root/"initial.pt")
    if action_state is not None:
        atomic_torch_save({**checkpoint_base, "state_dict":action_state, "update":action_updates,
                          "checkpoint_role":"completed action fitting optimizer opportunity"}, root/"action-stage.pt")
    for name,state,index,role in (("selected.pt",selected_state,selected_update,"validation-selected deployment"),
                                 ("final.pt",final_state,completed,"last optimizer iterate"),
                                 ("criterion-diagnostic.pt",criterion_state,criterion_update,"training criterion diagnostic only")):
        atomic_torch_save({**checkpoint_base, "state_dict":state, "update":index, "checkpoint_role":role}, root/name)
    model.load_state_dict(final_state)
    control_tick = time.perf_counter()
    fixed = conditional_readout_fit(train, model_config, initial_state=initial, budget=deadline,
        maximum_queries=int(config.get("conditional_max_queries", 64))) if config.get("conditional_readout", True) and status == "COMPLETED_UPDATES" else dict(status="NA", reason="Trial incomplete or control not declared")
    fixed["elapsed_seconds"] = time.perf_counter()-control_tick
    write_json(root/"fixed_controls.json", clean(fixed))
    for row in history: row.update(trial_id=config.get("id", family), family=family, seed=seed)
    # Private journal retains every commit; public observations are bounded.
    if reporting["enabled"]:
        curves, _, export = sampled_curve_rows(history, {**reporting, "compact_public_query_plan":False},
            chunk_start_update=start_update, pending_validation=pending)
    else:
        curves = [row for row in history if row["update"] > start_update or start_update == 0]; export = None
    shared._jsonl(root/"learning_curves.jsonl", curves)
    cumulative["total_seconds"] = previous_seconds+time.perf_counter()-started
    attempt.update(finished_unix=time.time(), elapsed_seconds=time.perf_counter()-started)
    checkpoint()
    result = dict(schema=SCHEMA, trial_id=config.get("id", family), family=family, seed=seed, status=status, failure=failure,
        updates=completed, requested_updates=updates, action_updates=action_updates, endpoint_target=config.get("endpoint_target", "oracle_tangent"),
        selected_update=selected_update, criterion_update=criterion_update, selected_initialization=selected_update == 0,
        selection_rule="validation-only physical MSE against declared endpoint target, then sampled maximum; earlier exact ties retained",
        initial_train=initial_train or _unavailable("Initialization incomplete"), initial_validation=initial_validation or _unavailable("Initialization incomplete"),
        selected_train=selected_train, selected_validation=best or _unavailable("Validation incomplete"), final_train=final_train,
        criterion_train=criterion or _unavailable("Training criterion incomplete"), criterion_role="single training checkpoint maximizing declared reconstruction coverage; diagnostic only",
        normalization=norm, model_config=model_config, architecture_metadata=model.architecture_metadata(), total_parameters=sum(value.numel() for value in model.parameters()),
        initial_state_sha256=shared._state_hash(initial), selected_state_sha256=shared._state_hash(selected_state), final_state_sha256=shared._state_hash(final_state),
        selected_checkpoint_sha256=file_digest(root/"selected.pt"), journal_sha256=file_digest(root/"journal.pt"),
        cost=cumulative, work=dict(work), work_scope="All attempted optimization queries and committed updates; validation/path observations additionally retain per-query work",
        attempts=attempts, optimization_history_summary=_history_summary(history), learning_curve_export=export,
        requested_training_queries=len(train), requested_validation_queries=len(validation), accepted_training_queries=len(train), accepted_validation_queries=len(validation),
        training_parent_ids=sorted({_parent(sample) for sample in train}), validation_parent_ids=sorted({_parent(sample) for sample in validation}),
        additional_supervision="accepted response-action labels then oracle endpoint" if family == "learned_response" else "oracle endpoint only",
        conditional_control=fixed,
        query_plan_sha256=plan_hash, precision=dict(network="float32", path="float64", labels="float64", loss="float64", tf32=False))
    write_json(root/"attempt.json", attempt)
    write_json(root/"trial.json", clean(result)); write_json(root/"training_results.json", clean(dict(schema=SCHEMA, trials=[result])))
    return result

