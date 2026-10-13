"""Complete inference callbacks for the six declared Tangent Response controls.

Only initial state, known physics/geometry and requested time enter a call.
Numerical paths, residuals, quadrature and oracle response solves are recomputed
inside that call. Supplied teachers/action labels are never inference inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import time

import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.trajectory import mean_propagator
from tdn.analysis.roadmap.numerics import quadrature
from .compression import ResponseFactors
from .contracts import digest, validate_target
from .tangent import AnchorPath, refined_action, refined_forced


NAMES = ("anchor", "G", "U", "fixed_shared_basis", "learned_response", "residual_fno")


class UnresolvedInference(ValueError):
    """An unresolved bounded oracle is not an observed numerical prediction."""
    def __init__(self, reason, metadata=None, work=None):
        super().__init__(reason)
        self.metadata, self.work = dict(metadata or {}), dict(work or {})


@dataclass
class EndpointMethod:
    name: str
    callback: object
    metadata: dict
    last_work: dict = field(default_factory=dict)
    last_parts: dict = field(default_factory=dict)
    calls: int = 0

    def __call__(self, *args, **kwargs):
        tick = time.perf_counter(); self.calls += 1
        try:
            result = self.callback(*args, **kwargs)
            if not isinstance(result, dict) or not torch.is_tensor(result.get("output")):
                raise ValueError("Complete inference adapter requires a tensor output")
            self.last_parts = {key: value for key, value in result.items() if key not in ("output", "work")}
            self.last_work = dict(result.get("work", {}), status="COMPLETED", call_index=self.calls,
                                  dispatch_seconds=time.perf_counter()-tick,
                                  timing_scope="host dispatch; synchronized benchmark encloses the complete call")
            return result["output"]
        except UnresolvedInference as error:
            self.last_work = dict(error.work, status="UNRESOLVED", reason=str(error), call_index=self.calls,
                                  dispatch_seconds=time.perf_counter()-tick)
            self.last_parts = dict(oracle_metadata=error.metadata)
            raise
        except (TimeoutError, InterruptedError) as error:
            self.last_work = dict(status="UNRESOLVED", reason=str(error), call_index=self.calls,
                                  dispatch_seconds=time.perf_counter()-tick, component_work="Partial allocation-budget interruption")
            self.last_parts = {}
            raise UnresolvedInference(str(error), work=self.last_work) from error
        except (RuntimeError, ValueError, FloatingPointError) as error:
            self.last_work = dict(status="FAILED", reason=str(error), call_index=self.calls,
                                  dispatch_seconds=time.perf_counter()-tick)
            self.last_parts = {}
            raise


class MethodCollection(dict):
    @property
    def metadata(self):
        return {name: dict(method.metadata) for name, method in self.items()}

    def performance_methods(self, *, deployable_only=False):
        """Performance's structured method interface, with oracle attribution."""
        return {name: {**method.metadata, "call": method, "method": name, "schedule": "frozen"}
                for name, method in self.items() if not deployable_only or method.metadata.get("deployable", False)}


class _Deadline:
    def __init__(self, seconds, outer=None):
        self.stop = time.monotonic()+float(seconds); self.outer = outer
    def check(self):
        if self.outer is not None:
            self.outer.check()
        if time.monotonic() >= self.stop:
            raise TimeoutError("Declared complete oracle-call budget exhausted")


def _settings(protocol):
    anchor, training, mechanism = (dict(protocol.get(key, {})) for key in ("anchor", "training", "mechanism"))
    comp = dict(protocol.get("compression", {}))
    cfg = dict(anchor=anchor.get("name", "df"), reaction_substeps=int(anchor.get("reaction_substeps", 8)),
                source_nodes=int(training.get("source_nodes", comp.get("source_nodes", 8))),
                mean_nodes=int(training.get("mean_nodes", 8)),
                initial_steps=int(mechanism.get("initial_steps", 32)), max_steps=int(mechanism.get("max_steps", 4096)),
                tolerance=float(mechanism.get("tolerance", 1e-10)), seconds=float(mechanism.get("seconds_per_query", 240)),
                control_output_dtype=training.get("control_output_dtype", "float32"))
    if cfg["control_output_dtype"] not in ("float32", "float64"):
        raise ValueError("Declared control output precision must be FP32 or FP64")
    if not math.isfinite(cfg["seconds"]) or cfg["seconds"] <= 0:
        raise ValueError("Oracle inference requires a finite positive per-call budget")
    return cfg


def _validate(u0, h, eq, geom, target):
    validate_target(target); n.validate(u0, h, geom, target)
    if u0.dtype != torch.float32:
        raise ValueError("Complete deployment callbacks require the actual FP32 current state")


def _graph(u0, eq, h):
    return torch.is_grad_enabled() and any(torch.is_tensor(value) and value.requires_grad
               for value in (u0, eq.kappa, eq.reaction_rate, h))


def _path(u0, eq, geom, target, cfg, budget=None, h=0.):
    return AnchorPath(u0.double(), eq, geom, target, anchor_name=cfg["anchor"],
                      reaction_substeps=cfg["reaction_substeps"], create_graph=_graph(u0, eq, h), budget=budget)


def _mean(path, source, s, h, eq, geom, target, nodes):
    start, stop = n.validate(source, s, geom, target), n.validate(source, h, geom, target)
    if bool((stop < start).any()):
        raise ValueError("Reversed tangent intervals are invalid")
    value = mean_propagator(path.mean, start, stop, eq, source, nodes=nodes)*n.heat(source, stop-start, eq, geom, target)
    return torch.where(stop == start, source, value)


def _normals(u0, eq, geom, target):
    variance = n.inner(n.center(u0.double()), n.center(u0.double()), geom, target)
    return (n.scalar(eq.reaction_rate, u0) != 0) & (variance != 0)


def _factor_entries(compression_result, private_factors):
    supplied = private_factors
    if supplied is None and compression_result:
        supplied = compression_result.get("_factors", compression_result.get("private_factors"))
    entries = {}
    if isinstance(supplied, dict):
        records = [dict(context=identity[0], method=identity[1], parent_id=identity[2] if len(identity)>2 else None,
                        factors=value) for identity, value in supplied.items()]
    else:
        records = list(supplied or [])
    for record in records:
        if record.get("parent_id"):
            continue
        factor = record["factors"] if isinstance(record["factors"], ResponseFactors) else ResponseFactors.from_payload(record["factors"])
        if factor.metadata.get("state_oracle") or factor.metadata.get("deployable") is not True or factor.metadata.get("training_split") != "train":
            continue
        # Freeze factors independently of their mutable acquisition container.
        payload = factor.to_payload()
        for field in ("alpha", "beta", "temporal_beta"):
            if payload[field] is not None:
                payload[field] = payload[field].clone()
        factor = ResponseFactors.from_payload(payload)
        context = record["context"]
        key = (str(context[0]), tuple(context[1]), tuple(float(v) for v in context[2]))
        if key != (factor.target, factor.geometry.grid, factor.geometry.lengths):
            raise ValueError("Frozen factor payload geometry differs from its declared context")
        entries[key, str(record["method"])] = factor
    return entries


def _selection(protocol, compression_result):
    cfg = protocol.get("compression", {})
    ranks = cfg.get("ranks", [2, 4, 8])
    default_rank = 4 if 4 in ranks else min(ranks)
    chosen = cfg.get("fixed_shared_method", f"shared_response_action_pod_rank{default_rank}")
    receipt = dict(method=chosen, selection_scope="prospectively declared shared response-action POD control",
                   source="protocol", diagnostic_labels_used=False)
    frozen = (compression_result or {}).get("shared_selection")
    if frozen is not None:
        if frozen.get("status") != "FROZEN" or frozen.get("selection_scope") not in ("validation_only", "validation only"):
            raise ValueError("Shared response selection must be frozen using validation only")
        if not frozen.get("source_identity") or not frozen.get("method"):
            raise ValueError("Validation selection needs an explicit source identity and method")
        chosen = str(frozen["method"]); receipt = dict(frozen, diagnostic_labels_used=False)
    if not chosen.startswith("shared_") or "oracle" in chosen:
        raise ValueError("Only a train-fitted shared control can become a fixed inference method")
    receipt["method"] = chosen; receipt["sha256"] = digest(receipt)
    return chosen, receipt


def _on_device(factor, device):
    return ResponseFactors(factor.alpha.to(device), factor.beta.to(device), factor.geometry, factor.target,
                           dict(factor.metadata), factor.temporal_beta.to(device) if factor.temporal_beta is not None else None)


def _factor_for(entries, method, geom, target, device):
    factor = entries.get(((target, tuple(geom.grid), tuple(geom.lengths)), method))
    if factor is None:
        raise ValueError("UNAVAILABLE: no frozen train-only shared factors for this target/geometry")
    return _on_device(factor, device)


def _model_entry(value):
    if isinstance(value, dict):
        model, metadata = value.get("model"), dict(value.get("metadata", {}))
    elif isinstance(value, tuple) and len(value) == 2:
        model, metadata = value[0], dict(value[1])
    else:
        model, metadata = value, {}
    role = metadata.get("checkpoint_role")
    if role is not None and "validation" not in str(role).lower():
        raise ValueError("Deployment callbacks require validation-selected checkpoints, not training-fit/final diagnostics")
    if not callable(model):
        return None, metadata
    if callable(getattr(model, "eval", None)):
        model.eval()
    return model, metadata


def _versions(model):
    if not callable(getattr(model, "parameters", None)):
        return None
    tensors = list(model.parameters())+list(model.buffers())
    return tuple((id(value), value._version, str(value.device), str(value.dtype)) for value in tensors)


def _unavailable(name, reason):
    def call(*args, **kwargs):
        raise ValueError("UNAVAILABLE: "+reason)
    return EndpointMethod(name, call, dict(method=name, status="UNAVAILABLE", deployable=False,
                                          inference_teacher_inputs=False, reason=reason))


def make_methods(samples, protocol, compression_result=None, learned_models=None, *, private_factors=None, budget=None):
    """Six complete initial-state callbacks, without selecting on diagnostics.

    ``samples`` supplies only device/context inventory; labels never enter a
    callback. Models must already be frozen by the controller. Factor records
    accept the controller's {context,method,parent_id,factors} payload schema.
    """
    cfg = _settings(protocol)
    entries = _factor_entries(compression_result, private_factors)
    chosen, selection = _selection(protocol, compression_result)
    methods = MethodCollection()
    common = dict(status="AVAILABLE", inference_teacher_inputs=False, scope="complete initial-state call; all numerical path/residual work recomputed",
                  target_contract="same-grid FD/Galerkin only", checkpoint_scope="controller verifies source/software/checkpoint seals")

    def anchor(u0, h, eq, geom, target):
        _validate(u0, h, eq, geom, target)
        step = n.scalar(h, u0)
        output = n.anchor(u0, step, eq, geom, target, name=cfg["anchor"], reaction_substeps=cfg["reaction_substeps"])
        output = torch.where(step == 0, u0, output)
        return dict(output=output, work=dict(deployed_anchor_evaluations=1, path_queries=0, residual_evaluations=0))
    methods["anchor"] = EndpointMethod("anchor", anchor, dict(common, deployable=True, precision_policy="actual FP32 deployed anchor", role="anchor"))

    def response(u0, h, eq, geom, target, *, shared=False):
        _validate(u0, h, eq, geom, target)
        path = _path(u0, eq, geom, target, cfg, budget, h)
        step = n.scalar(h, u0.double())
        smooth = path(step)
        correction = torch.zeros_like(smooth)
        factor = _factor_for(entries, chosen, geom, target, u0.device) if shared else None
        null = _normals(u0, eq, geom, target) if shared else None
        for node, weight in zip(*quadrature(cfg["source_nodes"])):
            source_time = step*node
            residual = path.evaluate(source_time)["residual"]
            action = _mean(path, residual, source_time, step, eq, geom, target, cfg["mean_nodes"])
            if factor is not None:
                # Known exact nulls; no claim that the fixed basis learns the
                # state-dependent spatial factors away from these nulls.
                extra = factor.apply(residual, source_time, step)
                action = action+torch.where(null, extra, torch.zeros_like(extra))
            correction = correction-step*weight*action
        physical = smooth+correction
        output = physical.float() if cfg["control_output_dtype"] == "float32" else physical
        output = torch.where(step == 0, u0.to(output), output)
        return dict(output=output, physical_prediction=physical, smooth_anchor64=smooth, physical_correction=correction,
                    work={**path.work(), "source_nodes":cfg["source_nodes"], "mean_nodes":cfg["mean_nodes"],
                          "response_action_calls":cfg["source_nodes"], "factor_contractions":cfg["source_nodes"] if shared else 0,
                          "deployed_anchor_evaluations":0, "teacher_queries":0})
    methods["G"] = EndpointMethod("G", response, dict(common, deployable=True, role="matched_anchor_mean_response",
             precision_policy="FP64 smooth path/residual/mean response; declared final FP32 cast", source_nodes=cfg["source_nodes"], mean_nodes=cfg["mean_nodes"]))
    available_contexts = [list(context) for context, name in entries if name == chosen]
    if available_contexts:
        methods["fixed_shared_basis"] = EndpointMethod("fixed_shared_basis", lambda *args, **kwargs: response(*args, **kwargs, shared=True),
                 dict(common, deployable=True, role="fixed_train_only_shared_response_basis", selection=selection,
                      available_contexts=available_contexts,
                      precision_policy="FP64 explicit path/source/factor response; final FP32 cast", source_nodes=cfg["source_nodes"],
                      factor_state_scope="fixed shared spatial basis; new anchor/path for every current state"))
    else:
        methods["fixed_shared_basis"] = _unavailable("fixed_shared_basis", "Frozen training-only shared response factors missing")

    def oracle(u0, h, eq, geom, target):
        _validate(u0, h, eq, geom, target)
        deadline = _Deadline(cfg["seconds"], budget)
        path = _path(u0, eq, geom, target, cfg, deadline, h)
        correction, metadata = refined_forced(path, eq, geom, target, h, mode="tangent",
              initial_steps=cfg["initial_steps"], max_steps=cfg["max_steps"], tolerance=cfg["tolerance"], budget=deadline)
        work = {**path.work(), "forced_solves":len(metadata.get("refinements", [])),
                "requested_forced_steps":sum(row["steps"] for row in metadata.get("refinements", [])), "teacher_queries":0}
        if not metadata.get("accepted", False) or correction is None:
            raise UnresolvedInference("UNRESOLVED: bounded tangent oracle not independently time resolved", metadata, work)
        smooth = path(h)
        return dict(output=smooth+correction, smooth_anchor64=smooth, physical_correction=correction,
                    oracle_metadata=metadata, work={**work, **path.work()})
    methods["U"] = EndpointMethod("U", oracle, dict(common, deployable=False, role="expensive_spatial_tangent_oracle",
                precision_policy="FP64 fixed-path independently refined forced response", refinement_is_certificate=False,
                initial_steps=cfg["initial_steps"], max_steps=cfg["max_steps"], tolerance=cfg["tolerance"], seconds_cap=cfg["seconds"]))

    for name in ("learned_response", "residual_fno"):
        model, metadata = _model_entry((learned_models or {}).get(name))
        if model is None:
            methods[name] = _unavailable(name, "Validation-selected learned checkpoint missing")
            continue
        identity = _versions(model)
        def call(u0, h, eq, geom, target, model=model, identity=identity):
            _validate(u0, h, eq, geom, target)
            if _versions(model) != identity:
                raise ValueError("Frozen inference checkpoint/device/dtype changed")
            if callable(getattr(model, "prediction_parts", None)):
                encoded = model.encode(u0, eq, geom, target, budget=budget)
                parts = model.prediction_parts(encoded, h)
                return dict(parts, output=parts["output"])
            return dict(output=model(u0, h, eq, geom, target), work=dict(complete_model_calls=1, teacher_queries=0))
        architecture = model.architecture_metadata() if callable(getattr(model, "architecture_metadata", None)) else {}
        methods[name] = EndpointMethod(name, call, dict(common, deployable=True, role=name,
               checkpoint_metadata=metadata, architecture=architecture,
               selection_scope="validation-selected model; controller seals identity before evaluation"))
    return MethodCollection((name, methods[name]) for name in NAMES)


def make_tangent_methods(u0, eq, geom, target, protocol, compression_result=None, learned_models=None,
                         *, private_factors=None, budget=None):
    """Action callbacks along one immutable anchor, distinct from macrosteps.

    Signature is ``action(source,s,h)``. Learned encodings are recomputed inside
    each call and use this SAME path, so encoder work is included. Endpoint-only
    FNO/anchor controls have no invented response operator.
    """
    _validate(u0, 0., eq, geom, target)
    cfg = _settings(protocol)
    initial = u0.clone()
    path = _path(initial, eq, geom, target, cfg, budget)
    entries = _factor_entries(compression_result, private_factors)
    chosen, selection = _selection(protocol, compression_result)
    physics = tuple(torch.as_tensor(value, dtype=torch.float64).detach().clone() for value in (eq.kappa, eq.reaction_rate))
    geometry_signature = (tuple(geom.grid), tuple(geom.lengths))
    identity = digest(dict(input_sha256=hashlib.sha256(initial.detach().cpu().numpy().tobytes()).hexdigest(),
                           physics=[value.cpu().tolist() for value in physics], geometry=list(geometry_signature),
                           target_kind=target, anchor=cfg["anchor"], reaction_substeps=cfg["reaction_substeps"],
                           dtype=str(initial.dtype), device=str(initial.device)))
    methods = MethodCollection()
    common = dict(status="AVAILABLE", path_identity=identity, scope="same immutable initial-state anchor path; no macrostep re-encoding",
                  inference_teacher_inputs=False, target_contract=target)
    def mean(source, s, h):
        value = source.double()
        result = _mean(path, value, s, h, eq, geom, target, cfg["mean_nodes"])
        return dict(output=result, work={**path.work(), "response_action_calls":1, "mean_nodes":cfg["mean_nodes"]})
    methods["G"] = EndpointMethod("G", mean, dict(common, deployable=True, role="same_path_anchor_mean_response"))
    def oracle(source, s, h):
        deadline = _Deadline(cfg["seconds"], budget)
        result, metadata = refined_action(path, eq, geom, target, source.double(), s, h,
                   initial_steps=cfg["initial_steps"], max_steps=cfg["max_steps"], tolerance=cfg["tolerance"], budget=deadline)
        work = {**path.work(), "homogeneous_solves":len(metadata.get("refinements", [])),
                "requested_tangent_steps":sum(row["steps"] for row in metadata.get("refinements", []))}
        if not metadata.get("accepted", False) or result is None:
            raise UnresolvedInference("UNRESOLVED: same-path tangent action not time resolved", metadata, work)
        return dict(output=result, oracle_metadata=metadata, work=work)
    methods["U"] = EndpointMethod("U", oracle, dict(common, deployable=False, role="same_path_spatial_tangent_oracle"))
    if ((target, tuple(geom.grid), tuple(geom.lengths)), chosen) in entries:
        def shared(source, s, h):
            factor = _factor_for(entries, chosen, geom, target, source.device)
            result = mean(source, s, h)
            extra = factor.apply(source.double(), s, h)
            result["output"] = result["output"]+torch.where(_normals(initial, eq, geom, target), extra, torch.zeros_like(extra))
            result["work"]["factor_contractions"] = 1
            return result
        methods["fixed_shared_basis"] = EndpointMethod("fixed_shared_basis", shared, dict(common, deployable=True, selection=selection))
    else:
        methods["fixed_shared_basis"] = _unavailable("fixed_shared_basis", "Frozen training-only shared factors missing")
    model, metadata = _model_entry((learned_models or {}).get("learned_response"))
    if model is not None and callable(getattr(model, "response_action", None)):
        checkpoint = _versions(model)
        def learned(source, s, h):
            if _versions(model) != checkpoint:
                raise ValueError("Frozen response checkpoint changed")
            if getattr(model, "anchor_name", cfg["anchor"]) != cfg["anchor"] or getattr(model, "reaction_substeps", cfg["reaction_substeps"]) != cfg["reaction_substeps"]:
                raise ValueError("Learned response and same-path anchor contracts differ")
            encoded = model.encode(initial, eq, geom, target, budget=budget)
            encoded.path = path
            output = model.response_action(encoded, source, s, h)
            return dict(output=output, work={**path.work(), "encoder_calls":1, "response_action_calls":1})
        methods["learned_response"] = EndpointMethod("learned_response", learned, dict(common, deployable=True, checkpoint_metadata=metadata))
    else:
        methods["learned_response"] = _unavailable("learned_response", "Learned response-action checkpoint missing")
    methods["anchor"] = _unavailable("anchor", "Endpoint anchor is not a response-action model")
    methods["residual_fno"] = _unavailable("residual_fno", "Endpoint-only FNO has no response-action contract")
    for method in methods.values():
        callback = method.callback
        def checked(*args, callback=callback, **kwargs):
            if (tuple(geom.grid), tuple(geom.lengths)) != geometry_signature:
                raise ValueError("Declared same-path geometry changed")
            if any(not torch.equal(torch.as_tensor(value, dtype=torch.float64, device=saved.device).detach(), saved)
                   for value, saved in zip((eq.kappa, eq.reaction_rate), physics)):
                raise ValueError("Declared same-path physics changed")
            return callback(*args, **kwargs)
        method.callback = checked
    methods.path = path
    methods.path_identity = identity
    return MethodCollectionWithPath(methods, path, identity)


class MethodCollectionWithPath(MethodCollection):
    def __init__(self, methods, path, identity):
        super().__init__(methods)
        self.path, self.path_identity = path, identity
