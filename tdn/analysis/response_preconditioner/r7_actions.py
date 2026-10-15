"""Five-branch identification and complete-call CPU timing for bounded R7.

Design extraction is an offline identification operation. Only the ordered,
fixed-step Lawson and sealed global readout actions are deployable timing
competitors. No teacher-informed local coefficients enter this module's
competitor menu. A repeated-source bundle deliberately rebuilds its path and
features for every source: it establishes a no-reuse cost, not amortization.
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence

import torch

from tdn.analysis.response import numerics as n
from .model import (BRANCH_NAMES, FEATURE_NAMES, FixedFeatureResponse,
                    _path_work, _prepare_path)
from .numerics import ordered_response

ORDERED_PANELS = (1, 2, 4, 8)
LAWSON_STEPS = (2, 4, 8)
WORK_SCOPE = (
    "Explicit response heat/FFT/stencil operations and path/feature calls; "
    "internal anchor/autograd transforms and actual peak memory are not "
    "instrumented. End-to-end CPU timing includes their execution."
)


def _model_contract(model, u):
    if model is None:
        model = FixedFeatureResponse().to(device=u.device)
    if not isinstance(model, FixedFeatureResponse):
        raise ValueError("R7 requires the existing FixedFeatureResponse")
    if (model.panels, model.potential_nodes, model.time_reference,
            model.output_precision) != (2, 2, 1., "float64"):
        raise ValueError("R7 freezes two panels, GL2, reference time 1 and FP64 output")
    return model


def design_context(u, z, start, stop, eq, geom, model=None):
    """Return the exact five branch columns and eight source-independent features.

    The public 40-column design remains authoritative. Every eighth column is
    its branch multiplied by eta[0]=1, so extraction introduces no inverse or
    division by a small feature. Feature construction is repeated once to
    expose eta through the existing model implementation; this additional
    offline extraction work is counted and is excluded from inference timing.
    A context contains one original parent, not a batch of different states.
    """
    if not torch.is_tensor(u) or u.ndim < 3 or u.shape[0] != 1:
        raise ValueError("R7 design_context requires exactly one parent")
    model = _model_contract(model, u)
    work = {}
    base, design = model.action_design(u, z, start, stop, eq, geom, work=work)
    initial, physics, path = model._prepare(u, eq, geom, "fd_same_grid")
    encoded = model._encode(initial, work)
    _, s, t = model._source_interval(initial, z, start, stop, geom, "fd_same_grid")
    _, eta = model._features(encoded, s, t, physics, geom, work)
    _path_work(work, path)
    features = eta.reshape(len(FEATURE_NAMES))
    branches = design[::len(FEATURE_NAMES)]
    if branches.shape[0] != len(BRANCH_NAMES) or not bool(features[0] == 1):
        raise ValueError("The frozen R7 branch-major constant-feature contract changed")
    # Detect drift in the defining source-independent factorization before
    # any local coefficients are fitted or their labels are acquired.
    reconstructed = (branches[:, None] * features.reshape(
        1, len(FEATURE_NAMES), *([1] * u.ndim))).flatten(0, 1)
    if not torch.equal(reconstructed, design):
        raise ValueError("R7 five-branch factorization no longer matches the model")
    work["offline_design_extractions"] = 1
    return dict(base=base, branches=branches, features=features, design=design,
                work=work, work_counter_scope=WORK_SCOPE)


def fresh_ordered_action(u, z, start, stop, eq, geom, *, panels=2, work=None):
    """Evaluate a complete ordered action about the original graph-capable path."""
    initial, physics, path = _prepare_path(u, eq, geom, "fd_same_grid")
    n.validate(z, 0., geom, "fd_same_grid")
    if z.shape != initial.shape or z.device != initial.device:
        raise ValueError("Source must match initial-state shape and device")
    result = ordered_response(path, z.to(dtype=torch.float64), start, stop,
                              physics, geom, panels=panels, potential_nodes=2,
                              work=work)
    _path_work(work, path)
    if work is not None:
        work["fresh_path_constructions"] = work.get("fresh_path_constructions", 0) + 1
    return result


def fresh_action(method, u, z, start, stop, eq, geom, *, model=None, work=None):
    """Run one deployable action with setup and precision bridges included.

    The selected global readout must already be loaded, as a deployed model
    would be. Its state conditioning is recomputed on every invocation. Local
    projection/oracle coefficients and refined teachers are not accepted.
    """
    if method in {f"ordered_{panels}" for panels in ORDERED_PANELS}:
        return fresh_ordered_action(u, z, start, stop, eq, geom,
                                    panels=int(method.split("_")[-1]), work=work)
    if method == "fixed_feature_fitted":
        if model is None:
            raise ValueError("The sealed global model is required for learned timing")
        model = _model_contract(model, u)
        result = model.response_action(u, z, start, stop, eq, geom, work=work)
        if work is not None:
            work["fresh_path_constructions"] = work.get("fresh_path_constructions", 0) + 1
        return result
    if method in {f"lawson_{steps}" for steps in LAWSON_STEPS}:
        from .r7_references import fixed_lawson_action
        return fixed_lawson_action(u, z, start, stop, eq, geom,
                                   steps=int(method.split("_")[-1]), work=work)
    raise ValueError(f"Unknown or nondeployable R7 action method: {method}")


def fresh_actions(u, sources: Sequence[torch.Tensor], start, stop, eq, geom,
                  model, *, lawson_steps=LAWSON_STEPS):
    """Build complete repeated-source callables, with no cross-source reuse.

    Every result has shape [sources, batch, channels, *grid]. Stacking outputs
    is included in the measured bundle cost. Sources execute sequentially at
    the same fixed context; this is not a vectorized block solver or endpoint.
    """
    if (not torch.is_tensor(u) or u.device.type != "cpu" or u.ndim < 3 or
            u.shape[:2] != (1, 1)):
        raise ValueError("R7 timing requires one scalar CPU parent")
    if model is None:
        raise ValueError("The sealed global model is required for learned timing")
    model = _model_contract(model, u)
    sources = tuple(sources)
    if not sources or any(not torch.is_tensor(z) or z.shape != u.shape or
                          z.device != u.device for z in sources):
        raise ValueError("A nonempty source sequence matching the parent is required")
    if len(set(lawson_steps)) != len(lawson_steps) or any(
            type(steps) is not int or steps not in LAWSON_STEPS for steps in lawson_steps):
        raise ValueError("R7 Lawson controls must be distinct choices from 2, 4, 8")
    names = [*(f"ordered_{panels}" for panels in ORDERED_PANELS),
             "fixed_feature_fitted", *(f"lawson_{steps}" for steps in lawson_steps)]

    def bundle(method):
        return torch.stack([fresh_action(method, u, z, start, stop, eq, geom, model=model)
                            for z in sources])

    return {name: (lambda method=name: bundle(method)) for name in names}


def _output_digest(value):
    if not torch.is_tensor(value) or value.device.type != "cpu":
        raise ValueError("R7 CPU timing callables must return a CPU tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite timed output; no clipping or silent fallback")
    data = value.detach().contiguous().numpy()
    return hashlib.sha256(str(data.dtype).encode() + str(data.shape).encode() +
                          data.tobytes()).hexdigest()


def measure_paired_actions(methods: Mapping[str, Callable[[], torch.Tensor]], *,
                           rounds=5, seed=2026101507, warmup=1, budget=None,
                           on_record=None):
    """Time fresh bundles in randomized paired rounds, retaining every attempt.

    ``budget`` is an optional zero-argument callback checked before each call;
    its exception propagates to the workflow's retained failure handler. Model
    exceptions remain rows with FAILED status and count in the denominator.
    ``on_record`` receives each completed row outside the timer, allowing a
    journal to retain partial observations before a later budget interruption.
    Output checks/digests occur after the timer. CPU synchronization is inherent;
    this helper rejects CUDA outputs and requires one PyTorch CPU thread.

    "Cold" means this benchmark's first invocation, not a fresh interpreter or
    guaranteed cold hardware cache. Raw rounds and min/median/max are empirical
    timing variation, not independent-sample confidence intervals or a speedup.
    """
    if (type(rounds) is not int or rounds < 1 or type(warmup) is not int or warmup < 0
            or type(seed) is not int or not methods
            or any(not isinstance(name, str) or not name or not callable(fn)
                   for name, fn in methods.items())):
        raise ValueError("Invalid paired timing method/round/warmup/seed contract")
    if torch.get_num_threads() != 1 or not torch.is_grad_enabled():
        raise ValueError("R7 timing requires one PyTorch CPU thread and gradients enabled")
    generator = random.Random(seed)
    records = []
    identities = {}

    def phase(label, index):
        order = list(methods)
        generator.shuffle(order)
        for position, name in enumerate(order):
            if budget is not None:
                budget()
            row = dict(phase=label, round=index, position=position, method=name)
            begin = time.perf_counter()
            elapsed = None
            output = None
            try:
                output = methods[name]()
                elapsed = time.perf_counter() - begin
                digest = _output_digest(output)
                if name in identities and identities[name] != digest:
                    raise ValueError("Timed method changed its output across repeated calls")
                identities[name] = digest
                row.update(status="PASS", seconds=elapsed, output_sha256=digest)
            except Exception as exc:
                if elapsed is None:
                    elapsed = time.perf_counter() - begin
                row.update(status="FAILED", seconds=elapsed,
                           error=f"{type(exc).__name__}: {exc}")
            finally:
                # Do not retain the preceding action's autograd graph while
                # constructing the next action in a timing round.
                del output
            if not math.isfinite(row["seconds"]) or row["seconds"] < 0:
                raise RuntimeError("Invalid monotonic timing observation")
            records.append(row)
            if on_record is not None:
                on_record(dict(row))

    phase("cold", 0)
    for index in range(warmup):
        phase("warmup", index)
    for index in range(rounds):
        phase("paired", index)
    summary = {}
    for name in methods:
        paired = [row for row in records if row["phase"] == "paired" and row["method"] == name]
        elapsed = [row["seconds"] for row in paired if row["status"] == "PASS"]
        summary[name] = dict(expected_rounds=rounds, passed_rounds=len(elapsed),
                             failed_rounds=rounds-len(elapsed),
                             median_seconds=statistics.median(elapsed) if elapsed else None,
                             minimum_seconds=min(elapsed) if elapsed else None,
                             maximum_seconds=max(elapsed) if elapsed else None)
    failures = sum(row["status"] != "PASS" for row in records)
    return dict(schema="tdn.response_preconditioner.r7.paired_timing.v1",
                status="PASS" if failures == 0 else "FAILED", seed=seed,
                rounds=rounds, warmup=warmup, records=records, summary=summary,
                expected_calls=(1+warmup+rounds)*len(methods), failed_calls=failures,
                methods=list(methods), device="cpu", torch_threads=torch.get_num_threads(),
                gradient_contract="grad enabled; fresh original AnchorPath create_graph=True per source",
                cost_scope="complete sequential source bundle; setup charged independently for every source; no cross-source reuse",
                cold_scope="first benchmark invocation, not a new process or guaranteed cold cache",
                timer_exclusions="output validation/digest and record serialization; one-time resident model loading",
                uncertainty_scope="raw paired rounds and empirical range; no confidence interval or superiority test",
                work_counter_scope=WORK_SCOPE, peak_memory="not measured by this helper; inspect worker telemetry separately",
                oracle_timing="excluded; teacher acquisition and fitting are offline costs",
                scientific_promotion=False)
