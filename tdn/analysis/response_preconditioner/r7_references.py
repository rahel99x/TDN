"""Bounded R7 action labels and a deployable fixed-work Lawson control.

Every source starts at its declared absolute time on the *original* DF anchor
from u0.  The refined Torch and independent NumPy integrators have separate
implementations and share one wall allowance per U/G source query. Their
agreement estimates uncertainty; it does not certify an exact response.
"""
from __future__ import annotations

import math
import time

import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.tangent_response.oracle import independent_solve
from tdn.analysis.tangent_response.tangent import (
    AnchorPath, physical_norms, refined_action, response_action,
)

TARGET = "fd_same_grid"
DEFAULT_REFERENCE = dict(tolerance=1e-10, initial_steps=4, max_steps=256,
                         max_seconds_per_query=20., independent_tolerance=1e-12,
                         independent_max_rhs_calls=4096)


def _configuration(config):
    source = config.get("reference", config)
    options = {key: source.get(key, value) for key, value in DEFAULT_REFERENCE.items()}
    for key in ("tolerance", "max_seconds_per_query", "independent_tolerance"):
        if isinstance(options[key], bool):
            raise ValueError(f"Reference {key} must be positive and finite")
        options[key] = float(options[key])
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"Reference {key} must be positive and finite")
    for key in ("initial_steps", "max_steps", "independent_max_rhs_calls"):
        if type(options[key]) is not int or not 1 <= options[key] <= 1_000_000:
            raise ValueError(f"Reference {key} must be a bounded positive integer")
    if options["initial_steps"] > options["max_steps"]:
        raise ValueError("Reference initial steps must not exceed the refinement cap")
    return options


def _inputs(u, z, start, stop, eq, geom):
    if not torch.is_tensor(u) or not torch.is_tensor(z):
        raise ValueError("Action inputs must be tensors")
    if u.dtype not in (torch.float32, torch.float64) or z.dtype not in (torch.float32, torch.float64):
        raise ValueError("Action inputs require FP32 or FP64")
    if u.device.type != "cpu" or z.device.type != "cpu":
        raise ValueError("R7 action implementation explicitly supports CPU only")
    n.validate(u, 0., geom, TARGET)
    n.validate(z, 0., geom, TARGET)
    if u.shape != z.shape or u.shape[:2] != (1, 1):
        raise ValueError("R7 action requires one scalar parent and one matching source")
    if not bool(torch.isfinite(u).all() and torch.isfinite(z).all()):
        raise ValueError("Action inputs must be finite")
    times = []
    for name, value in (("start", start), ("stop", stop)):
        scalar = torch.as_tensor(value, dtype=torch.float64)
        if scalar.numel() != 1 or not bool(torch.isfinite(scalar).all()):
            raise ValueError(f"Action {name} must be a finite scalar")
        times.append(float(scalar.detach()))
    if not 0 <= times[0] <= times[1]:
        raise ValueError("Action rejects reversed or negative intervals")
    for key in ("kappa", "reaction_rate"):
        value = torch.as_tensor(getattr(eq, key), dtype=torch.float64)
        if value.numel() != 1 or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"Action {key} must be a finite nonnegative scalar")
    return times


class _QueryBudget:
    def __init__(self, seconds, outer):
        self.started = time.monotonic()
        self.seconds, self.outer = seconds, outer

    def check(self):
        if self.outer is not None:
            self.outer.check()
        if time.monotonic()-self.started >= self.seconds:
            raise TimeoutError("R7 action query wall allowance exhausted (U and G combined)")


class _IndependentBudget:
    def __init__(self, query, maximum):
        self.query, self.maximum, self.started = query, maximum, 0

    def check(self):
        self.query.check()
        if self.started >= self.maximum:
            raise TimeoutError("Independent action RHS allowance exhausted")
        self.started += 1


def _one_mode(u, z, start, stop, eq, geom, mode, options, query):
    path = AnchorPath(u, eq, geom, TARGET, budget=query)
    refined, refinement = refined_action(
        path, eq, geom, TARGET, z, start, stop, mode=mode,
        initial_steps=options["initial_steps"], max_steps=options["max_steps"],
        tolerance=options["tolerance"]*.25, budget=query)
    independent = None
    cross = None
    difference = dict(rms=None, maximum=None)
    combined = dict(rms=None, maximum=None)
    independent_budget = _IndependentBudget(query, options["independent_max_rhs_calls"])
    reason = refinement["reason"]
    if refinement["accepted"]:
        try:
            query.check()
            independent, cross = independent_solve(
                u, stop, eq, geom, TARGET, kind="action", z=z, s=start, mode=mode,
                tolerance=options["independent_tolerance"], check=independent_budget.check)
            query.check()
            difference = physical_norms(refined-independent, geom, TARGET)
            combined = {key: refinement["uncertainty_"+key]+difference[key]
                        for key in ("rms", "maximum")}
            reason = "TIME_AND_INDEPENDENT_ACTION_BUDGET_ACCEPTED"
        except (TimeoutError, InterruptedError, FloatingPointError) as error:
            reason = type(error).__name__+": "+str(error)
    accepted = bool(refinement["accepted"] and independent is not None and
                    all(value is not None and math.isfinite(value) and value <= options["tolerance"]
                        for value in combined.values()))
    if not accepted and reason == "TIME_AND_INDEPENDENT_ACTION_BUDGET_ACCEPTED":
        reason = "COMBINED_ACTION_BUDGET_UNRESOLVED"
    return independent, refined, dict(
        accepted=accepted, status="ACCEPTED" if accepted else "UNRESOLVED", reason=reason,
        mode=mode, refinement=refinement, independent=cross, cross_difference=difference,
        uncertainty_rms=combined["rms"], uncertainty_maximum=combined["maximum"],
        tolerance=options["tolerance"], uncertainty_is_certificate=False,
        work=dict(path=path.work(), refined_rhs_calls_started=path.work()["path_queries"],
                  independent_rhs_calls_started=independent_budget.started,
                  independent_rhs_cap=independent_budget.maximum,
                  refinement_completed_step_counts=[row["steps"] for row in refinement["refinements"]],
                  incomplete_counters="Anchor/autograd FFT internals; failed independent inner operations"))


def acquire_reference(u, z, start, stop, eq, geom, config, budget=None):
    """Return U/G tensors and JSON metadata, retaining unresolved acquisitions.

    ``teacher`` and ``matched_mean`` are independent DOP853 endpoints, or None
    if unavailable. ``refined`` and ``matched_mean_refined`` retain the last
    completed Lawson approximations even when unresolved. No unresolved value
    may be admitted to fitting by the caller. ``reference.accepted`` qualifies
    U only; the independently assessed G status is in ``matched_mean``.

    The cap is one source query comprising U and G. Each mode independently
    caps the sum of attempted fixed refinements and independent RHS calls;
    both modes share the same wall allowance and enclosing study budget.
    """
    start, stop = _inputs(u, z, start, stop, eq, geom)
    options = _configuration(config)
    query = _QueryBudget(options["max_seconds_per_query"], budget)
    initial, source = u.detach().double(), z.detach().double()
    teacher, refined, reference = _one_mode(initial, source, start, stop, eq, geom,
                                          "tangent", options, query)
    mean, mean_refined, mean_reference = _one_mode(initial, source, start, stop, eq, geom,
                                                 "mean", options, query)
    reference.update(matched_mean=mean_reference, start=start, stop=stop,
                     path_contract="Original DF anchor a(t; u0); source initialized at absolute s",
                     target=TARGET, input_contract="Caller input promoted to FP64 without rerounding",
                     elapsed_seconds=time.monotonic()-query.started, caps=options,
                     uncertainty_aggregation="Lawson refinement estimate plus independent discrepancy",
                     wall_cap_scope="One source query: U and G together",
                     source_query_count=1, action_modes_attempted=2)
    levels = []
    steps = options["initial_steps"]
    while steps <= options["max_steps"]:
        levels.append(steps)
        steps *= 2
    reference["maximum_work_per_source_query"] = dict(
        action_modes=2, refinement_step_levels_per_mode=levels,
        lawson_steps_both_modes=2*sum(levels), lawson_rhs_calls_both_modes=8*sum(levels),
        independent_rhs_calls_both_modes=2*options["independent_max_rhs_calls"],
        wall_seconds_both_modes=options["max_seconds_per_query"])
    return dict(teacher=teacher, refined=refined, matched_mean=mean,
                matched_mean_refined=mean_refined, reference=reference)


def fixed_lawson(u, z, start, stop, eq, geom, steps, work=None):
    """Deployable fixed-work action, with a fresh differentiable original path.

    This control never reads a reference or adapts its number of steps. The
    caller measures the complete invocation, including FP64 bridges and path
    setup. The work counters are algorithmic counts, not hardware FLOPs.
    """
    _inputs(u, z, start, stop, eq, geom)
    if type(steps) is not int or not 1 <= steps <= 1_000_000:
        raise ValueError("Fixed Lawson steps must be a bounded positive integer")
    path = AnchorPath(u.double(), eq, geom, TARGET, create_graph=True)
    value = response_action(path, eq, geom, TARGET, z.double(), start, stop, steps=steps)
    if work is not None:
        work.update(method="fixed_Lawson_RK4", steps=steps, rhs_evaluations=4*steps,
                    integration_heat_actions=7*steps, integration_fft_transforms=14*steps,
                    path=path.work(), fresh_original_anchor=True,
                    gradient_contract="Complete fresh path; no detached cache",
                    incomplete_counters="Anchor/autograd FFT internals; end-to-end timing required")
    return value


fixed_lawson_action = fixed_lawson
