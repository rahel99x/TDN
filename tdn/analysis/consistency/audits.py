"""Training-free structural checks for the bounded consistency experiment.

The learned parameters are deliberately randomized: an exact-limit check at
zero-head initialization cannot establish that the architecture enforces it.
Unconstrained backbones are recorded as controls, never treated as failures for
properties that they do not claim. Correctness failures prevent an audit seal.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import laplacian, rhs
from tdn.numerics.splitting import split_step
from tdn.research.reaction import capacity_update
from tdn.runtime.metadata import write_json


SCHEMA = "tdn.consistency-neural/v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _state(grid, *, dtype, device):
    axes = torch.meshgrid(*(torch.arange(n, dtype=dtype, device=device) / n
                            for n in grid), indexing="ij")
    wave = .55 * torch.cos(2 * math.pi * 2 * axes[0] + .31)
    if len(grid) == 2:
        wave = wave + .45 * torch.sin(2 * math.pi * (axes[0] - 3 * axes[1]) - .7)
    return torch.stack((.4 + .22 * wave, .65 - .18 * wave)).unsqueeze(1)


def _scalar(value):
    answer = float(torch.as_tensor(value).detach().cpu())
    return answer if math.isfinite(answer) else None


def _maximum(value):
    return _scalar(value.abs().max())


def _randomize(model, seed):
    """Use a local CPU generator and leave the caller's RNG state unchanged."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            sample = torch.randn(parameter.shape, generator=generator,
                                 dtype=torch.float64) * .15
            parameter.copy_(sample.to(device=parameter.device, dtype=parameter.dtype))
    return model


def _row(identity, error, tolerance, *, required=True, **values):
    error = _scalar(error) if error is not None else None
    passed = error is not None and error <= tolerance
    status = ("PASS" if passed else "FAILED") if required else "OBSERVED"
    return dict(check_id=identity, case_id=identity, required=required,
                status=status, outcome=status,
                maximum_absolute_error=error, absolute_tolerance=float(tolerance),
                **values)


def _helper_cases(dtype, device, protocol):
    from tdn.research.consistency_neural import (
        calibrate_mean, commutator_gate, discrete_logistic_commutator,
        mean_shape_update,
    )

    precision = str(dtype).split(".")[-1]
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    equation = Equation(.003, 2.)
    for grid in ((16,), (9, 10)):
        geometry = Geometry(grid, tuple(1. for _ in grid))
        value = _state(grid, dtype=dtype, device=device)
        commutator = discrete_logistic_commutator(value, equation, geometry)
        # An independent R'(u)D(u)-D(R) construction checks the production
        # helper's stable edge formula, sign and physical scaling.
        reaction = equation.reaction_rate * value * (1 - value)
        direct = equation.reaction_rate * (1 - 2 * value) * equation.kappa * laplacian(value, geometry)
        direct = direct - equation.kappa * laplacian(reaction, geometry)
        prefix = f"helper/{precision}/{'x'.join(map(str, grid))}"
        yield _row(prefix + "/commutator_identity", _maximum(commutator - direct), tolerance,
                   equation=dict(kappa=equation.kappa, reaction_rate=equation.reaction_rate))
        yield _row(prefix + "/commutator_sign", max(0., -(_scalar(commutator.min()) or 0.)), tolerance,
                   minimum_commutator=_scalar(commutator.min()))
        axes = tuple(range(2, value.ndim))
        energy = sum(((torch.roll(value, -1, axis) - value) / dx).square().mean(axes, keepdim=True)
                     for axis, dx in enumerate(geometry.dx, start=2))
        telescoped = 2 * equation.kappa * equation.reaction_rate * energy
        mean_state = value.mean(axes, keepdim=True)
        variance = (value - mean_state).square().mean(axes, keepdim=True)
        physical_mean_derivative = rhs(value, equation, geometry).mean(axes, keepdim=True)
        expected_mean_derivative = equation.reaction_rate * (mean_state - mean_state.square() - variance)
        mean_law_error = _maximum(physical_mean_derivative - expected_mean_derivative)
        diffusion_mean_error = _maximum((equation.kappa * laplacian(value, geometry)).mean(axes, keepdim=True))
        commutator_mean_error = _maximum(commutator.mean(axes, keepdim=True) - telescoped)
        yield _row(prefix + "/periodic_telescoping",
                   max(commutator_mean_error, mean_law_error, diffusion_mean_error), tolerance,
                   algebraic_identity="mean(C)=2*kappa*r*sum(mean(forward_difference/dx)^2)",
                   mean_law_identity="mean(rhs(u))=r*(m-m^2-Var[u]); mean(diffusion)=0",
                   mean_law_error=mean_law_error, diffusion_mean_error=diffusion_mean_error,
                   commutator_mean_error=commutator_mean_error,
                   reaction_rate=equation.reaction_rate,
                   mean_state=[_scalar(x) for x in mean_state.flatten()],
                   variance=[_scalar(x) for x in variance.flatten()],
                   physical_mean_derivative=[_scalar(x) for x in physical_mean_derivative.flatten()])

    # C is exactly quadratic in a perturbation about a constant field on the
    # declared central-difference grid. The bounded tanh gate has the same
    # small-amplitude order, not a linear gradient-magnitude surrogate.
    geometry = Geometry((16, 16), (1., 1.))
    wave = _state(geometry.grid, dtype=dtype, device=device)[:1] - .4
    amplitudes = (.02, .01, .005)
    magnitudes, gates = [], []
    for amplitude in amplitudes:
        value = .45 + amplitude * wave
        commutator = discrete_logistic_commutator(value, equation, geometry)
        gate = commutator_gate(value, equation, geometry,
                               t_ref=protocol["t_ref"], U_ref=protocol["U_ref"])
        magnitudes.append(_scalar(commutator.double().square().mean().sqrt()))
        gates.append(_scalar(gate.double().square().mean().sqrt()))
    commutator_orders = [math.log(magnitudes[i] / magnitudes[i + 1], 2.) for i in range(2)]
    gate_orders = [math.log(gates[i] / gates[i + 1], 2.) for i in range(2)]
    order_tolerance = .004 if dtype == torch.float32 else 2e-8
    yield _row(f"helper/{precision}/small_amplitude_commutator_order",
               max(abs(value - 2) for value in commutator_orders), order_tolerance,
               amplitudes=list(amplitudes), rms=magnitudes, observed_orders=commutator_orders)
    yield _row(f"helper/{precision}/small_amplitude_gate_order",
               max(abs(value - 2) for value in gate_orders), max(order_tolerance, 2e-6),
               amplitudes=list(amplitudes), rms=gates, observed_orders=gate_orders)

    base = _state((8, 8), dtype=dtype, device=device)
    axes = (-2, -1)
    for index, target in enumerate((0., 1e-6, .24, .78, 1 - 1e-6, 1.)):
        targets = torch.full((base.shape[0], 1, 1, 1), target, dtype=dtype, device=device)
        calibrated = calibrate_mean(base, targets)
        error = max(_maximum(calibrated.mean(axes, keepdim=True) - targets),
                    max(0., -(_scalar(calibrated.min()) or 0.)),
                    max(0., (_scalar(calibrated.max()) or 0.) - 1.))
        yield _row(f"helper/{precision}/calibrate_mean/{index}", error,
                   5e-7 if dtype == torch.float32 else 3e-15,
                   target_mean=target, output_min=_scalar(calibrated.min()),
                   output_max=_scalar(calibrated.max()))

    for index, increment in enumerate((-1000., -.05, 0., .05, 1000.)):
        mean_increment = torch.full((2, 1, 1, 1), increment, dtype=dtype, device=device)
        spatial = 3 * (base - base.mean(axes, keepdim=True))
        actual = mean_shape_update(base, mean_increment, spatial)
        target = capacity_update(base.mean(axes, keepdim=True), mean_increment)
        error = max(_maximum(actual.mean(axes, keepdim=True) - target),
                    max(0., -(_scalar(actual.min()) or 0.)),
                    max(0., (_scalar(actual.max()) or 0.) - 1.))
        yield _row(f"helper/{precision}/mean_shape_target/{index}", error,
                   5e-7 if dtype == torch.float32 else 3e-15,
                   mean_increment=increment, target_mean=[_scalar(x) for x in target.flatten()],
                   output_min=_scalar(actual.min()), output_max=_scalar(actual.max()))

    # The moment map preserves identity exactly when both learned increments
    # vanish, including interior and endpoint fields.
    endpoints = torch.stack((torch.zeros_like(base[0]), torch.ones_like(base[0]), base[0]))
    identity = mean_shape_update(endpoints, torch.zeros_like(endpoints[:, :, :1, :1]),
                                 torch.zeros_like(endpoints))
    yield _row(f"helper/{precision}/mean_shape_zero_identity", _maximum(identity - endpoints), 0.)


def _model_cases(dtype, device, protocol):
    from tdn.research.consistency_neural import build_model

    precision = str(dtype).split(".")[-1]
    geometry = Geometry((8, 8), (1., 1.))
    rough = _state(geometry.grid, dtype=dtype, device=device)
    equation = Equation(.003, 2.)
    constant_values = (0., .37, .83, 1.)
    constant = torch.stack(tuple(torch.full_like(rough[0], value) for value in constant_values))
    cases = (
        ("constant", constant, .09, equation),
        ("zero_reaction", rough, .09, Equation(.003, 0.)),
        ("zero_diffusion", rough, .09, Equation(0., 2.)),
        ("zero_horizon", rough, 0., equation),
    )
    for index, family in enumerate(protocol["families"]):
        # Constructor RNG use is isolated too; this audit must not affect a
        # later paired training initialization in the same process.
        cuda_devices = [torch.device(device).index or torch.cuda.current_device()] if str(device).startswith("cuda") else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(48191 + index)
            model = build_model(family, width=protocol["width"], modes=protocol["modes"],
                                t_ref=protocol["t_ref"], U_ref=protocol["U_ref"])
        model = _randomize(model.to(device=device, dtype=dtype), 19231 + index).eval()
        parameter_norm = sum(float(parameter.detach().double().square().sum().cpu())
                             for parameter in model.parameters()) ** .5
        constrained = family not in ("premix", "fno", "precompress")
        prefix = f"model/{family}/{precision}"
        for condition, state, horizon, parameters in cases:
            with torch.no_grad():
                actual = model(state, horizon, parameters, geometry)
                expected = split_step(state, horizon, parameters, geometry)
            required = constrained or condition == "zero_horizon"
            tolerance = 5e-7 if dtype == torch.float32 else 3e-14
            yield _row(prefix + "/" + condition, _maximum(actual - expected), tolerance,
                       required=required, family=family, dtype=precision,
                       parameter_initialization="all learned weights sampled nonzero; local seed",
                       parameter_l2_norm=parameter_norm,
                       homogeneous_values=list(constant_values) if condition == "constant" else None,
                       claim="equals physical split for commuting/zero-time limits" if required else "unconstrained control observation")

        value = rough.clone().requires_grad_()
        horizon = torch.tensor([.04, .09], dtype=dtype, device=device, requires_grad=True)
        result = model(value, horizon, equation, geometry)
        result.square().mean().backward()
        missing, nonfinite = [], []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                missing.append(name)
            elif not bool(torch.isfinite(parameter.grad).all()):
                nonfinite.append(name)
        finite = (bool(torch.isfinite(result).all()) and value.grad is not None
                  and bool(torch.isfinite(value.grad).all()) and horizon.grad is not None
                  and bool(torch.isfinite(horizon.grad).all()) and not missing and not nonfinite)
        yield _row(prefix + "/finite_gradients", 0. if finite else None, 0., family=family,
                   missing_parameter_gradients=missing, nonfinite_parameter_gradients=nonfinite,
                   state_gradient_rms=_scalar(value.grad.double().square().mean().sqrt()) if value.grad is not None else None,
                   time_gradients=[_scalar(x) for x in horizon.grad] if horizon.grad is not None else None,
                   output_min=_scalar(result.min()), output_max=_scalar(result.max()))
        interval_error = max(0., -(_scalar(result.min()) or 0.), (_scalar(result.max()) or 0.) - 1.)
        yield _row(prefix + "/bounded_output", interval_error, 5e-7 if dtype == torch.float32 else 3e-14,
                   family=family, output_min=_scalar(result.min()), output_max=_scalar(result.max()))
        if family in ("premix_moment", "fno_moment"):
            with torch.no_grad():
                terms = model.correction_components(rough, horizon.detach(), equation, geometry)
                final = model(rough, horizon.detach(), equation, geometry)
            target = terms["target_mean"]
            measured = final.mean((-2, -1), keepdim=True)
            yield _row(prefix + "/final_mean_target", _maximum(measured - target),
                       5e-7 if dtype == torch.float32 else 3e-15, family=family,
                       target_mean=[_scalar(x) for x in target.flatten()],
                       final_mean=[_scalar(x) for x in measured.flatten()],
                       claim="bounded learned target; not exact mean ODE or mean conservation")


def _cases(protocol, device):
    for dtype in (torch.float32, torch.float64):
        yield from _helper_cases(dtype, device, protocol)
        yield from _model_cases(dtype, device, protocol)


def audit(protocol, run_dir, *, device="cpu", stop=None):
    """Run the frozen structural audit; failed or interrupted runs remain unsealed."""
    from .engine import _begin, _manifest, _status, _summary, _table
    from .protocol import validate_protocol

    protocol = deepcopy(protocol)
    validate_protocol(protocol)
    if str(device) not in ("cpu", "cuda"):
        raise ValueError("Consistency audit device must be cpu or cuda")
    directory = Path(run_dir)
    _begin(directory, protocol)
    started = time.monotonic()
    rows, error, status = [], None, "COMPLETED"
    try:
        iterator = iter(_cases(protocol, device))
        while True:
            requested = False
            if stop is not None:
                requested = stop() if callable(stop) else stop.requested
            if requested:
                signal_number = getattr(stop, "signal_number", None)
                raise InterruptedError(f"Consistency structural audit was interrupted by signal {signal_number}")
            if time.monotonic() - started >= protocol["audit_seconds"]:
                raise TimeoutError("Consistency structural audit exhausted its wall-time budget")
            try:
                row = next(iterator)
            except StopIteration:
                break
            rows.append(row)
            if row["outcome"] == "FAILED":
                raise RuntimeError(f"Consistency structural audit failed: {row['case_id']}")
        actual_ids = [row["check_id"] for row in rows]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(protocol["audit_case_ids"]):
            raise RuntimeError("Consistency structural audit coverage differs from the frozen check plan")
        required_ids = {row["check_id"] for row in rows if row["required"]}
        if required_ids != set(protocol["audit_required_case_ids"]):
            raise RuntimeError("Consistency structural audit requirements differ from the frozen check plan")
    except InterruptedError as failure:
        error, status = failure, _status(failure)
    except TimeoutError as failure:
        error, status = failure, _status(failure)
    except KeyboardInterrupt as failure:
        error, status = failure, _status(failure)
    except Exception as failure:
        error, status = failure, "FAILED"
        if not rows or rows[-1]["outcome"] != "FAILED":
            rows.append(dict(check_id="audit/runtime_failure", case_id="audit/runtime_failure",
                             required=True, status="FAILED", outcome="FAILED",
                             maximum_absolute_error=None, absolute_tolerance=None,
                             error=f"{type(failure).__name__}: {failure}"))
    _table(directory / "checks.json", rows)
    write_json(directory / "audit.json", dict(schema=SCHEMA, stage="audit",
               protocol_sha256=_digest(protocol), status=status, rows=rows,
               claim_scope="Training-free finite-case architecture identities; no performance or theorem claim"))
    counts = dict(Counter(row["outcome"] for row in rows))
    summary = _summary(directory, protocol, stage="audit", status=status, device=str(device),
                       start=started, error=error, case_count=len(rows), outcomes=counts,
                       coverage=dict(expected=len(protocol["audit_case_ids"]), reported=len(rows)),
                       counts=dict(checks=len(rows), required=sum(row["required"] for row in rows)),
                       correctness_failures=counts.get("FAILED", 0),
                       arbitrary_weights=True, training_performed=False)
    if error is not None:
        raise error
    _manifest(directory, ("protocol.json", "checks.json", "audit.json", "summary.json"),
              protocol, "audit_manifest.json")
    return summary


__all__ = ["audit"]
