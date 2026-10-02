"""Independent coupled RK4 teachers with fixed counts and explicit uncertainty.

Counts are selected at a boundary, then frozen for h/state derivative audits.
The stability criterion is deliberately conservative; stability is not an
accuracy certificate. Teacher acceptance still requires three refinement levels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .invariants import validate_state
from .operators import broadcast_h, check_shape, rhs, weighted_norm
from .splitting import split_step
from .types import Equation, Geometry


REFERENCE_METHOD = "coupled_periodic_central_difference_fixed_count_rk4_three_levels"


@dataclass(frozen=True)
class ReferenceResult:
    state: Tensor
    uncertainty: float
    defect_norm: float
    accepted: bool
    substeps: int
    refinement_substeps: tuple[int, int, int]
    refinement_differences: tuple[float, float]
    observed_order: float | None
    converged: bool
    reason: str


def choose_substeps(h_max: float, equation: Equation, geometry: Geometry) -> int:
    """Use dt*(4*kappa*sum(dx^-2)+rate) <= 0.5 on declared [0,1].

    The worst central-difference diffusion eigenvalue and reaction Jacobian
    bound are added conservatively. 0.5 lies well inside the RK4 negative-real
    stability interval and also resolves the logistic timescale. Refinement
    controls accuracy separately. Do not call this from a differentiable h path.
    """
    if not math.isfinite(h_max) or h_max < 0:
        raise ValueError("maximum horizon must be finite and nonnegative")
    stiffness = 4 * equation.kappa * sum(dx**-2 for dx in geometry.dx) + equation.reaction_rate
    return max(1, math.ceil(2 * h_max * stiffness))


def reference_step(u: Tensor, h: float | Tensor, equation: Equation, geometry: Geometry,
                   substeps: int) -> Tensor:
    """Coupled RK4 at a caller-frozen count, fully differentiable in u and h."""
    check_shape(u, geometry)
    if type(substeps) is not int or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    step = broadcast_h(h, u) / substeps
    result = u
    for _ in range(substeps):
        k1 = rhs(result, equation, geometry)
        k2 = rhs(result + step * k1 / 2, equation, geometry)
        k3 = rhs(result + step * k2 / 2, equation, geometry)
        k4 = rhs(result + step * k3, equation, geometry)
        result = result + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return result


def refined_reference(u: Tensor, h: float | Tensor, equation: Equation, geometry: Geometry,
                      substeps: int, error_fraction: float = 0.05,
                      tolerance: float | None = None, noise_floor: float = 1e-10) -> ReferenceResult:
    """Three levels n,2n,4n; report the finest FP64 state and acceptance.

    The conservative uncertainty is the full last refinement difference, with
    a dtype rounding allowance; it is not reduced by a Richardson factor.
    Resolved differences must decrease by at least 4x (effective order >=2),
    or both differences must be beneath the measured FP64 subtraction floor.
    Require uncertainty <= error_fraction*max(defect_norm,noise_floor), and,
    when supplied, <= error_fraction*tolerance. An h=0 or commuting-control
    zero defect is accepted at the declared noise floor, not as a useful label.
    This routine is a no-grad teacher builder, separate from reference_step.
    """
    if type(substeps) is not int or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if not math.isfinite(error_fraction) or not 0 < error_fraction < 1:
        raise ValueError("teacher error fraction must lie strictly between 0 and 1")
    if not math.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError("noise_floor must be finite and positive")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("numerical tolerance must be finite and positive")
    check_shape(u, geometry)
    with torch.no_grad():
        initial = u.to(torch.float64)
        step = broadcast_h(h, initial)
        if not bool(torch.isfinite(step).all()) or bool((step < 0).any()):
            raise ValueError("teacher horizons must be finite and nonnegative")
        validate_state(initial)
        counts = (substeps, 2 * substeps, 4 * substeps)
        states = [reference_step(initial, step, equation, geometry, n) for n in counts]
        finite = all(bool(torch.isfinite(state).all()) for state in states)
        if not finite:
            return ReferenceResult(states[-1], math.inf, math.inf, False, counts[-1], counts,
                                   (math.inf, math.inf), None, False, "nonfinite_refinement")
        differences = tuple(float(weighted_norm(states[i + 1] - states[i], geometry)) for i in range(2))
        rounding = 64 * torch.finfo(torch.float64).eps * max(1.0, float(weighted_norm(states[-1], geometry)))
        first, last = differences
        at_floor = max(first, last) <= rounding
        converged = at_floor or last <= first / 4
        order = math.log2(first / last) if first > 0 and last > 0 else None
        uncertainty = max(last, rounding)
        base = split_step(initial, step, equation, geometry, differentiable=False)
        defect_norm = float(weighted_norm(states[-1] - base, geometry))
        defect_ok = uncertainty <= error_fraction * max(defect_norm, noise_floor)
        tolerance_ok = tolerance is None or uncertainty <= error_fraction * tolerance
        try:
            validate_state(states[-1])
            admissible = True
        except ValueError:
            admissible = False
        accepted = converged and defect_ok and tolerance_ok and admissible
        reasons = []
        if not converged:
            reasons.append("unresolved_refinement")
        if not defect_ok:
            reasons.append("uncertainty_exceeds_defect_budget")
        if not tolerance_ok:
            reasons.append("uncertainty_exceeds_tolerance_budget")
        if not admissible:
            reasons.append("inadmissible_teacher")
        return ReferenceResult(states[-1], uncertainty, defect_norm, accepted, counts[-1], counts,
                               differences, order, converged, ",".join(reasons) if reasons else "accepted")
