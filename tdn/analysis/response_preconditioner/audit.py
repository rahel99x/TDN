"""Independent, bounded certificates for a cubic trajectory of the FD equation.

This module does not import a neural model, the response implementation, or a
numerical reference solver.  It audits the exact-real periodic finite-difference
ODE specified by the supplied binary floating-point inputs.  Fraction arithmetic
makes every polynomial coefficient and Bernstein enclosure exact.  Decimal.exp
is correctly rounded (ROUND_HALF_EVEN, as specified by Python); next_plus then
makes its result an upper bound.  All subsequent bound arithmetic is directed.

The certificate is for ONE panel and for the infinity norm.  ``true_initial_state``
is the actual initial condition of that ODE at the start of this panel, not an
approximation returned by an earlier panel.  Propagating certificates through a
multi-panel trajectory requires a separately checked continuity/error contract
and is not implemented here.  No continuum, gradient, or statistical guarantee
is implied.  The small degree-of-freedom cap is intentional: this is an independent
research checker, not a claim of affordable online certification.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from math import comb, isfinite, prod
from typing import Sequence

import numpy as np

MAX_SPATIAL_DOF = 64
MAX_DIMENSIONS = 3
MAX_GROWTH_EXPONENT = Fraction(1000)


def _fraction(value: object, name: str) -> Fraction:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if not isinstance(value, (float, np.floating)):
        raise ValueError(f"{name} must be a finite binary float or integer")
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise ValueError(f"{name} cannot silently narrow extended precision")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return Fraction.from_float(number)


def _field(value: object, name: str) -> tuple[tuple[int, ...], tuple[Fraction, ...]]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real spatial array") from exc
    if not 1 <= array.ndim <= MAX_DIMENSIONS or any(n == 0 for n in array.shape):
        raise ValueError(f"{name} must have one to three nonempty spatial axes")
    if array.size > MAX_SPATIAL_DOF:
        raise ValueError(f"auditor is capped at {MAX_SPATIAL_DOF} spatial degrees of freedom")
    if array.dtype.kind not in "fiu" or (array.dtype.kind == "f" and array.dtype.itemsize > 8):
        raise ValueError(f"{name} must contain finite floats of at most 64 bits or integers")
    return array.shape, tuple(_fraction(x, name) for x in array.flat)


def _bernstein(power: Sequence[Fraction], degree: int) -> tuple[Fraction, ...]:
    """Convert power coefficients on [0,1] to the declared Bernstein degree."""
    return tuple(
        sum((power[j] * Fraction(comb(k, j), comb(degree, j))
             for j in range(min(k, len(power) - 1) + 1)), Fraction(0))
        for k in range(degree + 1)
    )


def _laplacian(
    fields: Sequence[Sequence[Fraction]], shape: tuple[int, ...],
    rates: Sequence[Fraction],
) -> tuple[tuple[Fraction, ...], ...]:
    """Apply kappa*n_axis**2/length_axis**2 times the periodic FD stencil."""
    strides = tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))
    result = []
    for index, values in enumerate(fields):
        entry = [Fraction(0) for _ in values]
        for size, stride, rate in zip(shape, strides, rates):
            coordinate = (index // stride) % size
            plus = index + stride if coordinate + 1 < size else index - (size - 1) * stride
            minus = index - stride if coordinate else index + (size - 1) * stride
            for degree in range(len(values)):
                entry[degree] += rate * (
                    fields[plus][degree] + fields[minus][degree] - 2 * values[degree]
                )
        result.append(tuple(entry))
    return tuple(result)


@dataclass(frozen=True)
class CubicFDPolynomials:
    """Exact coefficients in normalized time q=t/step, flattened in C order."""

    shape: tuple[int, ...]
    step: Fraction
    kappa: Fraction
    reaction: Fraction
    lengths: tuple[Fraction, ...]
    state_power: tuple[tuple[Fraction, ...], ...]
    residual_power: tuple[tuple[Fraction, ...], ...]
    state_bernstein: tuple[tuple[Fraction, ...], ...]
    residual_bernstein: tuple[tuple[Fraction, ...], ...]


def build_cubic_fd_polynomials(
    y0: object, y1: object, slope0: object, slope1: object, *,
    step: float, kappa: float, reaction: float, lengths: Sequence[float],
) -> CubicFDPolynomials:
    """Build exact cubic y and degree-six d=y'-Ly-r*y*(1-y).

    Slopes are derivatives with respect to physical time, not normalized time.
    Geometry uses dx=length/n exactly; it does not adopt a floating-point FFT
    symbol or assume that a previously computed floating-point stencil is exact.
    """
    shape, left = _field(y0, "y0")
    data = []
    for value, name in ((y1, "y1"), (slope0, "slope0"), (slope1, "slope1")):
        other_shape, values = _field(value, name)
        if other_shape != shape:
            raise ValueError(f"{name} shape must equal y0 shape")
        data.append(values)
    right, left_slope, right_slope = data
    h = _fraction(step, "step")
    diffusion = _fraction(kappa, "kappa")
    r = _fraction(reaction, "reaction")
    if h <= 0 or diffusion < 0 or r < 0:
        raise ValueError("step must be positive; kappa and reaction must be nonnegative")
    try:
        sides = tuple(_fraction(x, "lengths") for x in lengths)
    except TypeError as exc:
        raise ValueError("lengths must contain one positive length per spatial axis") from exc
    if len(sides) != len(shape) or any(x <= 0 for x in sides):
        raise ValueError("lengths must contain one positive length per spatial axis")
    state = tuple(
        (a, h * da, 3 * (b - a) - h * (2 * da + db),
         2 * (a - b) + h * (da + db))
        for a, b, da, db in zip(left, right, left_slope, right_slope)
    )
    rates = tuple(diffusion * size * size / side**2 for size, side in zip(shape, sides))
    linear = _laplacian(state, shape, rates)
    residual = []
    for coefficients, laplace in zip(state, linear):
        entry = [Fraction(0) for _ in range(7)]
        for degree, coefficient in enumerate(coefficients):
            entry[degree] -= laplace[degree] + r * coefficient
            if degree:
                entry[degree - 1] += degree * coefficient / h
            for other_degree, other in enumerate(coefficients):
                entry[degree + other_degree] += r * coefficient * other
        residual.append(tuple(entry))
    return CubicFDPolynomials(
        shape=shape, step=h, kappa=diffusion, reaction=r, lengths=sides,
        state_power=state, residual_power=tuple(residual),
        state_bernstein=tuple(_bernstein(row, 3) for row in state),
        residual_bernstein=tuple(_bernstein(row, 6) for row in residual),
    )


def _decimal(value: Fraction, precision: int, rounding: str) -> Decimal:
    with localcontext() as context:
        context.prec = precision
        context.rounding = rounding
        return Decimal(value.numerator) / Decimal(value.denominator)


def _error_bound(
    initial_error: Fraction, residual: Fraction, alpha: Fraction,
    step: Fraction, precision: int,
) -> tuple[Decimal, Decimal, Decimal]:
    """Upper bounds for E(h), exp(alpha*h), and h*phi_1(alpha*h)."""
    if not initial_error and not residual:
        # The exact zero-error solution needs no exponential evaluation.
        return Decimal(0), Decimal(1), Decimal(0)
    exponent = alpha * step
    if exponent > MAX_GROWTH_EXPONENT:
        raise ValueError("bounded auditor requires alpha*step <= 1000 for nonzero error")
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_CEILING
        if not exponent:
            growth = Decimal(1)
            integral = _decimal(step, precision, ROUND_CEILING)
        else:
            exponent_upper = _decimal(exponent, precision, ROUND_CEILING)
            # Decimal.exp rounds correctly to nearest regardless of context's
            # rounding mode.  Its next representable value is therefore above
            # the mathematical exp(exponent_upper), including tie cases.
            growth = exponent_upper.exp().next_plus(context)
            if exponent <= 1:
                # Stable positive series, with an exact geometric bound on its
                # infinite tail.  No subtraction near zero or floating-point
                # test is used in this proof.
                term = Fraction(1)
                series = term
                for j in range(1, 65):
                    term *= exponent / (j + 1)
                    series += term
                next_term = term * exponent / 66
                tail = next_term / (1 - exponent / 67)
                integral = _decimal(step * (series + tail), precision, ROUND_CEILING)
            else:
                # Denominator is rounded down, numerator and quotient up.
                exponent_lower = _decimal(exponent, precision, ROUND_FLOOR)
                phi_upper = (growth - 1) / exponent_lower
                integral = _decimal(step, precision, ROUND_CEILING) * phi_upper
        bound = (
            growth * _decimal(initial_error, precision, ROUND_CEILING)
            + integral * _decimal(residual, precision, ROUND_CEILING)
        )
    return bound, growth, integral


def audit_cubic_fd_panel(
    y0: object, y1: object, slope0: object, slope1: object, *,
    step: float, kappa: float, reaction: float, lengths: Sequence[float],
    true_initial_state: object, initial_error_bound: float = 0.0,
    decimal_precision: int = 80,
) -> dict[str, object]:
    """Certify one finite-grid cubic candidate against its declared true IVP.

    The initial state is checked to lie in [0,1], which proves invariance for
    this periodic Markov FD diffusion with nonnegative logistic reaction.  The
    supplied initial-error bound must cover the exact infinity difference from
    y0.  With y>=m, alpha=max(0,r*(1-m)) bounds the infinity logarithmic norm of
    the exact error equation.  The final bound consequently covers every time
    in the panel, as well as its endpoint.  Returned decimal strings are upper
    bounds; converting them to float without outward rounding forfeits that
    last-bit guarantee.
    """
    if (isinstance(decimal_precision, bool) or not isinstance(decimal_precision, int)
            or not 34 <= decimal_precision <= 200):
        raise ValueError("decimal_precision must be an integer between 34 and 200")
    polynomial = build_cubic_fd_polynomials(
        y0, y1, slope0, slope1, step=step, kappa=kappa,
        reaction=reaction, lengths=lengths,
    )
    initial_shape, initial = _field(true_initial_state, "true_initial_state")
    if initial_shape != polynomial.shape:
        raise ValueError("true_initial_state shape must equal y0 shape")
    if any(value < 0 or value > 1 for value in initial):
        raise ValueError("true_initial_state must lie in [0,1]; invariant assumption is unverified")
    error0 = _fraction(initial_error_bound, "initial_error_bound")
    actual_error0 = max(abs(row[0] - value) for row, value in zip(polynomial.state_power, initial))
    if error0 < 0 or error0 < actual_error0:
        raise ValueError("initial_error_bound must cover the exact initial infinity error")
    lower = min(min(row) for row in polynomial.state_bernstein)
    upper = max(max(row) for row in polynomial.state_bernstein)
    defect = max(max(abs(value) for value in row) for row in polynomial.residual_bernstein)
    alpha = max(Fraction(0), polynomial.reaction * (1 - lower))
    bound, growth, integral = _error_bound(error0, defect, alpha, polynomial.step, decimal_precision)
    return {
        "status": "CERTIFIED_FINITE_GRID_PANEL",
        "certificate_proves": "upper_bound_not_tolerance_acceptance",
        "tolerance_checked": False,
        "scope": "single_panel_periodic_fd_infinity_norm",
        "spatial_shape": list(polynomial.shape),
        "degrees_of_freedom": prod(polynomial.shape),
        "reconstruction_degree": 3,
        "residual_degree_bound": 6,
        "residual_actual_degree": max(
            (index for row in polynomial.residual_power for index, value in enumerate(row) if value),
            default=0,
        ),
        "reconstruction_lower_bound": str(_decimal(lower, decimal_precision, ROUND_FLOOR)),
        "reconstruction_upper_bound": str(_decimal(upper, decimal_precision, ROUND_CEILING)),
        "residual_bound_infinity": str(_decimal(defect, decimal_precision, ROUND_CEILING)),
        "endpoint_error_bound_infinity": str(bound),
        "whole_panel_error_bound_infinity": str(bound),
        "exponential_growth_upper": str(growth) if error0 or defect else None,
        "duhamel_integral_upper": str(integral) if error0 or defect else None,
        "decimal_precision": decimal_precision,
        "arithmetic": "exact rational coefficients; correctly-rounded Decimal.exp enlarged by next_plus; directed bound arithmetic",
        "target": "exact-real periodic FD equation from supplied binary floating-point values; no continuum claim",
        "initial_invariant_checked": True,
        "multi_panel_propagation_supported": False,
        "exact": {
            "step": str(polynomial.step), "kappa": str(polynomial.kappa),
            "reaction": str(polynomial.reaction),
            "lengths": [str(value) for value in polynomial.lengths],
            "initial_error_bound": str(error0), "actual_initial_error": str(actual_error0),
            "reconstruction_lower": str(lower), "reconstruction_upper": str(upper),
            "residual_bound_infinity": str(defect), "logarithmic_norm_rate": str(alpha),
            "state_power_coefficients": [[str(value) for value in row] for row in polynomial.state_power],
            "residual_power_coefficients": [[str(value) for value in row] for row in polynomial.residual_power],
            "state_bernstein_coefficients": [[str(value) for value in row] for row in polynomial.state_bernstein],
            "residual_bernstein_coefficients": [[str(value) for value in row] for row in polynomial.residual_bernstein],
        },
    }
