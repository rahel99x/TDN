"""Independent mathematical checks for the bounded finite-grid auditor.

Dense samples below are test oracles for implementation errors.  The actual
certificate uses exact Bernstein control coefficients, not sampled maxima.
"""
from decimal import Decimal, localcontext
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from tdn.analysis.response_preconditioner.audit import (
    MAX_SPATIAL_DOF,
    audit_cubic_fd_panel,
    build_cubic_fd_polynomials,
)


def _options(**updates):
    result = dict(step=0.25, kappa=0.1, reaction=1.2, lengths=(1.0,))
    result.update(updates)
    return result


def _value(coefficients, point):
    return sum(coefficient * point**degree for degree, coefficient in enumerate(coefficients))


def _mp_fraction(value):
    return mp.mpf(value.numerator) / value.denominator


@pytest.mark.parametrize("stationary", [0.0, 1.0])
def test_exact_stationary_fields_have_zero_certified_error(stationary):
    state = np.full((3, 2), stationary)
    certificate = audit_cubic_fd_panel(
        state, state, np.zeros_like(state), np.zeros_like(state),
        true_initial_state=state, **_options(lengths=(1.5, 2.0)),
    )
    assert certificate["status"] == "CERTIFIED_FINITE_GRID_PANEL"
    assert certificate["certificate_proves"] == "upper_bound_not_tolerance_acceptance"
    assert certificate["tolerance_checked"] is False
    assert Decimal(certificate["endpoint_error_bound_infinity"]) == 0
    assert Decimal(certificate["whole_panel_error_bound_infinity"]) == 0
    assert Fraction(certificate["exact"]["residual_bound_infinity"]) == 0
    assert certificate["initial_invariant_checked"] is True
    assert certificate["multi_panel_propagation_supported"] is False


def test_cubic_endpoints_slopes_and_degree_six_are_exact():
    left = np.array([0.125, 0.375])
    right = np.array([0.1875, 0.3125])
    slope0 = np.array([0.25, -0.125])
    slope1 = np.array([-0.25, 0.125])
    polynomial = build_cubic_fd_polynomials(left, right, slope0, slope1, **_options())
    assert all(len(row) == 7 for row in polynomial.residual_power)
    for i, row in enumerate(polynomial.state_power):
        assert row[0] == Fraction(float(left[i]))
        assert sum(row) == Fraction(float(right[i]))
        assert row[1] / polynomial.step == Fraction(float(slope0[i]))
        assert sum(j * row[j] for j in range(1, 4)) / polynomial.step == Fraction(float(slope1[i]))
        assert polynomial.residual_power[i][6] == polynomial.reaction * row[3] ** 2
    assert any(row[6] != 0 for row in polynomial.residual_power)


def test_exact_bernstein_enclosures_contain_entire_sampled_polynomials():
    rng = np.random.default_rng(7182)
    values = [rng.uniform(-0.1, 0.9, (3, 2)) for _ in range(4)]
    polynomial = build_cubic_fd_polynomials(*values, **_options(lengths=(0.75, 1.25)))
    for power_rows, bernstein_rows in (
        (polynomial.state_power, polynomial.state_bernstein),
        (polynomial.residual_power, polynomial.residual_bernstein),
    ):
        for power, controls in zip(power_rows, bernstein_rows):
            for index in range(42):
                value = _value(power, Fraction(index, 41))
                assert min(controls) <= value <= max(controls)


def test_residual_matches_independent_stencil_and_differentiation():
    rng = np.random.default_rng(142)
    values = [rng.normal(size=(2, 3)) for _ in range(4)]
    parameters = _options(step=0.125, kappa=0.03125, reaction=0.75, lengths=(0.5, 1.25))
    polynomial = build_cubic_fd_polynomials(*values, **parameters)
    for q in np.linspace(0, 1, 51):
        y = np.array([float(_value(row, q)) for row in polynomial.state_power]).reshape(2, 3)
        derivative = np.array([
            sum(j * float(row[j]) * q ** (j - 1) for j in range(1, 4)) / parameters["step"]
            for row in polynomial.state_power
        ]).reshape(2, 3)
        laplace = np.zeros_like(y)
        for axis, (size, length) in enumerate(zip(y.shape, parameters["lengths"])):
            laplace += parameters["kappa"] * (size / length) ** 2 * (
                np.roll(y, 1, axis) + np.roll(y, -1, axis) - 2 * y
            )
        expected = derivative - laplace - parameters["reaction"] * y * (1 - y)
        actual = np.array([float(_value(row, q)) for row in polynomial.residual_power]).reshape(2, 3)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("shape", [(1,), (2,), (2, 1, 2)])
def test_singleton_and_two_cell_axes_follow_periodic_fd(shape):
    state = np.arange(np.prod(shape), dtype=float).reshape(shape) / 8
    zeros = np.zeros(shape)
    options = _options(kappa=1.0, reaction=0.0, lengths=tuple(float(n) for n in shape))
    polynomial = build_cubic_fd_polynomials(state, state, zeros, zeros, **options)
    expected = np.zeros(shape)
    for axis in range(len(shape)):
        expected -= np.roll(state, 1, axis) + np.roll(state, -1, axis) - 2 * state
    assert [row[0] for row in polynomial.residual_power] == [Fraction(float(x)) for x in expected.flat]
    assert all(all(value == 0 for value in row[1:]) for row in polynomial.residual_power)


@pytest.mark.parametrize("step,reaction", [(0.4, 1.2), (1e-10, 1.2), (1.5, 2.0)])
def test_certificate_bounds_analytic_logistic_error_at_all_sampled_times(step, reaction):
    initial = np.array([0.2])
    # This deliberately inaccurate endpoint makes the test independent of any
    # solver whose implementation the auditor could accidentally share.
    end = np.array([0.31])
    slope0 = reaction * initial * (1 - initial)
    slope1 = reaction * end * (1 - end)
    options = _options(step=step, reaction=reaction, kappa=0.25)
    certificate = audit_cubic_fd_panel(
        initial, end, slope0, slope1, true_initial_state=initial, **options,
    )
    with mp.workdps(100):
        bound = mp.mpf(certificate["whole_panel_error_bound_infinity"])
        coefficients = [Fraction(x) for x in certificate["exact"]["state_power_coefficients"][0]]
        initial_exact = _mp_fraction(Fraction(float(initial[0])))
        r = _mp_fraction(Fraction(reaction))
        h = _mp_fraction(Fraction(step))
        for index in range(101):
            q = mp.mpf(index) / 100
            truth = initial_exact / (initial_exact + (1 - initial_exact) * mp.exp(-r * h * q))
            candidate = sum(_mp_fraction(c) * q**j for j, c in enumerate(coefficients))
            assert abs(candidate - truth) <= bound
        exact = certificate["exact"]
        alpha = _mp_fraction(Fraction(exact["logarithmic_norm_rate"]))
        defect = _mp_fraction(Fraction(exact["residual_bound_infinity"]))
        mathematical_bound = h * defect if alpha == 0 else mp.expm1(alpha * h) / alpha * defect
        assert bound >= mathematical_bound


def test_zero_reaction_growth_limit_and_nonzero_initial_error():
    state = np.array([0.25, 0.5])
    truth = np.array([0.125, 0.5])
    certificate = audit_cubic_fd_panel(
        state, state, np.zeros(2), np.zeros(2), true_initial_state=truth,
        initial_error_bound=0.125, **_options(kappa=0.0, reaction=0.0),
    )
    assert Decimal(certificate["whole_panel_error_bound_infinity"]) == Decimal("0.125")
    assert Decimal(certificate["exponential_growth_upper"]) == 1
    assert Decimal(certificate["duhamel_integral_upper"]) == Decimal("0.25")


def test_negative_reconstruction_uses_enclosed_lower_bound_not_assumed_nonnegativity():
    left = np.array([0.25])
    end = np.array([-0.5])
    certificate = audit_cubic_fd_panel(
        left, end, np.zeros(1), np.zeros(1), true_initial_state=left, **_options(),
    )
    exact = certificate["exact"]
    lower = Fraction(exact["reconstruction_lower"])
    assert lower <= Fraction(-1, 2)
    assert Fraction(exact["logarithmic_norm_rate"]) == Fraction(1.2) * (1 - lower)


def test_decimal_outputs_round_outwards_at_low_allowed_precision():
    initial = np.array([0.2, 0.3])
    end = np.array([0.1, 0.7])
    certificate = audit_cubic_fd_panel(
        initial, end, np.zeros(2), np.ones(2), true_initial_state=initial,
        decimal_precision=34, **_options(step=0.8, reaction=2.0),
    )
    with localcontext() as context:
        context.prec = 120
        for output_key, exact_key, direction in (
            ("reconstruction_lower_bound", "reconstruction_lower", -1),
            ("reconstruction_upper_bound", "reconstruction_upper", 1),
            ("residual_bound_infinity", "residual_bound_infinity", 1),
        ):
            exact = Fraction(certificate["exact"][exact_key])
            reported = Fraction(Decimal(certificate[output_key]))
            assert direction * (reported - exact) >= 0


@pytest.mark.parametrize("name,value", [
    ("step", 0.0), ("step", -1.0), ("step", float("inf")),
    ("kappa", -0.1), ("reaction", -0.1), ("reaction", float("nan")),
    ("lengths", (0.0,)), ("lengths", (1.0, 1.0)),
    ("decimal_precision", 20), ("decimal_precision", True),
    ("initial_error_bound", -1.0),
])
def test_rejects_invalid_parameters(name, value):
    state = np.array([0.25])
    options = _options()
    options[name] = value
    with pytest.raises(ValueError):
        audit_cubic_fd_panel(state, state, np.zeros(1), np.zeros(1), true_initial_state=state, **options)


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf]), np.array([1j]),
                                  np.array([True]), np.array(0.2), np.zeros((0,)),
                                  np.zeros((1, 1, 1, 1)), np.zeros(MAX_SPATIAL_DOF + 1)])
def test_rejects_invalid_fields_and_work_above_cap(bad):
    with pytest.raises(ValueError):
        audit_cubic_fd_panel(bad, bad, bad, bad, true_initial_state=bad, **_options())


def test_rejects_unproved_invariance_shape_and_initial_error_assumptions():
    state = np.array([0.25])
    zero = np.zeros(1)
    for invalid in (np.array([-0.1]), np.array([1.01]), np.ones(2)):
        with pytest.raises(ValueError):
            audit_cubic_fd_panel(state, state, zero, zero, true_initial_state=invalid, **_options())
    with pytest.raises(ValueError, match="initial_error_bound"):
        audit_cubic_fd_panel(state, state, zero, zero, true_initial_state=np.array([0.5]), **_options())
    with pytest.raises(ValueError, match="shape"):
        audit_cubic_fd_panel(state, np.ones(2), zero, zero, true_initial_state=state, **_options())


def test_growth_work_limit_rejects_nonzero_case_without_fabricating_certificate():
    state = np.array([0.25])
    with pytest.raises(ValueError, match="alpha\\*step"):
        audit_cubic_fd_panel(state, state, np.zeros(1), np.zeros(1),
                             true_initial_state=state, **_options(step=10000.0))


def test_high_precision_independent_heat_solution_respects_full_panel_bound():
    initial = np.array([0.125, 0.75, 0.25])
    candidate = np.array([0.375, 0.375, 0.375])
    slopes = np.zeros(3)
    certificate = audit_cubic_fd_panel(
        initial, candidate, slopes, slopes, true_initial_state=initial,
        **_options(step=0.5, kappa=0.125, reaction=0.0),
    )
    # On the three-node periodic graph the two nonconstant eigenvalues are
    # exactly -3*kappa*n**2/length**2.  This independent spectral formula does
    # not call the audited FD implementation.
    with mp.workdps(100):
        initial_exact = [_mp_fraction(Fraction(float(value))) for value in initial]
        mean = sum(initial_exact) / 3
        bound = mp.mpf(certificate["whole_panel_error_bound_infinity"])
        rows = [[Fraction(value) for value in row]
                for row in certificate["exact"]["state_power_coefficients"]]
        for index in range(41):
            q = mp.mpf(index) / 40
            for value, row in zip(initial_exact, rows):
                truth = mean + (value - mean) * mp.exp(-mp.mpf(27) / 8 * mp.mpf("0.5") * q)
                reconstructed = sum(_mp_fraction(coefficient) * q**j for j, coefficient in enumerate(row))
                assert abs(truth - reconstructed) <= bound


def test_nonpositive_logarithmic_rate_is_safely_replaced_by_zero():
    state = np.array([1.125])
    certificate = audit_cubic_fd_panel(
        state, state, np.zeros(1), np.zeros(1), true_initial_state=np.array([1.0]),
        initial_error_bound=0.125, **_options(),
    )
    assert Fraction(certificate["exact"]["logarithmic_norm_rate"]) == 0
    assert Decimal(certificate["exponential_growth_upper"]) == 1


def test_caller_decimal_rounding_mode_does_not_change_certificate():
    from decimal import ROUND_DOWN

    state = np.array([0.2])
    with localcontext() as context:
        context.prec = 7
        context.rounding = ROUND_DOWN
        inherited = audit_cubic_fd_panel(
            state, np.array([0.3]), np.zeros(1), np.zeros(1),
            true_initial_state=state, **_options(),
        )
    standard = audit_cubic_fd_panel(
        state, np.array([0.3]), np.zeros(1), np.zeros(1),
        true_initial_state=state, **_options(),
    )
    assert inherited == standard
