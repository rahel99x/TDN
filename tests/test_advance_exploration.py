"""Mathematical and causal safeguards for the protected exploration branch."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tdn.analysis.advance import exploration as e


def test_nonaliased_convolution_matches_oversampled_physical_product():
    a, b = e.field(88000001, 7), e.field(88000002, 7)
    modes = np.arange(-7, 8)
    av, bv = np.zeros(61, complex), np.zeros(61, complex)
    av[modes % 61], bv[modes % 61] = a, b
    physical = np.fft.ifft(av, norm="forward")*np.fft.ifft(bv, norm="forward")
    expected = np.fft.fft(physical, norm="forward")[modes % 61]
    np.testing.assert_allclose(e.convolution(a, b), expected, atol=1e-16)


def test_real_coordinates_preserve_phase_reality_and_mean():
    a = e.field(88000003)
    np.testing.assert_array_equal(e.complex_coordinates(e.real_coordinates(a)), a)
    with pytest.raises(ValueError):
        e.complex_coordinates(np.ones(4))


def test_galerkin_mean_identity_and_logistic_constant_solution():
    a = e.field(88000004)
    m = a[len(a)//2].real
    variance = np.sum(abs(a)**2)-m*m
    assert abs(e.rhs(a)[len(a)//2]-.7*(m-m*m-variance)) < 1e-15
    a *= 0; a[len(a)//2] = .4
    sol, _ = e.integrate(a, .3)
    expected = .4*np.exp(.7*.3)/(1+.4*np.expm1(.7*.3))
    assert abs(sol.y[len(a)//2, -1]-expected) < 1e-12
    with pytest.raises(ValueError, match="forward"):
        e.integrate(a, -.01)


def test_variational_rhs_matches_independent_centered_difference():
    a, v = e.field(88000005), e.field(88000006)
    epsilon = 1e-6
    finite_difference = (e.rhs(a+epsilon*v)-e.rhs(a-epsilon*v))/(2*epsilon)
    np.testing.assert_allclose(e.tangent_rhs(a, v), finite_difference, atol=2e-11, rtol=1e-8)


def test_variational_solution_matches_perturbed_forward_solutions():
    a, v = e.field(88000007), e.field(88000008)
    v[len(v)//2] = 0
    coupled, _ = e.integrate(a, .15, tangent=v)
    plus, _ = e.integrate(a+1e-5*v, .15)
    minus, _ = e.integrate(a-1e-5*v, .15)
    expected = (plus.y[:, -1]-minus.y[:, -1])/2e-5
    np.testing.assert_allclose(coupled.y[len(a):, -1], expected, atol=2e-10, rtol=2e-8)


@pytest.mark.parametrize("method", ["exponential_etd1", "resolvent_backward_euler", "pade_11", "pade_02"])
def test_rational_responses_preserve_identity_and_zero_time(method):
    z = np.r_[0., -np.logspace(-12, 5, 30)]
    exp, phi = e.rational_response(z, method)
    np.testing.assert_allclose(exp, 1+z*phi, atol=3e-16)
    assert exp[0] == phi[0] == 1.
    a = e.field(88000009)
    np.testing.assert_array_equal(e.etd_step(a, 0., .02, .7, method), a)
    with pytest.raises(ValueError, match="z <= 0"):
        e.rational_response(np.array([1.]), method)


def test_pade11_stiff_failure_is_retained_not_clipped():
    exp, _ = e.rational_response(np.array([-1e6]), "pade_11")
    assert exp[0] < -.999
    exp, _ = e.rational_response(np.array([-1e6]), "pade_02")
    assert 0 < exp[0] < 1e-10


def test_etd1_has_first_order_nonlinear_convergence():
    a = e.field(88000010)
    teacher, _ = e.integrate(a, .2)
    errors = []
    for steps in (8, 16, 32):
        current = a.copy()
        for _ in range(steps):
            current = e.etd_step(current, .2/steps, .02, .7, "exponential_etd1")
        errors.append(np.linalg.norm(current-teacher.y[:, -1]))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2


def test_memory_feature_is_causal_and_recovers_constant_increment():
    previous = e.field(88000011, 2)
    h = .01
    closure = e.complex_coordinates(np.array([-.002, .001, 0., 0., .001]))
    current = e.rk4(previous, h)+h*closure
    features = e._features(current, previous, h, .02, .7, True)
    np.testing.assert_allclose(features[-5:], e.real_coordinates(closure), atol=2e-15)


class _Budget:
    def check(self):
        pass


@pytest.mark.parametrize("prototype,filename", [("XH", "XH_history.json"),
    ("XR", "XR_resolvent.json"), ("XT", "XT_tangent.json")])
def test_bounded_prototypes_execute_with_raw_negative_and_positive_evidence(tmp_path, prototype, filename):
    recorded = []
    ctx = SimpleNamespace(protocol={"profile": "smoke"}, unit={"prototype_id": prototype},
        stage="test", device="cpu", path=tmp_path, budget=_Budget(),
        record=lambda *args, **kwargs: recorded.append((args, kwargs)))
    result = e.run(ctx)
    assert result["status"] == "COMPLETED"
    assert recorded
    raw = json.loads((tmp_path/filename).read_text())
    assert raw["rows"]
    rows = json.loads((tmp_path/"prototype_rows.json").read_text())
    assert all({"math", "gap"} <= {v["category"] for v in row["checks"]} for row in rows)
    assert all(row["assessment"]["proof_status"] == "NOT_A_PROOF" for row in rows)
    assert all(v["verdict"] != "BAD" for row in rows for v in row["checks"] if v["category"] == "math")
    provenance = json.loads((tmp_path/"exploration_provenance.json").read_text())
    assert len(provenance["raw_sha256"]) == len(provenance["source_sha256"]) == 64
    if prototype == "XH":
        ids = raw["parent_ids"]
        assert set(ids["train"]).isdisjoint(ids["validation"])
        assert set(ids["train"]).isdisjoint(ids["test"])
        assert set(ids["validation"]).isdisjoint(ids["test"])
        assert all(0 <= v["history_time"] < v["observed_time"] < v["target_time"] for v in raw["training_rows"])
        assert all(v["complete_seconds"] >= v["startup_seconds"] for v in raw["rows"])
        assert {v["method"] for v in raw["rows"]} == {
            "coarse_rk4", "state_persistence", "instantaneous_fit", "secant_memory", "fitted_memory"}
        assert {v["scenario"] for v in raw["rows"]} >= {"normal", "step_shift", "coefficient_shift", "history_noise_0.0001"}
    if prototype == "XR":
        assert any(v["method"] == "classical_dop853_dense" for v in raw["timing_rows"])
        teachers = {v["teacher_id"] for v in raw["teacher_records"]}
        assert len(teachers) == len(raw["teacher_records"])
        assert {v["teacher_id"] for v in raw["rows"]} == teachers
        pade = next(v for v in rows if v["method"] == "pade_11")
        assert next(v for v in pade["checks"] if v["check_id"] == "negative-axis-stiff-damping")["verdict"] == "BAD"
    if prototype == "XT":
        assert all(v["initial_coarse_tangent_norm"] == 0 for v in raw["rows"])
        assert all(v["coarse_future_difference"] > 1e-8 for v in raw["twin_rows"])
        assert len({v["solve_id"] for v in raw["solve_records"]}) == len(raw["solve_records"])
        assert len(raw["solve_records"]) == len(raw["parent_ids"])*(3*(1+2*3+2)+2)
        assert raw["total_solve_seconds"] == sum(v["seconds"] for v in raw["solve_records"])


def test_exploration_rejects_gpu_label_and_undeclared_prototype(tmp_path):
    with pytest.raises(ValueError, match="CPU"):
        e.run(SimpleNamespace(device="cuda"))
    with pytest.raises(ValueError, match="exactly one"):
        e.run(SimpleNamespace(device="cpu", path=tmp_path, protocol={"profile": "smoke"}, stage="test"))
