"""R7 reference semantics and bounded controls; tiny CPU fixtures only."""
import json
import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from tdn.analysis.response import numerics as n
from tdn.analysis.response_preconditioner import r7_references as r
from tdn.analysis.tangent_response.oracle import independent_solve
from tdn.analysis.tangent_response.tangent import AnchorPath


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case():
    x = torch.arange(5, dtype=torch.float64).reshape(1, 1, 5)
    return (.31+.11*torch.cos(1.3*x)).float(), torch.sin(1.8*x)+.2, n.Physics(.023, 1.4), n.Geometry((5,), (1.,))


def config(**updates):
    return dict(r.DEFAULT_REFERENCE, **updates)


def test_nonzero_time_reference_matches_analytic_uniform_original_anchor():
    u, z, eq, geom = case()
    u = torch.full_like(u, .31)
    start, stop = .09, .23
    result = r.acquire_reference(u, z, start, stop, eq, geom, config())
    scalar = float(u.flatten()[0])
    def derivative(time):
        decay = math.exp(-eq.reaction_rate*time)
        return decay/(scalar+(1-scalar)*decay)**2
    columns = []
    for index in range(5):
        basis = np.eye(5)[:, index]
        columns.append(eq.kappa*(np.roll(basis, 1)-2*basis+np.roll(basis, -1))/geom.dx[0]**2)
    expected = (expm((stop-start)*np.stack(columns, axis=1))@z.numpy().ravel())*derivative(stop)/derivative(start)
    assert result["reference"]["accepted"]
    assert result["reference"]["matched_mean"]["accepted"]
    np.testing.assert_allclose(result["teacher"].numpy().ravel(), expected, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(result["teacher"], result["matched_mean"], rtol=1e-12, atol=1e-13)
    assert result["reference"]["start"] == start
    json.dumps(result["reference"], allow_nan=False)


def test_heterogeneous_nonzero_interval_preserves_original_anchor_and_mean_distinction():
    u, z, eq, geom = case()
    result = r.acquire_reference(u, z, .04, .18, eq, geom, config())
    assert result["reference"]["accepted"]
    assert result["reference"]["matched_mean"]["accepted"]
    exact, _ = independent_solve(u.double(), .18, eq, geom, r.TARGET, kind="action", z=z, s=.04)
    torch.testing.assert_close(result["teacher"], exact, rtol=0., atol=0.)
    restart = AnchorPath(u.double(), eq, geom, r.TARGET)(.04)
    wrong, _ = independent_solve(restart, .14, eq, geom, r.TARGET, kind="action", z=z)
    assert float((result["teacher"]-wrong).square().mean().sqrt()) > 100*result["reference"]["uncertainty_rms"]
    assert float((result["teacher"]-result["matched_mean"]).square().mean().sqrt()) > 1e-3
    assert result["reference"]["source_query_count"] == 1
    assert result["reference"]["action_modes_attempted"] == 2


def test_refinement_cap_retains_last_candidate_without_admitting_teacher():
    u, z, eq, geom = case()
    result = r.acquire_reference(u, z, .02, .08, eq, geom, config(initial_steps=4, max_steps=4))
    assert result["refined"] is not None
    assert result["teacher"] is None
    assert not result["reference"]["accepted"]
    assert result["reference"]["reason"] == "REFINEMENT_CAP"
    assert result["reference"]["work"]["refined_rhs_calls_started"] == 16
    assert result["reference"]["work"]["independent_rhs_calls_started"] == 0


def test_independent_rhs_cap_is_enforced_and_retained():
    u, z, eq, geom = case()
    result = r.acquire_reference(u, z, .02, .08, eq, geom, config(independent_max_rhs_calls=1))
    assert result["refined"] is not None
    assert result["teacher"] is None
    assert not result["reference"]["accepted"]
    assert "RHS allowance" in result["reference"]["reason"]
    assert result["reference"]["work"]["independent_rhs_calls_started"] == 1
    assert result["reference"]["matched_mean"]["work"]["independent_rhs_calls_started"] == 1


def test_outer_budget_exhaustion_preserves_both_missing_denominators():
    class Exhausted:
        def check(self):
            raise TimeoutError("study allowance")
    u, z, eq, geom = case()
    result = r.acquire_reference(u, z, .02, .08, eq, geom, config(), budget=Exhausted())
    assert result["teacher"] is None and result["refined"] is None
    assert result["matched_mean"] is None
    assert not result["reference"]["accepted"]
    assert "study allowance" in result["reference"]["matched_mean"]["reason"]
    assert result["reference"]["work"]["refined_rhs_calls_started"] == 0


def test_shared_wall_deadline_reaches_independent_rhs(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(r.time, "monotonic", lambda: clock[0])
    real = r.independent_solve
    def slow(*args, **kwargs):
        clock[0] = 21.
        return real(*args, **kwargs)
    monkeypatch.setattr(r, "independent_solve", slow)
    u, z, eq, geom = case()
    result = r.acquire_reference(u, z, .02, .08, eq, geom, config())
    assert result["teacher"] is None
    assert "wall allowance" in result["reference"]["reason"]
    assert result["reference"]["matched_mean"]["work"]["refined_rhs_calls_started"] == 0
    assert not result["reference"]["matched_mean"]["accepted"]


def test_acceptance_requires_maximum_as_well_as_rms(monkeypatch):
    real = r.independent_solve
    def localized_discrepancy(*args, **kwargs):
        value, metadata = real(*args, **kwargs)
        value = value.clone()
        value.flatten()[0] += 1.4e-10
        return value, metadata
    monkeypatch.setattr(r, "independent_solve", localized_discrepancy)
    u, z, eq, geom = case()
    result = r.acquire_reference(u, z, .02, .08, eq, geom, config())
    reference = result["reference"]
    assert reference["uncertainty_rms"] < 1e-10
    assert reference["uncertainty_maximum"] > 1e-10
    assert not reference["accepted"]
    assert reference["reason"] == "COMBINED_ACTION_BUDGET_UNRESOLVED"
    assert result["teacher"] is not None


def test_fixed_lawson_is_fresh_source_linear_and_converges_at_nonzero_time():
    u, z, eq, geom = case()
    truth, _ = independent_solve(u.double(), .23, eq, geom, r.TARGET, kind="action", z=z, s=.04)
    errors = []
    for steps in (1, 2, 4):
        work = {}
        value = r.fixed_lawson(u, z, .04, .23, eq, geom, steps, work=work)
        errors.append(float((value-truth).square().mean().sqrt()))
        assert work["fresh_original_anchor"]
        assert work["path"]["anchor_evaluations"] == 4*steps
        assert work["path"]["cache_hits"] == 0
        assert work["integration_heat_actions"] == 7*steps
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-8
    other = torch.cos(z)
    lhs = r.fixed_lawson(u, z-1.7*other, .04, .23, eq, geom, 3)
    rhs = r.fixed_lawson(u, z, .04, .23, eq, geom, 3)-1.7*r.fixed_lawson(u, other, .04, .23, eq, geom, 3)
    torch.testing.assert_close(lhs, rhs, rtol=2e-14, atol=2e-14)


def test_fixed_lawson_complete_path_derivative_matches_finite_difference():
    u, z, eq, geom = case()
    u = u.double().requires_grad_(True)
    delta = torch.cos(z)
    action = lambda state: r.fixed_lawson(state, z, .04, .13, eq, geom, 3)
    _, tangent = torch.autograd.functional.jvp(action, u, delta)
    epsilon = 1e-5
    finite = (action(u+epsilon*delta)-action(u-epsilon*delta))/(2*epsilon)
    assert float(tangent.norm()) > 1e-3
    torch.testing.assert_close(tangent, finite, rtol=1e-8, atol=2e-10)


@pytest.mark.parametrize("update", ({"max_seconds_per_query": 0}, {"max_steps": True},
                                  {"initial_steps": 16, "max_steps": 8},
                                  {"independent_max_rhs_calls": -1}, {"tolerance": math.nan}))
def test_invalid_reference_work_is_rejected(update):
    u, z, eq, geom = case()
    with pytest.raises(ValueError):
        r.acquire_reference(u, z, 0., .08, eq, geom, config(**update))


@pytest.mark.parametrize("interval", ((.1, .04), (-.1, .04), (0., math.inf)))
def test_invalid_source_intervals_are_rejected(interval):
    u, z, eq, geom = case()
    with pytest.raises(ValueError):
        r.acquire_reference(u, z, *interval, eq, geom, config())


def test_zero_interval_is_literal_identity_for_both_references_and_fixed_control():
    u, z, eq, geom = case()
    result = r.acquire_reference(u, z, .04, .04, eq, geom, config())
    assert result["reference"]["accepted"]
    torch.testing.assert_close(result["teacher"], z, rtol=0., atol=0.)
    torch.testing.assert_close(result["matched_mean"], z, rtol=0., atol=0.)
    torch.testing.assert_close(r.fixed_lawson(u, z, .04, .04, eq, geom, 2), z, rtol=0., atol=0.)
