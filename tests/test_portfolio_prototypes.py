"""Numerical safeguards for the protected development-only research path."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tdn.analysis.portfolio import prototypes as p


def test_interaction_integral_equal_rates_and_small_step():
    h = .3
    rates = np.array([-100., -1., 0., 2.])
    np.testing.assert_allclose(p.interaction_kernel(h, rates, rates), h*np.exp(h*rates), rtol=1e-14)
    np.testing.assert_array_equal(p.interaction_kernel(0., rates, rates+1), np.zeros(4))
    np.testing.assert_allclose(p.interaction_kernel(1e-10, rates, rates+1)/1e-10, np.ones(4), atol=1e-8)
    assert np.isfinite(p.interaction_kernel(.2, np.array([-1e7]), np.array([0.]))).all()
    with pytest.raises(ValueError):
        p.interaction_kernel(-.1, rates, rates)


def test_exact_kernel_matches_independent_high_order_quadrature():
    geometry = p.pair_geometry(10, 4, diffusion=.1, growth=.3)
    field = p._field(geometry, np.random.default_rng(7101))
    for h in (0., .002, .3):
        np.testing.assert_allclose(p.pair_query(field, geometry, h), p.pair_query(field, geometry, h, nodes=64),
                                   rtol=2e-13, atol=1e-15)


def test_nonaliased_fft_control_matches_pair_rule():
    geometry = p.pair_geometry(7, 6, diffusion=.2, growth=.3)
    field = p._field(geometry, np.random.default_rng(7102))
    np.testing.assert_allclose(p.fft_quadrature(field, geometry, .2),
                               p.pair_query(field, geometry, .2, nodes=4), rtol=1e-13, atol=1e-16)


def test_temporal_encoding_query_refresh_and_validity_interval():
    geometry = p.pair_geometry(7, 3, diffusion=.02)
    field = p._field(geometry, np.random.default_rng(7103))
    basis = p.temporal_basis(geometry, horizon=.3, degree=12)
    encoded = p.encode_temporal(field, geometry, basis)
    for h in (0., .0012, .173, .3):
        np.testing.assert_allclose(p.temporal_query(encoded, h), p.pair_query(field, geometry, h), rtol=1e-10, atol=1e-15)
    # A state update requires refresh; quadratic scaling must survive refresh.
    refreshed = p.encode_temporal(2*field, geometry, basis)
    np.testing.assert_allclose(p.temporal_query(refreshed, .2), 4*p.temporal_query(encoded, .2), atol=1e-15)
    with pytest.raises(ValueError, match="sealed encoding interval"):
        p.temporal_query(encoded, .301)


def test_dense_output_compares_same_response_not_nonlinear_solution():
    geometry = p.pair_geometry(5, 3, diffusion=.03)
    field = p._field(geometry, np.random.default_rng(7104))
    dense = p.dense_quadratic_solution(field, geometry, .2)
    for h in (.037, .129, .2):
        np.testing.assert_allclose(dense.sol(h)[len(field):], p.pair_query(field, geometry, h), rtol=2e-9, atol=1e-13)


def test_coarse_twins_share_state_and_variance_but_not_future():
    a = p.twin_coefficients(15, .4, .1, .08, .3, -.7, 1)
    b = p.twin_coefficients(15, .4, .1, .08, .3, -.7, -1)
    np.testing.assert_array_equal(a[13:18], b[13:18])
    assert np.sum(abs(a)**2) == np.sum(abs(b)**2)
    future_a, future_b = p.galerkin_solution(a, .01), p.galerkin_solution(b, .01)
    assert abs(future_a[16]-future_b[16]) > 1e-6
    # Separate actual histories, with identical current resolved fields.
    previous_a, previous_b = p.galerkin_solution(a, -.002), p.galerkin_solution(b, -.002)
    assert abs(previous_a[16]-previous_b[16]) > 1e-7


def test_galerkin_logistic_mean_limit_and_short_time_derivative():
    state = np.zeros(15, dtype=complex); state[7] = .4
    h, reaction = .13, .5
    future = p.galerkin_solution(state, h, reaction=reaction)
    expected = .4*np.exp(reaction*h)/(1+.4*np.expm1(reaction*h))
    np.testing.assert_allclose(future[7], expected, atol=1e-12)
    np.testing.assert_allclose(np.r_[future[:7], future[8:]], 0., atol=1e-15)


def test_selector_preserves_conjugacy_and_full_rank_identity():
    geometry = p.pair_geometry(12, 5)
    field = p._field(geometry, np.random.default_rng(7105))
    for rank in (1, 4, 12):
        output = p.pair_query(field, geometry, .1, active=p.select_modes(field, geometry, rank))
        np.testing.assert_allclose(output, output[::-1].conj(), atol=1e-15)
    np.testing.assert_array_equal(p.select_modes(field, geometry, 12), np.ones(len(geometry.p), dtype=bool))


def test_selection_fitted_rule_uses_only_field_features_and_bounded_candidates():
    geometry = p.pair_geometry(12, 5)
    field = p._field(geometry, np.random.default_rng(7106))
    candidates = [2, 4, 8, 12]
    assert p.predicted_rank(field, geometry, np.array([-100., 0., 0., 0.]), candidates) == 2
    assert p.predicted_rank(field, geometry, np.array([100., 0., 0., 0.]), candidates) == 12
    assert 1 <= p.heuristic_rank(field) <= 12


class _Budget:
    def check(self):
        pass


@pytest.mark.parametrize("prototype,artifact,minimum_rows", [
    ("X01", "temporal_query_rows.json", 7), ("X02", "closure_rows.json", 4),
    ("X03", "selection_rows.json", 3)])
def test_prototype_runs_real_science_and_keeps_raw_data(tmp_path, prototype, artifact, minimum_rows):
    recorded = []
    ctx = SimpleNamespace(protocol={"profile": "smoke", "units": {"test": {"prototype_ids": [prototype]}}},
                          stage="test", path=tmp_path, device="cpu", budget=_Budget(),
                          record=lambda *args, **kwargs: recorded.append((args, kwargs)))
    result = p.run(ctx)
    assert result["status"] == "COMPLETED"
    assert len(recorded) >= minimum_rows
    rows = json.loads((tmp_path/"prototype_rows.json").read_text())
    assert all(row["prototype_id"] == prototype for row in rows)
    assert all(row["status"] == "COMPLETED" for row in rows)
    assert all({"math", "gap"} <= {check["category"] for check in row["checks"]} for row in rows)
    raw = json.loads((tmp_path/artifact).read_text())
    assert raw
    if prototype in ("X02", "X03"):
        train = {v["parent_id"] for v in raw if v["split"] == "fit"}
        heldout = {v["parent_id"] for v in raw if v["split"] == "heldout_development"}
        assert train and heldout and train.isdisjoint(heldout)
    else:
        assert {v["method"] for v in raw} >= {"classical_dense_output", "fft_gauss_four", "compact_temporal_encoding"}


def test_protected_cpu_measurements_reject_gpu_mislabel(tmp_path):
    with pytest.raises(ValueError, match="explicitly measure CPU"):
        p.run(SimpleNamespace(device="cuda", path=tmp_path))
