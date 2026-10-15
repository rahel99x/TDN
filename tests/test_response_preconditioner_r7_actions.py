"""Tiny mathematical/timing fixtures; no R7 scientific bank is executed here."""
import pytest
import torch

from tdn.analysis.response.contracts import Physics
from tdn.analysis.response_preconditioner import r7_actions as actions
from tdn.analysis.response_preconditioner.model import FixedFeatureResponse
from tdn.analysis.response_preconditioner.numerics import ordered_response
from tdn.analysis.tangent_response.tangent import AnchorPath
from tdn.numerics.types import Geometry


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case():
    x = torch.arange(5, dtype=torch.float64) * (2 * torch.pi / 5)
    u = (.35 + .07 * torch.sin(x[:, None]) + .02 * torch.cos(x[None, :]))[None, None]
    z = (torch.cos(x[:, None]) + .2 * torch.sin(2 * x[None, :]))[None, None]
    return u, z, Physics(.02, 1.3), Geometry((5, 5), (1., 1.))


def fitted():
    return FixedFeatureResponse().set_readout(torch.linspace(-.12, .17, 40, dtype=torch.float64))


def test_five_branch_factorization_matches_global_model_and_is_source_independent():
    u, z, eq, geom = case()
    model = fitted()
    first = actions.design_context(u, z, .02, .06, eq, geom, model)
    other = actions.design_context(u, torch.roll(z, 1, -1) + .2, .02, .06, eq, geom, model)
    assert first["features"].shape == (8,)
    assert first["branches"].shape == (5, 1, 1, 5, 5)
    assert first["design"].shape == (40, 1, 1, 5, 5)
    assert torch.equal(first["features"], other["features"])
    c = model.readout.reshape(5, 8) @ first["features"]
    prediction = first["base"] + torch.einsum("j,j...->...", c, first["branches"])
    torch.testing.assert_close(prediction, model.response_action(u, z, .02, .06, eq, geom),
                               atol=3e-15, rtol=2e-14)
    assert first["work"]["affine_action_designs"] == 1
    assert first["work"]["fixed_feature_encodings"] == 2
    assert "not instrumented" in first["work_counter_scope"]


def test_extracted_branches_are_linear_in_source():
    u, z, eq, geom = case()
    other = torch.roll(z, 1, -2) + .1
    a = actions.design_context(u, z, .04, .08, eq, geom)
    b = actions.design_context(u, other, .04, .08, eq, geom)
    mixed = actions.design_context(u, 1.3*z-.7*other, .04, .08, eq, geom)
    torch.testing.assert_close(mixed["branches"], 1.3*a["branches"]-.7*b["branches"],
                               atol=3e-17, rtol=3e-12)


@pytest.mark.parametrize("invalid", ("batch", "panels", "precision"))
def test_context_rejects_changed_identification_contract(invalid):
    u, z, eq, geom = case()
    model = FixedFeatureResponse()
    if invalid == "batch":
        u, z = u.repeat(2, 1, 1, 1), z.repeat(2, 1, 1, 1)
    elif invalid == "panels":
        model = FixedFeatureResponse(panels=4)
    else:
        model = FixedFeatureResponse(output_precision="input")
    with pytest.raises(ValueError):
        actions.design_context(u, z, .02, .06, eq, geom, model)


@pytest.mark.parametrize("panels", (1, 2, 4, 8))
def test_fresh_ordered_uses_original_nonzero_time_path_and_complete_work(panels):
    u, z, eq, geom = case()
    work = {}
    actual = actions.fresh_ordered_action(u, z, .04, .08, eq, geom, panels=panels, work=work)
    path = AnchorPath(u, eq, geom, "fd_same_grid", create_graph=True)
    expected = ordered_response(path, z, .04, .08, eq, geom, panels=panels)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    restarted = AnchorPath(u, eq, geom, "fd_same_grid", create_graph=True)
    shifted = ordered_response(restarted, z, 0., .04, eq, geom, panels=panels)
    assert (actual-shifted).abs().max() > 1e-5
    assert work["fresh_path_constructions"] == 1
    assert work["heat_actions"] == panels
    assert work["path_anchor_evaluations"] == 2*panels


def test_fresh_ordered_rebuilds_state_graph_and_preserves_derivatives():
    u, z, eq, geom = case()
    function = lambda state: actions.fresh_ordered_action(state, z, .02, .06, eq, geom)
    first = function(u)
    changed = function(u+.03)
    assert (first-changed).abs().max() > 1e-4
    _, jvp = torch.autograd.functional.jvp(function, u, z)
    eps = 2e-5
    difference = (function(u+eps*z)-function(u-eps*z))/(2*eps)
    torch.testing.assert_close(jvp, difference, atol=2e-10, rtol=2e-8)


def test_fresh_learned_action_requires_frozen_model_and_counts_fresh_setup():
    u, z, eq, geom = case()
    with pytest.raises(ValueError, match="sealed global model"):
        actions.fresh_action("fixed_feature_fitted", u, z, .02, .06, eq, geom)
    with pytest.raises(ValueError, match="nondeployable"):
        actions.fresh_action("local_oracle", u, z, .02, .06, eq, geom)
    model = fitted()
    work = {}
    value = actions.fresh_action("fixed_feature_fitted", u, z, .02, .06, eq, geom,
                                 model=model, work=work)
    torch.testing.assert_close(value, model.response_action(u, z, .02, .06, eq, geom),
                               atol=0, rtol=0)
    assert work["fresh_path_constructions"] == 1
    assert work["fixed_feature_encodings"] == 1
    assert work["streamed_response_actions"] == 1


def test_source_bundle_charges_every_source_again_without_cross_call_reuse(monkeypatch):
    u, z, eq, geom = case()
    calls = []
    sources = [z, z+.1, z-.2]

    def fake(method, state, source, start, stop, physics, geometry, *, model):
        calls.append((method, id(state), id(source), start, stop))
        return source.double()

    monkeypatch.setattr(actions, "fresh_action", fake)
    methods = actions.fresh_actions(u, sources, .02, .06, eq, geom, fitted())
    assert len(methods) == 8
    for name, call in methods.items():
        for _ in range(2):
            value = call()
            assert value.shape == (3, *u.shape)
            assert [row[0] for row in calls[-3:]] == [name]*3
            assert [row[2] for row in calls[-3:]] == [id(z) for z in sources]
    assert len(calls) == 8*2*3


def test_all_deployable_methods_execute_in_one_bounded_tiny_timing_fixture():
    u, z, eq, geom = case()
    methods = actions.fresh_actions(u, [z], .02, .04, eq, geom, fitted())
    result = actions.measure_paired_actions(methods, rounds=1, warmup=0, seed=7)
    assert result["status"] == "PASS", result["records"]
    assert result["expected_calls"] == 16
    assert result["failed_calls"] == 0


def test_paired_timing_has_exact_denominator_and_reproducible_random_orders():
    counts = {"a": 0, "b": 0, "c": 0}

    def method(name):
        counts[name] += 1
        return torch.ones(2, dtype=torch.float64)

    methods = {name: (lambda n=name: method(n)) for name in counts}
    first = actions.measure_paired_actions(methods, rounds=5, seed=192, warmup=1)
    assert first["expected_calls"] == 21
    assert counts == dict(a=7, b=7, c=7)
    assert first["status"] == "PASS"
    assert len([row for row in first["records"] if row["phase"] == "cold"]) == 3
    assert len([row for row in first["records"] if row["phase"] == "warmup"]) == 3
    second = actions.measure_paired_actions(methods, rounds=5, seed=192, warmup=1)
    assert [(row["phase"], row["round"], row["method"]) for row in first["records"]] == [
        (row["phase"], row["round"], row["method"]) for row in second["records"]]
    for summary in first["summary"].values():
        assert summary["passed_rounds"] == 5
        assert 0 <= summary["minimum_seconds"] <= summary["median_seconds"] <= summary["maximum_seconds"]
    assert first["scientific_promotion"] is False


def test_timing_failures_remain_in_all_round_denominators():
    def failed():
        raise RuntimeError("fixture failed")

    result = actions.measure_paired_actions({"ok": lambda: torch.zeros(1), "bad": failed},
                                            rounds=2, warmup=0)
    assert result["status"] == "FAILED"
    assert result["failed_calls"] == 3
    assert result["expected_calls"] == 6
    assert result["summary"]["bad"]["failed_rounds"] == 2
    assert result["summary"]["bad"]["median_seconds"] is None
    assert all("fixture failed" in row["error"] for row in result["records"] if row["status"] == "FAILED")


def test_timing_detects_output_change_and_does_not_hide_it_as_fast_success():
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        return torch.tensor([calls], dtype=torch.float64)

    result = actions.measure_paired_actions({"changed": changed}, rounds=2, warmup=0)
    assert result["failed_calls"] == 2
    assert all("changed its output" in row["error"] for row in result["records"][1:])


def test_timing_budget_checks_between_every_call_and_propagates_stop():
    calls = []
    retained = []

    def budget():
        if len(calls) == 2:
            raise TimeoutError("bounded fixture")

    def method():
        calls.append(1)
        return torch.zeros(1)

    with pytest.raises(TimeoutError, match="bounded fixture"):
        actions.measure_paired_actions({"a": method}, rounds=5, budget=budget,
                                       on_record=retained.append)
    assert len(calls) == 2
    assert [row["phase"] for row in retained] == ["cold", "warmup"]
    assert all(row["status"] == "PASS" for row in retained)


def test_timing_rejects_changed_thread_or_gradient_contract():
    with torch.no_grad(), pytest.raises(ValueError, match="gradients enabled"):
        actions.measure_paired_actions({"x": lambda: torch.zeros(1)})
    torch.set_num_threads(2)
    with pytest.raises(ValueError, match="one PyTorch CPU thread"):
        actions.measure_paired_actions({"x": lambda: torch.zeros(1)})
