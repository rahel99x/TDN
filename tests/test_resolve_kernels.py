"""Independent sequential identities, gradients and measured kernel accounting."""
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.resolve.kernels import (BatchedQuadratureModel, batched_quadratic_defect,
    effective_rule, profile_call, run)
from tdn.analysis.roadmap.numerics import quadratic_df_defect
from tdn.numerics import Equation, Geometry


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def field(n=8, dtype=torch.float64):
    x = torch.arange(n, dtype=dtype)/n
    return (.41+.07*torch.cos(2*torch.pi*x[:, None]+.4)
            +.035*torch.sin(4*torch.pi*x[None, :]+.2)
            +.017*torch.cos(2*torch.pi*(x[:, None]+x[None, :])-.3))[None, None]


@pytest.mark.parametrize("track", ("discrete", "continuum"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("nodes", (2, 3, 4))
def test_resolve_batched_kernel_matches_sequential(track, dtype, nodes):
    # Independent fields, different per-field h, non-square domain lengths and
    # retained Nyquist content exercise batching and the Galerkin product.
    u = torch.cat((field(dtype=dtype), .2+field(dtype=dtype)*.4))
    u[..., ::2, ::2] += .003
    h = torch.tensor([.08, .03], dtype=dtype)
    eq, geo = Equation(.008, 4.), Geometry((8, 8), (1.3, .8))
    expected = quadratic_df_defect(u, h, eq, geo, nodes=nodes, target=track)
    actual = batched_quadratic_defect(u, h, eq, geo, track=track, nodes=nodes)
    torch.testing.assert_close(actual, expected, atol=3e-9 if dtype == torch.float32 else 3e-17,
                               rtol=3e-4 if dtype == torch.float32 else 3e-11)


@pytest.mark.parametrize("track", ("discrete", "continuum"))
@pytest.mark.parametrize("nodes", (2, 3, 4))
def test_resolve_kernel_state_and_time_gradients(track, nodes):
    u, h = field().requires_grad_(), torch.tensor(.08, dtype=torch.float64, requires_grad=True)
    eq, geo = Equation(.008, 4.), Geometry((8, 8), (1., 1.))
    weight = torch.cos(torch.arange(64, dtype=u.dtype)).reshape_as(u)
    reference = quadratic_df_defect(u, h, eq, geo, nodes=nodes, target=track)
    candidate = batched_quadratic_defect(u, h, eq, geo, track=track, nodes=nodes)
    expected = torch.autograd.grad((reference*weight).sum(), (u, h))
    actual = torch.autograd.grad((candidate*weight).sum(), (u, h))
    for value, wanted in zip(actual, expected):
        assert torch.isfinite(value).all() and value.abs().max() > 0
        torch.testing.assert_close(value, wanted, atol=5e-16, rtol=3e-10)


@pytest.mark.parametrize("track", ("discrete", "continuum"))
@pytest.mark.parametrize("nodes", (2, 3, 4))
def test_resolve_kernel_physical_nulls(track, nodes):
    u, geo = field(), Geometry((8, 8), (1., 1.))
    for state, h, eq in ((u, 0., Equation(.004, 3.)), (u, .1, Equation(0., 3.)),
                         (u, .1, Equation(.004, 0.)), (u*0+.7, .1, Equation(.004, 3.))):
        result = batched_quadratic_defect(state, h, eq, geo, track=track, nodes=nodes)
        assert torch.count_nonzero(result) == 0
    model = BatchedQuadratureModel(track, nodes=nodes).double()
    torch.testing.assert_close(model(u, 0., Equation(.004, 3.), geo), u, atol=0., rtol=0.)


def test_resolve_gl3_implicit_midpoint_moments_and_zero_integrand():
    rule = effective_rule(3)
    assert rule["active_response_count"] == 2
    assert rule["midpoint_weight"] == pytest.approx(4/9)
    assert sum(rule["active_weights"]) == pytest.approx(5/9)
    for degree in range(6):
        moment = sum(w*x**degree for x, w in zip(rule["active_positions"], rule["active_weights"]))
        moment += rule["midpoint_weight"]*.5**degree
        assert moment == pytest.approx(1/(degree+1), abs=2e-15)
    u, eq, geo = field(), Equation(.004, 3.), Geometry((8, 8), (1., 1.))
    for track in ("discrete", "continuum"):
        midpoint = quadratic_df_defect(u, .1, eq, geo, target=track,
                                      node_positions=[.5], node_weights=[1.])
        assert torch.count_nonzero(midpoint) == 0
        gl2 = quadratic_df_defect(u, .1, eq, geo, target=track, nodes=2)
        gl3 = batched_quadratic_defect(u, .1, eq, geo, track=track, nodes=3)
        assert not torch.allclose(gl3, gl2, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_resolve_kernel_odd_grid_translation(track):
    u, eq, geo = field(9), Equation(.02, 2.), Geometry((9, 9), (1., 1.))
    out = batched_quadratic_defect(u, .1, eq, geo, track=track, nodes=3)
    shifted = batched_quadratic_defect(torch.roll(u, (2, 1), (-2, -1)), .1, eq, geo, track=track, nodes=3)
    torch.testing.assert_close(shifted, torch.roll(out, (2, 1), (-2, -1)), atol=2e-16, rtol=2e-10)


def test_resolve_profiler_counts_are_separate_from_solver_timing():
    u, eq, geo = field(), Equation(.004, 3.), Geometry((8, 8), (1., 1.))
    report = profile_call(lambda:batched_quadratic_defect(u, .1, eq, geo, nodes=4))
    assert report["profile_seconds"] > 0
    assert report["invocation_counts"]["fft_dispatch"] == 4
    assert report["invocation_counts"]["allocations"] > 0
    assert report["energy_joules"] is None and report["gpu_utilization"] is None
    assert any(row["input_shapes"] for row in report["operators"])


def test_resolve_kernel_stage_records_raw_timing_and_unaccepted_references(tmp_path):
    import json
    records = []
    ctx = SimpleNamespace(path=tmp_path, unit=dict(track="discrete", grid=4), device="cpu",
        budget=SimpleNamespace(check=lambda:None), record=lambda *args, **kwargs:records.append((args, kwargs)),
        protocol={"resolve":{"kernels":dict(horizon=.04, regimes=["low"], seed=19002, schedules=[1, 2],
            repeats=2, warmup=1, reference_steps=2, reference_tolerance=0., rms_target=1e-4, max_target=1e-4)}})
    result = run(ctx)
    evidence = json.loads((tmp_path/"kernel_rows.json").read_text())
    assert result["endpoint_rows"] == 16 and len(records) == 28
    assert all(len(row["raw_seconds"]) == 2 for row in evidence["rows"])
    assert all(row["accuracy_status"] == "REFERENCE_UNACCEPTED" for row in evidence["rows"])
    assert all(row["accuracy_qualified_speed_ratio"] is None for row in evidence["comparisons"])
    assert len(evidence["representation_comparisons"]) == 6
    profiling = json.loads((tmp_path/"profiling.json").read_text())
    assert len(profiling["profiles"]) == 8
    assert len(profiling["paired_timing"]) == 2
    assert all(row["training_seconds"] == 0 for row in evidence["rows"])
