"""Mandatory actual-CUDA cases; skipped CPU collection is never GPU evidence."""
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.frontier.models import FAMILIES, TRAINABLE_FAMILIES, RANK_FAMILIES, make_model
from tdn.analysis.frontier import numerics
from tdn.analysis.roadmap.numerics import quadratic_df_defect
from tdn.numerics import Equation, Geometry

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def native_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_FRONTIER_GPU_TESTS") == "1":
            pytest.fail("Required frontier CUDA cases have no actual CUDA device")
        pytest.skip("No CUDA; this skip is not native GPU validation")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "frontier-gpu-tests")
    reference_precision()
    torch.manual_seed(770031)
    torch.cuda.manual_seed_all(770031)


def state(device="cpu", dtype=torch.float64, n=8):
    x = torch.arange(n, device=device, dtype=dtype) / n
    return (.43 + .08 * torch.cos(2 * torch.pi * x[:, None] + .2)
            + .03 * torch.sin(4 * torch.pi * x[None, :] - .4))[None, None]


def model(family, track):
    value = make_model(family, track, {"width": 3, "depth": 2, "modes": 2,
        "quad_nodes": 2, "cubic_nodes": 2, "reaction_substeps": 2}).double()
    with torch.no_grad():
        for parameter in value.parameters():
            parameter.copy_(.1 * torch.randn_like(parameter))
    return value


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_cuda_model_limits(track, family):
    for dtype in (torch.float32, torch.float64):
        cpu = model(family, track).to(dtype=dtype)
        gpu = deepcopy(cpu).cuda()
        u, eq, geo = state(dtype=dtype), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
        with torch.no_grad():
            actual, expected = gpu(u.cuda(), .03, eq, geo), cpu(u, .03, eq, geo)
            torch.testing.assert_close(actual.cpu(), expected,
                atol=5e-7 if dtype == torch.float32 else 2e-11,
                rtol=3e-6 if dtype == torch.float32 else 2e-9)
            assert torch.isfinite(actual).all() and actual.dtype == dtype
            torch.testing.assert_close(gpu(u.cuda(), 0., eq, geo).cpu(), u, atol=0., rtol=0.)
            if family in RANK_FAMILIES or family.startswith("analytic_quad"):
                for null_eq in (Equation(0., 2.), Equation(.003, 0.)):
                    increment = gpu.correction_components(u.cuda(), .03, null_eq, geo)["increment"]
                    torch.testing.assert_close(increment, torch.zeros_like(increment), atol=0., rtol=0.)


@pytest.mark.parametrize("family", TRAINABLE_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_cuda_model_gradient(track, family):
    cpu = model(family, track)
    gpu = deepcopy(cpu).cuda()
    eq, geo = Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    u, v = state().requires_grad_(), state("cuda").requires_grad_()
    h = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    hg = torch.tensor(.03, device="cuda", dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(cpu(u, h, eq, geo).square().mean(), (u, h, *cpu.parameters()))
    second = torch.autograd.grad(gpu(v, hg, eq, geo).square().mean(), (v, hg, *gpu.parameters()))
    for a, b in zip(first, second):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(b.cpu(), a, atol=3e-9, rtol=3e-5)
    assert any(float(v.abs().max()) > 0 for v in second[2:])


@pytest.mark.parametrize("kind", ("heat", "product", "quadratic", "etdrk4"))
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_cuda_numerical_parity(track, kind):
    eq, geo = Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    def apply(u):
        if kind == "heat": return numerics.heat_step(u, .03, eq, geo, track)
        if kind == "product": return numerics.product(u, u, track)
        if kind == "quadratic": return quadratic_df_defect(u, .03, eq, geo, target=track, nodes=3)
        return numerics.etdrk4_step(u, .03, eq, geo, track)
    torch.testing.assert_close(apply(state("cuda")).cpu(), apply(state()), atol=2e-11, rtol=2e-9)


def test_frontier_cuda_allocation_visible():
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_properties(0).total_memory > 0
    assert os.environ.get("TDN_EXECUTION_MODE") == "desktop-slurm"
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")


def forbid_independent_teachers(monkeypatch):
    from tdn.analysis.frontier import data
    from tdn.analysis.agenda import data as agenda_data
    def forbidden(*args, **kwargs):
        raise AssertionError("Operational GPU decisions must never access independent CPU teachers")
    monkeypatch.setattr(data, "lawson_reference", forbidden)
    monkeypatch.setattr(data, "generate_reference", forbidden)
    monkeypatch.setattr(agenda_data, "continuum_ifrk4", forbidden)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_cuda_policy_proposal(track, monkeypatch):
    from tdn.analysis.frontier import policy
    forbid_independent_teachers(monkeypatch)
    cpu = model("rank1", track).float()
    gpu = deepcopy(cpu).cuda()
    initial, eq, geo = state(dtype=torch.float32), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    budget = SimpleNamespace(check=lambda: None)
    expected, cpu_info = policy.proposal(cpu, initial, .03, eq, geo, spatial=True,
        sampler=lambda n: state(n=n), budget=budget, device="cpu")
    real_resample = policy.fourier_resample
    devices, sampled = [], []
    def require_device_resampling(value, grid):
        devices.append(value.device.type)
        assert value.device.type == "cuda"
        return real_resample(value, grid)
    def sampler(n):
        sampled.append(n)
        # The deployed initial function is sampled independently of truth.
        return state(n=n)
    monkeypatch.setattr(policy, "fourier_resample", require_device_resampling)
    actual, info = policy.proposal(gpu, initial.cuda(), .03, eq, geo, spatial=True,
        sampler=sampler, budget=budget, device="cuda")
    assert devices == ["cuda"] and sampled == [16]
    assert actual.device.type == "cuda" and actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, atol=6e-7, rtol=4e-6)
    assert info["proposal_seconds"] > 0 and info["estimator_seconds"] > 0
    for norm in ("rms", "max"):
        assert info["indicator"][norm] >= 1e-8
        assert abs(info["indicator"][norm] - cpu_info["indicator"][norm]) < 2e-6


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_cuda_policy_fallback(track, monkeypatch):
    from tdn.analysis.frontier import policy
    forbid_independent_teachers(monkeypatch)
    cpu, classical_cpu = model("rank1", track).float(), model("etdrk4", track).float()
    gpu, classical = deepcopy(cpu).cuda(), deepcopy(classical_cpu).cuda()
    initial, eq, geo = state(dtype=torch.float32), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    budget = SimpleNamespace(check=lambda: None)
    kwargs = dict(target=1e-3, max_refinements=2, budget=budget)
    expected, expected_info = policy.classical_fallback(classical_cpu, initial, .03, eq, geo, **kwargs)
    actual, info = policy.deploy(gpu, classical, initial.cuda(), .03, eq, geo,
        envelope={"rms": 1e30, "max": 1e30}, spatial=True, sampler=lambda n: state(n=n), device="cuda", **kwargs)
    assert info["fallback"] is True and info["accepted"] is False
    assert info["decision_uses_reference"] is False and info["deterministic_certificate"] is False
    assert actual.device.type == "cuda" and actual.dtype == torch.float32
    assert info["fallback_info"]["converged"] and expected_info["converged"]
    assert info["fallback_info"]["steps"] >= 3
    assert all(info[k] > 0 for k in ("proposal_seconds", "estimator_seconds", "fallback_seconds"))
    assert info["attributed_seconds"] == pytest.approx(sum(info[k] for k in
        ("proposal_seconds", "estimator_seconds", "fallback_seconds")), rel=1e-12)
    torch.testing.assert_close(actual.cpu(), expected, atol=6e-7, rtol=4e-6)
    accepted, accepted_info = policy.deploy(gpu, classical, initial.cuda(), .03, eq, geo,
        envelope={"rms": 0., "max": 0.}, spatial=True, sampler=lambda n: state(n=n), device="cuda", **kwargs)
    assert accepted_info["accepted"] and not accepted_info["fallback"]
    assert accepted_info["fallback_seconds"] == 0 and accepted_info["fallback_info"] is None
    assert accepted_info["attributed_seconds"] == pytest.approx(
        accepted_info["proposal_seconds"] + accepted_info["estimator_seconds"], rel=1e-12)
    assert accepted.device.type == "cuda" and torch.isfinite(accepted).all()


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_cuda_scaling_distinct_fields(track):
    from tdn.analysis.frontier import neural, scaling, data
    first = state(dtype=torch.float32)
    x = torch.arange(8, dtype=torch.float32) / 8
    second = (.52 + .05 * torch.sin(6 * torch.pi * x[:, None] + .2)
              + .04 * torch.cos(2 * torch.pi * x[None, :]))[None, None]
    parents = [dict(field_cluster=f"field{i}", kappa=.003, reaction_rate=2., lengths=[1., 1.], states={"8": u})
               for i, u in enumerate((first, second))]
    scaling.distinct_batch(parents, 2, 8)
    initial = torch.cat((first, second))
    eq, geo, schedule = Equation(.003, 2.), Geometry((8, 8), (1., 1.)), [.01, .02]
    # This teacher is evaluated only for the accuracy audit, never inside the
    # measured solver or deployment decision. It resolves the same-grid target.
    truth = data.lawson_reference(initial.double(), sum(schedule), eq, geo, 64, track)
    ref = dict(state=truth, accepted=True, uncertainty_rms=0., uncertainty_max_bound=0.)
    for family in ("rank1", "fno_small", "df", "etdrk4"):
        cpu = model(family, track).float()
        gpu = deepcopy(cpu).cuda()
        expected, _ = neural.rollout(cpu, initial, schedule, eq, geo)
        actual, _ = neural.rollout(gpu, initial.cuda(), schedule, eq, geo)
        assert actual.device.type == "cuda" and actual.dtype == torch.float32
        torch.testing.assert_close(actual.cpu(), expected, atol=7e-7, rtol=4e-6)
        errors, cpu_errors = neural.endpoint_errors(actual, ref), neural.endpoint_errors(expected, ref)
        assert errors["finite"] and errors["reference_accepted"]
        for norm in ("rms", "max"):
            assert abs(errors[f"error_{norm}"] - cpu_errors[f"error_{norm}"]) < 2e-6
        if family == "etdrk4":
            assert errors["error_max"] < 3e-6
        # No numerical accuracy threshold is imposed on arbitrary FNO weights:
        # those errors are measured, not silently declared successful.
