"""Physical metric, held-out linear actions, conditioning and source work."""
import json
import io
import math

import pytest
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.analysis.tangent_response.compression import (
    collect_experiment_actions, conditional_fit, field_metrics, fit_pod, fit_source_map, physical_unwhiten,
    physical_whiten, predeclare_probes, run_experiment,
)
from tdn.analysis.tangent_response.tangent import AnchorPath, response_action
from tdn.numerics.types import Geometry


def _state(geom, *, mean=.4):
    axes = [torch.arange(size, dtype=torch.float64)/size for size in geom.grid]
    mesh = torch.meshgrid(*axes, indexing="ij")
    value = mean+.15*torch.cos(2*math.pi*mesh[0])
    if geom.ndim == 2:
        value = value+.08*torch.sin(2*math.pi*mesh[1])
    return value.reshape(1, 1, *geom.grid)


@pytest.mark.parametrize("target", ["fd_same_grid", "galerkin_same_grid"])
@pytest.mark.parametrize("grid", [(5,), (6,), (5, 7), (6, 8)])
def test_whitening_is_physical_inner_product_with_odd_even_nyquist(target, grid):
    geom = Geometry(grid, (1.,)*len(grid))
    generator = torch.Generator().manual_seed(129)
    a = torch.randn(2, 1, *grid, generator=generator, dtype=torch.float64)
    b = torch.randn(2, 1, *grid, generator=generator, dtype=torch.float64)
    wa, wb = physical_whiten(a, geom, target), physical_whiten(b, geom, target)
    assert torch.allclose((wa*wb).flatten(1).sum(1), n.inner(a, b, geom, target).flatten(), atol=1e-14)
    assert torch.allclose(physical_unwhiten(wa, geom, target), a, atol=2e-14)
    metrics = field_metrics(a, geom, target)
    assert metrics["rms"]**2 == pytest.approx(metrics["mean_rms"]**2+metrics["centered_rms"]**2)


def test_nyquist_physical_mass_and_mean_are_retained():
    geom = Geometry((8,), (1.,))
    nyquist = (-1.)**torch.arange(8, dtype=torch.float64)
    value = (2.+nyquist).reshape(1, 1, 8)
    assert field_metrics(value, geom, "fd_same_grid")["rms"]**2 == pytest.approx(5.)
    assert field_metrics(value, geom, "galerkin_same_grid")["rms"]**2 == pytest.approx(4.5)


@pytest.mark.parametrize("target", ["fd_same_grid", "galerkin_same_grid"])
def test_conditional_fit_equilibrates_rank_deficiency_and_deployment(target):
    geom = Geometry((8,), (1.,))
    cosine = torch.cos(2*math.pi*torch.arange(8, dtype=torch.float64)/8).reshape(1, 1, 8)
    constant = torch.ones_like(cosine)
    design = torch.stack([1e-12*cosine, 2e-12*cosine, constant, torch.zeros_like(cosine)])
    target_field = .13*cosine+.07*constant
    fit = conditional_fit(design, target_field, geom, target, rcond=1e-10)
    assert fit["rank"] == 2
    assert fit["physical_error"]["rms"] < 2e-16
    assert torch.isfinite(fit["coefficients"]).all()
    assert fit["deployment_error"]["rms"] < 2e-8
    assert len(fit["gram"]) == 4 and len(fit["column_norms"]) == 4
    regularized = conditional_fit(design, target_field, geom, target, ridge=.1)
    assert regularized["physical_error"]["rms"] > fit["physical_error"]["rms"]
    assert regularized["ridge_scope"] == "squared coefficient norm in column-equilibrated coordinates"


def test_parent_query_weights_change_fit_without_dropping_queries():
    geom = Geometry((4,), (1.,))
    design = torch.ones(1, 2, 1, 4, dtype=torch.float64)
    target = torch.tensor([1., 3.], dtype=torch.float64).reshape(2, 1, 1).expand(2, 1, 4)
    unweighted = conditional_fit(design, target, geom, "fd_same_grid")
    weighted = conditional_fit(design, target, geom, "fd_same_grid", weights=[3., 1.])
    assert float(unweighted["coefficients"][0]) == pytest.approx(2.)
    assert float(weighted["coefficients"][0]) == pytest.approx(1.5)


@pytest.mark.parametrize("target", ["fd_same_grid", "galerkin_same_grid"])
def test_real_tangent_actions_fit_heldout_signed_direction_and_keep_mean(target):
    geom = Geometry((7,), (1.,)); eq = Physics(.005, 2.)
    path = AnchorPath(_state(geom), eq, geom, target)
    one = torch.ones_like(path.u0)
    cosine = n.center(path.u0)
    sources = torch.cat([one, cosine])
    actions = torch.cat([response_action(path, eq, geom, target, z[None], .003, .03, steps=16, mode="tangent")-
                         response_action(path, eq, geom, target, z[None], .003, .03, steps=16, mode="mean") for z in sources])
    model = fit_source_map(sources, actions, geom, target, 2)
    heldout = .31*one-.72*cosine
    exact = response_action(path, eq, geom, target, heldout, .003, .03, steps=16)-response_action(path, eq, geom, target, heldout, .003, .03, steps=16, mode="mean")
    prediction = model.apply(heldout)
    assert field_metrics(prediction-exact, geom, target)["rms"] < 2e-14
    assert n.mean(model.apply(cosine)).abs().max() > 1e-7
    assert torch.allclose(model.apply(-heldout), -prediction)
    assert model.metadata["deployable"]
    with pytest.raises(ValueError, match="training parents"):
        fit_source_map(sources, actions, geom, target, 2, split="diagnostic")
    oracle = fit_source_map(sources, actions, geom, target, 2, split="diagnostic", oracle=True)
    assert not oracle.metadata["deployable"]


def test_zero_actions_and_cutoff_report_rank_zero_without_inventing_directions():
    geom = Geometry((8,), (1.,))
    source = torch.randn(3, 1, 8, dtype=torch.float64)
    zero = torch.zeros_like(source)
    model = fit_source_map(source, zero, geom, "fd_same_grid", 8)
    assert model.metadata["requested_rank"] == 8
    assert model.metadata["retained_rank"] == 0
    assert torch.equal(model.apply(source), zero)
    with pytest.raises(ValueError, match="train parents"):
        fit_pod(source, geom, "fd_same_grid", 2, split="validation")


def test_time_conditioned_map_fits_distinct_intervals_with_same_spatial_probe():
    geom = Geometry((7,), (1.,)); source = torch.ones(1, 1, 7, dtype=torch.float64)
    shape = n.center(_state(geom))+.2
    pairs = [(0., .02), (.01, .04), (0., .08), (.04, .1)]
    # A representable time-conditioned affine control. This diagnoses the fit,
    # not the scientific rank of a tangent response; physical action tests above
    # independently use genuine numerical response actions.
    from tdn.analysis.tangent_response.compression import time_features
    coefficients = time_features([p[0] for p in pairs], [p[1] for p in pairs], channels=2)
    gains = coefficients@torch.tensor([.3, -.12], dtype=torch.float64)
    actions = gains[:, None, None]*shape
    model = fit_source_map(source.expand(4, 1, 7), actions, geom, "fd_same_grid", 2,
                           time_pairs=pairs, temporal_channels=2)
    unseen = (.02, .06)
    gain = time_features(*unseen, channels=2)@torch.tensor([.3, -.12], dtype=torch.float64)
    assert torch.allclose(model.apply(source, *unseen), gain.reshape(1, 1, 1)*shape, atol=1e-14)
    assert torch.equal(model.apply(source, .03, .03), torch.zeros_like(source))
    assert model.metadata["temporal_channels"] == 2
    assert model.metadata["retained_rank"] == 1
    buffer = io.BytesIO()
    torch.save(model.to_payload(), buffer)
    buffer.seek(0)
    from tdn.analysis.tangent_response.compression import ResponseFactors
    restored = ResponseFactors.from_payload(torch.load(buffer, weights_only=True))
    assert torch.equal(restored.apply(source, *unseen), model.apply(source, *unseen))
    with pytest.raises(ValueError, match="ordered"):
        model.apply(source, .04, .02)
    with pytest.raises(ValueError, match="require source"):
        model.apply(source)


def test_probes_are_frozen_independent_signed_and_have_roles_before_actions():
    geom = Geometry((8, 8), (1., 2.)); u0 = _state(geom).float()
    kwargs = dict(split="train", seed=17, source_nodes=2, band_probes=4, random_probes=4)
    first = predeclare_probes(u0, geom, "fd_same_grid", "parent", "query", .03, **kwargs)
    second = predeclare_probes(u0, geom, "fd_same_grid", "parent", "query", .03, **kwargs)
    assert len(first) == 18
    assert [p["probe_id"] for p in first] == [p["probe_id"] for p in second]
    assert all(0 < p["source_time"] < .03 for p in first)
    assert {p["probe_role"] for p in first} == {"fit", "heldout_direction"}
    for a, b in zip(first, second):
        if a["source"] is not None:
            assert torch.equal(a["source"], b["source"])
            assert field_metrics(a["source"], geom, "fd_same_grid")["rms"] == pytest.approx(1.)
    random = [p for p in first if p["probe_kind"] == "signed_random"]
    assert all(bool((p["source"] > 0).any()) and bool((p["source"] < 0).any()) for p in random)
    assert not torch.equal(random[0]["source"], random[1]["source"])
    separate = predeclare_probes(u0, geom, "fd_same_grid", "parent", "query", .03,
                                 source_nodes=2, source_fractions=[0., .5], band_probes=2, random_probes=2)
    assert len(separate) == 10
    assert {p["source_fraction"] for p in separate if p["probe_kind"] != "residual"} == {0., .5}
    assert sum(p["quadrature_weight"] for p in separate) == pytest.approx(.03)


def test_production_fits_are_thin_snapshot_svd_without_grid_square(monkeypatch):
    geom = Geometry((64, 64), (1., 1.))
    sources = torch.randn(5, 1, *geom.grid, dtype=torch.float64)
    actions = .2*sources+sources.mean((-2, -1), keepdim=True)
    original = torch.linalg.svd
    shapes = []
    def traced(value, **kwargs):
        shapes.append(tuple(value.shape))
        assert kwargs.get("full_matrices") is False
        assert min(value.shape) <= 5
        return original(value, **kwargs)
    monkeypatch.setattr(torch.linalg, "svd", traced)
    model = fit_source_map(sources, actions, geom, "fd_same_grid", 4)
    assert model.alpha.numel() <= 4*64*64
    assert shapes and all(shape != (4096, 4096) for shape in shapes)
    assert len(model.metadata["alpha_gram"]) <= 4


def test_compression_experiment_keeps_heldout_parents_uncertainty_and_real_work():
    geom = Geometry((5,), (1.,)); eq = Physics(.003, 1.)
    samples = []
    for index, split in enumerate(("train", "diagnostic")):
        u0 = _state(geom, mean=.35+.05*index).float()
        teacher = n.integrate(u0.double(), .02, eq, geom, "fd_same_grid", substeps=64)
        samples.append(dict(u0=u0, target=teacher, eq=eq, geom=geom,
            record=dict(parent_id=f"parent-{index}", query_id=f"query-{index}", split=split,
                        horizon=.02, target_kind="fd_same_grid")))
    result = run_experiment(samples, dict(ranks=[2], source_nodes=2, band_probes=2,
                           random_probes=2, steps=4, action_tolerance=1e-8, quadrature_tolerance=1e-8))
    assert result["coverage"]["requested_probes"] == 20
    assert result["coverage"]["independent_parent_count"] == 2
    assert result["work"]["homogeneous_solves"] == 80
    assert result["work"]["forced_solves"] == 8
    assert any(row["evaluation_scope"] == "heldout_direction" for row in result["action_rows"])
    assert any(row["evaluation_scope"] == "heldout_parent" for row in result["action_rows"])
    assert any(row["compression_object"] == "task_residual_family" for row in result["action_rows"])
    assert all(not row["deployable"] for row in result["action_rows"] if row["method"].startswith("state_oracle"))
    assert all("quadrature_check" in row for row in result["endpoint_rows"] if "mean_error" in row)
    public = {key: val for key, val in result.items() if not key.startswith("_")}
    json.dumps(public, allow_nan=False)
    config = dict(ranks=[2], source_nodes=2, band_probes=2, random_probes=2,
                  steps=4, action_tolerance=1e-8, quadrature_tolerance=1e-8)
    def forbidden(*args, **kwargs):
        raise AssertionError("Sealed numerical actions must not be reacquired")
    reused = run_experiment(samples, config, precomputed_actions=result["_actions"],
        precomputed_endpoint_checks=result["endpoint_quadrature_checks"],
        acquisition_receipts=[dict(work=result["work"], path_work=result["acquisition_path_work"], source_artifact_sha256="sealed-fixture")],
        action_callback=forbidden, endpoint_reference_callback=forbidden)
    assert reused["work"]["reused_numerical_acquisition"]
    assert reused["work"]["homogeneous_solves"] == result["work"]["homogeneous_solves"]
    assert reused["work"]["forced_solves"] == result["work"]["forced_solves"]
    assert reused["probe_plan_sha256"] == result["probe_plan_sha256"]
    assert reused["acquisition_path_work"] == result["acquisition_path_work"]
    assert reused["acquisition_path_work_status"] == "RECORDED"
    assert sum(row["derivative_evaluations"] for row in reused["acquisition_path_work"]) > 0
    assert sum(row["derivative_evaluations"] for row in reused["aggregation_path_work"]) < sum(row["derivative_evaluations"] for row in reused["acquisition_path_work"])
    for left, right in zip(result["action_rows"], reused["action_rows"]):
        assert left.get("relative_error") == pytest.approx(right.get("relative_error")) if left.get("relative_error") is not None else right.get("relative_error") is None
    missing_checks = run_experiment(samples, config, precomputed_actions=result["_actions"])
    assert all(row["status"] == "UNRESOLVED" for row in missing_checks["endpoint_rows"])
    assert missing_checks["acquisition_path_work"] == []
    assert missing_checks["acquisition_path_work_status"] == "PARTIAL_OR_UNAVAILABLE"
    json.dumps({key: val for key, val in missing_checks.items() if not key.startswith("_")}, allow_nan=False)
    with pytest.raises(ValueError, match="exactly cover"):
        run_experiment(samples, config, precomputed_actions=result["_actions"][:-1])
    strict = run_experiment(samples[:1], dict(ranks=[2], source_nodes=1, band_probes=0,
                           random_probes=0, steps=1, action_tolerance=1e-25))
    assert strict["coverage"]["resolved_probes"] == 0
    assert any(row["status"] == "UNRESOLVED" for row in strict["method_rows"])


def test_projected_fine_and_cross_split_parent_are_never_pooled():
    geom = Geometry((4,), (1.,)); eq = Physics(.01, 1.)
    with pytest.raises(ValueError, match="projected-fine"):
        physical_whiten(_state(geom), geom, "galerkin_projected_fine")
    base = dict(u0=_state(geom).float(), geom=geom, eq=eq)
    samples = [dict(base, record=dict(parent_id="same", query_id=split, split=split,
                    horizon=.01, target_kind="fd_same_grid")) for split in ("train", "diagnostic")]
    with pytest.raises(ValueError, match="crosses"):
        run_experiment(samples, dict(source_nodes=1, band_probes=0, random_probes=0))


def test_bounded_acquisition_preserves_requested_probes_on_budget_exhaustion():
    geom = Geometry((5,), (1.,)); eq = Physics(.01, 1.)
    sample = dict(u0=_state(geom).float(), geom=geom, eq=eq,
                  record=dict(parent_id="parent", query_id="query", split="train", horizon=.03, target_kind="fd_same_grid"))
    class Exhausted:
        def check(self):
            raise TimeoutError("Declared query work exhausted")
    collected = collect_experiment_actions([sample], dict(source_nodes=2, source_fractions=[0., .5],
                     probes_per_type=1, ranks=[2]), budget=Exhausted())
    assert collected["coverage"]["requested_probes"] == 10
    assert collected["coverage"]["resolved_probes"] == 0
    assert len(collected["measurement_rows"]) == 10
    assert all(row["status"] == "UNRESOLVED" for row in collected["measurement_rows"])
    assert collected["endpoint_quadrature_checks"][0]["check"]["status"] == "UNRESOLVED"
    json.dumps({key: val for key, val in collected.items() if not key.startswith("_")}, allow_nan=False)
