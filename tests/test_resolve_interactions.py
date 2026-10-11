"""Mathematical controls for independently paired finite-step diagnostics."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.resolve import interactions as d
from tdn.analysis.frontier import numerics as physical
from tdn.numerics import Equation, Geometry


@pytest.mark.parametrize("regime", d.REGIMES)
def test_continuous_field_is_identical_under_actual_grid_refinement(regime):
    declared = d.field_declaration(16, regime, .7)
    coarse = d.field_state(declared, 16)
    fine = d.field_state(declared, 64)
    torch.testing.assert_close(coarse, fine[..., ::4, ::4], atol=1e-14, rtol=0)
    assert float(coarse.mean()) == pytest.approx(.4)
    assert float(coarse.var(unbiased=False)) == pytest.approx(.04**2)
    assert float(fine.min()) > 0 and float(fine.max()) < 1
    if regime == "grid_relative":
        assert declared["frequency_scope"] == "grid_relative_stress"
        assert d.field_declaration(32, regime, .7)["terms"] != declared["terms"]
    else:
        assert d.field_declaration(32, regime, .7) == declared


@pytest.mark.parametrize("track", ["discrete", "continuum"])
def test_diffusion_and_reaction_interventions_are_independent_on_same_field(track):
    geom = Geometry((16, 16), (1., 1.))
    u = d.field_state(d.field_declaration(16, "overlapping_triads", 0), 16)
    original = u.clone()
    a, _ = d.physical_parameters(u, .12, geom, track, .1, .2)
    b, _ = d.physical_parameters(u, .12, geom, track, 5., .2)
    c, _ = d.physical_parameters(u, .12, geom, track, .1, 3.)
    assert a.reaction_rate == b.reaction_rate
    assert a.kappa == c.kappa
    assert b.kappa / a.kappa == pytest.approx(50.)
    assert c.reaction_rate / a.reaction_rate == pytest.approx(15.)
    v = u - u.mean()
    av = physical.apply_multiplier(v, physical.diffusion_symbol(u, b, geom, track))
    assert float(-.12 * (av * v).mean() / v.square().mean()) == pytest.approx(5.)
    torch.testing.assert_close(u, original, atol=0, rtol=0)


def test_decomposition_exposes_cancellation_instead_of_summing_error_norms():
    shape = (1, 1, 2, 2)
    truth = torch.zeros(shape, dtype=torch.float64)
    temporal_target = torch.ones(shape, dtype=torch.float64)
    galerkin_target = torch.full(shape, .75, dtype=torch.float64)
    prediction = torch.full(shape, .05, dtype=torch.float64)
    result = d.error_decomposition(prediction, temporal_target, galerkin_target, truth)
    assert result["component_rms"] == pytest.approx(dict(temporal=.95, discretization=.25, closure=.75))
    assert result["total_rms"] == pytest.approx(.05)
    assert result["pair_inner_products"]["temporal:closure"] < 0
    assert abs(result["squared_identity_residual"]) < 3e-16
    assert result["vector_identity_max"] < 3e-16


@pytest.mark.parametrize("track", ["discrete", "continuum"])
def test_declared_overlap_and_near_collinearity_break_the_previous_orthogonal_design(track):
    geom = Geometry((16, 16), (1., 1.))
    cosines = {}
    for regime in ("overlapping_triads", "aligned_harmonics"):
        u = d.field_state(d.field_declaration(16, regime, 0), 16)
        eq, _ = d.physical_parameters(u, .12, geom, track, 1., 1.)
        _, components, _ = d.predictions(u, .12, eq, geom, track)
        a, b = components["q2"].flatten(), components["filtered_c3"].flatten()
        cosines[regime] = float(torch.dot(a, b) / (a.norm() * b.norm()))
    assert abs(cosines["overlapping_triads"]) > .2
    assert abs(cosines["aligned_harmonics"]) > .9
    # The condition is measured on these cases, never imposed as a theorem.


def test_gram_reports_nonidentifiability_and_zero_directions():
    a = torch.tensor([1., 0., -1.], dtype=torch.float64)
    result = d.response_geometry(a, {"a": a, "duplicate": 2 * a, "zero": a * 0})
    assert result["pair_oracles"]["a:duplicate"]["rank"] == 1
    assert result["pair_oracles"]["a:duplicate"]["condition_number"] is None
    assert result["normalized_gram"][0][1] == pytest.approx(1.)
    assert result["normalized_gram"][0][2] is None


@pytest.mark.parametrize("track", ["discrete", "continuum"])
def test_refined_targets_separate_pure_diffusion_discretization_from_closure(track):
    declaration = d.field_declaration(8, "overlapping_triads", 0)
    geometry = Geometry((8, 8), (1., 1.))
    initial = d.field_state(declaration, 8)
    equation = Equation(.01, 0.)
    options = dict(substeps=2, max_substeps=8, tolerance=1e-9,
                   roundoff_floor=1e-12, spatial_factors=[2, 4])
    targets, references = d.refined_targets(declaration, 8, .12, equation, track, options,
        SimpleNamespace(check=lambda: None))
    expected = physical.heat_step(initial, .12, equation, geometry, track)
    torch.testing.assert_close(targets["same_grid"], expected, atol=3e-15, rtol=0)
    result = d.error_decomposition(expected, **targets)
    assert result["component_rms"]["temporal"] < 3e-15
    assert result["component_rms"]["closure"] < 3e-15
    assert result["component_rms"]["discretization"] > 1e-4 if track == "discrete" else result["component_rms"]["discretization"] == 0
    assert all(references[key]["accepted"] for key in targets)
    assert references["projected_fine"]["spatial_reference_grids"] == [16, 32]
    assert references["same_grid"]["refinement_substeps"] == [2, 4, 8]


def test_corrections_preserve_commuting_limits():
    u = d.field_state(d.field_declaration(8, "overlapping_triads", 0), 8)
    geom = Geometry((8, 8), (1., 1.))
    for eq in (Equation(0., 3.), Equation(.01, 0.)):
        _, components, _ = d.predictions(u, .12, eq, geom, "discrete")
        for name in ("q2", "q4", "q8", "q16", "raw_c3", "filtered_c3", "nested_c3"):
            assert torch.count_nonzero(components[name]) == 0


def test_unresolved_teacher_keeps_raw_error_but_cannot_earn_gap_success(tmp_path):
    config = dict(horizon=.12, regimes=["overlapping_triads"], phases=[0.],
                  diffusion_numbers=[.1], reaction_numbers=[.1],
                  reference=dict(substeps=2, max_substeps=8, tolerance=1e-30, spatial_factors=[2, 4]))
    recorded = []
    ctx = SimpleNamespace(protocol={"resolve": {"interactions": config}},
        unit=dict(track="discrete", grid=8), path=tmp_path,
        budget=SimpleNamespace(check=lambda: None),
        record=lambda identity, mechanisms, **row: recorded.append(row))
    summary = d.run(ctx)
    assert summary["cases"] == 1 and summary["endpoint_rows"] == len(d.METHODS)
    assert summary["accepted_cases"] == 0 and summary["unresolved_cases"] == 1
    rows = json.loads((tmp_path / "interaction_rows.json").read_text())["rows"]
    assert all(row["same_grid_metrics"]["error_rms"] >= 0 for row in rows)
    assert all(not row["same_grid_metrics"]["reference_accepted"] for row in rows)
    assert all(next(c for c in row["checks"] if c["check_id"] == "uncertainty-separated-rms-improvement-over-DF")["verdict"] == "NA" for row in recorded)
    cases = json.loads((tmp_path / "interaction_cases.json").read_text())["cases"]
    assert cases[0]["references"]["same_grid"]["temporal_refinements"]
    assert cases[0]["response_geometry"]["pair_oracles"]


def test_panel_retains_the_same_state_across_both_parameter_axes(tmp_path):
    config = dict(horizon=.12, regimes=["overlapping_triads"], phases=[0.],
                  diffusion_numbers=[.1, 1.], reaction_numbers=[.1, 1.],
                  reference=dict(substeps=4, max_substeps=64, tolerance=2e-7, spatial_factors=[2, 4]))
    recorded = []
    ctx = SimpleNamespace(protocol={"resolve": {"interactions": config}},
        unit=dict(track="continuum", grid=8), path=tmp_path,
        budget=SimpleNamespace(check=lambda: None),
        record=lambda identity, mechanisms, **row: recorded.append(row))
    summary = d.run(ctx)
    cases = json.loads((tmp_path / "interaction_cases.json").read_text())["cases"]
    assert summary["cases"] == 4 and summary["distinct_fields"] == 1
    assert summary["independent_sampling_claim"] is False
    assert len({case["field_state_sha256"] for case in cases}) == 1
    assert {(case["diffusion_number"], case["reaction_number"]) for case in cases} == {(.1, .1), (.1, 1.), (1., .1), (1., 1.)}
    assert all(case["response_geometry"]["basis_rms"]["q2"] > 0 for case in cases)
    assert all(next(c for c in row["checks"] if c["check_id"] == "error-vector-identity")["verdict"] == "GOOD" for row in recorded)


def test_failed_nonfinite_reference_is_retained_without_fake_accuracy_or_oracle(tmp_path, monkeypatch):
    # The failure is injected at the teacher boundary; methods still perform
    # their real numerical work. A nonfinite teacher must not reach LAPACK or
    # be silently replaced by the initial state as an apparently exact target.
    original = d.refined_targets
    def failed_teacher(*args, **kwargs):
        states, references = original(*args, **kwargs)
        states["same_grid"][:] = float("nan")
        references["same_grid"].update(accepted=False, uncertainty_rms=None,
                                       uncertainty_max_bound=None, reason="injected teacher failure")
        return states, references
    monkeypatch.setattr(d, "refined_targets", failed_teacher)
    config = dict(horizon=.12, regimes=["overlapping_triads"], phases=[0.],
                  diffusion_numbers=[.1, 1.], reaction_numbers=[.1],
                  reference=dict(substeps=2, max_substeps=8, tolerance=1e-6, spatial_factors=[2, 4]))
    recorded = []
    ctx = SimpleNamespace(protocol={"resolve": {"interactions": config}},
        unit=dict(track="discrete", grid=8), path=tmp_path,
        budget=SimpleNamespace(check=lambda: None),
        record=lambda identity, mechanisms, **row: recorded.append(row))
    summary = d.run(ctx)
    assert summary["cases"] == 2 and summary["unresolved_cases"] == 2
    assert summary["endpoint_rows"] == 2 * len(d.METHODS)
    rows = json.loads((tmp_path / "interaction_rows.json").read_text())["rows"]
    assert all(row["same_grid_metrics"]["error_rms"] is None for row in rows)
    assert all(row["same_grid_metrics"]["reference_status"] == "NONFINITE_REFERENCE" for row in rows)
    assert all(row["error_decomposition"]["vector_identity_max"] is None for row in rows)
    assert all(row["comparisons_to"]["gl4"]["uncertainty_separated_improvement"] is None for row in rows)
    assert all(next(c for c in row["checks"] if c["check_id"] == "uncertainty-separated-rms-improvement-over-DF")["verdict"] == "NA" for row in recorded)
    cases = json.loads((tmp_path / "interaction_cases.json").read_text())["cases"]
    assert all(case["response_geometry"]["status"] == "NA_NONFINITE_REFERENCE_OR_COMPONENT" for case in cases)


def test_one_method_numeric_failure_is_logged_and_does_not_remove_other_methods(tmp_path, monkeypatch):
    def failing_etdrk4(*args, **kwargs):
        raise FloatingPointError("injected method failure")
    monkeypatch.setattr(d.physical, "etdrk4_step", failing_etdrk4)
    config = dict(horizon=.12, regimes=["overlapping_triads"], phases=[0.],
                  diffusion_numbers=[.1], reaction_numbers=[.1],
                  reference=dict(substeps=2, max_substeps=16, tolerance=2e-6, spatial_factors=[2, 4]))
    recorded = []
    ctx = SimpleNamespace(protocol={"resolve": {"interactions": config}},
        unit=dict(track="discrete", grid=8), path=tmp_path,
        budget=SimpleNamespace(check=lambda: None),
        record=lambda identity, mechanisms, **row: recorded.append(row))
    summary = d.run(ctx)
    assert summary["endpoint_rows"] == len(d.METHODS)
    rows = json.loads((tmp_path / "interaction_rows.json").read_text())["rows"]
    failed = next(row for row in rows if row["method"] == "etdrk4")
    assert failed["same_grid_metrics"]["finite"] is False
    assert failed["same_grid_metrics"]["error_rms"] is None
    assert failed["component_failures_in_case"]["etdrk4"].startswith("FloatingPointError")
    assert all(row["same_grid_metrics"]["finite"] for row in rows if row["method"] != "etdrk4")
    cases = json.loads((tmp_path / "interaction_cases.json").read_text())["cases"]
    assert cases[0]["response_geometry"]["pair_oracles"]
    for row in recorded:
        if row["metrics"]["method"] == "etdrk4":
            assert next(c for c in row["checks"] if c["check_id"] == "finite-endpoint")["verdict"] == "BAD"
