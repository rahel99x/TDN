"""Convex attribution checks against known optima and native physical bases."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from tdn.analysis.resolve.fit import AffineObjective, convex_fit, adam_fit, run, uncertainty_adjusted_ratio
from tdn.analysis.advance.models import make_model
from tdn.analysis.frontier.core import Context
from tdn.analysis.frontier.neural import eq_geom
from tdn.research.experiment import Budget, horizon_key


def identifiable_objective(gains=(.6, 1.4)):
    basis = np.array([[[1., 0.], [0., 1.]], [[1., 1.], [1., -1.]]])
    return AffineObjective(basis, -basis @ np.array(gains), [1., 1.], ["a", "b"])


def test_convex_fit_encloses_known_interior_minimum():
    objective = identifiable_objective()
    fit = convex_fit(objective, atol=1e-11, rtol=1e-10, max_iterations=120)
    assert fit["converged"]
    assert fit["gains"] == pytest.approx([.6, 1.4], abs=3e-5)
    assert fit["lower_bound"] <= 1e-12
    assert fit["objective"] >= 0
    assert fit["objective_gap"] <= fit["tolerance"]
    assert fit["proof_status"] == "NOT_A_PROOF"


def test_convex_fit_finds_box_boundary_instead_of_tanh_saturation():
    objective = AffineObjective([[[1., 0.], [0., 1.]]], [[-3., .5]], [1.], ["a"], peak_weight=0.)
    fit = convex_fit(objective)
    assert fit["converged"]
    assert fit["gains"] == pytest.approx([1.75, .25], abs=1e-10)
    assert fit["objective"] == pytest.approx((1.25**2+.75**2)/2)
    assert fit["lower_bound"] == pytest.approx(fit["objective"])


def test_nonsmooth_tied_maximum_uses_valid_cut_and_tightens():
    # min max((g-.3)^2,(g-1.3)^2) occurs at .8, where max is nonsmooth.
    objective = AffineObjective([[[1.], [1.]]], [[-.3, -1.3]], [1.], ["a"], peak_weight=10.)
    fit = convex_fit(objective, free=(0,), fixed=(1.,), max_iterations=120, rtol=1e-8)
    assert fit["converged"]
    assert fit["gains"] == pytest.approx([.8], abs=1e-7)
    exact = objective.value_gradient([.8])[0]
    assert fit["lower_bound"] <= exact + 1e-12 <= fit["objective"] + 1e-12
    assert fit["objective_gap"] <= fit["tolerance"]


def test_subgradient_supports_all_sampled_points_including_max_ties():
    objective = identifiable_objective()
    x = np.array([1., 1.]); value, gradient = objective.value_gradient(x)
    rng = np.random.default_rng(13)
    for y in rng.uniform(.25, 1.75, size=(100, 2)):
        assert objective.value_gradient(y)[0] >= value + gradient @ (y-x) - 1e-12


def test_repeating_one_fields_queries_does_not_change_field_weight():
    objective = identifiable_objective()
    repeated = AffineObjective(objective.basis[[0, 0, 1]], objective.offset[[0, 0, 1]],
        [1., 1., 1.], ["a", "a", "b"])
    assert repeated.value_gradient([1., 1.])[0] == pytest.approx(objective.value_gradient([1., 1.])[0])
    assert repeated.gram()["gram"] == objective.gram()["gram"]


def test_identifiability_does_not_report_collinear_basis_as_independent():
    objective = AffineObjective([[[1., 2.], [2., 4.]]], [[0., 0.]], [1.], ["a"])
    result = objective.gram()
    assert result["numerical_rank"] == 1
    assert result["condition_number"] is None
    assert result["normalized_correlation"][0][1] == pytest.approx(1.)


def test_quadratic_only_and_fixed_cubic_controls_have_different_constraints():
    objective = identifiable_objective()
    omitted = convex_fit(objective, free=(0,), fixed=(1., 0.))
    fixed = convex_fit(objective, free=(0,), fixed=(1., 1.))
    assert omitted["gains"][1] == 0
    assert fixed["gains"][1] == 1
    assert omitted["objective"] > fixed["objective"]


def test_iteration_limit_cannot_be_reported_as_optimality():
    fit = convex_fit(identifiable_objective(), max_iterations=1, atol=0., rtol=0.)
    assert not fit["converged"]
    assert fit["status"] == "ITERATION_LIMIT"
    assert fit["objective_gap"] > 0


def test_adam_matches_same_affine_objective_and_never_uses_validation():
    objective = identifiable_objective()
    fit = adam_fit(objective, updates=100, learning_rate=.03)
    assert fit["objective"] < objective.value_gradient([1., 1.])[0]
    assert fit["objective"] == pytest.approx(objective.value_gradient(fit["gains"])[0], abs=1e-13)
    assert fit["selection"] == "Minimum training objective only"
    assert convex_fit(objective, atol=1e-10, rtol=1e-9)["objective"] <= fit["objective"] + 1e-7


@pytest.mark.parametrize("track", ["discrete", "continuum"])
def test_cached_basis_native_assembly_matches_model(track):
    torch.set_num_threads(1)
    model = make_model("two_basis", track, dict(modes=1, width=4, depth=1)).float()
    parent = dict(kappa=.01, reaction_rate=1.)
    eq, geom = eq_geom(parent, 8)
    x = torch.arange(8)/8
    u = (.3+.04*torch.cos(2*torch.pi*x[:, None])+.025*torch.sin(4*torch.pi*x[None, :]))[None, None]
    with torch.no_grad():
        model.gain_logits.copy_(torch.tensor([.12, -.2]))
        parts = model.correction_components(u, .03, eq, geom)
        gains = model.gains()
        native = parts["base"] + (gains[0]*parts["quadratic"]+gains[1]*parts["cubic"])
        assert torch.equal(native, model(u, .03, eq, geom))


def _test_context(tmp_path):
    config = dict(modes=1, width=4, depth=1, reaction_substeps=4)
    protocol = dict(profile="smoke", model_config=config, train_grid=4, train_horizons=[.03], validation_horizons=[.03],
        mechanisms={"R1": {}}, combinations={}, units={"fit": dict(track="discrete", seed=42, mechanism_ids=["R1"])},
        resolve=dict(fit=dict(direct_max_iterations=12, adam_updates=2)))
    ctx = Context(protocol, "fit", tmp_path / "fit", {"prepare": tmp_path / "prepare"}, "cpu", Budget(60))
    bank = []
    model = make_model("two_basis", "discrete", config).float()
    for index, split in enumerate(("train", "validation")):
        parent = dict(parent_id=split, field_cluster=split, split=split, kappa=.01, reaction_rate=1., regime="overlapping")
        eq, geom = eq_geom(parent, 4)
        x = torch.arange(4, dtype=torch.float64)/4
        u = (.3+.03*torch.cos(2*torch.pi*x[:, None]+index)+.02*torch.cos(2*torch.pi*(x[:, None]+x[None, :])))[None, None]
        with torch.no_grad():
            target = model(u.float(), .03, eq, geom).double()
        parent["states"] = {"4": u}
        parent["references"] = {f"discrete:4:{horizon_key(.03)}": dict(state=target,
            accepted=True, uncertainty_rms=0., uncertainty_max_bound=0.)}
        bank.append(parent)
    return ctx, bank


def test_stage_preserves_controls_raw_endpoint_rows_and_training_selection(tmp_path, monkeypatch):
    from tdn.analysis.resolve import fit
    torch.set_num_threads(1)
    ctx, bank = _test_context(tmp_path)
    monkeypatch.setattr(fit, "load_bank", lambda context, name: bank)
    run(ctx)
    result = json.loads((ctx.path / "fit_results.json").read_text())
    rows = json.loads((ctx.path / "fit_endpoints.json").read_text())["rows"]
    assert set(result["fits"]) == {"fixed_unit", "quadratic_only_direct", "quadratic_gain_direct", "cubic_gain_direct", "two_gain_direct", "two_gain_adam"}
    assert {r["method"] for r in rows} >= {"quad2_full", "quad4_full", "analytic_quad_cubic"}
    assert all(r["split"] == "validation" and r["reference_accepted"] for r in rows)
    assert result["parent_ids"] == {"train": ["train"], "validation": ["validation"]}
    assert all(v["inference_latency_seconds"] is None for v in result["validation"].values())
    assert result["train_gram"]["coefficient_count"] == 2
    assert ctx.rows and ctx.rows[0]["checks"][-1]["verdict"] == "NA"


def test_stage_rejects_confirmation_truth_before_fitting(tmp_path, monkeypatch):
    from tdn.analysis.resolve import fit
    ctx, bank = _test_context(tmp_path)
    bank[1]["split"] = "confirmation"
    monkeypatch.setattr(fit, "load_bank", lambda context, name: bank)
    with pytest.raises(ValueError, match="train/validation"):
        run(ctx)


def test_completed_optimizers_are_reused_without_duplicate_science_rows(tmp_path, monkeypatch):
    from tdn.analysis.resolve import fit
    ctx, bank = _test_context(tmp_path)
    monkeypatch.setattr(fit, "load_bank", lambda context, name: bank)
    run(ctx)
    original = (ctx.path / "two_gain_direct.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed optimizer was rerun")
    monkeypatch.setattr(fit, "convex_fit", forbidden)
    monkeypatch.setattr(fit, "adam_fit", forbidden)
    run(ctx)
    assert (ctx.path / "two_gain_direct.json").read_bytes() == original
    assert len(ctx.rows) == 1
    ledger = json.loads((ctx.path / "fit_attempt_costs.json").read_text())
    assert len(ledger["attempts"]) == 2
    assert all(a["status"] == "COMPLETED" for a in ledger["attempts"])


def test_interrupted_optimizer_cost_and_progress_remain_visible(tmp_path, monkeypatch):
    from tdn.analysis.resolve import fit
    ctx, bank = _test_context(tmp_path)
    monkeypatch.setattr(fit, "load_bank", lambda context, name: bank)
    actual_adam = fit.adam_fit
    def interrupted(*args, **kwargs):
        kwargs["progress"](dict(update=1, objective=1., elapsed_seconds=.1))
        raise TimeoutError("bounded trial interrupted")
    monkeypatch.setattr(fit, "adam_fit", interrupted)
    with pytest.raises(TimeoutError, match="bounded"):
        run(ctx)
    assert (ctx.path / "two_gain_direct.json").is_file()
    monkeypatch.setattr(fit, "adam_fit", actual_adam)
    run(ctx)
    ledger = json.loads((ctx.path / "fit_attempt_costs.json").read_text())
    assert [a["status"] for a in ledger["attempts"]] == ["INTERRUPTED_OR_FAILED", "COMPLETED"]
    assert ledger["observed_total_seconds"] >= sum(a["elapsed_seconds"] for a in ledger["attempts"])
    retained = json.loads((ctx.path / "two_gain_adam-interrupted-000.json").read_text())
    assert retained["last_completed_progress"]["update"] == 1


@pytest.mark.parametrize("scales,peak", [([0.], .1), ([float("nan")], .1), ([1.], -.1)])
def test_invalid_objective_cannot_produce_scores(scales, peak):
    with pytest.raises(ValueError, match="finite arrays"):
        AffineObjective([[[1., 1.]]], [[0.]], scales, ["a"], peak_weight=peak)


def test_uncertainty_adjustment_blocks_roundoff_gain_and_retains_na():
    common = dict(field_cluster="a", parent_id="a", grid=8, horizon=.03, track="discrete",
        finite=True, reference_accepted=True, reference_uncertainty_rms=1e-8)
    rows = [dict(common, method="control", error_rms=2e-8), dict(common, method="candidate", error_rms=1e-8)]
    assert uncertainty_adjusted_ratio(rows, "control", "candidate") == pytest.approx(.5)
    rows[0]["reference_uncertainty_rms"] = None
    assert uncertainty_adjusted_ratio(rows, "control", "candidate") is None
    rows[0]["reference_uncertainty_rms"] = 0
    rows[1]["reference_accepted"] = False
    assert uncertainty_adjusted_ratio(rows, "control", "candidate") is None
