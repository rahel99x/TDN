"""Adversarial policy/statistical checks: no hidden truth or duplicated units."""
from __future__ import annotations

import json
import math

import pytest
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.subflows import reaction_step
from tdn.analysis.roadmap.policy import (
    BareDiffusionFirst, acceptance_decision, adaptive_classical, calibrate_envelope,
    deploy_policy, estimate_attempt, interior_residual_indicator, spatial_discrepancy,
)
from tdn.analysis.roadmap.statistics import (
    assert_disjoint_cohorts, conformal_quantile, cost_distribution,
    independent_function_scores, paired_cluster_bootstrap, selective_risk_summary,
)


class IdentityFlow(torch.nn.Module):
    def forward(self, state, step, equation, geometry):
        # Keep a zero horizon derivative connected to the query for JVP.
        return state + step * torch.zeros_like(state)


class ReactionFlow(torch.nn.Module):
    def forward(self, state, step, equation, geometry):
        return reaction_step(state, step, equation)


def _calibration(multiplier=1., quantile=1.):
    return {"empirical_multiplier": multiplier, "conformal": {"quantile": quantile}}


def _field(grid):
    n, m = tuple(grid)
    x = torch.arange(n, dtype=torch.float64) / n
    y = torch.arange(m, dtype=torch.float64) / m
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return (.5 + .1 * torch.cos(2 * torch.pi * 2 * xx) + .03 * torch.sin(2 * torch.pi * yy))[None, None]


def test_finite_conformal_boundary_and_json_safe_na():
    short = conformal_quantile(range(98), .01)
    assert short["quantile"] is None
    assert short["status"] == "INSUFFICIENT_INDEPENDENT_FUNCTIONS"
    assert short["minimum_functions_for_finite_quantile"] == 99
    enough = conformal_quantile(range(99), .01)
    assert enough["quantile"] == 98
    assert enough["rank"] == 99
    assert not enough["conditional_on_acceptance_guarantee"]
    json.dumps(short, allow_nan=False)


@pytest.mark.parametrize("values,alpha", [([float('nan')], .1), ([-1.], .1), ([1.], 1.), ([1.], 0.)])
def test_conformal_refuses_invalid_inputs(values, alpha):
    with pytest.raises(ValueError):
        conformal_quantile(values, alpha)


def test_repeated_grids_seeds_are_one_conformal_function():
    repeated = [{"field_cluster": "one", "score": i / 10} for i in range(100)]
    collapsed = independent_function_scores(repeated)
    assert collapsed == [{"field_cluster": "one", "score": 9.9, "query_rows": 100}]
    assert conformal_quantile([r["score"] for r in collapsed], .01)["quantile"] is None


def test_joint_router_score_protects_data_dependent_candidate_selection():
    # Each marginal candidate errs on a different field. The bank score must
    # retain the bad maximum for both, rather than averaging it away.
    a = [{"field_cluster": "a", "score": 10.}, {"field_cluster": "b", "score": 1.}]
    b = [{"field_cluster": "a", "score": 1.}, {"field_cluster": "b", "score": 20.}]
    joint = independent_function_scores(a + b)
    assert [r["score"] for r in joint] == [10., 20.]
    assert conformal_quantile([r["score"] for r in joint], .4)["quantile"] == 20.


def test_calibration_rejects_confirmation_and_empty_bank():
    row = {"field_cluster": "a", "split": "confirmation", "indicator": {"rms": 1., "max": 1.},
           "upper_error": {"rms": 1., "max": 1.}}
    with pytest.raises(ValueError, match="calibration"):
        calibrate_envelope([row])
    with pytest.raises(ValueError, match="independent function"):
        calibrate_envelope([])


def test_calibration_joint_norm_maximum_not_duplicated_count():
    rows = [{"field_cluster": "a", "split": "calibration", "indicator": {"rms": 1., "max": 2.},
             "upper_error": {"rms": 2., "max": 12.}}] * 20
    envelope = calibrate_envelope(rows, safety_factor=1.1)
    assert envelope["empirical_multiplier"] == pytest.approx(6.6)
    assert envelope["conformal"]["independent_functions"] == 1
    assert envelope["conformal"]["quantile"] is None


def test_cohort_contamination_detects_renamed_continuous_field():
    terms = [{"mode": [1, 2], "weight": 1., "phase": .31}]
    a = {"parent_id": "train", "field_cluster": "train-field", "field_terms": terms}
    b = {"parent_id": "new-name", "field_cluster": "new-field", "field_terms": terms}
    with pytest.raises(ValueError, match="contamination"):
        assert_disjoint_cohorts(train=[a], confirmation=[b])
    b["field_terms"] = [{"mode": [1, 2], "weight": 1., "phase": .32}]
    assert assert_disjoint_cohorts(train=[a], confirmation=[b])["disjoint"]


def test_bootstrap_preserves_pairing_and_does_not_inflate_query_rows():
    rows = [{"field_cluster": f"f{f}", "seed": s, "difference": float(f + s)} for f in range(4) for s in range(3)]
    first = paired_cluster_bootstrap(rows, repeats=100)
    second = paired_cluster_bootstrap(rows * 20, repeats=100)
    assert first["independent_fields"] == second["independent_fields"] == 4
    assert first["training_seeds"] == second["training_seeds"] == 3
    for key in ("mean", "lower", "upper"):
        assert first[key] == second[key]
    assert first["seed_generalization_tested"]


def test_bootstrap_one_field_or_incomplete_cross_is_na():
    rows = [{"field_cluster": "one", "seed": s, "difference": float(s)} for s in range(20)]
    result = paired_cluster_bootstrap(rows, repeats=20)
    assert result["status"] == "INSUFFICIENT_INDEPENDENT_FIELDS"
    assert result["lower"] is None
    result = paired_cluster_bootstrap(rows + [{"field_cluster": "two", "seed": 0, "difference": 1.}], repeats=20)
    assert result["status"] == "INCOMPLETE_CROSSED_DESIGN"


def test_selective_risk_uses_accepted_functions_and_requires_299():
    duplicate = [{"field_cluster": "one", "accepted": True, "false_accept": False}] * 500
    result = selective_risk_summary(duplicate)
    assert result["accepted_independent_functions"] == 1
    assert result["one_sided_binomial_upper"] == pytest.approx(.95)
    assert result["zero_failure_functions_needed"] == 299
    enough = selective_risk_summary([{**duplicate[0], "field_cluster": str(i)} for i in range(299)])
    assert enough["one_sided_binomial_upper"] <= .01
    assert enough["target_supported_under_iid_assumptions"]
    assert not enough["conformal_marginal_coverage_is_conditional_risk"]


def test_selective_risk_aggregates_any_bad_query_and_no_accepts_are_na():
    rows = [{"field_cluster": "one", "accepted": True, "false_accept": False},
            {"field_cluster": "one", "accepted": True, "false_accept": True}]
    result = selective_risk_summary(rows)
    assert result["bad_accepted_functions"] == 1
    assert result["conditional_function_risk"] == 1.
    result = selective_risk_summary([{"field_cluster": "one", "accepted": False, "false_accept": False}])
    assert result["one_sided_binomial_upper"] is None
    json.dumps(result, allow_nan=False)


def test_cost_tails_can_reverse_a_favorable_median():
    learned, classical = cost_distribution([.1, .1, 5.]), cost_distribution([1., 1., 1.])
    assert learned["median_seconds"] < classical["median_seconds"]
    assert learned["total_seconds"] > classical["total_seconds"]
    assert learned["p95_seconds"] > classical["p95_seconds"]


def test_wrong_generator_semigroup_passes_doubling_but_residual_detects_bias():
    state = torch.full((1, 1, 4, 4), .2, dtype=torch.float64)
    eq, geo = Equation(.01, 1.), Geometry((4, 4), (1., 1.))
    _, indicators, _ = estimate_attempt(IdentityFlow(), state, [.2], eq, geo, estimator="step_doubling")
    assert indicators["total"]["max"] == 0.
    residual, diagnostics = interior_residual_indicator(IdentityFlow(), state, [.2], eq, geo)
    assert residual["max"] > .03
    assert diagnostics["sampled_reconstruction_in_domain"]
    assert not diagnostics["deterministic_certificate"]
    assert diagnostics["jvp_calls"] == 6


def test_exact_logistic_flow_residual_is_small():
    state = torch.full((1, 1, 4, 4), .2, dtype=torch.float64)
    estimates, info = interior_residual_indicator(ReactionFlow(), state, [.12, .08], Equation(0., 1.), Geometry((4, 4), (1., 1.)))
    assert estimates["max"] < 1e-12
    assert info["identity_jump"]["max"] == 0.


def test_spatial_bias_is_visible_when_temporal_doubling_is_zero():
    initial = _field((8, 8))
    eq, geo = Equation(.1, 0.), Geometry((8, 8), (1., 1.))
    model = BareDiffusionFirst()
    _, temporal, _ = estimate_attempt(model, initial, [.05], eq, geo)
    spatial, diagnostics = spatial_discrepancy(model, initial, [.05], eq, geo, _field)
    assert temporal["total"]["max"] < 1e-12
    assert spatial["max"] > 1e-3
    assert diagnostics["richardson_order_assumed"] is None
    assert not diagnostics["deterministic_certificate"]


def test_insufficient_conformal_quantile_rejects_before_acceptance():
    accepted, diagnostics = acceptance_decision(torch.full((1, 1, 2, 2), .5), {"rms": 0., "max": 0.},
        _calibration(quantile=None), {"rms": 1., "max": 1.}, calibration_mode="conformal")
    assert not accepted
    assert diagnostics["reason"] == "INSUFFICIENT_INDEPENDENT_FUNCTIONS"


def test_physical_violation_rejects_even_with_tiny_error_indicator():
    accepted, _ = acceptance_decision(torch.full((1, 1, 2, 2), 1.1), {"rms": 0., "max": 0.},
        _calibration(), {"rms": 1., "max": 1.})
    assert not accepted


def test_standalone_policy_charges_rejections_and_fallback():
    eq, geo = Equation(.1, 2.), Geometry((8, 8), (1., 1.))
    value, info = deploy_policy({"df": BareDiffusionFirst()}, ["df"], _field((8, 8)), .2, eq, geo,
        {"df": _calibration(multiplier=1e20)}, {"rms": 1e-6, "max": 1e-6}, spatial_guard=False,
        step_sizes=(.2, .1), max_attempts=2, max_refinements=3, track="discrete")
    assert torch.isfinite(value).all()
    assert info["fallback"]
    assert info["rejected_attempts"] == 2
    assert info["proposal_seconds"] > 0
    assert info["estimator_seconds"] > 0
    assert info["rejected_work_seconds"] > 0
    assert info["fallback_seconds"] > 0
    assert info["accounted_component_seconds"] <= info["standalone_seconds"] * 1.001
    assert not info["truth_used_in_decision"]


def test_formal_na_policy_really_executes_fallback_without_neural_proposal():
    class NeverCall(torch.nn.Module):
        def forward(self, *args):
            raise AssertionError("Conformal NA should skip a known-unusable proposal")
    initial = torch.full((1, 1, 4, 4), .2)
    _, info = deploy_policy({"neural": NeverCall()}, ["neural"], initial, .1, Equation(0., 1.), Geometry((4, 4), (1., 1.)),
        {"neural": _calibration(quantile=None)}, {"rms": 1e-4, "max": 1e-4}, spatial_guard=False,
        calibration_mode="conformal", step_sizes=(.1,), max_attempts=1, max_refinements=2, track="discrete")
    assert info["fallback"] and info["fallback_seconds"] > 0
    assert info["proposal_seconds"] == 0
    assert info["attempts"][0]["status"] == "NO_FINITE_CONFORMAL_QUANTILE"


def test_router_can_skip_expensive_model_without_truth():
    class CountingIdentity(IdentityFlow):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, *args):
            self.calls += 1
            return super().forward(*args)
    cheap, expensive = CountingIdentity(), CountingIdentity()
    initial = torch.full((1, 1, 4, 4), .5)
    models = {"cheap": cheap, "expensive": expensive}
    envelopes = {key: _calibration() for key in models}
    _, info = deploy_policy(models, ["cheap", "expensive"], initial, .1, Equation(0., 0.), Geometry((4, 4), (1., 1.)),
        envelopes, {"rms": 1e-4, "max": 1e-4}, spatial_guard=False, step_sizes=(.1,), max_attempts=1, track="discrete")
    assert info["selected_model"] == "cheap"
    assert cheap.calls > 0 and expensive.calls == 0
    _, info = deploy_policy(models, ["cheap", "expensive"], initial, .1, Equation(0., 0.), Geometry((4, 4), (1., 1.)),
        envelopes, {"rms": 1e-4, "max": 1e-4}, spatial_guard=False, routing=False,
        step_sizes=(.1,), max_attempts=1, track="discrete")
    assert info["selected_model"] == "expensive" and expensive.calls > 0


def test_classical_controller_refines_to_a_target_and_exposes_exhaustion():
    value, info = adaptive_classical(_field((8, 8)), .2, Equation(.1, 4.), Geometry((8, 8), (1., 1.)),
        {"rms": 1e-15, "max": 1e-15}, max_refinements=2)
    assert info["controller_status"] == "REFINEMENT_CAP_EXHAUSTED"
    assert info["refinements"] == 2
    assert info["attempts"][1]["fine_steps"] == 4
    assert not info["truth_used_in_decision"]
    assert torch.isfinite(value).all()


def test_finite_conformal_quantile_cannot_approve_distribution_shift():
    envelope = _calibration()
    envelope["conformal"]["exchangeability_supported"] = False
    accepted, diagnostics = acceptance_decision(torch.full((1, 1, 2, 2), .5), {"rms": 0., "max": 0.},
        envelope, {"rms": 1., "max": 1.}, calibration_mode="conformal")
    assert not accepted
    assert diagnostics["reason"] == "EXCHANGEABILITY_NOT_ESTABLISHED"


@pytest.mark.parametrize("availability", ["all", "none", "missing_first_seed"])
def test_policy_run_seals_before_confirmation_and_validates_each_scored_log(tmp_path, monkeypatch, availability):
    from tdn.analysis.roadmap import data, neural, policy
    from tdn.analysis.roadmap.core import Context, validate_row
    from tdn.analysis.roadmap.protocol import build_protocol
    from tdn.analysis.agenda.data import continuous_field
    from tdn.research.experiment import horizon_key

    class Budget:
        def check(self):
            pass
    class FrozenReaction(ReactionFlow):
        def __init__(self, family):
            super().__init__()
            self.family = family
        def architecture_metadata(self):
            return {"family": self.family, "parameters": 0}

    protocol = build_protocol("smoke")
    protocol["console_every"] = 10000
    protocol["policies"].update(grids=[4], horizons=[.02], estimators=["step_doubling"],
        max_attempts=1, attempted_step_sizes=[.02], max_refinements=1, bootstrap_repeats=20,
        max_calibration_parents=2, max_confirmation_clusters=1, tracks=["discrete", "continuum"])
    banks = {}
    for index, split in enumerate(("train", "validation", "calibration", "confirmation")):
        bank = []
        for number in range(2 if split == "calibration" else 1):
            parent = {"parent_id": f"{split}-{number}", "field_cluster": f"field-{split}-{number}",
                "continuous_field_id": f"continuous-{split}-{number}", "split": split,
                "kappa": 0., "reaction_rate": 1., "mean": .4, "variance": .0001,
                "field_terms": [{"mode": [1, 0], "weight": 1., "phase": float(index + number / 10)}]}
            state = continuous_field(parent, 4)
            truth = reaction_step(state, .02, Equation(0., 1.))
            parent["states"] = {"4": state}
            parent["references"] = {f"{track}:4:{horizon_key(.02)}": {"state": truth,
                "accepted": True, "uncertainty_rms": 0., "uncertainty_max_bound": 0.} for track in ("discrete", "continuum")}
            bank.append(parent)
        banks[split] = bank
    stage = tmp_path / "policy"
    loads = []
    def load(ctx, split):
        if split == "confirmation":
            assert (stage / "calibration.json").is_file()
            frozen = json.loads((stage / "calibration.json").read_text())
            assert not frozen["confirmation_seen_during_calibration"]
            assert frozen["empirical_safety_factor"] == 2.
            assert frozen["joint_router_conformal_quantile"]["independent_functions"] == 2
        loads.append(split)
        return banks[split]
    monkeypatch.setattr(data, "load_bank", load)
    frozen_seed = protocol["seeds"][0]
    available_models = {f"{family}-seed-{frozen_seed + (10 if availability == 'missing_first_seed' and family == 'source' else 0)}": FrozenReaction(family)
                        for family in ("source", "rank2", "cheap_fno", "c2_rank2")} if availability != "none" else {}
    monkeypatch.setattr(neural, "load_frozen_models", lambda ctx: available_models)
    context = Context(protocol, "policy", stage, {"train": tmp_path / "training"}, "cpu", Budget())
    result = policy.run(context)
    assert loads.index("confirmation") > loads.index("calibration")
    assert result["status"] == "COMPLETED"
    assert result["conformal_status"] == "INSUFFICIENT_INDEPENDENT_FUNCTIONS"
    assert not result["conformal_guarantee_available"]
    assert result["endpoints"] > 20
    assert all(validate_row(row) for row in context.rows)
    details = [json.loads(line) for line in (stage / "policy-details.jsonl").read_text().splitlines()]
    assert all(not row["truth_used_in_decision"] for row in details)
    formal = [row for row in details if row["calibration_mode"] == "conformal"]
    assert formal and all(row["fallback"] for row in formal)
    assert all(row["fallback_seconds"] > 0 for row in formal)
    assert any("c2_rank2" in row["variant_id"] for row in details)
    assert result["policy_variants"] == 21  # 4 fixed family slots + bank, four ablations + formal.
    missing_ids = {slot["model_id"] for slot in result["unavailable_model_slots"]}
    assert len(missing_ids) == {"all": 0, "none": 4, "missing_first_seed": 1}[availability]
    missing_details = [row for row in details if row["variant"]["group"] in missing_ids]
    assert all(row["fallback"] and not row["accepted"] and not row["requested_candidate_available"] for row in missing_details)
    assert all(row["fallback_seconds"] > 0 for row in missing_details)
    if availability == "none":
        assert result["models"] == ["df_base"]
        assert result["policy_bank_status"] == "CLASSICAL_ONLY_NO_ELIGIBLE_NEURAL_MODEL"
    elif availability == "missing_first_seed":
        assert f"source-seed-{frozen_seed + 10}" not in result["models"]
        assert result["policy_bank_status"] == "PARTIAL_NEURAL_BANK"
    assert policy.validate_policy_artifacts(stage)["status"] == "VERIFIED"
    # Re-scoring must detect an edited decision label even if a user retains
    # superficially plausible aggregate counts in the separate summary.
    detail_path = stage / "policy-details.jsonl"
    original = detail_path.read_text()
    changed = [json.loads(line) for line in original.splitlines()]
    changed[0]["false_accept"] = not changed[0]["false_accept"]
    detail_path.write_text("".join(json.dumps(row) + "\n" for row in changed))
    with pytest.raises(ValueError, match="false-accept"):
        policy.validate_policy_artifacts(stage)
    detail_path.write_text(original)
    changed = [json.loads(line) for line in original.splitlines()]
    changed[0]["rejected_work_seconds"] += 1.
    detail_path.write_text("".join(json.dumps(row) + "\n" for row in changed))
    with pytest.raises(ValueError, match="rejected work"):
        policy.validate_policy_artifacts(stage)
    detail_path.write_text(original)
    summary_rows = [row for row in context.rows if row["experiment_id"].startswith("policy-summary")]
    for row in summary_rows:
        risk = next(c for c in row["checks"] if c["check_id"] == "independent_cluster_risk_bound")
        assert risk["verdict"] == "NA"


@pytest.mark.parametrize("family", ["source", "rank2", "cheap_fno", "c2_rank2"])
def test_residual_estimator_supports_every_deployed_learned_family(family):
    from tdn.analysis.roadmap.models import build_model
    model = build_model(family, {"width": 4, "modes": 1}).double().eval().requires_grad_(False)
    value, indicators, costs = estimate_attempt(model, _field((8, 8)), [.02], Equation(.002, 2.), Geometry((8, 8), (1., 1.)),
        estimator="interior_residual", spatial_guard=True, sampler=_field)
    assert torch.isfinite(value).all()
    assert all(math.isfinite(v) and v >= 0 for v in indicators["total"].values())
    assert indicators["temporal_diagnostics"]["jvp_calls"] == 6
    assert costs["proposal_seconds"] > 0
    assert costs["estimator_seconds"] > 0
    assert costs["spatial_seconds"] > 0


def test_amortization_requires_recorded_costs_matched_accuracy_and_positive_savings():
    from tdn.analysis.roadmap.statistics import amortization_summary
    result = amortization_summary({"teachers": 20., "train": 70., "calibration": 10.}, .5, 1., matched_accuracy=True)
    assert result["break_even_queries"] == 200
    assert result["offline_total_seconds"] == 100.
    for result in (amortization_summary({"train": None}, .5, 1., matched_accuracy=True),
                   amortization_summary({"train": 10.}, 1.1, 1., matched_accuracy=True),
                   amortization_summary({"train": 10.}, .5, 1., matched_accuracy=False)):
        assert result["break_even_queries"] is None
        json.dumps(result, allow_nan=False)


def test_reference_error_preserves_sub_float32_difference():
    from tdn.analysis.roadmap.policy import _upper_error
    prediction = torch.full((1, 1, 2, 2), .4, dtype=torch.float32)
    reference = {"accepted": True, "state": prediction.double() + 1e-8,
                 "uncertainty_rms": 1e-12, "uncertainty_max_bound": 1e-12}
    result = _upper_error(prediction, reference)
    assert result["error"]["rms"] == pytest.approx(1e-8, abs=1e-15)
    assert result["error"]["max"] == pytest.approx(1e-8, abs=1e-15)
    assert result["upper"]["max"] > 1e-8
    assert result["lower"]["max"] > 9.99e-9


@pytest.mark.parametrize("change", [
    {"accepted": None}, {"accepted": False}, {"accepted": 1},
    {"uncertainty_rms": None}, {"uncertainty_rms": float('nan')},
    {"uncertainty_max_bound": -1.}, {"uncertainty_max_bound": float('inf')},
    {"uncertainty_max_bound": True},
])
def test_reference_metadata_cannot_default_to_perfect_teacher(change):
    from tdn.analysis.roadmap.policy import _upper_error
    state = torch.full((1, 1, 2, 2), .4, dtype=torch.float64)
    reference = {"accepted": True, "state": state, "uncertainty_rms": 1e-12, "uncertainty_max_bound": 1e-12}
    reference.update(change)
    with pytest.raises(ValueError):
        _upper_error(state.float(), reference)


@pytest.mark.parametrize("key", ["accepted", "uncertainty_rms", "uncertainty_max_bound"])
def test_reference_missing_metadata_is_rejected(key):
    from tdn.analysis.roadmap.policy import _upper_error
    state = torch.full((1, 1, 2, 2), .4, dtype=torch.float64)
    reference = {"accepted": True, "state": state, "uncertainty_rms": 0., "uncertainty_max_bound": 0.}
    del reference[key]
    with pytest.raises(ValueError):
        _upper_error(state.float(), reference)


@pytest.mark.parametrize("state", [torch.ones((1, 1, 2, 2), dtype=torch.float32),
                                   torch.ones((1, 1, 1, 2), dtype=torch.float64),
                                   torch.full((1, 1, 2, 2), float('nan'), dtype=torch.float64)])
def test_reference_state_dtype_shape_and_finiteness_are_required(state):
    from tdn.analysis.roadmap.policy import _upper_error
    reference = {"accepted": True, "state": state, "uncertainty_rms": 0., "uncertainty_max_bound": 0.}
    with pytest.raises(ValueError):
        _upper_error(torch.ones((1, 1, 2, 2)), reference)
