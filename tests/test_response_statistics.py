import copy

import pytest

from tdn.analysis.response.statistics import compare_endpoints, claims_payload, holm_adjust, reference_interval


def fixture(n=20):
    registry = dict(parents=[f"p{k}" for k in range(n)], query_ids=["short", "long"], seeds=[1, 2],
        target_kind="galerkin_same_grid", primary_tolerance=1., maximum_tolerance=2.,
        analysis_kind="development", draws=200, minimum_parents=16)
    rows = [dict(parent_id=p, query_id=q, seed=s, method=m, target_kind=registry["target_kind"],
                 rms=e, maximum=2*e, uncertainty_rms=1e-10, uncertainty_maximum=1e-10,
                 reference_accepted=True, reference_bounded=True, bound_violation=False)
            for p in registry["parents"] for q in registry["query_ids"] for s in registry["seeds"]
            for m, e in [("ours", .5), ("control", .8)]]
    return rows, registry


def test_reference_interval_is_not_automatic_pass_at_measured_threshold():
    assert reference_interval(.9, .2, 1.)["qualification"] == "UNRESOLVED"
    assert reference_interval(.7, .2, 1.)["qualification"] == "QUALIFIED"
    assert reference_interval(1.3, .2, 1.)["qualification"] == "FAILED"
    with pytest.raises(ValueError):
        reference_interval(float("nan"), 0., 1.)


def test_parent_clustering_and_frozen_descendant_weights():
    rows, plan = fixture()
    result = compare_endpoints(rows, "ours", "control", plan)
    assert result["attempted_parents"] == 20
    assert result["complete_paired_parents"] == 20
    assert result["denominators"]["ours"]["attempted"] == 80
    assert result["rms_geometric_ratio"] == pytest.approx(.625)
    # Duplicating a horizon only preserves the estimand with its split weight.
    plan2 = copy.deepcopy(plan)
    plan2["query_ids"].append("long_copy")
    plan2["query_weights"] = {"short": .5, "long": .25, "long_copy": .25}
    duplicated = rows+[dict(r, query_id="long_copy") for r in rows if r["query_id"] == "long"]
    same = compare_endpoints(duplicated, "ours", "control", plan2)
    assert same["rms_geometric_ratio"] == result["rms_geometric_ratio"]
    assert same["rms_ratio_interval"] == result["rms_ratio_interval"]
    assert same["attempted_parents"] == result["attempted_parents"]
    with pytest.raises(ValueError, match="Duplicate"):
        compare_endpoints(rows+[rows[0]], "ours", "control", plan)


def test_failed_predictions_and_missing_outputs_keep_denominator():
    rows, plan = fixture()
    rows = [r for r in rows if not (r["parent_id"] == "p0" and r["method"] == "ours")]
    result = compare_endpoints(rows, "ours", "control", plan)
    assert result["denominators"]["ours"]["failed_prediction_on_valid_reference"] == 4
    assert result["denominators"]["ours"]["attempted"] == 80
    assert result["complete_paired_parents"] == 19
    assert not result["sufficient_complete_parents"]
    assert not claims_payload([result])["comparisons"][0]["primary_rms_supported"]
    assert result["joint_coverage_difference"] == pytest.approx(-.05)


def test_unresolved_teachers_are_not_model_failures():
    rows, plan = fixture()
    for r in rows:
        if r["parent_id"] == "p0":
            r["reference_accepted"] = False
    result = compare_endpoints(rows, "ours", "control", plan)
    assert result["denominators"]["ours"]["unresolved_reference"] == 4
    assert result["denominators"]["ours"].get("failed_prediction_on_valid_reference", 0) == 0
    assert not result["sufficient_complete_parents"]


def test_small_samples_and_confirmation_never_relabel_development():
    rows, plan = fixture(2)
    result = compare_endpoints(rows, "ours", "control", plan)
    assert result["primary_one_sided_pvalue"] is None
    assert claims_payload([result])["comparisons"][0]["verdict"] == "UNRESOLVED_OR_NO_ADVANTAGE"
    plan["analysis_kind"] = "confirmation"
    with pytest.raises(ValueError, match="frozen"):
        compare_endpoints(rows, "ours", "control", plan)


def test_primary_rms_gain_does_not_hide_peak_harm():
    rows, plan = fixture()
    for r in rows:
        if r["method"] == "ours":
            r["maximum"] = 1.95
    result = claims_payload([compare_endpoints(rows, "ours", "control", plan)])["comparisons"][0]
    assert result["primary_rms_supported"]
    assert not result["tail_noninferiority_supported"]
    assert result["verdict"] == "RMS_ONLY_UNRESOLVED_TRADEOFF"


def test_holm_retains_missing_tests_and_monotonic_adjustment():
    assert holm_adjust([.01, .04, None]) == pytest.approx([.03, .08, 1.])
    assert holm_adjust([.04, .01]) == pytest.approx([.04, .02])


def test_reference_sensitivity_blocks_numerically_unresolved_gain():
    rows, plan = fixture()
    for r in rows:
        r["uncertainty_rms"] = .4
    result = claims_payload([compare_endpoints(rows, "ours", "control", plan)])["comparisons"][0]
    assert result["rms_geometric_ratio"] < .9
    assert result["reference_worst_case_rms_ratio"] > 1
    assert not result["primary_rms_supported"]


def test_mixed_targets_rejected():
    rows, plan = fixture()
    rows[0]["target_kind"] = "fd_same_grid"
    with pytest.raises(ValueError, match="target equations"):
        compare_endpoints(rows, "ours", "control", plan)


def test_missing_bounds_cannot_establish_composite_safety():
    rows, plan = fixture()
    rows[0]["reference_bounded"] = None
    result=claims_payload([compare_endpoints(rows,"ours","control",plan)])["comparisons"][0]
    assert result["primary_rms_supported"]
    assert not result["bound_measurement_complete"]
    assert not result["composite_advantage_supported"]


def test_paired_methods_cannot_use_different_reference_uncertainty():
    rows, plan = fixture()
    rows[0]["uncertainty_rms"] = .5
    with pytest.raises(ValueError,match="same deterministic"):
        compare_endpoints(rows,"ours","control",plan)
