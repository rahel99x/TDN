import copy

import pytest
import torch

from tdn.analysis.response.fields import make_parent, sample_parent
from tdn.analysis.response.interventions import (
    GROUPS, PARAMETER_VALUES, VARIANTS, transform_parent, validate_held_out_values,
)


def parent():
    return make_parent(111010001, "development_test", 2, regime="triads")


def test_physical_interventions_change_one_coefficient_and_preserve_input():
    original = parent()
    before = copy.deepcopy(original)
    field = sample_parent(original, [16, 16])
    for name, (parameter, value, _) in PARAMETER_VALUES.items():
        p = transform_parent(original, name)
        assert p[parameter] == value
        other = "kappa" if parameter == "reaction_rate" else "reaction_rate"
        assert p[other] == original[other]
        torch.testing.assert_close(sample_parent(p, [16, 16]), field, rtol=0, atol=0)
        assert p["parent_id"] == original["parent_id"]
        assert p["parent_sha256"] == original["parent_sha256"]
        assert p["descendant_sha256"] != original["parent_sha256"]
    assert original == before


def test_matched_mean_pair_preserves_identical_centered_field_even_after_cap():
    original = parent()
    assert original["amplitude_bound"] > .12
    low = transform_parent(original, "mean_pair_low")
    high = transform_parent(original, "mean_pair_high")
    assert low["terms"] == high["terms"]
    assert low["amplitude_bound"] == pytest.approx(.12)
    a, b = sample_parent(low, [32, 32]), sample_parent(high, [32, 32])
    torch.testing.assert_close(b-a, torch.full_like(a, .7), rtol=0, atol=3e-16)
    assert a.min() >= 0 and b.max() <= 1
    assert low["intervention"]["comparison_partner"] == "mean_pair_high"


def test_amplitude_intervention_is_not_clipping_and_keeps_modes_and_phases():
    original = parent()
    half = transform_parent(original, "amplitude_half")
    near = transform_parent(original, "amplitude_near_boundary")
    a, b = sample_parent(original, [32, 32]), sample_parent(half, [32, 32])
    torch.testing.assert_close(b-original["mean"], .5*(a-original["mean"]), rtol=0, atol=2e-16)
    assert near["amplitude_bound"] == pytest.approx(.94*min(original["mean"], 1-original["mean"]))
    for q, p in zip(near["terms"], original["terms"]):
        assert q["mode"] == p["mode"] and q["phase"] == p["phase"]
    assert near["intervention"]["guaranteed_lower"] >= 0
    assert near["intervention"]["guaranteed_upper"] <= 1


def test_constant_amplitude_is_retained_without_inventing_shape():
    p = make_parent(111010001, "development_test", 7)
    assert p["amplitude_bound"] == 0
    q = transform_parent(p, "amplitude_near_boundary")
    assert q["terms"] == p["terms"] and q["amplitude_bound"] == 0
    assert "Constant input" in q["intervention"]["note"]


def test_held_out_parameters_reject_training_and_selection_leakage():
    parents = [make_parent(111010001, split, i) for split in ("train", "validation", "development_test") for i in range(3)]
    assert validate_held_out_values(parents)["status"] == "VERIFIED"
    for split in ("train", "validation"):
        leaked = copy.deepcopy(parents)
        next(p for p in leaked if p["split"] == split)["reaction_rate"] = 1.5
        with pytest.raises(ValueError, match="entered training/validation"):
            validate_held_out_values(leaked)
    parents[-1]["reaction_rate"] = 1.5  # evaluated descendants cannot affect training guard
    assert validate_held_out_values(parents)["status"] == "VERIFIED"


def test_frozen_menus_are_complete_and_interventions_cannot_be_composed():
    assert set(sum((list(v) for v in GROUPS.values()), [])) == set(VARIANTS)
    p = parent()
    for name in VARIANTS:
        assert transform_parent(p, name) == transform_parent(copy.deepcopy(p), name)
    with pytest.raises(ValueError, match="not be composed"):
        transform_parent(transform_parent(p, "amplitude_half"), "reaction_interpolation")
    with pytest.raises(ValueError, match="Unknown"):
        transform_parent(p, "chosen_after_evaluation")
    p["mean"] = .01
    with pytest.raises(ValueError, match="clipping is forbidden"):
        transform_parent(p, "original")
