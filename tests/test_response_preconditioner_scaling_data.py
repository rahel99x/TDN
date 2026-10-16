"""P1 label-free cohort isolation, source nesting and frozen work contracts."""
import copy
from collections import Counter

import numpy as np
import pytest
import torch

from tdn.analysis.response.contracts import Physics
from tdn.analysis.response_preconditioner.scaling_data import parent_identity, parents, source_bank
from tdn.analysis.response_preconditioner.scaling_protocol import (
    FAMILIES, _fixture_config, planned_counts, protocol, validate_config,
)


def test_production_counts_and_balanced_nested_cohorts_are_fixed_before_labels():
    config = protocol()
    assert validate_config(config) == dict(reference_count=2688, expected_fit_cells=9,
        expected_withheld_queries=1152, expected_metric_rows=16128,
        expected_timing_bundles=1008, max_complete_source_calls=4032)
    assert config["ridge_menu"] == [1e-10, 1e-6]
    assert config["zero_candidate"] and not config["scientific_promotion"]
    bank = {split: list(parents(config, split)) for split in config["parent_seeds"]}
    assert [len(bank[split]) for split in ("train", "validation", "diagnostic")] == [32, 8, 8]
    for count in config["parent_counts"]:
        assert Counter(p["parent_family"] for p in bank["train"][:count]) == {
            family: count//4 for family in FAMILIES}
    for split in ("validation", "diagnostic"):
        assert Counter(p["parent_family"] for p in bank[split]) == {family: 2 for family in FAMILIES}
    timed = [bank["diagnostic"][i]["parent_family"] for i in config["timing"]["parent_indices"]]
    assert timed == list(FAMILIES)


def test_parent_inputs_are_disjoint_reproducible_and_in_declared_physical_ranges():
    config = protocol()
    originals = [p for split in config["parent_seeds"] for p in parents(config, split)]
    repeats = [p for split in config["parent_seeds"] for p in parents(config, split)]
    assert len({p["seed"] for p in originals}) == 48
    assert len({p["parent_id"] for p in originals}) == 48
    assert len({p["identity_sha256"] for p in originals}) == 48
    assert len({p["field_sha256"] for p in originals}) == 48
    assert not set(p["seed"] for p in originals) & set(range(2026101501, 2026101505))
    for left, right in zip(originals, repeats):
        assert left["u"].dtype == torch.float32 and left["u"].shape == (1, 1, 5, 5)
        assert bool(((left["u"] > 0) & (left["u"] < 1)).all())
        assert .009 <= left["eq"].kappa < .017 and 1.7 <= left["eq"].reaction_rate < 2.9
        assert .25 <= left["actual_mean"] <= .65
        assert left["actual_variance"] > 0
        assert left["parent_index"] == left["cohort_order"] == left["index"]
        assert left["parent_family"] == FAMILIES[left["parent_index"] % 4]
        assert left["identity_sha256"] == right["identity_sha256"]
        assert left["eq"] == right["eq"]
        torch.testing.assert_close(left["u"], right["u"], rtol=0, atol=0)


def test_parent_field_and_physics_do_not_depend_on_source_counts_or_cohort_length():
    config = protocol()
    modified = copy.deepcopy(config)
    modified["calibration_sources"] = 2
    modified["calibration_families"] = ["dense_random"]*2
    modified["withheld_sources"] = 1
    modified["withheld_families"] = ["dense_random"]
    modified["parent_seeds"]["train"] = modified["parent_seeds"]["train"][:8]
    left, right = list(parents(config, "train")), list(parents(modified, "train"))
    for a, b in zip(left, right):
        assert a["identity_sha256"] == b["identity_sha256"]
        assert a["eq"] == b["eq"]
        torch.testing.assert_close(a["u"], b["u"], rtol=0, atol=0)


def test_identity_detects_field_or_physics_change_without_using_parent_names():
    parent = next(parents(protocol(), "train"))
    original = parent_identity(parent["u"], parent["eq"], parent["geom"])
    modified = parent["u"].clone()
    modified[..., 0, 0] = torch.nextafter(modified[..., 0, 0], torch.ones_like(modified[..., 0, 0]))
    changed_field = parent_identity(modified, parent["eq"], parent["geom"])
    changed_eq = parent_identity(parent["u"], Physics(parent["eq"].kappa,
        parent["eq"].reaction_rate+.01), parent["geom"])
    assert original["field_sha256"] != changed_field["field_sha256"]
    assert original["identity_sha256"] != changed_field["identity_sha256"]
    assert original["identity_sha256"] != changed_eq["identity_sha256"]
    assert original["field_sha256"] == changed_eq["field_sha256"]
    with pytest.raises(ValueError, match="actual finite FP32"):
        parent_identity(parent["u"].double(), parent["eq"], parent["geom"])


def test_calibration_prefixes_are_exactly_nested_and_withheld_stream_is_separate():
    config = protocol()
    seed = config["parent_seeds"]["train"][0]
    largest = source_bank(seed, 2, "calibration", config)
    for count in config["calibration_counts"]:
        smaller = copy.deepcopy(config)
        smaller["calibration_sources"] = count
        smaller["calibration_families"] = ["dense_random"]*count
        for left, right in zip(largest, source_bank(seed, 2, "calibration", smaller)):
            assert left["source_index"] == right["source_index"]
            torch.testing.assert_close(left["z"], right["z"], rtol=0, atol=0)
    withheld = source_bank(seed, 2, "withheld", config)
    assert [s["family"] for s in withheld] == config["withheld_families"]
    assert not torch.equal(largest[0]["z"], withheld[0]["z"])
    assert not torch.equal(largest[0]["z"], source_bank(seed, 1, "calibration", config)[0]["z"])
    assert not torch.equal(largest[0]["z"], source_bank(seed+1, 2, "calibration", config)[0]["z"])
    for source in largest+withheld:
        assert source["z"].dtype == torch.float64
        assert source["z"].shape == (1, 1, 5, 5)
        assert float(source["z"].square().mean()) == pytest.approx(1., abs=8e-16)
    assert torch.equal(withheld[-1]["z"], torch.ones_like(withheld[-1]["z"]))
    assert abs(float(withheld[1]["z"].mean())) < 5e-16
    assert abs(float(withheld[2]["z"].mean())) < 5e-16


@pytest.mark.parametrize("mutation", ("budget", "ridge", "cap", "seed", "family", "grid", "promotion"))
def test_public_protocol_rejects_changed_science_or_resources(mutation):
    config = protocol()
    if mutation == "budget":
        config["budget"]["study_seconds"] = 901.
    elif mutation == "ridge":
        config["ridge_menu"] = [1e-12, 1e-6]
    elif mutation == "cap":
        config["max_reference_queries"] -= 1
    elif mutation == "seed":
        config["parent_seeds"]["diagnostic"][0] = config["parent_seeds"]["train"][0]
    elif mutation == "family":
        config["parent_families"].reverse()
    elif mutation == "grid":
        config["grid"] = [7, 7]
    else:
        config["scientific_promotion"] = True
    with pytest.raises(ValueError, match="frozen P1"):
        validate_config(config)


def test_private_reduced_fixture_needs_explicit_opt_in_and_exact_denominators():
    config = _fixture_config()
    assert planned_counts(config)["reference_count"] == 48
    assert validate_config(config, allow_fixture=True)["expected_fit_cells"] == 4
    with pytest.raises(ValueError, match="forbidden in production"):
        validate_config(config)
    config["reference_count"] -= 1
    with pytest.raises(ValueError, match="denominators"):
        validate_config(config, allow_fixture=True)


@pytest.mark.parametrize("mutation", ("budget", "physics", "seed", "timing", "cap", "source_family"))
def test_fixture_cannot_expand_science_or_hide_missing_work(mutation):
    config = _fixture_config()
    if mutation == "budget":
        config["budget"]["study_seconds"] = float("nan")
    elif mutation == "physics":
        config["parent_ranges"]["kappa"] = [.009, .17]
    elif mutation == "seed":
        config["parent_seeds"]["diagnostic"][0] = config["parent_seeds"]["train"][0]
    elif mutation == "timing":
        config["timing"]["parent_indices"] = [0, 0]
    elif mutation == "cap":
        config["max_reference_queries"] += 1
    else:
        config["withheld_families"] = ["dense_random"]*4
    with pytest.raises(ValueError):
        validate_config(config, allow_fixture=True)


@pytest.mark.parametrize("role,context", (("confirmation", 0), ("withheld", -1), ("calibration", 6)))
def test_sources_reject_unsupported_access_roles_and_contexts(role, context):
    with pytest.raises(ValueError):
        source_bank(2026101601, context, role, protocol())


def test_preparing_protocol_and_inputs_acquires_no_scientific_labels(monkeypatch):
    from tdn.analysis.response_preconditioner import r7_references

    def forbidden(*args, **kwargs):
        raise AssertionError("Label acquisition must not occur in protocol/parent preparation")

    monkeypatch.setattr(r7_references, "acquire_reference", forbidden)
    config = protocol("verify")
    validate_config(config)
    parent = next(parents(config, "train"))
    assert len(source_bank(parent["seed"], 0, "calibration", config)) == 8
