"""Independent teacher, immutable field, split and seal regressions."""
import json

import pytest
import torch

from tdn.analysis.frontier.core import Context
from tdn.analysis.frontier.data import (field_state, generate_reference, lawson_reference,
    load_split, prepare, validate_cohorts, reference_horizons, verify_frozen_training)
from tdn.analysis.frontier.protocol import build_protocol
from tdn.analysis.agenda.physics import fourier_resample
from tdn.numerics import Equation, Geometry
from tdn.numerics.reference import reference_step
from tdn.research.experiment import Budget


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_protocol():
    p = build_protocol("smoke")
    p["parents"] = [next(v for v in p["parents"] if v["split"] == s)
                    for s in ("development", "train", "validation", "calibration", "confirmation")]
    p.update(train_grid=8, grids=[8], train_horizons=[.01], validation_horizons=[.015],
        confirm_schedules=[[.01, .01]], reference_substeps=2, reference_max_substeps=32,
        reference_tolerance=1e-6, reference_spatial_factors=[1, 2])
    return p


def context(p, stage, path, prerequisites=None):
    return Context(p, stage, path, prerequisites or {}, "cpu", Budget(60))


@pytest.mark.parametrize("track", ["discrete", "continuum"])
def test_real_time_refinement_and_exact_homogeneous_teacher(track):
    p = tiny_protocol(); parent = dict(p["parents"][0], variance=0., mean=.3)
    result = generate_reference(parent, 8, .03, track, p, Budget(20))
    truth = .3 / (.3 + .7 * torch.exp(torch.tensor(-parent["reaction_rate"] * .03, dtype=torch.float64)))
    assert result["accepted"]
    assert float((result["state"] - truth).abs().max()) < result["uncertainty_max_bound"] + 1e-12
    refinements = result["temporal_refinements"]
    first = refinements[0] if track == "discrete" else refinements[0]["temporal_refinements"][0]
    assert first["counts"] == [2, 4, 8]
    assert result["uncertainty_rms"] > 0
    if track == "continuum":
        assert result["spatial_reference_grids"] == [8, 16]
        assert result["uncertainty_is_certificate"] is False


def test_independent_lawson_matches_coupled_fd_rk4_on_rectangular_physical_lengths():
    parent = tiny_protocol()["parents"][0]
    u = field_state(parent, 8)
    eq, geom = Equation(.02, 3.), Geometry((8, 8), (1.2, .7))
    actual = lawson_reference(u, .02, eq, geom, 64, "discrete")
    expected = reference_step(u, .02, eq, geom, 256)
    assert float((actual - expected).abs().max()) < 2e-10


def test_same_continuous_field_projects_consistently_across_grids():
    parent = tiny_protocol()["parents"][0]
    assert torch.allclose(field_state(parent, 8), fourier_resample(field_state(parent, 32), (8, 8)), atol=2e-15, rtol=0)


@pytest.mark.parametrize("key", ["field_cluster", "continuous_field_id", "field_seed"])
def test_split_leakage_rejected(key):
    p = tiny_protocol(); p["parents"][1][key] = p["parents"][0][key]
    with pytest.raises(ValueError, match="split-disjoint"):
        validate_cohorts(p)


def test_renaming_parent_ids_does_not_hide_identical_field_leakage():
    p = tiny_protocol()
    for key in ("field_terms", "mean", "variance"):
        p["parents"][1][key] = p["parents"][0][key]
    with pytest.raises(ValueError, match="Identical continuous coefficients"):
        validate_cohorts(p)


def test_shift_labels_correspond_to_actually_unseen_physics():
    p = build_protocol("full")
    validate_cohorts(p)
    row = next(v for v in p["parents"] if v.get("distribution") == "coefficient_shift")
    train = next(v for v in p["parents"] if v["split"] == "train")
    row.update(kappa=train["kappa"], reaction_rate=train["reaction_rate"])
    with pytest.raises(ValueError, match="Coefficient-shift"):
        validate_cohorts(p)


def test_unresolved_teacher_is_retained_and_cannot_be_training_label(tmp_path):
    p = tiny_protocol(); p["reference_max_substeps"] = 4
    prep = context(p, "prepare", tmp_path / "prepare")
    summary = prepare(prep)
    assert summary["unresolved_references"] > 0
    assert not summary["training_labels_ready"]
    with pytest.raises(ValueError, match="Unresolved reference"):
        load_split(context(p, "train", tmp_path / "train", {"prepare": prep.path}), "train")


def test_bank_safe_roundtrip_tamper_and_no_confirmation(tmp_path):
    p = tiny_protocol(); prep = context(p, "prepare", tmp_path / "prepare")
    summary = prepare(prep)
    assert summary["unresolved_references"] == 0
    ctx = context(p, "train", tmp_path / "train", {"prepare": prep.path})
    bank = load_split(ctx, "train")
    assert len(bank) == 1 and bank[0]["states"]["8"].dtype == torch.float64
    manifest = json.loads((prep.path / "data_manifest.json").read_text())
    assert "confirmation" not in manifest["splits"]
    assert not any(r["split"] == "confirmation" for r in manifest["records"])
    with pytest.raises(ValueError, match="overwrite"):
        prepare(prep)
    record = next(r for r in manifest["records"] if r["split"] == "train")
    metadata = prep.path / record["metadata"]
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError, match="bytes changed"):
        load_split(ctx, "train")


def test_fresh_confirmation_access_rejected_during_training(tmp_path):
    p = tiny_protocol()
    with pytest.raises(ValueError, match="forbidden"):
        load_split(context(p, "train", tmp_path / "train"), "confirmation")


def test_teacher_spatial_levels_cannot_be_identical():
    p = tiny_protocol(); p["reference_spatial_factors"] = [2, 2]
    with pytest.raises(ValueError, match="doubled"):
        generate_reference(p["parents"][0], 8, .01, "continuum", p, Budget(10))


def test_horizons_preserve_final_times_and_distinct_policy_scaling():
    p = tiny_protocol()
    p["scaling"]["horizon"] = .07
    p["policies"]["horizon"] = .09
    assert reference_horizons(p, "confirmation") == [.01, .02]
    assert reference_horizons(p, "scaling") == [.07]
    assert reference_horizons(p, "policy") == [.09]
    assert reference_horizons(p, "calibration") == [.09]


def test_all_irregular_schedule_prefixes_receive_teachers():
    p = tiny_protocol(); p["confirm_schedules"] = [[.02, .04, .06], [.06, .04, .02], [.03] * 4]
    assert reference_horizons(p, "confirmation") == [.02, .03, .06, .09, .10, .12]


def test_frozen_training_verifies_every_model_and_checkpoint_bytes(tmp_path):
    from tdn.analysis.frontier.neural import expected_model_specs
    from tdn.research.protocol import digest, file_digest
    from tdn.runtime.metadata import write_json
    p = tiny_protocol(); base = tmp_path / "train"; base.mkdir()
    write_json(base / "training_plan.json", {"models": expected_model_specs(p)})
    records = []
    for index, spec in enumerate(expected_model_specs(p)):
        name = f"model-{index}.bin"
        (base / name).write_bytes(f"own test checkpoint {index}".encode())
        records.append({**spec, "checkpoint": name, "checkpoint_sha256": file_digest(base / name)})
    write_json(base / "catalog.json", {"protocol_sha256": digest(p), "records": records,
        "frozen_before_confirmation": True, "selection_split": "validation",
        "training_plan_sha256": file_digest(base / "training_plan.json")})
    write_json(base / "freeze.json", {"protocol_sha256": digest(p), "catalog_sha256": file_digest(base / "catalog.json"),
        "checkpoint_hashes": {r["model_id"]: r["checkpoint_sha256"] for r in records}})
    ctx = context(p, "confirm_prepare", tmp_path / "confirm_prepare", {"train": base})
    assert len(verify_frozen_training(ctx)["checkpoint_hashes"]) == len(records)
    summary = prepare(ctx, confirmation=True)
    assert summary["accepted_references"] > 0
    assert ctx.rows and all({"G2", "G3"} <= set(r["mechanism_ids"]) for r in ctx.rows)
    (base / records[0]["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        verify_frozen_training(ctx)
