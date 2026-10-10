"""Real bounded CPU learning, frozen selection and recovery regressions."""
import copy
import json
from pathlib import Path
import pytest
import torch
from tdn.analysis.advance.protocol import build_protocol
from tdn.analysis.advance.learning import (model_specs, shared_schedules, validation_selection,
    evaluation_plan, locked_rows, train, freeze, verify_frozen, confirm, aggregate,
    _trial, _validate_completed_group, scaling_groups)
from tdn.analysis.advance.data import prepare, load_bank, declarations, teacher_protocol
from tdn.analysis.frontier.core import Context
from tdn.analysis.frontier.neural import eq_geom
from tdn.research.experiment import Budget
from tdn.research.protocol import digest


def tiny_protocol():
    p = build_protocol("smoke")
    p.update(models=["df", "quad2_full", "two_basis", "fno_zero"], tracks=["discrete"],
        grids=[4], train_grid=4, train_horizons=[.01], validation_horizons=[.01],
        evaluation_horizons=[.01], primary_schedules=[[.004, .006]],
        confirm_schedules=[[.004, .006], [.01], [.005, .005]], endpoint_steps=[1, 2])
    p["training"].update(updates=2, tuning_updates=1, learning_rates=[.001], validation_every=1,
        batch_size=1, optimizer_controls=[dict(id="default", clip_grad_norm=1., loss_scale=1.)],
        overfit_updates=2, overfit_seconds=5, equal_time_seconds=0)
    p["model_configs"] = {}
    p["confirmation"]["primary_controls"] = ["quad2_full", "df", "fno_zero"]
    p["parents"] = [r for r in p["parents"] if r["split"] in ("train", "validation", "confirmation")]
    for row in p["parents"]:
        row.update(regime="smooth", field_terms=[dict(mode=[1, 0], weight=1., phase=(row["field_seed"] % 997)/100)],
                   kappa=.01, reaction_rate=1., mean=.3, variance=.0025)
    ids = [r["parent_id"] for r in p["parents"] if r["split"] == "confirmation"]
    p["units"] = {"prepare": dict(kind="prepare"), "train": dict(kind="train", families=p["models"], tracks=p["tracks"], seeds=p["seeds"]),
        "freeze": dict(kind="freeze"), "confirm-prepare": dict(kind="confirm_prepare", parent_ids=ids),
        "confirm": dict(kind="confirm", parent_ids=ids, prepared_unit="confirm-prepare"), "aggregate": dict(kind="aggregate")}
    p["bootstrap_replicates"] = 10
    return p


def context(tmp_path, p, stage, prior=None, budget=None):
    return Context(p, stage, tmp_path / stage, prior or {}, "cpu", budget or Budget(120))


@pytest.fixture
def stable_source(monkeypatch):
    from tdn.analysis.advance import data
    snapshot = data.software_metadata()
    monkeypatch.setattr(data, "software_metadata", lambda: snapshot)
    torch.set_num_threads(1)


@pytest.fixture
def campaign(tmp_path, stable_source):
    p = tiny_protocol()
    prep = context(tmp_path, p, "prepare"); prepare(prep)
    fitting = context(tmp_path, p, "train", {"prepare": prep.path}); train(fitting)
    frozen = context(tmp_path, p, "freeze", {"train": fitting.path}); freeze(frozen)
    refs = context(tmp_path, p, "confirm-prepare", {"freeze": frozen.path}); prepare(refs)
    testing = context(tmp_path, p, "confirm", {"freeze": frozen.path, "confirm-prepare": refs.path}); confirm(testing)
    return p, dict(prepare=prep, train=fitting, freeze=frozen, refs=refs, confirm=testing)


def test_roster_units_cover_every_model_once():
    p = build_protocol("full")
    planned = model_specs(p)
    actual = [r for scope in p["units"].values() if scope["kind"] == "train" for r in model_specs(p, scope)]
    assert len(actual) == len({r["model_id"] for r in actual})
    assert set(r["model_id"] for r in planned) == set(r["model_id"] for r in actual)


def test_every_family_has_cheap_endpoint_opportunities():
    p = build_protocol("full"); schedules = shared_schedules(p)
    for final in p["evaluation_horizons"]:
        for count in (1, 2, 4, 8):
            assert any(r["schedule"] == [final/count]*count for r in schedules)


def test_schedule_selection_keeps_worst_validation_field():
    rows = [dict(model_id="m", track="discrete", final_time=.1, field_cluster=f, schedule_id=s,
        upper_rms=1e-2 if f == "b" and s == "cheap" else 1e-7, upper_max=1e-7,
        cost_seconds=1. if s == "cheap" else 2., reference_accepted=True) for f in ("a", "b") for s in ("cheap", "safe")]
    assert validation_selection(rows, [(1e-4, 1e-4)])[0]["schedule_id"] == "safe"
    with pytest.raises(ValueError, match="Duplicate"):
        validation_selection(rows + [rows[0]], [(1e-4, 1e-4)])


def test_real_training_freeze_fresh_confirm_and_aggregate(campaign, tmp_path):
    p, contexts = campaign
    catalog, _ = verify_frozen(contexts["confirm"])
    assert len(catalog["records"]) == 4
    trained = next(r for r in catalog["records"] if r["family"] == "fno_zero")
    assert trained["initial_df_max_difference"] == 0.
    assert trained["independent_training_fields"] == 2
    audits = json.loads((contexts["train"].path / "fairness_diagnostics.json").read_text())["rows"]
    assert audits and audits[0]["phase"] == "tiny_overfit"
    assert all(r["phase"] != "tiny_overfit" for r in catalog["records"])
    ctx = context(tmp_path, p, "aggregate", {"freeze": contexts["freeze"].path, "confirm": contexts["confirm"].path})
    result = aggregate(ctx)
    assert result["parents"] == 2 and result["endpoint_rows"] > 0
    locked = json.loads((ctx.path / "locked_frontiers.json").read_text())["rows"]
    assert all(r["seed"] == p["seeds"][0] for r in locked if r["family"] == "fno_zero")
    assert all(isinstance(r["grid"], int) for r in locked)


def test_completed_trials_are_reused_without_retraining(campaign):
    _, contexts = campaign
    before = (contexts["train"].path / "catalog.json").read_bytes()
    train(contexts["train"])
    assert (contexts["train"].path / "catalog.json").read_bytes() == before


def test_frozen_checkpoint_corruption_rejected(campaign):
    _, contexts = campaign
    catalog, _ = verify_frozen(contexts["confirm"])
    path = contexts["freeze"].path / catalog["records"][0]["checkpoint"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checkpoint"):
        verify_frozen(contexts["confirm"])


def test_fresh_data_is_not_available_to_training(campaign):
    _, contexts = campaign
    contexts["train"].prerequisites.update({"confirm-prepare": contexts["refs"].path, "freeze": contexts["freeze"].path})
    with pytest.raises(ValueError, match="forbidden"):
        load_bank(contexts["train"], "confirm-prepare")


def test_missing_frozen_schedule_preserves_fitted_seed():
    p = tiny_protocol(); spec = next(r for r in model_specs(p) if r["family"] == "two_basis")
    front = dict(model_id=spec["model_id"], track=spec["track"], final_time=.01,
                 rms_target=2e-5, max_target=2e-5, schedule_id=None, status="NO_FEASIBLE_VALIDATION_SCHEDULE")
    ids = [r["parent_id"] for r in p["parents"] if r["split"] == "confirmation"]
    rows = locked_rows([], [front], p, ids)
    assert all(r["seed"] == spec["seed"] and r["train_count"] == 2 for r in rows)
    assert all(r["cost_seconds"] is None and r["status"] == "NO_FEASIBLE_VALIDATION_SCHEDULE" for r in rows)


def test_confirmation_omits_unselected_cartesian_schedules():
    p = tiny_protocol(); catalog = model_specs(p); parents = [r for r in p["parents"] if r["split"] == "confirmation"]
    plan = evaluation_plan(p, catalog, [], parents)
    assert len(plan) == len(parents)
    assert all(scope["schedule"]["primary"] for scope in plan.values())


def test_confirmation_payload_tampering_rejected(campaign):
    _, contexts = campaign
    path = next((contexts["confirm"].path / "completed-groups").glob("*.json"))
    saved = json.loads(path.read_text()); saved["rows"][0]["cost_seconds"] *= 2
    with pytest.raises(ValueError, match="payload"):
        _validate_completed_group(saved, saved["identity"], saved["expected"])


def test_duplicate_parent_declaration_rejected(tmp_path):
    p = tiny_protocol(); p["units"]["confirm"]["parent_ids"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        declarations(context(tmp_path, p, "confirm"))


def test_teacher_projection_omits_unused_schedule_intermediates():
    from tdn.analysis.frontier.data import reference_horizons
    p = build_protocol("full"); original = copy.deepcopy(p)
    assert reference_horizons(teacher_protocol(p), "confirmation") == p["evaluation_horizons"]
    assert p == original


def test_family_specific_data_sizes_have_unique_specs():
    p = tiny_protocol(); p["training"]["subset_sizes_by_family"] = {"two_basis": [1, 2]}
    ids = [r["model_id"] for r in model_specs(p)]
    assert len(ids) == len(set(ids))
    assert sum("/two_basis/" in x for x in ids) == 2
    assert sum("/fno_zero/" in x for x in ids) == 1


def test_scaling_batch_one_covers_all_fields_without_padding_batch_four():
    p = build_protocol("full"); bank = [r for r in p["parents"] if r["split"] == "scaling"]
    one = list(scaling_groups(bank, 1)); four = list(scaling_groups(bank, 4))
    assert len(one) == 8
    assert {g[0]["parent_id"] for _, _, g in one} == {r["parent_id"] for r in bank}
    assert len(four) == 3 and sum(bool(g) for _, _, g in four) == 1
    assert all(len({p["parent_id"] for p in group}) == 4 for _, _, group in four if group)
    assert (len(one)+len(four))*len(p["scaling"]["grids"])*len(p["tracks"]) == 44


def test_interrupted_final_validation_is_replayed_exactly(tmp_path, stable_source, monkeypatch):
    from tdn.analysis.advance import learning
    from tdn.analysis.advance.models import make_model
    from tdn.analysis.frontier.neural import _samples
    p = tiny_protocol(); prep = context(tmp_path, p, "prepare"); prepare(prep)
    ctx = context(tmp_path, p, "train", {"prepare": prep.path})
    bank = load_bank(ctx, "prepare")
    base = make_model("df", "discrete", p["model_config"]).float()
    samples = _samples([r for r in bank if r["split"] == "train"], [.01], 4, "discrete", "cpu", base, 1e-6)
    spec = next(r for r in model_specs(p) if r["family"] == "two_basis")
    control = dict(id="default", clip_grad_norm=1., loss_scale=1.)
    original = learning.validation_metrics; calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("interrupt final validation")
        return original(*args)
    monkeypatch.setattr(learning, "validation_metrics", interrupted)
    with pytest.raises(TimeoutError, match="final validation"):
        _trial(ctx, spec, p["model_config"], samples, samples, .001, control, "final", updates=2, seconds=120)
    saved = torch.load(next((ctx.path / "progress").glob("*.pt")), weights_only=True)
    assert saved["completed"] == 2 and saved["pending_validation"] is True
    assert saved["curves"][-1]["update"] == 2 and saved["curves"][-1]["validation_loss"] is None
    monkeypatch.setattr(learning, "validation_metrics", original)
    resumed = _trial(ctx, spec, p["model_config"], samples, samples, .001, control, "final", updates=2, seconds=120)
    fresh = context(tmp_path / "uninterrupted", p, "train", {"prepare": prep.path})
    direct = _trial(fresh, spec, p["model_config"], samples, samples, .001, control, "final", updates=2, seconds=120)
    assert resumed["updates_selected"] == direct["updates_selected"]
    assert resumed["validation_objective"] == direct["validation_objective"]
    assert resumed["curves"][-1]["validation_loss"] is not None
    left = torch.load(ctx.path / resumed["checkpoint"], weights_only=True)["state_dict"]
    right = torch.load(fresh.path / direct["checkpoint"], weights_only=True)["state_dict"]
    assert all(torch.equal(left[k], right[k]) for k in left)
