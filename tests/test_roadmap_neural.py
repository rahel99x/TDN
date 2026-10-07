"""Checkpoint isolation, actual fitting and held-out reporting contracts."""
import copy
import json
from pathlib import Path

import pytest
import torch

from tdn.analysis.roadmap.core import Context, validate_row
from tdn.analysis.roadmap.protocol import build_protocol
from tdn.analysis.roadmap import neural
from tdn.numerics import Equation, Geometry
from tdn.numerics.reference import reference_step
from tdn.research.experiment import horizon_key


class Budget:
    def check(self):
        pass


def bank(split):
    n = 4
    x = torch.arange(n, dtype=torch.float64) / n
    u = (.45 + .07 * torch.cos(2 * torch.pi * x[:, None]) + .02 * torch.sin(2 * torch.pi * x[None, :]))[None, None]
    equation, geometry = Equation(.002, 2.), Geometry((n, n), (1., 1.))
    refs = {}
    for h in (.02, .03, .04, .06):
        refs[f"discrete:{n}:{horizon_key(h)}"] = dict(state=reference_step(u, h, equation, geometry, 24),
            uncertainty_rms=1.e-10, uncertainty_max_bound=1.e-10, accepted=True)
    return [dict(parent_id=f"{split}-1", field_cluster=f"{split}-cluster-1", split=split,
        kappa=.002, reaction_rate=2., regime="low", states={str(n): u}, references=refs)]


def context(tmp_path, stage, protocol, prerequisites=None):
    return Context(protocol, stage, tmp_path / stage, prerequisites or {}, "cpu", Budget())


def tiny_protocol():
    p = build_protocol("smoke")
    p.update(train_grid=4, grids=[4], width=2, modes=1,
             train_horizons=[.02, .04, .06], validation_horizons=[.03, .06],
             models=["source", "source_df_loss", "source_trust", "c2_rank2"],
             confirm_schedules=[[.02, .04], [.04, .02]])
    return p


def test_training_never_reads_confirmation_and_freezes_every_arm(tmp_path, monkeypatch):
    from tdn.analysis.roadmap import data
    accessed = []
    def load(ctx, split):
        accessed.append(split)
        assert split in ("train", "validation")
        return bank(split)
    monkeypatch.setattr(data, "load_bank", load)
    p = tiny_protocol()
    ctx = context(tmp_path, "train", p)
    result = neural.train(ctx)
    assert accessed == ["train", "validation"]
    assert result["selected_models"] == 5  # four requested plus shared-M11 comparator
    catalog = json.loads((ctx.path / "catalog.json").read_text())
    assert catalog["frozen_before_confirmation"]
    families = {r["family"] for r in catalog["records"]}
    assert families == {*p["models"], "source_df_loss_shared"}
    trust = next(r for r in catalog["records"] if r["family"] == "source_trust")
    assert len(trust["trust_validation_calibration"]) == 3
    source = next(r for r in catalog["records"] if r["family"] == "source")
    shared = next(r for r in catalog["records"] if r["family"] == "source_df_loss_shared")
    assert (shared["learning_rate"], shared["batch_size"]) == (source["learning_rate"], source["batch_size"])
    for row in ctx.rows:
        validate_row(row)
        assert row["metrics"]["gradient_norm_max"] > 0
        assert any(c["check_id"] == "generator_consistency" and c["verdict"] == "GOOD" for c in row["checks"])
        if row["metrics"]["selection"] == "SELECTED_INITIALIZATION":
            assert next(c for c in row["checks"] if c["check_id"] == "trained_parameter_change")["verdict"] == "NA"


def test_frozen_checkpoint_corruption_rejected_before_confirmation(tmp_path, monkeypatch):
    from tdn.analysis.roadmap import data
    monkeypatch.setattr(data, "load_bank", lambda ctx, split: bank(split))
    p = tiny_protocol(); p["models"] = ["rank1"]
    train_ctx = context(tmp_path, "train", p)
    neural.train(train_ctx)
    ctx = context(tmp_path, "confirm", p, {"train": train_ctx.path})
    models = neural.load_frozen_models(ctx)
    assert len(models) == 6
    assert all(not p.requires_grad for m in models.values() for p in m.parameters())
    checkpoint = next((train_ctx.path / "checkpoints").glob("rank1-seed-*.pt"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        neural.load_frozen_models(ctx)


def test_confirmation_retains_each_schedule_target_and_unaccepted_teacher_na(tmp_path, monkeypatch):
    from tdn.analysis.roadmap import data
    p = tiny_protocol(); p["models"] = ["source"]
    monkeypatch.setattr(data, "load_bank", lambda ctx, split: bank(split))
    train_ctx = context(tmp_path, "train", p)
    neural.train(train_ctx)
    accessed = []
    def heldout(ctx, split):
        assert (ctx.path / "frozen_model_ledger.json").exists()
        accessed.append(split)
        result = bank(split)
        result[0]["references"][f"discrete:4:{horizon_key(.06)}"]["accepted"] = False
        return result
    monkeypatch.setattr(data, "load_bank", heldout)
    ctx = context(tmp_path, "confirm", p, {"train": train_ctx.path})
    result = neural.confirm(ctx)
    assert accessed == ["confirmation"]
    assert result["endpoint_rows"] == 12
    assert result["all_schedule_groups"] == 6
    assert result["all_schedule_passes"] == 0
    endpoint_rows = [r for r in ctx.rows if r["experiment_id"].startswith("confirm/")]
    for row in endpoint_rows:
        targets = [c for c in row["checks"] if c["check_id"].startswith("target_")]
        assert len(targets) == 6 and all(c["verdict"] == "NA" for c in targets)
        assert row["metrics"]["median_seconds"] > 0
        validate_row(row)
    structure = [r for r in ctx.rows if r["experiment_id"].startswith("learned-structure/")]
    assert len(structure) == 6
    assert all("M21" in r["mechanism_ids"] for r in structure)


def test_train_validation_cluster_overlap_rejected(tmp_path, monkeypatch):
    from tdn.analysis.roadmap import data
    monkeypatch.setattr(data, "load_bank", lambda ctx, split: bank("shared"))
    with pytest.raises(ValueError, match="clusters overlap"):
        neural.train(context(tmp_path, "train", tiny_protocol()))


def test_failed_loss_retains_pre_failure_checkpoint_between_validations(tmp_path, monkeypatch):
    p = tiny_protocol(); p["training"]["validation_every"] = 2
    ctx = context(tmp_path, "train", p)
    original_loss, original_validation = neural._loss, neural.validation_metrics
    loss_calls, validation_calls = 0, 0
    def loss(*args, **kwargs):
        nonlocal loss_calls
        loss_calls += 1
        result = original_loss(*args, **kwargs)
        return result * float("nan") if loss_calls == 3 else result
    def validate(*args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        result = original_validation(*args, **kwargs)
        result["objective"] = 1. / validation_calls
        return result
    monkeypatch.setattr(neural, "_loss", loss)
    monkeypatch.setattr(neural, "validation_metrics", validate)
    record = neural._trial(ctx, "source", 14, .001, 1, 4, bank("train"), bank("validation"),
                           "late-failure", phase="test", config={"width": 2, "modes": 1})
    assert record["training_outcome"] == "NUMERICAL_FAILURE"
    assert record["updates_planned"] == 4
    assert record["updates_attempted"] == 3 and record["updates_completed"] == 2
    assert record["updates_selected"] == 2
    assert record["selection"] == "FALLBACK_PRE_FAILURE_CHECKPOINT"
    assert not record["eligible_for_optimizer_selection"]
    saved = torch.load(ctx.path / record["checkpoint"], weights_only=True)
    assert saved["selected_update"] == 2 and saved["checkpoint_validated"]
    assert all(torch.isfinite(v).all() for v in saved["state_dict"].values())
    row = ctx.rows[-1]
    assert row["status"] == "NUMERICAL_FAILURE" and row["assessment"]["verdict"] == "BAD"
    assert next(c for c in row["checks"] if c["check_id"] == "numerical_training_completed")["verdict"] == "BAD"
    validate_row(row)


def test_nonfinite_gradient_preserves_initialized_checkpoint(tmp_path, monkeypatch):
    ctx = context(tmp_path, "train", tiny_protocol())
    def nonfinite_gradient(model, *args, **kwargs):
        # Forward zero is finite; derivative of norm at zero is undefined.
        return model.head.weight.square().sum().sqrt()
    monkeypatch.setattr(neural, "_loss", nonfinite_gradient)
    record = neural._trial(ctx, "source", 15, .001, 1, 2, bank("train"), bank("validation"),
                           "gradient-failure", phase="test", config={"width": 2, "modes": 1})
    assert record["numerical_failure"]["code"] == "NONFINITE_GRADIENT_NORM"
    assert record["updates_attempted"] == 1 and record["updates_completed"] == 0
    assert record["selection"] == "FALLBACK_INITIALIZATION" and record["checkpoint_validated"]


def test_nonfinite_validation_is_recovered_without_promoting_bad_weights(tmp_path, monkeypatch):
    ctx = context(tmp_path, "train", tiny_protocol())
    original = neural.validation_metrics
    calls = 0
    def validation(model, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            with torch.no_grad():
                model.head.weight.fill_(float("inf"))
        return original(model, *args, **kwargs)
    monkeypatch.setattr(neural, "validation_metrics", validation)
    record = neural._trial(ctx, "source", 16, .001, 1, 2, bank("train"), bank("validation"),
                           "validation-failure", phase="test", config={"width": 2, "modes": 1})
    assert record["numerical_failure"]["phase"] == "validation"
    assert record["numerical_failure"]["code"] == "NONFINITE_VALIDATION_PREDICTION"
    assert record["updates_completed"] == 1 and record["updates_selected"] == 0
    state = torch.load(ctx.path / record["checkpoint"], weights_only=True)["state_dict"]
    assert torch.count_nonzero(state["head.weight"]) == 0


def test_failed_tuning_never_wins_and_all_failed_arm_does_not_stop_others(tmp_path, monkeypatch):
    from tdn.analysis.roadmap import data
    monkeypatch.setattr(data, "load_bank", lambda ctx, split: bank(split))
    p = tiny_protocol(); p["models"] = ["rank1", "source"]
    p["training"]["learning_rates"] = [.001, .003]
    original = neural._loss
    source_calls = 0
    def loss(model, *args, **kwargs):
        nonlocal source_calls
        value = original(model, *args, **kwargs)
        if model.family == "rank1":
            return value * float("nan")
        source_calls += 1
        return value * float("nan") if source_calls == 1 else value
    monkeypatch.setattr(neural, "_loss", loss)
    ctx = context(tmp_path, "train", p)
    result = neural.train(ctx)
    catalog = json.loads((ctx.path / "catalog.json").read_text())
    source = next(r for r in catalog["records"] if r["family"] == "source")
    rank = next(r for r in catalog["records"] if r["family"] == "rank1")
    assert source["tuning_selection"] == "source-tune-01"
    assert source["training_outcome"] == "COMPLETED"
    assert rank["training_outcome"] == "NUMERICAL_FAILURE"
    assert rank["updates_attempted"] == 0 and rank["updates_completed"] == 0
    assert rank["updates_planned"] == p["training"]["updates"]
    assert rank["selection"] == "FALLBACK_INITIALIZATION"
    assert result["selected_models"] == 2 and result["numerically_failed_trials"] == 4
    loaded = neural.load_frozen_models(context(tmp_path, "confirm", p, {"train": ctx.path}))
    rank_model = loaded[rank["model_id"]]
    assert rank_model.selection_metadata["training_outcome"] == "NUMERICAL_FAILURE"
    assert rank_model.selection_metadata["checkpoint_validated"]
    confirm_ctx = context(tmp_path, "confirm-fallback", p, {"train": ctx.path})
    neural.confirm(confirm_ctx)
    retained = [r for r in confirm_ctx.rows if r["experiment_id"].startswith(f"confirm/{rank['model_id']}/")]
    assert retained and any(r["metrics"]["passed_both"] for r in retained)
    assert all(next(c for c in r["checks"] if c["check_id"] == "numerical_training_completed")["verdict"] == "BAD" for r in retained)


def test_invalid_initialization_is_retained_as_unavailable_not_a_fake_checkpoint(tmp_path, monkeypatch):
    from tdn.analysis.roadmap import data
    monkeypatch.setattr(data, "load_bank", lambda ctx, split: bank(split))
    original = neural.validation_metrics
    def validation(model, *args, **kwargs):
        if model.family == "rank1":
            raise neural.TrialNumericalFailure("NONFINITE_VALIDATION_PREDICTION")
        return original(model, *args, **kwargs)
    monkeypatch.setattr(neural, "validation_metrics", validation)
    p = tiny_protocol(); p["models"] = ["rank1", "source"]
    ctx = context(tmp_path, "train", p)
    neural.train(ctx)
    catalog = json.loads((ctx.path / "catalog.json").read_text())
    failed = next(r for r in catalog["records"] if r["family"] == "rank1")
    assert failed["selection"] == "NO_VALIDATED_CHECKPOINT"
    assert not failed["checkpoint_validated"]
    future = context(tmp_path, "confirm", p, {"train": ctx.path})
    models = neural.load_frozen_models(future)
    assert failed["model_id"] not in models
    assert any(model.family == "source" for model in models.values())
    missing = future.rows[0]
    assert missing["status"] == "NUMERICAL_FAILURE"
    assert missing["gap_assessment"]["verdict"] == "NA"
    assert missing["math_assessment"]["verdict"] == "NA"
    validate_row(missing)


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), ValueError("invalid code shape"),
                                  TimeoutError("budget exhausted"), InterruptedError("signal")])
def test_resource_code_and_interrupt_errors_still_abort(tmp_path, monkeypatch, error):
    ctx = context(tmp_path, "train", tiny_protocol())
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(neural, "_loss", broken)
    with pytest.raises(type(error), match=str(error)):
        neural._trial(ctx, "source", 17, .001, 1, 2, bank("train"), bank("validation"),
                      "not-swallowed", phase="test", config={"width": 2, "modes": 1})
    assert not (ctx.path / "checkpoints" / "not-swallowed.pt").exists()


@pytest.mark.parametrize("bad_value,code", [(float("nan"), "NONFINITE_PREDICTED_STATE"),
                                          (1.1, "PREDICTED_STATE_OUTSIDE_ADMISSIBLE_INTERVAL")])
def test_failed_confirmation_case_retained_and_later_models_continue(tmp_path, monkeypatch, bad_value, code):
    from tdn.analysis.roadmap import data
    from tdn.analysis.roadmap.models import RoadmapSolver
    monkeypatch.setattr(data, "load_bank", lambda ctx, split: bank(split))
    p = tiny_protocol(); p["models"] = ["source"]
    training = context(tmp_path, "train", p)
    neural.train(training)
    original = RoadmapSolver.forward
    def bad_prediction(model, u, *args, **kwargs):
        return torch.full_like(u, bad_value) if model.family == "source" else original(model, u, *args, **kwargs)
    monkeypatch.setattr(RoadmapSolver, "forward", bad_prediction)
    ctx = context(tmp_path, "confirm", p, {"train": training.path})
    result = neural.confirm(ctx)
    assert result["endpoint_rows"] == 12
    assert result["numerically_failed_endpoints"] == 2
    failures = [r for r in ctx.rows if r["experiment_id"].startswith("confirm/source-")]
    assert len(failures) == 2
    for row in failures:
        assert row["status"] == "NUMERICAL_FAILURE" and row["assessment"]["verdict"] == "BAD"
        assert row["metrics"]["numerical_failure"] == code
        assert row["metrics"]["error_rms"] is None and row["metrics"]["median_seconds"] is None
        assert row["metrics"]["failed_attempt_seconds"] > 0
        assert row["metrics"]["completed_steps_before_failure"] == 0
        validate_row(row)
    successful = [r for r in ctx.rows if r["experiment_id"].startswith("confirm/") and r["status"] == "COMPLETED"]
    assert len(successful) == 10
    summary = json.loads((ctx.path / "confirmation_rows.json").read_text())
    source = next(r for r in summary["all_schedule_groups"] if r["model_id"].startswith("source-"))
    assert source["schedules_reported"] == 2 and source["schedules_completed"] == 0
    assert source["numerically_failed_schedules"] == 2 and not source["all_schedules_pass"]
    compared = json.loads((ctx.path / "matched_comparisons.json").read_text())["comparisons"]
    assert all(not r["candidate_feasible"] and r["candidate_cost_seconds"] is None for r in compared)


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), ValueError("code shape failure")])
def test_confirmation_resource_and_code_errors_are_not_reclassified(tmp_path, monkeypatch, error):
    from tdn.analysis.roadmap import data
    from tdn.analysis.roadmap.models import RoadmapSolver
    monkeypatch.setattr(data, "load_bank", lambda ctx, split: bank(split))
    p = tiny_protocol(); p["models"] = ["source"]
    training = context(tmp_path, "train", p)
    neural.train(training)
    original = RoadmapSolver.forward
    def broken(model, *args, **kwargs):
        if model.family == "source":
            raise error
        return original(model, *args, **kwargs)
    monkeypatch.setattr(RoadmapSolver, "forward", broken)
    with pytest.raises(type(error), match=str(error)):
        neural.confirm(context(tmp_path, "confirm", p, {"train": training.path}))
