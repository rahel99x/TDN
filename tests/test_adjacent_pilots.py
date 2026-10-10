"""Scientific separation, retained failures and recovery for channel pilots."""
import json
import math

import pytest
import torch

from tdn.analysis.adjacent import pilots
from tdn.analysis.adjacent.protocol import build_protocol


def small_protocol():
    p = build_protocol("smoke")
    p["pilot"].update(models=["quad2_fixed", "channel_global", "channel_neural"],
        tracks=["discrete"], train_fields=1, val_fields=1, evaluation_fields=1,
        endpoint_steps=[1, 2], max_seconds=120, evaluation_seconds=120, updates=1)
    return p


def test_pilot_aliases_keep_frozen_protocol_settings():
    p = small_protocol()
    options = pilots.settings(p)
    assert options["validation_fields"] == 1
    assert options["reference_max_substeps"] == 256
    assert options["seed_base"] == 71000000
    assert pilots._model_config(options, "channel_neural")["quad_nodes"] == 2
    assert pilots._model_config(options, "analytic_quad_cubic")["quad_nodes"] == 4
    assert pilots._model_config(options, "analytic_quad_cubic")["cubic_nodes"] == 4


def test_full_cannot_run_or_confirm_on_cpu(tmp_path):
    p = build_protocol("full")
    with pytest.raises(ValueError, match="native allocated CUDA"):
        pilots.run_pilot(p, tmp_path / "fit", device="cpu")
    with pytest.raises(ValueError, match="native allocated CUDA"):
        pilots.evaluate_pilot(p, tmp_path / "eval", tmp_path / "missing", device="cpu")


def test_missing_freeze_prevents_even_generating_fresh_parents(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Fresh parents were touched before verifying the frozen selection")
    monkeypatch.setattr(pilots, "_parents", forbidden)
    with pytest.raises(ValueError, match="Missing or unsafe"):
        pilots.evaluate_pilot(small_protocol(), tmp_path / "eval", tmp_path / "missing")


def test_real_pilot_freezes_disjoint_data_and_evaluation_recovers(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    p = small_protocol()
    train = tmp_path / "fit"
    result = pilots.run_pilot(p, train)
    assert result["status"] == "COMPLETED"
    assert result["model_count"] == 3
    assert result["reference_status"] == "ACCEPTED"
    frozen = pilots.verify_freeze(p, train)
    assert set(frozen["training_parents"]).isdisjoint(frozen["validation_parents"])
    assert "confirmation-plan.json" in frozen["artifacts"]
    assert "validation-rows.json" in frozen["artifacts"]
    catalog = json.loads((train / "catalog.json").read_text())
    fitted = next(r for r in catalog if r["family"] == "channel_global")
    assert fitted["parameter_count"] == 3
    assert fitted["trainable_parameter_count"] == 0
    assert fitted["fit_info"]["matched_objective"]
    for row in catalog:
        assert row["initial_checkpoint_sha256"]
        assert row["selection_status"] in {"FROZEN_CONTROL", "FITTED_CHECKPOINT", "SELECTED_INITIALIZATION"}
    evaluation = pilots.evaluate_pilot(p, tmp_path / "eval", train)
    assert evaluation["endpoint_count"] == 6
    assert all(r["scientific_verdict"] == "NA" for r in evaluation["rows"])
    assert {tuple(r["schedule"]) for r in evaluation["rows"]} == {(.06,), (.03, .03)}
    assert all(r["cost_seconds"] > 0 and len(r["raw_timing_seconds"]) == 2 for r in evaluation["rows"])
    assert {r["parent_id"] for r in evaluation["rows"]}.isdisjoint(
        frozen["training_parents"] + frozen["validation_parents"])
    def forbidden(*args, **kwargs):
        pytest.fail("A completed paired group must not be timed again during recovery")
    monkeypatch.setattr(pilots, "measure_paired", forbidden)
    recovered = pilots.evaluate_pilot(p, tmp_path / "eval", train)
    assert recovered["rows"] == evaluation["rows"]
    assert pilots.run_pilot(p, train)["model_count"] == 3
    (train / fitted["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        pilots.verify_freeze(p, train)


def test_failed_training_retains_verified_initialization(tmp_path, monkeypatch):
    from tdn.analysis.adjacent import models
    class BrokenTrain(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.gain = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, u, *args):
            return u + self.gain if not self.training else u + self.gain * float("nan")
    monkeypatch.setattr(models, "make_model", lambda *args, **kwargs: BrokenTrain())
    p = small_protocol()
    sample = dict(u=torch.ones(1, 1, 8, 8) * .4, target=torch.ones(1, 1, 8, 8) * .45,
                  h=.06, eq=None, geom=None, scale=1.)
    trial = pilots._fit_trial(p, tmp_path, "channel_neural", "discrete", 1, .001,
        [sample], [sample], pilots._Budget(60), torch.device("cpu"))
    assert trial["failure"] == "NONFINITE_TRAINING_LOSS"
    assert trial["selection_status"] == "SELECTED_INITIALIZATION"
    assert trial["checkpoint_validated"]
    assert trial["selected_update"] == 0
    assert (tmp_path / trial["initial_checkpoint"]).is_file()
    assert (tmp_path / trial["checkpoint"]).is_file()


def synthetic_rows(n=16, spread=.1, repeats=2):
    rows = []
    for i in range(n):
        for j in range(repeats):
            for family, error in (("channel_neural", 1e-5), ("channel_global", 1e-5 * math.exp(.5 + spread * (i % 2))),
                                  ("quad2_conditioned", 2e-5), ("band_gain", 3e-5)):
                rows.append(dict(family=family, track="discrete", seed=731001, field_cluster=f"parent-{i}",
                    horizon=[.04, .12][j % 2] if repeats == 2 else .04 + .01*j, steps=1, error_rms=error, reference_uncertainty_rms=1e-12, reference_accepted=True, finite=True))
    return rows


def test_confirmation_precision_uses_independent_fields_and_never_expands_cap():
    p = build_protocol("full"); p["pilot"]["tracks"] = ["discrete"]
    rows = synthetic_rows(n=4, spread=2., repeats=10)
    plan = pilots._plan_confirmation(p, rows)
    assert all(c["validation_parent_count"] == 4 for c in plan["contrasts"])
    assert plan["status"] == "INSUFFICIENT_PRECISION_BUDGET"
    assert plan["chosen_parents"] == 64
    assert plan["required_parents"] > 64
    assert plan["frozen_before_fresh_references"]


def test_descriptive_smoke_never_promotes_success_and_missing_inventory_is_na():
    smoke = build_protocol("smoke")
    rows = synthetic_rows()
    plan = pilots._plan_confirmation(smoke, rows)
    comparisons = pilots._primary_comparisons(smoke, rows, plan)
    assert all(c["verdict"] == "NA" for c in comparisons)
    full = build_protocol("full"); full["pilot"]["tracks"] = ["discrete"]
    plan = pilots._plan_confirmation(full, rows)
    assert plan["status"] == "PRECISION_PLAN_WITHIN_CAP"
    missing = [r for r in rows if r["field_cluster"] != "parent-0"]
    assert all(c["verdict"] == "NA" for c in pilots._primary_comparisons(full, missing, plan))


def test_stop_callback_and_budget_retain_distinct_failures():
    with pytest.raises(InterruptedError):
        pilots._Budget(60, lambda: True).check()
    with pytest.raises(TimeoutError, match="walltime"):
        pilots._Budget(-1).check()
    def source_budget():
        raise InterruptedError("upstream stop")
    with pytest.raises(InterruptedError, match="upstream"):
        pilots._Budget(60, source_budget).check()


def test_primary_rejects_incomplete_horizons_and_reference_unresolved_improvement():
    p = build_protocol("full"); p["pilot"]["tracks"] = ["discrete"]
    rows = synthetic_rows()
    plan = pilots._plan_confirmation(p, rows)
    incomplete = [r for r in rows if not (r["field_cluster"] == "parent-0" and r["horizon"] == .12)]
    result = pilots._primary_comparisons(p, incomplete, plan)
    assert all(r["verdict"] == "NA" and r["missing_parents"] for r in result)
    unresolved = [{**r, "reference_uncertainty_rms": 4e-5} for r in rows]
    assert all(r["verdict"] == "NA" and len(r["reference_unresolved_parents"]) == 16
               for r in pilots._primary_comparisons(p, unresolved, plan))


def test_roughness_generator_factor_is_counterbalanced_and_not_perfectly_confounded():
    p = build_protocol("full")
    parents = pilots._parents(pilots.settings(p), "train")
    for alpha in p["pilot"]["alpha_cycle"]:
        assert {parent.generator for parent in parents if parent.alpha == alpha} == {"ridge", "multiscale_2d"}


def test_frozen_pilot_ignores_outer_mutable_stage_metadata(tmp_path):
    p = small_protocol(); p["pilot"]["models"] = ["quad2_fixed"]
    (tmp_path / "execution.json").write_text('{"status":"RUNNING"}')
    pilots.run_pilot(p, tmp_path)
    (tmp_path / "execution.json").write_text('{"status":"COMPLETED"}')
    frozen = pilots.verify_freeze(p, tmp_path)
    assert "execution.json" not in frozen["artifacts"]
