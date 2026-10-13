"""Scientific controls: independent-parent weights, exact work resume and selection."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.response import training as tr
from tdn.analysis.response.training_control import control_config, learning_rate


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def sample(parent, offset=.1, *, split="train", horizon=.06, regime="low_mean"):
    u = torch.linspace(.2, .7, 16).reshape(1, 1, 4, 4)
    return dict(u0=u, target=u.double()+offset, anchor_fp32=u.clone(), anchor_fp64=u.double(),
        record=dict(parent_id=parent, query_id=f"{parent}/{horizon}", split=split, grid=[4, 4], lengths=[1., 1.],
            horizon=horizon, kappa=.01, reaction_rate=3., target_kind="fd_same_grid", tiny_regime=regime,
            reference=dict(accepted=True, uncertainty_rms=1e-11, uncertainty_maximum=1e-11)))


def config(updates=8):
    return dict(id="diagnostic", family="test", target_kind="fd_same_grid", phase="tiny_joint", seed=7,
        updates=updates, total_seconds=1000., validation_every=2, batch_mode="full_parent", objective_mode="physical",
        lr_schedule=dict(kind="cosine", minimum_ratio=.05, warmup_updates=1), model={},
        setting=dict(lr=.01, betas=[.9, .999], eps=1e-8, weight_decay=0., clip=None))


class ScalarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_head = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, value):
        return value+self.output_head


class DropoutNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Conv2d(1, 3, 1)
        self.dropout = torch.nn.Dropout(.3)
        self.output_head = torch.nn.Conv2d(3, 1, 1)
        torch.nn.init.zeros_(self.output_head.weight)
        torch.nn.init.zeros_(self.output_head.bias)

    def forward(self, value):
        return value+self.output_head(self.dropout(torch.tanh(self.trunk(value))))


def test_full_parent_weights_average_queries_within_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "predict", lambda model, value: model(value["u0"]))
    train = [sample("a", .1), sample("a", .3, horizon=.03), sample("b", .6)]
    cfg = config(1)
    cfg["lr_schedule"] = {"kind":"fixed"}
    cfg["setting"].update(betas=[0., 0.], eps=1.)
    model = ScalarNet()
    row = tr.train_trial(model, train, [sample("v", split="validation")], cfg,
                         tr.training_normalization(train), tmp_path, identity={"test":"weights"})
    # Half-MSE has gradient -(mean(.1,.3)+.6)/2 = -.4.
    assert float(model.output_head.detach()) == pytest.approx(.01*.4/1.4, rel=2e-6)
    curves = [json.loads(line) for line in (tmp_path/"learning_curves.jsonl").read_text().splitlines()]
    assert curves[-1]["query_weights"] == [.25, .25, .5]
    assert curves[-1]["gradient_norm"] == pytest.approx(.4, rel=2e-6)
    assert row["examples_seen"] == 3
    assert row["work"]["logical_optimizer_updates"] == 1


def test_full_parent_resume_restores_schedule_warmup_rng_and_exact_work(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "predict", lambda model, value: model(value["u0"]))
    train = [sample("b", .2), sample("a", .1), sample("a", .3, horizon=.03)]
    validation = [sample("v", .15, split="validation")]
    cfg = config(); cfg["head_warmup_updates"] = 2
    norm = tr.training_normalization(train)
    torch.manual_seed(51)
    full = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"full", identity={"same":True})
    torch.manual_seed(51)
    part = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"part", identity={"same":True},
                          max_chunk_updates=3, final_chunk=False)
    assert part["status"] == "CONTINUATION_REQUIRED"
    torch.manual_seed(91)
    resumed = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"resumed", identity={"same":True},
                             resume=tmp_path/"part/journal.pt")
    a = torch.load(tmp_path/"full/journal.pt", weights_only=True)
    b = torch.load(tmp_path/"resumed/journal.pt", weights_only=True)
    assert all(torch.equal(a["model"][name], b["model"][name]) for name in a["model"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["scheduler_state"] == b["scheduler_state"]
    assert [r.get("learning_rate") for r in a["history"]] == [r.get("learning_rate") for r in b["history"]]
    assert [r.get("train_loss") for r in a["history"]] == [r.get("train_loss") for r in b["history"]]
    assert resumed["examples_seen"] == full["examples_seen"] == 24
    assert full["selected_state_sha256"] == resumed["selected_state_sha256"]
    assert a["history"][1]["trunk_movement"] == a["history"][2]["trunk_movement"] == 0
    assert a["history"][3]["trunk_movement"] > 0


def test_criterion_checkpoint_cannot_replace_validation_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "predict", lambda model, value: model(value["u0"]))
    train = [sample("a", .2)]
    validation = [sample("v", 0., split="validation")]
    cfg = config(4); cfg["lr_schedule"] = {"kind":"fixed"}
    row = tr.train_trial(ScalarNet(), train, validation, cfg, tr.training_normalization(train), tmp_path, identity={"test":"selection"})
    assert row["selected_update"] == 0
    assert row["criterion_update"] == row["attainable_update"] == 4
    selected = torch.load(tmp_path/"selected.pt", weights_only=True)
    diagnostic = torch.load(tmp_path/"criterion-diagnostic.pt", weights_only=True)
    assert float(selected["state_dict"]["output_head"]) == 0
    assert float(diagnostic["state_dict"]["output_head"]) > 0
    assert "never deployment" in diagnostic["checkpoint_role"]


def test_unit_column_svd_recovers_span_despite_physical_scale_disparity():
    a = torch.tensor([[1e-14, 1.], [2e-14, -1.], [3e-14, .5]], dtype=torch.float64)
    expected = torch.tensor([2e13, .4], dtype=torch.float64)
    fitted, metadata = tr._conditioned_affine_solve(a, a@expected, ridge=0., rcond=1e-10)
    assert torch.allclose(a@fitted, a@expected, rtol=1e-12, atol=1e-12)
    assert metadata["rank"] == 2
    assert metadata["retained_condition_number"] < 10
    assert metadata["raw_coefficient_l2"] > 1e12


def test_gain_freeze_does_not_access_validation_or_teacher_values(monkeypatch):
    from tdn.analysis.response import temporal_basis
    monkeypatch.setattr(temporal_basis, "correction_gains", lambda u, h, *a, **k:dict(mean_gain=u.mean()*h, centered_gain=u.mean()*h*2))
    train, validation = sample("a"), sample("v", split="validation")
    cfg = config(); norm = tr.training_normalization([train])
    result = tr.fit_correction_gains([train, validation], norm, cfg)
    altered = deepcopy([train, validation])
    altered[0]["target"] *= 1e6
    altered[1]["u0"] *= 1e6
    assert tr.fit_correction_gains(altered, norm, cfg) == result
    assert result["mean_gain"] > 0
    assert result["gain_normalization"]["training_parent_ids"] == ["a"]


def test_screen_selection_is_parent_bound_and_missing_regime_is_explicit(tmp_path, monkeypatch):
    from tdn.analysis.response import references
    samples = [sample("chosen", regime="strong_reaction"), sample("other", regime="strong_reaction"),
               sample("valid", split="validation", regime="strong_reaction")]
    monkeypatch.setattr(references, "load_samples", lambda *a: samples)
    screen = dict(schema="tdn.response-screening/v1", selected_parents={"fd_same_grid":{
        "train":{"strong_reaction":"chosen"}, "validation":{"strong_reaction":"valid"}}})
    (tmp_path/"screening.json").write_text(json.dumps(screen))
    ctx = SimpleNamespace(unit={"data_stages":["prepare"]}, prerequisites={"prepare":tmp_path, "screening":tmp_path}, device="cpu")
    cfg = dict(target_kind="fd_same_grid", phase="tiny_individual", regime="strong_reaction", screen_stage="screening")
    chosen, _ = tr._data(ctx, cfg)
    assert [s["record"]["parent_id"] for s in chosen] == ["chosen", "valid"]
    assert cfg["screening"]["status"] == "ELIGIBLE"
    cfg["regime"] = "high_mean"
    chosen, _ = tr._data(ctx, cfg)
    assert chosen == []
    assert cfg["screening"]["status"] == "UNRESOLVED_SCREEN"
    assert len(cfg["screening"]["missing"]) == 2


def test_cosine_schedule_and_controls_reject_invalid_work_budgets():
    cfg = config(8)
    cfg["lr_schedule"]["warmup_updates"] = 2
    control = control_config(cfg)
    rates = [learning_rate(control, index) for index in range(8)]
    assert rates[:3] == [.005, .01, .01]
    assert rates[-1] == pytest.approx(.0005)
    assert all(a >= b for a, b in zip(rates[2:], rates[3:]))
    with pytest.raises(ValueError, match="budgets"):
        control_config({**cfg, "updates":0})
    with pytest.raises(ValueError, match="minimum_ratio"):
        control_config({**cfg, "lr_schedule":{"kind":"cosine", "minimum_ratio":float("nan")}})


def test_unresolved_screen_writes_a_complete_no_training_result(tmp_path, monkeypatch):
    from tdn.analysis.response import references, models
    monkeypatch.setattr(references, "load_samples", lambda *a:[sample("candidate"), sample("v", split="validation")])
    monkeypatch.setattr(models, "create_model", lambda *a:pytest.fail("Unresolved screening must not construct or train a model"))
    (tmp_path/"screening.json").write_text(json.dumps(dict(schema="tdn.response-screening/v1", selected_parents={})))
    cfg = {**config(), "screen_stage":"screening", "inference_benchmark":{"repeats":2, "warmup":1}}
    records = []
    ctx = SimpleNamespace(unit={"trial_id":"diagnostic", "data_stages":["prepare"]},
        protocol={"trials":{"diagnostic":cfg}}, prerequisites={"prepare":tmp_path, "screening":tmp_path},
        path=tmp_path, device="cpu", identities=set(), record=lambda *a, **k:records.append((a, k)))
    result = tr.run_training(ctx)
    row = json.loads((tmp_path/"trial.json").read_text())
    assert result["trial_status"] == row["status"] == "UNRESOLVED_SCREEN"
    assert row["updates"] == 0 and row["requested_updates"] == 8
    assert row["selected_train"]["mse"] is None
    assert row["inference_cost"]["status"] == "NA"
    assert row["postprocessing_cost"]["inference_benchmark_seconds"] >= 0
    assert not (tmp_path/"selected.pt").exists()
    assert len(records) == 1


def test_fp64_assembly_rollout_reencodes_in_parameter_precision_with_gradient():
    class Assembly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_head = torch.nn.Parameter(torch.tensor(.2))
            self.inputs = []

        def forward(self, u, h, *_):
            self.inputs.append(u.dtype)
            return u.double()*(1+self.output_head.double()*h)

    model = Assembly()
    value = tr._rollout_loss(model, sample("a"), {"mean_scale":1., "centered_scale":1.}, 3)
    gradient = torch.autograd.grad(value, model.output_head)[0]
    assert model.inputs == [torch.float32]*3
    assert float(gradient) != 0


def test_unsupported_affine_families_are_explicit_na():
    model = ScalarNet()
    model.family = "adaptive_temporal"
    row = tr.fit_frozen_encoder(model, [sample("a")], {"mean_scale":1., "centered_scale":1.})
    assert row["status"] == "NA"
    assert "affine" in row["reason"]


def test_inference_hook_uses_selected_checkpoint_and_canonical_validation_query(tmp_path, monkeypatch):
    selected_model = ScalarNet()
    with torch.no_grad():
        selected_model.output_head.fill_(.25)
    # This fixture is a new local artifact; no historical payload is loaded.
    (tmp_path/"selected.pt").write_bytes(b"locally generated selected checkpoint fixture")
    calls = []
    monkeypatch.setattr(tr, "load_model", lambda path, device: (selected_model, {"normalization":{"test":"selected"}, "update":2}))

    def benchmark(model, samples, normalization, **kwargs):
        calls.append((model, samples, normalization, kwargs))
        return dict(status="MEASURED", elapsed_seconds=.01)

    monkeypatch.setitem(sys.modules, "tdn.analysis.response.diagnostic_performance", SimpleNamespace(benchmark_model=benchmark))
    late = sample("z", split="validation")
    first = sample("a", split="validation")
    cfg = {"inference_benchmark":{"repeats":2, "warmup":1}}
    result = tr._postfit_inference_benchmark(tmp_path, [late, first], cfg, {"status":"COMPLETED_UPDATES"}, device="cpu")
    assert calls[0][0] is selected_model
    assert calls[0][1] == [first]
    assert calls[0][2] == {"test":"selected"}
    assert calls[0][3]["repeats"] == 2
    assert result["selected_update"] == 2
    assert result["checkpoint_role"] == "validation-selected checkpoint"
    assert json.loads((tmp_path/"inference_cost.json").read_text())["status"] == "MEASURED"


def test_inference_hook_marks_unresolved_trials_na_without_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda *a, **k:pytest.fail("Unresolved trial has no selected checkpoint"))
    result = tr._postfit_inference_benchmark(tmp_path, [], {"inference_benchmark":{"repeats":2, "warmup":1}},
                                            {"status":"UNRESOLVED_SCREEN"}, device="cpu")
    assert result["status"] == "NA"
    assert result["p99"] is None
    assert result["postprocessing_seconds"] >= 0
    assert tr._postfit_inference_benchmark(tmp_path, [], {}, {"status":"UNRESOLVED_SCREEN"}, device="cpu") is None


def test_reporting_cadence_compacts_plans_without_changing_training(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "predict", lambda model, value:model(value["u0"]))
    train = [sample("a", .1), sample("a", .2, horizon=.03), sample("b", .3)]
    validation = [sample("v", .2, split="validation")]
    norm = tr.training_normalization(train)
    cfg = config(12); cfg["validation_every"] = 3
    dense = {**cfg, "public_curve_every":1, "live_every":1, "compact_public_query_plan":True}
    sparse = {**cfg, "public_curve_every":4, "live_every":4, "compact_public_query_plan":True}
    torch.manual_seed(51)
    full = tr.train_trial(DropoutNet(), train, validation, dense, norm, tmp_path/"dense", identity={"test":"cadence"})
    torch.manual_seed(51)
    thin = tr.train_trial(DropoutNet(), train, validation, sparse, norm, tmp_path/"thin", identity={"test":"cadence"})
    a = torch.load(tmp_path/"dense/journal.pt", weights_only=True)
    b = torch.load(tmp_path/"thin/journal.pt", weights_only=True)
    assert all(torch.equal(a["model"][name], b["model"][name]) for name in a["model"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["scheduler_state"] == b["scheduler_state"]
    assert [r.get("query_ids") for r in a["history"]] == [r.get("query_ids") for r in b["history"]]
    assert [r.get("train_loss") for r in a["history"]] == [r.get("train_loss") for r in b["history"]]
    assert full["selected_state_sha256"] == thin["selected_state_sha256"]
    curves = [json.loads(line) for line in (tmp_path/"thin/learning_curves.jsonl").read_text().splitlines()]
    assert [r["update"] for r in curves] == [0, 3, 4, 6, 8, 9, 12]
    assert all("query_ids" not in row and "query_weights" not in row and "query_visits" not in row for row in curves)
    assert "query_ids" in b["history"][1]  # Full private journal is untouched.
    plans = json.loads((tmp_path/"thin/query_plan.json").read_text())["plans"]
    assert len(plans) == 1
    from tdn.research.protocol import digest
    assert all(digest(plan) == checksum for checksum, plan in plans.items())
    assert thin["optimization_history_summary"]["committed_updates"] == 12
    assert thin["optimization_history_summary"]["committed_query_visits"] == 36


def test_reporting_cadence_exact_resume_preserves_nonmultiple_attempt_end(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "predict", lambda model, value:model(value["u0"]))
    train = [sample("a"), sample("b", .2)]
    validation = [sample("v", split="validation")]
    cfg = {**config(12), "validation_every":3, "public_curve_every":4, "live_every":4, "compact_public_query_plan":True}
    norm = tr.training_normalization(train)
    torch.manual_seed(9)
    full = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"full", identity={"test":"resume"})
    torch.manual_seed(9)
    part = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"part", identity={"test":"resume"},
        max_chunk_updates=5, final_chunk=False)
    assert part["status"] == "CONTINUATION_REQUIRED"
    curves = [json.loads(line) for line in (tmp_path/"part/learning_curves.jsonl").read_text().splitlines()]
    assert curves[-1]["update"] == 5
    resumed = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"resumed", identity={"test":"resume"},
        resume=tmp_path/"part/journal.pt")
    a = torch.load(tmp_path/"full/journal.pt", weights_only=True)
    b = torch.load(tmp_path/"resumed/journal.pt", weights_only=True)
    assert all(torch.equal(a["model"][name], b["model"][name]) for name in a["model"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["work_counters"] == b["work_counters"]
    assert full["selected_state_sha256"] == resumed["selected_state_sha256"]
    assert len(b["history"]) == 13
    changed = {**cfg, "live_every":1}
    with pytest.raises(ValueError, match="reporting cadence"):
        tr.train_trial(DropoutNet(), train, validation, changed, norm, tmp_path/"changed", identity={"test":"resume"},
            resume=tmp_path/"part/journal.pt")


def test_whole_history_gradient_summary_retains_unsampled_spike_and_clipping():
    from tdn.analysis.response.training_control import reporting_control, sampled_curve_rows, whole_history_summary
    history = [{"update":0, "gradient_norm":200.}]+[dict(update=i, gradient_norm=100. if i == 17 else 1.,
        clipped=i == 17, query_visits=2, examples_seen=2*i) for i in range(1, 51)]
    policy = reporting_control({"public_curve_every":40, "live_every":40, "compact_public_query_plan":True})
    curves, _, _ = sampled_curve_rows(history, policy, chunk_start_update=0)
    assert [row["update"] for row in curves] == [0, 40, 50]
    summary = whole_history_summary(history)
    assert summary["metrics"]["gradient_norm"]["maximum"] == 100.
    assert summary["metrics"]["gradient_norm"]["maximum_update"] == 17
    assert summary["clipped_updates"] == 1
    assert summary["clipped_fraction"] == pytest.approx(1/50)
    assert summary["committed_query_visits"] == 100


def test_correction_diagnostics_timeout_retains_partial_rows_and_restores_mode():
    class Diagnostics(ScalarNet):
        family = "temporal"
        def correction_diagnostics(self, u, *_):
            assert not self.training
            return dict(raw_mean=u.mean(), raw_centered=u-u.mean(), total_correction=u*0., output=u)
    class OneQueryBudget:
        def __init__(self): self.calls=0
        def check(self):
            self.calls += 1
            if self.calls > 1: raise TimeoutError("diagnostic allowance exhausted")
    model = Diagnostics(); model.train()
    result = tr._correction_summary(model, [sample("a"), sample("b")], budget=OneQueryBudget())
    assert result["status"] == "UNRESOLVED"
    assert result["queries"] == 1 and result["requested_queries"] == 2
    assert model.training
    assert "analytic time response" in result["rows"][0]["raw_units_scope"]


@pytest.mark.parametrize("family", ("direct_fno", "residual_fno", "temporal", "adaptive_temporal", "spatial_temporal", "coupled_source"))
def test_real_correction_summary_all_six_learned_families(family):
    from tdn.analysis.response.models import create_model
    samples = [sample("a")]
    norm = tr.training_normalization(samples)
    model = create_model(dict(family=family, width=3, depth=1, modes=1, rank=4, normalization=norm))
    model.train()
    result = tr._correction_summary(model, samples)
    assert result["status"] == "MEASURED"
    assert result["queries"] == result["requested_queries"] == 1
    assert result["rows"][0]["mean_gain"]["finite"]
    assert result["rows"][0]["assembly_difference"]["finite"]
    assert result["rows"][0]["raw_units_scope"]
    assert model.training


def test_pending_validation_resume_keeps_both_observations_without_collector_conflict(tmp_path, monkeypatch):
    from tdn.analysis.response import diagnostic_metrics as dm
    from tdn.research.protocol import digest, file_digest
    from tdn.runtime.metadata import write_json
    monkeypatch.setattr(tr, "predict", lambda model, value:model(value["u0"]))
    train = [sample("a"), sample("b", .2)]
    validation = [sample("v", split="validation")]
    cfg = {**config(6), "validation_every":2, "public_curve_every":4, "live_every":4, "compact_public_query_plan":True}
    norm = tr.training_normalization(train)
    class InterruptSecondValidation:
        def __init__(self): self.calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 6:
                raise InterruptedError("Interrupt immediately before update-2 validation")
    torch.manual_seed(9)
    full = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"full", identity={"test":"pending"})
    torch.manual_seed(9)
    part = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"part", identity={"test":"pending"},
        budget=InterruptSecondValidation(), final_chunk=False)
    assert part["updates"] == 2 and part["status"] == "CONTINUATION_REQUIRED"
    saved = torch.load(tmp_path/"part/journal.pt", weights_only=True)
    assert saved["pending_validation"]
    old_curve = [json.loads(line) for line in (tmp_path/"part/learning_curves.jsonl").read_text().splitlines()]
    assert old_curve[-1]["observation_kind"] == "pending_validation"
    resumed = tr.train_trial(DropoutNet(), train, validation, cfg, norm, tmp_path/"resumed", identity={"test":"pending"},
        resume=tmp_path/"part/journal.pt")
    new_curve = [json.loads(line) for line in (tmp_path/"resumed/learning_curves.jsonl").read_text().splitlines()]
    finished = next(row for row in new_curve if row["update"] == 2)
    assert finished["validation_loss"] is not None
    assert finished.get("observation_kind", "scheduled") == "scheduled"
    assert resumed["selected_state_sha256"] == full["selected_state_sha256"]
    a = torch.load(tmp_path/"full/journal.pt", weights_only=True)
    b = torch.load(tmp_path/"resumed/journal.pt", weights_only=True)
    assert all(torch.equal(a["model"][name], b["model"][name]) for name in a["model"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    for key, values in a["optimizer"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, b["optimizer"]["state"][key][name])
    protocol = dict(profile="diagnostic", source_tree_sha256="a"*64,
        units={"part":{"kind":"train"}, "resumed":{"kind":"train"}, "analysis":{"kind":"diagnostic_metrics"}},
        trials={"diagnostic":cfg}, mechanisms={"M3":{}}, combinations={})
    analysis = tmp_path/"analysis"; analysis.mkdir()
    ctx = SimpleNamespace(protocol=protocol, stage="analysis", path=analysis, unit=protocol["units"]["analysis"],
        prerequisites={}, device="cpu")
    for stage in ("part", "resumed"):
        base = tmp_path/stage
        artifacts = {name:file_digest(base/name) for name in ("training_results.json", "learning_curves.jsonl", "query_plan.json")}
        manifest = dict(schema="tdn.response-science/v1", stage=stage, kind="train",
            source_tree_sha256="a"*64, protocol_sha256=digest(protocol), artifacts=artifacts)
        write_json(base/"science_manifest.json", manifest)
        (base/"COMPLETED").write_text(digest(manifest)+"\n")
        ctx.prerequisites[stage] = base
    data, _ = dm.collect(ctx)
    at_two = [row for row in data["learning_curves"] if row["update"] == 2]
    assert len(at_two) == 2
    assert {row.get("observation_kind", "scheduled") for row in at_two} == {"pending_validation", "scheduled"}
