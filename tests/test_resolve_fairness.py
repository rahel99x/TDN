"""Optimizer opportunity, selection separation and actual head-gradient checks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.resolve import fairness as f
from tdn.analysis.advance import learning
from tdn.analysis.advance.models import make_model
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import Budget, horizon_key


@pytest.fixture(autouse=True)
def bounded_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def settings():
    return dict(families=list(f.FAMILIES), optimizer_settings=[
        dict(id="clipped", learning_rate=.001, clip_grad_norm=1., loss_scale=1.),
        dict(id="unclipped", learning_rate=.001, clip_grad_norm=None, loss_scale=1.),
        dict(id="lower_lr", learning_rate=.0003, clip_grad_norm=None, loss_scale=1.)],
        updates=8, trial_seconds=20, equal_time_seconds=1, equal_time_max_updates=100,
        tiny_updates=8, tiny_seconds=10, tiny_relative_objective_target=.01,
        tiny_absolute_normalized_target=.01)


def row(phase="long_update", *, objective=1., identity="a", **kwargs):
    return dict(phase=phase, validation_objective=objective, checkpoint_validated=True,
                failure=None, trial_id=identity, initial_validation_objective=1.,
                stopping_reason="UPDATE_CAP", updates_selected=0, **kwargs)


def test_every_tied_optimizer_gets_long_opportunity_and_tiny_cannot_win():
    cfg = settings(); calls = []; samples = [dict(scale=2., parent_id="train", h=.1)]
    validation = [dict(parent_id="validation")]
    def trial(ctx, spec, config, actual_samples, actual_validation, rate, control, phase, **caps):
        calls.append((control["id"], phase, caps))
        if phase == "tiny_memorization":
            assert actual_samples is not samples and actual_validation == actual_samples
            return row(phase, objective=0., identity=control["id"] + phase)
        assert actual_validation is validation
        return row(phase, identity=control["id"] + phase)
    result = f.run_family_trials(None, {}, {}, samples, validation, cfg, trial_fn=trial)
    assert len(result["trials"]) == 6 and len(result["tiny"]) == 3
    assert {c[0] for c in calls if c[1] == "long_update"} == {"clipped", "unclipped", "lower_lr"}
    assert all(c[2] == dict(updates=8, seconds=20.) for c in calls if c[1] == "long_update")
    assert result["selected"]["validation_objective"] == 1.
    assert result["selected"]["phase"] in f.SELECTION_PHASES


def test_validation_only_selection_rejects_training_and_numerical_failures():
    entries = [row("tiny_memorization", objective=0.), row(objective=.7, identity="valid"),
               row(objective=.3, identity="failed"), row(objective=float("nan"), identity="nan")]
    entries[2]["failure"] = "NONFINITE_GRADIENT"
    entries[1]["training_loss"] = 999.
    assert f.select_validation_trial(entries)["trial_id"] == "valid"
    assert f.select_validation_trial(entries[:1]) is None


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_setting", "bad_rate", "bad_clip"])
def test_optimizer_plan_rejects_invalid_controls(change):
    cfg = settings()
    if change == "duplicate_id": cfg["optimizer_settings"][1]["id"] = "clipped"
    if change == "duplicate_setting":
        cfg["optimizer_settings"][1] = dict(cfg["optimizer_settings"][0], id="duplicate")
    if change == "bad_rate": cfg["optimizer_settings"][0]["learning_rate"] = float("inf")
    if change == "bad_clip": cfg["optimizer_settings"][0]["clip_grad_norm"] = 0.
    with pytest.raises(ValueError): f.optimizer_plan(cfg)


@pytest.mark.parametrize("objective,stop,expected", [(1e-3,"UPDATE_CAP","GOOD"), (.5,"UPDATE_CAP","BAD"),
                                                     (.5,"TIME_CAP","NA"), (1e-3,"TIME_CAP","GOOD")])
def test_tiny_target_is_low_residual_with_censoring(objective, stop, expected):
    trial = row(objective=objective); trial["stopping_reason"] = stop
    gate = f.tiny_gate(trial, settings())
    assert gate["verdict"] == expected and gate["selection_eligible"] is False
    assert gate["relative_objective_target"] == .01


def field(n=8, phase=.2):
    x = torch.arange(n, dtype=torch.float32)/n
    return (.43 + .12*torch.cos(2*torch.pi*x[:,None]+phase)
            + .06*torch.sin(4*torch.pi*x[None,:]-.4))[None,None]


@pytest.mark.parametrize("family", ["fno_scaled", "fno_mean"])
@pytest.mark.parametrize("track", ["discrete", "continuum"])
def test_scaled_and_mean_heads_activate_then_trunk_gradients_are_finite(family, track):
    torch.manual_seed(17)
    model = make_model(family, track, dict(width=3, depth=1, modes=2, reaction_substeps=2)).float()
    u = field(); eq = Equation(.003, 2.); geo = Geometry((8,8), (1.,1.)); h = .1
    with torch.no_grad():
        parts = model.correction_components(u, h, eq, geo)
        # Explicit nonzero mean + spatial signal exercises both independent heads.
        target = parts["base"] + parts["output_scale"]*(.4 + 2*(u-u.mean()))
    scale = max(float(parts["output_scale"].square().mean()), 1e-20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(u,h,eq,geo)-target).square().mean()/scale
        loss.backward()
        active = {name: float(p.grad.abs().sum()) for name,p in model.named_parameters() if p.grad is not None}
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(value>0 for name,value in active.items() if name.startswith("head"))
        if family == "fno_mean": assert any(value>0 for name,value in active.items() if name.startswith("mean_head"))
        if update: assert any(value>0 for name,value in active.items() if name.startswith("lift"))
        optimizer.step()


def test_real_trials_keep_initial_checkpoints_and_resume_completed_journals(tmp_path, monkeypatch):
    cfg = settings()
    cfg.update(families=["fno_scaled"], optimizer_settings=cfg["optimizer_settings"][:1], updates=2,
               tiny_updates=2, equal_time_seconds=0, train_count=1)
    protocol = dict(resolve=dict(fairness=cfg), training=dict(validation_every=1, batch_size=1,
        peak_weight=.1, normalization_floor=1e-6), model_config=dict(width=3, depth=1, modes=2, reaction_substeps=2),
        model_configs={}, train_grid=8, train_horizons=[.1], validation_horizons=[.1])
    bank = []
    eq = Equation(.003,2.); geom = Geometry((8,8),(1.,1.))
    teacher = make_model("etdrk4", "discrete", protocol["model_config"]).double()
    for split,phase in [("train",.2),("validation",.7)]:
        state = field(phase=phase).double(); truth = state
        with torch.no_grad():
            for _ in range(16): truth = teacher(truth,.1/16,eq,geom)
        bank.append(dict(split=split,parent_id=split,field_cluster=split,regime="smooth", kappa=.003,
            reaction_rate=2.,states={"8":state}, references={f"discrete:8:{horizon_key(.1)}":
                dict(state=truth,accepted=True,uncertainty_rms=1e-10,uncertainty_max_bound=1e-9)}))
    monkeypatch.setattr(f,"load_bank",lambda *_: bank)
    monkeypatch.setattr(f,"binding",lambda *_: dict(protocol_sha256="test",source_tree_sha256="test",software_sha256="test"))
    monkeypatch.setattr(learning,"binding",f.binding)
    records=[]
    ctx=SimpleNamespace(protocol=protocol,unit=dict(track="discrete",seed=17),stage="fairness-discrete-17",
        path=tmp_path,device="cpu",budget=Budget(60),identities=set(),record=lambda *a,**k:records.append((a,k)))
    before=deepcopy(protocol)
    result=f.run(ctx)
    assert protocol==before and result["completed_trials"]==2
    report=json.loads((tmp_path/"fairness_results.json").read_text())
    learned=report["records"][0]
    assert learned["selected_phase"] == "long_update"
    assert len(learned["trials"])==2
    assert all((tmp_path/r["initial_checkpoint"]).is_file() for r in learned["trials"])
    assert all(r["diagnostics"]["gradients_finite"] for r in learned["trials"])
    metrics=json.loads((tmp_path/"fairness_metrics.json").read_text())["rows"]
    assert all(r["reference_accepted"] and r["spectral_bands"] for r in metrics)
    assert {r["split"] for r in metrics}=={"train","validation"}
    original=f._trial
    def resumed(*args,**kwargs):
        value=original(*args,**kwargs)
        assert value["trial_id"] in {r["trial_id"] for r in learned["trials"]}
        return value
    monkeypatch.setattr(f,"_trial",resumed)
    again=f.run(ctx)
    assert again==result
    assert len(list((tmp_path/"trials").glob("*.json")))==2


def test_confirmation_truth_is_rejected_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(f,"load_bank",lambda *_: [dict(split="confirmation")])
    ctx=SimpleNamespace(protocol=dict(resolve=dict(fairness=settings())),unit=dict(track="discrete",seed=1))
    with pytest.raises(ValueError,match="never read confirmation"):
        f.run(ctx)
