"""Training contracts tested independently of the expensive physical decoder."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.response import training as tr


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def sample(parent, *, split="train", offset=.01, regime="low_mean", floor=False):
    u = torch.linspace(.2, .7, 16).reshape(1, 1, 4, 4).float()
    target = u.double()+offset*(u.double()-.3)
    if floor:
        target = u.double()+1e-10
    return dict(u0=u, target=target, anchor_fp32=u.clone(), anchor_fp64=u.double(),
        record=dict(parent_id=parent, query_id=parent+"/h", split=split, grid=[4, 4],
            lengths=[1., 1.], horizon=.06, kappa=.01, reaction_rate=3., target_kind="fd_same_grid",
            regime="triads", tiny_regime=regime,
            reference=dict(accepted=True, uncertainty_rms=1e-11, uncertainty_maximum=2e-11)))


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Conv2d(1, 3, 1)
        self.dropout = torch.nn.Dropout(.2)
        self.output_head = torch.nn.Conv2d(3, 1, 1)
        torch.nn.init.zeros_(self.output_head.weight)
        torch.nn.init.zeros_(self.output_head.bias)

    def forward(self, u):
        return u+self.output_head(self.dropout(torch.tanh(self.trunk(u))))


def trial_config(updates=12):
    return dict(id="test", family="test", phase="tiny_joint", target_kind="fd_same_grid", seed=37,
        updates=updates, validation_every=3, total_seconds=1000., batch_size=2,
        setting=dict(lr=3e-4, clip=1., loss_multiplier=1., weight_decay=0., eps=1e-8, betas=[.9, .999]), model={})


def test_training_normalization_uses_only_train_and_preserves_fp64_residuals():
    train = sample("train", floor=True)
    valid = sample("valid", split="validation", offset=1e6)
    values = tr.training_normalization([train, valid])
    assert values == tr.training_normalization([train])
    assert values["floor"] > 1e-10
    row = tr.endpoint_row(train["u0"], train, values)
    assert row["rms"] == pytest.approx(1e-10, rel=1e-5)
    assert row["floor_limited"]
    assert row["objective"] > 0.
    bad = deepcopy(train); bad["target"] = bad["target"].float()
    with pytest.raises(ValueError, match="FP64"):
        tr.training_normalization([bad])


def test_parent_split_overlap_and_confirmation_are_rejected_before_normalization():
    with pytest.raises(ValueError, match="more than one"):
        tr.training_normalization([sample("a"), sample("a", split="validation")])
    with pytest.raises(ValueError, match="confirmation"):
        tr.training_normalization([sample("a"), sample("b", split="confirmation")])


def test_exact_chunk_continuation_restores_optimizer_dropout_rng_and_sampler(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "predict", lambda model, s: model(s["u0"]))
    train = [sample("a", offset=.02), sample("b", offset=.03)]
    validation = [sample("v", split="validation", offset=.025)]
    norm = tr.training_normalization(train+validation)
    config = trial_config()
    identity = dict(source="frozen", config=config, data="frozen")
    torch.manual_seed(98); full = TinyNet()
    full_result = tr.train_trial(full, train, validation, config, norm, tmp_path/"full", identity=identity)
    torch.manual_seed(98); first = TinyNet()
    part = tr.train_trial(first, train, validation, config, norm, tmp_path/"part", identity=identity,
                          max_chunk_updates=5, final_chunk=False)
    assert part["status"] == "CONTINUATION_REQUIRED"
    torch.manual_seed(845); second = TinyNet()
    resumed = tr.train_trial(second, train, validation, config, norm, tmp_path/"continued", identity=identity,
                             resume=tmp_path/"part/journal.pt")
    a = torch.load(tmp_path/"full/journal.pt", weights_only=True)
    b = torch.load(tmp_path/"continued/journal.pt", weights_only=True)
    assert a["update"] == b["update"] == config["updates"]
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert a["rng"]["sampler"] == b["rng"]["sampler"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b["optimizer"]["state"][key][name])
    assert [r.get("parent_ids") for r in a["history"]] == [r.get("parent_ids") for r in b["history"]]
    assert [r.get("train_loss") for r in a["history"]] == [r.get("train_loss") for r in b["history"]]
    assert full_result["selected_state_sha256"] == resumed["selected_state_sha256"]
    assert full_result["trunk_movement"] > 0
    assert full_result["maximum_trunk_gradient"] > 0
    assert len(resumed["attempts"]) == 2
    assert resumed["cost"]["checkpoint_seconds"] > 0
    with pytest.raises(ValueError, match="identity changed"):
        tr.train_trial(TinyNet(), train, validation, config, norm, tmp_path/"wrong", identity={"different":True},
                        resume=tmp_path/"part/journal.pt")


def test_scaled_gradient_and_clipping_are_logged_without_equating_them(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "predict", lambda model, s: model(s["u0"]))
    train = [sample("a")]; valid = [sample("v", split="validation")]
    norm = tr.training_normalization(train)
    config = trial_config(updates=1)
    torch.manual_seed(1)
    row = tr.train_trial(TinyNet(), train, valid, config, norm, tmp_path, identity={"run":1})
    curves = [json.loads(line) for line in (tmp_path/"learning_curves.jsonl").read_text().splitlines()]
    assert curves[-1]["gradient_norm"] > 1
    assert curves[-1]["gradient_norm_after_clip"] <= 1.000001
    assert row["clipped_fraction"] == 1
    assert curves[-1]["trunk_gradient_norm"] == 0  # legitimate only on first zero-head update


def test_factorial_complete_eight_cells_and_validation_only_selection():
    settings = tr.factorial_settings()
    assert len(settings) == 8
    assert len({(s["lr"], s["clip"], s["loss_multiplier"]) for s in settings}) == 8
    rows = [dict(phase="factorial", target_kind="fd_same_grid", family="fno", trial_id=str(index),
        setting=s, status="COMPLETED_UPDATES", requested_updates=400, requested_seconds=120,
        selected_validation=dict(objective=float(index), rms=float(index), maximum=float(index)),
        selected_train=dict(objective=1e9-index)) for index, s in enumerate(settings)]
    result = tr.select_settings(rows)
    assert [r["trial_id"] for r in result["selected_settings"]["fd_same_grid/fno"]] == ["0", "1"]
    assert tr.select_settings(rows[:-1])["selected_settings"]["fd_same_grid/fno"] == []
    rows[-1]["requested_updates"] = 300
    with pytest.raises(ValueError, match="equal declared opportunity"):
        tr.select_settings(rows)


def test_fixed_affine_control_resolves_known_span_and_identifies_rank_loss():
    samples = [sample("a"), sample("b", offset=.01), sample("valid", split="validation", offset=.03)]
    normalization = tr.training_normalization(samples)
    fit = tr.fit_fixed_features(samples, normalization)
    assert fit["affine"]
    assert fit["rank"] < len(fit["coefficients"])  # all samples have one horizon
    assert fit["condition_number"] is None
    assert fit["train_objective"] < 1e-20
    altered = deepcopy(samples); altered[-1]["target"] += 1000
    assert tr.fit_fixed_features(altered, normalization)["coefficients"] == fit["coefficients"]


def test_m3_does_not_count_floor_or_censored_cells_as_resolved_success():
    rows = []
    for seed in (1, 2, 3):
        for regime in tr.CORE_REGIMES:
            metric = dict(mse=1e-10, rows=[dict(floor_limited=False, conservative_rms=1e-5,
                conservative_maximum=1e-5, reference_accepted=True)])
            rows.append(dict(phase="tiny_individual", family="x", target_kind="fd_same_grid", seed=seed,
                regime=regime, trial_id=f"{seed}/{regime}", initial_train=dict(mse=1e-6), selected_train=metric,
                status="COMPLETED_UPDATES", selected_initialization=False))
        initial_rows = [dict(parent_id=r, tiny_regime=r, mse=1e-6) for r in tr.CORE_REGIMES]
        final_rows = [dict(parent_id=r, tiny_regime=r, mse=1e-10, floor_limited=False,
            conservative_rms=1e-5, conservative_maximum=1e-5, reference_accepted=True) for r in tr.CORE_REGIMES]
        rows.append(dict(phase="tiny_joint", family="x", target_kind="fd_same_grid", seed=seed,
            trial_id=f"joint/{seed}", initial_train=dict(rows=initial_rows), selected_train=dict(rows=final_rows),
            status="COMPLETED_UPDATES"))
    assert tr.trainability_gate(rows, seeds=[1, 2, 3])["gate_passed"]
    assert not tr.trainability_gate(rows, seeds=[1, 2, 3], profile="smoke")["gate_passed"]
    assert not tr.trainability_gate(rows, seeds=[1, 2, 3], expected_groups=[("fd_same_grid", "missing")])["gate_passed"]
    rows[0]["selected_train"]["rows"][0]["floor_limited"] = True
    rows[1]["status"] = "TIME_CENSORED"
    gate = tr.trainability_gate(rows, seeds=[1, 2, 3])
    assert not gate["gate_passed"]
    assert gate["groups"][0]["status"] == "UNRESOLVED"


def test_sampling_subset_selects_tiny_regime_before_one_parent_count(tmp_path, monkeypatch):
    from tdn.analysis.response import references
    samples = [sample("a", regime="low_mean"), sample("b", regime="strong_reaction"), sample("v", split="validation")]
    monkeypatch.setattr(references, "load_samples", lambda *args: samples)
    ctx = SimpleNamespace(unit={"data_stages":["prepare"]}, prerequisites={"prepare":tmp_path}, device="cpu")
    selected, _ = tr._data(ctx, dict(target_kind="fd_same_grid", phase="tiny_individual", regime="strong_reaction", data_count=1))
    assert [s["record"]["parent_id"] for s in selected if s["record"]["split"] == "train"] == ["b"]


def test_phase_twins_are_distinct_queries_with_one_parent_statistical_weight(tmp_path, monkeypatch):
    from tdn.analysis.response import references
    first = sample("a")
    twin = deepcopy(first)
    first["record"].update(phase_variant="original", descendant_sha256="original-hash")
    twin["record"].update(phase_variant="phase_twin", descendant_sha256="twin-hash")
    samples = [first, twin, sample("v", split="validation")]
    monkeypatch.setattr(references, "load_samples", lambda *args:samples)
    ctx = SimpleNamespace(unit={"data_stages":["prepare"]}, prerequisites={"prepare":tmp_path}, device="cpu")
    chosen, _ = tr._data(ctx, dict(target_kind="fd_same_grid", phase="main", data_count=1))
    assert len([s for s in chosen if s["record"]["split"] == "train"]) == 2
    assert tr.training_normalization(chosen)["train_parent_ids"] == ["a"]
    samples.append(deepcopy(first))
    with pytest.raises(ValueError, match="Duplicated"):
        tr._data(ctx, dict(target_kind="fd_same_grid", phase="main", data_count=1))


def test_actual_frozen_neural_encoder_solve_and_fp32_assembly_are_distinguished():
    from tdn.analysis.response.models import create_model
    from tdn.analysis.response.references import equation_geometry
    samples = [sample("a"), sample("b"), sample("valid", split="validation")]
    samples[1]["record"]["horizon"] = .025
    norm = dict(mean_scale=.01, centered_scale=.01, full_scale=.1, full_mean=.4, floor=1e-7)
    torch.manual_seed(25)
    model = create_model(dict(family="residual_fno", width=3, depth=1, modes=1, frozen_encoder=True, normalization=norm))
    coefficients = torch.tensor([.1, -.2, .05, .2, .1, -.1, .03], dtype=torch.float64)
    for item in samples:
        eq, geom = equation_geometry(item["record"])
        with torch.no_grad():
            base, basis = model.affine_features(item["u0"], item["record"]["horizon"], eq, geom, "fd_same_grid")
        item["target"] = base.double()+torch.einsum("k,k...->...", coefficients, basis.double())
    initial_encoder = deepcopy(model.encoder.state_dict())
    fit = tr.fit_frozen_encoder(model, samples, norm)
    assert fit["fp64_affine_train_objective"] < 1e-18
    assert max(row["rms"] for row in fit["rows"]) < 1e-7
    assert all(torch.equal(initial_encoder[key], value) for key, value in model.encoder.state_dict().items())
    altered = deepcopy(samples); altered[-1]["target"] += 1
    assert tr.fit_frozen_encoder(model, altered, norm)["coefficients"] == fit["coefficients"]


def test_native_model_checkpoint_reloads_in_a_fresh_process_without_teacher_handles(tmp_path):
    from tdn.analysis.response.models import create_model
    train = [sample("a")]; validation = [sample("v", split="validation")]
    norm = tr.training_normalization(train)
    model_config = dict(family="residual_fno", width=3, depth=1, modes=1, normalization=norm)
    model = create_model(model_config)
    config = {**trial_config(updates=2), "family":"residual_fno", "model":model_config}
    tr.train_trial(model, train, validation, config, norm, tmp_path, identity={"test":"reload", "torch":torch.__version__})
    result = tr.checkpoint_reload_probe(tmp_path, train[0])
    assert result["independent_process"] and result["exact_equal"]
    probe = torch.load(tmp_path/"reload-input.pt", weights_only=True)
    assert set(probe) == {"u0", "record"}
    assert "reference" not in probe["record"]


def test_source_and_consistency_auxiliaries_backpropagate_and_require_declared_targets():
    from tdn.analysis.response.models import create_model
    item = sample("a")
    norm = tr.training_normalization([item])
    model = create_model(dict(family="coupled_source", width=3, depth=1, modes=1, rank=2, normalization=norm))
    with pytest.raises(ValueError, match="explicitly prepared"):
        tr._source_loss(model, item, norm)
    item["source_times"] = torch.tensor([.02, .04], dtype=torch.float64)
    source = item["u0"].double()-item["u0"].double().mean()
    item["source_targets"] = torch.stack([source*.001, source*.002])
    item["record"]["source_supervision"] = {"accepted":True}
    source_loss = tr._source_loss(model, item, norm)
    source_loss.backward()
    assert model.source_head.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    consistency = tr._consistency_loss(model, item, norm)
    consistency.backward()
    assert model.mean_head.weight.grad.abs().sum() > 0
    item["record"]["target_kind"] = "galerkin_projected_fine"
    with pytest.raises(ValueError, match="closure"):
        tr._consistency_loss(model, item, norm)


def test_short_rollout_gradient_includes_actual_intermediate_state():
    class Growth(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.tensor(.3))
        def forward(self, u, h, *_):
            return u*(1+self.p*h)
    item = sample("a")
    model = Growth(); norm = dict(mean_scale=1., centered_scale=1.)
    value = tr._rollout_loss(model, item, norm, 4)
    gradient = torch.autograd.grad(value, model.p)[0]
    direct = item["u0"]*(1+model.p*item["record"]["horizon"]/4)**4
    expected = torch.autograd.grad(tr.physical_loss(direct, item["target"], norm), model.p)[0]
    assert gradient == pytest.approx(float(expected), rel=2e-5)


def test_clock_consistency_penalty_uses_reconstructed_variance_and_has_gradient():
    from tdn.analysis.response.models import create_model
    from tdn.analysis.response.references import equation_geometry
    item=sample("clock"); norm=tr.training_normalization([item])
    network=create_model(dict(family="coupled_source",width=3,depth=1,modes=1,rank=2,
        mean_parameterization="clock",normalization=norm))
    with torch.no_grad():
        network.mean_head.bias.fill_(.3)
        network.source_head.bias.fill_(.4)
    loss=tr._consistency_loss(network,item,norm)
    eq,geom=equation_geometry(item["record"])
    t=torch.tensor(item["record"]["horizon"],requires_grad=True)
    prediction=network(item["u0"],t,eq,geom,"fd_same_grid")
    mean=prediction.mean()
    derivative=torch.autograd.grad(mean,t,create_graph=True)[0]
    variance=((prediction-mean)**2).mean()
    expected=((derivative-eq.reaction_rate*(mean*(1-mean)-variance))/(norm["mean_scale"]/network.t_ref))**2
    assert float(loss.detach()) == pytest.approx(float(expected.detach()),rel=2e-5)
    loss.backward()
    assert network.mean_head.weight.grad.abs().sum() > 0
    assert network.source_head.weight.grad.abs().sum() > 0


def test_galerkin_objective_uses_physical_nyquist_mass_and_retains_nodal_error():
    item = sample("a")
    item["record"]["target_kind"] = "galerkin_same_grid"
    alternating = (-1.)**torch.arange(4, dtype=torch.float64)
    error = (alternating[:, None]*alternating[None, :])[None, None]
    item["target"] = torch.zeros_like(error)
    norm = dict(mean_scale=1., centered_scale=1., floor=1e-8)
    row = tr.endpoint_row(error, item, norm)
    assert row["sampled_rms"] == 1
    assert row["rms"] == pytest.approx(.5)
    assert row["maximum"] == 1
    assert row["mse"] == pytest.approx(row["mean_error"]**2+row["centered_rms"]**2)
    assert tr.physical_loss(error, item["target"], norm, item["record"]) == pytest.approx(.125)
    coordinates = tr._physical_coordinates(error, item["record"])
    assert coordinates.square().sum() == pytest.approx(.25)


def test_main_selector_does_not_erase_entirely_missing_declared_variants(tmp_path):
    completed = tmp_path/"completed"; completed.mkdir()
    row = dict(phase="main", target_kind="fd_same_grid", family="x", variant_id="base", data_count=32,
        seed=1, trial_id="a", status="COMPLETED_UPDATES", updates=10, attempts=[], setting={},
        selected_validation={"objective":1.}, selected_checkpoint_sha256="hash", selected_initialization=False)
    (completed/"trial.json").write_text(json.dumps(row))
    output = tmp_path/"select"; output.mkdir()
    specs = {key:dict(id=key, phase="main", target_kind="fd_same_grid", family="x", variant_id=variant,
        data_count=32, seed=1) for key, variant in (("a", "base"), ("b", "coupling"))}
    ctx = SimpleNamespace(unit={"selection_phase":"main"}, protocol={"trials":specs, "mechanisms":{"M3":{}}},
        prerequisites={"a":completed}, path=output, identities=set(), record=lambda *a,**k:None)
    selection = tr.run_select(ctx)
    assert selection["expected_group_count"] == 2
    assert selection["selected_models"][0]["method_id"] == "x::base::n32"
    assert selection["missing"][0]["variant_id"] == "coupling"


def test_matched_fno_source_supervision_has_live_shared_encoder_gradients():
    from tdn.analysis.response.models import create_model
    item = sample("a"); norm = tr.training_normalization([item])
    model = create_model(dict(family="residual_fno", width=3, depth=1, modes=1, auxiliary_source_head=True, normalization=norm))
    item["source_times"] = torch.tensor([.02, .04], dtype=torch.float64)
    v = item["u0"].double()-item["u0"].double().mean()
    item["source_targets"] = torch.stack([v*.002, v*.003])
    item["record"]["source_supervision"] = {"accepted":True}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(); loss = tr._source_loss(model, item, norm); loss.backward(); optimizer.step()
    assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None) > 0


def test_source_label_path_identity_is_required_even_for_default_anchor():
    item = sample("a")
    config = dict(source_aux_weight=1., model={"anchor":"df", "reaction_substeps":8})
    with pytest.raises(ValueError, match="reaction_substeps identity"):
        tr._anchor_labels([item], config)
    item["record"]["source_supervision"] = dict(anchor="df", reaction_substeps=8)
    assert tr._anchor_labels([item], config)[0] is item
    with pytest.raises(ValueError, match="reaction_substeps identity"):
        tr._anchor_labels([item], {**config, "model":{"anchor":"df", "reaction_substeps":2}})
    with pytest.raises(ValueError, match="reaction_substeps identity"):
        tr._anchor_labels([item], {**config, "model":{"anchor":"shared_gl4", "reaction_substeps":8}})


def test_common_scale_ablation_retains_training_estimator_and_floor():
    original = dict(mean_scale=.01, centered_scale=.1, full_mean=.3, full_scale=.2, floor=1e-7)
    common = tr.apply_normalization_mode(original, "common")
    assert common["mean_scale"] == common["centered_scale"] == .1
    assert common["unmodified_channel_scales"] == {"mean_scale":.01, "centered_scale":.1}
    assert common["floor"] == original["floor"]
    assert original["mean_scale"] == .01
    assert tr.apply_normalization_mode(original)["mean_scale"] == .01


def test_direct_full_target_scale_includes_between_parent_means_without_validation_leakage():
    low, high = sample("low"), sample("high")
    low["target"] = torch.full_like(low["target"], .1)
    high["target"] = torch.full_like(high["target"], .9)
    valid = sample("validation", split="validation")
    valid["target"] = torch.full_like(valid["target"], 1000.)
    norm = tr.training_normalization([low, high, valid])
    assert norm["full_mean"] == pytest.approx(.5)
    assert norm["full_scale"] == pytest.approx(.4)
    assert norm["within_field_centered_scale"] == norm["floor"]
    assert norm == tr.training_normalization([low, high])


def test_direct_fno_default_loss_avoids_residual_denominators_but_shared_ablation_remains():
    norms = dict(mean_scale=1e-7, centered_scale=2e-7, full_scale=.2, full_mean=.5, floor=1e-8)
    direct = tr.configure_objective(norms, "direct_fno")
    residual = tr.configure_objective(norms, "residual_fno")
    shared = tr.configure_objective(norms, "direct_fno", "shared_physical")
    assert direct["objective_mode"] == "full_target"
    assert direct["objective_mean_scale"] == direct["objective_centered_scale"] == .2
    assert direct["mean_scale"] == 1e-7  # Source/physical residual attribution scales retained.
    error = torch.full((1, 1, 4, 4), .1, dtype=torch.float64)
    target = torch.zeros_like(error)
    assert tr.physical_loss(error, target, direct) == pytest.approx(.125)
    assert tr.physical_loss(error, target, shared) == tr.physical_loss(error, target, residual)
    assert tr.physical_loss(error, target, direct) < tr.physical_loss(error, target, shared)*1e-10
