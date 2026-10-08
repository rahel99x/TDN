"""Real CPU policy execution and gates; none of these tests is CUDA evidence."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.frontier import policy
from tdn.analysis.frontier.data import field_state, lawson_reference
from tdn.analysis.frontier.models import make_model
from tdn.analysis.frontier.protocol import build_protocol
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import horizon_key


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Budget:
    def check(self):
        pass


def state(n=8):
    x = torch.arange(n, dtype=torch.float32)/n
    return (.4+.04*torch.cos(2*torch.pi*x[:, None])+.03*torch.sin(2*torch.pi*x[None, :]))[None, None]


@pytest.mark.parametrize("track", ["discrete", "continuum"])
@pytest.mark.parametrize("spatial", [False, True])
def test_real_proposal_and_fallback_charge_all_work_without_truth(track, spatial):
    model = make_model("rank1", track, {"modes": 1}).float().eval()
    classical = make_model("etdrk4", track, {}).float().eval()
    initial, geom, eq = state(), Geometry((8, 8), (1., 1.)), Equation(.004, 2.)
    value, info = policy.deploy(model, classical, initial, .04, eq, geom,
        envelope={"rms": None, "max": None}, target=2e-5, spatial=spatial,
        sampler=state, max_refinements=3, budget=Budget(), device="cpu")
    assert value.shape == initial.shape and torch.isfinite(value).all()
    assert info["fallback"] and not info["accepted"]
    assert not info["decision_uses_reference"] and not info["deterministic_certificate"]
    assert all(info[key] > 0 for key in ("proposal_seconds", "estimator_seconds", "fallback_seconds"))
    assert info["attributed_seconds"] == sum(info[key] for key in ("proposal_seconds", "estimator_seconds", "fallback_seconds"))
    assert info["fallback_info"]["steps"] >= 3
    assert not info["clipping"]


def test_real_acceptance_uses_calibrated_indicator_not_truth():
    model = make_model("rank1", "discrete", {"modes": 1}).float().eval()
    classical = make_model("etdrk4", "discrete", {}).float().eval()
    value, info = policy.deploy(model, classical, state(), .02, Equation(.004, 2.), Geometry((8, 8), (1., 1.)),
        envelope={"rms": 1., "max": 1.}, target=1e-2, spatial=False,
        sampler=state, max_refinements=3, budget=Budget(), device="cpu")
    assert info["accepted"] and not info["fallback"]
    assert info["fallback_seconds"] == 0 and info["fallback_info"] is None
    assert torch.isfinite(value).all()


def test_invalid_physical_candidate_is_rejected_without_clipping():
    class BadModel(torch.nn.Module):
        family = "bad_test_control"
        def forward(self, u, *args):
            return torch.ones_like(u)*1.2
    classical = make_model("etdrk4", "discrete", {}).float().eval()
    value, info = policy.deploy(BadModel(), classical, state(), .02, Equation(.004, 2.), Geometry((8, 8), (1., 1.)),
        envelope={"rms": 0., "max": 0.}, target=1e-2, spatial=False,
        sampler=state, max_refinements=2, budget=Budget(), device="cpu")
    assert not info["physical_interval_admissible"] and info["fallback"]
    assert float(value.max()) < 1.


@pytest.mark.parametrize("indicator,envelope", [({"rms": math.nan, "max": 0.}, {"rms": 1., "max": 1.}),
    ({"rms": 1e-3, "max": 1e-3}, {"rms": None, "max": 1.}),
    ({"rms": -1., "max": 0.}, {"rms": 1., "max": 1.})])
def test_acceptance_never_turns_invalid_evidence_into_permission(indicator, envelope):
    assert not policy.can_accept(indicator, envelope, 1e-2)


def test_empirical_calibration_cannot_consume_policy_or_confirmation_fields():
    for split in ("policy", "confirmation", None):
        with pytest.raises(ValueError, match="calibration fields"):
            policy._envelope([{"field_cluster": "one", "split": split, "ratio_rms": 1., "ratio_max": 2.}], 2., .01)
    fitted = policy._envelope([{"field_cluster": "one", "split": "calibration", "ratio_rms": 1., "ratio_max": 2.}], 2., .01)
    assert fitted["empirical"] == {"rms": 2., "max": 4.}
    assert all(v["conformal"]["quantile"] is None for v in fitted["diagnostics"].values())


class Context:
    def __init__(self, root, tracks=("discrete", "continuum"), *, headroom=True, eligible=True):
        self.path = root/"policy"; self.path.mkdir()
        self.protocol = build_protocol("smoke")
        self.protocol["tracks"] = list(tracks)
        self.device, self.stage, self.budget, self.rows = "cpu", "policy", Budget(), []
        self.prerequisites = {}
        for stage in ("screen", "prepare", "train", "confirm_prepare", "confirm"):
            path = root/stage; path.mkdir()
            self.prerequisites[stage] = path
            (path/"summary.json").write_text(json.dumps({"elapsed_seconds": 1.}))
        n = self.protocol["policies"]["grid"]
        (self.prerequisites["screen"]/"screen.json").write_text(json.dumps({
            "selection_split": "development", "target": self.protocol["policies"]["target"],
            "conditions_detail": [{"track": t, "grid": n, "outcome": "headroom" if headroom else "easy"} for t in tracks]}))
        self.selected = {track: f"{track}/rank1" for track in tracks}
        (self.prerequisites["confirm"]/"gate.json").write_text(json.dumps({"tracks": {
            t: {"solver_utility_eligible": eligible, "selected_policy_model_id": m} for t, m in self.selected.items()}}))
        (self.prerequisites["train"]/"catalog.json").write_text(json.dumps({"selected_policy_models": self.selected}))
    def record(self, experiment_id, mechanisms, **row):
        self.rows.append({"experiment_id": experiment_id, **row})


def setup_real_bank(ctx, monkeypatch):
    from tdn.analysis.frontier import neural, data
    n, h = ctx.protocol["policies"]["grid"], ctx.protocol["policies"]["horizon"]
    banks = {}
    for split in ("calibration", "policy"):
        parents = [copy.deepcopy(p) for p in ctx.protocol["parents"] if p["split"] == split]
        for parent in parents:
            initial = field_state(parent, n)
            parent["states"] = {str(n): initial}
            parent["references"] = {}
            for track in ctx.protocol["tracks"]:
                eq, geom = neural.eq_geom(parent, n)
                truth = lawson_reference(initial.double(), h, eq, geom, 64, track)
                parent["references"][f"{track}:{n}:{horizon_key(h)}"] = dict(
                    state=truth, accepted=True, uncertainty_rms=1e-9, uncertainty_max_bound=1e-9)
        banks[split] = parents
    models = {}
    for track, identifier in ctx.selected.items():
        model = make_model("rank1", track, ctx.protocol["model_config"]).float().eval()
        model.selection_metadata = {"seed": 3600011, "updates_selected": 1}
        models[identifier] = model
    monkeypatch.setattr(neural, "load_models", lambda c: models)
    monkeypatch.setattr(data, "load_split", lambda c, split: banks[split])
    return banks, models


@pytest.mark.parametrize("headroom,eligible", [(False, True), (True, False), (False, False)])
def test_blocked_gate_never_loads_models_or_policy_truth(tmp_path, monkeypatch, headroom, eligible):
    from tdn.analysis.frontier import neural, data
    ctx = Context(tmp_path, headroom=headroom, eligible=eligible)
    def forbidden(*args):
        raise AssertionError("Blocked gate must not read models or reference data")
    monkeypatch.setattr(neural, "load_models", forbidden)
    monkeypatch.setattr(data, "load_split", forbidden)
    answer = policy.run(ctx)
    assert answer["status"] == "BLOCKED" and not answer["deployment_executed"]
    assert policy.validate_policy_artifacts(ctx.path)["endpoints"] == 0
    assert all(row["status"] == "BLOCKED" for row in ctx.rows)


def test_gate_model_cannot_change_after_confirmation_before_any_fresh_data_access(tmp_path, monkeypatch):
    from tdn.analysis.frontier import data
    ctx = Context(tmp_path, tracks=("discrete",))
    setup_real_bank(ctx, monkeypatch)
    (ctx.prerequisites["train"]/"catalog.json").write_text(json.dumps({"selected_policy_models": {"discrete": "another"}}))
    monkeypatch.setattr(data, "load_split", lambda *args: pytest.fail("Read fresh data before detecting changed model selection"))
    with pytest.raises(ValueError, match="fixed and available"):
        policy.run(ctx)


def test_calibration_and_policy_independent_fields_must_be_disjoint(tmp_path, monkeypatch):
    ctx = Context(tmp_path, tracks=("discrete",))
    banks, _ = setup_real_bank(ctx, monkeypatch)
    banks["policy"][0]["field_cluster"] = banks["calibration"][0]["field_cluster"]
    with pytest.raises(ValueError, match="contamination"):
        policy.run(ctx)


def test_complete_real_cpu_policy_has_frozen_track_calibrations_and_full_costs(tmp_path, monkeypatch):
    ctx = Context(tmp_path)
    setup_real_bank(ctx, monkeypatch)
    answer = policy.run(ctx)
    assert answer["endpoints"] == 8  # two fields x two modes x two target tracks
    assert policy.validate_policy_artifacts(ctx.path)["endpoints"] == 8
    rows = json.loads((ctx.path/"policy_rows.json").read_text())["rows"]
    assert {r["calibration_file"] for r in rows} == {"calibration-discrete.json", "calibration-continuum.json"}
    for row in rows:
        assert len(row["cost"]["samples_seconds"]) == ctx.protocol["timing_repeats"]
        assert row["attributed_seconds"] <= row["component_sample_seconds"] * 1.001
        assert not row["decision_uses_reference"]
    summaries = json.loads((ctx.path/"policy_summary.json").read_text())["rows"]
    assert all(r["risk"]["formal_conditional_risk_status"] == "NA" for r in summaries)
    assert all(r["amortization"]["offline_components_seconds"]["model_loading"] > 0 for r in summaries)
    assert all(r["amortization"]["break_even_queries"] is None for r in summaries if not r["acceptance_fraction"])
    assert all(c["verdict"] == "NA" for row in ctx.rows for c in row["checks"]
               if c["check_id"] == "formal-conditional-risk-guarantee")


def test_track_gate_isolation_does_not_borrow_other_tracks_headroom(tmp_path, monkeypatch):
    ctx = Context(tmp_path)
    setup_real_bank(ctx, monkeypatch)
    screen = json.loads((ctx.prerequisites["screen"]/"screen.json").read_text())
    screen["conditions_detail"][1]["outcome"] = "easy"
    (ctx.prerequisites["screen"]/"screen.json").write_text(json.dumps(screen))
    original = policy.proposal
    def guarded(model, *args, **kw):
        assert model.track == "discrete"
        return original(model, *args, **kw)
    monkeypatch.setattr(policy, "proposal", guarded)
    answer = policy.run(ctx)
    assert answer["endpoints"] == 4
    assert next(t for t in answer["tracks"] if t["track"] == "continuum")["status"] == "BLOCKED"


def test_false_acceptance_is_preserved_in_actual_policy_logs(tmp_path, monkeypatch):
    ctx = Context(tmp_path, tracks=("discrete",))
    banks, _ = setup_real_bank(ctx, monkeypatch)
    # Deliberately wrong fresh truth tests bookkeeping, not solver accuracy.
    for parent in banks["policy"]:
        for ref in parent["references"].values():
            ref["state"] = ref["state"] + .2
    original = policy._envelope
    def force_accept(rows, safety, alpha):
        fitted = original(rows, safety, alpha)
        fitted["empirical"] = {"rms": 0., "max": 0.}
        return fitted
    monkeypatch.setattr(policy, "_envelope", force_accept)
    result = policy.run(ctx)
    assert result["false_accepts"] == 4
    assert all(r["accuracy_status"] == "TARGET_EXCEEDED" and r["false_accept"]
               for r in json.loads((ctx.path/"policy_rows.json").read_text())["rows"])


def test_missing_offline_cost_stays_unknown(tmp_path, monkeypatch):
    ctx = Context(tmp_path, tracks=("discrete",))
    setup_real_bank(ctx, monkeypatch)
    (ctx.prerequisites["prepare"]/"summary.json").write_text("{}")
    policy.run(ctx)
    rows = json.loads((ctx.path/"policy_summary.json").read_text())["rows"]
    assert all(r["amortization"]["offline_components_seconds"]["prepare"] is None
               and r["amortization"]["offline_total_seconds"] is None for r in rows)


def test_validator_detects_rewritten_calibration(tmp_path, monkeypatch):
    ctx = Context(tmp_path, tracks=("discrete",))
    setup_real_bank(ctx, monkeypatch)
    policy.run(ctx)
    (ctx.path/"calibration-discrete.json").write_text("{}")
    with pytest.raises(ValueError, match="calibration differs"):
        policy.validate_policy_artifacts(ctx.path)
