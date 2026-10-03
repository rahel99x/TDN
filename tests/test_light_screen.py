"""Bounded physical screening and conservative evidence-accounting safeguards."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdn.analysis import light_screen as screen
from tdn.config import load_config
from tdn.numerics import Geometry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def light_config():
    return load_config(ROOT / "configs/carc-light.yaml")


@pytest.mark.parametrize("path,value", [
    (("problem", "grid"), [16, 16]),
    (("problem", "t_ref"), 2.),
    (("validation", "require_headroom"), False),
    (("validation", "tolerance"), .003),
    (("training", "max_steps"), 1000),
    (("model", "fixed_rates"), [.2, 1., 10., 100.]),
    (("teacher", "base_substeps"), 128),
    (("runtime", "intraop_threads"), 4),
])
def test_light_scope_rejects_expansion_or_gate_weakening(light_config, path, value):
    changed = copy.deepcopy(light_config)
    changed[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        screen.validate_light_config(changed)


def test_light_scope_rejects_gpu_and_confirmation(light_config):
    with pytest.raises(ValueError, match="CPU-only"):
        screen.validate_light_config(light_config, "cuda")
    light_config["runtime"]["confirmatory_authorized"] = True
    with pytest.raises(ValueError, match="development-only"):
        screen.validate_light_config(light_config)


def test_declared_horizons_overdetermine_four_columns_and_are_disjoint(light_config):
    plan = screen.make_plan(light_config)
    assert len(plan["cases"]) == 6
    assert len(plan["fit_horizons"]) == 8 > plan["temporal_amplitude_columns"]
    assert len(plan["heldout_horizons"]) == 6
    assert not set(plan["fit_horizons"]) & set(plan["heldout_horizons"])
    assert max(plan["heldout_horizons"]) == .4
    assert plan["rollout_time"] == .64
    assert plan["maximum_finest_reference_substeps"] == 324
    assert plan["screen_walltime_budget_seconds"] == 600
    assert plan["plan_hash"] == screen.make_plan(light_config)["plan_hash"]
    assert len({case["initial_state_sha256"] for case in plan["cases"]}) == 2


@pytest.mark.parametrize("name,seed", screen.INITIAL_STATES)
def test_richer_initial_states_are_reproducible_and_physically_bounded(name, seed):
    geometry = Geometry((8, 8), (1., 1.))
    state = screen.initial_state(name, seed, geometry)
    assert state.shape == (1, 1, 8, 8)
    assert state.dtype == torch.float64
    assert float(state.min()) >= .15 - 1e-14
    assert float(state.max()) <= .85 + 1e-14
    assert torch.equal(state, screen.initial_state(name, seed, geometry))
    assert not torch.equal(state, screen.initial_state(name, seed + 1, geometry))


def test_headroom_requires_both_controls_and_valid_reference():
    methods = [{"method": name, "failed": False, "error": .003} for name in screen.METHODS]
    reference = {"accepted": True, "uncertainty": .00001}
    assert screen.headroom_screen(methods, reference)["passed"]
    methods[1]["error"] = .001
    assert not screen.headroom_screen(methods, reference)["passed"]
    methods[1].update(error=None, failed=True)
    failed = screen.headroom_screen(methods, reference)
    assert not failed["passed"] and not failed["classical_methods_completed"]
    methods[1].update(error=.003, failed=False)
    assert not screen.headroom_screen(methods, {"accepted": False, "uncertainty": .00001})["passed"]
    assert not screen.headroom_screen(methods, {"accepted": True, "uncertainty": .002})["passed"]


def oracle_fixture(*, informative=False, fixed_error=.0001, learned_error=.00001):
    def model(error):
        return {"heldout_absolute_rms": [error] * 6,
                "amplitude_fit_overdetermined": True, "amplitude_design_full_rank": True}
    models = {"fixed_rate": model(fixed_error), "polynomial": model(.001),
              "rational": model(.0012), "learned_rate_oracle": model(learned_error)}
    models["learned_rate_oracle"]["informative_rate_fit"] = informative
    references = [{"role": "fit" if index < 8 else "heldout", "accepted": True,
                   "uncertainty": 1e-7, "defect_norm": .01} for index in range(14)]
    return {"models": models}, references


def test_saturated_or_noninformative_learned_fit_cannot_create_a_temporal_pass():
    oracle, references = oracle_fixture(fixed_error=.0011)
    result = screen.temporal_screen(oracle, references)
    assert result["eligible_temporal_models"] == ["fixed_rate"]
    assert not result["passed"]  # Tiny learned error is ineligible.
    oracle["models"]["learned_rate_oracle"]["informative_rate_fit"] = True
    assert screen.temporal_screen(oracle, references)["passed"]
    oracle["models"]["learned_rate_oracle"]["amplitude_fit_overdetermined"] = False
    assert not screen.temporal_screen(oracle, references)["passed"]


def test_temporal_screen_respects_reference_uncertainty_and_failed_teachers():
    oracle, references = oracle_fixture()
    assert screen.temporal_screen(oracle, references)["passed"]
    references[0]["accepted"] = False
    assert not screen.temporal_screen(oracle, references)["passed"]
    references[0]["accepted"] = True
    for row in references:
        row["uncertainty"] = .001
    result = screen.temporal_screen(oracle, references)
    assert not result["passed"]
    assert not result["defects_resolved_above_reference_floor"]


def test_run_snapshots_plan_and_retains_every_failed_case(light_config, tmp_path, monkeypatch):
    target = tmp_path / "screen"
    calls = []
    def fake_case(config, declared, geometry, deadline=None):
        plan = json.loads((target / "plan.json").read_text())
        assert len(plan["cases"]) == 6 and deadline is not None
        calls.append(declared["case_id"])
        return {**declared, "reference_passed": False,
                "numerical_headroom_screen": {"passed": False},
                "temporal_representation_screen": {"passed": False},
                "candidate_requires_full_audit": False, "failure_reason": "deliberately rejected reference"}
    monkeypatch.setattr(screen, "_case", fake_case)
    report = screen.run(light_config, target)
    assert report["status"] == "COMPLETED"
    assert report["completed_cases"] == 6
    assert report["references_passed_cases"] == 0
    assert report["decision"] == "no_joint_candidate"
    assert not report["pilot_authorized"] and not report["confirmatory_authorized"]
    assert len(list((target / "cases").glob("*.json"))) == 6
    assert calls == [case["case_id"] for case in screen.make_plan(light_config)["cases"]]
    with pytest.raises(FileExistsError, match="Preserve"):
        screen.run(light_config, target)


def test_incomplete_or_unexpected_failures_do_not_write_completed_summary(light_config, tmp_path, monkeypatch):
    target = tmp_path / "screen"
    def broken(*args, **kwargs):
        raise RuntimeError("unexpected implementation failure")
    monkeypatch.setattr(screen, "_case", broken)
    with pytest.raises(RuntimeError, match="unexpected implementation"):
        screen.run(light_config, target)
    assert (target / "plan.json").exists()
    assert not (target / "summary.json").exists()


def test_walltime_bound_stops_without_a_partial_success_summary(light_config, tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "MAX_SCREEN_SECONDS", 0)
    target = tmp_path / "screen"
    with pytest.raises(TimeoutError, match="walltime"):
        screen.run(light_config, target)
    assert (target / "plan.json").exists()
    assert not (target / "summary.json").exists()


def test_output_cannot_escape_project(light_config):
    with pytest.raises(ValueError, match="inside the project"):
        screen.run(light_config, ROOT.parent / "light-screen-external")


def test_rejected_reference_prevents_oracle_fit_without_removing_case(light_config, monkeypatch):
    def reference(state, horizon, *args):
        return state, {"h": horizon, "accepted": False, "uncertainty": .01, "defect_norm": .02}
    def rollout(state, steps, *args, **kwargs):
        return state, {"simulated_time": sum(steps)}
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected teachers must never be used for oracle fitting")
    monkeypatch.setattr(screen, "_reference", reference)
    monkeypatch.setattr(screen, "_rollout", rollout)
    monkeypatch.setattr(screen, "temporal_oracle_fits", forbidden)
    case = screen._case(light_config, screen.make_plan(light_config)["cases"][0], Geometry((8, 8), (1., 1.)))
    assert len(case["local_references"]) == 14
    assert len(case["classical_methods"]) == 2
    assert case["temporal_oracle"] is None
    assert not case["reference_passed"] and not case["candidate_requires_full_audit"]
