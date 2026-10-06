"""Finite-field deployment, rollout costs, and retained adverse outcomes."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tdn.analysis.interactions import experiment
from tdn.analysis.interactions.protocol import DEFAULTS, validate_config
from tdn.numerics.types import Equation, Geometry


def test_declared_states_valid_and_include_small_two_d():
    cases, rollouts = experiment.cases()
    assert {tuple(case["grid"]) for case in cases} == {(16,), (32,), (8, 8)}
    for case in cases + rollouts:
        u = experiment.state(case)
        assert u.device.type == "cpu" and u.dtype == torch.float64
        assert bool(((u >= 0) & (u <= 1)).all())


@pytest.mark.parametrize("variant", experiment.VARIANTS)
def test_complete_method_work_accumulates_over_rollout(variant):
    spec = experiment._spec("cost", h=.01)
    u = experiment.state(spec)
    geometry, equation = Geometry((16,), (1.,)), Equation(.03, 2.)
    single = {}
    experiment.method_step(u, .01, equation, geometry, variant, work=single)
    total = {}
    for _ in range(3):
        u = experiment.method_step(u, .01, equation, geometry, variant, work=total)
    for key in experiment.WORK_KEYS:
        assert total.get(key, 0) == 3 * single.get(key, 0), (variant, key, total, single)
    if variant in ("scalar_cubic", "output_phi"):
        assert single["fft_forward"] >= 1 and single["fft_inverse"] >= 1
        assert single["reaction_evaluations"] == 2
        assert single["laplacian_evaluations"] == 5


def test_real_reference_and_method_metrics():
    rows = []
    spec = experiment._spec("test", h=.03, steps=2)
    config = validate_config(deepcopy(DEFAULTS), smoke=True)
    experiment._case(rows, spec, "rollout", config, lambda: None)
    assert len(rows) == len(experiment.VARIANTS) + 1
    assert rows[0]["metrics"]["reference_accepted"]
    assert rows[0]["metrics"]["reference_4n"] == 4 * rows[0]["metrics"]["reference_n"]
    for row in rows[1:]:
        assert row["metrics"]["trajectory_completed"]
        assert row["metrics"]["completed_steps"] == 2
        assert row["metrics"]["complete_rollout_seconds"] >= row["metrics"]["first_step_seconds"] > 0
        assert row["metrics"]["error_l2"] is not None


def test_invalid_rollout_is_retained_without_false_final_time_error(monkeypatch):
    real_step = experiment.method_step

    def invalid(u, h, equation, geometry, variant, *, work):
        if variant == "scalar_cubic":
            work["fft_forward"] = work.get("fft_forward", 0) + 1
            return u + 2
        return real_step(u, h, equation, geometry, variant, work=work)

    monkeypatch.setattr(experiment, "method_step", invalid)
    rows = []
    experiment._case(rows, experiment._spec("invalid", h=.01, steps=3), "rollout",
                     validate_config(deepcopy(DEFAULTS), smoke=True), lambda: None)
    row = next(row for row in rows if row["variant"] == "scalar_cubic")
    assert row["outcome"] == "EXPECTED_LIMITATION"
    assert row["metrics"]["completed_steps"] == 1 and not row["metrics"]["trajectory_completed"]
    assert row["metrics"]["error_l2"] is None and row["metrics"]["mean_error"] is None
    assert row["metrics"]["work_fft_forward"] == 1


def test_unaccepted_teacher_retains_every_method(monkeypatch):
    real_teacher = experiment._teacher

    def rejected(*args):
        result, metrics = real_teacher(*args)
        return replace(result, accepted=False, reason="test_unresolved"), {**metrics, "reference_accepted": False}

    monkeypatch.setattr(experiment, "_teacher", rejected)
    rows = []
    experiment._case(rows, experiment._spec("rejected", h=.01), "one_step",
                     validate_config(deepcopy(DEFAULTS), smoke=True), lambda: None)
    assert len(rows) == len(experiment.VARIANTS) + 1
    assert {row["outcome"] for row in rows} == {"INCONCLUSIVE"}
    assert not any(row["metrics"].get("improves_strang_resolved") for row in rows)


def test_unexpected_method_failure_retained_with_time_and_unknown_partial_work(monkeypatch):
    real_step = experiment.method_step

    def broken(u, h, equation, geometry, variant, *, work):
        if variant == "scalar_cubic":
            raise RuntimeError("deliberate failure after partial computation")
        return real_step(u, h, equation, geometry, variant, work=work)

    monkeypatch.setattr(experiment, "method_step", broken)
    rows = []
    experiment._case(rows, experiment._spec("failure", h=.01), "one_step",
                     validate_config(deepcopy(DEFAULTS), smoke=True), lambda: None)
    row = next(row for row in rows if row["variant"] == "scalar_cubic")
    assert row["outcome"] == "FAIL"
    assert row["metrics"]["complete_rollout_seconds"] > 0
    assert not row["metrics"]["work_counts_complete"]
    assert row["metrics"]["output_mean"] is None and row["metrics"]["minimum"] is None
    assert row["metrics"]["error_l2"] is None
    assert len(rows) == len(experiment.VARIANTS) + 1
