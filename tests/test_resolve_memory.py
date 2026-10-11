"""Causality, paired perturbations, and held-out selection in protected RX."""
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tdn.analysis.resolve import memory as m


def test_forecast_cannot_read_future_reference(monkeypatch):
    previous = m.old.field(96500001, 2)
    current = m.old.rk4(previous, .02)
    def forbidden(*args, **kwargs):
        raise AssertionError("Deployment tried to obtain a teacher")
    monkeypatch.setattr(m.old, "integrate", forbidden)
    first, _ = m.forecast(previous, current, h=.02, diffusion=.02, reaction=.7,
                          steps=4, method="plain_secant")
    second, _ = m.forecast(previous, current, h=.02, diffusion=.02, reaction=.7,
                           steps=4, method="plain_secant")
    np.testing.assert_array_equal(first, second)
    assert not {"truth", "reference", "teacher"} & set(inspect.signature(m.forecast).parameters)


def test_noise_is_paired_scaled_and_preserves_reality():
    previous, current = m.old.field(96500002, 2), m.old.field(96500003, 2)
    low = m.noisy_observations(previous, current, 1e-4, 24)
    high = m.noisy_observations(previous, current, 1e-3, 24)
    for original, a, b in zip((previous, current), low, high):
        np.testing.assert_allclose(b-original, 10*(a-original), atol=6e-16)
        np.testing.assert_array_equal(a, a[::-1].conj())
    assert not np.array_equal(low[0]-previous, low[1]-current)
    zero = m.noisy_observations(previous, current, 0., 24)
    np.testing.assert_array_equal(zero[0], previous)
    with pytest.raises(ValueError, match="nonnegative"):
        m.noisy_observations(previous, current, -1., 24)


def test_damping_contains_plain_secant_and_rk4_limits():
    previous, current = m.old.field(96500004, 2), m.old.field(96500005, 2)
    kwargs = dict(h=.02, diffusion=.02, reaction=.7, steps=5)
    plain, _ = m.forecast(previous, current, method="plain_secant", **kwargs)
    undamped, _ = m.forecast(previous, current, method="validated_damped_secant", gain=1., decay=0., **kwargs)
    np.testing.assert_array_equal(plain, undamped)
    coarse, _ = m.forecast(previous, current, method="coarse_rk4", **kwargs)
    zero, _ = m.forecast(previous, current, method="validated_damped_secant", gain=0., decay=4., **kwargs)
    np.testing.assert_array_equal(coarse, zero)
    with pytest.raises(ValueError, match="at most one"):
        m.forecast(previous, current, method="validated_damped_secant", gain=2., **kwargs)


def test_validation_selection_can_reject_memory_without_looking_at_test():
    previous, current = m.old.field(96500006, 2), m.old.field(96500007, 2)
    common = dict(h=.02, diffusion=.02, reaction=.7, steps=3)
    truth, _ = m.forecast(previous, current, method="coarse_rk4", **common)
    item = dict(split="validation", parent_id=96501001, previous=previous,
                current=current, truth=truth, coarse_trajectory_rms=0.)
    chosen, candidates = m.select_damping([item], [0., .5, 1.], [0., 1.], **common)
    assert chosen == {"gain": 0., "decay": 0.}
    assert len(candidates) == 6
    with pytest.raises(ValueError, match="validation parents"):
        m.select_damping([dict(item, split="development_test")], [0., 1.], [0.], **common)


class Budget:
    def check(self):
        pass


def test_memory_smoke_keeps_noise_shifts_and_freezes_before_test(tmp_path, monkeypatch):
    settings = dict(seed=96600000, train_parents=3, validation_parents=2, test_parents=2,
        rollout_steps=3, damp_gains=[0., .5, 1.], decay_rates=[0., 1.], ridges=[1e-6, .01])
    recorded = []
    ctx = SimpleNamespace(path=tmp_path, device="cpu", budget=Budget(),
        protocol={"profile": "smoke", "resolve": {"memory": settings}},
        record=lambda *args, **kwargs: recorded.append((args, kwargs)))
    original = m._teacher
    def checked(parent, *args):
        if parent >= settings["seed"]+2000:
            assert (tmp_path/"memory_selection.json").is_file()
        return original(parent, *args)
    monkeypatch.setattr(m, "_teacher", checked)
    result = m.run(ctx)
    assert result["status"] == "COMPLETED"
    raw = json.loads((tmp_path/"memory_results.json").read_text())
    assert set(raw["parent_ids"]["train"]).isdisjoint(raw["parent_ids"]["validation"])
    assert set(raw["parent_ids"]["train"]).isdisjoint(raw["parent_ids"]["development_test"])
    assert set(raw["parent_ids"]["validation"]).isdisjoint(raw["parent_ids"]["development_test"])
    assert {v["method"] for v in raw["rows"]} == set(m.METHODS)
    assert {v["noise"] for v in raw["rows"]} == {0., 1e-4, 1e-3}
    assert {v["reaction_multiplier"] for v in raw["rows"]} == {1., 3.}
    assert all(v["history_time"] < v["observed_time"] < v["target_time"] for v in raw["training_rows"])
    assert all(v["complete_seconds"] >= v["startup_seconds"] for v in raw["rows"])
    assert raw["offline_costs"]["all_fit_and_selection_seconds"] > 0
    assert raw["offline_costs"]["unique_teacher_and_reference_seconds"] > 0
    assert len(recorded) == 6*len(m.METHODS)
    assert all(args[1] == ["RX"] for args, _ in recorded)
    assert all(v["verdict"] == "NA" for _, row in recorded for v in row["checks"]
               if v["category"] == "utility")
    assert all(v["verdict"] != "BAD" for _, row in recorded for v in row["checks"]
               if v["category"] == "math")


def test_missing_noise_control_and_gpu_label_rejected():
    with pytest.raises(ValueError, match="retain clean"):
        m._settings(SimpleNamespace(protocol={"resolve": {"memory": {"noise_levels": [0., 1e-4, .02]}}}))
    with pytest.raises(ValueError, match="CPU"):
        m.run(SimpleNamespace(device="cuda"))
