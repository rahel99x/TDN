"""Scientific isolation, selection and cost eligibility for research training."""
from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from tdn.numerics import Equation, Geometry, split_step
from tdn.research import experiment as research
from tdn.research.protocol import DEFAULT, assert_parent_disjoint, digest, parent_plan, validate_config


def simple_dataset():
    geometry = Geometry((4, 4), (1., 1.))
    parents = []
    for declared in parent_plan(smoke=True):
        state = research.initial_state(declared, geometry)
        equation = Equation(declared["kappa"], declared["reaction_rate"])
        references = {research.horizon_key(h): {"state": split_step(state, h, equation, geometry),
                       "accepted": True, "uncertainty": 1e-12} for h in (.03, .06, .11, .22)}
        parents.append({**declared, "initial": state, "references": references})
    return {"parents": parents}, geometry


def test_full_protocol_crosses_states_and_regimes_with_unique_parents():
    parents = parent_plan()
    assert len(parents) == 27
    assert len({row["seed"] for row in parents}) == 27
    for split in ("train", "validation", "diagnostic"):
        subset = [row for row in parents if row["split"] == split]
        assert len({(row["regime"], row["state_class"]) for row in subset}) == 9
    contaminated = copy.deepcopy(parents)
    contaminated[-1]["seed"] = contaminated[0]["seed"]
    with pytest.raises(ValueError, match="disjoint"):
        assert_parent_disjoint(contaminated)


@pytest.mark.parametrize("smoke", [False, True])
def test_actual_initial_states_are_unique_across_all_splits(smoke):
    geometry = Geometry((8, 8) if smoke else (32, 32), (1., 1.))
    hashes = [research.state_digest(research.initial_state(parent, geometry)) for parent in parent_plan(smoke=smoke)]
    assert len(hashes) == len(set(hashes))


def test_normalization_cannot_use_validation_or_diagnostic_states():
    dataset, geometry = simple_dataset()
    (mean, std), ids = research.training_normalization(dataset, geometry)
    altered = copy.deepcopy(dataset)
    for parent in altered["parents"]:
        if parent["split"] != "train":
            parent["initial"].fill_(.987)
    (other_mean, other_std), other_ids = research.training_normalization(altered, geometry)
    assert torch.equal(mean, other_mean) and torch.equal(std, other_std)
    assert ids == other_ids and all(name.startswith("train-") for name in ids)
    altered["parents"][0]["initial"].fill_(.2)
    (changed, _), _ = research.training_normalization(altered, geometry)
    assert not torch.equal(mean, changed)


def test_validation_rejects_invalid_hidden_stage_even_if_output_is_exact():
    dataset, geometry = simple_dataset()
    class BadIntermediate(torch.nn.Module):
        def audit_step(self, state, h, equation, geometry):
            result = split_step(state, h, equation, geometry)
            return result, {"injected": state - 2., "output": result}
    score = research.validation_loss(BadIntermediate(), [dataset["parents"][0]], [.03], geometry, "cpu", research.Budget(10))
    assert math.isinf(score)


def test_checkpoint_selection_uses_validation_and_freezes_best(tmp_path, monkeypatch):
    import tdn.research
    dataset, geometry = simple_dataset()
    for parent in dataset["parents"]:
        parent["references"] = {research.horizon_key(h): {"state": torch.full_like(parent["initial"], .7)}
                                for h in (.02, .04)}
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.5))
        def set_normalization(self, *_):
            return self
        def forward(self, state, *_):
            return self.scale.expand_as(state)
    monkeypatch.setattr(tdn.research, "build_research_model", lambda *args, **kwargs: TinyModel())
    seen = []
    scores = iter((3., 1., 2.))
    def validate(model, parents, *args):
        seen.extend(parent["split"] for parent in parents)
        return next(scores)
    monkeypatch.setattr(research, "validation_loss", validate)
    config = {**DEFAULT, "grid": [4, 4], "max_steps": 2, "validation_every": 1, "train_horizons": [.02]}
    protocol = {"config": config}
    model, record = research.train_family("tiny", dataset, protocol, tmp_path,
                                         (torch.zeros(12), torch.ones(12)), research.Budget(30))
    saved = torch.load(tmp_path / "checkpoints/tiny.pt", weights_only=True)
    assert record["selected_step"] == saved["selected_step"] == 1
    assert record["parameters_changed"] and record["steps"] == 2
    assert seen and set(seen) == {"validation"}
    assert torch.equal(model.scale, saved["state_dict"]["scale"])


def test_invalid_or_inaccurate_method_is_never_speedup_candidate():
    def row(name, error, seconds, *, valid=True):
        return {"parent_id": "diagnostic-x", "family": name, "h": .32,
                "status": "COMPLETED" if valid else "INVALID_TRAJECTORY", "error": error,
                "reference_uncertainty": .00001, "feasible": valid and error + .00001 <= .002,
                "timing": {"wall_seconds_median": seconds}}
    rows = [row("split", .004, .002), row("richardson_split", .003, .004),
            row("adaptive_split", .001, .005), row("fixed_decay", .001, .000001, valid=False),
            row("confluent_decay", .001, .002)]
    summary = research.summarize_frontier(rows, DEFAULT)
    assert summary["headroom_passed"]
    by_name = {row["family"]: row for row in summary["comparisons"]}
    assert by_name["fixed_decay"]["speedup"] is None
    assert by_name["confluent_decay"]["speedup"] == 2.5
    assert by_name["confluent_decay"]["best_classical_family"] == "adaptive_split"


@pytest.mark.parametrize("key,value", [("tolerance", .02), ("max_steps", 1000), ("grid", [64, 64]),
                                        ("families", ["confluent_decay"]), ("max_seconds", True)])
def test_research_config_cannot_expand_or_weaken_protocol(key, value):
    with pytest.raises(ValueError):
        validate_config({**copy.deepcopy(DEFAULT), key: value})


def test_boundary_parent_has_real_endpoints_and_budget_honors_stop():
    dataset, _ = simple_dataset()
    boundary = next(parent["initial"] for parent in dataset["parents"] if parent["state_class"] == "boundary")
    assert boundary.min() == 0 and boundary.max() == 1
    class Stop:
        requested = True
        signal_number = 10
    with pytest.raises(InterruptedError):
        research.Budget(10, Stop()).check()
