"""Regression tests for bounded training, independent fields and honest comparisons."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.frontier import neural
from tdn.analysis.frontier.models import make_model
from tdn.analysis.frontier.protocol import build_protocol
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import horizon_key


class Budget:
    def check(self):
        pass


def context(path, protocol, stage="train", prerequisites=None):
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    return SimpleNamespace(path=path, protocol=protocol, stage=stage, prerequisites=prerequisites or {},
                           budget=Budget(), device="cpu", rows=rows,
                           record=lambda *args, **kwargs: rows.append((args, kwargs)))


def bank(protocol, split):
    """Small manufactured split for orchestration tests, not scientific evidence."""
    n = protocol["train_grid"]
    x = torch.arange(n, dtype=torch.float64).reshape(1, 1, n, 1) / n
    base = make_model("df", "discrete", protocol["model_config"])
    parents = []
    horizons = sorted(set(protocol["train_horizons"] + protocol["validation_horizons"] +
        [round(sum(s[:i]), 12) for s in protocol["confirm_schedules"] for i in range(1, len(s) + 1)]))
    for index in range(2):
        u = (.4 + .05 * torch.sin(2 * torch.pi * x + index)).expand(1, 1, n, n).clone()
        parent = dict(parent_id=f"{split}-{index}", field_cluster=f"{split}-field-{index}",
                      regime=("low", "mixed")[index], kappa=.005, reaction_rate=1., states={str(n): u}, references={})
        for h in horizons:
            truth = base(u, h, Equation(.005, 1.), Geometry((n, n), (1., 1.)))
            parent["references"][f"discrete:{n}:{horizon_key(h)}"] = dict(
                state=truth, accepted=True, uncertainty_rms=1.e-10, uncertainty_max_bound=1.e-10)
        parents.append(parent)
    return parents


@pytest.fixture
def tiny():
    p = build_protocol("smoke")
    p.update(tracks=["discrete"], models=["rank1", "rank1_frozen", "fno_small", "df"], grids=[8],
             timing_repeats=2, bootstrap_replicates=20)
    p["training"].update(validation_every=2, updates=2)
    return p


def test_full_inventory_has_all_fractions_seeds_tracks_and_analytic_controls():
    rows = neural.expected_model_specs(build_protocol("full"))
    assert len(rows) == 80
    assert len({r["model_id"] for r in rows}) == 80
    assert sum(r["family"] == "rank1" for r in rows) == 12
    assert sum(r["family"] == "df" for r in rows) == 2
    assert all(r["seed"] is None and r["train_count"] == 0 for r in rows if r["family"] == "df")


def test_nested_subsets_retain_entire_independent_field_and_balance_regimes():
    parents = [{"parent_id": f"{regime}-{field}-{physics}", "field_cluster": f"{regime}-{field}",
                "regime": regime} for regime in ("a", "b") for field in range(3) for physics in range(2)]
    small, large = neural.nested_subset(parents, 2), neural.nested_subset(parents, 4)
    assert len(small) == 4 and len(large) == 8
    assert {p["parent_id"] for p in small} < {p["parent_id"] for p in large}
    assert {p["regime"] for p in small} == {"a", "b"}
    with pytest.raises(ValueError, match="independent fields"):
        neural.nested_subset(parents, 7)


def test_error_decomposition_and_spectral_parseval():
    generator = torch.Generator().manual_seed(55)
    value = .4 + .1 * torch.randn(1, 1, 8, 8, dtype=torch.float64, generator=generator)
    ref = {"state": torch.full_like(value, .4), "accepted": True,
           "uncertainty_rms": 1.e-5, "uncertainty_max_bound": 2.e-5}
    metrics = neural.endpoint_errors(value, ref)
    assert metrics["error_rms"] ** 2 == pytest.approx(metrics["centered_rms"] ** 2 + metrics["mean_error"] ** 2)
    assert metrics["error_rms"] ** 2 == pytest.approx(metrics["spectral_low_rms"] ** 2 + metrics["spectral_high_rms"] ** 2)
    assert metrics["upper_rms"] == pytest.approx(metrics["error_rms"] + 1.e-5)
    assert metrics["upper_max"] == pytest.approx(metrics["error_max"] + 2.e-5)


def test_nonfinite_endpoint_is_not_clipped_or_encoded_as_zero_error():
    value = torch.ones(1, 1, 8, 8) * float("nan")
    result = neural.endpoint_errors(value, {"state": torch.zeros_like(value), "accepted": True})
    assert not result["finite"] and result["error_rms"] is None


def test_unresolved_reference_does_not_manufacture_zero_uncertainty():
    value = torch.full((1, 1, 8, 8), .4)
    result = neural.endpoint_errors(value, {"state": value, "accepted": False,
        "uncertainty_rms": None, "uncertainty_max_bound": None})
    assert result["finite"] and result["error_rms"] == 0
    assert result["upper_rms"] is None and result["upper_max"] is None


def test_rollout_captures_unequal_intermediates_and_validates_schedule():
    model = lambda u, h, eq, geom: u + h
    endpoint, middle = neural.rollout(model, torch.zeros(1), [.01, .04, .02], None, None, capture_intermediates=True)
    assert float(endpoint) == pytest.approx(.07)
    assert [float(x) for x in middle] == pytest.approx([.01, .05, .07])
    with pytest.raises(ValueError, match="positive finite"):
        neural.rollout(model, torch.zeros(1), [0.], None, None)


@pytest.mark.parametrize("track", ["discrete", "continuum"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("schedule", [[.01] * 3, [.01, .04, .02]])
def test_classical_fusion_preserves_nonuniform_splitting_map(track, dtype, schedule):
    model = make_model("df", track, {"modes": 1, "reaction_substeps": 2})
    generator = torch.Generator().manual_seed(532)
    initial = .4 + .05 * torch.rand(1, 1, 8, 8, dtype=dtype, generator=generator)
    eq, geom = Equation(.003, 1.), Geometry((8, 8), (1., 1.))
    fused, _ = neural.rollout(model, initial, schedule, eq, geom)
    separate, middle = neural.rollout(model, initial, schedule, eq, geom, capture_intermediates=True)
    assert len(middle) == len(schedule)
    torch.testing.assert_close(fused, separate, atol=5.e-7 if dtype == torch.float32 else 2.e-14,
                               rtol=5.e-7 if dtype == torch.float32 else 2.e-14)


def test_training_retains_initialization_and_frozen_inventory(tmp_path, tiny, monkeypatch):
    from tdn.analysis.frontier import data
    banks = {s: bank(tiny, s) for s in ("train", "validation", "confirmation")}
    accesses = []
    def load(ctx, split):
        accesses.append(split)
        return banks[split]
    monkeypatch.setattr(data, "load_split", load)
    ctx = context(tmp_path / "train", tiny)
    result = neural.train(ctx)
    assert set(accesses) == {"train", "validation"}
    assert result["selected_models"] == 4 and result["tuning_trials"] == 2
    catalog = json.loads((ctx.path / "catalog.json").read_text())
    freeze = json.loads((ctx.path / "freeze.json").read_text())
    assert len(freeze["checkpoint_hashes"]) == 4
    records = {r["family"]: r for r in catalog["records"]}
    a = torch.load(ctx.path / records["rank1"]["initial_checkpoint"], weights_only=True)
    b = torch.load(ctx.path / records["rank1_frozen"]["checkpoint"], weights_only=True)
    assert a["state_dict"].keys() == b["state_dict"].keys()
    assert all(torch.equal(a["state_dict"][key], b["state_dict"][key]) for key in a["state_dict"])
    assert records["rank1_frozen"]["trainable_parameters"] == 0
    curves = [json.loads(line) for line in (ctx.path / "learning_curves.jsonl").read_text().splitlines()]
    arm = [r for r in curves if r["model_id"] == records["rank1"]["model_id"]]
    assert [r["update"] for r in arm] == [0, 1, 2]
    assert arm[1]["validation_loss"] is None and not arm[1]["validation_measured"]
    assert arm[2]["examples_seen"] == 2 and arm[2]["validation_measured"]
    models = neural.load_models(context(tmp_path / "confirm", tiny, "confirm", {"train": ctx.path}))
    assert len(models) == 4
    confirmation = context(tmp_path / "confirm", tiny, "confirm", {"train": ctx.path})
    outcome = neural.confirm(confirmation)
    assert outcome["endpoint_rows"] == 40 and outcome["failed_or_missing_endpoints"] == 0
    assert accesses[-1] == "confirmation"
    endpoint_rows = json.loads((confirmation.path / "confirmation_rows.json").read_text())["rows"]
    assert len(endpoint_rows) == len(confirmation.rows)
    assert {r["final_time"] for r in endpoint_rows} == {.06, .12}
    assert all(r["timing"]["repeats"] == 2 for r in endpoint_rows)
    extras = [r for r in endpoint_rows if r["schedule_kind"] == "classical_frontier"]
    assert len(extras) == 16 and {r["family"] for r in extras} == {"df"}
    assert all(r["requested_outputs"] == "endpoint_only" and not r["diagnostic_intermediates"]
               and r["intermediate_reference_count"] == 0 for r in extras)
    primary = [r for r in endpoint_rows if r["schedule_kind"] == "primary"]
    assert all(r["diagnostic_intermediates"] and r["intermediate_reference_count"] > 0 for r in primary)
    gate = json.loads((confirmation.path / "gate.json").read_text())
    assert gate["selected_policy_models"] == catalog["selected_policy_models"]
    checkpoint = ctx.path / records["rank1"]["checkpoint"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    accesses.clear()
    with pytest.raises(ValueError, match="bytes changed"):
        neural.confirm(context(tmp_path / "tamper", tiny, "confirm", {"train": ctx.path}))
    assert accesses == []


def test_train_validation_field_overlap_blocks_before_trials(tmp_path, tiny, monkeypatch):
    from tdn.analysis.frontier import data
    same = bank(tiny, "train")
    monkeypatch.setattr(data, "load_split", lambda ctx, split: same)
    with pytest.raises(ValueError, match="overlap"):
        neural.train(context(tmp_path, tiny))


def test_only_explicit_numerical_failure_is_recovered(tmp_path, tiny, monkeypatch):
    from tdn.analysis.frontier import models
    actual = models.make_model
    class Bad(torch.nn.Module):
        def __init__(self, error):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(())); self.error = error
        def forward(self, *args):
            if self.error:
                raise RuntimeError("programming failure")
            return args[0] * self.weight * float("nan")
        def architecture_metadata(self):
            return {"family": "rank1"}
    spec = neural.expected_model_specs(tiny)[0]
    ctx = context(tmp_path / "nan", tiny)
    samples = neural._samples(bank(tiny, "train"), [.02], 8, "discrete", "cpu",
                               actual("df", "discrete", tiny["model_config"]), 1.e-6)
    monkeypatch.setattr(models, "make_model", lambda *args, **kw: Bad(False))
    record = neural._trial(ctx, spec, tiny["model_config"], samples, samples, .001, 2, "final")
    assert record["training_outcome"] == "NUMERICAL_FAILURE"
    assert not record["checkpoint_validated"]
    monkeypatch.setattr(models, "make_model", lambda *args, **kw: Bad(True))
    with pytest.raises(RuntimeError, match="programming failure"):
        neural._trial(context(tmp_path / "program", tiny), spec, tiny["model_config"], samples, samples, .001, 2, "final")


def row(family, *, field="field-1", horizon=.12, grid=8, seed=1, cost=1., error=1.e-6, schedule_id="a"):
    return {"model_id": f"discrete/{family}/n2/seed{seed}", "family": family, "track": "discrete",
            "parent_id": field, "field_cluster": field, "grid": grid, "seed": seed, "train_count": 2,
            "schedule_id": schedule_id, "schedule": [horizon], "final_time": horizon, "status": "COMPLETED",
            "reference_accepted": True, "error_rms": error, "error_max": error,
            "upper_rms": error, "upper_max": error, "cost_seconds": cost,
            "target_results": {"2e-05": error <= 2.e-5}}


def test_frontiers_match_horizon_and_paired_seed(tmp_path, tiny):
    tiny["seeds"] = [1, 2]
    rows = [row("rank1", seed=1, cost=2), row("fno_small", seed=1, cost=4),
            row("fno_small", seed=2, cost=.01), row("rank1", seed=2, cost=2),
            row("fno_small", seed=1, cost=.02, horizon=.24), row("rank1", seed=1, cost=2, horizon=.24)]
    comparisons = neural.matched_comparisons(rows, tiny)
    selected = [r for r in comparisons if r["seed"] == 1 and r["final_time"] == .12 and r["rms_target"] == 2.e-5]
    assert len(selected) == 1
    assert selected[0]["control_over_candidate_speed_ratio"] == 2.


def test_cluster_summary_preserves_horizon_grid_and_independence():
    rows = []
    for horizon in (.12, .24):
        for grid in (8, 16):
            for seed in (1, 2, 3):
                rows.extend([row("rank1", horizon=horizon, grid=grid, seed=seed, error=1.e-6),
                             row("fno_small", horizon=horizon, grid=grid, seed=seed, error=2.e-6)])
    result = neural.paired_field_summary(rows, bootstrap=20)
    assert len(result) == 4
    assert all(r["independent_fields"] == 1 and r["paired_rows"] == 3 for r in result)
    assert all(r["error_ratio"] == pytest.approx(2) and r["error_ratio_ci95"] is None for r in result)


def test_parameter_free_analytic_control_is_paired_with_every_learning_seed():
    ours = [row("rank1", seed=seed, cost=2., error=1.e-6) for seed in (1, 2, 3)]
    control = row("analytic_quad", seed=None, cost=1., error=3.e-6)
    control["train_count"] = 0
    result = neural.paired_field_summary([*ours, control], bootstrap=20)
    assert len(result) == 1 and result[0]["competitor"] == "analytic_quad"
    assert result[0]["paired_rows"] == 3 and result[0]["independent_fields"] == 1
    assert result[0]["error_ratio"] == pytest.approx(3.)


def test_all_physical_controls_survive_neural_comparator_pairing():
    controls = ("analytic_quad", "analytic_quad_cubic", "df", "etdrk4")
    learned = ("rank1", "fno_small", "fno_standard", "direct_fno", "rank1_postcompression", "rank1_frozen")
    rows = []
    for field in ("first", "second"):
        for horizon in (.12, .24):
            for grid in (8, 16):
                for seed in (1, 2, 3):
                    rows.extend(row(family, field=field, horizon=horizon, grid=grid, seed=seed,
                        error=1.e-6 if family == "rank1" else 2.e-6) for family in learned)
                for family in controls:
                    control = row(family, field=field, horizon=horizon, grid=grid, seed=None, error=3.e-6)
                    control["train_count"] = 0
                    rows.append(control)
    result = neural.paired_field_summary(rows, bootstrap=20)
    assert len(result) == 9 * 2 * 2
    for horizon in (.12, .24):
        for grid in (8, 16):
            selected = [item for item in result if item["final_time"] == horizon and item["grid"] == grid]
            assert {item["competitor"] for item in selected} == (set(learned) - {"rank1"}) | set(controls)
            assert all(item["independent_fields"] == 2 and item["paired_rows"] == 6 for item in selected)
            assert all(item["error_ratio"] == pytest.approx(3.) for item in selected if item["competitor"] in controls)


def test_solver_gate_requires_every_cell_and_preregistered_model(tiny):
    chosen = "discrete/rank1/n2/seed3600011"
    catalog = {"selected_policy_models": {"discrete": chosen}, "records": []}
    rows = []
    for field in ("a", "b"):
        rows.append({"model_id": chosen, "comparator": "strongest_classical", "track": "discrete",
                     "grid": 8, "final_time": tiny["policies"]["horizon"], "rms_target": 2.e-5,
                     "field_cluster": field, "eligibility": "ELIGIBLE", "control_over_candidate_speed_ratio": 2.})
    gate = neural.confirmation_gate(catalog, [], rows, tiny)
    assert gate["solver_utility_eligible"] and not gate["neural_baseline_adequate"]
    rows[-1]["eligibility"] = "CANDIDATE_INFEASIBLE"
    assert not neural.confirmation_gate(catalog, [], rows, tiny)["solver_utility_eligible"]


def test_classical_uniform_plan_covers_all_counts_without_duplicate_noise_selection():
    protocol = build_protocol("full")
    plan = neural.confirmation_plan(protocol)
    coverage = plan["classical_frontier_coverage"]
    assert {(item["final_time"], item["steps"]) for item in coverage} == {
        (horizon, count) for horizon in (.12, .24) for count in (1, 2, 4, 8)}
    reused = [item for item in coverage if item["reused_primary_measurement"]]
    assert [(item["final_time"], item["steps"], item["schedule_id"]) for item in reused] == [(.12, 4, "schedule-2")]
    assert len(plan["schedules"]) == 12
    for specification in plan["schedules"]:
        if specification["kind"] == "classical_frontier":
            assert set(specification["families"]) == neural.CLASSICAL
            assert specification["requested_outputs"] == "endpoint_only"
            assert not specification["diagnostic_intermediates"]
    for item in coverage:
        specification = next(row for row in plan["schedules"] if row["schedule_id"] == item["schedule_id"])
        assert specification["schedule"] == pytest.approx([item["final_time"] / item["steps"]] * item["steps"])


def test_one_step_classical_control_prevents_false_solver_speedup(tiny):
    seed = tiny["seeds"][0]
    horizon = tiny["policies"]["horizon"]
    primary, extras = [], []
    for field in ("first", "second"):
        learned = row("rank1", field=field, horizon=horizon, seed=seed, cost=1.)
        classical = row("df", field=field, horizon=horizon, seed=None, cost=2.)
        classical["train_count"] = 0
        primary.extend([learned, classical])
        one_step = copy.deepcopy(classical)
        one_step.update(cost_seconds=.1, schedule_id="classical-uniform-one-step", requested_outputs="endpoint_only")
        extras.append(one_step)
    catalog = {"selected_policy_models": {"discrete": primary[0]["model_id"]}, "records": []}
    old = neural.matched_comparisons(primary, tiny)
    assert neural.confirmation_gate(catalog, primary, old, tiny)["solver_utility_eligible"]
    corrected = neural.matched_comparisons(primary + extras, tiny)
    relevant = [item for item in corrected if item["comparator"] == "strongest_classical"]
    assert all(item["control_schedule"] == "classical-uniform-one-step" for item in relevant)
    assert all(item["control_over_candidate_speed_ratio"] == pytest.approx(.1) for item in relevant)
    assert not neural.confirmation_gate(catalog, primary + extras, corrected, tiny)["solver_utility_eligible"]
