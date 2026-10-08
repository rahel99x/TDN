"""Scaling must retain accuracy, failed arms and genuinely different fields."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.frontier.models import make_model
from tdn.analysis.frontier import scaling, neural, data, numerics
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import horizon_key


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def protocol():
    return dict(seeds=[11, 12], tracks=["discrete"], models=["rank1", "df"],
        training={"subset_sizes": [1, 2]}, primary_target=1e-4, practical_speedup=1.2, timing_warmup=1, timing_seed=73,
        scaling=dict(families=["rank1", "df"], grids=[8], batches=[2],
                     horizon=.04, steps=2, repeats=2))


def selected_model(family, *, seed=None, count=0):
    model = make_model(family, "discrete", {"modes": 2}).float()
    model.selection_metadata = dict(seed=seed, train_count=count,
        selected_state="SELECTED_INITIALIZATION" if family == "rank1" else "ANALYTIC_CONTROL", updates_selected=0)
    return model


def parents(accepted=True):
    result = []
    x = torch.arange(8, dtype=torch.float64) / 8
    equation, geometry = Equation(.003, 0.), Geometry((8, 8), (1., 1.))
    for index in range(2):
        state = (.4 + .06 * torch.cos(2 * torch.pi * x[:, None] + .8 * index)).expand(8, 8).clone()[None, None]
        truth = numerics.heat_step(state, .04, equation, geometry)
        result.append(dict(parent_id=f"p{index}", field_cluster=f"f{index}", kappa=.003,
            reaction_rate=0., lengths=[1., 1.], states={"8": state},
            references={f"discrete:8:{horizon_key(.04)}": dict(state=truth, accepted=accepted,
                uncertainty_rms=1e-12, uncertainty_max_bound=1e-12)}))
    return result


def test_scaling_selection_is_frozen_seed_and_largest_data_not_fastest_model():
    models = {"first": selected_model("rank1", seed=11, count=2),
              "other_seed": selected_model("rank1", seed=12, count=2),
              "less_data": selected_model("rank1", seed=11, count=1),
              "df": selected_model("df"),
              "unrequested": selected_model("direct_fno", seed=11, count=2)}
    selected = scaling.select_models(models, protocol())
    assert list(selected) == ["first", "df"]
    assert selected["first"] is models["first"]


def test_scaling_rejects_duplicate_model_instances():
    value = selected_model("rank1", seed=11, count=2)
    with pytest.raises(ValueError, match="multiple"):
        scaling.select_models({"a": value, "b": deepcopy(value)}, protocol())


@pytest.mark.parametrize("failure", ["missing", "cluster", "copied_values", "physics", "geometry", "shape"])
def test_scaling_rejects_artificial_or_incompatible_batches(failure):
    values = parents()
    if failure == "missing": values.pop()
    elif failure == "cluster": values[1]["field_cluster"] = values[0]["field_cluster"]
    elif failure == "copied_values": values[1]["states"]["8"] = values[0]["states"]["8"].clone()
    elif failure == "physics": values[1]["kappa"] = .1
    elif failure == "geometry": values[1]["lengths"] = [2., 1.]
    elif failure == "shape": values[1]["states"]["8"] = values[1]["states"]["8"].expand(2, 1, 8, 8)
    with pytest.raises(ValueError):
        scaling.distinct_batch(values, 2, 8)


def test_scaling_accepts_distinct_fields_with_identical_equation():
    values = parents()
    assert scaling.distinct_batch(values, 2, 8) == values


@pytest.mark.parametrize("errors,wanted", [
    ([], None),
    ([dict(reference_accepted=False, upper_rms=0., upper_max=0.)], None),
    ([dict(reference_accepted=True, upper_rms=1e-6, upper_max=2e-6)], True),
    ([dict(reference_accepted=True, upper_rms=1e-6, upper_max=2e-4)], False),
    ([dict(reference_accepted=True, upper_rms=None, upper_max=None)], False),
    ([dict(reference_accepted=True, upper_rms=float("nan"), upper_max=1e-6)], False),
])
def test_scaling_accuracy_gate_requires_every_accepted_reference_and_both_norms(errors, wanted):
    assert scaling.accuracy_gate(errors, 1e-4) is wanted


@pytest.mark.parametrize("candidate_accuracy,control_accuracy", [(None, True), (True, None), (False, True), (True, False), (None, None)])
def test_scaling_no_accuracy_claim_from_throughput_only(candidate_accuracy, control_accuracy):
    candidate = dict(model_id="ours", accuracy_passed=candidate_accuracy, cost_seconds=.001)
    control = dict(model_id="theirs", accuracy_passed=control_accuracy, cost_seconds=10.)
    row = scaling.speed_comparison(candidate, control)
    assert row["speed_ratio"] is None and not row["matched_accuracy"]
    assert row["status"] == "ACCURACY_UNRESOLVED_OR_FAILED"
    assert row["throughput_advantage_established"] is False


@pytest.mark.parametrize("invalid", [None, 0., -1., float("inf"), float("nan")])
def test_scaling_invalid_cost_does_not_make_a_speedup(invalid):
    candidate = dict(model_id="ours", accuracy_passed=True, cost_seconds=invalid)
    control = dict(model_id="theirs", accuracy_passed=True, cost_seconds=1.)
    row = scaling.speed_comparison(candidate, control)
    assert row["speed_ratio"] is None and row["status"] == "INVALID_OR_MISSING_TIMING"


def test_scaling_speed_is_descriptive_even_when_accuracy_is_accepted():
    candidate = dict(model_id="ours", accuracy_passed=True, cost_seconds=.5)
    control = dict(model_id="theirs", accuracy_passed=True, cost_seconds=1.)
    row = scaling.speed_comparison(candidate, control)
    assert row["speed_ratio"] == 2. and row["status"] == "ELIGIBLE"
    assert row["throughput_advantage_established"] is False


@pytest.mark.parametrize("accepted,missing_model", [(True, False), (False, False), (True, True)])
def test_scaling_run_retains_references_caches_and_missing_arms(tmp_path, monkeypatch, accepted, missing_model):
    declared = protocol()
    values = parents(accepted)
    models = {"discrete/rank1/n2/seed11": selected_model("rank1", seed=11, count=2),
              "discrete/df/n0/seedNone": selected_model("df")}
    if missing_model:
        del models["discrete/rank1/n2/seed11"]
    monkeypatch.setattr(neural, "load_models", lambda ctx: models)
    monkeypatch.setattr(data, "load_split", lambda ctx, split: values)
    recorded = []
    ctx = SimpleNamespace(protocol=declared, device="cpu", path=tmp_path,
        budget=SimpleNamespace(check=lambda: None), record=lambda *args, **kw: recorded.append((args, kw)))
    result = scaling.run(ctx)
    rows = json.loads((tmp_path / "scaling_rows.json").read_text())["rows"]
    comparisons = json.loads((tmp_path / "scaling_comparisons.json").read_text())["rows"]
    assert result["workload_rows"] == 2 and result["comparison_rows"] == 1 and len(recorded) == 3
    assert len(comparisons) == 1
    assert all(row["independent_fields"] == ["f0", "f1"] for row in rows)
    assert all(row["final_time"] == .04 and row["schedule"] == [.02, .02] for row in rows)
    for row in rows:
        if row["status"] == "MODEL_UNAVAILABLE":
            assert missing_model and row["accuracy_passed"] is None and row["cost_seconds"] is None
            assert row["failure"] == "FROZEN_MODEL_UNAVAILABLE"
        else:
            assert isinstance(row["cache"], dict) and row["cache"]["state_dependent_cache"] is False
            assert row["cost"]["cold_seconds"] > 0 and len(row["cost"]["samples_seconds"]) == 2
            assert row["numerical_screening_seconds"] > 0
            assert row["total_measured_trial_seconds"] == pytest.approx(
                row["numerical_screening_seconds"] + row["cost"]["cold_seconds"]
                + row["cost"]["warmup_seconds"] + row["cost"]["total_seconds"])
            assert row["accuracy_passed"] is (True if accepted else None)
            assert row["selected_state"] in ("SELECTED_INITIALIZATION", "ANALYTIC_CONTROL")
            assert row["updates_selected"] == 0
    assert comparisons[0]["candidate_selected_state"] == (None if missing_model else "SELECTED_INITIALIZATION")
    assert comparisons[0]["comparator_selected_state"] == "ANALYTIC_CONTROL"
    if not accepted or missing_model:
        assert comparisons[0]["speed_ratio"] is None
    else:
        assert comparisons[0]["speed_ratio"] > 0


@pytest.mark.parametrize("family,kind", [("fno_standard", "neural"), ("df", "classical")])
def test_scaling_slow_accurate_candidate_scores_bad_utility_not_success(family, kind):
    from tdn.analysis.frontier.core import score_checks
    base = dict(track="discrete", grid=32, batch_size=2, final_time=.04, target=1e-4,
        independent_fields=["field-a", "field-b"], accuracy_passed=True, seed=11, train_count=2,
        parameters=43, schedule=[.02, .02])
    candidate = dict(base, model_id="rank-one", family="rank1", cost_seconds=2.)
    comparator = dict(base, model_id="control", family=family, cost_seconds=1., parameters=100)
    recorded = []
    ctx = SimpleNamespace(protocol=protocol(), record=lambda *args, **kw: recorded.append((args, kw)))
    comparison = scaling.record_crossover(ctx, candidate, comparator)
    _, record = recorded[0]
    utility = next(c for c in record["checks"] if c["category"] == "utility")
    assert utility["check_id"] == f"observed-{kind}-throughput-crossover"
    assert utility["measured"] == .5 and utility["target"] == 1.2
    assert utility["verdict"] == "BAD" and utility["required"]
    assert score_checks(record["checks"])["verdict"] == "BAD"
    assert score_checks(record["checks"])["score_1_100"] < 100
    assert record["metrics"] == comparison
    assert comparison["candidate_cost_seconds"] == 2. and comparison["comparator_cost_seconds"] == 1.
    assert comparison["candidate_parameters"] == 43 and comparison["comparator_parameters"] == 100
    assert comparison["independent_fields"] == base["independent_fields"]
    assert comparison["observed_threshold_passed"] is False


@pytest.mark.parametrize("candidate_accuracy,control_accuracy,cost", [(None, True, .01), (True, None, .01),
    (False, True, .01), (True, False, .01), (True, True, None), (True, True, 0.)])
def test_scaling_ineligible_speed_is_na_and_cannot_score_good_100(candidate_accuracy, control_accuracy, cost):
    from tdn.analysis.frontier.core import score_checks
    base = dict(track="continuum", grid=64, batch_size=1, final_time=.06, target=2e-5,
        independent_fields=["field-a"], seed=11, train_count=2, parameters=43, schedule=[.03, .03])
    candidate = dict(base, model_id="rank-one", family="rank1", accuracy_passed=candidate_accuracy, cost_seconds=cost)
    comparator = dict(base, model_id="fno", family="fno_small", accuracy_passed=control_accuracy, cost_seconds=1.)
    recorded = []
    ctx = SimpleNamespace(protocol=protocol(), record=lambda *args, **kw: recorded.append(kw))
    comparison = scaling.record_crossover(ctx, candidate, comparator)
    checks = recorded[0]["checks"]
    utility = next(c for c in checks if c["category"] == "utility")
    assert utility["verdict"] == "NA" and utility["measured"] is None
    assert score_checks(checks)["score_1_100"] < 100 and score_checks(checks)["verdict"] != "GOOD"
    assert comparison["speed_ratio"] is None and comparison["observed_threshold_passed"] is None
    assert comparison["throughput_advantage_established"] is False


@pytest.mark.parametrize("altered", [{"final_time": .08}, {"track": "continuum"},
    {"target": 1e-6}, {"grid": 64}, {"independent_fields": ["different"]}])
def test_scaling_crossover_rejects_mismatched_outputs_before_scoring(altered):
    base = dict(track="discrete", grid=32, batch_size=1, final_time=.04, target=1e-4,
        independent_fields=["field-a"], accuracy_passed=True, seed=11, train_count=2,
        parameters=43, schedule=[.04], cost_seconds=1.)
    candidate = dict(base, model_id="rank-one", family="rank1")
    comparator = dict(base, model_id="fno", family="fno_small", **altered)
    ctx = SimpleNamespace(protocol=protocol(), record=lambda *args, **kw: pytest.fail("Mismatched comparison was recorded"))
    with pytest.raises(ValueError, match="different requested outputs"):
        scaling.record_crossover(ctx, candidate, comparator)


def test_scaling_numerical_failure_is_retained_instead_of_claiming_missing_accuracy(tmp_path, monkeypatch):
    class Failed(torch.nn.Module):
        family, track = "rank1", "discrete"
        selection_metadata = dict(seed=11, train_count=2)
        cache_metadata = {}
        def forward(self, u, *args): return u * float("nan")
        def clear_cache(self): pass
    models = {"discrete/rank1/n2/seed11": Failed(), "discrete/df/n0/seedNone": selected_model("df")}
    monkeypatch.setattr(neural, "load_models", lambda ctx: models)
    monkeypatch.setattr(data, "load_split", lambda ctx, split: parents())
    ctx = SimpleNamespace(protocol=protocol(), device="cpu", path=tmp_path,
        budget=SimpleNamespace(check=lambda: None), record=lambda *args, **kw: None)
    scaling.run(ctx)
    rows = json.loads((tmp_path / "scaling_rows.json").read_text())["rows"]
    bad = next(row for row in rows if row["family"] == "rank1")
    assert bad["status"] == "NUMERICAL_FAILURE" and bad["accuracy_passed"] is False
    assert bad["failure"] == "Nonfinite scaling output"
    assert bad["cost_seconds"] is None
