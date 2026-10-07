"""Teacher eligibility, coefficient ownership and accuracy-matched cost logs."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdn.numerics import Equation, Geometry
from tdn.numerics.subflows import reaction_step
from tdn.analysis.roadmap import classical
from tdn.analysis.roadmap.core import Context, validate_row
from tdn.analysis.roadmap.protocol import build_protocol
from tdn.research.experiment import horizon_key


class Budget:
    def check(self):
        pass


def context(path, stage="scaling"):
    return Context(build_protocol("smoke"), stage, path, {}, "cpu", Budget())


def initial(n=8):
    x = torch.arange(n, dtype=torch.float64) / n
    return (.4 + .1 * torch.cos(2 * torch.pi * x[:, None]) + .02 * torch.sin(2 * torch.pi * x[None, :])).reshape(1, 1, n, n)


def candidate(parent, *, accepted=True, feasible=False, family="strang_diffusion_first", steps=1):
    return dict(parent_id=parent, grid=8, horizon=.04, reference_accepted=accepted,
                feasible=feasible, family=family, steps=steps)


def test_headroom_does_not_count_unresolved_or_entirely_infeasible_conditions():
    rows = [candidate("unresolved", accepted=False), candidate("infeasible"),
            candidate("bare_fine", feasible=True), candidate("eligible"),
            candidate("eligible", feasible=True, family="etdrk4", steps=2)]
    counts = classical.summarize_headroom(rows)
    assert counts["conditions"] == 4
    assert counts["eligible_conditions"] == 2
    assert counts["conditions_needing_more_than_one_bare_df_step"] == 1
    assert counts["unresolved_reference_conditions"] == 1
    assert counts["no_feasible_classical_conditions"] == 1
    assert counts["unresolved_conditions"] == 2


def test_unresolved_headroom_log_is_na_not_new_architecture_opportunity(monkeypatch, tmp_path):
    ctx = context(tmp_path, "headroom")
    p = ctx.protocol
    p["headroom"] = {"clusters": 1, "grids": [8, 16, 32], "horizons": [.04], "steps": [1], "target": 2e-4}
    monkeypatch.setattr(classical, "_continuum_reference", lambda *args: (initial(32), {"accepted": True}))
    monkeypatch.setattr(classical, "reference", lambda *args: (None, .01))
    monkeypatch.setattr(classical, "rollout", lambda model, u, *args, **kwargs: u)
    monkeypatch.setattr(classical, "_unfused_df", lambda u, *args: u)
    counts = classical.run_headroom(ctx)
    assert counts["headroom_conditions"] == 0 and counts["eligible_conditions"] == 0
    summary = next(row for row in ctx.rows if row["experiment_id"].startswith("C0/"))
    gap = next(c for c in summary["checks"] if c["check_id"] == "some-classical-accuracy-headroom")
    assert gap["verdict"] == "NA" and gap["measured"] is None
    validate_row(summary)


@pytest.mark.parametrize("family", ["etdrk4", "gl3_fused"])
def test_prepared_cache_charges_setup_and_preserves_complete_outputs(family):
    u, eq, g = initial(), Equation(.002, 2.), Geometry((8, 8), (1., 1.))
    schedule = [.01, .02, .01]
    cached, cm = classical.prepared_workload(u, schedule, eq, g, family, queries=3)
    fresh, fm = classical.prepared_workload(u, schedule, eq, g, family, cached=False, queries=3)
    torch.testing.assert_close(cached, fresh, rtol=0, atol=0)
    assert cm["coefficient_preparations"] == 2
    assert cm["cache_hits"] == 7
    assert fm["coefficient_preparations"] == 9
    assert cm["cached_tensor_bytes"] > 0 and not cm["state_dependent_cache"]


def test_prepared_cache_never_reuses_changed_operator_or_geometry():
    u, eq, g = initial(), Equation(.002, 2.), Geometry((8, 8), (1., 1.))
    cache = classical.PreparedCache()
    first = cache.get(u, .01, eq, g, "etdrk4")
    assert cache.get(u.repeat(2, 1, 1, 1), .01, eq, g, "etdrk4") is first
    cases = [(u, .02, eq, g, "etdrk4"),
             (u, .01, Equation(.01, 2.), g, "etdrk4"),
             (u, .01, Equation(.002, 3.), g, "etdrk4"),
             (u, .01, eq, Geometry((8, 8), (2., 1.)), "etdrk4"),
             (initial(10), .01, eq, Geometry((10, 10), (1., 1.)), "etdrk4"),
             (u.float(), .01, eq, g, "etdrk4"),
             (u, .01, eq, g, "gl3_fused")]
    for state, h, equation, geometry, family in cases:
        prepared = cache.get(state, h, equation, geometry, family)
        assert prepared is not first
        expected = classical.PreparedCache().get(state, h, equation, geometry, family)(state)
        torch.testing.assert_close(prepared(state), expected, rtol=0, atol=0)
    assert cache.metadata["cache_entries"] == 8
    assert cache.metadata["cache_hits"] == 1


@pytest.mark.parametrize("h", [-.1, float("nan"), torch.tensor(.1, requires_grad=True), torch.tensor([.1])])
def test_prepared_cache_rejects_invalid_or_differentiable_time(h):
    with pytest.raises(ValueError):
        classical.PreparedCache().get(initial(), h, Equation(.002, 2.), Geometry((8, 8), (1., 1.)), "etdrk4")


def test_scaling_selects_first_metadata_seed_not_checkpoint_id_spelling():
    models = {"source-seed-20": SimpleNamespace(family="source", selection_metadata={"seed": 20}),
              "arbitrary/slash": SimpleNamespace(family="source", selection_metadata={"seed": 10}),
              "source-prefix-but-wrong-family": SimpleNamespace(family="deep_fno", selection_metadata={"seed": 30}),
              "rank2-seed-10": SimpleNamespace(family="rank2", selection_metadata={"seed": 10}),
              "initialization": SimpleNamespace(family="df_base", selection_metadata={"seed": None})}
    selected = classical.select_scaling_models(models, families=("source", "rank2"))
    assert set(selected) == {"arbitrary/slash", "rank2-seed-10"}


@pytest.mark.parametrize("accepted,uncertainty", [(False, 1e-9), (True, float("nan")), (True, -.1)])
def test_accuracy_requires_accepted_finite_nonnegative_reference_uncertainty(accepted, uncertainty):
    ref = {"state": initial(), "accepted": accepted, "uncertainty_rms": uncertainty, "uncertainty_max_bound": uncertainty}
    assert classical._reference_errors(initial(), ref) is None


class Solver(nn.Module):
    family = "source"
    selection_metadata = {"seed": 10}

    def __init__(self, wrong=False):
        super().__init__()
        self.wrong = wrong

    def forward(self, u, h, equation, geometry):
        return u + .01 if self.wrong else reaction_step(u, h, equation)


@pytest.mark.parametrize("wrong", [False, True])
def test_scaling_logs_accuracy_where_teacher_exists_and_na_elsewhere(monkeypatch, tmp_path, wrong):
    from tdn.analysis.roadmap import neural, data
    ctx = context(tmp_path)
    p = ctx.protocol
    p["scaling"] = {"grids": [8, 16], "batches": [1], "repeats": 1, "maximum_cases": 2}
    eq = Equation(.002, 2.)
    h = sum(p["confirm_schedules"][0])
    parent = dict(parent_id="confirmation-engineering", kappa=eq.kappa, reaction_rate=eq.reaction_rate,
                  references={f"discrete:8:{horizon_key(h)}": {"accepted": True,
                    "state": reaction_step(torch.full((1, 1, 8, 8), .4, dtype=torch.float64), h, eq),
                    "uncertainty_rms": 1e-10, "uncertainty_max_bound": 1e-10}})
    monkeypatch.setattr(neural, "load_frozen_models", lambda context: {"source-seed-10": Solver(wrong)})
    monkeypatch.setattr(data, "load_bank", lambda *args: [parent])
    monkeypatch.setattr(classical, "continuous_field", lambda parent, n: torch.full((1, 1, n, n), .4, dtype=torch.float64))
    monkeypatch.setattr(classical, "run_preparation_reuse", lambda *args: 0)
    calls = 0
    def measured(call, **kwargs):
        nonlocal calls
        calls += 1
        value = call()
        return value, {"median_seconds": 2. if calls % 2 else 1., "peak_allocated_bytes": None}
    ctx.measure = measured
    result = classical.run_scaling(ctx)
    assert result["models"] == 1 and result["teacher_checked_rows"] == 1 and result["throughput_only_rows"] == 1
    small, large = ctx.rows
    assert small["metrics"]["speedup_over_bare_df"] == 2
    small_gap = next(c for c in small["checks"] if c["category"] == "gap")
    assert small_gap["verdict"] == ("NA" if wrong else "GOOD")
    assert all(c["verdict"] == "NA" for c in large["checks"] if c["category"] in {"math", "gap"})
    assert (small["math_assessment"]["verdict"] == "BAD") is wrong
    assert large["math_assessment"]["verdict"] == "NA"
    for row in ctx.rows:
        validate_row(row)
