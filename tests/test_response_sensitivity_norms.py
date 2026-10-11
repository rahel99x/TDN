"""Independent Nyquist checks at the evaluation/physical-norm boundary."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.response import evaluation


def _case(tmp_path, target, size=4):
    u = torch.full((1, 1, size, size), .5, dtype=torch.float32)
    record = dict(parent_id="paired-parent", target_kind=target, grid=[size, size],
                  lengths=[1., 1.], horizon=.04, kappa=.003, reaction_rate=2.)
    ctx = SimpleNamespace(path=tmp_path, device="cpu", budget=SimpleNamespace(check=lambda: None),
        protocol={"evaluation": {"sensitivity_epsilons": [1/128, 1/256]}})
    return ctx, {"u0": u, "record": record}


@pytest.mark.parametrize("target,mass", [("fd_same_grid", 1.), ("galerkin_same_grid", .25)])
def test_tangent_error_and_refinement_use_correct_nyquist_mass(tmp_path, monkeypatch, target, mass):
    ctx, sample = _case(tmp_path, target)
    mode = torch.tensor([1., -1., 1., -1.], dtype=torch.float64)
    corner = (mode[:, None]*mode[None, :])[None, None]
    # Independent artificial teacher: its centered secant has an epsilon²
    # Nyquist contribution. The prediction has a different exact derivative.
    def teacher(ctx, state, record, horizon, purpose):
        delta = state.double().mean()-.5
        return state.double()+delta**3*corner, {"accepted": True, "uncertainty_rms": 0.}
    def predict(inp):
        state = inp["u0"].double()
        return state+2*(state.mean()-.5)*corner
    predict.metadata = {"method_id": "linear-probe"}
    monkeypatch.setattr(evaluation, "diagnostic_teacher", teacher)
    rows = evaluation.run_sensitivity_panel(ctx, [sample], {"probe": predict})
    mean_rows = [row for row in rows if row["direction"] == "mean"]
    assert len(mean_rows) == 2
    for row in mean_rows:
        expected_nodal = 2-row["epsilon"]**2
        assert row["tangent_error_rms"] == pytest.approx(expected_nodal*math.sqrt(mass), abs=1e-13)
        assert row["nodal_tangent_error_rms"] == pytest.approx(expected_nodal, abs=1e-13)
        nodal_change = (1/128)**2-(1/256)**2
        assert row["teacher_tangent_halving_change_rms"] == pytest.approx(nodal_change*math.sqrt(mass), abs=1e-13)
        assert row["nodal_teacher_tangent_halving_change_rms"] == pytest.approx(nodal_change, abs=1e-13)
        assert row["teacher_tangent_rms"] == pytest.approx(math.sqrt(1+mass*(1/256)**4), abs=1e-13)
        assert row["norm"] == ("nodal_rms" if target == "fd_same_grid" else "trigonometric_interpolant_rms")
    saved = json.loads((tmp_path/"sensitivity_metrics.json").read_text())
    assert saved["rows"] == rows


def test_nyquist_direction_is_unit_physical_norm_and_failures_keep_contract(tmp_path, monkeypatch):
    # On N=2 the first cosine is Nyquist in one axis: physical RMS=nodal/sqrt(2).
    ctx, sample = _case(tmp_path, "galerkin_same_grid", size=2)
    observations = []
    def teacher(ctx, state, record, horizon, purpose):
        observations.append(state.clone())
        return state.double(), {"accepted": True, "uncertainty_rms": 0.}
    def failed(inp):
        raise ValueError("Deliberate unavailable prediction")
    failed.metadata = {"method_id": "failed-probe"}
    monkeypatch.setattr(evaluation, "diagnostic_teacher", teacher)
    rows = evaluation.run_sensitivity_panel(ctx, [sample], {"failed": failed})
    # Four signed mean perturbations precede four signed first-mode probes.
    displacement = observations[4].double()-sample["u0"].double()
    nodal_size = float(displacement.square().mean().sqrt())
    assert nodal_size == pytest.approx(math.sqrt(2)/128, rel=3e-6)
    for row in rows:
        assert row["direction_normalization_norm"] == "trigonometric_interpolant_rms"
        assert row["direction_rms"] == pytest.approx(1., abs=1e-7)
        assert row["status"] == "FAILED" and row["tangent_error_rms"] is None
        assert row["norm"] == "trigonometric_interpolant_rms"
